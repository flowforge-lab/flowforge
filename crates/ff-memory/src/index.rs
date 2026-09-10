//! SQLite FTS5 recall index (RFC 0006 §6). The **default and floor** backend:
//! local, fast, no model, no cloud. The index is a **derived artifact** — it can
//! be deleted and rebuilt from the Markdown at any time (RFC 0006 §4).
//!
//! Layout is the standard FTS5 *external-content* pattern: a plain `chunks` table
//! holds the full rows (so search can return line spans and the reserved
//! `embedding` column lives here for the M5.3 hybrid backend), and a contentless
//! `chunks_fts` virtual table indexes only `text`, kept in sync by triggers.
//! Search joins the BM25 hit back to its row.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{NaiveDate, Utc};
use rusqlite::{params, Connection, OptionalExtension};

use crate::consolidate::{chunk_key, content_key_of};
use crate::embed::Embedder;

use crate::error::Result;
use crate::{DecayConfig, MemoryChunk, MemorySource};

/// Milliseconds in a day, for converting `last_accessed` deltas to fractional
/// decay days (RFC 0007 §2).
const ONE_DAY_MS: f32 = 86_400_000.0;

/// Fraction of the max score used as the retrieve-based boost ceiling (B2).
/// At `SATURATION_HITS` retrieves the additive term reaches its maximum:
/// `RETRIEVE_FRACTION * max_score`. Below saturation the boost follows a
/// sub-linear curve: `√(hits / SATURATION_HITS)`.
const RETRIEVE_FRACTION: f32 = 0.1;
/// Number of retrieves at which the boost saturates.
/// `√(SATURATION_HITS / SATURATION_HITS) = 1.0`.
const SATURATION_HITS: f32 = 10.0;

/// A search hit: the chunk plus a relevance score (higher = more relevant; it is
/// the negated BM25 distance, so callers can sort descending intuitively).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredChunk {
    pub chunk: MemoryChunk,
    pub score: f32,
    /// Effective (lazily-decayed) usage weight at search time (RFC 0007 §3).
    /// `1.0` when decay is disabled or the chunk has no `chunk_stats` row, so a
    /// fresh / never-recalled chunk is never dormant.
    pub weight: f32,
    /// Epoch-ms of the last recorded access, if any — drives the dormant age
    /// tag in `memory_search`. `None` when decay is disabled or no row exists.
    pub last_accessed_ms: Option<i64>,
}

/// Read-time usage stats for a chunk: `weight` lazily decayed to the query
/// instant (RFC 0007 §3 — dormancy is a *derived* predicate, computed not
/// stored), plus the raw `last_accessed` for age display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectiveStat {
    pub weight: f32,
    pub last_accessed_ms: i64,
}

/// Full read-time usage snapshot for one chunk, backing the M6.2 "Salience"
/// surface (#293). Unlike [`EffectiveStat`] this is emitted for *every* keyed
/// row regardless of the decay flag, and carries the raw `access_count` and
/// `pinned` flag plus a server-computed `dormant` predicate — so the frontend
/// never re-derives the threshold (RFC 0007 §7). `weight` is pin-aware: a
/// pinned chunk reads `1.0` (decay skipped), matching [`effective_stats`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkStatSnapshot {
    /// Effective (pin-aware, lazily decayed) weight at the query instant.
    pub weight: f32,
    pub last_accessed_ms: i64,
    pub access_count: u32,
    pub pinned: bool,
    /// `decay.enabled && weight < dormant_threshold && !pinned` (RFC 0007 §3).
    pub dormant: bool,
}

/// A directed edge in the `chunk_links` graph layer (memory sifting, #1293).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub from_key: String,
    pub to_key: String,
    pub kind: String,
    pub created_at: i64,
}

/// Recall backend (RFC 0006 §6). Swappable so M5.3 can add a hybrid vector index
/// behind the same seam; v1 ships [`Fts5Index`].
pub trait MemoryIndex: Send + Sync {
    /// Replace the entire index with `chunks` (full rebuild).
    fn reindex(&self, chunks: &[MemoryChunk]) -> Result<()>;
    /// Replace just the chunks for one file (used after a targeted write so a
    /// same-turn search sees the change without a full rebuild).
    fn reindex_path(&self, path: &Path, chunks: &[MemoryChunk]) -> Result<()>;
    /// Drop all chunks for a file that no longer exists.
    fn remove_path(&self, path: &Path) -> Result<()>;
    /// Ranked BM25 search. An empty/whitespace query yields no hits.
    fn search(&self, query: &str, k: usize) -> Result<Vec<ScoredChunk>>;
    /// Reinforce usage stats for the surfaced top-k `hits` (RFC 0007 §2). The
    /// default is a no-op so backends without a `chunk_stats` table (e.g. a null
    /// index) need no change; [`Fts5Index`] overrides it.
    fn reinforce(&self, _hits: &[ScoredChunk]) -> Result<()> {
        Ok(())
    }
    /// Weak reinforcement for chunks that were *ambient-injected* and the turn
    /// produced a reply (RFC 0007 §10.1). Bumps each key's `weight` toward `1.0`
    /// by the smaller `decay.ambient_gain` (vs `reinforce_gain` for a recall) and
    /// refreshes `last_accessed`, so a still-shown chunk ages more slowly. A
    /// complete no-op unless decay is enabled *and* `ambient_gain > 0`, and it
    /// never creates a row for a never-recalled chunk — so the RFC §3
    /// never-recalled-stays-`1.0` invariant is preserved. The default is a no-op
    /// so backends without a `chunk_stats` table need no change.
    fn reinforce_ambient(&self, _keys: &[String]) -> Result<()> {
        Ok(())
    }
    /// Record that a compacted tool result was pulled back verbatim via
    /// `compaction_retrieve` (RFC 0022 §4.3): a strong, use-time reinforcement
    /// signal. Aggregates a cross-session count per content key in the
    /// `retrieve_stats` table. The default is a no-op so backends without a
    /// stats table need no change; [`Fts5Index`] overrides it.
    fn record_retrieve(&self, _content_key: &str) -> Result<()> {
        Ok(())
    }
    /// Cross-session count of how many times `content_key` was pulled back via
    /// `compaction_retrieve`. `0` when never retrieved. The default is `0` for
    /// stats-less backends.
    fn retrieve_count(&self, _content_key: &str) -> Result<u64> {
        Ok(0)
    }
    /// Store that `content_key` maps to `chunk_keys` (found by FTS5 text
    /// overlap at retrieve time). The default is a no-op so backends without a
    /// stats table need no change; [`Fts5Index`] overrides it.
    fn map_retrieve_to_chunks(
        &self,
        _content_key: &str,
        _mappings: &[(String, f32)],
    ) -> Result<()> {
        Ok(())
    }
    /// Summed `retrieve_stats.count` per `chunk_key` across all content keys
    /// that map to it. Absent keys are omitted (caller treats as zero boost).
    /// The default is empty for stats-less backends.
    fn chunk_retrieve_hits(&self, _chunk_keys: &[String]) -> Result<HashMap<String, u64>> {
        Ok(HashMap::new())
    }
    /// Read-time effective (decayed) usage stats for `keys`, computed against
    /// `now_ms` without persisting (RFC 0007 §3). Keys with no `chunk_stats`
    /// row — and *all* keys when decay is disabled — are omitted from the map;
    /// callers treat an absent key as weight `1.0` (never dormant). The default
    /// is empty so backends without a stats table need no change.
    fn effective_stats(
        &self,
        _keys: &[String],
        _now_ms: i64,
    ) -> Result<HashMap<String, EffectiveStat>> {
        Ok(HashMap::new())
    }
    /// Full per-chunk usage snapshot for the M6.2 Salience surface (#293):
    /// effective (pin-aware) weight, `last_accessed_ms`, `access_count`, `pinned`,
    /// and the server-computed `dormant` predicate, for every key that has a
    /// `chunk_stats` row — emitted regardless of the decay flag (unlike
    /// [`effective_stats`](Self::effective_stats), which is empty when decay is
    /// off). Keys with no row are omitted; callers treat an absent key as a
    /// never-recalled chunk (weight `1.0`, never dormant, not pinned). The
    /// default is empty so backends without a stats table need no change.
    fn chunk_stats_snapshot(
        &self,
        _keys: &[String],
        _now_ms: i64,
    ) -> Result<HashMap<String, ChunkStatSnapshot>> {
        Ok(HashMap::new())
    }
    /// The set of `chunk_key`s whose `suppress_promotion` flag is set (#1292
    /// block A). The consolidation pass reads this to keep a failed session's
    /// touched chunks out of curated (see
    /// [`ChunkStatsSalience`](crate::consolidate::ChunkStatsSalience)). The
    /// default is empty so backends without a stats table need no change.
    fn suppressed_promotion_keys(&self) -> Result<HashSet<String>> {
        Ok(HashSet::new())
    }
    /// Reset (wake) a chunk: restore `weight` to `1.0` and stamp `last_accessed`
    /// to now, creating the row if absent (RFC 0007 §7). The default is a no-op
    /// so backends without a stats table need no change.
    fn reset_chunk(&self, _key: &str) -> Result<()> {
        Ok(())
    }
    /// Sleep a chunk: force `weight` to `0.0` — below any sane
    /// `dormant_threshold` — so it is skipped from ambient injection right away,
    /// without waiting out disuse-decay (#1239). The exact inverse of
    /// [`reset_chunk`](Self::reset_chunk), and equally reversible: waking it or
    /// a search recall reinforces the weight back up. Never deletes anything and
    /// never touches Markdown; the chunk stays fully searchable.
    ///
    /// Honest about pins rather than clever: this writes weight `0.0` even for a
    /// pinned chunk, but a pin overrides weight at read time
    /// ([`chunk_stats_snapshot`](Self::chunk_stats_snapshot)), so the chunk keeps
    /// reading `1.0` and never-dormant until it is unpinned. Callers that want
    /// sleeping to be visible must unpin first — the UI disables the control.
    ///
    /// Dormancy is also gated on `decay.enabled`: with decay off nothing is ever
    /// dormant, so this records the weight but changes no behaviour.
    ///
    /// The default is a no-op so backends without a stats table need no change.
    fn sleep_chunk(&self, _key: &str) -> Result<()> {
        Ok(())
    }
    /// Set a chunk's `pinned` flag, creating the row if absent. A pinned chunk
    /// reads effective weight `1.0` (decay skipped) and is never dormant (#293).
    /// The default is a no-op so backends without a stats table need no change.
    fn set_chunk_pinned(&self, _key: &str, _pinned: bool) -> Result<()> {
        Ok(())
    }
    /// Reinforce chunks a *successful* session/goal touched (#1292 block A,
    /// outcome-gated reinforcement). A success verdict promotes the goal-relevant
    /// chunks the session leaned on — the same weight bump a recall gives — and
    /// clears any `suppress_promotion` flag, so a chunk buried by an earlier
    /// failure is rehabilitated the moment a later success reuses it. Keyed by
    /// [`chunk_key`], like [`reinforce_ambient`](Self::reinforce_ambient).
    ///
    /// Upserts: a chunk written *this* run has no stats row yet (a write only
    /// populates `chunks`), and those freshly-written keys are exactly what a
    /// goal's touch log collects, so the row is created at the recall baseline
    /// and takes the gain rather than the signal being silently dropped.
    ///
    /// The default is a no-op so backends without a stats table need no change.
    fn reinforce_outcome(&self, _keys: &[String]) -> Result<()> {
        Ok(())
    }
    /// Set a chunk's `suppress_promotion` flag, creating the row if absent
    /// (#1292 block A). A suppressed chunk is scored `0.0` by
    /// [`ChunkStatsSalience`](crate::consolidate::ChunkStatsSalience), so the
    /// consolidation pass never promotes it out of Daily into Curated. Used to
    /// gate promotion on a *failed* session/goal outcome without deleting the
    /// chunk (a later success clears the flag via
    /// [`reinforce_outcome`](Self::reinforce_outcome)).
    ///
    /// The default is a no-op so backends without a stats table need no change.
    fn set_suppress_promotion(&self, _key: &str, _suppress: bool) -> Result<()> {
        Ok(())
    }
    /// Record a directed edge between two chunks in the `chunk_links` graph layer
    /// (memory sifting, #1293). `kind` names the relationship (`"supersession"`
    /// in C1; `"wikilink"`/`"cooccur"` in C2). Idempotent on
    /// `(from_key, to_key, kind)`. Unlike `chunk_stats`, edges describe *events*,
    /// not current content, so they are never GC'd by `reindex`. The default is a
    /// no-op so backends without the table need no change; [`Fts5Index`] overrides it.
    fn record_link(&self, _from_key: &str, _to_key: &str, _kind: &str) -> Result<()> {
        Ok(())
    }
    /// Edges originating at `key` (i.e. `from_key = key`). The default is empty.
    fn links_from(&self, _key: &str) -> Result<Vec<Link>> {
        Ok(Vec::new())
    }
    /// Edges pointing at `key` (i.e. `to_key = key`). The default is empty.
    fn links_to(&self, _key: &str) -> Result<Vec<Link>> {
        Ok(Vec::new())
    }
    /// All currently-indexed curated chunks (`source = 'curated'`), so
    /// supersession detection can match a promotion against the full committed
    /// curated set — not just the batch handed to [`Memory::consolidate`] this
    /// pass (memory sifting C1, #1293). A chunk demoted-then-re-promoted, or one
    /// present in the index but absent from the current in-memory batch, is
    /// still a valid supersession target. The default is empty so backends
    /// without a chunks table need no change; [`Fts5Index`] overrides it.
    fn curated_chunks(&self) -> Result<Vec<MemoryChunk>> {
        Ok(Vec::new())
    }
}

/// FTS5/BM25 index over a SQLite database. `open` on a path persists to disk;
/// [`open_in_memory`](Self::open_in_memory) is for tests.
pub struct Fts5Index {
    conn: Mutex<Connection>,
    decay: DecayConfig,
}

impl Fts5Index {
    /// Open (creating if absent) the index database at `path`, ensuring the schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|source| crate::error::MemoryError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let conn = Connection::open(path)?;
        Self::from_conn(conn)
    }

    /// An ephemeral in-memory index (tests).
    pub fn open_in_memory() -> Result<Self> {
        Self::from_conn(Connection::open_in_memory()?)
    }

    fn from_conn(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                 id         INTEGER PRIMARY KEY,
                 source     TEXT NOT NULL,
                 path       TEXT NOT NULL,
                 heading    TEXT,
                 text       TEXT NOT NULL,
                 line_start INTEGER NOT NULL,
                 line_end   INTEGER NOT NULL,
                 embedding  BLOB
             );
             CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
             CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                 USING fts5(text, content='chunks', content_rowid='id');
             CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                 INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
             END;
             CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                 INSERT INTO chunks_fts(chunks_fts, rowid, text)
                     VALUES('delete', old.id, old.text);
             END;
             CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                 INSERT INTO chunks_fts(chunks_fts, rowid, text)
                     VALUES('delete', old.id, old.text);
                 INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
             END;
             CREATE TABLE IF NOT EXISTS chunk_stats (
                 chunk_key     TEXT PRIMARY KEY,
                 weight        REAL    NOT NULL DEFAULT 1.0,
                 last_accessed INTEGER NOT NULL,
                 access_count  INTEGER NOT NULL DEFAULT 0,
                 pinned        INTEGER NOT NULL DEFAULT 0,
                 suppress_promotion INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS retrieve_stats (
                 content_key TEXT PRIMARY KEY,
                 count       INTEGER NOT NULL DEFAULT 0,
                 last_ms     INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS content_chunk_map (
                 content_key TEXT NOT NULL,
                 chunk_key   TEXT NOT NULL,
                 similarity  REAL NOT NULL DEFAULT 1.0,
                 last_mapped INTEGER NOT NULL,
                 PRIMARY KEY (content_key, chunk_key)
             );
             -- Directed edge graph over chunk keys (memory sifting, #1293). `kind`
             -- is one of three values: 'supersession' (C1: from=older fact,
             -- to=the promotion that replaced it), 'wikilink', or 'cooccur'
             -- (both C2). Kept as free TEXT rather than a CHECK constraint so a
             -- new edge kind needs no migration; writers pass a fixed literal.
             CREATE TABLE IF NOT EXISTS chunk_links (
                 from_key   TEXT NOT NULL,
                 to_key     TEXT NOT NULL,
                 kind       TEXT NOT NULL,
                 created_at INTEGER NOT NULL,
                 PRIMARY KEY (from_key, to_key, kind)
             );
             CREATE INDEX IF NOT EXISTS idx_chunk_links_to ON chunk_links(to_key);",
        )?;
        Self::ensure_embedding_column(&conn)?;
        Self::ensure_pinned_column(&conn)?;
        Self::ensure_suppress_promotion_column(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            decay: DecayConfig::default(),
        })
    }

    /// Attach decay configuration (RFC 0007 §5). Builder so existing call sites
    /// and tests keep the disabled-by-default behaviour.
    pub fn with_decay(mut self, decay: DecayConfig) -> Self {
        self.decay = decay;
        self
    }

    /// Back-fill the `embedding` column on indexes created before M5.3.0 (#196).
    /// The M5.1 schema (#176) had no `embedding` column, and the `CREATE TABLE IF
    /// NOT EXISTS` above is a no-op against that pre-existing table — so an old
    /// on-disk `index.db` would lack the column and every `reindex` insert would
    /// fail, silently freezing recall. The index is a derived cache, but adding
    /// the column in place is cheaper than a full rebuild and keeps the FTS data.
    /// FTS5 only indexes `text`, so the new column is invisible to `chunks_fts`.
    fn ensure_embedding_column(conn: &Connection) -> Result<()> {
        let has_embedding = conn
            .prepare("PRAGMA table_info(chunks)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<String>, _>>()?
            .iter()
            .any(|name| name == "embedding");
        if !has_embedding {
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB", [])?;
        }
        Ok(())
    }

    /// Back-fill the `pinned` column on `chunk_stats` tables created before M6.2
    /// (#293). The M6.0 schema had no `pinned` column, and the `CREATE TABLE IF
    /// NOT EXISTS` above is a no-op against a pre-existing table — so an old
    /// on-disk index would lack the column and every pin write would fail. Mirror
    /// of [`ensure_embedding_column`] for the stats side table.
    fn ensure_pinned_column(conn: &Connection) -> Result<()> {
        let has_pinned = conn
            .prepare("PRAGMA table_info(chunk_stats)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<String>, _>>()?
            .iter()
            .any(|name| name == "pinned");
        if !has_pinned {
            conn.execute(
                "ALTER TABLE chunk_stats ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        Ok(())
    }

    /// Back-fill the `suppress_promotion` column on `chunk_stats` tables created
    /// before outcome-gated reinforcement (#1292 block A). Mirror of
    /// [`ensure_pinned_column`]: the `CREATE TABLE IF NOT EXISTS` above is a no-op
    /// against a pre-existing table, so an old on-disk index would lack the
    /// column and every suppress write would fail. A suppressed chunk is scored
    /// zero by [`ChunkStatsSalience`](crate::consolidate::ChunkStatsSalience) so
    /// promotion (block D) skips it.
    fn ensure_suppress_promotion_column(conn: &Connection) -> Result<()> {
        let has_col = conn
            .prepare("PRAGMA table_info(chunk_stats)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<String>, _>>()?
            .iter()
            .any(|name| name == "suppress_promotion");
        if !has_col {
            conn.execute(
                "ALTER TABLE chunk_stats ADD COLUMN suppress_promotion INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        Ok(())
    }

    fn insert_chunks(conn: &Connection, chunks: &[MemoryChunk]) -> Result<()> {
        let mut stmt = conn.prepare(
            "INSERT INTO chunks (source, path, heading, text, line_start, line_end, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for c in chunks {
            stmt.execute(params![
                source_to_str(&c.source),
                c.path.to_string_lossy(),
                c.heading,
                c.text,
                c.line_start,
                c.line_end,
                c.embedding.as_deref().map(vec_to_blob),
            ])?;
        }
        Ok(())
    }

    /// Stored embeddings for the given chunk ids (NULL/absent entries omitted).
    /// Used by [`HybridIndex`] to fuse vector similarity over a BM25 candidate
    /// set; with the default [`NoopEmbedder`](crate::embed::NoopEmbedder) the
    /// `embedding` column is always NULL so this returns empty.
    fn embeddings_for(&self, ids: &[i64]) -> Result<HashMap<i64, Vec<f32>>> {
        let mut out = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT embedding FROM chunks WHERE id = ?1")?;
        for &id in ids {
            let blob: Option<Vec<u8>> = stmt.query_row(params![id], |row| row.get(0))?;
            if let Some(bytes) = blob {
                out.insert(id, blob_to_vec(&bytes));
            }
        }
        Ok(out)
    }

    /// Reinforce `hits` against `now_ms` (RFC 0007 §2). Split from the trait
    /// method so tests can inject a deterministic clock. For each hit, looks up
    /// the `chunk_stats` row by stable [`chunk_key`]:
    /// - **new key**: insert at `weight = 1.0`, `access_count = 1`.
    /// - **existing, decay enabled**: lazily decay from `last_accessed`, then
    ///   reinforce, bump `access_count`, stamp `now_ms`.
    /// - **existing, decay disabled**: record the access (count + timestamp) but
    ///   leave `weight` untouched -- behaviour is byte-identical to M5.
    pub(crate) fn reinforce_at(&self, hits: &[ScoredChunk], now_ms: i64) -> Result<()> {
        if hits.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut sel = tx.prepare(
                "SELECT weight, last_accessed, access_count
                 FROM chunk_stats WHERE chunk_key = ?1",
            )?;
            let mut up = tx.prepare(
                "INSERT INTO chunk_stats (chunk_key, weight, last_accessed, access_count)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(chunk_key) DO UPDATE SET
                     weight = ?2, last_accessed = ?3, access_count = ?4",
            )?;
            for hit in hits {
                let key = chunk_key(&hit.chunk);
                let existing: Option<(f64, i64, i64)> = sel
                    .query_row(params![key], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    })
                    .optional()?;
                let (weight, count) = match existing {
                    None => (1.0_f32, 1_i64),
                    Some((w, last, c)) => {
                        let w = w as f32;
                        let new_w = if self.decay.enabled {
                            reinforced_weight(
                                decayed_weight(w, last, now_ms, self.decay.factor),
                                self.decay.reinforce_gain,
                            )
                        } else {
                            w
                        };
                        (new_w, c + 1)
                    }
                };
                up.execute(params![key, weight as f64, now_ms, count])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Time-injectable core of [`reinforce_ambient`](MemoryIndex::reinforce_ambient).
    /// Gated: no-op unless decay is enabled and `ambient_gain > 0`. Only updates
    /// keys that already have a `chunk_stats` row (a never-recalled chunk has no
    /// row and is left untouched, preserving RFC §3).
    pub(crate) fn reinforce_ambient_at(&self, keys: &[String], now_ms: i64) -> Result<()> {
        if keys.is_empty() || !self.decay.enabled || self.decay.ambient_gain <= 0.0 {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut sel = tx.prepare(
                "SELECT weight, last_accessed, access_count
                 FROM chunk_stats WHERE chunk_key = ?1",
            )?;
            let mut up = tx.prepare(
                "UPDATE chunk_stats
                 SET weight = ?2, last_accessed = ?3, access_count = ?4
                 WHERE chunk_key = ?1",
            )?;
            for key in keys {
                let existing: Option<(f64, i64, i64)> = sel
                    .query_row(params![key], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    })
                    .optional()?;
                // Never-recalled chunks (no row) are intentionally skipped: ambient
                // injection does not start the age clock (RFC §3).
                if let Some((w, last, c)) = existing {
                    let new_w = reinforced_weight(
                        decayed_weight(w as f32, last, now_ms, self.decay.factor),
                        self.decay.ambient_gain,
                    );
                    up.execute(params![key, new_w as f64, now_ms, c + 1])?;
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Time-injectable core of [`record_retrieve`](MemoryIndex::record_retrieve).
    /// Upserts a per-content-key row in `retrieve_stats`, incrementing the
    /// cross-session count and stamping `last_ms`. Decay-independent: a retrieve
    /// is a factual use event, always recorded (unlike ambient reinforcement,
    /// which is gated on the decay config).
    pub(crate) fn record_retrieve_at(&self, content_key: &str, now_ms: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO retrieve_stats (content_key, count, last_ms)
             VALUES (?1, 1, ?2)
             ON CONFLICT(content_key) DO UPDATE SET count = count + 1, last_ms = ?2",
            params![content_key, now_ms],
        )?;
        Ok(())
    }

    /// Time-injectable core of [`map_retrieve_to_chunks`](MemoryIndex::map_retrieve_to_chunks).
    /// Upserts mapping rows in `content_chunk_map` (B2).
    pub(crate) fn map_retrieve_to_chunks_at(
        &self,
        content_key: &str,
        mappings: &[(String, f32)],
        now_ms: i64,
    ) -> Result<()> {
        if mappings.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "INSERT INTO content_chunk_map (content_key, chunk_key, similarity, last_mapped)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(content_key, chunk_key) DO UPDATE SET
                 similarity = ?3, last_mapped = ?4",
        )?;
        for (chunk_key, similarity) in mappings {
            stmt.execute(params![content_key, chunk_key, *similarity as f64, now_ms])?;
        }
        Ok(())
    }

    /// Time-injectable core of [`reset_chunk`](MemoryIndex::reset_chunk) (#293).
    /// Restores `weight` to `1.0` and stamps `last_accessed = now_ms`, creating
    /// the row if absent (a never-recalled chunk has no row). `access_count` and
    /// `pinned` are preserved on an existing row; a fresh row starts at count `0`,
    /// unpinned.
    pub(crate) fn reset_chunk_at(&self, key: &str, now_ms: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chunk_stats (chunk_key, weight, last_accessed, access_count, pinned)
             VALUES (?1, 1.0, ?2, 0, 0)
             ON CONFLICT(chunk_key) DO UPDATE SET weight = 1.0, last_accessed = ?2",
            params![key, now_ms],
        )?;
        Ok(())
    }

    /// Time-injectable core of [`sleep_chunk`](MemoryIndex::sleep_chunk) (#1239).
    /// Forces `weight` to `0.0`, creating the row if absent. `access_count` and
    /// `pinned` are preserved on an existing row; a fresh row starts at count `0`,
    /// unpinned.
    ///
    /// Unlike [`reset_chunk_at`](Self::reset_chunk_at), an existing row keeps its
    /// stored `last_accessed` — sleeping is a curation act, not an access, so the
    /// Salience panel should keep showing how long it has really been idle. Only
    /// a fresh row takes `now_ms`, and only so a never-recalled chunk does not
    /// fabricate an epoch-0 timestamp (same reasoning as
    /// [`set_chunk_pinned_at`](Self::set_chunk_pinned_at)). The timestamp does not
    /// affect decay either way: `0.0 * factor^days` is `0.0`.
    pub(crate) fn sleep_chunk_at(&self, key: &str, now_ms: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chunk_stats (chunk_key, weight, last_accessed, access_count, pinned)
             VALUES (?1, 0.0, ?2, 0, 0)
             ON CONFLICT(chunk_key) DO UPDATE SET weight = 0.0",
            params![key, now_ms],
        )?;
        Ok(())
    }

    /// Time-injectable core of
    /// [`set_chunk_pinned`](MemoryIndex::set_chunk_pinned) (#293). Sets the
    /// `pinned` flag, creating the row if absent. A new row starts at weight
    /// `1.0` with `last_accessed = now_ms` (so pinning a never-recalled chunk
    /// does not fabricate an epoch-0 timestamp); an existing row keeps its
    /// stored weight/timestamp — the pin overrides at read time anyway.
    pub(crate) fn set_chunk_pinned_at(&self, key: &str, pinned: bool, now_ms: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chunk_stats (chunk_key, weight, last_accessed, access_count, pinned)
             VALUES (?1, 1.0, ?2, 0, ?3)
             ON CONFLICT(chunk_key) DO UPDATE SET pinned = ?3",
            params![key, now_ms, pinned as i64],
        )?;
        Ok(())
    }

    /// Time-injectable core of
    /// [`set_suppress_promotion`](MemoryIndex::set_suppress_promotion). Upserts
    /// the flag, creating a full-weight row if absent (mirrors
    /// [`set_chunk_pinned_at`](Self::set_chunk_pinned_at)).
    pub(crate) fn set_suppress_promotion_at(
        &self,
        key: &str,
        suppress: bool,
        now_ms: i64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chunk_stats (chunk_key, weight, last_accessed, access_count, suppress_promotion)
             VALUES (?1, 1.0, ?2, 0, ?3)
             ON CONFLICT(chunk_key) DO UPDATE SET suppress_promotion = ?3",
            params![key, now_ms, suppress as i64],
        )?;
        Ok(())
    }

    /// Time-injectable core of [`record_link`](MemoryIndex::record_link) (#1293).
    /// Inserts a directed edge, idempotent on `(from_key, to_key, kind)`;
    /// a repeat refreshes `created_at` to the latest occurrence.
    pub(crate) fn record_link_at(
        &self,
        from_key: &str,
        to_key: &str,
        kind: &str,
        now_ms: i64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chunk_links (from_key, to_key, kind, created_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(from_key, to_key, kind) DO UPDATE SET created_at = ?4",
            params![from_key, to_key, kind, now_ms],
        )?;
        Ok(())
    }

    /// Time-injectable core of
    /// [`reinforce_outcome`](MemoryIndex::reinforce_outcome). A success has two
    /// effects, both keyed to keep it symmetric with how suppression is applied
    /// and read:
    ///
    /// 1. **Clears `suppress_promotion` by content identity.** Suppression is
    ///    matched by content key (a fact flagged on one day stays suppressed
    ///    when it recurs under a later daily date), so the lift must use the
    ///    same identity — every flagged row sharing a touched key's content key
    ///    is cleared, not just the exact `chunk_key` (#1304 F1). This is
    ///    unconditional on decay config: an earlier failure's flag must lift
    ///    even when decay is off.
    /// 2. **Reinforces weight on rows that already exist.** The weight model
    ///    caps at the recall baseline (`1.0`), so a freshly written chunk (which
    ///    has no `chunk_stats` row yet) cannot be pushed *above* baseline by a
    ///    success — creating a row would only start a decay clock and leave it
    ///    worse off than an untracked sibling (#1304 F2). For those chunks
    ///    "success" simply means "not suppressed", delivered by effect 1. Only
    ///    already-tracked chunks (previously recalled) take the `reinforce_gain`.
    pub(crate) fn reinforce_outcome_at(&self, keys: &[String], now_ms: i64) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        // Promotion suppression is matched by content identity (a fact flagged
        // on one day stays suppressed under a later daily date), so a success
        // must lift the flag by the same identity — clearing only the exact
        // `chunk_key` would leave an earlier day's failure row buried forever
        // (#1304 F1). Reduce the touched keys to their content keys and clear
        // every flagged row that shares one.
        let touched_content: HashSet<&str> = keys.iter().map(|k| content_key_of(k)).collect();
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let flagged: Vec<String> = {
                let mut sel =
                    tx.prepare("SELECT chunk_key FROM chunk_stats WHERE suppress_promotion != 0")?;
                let rows = sel.query_map([], |row| row.get::<_, String>(0))?;
                rows.filter_map(|r| r.ok())
                    .filter(|ck| touched_content.contains(content_key_of(ck)))
                    .collect()
            };
            {
                let mut clr = tx.prepare(
                    "UPDATE chunk_stats SET suppress_promotion = 0 WHERE chunk_key = ?1",
                )?;
                for ck in &flagged {
                    clr.execute(params![ck])?;
                }
            }

            // Weight reinforcement applies only to rows that already exist. A
            // freshly written chunk has no `chunk_stats` row, and the weight
            // model caps at the recall baseline (`1.0`), so creating a row here
            // could not raise the weight — it would only start a decay clock and
            // leave a written-then-succeeded chunk *worse off* than an untracked
            // sibling (#1304 F2). For such chunks, "success" means "not
            // suppressed", which the clear above already delivers.
            let mut sel = tx.prepare(
                "SELECT weight, last_accessed, access_count
                 FROM chunk_stats WHERE chunk_key = ?1",
            )?;
            let mut upd = tx.prepare(
                "UPDATE chunk_stats
                 SET weight = ?2, last_accessed = ?3, access_count = ?4
                 WHERE chunk_key = ?1",
            )?;
            for key in keys {
                let existing: Option<(f64, i64, i64)> = sel
                    .query_row(params![key], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    })
                    .optional()?;
                let Some((w, last, c)) = existing else {
                    continue;
                };
                let (raw_w, count) = (w as f32, c + 1);
                // With decay off the stored weight is frozen; with decay on we
                // lazily decay it to `now_ms` before applying the success gain.
                let new_w = if self.decay.enabled {
                    let decayed = decayed_weight(raw_w, last, now_ms, self.decay.factor);
                    reinforced_weight(decayed, self.decay.reinforce_gain)
                } else {
                    raw_w
                };
                upd.execute(params![key, new_w as f64, now_ms, count])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Edges whose `from_key` is `key` (outgoing). SQL is a compile-time literal
    /// with the column fixed — no identifier is interpolated from a caller.
    fn query_links_from(&self, key: &str) -> Result<Vec<Link>> {
        self.query_links(
            "SELECT from_key, to_key, kind, created_at FROM chunk_links
             WHERE from_key = ?1 ORDER BY created_at DESC, kind, from_key, to_key",
            key,
        )
    }

    /// Edges whose `to_key` is `key` (incoming). SQL is a compile-time literal
    /// with the column fixed — no identifier is interpolated from a caller.
    fn query_links_to(&self, key: &str) -> Result<Vec<Link>> {
        self.query_links(
            "SELECT from_key, to_key, kind, created_at FROM chunk_links
             WHERE to_key = ?1 ORDER BY created_at DESC, kind, from_key, to_key",
            key,
        )
    }

    fn query_links(&self, sql: &str, key: &str) -> Result<Vec<Link>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![key], |row| {
            Ok(Link {
                from_key: row.get(0)?,
                to_key: row.get(1)?,
                kind: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// All curated chunks currently in the index (memory sifting C1, #1293).
    /// Backs [`MemoryIndex::curated_chunks`]; `embedding` is left `None` since
    /// supersession detection only needs source + heading + text to key.
    fn curated_chunks_query(&self) -> Result<Vec<MemoryChunk>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source, path, heading, text, line_start, line_end
             FROM chunks WHERE source = 'curated'",
        )?;
        let rows = stmt.query_map([], |row| {
            let source: String = row.get(1)?;
            let path: String = row.get(2)?;
            Ok(MemoryChunk {
                id: row.get(0)?,
                source: source_from_str(&source),
                path: PathBuf::from(path),
                heading: row.get(3)?,
                text: row.get(4)?,
                line_start: row.get(5)?,
                line_end: row.get(6)?,
                embedding: None,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl MemoryIndex for Fts5Index {
    fn reindex(&self, chunks: &[MemoryChunk]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM chunks", [])?;
        Self::insert_chunks(&tx, chunks)?;
        // Orphan sweep (RFC 0007 §4): a full rebuild knows the complete keyset, so
        // drop stats whose chunk no longer exists. Stats for surviving keys are
        // kept untouched -- chunk_key is stable across reindex, so the row re-joins
        // by key with no work. (reindex_path is partial and cannot sweep globally;
        // the next full reindex reconciles.)
        let keys: Vec<String> = chunks.iter().map(chunk_key).collect();
        if keys.is_empty() {
            tx.execute("DELETE FROM chunk_stats", [])?;
            tx.execute("DELETE FROM content_chunk_map", [])?;
        } else {
            tx.execute("CREATE TEMP TABLE valid_keys (k TEXT PRIMARY KEY)", [])?;
            {
                let mut stmt = tx.prepare("INSERT OR IGNORE INTO valid_keys (k) VALUES (?1)")?;
                for k in &keys {
                    stmt.execute(params![k])?;
                }
            }
            tx.execute(
                "DELETE FROM chunk_stats WHERE chunk_key NOT IN (SELECT k FROM valid_keys)",
                [],
            )?;
            tx.execute(
                "DELETE FROM content_chunk_map WHERE chunk_key NOT IN (SELECT k FROM valid_keys)",
                [],
            )?;
            tx.execute("DROP TABLE valid_keys", [])?;
        }
        tx.commit()?;
        Ok(())
    }

    fn reindex_path(&self, path: &Path, chunks: &[MemoryChunk]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM chunks WHERE path = ?1",
            params![path.to_string_lossy()],
        )?;
        Self::insert_chunks(&tx, chunks)?;
        tx.commit()?;
        Ok(())
    }

    fn remove_path(&self, path: &Path) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM chunks WHERE path = ?1",
            params![path.to_string_lossy()],
        )?;
        Ok(())
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<ScoredChunk>> {
        let Some(match_query) = fts_query(query) else {
            return Ok(Vec::new());
        };
        // Scope the connection lock so it is released before `effective_stats`
        // re-locks it below (the Mutex is not reentrant).
        let mut out = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT c.id, c.source, c.path, c.heading, c.text, c.line_start, c.line_end,
                        bm25(chunks_fts) AS score
                 FROM chunks_fts
                 JOIN chunks c ON c.id = chunks_fts.rowid
                 WHERE chunks_fts MATCH ?1
                 ORDER BY score
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![match_query, k as i64], |row| {
                let source: String = row.get(1)?;
                let path: String = row.get(2)?;
                let bm25: f64 = row.get(7)?;
                Ok(ScoredChunk {
                    chunk: MemoryChunk {
                        id: row.get(0)?,
                        source: source_from_str(&source),
                        path: PathBuf::from(path),
                        heading: row.get(3)?,
                        text: row.get(4)?,
                        line_start: row.get(5)?,
                        line_end: row.get(6)?,
                        embedding: None,
                    },
                    // bm25() is a distance (lower = better, typically negative); negate
                    // so a larger score means more relevant.
                    score: -bm25 as f32,
                    // Annotated below from chunk_stats; defaults keep a fresh chunk
                    // non-dormant.
                    weight: 1.0,
                    last_accessed_ms: None,
                })
            })?;
            let mut v = Vec::new();
            for r in rows {
                v.push(r?);
            }
            v
        };

        // Annotate each hit with its read-time effective weight + last access so
        // `memory_search` can tag dormant chunks (RFC 0007 §3). No-op when decay
        // is disabled (`effective_stats` returns empty), keeping output identical
        // to M5.
        if !out.is_empty() {
            let keys: Vec<String> = out.iter().map(|s| chunk_key(&s.chunk)).collect();
            let now_ms = Utc::now().timestamp_millis();
            let stats = self.effective_stats(&keys, now_ms)?;
            for (s, key) in out.iter_mut().zip(&keys) {
                if let Some(es) = stats.get(key) {
                    s.weight = es.weight;
                    s.last_accessed_ms = Some(es.last_accessed_ms);
                }
            }
            // Retrieve-based score boost (B2): bump the score of chunks that
            // are mapped from repeatedly-retrieved content keys.
            let max_score = out.iter().map(|s| s.score).fold(0.0_f32, f32::max);
            if max_score > 0.0 {
                let retrieve_hits = self.chunk_retrieve_hits(&keys)?;
                for (s, key) in out.iter_mut().zip(&keys) {
                    if let Some(&hits) = retrieve_hits.get(key) {
                        s.score += RETRIEVE_FRACTION * max_score * retrieve_boost_factor(hits);
                    }
                }
            }
        }
        Ok(out)
    }

    fn reinforce(&self, hits: &[ScoredChunk]) -> Result<()> {
        self.reinforce_at(hits, Utc::now().timestamp_millis())
    }

    fn reinforce_ambient(&self, keys: &[String]) -> Result<()> {
        self.reinforce_ambient_at(keys, Utc::now().timestamp_millis())
    }

    fn record_retrieve(&self, content_key: &str) -> Result<()> {
        self.record_retrieve_at(content_key, Utc::now().timestamp_millis())
    }

    fn retrieve_count(&self, content_key: &str) -> Result<u64> {
        let conn = self.conn.lock().unwrap();
        let count: Option<i64> = conn
            .query_row(
                "SELECT count FROM retrieve_stats WHERE content_key = ?1",
                params![content_key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(count.unwrap_or(0).max(0) as u64)
    }

    fn map_retrieve_to_chunks(&self, content_key: &str, mappings: &[(String, f32)]) -> Result<()> {
        self.map_retrieve_to_chunks_at(content_key, mappings, Utc::now().timestamp_millis())
    }

    fn chunk_retrieve_hits(&self, chunk_keys: &[String]) -> Result<HashMap<String, u64>> {
        if chunk_keys.is_empty() {
            return Ok(HashMap::new());
        }
        let conn = self.conn.lock().unwrap();
        let placeholders: Vec<String> = (0..chunk_keys.len())
            .map(|i| format!("?{}", i + 1))
            .collect();
        let sql = format!(
            "SELECT cm.chunk_key, SUM(rs.count)
             FROM content_chunk_map cm
             JOIN retrieve_stats rs USING (content_key)
             WHERE cm.chunk_key IN ({})
             GROUP BY cm.chunk_key",
            placeholders.join(", ")
        );
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::types::ToSql> = chunk_keys
            .iter()
            .map(|k| k as &dyn rusqlite::types::ToSql)
            .collect();
        let mut out = HashMap::new();
        let rows = stmt.query_map(params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (key, count) = row?;
            out.insert(key, count.max(0) as u64);
        }
        Ok(out)
    }

    fn effective_stats(
        &self,
        keys: &[String],
        now_ms: i64,
    ) -> Result<HashMap<String, EffectiveStat>> {
        let mut out = HashMap::new();
        if !self.decay.enabled || keys.is_empty() {
            return Ok(out);
        }
        let conn = self.conn.lock().unwrap();
        let mut sel = conn.prepare(
            "SELECT weight, last_accessed, pinned FROM chunk_stats WHERE chunk_key = ?1",
        )?;
        for key in keys {
            let row: Option<(f64, i64, i64)> = sel
                .query_row(params![key], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .optional()?;
            if let Some((w, last, pinned)) = row {
                // A pinned chunk holds weight 1.0 (decay skipped), so the
                // ambient-injection skip path (`curated_filter`) keeps it live —
                // pinned facts never decay (RFC 0007 §7).
                let weight = if pinned != 0 {
                    1.0
                } else {
                    decayed_weight(w as f32, last, now_ms, self.decay.factor)
                };
                out.insert(
                    key.clone(),
                    EffectiveStat {
                        weight,
                        last_accessed_ms: last,
                    },
                );
            }
        }
        Ok(out)
    }

    fn chunk_stats_snapshot(
        &self,
        keys: &[String],
        now_ms: i64,
    ) -> Result<HashMap<String, ChunkStatSnapshot>> {
        let mut out = HashMap::new();
        if keys.is_empty() {
            return Ok(out);
        }
        let threshold = self.decay.dormant_threshold;
        let conn = self.conn.lock().unwrap();
        let mut sel = conn.prepare(
            "SELECT weight, last_accessed, access_count, pinned
             FROM chunk_stats WHERE chunk_key = ?1",
        )?;
        for key in keys {
            let row: Option<(f64, i64, i64, i64)> = sel
                .query_row(params![key], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })
                .optional()?;
            if let Some((w, last, count, pinned)) = row {
                let pinned = pinned != 0;
                // Effective weight is pin-aware, and decay applies only when the
                // flag is on — mirrors `effective_stats`, but emitted regardless
                // of the decay flag so the Salience panel is not all-or-nothing.
                let weight = if pinned {
                    1.0
                } else if self.decay.enabled {
                    decayed_weight(w as f32, last, now_ms, self.decay.factor)
                } else {
                    w as f32
                };
                let dormant = self.decay.enabled && !pinned && weight < threshold;
                out.insert(
                    key.clone(),
                    ChunkStatSnapshot {
                        weight,
                        last_accessed_ms: last,
                        access_count: count.max(0) as u32,
                        pinned,
                        dormant,
                    },
                );
            }
        }
        Ok(out)
    }

    fn suppressed_promotion_keys(&self) -> Result<HashSet<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT chunk_key FROM chunk_stats WHERE suppress_promotion != 0")?;
        let keys = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<HashSet<String>, _>>()?;
        Ok(keys)
    }

    fn reset_chunk(&self, key: &str) -> Result<()> {
        self.reset_chunk_at(key, Utc::now().timestamp_millis())
    }

    fn sleep_chunk(&self, key: &str) -> Result<()> {
        self.sleep_chunk_at(key, Utc::now().timestamp_millis())
    }

    fn set_chunk_pinned(&self, key: &str, pinned: bool) -> Result<()> {
        self.set_chunk_pinned_at(key, pinned, Utc::now().timestamp_millis())
    }

    fn reinforce_outcome(&self, keys: &[String]) -> Result<()> {
        self.reinforce_outcome_at(keys, Utc::now().timestamp_millis())
    }

    fn set_suppress_promotion(&self, key: &str, suppress: bool) -> Result<()> {
        self.set_suppress_promotion_at(key, suppress, Utc::now().timestamp_millis())
    }
    fn record_link(&self, from_key: &str, to_key: &str, kind: &str) -> Result<()> {
        self.record_link_at(from_key, to_key, kind, Utc::now().timestamp_millis())
    }
    fn links_from(&self, key: &str) -> Result<Vec<Link>> {
        self.query_links_from(key)
    }
    fn links_to(&self, key: &str) -> Result<Vec<Link>> {
        self.query_links_to(key)
    }
    fn curated_chunks(&self) -> Result<Vec<MemoryChunk>> {
        self.curated_chunks_query()
    }
}

/// Lazy exponential decay (RFC 0007 §2): `weight * factor^days`, where `days` is
/// the fractional idle days since `last_ms`. Path-independent -- one lazy
/// application over N days equals N daily applications (`factor^(a+b) ==
/// factor^a * factor^b`), so no per-day cron is needed.
fn decayed_weight(weight: f32, last_ms: i64, now_ms: i64, factor: f32) -> f32 {
    let days = ((now_ms - last_ms) as f32 / ONE_DAY_MS).max(0.0);
    weight * factor.powf(days)
}

/// Hebbian reinforcement (RFC 0007 §2): bump `weight` toward 1.0, clamped.
fn reinforced_weight(weight: f32, gain: f32) -> f32 {
    (weight + gain * (1.0 - weight)).min(1.0)
}

/// Retrieve-based score boost (B2). Returns an additive term in `[0, 1]`
/// that when multiplied by `max_score * RETRIEVE_FRACTION` gives the boost.
/// Saturates at `√(1.0) = 1.0` when `hits >= SATURATION_HITS`.
fn retrieve_boost_factor(hits: u64) -> f32 {
    (hits as f32 / SATURATION_HITS).sqrt().min(1.0)
}

fn source_to_str(source: &MemorySource) -> String {
    match source {
        MemorySource::Curated => "curated".to_string(),
        MemorySource::Daily { date } => format!("daily:{}", date.format("%Y-%m-%d")),
    }
}

fn source_from_str(s: &str) -> MemorySource {
    match s.strip_prefix("daily:") {
        Some(date) => NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map(|date| MemorySource::Daily { date })
            .unwrap_or(MemorySource::Curated),
        None => MemorySource::Curated,
    }
}

/// Hybrid recall (RFC 0006 §6): FTS5/BM25 fused with vector similarity from an
/// [`Embedder`]. The fusion only engages when the embedder yields a query vector;
/// otherwise `search` returns the inner [`Fts5Index`] result verbatim, so with
/// the default [`NoopEmbedder`](crate::embed::NoopEmbedder) it is byte-identical
/// to a bare FTS5 index — the "never a hard failure" floor.
///
/// Fusion is Reciprocal Rank Fusion (RRF) over the BM25 candidate pool: a chunk's
/// final score sums `1/(C + rank)` from its BM25 rank and its cosine-similarity
/// rank. RRF is scale-independent, so exact ID/symbol hits (BM25's strength) and
/// semantic neighbours (vectors' strength) both surface. M5.3.1 supplies a real
/// local embedder; M5.3.0 only opens this seam.
pub struct HybridIndex<E: Embedder> {
    inner: Fts5Index,
    embedder: E,
}

/// RRF damping constant (the standard default). Larger = flatter rank weighting.
const RRF_C: f32 = 60.0;
/// How many BM25 candidates to over-fetch (relative to `k`) before fusing, so
/// vector re-ranking has room to promote a semantically-close low-BM25 hit.
const FUSION_POOL_FACTOR: usize = 4;

impl<E: Embedder> HybridIndex<E> {
    /// Wrap an [`Fts5Index`] with an [`Embedder`]. The index keeps full BM25
    /// behaviour; the embedder adds optional vector fusion.
    pub fn new(inner: Fts5Index, embedder: E) -> Self {
        Self { inner, embedder }
    }

    /// Attach a chunk embedding (when the embedder produces one) before indexing,
    /// leaving any pre-set embedding untouched. With [`NoopEmbedder`] this is a
    /// clone with every embedding left `None`.
    fn with_embeddings(&self, chunks: &[MemoryChunk]) -> Result<Vec<MemoryChunk>> {
        let mut out = Vec::with_capacity(chunks.len());
        for c in chunks {
            let mut c = c.clone();
            if c.embedding.is_none() {
                c.embedding = self.embedder.embed_chunk(&c.text)?;
            }
            out.push(c);
        }
        Ok(out)
    }
}

impl<E: Embedder> MemoryIndex for HybridIndex<E> {
    fn reindex(&self, chunks: &[MemoryChunk]) -> Result<()> {
        self.inner.reindex(&self.with_embeddings(chunks)?)
    }

    fn reindex_path(&self, path: &Path, chunks: &[MemoryChunk]) -> Result<()> {
        self.inner
            .reindex_path(path, &self.with_embeddings(chunks)?)
    }

    fn remove_path(&self, path: &Path) -> Result<()> {
        self.inner.remove_path(path)
    }

    fn reinforce(&self, hits: &[ScoredChunk]) -> Result<()> {
        self.inner.reinforce(hits)
    }

    fn reinforce_ambient(&self, keys: &[String]) -> Result<()> {
        self.inner.reinforce_ambient(keys)
    }

    fn record_retrieve(&self, content_key: &str) -> Result<()> {
        self.inner.record_retrieve(content_key)
    }

    fn retrieve_count(&self, content_key: &str) -> Result<u64> {
        self.inner.retrieve_count(content_key)
    }

    fn map_retrieve_to_chunks(&self, content_key: &str, mappings: &[(String, f32)]) -> Result<()> {
        self.inner.map_retrieve_to_chunks(content_key, mappings)
    }

    fn chunk_retrieve_hits(&self, chunk_keys: &[String]) -> Result<HashMap<String, u64>> {
        self.inner.chunk_retrieve_hits(chunk_keys)
    }

    fn effective_stats(
        &self,
        keys: &[String],
        now_ms: i64,
    ) -> Result<HashMap<String, EffectiveStat>> {
        self.inner.effective_stats(keys, now_ms)
    }

    fn chunk_stats_snapshot(
        &self,
        keys: &[String],
        now_ms: i64,
    ) -> Result<HashMap<String, ChunkStatSnapshot>> {
        self.inner.chunk_stats_snapshot(keys, now_ms)
    }

    fn suppressed_promotion_keys(&self) -> Result<HashSet<String>> {
        self.inner.suppressed_promotion_keys()
    }

    fn reset_chunk(&self, key: &str) -> Result<()> {
        self.inner.reset_chunk(key)
    }

    fn sleep_chunk(&self, key: &str) -> Result<()> {
        self.inner.sleep_chunk(key)
    }

    fn set_chunk_pinned(&self, key: &str, pinned: bool) -> Result<()> {
        self.inner.set_chunk_pinned(key, pinned)
    }

    fn reinforce_outcome(&self, keys: &[String]) -> Result<()> {
        self.inner.reinforce_outcome(keys)
    }

    fn set_suppress_promotion(&self, key: &str, suppress: bool) -> Result<()> {
        self.inner.set_suppress_promotion(key, suppress)
    }
    fn record_link(&self, from_key: &str, to_key: &str, kind: &str) -> Result<()> {
        self.inner.record_link(from_key, to_key, kind)
    }

    fn links_from(&self, key: &str) -> Result<Vec<Link>> {
        self.inner.links_from(key)
    }

    fn links_to(&self, key: &str) -> Result<Vec<Link>> {
        self.inner.links_to(key)
    }

    fn curated_chunks(&self) -> Result<Vec<MemoryChunk>> {
        self.inner.curated_chunks()
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<ScoredChunk>> {
        // No query vector (embeddings off / unavailable) -> pure BM25, identical
        // to Fts5Index. This is the fallback guarantee and the common path.
        let Some(qvec) = self.embedder.embed_query(query)? else {
            return self.inner.search(query, k);
        };

        // Over-fetch BM25, then fuse. The pool is the recall set; vectors only
        // re-rank within it (full vector recall is a later slice).
        let pool_k = k.saturating_mul(FUSION_POOL_FACTOR).max(k);
        let bm25 = self.inner.search(query, pool_k)?;
        if bm25.is_empty() {
            return Ok(bm25);
        }

        let ids: Vec<i64> = bm25.iter().map(|s| s.chunk.id).collect();
        let embs = self.inner.embeddings_for(&ids)?;

        // Vector rank: candidates with a stored embedding and a positive cosine,
        // ordered by cosine desc. Orthogonal/negative chunks earn no vector
        // credit just for being in the BM25 pool, so a true semantic neighbour
        // is not diluted into a tie with an unrelated high-BM25 hit.
        let mut by_cosine: Vec<(usize, f32)> = bm25
            .iter()
            .enumerate()
            .filter_map(|(i, s)| embs.get(&s.chunk.id).map(|e| (i, cosine(&qvec, e))))
            .filter(|(_, c)| *c > 0.0)
            .collect();
        by_cosine.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut vec_rank = vec![usize::MAX; bm25.len()];
        for (rank, (cand_idx, _)) in by_cosine.iter().enumerate() {
            vec_rank[*cand_idx] = rank;
        }

        // RRF: BM25 rank is the candidate's position (already best-first); the
        // vector term is omitted for candidates with no embedding.
        let mut fused: Vec<(usize, f32)> = (0..bm25.len())
            .map(|i| {
                let bm = 1.0 / (RRF_C + (i as f32 + 1.0));
                let vc = if vec_rank[i] == usize::MAX {
                    0.0
                } else {
                    1.0 / (RRF_C + (vec_rank[i] as f32 + 1.0))
                };
                (i, bm + vc)
            })
            .collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut results: Vec<ScoredChunk> = fused
            .into_iter()
            .take(k)
            .map(|(i, score)| ScoredChunk {
                chunk: bm25[i].chunk.clone(),
                score,
                // Usage stats are populated by the inner BM25 search; carry them
                // through the fusion re-rank unchanged.
                weight: bm25[i].weight,
                last_accessed_ms: bm25[i].last_accessed_ms,
            })
            .collect();

        // Retrieve-based score boost (B2): same proportional boost as Fts5Index.
        let max_score = results.iter().map(|s| s.score).fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            let keys: Vec<String> = results.iter().map(|s| chunk_key(&s.chunk)).collect();
            let retrieve_hits = self.chunk_retrieve_hits(&keys)?;
            for (s, key) in results.iter_mut().zip(&keys) {
                if let Some(&hits) = retrieve_hits.get(key) {
                    s.score += RETRIEVE_FRACTION * max_score * retrieve_boost_factor(hits);
                }
            }
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(results)
    }
}

/// Cosine similarity in `[-1, 1]`; `0.0` for mismatched-length or zero vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Serialize an embedding to a little-endian `f32` BLOB.
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Inverse of [`vec_to_blob`]; trailing partial bytes (never produced by us) are
/// ignored.
fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    // `as_chunks` rather than `chunks_exact(4)`: the width is a constant, so
    // this hands `from_le_bytes` a `&[u8; 4]` directly instead of re-indexing a
    // slice it cannot prove the length of. `clippy::chunks_exact_to_as_chunks`
    // (new in the 1.98 toolchain) fires on the old form.
    let (chunks, _partial) = b.as_chunks::<4>();
    chunks.iter().copied().map(f32::from_le_bytes).collect()
}

/// Turn arbitrary user text into a safe FTS5 MATCH expression: each whitespace
/// token becomes a quoted string (so `:`/`-`/`*` etc. can't trigger a syntax
/// error or an unintended operator), joined by implicit AND. `None` if empty.
fn fts_query(raw: &str) -> Option<String> {
    // Keep only alphanumeric runs as quoted terms (implicit AND). Dropping all
    // punctuation means FTS5 operators (`:` `-` `*` `"` `(`) in arbitrary user
    // text can never form a malformed MATCH or an empty phrase.
    let tokens: Vec<String> = raw
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{t}\""))
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

#[cfg(test)]
mod tests;
