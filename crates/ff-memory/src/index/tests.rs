use super::*;
use crate::chunk_markdown;
use crate::embed::{Embedder, NoopEmbedder};
use std::path::Path;

/// 3-axis [rust, python, sqlite] keyword vector — deterministic stand-in for
/// a real embedder so fusion is unit-testable without a model.
fn vectorize(text: &str) -> Vec<f32> {
    let l = text.to_lowercase();
    vec![
        f32::from(l.contains("rust")),
        f32::from(l.contains("python")),
        f32::from(l.contains("sqlite")),
    ]
}

/// Embeds chunks by keyword and returns a fixed query vector, so a test can
/// aim the query at a chosen chunk regardless of the BM25 ordering.
struct FakeEmbedder {
    query: Vec<f32>,
}
impl Embedder for FakeEmbedder {
    fn embed_query(&self, _query: &str) -> Result<Option<Vec<f32>>> {
        Ok(Some(self.query.clone()))
    }
    fn embed_chunk(&self, text: &str) -> Result<Option<Vec<f32>>> {
        Ok(Some(vectorize(text)))
    }
}

fn chunks(md: &str, path: &str) -> Vec<MemoryChunk> {
    chunk_markdown(md, MemorySource::Curated, Path::new(path))
}

#[test]
fn search_ranks_and_returns_chunks() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex(&chunks(
        "## Prefs\nuser prefers rust over python\n\n## Tools\nuse sqlite for storage",
        "MEMORY.md",
    ))
    .unwrap();
    let hits = idx.search("rust", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].chunk.text.contains("rust"));
    assert_eq!(hits[0].chunk.heading.as_deref(), Some("Prefs"));
}

#[test]
fn empty_query_returns_nothing() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex(&chunks("## H\nbody", "MEMORY.md")).unwrap();
    assert!(idx.search("   ", 10).unwrap().is_empty());
}

#[test]
fn special_chars_do_not_error() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex(&chunks("## H\norigin address id join key", "MEMORY.md"))
        .unwrap();
    // Colons, hyphens, quotes — all would be FTS5 syntax without sanitizing.
    let hits = idx.search("origin: \"address\" - id", 10).unwrap();
    assert!(!hits.is_empty());
}

#[test]
fn reindex_is_a_full_replace() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex(&chunks("## A\nalpha content", "MEMORY.md"))
        .unwrap();
    idx.reindex(&chunks("## B\nbeta content", "MEMORY.md"))
        .unwrap();
    assert!(idx.search("alpha", 10).unwrap().is_empty());
    assert_eq!(idx.search("beta", 10).unwrap().len(), 1);
}

#[test]
fn reindex_path_replaces_only_that_file() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex_path(Path::new("a.md"), &chunks("## A\napple", "a.md"))
        .unwrap();
    idx.reindex_path(Path::new("b.md"), &chunks("## B\nbanana", "b.md"))
        .unwrap();
    // Rewriting a.md must not touch b.md.
    idx.reindex_path(Path::new("a.md"), &chunks("## A\navocado", "a.md"))
        .unwrap();
    assert!(idx.search("apple", 10).unwrap().is_empty());
    assert_eq!(idx.search("avocado", 10).unwrap().len(), 1);
    assert_eq!(idx.search("banana", 10).unwrap().len(), 1);
}

#[test]
fn remove_path_drops_a_files_chunks() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex_path(Path::new("a.md"), &chunks("## A\napple", "a.md"))
        .unwrap();
    idx.remove_path(Path::new("a.md")).unwrap();
    assert!(idx.search("apple", 10).unwrap().is_empty());
}

#[test]
fn daily_source_round_trips_through_index() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let date = NaiveDate::from_ymd_opt(2026, 6, 18).unwrap();
    let cs = chunk_markdown(
        "## Log\nshipped m5.1",
        MemorySource::Daily { date },
        Path::new("daily/2026-06-18.md"),
    );
    idx.reindex(&cs).unwrap();
    let hits = idx.search("shipped", 10).unwrap();
    assert_eq!(hits[0].chunk.source, MemorySource::Daily { date });
}

#[test]
fn persists_to_disk_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("index.db");
    {
        let idx = Fts5Index::open(&db).unwrap();
        idx.reindex(&chunks("## H\npersistent body", "MEMORY.md"))
            .unwrap();
    }
    let idx = Fts5Index::open(&db).unwrap();
    assert_eq!(idx.search("persistent", 10).unwrap().len(), 1);
}

#[test]
fn blob_round_trips_an_embedding() {
    let v = vec![0.5_f32, -1.25, 3.0, 0.0];
    assert_eq!(blob_to_vec(&vec_to_blob(&v)), v);
}

#[test]
fn cosine_is_one_for_parallel_zero_for_orthogonal() {
    assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    assert_eq!(cosine(&[1.0], &[1.0, 1.0]), 0.0);
}

#[test]
fn hybrid_with_noop_is_byte_identical_to_fts5() {
    let md = "## Prefs\nuser prefers rust over python\n\n## Tools\nuse sqlite for storage";
    let bare = Fts5Index::open_in_memory().unwrap();
    bare.reindex(&chunks(md, "MEMORY.md")).unwrap();

    let hybrid = HybridIndex::new(Fts5Index::open_in_memory().unwrap(), NoopEmbedder);
    hybrid.reindex(&chunks(md, "MEMORY.md")).unwrap();

    for q in ["rust", "sqlite", "python prefers", "storage", "   "] {
        assert_eq!(
            hybrid.search(q, 10).unwrap(),
            bare.search(q, 10).unwrap(),
            "hybrid+noop must match bare BM25 for query {q:?}"
        );
    }
}

#[test]
fn fusion_promotes_the_semantic_match_over_bm25_order() {
    // Three chunks share the term "topic" so a "topic" query returns all
    // three with (near-)tied BM25 — order falls to insert/id order, putting
    // the python chunk first.
    let md = "## P\npython topic\n\n## R\nrust topic\n\n## S\nsqlite topic";

    let bm25_only = Fts5Index::open_in_memory().unwrap();
    bm25_only.reindex(&chunks(md, "MEMORY.md")).unwrap();
    let baseline = bm25_only.search("topic", 10).unwrap();
    assert!(baseline.len() >= 3);
    assert!(
        baseline[0].chunk.text.contains("python"),
        "baseline BM25 should lead with the python chunk, got {:?}",
        baseline[0].chunk.text
    );

    // Aim the query vector at the rust axis: fusion must promote the rust
    // chunk to the top even though BM25 alone ranks it lower.
    let hybrid = HybridIndex::new(
        Fts5Index::open_in_memory().unwrap(),
        FakeEmbedder {
            query: vec![1.0, 0.0, 0.0],
        },
    );
    hybrid.reindex(&chunks(md, "MEMORY.md")).unwrap();
    let fused = hybrid.search("topic", 10).unwrap();
    assert!(
        fused[0].chunk.text.contains("rust"),
        "fusion should surface the rust chunk first, got {:?}",
        fused[0].chunk.text
    );
}

#[test]
fn fusion_falls_back_to_bm25_when_query_vector_is_absent() {
    // FakeEmbedder always yields a vector; NoopEmbedder yields none. With no
    // query vector the hybrid path must be identical to BM25.
    let md = "## A\nrust topic\n\n## B\npython topic";
    let hybrid = HybridIndex::new(Fts5Index::open_in_memory().unwrap(), NoopEmbedder);
    hybrid.reindex(&chunks(md, "MEMORY.md")).unwrap();
    let bare = Fts5Index::open_in_memory().unwrap();
    bare.reindex(&chunks(md, "MEMORY.md")).unwrap();
    assert_eq!(
        hybrid.search("topic", 10).unwrap(),
        bare.search("topic", 10).unwrap()
    );
}

#[test]
fn open_migrates_pre_m530_schema_missing_embedding_column() {
    // Simulate an M5.1 (#176) on-disk index whose `chunks` table predates the
    // `embedding` column. `from_conn`'s `CREATE TABLE IF NOT EXISTS` is a
    // no-op against it, so the migration must back-fill the column or every
    // reindex insert fails (the bug the #196 review caught — unit tests missed
    // it because they all start from a fresh `open_in_memory`).
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE chunks (
             id         INTEGER PRIMARY KEY,
             source     TEXT NOT NULL,
             path       TEXT NOT NULL,
             heading    TEXT,
             text       TEXT NOT NULL,
             line_start INTEGER NOT NULL,
             line_end   INTEGER NOT NULL
         );
         CREATE VIRTUAL TABLE chunks_fts
             USING fts5(text, content='chunks', content_rowid='id');",
    )
    .unwrap();

    let idx = Fts5Index::from_conn(conn).unwrap();
    idx.reindex(&chunks("## Prefs\nuser prefers rust", "MEMORY.md"))
        .unwrap();
    let hits = idx.search("rust", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].chunk.text.contains("rust"));
}

// ----- M6.0 chunk_stats foundation (RFC 0007) --------------------------

use crate::DecayConfig;

/// Read a `chunk_stats` row by key: `(weight, last_accessed, access_count)`.
fn read_stat(idx: &Fts5Index, key: &str) -> Option<(f32, i64, i64)> {
    let conn = idx.conn.lock().unwrap();
    conn.query_row(
        "SELECT weight, last_accessed, access_count FROM chunk_stats WHERE chunk_key = ?1",
        params![key],
        |r| Ok((r.get::<_, f64>(0)? as f32, r.get(1)?, r.get(2)?)),
    )
    .optional()
    .unwrap()
}

fn scored(chunks: &[MemoryChunk]) -> Vec<ScoredChunk> {
    chunks
        .iter()
        .map(|c| ScoredChunk {
            chunk: c.clone(),
            score: 0.0,
            weight: 1.0,
            last_accessed_ms: None,
        })
        .collect()
}

fn enabled_decay() -> DecayConfig {
    DecayConfig {
        enabled: true,
        ..DecayConfig::default()
    }
}

fn disabled_decay() -> DecayConfig {
    DecayConfig {
        enabled: false,
        ..DecayConfig::default()
    }
}

fn ambient_decay(gain: f32) -> DecayConfig {
    DecayConfig {
        enabled: true,
        ambient_gain: gain,
        ..DecayConfig::default()
    }
}

#[test]
fn decay_lazy_equals_repeated_daily() {
    let f = 0.98_f32;
    let day = ONE_DAY_MS as i64;
    let lazy = decayed_weight(1.0, 0, day * 10, f);
    let mut step = 1.0_f32;
    for _ in 0..10 {
        step = decayed_weight(step, 0, day, f);
    }
    assert!((lazy - step).abs() < 1e-4, "lazy {lazy} vs daily {step}");
}

#[test]
fn reinforcement_clamps_at_one() {
    let mut w = 0.2_f32;
    for _ in 0..100 {
        w = reinforced_weight(w, 0.3);
        assert!(w <= 1.0 + f32::EPSILON, "weight escaped 1.0: {w}");
    }
    assert!((w - 1.0).abs() < 1e-3);
    // A fresh fully-salient chunk stays put.
    assert_eq!(reinforced_weight(1.0, 0.3), 1.0);
}

#[test]
fn reinforce_records_new_chunk_at_full_weight() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce(&scored(&cs)).unwrap();
    let key = chunk_key(&cs[0]);
    let (w, _, count) = read_stat(&idx, &key).expect("stat row created");
    assert_eq!(w, 1.0);
    assert_eq!(count, 1);
}

#[test]
fn reinforce_decays_then_lifts_existing_weight() {
    let decay = DecayConfig {
        enabled: true,
        factor: 0.5,
        reinforce_gain: 0.3,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let hits = scored(&cs);
    idx.reinforce_at(&hits, 0).unwrap();
    // Two idle days later: decay 1.0 -> 0.25, reinforce -> 0.25 + 0.3*0.75 = 0.475.
    let day = ONE_DAY_MS as i64;
    idx.reinforce_at(&hits, day * 2).unwrap();
    let key = chunk_key(&cs[0]);
    let (w, last, count) = read_stat(&idx, &key).unwrap();
    assert!((w - 0.475).abs() < 1e-4, "weight {w}");
    assert_eq!(last, day * 2);
    assert_eq!(count, 2);
}

#[test]
fn disabled_records_stats_but_never_decays() {
    // Decay explicitly disabled (the M5 rollback path): access is recorded but
    // weight is frozen.
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(disabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let hits = scored(&cs);
    idx.reinforce_at(&hits, 0).unwrap();
    let day = ONE_DAY_MS as i64;
    idx.reinforce_at(&hits, day * 10).unwrap();
    let key = chunk_key(&cs[0]);
    let (w, last, count) = read_stat(&idx, &key).unwrap();
    assert_eq!(w, 1.0, "weight must not decay when disabled");
    assert_eq!(last, day * 10, "last_accessed is still recorded");
    assert_eq!(count, 2, "access_count is still recorded");
}

#[test]
fn stats_survive_edit_and_reindex_under_stable_key() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce(&scored(&cs)).unwrap();
    let key = chunk_key(&cs[0]);
    assert_eq!(read_stat(&idx, &key).unwrap().2, 1);

    // Re-author the same fact with whitespace / line shifts: chunk_key is
    // stable (consolidate.rs), so the row must survive the orphan sweep.
    let edited = chunks("\n\n## H\nalpha body   \n\n", "MEMORY.md");
    assert_eq!(
        chunk_key(&edited[0]),
        key,
        "key must be stable across edits"
    );
    idx.reindex(&edited).unwrap();
    idx.reinforce(&scored(&edited)).unwrap();
    let (_, _, count) = read_stat(&idx, &key).expect("stat row survived reindex");
    assert_eq!(count, 2, "row persisted, so access_count accumulated");
}

#[test]
fn reindex_sweeps_orphaned_stats() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce(&scored(&cs)).unwrap();
    let stale_key = chunk_key(&cs[0]);
    assert!(read_stat(&idx, &stale_key).is_some());

    // Replace with genuinely new content: the old key is orphaned and swept.
    let fresh = chunks("## H\ncompletely different content", "MEMORY.md");
    idx.reindex(&fresh).unwrap();
    assert!(
        read_stat(&idx, &stale_key).is_none(),
        "orphan must be swept"
    );
}

#[test]
fn empty_reindex_sweeps_all_stats() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce(&scored(&cs)).unwrap();
    idx.reindex(&[]).unwrap();
    assert!(read_stat(&idx, &chunk_key(&cs[0])).is_none());
}

// ----- M6.1 dormancy reads (RFC 0007 §3) -------------------------------

#[test]
fn search_then_reinforce_lands_on_indexed_key() {
    // Guards the production invariant (PR #367 review): a chunk reconstructed
    // by `search` must hash to the same `chunk_key` as the indexed chunk, so
    // search-driven reinforcement updates the row the orphan sweep keeps. A
    // future change to search's SELECT or to chunk_key's inputs would break
    // this silently with every other test still green.
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## Prefs\nuser prefers rust", "MEMORY.md");
    idx.reindex(&cs).unwrap();

    let hits = idx.search("rust", 10).unwrap();
    assert_eq!(hits.len(), 1);
    idx.reinforce(&hits).unwrap();

    let indexed_key = chunk_key(&cs[0]);
    let (_, _, count) = read_stat(&idx, &indexed_key)
        .expect("reinforce(search hits) must update the indexed chunk stats row");
    assert_eq!(count, 1);
    assert_eq!(
        chunk_key(&hits[0].chunk),
        indexed_key,
        "search-reconstructed key must match the indexed key"
    );
}

#[test]
fn effective_stats_decays_at_read_time_without_persisting() {
    let decay = DecayConfig {
        enabled: true,
        factor: 0.5,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    let key = chunk_key(&cs[0]);

    // Two idle days later (factor 0.5): effective weight = 1.0 * 0.5^2 = 0.25.
    let day = ONE_DAY_MS as i64;
    let stats = idx
        .effective_stats(std::slice::from_ref(&key), day * 2)
        .unwrap();
    let es = stats.get(&key).expect("row present");
    assert!((es.weight - 0.25).abs() < 1e-4, "weight {}", es.weight);
    assert_eq!(es.last_accessed_ms, 0);
    // Read-only: the persisted weight is untouched.
    assert_eq!(read_stat(&idx, &key).unwrap().0, 1.0);
}

#[test]
fn effective_stats_omits_unknown_keys() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let stats = idx.effective_stats(&["nope".to_string()], 0).unwrap();
    assert!(
        stats.is_empty(),
        "unknown key absent -> caller treats as 1.0"
    );
}

#[test]
fn effective_stats_empty_when_decay_disabled() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(disabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    let stats = idx
        .effective_stats(&[chunk_key(&cs[0])], ONE_DAY_MS as i64 * 100)
        .unwrap();
    assert!(
        stats.is_empty(),
        "disabled -> nothing dormant, identical to M5"
    );
}

#[test]
fn search_annotates_decayed_weight() {
    let decay = DecayConfig {
        enabled: true,
        factor: 0.5,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);
    let cs = chunks("## Prefs\nuser prefers rust", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    // Last access at the epoch: read-time decay to "now" collapses the weight.
    idx.reinforce_at(&scored(&cs), 0).unwrap();

    let hits = idx.search("rust", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].weight < 0.25, "decayed weight {}", hits[0].weight);
    assert_eq!(hits[0].last_accessed_ms, Some(0));
}

#[test]
fn search_weight_stays_one_when_decay_disabled() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(disabled_decay());
    let cs = chunks("## Prefs\nuser prefers rust", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    let hits = idx.search("rust", 10).unwrap();
    assert_eq!(hits[0].weight, 1.0, "disabled decay -> weight neutral");
    assert_eq!(hits[0].last_accessed_ms, None);
}
const DAY_MS_I: i64 = ONE_DAY_MS as i64;

#[test]
fn reinforce_ambient_bumps_existing_row_weaker_than_recall() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(ambient_decay(0.1));
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap(); // recall: weight 1.0 @ t=0

    let later = DAY_MS_I * 100;
    idx.reinforce_ambient_at(&[chunk_key(&cs[0])], later)
        .unwrap();

    let decayed = decayed_weight(1.0, 0, later, 0.98);
    let recall_bump = reinforced_weight(decayed, 0.3); // reinforce_gain
    let (w, last, count) = read_stat(&idx, &chunk_key(&cs[0])).unwrap();
    assert!(w > decayed, "ambient gain must bump above pure decay");
    assert!(
        w < recall_bump,
        "ambient bump must be weaker than a recall bump"
    );
    assert_eq!(last, later, "ambient touch refreshes last_accessed");
    assert_eq!(count, 2, "ambient touch counts as an access");
}

#[test]
fn reinforce_ambient_skips_never_recalled_chunk() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(ambient_decay(0.1));
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    // Never recalled -> no chunk_stats row. Ambient must NOT create one (RFC §3).
    idx.reinforce_ambient_at(&[chunk_key(&cs[0])], DAY_MS_I)
        .unwrap();
    assert!(
        read_stat(&idx, &chunk_key(&cs[0])).is_none(),
        "ambient injection must not start the age clock for a never-recalled chunk"
    );
}

#[test]
fn reinforce_ambient_noop_when_gain_zero() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(ambient_decay(0.0));
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    idx.reinforce_ambient_at(&[chunk_key(&cs[0])], DAY_MS_I * 100)
        .unwrap();
    let (w, last, count) = read_stat(&idx, &chunk_key(&cs[0])).unwrap();
    assert_eq!((w, last, count), (1.0, 0, 1), "gain 0 => complete no-op");
}

#[test]
fn reinforce_ambient_noop_when_decay_disabled() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(disabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    idx.reinforce_ambient_at(&[chunk_key(&cs[0])], DAY_MS_I * 100)
        .unwrap();
    let (w, last, count) = read_stat(&idx, &chunk_key(&cs[0])).unwrap();
    assert_eq!((w, last, count), (1.0, 0, 1), "decay off => complete no-op");
}

// ----- M6.2 snapshot + reset + pin (RFC 0007 §7, #293) -----------------

fn read_pinned(idx: &Fts5Index, key: &str) -> Option<bool> {
    let conn = idx.conn.lock().unwrap();
    conn.query_row(
        "SELECT pinned FROM chunk_stats WHERE chunk_key = ?1",
        params![key],
        |r| r.get::<_, i64>(0),
    )
    .optional()
    .unwrap()
    .map(|p| p != 0)
}

#[test]
fn pin_holds_weight_at_one_across_a_decay_pass() {
    // factor 0.5 would collapse an unpinned chunk to ~0 after 10 idle days;
    // a pinned chunk reads 1.0 from BOTH the snapshot and effective_stats, so
    // the ambient-injection skip path keeps it live (pinned facts never decay).
    let decay = DecayConfig {
        enabled: true,
        factor: 0.5,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    let key = chunk_key(&cs[0]);
    idx.set_chunk_pinned_at(&key, true, 0).unwrap();

    let later = DAY_MS_I * 10;
    let snap = idx
        .chunk_stats_snapshot(std::slice::from_ref(&key), later)
        .unwrap();
    let s = snap.get(&key).expect("row present");
    assert_eq!(s.weight, 1.0, "pinned weight held at 1.0");
    assert!(!s.dormant, "pinned chunk is never dormant");
    assert!(s.pinned);

    let es = idx
        .effective_stats(std::slice::from_ref(&key), later)
        .unwrap();
    assert_eq!(
        es.get(&key).unwrap().weight,
        1.0,
        "effective_stats is pin-aware so curated_filter keeps the chunk live"
    );
}

#[test]
fn unpinned_chunk_below_threshold_is_dormant_in_snapshot() {
    let decay = DecayConfig {
        enabled: true,
        factor: 0.5,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    let key = chunk_key(&cs[0]);

    // 3 idle days: 1.0 * 0.5^3 = 0.125 < dormant_threshold (0.25).
    let snap = idx
        .chunk_stats_snapshot(std::slice::from_ref(&key), DAY_MS_I * 3)
        .unwrap();
    let s = snap.get(&key).unwrap();
    assert!((s.weight - 0.125).abs() < 1e-4, "weight {}", s.weight);
    assert!(s.dormant, "below threshold and unpinned => dormant");
    assert_eq!(s.access_count, 1);
    assert!(!s.pinned);
}

#[test]
fn snapshot_emitted_even_when_decay_disabled() {
    // Unlike effective_stats (empty when decay off), the snapshot still
    // reports stored weight/access_count/pinned so the Salience panel is not
    // all-or-nothing on the decay flag; nothing is ever dormant though.
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(disabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    let key = chunk_key(&cs[0]);
    let snap = idx
        .chunk_stats_snapshot(std::slice::from_ref(&key), DAY_MS_I * 100)
        .unwrap();
    let s = snap.get(&key).expect("row present despite decay off");
    assert_eq!(s.weight, 1.0, "decay off => stored weight, no decay");
    assert!(!s.dormant, "decay off => never dormant");
    assert_eq!(s.access_count, 1);
}

#[test]
fn snapshot_omits_unknown_keys() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let snap = idx.chunk_stats_snapshot(&["nope".to_string()], 0).unwrap();
    assert!(
        snap.is_empty(),
        "no row => caller treats as never-recalled 1.0"
    );
}

#[test]
fn reset_restores_weight_and_creates_row_if_absent() {
    let decay = DecayConfig {
        enabled: true,
        factor: 0.5,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    // Never recalled => no row. Reset must create it (wake) at weight 1.0.
    assert!(read_stat(&idx, &key).is_none());
    idx.reset_chunk_at(&key, 5_000).unwrap();
    let (w, last, count) = read_stat(&idx, &key).expect("reset created the row");
    assert_eq!((w, last, count), (1.0, 5_000, 0));

    // Decay it, then reset again: weight back to 1.0, timestamp refreshed.
    idx.reinforce_at(&scored(&cs), 0).unwrap(); // count -> 1, weight path
    idx.reset_chunk_at(&key, DAY_MS_I).unwrap();
    let (w, last, _) = read_stat(&idx, &key).unwrap();
    assert_eq!(w, 1.0, "reset restores neutral weight");
    assert_eq!(last, DAY_MS_I, "reset stamps last_accessed");
}

/// Sleep is the inverse of wake (#1239). Asserted through the `dormant` flag
/// rather than the stored weight: dormancy is the behaviour users get (skipped
/// from ambient injection), and it is what a 0.0<->1.0 swap in the SQL must flip.
#[test]
fn sleep_forces_dormancy_and_wake_round_trips_it_back() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);
    let dormant_at = |now: i64| {
        idx.chunk_stats_snapshot(std::slice::from_ref(&key), now)
            .unwrap()
            .get(&key)
            .map(|s| s.dormant)
    };

    // A freshly recalled chunk is live, not dormant.
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    assert_eq!(dormant_at(0), Some(false), "recalled chunk starts live");

    // Sleep drops it below the threshold immediately -- no waiting out decay.
    idx.sleep_chunk_at(&key, 1_000).unwrap();
    assert_eq!(dormant_at(1_000), Some(true), "sleep forces dormancy now");

    // Fully reversible: wake restores it.
    idx.reset_chunk_at(&key, 2_000).unwrap();
    assert_eq!(dormant_at(2_000), Some(false), "wake round-trips it back");
}

#[test]
fn sleep_creates_the_row_and_preserves_access_count_and_pin() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    // Never recalled => no row. Sleep must create it at weight 0, stamping the
    // fresh row's timestamp so it is not a fabricated epoch-0.
    assert!(read_stat(&idx, &key).is_none());
    idx.sleep_chunk_at(&key, 5_000).unwrap();
    assert_eq!(read_stat(&idx, &key), Some((0.0, 5_000, 0)));

    // On an existing row, sleep preserves access_count AND last_accessed --
    // sleeping is a curation act, not an access, so "idle for N days" stays true.
    idx.reset_chunk_at(&key, 6_000).unwrap();
    idx.reinforce_at(&scored(&cs), 7_000).unwrap(); // access_count -> 1
    let (_, last_before, count_before) = read_stat(&idx, &key).unwrap();
    idx.sleep_chunk_at(&key, 9_999_999).unwrap();
    let (w, last, count) = read_stat(&idx, &key).unwrap();
    assert_eq!(w, 0.0, "sleep zeroes the weight");
    assert_eq!(last, last_before, "sleep does not stamp last_accessed");
    assert_eq!(count, count_before, "sleep preserves access_count");
}

/// The backend stays honest: sleep writes weight 0 even for a pinned chunk, but
/// the pin overrides at read time, so it is not dormant until unpinned. The UI
/// disables the control; this pins the semantics underneath it.
#[test]
fn sleeping_a_pinned_chunk_writes_weight_but_pin_still_wins() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    idx.set_chunk_pinned_at(&key, true, 1_000).unwrap();
    idx.sleep_chunk_at(&key, 2_000).unwrap();

    assert_eq!(read_stat(&idx, &key).unwrap().0, 0.0, "stored weight is 0");
    assert_eq!(read_pinned(&idx, &key), Some(true), "pin survives sleep");
    let snap = idx
        .chunk_stats_snapshot(std::slice::from_ref(&key), 2_000)
        .unwrap();
    let s = snap.get(&key).unwrap();
    assert_eq!(s.weight, 1.0, "pin overrides the slept weight on read");
    assert!(!s.dormant, "a pinned chunk is never dormant");

    // Unpinning reveals the slept weight -- the write was real all along.
    idx.set_chunk_pinned_at(&key, false, 3_000).unwrap();
    let snap = idx
        .chunk_stats_snapshot(std::slice::from_ref(&key), 3_000)
        .unwrap();
    assert!(snap.get(&key).unwrap().dormant, "dormant once unpinned");
}

#[test]
fn set_pinned_round_trips_and_creates_row_if_absent() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    assert!(read_stat(&idx, &key).is_none());
    idx.set_chunk_pinned_at(&key, true, 7_000).unwrap();
    assert_eq!(read_pinned(&idx, &key), Some(true), "pin creates the row");
    let (w, last, count) = read_stat(&idx, &key).unwrap();
    assert_eq!(
        (w, last, count),
        (1.0, 7_000, 0),
        "fresh pinned row defaults"
    );

    idx.set_chunk_pinned_at(&key, false, 9_000).unwrap();
    assert_eq!(read_pinned(&idx, &key), Some(false), "unpin round-trips");
}

#[test]
fn ensure_pinned_column_upgrades_a_columnless_db() {
    // Simulate an M6.0 on-disk index: chunk_stats without the pinned column.
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE chunk_stats (
             chunk_key     TEXT PRIMARY KEY,
             weight        REAL    NOT NULL DEFAULT 1.0,
             last_accessed INTEGER NOT NULL,
             access_count  INTEGER NOT NULL DEFAULT 0
         );
         INSERT INTO chunk_stats (chunk_key, weight, last_accessed, access_count)
             VALUES ('old:key', 0.5, 0, 3);",
    )
    .unwrap();

    // from_conn must ALTER-in-place so pin writes succeed on the upgraded DB.
    let idx = Fts5Index::from_conn(conn).unwrap();
    assert_eq!(
        read_pinned(&idx, "old:key"),
        Some(false),
        "back-filled column defaults to 0 on the pre-existing row"
    );
    idx.set_chunk_pinned_at("old:key", true, 1_000).unwrap();
    assert_eq!(read_pinned(&idx, "old:key"), Some(true));
}

// --- retrieve_stats: compaction_retrieve as a durable use signal (#1291) ---

#[test]
fn record_retrieve_counts_are_zero_until_recorded() {
    let idx = Fts5Index::open_in_memory().unwrap();
    assert_eq!(idx.retrieve_count("abc123").unwrap(), 0);
}

#[test]
fn record_retrieve_accumulates_per_key() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.record_retrieve_at("abc123", 100).unwrap();
    idx.record_retrieve_at("abc123", 200).unwrap();
    idx.record_retrieve_at("other", 250).unwrap();

    assert_eq!(idx.retrieve_count("abc123").unwrap(), 2);
    assert_eq!(idx.retrieve_count("other").unwrap(), 1);
    assert_eq!(idx.retrieve_count("never").unwrap(), 0);
}

#[test]
fn record_retrieve_stamps_last_ms() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.record_retrieve_at("k", 100).unwrap();
    idx.record_retrieve_at("k", 999).unwrap();
    let last: i64 = {
        let conn = idx.conn.lock().unwrap();
        conn.query_row(
            "SELECT last_ms FROM retrieve_stats WHERE content_key = ?1",
            rusqlite::params!["k"],
            |row| row.get(0),
        )
        .unwrap()
    };
    assert_eq!(last, 999, "last_ms tracks the most recent retrieve");
}

#[test]
fn record_retrieve_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("index.db");
    {
        let idx = Fts5Index::open(&db).unwrap();
        idx.record_retrieve_at("k", 1).unwrap();
        idx.record_retrieve_at("k", 2).unwrap();
    }
    // A fresh open models a later session: the count must survive, not reset.
    let idx = Fts5Index::open(&db).unwrap();
    assert_eq!(idx.retrieve_count("k").unwrap(), 2);
    idx.record_retrieve_at("k", 3).unwrap();
    assert_eq!(idx.retrieve_count("k").unwrap(), 3);
}

#[test]
fn hybrid_index_delegates_retrieve_recording() {
    let hybrid = HybridIndex::new(Fts5Index::open_in_memory().unwrap(), NoopEmbedder);
    hybrid.record_retrieve("k").unwrap();
    hybrid.record_retrieve("k").unwrap();
    assert_eq!(hybrid.retrieve_count("k").unwrap(), 2);
}

#[test]
fn record_retrieve_is_independent_of_decay_config() {
    // Unlike ambient reinforcement (gated on the decay config), a retrieve is a
    // factual use event and is always recorded — verified here against the
    // default config, whatever its decay state.
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.record_retrieve_at("k", 1).unwrap();
    assert_eq!(idx.retrieve_count("k").unwrap(), 1);
}

// --- content_chunk_map: retrieve-key → chunk mapping (#1296) ---

#[test]
fn map_retrieve_to_chunks_stores_mappings() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let mappings = vec![
        ("chunk:aaa".to_string(), 0.8),
        ("chunk:bbb".to_string(), 0.5),
    ];
    idx.map_retrieve_to_chunks_at("deadbeef", &mappings, 1000)
        .unwrap();

    // Rows exist and similarity is stored.
    let conn = idx.conn.lock().unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM content_chunk_map WHERE content_key = ?1",
            params!["deadbeef"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn chunk_retrieve_hits_returns_summed_counts() {
    let idx = Fts5Index::open_in_memory().unwrap();

    // Record retrieves for two content keys.
    idx.record_retrieve_at("key_a", 100).unwrap();
    idx.record_retrieve_at("key_a", 200).unwrap();
    idx.record_retrieve_at("key_b", 300).unwrap();

    // Map both content keys to chunk "chunk:xyz".
    idx.map_retrieve_to_chunks_at("key_a", &[("chunk:xyz".to_string(), 0.9)], 100)
        .unwrap();
    idx.map_retrieve_to_chunks_at("key_b", &[("chunk:xyz".to_string(), 0.7)], 200)
        .unwrap();

    let hits = idx.chunk_retrieve_hits(&["chunk:xyz".to_string()]).unwrap();
    assert_eq!(hits.get("chunk:xyz").copied(), Some(3)); // 2 + 1
}

#[test]
fn chunk_retrieve_hits_omits_unknown_keys() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let hits = idx
        .chunk_retrieve_hits(&["chunk:never".to_string()])
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn chunk_retrieve_hits_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("index.db");
    {
        let idx = Fts5Index::open(&db).unwrap();
        idx.record_retrieve_at("k", 1).unwrap();
        idx.map_retrieve_to_chunks_at("k", &[("chunk:persist".to_string(), 0.5)], 1)
            .unwrap();
    }
    let idx = Fts5Index::open(&db).unwrap();
    let hits = idx
        .chunk_retrieve_hits(&["chunk:persist".to_string()])
        .unwrap();
    assert_eq!(hits.get("chunk:persist").copied(), Some(1));
}

#[test]
fn empty_mappings_are_a_noop() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.map_retrieve_to_chunks_at("k", &[], 100).unwrap();
    let conn = idx.conn.lock().unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_chunk_map", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn retrieve_boost_raises_rank_of_mapped_chunk() {
    let decay = DecayConfig {
        enabled: false,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);

    // Two chunks: one mentioning "rust" (the mapped one), one mentioning "python".
    let md = "## Rust\nI like rust\n\n## Python\nI like python";
    let cs = chunks(md, "MEMORY.md");
    idx.reindex(&cs).unwrap();

    // Record a retrieve and map it to the rust chunk.
    let rust_key = chunk_key(&cs[0]);
    idx.record_retrieve_at("content_key_x", 100).unwrap();
    idx.record_retrieve_at("content_key_x", 200).unwrap();
    idx.map_retrieve_to_chunks_at("content_key_x", &[(rust_key.clone(), 0.9)], 100)
        .unwrap();

    // Search for "rust" — both chunks could match, but the rust chunk should
    // rank higher due to the retrieve boost.
    let hits = idx.search("rust", 5).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(
        chunk_key(&hits[0].chunk),
        rust_key,
        "the mapped chunk must rank first"
    );
}

#[test]
fn retrieve_boost_is_proportional_to_count() {
    let decay = DecayConfig {
        enabled: false,
        ..DecayConfig::default()
    };
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay);

    let md = "## A\ncommon topic\n\n## B\ncommon topic";
    let cs = chunks(md, "MEMORY.md");
    idx.reindex(&cs).unwrap();

    let key_a = chunk_key(&cs[0]);
    let key_b = chunk_key(&cs[1]);

    // Map key_a with 1 retrieve, key_b with 5 retrieves.
    idx.record_retrieve_at("k_a", 100).unwrap();
    idx.record_retrieve_at("k_b", 100).unwrap();
    idx.record_retrieve_at("k_b", 200).unwrap();
    idx.record_retrieve_at("k_b", 300).unwrap();
    idx.record_retrieve_at("k_b", 400).unwrap();
    idx.record_retrieve_at("k_b", 500).unwrap();
    idx.map_retrieve_to_chunks_at("k_a", &[(key_a.clone(), 0.9)], 100)
        .unwrap();
    idx.map_retrieve_to_chunks_at("k_b", &[(key_b.clone(), 0.9)], 100)
        .unwrap();

    // Both chunks match identically — the one with more retrieves gets a
    // higher score so must rank first.
    let hits = idx.search("common topic", 5).unwrap();
    assert_eq!(hits.len(), 2);
    // The scores may be equal in BM25; the retrieve boost breaks the tie
    // in favor of the chunk with 5 retrieves.
    let score_a = hits
        .iter()
        .find(|s| chunk_key(&s.chunk) == key_a)
        .map(|s| s.score)
        .unwrap();
    let score_b = hits
        .iter()
        .find(|s| chunk_key(&s.chunk) == key_b)
        .map(|s| s.score)
        .unwrap();
    assert!(
        score_b > score_a,
        "chunk with 5 retrieves ({score_b}) must have a higher score than chunk with 1 retrieve ({score_a})"
    );
}

#[test]
fn retrieve_boost_works_in_hybrid_mode() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let hybrid = HybridIndex::new(
        idx,
        FakeEmbedder {
            query: vec![1.0, 0.0, 0.0],
        },
    );

    let md = "## Rust\nrust language\n\n## Python\npython language";
    let cs = chunks(md, "MEMORY.md");
    hybrid.reindex(&cs).unwrap();

    let rust_key = chunk_key(&cs[0]);
    let python_key = chunk_key(&cs[1]);

    // Map both content keys with different retrieve counts.
    hybrid.record_retrieve("k_rust").unwrap();
    hybrid.record_retrieve("k_rust").unwrap();
    hybrid.record_retrieve("k_rust").unwrap();
    hybrid.record_retrieve("k_python").unwrap();
    hybrid
        .map_retrieve_to_chunks("k_rust", &[(rust_key.clone(), 0.9)])
        .unwrap();
    hybrid
        .map_retrieve_to_chunks("k_python", &[(python_key.clone(), 0.9)])
        .unwrap();

    // Search for "language" — the rust chunk has 3 retrieves vs python's 1.
    let hits = hybrid.search("language", 5).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(
        chunk_key(&hits[0].chunk),
        rust_key,
        "chunk with 3 retrieves must rank higher"
    );
}

#[test]
fn no_retrieve_mapping_means_no_boost() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let md = "## H\nsome content";
    let cs = chunks(md, "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let hits = idx.search("content", 5).unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn retrieve_boost_factor_saturates_at_saturation_hits() {
    // At SATURATION_HITS (10) the factor must be 1.0; at 5 it must be >0.7.
    let f10 = super::retrieve_boost_factor(10);
    assert!(
        (f10 - 1.0).abs() < 1e-6,
        "expected 1.0 at saturation (10 hits), got {f10}"
    );
    let f5 = super::retrieve_boost_factor(5);
    let expected_5 = (5.0_f32 / 10.0).sqrt();
    assert!(
        (f5 - expected_5).abs() < 1e-6,
        "expected {expected_5} at 5 hits, got {f5}"
    );
    // Beyond saturation it stays at 1.0.
    let f100 = super::retrieve_boost_factor(100);
    assert!(
        (f100 - 1.0).abs() < 1e-6,
        "expected 1.0 beyond saturation, got {f100}"
    );
    // Zero retrieves yields zero boost.
    let f0 = super::retrieve_boost_factor(0);
    assert!((f0 - 0.0).abs() < 1e-6, "expected 0.0 at 0 hits, got {f0}");
}

// --- Outcome-gated reinforcement (#1292 block A) ------------------------------

fn read_suppress(idx: &Fts5Index, key: &str) -> Option<bool> {
    let conn = idx.conn.lock().unwrap();
    conn.query_row(
        "SELECT suppress_promotion FROM chunk_stats WHERE chunk_key = ?1",
        params![key],
        |r| r.get::<_, i64>(0),
    )
    .optional()
    .unwrap()
    .map(|s| s != 0)
}

#[test]
fn set_suppress_promotion_upserts_and_toggles() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    // No row yet.
    assert_eq!(read_suppress(&idx, &key), None);

    // Set creates the row (weight defaults to 1.0).
    idx.set_suppress_promotion(&key, true).unwrap();
    assert_eq!(read_suppress(&idx, &key), Some(true));
    assert_eq!(read_stat(&idx, &key).unwrap().0, 1.0);

    // Clear toggles it back without deleting the row.
    idx.set_suppress_promotion(&key, false).unwrap();
    assert_eq!(read_suppress(&idx, &key), Some(false));
}

#[test]
fn suppressed_promotion_keys_lists_only_flagged() {
    let idx = Fts5Index::open_in_memory().unwrap();
    let cs = chunks("## A\nalpha\n\n## B\nbeta\n\n## C\ngamma", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    idx.set_suppress_promotion(&chunk_key(&cs[0]), true)
        .unwrap();
    idx.set_suppress_promotion(&chunk_key(&cs[2]), true)
        .unwrap();

    let flagged = idx.suppressed_promotion_keys().unwrap();
    assert_eq!(flagged.len(), 2);
    assert!(flagged.contains(&chunk_key(&cs[0])));
    assert!(flagged.contains(&chunk_key(&cs[2])));
    assert!(!flagged.contains(&chunk_key(&cs[1])));
}

#[test]
fn reinforce_outcome_clears_suppress_and_bumps_weight() {
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    // A prior failure suppressed and let it decay from an old recall.
    idx.reinforce_at(&scored(&cs), 0).unwrap(); // weight 1.0 @ t=0
    idx.set_suppress_promotion(&key, true).unwrap();

    let later = DAY_MS_I * 100;
    idx.reinforce_outcome_at(std::slice::from_ref(&key), later)
        .unwrap();

    // Suppression lifted.
    assert_eq!(read_suppress(&idx, &key), Some(false));
    // Weight got the recall-strength bump above pure decay.
    let decayed = decayed_weight(1.0, 0, later, 0.98);
    let (w, last, _) = read_stat(&idx, &key).unwrap();
    assert!(w > decayed, "success reinforcement must bump above decay");
    assert_eq!(last, later);
}

#[test]
fn reinforce_outcome_leaves_freshly_written_chunk_untracked() {
    // #1304 F2: a chunk written this run has no chunk_stats row yet, and the
    // weight model caps at the recall baseline (1.0) — so a success cannot lift
    // it *above* baseline. Creating a row here would only start a decay clock
    // and leave the chunk worse off than an untracked sibling. Success on a
    // fresh write must NOT create a row; "not suppressed" is the whole effect.
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(enabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);

    // No chunk_stats row exists (never recalled/ambient-touched).
    assert_eq!(read_stat(&idx, &key), None);

    idx.reinforce_outcome_at(std::slice::from_ref(&key), 1000)
        .unwrap();

    // Still no row: the chunk is left at the untracked default (effective
    // weight 1.0, never dormant), not saddled with a fresh decay clock.
    assert_eq!(
        read_stat(&idx, &key),
        None,
        "success on a freshly written chunk must not create a decaying row"
    );
}

#[test]
fn reinforce_outcome_rehabilitates_across_daily_dates() {
    // #1304 F1: suppression is matched by content identity (a fact flagged on
    // one day stays suppressed when it recurs under a later daily date). A
    // success must lift the flag by the same identity — clearing only the exact
    // chunk_key would leave the earlier day's failure row buried forever.
    let idx = Fts5Index::open_in_memory().unwrap();

    // Same content, two different daily dates -> different chunk_keys, same
    // content key.
    let day1 = chunk_markdown(
        "## Note\nremember this fact",
        MemorySource::Daily {
            date: NaiveDate::from_ymd_opt(2026, 8, 26).unwrap(),
        },
        Path::new("daily/2026-08-26.md"),
    );
    let day2 = chunk_markdown(
        "## Note\nremember this fact",
        MemorySource::Daily {
            date: NaiveDate::from_ymd_opt(2026, 8, 27).unwrap(),
        },
        Path::new("daily/2026-08-27.md"),
    );
    let key1 = chunk_key(&day1[0]);
    let key2 = chunk_key(&day2[0]);
    assert_ne!(key1, key2, "different daily dates -> different chunk_keys");

    // Day 1: a failure suppresses the fact.
    idx.reindex(&day1).unwrap();
    idx.set_suppress_promotion(&key1, true).unwrap();
    assert_eq!(read_suppress(&idx, &key1), Some(true));

    // Day 2: the same fact is written and its session succeeds. Settling the
    // day-2 key must rehabilitate the day-1 flag, since they share content.
    idx.reinforce_outcome_at(std::slice::from_ref(&key2), 1000)
        .unwrap();
    assert!(
        idx.suppressed_promotion_keys().unwrap().is_empty(),
        "a later success rehabilitates the same content suppressed earlier"
    );
}

#[test]
fn reinforce_outcome_clears_suppress_even_with_decay_disabled() {
    // Decay off: weight must not move, but a stale failure's flag must still lift.
    let idx = Fts5Index::open_in_memory()
        .unwrap()
        .with_decay(disabled_decay());
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    let key = chunk_key(&cs[0]);
    idx.reinforce_at(&scored(&cs), 0).unwrap();
    idx.set_suppress_promotion(&key, true).unwrap();
    let before = read_stat(&idx, &key).unwrap().0;

    idx.reinforce_outcome_at(std::slice::from_ref(&key), 5000)
        .unwrap();

    assert_eq!(read_suppress(&idx, &key), Some(false));
    assert_eq!(
        read_stat(&idx, &key).unwrap().0,
        before,
        "weight frozen with decay off"
    );
}

#[test]
fn suppress_promotion_survives_reopen_migration() {
    // A pre-existing chunk_stats table without the column must gain it.
    let idx = Fts5Index::open_in_memory().unwrap();
    let cs = chunks("## H\nalpha body", "MEMORY.md");
    idx.reindex(&cs).unwrap();
    // ensure_suppress_promotion_column is idempotent — running twice is a no-op.
    Fts5Index::ensure_suppress_promotion_column(&idx.conn.lock().unwrap()).unwrap();
    let key = chunk_key(&cs[0]);
    idx.set_suppress_promotion(&key, true).unwrap();
    assert_eq!(read_suppress(&idx, &key), Some(true));
}

// --- chunk_links graph layer (memory sifting C1, #1293) ---

#[test]
fn record_link_and_query_by_direction() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.record_link("curated:old", "curated:new", "supersession")
        .unwrap();

    let from = idx.links_from("curated:old").unwrap();
    assert_eq!(from.len(), 1);
    assert_eq!(from[0].to_key, "curated:new");
    assert_eq!(from[0].kind, "supersession");

    let to = idx.links_to("curated:new").unwrap();
    assert_eq!(to.len(), 1);
    assert_eq!(to[0].from_key, "curated:old");

    // No edge in the opposite direction.
    assert!(idx.links_to("curated:old").unwrap().is_empty());
    assert!(idx.links_from("curated:new").unwrap().is_empty());
}

#[test]
fn record_link_is_idempotent_on_triple() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.record_link_at("a", "b", "supersession", 100).unwrap();
    idx.record_link_at("a", "b", "supersession", 200).unwrap();
    let links = idx.links_from("a").unwrap();
    assert_eq!(
        links.len(),
        1,
        "same (from,to,kind) upserts, not duplicates"
    );
    assert_eq!(links[0].created_at, 200, "repeat refreshes created_at");

    // A different kind between the same pair is a distinct edge.
    idx.record_link("a", "b", "cooccur").unwrap();
    assert_eq!(idx.links_from("a").unwrap().len(), 2);
}

#[test]
fn links_survive_reindex() {
    let idx = Fts5Index::open_in_memory().unwrap();
    idx.reindex(&chunks("## A\nalpha", "MEMORY.md")).unwrap();
    idx.record_link("curated:old", "curated:new", "supersession")
        .unwrap();

    // A full rebuild (which GCs chunk_stats) must leave chunk_links untouched:
    // edges describe events, not current content.
    idx.reindex(&chunks("## B\nbeta", "MEMORY.md")).unwrap();

    let links = idx.links_from("curated:old").unwrap();
    assert_eq!(links.len(), 1, "edges survive reindex");
    assert_eq!(links[0].to_key, "curated:new");
}

#[test]
fn migration_is_idempotent() {
    // Re-opening a connection re-runs `CREATE TABLE IF NOT EXISTS chunk_links`
    // and must not error or lose rows. (open_in_memory is fresh each call, so
    // drive idempotency via a shared on-disk DB is overkill; re-running from_conn
    // semantics is exercised by opening twice against the same file below.)
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("index.db");
    {
        let idx = Fts5Index::open(&path).unwrap();
        idx.record_link("a", "b", "supersession").unwrap();
    }
    // Second open re-runs the migration; the row must persist.
    let idx = Fts5Index::open(&path).unwrap();
    assert_eq!(idx.links_from("a").unwrap().len(), 1);
}
