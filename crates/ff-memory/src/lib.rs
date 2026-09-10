//! Durable, local-first user memory (RFC 0006, M5).
//!
//! Memory is **plain Markdown the user owns**, under `~/.flowforge/memory/`:
//! a small curated `MEMORY.md` (facts, preferences, decisions) and an
//! append-only `daily/YYYY-MM-DD.md` working log. This crate owns those files,
//! the (later) index, and recall.
//!
//! **Ambient injection** ([`Memory::ambient_block`]) renders a compact,
//! budget-bounded block — curated memory plus today + yesterday's daily log —
//! that the host prepends to the system prompt through the RFC 0001 §4 hook (the
//! same seam RFC 0002 uses for ambient context).
//!
//! **Recall** (M5.1) reaches past the ambient window: an [`Fts5Index`] (SQLite
//! FTS5/BM25, a deletable/rebuildable derived artifact) backs [`Memory::get`] and
//! the search/write helpers the `memory_search` / `memory_get` / `memory_write`
//! tools expose. A debounced [`watch`] reindexes on edit.
//!
//! Reads are **lenient**: a missing file is "nothing recorded yet" (empty), never
//! an error, so callers never need a try/catch around a fresh install.

pub mod consolidate;
mod embed;
mod error;
pub mod flush;
pub mod index;
pub mod outcome;
pub mod watch;

pub use consolidate::{
    chunk_key, ChunkStatsSalience, ConsolidationReport, RecencyFrequencySalience, Salience,
    Supersession,
};
pub use embed::{Embedder, NoopEmbedder, OpenAiEmbedder, INTERACTIVE_EMBED_TIMEOUT};
pub use error::{MemoryError, Result};
pub use flush::{FlushLedger, FlushRecord};
pub use index::{ChunkStatSnapshot, Fts5Index, HybridIndex, Link, MemoryIndex, ScoredChunk};
pub use outcome::{MemoryOutcomeSink, TouchLog, Verdict};

use std::path::{Path, PathBuf};

use chrono::{Local, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Where a chunk came from (RFC 0006 §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemorySource {
    /// The curated `MEMORY.md`.
    Curated,
    /// A dated daily log.
    Daily { date: NaiveDate },
}

/// Where a `memory_write` lands (RFC 0006 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteTarget {
    /// Append to today's daily log — the agent's free, frequent write.
    Daily,
    /// Append to the curated `MEMORY.md` — conservative; only durable facts.
    Curated,
}

/// A durable-memory stratum — a curated `MEMORY.md` section in the Biosphere
/// who/how/what convention (RFC 0008 §3/§4). The headings are a soft-contract:
/// where structured facts go, not a schema the file must obey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stratum {
    /// WHO: role, stable traits, hard preferences.
    Identity,
    /// HOW: conventions, working style, recurring decisions.
    Patterns,
    /// WHAT: current priorities and active work.
    Focus,
}

impl Stratum {
    /// The canonical Markdown heading line for this stratum.
    pub fn heading(self) -> &'static str {
        match self {
            Stratum::Identity => "## Identity",
            Stratum::Patterns => "## Patterns",
            Stratum::Focus => "## Focus",
        }
    }

    /// Every stratum, in canonical render order.
    pub const ALL: [Stratum; 3] = [Stratum::Identity, Stratum::Patterns, Stratum::Focus];

    /// Parse the lowercase tool-facing name (`identity` / `patterns` / `focus`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "identity" => Some(Stratum::Identity),
            "patterns" => Some(Stratum::Patterns),
            "focus" => Some(Stratum::Focus),
            _ => None,
        }
    }

    /// Classify a Markdown line as a stratum heading, case-insensitively.
    ///
    /// **This is the section boundary, and it is deliberately the only one.** A
    /// `## X` that is not one of the three strata is a note's own sub-heading —
    /// `memory_write` encourages those — so it belongs to the current stratum's
    /// body rather than ending the section (#774).
    ///
    /// The read side (the frontend's `parseCategories`, mirrored by
    /// `lib/memory-view.ts`) and the write side ([`replace_under_heading`]) must
    /// agree on this rule or a stratum containing a sub-heading round-trips
    /// lossily: read renders the whole body, write truncates at the sub-heading
    /// and re-appends the tail as a sibling section. Routing both through this
    /// one predicate makes that class of bug unrepresentable.
    pub fn from_heading_line(line: &str) -> Option<Self> {
        let text = line.trim_end();
        let rest = text.strip_prefix("##")?;
        // Exactly the `## <Name>` shape — `###` and `##Name` are not stratum
        // headings, matching the read side's `^##\s+(.+?)\s*$`.
        if !rest.starts_with(char::is_whitespace) {
            return None;
        }
        let name = rest.trim();
        Stratum::ALL
            .into_iter()
            .find(|s| s.heading()[3..].eq_ignore_ascii_case(name))
    }
}

/// Which memory file a [`MemoryFile`] is. Mirrors [`MemorySource`] but is a plain,
/// flat enum for the read-only Settings browse surface (M5.1e).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFileKind {
    /// The curated `MEMORY.md`.
    Curated,
    /// A dated daily log under `daily/`.
    Daily,
}

/// A memory file as the Settings pane sees it (RFC 0006 §8, #131): name, root-
/// relative path, kind, size, and mtime. Read-only; no contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFile {
    /// File name, e.g. `MEMORY.md` or `2026-06-18.md`.
    pub name: String,
    /// Path relative to the memory root, e.g. `MEMORY.md` or `daily/2026-06-18.md`.
    pub rel_path: String,
    pub kind: MemoryFileKind,
    pub size_bytes: u64,
    /// Last-modified time in epoch milliseconds, or 0 if unavailable.
    pub modified_ms: i64,
}

/// One indexed unit of memory, chunked from a Markdown file (RFC 0006 §4).
///
/// M5.0 defines the shape but only chunks for tests; the FTS5 index (M5.1)
/// consumes these. `embedding` is reserved for the optional hybrid backend
/// (M5.3) and is always `None` today.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryChunk {
    pub id: i64,
    pub source: MemorySource,
    pub path: PathBuf,
    /// Nearest Markdown heading, for context. `None` for pre-heading preamble.
    pub heading: Option<String>,
    /// The chunk body, including its heading line (FTS5-indexed in M5.1).
    pub text: String,
    /// 1-based, inclusive line span — lets `memory_get` target a chunk (M5.1).
    pub line_start: u32,
    pub line_end: u32,
    /// Reserved for the hybrid backend (M5.3); always `None` in FTS-only mode.
    pub embedding: Option<Vec<f32>>,
}

/// Memory behaviour knobs. Defaults make memory on with a small ambient budget;
/// `enabled = false` is the full disable path (RFC 0006 §8).
///
/// Note (M5.3.x persistence): `rename_all = "camelCase"` is safe today because
/// this is only ever built via `Default` and never deserialized from disk. When
/// on-disk persistence lands, keep the file camelCase too (or add an explicit
/// migration) so a snake_case settings file isn't silently reintroduced.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct MemoryConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Byte budget for the curated section of the ambient block (a pragmatic
    /// stand-in for a token budget). The curated file is truncated to a
    /// line-boundary head past this, with a pointer to `memory_search`.
    #[serde(default = "default_budget")]
    #[ts(type = "number")]
    pub injection_budget_bytes: usize,
    /// Hybrid (semantic) recall settings (M5.3). Off by default: recall stays
    /// pure FTS5/BM25 until a user opts in and an embedder is wired.
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    /// Whether the consolidation pass may hard-evict (demote) the lowest-salience
    /// curated facts back to the daily log when merge + promote alone cannot bring
    /// the curated file under `injection_budget_bytes` (RFC 0007 sec 6). Demotion
    /// never deletes: the entry is appended to today's daily log (still FTS-indexed,
    /// still found by `memory_search`) and removed from curated. Default on now;
    /// M6.1 flips this off once decay/dormancy governs injection instead.
    #[serde(default = "default_evict_to_budget")]
    pub evict_to_budget: bool,
    /// Usage-driven decay knobs (RFC 0007 §5). Disabled by default in M6.0:
    /// statistics are recorded but never decayed, so behaviour is byte-identical
    /// to M5. M6.1 flips this default on once dormancy consumes `weight`.
    #[serde(default)]
    pub decay: DecayConfig,
}

/// Usage-driven decay configuration (RFC 0007 §5). Each knob has a conservative
/// default; the whole mechanism is gated by `enabled`.
///
/// `weight` only decays/reinforces when `enabled = true`. M6.0 (#291) shipped
/// `enabled = false` as a no-op rollback path; M6.1 (#292) flips the default on
/// and consumes `dormant_threshold` to skip dormant chunks from ambient
/// injection. `ambient_gain` (weak ambient reinforcement, RFC §10.1, #387) is
/// consumed but defaults to `0` — opt-in (see [`default_ambient_gain`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct DecayConfig {
    /// Master switch for decay + reinforcement (RFC 0007 §5). When `false`,
    /// stats are still recorded but `weight` is never modified.
    #[serde(default = "default_decay_enabled")]
    pub enabled: bool,
    /// Daily multiplier applied per idle day: `weight *= factor.powf(days)`
    /// (~35-day half-life at 0.98).
    #[serde(default = "default_decay_factor")]
    pub factor: f32,
    /// Strength of a `memory_search` hit: `weight += gain * (1.0 - weight)`.
    #[serde(default = "default_reinforce_gain")]
    pub reinforce_gain: f32,
    /// Weak reinforcement for a chunk that was ambient-injected and the turn
    /// produced a reply (RFC 0007 §10.1). Defaults to `0` (off) — a nonzero value
    /// keeps still-shown curated chunks fresh; set > 0 to opt in.
    #[serde(default = "default_ambient_gain")]
    pub ambient_gain: f32,
    /// `weight` below this marks a chunk dormant — skipped from ambient
    /// injection while staying recallable (RFC 0007 §M6.1).
    #[serde(default = "default_dormant_threshold")]
    pub dormant_threshold: f32,
}

/// Hybrid recall configuration (RFC 0006, M5.3). When `enabled = false`
/// (the default) recall is byte-identical to the FTS5/BM25 path.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct EmbeddingsConfig {
    /// Opt in to semantic recall. Off keeps the FTS-only behaviour.
    pub enabled: bool,
    /// Which embedder backs semantic recall when `enabled`.
    pub provider: EmbeddingProvider,
}

/// Where chunk embeddings come from (M5.3). `Local` is the default on-device
/// path; `Cloud` is reserved for the M5.3.2 hosted embedder.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub enum EmbeddingProvider {
    /// On-device embedder (default).
    #[default]
    Local,
    /// Hosted embedder (reserved for M5.3.2).
    Cloud,
}

fn default_enabled() -> bool {
    true
}
fn default_budget() -> usize {
    4096
}
fn default_evict_to_budget() -> bool {
    true
}
fn default_decay_enabled() -> bool {
    true
}
fn default_decay_factor() -> f32 {
    0.98
}
fn default_reinforce_gain() -> f32 {
    0.3
}
fn default_ambient_gain() -> f32 {
    // Off by default (RFC 0007 §10 Open Q#1): with dormant-skip, a nonzero gain
    // would refresh every still-shown curated chunk every turn, so curated facts
    // would never go dormant during active use. Opt-in; set > 0 to enable.
    0.0
}
fn default_dormant_threshold() -> f32 {
    0.25
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            injection_budget_bytes: default_budget(),
            embeddings: EmbeddingsConfig::default(),
            evict_to_budget: default_evict_to_budget(),
            decay: DecayConfig::default(),
        }
    }
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            enabled: default_decay_enabled(),
            factor: default_decay_factor(),
            reinforce_gain: default_reinforce_gain(),
            ambient_gain: default_ambient_gain(),
            dormant_threshold: default_dormant_threshold(),
        }
    }
}

/// Owns the memory directory and renders the ambient block.
#[derive(Debug, Clone)]
pub struct Memory {
    root: PathBuf,
    config: MemoryConfig,
}

impl Memory {
    /// Bind to an explicit memory root (used in tests).
    pub fn new(root: impl Into<PathBuf>, config: MemoryConfig) -> Self {
        Self {
            root: root.into(),
            config,
        }
    }

    /// Bind to the default `~/.flowforge/memory/` root.
    pub fn with_default_root(config: MemoryConfig) -> Self {
        Self::new(default_root(), config)
    }

    /// Path to the curated `MEMORY.md`.
    pub fn curated_path(&self) -> PathBuf {
        self.root.join("MEMORY.md")
    }

    /// Path to a given day's log, `daily/YYYY-MM-DD.md`.
    pub fn daily_path(&self, date: NaiveDate) -> PathBuf {
        self.root
            .join("daily")
            .join(format!("{}.md", date.format("%Y-%m-%d")))
    }

    /// The derived FTS5 index database, `~/.flowforge/memory/index.db`.
    pub fn index_path(&self) -> PathBuf {
        self.root.join("index.db")
    }

    /// The memory root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether memory is enabled (RFC 0006 §8). Recall tools no-op when `false`.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Read a memory file, optionally sliced to a 1-based inclusive line range.
    /// Backs `memory_get`: a missing file is empty (never an error), and a `path`
    /// outside the memory root is rejected as empty so the tool can't read
    /// arbitrary files.
    pub fn get(&self, path: &Path, line_start: Option<u32>, line_end: Option<u32>) -> String {
        if !self.within_root(path) {
            return String::new();
        }
        let raw = read_lenient(path);
        match (line_start, line_end) {
            (None, None) => raw,
            _ => {
                let start = line_start.unwrap_or(1).max(1) as usize;
                let end = line_end.unwrap_or(u32::MAX) as usize;
                raw.lines()
                    .enumerate()
                    .filter(|(i, _)| {
                        let n = i + 1;
                        n >= start && n <= end
                    })
                    .map(|(_, l)| l)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
    }

    /// Append `text` to the chosen target, creating the file/dirs if needed.
    /// Returns the file written so the caller can reindex just that path.
    pub fn write(&self, text: &str, target: WriteTarget) -> Result<PathBuf> {
        let path = match target {
            WriteTarget::Daily => self.daily_path(Local::now().date_naive()),
            WriteTarget::Curated => self.curated_path(),
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| MemoryError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let existing = read_lenient(&path);
        let mut body = String::new();
        if !existing.is_empty() && !existing.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(text.trim_end());
        body.push('\n');
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| MemoryError::Io {
                path: path.clone(),
                source,
            })?;
        file.write_all(body.as_bytes())
            .map_err(|source| MemoryError::Io {
                path: path.clone(),
                source,
            })?;
        Ok(path)
    }

    /// Append a durable fact to the curated `MEMORY.md` under a Biosphere
    /// stratum heading (RFC 0008 §4): the text lands at the end of the matching
    /// `## Identity` / `## Patterns` / `## Focus` section, creating that section
    /// if it does not exist yet. Goes through [`rewrite_curated`] because placing
    /// text under a heading is a structured edit, not a blind append.
    pub fn write_curated_stratum(&self, text: &str, stratum: Stratum) -> Result<PathBuf> {
        let existing = read_lenient(&self.curated_path());
        let updated = insert_under_heading(&existing, stratum.heading(), text);
        self.rewrite_curated(&updated)?;
        Ok(self.curated_path())
    }

    /// Replace the **whole body** of a curated stratum with `text` (#969), for the
    /// Settings → Memory editor (#868): the editable view sends back the full section
    /// content, so this overwrites `## Identity` / `## Patterns` / `## Focus` rather
    /// than appending like [`write_curated_stratum`]. Sibling sections are preserved;
    /// a missing heading is created; empty `text` clears the stratum's body but keeps
    /// the heading. Goes through [`rewrite_curated`] (atomic, single-writer).
    pub fn replace_curated_stratum(&self, text: &str, stratum: Stratum) -> Result<PathBuf> {
        let existing = read_lenient(&self.curated_path());
        let updated = replace_under_heading(&existing, stratum, text);
        self.rewrite_curated(&updated)?;
        Ok(self.curated_path())
    }

    /// The memory config.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Atomically rewrite the curated `MEMORY.md` with new content.
    ///
    /// **Invariant**: this is the sole *full-file rewrite* path for curated
    /// Markdown. Only the consolidation pass calls this. The normal capture
    /// path (`Memory::write` with `WriteTarget::Curated`) still appends — that
    /// is expected. Decay/dormancy (M6) never touches Markdown (RFC 0007 §7).
    ///
    /// Uses write-to-temp + atomic rename so a crash mid-write never corrupts
    /// the curated file. The temp file is created in the same directory as
    /// `MEMORY.md` to guarantee same-filesystem rename semantics.
    pub fn rewrite_curated(&self, content: &str) -> Result<()> {
        use std::io::Write as _;
        let curated = self.curated_path();
        if let Some(parent) = curated.parent() {
            std::fs::create_dir_all(parent).map_err(|source| MemoryError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        // Temp file in the same dir — same filesystem guarantees atomic rename.
        let dir = curated
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(dir).map_err(|source| MemoryError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        tmp.write_all(content.as_bytes())
            .map_err(|source| MemoryError::Io {
                path: tmp.path().to_path_buf(),
                source,
            })?;
        tmp.flush().map_err(|source| MemoryError::Io {
            path: tmp.path().to_path_buf(),
            source,
        })?;
        // persist() does an atomic rename on Unix; on Windows it falls back to
        // a non-atomic overwrite, which is acceptable for a desktop app.
        tmp.persist(&curated).map_err(|e| MemoryError::Io {
            path: curated.clone(),
            source: e.error,
        })?;
        Ok(())
    }

    /// Whether the curated file exceeds the injection budget, indicating that
    /// consolidation should run. Includes a 10% hysteresis band to avoid
    /// flip-flopping right at the boundary.
    pub fn needs_consolidation(&self) -> bool {
        let curated = self.curated_path();
        let size = std::fs::metadata(&curated).map(|m| m.len()).unwrap_or(0) as usize;
        // Trigger at 110% of budget (hysteresis)
        size > self.config.injection_budget_bytes + self.config.injection_budget_bytes / 10
    }

    /// Chunk every memory file (curated + all daily logs) — the input to a full
    /// [`MemoryIndex::reindex`].
    pub fn all_chunks(&self) -> Vec<MemoryChunk> {
        let mut out = Vec::new();
        let curated = self.curated_path();
        out.extend(chunk_markdown(
            &read_lenient(&curated),
            MemorySource::Curated,
            &curated,
        ));
        if let Ok(entries) = std::fs::read_dir(self.root.join("daily")) {
            let mut files: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
                .collect();
            files.sort();
            for path in files {
                let source = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
                    .map(|date| MemorySource::Daily { date })
                    .unwrap_or(MemorySource::Curated);
                out.extend(chunk_markdown(&read_lenient(&path), source, &path));
            }
        }
        out
    }

    /// List every memory file for the read-only Settings browse surface (M5.1e,
    /// #131): the curated `MEMORY.md` first, then daily logs newest-first. Skips
    /// the derived `index.db` and any non-Markdown entry. Missing files/dirs yield
    /// an empty list (never an error).
    pub fn list_files(&self) -> Vec<MemoryFile> {
        let mut out = Vec::new();
        let curated = self.curated_path();
        if curated.is_file() {
            if let Some(f) = self.describe_file(&curated, MemoryFileKind::Curated) {
                out.push(f);
            }
        }
        if let Ok(entries) = std::fs::read_dir(self.root.join("daily")) {
            let mut daily: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("md"))
                .collect();
            // Newest-first: file names are `YYYY-MM-DD.md`, so a reverse lexical
            // sort is chronological.
            daily.sort();
            daily.reverse();
            for path in daily {
                if let Some(f) = self.describe_file(&path, MemoryFileKind::Daily) {
                    out.push(f);
                }
            }
        }
        out
    }

    /// Read a memory file by its root-relative path (as handed out by
    /// [`list_files`](Self::list_files)). Returns `None` when the path escapes the
    /// root (reuses the hardened [`within_root`](Self::within_root) from #176);
    /// a missing file within the root reads as `Some(String::new())`.
    pub fn read_file(&self, rel_path: &str) -> Option<String> {
        let path = self.root.join(rel_path);
        if !self.within_root(&path) {
            return None;
        }
        Some(read_lenient(&path))
    }

    /// Build a [`MemoryFile`] from a path on disk; `None` if metadata is
    /// unreadable. `rel_path` strips the root prefix for display.
    fn describe_file(&self, path: &Path, kind: MemoryFileKind) -> Option<MemoryFile> {
        let meta = std::fs::metadata(path).ok()?;
        let modified_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        // Forward slashes so the wire contract is platform-stable; `read_file`
        // joins it back onto the root regardless of OS separator.
        let rel_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        Some(MemoryFile {
            name: path.file_name()?.to_string_lossy().into_owned(),
            rel_path,
            kind,
            size_bytes: meta.len(),
            modified_ms,
        })
    }

    /// Containment check. Paths may not exist yet (so no `canonicalize`); reject
    /// any `..` component outright — legit memory paths are flat (MEMORY.md,
    /// daily/YYYY-MM-DD.md) — so `<root>/../../etc/passwd` can't slip past the
    /// component-wise `starts_with`.
    fn within_root(&self, path: &Path) -> bool {
        use std::path::Component;
        if path.components().any(|c| c == Component::ParentDir) {
            return false;
        }
        path.starts_with(&self.root)
    }

    /// The ambient block to prepend to the system prompt, or `None` when memory
    /// is disabled or nothing has been recorded yet. Curated memory comes first
    /// (budget-bounded), then today + yesterday's daily log.
    pub fn ambient_block(&self) -> Option<String> {
        self.ambient_block_for(Local::now().date_naive())
    }

    /// [`ambient_block`](Self::ambient_block) with an injected "today" — the
    /// testable core (the public method just supplies the host clock).
    pub fn ambient_block_for(&self, today: NaiveDate) -> Option<String> {
        if !self.config.enabled {
            return None;
        }
        let curated = self.curated_section();
        let daily = self.daily_section(today);
        if curated.is_none() && daily.is_none() {
            return None;
        }
        let mut out = String::from("## Memory\n");
        if let Some(c) = curated {
            out.push('\n');
            out.push_str(&c);
            out.push('\n');
        }
        if let Some(d) = daily {
            out.push_str(&d);
        }
        Some(out)
    }

    /// [`ambient_block`](Self::ambient_block) with dormant curated chunks excised
    /// (RFC 0007 §M6.1). A curated chunk whose effective (read-time decayed)
    /// weight has fallen below `decay.dormant_threshold` is dropped from the
    /// ambient block — the SNR / token-budget win — while staying fully
    /// recallable: a `memory_search` hit reinforces it back above threshold (wake
    /// via recall, no "undelete"). Daily logs are never filtered (yesterday/today
    /// are never dormant).
    ///
    /// **Byte-identical when decay is disabled.** [`MemoryIndex::effective_stats`]
    /// returns an empty map when `decay.enabled = false`, so no chunk is ever
    /// dormant and this returns the same bytes as
    /// [`ambient_block`](Self::ambient_block) — the M5 rollback path.
    ///
    /// **Never-recalled stays present.** A chunk with no `chunk_stats` row has
    /// effective weight `1.0` (the age clock starts at first *recall*, not
    /// creation — RFC 0007 §3), so a fresh `MEMORY.md` entry is never skipped.
    /// This is intended and preserved by weak ambient reinforcement (`ambient_gain`,
    /// RFC §10.1): ambient reinforcement only bumps chunks that already have a
    /// `chunk_stats` row, so it never starts the age clock for a never-recalled
    /// chunk — a fresh entry stays at weight `1.0` until its first real recall.
    pub fn ambient_block_filtered(&self, index: &dyn MemoryIndex) -> Option<String> {
        self.ambient_block_filtered_keyed(index).0
    }

    /// [`ambient_block_filtered`](Self::ambient_block_filtered) returning the
    /// block *and* the `chunk_key`s of the curated chunks that were injected (not
    /// dormant). The host reinforces those keys after a successful turn (weak
    /// ambient reinforcement, RFC 0007 §10.1) — daily chunks are excluded (they
    /// are never dormant, so reinforcing them is meaningless).
    pub fn ambient_block_filtered_keyed(
        &self,
        index: &dyn MemoryIndex,
    ) -> (Option<String>, Vec<String>) {
        self.ambient_block_filtered_keyed_for(
            index,
            Local::now().date_naive(),
            Utc::now().timestamp_millis(),
        )
    }

    /// [`ambient_block_filtered`](Self::ambient_block_filtered) with injected
    /// clocks — the testable core (the public method supplies the host clock).
    pub fn ambient_block_filtered_for(
        &self,
        index: &dyn MemoryIndex,
        today: NaiveDate,
        now_ms: i64,
    ) -> Option<String> {
        self.ambient_block_filtered_keyed_for(index, today, now_ms)
            .0
    }

    /// [`ambient_block_filtered_keyed`](Self::ambient_block_filtered_keyed) with
    /// injected clocks — the testable core.
    pub fn ambient_block_filtered_keyed_for(
        &self,
        index: &dyn MemoryIndex,
        today: NaiveDate,
        now_ms: i64,
    ) -> (Option<String>, Vec<String>) {
        if !self.config.enabled {
            return (None, Vec::new());
        }
        let (curated, curated_keys) = self.curated_filter(index, now_ms);
        let daily = self.daily_section(today);
        if curated.is_none() && daily.is_none() {
            return (None, Vec::new());
        }
        let mut out = String::from("## Memory\n");
        if let Some(c) = curated {
            out.push('\n');
            out.push_str(&c);
            out.push('\n');
        }
        if let Some(d) = daily {
            out.push_str(&d);
        }
        (Some(out), curated_keys)
    }

    /// The dormancy-filtered curated section *and* the `chunk_key`s of the curated
    /// chunks that survived (were actually injected) — the latter is the input to
    /// weak ambient reinforcement (RFC 0007 §10.1).
    ///
    /// Dormant chunks' line ranges are excised from the *original* curated text
    /// before truncation (line-range deletion, not chunk-rejoin, so surrounding
    /// text stays verbatim). Chunks are derived exactly as the index sees them
    /// (`chunk_markdown` over the raw file) so `chunk_key` correlation holds. With
    /// no dormant chunks the raw text is passed through unchanged, giving the
    /// byte-identical guarantee.
    /// Build a [`ChunkStatsSalience`] for a consolidation pass from a live index
    /// (M6.3, issue #294). Gathers the pin-aware, lazily-decayed effective
    /// `weight` of every curated chunk at `now_ms` so demotion ranks by real
    /// retrieval usage. Daily (promote-side) chunks need no stats — the salience
    /// scores them by recency × frequency regardless.
    ///
    /// Curated keys with no `chunk_stats` row are simply absent from the map;
    /// [`ChunkStatsSalience`] reads an absent key as full weight `1.0` (RFC 0007
    /// §3: never-recalled ⇒ not dormant).
    pub fn chunk_stats_salience(&self, index: &dyn MemoryIndex, now_ms: i64) -> ChunkStatsSalience {
        // Promotion-suppressed keys gate the Daily (promote) side, so they apply
        // even when curated is empty (nothing to demote, but Daily chunks may
        // still be up for promotion). #1292 block A.
        let suppressed = index.suppressed_promotion_keys().unwrap_or_default();
        let raw = read_lenient(&self.curated_path());
        if raw.trim().is_empty() {
            return ChunkStatsSalience::with_suppressed(
                std::collections::HashMap::new(),
                suppressed,
            );
        }
        let chunks = chunk_markdown(&raw, MemorySource::Curated, &self.curated_path());
        let keys: Vec<String> = chunks.iter().map(chunk_key).collect();
        let stats = index.effective_stats(&keys, now_ms).unwrap_or_default();
        let weights = stats.into_iter().map(|(k, s)| (k, s.weight)).collect();
        ChunkStatsSalience::with_suppressed(weights, suppressed)
    }

    fn curated_filter(
        &self,
        index: &dyn MemoryIndex,
        now_ms: i64,
    ) -> (Option<String>, Vec<String>) {
        let raw = read_lenient(&self.curated_path());
        if raw.trim().is_empty() {
            return (None, Vec::new());
        }
        let curated_path = self.curated_path();
        let chunks = chunk_markdown(&raw, MemorySource::Curated, &curated_path);
        let keys: Vec<String> = chunks.iter().map(chunk_key).collect();
        let stats = index.effective_stats(&keys, now_ms).unwrap_or_default();
        let threshold = self.config.decay.dormant_threshold;

        // A line is removed only if it is covered by a dormant chunk and by no
        // live chunk — chunked large sections produce overlapping line-windows, so
        // a line shared with a still-live window must survive.
        let mut dormant_lines: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut live_lines: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut live_keys: Vec<String> = Vec::new();
        for (chunk, key) in chunks.iter().zip(keys.iter()) {
            let is_dormant = stats.get(key).is_some_and(|s| s.weight < threshold);
            if !is_dormant && !live_keys.contains(key) {
                live_keys.push(key.clone());
            }
            let target = if is_dormant {
                &mut dormant_lines
            } else {
                &mut live_lines
            };
            for ln in chunk.line_start..=chunk.line_end {
                target.insert(ln);
            }
        }
        let remove: std::collections::HashSet<u32> =
            dormant_lines.difference(&live_lines).copied().collect();

        let kept = if remove.is_empty() {
            raw.clone()
        } else {
            raw.lines()
                .enumerate()
                .filter(|(i, _)| !remove.contains(&((*i as u32) + 1)))
                .map(|(_, l)| l)
                .collect::<Vec<_>>()
                .join("\n")
        };

        let curated = kept.trim();
        if curated.is_empty() {
            return (None, Vec::new());
        }
        let budget = self.config.injection_budget_bytes;
        let text = if curated.len() > budget {
            let head = head_within(curated, budget);
            format!("{head}\n\n_(memory truncated — use `memory_search` for the rest)_")
        } else {
            curated.to_string()
        };
        (Some(text), live_keys)
    }

    fn curated_section(&self) -> Option<String> {
        let raw = read_lenient(&self.curated_path());
        let curated = raw.trim();
        if curated.is_empty() {
            return None;
        }
        let budget = self.config.injection_budget_bytes;
        if curated.len() > budget {
            let head = head_within(curated, budget);
            Some(format!(
                "{head}\n\n_(memory truncated — use `memory_search` for the rest)_"
            ))
        } else {
            Some(curated.to_string())
        }
    }

    fn daily_section(&self, today: NaiveDate) -> Option<String> {
        let yesterday = today.pred_opt().unwrap_or(today);
        let mut recent = String::new();
        // Oldest first so the most recent context is closest to the prompt tail.
        let mut days = vec![(yesterday, "Yesterday")];
        if yesterday != today {
            days.push((today, "Today"));
        }
        for (date, label) in days {
            let raw = read_lenient(&self.daily_path(date));
            let log = raw.trim();
            if !log.is_empty() {
                recent.push_str(&format!("\n#### {label} ({date})\n{log}\n"));
            }
        }
        if recent.is_empty() {
            None
        } else {
            Some(format!("\n### Recent daily log\n{recent}"))
        }
    }
}

/// The default memory root, `~/.flowforge/memory/` (falls back to `./.flowforge`
/// if the home directory cannot be resolved — matches the `~/.flowforge`
/// convention of `mcp.json` and `skills/`).
pub fn default_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".flowforge")
        .join("memory")
}

/// Read a file, treating a missing file as empty. Any other I/O error also
/// degrades to empty so a single unreadable file never breaks a turn.
fn read_lenient(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Largest line-boundary prefix of `text` within `budget` bytes; falls back to a
/// char-boundary cut when a single line already exceeds the budget.
fn head_within(text: &str, budget: usize) -> &str {
    if text.len() <= budget {
        return text;
    }
    let mut cut = 0;
    for (i, _) in text.match_indices('\n') {
        if i > budget {
            break;
        }
        cut = i;
    }
    if cut == 0 {
        let mut b = budget.min(text.len());
        while b > 0 && !text.is_char_boundary(b) {
            b -= 1;
        }
        &text[..b]
    } else {
        &text[..cut]
    }
}

/// Target byte size for a memory chunk, and the overlap carried between windows
/// when a section must be split (RFC 0006 §11.4). Heading-anchored sections are
/// the primary boundary; only a section whose joined text exceeds the target is
/// broken into overlapping line-windows, so today's small memory files stay a
/// single chunk (byte-identical BM25 behaviour) and only genuinely large sections
/// get windowed for focused embeddings. ~2 KB approximates 512 tokens; ~15%
/// overlap preserves context across a split.
const CHUNK_TARGET_BYTES: usize = 2048;
const CHUNK_OVERLAP_BYTES: usize = 307;

/// Greedy line-windows over `lines`: each window's joined byte length stays within
/// `target` where it can (a single oversized line still forms its own window), and
/// successive windows overlap by about `overlap` bytes. Returns half-open
/// `[start, end)` index ranges that cover every line and always advance, so a
/// pathological section can never loop. Splitting on line boundaries keeps every
/// chunk's reported line span exact for `memory_get`.
fn window_line_ranges(lines: &[&str], target: usize, overlap: usize) -> Vec<(usize, usize)> {
    if lines.is_empty() {
        return Vec::new();
    }
    // +1 approximates the `\n` join separator that sits between two lines.
    let len_of = |i: usize| lines[i].len() + 1;
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let mut end = start;
        let mut acc = 0;
        while end < lines.len() && (end == start || acc + len_of(end) <= target) {
            acc += len_of(end);
            end += 1;
        }
        ranges.push((start, end));
        if end >= lines.len() {
            break;
        }
        // Carry ~`overlap` bytes into the next window, but never step back to or
        // past `start`, so the loop always makes forward progress.
        let mut back = end;
        let mut carried = 0;
        while back > start + 1 && carried < overlap {
            back -= 1;
            carried += len_of(back);
        }
        start = back;
    }
    ranges
}

/// Split Markdown into heading-anchored chunks (RFC 0006 §4). Each chunk runs from
/// a heading line to just before the next heading; content before the first
/// heading becomes a preamble chunk. Empty chunks are dropped. Used by the M5.1
/// index; defined here with the types it produces.
pub fn chunk_markdown(text: &str, source: MemorySource, path: &Path) -> Vec<MemoryChunk> {
    let mut chunks: Vec<MemoryChunk> = Vec::new();
    let mut heading: Option<String> = None;
    let mut start_line: u32 = 1;
    let mut body: Vec<&str> = Vec::new();

    let flush = |chunks: &mut Vec<MemoryChunk>,
                 heading: &Option<String>,
                 start: u32,
                 end: u32,
                 body: &[&str]| {
        let joined = body.join("\n");
        if joined.trim().is_empty() {
            return;
        }
        // Small sections (the common case) stay a single chunk -> byte-identical
        // BM25 behaviour. Only a section past the target is split into overlapping
        // line-windows so its embeddings stay focused (RFC 0006 sec 11.4).
        if joined.len() <= CHUNK_TARGET_BYTES {
            let id = chunks.len() as i64;
            chunks.push(MemoryChunk {
                id,
                source: source.clone(),
                path: path.to_path_buf(),
                heading: heading.clone(),
                text: joined,
                line_start: start,
                line_end: end,
                embedding: None,
            });
            return;
        }
        for (s, e) in window_line_ranges(body, CHUNK_TARGET_BYTES, CHUNK_OVERLAP_BYTES) {
            let text = body[s..e].join("\n");
            if text.trim().is_empty() {
                continue;
            }
            let id = chunks.len() as i64;
            chunks.push(MemoryChunk {
                id,
                source: source.clone(),
                path: path.to_path_buf(),
                heading: heading.clone(),
                text,
                line_start: start + s as u32,
                line_end: start + e as u32 - 1,
                embedding: None,
            });
        }
    };

    let total = text.lines().count() as u32;
    // A `#` line inside a fenced code block (``` or ~~~) is code, not a heading —
    // a shell/python comment pasted into a daily log must not split a chunk.
    let mut in_fence = false;
    for (i, raw) in text.lines().enumerate() {
        let lineno = (i + 1) as u32;
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            body.push(raw);
        } else if !in_fence && trimmed.starts_with('#') {
            flush(
                &mut chunks,
                &heading,
                start_line,
                lineno.saturating_sub(1),
                &body,
            );
            heading = Some(trimmed.trim_start_matches('#').trim().to_string());
            start_line = lineno;
            body = vec![raw];
        } else {
            body.push(raw);
        }
    }
    flush(
        &mut chunks,
        &heading,
        start_line,
        total.max(start_line),
        &body,
    );
    chunks
}

/// Insert `text` at the end of the `heading` section of a Markdown document,
/// creating the section at the end of the file if the heading is absent. A
/// "section" runs from its heading line to the next sibling `## ` heading (or
/// EOF). Pure and lenient — used by [`Memory::write_curated_stratum`] (RFC 0008).
fn insert_under_heading(content: &str, heading: &str, text: &str) -> String {
    let text = text.trim_end();
    let lines: Vec<&str> = content.lines().collect();
    let head_idx = lines.iter().position(|l| l.trim_end() == heading);

    match head_idx {
        None => {
            // Append a fresh section at the end of the file.
            let mut out = content.trim_end().to_string();
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(heading);
            out.push('\n');
            out.push_str(text);
            out.push('\n');
            out
        }
        Some(h) => {
            // The section ends at the next top-level `## ` heading, or EOF.
            let end = lines[h + 1..]
                .iter()
                .position(|l| l.starts_with("## "))
                .map(|rel| h + 1 + rel)
                .unwrap_or(lines.len());

            // Drop trailing blank lines inside the section so the new text joins
            // cleanly, then re-emit the section body + the appended text.
            let mut body_end = end;
            while body_end > h + 1 && lines[body_end - 1].trim().is_empty() {
                body_end -= 1;
            }

            let mut out: Vec<String> = lines[..body_end].iter().map(|l| l.to_string()).collect();
            out.push(text.to_string());
            if end < lines.len() {
                // Blank line before the next sibling heading.
                out.push(String::new());
                out.extend(lines[end..].iter().map(|l| l.to_string()));
            }
            let mut joined = out.join("\n");
            joined.push('\n');
            joined
        }
    }
}

/// Replace the entire body under `stratum`'s heading with `text` (#969,
/// whole-stratum replace), preserving the heading line and all sibling stratum
/// sections. When the heading is absent it is created at the end (same as
/// [`insert_under_heading`]), so a first-time edit of an empty stratum works.
/// Empty `text` leaves the heading with an empty body (the stratum is cleared,
/// not removed).
///
/// Section bounds — both the heading match and the end — go through
/// [`Stratum::from_heading_line`], the single rule the read side also uses. A
/// user's own `## sub-heading` inside a stratum is body, not a boundary, so
/// editing such a stratum round-trips conservatively instead of truncating at
/// the sub-heading and re-appending the tail as a sibling section (#774/#868).
fn replace_under_heading(content: &str, stratum: Stratum, text: &str) -> String {
    let heading = stratum.heading();
    let text = text.trim_end();
    let lines: Vec<&str> = content.lines().collect();
    // Case-insensitive, like the read side: a hand-written `## identity` is the
    // Identity stratum, and must be rewritten in place rather than duplicated.
    let head_idx = lines
        .iter()
        .position(|l| Stratum::from_heading_line(l) == Some(stratum));

    match head_idx {
        None => {
            // No such section yet — create it at the end (mirrors insert's append).
            let mut out = content.trim_end().to_string();
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(heading);
            out.push('\n');
            if !text.is_empty() {
                out.push_str(text);
                out.push('\n');
            }
            out
        }
        Some(h) => {
            // Section ends at the next *known stratum* heading, or EOF — NOT at
            // any `## `, which would swallow a note's own sub-heading boundary.
            let end = lines[h + 1..]
                .iter()
                .position(|l| Stratum::from_heading_line(l).is_some())
                .map(|rel| h + 1 + rel)
                .unwrap_or(lines.len());

            // Keep everything up to and including the heading line, drop the old
            // body [h+1..end), splice in the new text, then re-emit any siblings.
            let mut out: Vec<String> = lines[..=h].iter().map(|l| l.to_string()).collect();
            if !text.is_empty() {
                out.push(text.to_string());
            }
            if end < lines.len() {
                // Blank line before the next sibling heading.
                out.push(String::new());
                out.extend(lines[end..].iter().map(|l| l.to_string()));
            }
            let mut joined = out.join("\n");
            joined.push('\n');
            joined
        }
    }
}

#[cfg(test)]
mod tests;
