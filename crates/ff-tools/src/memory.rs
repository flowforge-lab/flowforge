//! Durable-memory recall tools (RFC 0006 §6–§7): `memory_search`, `memory_get`,
//! and `memory_write`. They operate on the user's memory store at
//! `~/.flowforge/memory` — *outside* the per-session workspace jail — so they
//! deliberately ignore the `root` argument every other tool is confined to.
//!
//! Search and get are [`Safety::ReadOnly`]; write is [`Safety::Write`] (the host
//! approval gate prompts before the model edits durable memory). All three no-op
//! gracefully when memory is disabled so a turn never fails on a recall call.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use ff_memory::{
    chunk_key, chunk_markdown, Memory, MemoryIndex, MemorySource, ScoredChunk, Stratum, WriteTarget,
};
use serde_json::Value;

use crate::registry::{Safety, Tool, ToolOutcome};

const DEFAULT_SEARCH_LIMIT: usize = 5;
const MAX_SEARCH_LIMIT: usize = 20;

/// Display a chunk's path relative to the memory root (e.g. `MEMORY.md`,
/// `daily/2026-06-18.md`) so the model gets back a value it can feed to
/// `memory_get`.
fn rel_path(memory: &Memory, path: &Path) -> String {
    path.strip_prefix(memory.root())
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Resolve a model-supplied path against the memory root. `memory_search` only
/// hands back root-relative paths, so all input is treated as relative;
/// `Memory::get` rejects anything that escapes the root (incl. `..` traversal).
fn resolve(memory: &Memory, raw: &str) -> PathBuf {
    memory.root().join(raw)
}

/// Render ranked search hits for display. Shared by `memory_search` (model-
/// facing) and the CLI `memory search` subcommand (human-facing) so both
/// surfaces emit the same path / heading / line-range / dormant-tag shape.
pub fn format_hits(memory: &Memory, hits: &[ScoredChunk]) -> String {
    let threshold = memory.config().decay.dormant_threshold;
    let now_ms = Utc::now().timestamp_millis();
    let mut out = String::new();
    for (i, sc) in hits.iter().enumerate() {
        let chunk = &sc.chunk;
        let path = rel_path(memory, &chunk.path);
        let heading = chunk
            .heading
            .as_deref()
            .map(|h| format!(" \u{203a} {h}"))
            .unwrap_or_default();
        // Dormancy is a derived predicate (RFC 0007 §3): a hit below the
        // threshold is still returned, but tagged so the model knows the fact is
        // old. A search hit also reinforces the chunk (caller-side), so recall
        // can lift it back above the threshold. The tag never appears when decay
        // is disabled (weight stays 1.0).
        let tag = if sc.weight < threshold {
            let age = sc
                .last_accessed_ms
                .map(|ms| human_age_ms(now_ms - ms))
                .unwrap_or_else(|| "long ago".to_string());
            format!(" [dormant \u{b7} last recalled ~{age}]")
        } else {
            String::new()
        };
        out.push_str(&format!(
            "[{}] {path}{heading}{tag} (lines {}-{})\n{}\n\n",
            i + 1,
            chunk.line_start,
            chunk.line_end,
            chunk.text.trim()
        ));
    }
    out.trim_end().to_string()
}

/// Coarse human-readable age for a positive epoch-ms delta, e.g. `6 months ago`.
/// Used only for the dormant tag, so approximate buckets are fine.
fn human_age_ms(delta_ms: i64) -> String {
    let days = (delta_ms.max(0) as f64 / 86_400_000.0).floor() as i64;
    let (n, unit) = if days >= 365 {
        (days / 365, "year")
    } else if days >= 30 {
        (days / 30, "month")
    } else if days >= 7 {
        (days / 7, "week")
    } else if days >= 1 {
        (days, "day")
    } else {
        return "today".to_string();
    };
    let plural = if n == 1 { "" } else { "s" };
    format!("{n} {unit}{plural} ago")
}

/// `memory_search` — ranked BM25 recall over durable memory.
pub struct MemorySearchTool {
    memory: Arc<Memory>,
    index: Arc<dyn MemoryIndex>,
}

impl MemorySearchTool {
    pub fn new(memory: Arc<Memory>, index: Arc<dyn MemoryIndex>) -> Self {
        Self { memory, index }
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn reaches_network(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "memory_search"
    }

    fn description(&self) -> &str {
        "Search your durable memory (notes you've kept across sessions about the user, \
         their projects, and decisions) for text relevant to a query. Returns ranked \
         snippets with their file path and line range; follow up with `memory_get` to \
         read more around a hit."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to recall — keywords or a short phrase."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5, max 20).",
                    "minimum": 1
                }
            },
            "required": ["query"]
        })
    }

    fn safety(&self, _args: &Value) -> Safety {
        Safety::ReadOnly
    }

    fn max_safety(&self) -> Safety {
        Safety::ReadOnly
    }

    async fn run(&self, args: Value, _root: &Path) -> ToolOutcome {
        if !self.memory.is_enabled() {
            return ToolOutcome::ok("(memory is disabled)");
        }
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return ToolOutcome::error("missing required argument: query (a string)");
        };
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| (n as usize).clamp(1, MAX_SEARCH_LIMIT))
            .unwrap_or(DEFAULT_SEARCH_LIMIT);

        // The hybrid index may make a blocking embedding HTTP call inside `search`;
        // run it off the async worker so a slow/unreachable embedder never parks a
        // runtime thread (it still degrades to BM25 internally on failure).
        let index = self.index.clone();
        let query = query.to_string();
        match tokio::task::spawn_blocking(move || {
            let hits = index.search(&query, limit)?;
            // Reinforce the surfaced top-k (RFC 0007 sec 2, M6.0). Best-effort: a
            // stats write must never fail recall. It is a no-op for backends
            // without a chunk_stats table, and weight-neutral when decay is off.
            let _ = index.reinforce(&hits);
            ff_memory::Result::Ok(hits)
        })
        .await
        {
            Ok(Ok(hits)) if hits.is_empty() => ToolOutcome::ok("No matching memory."),
            Ok(Ok(hits)) => ToolOutcome::ok(format_hits(&self.memory, &hits)),
            Ok(Err(e)) => ToolOutcome::error(format!("memory search failed: {e}")),
            Err(e) => ToolOutcome::error(format!("memory search task failed: {e}")),
        }
    }
}

/// `memory_get` — read a memory file (optionally a line range).
pub struct MemoryGetTool {
    memory: Arc<Memory>,
}

impl MemoryGetTool {
    pub fn new(memory: Arc<Memory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for MemoryGetTool {
    fn reaches_network(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "memory_get"
    }

    fn description(&self) -> &str {
        "Read a durable-memory file, optionally limited to a 1-based inclusive line \
         range. Use the path and line numbers from a `memory_search` hit to read the \
         surrounding context."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Memory file path, e.g. `MEMORY.md` or `daily/2026-06-18.md`."
                },
                "line_start": {
                    "type": "integer",
                    "description": "First line to read (1-based, inclusive).",
                    "minimum": 1
                },
                "line_end": {
                    "type": "integer",
                    "description": "Last line to read (1-based, inclusive).",
                    "minimum": 1
                }
            },
            "required": ["path"]
        })
    }

    fn safety(&self, _args: &Value) -> Safety {
        Safety::ReadOnly
    }

    fn max_safety(&self) -> Safety {
        Safety::ReadOnly
    }

    async fn run(&self, args: Value, _root: &Path) -> ToolOutcome {
        if !self.memory.is_enabled() {
            return ToolOutcome::ok("(memory is disabled)");
        }
        let Some(raw) = args.get("path").and_then(Value::as_str) else {
            return ToolOutcome::error("missing required argument: path (a string)");
        };
        let line_start = args
            .get("line_start")
            .and_then(Value::as_u64)
            .map(|n| n as u32);
        let line_end = args
            .get("line_end")
            .and_then(Value::as_u64)
            .map(|n| n as u32);

        let path = resolve(&self.memory, raw);
        let content = self.memory.get(&path, line_start, line_end);
        if content.is_empty() {
            ToolOutcome::ok("(no such memory file or empty)")
        } else {
            ToolOutcome::ok(content)
        }
    }
}

/// `memory_write` — append to durable memory and reindex so the new text is
/// searchable within the same turn.
pub struct MemoryWriteTool {
    memory: Arc<Memory>,
    index: Arc<dyn MemoryIndex>,
    /// Optional session-scoped log of the Daily `chunk_key`s this session wrote.
    /// When wired (goal mode), the termination hook drains it and settles the
    /// keys against the session's verdict (#1292). `None` keeps the historic
    /// fire-and-forget behaviour for one-shot / non-goal runs.
    touched: Option<ff_memory::TouchLog>,
}

impl MemoryWriteTool {
    pub fn new(memory: Arc<Memory>, index: Arc<dyn MemoryIndex>) -> Self {
        Self {
            memory,
            index,
            touched: None,
        }
    }

    /// Attach a shared [`TouchLog`](ff_memory::TouchLog) so writes register the
    /// Daily chunks they produce for later outcome settlement.
    pub fn with_touch_log(mut self, touched: ff_memory::TouchLog) -> Self {
        self.touched = Some(touched);
        self
    }
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn reaches_network(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "memory_write"
    }

    fn description(&self) -> &str {
        "Append a note to your durable memory so you remember it in future sessions. \
         `target` is `daily` (today's log — for time-stamped observations, the default) \
         or `curated` (the long-lived MEMORY.md — for stable facts about the user and \
         their projects). For curated facts, set `stratum` to file the note under the \
         right section: `identity` (who they are), `patterns` (how they work), or \
         `focus` (what they are working on). Write Markdown; a `## Heading` makes the \
         note easier to recall."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The Markdown note to append."
                },
                "target": {
                    "type": "string",
                    "enum": ["daily", "curated"],
                    "description": "Where to write (default `daily`)."
                },
                "stratum": {
                    "type": "string",
                    "enum": ["identity", "patterns", "focus"],
                    "description": "Curated section to file the note under (implies `curated`)."
                }
            },
            "required": ["text"]
        })
    }

    fn safety(&self, _args: &Value) -> Safety {
        Safety::Write
    }

    async fn run(&self, args: Value, _root: &Path) -> ToolOutcome {
        if !self.memory.is_enabled() {
            return ToolOutcome::ok("(memory is disabled)");
        }
        let Some(text) = args.get("text").and_then(Value::as_str) else {
            return ToolOutcome::error("missing required argument: text (a string)");
        };
        if text.trim().is_empty() {
            return ToolOutcome::error("nothing to write: text is empty");
        }
        let stratum = match args.get("stratum").and_then(Value::as_str) {
            None => None,
            Some(s) => match Stratum::parse(s) {
                Some(st) => Some(st),
                None => {
                    return ToolOutcome::error(format!(
                        "invalid stratum `{s}` (expected `identity`, `patterns`, or `focus`)"
                    ));
                }
            },
        };

        let target_arg = args.get("target").and_then(Value::as_str);
        if stratum.is_some() && target_arg == Some("daily") {
            return ToolOutcome::error(
                "stratum applies only to curated memory; got target `daily`",
            );
        }
        let target = match target_arg {
            None if stratum.is_some() => WriteTarget::Curated,
            None | Some("daily") => WriteTarget::Daily,
            Some("curated") => WriteTarget::Curated,
            Some(other) => {
                return ToolOutcome::error(format!(
                    "invalid target `{other}` (expected `daily` or `curated`)"
                ));
            }
        };

        // Snapshot the Daily file's existing chunk keys before the write, so we
        // can touch only the chunk(s) this write actually produces — not every
        // fact already in today's file (#1292 review F4). The path is
        // date-derived and stable across the append. Skipped unless a touch log
        // is wired and we're writing Daily (only Daily chunks are promotion
        // candidates).
        let pre_keys: Option<HashSet<String>> =
            if matches!(target, WriteTarget::Daily) && self.touched.is_some() {
                let dpath = self.memory.daily_path(chrono::Local::now().date_naive());
                let before = self.memory.get(&dpath, None, None);
                let src = MemorySource::Daily {
                    date: chrono::Local::now().date_naive(),
                };
                Some(
                    chunk_markdown(&before, src, &dpath)
                        .iter()
                        .map(chunk_key)
                        .collect(),
                )
            } else {
                None
            };

        let write_result = match stratum {
            Some(st) => self.memory.write_curated_stratum(text, st),
            None => self.memory.write(text, target),
        };
        let path = match write_result {
            Ok(p) => p,
            Err(e) => return ToolOutcome::error(format!("memory write failed: {e}")),
        };

        let source = match target {
            WriteTarget::Daily => MemorySource::Daily {
                date: chrono::Local::now().date_naive(),
            },
            WriteTarget::Curated => MemorySource::Curated,
        };
        let full = self.memory.get(&path, None, None);
        let chunks = chunk_markdown(&full, source, &path);

        // Register only the Daily chunks this write *added* (post-write set minus
        // the pre-write snapshot), so a later outcome settlement reinforces or
        // suppresses exactly what this write introduced — not co-located facts it
        // merely sat beside (#1292 review F4). No-op when no log is wired.
        if let Some(pre) = &pre_keys {
            if let Some(touched) = &self.touched {
                touched.extend(chunks.iter().map(chunk_key).filter(|k| !pre.contains(k)));
            }
        }

        // The hybrid index may make a blocking embedding HTTP call inside
        // `reindex_path`; run it off the async worker (mirrors `memory_search`)
        // so `reqwest::blocking` never drops its runtime in async context, which
        // would panic the turn task. It still degrades to BM25 internally on
        // embed failure.
        let index = self.index.clone();
        let path2 = path.clone();
        match tokio::task::spawn_blocking(move || index.reindex_path(&path2, &chunks)).await {
            Ok(Ok(())) => ToolOutcome::ok(format!("Wrote to {}", rel_path(&self.memory, &path))),
            Ok(Err(e)) => ToolOutcome::ok(format!(
                "Wrote to {} (warning: reindex failed: {e})",
                rel_path(&self.memory, &path)
            )),
            Err(e) => ToolOutcome::ok(format!(
                "Wrote to {} (warning: reindex task failed: {e})",
                rel_path(&self.memory, &path)
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// memory_consolidate — trigger the consolidation pass (#223)
// ---------------------------------------------------------------------------

/// `memory_consolidate` — manually trigger the consolidation pass (issue #223).
///
/// Verifies the trigger condition (or `force`), runs [`Memory::consolidate`]
/// (merge / promote / demote), then rebuilds the recall index off the rewritten
/// files. Idempotent: a re-run with nothing to change reports a no-op.
///
/// **Invariant**: consolidation is the sole full-file writer of curated
/// Markdown; decay/dormancy (M6) never edits Markdown.
pub struct MemoryConsolidateTool {
    memory: Arc<Memory>,
    /// Reindexed after the atomic curated rewrite so recall sees the new shape.
    index: Arc<dyn MemoryIndex>,
}

impl MemoryConsolidateTool {
    pub fn new(memory: Arc<Memory>, index: Arc<dyn MemoryIndex>) -> Self {
        Self { memory, index }
    }
}

#[async_trait]
impl Tool for MemoryConsolidateTool {
    fn reaches_network(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "memory_consolidate"
    }

    fn description(&self) -> &str {
        "Trigger a memory consolidation pass: merges near-identical curated facts, promotes recurring high-signal daily-log entries into MEMORY.md, and bounds the curated file to the injection budget. Idempotent — a re-run with no growth is a no-op. Raw daily logs are never deleted."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Run even if the trigger threshold is not met (default false)."
                }
            }
        })
    }

    fn safety(&self, _args: &Value) -> Safety {
        Safety::Write
    }

    async fn run(&self, args: Value, _root: &Path) -> ToolOutcome {
        if !self.memory.is_enabled() {
            return ToolOutcome::ok("(memory is disabled)");
        }
        let force = args.get("force").and_then(Value::as_bool).unwrap_or(false);

        if !force && !self.memory.needs_consolidation() {
            return ToolOutcome::ok("Consolidation not needed: curated file is within budget.");
        }

        let salience = self
            .memory
            .chunk_stats_salience(self.index.as_ref(), Utc::now().timestamp_millis());
        let salience_index = self.index.as_ref();
        let report = match self.memory.consolidate(&salience, Some(salience_index)) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::error(format!("consolidation failed: {e}")),
        };

        if !report.ran {
            return ToolOutcome::ok("Consolidation ran: nothing to change (already consolidated).");
        }

        // Rebuild the recall index off the rewritten files. The hybrid index may
        // make a blocking embedding HTTP call, so run it off the async worker
        // (mirrors `memory_write`); it degrades to BM25 internally on failure.
        let index = self.index.clone();
        let memory = self.memory.clone();
        let chunks = self.memory.all_chunks();
        let summary = format!(
            "Consolidated: merged {}, promoted {}, demoted {} ({} -> {} bytes).",
            report.merged, report.promoted, report.demoted, report.bytes_before, report.bytes_after
        );
        match tokio::task::spawn_blocking(move || {
            index.reindex(&chunks)?;
            // Record supersession edges after reindex, when both endpoint keys are
            // live in the rebuilt index (memory sifting C1, #1293). Edges describe
            // events, so reindex never GCs them. Best-effort, owned by ff-memory.
            memory.record_supersession_edges(index.as_ref(), &report);
            Ok::<(), ff_memory::MemoryError>(())
        })
        .await
        {
            Ok(Ok(())) => ToolOutcome::ok(summary),
            Ok(Err(e)) => ToolOutcome::ok(format!("{summary} (warning: reindex failed: {e})")),
            Err(e) => ToolOutcome::ok(format!("{summary} (warning: reindex task failed: {e})")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_memory::{Fts5Index, HybridIndex, MemoryChunk, MemoryConfig, OpenAiEmbedder};
    use std::sync::Arc;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn setup() -> (TempDir, Arc<Memory>, Arc<dyn MemoryIndex>) {
        let dir = TempDir::new().unwrap();
        let memory = Arc::new(Memory::new(
            dir.path().to_path_buf(),
            MemoryConfig::default(),
        ));
        let index: Arc<dyn MemoryIndex> = Arc::new(Fts5Index::open_in_memory().unwrap());
        (dir, memory, index)
    }

    fn disabled() -> (TempDir, Arc<Memory>, Arc<dyn MemoryIndex>) {
        let dir = TempDir::new().unwrap();
        let memory = Arc::new(Memory::new(
            dir.path().to_path_buf(),
            MemoryConfig {
                enabled: false,
                ..MemoryConfig::default()
            },
        ));
        let index: Arc<dyn MemoryIndex> = Arc::new(Fts5Index::open_in_memory().unwrap());
        (dir, memory, index)
    }

    fn hit(weight: f32, last_accessed_ms: Option<i64>) -> ScoredChunk {
        ScoredChunk {
            chunk: MemoryChunk {
                id: 1,
                source: MemorySource::Curated,
                path: PathBuf::from("MEMORY.md"),
                heading: Some("Prefs".to_string()),
                text: "user prefers rust".to_string(),
                line_start: 1,
                line_end: 2,
                embedding: None,
            },
            score: 1.0,
            weight,
            last_accessed_ms,
        }
    }

    #[test]
    fn format_hits_tags_dormant_chunk_with_age() {
        // Default dormant_threshold is 0.25; a 0.1-weight hit recalled ~6 months
        // ago must be returned but tagged (RFC 0007 §3).
        let (_dir, memory, _index) = setup();
        let six_months_ago = Utc::now().timestamp_millis() - 180 * 86_400_000;
        let out = format_hits(&memory, &[hit(0.1, Some(six_months_ago))]);
        assert!(out.contains("[dormant"), "missing dormant tag: {out}");
        assert!(out.contains("last recalled ~"), "{out}");
        assert!(out.contains("months ago"), "{out}");
        // The chunk body is still present -- dormant means tagged, not dropped.
        assert!(out.contains("user prefers rust"), "{out}");
    }

    #[test]
    fn format_hits_leaves_live_chunk_untagged() {
        let (_dir, memory, _index) = setup();
        let out = format_hits(&memory, &[hit(0.9, Some(Utc::now().timestamp_millis()))]);
        assert!(
            !out.contains("dormant"),
            "live chunk must not be tagged: {out}"
        );
    }

    #[test]
    fn format_hits_dormant_without_timestamp_says_long_ago() {
        let (_dir, memory, _index) = setup();
        let out = format_hits(&memory, &[hit(0.0, None)]);
        assert!(out.contains("long ago"), "{out}");
    }

    #[test]
    fn human_age_ms_buckets() {
        let day = 86_400_000_i64;
        assert_eq!(human_age_ms(0), "today");
        assert_eq!(human_age_ms(day * 3 / 2), "1 day ago");
        assert_eq!(human_age_ms(day * 2), "2 days ago");
        assert_eq!(human_age_ms(day * 14), "2 weeks ago");
        assert_eq!(human_age_ms(day * 180), "6 months ago");
        assert_eq!(human_age_ms(day * 400), "1 year ago");
    }

    #[tokio::test]
    async fn write_then_search_then_get_round_trip() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory.clone(), index.clone());
        let search = MemorySearchTool::new(memory.clone(), index.clone());
        let get = MemoryGetTool::new(memory.clone());

        let out = write
            .run(
                serde_json::json!({
                    "text": "## Join key\nThe origin address id is the donor join key.",
                    "target": "curated"
                }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);
        assert!(out.content.contains("MEMORY.md"), "{}", out.content);

        let hit = search
            .run(
                serde_json::json!({ "query": "origin address join key" }),
                Path::new("."),
            )
            .await;
        assert!(hit.success, "{}", hit.content);
        assert!(
            hit.content.contains("Join key"),
            "search miss: {}",
            hit.content
        );
        assert!(hit.content.contains("MEMORY.md"), "{}", hit.content);

        let read = get
            .run(serde_json::json!({ "path": "MEMORY.md" }), Path::new("."))
            .await;
        assert!(read.success);
        assert!(read.content.contains("donor join key"), "{}", read.content);
    }

    #[tokio::test]
    async fn daily_write_registers_touched_chunks_in_log() {
        let (_dir, memory, index) = setup();
        let log = ff_memory::TouchLog::new();
        let write = MemoryWriteTool::new(memory.clone(), index.clone()).with_touch_log(log.clone());

        let out = write
            .run(
                serde_json::json!({
                    "text": "## Observation\nthe dishwasher still has standing water",
                    "target": "daily"
                }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);
        assert!(
            !log.is_empty(),
            "a daily write should register its chunk_key(s)"
        );
    }

    #[tokio::test]
    async fn write_then_settle_success_leaves_no_stats_row() {
        // #1304 F2: a chunk written *this* run has no `chunk_stats` row, and the
        // weight model caps at the recall baseline (1.0) — so a success cannot
        // lift it above baseline. Creating a row would only start a decay clock
        // and leave the chunk worse off than an untracked sibling. Success on a
        // freshly written chunk must NOT create a row; "not suppressed" is the
        // whole effect. This also pins the end-to-end wiring: the goal-mode
        // `MemoryWriteTool` fills the touch log, and `settle` consumes it.
        let (_dir, memory, index) = setup();
        let log = ff_memory::TouchLog::new();
        let write = MemoryWriteTool::new(memory.clone(), index.clone()).with_touch_log(log.clone());

        let out = write
            .run(
                serde_json::json!({ "text": "Prefer nextest over cargo test for the full run." }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);

        let touched = log.drain();
        assert_eq!(touched.len(), 1, "one chunk written -> one touched key");

        // Before settle there is no stats row for the written key.
        let before = index
            .chunk_stats_snapshot(&touched, chrono::Utc::now().timestamp_millis())
            .unwrap();
        assert!(
            before.is_empty(),
            "a fresh write only populates `chunks`, never `chunk_stats`"
        );

        ff_memory::MemoryOutcomeSink::new(index.as_ref())
            .settle(ff_memory::Verdict::Success, &touched)
            .unwrap();

        // Still no row: a fresh written-then-succeeded chunk is left at the
        // untracked default (effective weight 1.0, never dormant), not saddled
        // with a decay clock that would penalise it later.
        let after = index
            .chunk_stats_snapshot(&touched, chrono::Utc::now().timestamp_millis())
            .unwrap();
        assert!(
            after.is_empty(),
            "success on a freshly written chunk must not create a decaying row"
        );
    }

    #[tokio::test]
    async fn second_daily_write_touches_only_the_new_chunk() {
        // #1292 review F4: a Daily write rebuilds the whole file, so chunking the
        // rebuilt file would register every fact already in it. Only the chunk
        // this write *added* should be touched, so settlement reinforces exactly
        // what this write introduced.
        let (_dir, memory, index) = setup();

        // First write establishes an existing chunk under its own heading.
        let first = ff_memory::TouchLog::new();
        MemoryWriteTool::new(memory.clone(), index.clone())
            .with_touch_log(first.clone())
            .run(
                serde_json::json!({ "text": "## First\nAn earlier note." }),
                Path::new("."),
            )
            .await;
        let first_keys = first.drain();
        assert_eq!(first_keys.len(), 1);

        // Second write into the same daily file must touch only its own chunk.
        let second = ff_memory::TouchLog::new();
        let out = MemoryWriteTool::new(memory.clone(), index.clone())
            .with_touch_log(second.clone())
            .run(
                serde_json::json!({ "text": "## Second\nA later note." }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);

        let second_keys = second.drain();
        assert_eq!(
            second_keys.len(),
            1,
            "only the chunk this write added is touched, not the whole file"
        );
        assert!(
            !second_keys.iter().any(|k| first_keys.contains(k)),
            "the pre-existing chunk must not be re-touched by a later write"
        );
    }

    #[tokio::test]
    async fn curated_write_does_not_touch_log() {
        let (_dir, memory, index) = setup();
        let log = ff_memory::TouchLog::new();
        let write = MemoryWriteTool::new(memory.clone(), index.clone()).with_touch_log(log.clone());

        let out = write
            .run(
                serde_json::json!({
                    "text": "## Stable fact\nTony owns FlowForge front to back.",
                    "target": "curated"
                }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);
        assert!(
            log.is_empty(),
            "curated writes are not promotion candidates, so nothing is touched"
        );
    }

    #[tokio::test]
    async fn write_without_touch_log_still_succeeds() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory.clone(), index.clone());
        let out = write
            .run(
                serde_json::json!({ "text": "## Note\nno log wired", "target": "daily" }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);
    }

    #[tokio::test]
    async fn search_empty_yields_no_match() {
        let (_dir, memory, index) = setup();
        let search = MemorySearchTool::new(memory, index);
        let out = search
            .run(
                serde_json::json!({ "query": "nonexistent" }),
                Path::new("."),
            )
            .await;
        assert!(out.success);
        assert_eq!(out.content, "No matching memory.");
    }

    #[tokio::test]
    async fn search_missing_query_is_error() {
        let (_dir, memory, index) = setup();
        let search = MemorySearchTool::new(memory, index);
        let out = search.run(serde_json::json!({}), Path::new(".")).await;
        assert!(!out.success);
        assert!(out.content.contains("missing required argument: query"));
    }

    #[tokio::test]
    async fn get_missing_file_is_ok_empty() {
        let (_dir, memory, _index) = setup();
        let get = MemoryGetTool::new(memory);
        let out = get
            .run(serde_json::json!({ "path": "MEMORY.md" }), Path::new("."))
            .await;
        assert!(out.success);
        assert!(out.content.contains("no such memory file"));
    }

    #[tokio::test]
    async fn get_rejects_path_traversal() {
        let dir = TempDir::new().unwrap();
        // A secret sibling outside the memory root.
        let secret = dir.path().join("secret.txt");
        std::fs::write(&secret, "TOP-SECRET-KEY-12345").unwrap();
        let root = dir.path().join("memory");
        let memory = Arc::new(Memory::new(root, MemoryConfig::default()));
        let get = MemoryGetTool::new(memory);

        for p in [
            "../secret.txt",
            "../../secret.txt",
            "daily/../../secret.txt",
        ] {
            let out = get
                .run(serde_json::json!({ "path": p }), Path::new("."))
                .await;
            assert!(out.success, "{}", out.content);
            assert!(
                !out.content.contains("TOP-SECRET"),
                "path `{p}` leaked a file outside the memory root: {}",
                out.content
            );
        }
    }

    #[tokio::test]
    async fn write_empty_text_is_error() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory, index);
        let out = write
            .run(serde_json::json!({ "text": "   " }), Path::new("."))
            .await;
        assert!(!out.success);
        assert!(out.content.contains("nothing to write"));
    }

    #[tokio::test]
    async fn write_invalid_target_is_error() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory, index);
        let out = write
            .run(
                serde_json::json!({ "text": "x", "target": "weekly" }),
                Path::new("."),
            )
            .await;
        assert!(!out.success);
        assert!(out.content.contains("invalid target"));
    }

    #[tokio::test]
    async fn write_with_stratum_files_under_heading_and_is_searchable() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory.clone(), index.clone());
        let out = write
            .run(
                serde_json::json!({ "text": "L5 SDE on Maps", "stratum": "identity" }),
                Path::new("."),
            )
            .await;
        assert!(out.success, "{}", out.content);
        // Stratum implies curated: the note lands under `## Identity` in MEMORY.md.
        let curated = memory.get(&memory.curated_path(), None, None);
        assert!(curated.contains("## Identity"), "{curated}");
        assert!(curated.contains("L5 SDE on Maps"), "{curated}");
    }

    #[tokio::test]
    async fn write_stratum_with_daily_target_is_error() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory, index);
        let out = write
            .run(
                serde_json::json!({ "text": "x", "stratum": "identity", "target": "daily" }),
                Path::new("."),
            )
            .await;
        assert!(!out.success);
        assert!(
            out.content.contains("stratum applies only to curated"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn write_invalid_stratum_is_error() {
        let (_dir, memory, index) = setup();
        let write = MemoryWriteTool::new(memory, index);
        let out = write
            .run(
                serde_json::json!({ "text": "x", "stratum": "vibes" }),
                Path::new("."),
            )
            .await;
        assert!(!out.success);
        assert!(out.content.contains("invalid stratum"), "{}", out.content);
    }

    #[tokio::test]
    async fn disabled_memory_tools_no_op() {
        let (_dir, memory, index) = disabled();
        let search = MemorySearchTool::new(memory.clone(), index.clone());
        let write = MemoryWriteTool::new(memory.clone(), index.clone());
        let get = MemoryGetTool::new(memory.clone());

        for out in [
            search
                .run(serde_json::json!({ "query": "x" }), Path::new("."))
                .await,
            write
                .run(serde_json::json!({ "text": "x" }), Path::new("."))
                .await,
            get.run(serde_json::json!({ "path": "MEMORY.md" }), Path::new("."))
                .await,
        ] {
            assert!(out.success);
            assert_eq!(out.content, "(memory is disabled)");
        }
        // Disabled write must not create the file.
        assert!(!memory.curated_path().exists());
    }

    #[tokio::test]
    async fn safety_classifications() {
        let (_dir, memory, index) = setup();
        let search = MemorySearchTool::new(memory.clone(), index.clone());
        let get = MemoryGetTool::new(memory.clone());
        let write = MemoryWriteTool::new(memory, index);
        let empty = serde_json::json!({});
        assert_eq!(search.safety(&empty), Safety::ReadOnly);
        assert_eq!(get.safety(&empty), Safety::ReadOnly);
        assert_eq!(write.safety(&empty), Safety::Write);
    }

    // Regression for the M5.3.1 review (#215): `MemoryWriteTool::run` is async and
    // must run the (blocking) hybrid reindex off the async worker. With embeddings
    // enabled, `reindex_path` -> `embed_chunk` -> `reqwest::blocking`; invoking that
    // directly on a tokio worker drops reqwest's internal runtime in async context
    // and panics the turn. Driving the real async call site here (embed runs ON a
    // tokio worker, not via an off-runtime thread) would panic before the fix and
    // succeeds after it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_with_hybrid_embedder_does_not_panic_on_async_worker() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "embedding": [0.1f32, 0.2, 0.3] }]
            })))
            .mount(&server)
            .await;

        let dir = TempDir::new().unwrap();
        let memory = Arc::new(Memory::new(
            dir.path().to_path_buf(),
            MemoryConfig::default(),
        ));
        // Build the embedder off the async worker, exactly like production
        // (`build_memory` runs before Tauri enters its runtime): `reqwest::blocking`
        // spins up and drops a temporary runtime during construction, which itself
        // panics inside an async context.
        let uri = server.uri();
        let index: Arc<dyn MemoryIndex> = tokio::task::spawn_blocking(move || {
            let embedder = OpenAiEmbedder::new(format!("{uri}/v1"), "test-embed", None);
            Arc::new(HybridIndex::new(
                Fts5Index::open_in_memory().unwrap(),
                embedder,
            )) as Arc<dyn MemoryIndex>
        })
        .await
        .unwrap();

        let write = MemoryWriteTool::new(memory.clone(), index.clone());
        let out = write
            .run(
                serde_json::json!({
                    "text": "## Embeddings\nlocal vector recall is on for this write.",
                    "target": "curated"
                }),
                Path::new("."),
            )
            .await;

        assert!(out.success, "{}", out.content);
        assert!(out.content.contains("MEMORY.md"), "{}", out.content);
        assert!(
            !out.content.contains("reindex task failed"),
            "{}",
            out.content
        );

        // The embedded chunk is searchable (also off the worker), proving the
        // hybrid path ran end-to-end without panicking.
        let search = MemorySearchTool::new(memory.clone(), index.clone());
        let hit = search
            .run(
                serde_json::json!({"query": "vector recall"}),
                Path::new("."),
            )
            .await;
        assert!(hit.success, "{}", hit.content);
    }

    #[tokio::test]
    async fn consolidate_noop_when_under_budget() {
        let (_dir, memory, index) = setup();
        let consolidate = MemoryConsolidateTool::new(memory.clone(), index);
        let out = consolidate.run(serde_json::json!({}), Path::new(".")).await;
        assert!(out.success);
        assert!(out.content.contains("not needed"), "{}", out.content);
    }

    #[tokio::test]
    async fn consolidate_triggers_when_forced() {
        let (_dir, memory, index) = setup();
        let consolidate = MemoryConsolidateTool::new(memory.clone(), index);
        let out = consolidate
            .run(serde_json::json!({ "force": true }), Path::new("."))
            .await;
        assert!(out.success);
        // Empty memory: the forced pass runs but has nothing to change.
        assert!(out.content.contains("nothing to change"), "{}", out.content);
    }

    #[tokio::test]
    async fn consolidate_disabled_memory_no_op() {
        let (_dir, memory, index) = disabled();
        let consolidate = MemoryConsolidateTool::new(memory, index);
        let out = consolidate
            .run(serde_json::json!({ "force": true }), Path::new("."))
            .await;
        assert!(out.success);
        assert_eq!(out.content, "(memory is disabled)");
    }
}
