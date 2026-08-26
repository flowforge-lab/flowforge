//! Consolidation pass foundation (RFC 0006 §7.3, issue #223 P1).
//!
//! **Invariant**: Consolidation (via `rewrite_curated`) is the sole path that
//! does a **full-file rewrite** of curated `MEMORY.md`. The normal agent/user
//! capture path (`memory_write` → `WriteTarget::Curated`) still **appends** —
//! that's expected and not a violation. Decay/dormancy (M6, RFC 0007) NEVER
//! edits Markdown — it only controls what gets *injected*. Both consume
//! `chunk_key` and `Salience`.
//!
//! This module provides:
//! - [`chunk_key`] — stable identity for dedup and M6's `chunk_stats` table.
//! - [`Salience`] trait — ranks chunks for promotion/demotion decisions.
//! - [`RecencyFrequencySalience`] — mechanical default (recency × frequency).

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use chrono::{Local, NaiveDate};

use crate::{read_lenient, MemoryChunk, MemorySource, Result, Stratum, WriteTarget};

/// Canonical strata, emitted in this fixed who/how/what order during the
/// consolidate rewrite (RFC 0008 §4, issue #254).
const CANONICAL_STRATA: [Stratum; 3] = [Stratum::Identity, Stratum::Patterns, Stratum::Focus];

/// Stable chunk identity: `source + heading-path + hash(normalized text)`.
///
/// This key is shared between the consolidation pass (dedup) and M6's future
/// `chunk_stats` side table (RFC 0007 §4). It must be stable across:
/// - line-number shifts (reindex)
/// - trailing whitespace changes
/// - reordering of chunks in the same file
///
/// Changing the *content* of a chunk intentionally produces a new key (the old
/// stats are orphaned and swept on rebuild — the right default for genuinely
/// new content, per RFC 0007 §4).
pub fn chunk_key(chunk: &MemoryChunk) -> String {
    let source_tag = match &chunk.source {
        MemorySource::Curated => "curated".to_string(),
        MemorySource::Daily { date } => format!("daily:{}", date.format("%Y-%m-%d")),
    };
    format!("{source_tag}:{}", content_key(chunk))
}

/// Source-agnostic content identity: `heading-path + hash(normalized text)`.
///
/// Unlike [`chunk_key`] this deliberately omits the source tag, so the *same
/// fact* captured in a daily log and again in curated Markdown shares one key.
/// Consolidation uses it to count how often a fact recurs (promotion) and to
/// detect a curated fact that is already present (merge / skip).
fn content_key(chunk: &MemoryChunk) -> String {
    let heading_path = chunk.heading.as_deref().unwrap_or("");
    let normalized = normalize_text(&chunk.text);

    // SHA-256 is stable across Rust versions - critical because RFC 0007 sec 4
    // persists these keys in the chunk_stats side table. Truncated to 16 hex
    // chars (64 bits) for compact keys; collision risk is negligible at memory
    // scale (~thousands of chunks).
    let hash = Sha256::digest(normalized.as_bytes());
    let text_hash: String = hash.iter().take(8).map(|b| format!("{b:02x}")).collect();

    format!("{heading_path}:{text_hash}")
}
/// The source-agnostic content portion of a [`chunk_key`]. A `chunk_key` is
/// `"{source_tag}:{content_key}"`, so stripping the tag yields the same value
/// [`content_key`] would produce for that chunk. Used by promotion suppression
/// (#1292 F3) so a fact flagged on one day stays suppressed even when it recurs
/// under a later daily date (a different `chunk_key`, identical content).
pub(crate) fn content_key_of(chunk_key: &str) -> &str {
    if let Some(rest) = chunk_key.strip_prefix("daily:") {
        // `rest` is `YYYY-MM-DD:{content_key}`; drop the fixed 10-char date and
        // its trailing colon.
        rest.get(11..).unwrap_or(rest)
    } else if let Some(rest) = chunk_key.strip_prefix("curated:") {
        rest
    } else {
        chunk_key
    }
}

/// Normalize text for hashing: trim each line, collapse runs of whitespace,
/// strip trailing newlines. This makes the key resilient to formatting drift
/// while still changing when the semantic content changes.
fn normalize_text(text: &str) -> String {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Salience trait
// ---------------------------------------------------------------------------

/// Ranks a chunk's importance for consolidation decisions (promotion into or
/// demotion out of curated Markdown).
///
/// **Default impl**: [`RecencyFrequencySalience`] — mechanical recency × frequency.
/// **M6.3** (RFC 0007 §6, issue #294): [`ChunkStatsSalience`] is the promised
/// `chunk_stats`-backed impl — real decayed `weight` ranks demotion, with zero
/// change to the consolidation loops (swap the trait object at the call site).
///
/// TODO(M6.3): Add an LLM-driven Salience impl that uses semantic similarity
/// and importance scoring beyond mechanical heuristics.
///
/// Generic over the scored item `T` (RFC 0022 Step 2a): memory scores
/// `MemoryChunk` for consolidation, while `ff-agent` scores `Message` for
/// value-aware compaction band selection — one decay model, two call sites.
pub trait Salience<T>: Send + Sync {
    /// Score an item in `[0.0, 1.0]`. Higher = more salient = keep/promote.
    fn score(&self, item: &T, occurrences: u32) -> f32;
}

/// Mechanical recency × frequency ranking (the P1 default).
///
/// - **Recency**: exponential decay from chunk date (daily logs) or a fixed
///   high recency for curated (already promoted = presumed relevant).
/// - **Frequency**: `min(1.0, occurrences / saturation)` — caps so a single
///   repeated fact doesn't dominate.
///
/// Both factors are in `[0, 1]`; the product is the score.
pub struct RecencyFrequencySalience {
    /// Half-life in days for recency decay.
    pub half_life_days: f32,
    /// Number of occurrences at which frequency saturates to 1.0.
    pub saturation: u32,
}

impl Default for RecencyFrequencySalience {
    fn default() -> Self {
        Self {
            half_life_days: 14.0,
            saturation: 3,
        }
    }
}

impl Salience<MemoryChunk> for RecencyFrequencySalience {
    fn score(&self, chunk: &MemoryChunk, occurrences: u32) -> f32 {
        let recency = match &chunk.source {
            MemorySource::Curated => 1.0_f32, // already promoted
            MemorySource::Daily { date } => {
                let today = Local::now().date_naive();
                let age_days = (today - *date).num_days().max(0) as f32;
                // Exponential decay: 0.5^(age / half_life)
                (0.5_f32).powf(age_days / self.half_life_days)
            }
        };
        let freq = (occurrences as f32 / self.saturation as f32).min(1.0);
        recency * freq
    }
}

/// M6.3: retrieval-reinforced salience (RFC 0007 §6). The `chunk_stats`-backed
/// impl the [`Salience`] doc-comment promised — feeds the real, lazily-decayed
/// `weight` accrued by `reinforce` (every `memory_search` hit) and ambient
/// injection into the **demote** decision, so a sustained-*dormant* curated fact
/// is evicted before a recently-recalled one.
///
/// **Source-aware split** (issue #294). The two consolidation loops feed
/// disjoint sources: *promote* only ever scores `Daily` chunks, *demote* only
/// `Curated` chunks. So this type routes by [`MemorySource`]:
///
/// - **Curated** (the demote side): score by the chunk's decayed `weight`. Real
///   usage now ranks eviction. Before this, every curated chunk scored an
///   identical `recency(1.0) × freq(0) = 0` — `occurrences` counts *distinct
///   daily-log days* and a curated key is never in that map — so demotion evicted
///   by array index, not importance.
/// - **Daily** (the promote side): delegate verbatim to
///   [`RecencyFrequencySalience`]. Promote candidates are never-recalled by
///   construction (`reinforce` only creates a `chunk_stats` row on a search hit),
///   so `access_count`/`weight` is ~always absent there and must not drag
///   promotion down. Promotion stays on recency × frequency.
///
/// A curated key absent from the weight map reads `1.0` (never-recalled ⇒ not
/// dormant), matching the `chunk_stats_snapshot` / `effective_stats` invariant
/// (RFC 0007 §3): an absent row is treated as full weight, never as zero.
pub struct ChunkStatsSalience {
    /// `chunk_key` → lazily-decayed, pin-aware effective weight at pass time.
    /// Absent key ⇒ treated as `1.0`.
    weights: HashMap<String, f32>,
    /// `chunk_key`s whose `suppress_promotion` flag is set (#1292 block A). A
    /// `Daily` chunk in this set scores `0.0` — a failed session/goal outcome
    /// keeps its touched chunks out of curated until a later success clears the
    /// flag. Curated (demote-side) scoring is unaffected.
    suppressed: HashSet<String>,
    /// Delegate for `Daily` (promote-side) scoring.
    daily: RecencyFrequencySalience,
}

impl ChunkStatsSalience {
    /// Build from a `chunk_key → effective weight` map (curated keys only need
    /// be present; absent ⇒ `1.0`). See [`Memory::chunk_stats_salience`] for the
    /// production constructor that gathers the snapshot from a live index.
    pub fn new(weights: HashMap<String, f32>) -> Self {
        Self::with_suppressed(weights, HashSet::new())
    }

    /// Build with an explicit set of promotion-suppressed `chunk_key`s (#1292
    /// block A). [`new`](Self::new) is this with an empty set.
    ///
    /// The keys are stored reduced to their [`content_key_of`] identity, so
    /// suppression follows the *fact*, not the daily date it was flagged under
    /// (#1292 F3): a fact buried by a failed run stays buried even when it recurs
    /// under a later date — matching the `content_key` granularity the promote
    /// loop already dedups on.
    pub fn with_suppressed(weights: HashMap<String, f32>, suppressed: HashSet<String>) -> Self {
        let suppressed = suppressed
            .iter()
            .map(|k| content_key_of(k).to_string())
            .collect();
        Self {
            weights,
            suppressed,
            daily: RecencyFrequencySalience::default(),
        }
    }
}

impl Salience<MemoryChunk> for ChunkStatsSalience {
    fn score(&self, chunk: &MemoryChunk, occurrences: u32) -> f32 {
        match &chunk.source {
            // Demote side: rank by real, decayed usage weight.
            MemorySource::Curated => *self.weights.get(&chunk_key(chunk)).unwrap_or(&1.0),
            // Promote side: unchanged recency × frequency, unless a failed
            // outcome suppressed this fact — then it must not be promoted.
            // Matched by content identity so an identical note under a newer
            // daily date is still recognised as the suppressed fact (#1292 F3).
            MemorySource::Daily { .. } => {
                if self.suppressed.contains(content_key_of(&chunk_key(chunk))) {
                    0.0
                } else {
                    self.daily.score(chunk, occurrences)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Consolidation pass (issue #223 P2, RFC 0007 sec 6)
// ---------------------------------------------------------------------------

/// Minimum [`Salience`] score for a recurring daily fact to be promoted into
/// curated Markdown. With the default recency x frequency salience this means a
/// fact must be both recent and seen on more than one day -- a one-off daily
/// note scores ~`1/saturation` and stays in the daily log.
const PROMOTION_SCORE_CUTOFF: f32 = 0.5;

/// Outcome of a [`Memory::consolidate`] pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationReport {
    /// Whether the pass changed anything (and thus rewrote curated Markdown).
    /// A re-run with nothing to do returns `ran = false` and touches no files.
    pub ran: bool,
    /// Near-identical curated facts collapsed into one.
    pub merged: usize,
    /// Recurring, recent daily facts lifted into curated.
    pub promoted: usize,
    /// Lowest-salience curated facts evicted back to the daily log.
    pub demoted: usize,
    /// Curated file size before the pass, in bytes.
    pub bytes_before: usize,
    /// Curated file size after the pass (== `bytes_before` when `!ran`).
    pub bytes_after: usize,
    /// Supersession edges detected this pass (memory sifting C1, #1293): each
    /// `(from_key, to_key)` says the curated chunk `from_key` was superseded by
    /// the freshly promoted curated chunk `to_key` (same heading, changed text).
    /// The caller records these into `chunk_links` after reindex, once both keys
    /// are live. Empty unless a promotion superseded exactly one same-heading fact.
    pub superseded: Vec<(String, String)>,
}

/// Map a chunk's (stripped) heading to a canonical [`Stratum`], if it is one.
/// `chunk.heading` holds the heading text without leading `#`s (e.g. `Identity`),
/// so we compare against each stratum's `## Heading` line.
fn stratum_for_heading(heading: &str) -> Option<Stratum> {
    let line = format!("## {}", heading.trim());
    CANONICAL_STRATA.into_iter().find(|s| s.heading() == line)
}

/// The body of a chunk with its leading heading line removed. `chunk_markdown`
/// prepends the raw heading line to a section's `text`, so when the first line
/// is that heading we drop it; otherwise (a preamble chunk, or a non-first
/// window of a large split section) the whole trimmed text is the body.
fn body_without_heading(chunk: &MemoryChunk) -> &str {
    let text = chunk.text.trim();
    let Some(h) = chunk.heading.as_deref() else {
        return text;
    };
    let mut parts = text.splitn(2, '\n');
    let first = parts.next().unwrap_or("");
    if first.trim_start().starts_with('#')
        && first.trim_start().trim_start_matches('#').trim() == h.trim()
    {
        parts.next().unwrap_or("").trim()
    } else {
        text
    }
}

/// Rebuild curated Markdown from the kept chunks, **grouping** them under the
/// canonical strata headings (RFC 0008 §4, issue #254): a preamble (heading-less
/// chunks) first, then `## Identity` / `## Patterns` / `## Focus` in that fixed
/// order, then any freeform headings in first-seen order. Chunks that share a
/// heading collapse into a single section (no duplicate `## Identity`), and
/// freeform headings are preserved rather than dropped.
///
/// This is a **fixpoint**: re-chunking the output yields one chunk per heading,
/// and re-grouping is the identity — which is what keeps the consolidate re-run
/// idempotent and `chunk_key` stable.
fn render_curated(chunks: &[MemoryChunk]) -> String {
    let mut preamble: Vec<&str> = Vec::new();
    // Canonical strata: heading line -> joined bodies.
    let mut canonical: HashMap<&'static str, Vec<&str>> = HashMap::new();
    // Freeform headings: first-seen order + (original heading line, bodies).
    let mut freeform_order: Vec<String> = Vec::new();
    let mut freeform: HashMap<String, (String, Vec<&str>)> = HashMap::new();

    for c in chunks {
        let body = body_without_heading(c);
        match c.heading.as_deref() {
            None => {
                if !body.is_empty() {
                    preamble.push(body);
                }
            }
            Some(h) => {
                if let Some(strat) = stratum_for_heading(h) {
                    canonical.entry(strat.heading()).or_default().push(body);
                } else {
                    let original = c.text.trim().lines().next().unwrap_or("").trim();
                    let entry = freeform.entry(h.to_string()).or_insert_with(|| {
                        freeform_order.push(h.to_string());
                        (original.to_string(), Vec::new())
                    });
                    entry.1.push(body);
                }
            }
        }
    }

    let join_section = |heading: Option<&str>, bodies: &[&str]| -> String {
        let body = bodies
            .iter()
            .filter(|b| !b.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        match (heading, body.is_empty()) {
            (None, _) => body,
            (Some(h), true) => h.to_string(),
            (Some(h), false) => format!("{h}\n{body}"),
        }
    };

    let mut sections: Vec<String> = Vec::new();
    let pre = preamble.join("\n\n");
    if !pre.is_empty() {
        sections.push(pre);
    }
    for strat in CANONICAL_STRATA {
        if let Some(bodies) = canonical.get(strat.heading()) {
            sections.push(join_section(Some(strat.heading()), bodies));
        }
    }
    for key in &freeform_order {
        let (hline, bodies) = &freeform[key];
        sections.push(join_section(Some(hline), bodies));
    }

    let mut out = sections.join("\n\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Re-join the overlapping line-windows that [`chunk_markdown`] emits for a
/// section larger than `CHUNK_TARGET_BYTES` back into one whole-section chunk.
///
/// Windowing is an index/embedding concern (focused vectors); the rewrite path
/// must see each section's *whole* body. Otherwise [`render_curated`] re-joins
/// the windows' bodies and re-emits their ~15% overlap as duplicated lines, and
/// re-chunking the larger output never reaches a fixpoint (#254).
///
/// Windows of one section arrive consecutively, share `(source, path, heading)`,
/// and have overlapping or adjacent line spans -- exactly the run folded here.
/// Distinct same-heading sections are separated by other chunks, so they stay
/// apart (and `render_curated` collapses them by heading, as intended).
fn coalesce_windows(chunks: Vec<MemoryChunk>) -> Vec<MemoryChunk> {
    let mut out: Vec<MemoryChunk> = Vec::with_capacity(chunks.len());
    for c in chunks {
        if let Some(last) = out.last_mut() {
            // A true window genuinely *overlaps* the previous chunk's line span
            // (chunk_markdown carries ~15% of bytes into the next window). Two
            // distinct same-heading sections are only adjacent (blank line
            // between them), never overlapping -- so require strict overlap to
            // avoid folding separate sections that `render_curated` should keep
            // apart and collapse by heading.
            let is_window_of_last = last.source == c.source
                && last.path == c.path
                && last.heading == c.heading
                && c.line_start <= last.line_end;
            if is_window_of_last {
                // Drop the leading lines of `c` already covered by `last`, then
                // append only the new tail so the overlapping lines appear once.
                let skip = (last.line_end + 1).saturating_sub(c.line_start) as usize;
                let tail: Vec<&str> = c.text.lines().skip(skip).collect();
                if !tail.is_empty() {
                    last.text.push('\n');
                    last.text.push_str(&tail.join("\n"));
                }
                last.line_end = last.line_end.max(c.line_end);
                continue;
            }
        }
        out.push(c);
    }
    out
}

impl crate::Memory {
    /// Run the consolidation pass (issue #223 P2; RFC 0006 sec 7.3 / RFC 0007 sec 6).
    ///
    /// Decides merge / promote / demote entirely in memory, then performs side
    /// effects: daily-log appends for any demoted entries, followed by a single
    /// atomic curated rewrite. **Idempotent** -- a re-run with nothing to change
    /// returns `ran = false` and writes nothing.
    ///
    /// - **Merge**: collapse curated facts with identical content.
    /// - **Promote**: lift recurring, recent daily facts (score past
    ///   [`PROMOTION_SCORE_CUTOFF`]) into curated.
    /// - **Demote**: when `evict_to_budget` is set and curated still exceeds
    ///   `injection_budget_bytes`, append the lowest-salience curated facts back
    ///   to today's daily log (history, still FTS-indexed) and drop them from
    ///   curated. Never deletes -- no data loss.
    ///
    /// Does **not** reindex: the caller owns the index (and the blocking embed
    /// call it may make). See `MemoryConsolidateTool`.
    pub fn consolidate(&self, salience: &dyn Salience<MemoryChunk>) -> Result<ConsolidationReport> {
        let bytes_before = std::fs::metadata(self.curated_path())
            .map(|m| m.len())
            .unwrap_or(0) as usize;
        let mut report = ConsolidationReport {
            bytes_before,
            bytes_after: bytes_before,
            ..Default::default()
        };
        if !self.is_enabled() {
            return Ok(report);
        }

        // Gather and split curated vs daily. Coalesce windowed sections back to
        // whole bodies first so the rewrite path never re-emits window overlap.
        let mut curated: Vec<MemoryChunk> = Vec::new();
        let mut daily: Vec<MemoryChunk> = Vec::new();
        for c in coalesce_windows(self.all_chunks()) {
            match c.source {
                MemorySource::Curated => curated.push(c),
                MemorySource::Daily { .. } => daily.push(c),
            }
        }

        // Recurrence is counted per distinct day so a fact repeated within a
        // single daily log doesn't inflate its frequency.
        let mut daily_days: HashMap<String, HashSet<NaiveDate>> = HashMap::new();
        for c in &daily {
            if let MemorySource::Daily { date } = c.source {
                daily_days.entry(content_key(c)).or_default().insert(date);
            }
        }
        let occurrences =
            |key: &str| -> u32 { daily_days.get(key).map(|d| d.len() as u32).unwrap_or(0) };

        // --- Merge: keep the first of each duplicate curated content. ---
        let mut seen: HashSet<String> = HashSet::new();
        let mut kept: Vec<MemoryChunk> = Vec::new();
        for c in curated {
            if seen.insert(content_key(&c)) {
                kept.push(c);
            } else {
                report.merged += 1;
            }
        }
        let mut curated = kept;
        let curated_keys: HashSet<String> = curated.iter().map(content_key).collect();

        // Supersession detection (memory sifting C1, #1293). Map each distinctive
        // heading to the curated chunk keys under it, so a promotion whose heading
        // matches exactly one existing curated fact can be recorded as superseding
        // it. Canonical stratum headings (Identity/Patterns/Focus) are structural
        // containers for bare bullets, not topics, so they are excluded -- only a
        // distinctive sub-heading names "the same thing".
        let mut curated_by_heading: HashMap<String, Vec<String>> = HashMap::new();
        for c in &curated {
            if let Some(h) = c.heading.as_deref() {
                if stratum_for_heading(h).is_none() {
                    curated_by_heading
                        .entry(h.to_string())
                        .or_default()
                        .push(chunk_key(c));
                }
            }
        }

        // --- Promote: recurring, recent daily facts not already curated. Pick
        //     the highest-scoring (freshest) copy per content key. ---
        let mut best: HashMap<String, (f32, MemoryChunk)> = HashMap::new();
        for c in &daily {
            let key = content_key(c);
            if curated_keys.contains(&key) {
                continue;
            }
            let score = salience.score(c, occurrences(&key));
            if score < PROMOTION_SCORE_CUTOFF {
                continue;
            }
            match best.get(&key) {
                Some((s, _)) if *s >= score => {}
                _ => {
                    best.insert(key, (score, c.clone()));
                }
            }
        }
        let mut promoted_keys: Vec<String> = best.keys().cloned().collect();
        promoted_keys.sort();
        for key in promoted_keys {
            let (_, mut chunk) = best.remove(&key).unwrap();
            chunk.source = MemorySource::Curated;
            // Supersession: if this promotion's heading matches exactly one
            // existing curated fact (1:1 guard), record the old fact as superseded
            // by the new one. Ambiguous headings (>1 curated chunk) are skipped --
            // we cannot tell which was replaced, so never mis-record.
            if let Some(h) = chunk.heading.as_deref() {
                if let Some(olds) = curated_by_heading.get(h) {
                    if olds.len() == 1 {
                        report.superseded.push((olds[0].clone(), chunk_key(&chunk)));
                    }
                }
            }
            curated.push(chunk);
            report.promoted += 1;
        }

        // --- Demote: hard-evict lowest-salience curated facts to fit the budget
        //     (config-gated). Append each back to today's daily log first. ---
        if self.config().evict_to_budget {
            let budget = self.config().injection_budget_bytes;
            while render_curated(&curated).len() > budget && curated.len() > 1 {
                let mut worst = 0usize;
                let mut worst_score = f32::INFINITY;
                for (i, c) in curated.iter().enumerate() {
                    let score = salience.score(c, occurrences(&content_key(c)));
                    if score < worst_score {
                        worst_score = score;
                        worst = i;
                    }
                }
                let evicted = curated.remove(worst);
                self.write(&evicted.text, WriteTarget::Daily)?;
                report.demoted += 1;
            }
        }

        // Group the kept chunks under the canonical strata headings (#254). A
        // pure regroup (no merge/promote/demote) still rewrites when it changes
        // the file, so a scattered curated file self-heals into who/how/what
        // shape. Grouping is a fixpoint, so the next run finds nothing to do.
        let rendered = render_curated(&curated);
        let regrouped = rendered != read_lenient(&self.curated_path());

        report.ran = report.merged > 0 || report.promoted > 0 || report.demoted > 0 || regrouped;
        if report.ran {
            self.rewrite_curated(&rendered)?;
            report.bytes_after = std::fs::metadata(self.curated_path())
                .map(|m| m.len())
                .unwrap_or(0) as usize;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
