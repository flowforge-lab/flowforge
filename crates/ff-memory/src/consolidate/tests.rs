use super::*;
use chrono::NaiveDate;
use std::path::PathBuf;

fn make_chunk(source: MemorySource, heading: Option<&str>, text: &str) -> MemoryChunk {
    MemoryChunk {
        id: 0,
        source,
        path: PathBuf::from("MEMORY.md"),
        heading: heading.map(String::from),
        text: text.to_string(),
        line_start: 1,
        line_end: 1,
        embedding: None,
    }
}

#[test]
fn chunk_key_stable_across_whitespace_changes() {
    let c1 = make_chunk(
        MemorySource::Curated,
        Some("Prefs"),
        "likes rust\nhates yaml",
    );
    let c2 = make_chunk(
        MemorySource::Curated,
        Some("Prefs"),
        "  likes rust  \n  hates yaml  \n",
    );
    assert_eq!(chunk_key(&c1), chunk_key(&c2));
}

#[test]
fn chunk_key_changes_on_content_change() {
    let c1 = make_chunk(MemorySource::Curated, Some("Prefs"), "likes rust");
    let c2 = make_chunk(MemorySource::Curated, Some("Prefs"), "likes python");
    assert_ne!(chunk_key(&c1), chunk_key(&c2));
}

#[test]
fn chunk_key_differs_by_source() {
    let c1 = make_chunk(MemorySource::Curated, Some("A"), "same text");
    let c2 = make_chunk(
        MemorySource::Daily {
            date: NaiveDate::from_ymd_opt(2026, 6, 19).unwrap(),
        },
        Some("A"),
        "same text",
    );
    assert_ne!(chunk_key(&c1), chunk_key(&c2));
}

#[test]
fn chunk_key_differs_by_heading() {
    let c1 = make_chunk(MemorySource::Curated, Some("A"), "same text");
    let c2 = make_chunk(MemorySource::Curated, Some("B"), "same text");
    assert_ne!(chunk_key(&c1), chunk_key(&c2));
}

#[test]
fn chunk_key_stable_across_line_number_shifts() {
    let mut c1 = make_chunk(MemorySource::Curated, Some("Prefs"), "likes rust");
    c1.line_start = 1;
    c1.line_end = 1;
    let mut c2 = make_chunk(MemorySource::Curated, Some("Prefs"), "likes rust");
    c2.line_start = 42;
    c2.line_end = 42;
    assert_eq!(chunk_key(&c1), chunk_key(&c2));
}

#[test]
fn salience_curated_scores_high_with_occurrences() {
    let s = RecencyFrequencySalience::default();
    let c = make_chunk(MemorySource::Curated, Some("H"), "fact");
    // 3 occurrences saturates frequency to 1.0; curated recency = 1.0
    assert!((s.score(&c, 3) - 1.0).abs() < 0.001);
}

#[test]
fn salience_old_daily_scores_low() {
    let s = RecencyFrequencySalience::default();
    let old_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let c = make_chunk(MemorySource::Daily { date: old_date }, Some("H"), "fact");
    let score = s.score(&c, 3);
    assert!(
        score < 0.01,
        "old daily chunk should score very low: {score}"
    );
}

#[test]
fn salience_zero_occurrences_is_zero() {
    let s = RecencyFrequencySalience::default();
    let c = make_chunk(MemorySource::Curated, Some("H"), "fact");
    assert_eq!(s.score(&c, 0), 0.0);
}

// -- consolidation pass (issue #223 P2) --

use crate::{Memory, MemoryConfig};

fn mem_with(root: &std::path::Path, budget: usize, evict: bool) -> Memory {
    let config = MemoryConfig {
        injection_budget_bytes: budget,
        evict_to_budget: evict,
        ..Default::default()
    };
    Memory::new(root, config)
}

fn days_ago(n: i64) -> NaiveDate {
    chrono::Local::now().date_naive() - chrono::Duration::days(n)
}

fn write_daily(m: &Memory, date: NaiveDate, content: &str) {
    let path = m.daily_path(date);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

#[test]
fn consolidate_merges_duplicate_curated_facts() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, true);
    m.rewrite_curated("# Prefs\nlikes rust\n\n# Prefs\nlikes rust\n")
        .unwrap();

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.merged, 1, "the duplicate section should collapse");
    assert!(report.ran);
    let curated = std::fs::read_to_string(m.curated_path()).unwrap();
    assert_eq!(
        curated.matches("likes rust").count(),
        1,
        "only one copy left"
    );
}

#[test]
fn consolidate_promotes_recurring_daily_fact() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, true);
    // Same fact captured on three recent days -> recurring -> promote.
    for n in 0..3 {
        write_daily(&m, days_ago(n), "# Project\nuses tauri\n");
    }

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(
        report.promoted, 1,
        "recurring daily fact should be promoted"
    );
    assert!(report.ran);
    let curated = std::fs::read_to_string(m.curated_path()).unwrap();
    assert!(
        curated.contains("uses tauri"),
        "promoted into curated: {curated}"
    );
}

#[test]
fn consolidate_skips_one_off_daily_fact() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, true);
    write_daily(&m, days_ago(0), "# One\nseen once\n");

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.promoted, 0, "a one-off fact stays in the daily log");
    assert!(!report.ran);
}

#[test]
fn consolidate_demotes_to_daily_when_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 40, true);
    m.rewrite_curated("# A\nalpha fact\n\n# B\nbeta fact\n\n# C\ngamma fact\n")
        .unwrap();

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert!(report.demoted >= 1, "over-budget curated should demote");
    assert!(report.ran);
    // Demoted entries are appended to TODAY's daily log (history, not deleted).
    let today = std::fs::read_to_string(m.daily_path(days_ago(0))).unwrap();
    assert!(
        today.contains("fact"),
        "evicted text lands in daily: {today}"
    );
    assert!(report.bytes_after < report.bytes_before, "curated shrank");
}

#[test]
fn consolidate_demote_gated_off_by_config() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 40, false);
    let curated = "# A\nalpha fact\n\n# B\nbeta fact\n\n# C\ngamma fact\n";
    m.rewrite_curated(curated).unwrap();

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.demoted, 0, "eviction is disabled");
    assert!(
        !report.ran,
        "nothing to do when eviction is off and no merge/promote"
    );
    assert_eq!(std::fs::read_to_string(m.curated_path()).unwrap(), curated);
}

#[test]
fn consolidate_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, true);
    m.rewrite_curated("# Prefs\nlikes rust\n\n# Prefs\nlikes rust\n")
        .unwrap();

    let first = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert!(first.ran);
    let after_first = std::fs::read_to_string(m.curated_path()).unwrap();

    let second = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert!(!second.ran, "re-run must be a no-op");
    assert_eq!(second.merged + second.promoted + second.demoted, 0);
    assert_eq!(
        std::fs::read_to_string(m.curated_path()).unwrap(),
        after_first
    );
}

// --- #254: strata grouping in render_curated -------------------------

fn curated_chunk(heading: Option<&str>, text: &str) -> MemoryChunk {
    make_chunk(MemorySource::Curated, heading, text)
}

#[test]
fn render_curated_groups_canonical_strata_in_fixed_order() {
    // Deliberately out of order: Focus before Identity.
    let chunks = vec![
        curated_chunk(Some("Focus"), "## Focus\nmaps work"),
        curated_chunk(Some("Identity"), "## Identity\nL5 SDE"),
        curated_chunk(Some("Patterns"), "## Patterns\nprefers Python"),
    ];
    assert_eq!(
        render_curated(&chunks),
        "## Identity\nL5 SDE\n\n## Patterns\nprefers Python\n\n## Focus\nmaps work\n"
    );
}

#[test]
fn render_curated_collapses_duplicate_heading_sections() {
    let chunks = vec![
        curated_chunk(Some("Identity"), "## Identity\nL5 SDE"),
        curated_chunk(Some("Identity"), "## Identity\nbased in Austin"),
    ];
    let out = render_curated(&chunks);
    assert_eq!(out, "## Identity\nL5 SDE\nbased in Austin\n");
    assert_eq!(
        out.matches("## Identity").count(),
        1,
        "no duplicate heading"
    );
}

#[test]
fn render_curated_preserves_freeform_headings_after_canonical() {
    let chunks = vec![
        curated_chunk(Some("Projects"), "## Projects\nflowforge"),
        curated_chunk(Some("Identity"), "## Identity\nL5 SDE"),
    ];
    assert_eq!(
        render_curated(&chunks),
        "## Identity\nL5 SDE\n\n## Projects\nflowforge\n"
    );
}

#[test]
fn render_curated_keeps_preamble_first() {
    let chunks = vec![
        curated_chunk(None, "intro line"),
        curated_chunk(Some("Identity"), "## Identity\nL5 SDE"),
    ];
    assert_eq!(
        render_curated(&chunks),
        "intro line\n\n## Identity\nL5 SDE\n"
    );
}

#[test]
fn render_curated_is_a_fixpoint() {
    let chunks = vec![
        curated_chunk(Some("Focus"), "## Focus\nmaps work"),
        curated_chunk(Some("Identity"), "## Identity\nL5 SDE"),
        curated_chunk(Some("Identity"), "## Identity\nbased in Austin"),
    ];
    let once = render_curated(&chunks);
    let rechunked = crate::chunk_markdown(
        &once,
        MemorySource::Curated,
        std::path::Path::new("MEMORY.md"),
    );
    assert_eq!(render_curated(&rechunked), once, "regroup must be stable");
}

#[test]
fn consolidate_groups_scattered_strata_into_canonical_shape() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, false);
    m.rewrite_curated(
        "## Focus\nmaps work\n\n## Identity\nL5 SDE\n\n## Identity\nbased in Austin\n",
    )
    .unwrap();

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert!(report.ran, "a scattered curated file should regroup");

    let curated = std::fs::read_to_string(m.curated_path()).unwrap();
    assert_eq!(
        curated,
        "## Identity\nL5 SDE\nbased in Austin\n\n## Focus\nmaps work\n"
    );
    assert_eq!(curated.matches("## Identity").count(), 1);

    // Second run: already grouped -> no-op.
    let second = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert!(!second.ran, "re-run on grouped file must be a no-op");
    assert_eq!(std::fs::read_to_string(m.curated_path()).unwrap(), curated);
}

#[test]
fn consolidate_disabled_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig {
        enabled: false,
        ..Default::default()
    };
    let m = Memory::new(dir.path(), config);
    m.rewrite_curated("# Prefs\nlikes rust\n\n# Prefs\nlikes rust\n")
        .unwrap();
    let before = std::fs::read_to_string(m.curated_path()).unwrap();

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert!(!report.ran);
    assert_eq!(std::fs::read_to_string(m.curated_path()).unwrap(), before);
}

#[test]
fn consolidate_coalesces_windowed_section_without_duplication() {
    // A single canonical section larger than CHUNK_TARGET_BYTES is windowed
    // by chunk_markdown into overlapping chunks. Consolidation must rebuild
    // the whole section once -- not re-emit the ~15% overlap -- and reach a
    // fixpoint even though the section stays under the injection budget.
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 8192, false);

    let mut body = String::from("## Patterns\n");
    for i in 0..120 {
        body.push_str(&format!(
            "- fact number {i:03} with enough descriptive text to grow the section\n"
        ));
    }
    assert!(
        body.len() > 2048 && body.len() < 8192,
        "section must be windowed (>2KB) yet under budget, got {}",
        body.len()
    );
    m.rewrite_curated(&body).unwrap();

    // Already canonical and whole: the windowed read must not be mistaken
    // for a regroup, and nothing may be duplicated.
    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert!(
        !report.ran,
        "a whole, already-grouped section must be a no-op even when windowed"
    );

    let curated = std::fs::read_to_string(m.curated_path()).unwrap();
    assert_eq!(
        curated, body,
        "file must be byte-identical (no overlap re-emitted)"
    );
    let max_dup = {
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for line in curated.lines().filter(|l| l.starts_with("- fact")) {
            *counts.entry(line).or_default() += 1;
        }
        counts.values().copied().max().unwrap_or(0)
    };
    assert_eq!(max_dup, 1, "no fact line may appear more than once");

    // Re-run is a fixpoint.
    let second = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert!(
        !second.ran,
        "re-run on the windowed section must stay a no-op"
    );
    assert_eq!(std::fs::read_to_string(m.curated_path()).unwrap(), curated);
}

// --- #294: retrieval-reinforced demotion (ChunkStatsSalience) --------------

use crate::index::{Fts5Index, MemoryIndex, ScoredChunk};
use crate::DecayConfig;

/// Reindex the curated file into `idx` and reinforce a single chunk (by content
/// substring) at `at_ms`, so decay can drive it dormant relative to `now_ms`.
fn reinforce_curated_chunk(idx: &Fts5Index, m: &Memory, needle: &str, at_ms: i64) {
    let raw = std::fs::read_to_string(m.curated_path()).unwrap();
    let chunks = crate::chunk_markdown(&raw, MemorySource::Curated, &m.curated_path());
    idx.reindex(&chunks).unwrap();
    let hit = chunks
        .into_iter()
        .find(|c| c.text.contains(needle))
        .map(|chunk| ScoredChunk {
            chunk,
            score: 1.0,
            weight: 1.0,
            last_accessed_ms: None,
        })
        .expect("needle not found among curated chunks");
    idx.reinforce_at(&[hit], at_ms).unwrap();
}

fn decay_on() -> DecayConfig {
    DecayConfig {
        enabled: true,
        ..Default::default()
    }
}

/// Three curated facts A, B, C over budget by one. C has been dormant for ~150
/// idle days (weight decayed well below the dormant threshold); A and B were
/// never recalled (absent stats row ⇒ weight 1.0). `ChunkStatsSalience` must
/// evict **C** (the genuinely stale fact), not the array-first chunk.
#[test]
fn demote_evicts_dormant_curated_with_chunk_stats_salience() {
    let dir = tempfile::tempdir().unwrap();
    // Budget fits two of the three ~15-byte curated bodies, forcing one eviction.
    let m = mem_with(dir.path(), 44, true);
    m.rewrite_curated("# A\nalpha fact\n\n# B\nbeta fact\n\n# C\ngamma fact\n")
        .unwrap();

    let now_ms: i64 = 10_000_000_000_000;
    let idx = Fts5Index::open_in_memory().unwrap().with_decay(decay_on());
    // Reinforce C 150 days in the past: 0.98^150 ≈ 0.048 << 0.25 threshold.
    let past = now_ms - 150 * 86_400_000;
    reinforce_curated_chunk(&idx, &m, "gamma fact", past);

    let salience = m.chunk_stats_salience(&idx, now_ms);
    let report = m.consolidate(&salience, None).unwrap();

    assert_eq!(report.demoted, 1, "exactly one over-budget chunk demoted");
    let curated = std::fs::read_to_string(m.curated_path()).unwrap();
    assert!(
        !curated.contains("gamma fact"),
        "dormant C must be evicted, got:\n{curated}"
    );
    assert!(
        curated.contains("alpha fact") && curated.contains("beta fact"),
        "recently-relevant A and B must be retained, got:\n{curated}"
    );
}

/// Companion: the mechanical default (`RecencyFrequencySalience`) scores every
/// curated chunk an identical `1.0 × 0 = 0` (curated keys are never in the
/// daily-day map), so it evicts the **array-first** chunk (A) regardless of real
/// usage. This pins the pre-#294 behaviour the new salience corrects.
#[test]
fn demote_with_mechanical_salience_evicts_array_first_not_dormant() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 44, true);
    m.rewrite_curated("# A\nalpha fact\n\n# B\nbeta fact\n\n# C\ngamma fact\n")
        .unwrap();

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.demoted, 1);
    let curated = std::fs::read_to_string(m.curated_path()).unwrap();
    assert!(
        !curated.contains("alpha fact"),
        "mechanical salience evicts the array-first chunk A, got:\n{curated}"
    );
}

// --- Promotion suppression (#1292 block A) -----------------------------------

#[test]
fn suppression_follows_content_across_daily_dates() {
    // #1292 F3: a fact flagged on one day must stay suppressed when it recurs
    // under a later daily date. The keys differ (date-prefixed), but the content
    // identity — the granularity the promote loop dedups on — is the same.
    let old_day: chrono::NaiveDate = "2026-08-20".parse().unwrap();
    let new_day: chrono::NaiveDate = "2026-08-31".parse().unwrap();
    let flagged_old = make_chunk(MemorySource::Daily { date: old_day }, Some("H"), "dead end");
    let recur_new = make_chunk(MemorySource::Daily { date: new_day }, Some("H"), "dead end");
    assert_ne!(
        chunk_key(&flagged_old),
        chunk_key(&recur_new),
        "different dates -> different chunk_key"
    );

    let mut flagged = HashSet::new();
    flagged.insert(chunk_key(&flagged_old));
    let s = ChunkStatsSalience::with_suppressed(HashMap::new(), flagged);

    assert_eq!(
        s.score(&recur_new, 3),
        0.0,
        "the same fact under a newer date is still recognised as suppressed"
    );
}

#[test]
fn suppressed_daily_chunk_scores_zero() {
    // A fresh daily chunk normally scores high; suppression must force it to 0
    // so promotion skips it, without touching the unsuppressed sibling.
    let today = chrono::Local::now().date_naive();
    let suppressed = make_chunk(MemorySource::Daily { date: today }, Some("H"), "dead end");
    let live = make_chunk(MemorySource::Daily { date: today }, Some("H"), "good lead");

    let mut flagged = HashSet::new();
    flagged.insert(chunk_key(&suppressed));
    let s = ChunkStatsSalience::with_suppressed(HashMap::new(), flagged);

    assert_eq!(
        s.score(&suppressed, 3),
        0.0,
        "suppressed daily must score 0"
    );
    assert!(
        s.score(&live, 3) > 0.5,
        "unsuppressed daily keeps its recency×frequency score"
    );
}

// --- Supersession detection (memory sifting C1, #1293) ---

#[test]
fn consolidate_records_supersession_when_heading_uniquely_matches() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, false);
    // One existing curated fact under a distinctive heading.
    m.rewrite_curated("# Project\nold approach\n").unwrap();
    // A recurring daily fact under the SAME heading, different text.
    for n in 0..3 {
        write_daily(&m, days_ago(n), "# Project\nnew approach\n");
    }

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.promoted, 1);
    assert_eq!(
        report.superseded.len(),
        1,
        "the promotion supersedes the one same-heading curated fact"
    );

    // from = old curated key, to = new promoted (now-curated) key; they differ
    // (different text) and are both curated-sourced.
    let Supersession {
        superseded_key: old_key,
        superseded_by: new_key,
    } = report.superseded[0].clone();
    assert_ne!(old_key, new_key);
    assert!(old_key.starts_with("curated:"), "from key: {old_key}");
    assert!(new_key.starts_with("curated:"), "to key: {new_key}");

    // Recording the edges via the ff-memory seam is queryable in both directions.
    let idx = Fts5Index::open_in_memory().unwrap();
    let recorded = m.record_supersession_edges(&idx, &report);
    assert_eq!(recorded, 1);
    assert_eq!(idx.links_from(&old_key).unwrap()[0].to_key, new_key);
    assert_eq!(idx.links_to(&new_key).unwrap()[0].from_key, old_key);
}

#[test]
fn consolidate_supersedes_chunk_present_in_index_but_absent_from_batch() {
    // Spec (#1293): detection must query the full index, not just this batch. An
    // old curated fact that lingers in the index but has fallen out of the
    // current curated file (e.g. demoted, then the same topic re-promoted) is
    // still a valid supersession target.
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, false);

    // The index holds an old curated fact under a distinctive heading, but the
    // on-disk curated file (this pass's batch) does NOT contain it.
    let idx = Fts5Index::open_in_memory().unwrap();
    let old = MemoryChunk {
        id: 0,
        source: crate::MemorySource::Curated,
        path: m.curated_path(),
        heading: Some("Project".to_string()),
        text: "# Project\nold approach\n".to_string(),
        line_start: 1,
        line_end: 2,
        embedding: None,
    };
    idx.reindex(std::slice::from_ref(&old)).unwrap();
    let old_key = chunk_key(&old);

    // Curated file is empty; a recurring daily fact under the same heading.
    m.rewrite_curated("").unwrap();
    for n in 0..3 {
        write_daily(&m, days_ago(n), "# Project\nnew approach\n");
    }

    // Batch-only detection (None) cannot see the old fact -> no edge.
    let batch_only = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();
    assert_eq!(batch_only.promoted, 1);
    assert!(
        batch_only.superseded.is_empty(),
        "batch-local detection cannot see the index-only fact"
    );

    // Re-stage and run again, this time consulting the index -> edge is found.
    m.rewrite_curated("").unwrap();
    for n in 0..3 {
        write_daily(&m, days_ago(n), "# Project\nnew approach\n");
    }
    let report = m
        .consolidate(&RecencyFrequencySalience::default(), Some(&idx))
        .unwrap();

    assert_eq!(
        report.superseded.len(),
        1,
        "index-wide match yields the edge"
    );
    let edge = &report.superseded[0];
    assert_eq!(edge.superseded_key, old_key);
    assert_ne!(edge.superseded_by, old_key);
}

#[test]
fn consolidate_skips_supersession_when_heading_is_ambiguous() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, false);
    // TWO curated facts share the same heading -> cannot tell which is superseded.
    m.rewrite_curated("# Notes\nfirst note\n\n# Notes\nsecond note\n")
        .unwrap();
    for n in 0..3 {
        write_daily(&m, days_ago(n), "# Notes\nthird note\n");
    }

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.promoted, 1);
    assert!(
        report.superseded.is_empty(),
        "ambiguous heading (>1 curated chunk) must not mis-record a supersession"
    );
}

#[test]
fn suppression_does_not_affect_curated_demotion() {
    // Suppression is a promote-side gate; a curated chunk with the same key must
    // still score by its weight (demote side), never forced to 0.
    let c = make_chunk(MemorySource::Curated, Some("H"), "curated fact");
    let mut flagged = HashSet::new();
    flagged.insert(chunk_key(&c));
    let mut weights = HashMap::new();
    weights.insert(chunk_key(&c), 0.8);
    let s = ChunkStatsSalience::with_suppressed(weights, flagged);

    assert_eq!(
        s.score(&c, 1),
        0.8,
        "curated (demote) scoring ignores the promote-side suppress flag"
    );
}

#[test]
fn consolidate_skips_supersession_for_stratum_container_heading() {
    let dir = tempfile::tempdir().unwrap();
    let m = mem_with(dir.path(), 4096, false);
    // A canonical stratum heading is a structural container for bare bullets,
    // not a topic; a promotion under it must not be treated as supersession.
    m.rewrite_curated("## Focus\n- shipping C1\n").unwrap();
    for n in 0..3 {
        write_daily(&m, days_ago(n), "## Focus\n- shipping C2\n");
    }

    let report = m
        .consolidate(&RecencyFrequencySalience::default(), None)
        .unwrap();

    assert_eq!(report.promoted, 1);
    assert!(
        report.superseded.is_empty(),
        "stratum container heading is not a supersession signal"
    );
}
