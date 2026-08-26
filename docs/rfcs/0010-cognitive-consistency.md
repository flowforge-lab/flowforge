# 0010 — Cognitive Consistency: Temporal Fact Tracking

- **Status:** Proposed — **partially superseded in implementation** (see note below)
- **Milestone:** _Open (see §11.1)_
- **Author:** tonytan4ever
- **Depends on:** RFC 0006 (Markdown source of truth, `MemoryChunk`, `chunk_key`,
  ambient injection, recall tools, consolidation §7.3), RFC 0007 (durable side-table
  pattern, reinforcement/decay), RFC 0008 (strata headings as fact homes)
- **Supersedes:** —
- **Tracking issue:** _TBD_

> **Implementation note (2026-08, #1293).** The "memory sifting + indexing" plan
> lands supersession first, but via a **generic `chunk_links` edge table**
> (`from_key, to_key, kind, created_at`) rather than this RFC's dedicated
> `chunk_history` interval table. The edge table is shared with wiki-link and
> co-occurrence edges (block C2, #1294), so it is the durable graph layer C
> depends on. C1 (#1293) records `kind='supersession'` edges during consolidate
> promotion, guarded by single-heading identity (1:1 heading match; ambiguous and
> stratum-container headings are skipped). The `valid_from/valid_to/reason`
> interval model and inspectability surface in §3–§8 below are **not** implemented
> by C1; treat them as a possible future enrichment layered on the edge table, not
> as the shipped design.


## 1. Summary & Goals

FlowForge memory today is **point-in-time**. When a curated fact is replaced — a job
title changes, a project ships, a preference flips — the prior assertion is silently
overwritten. The Markdown reflects only the latest belief; the recall chain that led
there is gone. RFC 0006 §7.3 (consolidation) and RFC 0007 (decay) both treat content
as a single live state, with no model of *what was true before*.

This RFC adds a **temporal layer** for curated facts. When a curated assertion is
superseded, the prior assertion is **time-bounded** (`valid_from` / `valid_to`) in a
derived side table rather than discarded. Memory gains history, and the recall chain
becomes traceable: "what did I believe about X, and when did that change?"

The first slice is deliberately small — **supersession capture only**. A full
subject/predicate/object relation graph and a `memory_evolution` recall tool are
named follow-on phases (§6) so this RFC can ship in a single milestone without
prejudging the larger graph design.

## 2. Principle: Derived & Advisory, Markdown Canonical

RFC 0006 §2 establishes the invariant this RFC must not violate: **the user's
Markdown files are the single source of truth**, and every other store (FTS5,
embeddings, `chunk_stats` from RFC 0007 §4) is a derived, rebuildable artifact.

The temporal layer follows the same rule:

- **Markdown is unchanged.** Supersession does not edit, annotate, or rewrite
  curated files. The current `MEMORY.md` continues to reflect the latest curated
  belief, exactly as today.
- **The history is a derived side table** (§4), rebuildable in principle from
  prior commits / consolidation logs, and lossy-acceptable: a wiped `~/.flowforge`
  costs the user the audit trail, never the canonical content.
- **The temporal layer is advisory.** It informs surfacing and the future
  `memory_evolution` tool; it never gates ambient injection or recall ranking.
  RFC 0007 dormancy stays the only mechanism that quiets a chunk.

This containment is what lets us add temporal tracking without renegotiating the
RFC 0006 contract: the Markdown view of memory is identical to today's, with or
without M-cogcons enabled.

## 3. The Supersession Model

A **supersession event** is the moment a curated fact replaces a prior curated
fact about the same thing. Concretely, during a curated write
(`memory_write` with `WriteTarget::Curated` / `write_curated_stratum`, RFC 0008 §4)
or a consolidation pass (RFC 0006 §7.3), one of:

1. A chunk under a stable heading-path is **rewritten** with materially different
   text, or
2. A chunk is **removed** while a new chunk under the same heading-path takes its
   conceptual place, or
3. The model / user **explicitly signals** "this replaces that" during a curated
   edit.

Today, case (1) is exactly the boundary RFC 0007 §4 calls out as "materially
rewriting a chunk's text intentionally starts it fresh at `weight = 1.0`." That
fresh start is the right behaviour for *salience*, but it is the wrong behaviour
for *history*: the moment we currently lose the prior fact is precisely the moment
we should capture it.

When a supersession event fires, the prior assertion is **time-bounded**:

- `valid_from` — when the prior assertion first appeared (best-effort: timestamp
  of the earliest write we have a record of, falling back to "first observed").
- `valid_to` — the timestamp of the supersession event.
- `superseded_by` — the new chunk's `chunk_key`, when one exists.

The new assertion has `valid_from = now`, `valid_to = NULL` (still live). When it
in turn is superseded, its own `valid_to` closes and a third row opens. A curated
fact's history is the chain of rows sharing a heading-path, ordered by `valid_to`.

Phase 1 captures cases (1) and (2). Case (3) — explicit user/model signalling —
is deferred to the relation-graph phase (§6) where the signal has somewhere to go.

## 4. Data Model: A Derived Side Table

Following RFC 0007 §4 exactly, the temporal layer is a **durable side table**
keyed off `chunk_key`. It survives `reindex` (RFC 0006 §4) by construction — its
rows describe events, not current content, so a Markdown rebuild does not
invalidate them.

```sql
CREATE TABLE chunk_history (
    id            INTEGER PRIMARY KEY,
    chunk_key     TEXT NOT NULL,            -- the prior chunk's stable identity
    heading_path  TEXT NOT NULL,            -- e.g. "who.md > Roles > Job"
    text          TEXT NOT NULL,            -- the prior assertion verbatim
    valid_from    INTEGER NOT NULL,         -- epoch seconds
    valid_to      INTEGER NOT NULL,         -- epoch seconds, supersession time
    superseded_by TEXT,                     -- new chunk_key, or NULL if deleted
    reason        TEXT                      -- 'rewrite' | 'remove' | 'consolidate'
);

CREATE INDEX chunk_history_by_path ON chunk_history(heading_path, valid_to);
CREATE INDEX chunk_history_by_key  ON chunk_history(chunk_key);
```

Notes:

- `chunk_key` is the prior chunk's key (RFC 0007 §4: source + heading-path +
  hash of normalized text). Because the hash changes when the text materially
  changes, the prior `chunk_key` is exactly the natural row identity for "the
  fact as it was."
- `heading_path` is duplicated rather than derived so a chain is queryable
  without joining live FTS5 rows (the live row no longer exists for the
  superseded chunk).
- Storing `text` verbatim is the whole point — the recall chain has to survive
  the rewrite. Privacy posture (§9) covers this.
- No row is ever updated. Closing a `valid_to` is a single `INSERT` at
  supersession time; the prior live state had no row.

The live (current) state is **not** in `chunk_history` — it lives in the FTS5
index as today. A row appears here only at the moment a fact is closed out.

## 5. Phase 1 — Supersession Only

Phase 1 ships:

1. The `chunk_history` side table and its migrations.
2. A supersession-detection hook in the curated-write path
   (`Memory::write_curated_stratum`, `Memory::rewrite_curated`) and in the
   consolidation path (RFC 0006 §7.3 `consolidate`):
   - On a curated write, compute the new chunk set and diff against the prior
     chunk set under the same heading-path.
   - For each chunk that disappears (case 2) or is replaced by a chunk under
     the same heading-path with different text (case 1), insert a
     `chunk_history` row capturing the prior text with `valid_to = now`.
3. Inspectability surface (§8): the user can see history rows for any heading.

What Phase 1 explicitly does **not** ship:

- No new recall tool (`memory_search` is unchanged; history rows are not in
  FTS5).
- No relation graph (no S/P/O extraction).
- No ambient-injection change (history is never injected).
- No automatic detection of "same fact, different wording" — Phase 1 detection
  is purely **heading-path identity**. Two chunks under the same heading-path
  are presumed to be about the same thing; chunks under different heading-paths
  are presumed unrelated. This is conservative and predictable; §6 relaxes it.

This minimum gives the recall-chain trace the user asked for, without committing
to the harder design questions in §6.

## 6. Later Phase — Relation Graph & `memory_evolution`

Once Phase 1 has accumulated real history, two follow-ons become tractable:

**6.1 Subject/predicate/object relation graph (advisory).** Curated facts under
a stratum heading (RFC 0008) are extracted into S/P/O triples — e.g.
`(Tony, role, "L5 SDE2")` — and the same temporal envelope (`valid_from` /
`valid_to`) is attached to each triple. This handles the "same fact, different
wording" case Phase 1 punts on, and lets supersession be detected by S+P
collision rather than by heading-path identity. The graph is **derived** (it
can be rebuilt from `chunk_history` + the live curated Markdown) and
**advisory** (it does not gate ambient injection or recall ranking).

**6.2 `memory_evolution` recall tool.** A new tool the model can call to ask
"how has belief about X changed?" — backed by `chunk_history` (Phase 1) and
later the relation graph (6.1). Returns the chain ordered by time, with a
ranged-time annotation per assertion. This is the explicit user-visible payoff
of "trace memory's recall chain."

Both are deferred because they each warrant their own RFC: the graph design has
real choices (extraction strategy, schema, refresh trigger), and the tool's
surface-area lands in the same family as `memory_search` / `memory_get` and
should be designed alongside any other recall additions.

## 7. Interaction with Consolidation & Hygiene

**RFC 0006 §7.3 consolidation.** Consolidation is the natural place for
supersession to fire: it is exactly when stale curated content gets pruned in
favour of a current summary. The hook in §5 wraps the rewrite that consolidation
already performs, so consolidation's output is unchanged but its discarded
content lands in `chunk_history` instead of vanishing.

**RFC 0007 reinforcement & decay.** A superseded chunk leaves the live FTS5
index, so it has no `chunk_stats` row to decay — the question doesn't arise.
The prior chunk's `chunk_stats` row is **swept on rebuild** (RFC 0007 §4: "Orphaned
rows ... are swept on rebuild"), which is correct: the historical fact's
*salience* is over; its *historicity* is what `chunk_history` carries forward.

**RFC 0008 strata.** Strata headings are the fact homes. Phase 1 supersession
detection by heading-path means a fact's history is naturally scoped to its
stratum (Identity / Patterns / Focus); a Focus item rotating out of `what.md`
generates a history row in its Focus heading, not crossing strata.

## 8. Surfacing & Inspectability

The user-visible payoff for Phase 1, mirroring RFC 0007 §7:

- A "History" expander next to any curated heading in the memory Settings pane
  shows the chain of prior assertions with their time bounds.
- A CLI inspection: `flowforge memory history <heading-path>` prints the chain.
- History rows are read-only from the UI; the user **owns** them (§9) and can
  clear them per-heading or globally, but cannot edit them (the whole point is
  that they record what was, not what should be).
- History is never auto-injected and never appears in `memory_search` results
  (Phase 1). The only path from a model turn to history is the future
  `memory_evolution` tool (§6.2).

## 9. Privacy Posture

Inherits RFC 0006 §8 verbatim — local-first, single-user, no telemetry, never
leaves the box. Three additional clauses specific to history:

- **History is more sensitive than live memory.** A fact the user has retracted
  may be more private than a fact they currently hold. `chunk_history` lives
  in the same `~/.flowforge` SQLite as the live index and shares its file
  permissions; no separate egress path is added.
- **User-clearable.** The Settings pane and CLI must expose a per-heading and
  a global "forget history" action. Clearing history is irreversible and
  intentionally so.
- **Never sent upstream.** History rows are excluded from any future cloud
  embedding / sync feature by default; that gate is opt-in per row, not per
  feature.

## 10. Non-Goals

- **No Markdown rewrite of history.** The user's curated files are never
  annotated with timestamps or strikethroughs. Markdown stays clean.
- **No automatic truth arbitration.** The temporal layer records *that* a
  belief changed; it never decides which version was correct.
- **No relation graph in Phase 1.** Deferred to §6.1.
- **No `memory_evolution` tool in Phase 1.** Deferred to §6.2.
- **No cross-source supersession.** Phase 1 only fires for curated writes; an
  ambient daily-log chunk is never recorded as superseding a curated fact.
- **No deletion-based history.** RFC 0007 §8 ("No deletion") still holds for
  the live store; this RFC adds *capture*, not *deletion*.

## 11. Open Questions

1. **Milestone label.** The README roadmap currently lists `M6 = Cold-start
   <200ms` and `M7 = Workflow canvas`, but RFC 0007 internally calls itself M6
   and RFC 0008 routes signals to M8. The README and the RFC lineage have
   already drifted, and this RFC adds another claimant. The label is left open
   pending an explicit roadmap reconciliation; "M-cogcons" is used as a
   placeholder above.
2. **Supersession detection signal.** Phase 1 uses heading-path identity. Is
   that strong enough on its own, or do we need an explicit "this replaces
   that" hint at write time (case 3 in §3) before Phase 1 ships? Current
   recommendation: ship heading-path-only and let real usage tell us whether
   case 3 is needed before §6.
3. **Side-table format and migration.** SQLite is the obvious home, alongside
   the FTS5 index. Question: same database file (`memory.db`) or a separate
   `history.db`? Same-file is simpler and shares the file-permission story
   (§9); separate-file makes "forget history" a single `unlink`. Phase 1
   recommends same-file.
4. **History and FTS5 search.** Phase 1 keeps history out of `memory_search`
   to preserve current behaviour. Should a future toggle let advanced users
   include history in search results, or is that strictly the
   `memory_evolution` tool's job (§6.2)?
5. **`valid_from` for pre-existing chunks.** When the feature ships, every
   live curated chunk has unknown `valid_from`. Options: backfill with
   "feature install time" (honest but loses prior provenance), or leave
   `valid_from` NULL meaning "no earlier than this row's `valid_to` of its
   predecessor, which doesn't exist." Recommendation: backfill with install
   time and document it.
