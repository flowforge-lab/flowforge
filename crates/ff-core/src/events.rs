//! Payloads for backend -> frontend Tauri events. Names mirror the SOP event table.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::{McpServerStatus, ProviderKind, SessionStatus, StopReason};

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct TokenEvent {
    pub session_id: String,
    pub message_id: String,
    pub delta: String,
}

/// Reasoning/thinking stream delta for an in-flight assistant message (#181).
/// Emitted only when the provider sends reasoning content; never persisted on
/// [`crate::Message`] — the frontend accumulates it separately from `content`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ReasoningEvent {
    pub session_id: String,
    pub message_id: String,
    pub delta: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ToolCallEvent {
    pub session_id: String,
    /// Assistant message the call belongs to.
    pub message_id: String,
    /// Correlates the call with its [`ToolResultEvent`].
    pub call_id: String,
    pub tool: String,
    #[ts(type = "unknown")]
    pub args: serde_json::Value,
}

/// Trust level of a tool call that requires user approval. Read-only calls never
/// reach approval, so this enum carries only the gated levels — it is the typed
/// contract for [`ToolApprovalRequestEvent::safety`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub enum ApprovalSafety {
    Write,
    Sensitive,
    Dangerous,
    Publish,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ToolApprovalRequestEvent {
    pub session_id: String,
    /// Assistant message the call belongs to.
    pub message_id: String,
    /// Correlates with the [`ToolCallEvent`] / [`ToolResultEvent`] for this call.
    pub call_id: String,
    pub tool: String,
    #[ts(type = "unknown")]
    pub args: serde_json::Value,
    /// Trust level — read-only calls never require approval.
    pub safety: ApprovalSafety,
    /// Scope-at-a-glance for a `propose_pr` gate (#1252): the files/lines about to
    /// be pushed, computed by the host at approval-emit time. `None` for every
    /// other tool, and best-effort — a git failure yields `None` rather than
    /// blocking the gate. The line-level review still happens on the draft PR;
    /// this is only so the human sees *what's about to be pushed* without leaving
    /// the window.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub scope_summary: Option<ProposePrScope>,
}

/// The `git diff --stat`-style scope shown on a `propose_pr` approval gate
/// (#1252). `propose_pr` commits with `add -A`, so untracked files are counted
/// separately (`git diff --stat HEAD` alone would undercount an all-new-file
/// proposal to an empty scope).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ProposePrScope {
    /// Tracked files with staged/working-tree changes vs `HEAD`.
    pub files_changed: u32,
    /// Total lines added across those files.
    pub insertions: u32,
    /// Total lines removed across those files.
    pub deletions: u32,
    /// Untracked, non-ignored files that `add -A` will bring into the commit
    /// (invisible to `git diff --stat HEAD`).
    pub new_files: u32,
    /// Per-file churn (#1252 AC2), capped and ordered by descending churn so the
    /// card shows the biggest touches first. Binary files report `+0 −0` (git
    /// numstat emits `-` for them). Untracked new files are not listed here —
    /// they are only counted in [`Self::new_files`].
    pub per_file: Vec<FileScope>,
}

/// One row of [`ProposePrScope::per_file`] — a single file's line churn (#1252).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct FileScope {
    /// Repo-relative path as git reports it.
    pub path: String,
    /// Lines added in this file.
    pub insertions: u32,
    /// Lines removed in this file.
    pub deletions: u32,
}

/// Backend -> frontend request to put a clarifying question to the user (the
/// `ask_user` tool, #44). The turn pauses until the frontend replies via the
/// `respond_ask(session_id, call_id, answer)` command. Dismissing it (turn cancel)
/// resolves the question as "[no answer: question dismissed]" — never a hang. `call_id` correlates with
/// the [`ToolCallEvent`] / [`ToolResultEvent`] for the same step.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ToolAskRequestEvent {
    pub session_id: String,
    pub message_id: String,
    pub call_id: String,
    pub question: String,
    /// The model requested a secret (#562): the frontend renders a masked field
    /// and the resolved answer is redacted to a placeholder when persisted. `false`
    /// for an ordinary question.
    pub secret: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ToolResultEvent {
    pub session_id: String,
    pub message_id: String,
    pub call_id: String,
    pub success: bool,
    pub result: String,
}

/// Which standard stream a live output chunk (#680) came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub enum OutputStreamKind {
    Stdout,
    Stderr,
}

/// A live chunk of a running command's output (#680), emitted as the process
/// produces it — before (and in addition to) the final [`ToolResultEvent`]. The
/// frontend appends `delta` to the running tool-call block so long builds/tests
/// show progress instead of appearing frozen. `call_id` correlates with the
/// [`ToolCallEvent`] / [`ToolResultEvent`] for the same step.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ToolOutputChunkEvent {
    pub session_id: String,
    pub message_id: String,
    pub call_id: String,
    pub stream: OutputStreamKind,
    pub delta: String,
}

/// The `chars/4` context-size estimate at turn end, split into the three
/// component buckets the context-usage popover renders (#931): the transient
/// system prompt, the advertised tool schemas, and the persisted message
/// transcript. Proportions drive the segmented bar; the popover shows each
/// bucket's token count and share of the budget.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ContextBreakdown {
    /// Estimated tokens of the system prompt (persona + skills + ambient context).
    pub system_tokens: u32,
    /// Estimated tokens of the advertised tool schemas.
    pub tool_tokens: u32,
    /// Number of tool specs advertised this turn.
    pub tool_specs: u32,
    /// #1179 3A: how many of this turn's advertised specs were admitted up front
    /// by declaration rather than found by `tool_search`. The size of the bet --
    /// on its own it says nothing about whether the bet paid, which is why it is
    /// never reported without `preheated_used`. `None` when nothing was preheated,
    /// distinct from `Some(0)`, which would mean a preheat list resolved to
    /// nothing (every name unknown).
    #[serde(default)]
    #[ts(optional)]
    pub preheated_count: Option<u32>,
    /// #1179 3A: how many preheated tools the model actually called this turn.
    /// The number that makes a bad preheat list falsifiable: well below
    /// `preheated_count` means the resident block is carrying schemas nobody
    /// wants -- exactly what Phase 2 (RFC 0024) spent 16.8% of the prompt to
    /// avoid. Must be an intersection with the called set; deriving it from
    /// `preheated_count` would leave it unable to ever report a miss.
    #[serde(default)]
    #[ts(optional)]
    pub preheated_used: Option<u32>,
    /// #1179 3A: bytes of advertised schema the preheat added. Bytes rather than a
    /// tool count because specs differ by more than an order of magnitude
    /// (codegraph's single tool is 1726 B, `glob` a few hundred), so a count
    /// cannot be checked against a prompt budget.
    #[serde(default)]
    #[ts(optional)]
    pub preheated_bytes: Option<u32>,
    /// Estimated tokens of the **verbatim** persisted message transcript
    /// (user/assistant/tool) — the store, before any compaction. Formerly
    /// `messageTokens`; renamed to distinguish from `wireTokens` (#997).
    pub verbatim_tokens: u32,
    /// Estimated tokens of the **compacted wire** actually sent to the model this
    /// turn (post Tier-1 extractive + Tier-2 abstractive). This is what `pctUsed`
    /// should be computed from — it reflects prefill cost, not store size (#997).
    pub wire_tokens: u32,
    /// Number of messages in the verbatim transcript.
    pub message_count: u32,
    /// Estimated tokens of the **Mid** layer of the wire (#1045): the folded
    /// timeline covering everything older than the verbatim tail. `None` (not
    /// `0`) when no fold has happened yet -- the whole transcript is still Near,
    /// and the popover should render "no fold yet", distinct from "folded to
    /// nothing".
    #[serde(default)]
    #[ts(optional)]
    pub mid_tokens: Option<u32>,
    /// Estimated tokens of the **Near** layer of the wire (#1045): the
    /// token-budgeted verbatim tail actually sent. Measured on the wire at send
    /// time, so `mid_tokens + near_tokens` equals the message portion of the
    /// wire (`wire_tokens` minus the separate system/tool buckets). `None` when
    /// not assessed.
    #[serde(default)]
    #[ts(optional)]
    pub near_tokens: Option<u32>,
}

/// Provider-reported token usage for a turn (#931), summed across the turn's
/// round-trips. Distinct from the `chars/4` proxy: these are authoritative
/// counts from the provider's usage metadata (Bedrock `ConverseStream`
/// `TokenUsage`, OpenAI `usage`). The frontend accumulates these across turns to
/// render the SESSION TOTALS block; all fields are 0 when the provider does not
/// report usage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct TurnUsage {
    /// Prompt tokens sent to the model this turn (cumulative over round-trips).
    pub input_tokens: u32,
    /// Completion tokens generated by the model this turn.
    pub output_tokens: u32,
    /// Prompt-prefix cache-read (hit) tokens this turn.
    pub cache_read_tokens: u32,
    /// Prompt-prefix cache-write (miss) tokens this turn.
    pub cache_write_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct TurnDoneEvent {
    pub session_id: String,
    pub message_id: String,
    /// Estimated token count of the session context at turn end, for a
    /// frontend context-usage indicator. `None` when not assessed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u32>,
    /// Why the turn stopped without a usable answer (#658), when it did. `None`
    /// for a normal completion. Lets the frontend offer the Continue affordance
    /// and render the Cancelled banner without string-matching the notice text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Component breakdown of the context-size estimate (#931), for the
    /// context-usage popover's segmented bar and rows. `None` when not assessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub breakdown: Option<ContextBreakdown>,
    /// Provider-reported token usage for this turn (#931). `None` when the
    /// provider does not report usage (e.g. no metadata frame).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub usage: Option<TurnUsage>,
    /// Effective compaction budget (context_window * safety_factor) the agent loop
    /// compacts against (#945). The denominator for the popover's usage-% bar.
    /// `None` only for events not originating from `run_turn`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub budget_tokens: Option<u32>,
}

/// Per-turn timing baseline (F1, #427): the wall-clock breakdown the performance
/// epic (#426) measures every later change against. `round_trips` is the number of
/// provider responses (agent loop iterations) this turn; `iter_ms` is the
/// per-iteration wall-clock in arrival order; `flushes` counts silent mid-turn
/// memory flushes (each an extra provider round-trip); `output_tokens` is the
/// estimated assistant output tokens (tokenx-rs); `prefill_estimates` is the per-
/// round-trip projected request size and `tier1_fires`/`tier2_fires` count how
/// often each compaction tier engaged (F1b, #441). The F1b fields are optional on
/// the wire -- the desktop always populates them, but a non-desktop emitter may
/// omit them cleanly (#475 follow-up). Emitted once at turn end.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct TurnStatsEvent {
    pub session_id: String,
    pub round_trips: u32,
    pub total_ms: u32,
    pub iter_ms: Vec<u32>,
    pub flushes: u32,
    pub output_tokens: u32,
    /// F1b (#441): projected prefill-token estimate of each round-trip's outgoing
    /// request (post-compaction wire), in iteration order. Omitted by emitters that
    /// do not compute F1b telemetry.
    ///
    /// Invariant when present: `prefill_estimates.len() == round_trips` (one
    /// estimate per round-trip). The frontend should assert this when it eventually
    /// consumes `turn:stats` events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub prefill_estimates: Option<Vec<u32>>,
    /// F1b (#441) / #1045: number of fold **ticks** this turn -- times the
    /// layered Tier-1 pass advanced the frozen boundary. With the Near-budget
    /// hysteresis (#1045) this is `<= 1` for most turns; it is NOT a
    /// per-iteration "the pass ran" count. Omitted by emitters that do not
    /// compute F1b telemetry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub tier1_fires: Option<u32>,
    /// F1b (#441): iterations that engaged the Tier-2 abstractive cold-tail summary.
    /// Omitted by emitters that do not compute F1b telemetry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub tier2_fires: Option<u32>,
    /// #1045: `compaction_retrieve` calls the model made this turn -- the recall
    /// cost of the layered fold. Omitted by emitters that do not compute it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub retrieve_calls: Option<u32>,
    /// TTFT (end-to-end): milliseconds from the moment the host handed the
    /// request to `run_turn` to the arrival of the first assistant token.
    /// Anchored at `turn_start`, so it *includes* any pre-first-token work the
    /// turn did before the model spoke — most notably a context-pressure memory
    /// flush (a full extra provider round-trip) and planning-step reasoning.
    /// Pair it with [`Self::prompt_latency_ms`] to separate prefill from that
    /// side work. `None` when the turn produced no assistant message (e.g. an
    /// early error or cancel before the first token streamed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub first_token_ms: Option<u32>,
    /// #960: pure provider prefill latency of round-trip 0, in milliseconds —
    /// measured inside `run_turn` from the instant the provider stream is
    /// returned to the first output-carrying chunk. Unlike
    /// [`Self::first_token_ms`], it excludes the pre-first-token memory flush and
    /// the `turn_start`→stream-start gap, so it isolates the cache-addressable
    /// prefill cost. `promptLatencyMs / firstTokenMs` is the "prefill share": a
    /// value near 100% means the wait was almost all prefill; a small value means
    /// it was dominated by flush/reasoning side work. `None` when the turn
    /// produced no token, or for emitters that do not compute it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub prompt_latency_ms: Option<u32>,
    /// #971: wall-clock (ms) spent in the pre-main-call **Tier-2 abstractive
    /// summarize** this turn (the uncached `summarize_cold` LLM call; the
    /// cross-turn-cache reuse path is excluded). The dominant "other" latency on
    /// an over-budget re-trigger turn. `None` when no summarize ran this turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub tier2_ms: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct TurnErrorEvent {
    pub session_id: String,
    pub message: String,
}

/// A transient transport drop before any token; the turn is auto-retrying (#928).
/// The frontend renders "Reconnecting... {attempt}/{max_attempts}" on the in-progress
/// turn and clears it on the next `TokenEvent`/`TurnDoneEvent`. `attempt` is the
/// upcoming retry (1-based).
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ReconnectingEvent {
    pub session_id: String,
    pub message_id: String,
    pub attempt: u32,
    pub max_attempts: u32,
}

/// A transient connection failure ended the turn -- budget exhausted or a mid-stream
/// drop (no resume) (#928). Distinct from `TurnErrorEvent` so the frontend can render
/// a connection-specific error + "Try again" (a re-run). `message` is the underlying
/// error for detail; the user-facing headline copy is owned by the frontend and must
/// stay provider-neutral (offline vs provider-down are indistinguishable here).
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ConnectionFailedEvent {
    pub session_id: String,
    pub message_id: String,
    pub message: String,
}

/// A silent context-pressure memory flush wrote durable facts to the user's
/// on-disk memory mid-turn (#283, follow-up to #244 R5). Emitted only when the
/// flush actually wrote something, so the memory browser can surface provenance
/// ("memory auto-updated this turn").
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct MemoryFlushedEvent {
    pub session_id: String,
    pub message_id: String,
    /// Number of durable facts written this turn (always > 0 when emitted).
    pub writes: u32,
}

/// The active phenotype is `LocalOnly` but the resolved inference path is a
/// hosted provider (#888). The egress policy strips network-capable *tools*
/// (RFC 0013 / #883) but the inference call itself still leaves the machine
/// when the model is hosted — prompt content (potentially PII) reaches the
/// cloud regardless of the tool layer. The frontend renders this as a
/// privacy warning (badge, banner, or inline notice) so the user can either
/// switch to a local connection or accept the inference egress explicitly.
/// Mirrors the `AttachmentsDropped` precedent: surfaced via a typed event so
/// the silent capability gap doesn't go unnoticed. Emitted once per turn
/// (first iteration only) on the `egress:mismatch` IPC channel.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct EgressMismatchEvent {
    pub session_id: String,
    pub message_id: String,
    /// The resolved provider kind (e.g. `openAi`, `siliconFlow`, `bedrock`).
    /// Frontends use this to render the warning with the right "hosted by"
    /// label.
    pub kind: ProviderKind,
    /// The model id resolved at turn start. Useful when a phenotype override
    /// swaps the model away from the global default, so the warning can name
    /// the actual model leaving the machine.
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct IntentionSignal {
    pub session_id: String,
    pub goal: String,
}

/// A session's title was regenerated as an LLM one-line summary after its first
/// turn (#671 item 2b). The heuristic `auto_title` seeds an instant title on the
/// first user message; this replaces it with a better summary once a reply exists.
/// The frontend patches the cached session title in place -- no refetch.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct SessionTitleUpdatedEvent {
    pub session_id: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct OutcomeSignal {
    pub session_id: String,
    pub status: SessionStatus,
    /// Daily `chunk_key`s the session touched, for the signal pipeline to settle
    /// memory (#1307). Carried on the signal so the consumer need not look up
    /// session state by `session_id`; typically <100 keys per session.
    pub touched: Vec<String>,
}

/// Backend -> frontend request to approve installing a skill (M3.2). Emitted after
/// the bundle is fetched and validated, so the user approves with the real declared
/// `manifest` (name, version, tools/permissions) in hand. The frontend replies via
/// the shared `respond_approval(session_id, call_id, approved)` command — the same
/// gate as dangerous tool calls. An install has no turn, so the backend keys it by
/// `request_id`: reply with `request_id` for BOTH `session_id` and `call_id`.
/// `warnings` carries non-fatal validation notes.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct SkillInstallApprovalRequestEvent {
    pub request_id: String,
    pub source: String,
    pub manifest: crate::SkillManifest,
    pub warnings: Vec<String>,
}

/// Backend -> frontend notice that the active skill set changed (an
/// activate/deactivate, or an install/uninstall reload that pruned a missing
/// skill). Carries the full active set so the frontend replaces its state rather
/// than diffing. The installed-skill list itself is re-fetched via `list_skills`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct SkillsChangedEvent {
    pub active: Vec<String>,
}

/// Backend -> frontend notice that one or more MCP servers changed status (a
/// start/stop/restart, a connect failure, or an enable/disable/add/remove reload).
/// Carries the full status snapshot so the frontend replaces its state rather than
/// diffing, mirroring [`SkillsChangedEvent`]. The server definitions themselves are
/// re-fetched via `list_mcp_servers`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct McpStatusChangedEvent {
    pub servers: Vec<McpServerStatus>,
}

/// Backend -> frontend notice that a just-activated phenotype lists a skill whose
/// declared MCP server is unavailable (#301/#235) -- absent from `mcp.json`, or
/// present but not `Running`. Mirrors the warn-only signal `warn_missing_skill_mcp`
/// logs; emitted only when the unavailable list is non-empty so the frontend toast
/// fires exactly when there is something to report. Non-fatal: activation never
/// blocks and the skill's grep/glob fallbacks still work. `servers` is name-sorted
/// and deduplicated.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct PhenotypeMcpUnavailableEvent {
    pub phenotype: String,
    pub servers: Vec<String>,
}

/// Backend -> frontend notice that entries in an active phenotype's `preheat` list
/// were dropped (#1179 3B) -- an unknown/non-deferrable tool name, or a name that
/// did not fit [`MAX_PREHEAT_BYTES`](ff_tools::MAX_PREHEAT_BYTES).
///
/// Exists because the alternative is indistinguishable from success: a phenotype
/// with a typo'd tool name preheats nothing, behaves exactly like one with no
/// preheat list, and gives its author no reason to look. Non-fatal -- the tool is
/// still reachable the ordinary way, via `tool_search`.
///
/// Emitted only when something was actually dropped, so the frontend can toast
/// unconditionally on receipt.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct PhenotypePreheatDroppedEvent {
    pub phenotype: String,
    pub session_id: String,
    /// Names no registered deferred tool matched.
    pub unknown: Vec<String>,
    /// Names dropped for exceeding the byte budget, in declaration order.
    pub over_budget: Vec<String>,
    /// Bytes the surviving entries cost, for comparison against the budget.
    pub admitted_bytes: u32,
}

/// Backend -> frontend telemetry: a skill became active for a turn (M3.5, RFC 0001
/// §8). Emitted once per active skill at the start of each agent turn. `ff-signals`
/// folds these into per-skill aggregates (activation counts); the frontend may also
/// surface live activity. Pairs with [`SkillCompleted`] at turn end.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct SkillActivated {
    pub skill: String,
    pub session_id: String,
}

/// Backend -> frontend telemetry: a turn that had `skill` active finished (M3.5, RFC
/// 0001 §8). Emitted once per active skill when the turn ends. `tokens` is a coarse
/// cost proxy (streamed assistant characters / 4) until real provider usage is wired
/// (deferred with the M4 autonomous trigger); `turns` counts agent loop iterations;
/// `latency_ms` is wall-clock turn duration; `success` is true when the turn ended
/// cleanly (not error/cancel). `ff-signals` folds these into rolling per-skill
/// aggregates (mean tokens/turns, success rate).
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct SkillCompleted {
    pub skill: String,
    pub session_id: String,
    pub tokens: u32,
    pub latency_ms: u32,
    pub turns: u32,
    pub success: bool,
}

/// Rough cost projection shown alongside an optimize proposal (RFC 0001 §8). Both
/// values use the same coarse token proxy as the telemetry substrate, so they are
/// comparable to each other but approximate in absolute terms (real provider usage
/// lands in M4). `estimatedMeanTokens` scales the current rolling mean by the body's
/// size change; `0.0` when the skill has no telemetry yet.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct EvolveCostEstimate {
    pub current_mean_tokens: f64,
    pub estimated_mean_tokens: f64,
}

/// Backend -> frontend request to approve an optimize/evolve rewrite of a skill
/// (M3.5, RFC 0001 §8). The model has proposed `after_body`; the frontend shows the
/// before→after diff (it has both bodies) plus the cost estimate, and the user
/// approves or rejects. Like a skill install, this is a standalone approval with no
/// turn, so it is keyed by `request_id` and answered via the shared
/// `respond_approval(request_id, request_id, approved)` command. On approval the
/// skill is version-bumped to `new_version`, retaining the previous version for
/// rollback — it is never silently overwritten.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct SkillEvolveApprovalRequestEvent {
    pub request_id: String,
    pub skill: String,
    pub current_version: String,
    pub new_version: String,
    pub before_body: String,
    pub after_body: String,
    pub cost_estimate: EvolveCostEstimate,
}

/// Download progress for an in-flight self-update install (#566, RFC 0014 section
/// 12.2). Emitted as `update:progress` from `install_update` per downloaded chunk;
/// `total` is the content length, absent when the feed does not send one (the UI
/// then renders an indeterminate bar). App-global -- there is one install at a time,
/// so no session id. A terminal `update:download-finished` event (empty payload)
/// follows the last chunk, before the app relaunches.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct UpdateProgressEvent {
    pub downloaded: u64,
    pub total: Option<u64>,
}

/// One chunk of live stdout/stderr from a background process started via
/// `process_manager action=start` (#873). Emitted as `process:output`,
/// independently of any assistant turn, by a desktop bridge task that
/// subscribes to the process's output broadcast the moment it starts. The
/// frontend appends `delta` to the process's output panel keyed by
/// `process_id`. `stream` is `"stdout"` or `"stderr"`. Unlike the per-turn
/// `token`/`tool:output` events, these keep flowing across turns for the
/// life of the process.
///
/// `process_id` is the small sequential id `start` returns (fits a `u32`,
/// so the frontend gets a plain `number`, not a `bigint`). `stream` reuses the
/// same [`OutputStreamKind`] the per-turn `tool:output` chunks use.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ProcessOutputEvent {
    pub session_id: String,
    pub process_id: u32,
    pub stream: OutputStreamKind,
    pub delta: String,
}

/// A background process ended (exited, was killed, or failed to run) (#873).
/// Emitted once as `process:exited` after the last `process:output`, when the
/// bridge task sees the output broadcast close. `status` is the supervisor's
/// human-readable label -- `"exited(0)"`, `"exited(3)"`, `"killed"`, or
/// `"failed: <reason>"` -- which the frontend renders as the process's
/// terminal status badge.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ProcessExitedEvent {
    pub session_id: String,
    pub process_id: u32,
    pub status: String,
}

/// An embedded terminal's shell exited (#1284) -- the user typed `exit`, the
/// shell crashed, or we killed it because its tab/pane/session went away.
/// Emitted once, when the PTY reader sees EOF.
///
/// This is an `app.emit` event where the terminal's *output* is a
/// `tauri::ipc::Channel` (see `terminal.rs`): one message per terminal for its
/// whole life is exactly the low-frequency signal the event system is for,
/// whereas the byte stream is not. `terminal_id` is the id `terminal_open`
/// returned; the frontend marks that tab dead. An id the frontend no longer
/// knows is expected -- closing a tab kills the shell, so a `terminal:exited`
/// races the close it caused -- and is ignored.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct TerminalExitedEvent {
    pub session_id: String,
    pub terminal_id: String,
}

/// The set of active observers for a session changed (#1038, epic #954 M2):
/// one started, was stopped, or fired. Coarse by design -- the frontend
/// re-runs `list_observers(sessionId)` on receipt rather than diffing. A
/// finer started/fired/stopped event is deferred (would be needed for a
/// fired-observer highlight).
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export, export_to = "../../../apps/desktop/src/bindings/")]
pub struct ObserverChangedEvent {
    pub session_id: String,
}

#[cfg(test)]
mod tests;
