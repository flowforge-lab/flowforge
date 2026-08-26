//! Thin Tauri shell. Per the SOP, this layer contains only command/event glue —
//! all business logic lives in the `ff-*` crates. Each handler deserializes,
//! calls into a crate, and returns. Streaming responses go out as Tauri events.

mod dev_update_watcher;
mod git_watch;
mod optimize;
mod secrets;
mod state;
mod terminal;
mod tools;

use crate::terminal::{terminal_close, terminal_open, terminal_resize, terminal_write, Terminals};
use async_trait::async_trait;
use ff_agent::{
    drive_goal, AgentEvent, ApprovalOutcome, Approver, CancelToken, DenyReason, GateDecision,
    GoalIteration, IterationOutcome, ToolContext, VerifyOutcome,
};
use ff_core::events::{
    ApprovalSafety, ConnectionFailedEvent, EgressMismatchEvent, EvolveCostEstimate,
    IntentionSignal, McpStatusChangedEvent, MemoryFlushedEvent, ObserverChangedEvent,
    OutputStreamKind, PhenotypeMcpUnavailableEvent, ProcessExitedEvent, ProcessOutputEvent,
    ReasoningEvent, ReconnectingEvent, SessionTitleUpdatedEvent, SkillActivated, SkillCompleted,
    SkillEvolveApprovalRequestEvent, SkillInstallApprovalRequestEvent, SkillsChangedEvent,
    TokenEvent, ToolApprovalRequestEvent, ToolAskRequestEvent, ToolCallEvent, ToolOutputChunkEvent,
    ToolResultEvent, TurnDoneEvent, TurnErrorEvent, TurnStatsEvent, UpdateProgressEvent,
};
use ff_core::{
    pre_prompt_decision, resolve_tool_arg, Attachment, BedrockAuth, CreateScheduledTaskInput,
    DirEntry, FileContent, Format, Goal, GoalStatus, McpServerConfig, McpServerStatus,
    MemoryFileInfo, MemoryFileKind, MemoryOverview, Message, Mode, ModelSelection, PermissionCell,
    PermissionMatrixView, Phenotype, ProviderConfig, ProviderConnection, ProviderKind,
    ProviderRegistry, ResolvedModel, Role, RunRecord, RunStatus, ScheduledTask, SearchConfig,
    SecretKind, Session, SessionWorkspace, Skill, SkillInfo, SkillManifest, TaskKind,
};
use ff_observer::{ObserverEvent, ObserverInfo};
use ff_scheduled::ScheduledApprover;
use ff_signals::SkillAggregate;
use ff_tools::{NotebookKernelState, Safety};
use state::AppState;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tauri::{Emitter, Manager, State};
use tauri_plugin_shell::ShellExt;
use uuid::Uuid;

// Boot-timing trace (#599 item 0). A single shared clock for the cold-start
// milestones — webview page-load, FE first paint, `AppState::new()` (now run
// post-paint on the background hydrate task — see `run`), and `app_ready` — so
// the <200ms-to-paint target is measured against real numbers instead of
// guesses. Every `[boot-trace]` line reports milliseconds since `BOOT_T0` (set
// as the first line of `run()`). Entirely gated behind the `FF_BOOT_TRACE` env
// var: unset, each stamp is a cheap `OnceLock` load and returns before
// formatting or I/O, so a normal launch pays nothing.
static BOOT_T0: OnceLock<std::time::Instant> = OnceLock::new();
static BOOT_TRACE_ON: OnceLock<bool> = OnceLock::new();

// Paint-first boot (#599): flipped by the background hydrate task in `run` once
// `AppState::new()` has finished and the state has been managed. The FE gates
// its backend-dependent work on the `app:ready` event plus this flag
// (subscribe-then-check) so the invoke handlers — which read
// `State<'_, Arc<AppState>>` — are never hit before the state exists.
static APP_READY: AtomicBool = AtomicBool::new(false);

fn boot_trace_enabled() -> bool {
    *BOOT_TRACE_ON.get_or_init(|| std::env::var_os("FF_BOOT_TRACE").is_some())
}

/// Emit `[boot-trace] <label> +<ms>ms[ (<extra>)]`, the elapsed since `BOOT_T0`.
/// Used for the cumulative milestones (window-relative), not per-step durations.
pub(crate) fn boot_trace(label: &str, extra: Option<&str>) {
    if !boot_trace_enabled() {
        return;
    }
    let ms = BOOT_T0
        .get()
        .map(|t0| t0.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    match extra {
        Some(e) => eprintln!("[boot-trace] {label} +{ms:.1}ms ({e})"),
        None => eprintln!("[boot-trace] {label} +{ms:.1}ms"),
    }
}

/// Emit `[boot-trace] <label> <ms>ms` for a single step's own duration (not
/// since-`BOOT_T0`). Used by `AppState::new()` to break down each blocking open
/// so items 1-4 are driven by which SQLite open / scan actually costs.
pub(crate) fn boot_trace_step(label: &str, dur: Duration) {
    if !boot_trace_enabled() {
        return;
    }
    eprintln!("[boot-trace] {label} {:.1}ms", dur.as_secs_f64() * 1000.0);
}

/// Per-turn telemetry accumulator (RFC 0001 §8), filled by the agent-event closure
/// and folded into per-skill aggregates when the turn ends. `message_ids` counts
/// distinct assistant messages -- one per agent loop iteration, i.e. the turn count;
/// `tokens` is the estimated assistant output tokens (via tokenx-rs, ~96% accurate).
#[derive(Default)]
struct TurnMetrics {
    tokens: usize,
    message_ids: std::collections::HashSet<String>,
    /// First-seen instant of each distinct assistant message, in arrival order.
    /// One per agent loop iteration (provider round-trip); consecutive deltas
    /// give the per-iteration wall-clock baseline (F1, #427).
    iter_marks: Vec<std::time::Instant>,
    /// Silent mid-turn memory flushes, each an extra provider round-trip (#427).
    flushes: u32,
    /// F1b (#441): captured from the turn's `Done` event -- the per-round-trip
    /// prefill estimate and how often each compaction tier engaged this turn.
    prefill_estimates: Vec<u32>,
    tier1_fires: u32,
    tier2_fires: u32,
    /// #1045: `compaction_retrieve` calls the model made this turn, carried on
    /// `Done` -- the recall cost of the layered fold.
    retrieve_calls: u32,
    /// #960: pure provider round-0 prefill latency (ms) carried on the turn's
    /// `Done` event. Distinct from the host-computed `first_token_ms`, which is
    /// anchored at `turn_start` and so also absorbs any pre-first-token memory
    /// flush / planning reasoning.
    prompt_latency_ms: Option<u32>,
    /// #971: pre-main-call Tier-2 abstractive-summarize wall-clock (ms) carried on `Done`.
    tier2_ms: Option<u32>,
}

impl TurnMetrics {
    fn note_turn(&mut self, message_id: &str) {
        if self.message_ids.insert(message_id.to_string()) {
            self.iter_marks.push(std::time::Instant::now());
        }
    }

    fn note_flush(&mut self) {
        self.flushes += 1;
    }

    /// Fold the agent-side F1b signal carried by the turn's `Done` event (#441).
    /// Fires once per turn, so a plain assign is correct.
    fn note_done(
        &mut self,
        prefill_estimates: &[u32],
        prompt_latency_ms: Option<u32>,
        tier2_ms: Option<u32>,
        tier1_fires: u32,
        tier2_fires: u32,
        retrieve_calls: u32,
    ) {
        self.prefill_estimates = prefill_estimates.to_vec();
        self.prompt_latency_ms = prompt_latency_ms;
        self.tier2_ms = tier2_ms;
        self.tier1_fires = tier1_fires;
        self.tier2_fires = tier2_fires;
        self.retrieve_calls = retrieve_calls;
    }

    /// `(estimated assistant output tokens, distinct turn count)`.
    fn snapshot(&self) -> (usize, usize) {
        (self.tokens, self.message_ids.len())
    }

    /// Per-turn timing breakdown for the #427 baseline: `(round_trips, per-iteration
    /// ms in arrival order, flushes, first_token_ms)`. `turn_start` anchors the
    /// TTFT measurement (time from the moment we hand the request to `run_turn`
    /// to the first assistant token arriving); `turn_end` closes the final
    /// iteration. `first_token_ms` is `None` when the turn produced no assistant
    /// message (e.g. an early error before any token streamed).
    fn timing(
        &self,
        turn_start: std::time::Instant,
        turn_end: std::time::Instant,
    ) -> (u32, Vec<u32>, u32, Option<u32>) {
        let iter_ms = self
            .iter_marks
            .iter()
            .enumerate()
            .map(|(i, start)| {
                let end = self.iter_marks.get(i + 1).copied().unwrap_or(turn_end);
                u32::try_from(end.saturating_duration_since(*start).as_millis()).unwrap_or(u32::MAX)
            })
            .collect();
        let round_trips = u32::try_from(self.iter_marks.len()).unwrap_or(u32::MAX);
        let first_token_ms = self.iter_marks.first().map(|first| {
            u32::try_from(first.saturating_duration_since(turn_start).as_millis())
                .unwrap_or(u32::MAX)
        });
        (round_trips, iter_ms, self.flushes, first_token_ms)
    }
}

/// The invocation-time gate for a resolved permission cell (#702): `Some(true)`
/// auto-approves, `Some(false)` rejects, `None` means prompt the user (`Ask`).
/// Superseded by [`pre_prompt_decision`] in production, but kept for the
/// `edited_cell_flips_the_invocation_gate` test which exercises the raw cell semantics.
#[cfg(test)]
fn matrix_gate(cell: PermissionCell) -> Option<bool> {
    if cell.is_allow() {
        Some(true)
    } else if cell.is_deny() {
        Some(false)
    } else {
        None
    }
}

/// Routes write/dangerous tool calls through a UI confirmation. Read-only calls
/// never reach this approver — the agent loop short-circuits them.
struct UiApprover<R: tauri::Runtime> {
    app: tauri::AppHandle<R>,
    state: Arc<AppState>,
    session_id: String,
    /// The session's resolved autonomy mode for this turn (#265).
    mode: Mode,
    /// Autonomous goal mode (#1256): there is no user watching the loop, so a
    /// `Prompt` decision cannot be surfaced as an interactive request — it would
    /// block indefinitely (#1270). When true, `Prompt` collapses to a clean deny
    /// that routes the model to the report-only branch.
    autonomous: bool,
    /// Goal authorised to open a draft PR on completion (#1256). Applies a
    /// per-call `propose_pr -> Allow` override to the matrix *snapshot* only —
    /// never the shared singleton — so authorisation is scoped to this goal loop.
    allow_propose_pr: bool,
    /// How long to wait for a human before giving up on a prompt (#1270). Set
    /// per construction site: generous for a foreground chat turn, short inside
    /// a goal iteration where nobody may be present at all.
    approval_timeout: Duration,
}

#[async_trait]
impl<R: tauri::Runtime> Approver for UiApprover<R> {
    async fn approve(
        &self,
        message_id: &str,
        call_id: &str,
        name: &str,
        safety: Safety,
        args: &serde_json::Value,
    ) -> ApprovalOutcome {
        // Snapshot the matrix once for this call, read live (#702/#742) so a
        // Control-panel edit takes effect on the next tool invocation. An
        // authorised goal (#1256) gets the per-tool `propose_pr -> Allow`
        // override via the shared `goal_matrix` helper — applied to this local
        // snapshot only, never the singleton, so authorisation stays scoped to
        // the goal loop and stays identical to the CLI host.
        let matrix = ff_agent::goal_matrix(self.state.permission_matrix(), self.allow_propose_pr);
        let cell = matrix.effective_cell(name, self.mode, safety);
        let allowlisted = self.state.allowlist_covers(&self.session_id, name, safety);
        let resolved_arg = resolve_tool_arg(name, args);
        let scoped_effect = matrix.evaluate_rules(name, resolved_arg.as_deref(), self.mode);

        // The synchronous pre-prompt decision encodes the canonical gate order
        // (#827/#828 Part C). Extracted so it is unit-testable without an AppHandle.
        let scoped_deny_rule = matrix.matching_deny_rule(name, resolved_arg.as_deref());
        let scoped_deny_rule_desc =
            scoped_deny_rule.map(|r| format!("{} ({})", r.tool, r.matcher.description()));
        match pre_prompt_decision(
            cell,
            allowlisted,
            scoped_effect,
            safety,
            self.mode,
            scoped_deny_rule_desc,
        ) {
            ff_core::PrePromptDecision::Deny(reason) => {
                if matches!(reason, DenyReason::Mode { .. }) {
                    tracing::info!(
                        tool = name,
                        mode = ?self.mode,
                        ?safety,
                        "matrix cell denied tool call"
                    );
                }
                return ApprovalOutcome::Denied(reason);
            }
            ff_core::PrePromptDecision::Allow => {
                if scoped_effect == Some(ff_core::RuleEffect::Allow) {
                    tracing::info!(
                        tool = name,
                        arg = ?resolved_arg,
                        "scoped rule auto-approved"
                    );
                }
                return ApprovalOutcome::Allowed;
            }
            ff_core::PrePromptDecision::Prompt => {}
        }

        // Autonomous goal loop (#1256): no user to prompt, so a Prompt decision
        // becomes a clean deny rather than an indefinite block (#1270). The model
        // reads the refusal and finishes via the report-only branch.
        if self.autonomous {
            return ApprovalOutcome::Denied(DenyReason::NoInteractiveTerminal);
        }

        let approval_safety = match safety {
            Safety::Write => ApprovalSafety::Write,
            Safety::Sensitive => ApprovalSafety::Sensitive,
            Safety::Dangerous => ApprovalSafety::Dangerous,
            Safety::Publish => ApprovalSafety::Publish,
            Safety::ReadOnly => return ApprovalOutcome::Allowed,
        };
        let rx = self.state.register_approval(&self.session_id, call_id);
        // #1252: decorate a propose_pr gate with what is about to be pushed.
        // Computed here (the tool is atomic and gated before run(), so nothing is
        // committed yet) against the session's working dir. Best-effort — a git
        // failure leaves it None and never blocks the gate.
        let scope_summary = if name == ff_tools::PROPOSE_PR_TOOL_NAME {
            let workspace = self.state.session_root(&self.session_id);
            ff_tools::propose_pr_scope_summary(args, &workspace).await
        } else {
            None
        };
        let _ = self.app.emit(
            "tool:approval-request",
            ToolApprovalRequestEvent {
                session_id: self.session_id.clone(),
                message_id: message_id.to_string(),
                call_id: call_id.to_string(),
                tool: name.to_string(),
                args: args.clone(),
                safety: approval_safety,
                scope_summary,
            },
        );
        match tokio::time::timeout(self.approval_timeout, rx).await {
            Ok(Ok(true)) => ApprovalOutcome::Allowed,
            // Explicit deny, or sender dropped (cancel) -> RecvError -> deny.
            Ok(_) => ApprovalOutcome::Denied(DenyReason::User),
            // Nobody answered before the deadline (#1270). Distinct from a user
            // deny: attributing this to the user records a decision that never
            // happened, and the model should be told it can stop waiting.
            Err(_elapsed) => {
                tracing::warn!(
                    tool = %name,
                    timeout_secs = self.approval_timeout.as_secs(),
                    "no approval answer within the timeout; denying as unanswered"
                );
                self.state.discard_approval(&self.session_id, call_id);
                ApprovalOutcome::Denied(DenyReason::Unanswered)
            }
        }
    }

    async fn ask(
        &self,
        message_id: &str,
        call_id: &str,
        args: &serde_json::Value,
    ) -> Option<String> {
        // The loop forwards the tool args; the host reads the `question` field and
        // the optional `secret` flag (#562).
        let question = args
            .get("question")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        let secret = args
            .get("secret")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let rx = self.state.register_ask(&self.session_id, call_id);
        let _ = self.app.emit(
            "tool:ask-request",
            ToolAskRequestEvent {
                session_id: self.session_id.clone(),
                message_id: message_id.to_string(),
                call_id: call_id.to_string(),
                question,
                secret,
            },
        );
        match tokio::time::timeout(self.approval_timeout, rx).await {
            // Sender dropped (cancel/teardown) -> RecvError -> dismissed (None).
            Ok(answered) => answered.ok(),
            // Unanswered by the deadline resolves the same way a dismissal does:
            // the loop already renders `None` as "[no answer: question dismissed]".
            Err(_elapsed) => {
                self.state.discard_ask(&self.session_id, call_id);
                None
            }
        }
    }
}

type CmdResult<T> = Result<T, String>;

#[tauri::command]
fn create_session(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    goal: Option<String>,
) -> Session {
    // A bare `＋` (no goal) is a deferred draft (#671 item 2a): not written to
    // disk until its first message, so clicking `＋` never accrues empty rows. A
    // goal-bearing session (scheduled/intention run, #543) commits immediately and
    // announces its intention so it lists and starts right away.
    match goal {
        Some(goal) => {
            let session = state.store.create_session(Some(goal.clone()));
            let _ = app.emit(
                "signal:intention",
                IntentionSignal {
                    session_id: session.id.clone(),
                    goal,
                },
            );
            session
        }
        None => state.store.create_draft_session(),
    }
}

#[tauri::command]
fn list_sessions(state: State<'_, Arc<AppState>>) -> Vec<Session> {
    state.store.list_sessions()
}

#[tauri::command]
fn get_messages(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    limit: Option<usize>,
) -> Vec<Message> {
    // #1142: read-only. Orphaned rows are reconciled once at app startup
    // (see AppState::with_registry), not on every session switch.
    match limit {
        Some(n) => state.store.get_messages_tail(&session_id, n),
        None => state.store.get_messages(&session_id),
    }
}

/// Full-text search across all sessions (#679). Returns ranked hits with snippets.
#[tauri::command]
fn search_messages(
    state: State<'_, Arc<AppState>>,
    query: String,
    limit: Option<usize>,
) -> Vec<ff_session::SearchHit> {
    state.store.search_messages(&query, limit.unwrap_or(50))
}

/// Full-text search within a single session (#679). For in-thread find.
#[tauri::command]
fn search_in_session(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    query: String,
) -> Vec<ff_session::SearchHit> {
    state.store.search_in_session(&session_id, &query)
}

/// Export a session and its transcript as Markdown or JSON (#278). Returns the
/// rendered string for the frontend to save via a file dialog. Errors when the
/// session id is unknown so the UI can surface it rather than write an empty file.
#[tauri::command]
fn export_session(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    format: Format,
) -> Result<String, String> {
    state
        .store
        .export_session(&session_id, format)
        .ok_or_else(|| format!("session not found: {session_id}"))
}

/// Set a session's display title (server-truth). Used by the sidebar rename and
/// to lift legacy localStorage titles to the backend.
#[tauri::command]
fn rename_session(state: State<'_, Arc<AppState>>, session_id: String, title: String) {
    state.store.set_title(&session_id, title);
}

/// Permanently remove a session and its transcript. Cancels any in-flight turn and
/// pending approvals first, so a stream cannot write back to a deleted session.
#[tauri::command]
fn delete_session(
    state: State<'_, Arc<AppState>>,
    terminals: State<'_, Terminals>,
    session_id: String,
) {
    if let Some(token) = state.take_cancel(&session_id) {
        token.cancel();
    }
    state.cancel_pending_approvals(&session_id);
    state.clear_session_approvals(&session_id);
    state.compaction_cache.invalidate(&session_id);
    state.store.delete_session(&session_id);
    state.reap_session_processes(&session_id);
    state.reap_session_kernels(&session_id);
    // Kill this session's embedded terminals (#1284). Synchronous, unlike the
    // reaps around it: killing a PTY child is a signal, not an await, so there
    // is no runtime to enter from this off-reactor sync command.
    let reaped = terminals.reap_session(&session_id);
    if reaped > 0 {
        tracing::info!(session_id = %session_id, reaped, "reaped session terminals");
    }
    // Stop and reap session-scoped background observers (#891 Phase 1).
    // Same fire-and-forget, off-reactor pattern as the reaps above.
    state.reap_session_observers(&session_id);
    // Release this session's per-workspace MCP instance refs, evicting any instance no
    // live session references (RFC 0018 §4.3). Spawned (not awaited) like the process
    // reap above: `delete_session` is a sync Tauri command that runs off the reactor on
    // macOS (#117), so a bare block_on/await would panic -- `tauri::async_runtime::spawn`
    // enters the shared runtime.
    let state_for_mcp = state.inner().clone();
    let sid = session_id.clone();
    tauri::async_runtime::spawn(async move {
        state_for_mcp.release_session_mcp(&sid).await;
    });
}

/// All scheduled tasks, newest first, with derived cadence label / next run
/// (RFC 0017 #540). Built-in + user-created.
#[tauri::command]
fn list_scheduled_tasks(state: State<'_, Arc<AppState>>) -> Vec<ScheduledTask> {
    state.scheduled.list()
}

/// Create a scheduled task. Validates the cron expression; the human cadence
/// label and next-run are derived server-side, never sent by the FE.
#[tauri::command]
fn create_scheduled_task(
    state: State<'_, Arc<AppState>>,
    input: CreateScheduledTaskInput,
) -> Result<ScheduledTask, String> {
    state.scheduled.create(input)
}

/// Pause/resume a task; returns the updated task. Errors on an unknown id.
#[tauri::command]
fn toggle_scheduled_task(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<ScheduledTask, String> {
    state
        .scheduled
        .toggle_paused(&id)
        .ok_or_else(|| format!("unknown scheduled task: {id}"))
}

/// Delete a task. Rejected for built-in tasks (they ship with the app).
#[tauri::command]
fn delete_scheduled_task(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
    state
        .scheduled
        .delete(&id)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Preview the human cadence label a cron expression would produce, for the
/// New-task form's Custom-cron field. Derived from the same source as `list`.
#[tauri::command]
fn preview_cadence(cron: String) -> Result<String, String> {
    ff_scheduled::cron::parse(&cron)?;
    Ok(ff_scheduled::cron::cadence_label(&cron))
}

/// A task's fire history, newest first, capped at 50. Backs the run-history list
/// and the ↗ open-session affordance (RFC 0017 §6.2, #544).
#[tauri::command]
fn list_scheduled_runs(state: State<'_, Arc<AppState>>, id: String) -> Vec<RunRecord> {
    state.scheduled.runs(&id, 50)
}

/// Engage or release the global pause-all kill-switch (RFC 0017 §8.3, #544). A
/// true switch: the sweep is gated regardless of per-task `paused`, so tasks
/// created while engaged stay held too. Emits `scheduled:changed` so the UI
/// reflects the new gate.
#[tauri::command]
fn set_scheduled_paused_all(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    paused: bool,
) -> bool {
    state.scheduled.set_all_paused(paused);
    let _ = app.emit("scheduled:changed", state.scheduled.list());
    paused
}

// ===== Goal mode (RFC 0020, #716) =====
// The five `goal_*` IPC commands are the FE-facing half of the goal lifecycle;
// the `goal_complete` capability is also an agent tool (dual-surface, §7). Each
// mutation persists through the path-injected `GoalStore` and emits
// `goal:updated` so the FE panel (#717) live-refreshes without polling. The
// self-continue loop is spawned by `goal_set` / `goal_resume` — see
// `spawn_goal_loop`.

/// Begin (or replace) the active goal for a session and start the self-continue
/// loop (RFC 0020 §5.1). An empty objective is rejected. A pre-existing goal for
/// the session is overwritten — starting a new objective is a deliberate reset.
#[tauri::command]
// Tauri commands take flat named IPC params (not a bundled struct), so the
// budget dimensions + authorisation flag exceed clippy's arg-count heuristic.
#[allow(clippy::too_many_arguments)]
async fn goal_set(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
    objective: String,
    max_iterations: Option<u32>,
    max_tokens: Option<u64>,
    max_wall_ms: Option<i64>,
    allow_propose_pr: Option<bool>,
) -> Result<Goal, String> {
    let objective = objective.trim();
    if objective.is_empty() {
        return Err("goal objective must not be empty".into());
    }
    // Re-setting a goal is a deliberate reset, but a loop may still be driving the
    // OLD goal on an in-memory copy (#753 review blocker 2). Stop it first — cancel
    // the in-flight turn and wait (bounded) for the loop to release its slot — so
    // its final checkpoint can't clobber the fresh goal we're about to write.
    stop_goal_loop(state.inner(), &session_id).await;

    let goal = build_goal(
        &session_id,
        objective,
        max_iterations,
        max_tokens,
        max_wall_ms,
        allow_propose_pr,
    );
    state
        .goals
        .save(&goal)
        .map_err(|e| format!("failed to persist goal: {e}"))?;
    let _ = app.emit("goal:updated", &goal);
    spawn_goal_loop(state.inner().clone(), app, session_id);
    Ok(goal)
}

/// Build the `Goal` a `goal_set` call persists — the pure mapping from the
/// command's optional args onto a fresh goal, split out so a test can pin the
/// authorisation plumbing (`allow_propose_pr`, #1256) end-to-end without
/// constructing a tauri `State` or spawning the background loop. `objective` is
/// assumed already trimmed and non-empty by the caller.
fn build_goal(
    session_id: &str,
    objective: &str,
    max_iterations: Option<u32>,
    max_tokens: Option<u64>,
    max_wall_ms: Option<i64>,
    allow_propose_pr: Option<bool>,
) -> Goal {
    let mut goal = Goal::new(session_id, objective, now_ms());
    if let Some(m) = max_iterations {
        goal.budget.max_iterations = m;
    }
    goal.budget.max_tokens = max_tokens;
    goal.budget.max_wall_ms = max_wall_ms;
    goal.allow_propose_pr = allow_propose_pr.unwrap_or(false);
    goal
}

/// Snapshot the current goal for a session, or `None` if there is no goal
/// checkpoint (RFC 0020 §7 — panel poll / event join). A corrupt checkpoint file
/// surfaces as an error rather than a silent `None`.
#[tauri::command]
fn goal_status(
    state: State<'_, Arc<AppState>>,
    session_id: String,
) -> Result<Option<Goal>, String> {
    state
        .goals
        .load(&session_id)
        .map_err(|e| format!("failed to read goal: {e}"))
}

/// Pause a running goal at the next boundary (RFC 0020 §5.3). Idempotent: pausing
/// an already-paused/terminal goal just returns its current state. The loop
/// observes the persisted `Paused` status at its next boundary check and stops
/// resumably; this command also flips it eagerly so the FE reflects intent now.
#[tauri::command]
async fn goal_pause(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<Option<Goal>, String> {
    // Stop the loop first (cancel the in-flight turn + wait for it to exit).
    // The loop owns the goal in memory and never re-reads it, so writing Paused
    // here while it runs would be clobbered by its next checkpoint save. A
    // cancelled turn already transitions the goal to Paused (resumable) via
    // drive_goal; we then ensure Paused for the no-loop-running case.
    stop_goal_loop(state.inner(), &session_id).await;
    let Some(mut goal) = state
        .goals
        .load(&session_id)
        .map_err(|e| format!("failed to read goal: {e}"))?
    else {
        return Ok(None);
    };
    if goal.status == GoalStatus::Active {
        goal.status = GoalStatus::Paused;
        goal.updated_ms = now_ms();
        state
            .goals
            .save(&goal)
            .map_err(|e| format!("failed to persist goal: {e}"))?;
        let _ = app.emit("goal:updated", &goal);
    }
    Ok(Some(goal))
}

/// Resume a paused goal and restart the loop from the last persisted checkpoint
/// (RFC 0020 §5.3 — resume replays from the last completed iteration, never a
/// partial turn). Only a `Paused` goal resumes; a terminal goal is returned
/// unchanged.
#[tauri::command]
fn goal_resume(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<Option<Goal>, String> {
    let Some(mut goal) = state
        .goals
        .load(&session_id)
        .map_err(|e| format!("failed to read goal: {e}"))?
    else {
        return Ok(None);
    };
    if goal.status == GoalStatus::Paused {
        goal.status = GoalStatus::Active;
        goal.updated_ms = now_ms();
        state
            .goals
            .save(&goal)
            .map_err(|e| format!("failed to persist goal: {e}"))?;
        let _ = app.emit("goal:updated", &goal);
        spawn_goal_loop(state.inner().clone(), app, session_id);
    }
    Ok(Some(goal))
}

/// Delete the goal for a session entirely (RFC 0020 §7 — dismiss/clear). Stops
/// any running loop first (so its next checkpoint can't recreate the file we're
/// deleting), then removes the checkpoint file. Idempotent — clearing a
/// nonexistent goal is fine.
#[tauri::command]
async fn goal_clear(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<(), String> {
    stop_goal_loop(state.inner(), &session_id).await;
    state
        .goals
        .delete(&session_id)
        .map_err(|e| format!("failed to clear goal: {e}"))?;
    let _ = app.emit("goal:cleared", &session_id);
    Ok(())
}

/// Mark a session's goal complete from the FE (RFC 0020 §7 — dual-surface: this
/// is the IPC half of the `goal_complete` capability the agent tool also
/// exposes). Stops any running loop first, then transitions the goal to
/// `Completed` and persists. Returns `None` if there is no goal. Idempotent on a
/// terminal goal (returns it unchanged).
#[tauri::command]
async fn goal_complete(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<Option<Goal>, String> {
    stop_goal_loop(state.inner(), &session_id).await;
    let Some(mut goal) = state
        .goals
        .load(&session_id)
        .map_err(|e| format!("failed to read goal: {e}"))?
    else {
        return Ok(None);
    };
    if !matches!(goal.status, GoalStatus::Completed) {
        goal.status = GoalStatus::Completed;
        goal.updated_ms = now_ms();
        state
            .goals
            .save(&goal)
            .map_err(|e| format!("failed to persist goal: {e}"))?;
        let _ = app.emit("goal:updated", &goal);
    }
    Ok(Some(goal))
}

/// Snapshot a session's `notebook_runner` kernel for the status panel (#871).
/// Read-only, so the panel can poll it while a kernel runs. Returns
/// `has_kernel: false` when the session has no kernel.
#[tauri::command]
async fn notebook_status(
    state: State<'_, Arc<AppState>>,
    session_id: String,
) -> Result<NotebookKernelState, String> {
    Ok(state.notebook_snapshot(&session_id).await)
}

/// Stop the session's `notebook_runner` kernel(s) — the panel's Stop button
/// (#871). With no `kernel_id`, reaps the whole session (FE-1 Stop, back-compat);
/// with a `kernel_id`, stops just that kernel — the multi-kernel switcher's
/// per-tab Stop (#871 FE-2 / #923). Idempotent: a no-op when the target kernel is
/// already gone. The FE follows up with `notebook_status` to render the result.
#[tauri::command]
async fn notebook_stop(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    kernel_id: Option<String>,
) -> Result<(), String> {
    state.notebook_stop(&session_id, kernel_id.as_deref()).await;
    Ok(())
}

/// Restart a session's `notebook_runner` kernel (the panel's Restart button,
/// #871 FE-2 / #922): stop the current kernel and spawn a fresh one, discarding
/// all in-kernel state (a new kernel id, execution count reset to 0). Returns
/// the post-restart snapshot so the FE renders the fresh kernel without a
/// follow-up `notebook_status`. `kernel_id` targets a specific kernel when
/// given (forward-compat for Phase 3 multi-kernel), else the representative one.
#[tauri::command]
async fn notebook_restart(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    kernel_id: Option<String>,
) -> Result<NotebookKernelState, String> {
    state
        .notebook_restart(&session_id, kernel_id.as_deref())
        .await
}

/// List a session's active background observers (#1038, epic #954 M2) — backs
/// the `👁 Observers` panel. Oldest id first; only the caller's session.
#[tauri::command]
async fn list_observers(
    state: State<'_, Arc<AppState>>,
    session_id: String,
) -> Result<Vec<ObserverInfo>, String> {
    Ok(state.list_observers(&session_id))
}

/// Stop one observer by id (#1038 M2) — the panel's `[×]`. Only the owning
/// session may stop it; an unknown/foreign id is a no-op error. Emits
/// `observer:changed` so the panel (and any other view) re-lists.
#[tauri::command]
async fn stop_observer(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    id: u64,
    session_id: String,
) -> Result<(), String> {
    state.stop_observer(id, &session_id).await?;
    let _ = app.emit("observer:changed", ObserverChangedEvent { session_id });
    Ok(())
}

/// Fire a scheduled task immediately, off-schedule (RFC 0017 §8.3). Runs the
/// same bounded headless turn the scheduler would, records the run, and stamps
/// `last_run` so the manual fire counts as the most recent run (and the
/// background sweep will not immediately re-fire it). Returns the `RunRecord`
/// and emits `scheduled:fired` + `scheduled:changed` so the UI live-updates.
#[tauri::command]
async fn run_scheduled_task_now(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    id: String,
) -> Result<RunRecord, String> {
    let task = state
        .scheduled
        .get(&id)
        .ok_or_else(|| format!("unknown scheduled task: {id}"))?;
    // The global kill-switch blocks manual fires too — "pause all" must mean
    // nothing fires unattended, including a stray "run now" (RFC 0017 §8.3).
    if state.scheduled.is_all_paused() {
        return Err("scheduled tasks are globally paused".into());
    }
    let runner = DesktopTaskRunner {
        state: state.inner().clone(),
        app: app.clone(),
    };
    let fired_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let outcome = ff_scheduled::TaskRunner::fire(&runner, &task).await;
    let record =
        state
            .scheduled
            .append_run(&task.id, outcome.session_id.as_deref(), outcome.status);
    state.scheduled.stamp_last_run(&task.id, fired_ms);
    emit_scheduled_fired(&app, &state.scheduled, &record);
    Ok(record)
}

/// Emit `scheduled:fired` (the full `RunRecord`, matching the FE binding) plus a
/// `scheduled:changed` snapshot of the task list, so the UI updates run history,
/// the ↗ session link, and the derived next/last stamps without polling (#544).
fn emit_scheduled_fired(
    app: &tauri::AppHandle,
    store: &ff_scheduled::ScheduledStore,
    run: &RunRecord,
) {
    let _ = app.emit("scheduled:fired", run);
    let _ = app.emit("scheduled:changed", store.list());
}

/// Clone a session and its transcript into a fresh session (server-truth).
/// Backs the sidebar/split Duplicate action.
#[tauri::command]
fn fork_session(state: State<'_, Arc<AppState>>, session_id: String) -> Result<Session, String> {
    state
        .store
        .fork_session(&session_id)
        .ok_or_else(|| format!("unknown session: {session_id}"))
}

/// The working directory a session's tools run in (slice 3b, issue #200).
/// Returns the session's chosen workspace (or the global default when unset)
/// together with its current git branch when the cwd is a repo (#211).
#[tauri::command]
fn get_session_workspace(state: State<'_, Arc<AppState>>, session_id: String) -> SessionWorkspace {
    let root = state.session_root(&session_id);
    SessionWorkspace {
        git_branch: git_branch(&root),
        path: root.display().to_string(),
    }
}

/// The current git branch of `dir`, or `None` when it is not a git working tree
/// (no `.git/HEAD`) or is in detached HEAD. Reads `.git/HEAD` directly -- cheaper
/// than spawning `git` and dependency-free; `ref: refs/heads/<name>` yields the
/// branch, a bare commit SHA (detached) yields `None`. Extracted as a free fn so
/// it is unit-testable without a Tauri `State`.
pub(crate) fn git_branch(dir: &std::path::Path) -> Option<String> {
    let head = std::fs::read_to_string(dir.join(".git").join("HEAD")).ok()?;
    head.trim()
        .strip_prefix("ref: refs/heads/")
        .map(str::to_string)
}

/// Set a session's working directory. Validates that `path` exists and is a
/// directory (canonicalized so the stored root is absolute and symlink-resolved),
/// then returns the canonical path the UI should display. The chosen directory
/// becomes the `root` for that session's tools -- file tools are jailed to it and
/// `bash` runs in it.
#[tauri::command]
fn set_session_workspace(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    path: String,
) -> Result<String, String> {
    let canonical = resolve_workspace_dir(&path)?;
    let display = canonical.display().to_string();
    state.set_session_cwd(&session_id, canonical);
    Ok(display)
}

/// Canonicalize `path` and confirm it is an existing directory. Returns the
/// absolute, symlink-resolved path on success. Extracted from
/// [`set_session_workspace`] so the validation is unit-testable without a
/// Tauri `State`.
fn resolve_workspace_dir(path: &str) -> Result<std::path::PathBuf, String> {
    let canonical =
        std::fs::canonicalize(path).map_err(|e| format!("cannot resolve directory: {e}"))?;
    if !canonical.is_dir() {
        return Err(format!("not a directory: {}", canonical.display()));
    }
    Ok(canonical)
}

/// Local branch names in `dir`'s git work tree, refname-sorted, for the branch
/// picker (#628). `Ok(vec![])` when `dir` is not a repo; `Err` only on an actual
/// git failure. Uses `for-each-ref` (stable plumbing) so packed refs are included
/// -- reading `.git/refs/heads/` directly, like [`git_branch`], would miss them.
/// Extracted as a free fn so it is unit-testable against a temp repo without a
/// Tauri `State`.
pub(crate) fn list_local_branches(dir: &std::path::Path) -> Result<Vec<String>, String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["for-each-ref", "--format=%(refname:short)", "refs/heads/"])
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !out.status.success() {
        // Not a repo (or a bare/broken one): no branches to offer rather than a
        // hard error, mirroring how `git_branch` returns `None` off-repo.
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Check out `branch` in `dir`. Validates that `branch` is one of the repo's local
/// branches *before* spawning git: this yields a clean error for a stale picker
/// entry and closes the `git checkout -<flag>` argument-injection vector (a branch
/// literally named like a flag can never reach the checkout). On a checkout that
/// git rejects (e.g. the working tree would be overwritten) the trimmed stderr is
/// surfaced verbatim. Free fn for the same testability reason as above.
pub(crate) fn switch_branch(dir: &std::path::Path, branch: &str) -> Result<(), String> {
    if !list_local_branches(dir)?.iter().any(|b| b == branch) {
        return Err(format!("unknown branch: {branch}"));
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["checkout", branch])
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr);
        let msg = msg.trim();
        return Err(if msg.is_empty() {
            format!("git checkout {branch} failed")
        } else {
            msg.to_string()
        });
    }
    Ok(())
}

/// List the local branches of a session's cwd repo, for a switch picker (#628).
/// Read-only; empty when the cwd is not a git work tree.
#[tauri::command]
fn list_branches(
    state: State<'_, Arc<AppState>>,
    session_id: String,
) -> Result<Vec<String>, String> {
    list_local_branches(&state.session_root(&session_id))
}

/// Check out `branch` in a session's cwd and return the updated workspace (#628,
/// item #2 of #618).
///
/// DESIGN -- seamless connection with the #627 flash: this command *emits*
/// `workspace:branch-changed` itself rather than leaning solely on `GitHeadWatcher`
/// to notice the `.git/HEAD` write. The watcher is only re-pointed at turn-start
/// (`align_git_watcher`), so a watcher-only path would (a) fire ~200ms late behind
/// the debounce and (b) miss a switch made in a session that has not run a turn
/// yet. Emitting here routes the change through the exact reactive channel an
/// external checkout uses -- `onWorkspaceBranchChanged` -> store `applyBranchChanged`
/// -> `ff-branch-flash` (#627) -- so the chip updates and flashes immediately and
/// deterministically, with no extra FE wiring. When the watcher *is* pointed it may
/// re-emit the same branch ~200ms later; that is harmless: `applyBranchChanged` is
/// idempotent and #627's transition guard suppresses a second flash on an unchanged
/// value.
#[tauri::command]
fn checkout_branch(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    session_id: String,
    branch: String,
) -> Result<SessionWorkspace, String> {
    let root = state.session_root(&session_id);
    switch_branch(&root, &branch)?;
    let ws = SessionWorkspace {
        git_branch: git_branch(&root),
        path: root.display().to_string(),
    };
    let _ = app.emit("workspace:branch-changed", ws.clone());
    Ok(ws)
}

/// Map an internal [`ff_memory::MemoryFile`] to the IPC [`MemoryFileInfo`] contract.
fn to_file_info(f: ff_memory::MemoryFile) -> MemoryFileInfo {
    MemoryFileInfo {
        name: f.name,
        rel_path: f.rel_path,
        kind: match f.kind {
            ff_memory::MemoryFileKind::Curated => MemoryFileKind::Curated,
            ff_memory::MemoryFileKind::Daily => MemoryFileKind::Daily,
        },
        size_bytes: f.size_bytes as i64,
        modified_ms: f.modified_ms,
    }
}

/// List the curated + daily memory files for the Settings memory pane (Issue #131).
/// Read-only: curated first, then daily newest-first.
#[tauri::command]
fn list_memory_files(state: State<'_, Arc<AppState>>) -> Vec<MemoryFileInfo> {
    state
        .memory()
        .list_files()
        .into_iter()
        .map(to_file_info)
        .collect()
}

/// Read one memory file's body by its root-relative path (from `list_memory_files`).
/// Errors if the path escapes the memory root.
#[tauri::command]
fn read_memory_file(state: State<'_, Arc<AppState>>, rel_path: String) -> Result<String, String> {
    state
        .memory()
        .read_file(&rel_path)
        .ok_or_else(|| "invalid memory path".to_string())
}

/// Default cap for [`read_file`] when the caller passes no `max_bytes`: 512 KiB.
/// Large files are truncated to this prefix so the viewer stays responsive.
const DEFAULT_READ_FILE_BYTES: u64 = 512 * 1024;

/// List one directory level under a session's workspace, for the Files panel
/// (Issue #872). `path` is relative to the session workspace root (`""` or `"."`
/// is the root). Jailed via [`ff_tools::resolve_in_root`] so `..`/symlink
/// escapes are rejected, and `.gitignore`-aware (so `node_modules`, `target/`,
/// etc. are omitted) via the same `ignore` walker the `tree` tool uses. Entries
/// are sorted directories-first, then case-insensitively by name.
#[tauri::command]
fn list_directory(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    path: String,
) -> Result<Vec<DirEntry>, String> {
    list_directory_in(&state.session_root(&session_id), &path)
}

/// The listing logic behind [`list_directory`], split out so it is unit-testable
/// against a temp workspace without a Tauri `State`.
fn list_directory_in(root: &std::path::Path, path: &str) -> Result<Vec<DirEntry>, String> {
    let rel = if path.is_empty() { "." } else { path };
    let dir = ff_tools::resolve_in_root(root, rel)?;
    if !dir.is_dir() {
        return Err(format!("not a directory: {rel}"));
    }

    // `ignore` counts the walk root as depth 0; depth 1 is its direct children.
    let mut walk = ignore::WalkBuilder::new(&dir);
    walk.require_git(false).max_depth(Some(1));

    let mut entries: Vec<DirEntry> = Vec::new();
    for entry in walk.build().flatten() {
        if entry.depth() == 0 {
            continue;
        }
        let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
        let name = entry.file_name().to_string_lossy().into_owned();
        let size = if is_dir {
            0
        } else {
            entry.metadata().map(|m| m.len()).unwrap_or(0)
        };
        entries.push(DirEntry { name, is_dir, size });
    }

    entries.sort_by(|a, b| {
        b.is_dir
            .cmp(&a.is_dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(entries)
}

/// Read one file's body under a session's workspace, for the Files panel viewer
/// (Issue #872). `path` is relative to the session workspace root and jailed like
/// [`list_directory`]. Reads at most `max_bytes` (default [`DEFAULT_READ_FILE_BYTES`]);
/// `truncated` is set when the file is larger. Non-UTF-8 content returns
/// `is_binary: true` with `text: None` so the viewer shows a placeholder instead
/// of raw bytes.
#[tauri::command]
fn read_file(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    path: String,
    max_bytes: Option<u64>,
) -> Result<FileContent, String> {
    read_file_in(&state.session_root(&session_id), &path, max_bytes)
}

/// The read logic behind [`read_file`], split out so it is unit-testable against a
/// temp workspace without a Tauri `State`.
fn read_file_in(
    root: &std::path::Path,
    path: &str,
    max_bytes: Option<u64>,
) -> Result<FileContent, String> {
    use std::io::Read;

    let file_path = ff_tools::resolve_in_root(root, path)?;
    let meta = std::fs::metadata(&file_path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("not a file: {path}"));
    }
    let size = meta.len();
    let cap = max_bytes.unwrap_or(DEFAULT_READ_FILE_BYTES);
    let truncated = size > cap;

    let file = std::fs::File::open(&file_path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let mut bytes = Vec::new();
    file.take(cap)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {path}: {e}"))?;

    // Decode as UTF-8. When truncated, an "unexpected end of input" error
    // (`error_len() == None`) just means the cap fell mid-multibyte-char, so we
    // keep the valid prefix as text rather than mislabeling a text file binary.
    let (text, is_binary) = match std::str::from_utf8(&bytes) {
        Ok(s) => (Some(s.to_owned()), false),
        Err(e) if truncated && e.error_len().is_none() => {
            let valid = &bytes[..e.valid_up_to()];
            (Some(String::from_utf8_lossy(valid).into_owned()), false)
        }
        Err(_) => (None, true),
    };

    Ok(FileContent {
        text,
        is_binary,
        truncated,
        size,
    })
}

/// Summarize the memory store (file/byte counts, root, enabled flag) for the
/// Settings pane header. Read-only — the enable toggle lands with Issue #166.
#[tauri::command]
fn memory_overview(state: State<'_, Arc<AppState>>) -> MemoryOverview {
    let mem = state.memory();
    let files = mem.list_files();
    let total_bytes: i64 = files.iter().map(|f| f.size_bytes as i64).sum();
    MemoryOverview {
        enabled: mem.is_enabled(),
        file_count: files.len() as i64,
        total_bytes,
        root_path: mem.root().display().to_string(),
        decay_enabled: mem.config().decay.enabled,
    }
}

/// Wall-clock now in epoch milliseconds (the read instant for lazy decay). Uses
/// `SystemTime` to avoid pulling `chrono` into this crate just for one call.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A chunk path made relative to the memory root, with forward slashes — the
/// same shape `MemoryFileInfo::rel_path` uses (e.g. `MEMORY.md`,
/// `daily/2026-06-25.md`).
fn chunk_rel_path(root: &std::path::Path, path: &std::path::Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// First non-empty, non-heading line of a chunk, trimmed and capped to ~80
/// chars — the human-readable summary for a Salience list row (#293). Heading
/// lines (hash-prefixed) are skipped because the heading is already its own
/// field; a char slice would cut mid-word and start on blank/heading lines.
fn chunk_preview(text: &str) -> String {
    const MAX: usize = 80;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("#"))
        .unwrap_or("");
    if line.chars().count() > MAX {
        let head: String = line.chars().take(MAX).collect();
        format!("{}…", head.trim_end())
    } else {
        line.to_string()
    }
}

/// Per-chunk salience stats for the Settings Salience surface (#293). Joins every
/// indexed chunk (`all_chunks`) with its `chunk_stats` usage
/// (`chunk_stats_snapshot`); a chunk with no stats row is a never-recalled chunk
/// (weight `1.0`, never dormant, not pinned, no `last_accessed`).
#[tauri::command]
fn list_memory_chunks(state: State<'_, Arc<AppState>>) -> Vec<ff_core::MemoryChunkStat> {
    let mem = state.memory();
    let root = mem.root().to_path_buf();
    let chunks = mem.all_chunks();
    let keys: Vec<String> = chunks.iter().map(ff_memory::chunk_key).collect();
    let stats = state
        .index()
        .chunk_stats_snapshot(&keys, now_ms())
        .unwrap_or_default();
    chunks
        .iter()
        .zip(keys.iter())
        .map(|(chunk, key)| {
            let snap = stats.get(key);
            ff_core::MemoryChunkStat {
                chunk_key: key.clone(),
                rel_path: chunk_rel_path(&root, &chunk.path),
                heading: chunk.heading.clone(),
                preview: chunk_preview(&chunk.text),
                weight: snap.map(|s| s.weight).unwrap_or(1.0),
                access_count: snap.map(|s| s.access_count).unwrap_or(0),
                last_accessed_ms: snap.map(|s| s.last_accessed_ms),
                dormant: snap.is_some_and(|s| s.dormant),
                pinned: snap.is_some_and(|s| s.pinned),
            }
        })
        .collect()
}

/// Replace the whole body of a curated `MEMORY.md` stratum (#969/#868): the
/// Settings → Memory editor sends back the full `## Identity`/`## Patterns`/`## Focus`
/// section content, and this overwrites it (siblings preserved, heading kept even
/// when empty). Reindexes afterward so recall reflects the edit; the atomic
/// single-writer rewrite lives in `ff-memory`.
#[tauri::command]
fn write_curated_memory(
    state: State<'_, Arc<AppState>>,
    stratum: ff_core::Stratum,
    text: String,
) -> CmdResult<()> {
    let mem = state.memory();
    // Wire enum → ff-memory domain enum (same split as MemoryFileKind above).
    let domain = match stratum {
        ff_core::Stratum::Identity => ff_memory::Stratum::Identity,
        ff_core::Stratum::Patterns => ff_memory::Stratum::Patterns,
        ff_core::Stratum::Focus => ff_memory::Stratum::Focus,
    };
    mem.replace_curated_stratum(&text, domain)
        .map_err(|e| e.to_string())?;
    // Best-effort reindex so search/ambient recall reflects the edit; a failure
    // leaves the file written (the source of truth) — mirrors the consolidate path.
    let chunks = mem.all_chunks();
    if let Err(e) = state.index().reindex(&chunks) {
        tracing::warn!(error = %e, "reindex after curated edit failed");
    }
    Ok(())
}

/// Reset (wake) a chunk: restore its weight to `1.0` and stamp `last_accessed`
/// now, creating the stats row if absent (#293). Never edits Markdown.
#[tauri::command]
fn reset_memory_chunk(state: State<'_, Arc<AppState>>, chunk_key: String) -> Result<(), String> {
    state
        .index()
        .reset_chunk(&chunk_key)
        .map_err(|e| e.to_string())
}

/// Sleep a chunk: force its weight to `0.0` so it goes dormant immediately and
/// is skipped from ambient injection, instead of waiting out disuse-decay
/// (#1239). The inverse of [`reset_memory_chunk`], and just as reversible — the
/// chunk stays searchable and is never deleted. Never edits Markdown.
#[tauri::command]
fn sleep_memory_chunk(state: State<'_, Arc<AppState>>, chunk_key: String) -> Result<(), String> {
    state
        .index()
        .sleep_chunk(&chunk_key)
        .map_err(|e| e.to_string())
}

/// Pin/unpin a chunk: a pinned chunk holds effective weight `1.0` (decay
/// skipped) and is never dormant (#293). Never edits Markdown.
#[tauri::command]
fn set_memory_chunk_pinned(
    state: State<'_, Arc<AppState>>,
    chunk_key: String,
    pinned: bool,
) -> Result<(), String> {
    state
        .index()
        .set_chunk_pinned(&chunk_key, pinned)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_turn(state: State<'_, Arc<AppState>>, session_id: String) {
    if let Some(token) = state.take_cancel(&session_id) {
        token.cancel();
    }
    // Pending approvals for this session would block the turn forever otherwise.
    state.cancel_pending_approvals(&session_id);
}

/// Frontend response to a [`ToolApprovalRequestEvent`]. Wakes the awaiting approver.
#[tauri::command]
fn respond_approval(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    call_id: String,
    approved: bool,
) {
    state.resolve_approval(&session_id, &call_id, approved);
}

/// Frontend response to a [`ToolAskRequestEvent`] (#44): the user's answer to an
/// `ask_user` question. Wakes the awaiting tool call, which resumes the turn.
#[tauri::command]
fn respond_ask(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    call_id: String,
    answer: String,
) {
    state.resolve_ask(&session_id, &call_id, answer);
}

// --- Four-option tool approval commands (#229) ---

/// Approve a tool for this session only (in-memory).
#[tauri::command]
fn set_session_approve(state: State<'_, Arc<AppState>>, session_id: String, tool: String) {
    state.set_session_approve(&session_id, &tool);
}

/// Add a tool to the persistent always-approved set.
#[tauri::command]
fn set_always_approve(state: State<'_, Arc<AppState>>, tool: String) {
    state.set_always_approve(&tool);
}

/// Remove a tool from the persistent always-approved set.
#[tauri::command]
fn remove_always_approve(state: State<'_, Arc<AppState>>, tool: String) {
    state.remove_always_approve(&tool);
}

/// List all persistently always-approved tools (sorted).
#[tauri::command]
fn list_always_approved(state: State<'_, Arc<AppState>>) -> Vec<String> {
    state.list_always_approved()
}

/// Persists the user message, then spawns the assistant turn. Tokens stream back
/// over `turn:token`; completion via `turn:done`; failures via `turn:error`.
/// Returns the user message id immediately.
#[tauri::command]
fn send_message(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
    content: String,
    // Sent by the composer (#399); persisted on the user message so `run_turn`
    // forwards them to vision-capable providers. `None` for a plain text turn.
    attachments: Option<Vec<Attachment>>,
) -> CmdResult<String> {
    // Cancel any in-flight turn so a cancel+resend sequence never races two
    // parallel agent loops into the same session (#744). Mirrors edit_message.
    if let Some(token) = state.take_cancel(&session_id) {
        token.cancel();
    }
    state.cancel_pending_approvals(&session_id);

    let user_msg =
        match attachments {
            Some(attachments) if !attachments.is_empty() => state
                .store
                .add_message_with_attachments(&session_id, Role::User, content, attachments),
            _ => state.store.add_message(&session_id, Role::User, content),
        };

    spawn_assistant_turn(state.inner().clone(), app, session_id, 0);

    Ok(user_msg.id)
}

/// Wake `session_id` because an observer fired (#891 Phase 1).
///
/// Two paths, picked on a **non-destructive** liveness probe
/// (`has_active_turn`) of the session's registered cancel token
/// (#1018 Path A — the probe must never strip the live turn's token):
///
/// - **Turn in flight** (`has_active_turn` is `true`): the in-flight
///   turn owns the cancel, so the event is deferred — appended to the
///   supervisor's per-session queue. `spawn_assistant_turn` drains the
///   queue on the next `send_message` / `edit_message` / observer wake.
///   We DO NOT cancel the in-flight turn (`cancel_turn` would race the
///   user's own cancel and could leave the transcript inconsistent),
///   we DO NOT touch its cancel token (probing with the destructive
///   `take_cancel` used to strip it, corrupting the running tool call —
///   `[no result recorded]`), and we DO NOT drop the event (the user
///   explicitly registered the observer to see this change).
/// - **No turn in flight** (`has_active_turn` is `false`): we own the
///   wake. Buffer the event and spawn a fresh assistant turn.
///   `spawn_assistant_turn` is the single drain point: it surfaces the
///   buffered wakes to that turn as **transient, request-only context**
///   (the volatile system block), never as a persisted `Role::User`
///   message (#1018 Path B — persisting a wake pollutes history across
///   relaunch and is semantically wrong: the app can't keep observing
///   once closed).
///
/// Liveness: `has_active_turn` reads the same cancel-token registry
/// `delete_session`, `send_message`, and `edit_message` gate on, so the
/// wake can never race a user-driven turn cancel. If a turn completes
/// between the probe and the buffer push, the next `spawn_assistant_turn`
/// (from any source) will surface the event — the event is never lost.
pub(crate) async fn wake_session_for_observer(
    state: &Arc<AppState>,
    event: ObserverEvent,
    app: &tauri::AppHandle,
) {
    let session_id = event.session_id.clone();
    let observer_id = event.id;
    // #1038 M2: an observer firing is a change the panel wants to reflect
    // (coarse — the FE re-lists). Emit regardless of the defer/spawn branch below.
    let _ = app.emit(
        "observer:changed",
        ObserverChangedEvent {
            session_id: session_id.clone(),
        },
    );
    if state.has_active_turn(&session_id) {
        // Turn in flight: defer — but only *probe* liveness, never
        // strip the live turn's cancel token (#1018 Path A). The old
        // `take_cancel` probe removed the token as a side effect, which
        // desynced the in-flight turn's completion bookkeeping and let a
        // rapid follow-up wake spawn a competing turn, dropping the
        // running tool call's result (`[no result recorded]`).
        // `buffer_event` re-inserts into the same queue the next
        // `spawn_assistant_turn` will drain.
        state.buffer_observer_event(&session_id, event);
        tracing::info!(
            session_id = %session_id,
            observer_id = observer_id,
            "observer event deferred (turn in flight)"
        );
        return;
    }

    // No turn in flight: buffer the current event and spawn a turn.
    // `spawn_assistant_turn` is the single point that drains the buffer
    // and surfaces the wakes transiently (#1018 Path B) — request-only
    // context, never persisted to the transcript. Buffering here (rather
    // than persisting via `add_message`) keeps a wake ephemeral: it must
    // not survive relaunch as a `Role::User` row polluting history.
    state.buffer_observer_event(&session_id, event);
    tracing::info!(
        session_id = %session_id,
        observer_id = observer_id,
        "observer wake spawning turn"
    );
    spawn_assistant_turn(state.clone(), app.clone(), session_id, 0);
}

/// Bridge one background process's live output to the frontend (#873). Spawned
/// by `start_process_output_pump` on each `process_manager start`, this task
/// forwards every [`ProcessChunk`] the process emits as a `process:output`
/// event — independently of any assistant turn, for the life of the process —
/// then emits a terminal `process:exited` when the output broadcast closes
/// (the exit-watcher drops the sender on exit/kill). `chunks` was subscribed
/// inside `ProcessSupervisor::start` before the drain tasks spawned, so no
/// output is missed.
pub(crate) fn spawn_process_output_bridge(
    app: tauri::AppHandle,
    supervisor: Arc<ff_tools::process::ProcessSupervisor>,
    id: u64,
    session_id: String,
    mut chunks: tokio::sync::broadcast::Receiver<ff_tools::process::ProcessChunk>,
) {
    use tokio::sync::broadcast::error::RecvError;
    tokio::spawn(async move {
        let process_id = id as u32;
        loop {
            match chunks.recv().await {
                Ok(chunk) => {
                    let stream = match chunk.stream {
                        ff_tools::OutputStream::Stderr => OutputStreamKind::Stderr,
                        ff_tools::OutputStream::Stdout => OutputStreamKind::Stdout,
                    };
                    let _ = app.emit(
                        "process:output",
                        ProcessOutputEvent {
                            session_id: session_id.clone(),
                            process_id,
                            stream,
                            delta: String::from_utf8_lossy(&chunk.bytes).into_owned(),
                        },
                    );
                }
                // The UI fell behind and the bounded broadcast dropped `n`
                // chunks. Surface the gap as a stderr notice rather than
                // silently losing output, and keep forwarding.
                Err(RecvError::Lagged(n)) => {
                    let _ = app.emit(
                        "process:output",
                        ProcessOutputEvent {
                            session_id: session_id.clone(),
                            process_id,
                            stream: OutputStreamKind::Stderr,
                            delta: format!(
                                "\n[... {n} output chunk(s) dropped (UI fell behind) ...]\n"
                            ),
                        },
                    );
                }
                Err(RecvError::Closed) => break,
            }
        }
        // Process ended: the map entry lives until the reaper, so the status
        // label is still readable here.
        let status = supervisor
            .status_label(id, &session_id)
            .unwrap_or_else(|| "exited".to_string());
        let _ = app.emit(
            "process:exited",
            ProcessExitedEvent {
                session_id,
                process_id,
                status,
            },
        );
    });
}

/// Render deferred observer wakes into a single transient, request-only
/// context block for a turn (#1018 Path B). Returns `None` when there are
/// no wakes. The result is folded into the turn's `extra_instructions` (the
/// volatile system block) — seen by the turn but never persisted, so wakes
/// can't survive relaunch as `Role::User` rows polluting history.
fn observer_wake_context(events: &[ff_observer::ObserverEvent]) -> Option<String> {
    if events.is_empty() {
        return None;
    }
    let lines: Vec<String> = events
        .iter()
        .map(|ev| format!("- [Observer \"{}\"]: {}", ev.label, ev.summary))
        .collect();
    Some(format!(
        "## Observer wakes\nWhile you were busy, these watched targets changed. \
         Treat as already-acknowledged background context — act on them only if relevant:\n{}",
        lines.join("\n")
    ))
}

/// Max consecutive observer-driven drain turns before we stop auto-spawning
/// successors (#1096). Bounds the self-sustaining loop where a drain turn writes
/// a file that an observer watches, which re-buffers a wake, which would spawn
/// another drain turn, and so on. On cap the buffer is *retained* (not spawned,
/// not cleared) so the next real user turn drains it — no silent wake drop.
const MAX_DRAIN_TURNS: u32 = 3;

/// Decide whether the post-turn tail should spawn a successor "drain" turn to
/// surface observer wakes buffered during the turn (#1095). Pure so the cap
/// boundary is unit-testable without driving `run_turn`. Spawn only when the
/// session went genuinely idle (#1018), wakes are actually buffered, and we
/// haven't hit the consecutive-drain cap (#1096).
fn should_spawn_drain(went_idle: bool, has_buffered: bool, drain_count: u32) -> bool {
    went_idle && has_buffered && drain_count < MAX_DRAIN_TURNS
}

/// Set up and spawn the assistant turn for `session_id`: snapshots the provider,
/// resolves the session's phenotype/mode, builds the tool registry + system
/// prompt, runs the turn (streaming over `turn:*` / `tool:*`), and folds the
/// per-turn telemetry. Shared by `send_message` (after persisting the user turn)
/// and `edit_message` (after editing + truncating), so both paths run identical
/// turn semantics.
///
/// `drain_count` is the number of consecutive observer-driven "drain" turns that
/// led here (#1096): `0` for a normal user- or observer-initiated turn, and
/// `n+1` when the post-turn tail spawns a successor to surface wakes buffered
/// during a turn that went idle. Bounded by [`MAX_DRAIN_TURNS`] via
/// [`should_spawn_drain`] so a self-sustaining wake loop can't run unbounded.
fn spawn_assistant_turn(
    state: Arc<AppState>,
    app: tauri::AppHandle,
    session_id: String,
    drain_count: u32,
) {
    // Drain any observer wakes deferred while a turn was in flight and
    // surface them to THIS turn as transient, request-only context
    // (#1018 Path B) — never as persisted `Role::User` rows. A wake is
    // ephemeral ("state X changed while you were busy"); persisting it
    // pollutes history across relaunch and is semantically wrong (the
    // app can't keep observing once closed). This is the single drain
    // point: the wake path buffers and delegates here.
    let buffered = state.drain_observer_buffer(&session_id);
    let wake_context: Option<String> = observer_wake_context(&buffered);
    // Captured before `wake_context` is moved into `extra_instructions` below, so
    // the turn's outcome log can say whether this was an observer-initiated turn.
    let is_observer_wake = wake_context.is_some();
    if wake_context.is_some() {
        tracing::info!(
            session_id = %session_id,
            count = buffered.len(),
            "surfacing observer wakes as transient turn context"
        );
    }

    let cancel = CancelToken::new();
    state.register_cancel(&session_id, cancel.clone());
    // A clone kept by the host so the post-turn telemetry can tell a clean finish
    // from a user cancel (the original is moved into `run_turn`).
    let cancel_probe = cancel.clone();

    // Snapshot the provider from the current config for this turn; a settings
    // change between turns is picked up on the next `send_message`.
    // Resolve THIS session's phenotype (#246): an explicit per-pane binding, else
    // the global active one. It supplies the persona, active skills, and the loop
    // iteration cap for the turn (RFC 0001 §7) — so two panes can run different Phenos.
    let pheno = state.session_phenotype(&session_id);
    // Resolve the (connection, model) for the turn via the three-tier precedence
    // session > phenotype > global (RFC 0005 §11.2) and build the provider for the
    // RESOLVED connection. This routes a phenotype model override through its intended
    // endpoint rather than the global active one (fixes RFC 0005 §11.1).
    let selection = state.resolve_model_selection(&session_id);
    let (mut provider, _) =
        state.build_provider_for(Some(&selection.connection), Some(&selection.model));
    let model = selection.model;
    let conn_id = selection.connection;
    let persona = pheno.persona.clone();
    // Per-pane iteration cap (#244-R3 x #246): the resolved phenotype carries
    // `max_iterations`, so the bound Pheno governs the loop cap with no extra plumbing.
    let max_iterations = pheno
        .max_iterations
        .unwrap_or(ff_agent::DEFAULT_MAX_ITERATIONS);
    // Resolve THIS session's autonomy mode (#265): an explicit per-pane binding, else
    // the global default. Governs tool capability (Plan), the prompt steer, and the
    // approval policy for the turn.
    let mode = state.session_mode(&session_id);
    tauri::async_runtime::spawn(async move {
        let sid = session_id.clone();
        let approver = UiApprover {
            app: app.clone(),
            state: state.clone(),
            session_id: sid.clone(),
            mode,
            autonomous: false,
            allow_propose_pr: false,
            approval_timeout: CHAT_APPROVAL_TIMEOUT,
        };
        // Point the workspace-aware MCP server (codegraph) at this session's workspace
        // before snapshotting tools, so its code graph reflects the active checkout
        // (#548 W1b). Idempotent; restarts codegraph only when the path changed.
        let session_root = state.session_root(&sid);
        state.align_session_mcp(&sid, &session_root).await;
        // Re-aim the git HEAD watcher at this session's checkout (#561 BE half),
        // mirroring the codegraph alignment above. Idempotent; a no-op when the path
        // is unchanged. Drives the live `workspace:branch-changed` event the FE
        // listener (PR #581) patches into the composer chip.
        state.align_git_watcher(&session_root);
        // Snapshot built-in + MCP-bridged tools for this turn (RFC 0003 §6).
        let registry = state.build_tool_registry(&session_root);
        // Snapshot the advertised-tools matrix for this turn (#702) so the model
        // sees a stable tool list. The approval gate reads the matrix live (see
        // `UiApprover::approve`), so Control-panel edits still take effect on the
        // next tool call — only the advertised set is turn-bounded.
        let permission_matrix = state.permission_matrix();
        let mut tool_ctx = ToolContext::new(
            &registry,
            &session_root,
            &approver,
            max_iterations,
            &permission_matrix,
        );
        tool_ctx.mode = mode;
        tool_ctx.egress = state.session_phenotype(&session_id).egress;
        tool_ctx.search_sources = state.session_phenotype(&session_id).search_sources.clone();
        tool_ctx.abstractive = crate::state::abstractive_config_from_env();
        tool_ctx.compaction_model = state.compaction_model(&conn_id);
        tool_ctx.compaction_budget = state.compaction_budget(&conn_id);
        tool_ctx.near_budget_tokens = state.near_budget(&conn_id);
        tool_ctx.compaction_cache = Some(&state.compaction_cache);
        tool_ctx.tool_search = Some(state.tool_search());
        // Skills + ambient context for this turn (RFC 0001 §4, RFC 0002 phase 1):
        // the resolved persona, installed-skill descriptions, the bodies of the
        // active skills, and the current local time.
        let skills = state.skills_snapshot();
        let user_ctx =
            ff_agent::UserContext::now().with_working_dir(session_root.display().to_string());
        // Active-skill source (#246): an explicit per-pane binding uses the
        // phenotype's declared skills; an unbound session keeps the global active
        // set so the command palette still affects turns. See `turn_active_skills`.
        let active: Vec<String> = state.turn_active_skills(&sid);
        let mcp_guidance = state.mcp_guidance(&sid);
        let (inject_mem, extra_instructions) = state.turn_prompt_injection(&session_root);
        // Fold any deferred observer wakes into the turn's request-only
        // instructions (#1018 Path B): they ride in the volatile system
        // block, seen by this turn but never persisted to the transcript.
        let extra_instructions = match (extra_instructions, wake_context) {
            (Some(base), Some(wake)) => Some(format!("{base}\n\n{wake}")),
            (Some(base), None) => Some(base),
            (None, wake) => wake,
        };
        let (memory, ambient_keys) = if inject_mem {
            state
                .memory()
                .ambient_block_filtered_keyed(state.index().as_ref())
        } else {
            (None, vec![])
        };
        let system_prompt_inputs = ff_agent::SystemPromptInputs {
            persona: persona.as_deref(),
            skills: &skills,
            active: &active,
            user: &user_ctx,
            memory: memory.as_deref(),
            extra_instructions: extra_instructions.as_deref(),
            goal: None,
            mode,
            mcp_guidance: &mcp_guidance,
        };

        // Telemetry (RFC 0001 §8): one SkillActivated per active skill, plus a
        // wall-clock start and a per-turn metrics accumulator the event closure
        // folds into. `turns` = distinct assistant message ids (one per loop
        // iteration); `tokens` = a coarse char/4 proxy over streamed assistant text.
        for skill in &active {
            state.record_skill_activated(skill);
            let _ = app.emit(
                "skill:activated",
                SkillActivated {
                    skill: skill.clone(),
                    session_id: sid.clone(),
                },
            );
        }
        let turn_start = std::time::Instant::now();
        let metrics = std::sync::Arc::new(std::sync::Mutex::new(TurnMetrics::default()));
        let metrics_for_events = metrics.clone();

        // Prime the compaction budget against the *served* context window (#612).
        // For Ollama this is the already-cached `/api/ps` probe (the same value
        // the model chip displays); other providers no-op. Before the model is
        // resident `/api/ps` reports nothing, so this falls to the conservative
        // default -- a safe under-fill -- and picks up the real window once loaded.
        provider.set_context_budget(state.served_window(&sid).await.window);

        let thinking = state.provider_config().thinking;
        let reasoning_visibility = state.provider_config().reasoning_visibility;
        let result = ff_agent::run_session_turn(
            provider.as_ref(),
            state.store.as_ref(),
            &tool_ctx,
            &sid,
            &model,
            &system_prompt_inputs,
            thinking,
            reasoning_visibility,
            cancel,
            |event| {
                // Telemetry (RFC 0001 §8): fold per-turn metrics for events
                // that carry a message_id. Token also counts streamed chars as
                // a coarse token-cost proxy. In-process only — the sidecar path
                // has no local accumulator and goes straight to emit_agent_event.
                if let Ok(mut m) = metrics_for_events.lock() {
                    match &event {
                        AgentEvent::Token { message_id, delta } => {
                            m.note_turn(message_id);
                            m.tokens += ff_llm::count_tokens(delta);
                        }
                        AgentEvent::Reasoning { message_id, .. }
                        | AgentEvent::ToolCallStarted { message_id, .. } => {
                            m.note_turn(message_id);
                        }
                        AgentEvent::Done {
                            message_id,
                            prefill_estimates,
                            prompt_latency_ms,
                            tier2_ms,
                            tier1_fires,
                            tier2_fires,
                            retrieve_calls,
                            ..
                        } => {
                            m.note_turn(message_id);
                            m.note_done(
                                prefill_estimates.as_deref().unwrap_or(&[]),
                                *prompt_latency_ms,
                                *tier2_ms,
                                tier1_fires.unwrap_or(0),
                                tier2_fires.unwrap_or(0),
                                retrieve_calls.unwrap_or(0),
                            );
                        }
                        AgentEvent::MemoryFlushed { message_id, .. } => {
                            m.note_turn(message_id);
                            m.note_flush();
                        }
                        _ => {}
                    }
                }
                // Wire mapping is shared with the sidecar path via
                // emit_agent_event, so the two cannot drift — the whole point
                // of the sidecar parity test (RFC 0004 §5).
                emit_agent_event(&app, &sid, event);
            },
        )
        .await;

        if let Err(ref e) = result {
            // The blind spot that stalled the #1117 diagnosis: this path emitted
            // `turn:error` to the FE and logged NOTHING. A user-initiated turn
            // still shows the error in the transcript, but an observer-initiated
            // turn has no one watching — it failed, wrote no row, and left no
            // trace, so the log could not say whether the wake turn ran and
            // failed or never ran at all. Log every turn failure, tagged with
            // whether a wake initiated it.
            tracing::error!(
                session_id = %session_id,
                observer_wake = is_observer_wake,
                drain_count,
                error = %e,
                "turn failed"
            );
            let _ = app.emit(
                "turn:error",
                TurnErrorEvent {
                    session_id: session_id.clone(),
                    message: e.to_string(),
                },
            );
        } else if is_observer_wake {
            // Prove a wake turn that produced no visible output actually *succeeded*,
            // rather than failing silently.
            tracing::info!(
                session_id = %session_id,
                drain_count,
                // Record the shape of the returned message. The original silent wake
                // (#1117) is fixed — it was a Bedrock 400 on an assistant-terminated
                // request, which now lands in the `turn failed` branch above rather
                // than here. So `content_len = 0` with no tool calls no longer means
                // "unexplained"; it means the turn genuinely completed and chose to
                // say nothing (an empty/NO_REPLY reply), which is its own bug class.
                message_id = result.as_ref().map(|m| m.id.as_str()).unwrap_or(""),
                content_len = result.as_ref().map(|m| m.content.len()).unwrap_or(0),
                tool_calls = result
                    .as_ref()
                    .ok()
                    .and_then(|m| m.tool_calls.as_ref())
                    .map_or(0, |c| c.len()),
                "observer wake turn completed without error"
            );
        }

        // #1038 M2: the `observer` tool's start/stop run inside `ff-observer`,
        // which has no `AppHandle` and so cannot emit — meaning the panel would
        // never learn about observers the agent attached or stopped *via the
        // tool* this turn (its only other refresh signals are an observer
        // *firing* and a panel-driven `stop_observer`). Emit one coarse
        // `observer:changed` at turn end so the panel re-lists whatever the turn
        // changed. Cheap and idempotent: a session with no observers just
        // re-reads an empty list (panel stays hidden). Fires on both Ok and Err
        // since a start can precede a later turn error.
        let _ = app.emit(
            "observer:changed",
            ObserverChangedEvent {
                session_id: sid.clone(),
            },
        );

        // Telemetry (RFC 0001 §8): fold this turn's metrics into each active skill's
        // aggregate and emit a SkillCompleted per skill. Success = a clean finish
        // (run_turn returned Ok and the turn was not cancelled).
        let turn_end = std::time::Instant::now();
        let (
            output_tokens,
            turn_count,
            round_trips,
            iter_ms,
            flushes,
            prefill_estimates,
            tier1_fires,
            tier2_fires,
            retrieve_calls,
            first_token_ms,
            prompt_latency_ms,
            tier2_ms,
        ) = metrics
            .lock()
            .map(|m| {
                let (c, t) = m.snapshot();
                let (rt, ims, fl, ttft) = m.timing(turn_start, turn_end);
                (
                    c,
                    t,
                    rt,
                    ims,
                    fl,
                    m.prefill_estimates.clone(),
                    m.tier1_fires,
                    m.tier2_fires,
                    m.retrieve_calls,
                    ttft,
                    m.prompt_latency_ms,
                    m.tier2_ms,
                )
            })
            .unwrap_or_default();
        let success = result.is_ok() && !cancel_probe.is_cancelled();
        let latency_ms = u32::try_from(turn_end.saturating_duration_since(turn_start).as_millis())
            .unwrap_or(u32::MAX);
        let tokens = u32::try_from(output_tokens).unwrap_or(u32::MAX);
        let turns = u32::try_from(turn_count).unwrap_or(u32::MAX);

        // F1 (#427): emit the per-turn timing baseline the performance epic (#426)
        // measures every later change against. Additive telemetry -- never alters
        // turn behavior. `tracing` mirrors it to the dev log for a `tauri dev` run.
        let stats = TurnStatsEvent {
            session_id: sid.clone(),
            round_trips,
            total_ms: latency_ms,
            iter_ms,
            flushes,
            output_tokens: u32::try_from(output_tokens).unwrap_or(u32::MAX),
            // F1b fields are Option on the wire (#475 follow-up); the desktop
            // always populates them.
            prefill_estimates: Some(prefill_estimates),
            tier1_fires: Some(tier1_fires),
            tier2_fires: Some(tier2_fires),
            // #1045: recall cost of the layered fold.
            retrieve_calls: Some(retrieve_calls),
            // TTFT: `None` when the turn produced no assistant message (early
            // error / cancel before the first token streamed). Otherwise the ms
            // from `run_turn` dispatch to the first assistant token arriving --
            // the answer to "why is first-byte slow?".
            first_token_ms,
            // #960: pure provider round-0 prefill latency. `promptLatencyMs /
            // firstTokenMs` is the prefill share -- how much of the wait was
            // prefill (cache-addressable) vs pre-first-token flush/reasoning.
            prompt_latency_ms,
            // #971: per-phase compaction wall-clock split out of `first_token_ms`.
            tier2_ms,
        };
        tracing::info!(
            target: "turn_metrics",
            session_id = %sid,
            round_trips,
            total_ms = latency_ms,
            first_token_ms = ?stats.first_token_ms,
            flushes,
            output_tokens = stats.output_tokens,
            iter_ms = ?stats.iter_ms,
            tier1_fires = stats.tier1_fires,
            tier2_fires = stats.tier2_fires,
            prefill_estimates = ?stats.prefill_estimates,
            "turn metrics (F1 baseline)"
        );
        let _ = app.emit("turn:stats", stats);

        for skill in &active {
            let ev = SkillCompleted {
                skill: skill.clone(),
                session_id: sid.clone(),
                tokens,
                latency_ms,
                turns,
                success,
            };
            state.record_skill_completed(&ev);
            let _ = app.emit("skill:completed", ev);
        }
        // Persist the turn's telemetry once, lock-free (addresses #77 nit 1).
        state.persist_signals();

        // Drop the session's cancel token *before* the flush — but only if it is
        // still THIS turn's token. The map is keyed by session_id alone, so a
        // successor turn (e.g. the re-run that `edit_message` spawns after
        // cancelling this one) may have already replaced it via register_cancel.
        // Removing unconditionally would strip the live successor's token, killing
        // its Stop button and auto-denying its tool approvals. The identity check
        // on cancel_probe (the task-local clone this turn owns) leaves a
        // successor's token intact and also subsumes the single-turn case where
        // the next turn registers during this turn's multi-second silent flush.
        //
        // `Some` here is the precise "this turn landed and no successor replaced
        // it" signal — i.e. the session just went genuinely idle. We use it
        // below to drain any observer wakes that were buffered while this turn
        // ran (#1095): without it, a wake that fires during a turn with no
        // following user input sits in the buffer indefinitely.
        let went_idle = state.take_cancel_if(&session_id, &cancel_probe).is_some();

        // Persist any mode-switch marker deferred because the user switched
        // autonomy mode while THIS turn was in flight (#1066). The turn has
        // settled and its tool batch is complete, so the marker now lands as a
        // well-formed trailing row rather than interposed in a tool_use/
        // tool_result pair. Runs on every terminal path (success, error, cancel)
        // so the signal is never silently lost.
        state.flush_deferred_mode_markers(&session_id);

        // Pre-compaction memory flush (RFC 0006 §7.2): once the visible turn has
        // finished cleanly, persist any durable facts before context pressure forces
        // a summarization that would drop them. Silent — never adds to the transcript.
        if success {
            // Weak ambient reinforcement (RFC 0007 §10.1): the turn replied, so
            // refresh the curated chunks that were ambient-injected. No-op unless
            // `decay.ambient_gain > 0`.
            let _ = state.index().reinforce_ambient(&ambient_keys);
            // LLM-summarized session title (#671 item 2b): after the first turn,
            // replace the heuristic first-message title with a one-line summary and
            // announce it so the sidebar re-titles in place. Best-effort — gated to
            // fire once per session, and a failure/timeout leaves the heuristic title.
            if let Some(title) = state
                .generate_session_title(provider.as_ref(), &sid, &model, cancel_probe.clone())
                .await
            {
                state.store.set_title(&sid, title.clone());
                let _ = app.emit(
                    "session:title-updated",
                    SessionTitleUpdatedEvent {
                        session_id: sid.clone(),
                        title,
                    },
                );
            }
            if let Some(writes) = state
                .maybe_flush_memory(provider.as_ref(), &registry, &sid, &model, cancel_probe)
                .await
            {
                // #991: the flush moved off run_turn's critical path, and with it the
                // only MemoryFlushed emission. Re-emit here (same wire event) so the
                // FE "memory auto-updated" provenance notice (#283) is preserved,
                // correlated to the turn's final assistant message.
                if let Ok(msg) = &result {
                    emit_agent_event(
                        &app,
                        &sid,
                        AgentEvent::MemoryFlushed {
                            message_id: msg.id.clone(),
                            writes,
                        },
                    );
                }
            }
        }

        // #1095: if this turn just took the session to genuinely idle (we
        // reclaimed our own cancel token above — no successor turn replaced
        // us) and observer wakes were buffered while we ran, nothing else will
        // deliver them: the buffer's only drain point is a *new*
        // `spawn_assistant_turn`, and none is coming without user input. Spawn
        // one now so the buffered wakes surface as a fresh turn.
        //
        // Safe w.r.t. #1018: gated on `went_idle`, so if a successor turn is
        // already live we don't spawn a competitor — that successor will drain
        // the buffer on its own start. Terminates: the spawned turn drains
        // (empties) the buffer at its entry, so it only re-spawns again if a
        // *new* wake fires during it, which is exactly when another report is
        // warranted — and even then `MAX_DRAIN_TURNS` (#1096) caps consecutive
        // drain turns so a wake that a drain turn itself re-triggers (e.g. it
        // writes a watched file) can't loop unbounded.
        let has_buffered = state.has_buffered_observer_events(&sid);
        if should_spawn_drain(went_idle, has_buffered, drain_count) {
            tracing::info!(
                session_id = %sid,
                drain_count,
                "turn idle with buffered observer wakes; spawning drain turn (#1095)"
            );
            spawn_assistant_turn(state.clone(), app.clone(), sid.clone(), drain_count + 1);
        } else if went_idle && has_buffered {
            // Cap reached: stop spawning, but do NOT drain/clear the buffer.
            // Leaving it intact means the next real user turn surfaces these
            // wakes on its own entry — so we bound the loop without recreating
            // the silent-wake-drop this whole bug class (#1090/#1095) is about.
            tracing::warn!(
                session_id = %sid,
                drain_count,
                "observer drain-turn cap (MAX_DRAIN_TURNS) reached; \
                 buffered wakes retained for the next user turn (#1096)"
            );
        }
    });
}

/// Wall-clock ceiling on a single goal iteration's turn, mirroring
/// `SCHEDULED_FIRE_TIMEOUT`: a stuck provider must not wedge the loop. On
/// timeout the turn is cancelled and the iteration is recorded as failed.
const GOAL_ITERATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// How long a foreground chat turn waits for a human to answer an approval
/// prompt (#1270). Generous — the user may legitimately step away mid-turn —
/// but not infinite, so an abandoned turn eventually releases instead of parking
/// forever holding its context.
const CHAT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(1800);

/// The same wait inside a goal iteration, where the whole point of the loop is to
/// make progress with nobody present, so giving up early is the useful behaviour.
///
/// Since #1256, `approve` short-circuits before this ever applies: an `autonomous`
/// approver turns a `Prompt` into an immediate deny rather than raising a request
/// nobody can answer. So the live consumer of this constant is [`UiApprover::ask`],
/// which has no such short-circuit — a goal-loop `ask_user` still raises a real
/// question, and without a bound it would park the loop exactly as #1270 describes.
///
/// MUST stay below [`GOAL_ITERATION_TIMEOUT`]: if the iteration budget expired
/// first, the turn is recorded as a *failure* rather than the model converging on
/// a stated dismissal — which is the entire point of the fix.
const GOAL_APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

// That ordering is the whole fix, so pin it at compile time rather than trusting
// the comment: lowering the iteration budget under the approval budget would
// silently restore the "iteration failed" behaviour this issue set out to remove.
const _: () = assert!(
    GOAL_APPROVAL_TIMEOUT.as_secs() < GOAL_ITERATION_TIMEOUT.as_secs(),
    "GOAL_APPROVAL_TIMEOUT must expire before GOAL_ITERATION_TIMEOUT, or an \
     unanswered prompt fails the iteration instead of denying the call"
);

/// The neutral continuation nudge seeded as the user turn when a goal iteration
/// has no pending steer (#778). It deliberately does NOT repeat the objective:
/// the system-prompt goal block (#718, `ff_agent::system_prompt`) already carries
/// the objective, progress ledger, and the `goal_complete` instruction, so
/// inlining the objective here would duplicate it every iteration (and could
/// drift from the sysprompt wording). Keep this generic — it only tells the agent
/// to take the next step against the goal already described in its instructions.
const GOAL_CONTINUE_NUDGE: &str =
    "Continue toward the goal described in your instructions. Take the next \
     concrete step, or call the `goal_complete` tool if it is fully met and verified.";

/// The host-side [`GoalIteration`] (RFC 0020 §5.2, #716): drives one headless
/// agent turn toward the objective, mirroring the scheduled runner's `fire` but
/// against the live session with the interactive [`UiApprover`] — so a mid-loop
/// approval / `ask_user` still surfaces in the UI (the RFC 0017 join point).
/// Per-tool safety gating already happens inside `run_turn` via the shared
/// `permission_matrix`; the loop-level [`GoalIteration::gate`] is a coarse
/// per-iteration pre-flight (#719/#682) that halts the *whole* loop when the
/// active mode's matrix posture says an autonomous iteration shouldn't run
/// unattended.
struct GoalLoopIteration {
    state: Arc<AppState>,
    app: tauri::AppHandle,
    session_id: String,
    /// Daily `chunk_key`s the loop's `memory_write` calls touched, settled
    /// against the run's verdict when the loop ends (#1292).
    touch_log: ff_memory::TouchLog,
}

/// Coarse per-iteration goal gate (#719): decide whether to spend another
/// unattended iteration given the active `mode`'s permission-matrix posture.
///
/// The loop is headless-autonomous, so we gate on the `Sensitive` tier — the
/// representative "externally-visible autonomous work" a goal iteration is
/// expected to do (network egress, sub-agent spawn; RFC 0019 §Safety). Per-tool
/// gating (including `Dangerous`→deny and the Ask *prompt*) still runs inside the
/// turn via the shared matrix; this pre-flight only governs loop continuation:
/// - `Allow`  → [`GateDecision::Proceed`] (e.g. Act: run autonomously).
/// - `Ask`    → [`GateDecision::Pause`]  (e.g. Auto: pause & surface, resumable —
///   an unattended loop must not silently auto-approve a stream of Ask calls).
/// - `Deny`   → [`GateDecision::Deny`]   (a matrix edited to deny Sensitive in
///   this mode -- no default mode does, post-#793 -- halts the loop).
fn goal_gate_for(mode: Mode, matrix: &ff_core::PermissionMatrix) -> GateDecision {
    match matrix.cell(mode, ff_core::Safety::Sensitive) {
        PermissionCell::Allow => GateDecision::Proceed,
        PermissionCell::Ask => GateDecision::Pause,
        PermissionCell::Deny => GateDecision::Deny,
    }
}

#[async_trait::async_trait]
impl GoalIteration for GoalLoopIteration {
    fn gate(&self, _goal: &Goal) -> GateDecision {
        // Coarse loop-continuation gate keyed on the active mode's matrix posture
        // for the Sensitive tier (#719). Per-tool matrix gating — including the
        // Ask *prompt* and Dangerous deny — is still enforced inside the turn via
        // the shared `permission_matrix`; a paused/denied goal is checkpointed by
        // `drive_goal` from the returned decision. Read live so a Control-panel
        // matrix edit takes effect on the next boundary (#702/#742).
        let mode = self.state.session_mode(&self.session_id);
        goal_gate_for(mode, &self.state.permission_matrix())
    }

    async fn verify(&self, goal: &Goal) -> VerifyOutcome {
        // #684 D3: run the goal's verify_cmd against the session's working
        // directory so a `goal_complete` claim is proven (green) before the loop
        // accepts it. Uses the shared runner so desktop + CLI cannot diverge.
        let workspace = self.state.session_root(&self.session_id);
        ff_agent::run_verify_command(goal, &workspace).await
    }

    async fn run_once(&self, goal: &Goal) -> IterationOutcome {
        // Seed the continuation turn. A pending steer (a message the user typed
        // while the goal ran) takes priority as the turn content; otherwise a
        // neutral "continue" nudge. The objective is NOT repeated here — the
        // system-prompt goal block (#718) already carries the objective,
        // progress, and the `goal_complete` instruction, so inlining it again
        // duplicates it every iteration and drifts if it were ever reworded
        // (#778). The steer is one-shot: the loop clears `pending_steer` on the
        // in-memory goal (via `steer_consumed`) before it checkpoints, so it is
        // applied once and not re-persisted next boundary (#753 review nit 1).
        let steer = goal
            .pending_steer
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let steer_consumed = steer.is_some();
        let prompt = match steer {
            Some(s) => s.to_string(),
            None => GOAL_CONTINUE_NUDGE.to_string(),
        };
        self.state
            .store
            .add_message(&self.session_id, Role::User, prompt.clone());

        let sid = self.session_id.clone();
        let selection = self.state.resolve_model_selection(&sid);
        let (mut provider, model) = self
            .state
            .build_provider_for(Some(&selection.connection), Some(&selection.model));
        let pheno = self.state.session_phenotype(&sid);
        let persona = pheno.persona.clone();
        let mode = self.state.session_mode(&sid);
        let max_iterations = pheno
            .max_iterations
            .unwrap_or(ff_agent::DEFAULT_MAX_ITERATIONS);

        let session_root = self.state.session_root(&sid);
        self.state.align_session_mcp(&sid, &session_root).await;
        self.state.align_git_watcher(&session_root);
        let registry = self
            .state
            .build_tool_registry_with_touch_log(&session_root, Some(self.touch_log.clone()));
        // #1179 3B: before the agent reads the unlocked set, so declared tools land
        // in the stable prompt region rather than looking like a mid-turn unlock.
        if let Some(dropped) = self.state.align_session_preheat(&sid, &registry) {
            let _ = self.app.emit("phenotype:preheat-dropped", dropped);
        }

        let approver = UiApprover {
            app: self.app.clone(),
            state: self.state.clone(),
            session_id: sid.clone(),
            mode,
            autonomous: true,
            allow_propose_pr: goal.allow_propose_pr,
            approval_timeout: GOAL_APPROVAL_TIMEOUT,
        };
        // Snapshot the matrix for this turn (#702); see the other turn paths. An
        // authorised goal (#1256) carries the same per-tool `propose_pr -> Allow`
        // override the approver applies, via the shared helper, kept local to
        // this loop.
        let permission_matrix =
            ff_agent::goal_matrix(self.state.permission_matrix(), goal.allow_propose_pr);
        let mut tool_ctx = ff_agent::ToolContext::new(
            &registry,
            &session_root,
            &approver,
            max_iterations,
            &permission_matrix,
        );
        tool_ctx.mode = mode;
        tool_ctx.egress = pheno.egress;
        tool_ctx.search_sources = pheno.search_sources.clone();
        tool_ctx.abstractive = crate::state::abstractive_config_from_env();
        tool_ctx.tool_search = Some(self.state.tool_search());

        let skills = self.state.skills_snapshot();
        let user_ctx =
            ff_agent::UserContext::now().with_working_dir(session_root.display().to_string());
        let active: Vec<String> = self.state.turn_active_skills(&sid);
        let mcp_guidance = self.state.mcp_guidance(&sid);
        let (inject_mem, extra_instructions) = self.state.turn_prompt_injection(&session_root);
        let (memory, _ambient_keys) = if inject_mem {
            self.state
                .memory()
                .ambient_block_filtered_keyed(self.state.index().as_ref())
        } else {
            (None, vec![])
        };
        let system_prompt_inputs = ff_agent::SystemPromptInputs {
            persona: persona.as_deref(),
            skills: &skills,
            active: &active,
            user: &user_ctx,
            memory: memory.as_deref(),
            extra_instructions: extra_instructions.as_deref(),
            goal: Some(goal),
            mode,
            mcp_guidance: &mcp_guidance,
        };

        provider.set_context_budget(self.state.served_window(&sid).await.window);

        let cancel = CancelToken::new();
        self.state.register_cancel(&sid, cancel.clone());
        let cancel_probe = cancel.clone();
        let thinking = self.state.provider_config().thinking;
        let reasoning_visibility = self.state.provider_config().reasoning_visibility;

        // Capture per-turn tokens (from `AgentEvent::Done`) directly in the event
        // closure, so the loop gets them without re-reading the store.
        let tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Goal-signal collection (`goal_complete` / `goal_step`) runs the same
        // state machine as the CLI host, shared via `TurnLedger` so the two cannot
        // drift (#1226). One `Mutex` replaces the four it used to scatter across
        // match arms.
        let ledger = Arc::new(std::sync::Mutex::new(ff_agent::TurnLedger::new()));
        let tokens_ev = tokens.clone();
        let ledger_ev = ledger.clone();
        let app_ev = self.app.clone();
        let sid_ev = sid.clone();
        let turn_start = std::time::Instant::now();

        let turn = ff_agent::run_session_turn(
            provider.as_ref(),
            self.state.store.as_ref(),
            &tool_ctx,
            &sid,
            &model,
            &system_prompt_inputs,
            thinking,
            reasoning_visibility,
            cancel.clone(),
            move |event| {
                match &event {
                    ff_agent::AgentEvent::Done {
                        token_count: Some(t),
                        ..
                    } => {
                        tokens_ev.store(*t as u64, std::sync::atomic::Ordering::SeqCst);
                    }
                    _ => ledger_ev.lock().unwrap().observe(&event),
                }
                emit_agent_event(&app_ev, &sid_ev, event);
            },
        );
        let result = tokio::time::timeout(GOAL_ITERATION_TIMEOUT, turn).await;
        self.state.take_cancel_if(&sid, &cancel);

        // #991: post-turn memory flush (off critical path), matching spawn_assistant_turn.
        if matches!(result, Ok(Ok(_))) {
            self.state
                .maybe_flush_memory(
                    provider.as_ref(),
                    &registry,
                    &sid,
                    &model,
                    cancel_probe.clone(),
                )
                .await;
        }

        // Distinguish a user Stop (cancel) — the goal should PAUSE resumably — from
        // an unrecoverable failure (timeout / provider error) — which FAILS it.
        // A timeout is not a user cancel: the user didn't stop it, so it fails.
        let user_cancelled = cancel_probe.is_cancelled();
        let (cancelled, failed) = match result {
            _ if user_cancelled => (true, false),
            Err(_elapsed) => {
                cancel.cancel();
                (false, true)
            }
            Ok(Err(_)) => (false, true),
            Ok(Ok(_)) => (false, false),
        };

        // Consume the shared ledger once, at the turn boundary. `mem::take` leaves
        // a fresh empty ledger behind so this does not depend on the event closure
        // (the other Arc holder) already being dropped.
        let ledger = std::mem::take(&mut *ledger.lock().unwrap());
        let goal_complete = ledger.completed();
        let proposed_pr = ledger.proposed_pr();
        IterationOutcome {
            tokens: tokens.load(std::sync::atomic::Ordering::SeqCst),
            wall_ms: turn_start.elapsed().as_millis() as i64,
            goal_complete,
            proposed_pr,
            cancelled,
            failed,
            steer_consumed,
            ledger_steps: ledger.into_steps(),
        }
    }

    fn save(&self, goal: &Goal) {
        if let Err(e) = self.state.goals.save(goal) {
            tracing::warn!(error = %e, session = %self.session_id, "failed to persist goal checkpoint");
        }
        let _ = self.app.emit("goal:updated", goal);
    }

    fn now_ms(&self) -> i64 {
        now_ms()
    }
}

/// Stop a running goal loop for a session and wait (bounded) for it to fully
/// exit before returning (#753 review blocker 2). Cancels the in-flight turn so
/// `drive_goal` breaks at its next boundary, then polls the single-flight slot
/// until it clears (or a short timeout). Callers that overwrite goal state
/// (`goal_set`) must await this first so the old loop's final checkpoint can't
/// race the new goal. Safe to call when no loop is running (returns promptly).
async fn stop_goal_loop(state: &Arc<AppState>, session_id: &str) {
    if !state.goal_loop_running(session_id) {
        return;
    }
    if let Some(token) = state.take_cancel(session_id) {
        token.cancel();
    }
    state.cancel_pending_approvals(session_id);
    // Bounded wait: the loop clears its slot on the next boundary after the
    // cancelled turn returns. Cap so a wedged provider can't hang goal_set.
    for _ in 0..600 {
        if !state.goal_loop_running(session_id) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    tracing::warn!(session = %session_id, "goal loop did not stop within timeout; proceeding");
}

/// Spawn the self-continue loop for a session's active goal (RFC 0020 §5). Loads
/// the goal and drives [`drive_goal`] to a terminal state on a background task.
/// Single-flight (#753 review): claims the session's goal-loop slot first and
/// refuses to spawn if a loop is already running, so `goal_set` / `goal_resume`
/// can never stack two loops racing the same transcript. The slot is released on
/// any terminal stop (including an early return / panic) via a drop guard.
fn spawn_goal_loop(state: Arc<AppState>, app: tauri::AppHandle, session_id: String) {
    if !state.try_start_goal_loop(&session_id) {
        tracing::debug!(session = %session_id, "goal loop already running; not spawning another");
        return;
    }
    tauri::async_runtime::spawn(async move {
        // Release the single-flight slot no matter how the task exits.
        struct LoopGuard {
            state: Arc<AppState>,
            session_id: String,
        }
        impl Drop for LoopGuard {
            fn drop(&mut self) {
                self.state.end_goal_loop(&self.session_id);
            }
        }
        let _guard = LoopGuard {
            state: state.clone(),
            session_id: session_id.clone(),
        };

        let mut goal = match state.goals.load(&session_id) {
            Ok(Some(g)) => g,
            Ok(None) => return,
            Err(e) => {
                tracing::warn!(error = %e, session = %session_id, "goal loop: cannot load goal");
                return;
            }
        };
        if goal.status != GoalStatus::Active {
            return;
        }
        let touch_log = ff_memory::TouchLog::default();
        let iter = GoalLoopIteration {
            state: state.clone(),
            app: app.clone(),
            session_id: session_id.clone(),
            touch_log: touch_log.clone(),
        };
        let stop = drive_goal(&mut goal, &iter).await;
        // Settle the memory the loop touched against its verdict (#1292): a
        // completed goal reinforces those Daily chunks, an exhausted/failed one
        // suppresses their promotion, a paused one leaves them untouched. Mirrors
        // the CLI goal loop so both surfaces reinforce identically.
        let touched = touch_log.drain();
        if !touched.is_empty() {
            let index = state.index();
            let sink = ff_memory::MemoryOutcomeSink::new(index.as_ref());
            if let Err(e) = sink.settle(stop.verdict(), &touched) {
                tracing::warn!(error = %e, session = %session_id, "outcome-gated reinforcement failed");
            }
        }
        tracing::info!(session = %session_id, ?stop, "goal loop finished");
    });
}

/// bounded single `run_turn`; if it overruns it is cancelled and recorded
/// `cancelled`, so one hung fire cannot starve later due tasks.
const SCHEDULED_FIRE_TIMEOUT: Duration = Duration::from_secs(600);

/// Fires a due scheduled task as a headless agent turn (RFC 0017 §4). Mirrors
/// `spawn_assistant_turn`: create a session, bind the task's workspace + profile,
/// resolve the provider + phenotype, run one bounded `run_turn` under a
/// `ScheduledApprover` at the task's safety ceiling, and map the result to a
/// terminal `RunStatus`. Lives here (not in `ff-scheduled`) so that crate stays
/// Tauri-free; this is the host-supplied `TaskRunner`.
struct DesktopTaskRunner {
    state: Arc<AppState>,
    app: tauri::AppHandle,
}

impl DesktopTaskRunner {
    /// Run a built-in action directly, mapping its result to a terminal
    /// `RunStatus` (#544). A built-in has no free-text prompt and no agent loop,
    /// so it never creates a session (`session_id: None`).
    async fn fire_builtin(&self, action: ff_core::BuiltinAction) -> RunStatus {
        match action {
            ff_core::BuiltinAction::MemoryConsolidate => {
                let memory = self.state.memory();
                if !memory.is_enabled() {
                    // A disabled memory store has nothing to organize; the fire
                    // succeeded as a no-op rather than failing.
                    return RunStatus::Ok;
                }
                let index = self.state.index();
                // Both `consolidate` (file rewrite) and `reindex` (a possible
                // blocking embed HTTP call) are sync/blocking, so run them off the
                // async worker — mirrors `MemoryConsolidateTool::run`.
                let result = tokio::task::spawn_blocking(move || {
                    let salience = memory.chunk_stats_salience(index.as_ref(), now_ms());
                    let report = memory.consolidate(&salience)?;
                    if report.ran {
                        // Best-effort reindex; a recall-cache failure must not fail
                        // the consolidation pass itself.
                        let _ = index.reindex(&memory.all_chunks());
                        // Record supersession edges after reindex (#1293).
                        for (from, to) in &report.superseded {
                            let _ = index.record_link(from, to, "supersession");
                        }
                    }
                    ff_memory::Result::Ok(())
                })
                .await;
                match result {
                    Ok(Ok(())) => RunStatus::Ok,
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "scheduled builtin: memory_consolidate failed");
                        RunStatus::Error
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "scheduled builtin: consolidate task panicked");
                        RunStatus::Error
                    }
                }
            }
        }
    }
}

#[async_trait]
impl ff_scheduled::TaskRunner for DesktopTaskRunner {
    async fn fire(&self, task: &ScheduledTask) -> ff_scheduled::RunOutcome {
        // A built-in runs it's named action directly (no agent loop / session);
        // a prompt task drives a headless `run_turn` below (#544).
        let prompt = match &task.kind {
            TaskKind::Prompt(text) => text.clone(),
            TaskKind::Builtin(action) => {
                let status = self.fire_builtin(*action).await;
                return ff_scheduled::RunOutcome {
                    session_id: None,
                    status,
                };
            }
        };

        // Create the fire's session and bind the task's workspace + profile so the
        // turn runs in the intended checkout under the intended persona. A stale
        // profile name fails the fire (`error`), surfaced on the run record.
        let session = self.state.store.create_session(Some(task.name.clone()));
        let sid = session.id;
        if let Some(ws) = &task.workspace {
            self.state
                .store
                .set_session_workspace(&sid, Some(ws.clone()));
        }
        if let Some(profile) = &task.profile {
            if let Err(e) = self
                .state
                .set_session_phenotype(&sid, Some(profile.clone()))
            {
                tracing::warn!(task = %task.id, error = %e, "scheduled fire: unresolved profile");
                return ff_scheduled::RunOutcome {
                    session_id: Some(sid),
                    status: RunStatus::Error,
                };
            }
        }
        self.state.store.add_message(&sid, Role::User, prompt);

        // The safety ceiling maps to the run mode: a read-only task runs in Plan
        // (read-capable tools advertised; the headless approver auto-runs ReadOnly
        // and denies Write/Sensitive/Dangerous), a write task in Act (write tools
        // advertised; the approver allows Write and denies Dangerous).
        let mode = match task.safety_ceiling {
            ff_core::SafetyCeiling::ReadOnly => Mode::Plan,
            ff_core::SafetyCeiling::Write => Mode::Act,
        };

        let pheno = self.state.session_phenotype(&sid);
        let selection = self.state.resolve_model_selection(&sid);
        let (mut provider, _) = self
            .state
            .build_provider_for(Some(&selection.connection), Some(&selection.model));
        let model = selection.model;
        let max_iterations = pheno
            .max_iterations
            .unwrap_or(ff_agent::DEFAULT_MAX_ITERATIONS);

        let session_root = self.state.session_root(&sid);
        self.state.align_session_mcp(&sid, &session_root).await;
        // Keep the git HEAD watcher aimed at the active checkout here too (#561), so
        // a scheduled-task turn that switches branches live-updates the FE chip.
        self.state.align_git_watcher(&session_root);
        let registry = self.state.build_tool_registry(&session_root);
        // #1179 3B; see the interactive path. A scheduled turn has no UI to toast,
        // so the drop notice is warn-only there.
        if let Some(dropped) = self.state.align_session_preheat(&sid, &registry) {
            tracing::warn!(
                session_id = %dropped.session_id,
                phenotype = %dropped.phenotype,
                unknown = ?dropped.unknown,
                over_budget = ?dropped.over_budget,
                "phenotype preheat entries dropped"
            );
        }
        let approver = ScheduledApprover::new(task.safety_ceiling);
        // Snapshot the matrix for this turn (#702); see the interactive path above.
        let permission_matrix = self.state.permission_matrix();
        let mut tool_ctx = ToolContext::new(
            &registry,
            &session_root,
            &approver,
            max_iterations,
            &permission_matrix,
        );
        tool_ctx.mode = mode;
        tool_ctx.egress = pheno.egress;
        tool_ctx.search_sources = pheno.search_sources.clone();
        tool_ctx.abstractive = crate::state::abstractive_config_from_env();
        tool_ctx.compaction_model = self.state.compaction_model(&selection.connection);
        tool_ctx.compaction_budget = self.state.compaction_budget(&selection.connection);
        tool_ctx.near_budget_tokens = self.state.near_budget(&selection.connection);
        tool_ctx.compaction_cache = Some(&self.state.compaction_cache);
        tool_ctx.tool_search = Some(self.state.tool_search());

        let skills = self.state.skills_snapshot();
        let user_ctx =
            ff_agent::UserContext::now().with_working_dir(session_root.display().to_string());
        let active: Vec<String> = self.state.turn_active_skills(&sid);
        let mcp_guidance = self.state.mcp_guidance(&sid);
        let (inject_mem, extra_instructions) = self.state.turn_prompt_injection(&session_root);
        let (memory, _ambient_keys) = if inject_mem {
            self.state
                .memory()
                .ambient_block_filtered_keyed(self.state.index().as_ref())
        } else {
            (None, vec![])
        };
        let system_prompt_inputs = ff_agent::SystemPromptInputs {
            persona: pheno.persona.as_deref(),
            skills: &skills,
            active: &active,
            user: &user_ctx,
            memory: memory.as_deref(),
            extra_instructions: extra_instructions.as_deref(),
            goal: None,
            mode,
            mcp_guidance: &mcp_guidance,
        };

        // Prime the compaction budget against the served context window (#612),
        // mirroring the interactive `send_message` path; Ollama uses the cached
        // `/api/ps` probe, other providers no-op.
        provider.set_context_budget(self.state.served_window(&sid).await.window);

        let cancel = CancelToken::new();
        let thinking = self.state.provider_config().thinking;
        let reasoning_visibility = self.state.provider_config().reasoning_visibility;
        let app = self.app.clone();
        let sid_for_events = sid.clone();
        let turn = ff_agent::run_session_turn(
            provider.as_ref(),
            self.state.store.as_ref(),
            &tool_ctx,
            &sid,
            &model,
            &system_prompt_inputs,
            thinking,
            reasoning_visibility,
            cancel.clone(),
            move |event| emit_agent_event(&app, &sid_for_events, event),
        );
        let result = tokio::time::timeout(SCHEDULED_FIRE_TIMEOUT, turn).await;

        // #991: post-turn memory flush (off critical path), matching spawn_assistant_turn.
        if matches!(&result, Ok(Ok(_))) {
            self.state
                .maybe_flush_memory(provider.as_ref(), &registry, &sid, &model, cancel.clone())
                .await;
        }

        // Outcome precedence (RFC 0017 §8.4): an ask_user dismissal surfaces as
        // needs_attention regardless of how the rest of the turn ended; else a
        // timeout is a cancellation; else a run_turn error; else ok (a denied
        // write within an otherwise-complete run is still ok).
        let status = if approver.needs_attention() {
            RunStatus::NeedsAttention
        } else {
            match result {
                Err(_elapsed) => {
                    cancel.cancel();
                    RunStatus::Cancelled
                }
                Ok(Err(_)) => RunStatus::Error,
                Ok(Ok(_)) => RunStatus::Ok,
            }
        };

        ff_scheduled::RunOutcome {
            session_id: Some(sid),
            status,
        }
    }
}

/// Edit a prior user message in place, truncate the transcript after it, and
/// re-run the turn from the edited prompt (#464, backend for #463). Cancels any
/// in-flight turn + pending approvals for the session first so a running turn
/// cannot append after the truncation, validates the edit in the store (rejecting
/// an unknown id, a wrong-session id, or a non-user message), then spawns a fresh
/// assistant turn over the existing `turn:*` / `tool:*` events. Returns the edited
/// message's id.
#[tauri::command]
fn edit_message(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
    message_id: String,
    content: String,
    attachments: Option<Vec<Attachment>>,
) -> CmdResult<String> {
    // Stop any turn already streaming into this session before we mutate its
    // transcript (mirrors `cancel_turn`): otherwise the running turn would keep
    // appending messages past the cut we are about to make.
    if let Some(token) = state.take_cancel(&session_id) {
        token.cancel();
    }
    state.cancel_pending_approvals(&session_id);
    // Invalidate the cross-turn summary cache (#757): the transcript is being
    // truncated, so the old boundary is no longer valid.
    state.compaction_cache.invalidate(&session_id);

    let edited_id = state
        .store
        .edit_user_message(&session_id, &message_id, content, attachments)
        .map_err(|e| e.to_string())?;

    spawn_assistant_turn(state.inner().clone(), app, session_id, 0);

    Ok(edited_id)
}

/// Current LLM provider settings for the settings panel.
#[tauri::command]
fn get_provider_config(state: State<'_, Arc<AppState>>) -> ProviderConfig {
    state.provider_config()
}

/// Persist new provider settings. Returns the stored config so the UI can confirm
/// the applied state (e.g. `has_key`, which the frontend never sets itself).
#[tauri::command]
fn set_provider_config(
    state: State<'_, Arc<AppState>>,
    kind: ProviderKind,
    base_url: Option<String>,
    model: String,
    thinking: bool,
) -> ProviderConfig {
    let current = state.provider_config();
    let config = ProviderConfig {
        kind,
        // Treat an empty string from the UI the same as "use the default endpoint".
        base_url: base_url.filter(|u| !u.trim().is_empty()),
        model,
        // Secrets are a later phase; preserve whatever the backend already knows.
        has_key: current.has_key,
        thinking,
        // This legacy shim has no effort control; preserve the persisted dial.
        reasoning_effort: current.reasoning_effort,
        reasoning_visibility: current.reasoning_visibility,
        // No warmup control on this shim either; preserve the persisted value.
        warmup_enabled: current.warmup_enabled,
        // No num_ctx control on this shim; preserve the persisted window (#651).
        num_ctx: current.num_ctx,
    };
    state.set_provider_config(config.clone());
    // Model/provider changed — invalidate all cached summaries (#757) since
    // a summary generated by the old model may be incoherent for the new one.
    state.compaction_cache.invalidate_all();
    config
}

/// The full provider connection registry for the settings panel (RFC 0005 Phase A).
#[tauri::command]
fn get_provider_registry(state: State<'_, Arc<AppState>>) -> ProviderRegistry {
    state.provider_registry()
}

/// Select the active connection by id. Errors on an unknown id.
#[tauri::command]
fn set_active_connection(state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    state.set_active_connection(&id)
}

/// Add or update a connection (keyed by `id`, derived from vendor/name when blank).
/// Returns the stored connection so the UI sees the resolved id.
#[tauri::command]
fn upsert_connection(
    state: State<'_, Arc<AppState>>,
    conn: ProviderConnection,
) -> CmdResult<ProviderConnection> {
    let result = state.upsert_connection(conn);
    // Connection model/endpoint may have changed — invalidate summaries (#757).
    state.compaction_cache.invalidate_all();
    Ok(result)
}

/// Remove a connection by id. Errors when removing the last one.
#[tauri::command]
fn remove_connection(state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    state.remove_connection(&id)
}

/// Store a provider secret for a connection in the OS keychain and flip its hasKey
/// flag. The secret value is never returned to the frontend.
#[tauri::command]
fn set_provider_secret(
    state: State<'_, Arc<AppState>>,
    connection_id: String,
    kind: SecretKind,
    value: String,
) -> CmdResult<()> {
    state.set_connection_secret(&connection_id, kind, &value)
}

/// Clear a provider secret for a connection and recompute its hasKey flag from the
/// remaining stored secrets.
#[tauri::command]
fn clear_provider_secret(
    state: State<'_, Arc<AppState>>,
    connection_id: String,
    kind: SecretKind,
) -> CmdResult<()> {
    state.clear_connection_secret(&connection_id, kind)
}

/// Store an API key for a search backend in the OS keychain (#1010). The value is
/// never returned to the frontend; `getSearchConfig`'s `hasKey` reflects presence.
#[tauri::command]
fn set_search_secret(
    state: State<'_, Arc<AppState>>,
    backend: ff_core::SearchBackend,
    value: String,
) -> CmdResult<()> {
    state.set_search_secret(backend, &value)
}

/// Clear a search backend's stored API key (#1010).
#[tauri::command]
fn clear_search_secret(
    state: State<'_, Arc<AppState>>,
    backend: ff_core::SearchBackend,
) -> CmdResult<()> {
    state.clear_search_secret(backend)
}

/// Per-backend key presence for the Settings Search key panel (#1015). Boolean
/// only — no secret value crosses the wire.
#[tauri::command]
fn search_secret_presence(state: State<'_, Arc<AppState>>) -> Vec<ff_core::SearchSecretPresence> {
    state.search_secret_presence()
}

/// Which secret kinds are stored for a connection (#320), so each Bedrock secret
/// field shows its own Stored/Clear state. Presence only — no value is returned.
#[tauri::command]
fn provider_secret_presence(
    state: State<'_, Arc<AppState>>,
    connection_id: String,
) -> CmdResult<Vec<SecretKind>> {
    state.connection_secret_presence(&connection_id)
}

/// The Bedrock auth a connection resolves to right now (#320) — the explicit mode,
/// or the `Auto` precedence winner (API key > profile > IAM keys). `None` for
/// non-Bedrock or unknown connections; lets the UI badge the active credential.
#[tauri::command]
fn resolved_bedrock_auth(
    state: State<'_, Arc<AppState>>,
    connection_id: String,
) -> Option<BedrockAuth> {
    state.resolved_bedrock_auth(&connection_id)
}

/// The Control-panel settings blob (#147). Opaque JSON round-tripped verbatim;
/// the frontend (`lib/control.ts`) owns the shape. Returns the factory default on
/// first load.
#[tauri::command]
fn get_control_config(state: State<'_, Arc<AppState>>) -> serde_json::Value {
    state.control_config()
}

/// Persist the Control-panel settings blob and echo back the stored value so the
/// UI can confirm the applied state (mirrors `set_search_config`).
#[tauri::command]
fn set_control_config(
    state: State<'_, Arc<AppState>>,
    config: serde_json::Value,
) -> serde_json::Value {
    state.set_control_config(config)
}

/// Current persisted web-search settings.
#[tauri::command]
fn get_search_config(state: State<'_, Arc<AppState>>) -> SearchConfig {
    state.search_config()
}

/// Persist new web-search settings. Returns the stored config so the UI can confirm
/// the applied state (e.g. `hasKey`, which the frontend never sets itself).
#[tauri::command]
fn set_search_config(
    state: State<'_, Arc<AppState>>,
    backend: ff_core::SearchBackend,
    base_url: Option<String>,
    email: Option<String>,
) -> SearchConfig {
    let current = state.search_config();
    let config = SearchConfig {
        backend,
        // Treat an empty string from the UI the same as "no endpoint configured".
        base_url: base_url.filter(|u| !u.trim().is_empty()),
        // User email for NCBI requests (#1021). Treat empty string as "not set".
        email: email.filter(|e| !e.trim().is_empty()),
        // Secrets are a later phase; preserve whatever the backend already knows.
        has_key: current.has_key,
    };
    state.set_search_config(config.clone());
    config
}

/// Best-effort model list for a connection's endpoint (`id` defaults to the active
/// connection). Returns an empty list (never an error) when the server is
/// unreachable so the picker degrades to free-text entry.
#[tauri::command]
async fn list_models(
    state: State<'_, Arc<AppState>>,
    id: Option<String>,
) -> CmdResult<Vec<String>> {
    let (provider, _model) = state.build_provider_for(id.as_deref(), None);
    Ok(provider.list_models().await.unwrap_or_default())
}

/// Probe a connection for the settings "Test Connection" button. `id` defaults to
/// the active connection. Returns `Ok(())` on a successful round-trip, or an
/// `Err(String)` message the UI can show. Unlike `list_models`, the error is
/// surfaced (the button reports failure) rather than swallowed.
#[tauri::command]
async fn test_connection(state: State<'_, Arc<AppState>>, id: Option<String>) -> CmdResult<()> {
    let (provider, model) = state.build_provider_for(id.as_deref(), None);
    provider
        .test_connection(&model)
        .await
        .map_err(|e| e.to_string())
}

/// Wake the configured model server so its GPU/compute pipelines are hot before
/// the user's first message. The composer fires this (debounced) when it gains
/// focus. Fully best-effort: any failure (server down, busy) is swallowed so
/// warmup never blocks the UI or surfaces an error.
///
/// Gated to local kinds with warmup enabled (#61): warming a *hosted* endpoint
/// would fire a billed request on every composer focus, and a user who disabled
/// warmup (e.g. on laptop battery) must not pay for it. The frontend also gates
/// the nudge, so this is defense in depth.
#[tauri::command]
async fn warmup(state: State<'_, Arc<AppState>>) -> CmdResult<()> {
    let cfg = state.provider_config();
    if !should_warmup(cfg.kind, cfg.warmup_enabled) {
        return Ok(());
    }
    let (provider, model) = state.build_provider();
    let _ = provider.warmup(&model).await;
    Ok(())
}

/// Whether the composer warmup nudge should fire for a connection: only for
/// local backends (#61) with warmup enabled. Pure so it is unit-testable.
fn should_warmup(kind: ProviderKind, warmup_enabled: bool) -> bool {
    kind.is_local() && warmup_enabled
}

/// Install a skill from a path / git URL / raw-Markdown URL. The bundle is fetched
/// and validated first (a bad bundle returns an error and never prompts), then the
/// user approves the real declared manifest via the shared approval gate, and only
/// on approval is it moved into `~/.flowforge/skills/`. Returns the installed
/// manifest. Errors (validation, denial, IO) come back as `Err(String)`.
#[tauri::command]
async fn install_skill(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    source: String,
) -> CmdResult<SkillManifest> {
    // Fetch + validate off the async runtime; a bad bundle fails here, pre-approval.
    let prep_source = source.clone();
    let staged = tokio::task::spawn_blocking(move || ff_skills::prepare_install(&prep_source))
        .await
        .map_err(|e| format!("install task failed: {e}"))?
        .map_err(|e| e.to_string())?;

    let manifest = staged.manifest().clone();
    let request_id = Uuid::new_v4().to_string();

    // Present the validated manifest for approval, reusing the dangerous-tool gate.
    // An install is a live, standalone approval (no turn), so register a liveness
    // token under `request_id` to satisfy `register_approval`'s TOCTOU guard, and
    // key the prompt (request_id, request_id). The token is released after the
    // await — and lets a shutdown/`cancel_pending_approvals` deny a stuck install.
    state.register_cancel(&request_id, CancelToken::new());
    let rx = state.register_approval(&request_id, &request_id);
    let _ = app.emit(
        "skill:install-approval-request",
        SkillInstallApprovalRequestEvent {
            request_id: request_id.clone(),
            source,
            manifest: manifest.clone(),
            warnings: Vec::new(),
        },
    );
    let approved = rx.await.unwrap_or(false);
    state.take_cancel(&request_id);
    if !approved {
        return Err("install was not approved".to_string());
    }

    let skills_root = state.skills_root();
    tokio::task::spawn_blocking(move || ff_skills::commit_install(staged, &skills_root))
        .await
        .map_err(|e| format!("install task failed: {e}"))?
        .map_err(|e| e.to_string())?;

    state.reload_skills();
    emit_skills_changed(&app, &state);
    Ok(manifest)
}

/// Build a frontend [`SkillInfo`] from a loaded skill, folding in active state and
/// a search score (`0` for the unranked list).
fn to_skill_info(skill: &Skill, active: &[String], score: u32) -> SkillInfo {
    SkillInfo {
        name: skill.manifest.name.clone(),
        description: skill.manifest.description.clone(),
        version: skill.manifest.version.clone(),
        keywords: skill.manifest.keywords.clone(),
        active: active.contains(&skill.manifest.name),
        score,
    }
}

/// Emit the current active set so the frontend can reconcile its state.
fn emit_skills_changed(app: &tauri::AppHandle, state: &AppState) {
    let _ = app.emit(
        "skills:changed",
        SkillsChangedEvent {
            active: state.active_skills(),
        },
    );
}

/// Emit `phenotype:mcp-unavailable` when `phenotype`'s skills declare an MCP server
/// that is absent or not running (#301), so the frontend can surface a non-blocking
/// toast over the warn-only backend signal. Fires only when the list is non-empty,
/// matching the frontend listener contract; best-effort (a dropped emit is harmless).
fn emit_pheno_mcp_unavailable(app: &tauri::AppHandle, state: &AppState, phenotype: &str) {
    let servers = state.unavailable_skill_mcp_servers(phenotype);
    if !servers.is_empty() {
        let _ = app.emit(
            "phenotype:mcp-unavailable",
            PhenotypeMcpUnavailableEvent {
                phenotype: phenotype.to_string(),
                servers,
            },
        );
    }
}

/// All installed skills, name-sorted, each flagged with its active state. `score`
/// is always `0` here — ranking is `search_skills`' job. Backs the command palette
/// skill source (#11/#16).
#[tauri::command]
fn list_skills(state: State<'_, Arc<AppState>>) -> Vec<SkillInfo> {
    let active = state.active_skills();
    let reg = state.skills_snapshot();
    let mut skills: Vec<&Skill> = reg.list().collect();
    skills.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    skills
        .into_iter()
        .map(|s| to_skill_info(s, &active, 0))
        .collect()
}

/// Rank installed skills for `query`, sharing `ff_skills::search_skills` with the
/// agent tool of the same name. An empty query lists all skills (name-sorted).
#[tauri::command]
fn search_skills(state: State<'_, Arc<AppState>>, query: String) -> Vec<SkillInfo> {
    let active = state.active_skills();
    let reg = state.skills_snapshot();
    ff_skills::search_skills(&reg, &query)
        .into_iter()
        .map(|hit| SkillInfo {
            name: hit.manifest.name.clone(),
            description: hit.manifest.description.clone(),
            version: hit.manifest.version.clone(),
            keywords: hit.manifest.keywords.clone(),
            active: active.contains(&hit.manifest.name),
            score: hit.score,
        })
        .collect()
}

/// Per-skill telemetry aggregate (RFC 0001 §8): activation/completion counts, mean
/// token cost, mean turns, mean latency, and success rate. `None` when the skill has
/// no recorded signals yet. Backs the optimize flow's before/after cost estimate.
#[tauri::command]
fn get_skill_telemetry(state: State<'_, Arc<AppState>>, skill: String) -> Option<SkillAggregate> {
    state.skill_telemetry(&skill)
}

/// A compact recent-transcript sample for an optimize prompt: the last `n` messages
/// of the session, each as `role: content` with the body truncated. Current-session
/// only for M3 (durable cross-session transcripts are deferred to M5).
fn recent_transcript(state: &AppState, session_id: &str, n: usize) -> Vec<String> {
    // #1142 P1: use tail query so long sessions don't pay for full-history fetch.
    let messages = state.store.get_messages_tail(session_id, n);
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };
            let mut body = m.content.replace('\n', " ");
            if body.chars().count() > 200 {
                body = body.chars().take(200).collect::<String>() + "…";
            }
            format!("{role}: {body}")
        })
        .collect()
}

/// Manual skill optimize/evolve (M3.5, RFC 0001 §8). Gathers the skill body, its
/// telemetry aggregate, and a recent-transcript sample; asks the model for a
/// streamlined rewrite; then presents a before->after proposal with a cost estimate
/// for approval. On approval the skill is version-bumped (previous version retained
/// for rollback); on rejection nothing changes. Reuses the standalone-approval gate
/// (keyed by `request_id`, answered via `respond_approval`), exactly like install.
#[tauri::command]
async fn optimize_skill(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
    skill: String,
) -> CmdResult<String> {
    // Snapshot the current body + version (reject an unknown skill before any model
    // call), plus telemetry and a short transcript sample for context.
    let (before_body, current_version) = {
        let reg = state.skills_snapshot();
        let s = reg
            .get(&skill)
            .ok_or_else(|| format!("unknown skill: {skill}"))?;
        (s.body.clone(), s.manifest.version.clone())
    };
    let aggregate = state.skill_telemetry(&skill);
    let transcript = recent_transcript(&state, &session_id, 6);

    let (provider, default_model) = state.build_provider();
    let model = state.active_model_override().unwrap_or(default_model);
    let after_body = optimize::propose_rewrite(
        provider.as_ref(),
        &model,
        &skill,
        &before_body,
        aggregate.as_ref(),
        &transcript,
    )
    .await?;

    let new_version = ff_skills::bump_patch(&current_version);
    let (current_mean_tokens, estimated_mean_tokens) =
        optimize::estimate_cost(aggregate.as_ref(), &before_body, &after_body);

    // Standalone approval, same pattern as install_skill.
    let request_id = Uuid::new_v4().to_string();
    state.register_cancel(&request_id, CancelToken::new());
    let rx = state.register_approval(&request_id, &request_id);
    let _ = app.emit(
        "skill:evolve-approval-request",
        SkillEvolveApprovalRequestEvent {
            request_id: request_id.clone(),
            skill: skill.clone(),
            current_version,
            new_version,
            before_body,
            after_body: after_body.clone(),
            cost_estimate: EvolveCostEstimate {
                current_mean_tokens,
                estimated_mean_tokens,
            },
        },
    );
    let approved = rx.await.unwrap_or(false);
    state.take_cancel(&request_id);
    if !approved {
        return Err("optimize was not approved".to_string());
    }

    let skills_root = state.skills_root();
    let history_root = state.skill_history_root();
    let skill_name = skill.clone();
    let applied = tokio::task::spawn_blocking(move || {
        ff_skills::bump_skill(&skills_root, &history_root, &skill_name, &after_body)
    })
    .await
    .map_err(|e| format!("optimize task failed: {e}"))?
    .map_err(|e| e.to_string())?;

    state.reload_skills();
    emit_skills_changed(&app, &state);
    Ok(applied)
}

/// Restore a retained previous version of a skill as the live version (RFC 0001 §8).
/// The current live version is archived first, so a rollback is itself reversible.
/// Emits `skills:changed`.
#[tauri::command]
async fn rollback_skill(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    skill: String,
    version: String,
) -> CmdResult<()> {
    let skills_root = state.skills_root();
    let history_root = state.skill_history_root();
    tokio::task::spawn_blocking(move || {
        ff_skills::rollback_skill(&skills_root, &history_root, &skill, &version)
    })
    .await
    .map_err(|e| format!("rollback task failed: {e}"))?
    .map_err(|e| e.to_string())?;
    state.reload_skills();
    emit_skills_changed(&app, &state);
    Ok(())
}

/// Retained version names for a skill, newest-looking first. Empty when the skill
/// has never been optimized (RFC 0001 §8). Backs the rollback picker.
#[tauri::command]
fn list_skill_versions(state: State<'_, Arc<AppState>>, skill: String) -> CmdResult<Vec<String>> {
    ff_skills::list_skill_versions(&state.skill_history_root(), &skill).map_err(|e| e.to_string())
}

/// Add a skill to the global active set (its body is injected next turn). Errors on
/// an unknown name. Emits `skills:changed`.
#[tauri::command]
fn activate_skill(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    name: String,
) -> CmdResult<()> {
    state.activate_skill(&name)?;
    emit_skills_changed(&app, &state);
    Ok(())
}

/// Remove a skill from the active set (idempotent). Emits `skills:changed`.
#[tauri::command]
fn deactivate_skill(state: State<'_, Arc<AppState>>, app: tauri::AppHandle, name: String) {
    state.deactivate_skill(&name);
    emit_skills_changed(&app, &state);
}

/// Uninstall a skill by manifest name. Local and reversible, so it is not gated.
#[tauri::command]
fn uninstall_skill(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    name: String,
) -> CmdResult<()> {
    ff_skills::uninstall(&name, &state.skills_root()).map_err(|e| e.to_string())?;
    state.reload_skills();
    // An uninstalled skill must not linger in the active set.
    state.prune_active_skills();
    emit_skills_changed(&app, &state);
    Ok(())
}

/// All selectable phenotypes (built-in `default` + `~/.flowforge/phenos/`),
/// name-sorted. Backs the `pheno` command palette.
#[tauri::command]
fn list_phenotypes(state: State<'_, Arc<AppState>>) -> Vec<Phenotype> {
    state.list_phenotypes()
}

/// The active phenotype.
#[tauri::command]
fn get_phenotype(state: State<'_, Arc<AppState>>) -> Phenotype {
    state.active_phenotype()
}

/// Switch the active phenotype: replaces the active-skill set with the phenotype's
/// skills and persists the choice across restarts (RFC 0001 §7). Errors on an unknown
/// name. Emits `skills:changed` so the FE active set updates. Returns the phenotype now
/// active.
#[tauri::command]
fn switch_phenotype(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    name: String,
) -> CmdResult<Phenotype> {
    let pheno = state.switch_phenotype(&name)?;
    emit_skills_changed(&app, &state);
    emit_pheno_mcp_unavailable(&app, &state, &pheno.name);
    Ok(pheno)
}

/// Persist an edited phenotype to disk (RFC 0005 Phase D / #525). Accepts the whole
/// `Phenotype` (lossless read-modify-write). The built-in `default` is immutable and
/// rejected. When the edited phenotype is the one currently active, its skills and
/// overrides are re-applied immediately and `skills:changed` is emitted so the FE
/// active set updates. Returns the saved phenotype.
#[tauri::command]
fn update_phenotype(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    phenotype: Phenotype,
) -> CmdResult<Phenotype> {
    let pheno = state.update_phenotype(phenotype)?;
    if pheno.name == state.active_phenotype().name {
        emit_skills_changed(&app, &state);
        emit_pheno_mcp_unavailable(&app, &state, &pheno.name);
    }
    Ok(pheno)
}

/// Bind a single session to a phenotype, or clear the binding (`name: None`) so it
/// inherits the global active one (#246). Unlike `switch_phenotype`, this changes
/// only the named session's Pheno — other panes are untouched — and persists
/// nothing globally. Errors on an unknown phenotype name. The FE pane selector
/// (#245) calls this to make a pane's Pheno truly per-session.
#[tauri::command]
fn set_session_phenotype(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    session_id: String,
    name: Option<String>,
) -> CmdResult<()> {
    state.set_session_phenotype(&session_id, name.clone())?;
    if let Some(n) = name {
        emit_pheno_mcp_unavailable(&app, &state, &n);
    }
    Ok(())
}

/// Bind a single session's autonomy mode, or clear it (`mode: None`) so it inherits
/// the global default (#265). Per-pane, like `set_session_phenotype`. When the
/// resolved mode actually changes, a System-role message is injected into the
/// transcript so the model sees an in-context signal (#828) — breaking behavioral
/// inertia from prior assistant turns that referenced the old mode.
#[tauri::command]
fn set_session_mode(state: State<'_, Arc<AppState>>, session_id: String, mode: Option<Mode>) {
    let old = state.session_mode(&session_id);
    state.set_session_mode(&session_id, mode);
    let new = state.session_mode(&session_id);
    if old != new {
        let label = match new {
            Mode::Plan => "Plan. Read-only tools only; writes are denied.",
            Mode::Auto => "Auto. Writes auto-approved; sensitive actions prompt.",
            Mode::Act => "Act. Full tool access enabled.",
        };
        // Use Role::User so the message stays IN-POSITION in the conversation
        // flow. Bedrock's `to_converse` lifts all Role::System messages out of
        // sequence into a flat system parameter, losing the temporal signal that the
        // mode changed between the assistant's prior response and the next user
        // message (#848). A user-role marker stays in the conversation and breaks
        // the model's self-consistency anchoring to its own prior "I'm in Plan" text.
        let marker = format!(
            "{} Mode switched to {label}]",
            ff_core::MODE_SWITCH_MARKER_PREFIX
        );
        // But if a turn is IN FLIGHT, appending now would land the marker between
        // an assistant `tool_use` and its not-yet-persisted `tool_result`,
        // wedging the session with an Anthropic 422 (#1066). Defer it: it is
        // drained and persisted once the turn settles, landing after the tool
        // batch. The read-side self-heal (#1067) is the safety net for any bad
        // order that still reaches the wire.
        if state.has_active_turn(&session_id) {
            state.defer_mode_marker(&session_id, marker);
        } else {
            state.store.add_message(&session_id, Role::User, marker);
        }
    }
}

/// Pin a single session's model, or clear it (`selection: None`) so it inherits its
/// phenotype's model again (RFC 0005 §11 Phase D, #499). Per-pane, like
/// `set_session_phenotype`. Errors on an unknown connection id so a stale UI cannot
/// pin a phantom endpoint. The FE per-pane model chip (#523) calls this.
#[tauri::command]
fn set_session_model_selection(
    state: State<'_, Arc<AppState>>,
    session_id: String,
    selection: Option<ModelSelection>,
) -> CmdResult<()> {
    state.set_session_model(&session_id, selection)?;
    Ok(())
}

/// A session's explicit per-pane model selection, or `None` if it inherits its
/// phenotype's model (#499). The raw binding, *not* the resolved selection -- use
/// `resolve_model_selection` for what a turn actually runs on.
#[tauri::command]
fn get_session_model_selection(
    state: State<'_, Arc<AppState>>,
    session_id: String,
) -> Option<ModelSelection> {
    state.session_model(&session_id)
}

/// The resolved `(connection, model)` a turn for this session runs on, applying the
/// RFC 0005 §11.2 three-tier precedence session > phenotype > global (#499). The FE
/// renders this on the per-pane model chip so the displayed model matches what the
/// next turn will actually use.
#[tauri::command]
async fn resolve_model_selection(
    state: State<'_, Arc<AppState>>,
    session_id: String,
) -> Result<ResolvedModel, String> {
    let mut resolved = state.resolve_model_selection(&session_id);
    // Fold the async-only served-window probe onto the sync resolver's output
    // (#602). Cast u64 -> u32 so the ts-rs binding stays `number` (u64 maps to
    // `bigint`, which the FE `ServedWindow.window: number` cannot consume); a
    // window above `u32::MAX` (~4.29B tokens) is absurd, so try_from is just a
    // safety net rather than a real constraint. The `Result` wrapper is required
    // for async tauri commands with reference inputs; we never actually err.
    let probe = state.served_window(&session_id).await;
    // #1023: the probe only yields a window for Ollama; for every other provider it
    // returns None. Preserve the sync resolver's model-spec fallback in that case
    // (`.or(...)`) instead of clobbering it back to null — otherwise the FE context
    // gauge has no real denominator and falls back to the misleading soft-budget
    // estimate (the phantom pctUsed>100% on Bedrock/Anthropic/global).
    resolved.context_window = probe
        .window
        .and_then(|n| u32::try_from(n).ok())
        .or(resolved.context_window);
    resolved.trained_context_window = probe
        .trained
        .and_then(|n| u32::try_from(n).ok())
        .or(resolved.trained_context_window);
    resolved.context_window_source = probe.source.or(resolved.context_window_source);
    // Ollama reports vision capability via `/api/show`; OR it onto the name-based
    // gate so a daemon-reported multimodal model (e.g. a qwen3-vl MoE) isn't
    // blocked by a stale name allow-list (#625). `None`/`Some(false)` leave the
    // name-based result untouched.
    resolved.supports_vision = resolved.supports_vision || probe.supports_vision.unwrap_or(false);
    Ok(resolved)
}

/// The global default autonomy mode (#265), inherited by sessions with no explicit
/// binding. Factory value `Auto`.
#[tauri::command]
fn get_default_mode(state: State<'_, Arc<AppState>>) -> Mode {
    state.default_mode()
}

/// Set and persist the global default autonomy mode (#265).
#[tauri::command]
fn set_default_mode(state: State<'_, Arc<AppState>>, mode: Mode) {
    state.set_default_mode(mode);
}

/// The Control-panel view of the permission matrix (#702, RFC 0019 §3): every
/// Mode × Safety cell as a flat, self-describing list.
#[tauri::command]
fn get_permission_matrix(state: State<'_, Arc<AppState>>) -> PermissionMatrixView {
    state.permission_matrix_view()
}

/// Edit and persist a single matrix cell (#702), returning the updated view. Takes
/// effect on the next tool invocation: the approval gate reads the matrix live, and
/// the advertised-tools gate picks it up on the next turn.
#[tauri::command]
fn set_permission_cell(
    state: State<'_, Arc<AppState>>,
    mode: Mode,
    safety: Safety,
    cell: PermissionCell,
) -> PermissionMatrixView {
    state.set_permission_cell(mode, safety, cell)
}

/// Set and persist a per-tool override (#700/#702), returning the updated view.
/// The override replaces the matrix cell for the named tool across every
/// Mode × Safety combination; it is read live by the approval gate.
#[tauri::command]
fn set_tool_override(
    state: State<'_, Arc<AppState>>,
    tool: String,
    cell: PermissionCell,
) -> PermissionMatrixView {
    state.set_tool_override(tool, cell)
}

/// Remove a per-tool override (#700/#702), returning the updated view.
#[tauri::command]
fn remove_tool_override(state: State<'_, Arc<AppState>>, tool: String) -> PermissionMatrixView {
    state.remove_tool_override(&tool)
}

// ---- MCP server control (M4.4, RFC 0003 §3,5) ----
//
// Enable/disable/add/remove write `mcp.json` via `ff_mcp::config`; the existing config
// watcher then reloads and reconciles the supervisor, so these commands deliberately
// do NOT touch the supervisor directly. Only `restart_mcp_server` drives the actor (an
// immediate restart has no config delta for the watcher to observe). Status changes are
// pushed to the FE via the `mcp:status-changed` forwarder wired in `run`.

/// Current status snapshot of every configured MCP server. Empty (not an error) when
/// the MCP host never started (no `mcp.json`, no home dir).
#[tauri::command]
fn list_mcp_servers(state: State<'_, Arc<AppState>>) -> CmdResult<Vec<McpServerStatus>> {
    Ok(state
        .mcp_handle()
        .map(|h| h.status_snapshot())
        .unwrap_or_default())
}

/// Drive an immediate restart of one server, bypassing backoff and reviving a `Failed`
/// server. Unknown ids are a no-op inside the supervisor.
#[tauri::command]
async fn restart_mcp_server(state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    let handle = state
        .mcp_handle()
        .ok_or_else(|| "mcp host is not running".to_string())?;
    handle.restart(&id).await;
    Ok(())
}

/// Enable or disable one server by flipping `disabled` in `mcp.json`. The config watcher
/// reconciles the supervisor; the resulting status change arrives via `mcp:status-changed`.
#[tauri::command]
fn set_mcp_server_enabled(
    state: State<'_, Arc<AppState>>,
    id: String,
    enabled: bool,
) -> CmdResult<()> {
    let path = state
        .mcp_config_path()
        .ok_or_else(|| "mcp host is not running".to_string())?;
    ff_mcp::set_disabled(&path, &id, !enabled).map_err(|e| e.to_string())
}

/// Add (or replace) a server definition in `mcp.json`. `def` is **raw user input** from
/// the add form, never a `load()`-resolved config, so its `env` holds literal strings or
/// `${env:}` templates — mapping it to the un-resolved `McpServerInput` keeps a resolved
/// secret from ever being written back (enforced by the distinct type).
#[tauri::command]
fn add_mcp_server(state: State<'_, Arc<AppState>>, def: McpServerConfig) -> CmdResult<()> {
    let path = state
        .mcp_config_path()
        .ok_or_else(|| "mcp host is not running".to_string())?;
    let input = ff_mcp::McpServerInput {
        id: def.id,
        command: def.command,
        args: def.args,
        env: def.env,
        disabled: def.disabled,
        scope: def.scope,
    };
    ff_mcp::upsert(&path, &input).map_err(|e| e.to_string())
}

/// Remove a server definition from `mcp.json`. Idempotent: absent id is a no-op.
#[tauri::command]
fn remove_mcp_server(state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    let path = state
        .mcp_config_path()
        .ok_or_else(|| "mcp host is not running".to_string())?;
    ff_mcp::remove(&path, &id).map_err(|e| e.to_string())
}

/// Outcome of an update check, serialized to the FE-owned `UpdateStatus` contract
/// in `lib/about.ts` (`{ kind: "upToDate", version }` | `{ kind: "available",
/// version, notes }` | `{ kind: "olderAvailable", version, notes }`). The FE owns
/// the toast copy; the backend reports only the structured outcome (#159, RFC 0014).
#[derive(serde::Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum UpdateStatus {
    UpToDate {
        version: String,
    },
    Available {
        version: String,
        notes: Option<String>,
    },
    /// The feed's build is **older** than the running one (#1034). Only reachable on
    /// the local dogfood channel, where rebuilding an earlier commit is a legitimate
    /// thing to do while bisecting. Deliberately distinct from `Available` so the FE
    /// never banners it and requires an explicit downgrade confirmation first.
    OlderAvailable {
        version: String,
        notes: Option<String>,
    },
}

/// Default port for the local dev-release HTTP server (matches `scripts/dev-release.sh`).
const DEV_RELEASE_PORT: u16 = 8787;

/// Which update feed a check/install should target (#1033). Passed explicitly from
/// the FE on every call so the endpoint is never decided by a global side-effect
/// flag — that ordering-dependent atomic is what let the first boot check race the
/// dev-update watcher and pin the GitHub release into the UI. `github` is the
/// compiled-in default feed; `local` is the `dev-release.sh` server on localhost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
enum UpdateChannel {
    #[default]
    Github,
    Local,
}

/// Where a feed's build sits relative to the running one (#1034).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VersionDirection {
    /// Install it frictionlessly — this is the normal update.
    Newer,
    /// A deliberate downgrade; never bannered, needs explicit confirmation.
    Older,
    Same,
}

/// Classify `remote` against `current`.
///
/// With `dev-release.sh` versioning builds as `0.0.0-dev.<committer epoch>` (#1034),
/// semver ordering of the prerelease is exactly commit recency, so plain `Version`
/// comparison answers the question for two dogfood builds.
///
/// The one deliberate exception is the **dev lineage**: semver ranks every
/// `0.0.0-dev.*` *below* a released `0.1.0`, so a dev running a release build would
/// be asked to confirm a "downgrade" on every single dogfood install. When the feed
/// offers a dev build and the running app is not one, treat it as `Newer` — entering
/// the dev lineage is the point of turning the channel on, not a downgrade.
///
/// An unparseable version counts as `Newer`: fail toward the pre-#1034 frictionless
/// behavior rather than silently hiding a build the dev asked for.
pub(crate) fn compare_build(current: &str, remote: &str) -> VersionDirection {
    if current == remote {
        return VersionDirection::Same;
    }
    let (Ok(current), Ok(remote)) = (
        semver::Version::parse(current),
        semver::Version::parse(remote),
    ) else {
        return VersionDirection::Newer;
    };
    let is_dev = |v: &semver::Version| v.pre.as_str().starts_with("dev");
    if is_dev(&remote) && !is_dev(&current) {
        return VersionDirection::Newer;
    }
    match remote.cmp(&current) {
        std::cmp::Ordering::Greater => VersionDirection::Newer,
        std::cmp::Ordering::Less => VersionDirection::Older,
        std::cmp::Ordering::Equal => VersionDirection::Same,
    }
}

fn updater(
    app: &tauri::AppHandle,
    channel: UpdateChannel,
) -> CmdResult<tauri_plugin_updater::Updater> {
    use tauri_plugin_updater::UpdaterExt;
    // An explicit `FF_UPDATER_ENDPOINT` still wins (RFC 0014 D1 override), then the
    // caller-supplied channel — never an ambient flag.
    if let Ok(endpoint) = std::env::var("FF_UPDATER_ENDPOINT") {
        let endpoint = url::Url::parse(&endpoint).map_err(|e| e.to_string())?;
        app.updater_builder()
            .endpoints(vec![endpoint])
            .map_err(|e| e.to_string())?
            // Permissive on purpose (#1034): the comparator is the plugin's only gate,
            // so accepting either direction is what lets `check()` hand us an older
            // build at all. Direction is classified by `compare_build` below.
            .version_comparator(|current, update| update.version != current)
            .build()
            .map_err(|e| e.to_string())
    } else if channel == UpdateChannel::Local {
        let endpoint = url::Url::parse(&format!("http://localhost:{DEV_RELEASE_PORT}/latest.json"))
            .map_err(|e| e.to_string())?;
        app.updater_builder()
            .endpoints(vec![endpoint])
            .map_err(|e| e.to_string())?
            // Any different version reaches us in either direction (#1034); whether it
            // is an upgrade or a downgrade is decided by `compare_build`, and a
            // downgrade additionally needs the caller's explicit opt-in.
            .version_comparator(|current, update| update.version != current)
            .build()
            .map_err(|e| e.to_string())
    } else {
        app.updater().map_err(|e| e.to_string())
    }
}

/// Check the given update feed. Returns the structured `UpdateStatus` so the UI
/// can branch (offer "Update now" on `available`). Errors (offline, malformed
/// manifest, local feed unreachable) surface as `Err(String)` for the FE to toast.
/// The `channel` is explicit (#1033) — the endpoint is never inferred from a global
/// flag, so a boot check can't race the dev-update watcher onto the wrong feed.
///
/// The feed's build is classified by direction (#1034): only a genuinely newer build
/// is `Available` (bannered, one-click); an older one is `OlderAvailable`, which the
/// FE shows only in Settings → About behind a downgrade confirmation.
#[tauri::command]
async fn check_for_updates(
    app: tauri::AppHandle,
    channel: UpdateChannel,
) -> CmdResult<UpdateStatus> {
    let current = app.package_info().version.to_string();
    match updater(&app, channel)?
        .check()
        .await
        .map_err(|e| e.to_string())?
    {
        Some(update) => Ok(match compare_build(&current, &update.version) {
            VersionDirection::Newer => UpdateStatus::Available {
                version: update.version.clone(),
                notes: update.body.clone(),
            },
            VersionDirection::Older => UpdateStatus::OlderAvailable {
                version: update.version.clone(),
                notes: update.body.clone(),
            },
            VersionDirection::Same => UpdateStatus::UpToDate { version: current },
        }),
        None => Ok(UpdateStatus::UpToDate { version: current }),
    }
}

/// Whether the re-checked feed may be installed, given what the user actually agreed to.
///
/// `install_update` re-checks the feed rather than caching an `Update` handle across IPC
/// calls, which opens a check→install window: the version the user saw (and, for a
/// downgrade, explicitly confirmed) came from an *earlier* `check_for_updates`, and
/// another `dev-release.sh` run can move the feed in between. So consent is checked
/// against `expected` — the exact version the FE displayed — not merely against the
/// direction (#1034 review). A moved feed is refused so the FE can re-prompt with the
/// build that is actually there, instead of installing something the user never saw.
///
/// Pure so both guards are unit-testable without an `AppHandle`.
pub(crate) fn install_guard(
    current: &str,
    remote: &str,
    expected: &str,
    allow_downgrade: bool,
) -> Result<(), String> {
    if remote != expected {
        return Err(format!(
            "update feed moved: you confirmed {expected}, but the feed now offers {remote}; re-check before installing"
        ));
    }
    if !allow_downgrade && compare_build(current, remote) == VersionDirection::Older {
        return Err(format!(
            "refusing to install {remote} — it is older than the running {current}; confirm the downgrade first"
        ));
    }
    Ok(())
}

/// Download and install the available update from `channel`, then relaunch. Re-checks
/// rather than caching the `Update` handle across IPC calls. A no-op if nothing is
/// available. `channel` must match the check that surfaced the update (#1033).
///
/// `expected_version` is the version the FE showed the user, and installing an **older**
/// build additionally requires `allow_downgrade` (#1034) — the FE only passes it after
/// the user confirms the dialog in Settings → About. Both are enforced by
/// [`install_guard`], so no code path (banner, dogfood auto-install, a feed that moved
/// mid-confirmation) can install a build the user did not agree to.
#[tauri::command]
async fn install_update(
    app: tauri::AppHandle,
    channel: UpdateChannel,
    expected_version: String,
    allow_downgrade: bool,
) -> CmdResult<()> {
    let Some(update) = updater(&app, channel)?
        .check()
        .await
        .map_err(|e| e.to_string())?
    else {
        return Ok(());
    };
    let current = app.package_info().version.to_string();
    install_guard(
        &current,
        &update.version,
        &expected_version,
        allow_downgrade,
    )?;
    let progress_app = app.clone();
    let finish_app = app.clone();
    let mut downloaded: u64 = 0;
    update
        .download_and_install(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ =
                    progress_app.emit("update:progress", UpdateProgressEvent { downloaded, total });
            },
            move || {
                let _ = finish_app.emit("update:download-finished", ());
            },
        )
        .await
        .map_err(|e| e.to_string())?;
    app.restart();
}

/// Start the local dev-update file watcher (#705, Phase 2). Called by the FE on
/// boot when the `localUpdateChannel` experimental flag is on. The watcher observes
/// `~/.config/flowforge/dev-update/latest.json` via kqueue/inotify and emits
/// `update:local-feed-changed` instantly when the file is written, so the FE can
/// trigger an immediate `refresh()` without waiting for the 15s poll tick.
///
/// This only starts the FS observer; it does NOT decide the update endpoint — that
/// is chosen per-call via the explicit `UpdateChannel` (#1033). Decoupling the two
/// removes the boot race where a check could fire before this ran.
///
/// Idempotent: calling twice is a no-op (the watcher is stored in managed state and
/// lives for the app lifetime; there is no stop — it is zero-cost when idle).
#[tauri::command]
fn start_dev_update_watcher(app: tauri::AppHandle) {
    use std::sync::Once;
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        match dev_update_watcher::DevUpdateWatcher::spawn() {
            Ok((_watcher, mut rx)) => {
                // Leak the watcher so it lives for the process lifetime (zero-cost
                // when idle; the OS kqueue fd stays open).
                std::mem::forget(_watcher);
                let emit_app = app.clone();
                tauri::async_runtime::spawn(async move {
                    while rx.recv().await.is_some() {
                        let _ = emit_app.emit("update:local-feed-changed", ());
                    }
                });
                tracing::info!("dev-update watcher started");
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to start dev-update watcher");
            }
        }
    });
}

/// Map an [`AgentEvent`] to its frontend Tauri event and emit it.
///
/// This is the single source of truth for the `AgentEvent → app.emit(…)` wire
/// mapping shared by both turn paths — the in-process `run_turn` closure (in
/// `send_message`) and the CLI sidecar loop below (`run_sidecar_turn`).
/// Keeping both paths on one helper is exactly what the sidecar parity
/// smoke-test (RFC 0004 §5) guards against drift: if the mapping changes, it
/// changes in one place.
///
/// Generic over `R: tauri::Runtime` so the sidecar parity integration test can
/// drive it through a `MockRuntime` app handle (see `tests::sidecar_turn_*`).
fn emit_agent_event<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    session_id: &str,
    event: AgentEvent,
) {
    match event {
        AgentEvent::Token { message_id, delta } => {
            let _ = app.emit(
                "turn:token",
                TokenEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    delta,
                },
            );
        }
        AgentEvent::Reasoning { message_id, delta } => {
            let _ = app.emit(
                "turn:reasoning",
                ReasoningEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    delta,
                },
            );
        }
        AgentEvent::ToolCallStarted {
            message_id,
            call_id,
            name,
            args,
        } => {
            let _ = app.emit(
                "tool:call",
                ToolCallEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    call_id,
                    tool: name,
                    args,
                },
            );
        }
        AgentEvent::ToolCallFinished {
            message_id,
            call_id,
            success,
            result,
            observer_intent,
        } => {
            // #1039: a long-running tool (process_manager wake_on / test_runner
            // background) can declare a background observer. Attach it here — the
            // host owns the ObserverSupervisor; the tool crate can't (ff-observer
            // depends on ff-tools). Best-effort: a failure only logs.
            if let Some(intent) = observer_intent {
                if let Some(state) = app.try_state::<Arc<AppState>>() {
                    state.attach_observer_intent(session_id, *intent);
                    // #1038 M2: a newly attached observer is a change the panel
                    // wants to show without the user asking (coarse — FE re-lists).
                    let _ = app.emit(
                        "observer:changed",
                        ObserverChangedEvent {
                            session_id: session_id.to_string(),
                        },
                    );
                }
            }
            let _ = app.emit(
                "tool:result",
                ToolResultEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    call_id,
                    success,
                    result,
                },
            );
        }
        AgentEvent::Done {
            message_id,
            token_count,
            stop_reason,
            breakdown,
            usage,
            budget_tokens,
            ..
        } => {
            // #1107: persist the preheat attribution before the event carries the
            // breakdown away. The counters ride `turn:done`, which the UI
            // overwrites next turn, so without this row nothing can say whether
            // the 2500 B preheat budget earned its keep beyond the current turn.
            // Best-effort, mirroring the observer-intent hop above: losing a row
            // costs analysis fidelity, never the turn.
            if let Some(pre) = breakdown.as_ref() {
                if let Some(state) = app.try_state::<Arc<AppState>>() {
                    state.store.put_turn_preheat(
                        session_id,
                        &message_id,
                        pre.preheated_count.unwrap_or(0) as usize,
                        pre.preheated_used.unwrap_or(0) as usize,
                        pre.preheated_bytes.unwrap_or(0) as usize,
                    );
                }
            }
            let _ = app.emit(
                "turn:done",
                TurnDoneEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    token_count,
                    stop_reason,
                    breakdown,
                    usage,
                    budget_tokens,
                },
            );
        }
        AgentEvent::MemoryFlushed { message_id, writes } => {
            let _ = app.emit(
                "memory:flushed",
                MemoryFlushedEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    writes,
                },
            );
        }
        AgentEvent::ToolOutputChunk {
            message_id,
            call_id,
            stream,
            delta,
        } => {
            let stream = match stream {
                ff_tools::OutputStream::Stdout => OutputStreamKind::Stdout,
                ff_tools::OutputStream::Stderr => OutputStreamKind::Stderr,
            };
            let _ = app.emit(
                "tool:output",
                ToolOutputChunkEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    call_id,
                    stream,
                    delta,
                },
            );
        }
        AgentEvent::AttachmentsDropped { .. } => {
            // User-facing notice deferred to PR-2 / #342 (transcript render).
        }
        AgentEvent::EgressMismatch {
            message_id,
            kind,
            model,
        } => {
            // LocalOnly-but-cloud-inference notice (#888). The agent surfaces a
            // typed event so the frontend can render a privacy badge / banner
            // and so the host can record telemetry. No in-process consumer yet;
            // the FE is the primary surface. Distinct from
            // [`AttachmentsDropped`] because the gap here is policy-vs-reality
            // (the user asked for local-only; the connection is hosted), not a
            // capability strip.
            let _ = app.emit(
                "egress:mismatch",
                EgressMismatchEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    kind,
                    model,
                },
            );
        }
        AgentEvent::Error { message } => {
            let _ = app.emit(
                "turn:error",
                TurnErrorEvent {
                    session_id: session_id.to_string(),
                    message,
                },
            );
        }
        AgentEvent::Reconnecting {
            message_id,
            attempt,
            max_attempts,
        } => {
            let _ = app.emit(
                "turn:reconnecting",
                ReconnectingEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    attempt,
                    max_attempts,
                },
            );
        }
        AgentEvent::ConnectionFailed {
            message_id,
            message,
        } => {
            let _ = app.emit(
                "turn:connection-failed",
                ConnectionFailedEvent {
                    session_id: session_id.to_string(),
                    message_id,
                    message,
                },
            );
        }
    }
}

/// CLI.7 — Sidecar parity smoke-test (RFC 0004 §5).
///
/// Spawns the bundled `flowforge` CLI as a Tauri sidecar (`externalBin`),
/// invokes `flowforge run "<prompt>" --json`, and re-emits every parsed
/// `AgentEvent` as the same Tauri events the in-process `run_turn` path
/// emits (`turn:token`, `tool:call`, `turn:done`, …).  This lets the
/// frontend verify that the sidecar produces an event stream equivalent
/// to the in-process path.
///
/// **PATH caveat (RFC 0004 §5):** the bundled sidecar binary lives inside
/// the app bundle and is NOT on the user's PATH.  Users who want
/// `flowforge` on the command line must install it separately or symlink
/// it manually — see the caveat doc in `docs/rfcs/0004-cli.md`.
#[tauri::command]
async fn run_sidecar_turn<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    prompt: String,
) -> Result<serde_json::Value, String> {
    let session_id = Uuid::new_v4().to_string();

    let command = app
        .shell()
        .sidecar("flowforge")
        .map_err(|e| format!("failed to resolve sidecar: {e}"))?
        .args(["run", prompt.as_str(), "--json"]);

    let (mut rx, _child) = command
        .spawn()
        .map_err(|e| format!("failed to spawn sidecar: {e}"))?;

    let mut event_count = 0usize;

    while let Some(event) = rx.recv().await {
        match event {
            tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let agent_event: AgentEvent = match serde_json::from_str(line) {
                    Ok(e) => e,
                    // The CLI emits only `AgentEvent` lines under `--json`, so a parse
                    // failure normally means a schema drift between the two binaries.
                    // Break loudly rather than swallowing it without a trace.
                    Err(e) => {
                        eprintln!(
                            "[sidecar] failed to parse stdout line as AgentEvent ({e}); \
                             this usually signals a schema drift between the desktop and \
                             CLI binaries. line: {line}"
                        );
                        continue;
                    }
                };
                event_count += 1;
                emit_agent_event(&app, &session_id, agent_event);
            }
            tauri_plugin_shell::process::CommandEvent::Stderr(bytes) => {
                eprintln!("[sidecar stderr] {}", String::from_utf8_lossy(&bytes));
            }
            tauri_plugin_shell::process::CommandEvent::Terminated(payload) => {
                if payload.code != Some(0) {
                    return Err(format!("sidecar exited with code {:?}", payload.code));
                }
                break;
            }
            tauri_plugin_shell::process::CommandEvent::Error(err) => {
                return Err(format!("sidecar error: {err}"));
            }
            _ => {}
        }
    }

    Ok(serde_json::json!({
        "session_id": session_id,
        "events": event_count,
    }))
}

/// Boot trace (#599 item 0): the FE reports first paint on the same `BOOT_T0`
/// clock. `fe_nav_ms` is the webview-internal `performance.now()` (≈ navigation
/// start → paint), which isolates the WKWebView/process floor from our own work.
#[tauri::command]
fn mark_fe_ready(phase: String, fe_nav_ms: f64) {
    boot_trace(
        &format!("fe.{phase}"),
        Some(&format!("fe_nav {fe_nav_ms:.1}ms")),
    );
}

/// Paint-first boot (#599): has the background hydrate task finished
/// `AppState::new()` and published the managed state yet? Reads a static flag
/// (no `AppState` dependency) so it is safe to call before the state is managed
/// — the FE uses it as the "subscribe-then-check" half of the `app:ready` gate,
/// closing the race where the event fired before its listener attached.
#[tauri::command]
fn is_app_ready() -> bool {
    APP_READY.load(Ordering::SeqCst)
}

// Boot finalization seam (#599 review nit): the three side effects of "hydrate
// done" — `manage(state)`, `APP_READY = true`, `emit("app:ready")` — must run in
// exactly that order, or the FE's subscribe-then-check gate silently breaks:
//   * `store` before `manage` → `is_app_ready()` is true while no state is
//     managed, so a command the FE fires on the gate reads an unresolved
//     `State<'_, Arc<AppState>>`.
//   * `emit` before `store` → the FE's `isAppReady()` poll (run on receiving
//     `app:ready`) returns false and the loading state hangs.
// The order was correct as written but nothing guarded it — a future reorder of
// those three lines would slip past review. `publish_app_ready` makes the order
// a single reviewable unit, and `BootFinalize` lets a test substitute a spy
// that asserts the order at call time (see `tests::boot_finalize_orders_*`).
trait BootFinalize {
    /// Publish the managed state to the command layer. Runs BEFORE the
    /// `APP_READY` flag flips.
    fn manage_state(&self, state: Arc<AppState>);
    /// Notify the FE the loading gate may drop. Runs AFTER the `APP_READY`
    /// flag flips AND after `manage_state`.
    fn emit_ready(&self);
}

/// Render a caught panic payload (`Box<dyn Any + Send>`) into a human-readable,
/// `context`-prefixed message. A panic payload is one of two common stringy forms
/// (`&'static str` from `panic!("literal")`, `String` from `panic!("{fmt}")`) or,
/// rarely, something else; fall back to a generic note so the FE never gets an
/// opaque "unknown" with no clue where to look. Shared by the detached MCP-init
/// task and the outer `post-init` `catch_unwind` so both format identically.
fn panic_message(context: &str, payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        format!("{context} panicked: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("{context} panicked: {s}")
    } else {
        format!("{context} panicked (non-string panic payload)")
    }
}

/// Run the boot-finalization sequence in the order the FE gate requires:
/// `manage` → `APP_READY = true` → `emit("app:ready")`. The only call site is
/// the hydrate task in `run()`; the order is the invariant guarded by
/// `tests::boot_finalize_orders_manage_before_flag_before_emit`.
fn publish_app_ready<F: BootFinalize>(finalizer: &F, state: Arc<AppState>) {
    finalizer.manage_state(state);
    APP_READY.store(true, Ordering::SeqCst);
    boot_trace("app_ready", None);
    finalizer.emit_ready();
}

impl<R: tauri::Runtime> BootFinalize for tauri::AppHandle<R> {
    fn manage_state(&self, state: Arc<AppState>) {
        self.manage(state);
    }
    fn emit_ready(&self) {
        let _ = self.emit("app:ready", ());
    }
}

/// Outer bound on the MCP teardown at quit (#1195). The previous handler awaited
/// `stop_all()` unbounded, so one wedged MCP server could hold a quit open indefinitely.
///
/// Must stay comfortably above `ff_mcp`'s per-server `SHUTDOWN_TIMEOUT` (2s), which
/// already bounds each graceful `client.shutdown()` — this budget covers the actor
/// round-trip *around* those closes, so setting it at or below 2s would clip a
/// perfectly healthy shutdown. It exists because the per-server bound is not enough
/// on its own: the supervisor actor can be busy elsewhere and never reach the stop.
const MCP_SHUTDOWN_BUDGET: Duration = Duration::from_secs(3);

/// The name of the quit path `event` represents, or `None` if it isn't one.
///
/// Returns a label rather than a bool so the teardown can say *which* path it ran on. That
/// label is not decoration: #1195 was filed on a static trace claiming ⌘Q reaches only
/// `RunEvent::Exit`, and it took this log line to establish that on tauri 2.11.2 / Darwin
/// 25.5 ⌘Q in fact emits `ExitRequested` first and then `Exit`. AppKit's termination
/// destroys the window, which trips `tauri-runtime-wry`'s last-window-`Destroyed` branch —
/// the very `ExitRequested` source the issue lists — before `LoopDestroyed` maps to `Exit`.
/// Keep it: quit-path behaviour here is version-dependent and not reliably derivable by
/// reading the vendored sources.
///
/// Both arms are handled because neither event alone is guaranteed:
///
/// - `ExitRequested` is emitted from `Message::RequestExit` (an explicit `app.exit()`) and
///   from the last-window-`Destroyed` branch. A quit that destroys no window — or a tao
///   version that skips `Destroyed` on termination, which is what #1195 predicted — never
///   produces it.
/// - `Exit` comes from `LoopDestroyed` and is the last thing the loop emits on every path.
///   It is what `tauri-plugin-store` hangs its own on-exit save off, and it cannot be
///   prevented (tao exposes no `applicationShouldTerminate`), so the work behind it must be
///   bounded — hence `MCP_SHUTDOWN_BUDGET`.
///
/// Firing on both is idempotent, and measured to be: `Cmd::StopAll` makes the supervisor
/// actor return, so the second `stop_all()` finds the command channel closed and returns
/// at once (observed ~6ms apart, both reporting completion) rather than tearing down twice.
fn mcp_quit_path(event: &tauri::RunEvent) -> Option<&'static str> {
    match event {
        tauri::RunEvent::Exit => Some("Exit"),
        tauri::RunEvent::ExitRequested { .. } => Some("ExitRequested"),
        _ => None,
    }
}

/// Block the calling thread on `fut`, giving up after `budget`. Returns whether it
/// completed.
///
/// The quit paths call this from the event-loop callback, which is not itself inside a
/// runtime — `tauri::async_runtime::block_on` enters the shared one, so the timer works.
/// Generic over the future so the budget can be tested against a wedged future without
/// standing up a real supervisor.
fn bounded_block_on<F: std::future::Future<Output = ()>>(fut: F, budget: Duration) -> bool {
    let completed =
        tauri::async_runtime::block_on(async move { tokio::time::timeout(budget, fut).await })
            .is_ok();
    if !completed {
        tracing::warn!(
            budget_secs = budget.as_secs(),
            "mcp teardown timed out at exit; abandoning it to let the quit proceed"
        );
    }
    completed
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Boot-trace origin (#599 item 0): the earliest FlowForge-controlled point.
    // OS process spawn + WKWebView init happen before any of our code and are
    // unmeasurable from here (the platform floor the issue calls out); the FE's
    // `performance.now()` in `mark_fe_ready` gives the closest proxy for that.
    let _ = BOOT_T0.set(std::time::Instant::now());
    // #1118: install the tracing subscriber FIRST, before any instrumented code
    // runs — the observer pump, process reaper and scheduler all emit on paths
    // reached during `setup`, and events emitted before this line are lost.
    // On by default at the warn floor (#1118); `FF_LOG` raises or lowers it and
    // `FF_LOG=off` opts out entirely. The guard must outlive the app, so it's
    // bound for the whole of `run()`: dropping it flushes and stops the writer
    // thread.
    let _log_guard = state::flowforge_config_dir().and_then(|dir| ff_logging::init(&dir));
    // Paint-first boot (#599): `AppState::new()` and the supervisor / watcher /
    // reaper / scheduler wiring are deferred to a background hydrate task (spawned
    // from `setup` below) so the window is created and painted FIRST — the FE
    // renders a loading state until the `app:ready` event (mirrors the
    // `mcp:status-changed` / `workspace:branch-changed` hydrate pattern) and the
    // `is_app_ready` flag. This is a reordering, not new logic: the same work runs,
    // just off the synchronous pre-window path.
    let app = tauri::Builder::default()
        // The open PTYs (#1284). Managed here rather than with `AppState` (which
        // only lands after the async boot) because it has no initialization to
        // wait on -- it starts empty and fills the first time a drawer opens.
        .manage(terminal::Terminals::default())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        // Boot trace (#599 item 0): stamp when the webview begins loading our
        // HTML (webview process up) and when the document finishes loading. These
        // now land BEFORE `app_state_new` — the win this reordering buys.
        .on_page_load(|_webview, payload| {
            let url = payload.url().to_string();
            match payload.event() {
                tauri::webview::PageLoadEvent::Started => {
                    boot_trace("webview.page-load-started", Some(&url))
                }
                tauri::webview::PageLoadEvent::Finished => {
                    boot_trace("webview.page-load-finished", Some(&url))
                }
            }
        })
        .setup(move |app| {
            // Spawn the heavy init off the synchronous pre-window path. `setup`
            // returns immediately, the window paints, and this task runs to
            // completion before publishing the state and emitting `app:ready`.
            //
            // `AppState::new()` is sync blocking I/O (SQLite opens, skill scans,
            // the FTS5 index build, fs seed); `spawn_blocking` runs it on the
            // blocking pool so it never stalls an async worker, and — like the
            // main-thread path it replaces — needs no entered reactor (the skill /
            // memory / git watchers are `std::thread`, not `tokio::spawn`).
            // `init_mcp` / `start_process_reaper` enter the runtime themselves
            // (issue #117), so they are safe to call from this async task too.
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = match tokio::task::spawn_blocking(AppState::new).await {
                    Ok(s) => Arc::new(s),
                    Err(e) => {
                        let msg = format!("AppState::new() panicked: {e}");
                        tracing::error!("{msg}");
                        let _ = app_handle.emit("app:init-error", &msg);
                        return;
                    }
                };
                // Cumulative since t0 ≈ the blocking `AppState::new()` cost (now
                // post-paint); per-step durations are emitted from `state.rs`
                // (registry/seed/skills/memory/DBs).
                boot_trace("app_state_new", None);

                // Guard the post-spawn_blocking body so no panic goes unlogged.
                // This task's `JoinHandle` is dropped, so a panic in
                // `init_git_watcher` / `start_process_reaper` / the scheduler
                // wiring would be silently discarded by Tokio — leaving
                // `APP_READY` false and the FE on a dead `<BootSplash>` spinner
                // with no clue where it stopped (regression: previously this ran
                // on the main thread, where a panic was a visible crash). Catch
                // the unwind, log it, and emit `app:init-error` (the same wire the
                // `AppState::new()` panic path above uses) so the FE can surface an
                // actionable error. `error_handle` stays outside the closure so it
                // survives the panic to emit. (The detached MCP-init task below
                // guards its own body the same way — the outer catch covers only
                // its `spawn` call, not the task's later execution.)
                let error_handle = app_handle.clone();
                let post_init = move || {
                    // MCP servers are not needed for first paint (RFC 0003; the
                    // composer works with zero tools, and `list_mcp_servers`
                    // returns empty-not-error before the host is up), so init runs
                    // in its OWN spawned task rather than inline here — keeping the
                    // synchronous `mcp.json` read + OS watcher registration inside
                    // `init_mcp` off the path to `publish_app_ready` (#599 item 5).
                    // `init_mcp` enters the shared Tokio runtime itself, so it's
                    // safe from this task (issue #117). The `mcp:status-changed`
                    // forwarder stays in the SAME task, sequenced AFTER `init_mcp`
                    // returns, so `mcp_handle()` is populated before it is read —
                    // no status updates are lost to a race. This task is detached,
                    // so guard its body: a panic here would otherwise be discarded
                    // by Tokio (the outer `catch_unwind` covers only the spawn).
                    {
                        let state = state.clone();
                        let app_handle = app_handle.clone();
                        tauri::async_runtime::spawn(async move {
                            // `&AppState` is not `UnwindSafe`, so assert it across
                            // the catch boundary (the state is only read here).
                            let guarded = AssertUnwindSafe(|| state.init_mcp());
                            if let Err(payload) = catch_unwind(guarded) {
                                let msg = panic_message("mcp init", &payload);
                                tracing::error!("{msg}");
                                let _ = app_handle.emit("app:init-error", &msg);
                                return;
                            }
                            // Forward supervisor status changes to the FE as
                            // `mcp:status-changed`, so the servers panel reflects
                            // start/stop/restart/health without polling. The watch
                            // tick coalesces; we re-snapshot on each wake.
                            if let Some(handle) = state.mcp_handle() {
                                let mut rx = handle.status_changed_rx();
                                while rx.changed().await.is_ok() {
                                    let servers = handle.status_snapshot();
                                    let _ = app_handle.emit(
                                        "mcp:status-changed",
                                        McpStatusChangedEvent { servers },
                                    );
                                }
                            }
                        });
                    }
                    // Live-sync the active session's git branch (#561 BE half): the
                    // GitHeadWatcher observes the workspace's `.git/HEAD` and emits
                    // `workspace:branch-changed` on a real branch change, which the FE
                    // listener merged in PR #581 patches into the composer chip with no
                    // remount. A `None` rx means the watcher could not start -- live sync
                    // degrades to off rather than failing the app.
                    if let Some(mut rx) = state.init_git_watcher() {
                        let app_handle = app_handle.clone();
                        tauri::async_runtime::spawn(async move {
                            while let Some(ws) = rx.recv().await {
                                let _ = app_handle.emit("workspace:branch-changed", ws);
                            }
                        });
                    }
                    // Drive the periodic idle-process reaper
                    // (`ProcessSupervisor::reap_idle`): finished processes and ones the
                    // agent abandoned (started but never polled again) are cleaned up on
                    // a timer. Enters the runtime itself, so it's safe here too.
                    state.start_process_reaper();
                    // Drive the observer event pump (#891 Phase 1): a single
                    // long-lived task drains the supervisor's event channel
                    // and turns each event into either a fresh turn (when the
                    // session is idle) or a deferred wake queued for the
                    // next turn. Enters the runtime itself, like
                    // `start_process_reaper` — safe here.
                    state.start_observer_pump(&app_handle);
                    // Drive the process-output pump (#873): a single long-lived
                    // task bridges each background process's live stdout/stderr
                    // to the frontend as `process:output` events across turns.
                    // Enters the runtime itself, like the pumps above.
                    state.start_process_output_pump(&app_handle);
                    // Drive the scheduled-task firing engine (RFC 0017 section 4, #542):
                    // a background sweep fires due tasks through the desktop runner. The
                    // tick is coarse; the due predicate is minute-granular, so a 30s sweep
                    // never misses a slot and never double-fires (stamped last_run gates
                    // it). The sweep loop is inlined here rather than calling
                    // `ff_scheduled::spawn_scheduler` (which does its own internal
                    // `tokio::spawn`), so this single task owns it directly.
                    let scheduler_runner: Arc<dyn ff_scheduled::TaskRunner> =
                        Arc::new(DesktopTaskRunner {
                            state: state.clone(),
                            app: app_handle.clone(),
                        });
                    {
                        let sched_store = state.scheduled.clone();
                        let sched_app = app_handle.clone();
                        tauri::async_runtime::spawn(async move {
                            let mut interval = tokio::time::interval(Duration::from_secs(30));
                            loop {
                                interval.tick().await;
                                let fired = ff_scheduled::run_due_once(
                                    sched_store.as_ref(),
                                    scheduler_runner.as_ref(),
                                )
                                .await;
                                // Emit one `scheduled:fired` per appended run, plus a
                                // single `scheduled:changed` snapshot when anything fired,
                                // so the UI live-updates without polling (#544).
                                for run in &fired {
                                    emit_scheduled_fired(&sched_app, sched_store.as_ref(), run);
                                }
                            }
                        });
                    }

                    // Resume any goal that was `Active` when the app last closed
                    // (RFC 0020 §5.3, #802). `spawn_goal_loop` is single-flight
                    // guarded and re-checks `status`, so a racing IPC `goal_resume`
                    // can never double-spawn. This is best-effort and off the
                    // first-paint path (post_init already runs after `app:ready` is
                    // scheduled), so a slow or failed scan never blocks boot. The
                    // loops spawn before `publish_app_ready` (they hold their own
                    // `Arc<AppState>`, not the managed state), but the initial
                    // `goal:updated` emit is deferred until after it so the FE panel
                    // is already listening when the "running" snapshot arrives.
                    let resumed = state.goals.list_active();
                    for goal in &resumed {
                        spawn_goal_loop(state.clone(), app_handle.clone(), goal.session_id.clone());
                    }

                    // State is now usable: publish it to the command layer and notify
                    // the FE to drop its loading gate. Commands read
                    // `State<'_, Arc<AppState>>`, which only resolves once `manage` runs
                    // — the FE never invokes them before `app:ready` (gated by
                    // `is_app_ready`), so the pre-ready window sees no unmanaged-state
                    // error. `publish_app_ready` runs the three steps in the order the
                    // gate requires (see its invariant); doing it inline would let a
                    // future reorder break the gate silently.
                    publish_app_ready(&app_handle, state);

                    for goal in &resumed {
                        let _ = app_handle.emit("goal:updated", goal);
                    }
                };
                if let Err(payload) = catch_unwind(AssertUnwindSafe(post_init)) {
                    let msg = panic_message("post-init", &payload);
                    tracing::error!("{msg}");
                    let _ = error_handle.emit("app:init-error", &msg);
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            create_session,
            list_sessions,
            get_messages,
            search_messages,
            search_in_session,
            export_session,
            rename_session,
            delete_session,
            fork_session,
            list_scheduled_tasks,
            create_scheduled_task,
            toggle_scheduled_task,
            delete_scheduled_task,
            run_scheduled_task_now,
            list_scheduled_runs,
            set_scheduled_paused_all,
            goal_set,
            goal_status,
            goal_pause,
            goal_resume,
            goal_clear,
            goal_complete,
            notebook_status,
            notebook_stop,
            notebook_restart,
            list_observers,
            stop_observer,
            preview_cadence,
            get_session_workspace,
            set_session_workspace,
            list_branches,
            checkout_branch,
            list_memory_files,
            read_memory_file,
            memory_overview,
            write_curated_memory,
            list_directory,
            read_file,
            terminal_open,
            terminal_write,
            terminal_resize,
            terminal_close,
            list_memory_chunks,
            reset_memory_chunk,
            sleep_memory_chunk,
            set_memory_chunk_pinned,
            send_message,
            edit_message,
            run_sidecar_turn,
            cancel_turn,
            respond_approval,
            respond_ask,
            set_session_approve,
            set_always_approve,
            remove_always_approve,
            list_always_approved,
            get_provider_config,
            set_provider_config,
            get_provider_registry,
            set_active_connection,
            upsert_connection,
            remove_connection,
            set_provider_secret,
            clear_provider_secret,
            provider_secret_presence,
            resolved_bedrock_auth,
            get_control_config,
            set_control_config,
            get_search_config,
            set_search_config,
            set_search_secret,
            clear_search_secret,
            search_secret_presence,
            list_models,
            test_connection,
            warmup,
            install_skill,
            uninstall_skill,
            list_skills,
            search_skills,
            get_skill_telemetry,
            optimize_skill,
            rollback_skill,
            list_skill_versions,
            activate_skill,
            deactivate_skill,
            list_phenotypes,
            get_phenotype,
            switch_phenotype,
            update_phenotype,
            set_session_phenotype,
            set_session_mode,
            set_session_model_selection,
            get_session_model_selection,
            resolve_model_selection,
            get_default_mode,
            set_default_mode,
            get_permission_matrix,
            set_permission_cell,
            set_tool_override,
            remove_tool_override,
            list_mcp_servers,
            restart_mcp_server,
            set_mcp_server_enabled,
            add_mcp_server,
            remove_mcp_server,
            check_for_updates,
            install_update,
            start_dev_update_watcher,
            mark_fe_ready,
            is_app_ready,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");
    app.run(|app, event| {
        // Stop every MCP child cleanly while the runtime is still alive — drop alone
        // reaps the child via `process_wrap`, but its background task needs a live Tokio
        // runtime, which exits with the app (RFC 0003 §5). Which events qualify, and why
        // the wait is bounded, live in `mcp_quit_path` / `bounded_block_on` so both are
        // covered by tests; keep this dispatch a thin call to them.
        if let Some(quit_path) = mcp_quit_path(&event) {
            if let Some(state) = app.try_state::<Arc<AppState>>() {
                if let Some(handle) = state.mcp_handle() {
                    let completed = bounded_block_on(handle.stop_all(), MCP_SHUTDOWN_BUDGET);
                    tracing::info!(quit_path, completed, "mcp teardown at quit");
                }
            }
        }
    });
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
