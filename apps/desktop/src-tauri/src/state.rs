use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ff_agent::{
    flush_due, AbstractiveConfig, CancelToken, CompactionCache, CompactionContext,
    CompactionOutcome, CompactionStrategy, ContextPressureEstimator, MemoryFlush,
    ProxyTokenEstimator, DEFAULT_FLUSH_AT_FRACTION, INTERRUPTED_NOTICE,
};
use ff_core::{
    model_supports_documents, BedrockAuth, ConnectionId, ContextWindowSource, GoalStore, McpScope,
    McpServerConfig, McpServerState, McpServerStatus, Mode, ModelSelection, Phenotype,
    ProviderConfig, ProviderConnection, ProviderKind, ProviderRegistry, ResolvedModel,
    SearchBackend, SearchConfig, SecretKind, SessionWorkspace,
};
use ff_llm::{
    model_context_window, ollama_num_ctx_from_env, reasoning_control, wire_dialect, BedrockCreds,
    BedrockProvider, OllamaProvider, OpenAiProvider, Provider, ServedWindowProbe,
};
use ff_mcp::{McpConfigWatcher, SupervisorHandle};

use crate::git_watch::GitHeadWatcher;
use ff_memory::watch::MemoryWatcher;
use ff_memory::{
    DecayConfig, EmbeddingProvider, FlushLedger, Fts5Index, HybridIndex, Memory, MemoryConfig,
    MemoryIndex, NoopEmbedder, OpenAiEmbedder,
};
use ff_observer::{ObserverEvent, ObserverSupervisor, ObserverTool};
use ff_scheduled::ScheduledStore;
use ff_session::SessionStore;
use ff_signals::{SignalStore, SkillAggregate, SkillCompleted};
use ff_skills::{
    default_phenotype, load_phenotypes, save_phenotype, SharedRegistry, SkillRegistry,
    SkillWatcher, DEFAULT_PHENOTYPE,
};
use ff_tools::memory::{MemoryConsolidateTool, MemoryGetTool, MemorySearchTool, MemoryWriteTool};
use ff_tools::notebook::{KernelSupervisor, NotebookKernelState, NotebookTool};
use ff_tools::process::{ProcessManagerTool, ProcessSupervisor};
use ff_tools::{Safety, ToolRegistry};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::oneshot;

/// Registry of in-flight turn cancellation tokens and tool-approval prompts, kept
/// behind a single lock so a cancel and an approval registration can never race
/// (TOCTOU): `register_approval` checks for a live cancel token and inserts the
/// pending slot under the same guard.
#[derive(Default)]
struct ApprovalRegistry {
    /// session_id -> live turn cancellation token. Present only while a turn runs.
    cancels: HashMap<String, CancelToken>,
    /// (session_id, call_id) -> pending UI approval. Keyed by both so colliding
    /// LLM-supplied `call_id`s across concurrent sessions never overwrite each
    /// other. Removed when the user responds, or dropped (denying the call) when
    /// the turn is cancelled. The sender wakes the awaiting approver.
    pending: HashMap<(String, String), oneshot::Sender<bool>>,
    /// (session_id, call_id) -> pending `ask_user` question (#44). Same keying,
    /// liveness, and cancel-via-drop semantics as `pending`, but carries the typed
    /// answer string instead of an approve/deny bool.
    pending_asks: HashMap<(String, String), oneshot::Sender<String>>,
    /// session_id -> set of tool names approved for this session only (#229).
    session_approved: HashMap<String, HashSet<String>>,
    /// Tools approved globally across all sessions (persisted to tool_permissions.json).
    always_approved: HashSet<String>,
}

/// On-disk shape of `tool_permissions.json` (#229).
#[derive(serde::Serialize, serde::Deserialize)]
struct ToolPermissions {
    always_approved: Vec<String>,
}

impl ApprovalRegistry {
    fn set_session_approve(&mut self, session_id: &str, tool: &str) {
        self.session_approved
            .entry(session_id.to_string())
            .or_default()
            .insert(tool.to_string());
    }

    fn is_session_approved(&self, session_id: &str, tool: &str) -> bool {
        self.session_approved
            .get(session_id)
            .is_some_and(|s| s.contains(tool))
    }

    fn set_always_approve(&mut self, tool: &str) {
        self.always_approved.insert(tool.to_string());
    }

    fn remove_always_approve(&mut self, tool: &str) {
        self.always_approved.remove(tool);
    }

    fn is_always_approved(&self, tool: &str) -> bool {
        self.always_approved.contains(tool)
    }

    fn list_always_approved(&self) -> Vec<String> {
        let mut v: Vec<String> = self.always_approved.iter().cloned().collect();
        v.sort();
        v
    }

    fn load_always_approved(path: &Path) -> HashSet<String> {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<ToolPermissions>(&s).ok())
            .map(|tp| tp.always_approved.into_iter().collect())
            .unwrap_or_default()
    }

    fn save_always_approved(path: &Path, set: &HashSet<String>) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut sorted: Vec<String> = set.iter().cloned().collect();
        sorted.sort();
        let perms = ToolPermissions {
            always_approved: sorted,
        };
        let json = serde_json::to_string_pretty(&perms).map_err(io::Error::other)?;
        // Atomic write: temp file + rename
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &json)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// System prompt for the post-turn LLM title (#671 item 2b).
const TITLE_SYSTEM_PROMPT: &str = "You write a concise title for a conversation. \
Summarize it in 3 to 6 words. Reply with only the title text: no quotes, no \
trailing punctuation, no preamble.";
/// Hard wall-clock ceiling on the title call so a stuck provider never leaves a
/// session mid-flight; the heuristic title stands on timeout.
const TITLE_TIMEOUT: Duration = Duration::from_secs(20);
/// Output-token cap: a title is a handful of words.
const TITLE_MAX_TOKENS: u32 = 32;
/// Per-message char budget when rendering the transcript into the title prompt, so
/// a giant first message/reply cannot balloon the request.
const TITLE_PROMPT_MSG_CHARS: usize = 2000;

/// Render the (short, first-turn) transcript into a compact prompt body for the
/// title model: one `Role: text` line per user/assistant message, each capped at
/// [`TITLE_PROMPT_MSG_CHARS`]. Tool/other roles are skipped — a title summarizes
/// the human-visible exchange.
fn render_title_transcript(history: &[ff_core::Message]) -> String {
    let mut out = String::new();
    for m in history {
        let label = match m.role {
            ff_core::Role::User => "User",
            ff_core::Role::Assistant => "Assistant",
            _ => continue,
        };
        let body = m.content.trim();
        if body.is_empty() {
            continue;
        }
        let capped: String = body.chars().take(TITLE_PROMPT_MSG_CHARS).collect();
        out.push_str(label);
        out.push_str(": ");
        out.push_str(&capped);
        out.push('\n');
    }
    out
}

/// Drain a provider stream into its concatenated text deltas, aborting early if the
/// turn's cancel token trips. Returns `None` on a transport/decode error or cancel,
/// so the caller keeps the heuristic title. Tool-call fragments are ignored — the
/// title request advertises no tools.
async fn collect_stream_text(
    provider: &dyn Provider,
    req: ff_llm::ChatRequest,
    cancel: &CancelToken,
) -> Option<String> {
    use futures_util::StreamExt;
    let mut stream = match provider.chat_stream(req).await {
        Ok(s) => s,
        Err(e) => {
            // `log_kind` rather than `%e`: the full error carries the provider's
            // response body, and this file is written by default (#1118).
            tracing::warn!(error = %e.log_kind(), "title generation stream failed");
            tracing::debug!(error = %e, "title generation stream failed (detail)");
            return None;
        }
    };
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if cancel.is_cancelled() {
            return None;
        }
        match item {
            Ok(chunk) => text.push_str(&chunk.delta),
            Err(e) => {
                tracing::warn!(error = %e.log_kind(), "title generation chunk failed");
                tracing::debug!(error = %e, "title generation chunk failed (detail)");
                return None;
            }
        }
    }
    Some(text)
}

/// Clean a raw model title into a display string, or `None` if nothing usable
/// remains (#671 item 2b). Takes the first non-empty line, strips surrounding
/// quotes/backticks and trailing sentence punctuation, collapses inner whitespace,
/// and caps the length so a runaway response cannot overflow the sidebar.
fn sanitize_generated_title(raw: &str) -> Option<String> {
    const MAX_TITLE_CHARS: usize = 60;
    let line = raw.lines().map(str::trim).find(|l| !l.is_empty())?;
    // Strip wrapping quotes/backticks and any surrounding sentence punctuation
    // together, so a quote nested inside a trailing period (e.g. `"Title".`) is
    // fully removed rather than leaving a stray quote when the period is trimmed.
    let stripped = line.trim_matches(|c: char| c.is_whitespace() || "\"'`.,;:!?".contains(c));
    let collapsed = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    let capped: String = collapsed.chars().take(MAX_TITLE_CHARS).collect();
    let capped = capped.trim().to_string();
    if capped.is_empty() {
        None
    } else {
        Some(capped)
    }
}

/// Builds a fresh [`Provider`] from a [`ProviderConnection`]. Called once per turn
/// so a runtime provider switch takes effect on the next message — there is no
/// shared, mutable provider to swap, only the persisted registry.
fn build_provider(conn: &ProviderConnection, model: &str) -> Box<dyn Provider> {
    let base_url = conn.resolved_base_url().to_string();
    // Per-gateway wire-dialect choices (#375). Resolved once here so the per-turn
    // hot path only carries a `Copy` struct; defaults are no-ops for vanilla
    // OpenAI / candle-vllm / Ollama / LM Studio.
    let dialect = wire_dialect(conn.kind, model);
    // Reasoning depth dial (#394/#395). The per-connection user override now
    // drives it: it both caps SiliconFlow's auto-`max` escalation and bounds
    // Bedrock/Anthropic extended thinking. Medium for pre-#395 registries.
    let effort = conn.reasoning_effort;
    // OpenAI-wire reasoning controls (#394). No-op except for the SiliconFlow
    // gateway; native providers take the effort dial directly below.
    let reasoning = reasoning_control(conn.kind, model, effort);
    // Attachment capabilities are derived from the resolved `(kind, model)` (RFC
    // 0005 §11.3), never a stored connection flag, so a per-session model override
    // is gated by the model actually running. Fail-closed on unknown models.
    // `ff_llm`'s lookup (not `ff_core`'s) so the user's `model-specs.json` override
    // is honoured -- the bundled name matching alone misclassifies flagships that
    // ship vision without advertising it in their id (#1137).
    let vision = ff_llm::model_supports_vision(conn.kind, model);
    let documents = model_supports_documents(conn.kind, model);
    match conn.kind {
        ProviderKind::CandleVllm => Box::new(
            OpenAiProvider::new(base_url, None)
                .with_vision(vision)
                .with_documents(documents)
                .with_dialect(dialect)
                .with_reasoning_control(reasoning)
                // CandleVllm is local (#888): the egress-mismatch warning stays
                // silent even when the phenotype is `egress = local-only`.
                .with_kind(conn.kind),
        ),
        ProviderKind::Ollama => Box::new(
            OllamaProvider::new(base_url)
                .with_vision(vision)
                .with_documents(documents)
                // Per-connection window (#651) wins; the env var stays as a
                // global override for connections that leave it unset.
                .with_num_ctx(conn.num_ctx.map(u64::from).or_else(ollama_num_ctx_from_env))
                // Ollama is local (#888); the native adapter's `with_kind` is a
                // no-op for correctness but kept symmetric with the other arms.
                .with_kind(conn.kind),
        ),
        // Bedrock resolves credentials by auth mode, pulling secret material from the
        // OS keychain here so the provider crate stays keychain-free (#202 PR-2).
        ProviderKind::Bedrock => {
            let region = conn
                .region
                .clone()
                .unwrap_or_else(|| "us-east-1".to_string());
            let id = conn.id.as_str();
            // Auto (the default) resolves to a concrete mode by precedence
            // (API key > profile > IAM keys, #320); explicit modes pin themselves.
            // The probe path (build_provider_for) flows through here too, so it
            // validates the exact credential a run will use.
            let mode = match conn.auth_mode.unwrap_or(BedrockAuth::Auto) {
                BedrockAuth::Auto => resolve_bedrock_auth(conn),
                other => other,
            };
            let creds = match mode {
                BedrockAuth::Auto => unreachable!("Auto resolved above"),
                BedrockAuth::Profile => BedrockCreds::Profile {
                    name: conn
                        .aws_profile
                        .clone()
                        .unwrap_or_else(|| "default".to_string()),
                },
                BedrockAuth::IamKeys => BedrockCreds::IamKeys {
                    access_key_id: conn.access_key_id.clone().unwrap_or_default(),
                    secret_access_key: crate::secrets::get(id, SecretKind::SecretAccessKey)
                        .unwrap_or_default(),
                    session_token: crate::secrets::get(id, SecretKind::SessionToken),
                },
                BedrockAuth::ApiKey => BedrockCreds::ApiKey {
                    token: crate::secrets::get(id, SecretKind::ApiKey).unwrap_or_default(),
                },
            };
            Box::new(
                BedrockProvider::new(region, creds)
                    .with_vision(vision)
                    .with_documents(documents)
                    .with_reasoning_effort(effort)
                    // Bedrock is hosted (#888): the egress-mismatch warning
                    // fires correctly when the phenotype is `egress = local-only`.
                    .with_kind(conn.kind),
            )
        }
        // Hosted OpenAI (-compatible). Bearer key pulled from the keychain here so
        // the provider crate stays keychain-free, mirroring the Bedrock arm (#311).
        ProviderKind::OpenAi => {
            let key = crate::secrets::get(conn.id.as_str(), SecretKind::ApiKey);
            Box::new(
                OpenAiProvider::new(base_url, key)
                    .with_vision(vision)
                    .with_documents(documents)
                    .with_dialect(dialect)
                    .with_reasoning_control(reasoning)
                    // OpenAi is hosted (#888): the egress-mismatch warning
                    // fires correctly when the phenotype is `egress = local-only`.
                    .with_kind(conn.kind),
            )
        }
        // SiliconFlow is OpenAI-compatible; the bearer key is pulled from the OS
        // keychain here so the provider crate stays keychain-free (mirrors Bedrock).
        ProviderKind::SiliconFlow => {
            let key = crate::secrets::get(conn.id.as_str(), SecretKind::ApiKey);
            Box::new(
                OpenAiProvider::new(base_url, key)
                    .with_vision(vision)
                    .with_documents(documents)
                    .with_dialect(dialect)
                    .with_reasoning_control(reasoning)
                    // SiliconFlow is hosted (#888): the egress-mismatch warning
                    // fires correctly when the phenotype is `egress = local-only`.
                    .with_kind(conn.kind),
            )
        }
        // OpenRouter is OpenAI-compatible (#807); bearer key from the OS keychain.
        ProviderKind::OpenRouter => {
            let key = crate::secrets::get(conn.id.as_str(), SecretKind::ApiKey);
            Box::new(
                OpenAiProvider::new(base_url, key)
                    .with_vision(vision)
                    .with_documents(documents)
                    .with_dialect(dialect)
                    .with_reasoning_control(reasoning)
                    // OpenRouter is hosted (#888): the egress-mismatch warning
                    // fires correctly when the phenotype is `egress = local-only`.
                    .with_kind(conn.kind),
            )
        }
    }
}

/// Concrete auth a Bedrock connection in `Auto` mode resolves to, reading live
/// keychain presence (#320). A profile counts only when its name is non-empty;
/// IAM keys count only when both the access key id and the secret access key are
/// present. Pure delegation to [`BedrockAuth::resolve_auto`] for the precedence.
fn resolve_bedrock_auth(conn: &ProviderConnection) -> BedrockAuth {
    let id = conn.id.as_str();
    BedrockAuth::resolve_auto(
        crate::secrets::get(id, SecretKind::ApiKey).is_some(),
        conn.aws_profile.as_deref().is_some_and(|p| !p.is_empty()),
        conn.access_key_id.is_some()
            && crate::secrets::get(id, SecretKind::SecretAccessKey).is_some(),
    )
}

/// The active connection, or the built-in default when the `active` pointer dangles
/// (registry invariants forbid this, but turns must still resolve a provider).
fn active_connection_or_default(registry: &ProviderRegistry) -> ProviderConnection {
    registry.active_connection().cloned().unwrap_or_else(|| {
        ProviderRegistry::default()
            .connections
            .into_iter()
            .next()
            .expect("default registry is non-empty")
    })
}

/// Human-facing label for a bare local kind, used when wrapping a legacy
/// [`ProviderConfig`] as a connection (migration / `with_config`).
fn display_name_for(kind: ProviderKind) -> String {
    match kind {
        ProviderKind::CandleVllm => "candle-vLLM",
        ProviderKind::Ollama => "Ollama",
        ProviderKind::Bedrock => "Amazon Bedrock",
        ProviderKind::OpenAi => "OpenAI",
        ProviderKind::SiliconFlow => "SiliconFlow",
        ProviderKind::OpenRouter => "OpenRouter",
    }
    .to_string()
}

/// Wrap a legacy single [`ProviderConfig`] as a [`ProviderConnection`]. The id is
/// the kind slug so it is stable across migrations.
fn config_to_connection(config: ProviderConfig) -> ProviderConnection {
    ProviderConnection {
        id: config.kind.slug().to_string(),
        kind: config.kind,
        display_name: display_name_for(config.kind),
        vendor: None,
        base_url: config.base_url,
        model: config.model,
        has_key: config.has_key,
        secret_missing: false,
        thinking: config.thinking,
        // Carry the depth dial through migration; a legacy `provider.json` without
        // the field deserializes to Medium (`#[serde(default)]`), same as before.
        reasoning_effort: config.reasoning_effort,
        reasoning_visibility: config.reasoning_visibility,
        warmup_enabled: config.warmup_enabled,
        // Carry the served window through migration (#651); a legacy file without
        // the field deserializes to `None`, same env→probe→default behavior as before.
        num_ctx: config.num_ctx,
        region: None,
        auth_mode: None,
        aws_profile: None,
        access_key_id: None,
        compaction_model: None,
        compaction_budget: None,
        near_budget: None,
    }
}

/// Project a connection back to the legacy [`ProviderConfig`] shape for the
/// `get/set_provider_config` shims kept during the frontend cutover (#126).
fn connection_to_config(conn: &ProviderConnection) -> ProviderConfig {
    ProviderConfig {
        kind: conn.kind,
        base_url: conn.base_url.clone(),
        model: conn.model.clone(),
        has_key: conn.has_key,
        thinking: conn.thinking,
        reasoning_effort: conn.reasoning_effort,
        reasoning_visibility: conn.reasoning_visibility,
        warmup_enabled: conn.warmup_enabled,
        num_ctx: conn.num_ctx,
    }
}

/// The root of FlowForge's persisted config: `<OS config dir>/flowforge`
/// (`~/Library/Application Support/flowforge` on macOS, `~/.config/flowforge` on
/// Linux). Returns `None` when the OS exposes no config dir — settings then stay
/// in-memory for the session. Under `cfg!(test)` this always returns `None` so the
/// test suite can never read or clobber the developer's real config (the same
/// isolation `build_session_store` / `build_scheduled_store` apply to their stores).
pub(crate) fn flowforge_config_dir() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    dirs::config_dir().map(|d| d.join("flowforge"))
}

/// `<config dir>/flowforge/provider.json` — the legacy single-provider file
/// (`~/Library/Application Support` on macOS, `~/.config` on Linux). Still
/// read for one-time migration into the registry, and left in place afterward as a
/// backup. `None` only when the OS exposes no config dir.
fn config_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("provider.json"))
}

/// `<config dir>/flowforge/provider-registry.json` — the persisted connection registry
/// (replaces `provider.json`). `None` only when the OS exposes no config dir, in
/// which case settings stay in-memory for the session.
fn registry_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("provider-registry.json"))
}

/// Build the registry to start from: the saved registry if present, else a one-time
/// migration of a legacy `provider.json` (the saved provider becomes the *active*
/// connection, with the other local vendor seeded keyless + inactive), else the
/// built-in default. Pure and idempotent — persistence happens lazily on the first
/// mutation, so construction (including in tests) never writes to the config dir.
fn load_or_migrate_registry() -> ProviderRegistry {
    load_or_migrate_registry_at(registry_path(), config_path())
}

/// Outcome of reading the persisted registry file: cleanly loaded, genuinely
/// absent, or present-but-unreadable. Distinguishing the last case is what keeps
/// a corrupt or half-written file from silently masquerading as "no config" and
/// wiping the user's connections back to the factory default.
enum RegistryRead {
    Loaded(ProviderRegistry),
    Absent,
    Corrupt,
}

/// Read and parse the registry file without ever destroying data on failure. A
/// file that exists but cannot be read or parsed (e.g. truncated by a crash
/// mid-write) is renamed to a `*.corrupt-<unix>.json` sibling and reported as
/// [`RegistryRead::Corrupt`] so the caller seeds a fresh default without
/// overwriting the preserved bytes.
fn read_registry_file(path: Option<&Path>) -> RegistryRead {
    let Some(path) = path else {
        return RegistryRead::Absent;
    };
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return RegistryRead::Absent,
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e,
                "provider registry unreadable; quarantining and seeding default");
            quarantine_registry(path);
            return RegistryRead::Corrupt;
        }
    };
    // Lenient parse: salvage every connection that still deserializes rather than
    // wiping the user back to the factory default over one bad/forward-incompatible
    // field (#811). Only a payload with zero salvageable connections is quarantined.
    match ProviderRegistry::parse_lenient(&raw) {
        Some(registry) => RegistryRead::Loaded(registry),
        None => {
            tracing::error!(path = %path.display(),
                "provider registry unparseable; quarantining and seeding default");
            quarantine_registry(path);
            RegistryRead::Corrupt
        }
    }
}

/// Current unix timestamp in seconds (shared by quarantine + backup filenames).
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// How many timestamped registry backups to retain before pruning the oldest.
const REGISTRY_BACKUP_RETENTION: usize = 5;

/// Copy the current on-disk registry to `provider-registry.<unix_secs>.bak` before
/// it is overwritten, keeping the newest `retention` backups and pruning older
/// ones. This is a deterministic, built-in recovery net (#509) independent of any
/// leftover-file luck from older builds. Best-effort: any I/O error is logged,
/// never fatal — a failed backup must not block the save it protects. Injectable
/// `now_secs` / `retention` so the test suite (where `registry_path()` is `None`)
/// can exercise the logic in a tempdir.
fn backup_registry_at(path: &Path, now_secs: u64, retention: usize) {
    if !path.exists() {
        return;
    }
    let Some(parent) = path.parent() else {
        return;
    };
    // Copy (not rename) so the live file stays intact for the upcoming write_atomic.
    let backup_name = format!("provider-registry.{now_secs}.bak");
    let backup_path = parent.join(&backup_name);
    if let Err(e) = fs::copy(path, &backup_path) {
        tracing::warn!(path = %path.display(), error = %e,
            "registry backup failed; proceeding with save");
        return;
    }

    // Prune: keep only the newest `retention` .bak files.
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    let mut baks: Vec<(u64, std::path::PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with("provider-registry.") || !name.ends_with(".bak") {
                return None;
            }
            // Extract the timestamp between the dots: "provider-registry.<ts>.bak"
            let ts_str = name
                .strip_prefix("provider-registry.")?
                .strip_suffix(".bak")?;
            let ts: u64 = ts_str.parse().ok()?;
            Some((ts, e.path()))
        })
        .collect();

    if baks.len() <= retention {
        return;
    }
    // Sort descending by timestamp; remove everything past `retention`.
    baks.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
    for (_, stale) in baks.drain(retention..) {
        let _ = fs::remove_file(&stale);
    }
}

/// Preserve an unreadable registry file by renaming it alongside the original
/// rather than letting the next save truncate it. Best-effort: a rename failure
/// is logged but never fatal.
fn quarantine_registry(path: &Path) {
    let unix = now_unix_secs();
    let preserved = path.with_extension(format!("corrupt-{unix}.json"));
    match fs::rename(path, &preserved) {
        Ok(()) => {
            tracing::warn!(preserved = %preserved.display(), "preserved unreadable provider registry")
        }
        Err(e) => tracing::warn!(path = %path.display(), error = %e,
            "could not preserve unreadable provider registry"),
    }
}

/// When the live registry is corrupt or absent, try to recover from the newest
/// parseable backup. Scans the registry's parent dir for `provider-registry.<ts>.bak`
/// files (the same glob that [`backup_registry_at`] writes), tries each in
/// timestamp-descending order, and returns the first one that parses successfully.
/// Returns `None` if no valid backup is found, and the caller should fall through
/// to legacy migration or the default registry.
fn try_recover_from_backup(reg_path: &Path) -> Option<ProviderRegistry> {
    let parent = reg_path.parent()?;
    let entries = fs::read_dir(parent).ok()?;
    let mut baks: Vec<(u64, std::path::PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with("provider-registry.") || !name.ends_with(".bak") {
                return None;
            }
            let ts_str = name
                .strip_prefix("provider-registry.")?
                .strip_suffix(".bak")?;
            let ts: u64 = ts_str.parse().ok()?;
            Some((ts, e.path()))
        })
        .collect();

    if baks.is_empty() {
        return None;
    }

    baks.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));

    for (_, backup_path) in baks {
        if let Ok(raw) = fs::read_to_string(&backup_path) {
            if let Some(registry) = ProviderRegistry::parse_lenient(&raw) {
                tracing::info!(backup = %backup_path.display(),
                    "recovered provider registry from backup");
                return Some(registry);
            }
        }
    }

    None
}

/// Path-injectable core of [`load_or_migrate_registry`] so tests can drive it with
/// tempdir paths instead of the real config dir.
fn load_or_migrate_registry_at(
    reg_path: Option<PathBuf>,
    cfg_path: Option<PathBuf>,
) -> ProviderRegistry {
    let mut registry = match read_registry_file(reg_path.as_deref()) {
        RegistryRead::Loaded(r) => r,
        RegistryRead::Absent => reg_path
            .as_deref()
            .and_then(try_recover_from_backup)
            .unwrap_or_else(|| {
                cfg_path
                    .as_ref()
                    .and_then(|p| fs::read_to_string(p).ok())
                    .and_then(|s| serde_json::from_str::<ProviderConfig>(&s).ok())
                    .map(build_migrated_registry)
                    .unwrap_or_default()
            }),
        RegistryRead::Corrupt => reg_path
            .as_deref()
            .and_then(try_recover_from_backup)
            .unwrap_or_default(),
    };
    registry.migrate();
    registry
}

/// Migrate a legacy single [`ProviderConfig`] into a registry: it becomes the
/// active connection, and the *other* built-in local vendor is added keyless and
/// inactive so the user can still switch (#139 review nit 2).
fn build_migrated_registry(config: ProviderConfig) -> ProviderRegistry {
    let active = config_to_connection(config);
    let mut connections = vec![active.clone()];
    for seed in ProviderRegistry::default().connections {
        if seed.kind != active.kind {
            connections.push(seed);
        }
    }
    ProviderRegistry {
        active: active.id,
        connections,
        // Legacy `provider.json` predates #633; `schema_version: 0` lets the
        // load-path migration flip its local connection's thinking default off.
        schema_version: 0,
    }
}

/// Atomically write `contents` to `path`: write a sibling `.tmp` file, then
/// rename it over the target. Rename is atomic on the same filesystem, so a
/// crash or kill mid-write leaves the previous (valid) file intact instead of a
/// truncated one — the root cause behind config silently resetting to default.
fn write_atomic(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)
}

/// Persists the connection registry. Best-effort: a write failure leaves the
/// in-memory registry authoritative for this session rather than failing the
/// command (mirrors the search-config write path). Written atomically so an
/// interrupted save never corrupts the existing file.
fn save_registry(registry: &ProviderRegistry) {
    let Some(path) = registry_path() else {
        return;
    };
    let Ok(json) = serde_json::to_string_pretty(registry) else {
        return;
    };
    backup_registry_at(&path, now_unix_secs(), REGISTRY_BACKUP_RETENTION);
    if let Err(e) = write_atomic(&path, &json) {
        tracing::warn!(path = %path.display(), error = %e,
            "provider registry save failed; in-memory state authoritative this session");
    }
}

/// `<config dir>/flowforge/search.json` — persisted, non-secret web-search settings.
/// `None` only when the OS exposes no config dir (settings stay in-memory then).
fn search_config_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("search.json"))
}

/// Loads persisted web-search settings, falling back to the default (SearXNG, no
/// endpoint) when the file is missing or unparseable.
fn load_search_config() -> SearchConfig {
    search_config_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persists web-search settings. Best-effort and atomic, like [`save_registry`].
fn save_search_config(config: &SearchConfig) {
    let Some(path) = search_config_path() else {
        return;
    };
    let Ok(json) = serde_json::to_string_pretty(config) else {
        return;
    };
    if let Err(e) = write_atomic(&path, &json) {
        tracing::warn!(path = %path.display(), error = %e,
            "search config save failed; in-memory state authoritative this session");
    }
}

/// Synthetic keychain "connection id" for a search backend's API key (#1010).
/// Search sources are not provider connections, so they get their own `search:*`
/// account namespace, reusing [`SecretKind::ApiKey`] — no `SecretKind` change.
fn search_secret_conn_id(backend: SearchBackend) -> &'static str {
    match backend {
        SearchBackend::Tavily => "search:tavily",
        SearchBackend::SearxNg => "search:searxng",
        SearchBackend::Brave => "search:brave",
        SearchBackend::OpenAiCompatible => "search:openai-compatible",
    }
}

/// Whether a search backend has an API key stored in the OS keychain (#1010).
fn search_backend_has_key(backend: SearchBackend) -> bool {
    crate::secrets::get(search_secret_conn_id(backend), SecretKind::ApiKey).is_some()
}

/// Host [`SearchKeyProvider`](ff_tools::SearchKeyProvider): resolves a search
/// backend's API key from the OS keychain and injects it into the `web_search`
/// tool (which cannot depend on the host `secrets` module). #1010.
struct KeychainSearchKeys;

impl ff_tools::SearchKeyProvider for KeychainSearchKeys {
    fn key_for(&self, backend: SearchBackend) -> Option<String> {
        crate::secrets::get(search_secret_conn_id(backend), SecretKind::ApiKey)
    }
}

/// Host [`SearchUserInfoProvider`](ff_tools::SearchUserInfoProvider): resolves the
/// user email from the search config and injects it into the `PubMedSource` tool
/// (#1021). If no email is configured, anonymous (keyless) mode is used.
struct ConfigUserInfo {
    config: Arc<Mutex<SearchConfig>>,
}

impl ff_tools::SearchUserInfoProvider for ConfigUserInfo {
    fn user_email(&self) -> Option<String> {
        self.config.lock().unwrap().email.clone()
    }
}

/// `<config dir>/flowforge/control.json` — the Control panel's settings blob
/// (#147). `None` only when the OS exposes no config dir.
/// Soft cap on the resolved extra-instructions blob (#1002). The block lands in
/// the volatile system-prompt tail and is re-sent every turn, so an oversized
/// value quietly inflates token cost; past this we warn but still inject.
const MAX_EXTRA_INSTRUCTIONS_BYTES: usize = 32 * 1024;

/// Whether `injectMemory` is enabled in the Control config; defaults `true` to
/// match [`default_control_config`]. Pure over the config blob so it is unit
/// testable without a config dir (which `flowforge_config_dir` denies under test).
fn inject_memory_enabled_from(cfg: &serde_json::Value) -> bool {
    cfg.get("injectMemory")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

/// Resolve the extra prompt content from a Control config blob (#1002): the
/// trimmed `userInstructions` plus the contents of each readable `promptFiles`
/// path. Missing or unreadable files are warned and skipped (never hard-fail),
/// mirroring the other best-effort file reads in this module. Returns `None` when
/// the combined result is empty. The block lands in the volatile system-prompt
/// tail and is re-sent every turn, so past [`MAX_EXTRA_INSTRUCTIONS_BYTES`] we
/// warn but still inject -- honoring the user's explicit instruction rather than
/// truncating it mid-content.
///
/// `promptFiles` entries may use a `{workspace}`/`${workspace}` placeholder
/// (the UI suggests `{workspace}/AGENTS.md`), expanded here against
/// `session_root` — the same substitution ff-mcp applies to server configs
/// (#544). Storing the placeholder rather than an absolute path keeps a synced
/// Control config portable across machines and per-session checkouts.
fn resolve_extra_instructions_from(cfg: &serde_json::Value, session_root: &Path) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(instr) = cfg.get("userInstructions").and_then(|v| v.as_str()) {
        let instr = instr.trim();
        if !instr.is_empty() {
            parts.push(instr.to_string());
        }
    }

    if let Some(files) = cfg.get("promptFiles").and_then(|v| v.as_array()) {
        let root = session_root.to_string_lossy();
        for entry in files {
            let Some(raw) = entry.as_str() else { continue };
            let path = raw
                .replace("${workspace}", &root)
                .replace("{workspace}", &root);
            match fs::read_to_string(&path) {
                Ok(body) => {
                    let body = body.trim();
                    if !body.is_empty() {
                        let label = Path::new(&path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&path);
                        parts.push(format!("### {label}\n{body}"));
                    }
                }
                Err(e) => tracing::warn!(error = %e, path, "prompt file unreadable; skipping"),
            }
        }
    }

    if parts.is_empty() {
        return None;
    }
    let joined = parts.join("\n\n");
    if joined.len() > MAX_EXTRA_INSTRUCTIONS_BYTES {
        tracing::warn!(
            bytes = joined.len(),
            cap = MAX_EXTRA_INSTRUCTIONS_BYTES,
            "extra prompt instructions exceed cap; injecting anyway (inflates every turn)"
        );
    }
    Some(joined)
}

fn control_config_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("control.json"))
}

/// The factory `ControlConfig`, returned on first load before the user has saved
/// anything. The frontend (`lib/control.ts`) owns the config shape, so the backend
/// persists it verbatim as an opaque JSON blob and only bakes in the default here;
/// keeping the value untyped means the two can never drift. Mirrors
/// `CONTROL_DEFAULTS` in `lib/control.ts`.
fn default_control_config() -> serde_json::Value {
    serde_json::json!({
        "defaultMode": "auto",
        "permissionPolicy": {
            "read": "allow",
            "localWrites": "allow",
            "externalChanges": "ask",
            "dangerous": "deny"
        },
        "injectMemory": true,
        "userInstructions": "",
        "promptFiles": [],
        "teammates": [
            {
                "id": "reviewer",
                "name": "Riley Reviewer",
                "slug": "reviewer",
                "description": "Scans diffs and flags risky changes before they land."
            },
            {
                "id": "scribe",
                "name": "Sam Scribe",
                "slug": "scribe",
                "description": "Drafts docs and changelogs from the session."
            }
        ],
        "ui": {
            "accentColor": "#6366f1",
            "logoPath": "",
            "faviconPath": "",
            "contextualGreeting": true
        }
    })
}

/// Loads the persisted Control config, falling back to [`default_control_config`]
/// when the file is missing or unparseable. Opaque blob (see the module note).
fn load_control_config() -> serde_json::Value {
    control_config_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(default_control_config)
}

/// Persists the Control config blob. Best-effort and atomic, like
/// [`save_search_config`].
fn save_control_config(config: &serde_json::Value) {
    let Some(path) = control_config_path() else {
        return;
    };
    let Ok(json) = serde_json::to_string_pretty(config) else {
        return;
    };
    if let Err(e) = write_atomic(&path, &json) {
        tracing::warn!(path = %path.display(), error = %e,
            "control config save failed; in-memory state authoritative this session");
    }
}

/// `~/.config/flowforge/tool_permissions.json` — the persistent tool allowlist (#229).
/// `None` only when the OS exposes no config dir.
fn tool_permissions_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("tool_permissions.json"))
}

/// The on-disk session database (RFC 0012 / #277). `None` if no config dir resolves,
/// in which case the store falls back to `:memory:`.
fn sessions_db_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("flowforge").join("sessions.db"))
}

/// Open the persistent session store, falling back to an ephemeral in-memory store
/// (with a warning) if the path is unavailable or the file cannot be opened — same
/// resilience as the FlushLedger open.
fn build_session_store() -> SessionStore {
    // Tests must never write to the real config dir (see load_or_migrate_registry).
    // An on-disk session db is exercised directly in `session_db_survives_restart`.
    if cfg!(test) {
        // Allow restart-scenario tests to point AppState::with_registry at a
        // pre-populated file-backed db without touching the real config dir.
        if let Ok(path) = std::env::var("FF_TEST_SESSION_DB_PATH") {
            return SessionStore::open(&path).unwrap_or_else(|e| {
                tracing::warn!(error = %e, path = %path, "test session db unavailable; falling back to in-memory");
                SessionStore::new()
            });
        }
        return SessionStore::new();
    }
    let Some(path) = sessions_db_path() else {
        tracing::warn!("no config dir; sessions will not persist across restarts");
        return SessionStore::new();
    };
    SessionStore::open(&path).unwrap_or_else(|e| {
        tracing::warn!(error = %e, path = %path.display(), "session db unavailable; sessions will not persist");
        SessionStore::new()
    })
}

/// `~/.config/flowforge/scheduled.db` — the durable scheduled-task store.
fn scheduled_db_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("flowforge").join("scheduled.db"))
}

/// `~/.flowforge/goals/` — the directory of `<session_id>.json` goal checkpoints
/// (RFC 0020 §5, #715/#716). In tests, a per-process temp dir so goal I/O never
/// touches the real config dir or leaks between test runs.
fn build_goal_store() -> GoalStore {
    if cfg!(test) {
        let dir = std::env::temp_dir().join(format!("ff-goals-test-{}", std::process::id()));
        return GoalStore::new(dir);
    }
    GoalStore::new(ff_core::goal_store_dir())
}

/// Open the scheduled-task store, falling back to an ephemeral in-memory store (with
/// a warning) if the path is unavailable — same resilience as `build_session_store`.
fn build_scheduled_store() -> ScheduledStore {
    let store = build_scheduled_store_inner();
    // Seed the app's built-in tasks (e.g. Memory Organizer) on first run.
    // Idempotent, so it is safe to call on every startup (RFC 0017 §6.3, #544).
    store.seed_builtins();
    store
}

fn build_scheduled_store_inner() -> ScheduledStore {
    if cfg!(test) {
        return ScheduledStore::open_in_memory().expect("in-memory scheduled store");
    }
    let Some(path) = scheduled_db_path() else {
        tracing::warn!("no config dir; scheduled tasks will not persist across restarts");
        return ScheduledStore::open_in_memory().expect("in-memory scheduled store");
    };
    ScheduledStore::open(&path).unwrap_or_else(|e| {
        tracing::warn!(error = %e, path = %path.display(), "scheduled db unavailable; tasks will not persist");
        ScheduledStore::open_in_memory().expect("in-memory scheduled store")
    })
}

/// `~/.config/flowforge/phenotype.json` — the name of the active phenotype, persisted
/// so a switch survives a restart (RFC 0001 §7). Separate from the phenotype
/// *definitions* in `~/.flowforge/phenos/`; this only records which one is active.
fn active_phenotype_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("phenotype.json"))
}

/// The persisted active phenotype name, or `None` if never set / unreadable. Falls
/// back to the built-in default at the call site.
fn load_active_phenotype_name() -> Option<String> {
    let raw = active_phenotype_path().and_then(|p| fs::read_to_string(p).ok())?;
    serde_json::from_str::<ActivePhenotypeFile>(&raw)
        .ok()
        .map(|f| f.active)
}

/// Persist the active phenotype name. Best-effort, like `save_registry`.
fn save_active_phenotype_name(name: &str) {
    let Some(path) = active_phenotype_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = ActivePhenotypeFile {
        active: name.to_string(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&file) {
        let _ = fs::write(path, json);
    }
}

/// On-disk shape of `phenotype.json`.
#[derive(serde::Serialize, serde::Deserialize)]
struct ActivePhenotypeFile {
    active: String,
}

/// `~/.config/flowforge/mode.json` — the default agent autonomy mode (RFC 0011 P2,
/// #265), persisted so the user's choice survives a restart. New sessions with no
/// explicit binding inherit this; the factory value is [`Mode::Auto`].
fn default_mode_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("mode.json"))
}

/// On-disk shape of `mode.json`.
#[derive(serde::Serialize, serde::Deserialize)]
struct DefaultModeFile {
    default: Mode,
}

/// The persisted default mode, or [`Mode::Auto`] if never set / unreadable.
fn load_default_mode() -> Mode {
    default_mode_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|raw| serde_json::from_str::<DefaultModeFile>(&raw).ok())
        .map(|f| f.default)
        .unwrap_or(Mode::Auto)
}

/// Persist the default mode. Best-effort, like `save_active_phenotype_name`.
fn save_default_mode(mode: Mode) {
    let Some(path) = default_mode_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&DefaultModeFile { default: mode }) {
        let _ = fs::write(path, json);
    }
}

// ─── Permission matrix persistence (#699) ────────────────────────────────────

fn permission_matrix_path() -> Option<PathBuf> {
    flowforge_config_dir().map(|d| d.join("permissions.json"))
}

fn load_permission_matrix() -> ff_core::PermissionMatrix {
    let matrix: ff_core::PermissionMatrix = permission_matrix_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    // Surface malformed rule patterns loudly (#768 review nit 2) instead of
    // silently swallowing them — a typo'd deny backstop fails closed at
    // evaluation, but the operator still needs to see why.
    for (i, err) in matrix.validate_rules() {
        tracing::warn!(rule = i, error = %err, "invalid permission rule pattern");
    }
    matrix
}

/// Persist the permission matrix (#702). Best-effort and atomic, like
/// `save_search_config`: a write failure leaves the in-memory matrix authoritative
/// for this session rather than failing the edit.
fn save_permission_matrix(matrix: &ff_core::PermissionMatrix) {
    let Some(path) = permission_matrix_path() else {
        return;
    };
    let Ok(json) = serde_json::to_string_pretty(matrix) else {
        return;
    };
    if let Err(e) = write_atomic(&path, &json) {
        eprintln!("failed to persist permission matrix: {e}");
    }
}

/// `~/.flowforge/phenos`, where phenotype definition TOML files live. Under
/// `cfg!(test)` this points at a per-process temp dir that is never created, so
/// `resolve_phenotype` sees no installed definitions and falls back to the built-in
/// `default` — the suite must never read the developer's real phenotypes (which
/// carry their own model pins and would otherwise leak into resolution, e.g.
/// `unbound_session_resolves_to_active_connection_and_its_model`) (#811).
fn phenotypes_root() -> PathBuf {
    if cfg!(test) {
        return std::env::temp_dir().join(format!("ff-phenos-test-{}", std::process::id()));
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".flowforge")
        .join("phenos")
}

/// The Codon phenotype, bundled verbatim from the committed `docs/examples/codon/`
/// tree (the single source of truth) so the shipped binary and the documented
/// example can never drift.
const CODON_PHENOTYPE_TOML: &str =
    include_str!("../../../../docs/examples/codon/phenos/codon.toml");
/// The RFC 0013 phenotype family, seeded write-if-absent alongside `codon`.
/// `orchestrator` is the factory-active default (see [`initial_phenotype`]);
/// `enclave` carries `egress = "local-only"`.
const ORCHESTRATOR_PHENOTYPE_TOML: &str =
    include_str!("../../../../docs/examples/phenos/orchestrator.toml");
const ERUDITE_PHENOTYPE_TOML: &str = include_str!("../../../../docs/examples/phenos/erudite.toml");
const ENCLAVE_PHENOTYPE_TOML: &str = include_str!("../../../../docs/examples/phenos/enclave.toml");
/// The codegraph skill Codon depends on, bundled from the same example tree.
const CODEGRAPH_SKILL_MD: &str =
    include_str!("../../../../docs/examples/codon/skills/codegraph/SKILL.md");

/// Manual revision of the seed/retire *logic* (as opposed to the bundled
/// content, which [`SEED_FINGERPRINT`] tracks automatically). Bump this when the
/// shape of [`seed_builtin_content_at`] or [`retire_seeded_codegraph_if_unmodified`]
/// changes so a user whose stamp already matches the (unchanged) content re-runs
/// the pass exactly once. A pure change to the bundled `include_str!` sources
/// needs no bump — it alters [`SEED_FINGERPRINT`] on its own.
const SEED_LOGIC_VERSION: &str = "v1";

/// FNV-1a-fold a byte slice into an accumulator (const-context helper for
/// [`SEED_FINGERPRINT`]). Our own hasher, not `std`'s `DefaultHasher`, so the
/// persisted stamp is stable across Rust toolchain updates — `DefaultHasher` is
/// only guaranteed stable *within* a binary, not across toolchains, and a
/// toolchain-driven mismatch would silently re-run the idempotent pass.
const fn fnv1a_mix(mut h: u64, bytes: &[u8]) -> u64 {
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += 1;
    }
    h
}

/// Compile-time fingerprint of the bundled seed content (Codon + the RFC 0013
/// phenotype family + codegraph skill) folded with [`SEED_LOGIC_VERSION`].
/// Persisted to `~/.flowforge/.seed_version` after a successful seed pass; a
/// matching stamp on the next launch short-circuits the whole pass
/// ([`seed_builtin_content`]) — no `exists()`/stat calls, no `mcp.json` read, no
/// dir-walks (#599 item 3). Editing any bundled `include_str!` source, or bumping
/// [`SEED_LOGIC_VERSION`], yields a new fingerprint and re-runs the pass exactly
/// once. Every seeded file MUST be folded in here, or an already-stamped user
/// never receives it on upgrade (the gate would short-circuit on the stale stamp).
const SEED_FINGERPRINT: u64 = {
    let h = 0xcbf2_9ce4_8422_2325u64;
    let h = fnv1a_mix(h, CODON_PHENOTYPE_TOML.as_bytes());
    let h = fnv1a_mix(h, ORCHESTRATOR_PHENOTYPE_TOML.as_bytes());
    let h = fnv1a_mix(h, ERUDITE_PHENOTYPE_TOML.as_bytes());
    let h = fnv1a_mix(h, ENCLAVE_PHENOTYPE_TOML.as_bytes());
    let h = fnv1a_mix(h, CODEGRAPH_SKILL_MD.as_bytes());
    let h = fnv1a_mix(h, SEED_LOGIC_VERSION.as_bytes());
    h
};

/// `~/.flowforge/.seed_version` — the version stamp written after a successful
/// built-in seed pass. A dotfile at the `~/.flowforge` root, deliberately
/// outside `phenos/` and `skills/` so the phenotype and skill loaders never try
/// to parse it; and *inside* `~/.flowforge` (not `~/.config/flowforge`) so
/// deleting `~/.flowforge` to reset also clears the stamp and forces a re-seed.
/// `None` when the home directory cannot be resolved.
#[cfg(not(test))]
fn seed_stamp_path() -> Option<PathBuf> {
    dirs::home_dir().map(|d| d.join(".flowforge").join(".seed_version"))
}

/// Seed the built-in content (the Codon phenotype and the codegraph skill it
/// requires) into the real `~/.flowforge/` tree. Runs once at startup, before the
/// skill watcher spawns and the persisted phenotype resolves, so a user who has
/// selected Codon finds its skill already present. Gated by the on-disk version
/// stamp ([`seed_stamp_path`]): when the stamp matches [`SEED_FINGERPRINT`] the
/// entire pass is skipped, so a steady-state launch does no `exists()`/stat
/// calls and no `mcp.json` read (#599 item 3).
#[cfg(not(test))]
fn seed_builtin_content() {
    seed_builtin_content_gated(
        seed_stamp_path().as_deref(),
        &phenotypes_root(),
        &skills_root(),
        ff_mcp::config_path().as_deref(),
    );
}

/// Gated, path-injectable core of [`seed_builtin_content`]. Compares the on-disk
/// stamp at `stamp_path` to [`SEED_FINGERPRINT`]; on a match the whole pass is
/// skipped. Otherwise runs [`seed_builtin_content_at`] and (best-effort) writes
/// the new stamp so the next launch short-circuits. `stamp_path` `None` (no home
/// dir) skips the gate and always runs, matching the pre-gate behaviour. The
/// stamp is written *after* the pass and only best-effort: a home so read-only
/// that the seed writes were swallowed also rejects the stamp, so the next launch
/// retries — a pass that succeeds but then fails to stamp simply re-runs a
/// no-op seed next launch until the stamp lands.
fn seed_builtin_content_gated(
    stamp_path: Option<&Path>,
    phenotypes_root: &Path,
    skills_root: &Path,
    mcp_path: Option<&Path>,
) {
    if let Some(stamp) = stamp_path {
        if stamp_matches(stamp) {
            tracing::debug!(stamp = %stamp.display(), "seed built-ins: stamp matches, skipping pass");
            return;
        }
    }
    seed_builtin_content_at(phenotypes_root, skills_root, mcp_path);
    if let Some(stamp) = stamp_path {
        write_seed_stamp(stamp);
    }
}

/// True when the stamp at `path` matches the current [`SEED_FINGERPRINT`]. A
/// missing, corrupt, or unreadable stamp is treated as a mismatch (so the pass
/// re-runs) rather than an error — the seed is idempotent, so the only cost of a
/// spurious mismatch is one redundant no-op pass.
fn stamp_matches(path: &Path) -> bool {
    let Ok(raw) = fs::read_to_string(path) else {
        return false;
    };
    raw.trim() == format!("{:016x}", SEED_FINGERPRINT)
}

/// Persist [`SEED_FINGERPRINT`] to `path` so the next launch can short-circuit.
/// Best-effort: a failure (read-only home, racing process) is logged and skipped
/// — the next launch simply re-runs the idempotent pass.
fn write_seed_stamp(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            tracing::warn!(path = %path.display(), error = %e, "seed stamp: create dir");
            return;
        }
    }
    if let Err(e) = fs::write(path, format!("{:016x}\n", SEED_FINGERPRINT)) {
        tracing::warn!(path = %path.display(), error = %e, "seed stamp: write");
    }
}

/// Path-injectable, un-gated core of the seed pass so tests can drive it against
/// a tempdir instead of the real home. Each built-in is written only when absent,
/// leaving a user-edited copy untouched; the codegraph skill body is written at
/// `skills/codegraph/SKILL.md` (the layout [`SkillRegistry`] scans). When an
/// `mcp.json` path is known we retire a previously seeded, unmodified disabled
/// codegraph entry (RFC 0018 C3 #590) -- codegraph now travels with the codon
/// phenotype, not the global file; a user-edited entry is left intact. `None`
/// skips it (no home dir). The startup path gates this behind
/// [`seed_builtin_content_gated`]; tests drive it directly to exercise the
/// writes without the version-stamp short-circuit.
fn seed_builtin_content_at(phenotypes_root: &Path, skills_root: &Path, mcp_path: Option<&Path>) {
    seed_if_absent(&phenotypes_root.join("codon.toml"), CODON_PHENOTYPE_TOML);
    // RFC 0013 phenotype family (write-if-absent; user edits are never clobbered).
    seed_if_absent(
        &phenotypes_root.join("orchestrator.toml"),
        ORCHESTRATOR_PHENOTYPE_TOML,
    );
    seed_if_absent(
        &phenotypes_root.join("erudite.toml"),
        ERUDITE_PHENOTYPE_TOML,
    );
    seed_if_absent(
        &phenotypes_root.join("enclave.toml"),
        ENCLAVE_PHENOTYPE_TOML,
    );
    seed_if_absent(
        &skills_root.join("codegraph").join("SKILL.md"),
        CODEGRAPH_SKILL_MD,
    );
    if let Some(mcp_path) = mcp_path {
        retire_seeded_codegraph_if_unmodified(mcp_path);
    }
}

/// The exact shape the pre-C3 seed wrote for codegraph. A live `mcp.json` entry that
/// matches this is an unmodified seed safe to retire; any difference (a user-set
/// command/args/env, or an enabled entry the user turned on) means the user owns it
/// and it is left alone.
///
/// This canonical shape is *not* covered by [`SEED_FINGERPRINT`] (only the
/// `include_str!` bodies are); changing it without bumping
/// [`SEED_LOGIC_VERSION`] would silently skip the corrected pass for
/// already-stamped users.
fn is_unmodified_codegraph_seed(srv: &McpServerConfig) -> bool {
    srv.id == "codegraph"
        && srv.command == "codegraph"
        && srv.args == ["serve", "--mcp"]
        && srv.env.is_empty()
        && srv.disabled
        && srv.scope == McpScope::Workspace
}

/// Remove the pre-C3 seeded codegraph entry from the global `mcp.json` now that
/// codegraph travels with the codon phenotype (RFC 0018 C3 #590). Removes ONLY an
/// unmodified, disabled seed ([`is_unmodified_codegraph_seed`]); a user-edited entry
/// (e.g. the #573 absolute-`command` workaround) keeps working as a global-tier
/// override. Best-effort: a parse or write failure is logged and skipped so a
/// hand-managed or read-only `mcp.json` is never clobbered or blocks startup.
fn retire_seeded_codegraph_if_unmodified(mcp_path: &Path) {
    let servers = match ff_mcp::load(mcp_path) {
        Ok(servers) => servers,
        Err(e) => {
            tracing::warn!(error = %e, "retire codegraph mcp seed: read existing mcp.json");
            return;
        }
    };
    let is_seed = servers
        .iter()
        .find(|s| s.id == "codegraph")
        .is_some_and(is_unmodified_codegraph_seed);
    if !is_seed {
        return;
    }
    if let Err(e) = ff_mcp::remove(mcp_path, "codegraph") {
        tracing::warn!(error = %e, "retire codegraph mcp seed: write");
    }
}

/// Write `contents` to `path` only if it does not already exist. Best-effort: a
/// failure (read-only home, racing process) is logged and skipped so startup is
/// never blocked -- the affected built-in simply will not appear until a later
/// successful launch.
fn seed_if_absent(path: &Path, contents: &str) {
    if path.exists() {
        return;
    }
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            tracing::warn!(path = %path.display(), error = %e, "seed built-in: create dir");
            return;
        }
    }
    if let Err(e) = fs::write(path, contents) {
        tracing::warn!(path = %path.display(), error = %e, "seed built-in: write");
    }
}

/// Resolve a phenotype by name: the built-in `default`, otherwise a definition from
/// `~/.flowforge/phenos/`. Returns `None` for an unknown name.
fn resolve_phenotype(name: &str) -> Option<Phenotype> {
    if name == DEFAULT_PHENOTYPE {
        return Some(default_phenotype());
    }
    let (mut map, errors) = load_phenotypes(&phenotypes_root());
    for e in &errors {
        tracing::warn!(error = %e, "phenotype load");
    }
    map.remove(name)
}

/// The factory-active phenotype (RFC 0013, revisiting #298): `orchestrator` is the
/// out-of-box default working set, seeded into `~/.flowforge/phenos/` on first run.
const FACTORY_ACTIVE_PHENOTYPE: &str = "orchestrator";

/// First-run phenotype selection. A persisted user choice always wins; otherwise we
/// prefer the factory-active `orchestrator` default (seeded into `~/.flowforge/phenos/`
/// on first run), falling back to the built-in `default` when it isn't installed (e.g.
/// a read-only home where the seed couldn't land). Pure over its inputs so the branch
/// matrix is unit-testable without touching `~/.flowforge`.
fn initial_phenotype(
    persisted: Option<String>,
    resolve: impl Fn(&str) -> Option<Phenotype>,
) -> Phenotype {
    persisted
        .and_then(|n| resolve(n.as_str()))
        .or_else(|| resolve(FACTORY_ACTIVE_PHENOTYPE))
        .unwrap_or_else(default_phenotype)
}

pub struct AppState {
    pub store: Arc<SessionStore>,
    /// Durable scheduled-task store (RFC 0017, #539/#540). Shared (via `Arc`) so a
    /// later headless runner (#542) can read the due set without rebuilding state.
    pub scheduled: Arc<ScheduledStore>,
    /// Durable goal-mode store (RFC 0020, #715/#716): a directory of
    /// `<session_id>.json` checkpoints. `Clone` + cheap (holds only a path), so
    /// the self-continue loop and the `goal_*` IPC commands share it directly.
    pub goals: GoalStore,
    /// Persisted, non-secret LLM provider connection registry (RFC 0005 Phase A).
    /// The active connection drives each turn; snapshotted (never held across an
    /// await) per turn. Mutated by the connection commands and the legacy
    /// `set_provider_config` shim.
    registry: Mutex<ProviderRegistry>,
    /// Persisted, non-secret web-search settings, shared (via `Arc`) with the
    /// registered `web_search` tool so a runtime backend switch is visible without
    /// rebuilding the registry.
    search_config: Arc<Mutex<SearchConfig>>,
    /// Per-session workspace roots are an M3 concern (folder picker). For M2 every
    /// session shares one default root; the field is threaded so the picker is
    /// purely additive later.
    pub workspace_root: PathBuf,
    /// Installed skills, kept current by `_skill_watcher`. Snapshotted per turn
    /// (`skills_snapshot`) so a mid-turn reload never races (RFC 0001 §9).
    skills: SharedRegistry,
    /// Owns the filesystem watcher; dropping it stops hot-reload. `Mutex` keeps
    /// `AppState` `Sync` (the `notify` watcher is `Send` but not `Sync`). `Option`
    /// covers the fallback when the watcher cannot start.
    _skill_watcher: Mutex<Option<SkillWatcher>>,
    /// Turn cancellation tokens + pending approvals under one lock (see
    /// [`ApprovalRegistry`]).
    approvals: Mutex<ApprovalRegistry>,
    /// Session ids with a goal self-continue loop currently running (#716). A
    /// single-flight guard: `try_start_goal_loop` refuses to spawn a second loop
    /// for a session, so `goal_set`/`goal_resume` cannot stack overlapping loops
    /// that would race on the transcript and double-checkpoint.
    goal_loops: Mutex<HashSet<String>>,
    /// Globally active skills, whose bodies are injected into the system prompt
    /// (RFC 0001 §4). A `BTreeSet` keeps the set deduplicated and name-sorted.
    /// Replaced wholesale by `switch_phenotype`; tweaked individually by
    /// `activate_skill`/`deactivate_skill`.
    active_skills: Mutex<BTreeSet<String>>,
    /// The active phenotype, resolved at startup from the persisted pointer (RFC
    /// 0001 §7). Supplies the model and persona overrides for each turn.
    active_phenotype: Mutex<Phenotype>,
    /// The default agent autonomy mode (RFC 0011 P2, #265), loaded at startup from
    /// `mode.json`. A session with no explicit mode binding inherits this.
    default_mode: Mutex<Mode>,
    /// Permission matrix (#699, RFC 0019 §3): Mode × Safety → Allow/Ask/Deny.
    /// Persisted to `permissions.json`; falls back to Default on missing/corrupt file.
    /// Editable at runtime from the Control panel (#702) via [`set_permission_cell`].
    permission_matrix: Mutex<ff_core::PermissionMatrix>,
    /// Per-skill telemetry aggregates (RFC 0001 §8), persisted to
    /// `~/.flowforge/skill_signals.json`. Updated at each turn's start/end; read by
    /// the manual optimize flow's cost estimates.
    signals: Mutex<SignalStore>,
    /// MCP host plumbing (RFC 0003). Populated lazily from Tauri's `setup` closure
    /// (the supervisor needs a live Tokio runtime to spawn its actor) and `None`
    /// when no `mcp.json` is present or the watcher cannot start.
    _mcp_watcher: Mutex<Option<McpConfigWatcher>>,
    /// Owns the git HEAD watcher (#561); dropping it stops live branch sync. `Mutex`
    /// for `Sync` (the `notify` watcher is `Send` but not `Sync`); `None` when the
    /// watcher could not start. Re-pointed per active session by [`align_git_watcher`](Self::align_git_watcher).
    _git_watcher: Mutex<Option<GitHeadWatcher>>,
    mcp: Mutex<Option<SupervisorHandle>>,
    /// Path to the watched `mcp.json`, captured when [`init_mcp_at`](Self::init_mcp_at)
    /// runs. The MCP control commands write back to this exact file so their edits flow
    /// through the same watcher that drives reconcile. `None` until MCP is initialized.
    mcp_config_path: Mutex<Option<PathBuf>>,
    /// Durable agent memory (RFC 0006): the Markdown store at `~/.flowforge/memory`
    /// and its FTS5 recall index, shared (via `Arc`) with the registered
    /// `memory_*` tools. The index is a derived cache rebuilt from the Markdown on
    /// startup and kept current by `_memory_watcher`.
    memory: Arc<Memory>,
    memory_index: Arc<dyn MemoryIndex>,
    /// Durable per-session flush ledger (RFC 0006 §7.2). `None` if its sidecar db
    /// could not be opened — the flush then degrades to off rather than failing.
    flush_ledger: Option<Arc<FlushLedger>>,
    /// Owns the memory filesystem watcher; dropping it stops debounced reindex.
    _memory_watcher: Mutex<Option<MemoryWatcher>>,
    /// App-global table of agent-started background processes (#218), shared (via
    /// `Arc`) with the registered `process_manager` tool so a process started in
    /// one turn can be polled or stopped in a later one. Children are killed when
    /// the last `Arc` drops at app exit.
    process_supervisor: Arc<ProcessSupervisor>,
    /// Receive end of the process lifecycle channel (#873). Populated at
    /// construction (so the supervisor's `Arc` can be shared with the
    /// `process_manager` tool immediately) and consumed once by
    /// `start_process_output_pump`, which spawns a per-process bridge that
    /// forwards live output to the frontend as `process:output` events. Same
    /// `Mutex<Option<…>>` take-once idempotency shape as `observer_events_rx`.
    process_lifecycle_rx: Mutex<Option<UnboundedReceiver<ff_tools::process::ProcessLifecycle>>>,
    /// Persistent Python kernels for the `notebook_runner` tool (#859), one per
    /// session. Long-lived child processes, so they are reaped on session end
    /// (`reap_session_kernels`) and killed when the last `Arc` drops at app exit.
    kernel_supervisor: Arc<KernelSupervisor>,
    /// Session-scoped supervisor of background observers (#891 Phase 1): a
    /// `start / stop / list / reap_session` table that owns one
    /// `ObserverSource` per live observer. The receiver of its
    /// `mpsc::UnboundedReceiver<ObserverEvent>` is held in
    /// [`observer_events_rx`](Self::observer_events_rx) and drained by
    /// `start_observer_pump` on a single long-lived task.
    observer_supervisor: Arc<ObserverSupervisor>,
    /// Per-session set of deferred tools unlocked by `tool_search` (RFC 0024 Layer 1).
    ///
    /// Held on `AppState` rather than rebuilt per turn because admissions must
    /// **outlive the turn**: a tool the model searched for in one turn stays callable
    /// in the next, which is also what keeps the tools block append-only across a
    /// session (§6). Shared as an `Arc` with the `tool_search` tool instance the same
    /// way `observer_supervisor` is shared with the `observer` tool.
    tool_search: Arc<ff_tools::ToolSearchState>,
    /// The receive end of the observer event channel. Populated at
    /// construction (so the supervisor's `Arc` can be shared with the
    /// `observer` tool immediately); consumed once at
    /// `start_observer_pump` time. The `Mutex<Option<…>>` is the same
    /// idempotency shape `init_mcp_at` uses for its supervisor handle.
    observer_events_rx: Mutex<Option<UnboundedReceiver<ObserverEvent>>>,
    /// Short-TTL cache for the Ollama served-window probe (#602), keyed by the
    /// resolved `(connection, model)`. The chip resolves on every render, but the
    /// served window changes only when the model is (re)loaded, so a probe per
    /// resolve would spam `/api/ps`. Entries expire after [`SERVED_WINDOW_TTL`].
    served_window_cache: Mutex<HashMap<(ConnectionId, String), (Instant, ServedWindowProbe)>>,
    /// Cross-turn abstractive summary cache (#757). Survives across turns so
    /// the compaction summarizer can skip redundant work when only a few
    /// messages were appended since the last summary.
    pub compaction_cache: CompactionCache,
    /// Mode-switch markers deferred because a turn was in flight when the user
    /// switched autonomy mode (#1066). Appending the `[system: Mode switched …]`
    /// row directly (#828) mid-turn would interpose it between an assistant
    /// `tool_use` and its `tool_result`, wedging the session (Anthropic 422s a
    /// non-adjacent pair). Keyed by session; drained and persisted once the turn
    /// settles so the marker lands after the completed tool batch.
    deferred_mode_markers: Mutex<HashMap<String, Vec<String>>>,
}

/// How long a probed served window stays fresh before the next resolve re-probes.
const SERVED_WINDOW_TTL: Duration = Duration::from_secs(10);

impl AppState {
    pub fn new() -> Self {
        // Boot trace (#599 item 0): time the provider.json read/migrate.
        let t = std::time::Instant::now();
        let registry = load_or_migrate_registry();
        crate::boot_trace_step("app_state.registry", t.elapsed());
        Self::with_registry(registry)
    }

    pub fn with_registry(registry: ProviderRegistry) -> Self {
        // Seed the bundled built-ins (Codon + codegraph) before the watcher loads
        // the skills dir, so the codegraph skill is present when a persisted Codon
        // phenotype resolves below. Gated by a version stamp at
        // `~/.flowforge/.seed_version`: a steady-state launch whose stamp matches
        // the bundled-content fingerprint skips the whole pass (no `exists()`/stat
        // calls, no `mcp.json` read) — #599 item 3. Gated out of tests, which must
        // not write to the real `~/.flowforge/`; the seed core is exercised
        // directly via tempdirs.
        #[cfg(not(test))]
        {
            let t = std::time::Instant::now();
            seed_builtin_content();
            crate::boot_trace_step("app_state.seed", t.elapsed());
        }
        let t = std::time::Instant::now();
        let (watcher, skills) = load_skills();
        crate::boot_trace_step("app_state.skills", t.elapsed());
        // The installer tools are agent-callable, so they own the skills root and a
        // handle to the shared registry to refresh it on a successful install.
        // Shared so the registered `web_search` tool and `set_search_config` see the
        // same cell; a settings change takes effect on the next call.
        let search_config = Arc::new(Mutex::new(load_search_config()));

        // The four boot stores open independent SQLite files (memory index,
        // flush ledger, session, scheduled), so open them concurrently rather
        // than serially to shorten time-to-ready (#599 item 2). `Memory` itself
        // is I/O-free (a root path plus config), so it is built first and shared
        // by reference into the memory-index and flush-ledger jobs.
        let mem_config = memory_config_from_env();
        let embedder = local_embedder_from_env(&mem_config);
        let decay = mem_config.decay.clone();
        let memory = Arc::new(Memory::with_default_root(mem_config));

        let t = std::time::Instant::now();
        let ((memory_index, memory_watcher), flush_ledger, store, scheduled) = std::thread::scope(
            |s| {
                let mem_job = s.spawn(|| {
                    let t = std::time::Instant::now();
                    let out = open_memory_index(&memory, embedder, decay);
                    crate::boot_trace_step("app_state.memory_fts5", t.elapsed());
                    out
                });
                let flush_job = s.spawn(|| {
                    let t = std::time::Instant::now();
                    let out = FlushLedger::open(memory.root().join("flush.db"))
                        .map(Arc::new)
                        .map_err(|e| {
                            tracing::warn!(error = %e, "flush ledger unavailable; memory flush disabled");
                        })
                        .ok();
                    crate::boot_trace_step("app_state.flush_db", t.elapsed());
                    out
                });
                let session_job = s.spawn(|| {
                    let t = std::time::Instant::now();
                    let out = Arc::new(build_session_store());
                    crate::boot_trace_step("app_state.session_db", t.elapsed());
                    out
                });
                let scheduled_job = s.spawn(|| {
                    let t = std::time::Instant::now();
                    let out = Arc::new(build_scheduled_store());
                    crate::boot_trace_step("app_state.scheduled_db", t.elapsed());
                    out
                });
                (
                    mem_job.join().unwrap(),
                    flush_job.join().unwrap(),
                    session_job.join().unwrap(),
                    scheduled_job.join().unwrap(),
                )
            },
        );
        // #1142: reconcile orphaned rows once at startup, not on every session switch.
        // At this point no turns are in flight, so the sweep is safe for all sessions.
        for s in store.list_sessions() {
            store.reconcile_orphaned_assistant_rows(&s.id, INTERRUPTED_NOTICE);
        }
        crate::boot_trace_step("app_state.stores_parallel", t.elapsed());
        // Build the observer supervisor and its event receiver
        // together so the supervisor's sender side and the pump's
        // receiver side are on the same channel (#891 Phase 1). The
        // observer supervisor also borrows the same
        // `ProcessSupervisor` `process_manager` uses, so the `process`
        // observer kind (Phase 3, #893) can subscribe to a running
        // process's bytes. Build the process supervisor first so the
        // observer can be wired with it in one expression.
        let process_supervisor = Arc::new(ProcessSupervisor::new());
        // #873: install the lifecycle listener now, so every `process_manager`
        // start (from any turn) is bridged to the frontend. Consumed once by
        // `start_process_output_pump`.
        let process_lifecycle_rx = process_supervisor.lifecycle_channel();
        let (observer_supervisor, observer_events_rx) = {
            let (sup, rx) = ObserverSupervisor::new();
            (
                Arc::new(sup.with_process_supervisor(process_supervisor.clone())),
                rx,
            )
        };
        let state = Self {
            store,
            scheduled,
            goals: build_goal_store(),
            registry: Mutex::new(registry),
            search_config,
            workspace_root: default_workspace_root(),
            skills,
            _skill_watcher: Mutex::new(watcher),
            approvals: Mutex::new({
                let mut reg = ApprovalRegistry::default();
                if let Some(path) = tool_permissions_path() {
                    reg.always_approved = ApprovalRegistry::load_always_approved(&path);
                }
                reg
            }),
            goal_loops: Mutex::new(HashSet::new()),
            active_skills: Mutex::new(BTreeSet::new()),
            active_phenotype: Mutex::new(default_phenotype()),
            default_mode: Mutex::new(load_default_mode()),
            permission_matrix: Mutex::new(load_permission_matrix()),
            signals: Mutex::new(load_signals()),
            _mcp_watcher: Mutex::new(None),
            _git_watcher: Mutex::new(None),
            mcp: Mutex::new(None),
            mcp_config_path: Mutex::new(None),
            memory,
            memory_index,
            flush_ledger,
            _memory_watcher: Mutex::new(memory_watcher),
            process_supervisor,
            process_lifecycle_rx: Mutex::new(Some(process_lifecycle_rx)),
            kernel_supervisor: Arc::new(KernelSupervisor::new()),
            observer_supervisor,
            tool_search: Arc::new({
                // Persist the embedding cache so an embed computed once survives a
                // restart (#1138 step 5). `ff-tools` does not choose its own data
                // path, and under `cfg!(test)` there is no path at all, so the cache
                // stays in-process and no test can clobber the real one.
                let state = ff_tools::ToolSearchState::new();
                match flowforge_config_dir() {
                    Some(dir) => state.with_cache_path(dir.join("tool_embeddings.json")),
                    None => state,
                }
            }),
            observer_events_rx: Mutex::new(Some(observer_events_rx)),
            served_window_cache: Mutex::new(HashMap::new()),
            compaction_cache: CompactionCache::new(),
            deferred_mode_markers: Mutex::new(HashMap::new()),
        };
        // Restore the persisted phenotype so its active skills survive a restart.
        // With no persisted choice, prefer the out-of-box `codon` default (#298),
        // falling back to the built-in `default` when codon isn't installed.
        let initial = initial_phenotype(state.persisted_phenotype_name(), resolve_phenotype);
        state.apply_phenotype(initial);
        state
    }

    /// The persisted active phenotype name, if any. Indirection so tests can observe
    /// the load path.
    fn persisted_phenotype_name(&self) -> Option<String> {
        load_active_phenotype_name()
    }

    /// Make `pheno` the active phenotype: replace the active-skill set with its
    /// (registry-validated) skills, recording the resolved phenotype for model and
    /// persona overrides. Unknown skill names are dropped with a warning rather than
    /// failing — the installed set can drift from a phenotype definition.
    fn apply_phenotype(&self, pheno: Phenotype) -> BTreeSet<String> {
        let next = self.resolve_skills(&pheno);
        *self.active_skills.lock().unwrap() = next.clone();
        *self.active_phenotype.lock().unwrap() = pheno;
        next
    }

    /// Resolve a phenotype's declared skills against the installed registry,
    /// dropping (with a warning) any name that is not installed — the installed
    /// set can drift from a phenotype definition, and a missing skill must not
    /// fail the turn. Returns a name-sorted, deduplicated set. Shared by
    /// [`apply_phenotype`](Self::apply_phenotype) (global switch) and the
    /// per-session resolver (#246) so both paths validate identically.
    pub(crate) fn resolve_skills(&self, pheno: &Phenotype) -> BTreeSet<String> {
        let known = self.skills.read().unwrap();
        let mut next = BTreeSet::new();
        for name in &pheno.skills {
            if known.get(name).is_some() {
                next.insert(name.clone());
            } else {
                tracing::warn!(skill = %name, phenotype = %pheno.name, "phenotype names unknown skill; skipping");
            }
        }
        next
    }

    /// MCP server ids declared by `skills` (via their manifests) whose tools are NOT
    /// currently available — the server is either absent from `~/.flowforge/mcp.json`
    /// or present but not `Running` (a `Failed`/`Disabled`/`Restarting`/`Starting`
    /// server advertises no tools).
    ///
    /// A phenotype carries its MCP servers as "DNA" (#235): a skill declares the
    /// servers it needs, and a phenotype that lists the skill expects those servers
    /// available. The first cut *requires* the server to already exist in `mcp.json`
    /// and reports the unavailable ones; it never injects a server definition on
    /// activation (that would mutate the `mcp.json` source-of-truth — a follow-up).
    ///
    /// Name-sorted and deduplicated. Empty only when every required server is present
    /// AND running, or no listed skill requires one. When MCP is uninitialized every
    /// required server is reported (none can be running).
    fn missing_skill_mcp_servers(&self, skills: &BTreeSet<String>) -> Vec<String> {
        let required: BTreeSet<String> = {
            let registry = self.skills.read().unwrap();
            skills
                .iter()
                .filter_map(|name| registry.get(name))
                .flat_map(|skill| skill.manifest.mcp.iter().cloned())
                .collect()
        };
        if required.is_empty() {
            return Vec::new();
        }
        let snapshot = self
            .mcp_handle()
            .map(|handle| handle.status_snapshot())
            .unwrap_or_default();
        Self::unavailable_required_servers(&required, &snapshot)
    }

    /// Pure diff of `required` server ids against a supervisor `snapshot`: an id is
    /// unavailable unless a server with that id is present and in the `Running` state.
    /// Name-sorted and deduplicated (`required` is a `BTreeSet`).
    fn unavailable_required_servers(
        required: &BTreeSet<String>,
        snapshot: &[McpServerStatus],
    ) -> Vec<String> {
        let running: BTreeSet<&str> = snapshot
            .iter()
            .filter(|s| s.state == McpServerState::Running)
            .map(|s| s.id.as_str())
            .collect();
        required
            .iter()
            .filter(|id| !running.contains(id.as_str()))
            .cloned()
            .collect()
    }

    /// Warn (once per server) when an activated phenotype lists skills whose declared
    /// MCP servers are unavailable — absent from `mcp.json` or present but not running
    /// (#235). Best-effort and non-fatal: this must not block activation — the skill's
    /// grep/glob fallbacks still work; only the bridged MCP tools are unavailable until
    /// the user adds/repairs the server. No server is injected.
    fn warn_missing_skill_mcp(&self, phenotype: &str, skills: &BTreeSet<String>) {
        for server in self.missing_skill_mcp_servers(skills) {
            tracing::warn!(
                phenotype = %phenotype,
                server = %server,
                "phenotype skill requires an MCP server whose tools are unavailable (not present in mcp.json, or present but not running); add or repair it to enable them (no server is injected)"
            );
        }
    }

    /// The required-but-unavailable MCP servers for `phenotype_name`'s resolved skills
    /// (#301). Same list [`warn_missing_skill_mcp`](Self::warn_missing_skill_mcp) logs,
    /// surfaced read-only so the command layer (which holds the `AppHandle`) can emit
    /// `phenotype:mcp-unavailable` alongside the warn. Name-sorted and deduplicated;
    /// empty for an unknown phenotype or when every required server is present and
    /// running.
    pub fn unavailable_skill_mcp_servers(&self, phenotype_name: &str) -> Vec<String> {
        match resolve_phenotype(phenotype_name) {
            Some(pheno) => self.missing_skill_mcp_servers(&self.resolve_skills(&pheno)),
            None => Vec::new(),
        }
    }

    /// Run a pre-compaction memory flush if the session is now under context
    /// pressure (RFC 0006 §7.2). Best-effort and silent: it persists durable facts
    /// to memory before older detail is summarized away, and never touches the
    /// visible transcript. No-op when memory is disabled or the flush ledger is
    /// unavailable.
    ///
    /// Cycle policy (v1): flush the first time a session crosses the budget, then at
    /// most once per [`REFLUSH_INTERVAL_MESSAGES`] of further growth while still over
    /// budget. A real auto-compaction trigger (its own work) will later advance the
    /// cycle marker; the estimator/strategy seams in `ff-agent` already accommodate
    /// per-model windows and richer strategies.
    /// Run the post-turn memory flush if the session is over budget and hasn't
    /// flushed recently (ledger-gated). Returns `Some(writes)` when the flush wrote
    /// `writes > 0` durable facts, so the caller can emit `MemoryFlushed` (#991 —
    /// the flush moved off `run_turn`'s critical path, and with it the only
    /// `MemoryFlushed` emission; the host now emits it from the returned count).
    /// `None` when no flush ran, it wrote nothing, or it failed.
    pub async fn maybe_flush_memory(
        &self,
        provider: &dyn Provider,
        registry: &ToolRegistry,
        session_id: &str,
        model: &str,
        cancel: CancelToken,
    ) -> Option<u32> {
        let ledger = self.flush_ledger.as_ref()?;
        if !self.memory.is_enabled() {
            return None;
        }
        let history = self.store.get_messages(session_id);
        let pressure = ProxyTokenEstimator::default().assess(&history, model);
        let message_count = history.len() as u64;
        let last_flush_count = match ledger.last_flush(session_id) {
            Ok(rec) => rec.map(|r| r.message_count),
            Err(e) => {
                tracing::warn!(error = %e, "flush ledger read failed; skipping flush");
                return None;
            }
        };
        if !flush_due(
            pressure,
            message_count,
            last_flush_count,
            DEFAULT_FLUSH_AT_FRACTION,
            REFLUSH_INTERVAL_MESSAGES,
        ) {
            return None;
        }

        let session_root = self.session_root(session_id);
        let flush_clock = std::time::Instant::now();
        let outcome = MemoryFlush
            .compact(CompactionContext {
                provider,
                store: self.store.as_ref(),
                registry,
                root: &session_root,
                session_id,
                model,
                cancel,
            })
            .await;
        let flush_elapsed_ms = flush_clock.elapsed().as_millis() as u64;
        match outcome {
            Ok(o) => {
                tracing::info!(?o, flush_elapsed_ms, session = %session_id, "post-turn memory flush (#993 instrument)");
                if let Err(e) = ledger.record_flush(session_id, message_count, now_ms()) {
                    tracing::warn!(error = %e, "flush ledger write failed");
                }
                // Surface provenance (#283) only when facts were actually written.
                match o {
                    CompactionOutcome::Wrote { writes } if writes > 0 => u32::try_from(writes).ok(),
                    _ => None,
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "memory flush failed");
                None
            }
        }
    }

    /// Regenerate a session's title as a one-line LLM summary after its first turn
    /// (#671 item 2b). The heuristic [`auto_title`] seeds an instant title on the
    /// first user message; this replaces it with a better summary once a reply
    /// exists. Non-blocking and best-effort: any failure, timeout, or cancel leaves
    /// the heuristic title in place and returns `None`.
    ///
    /// Gated to run exactly once per session: it fires only when the transcript
    /// holds a single user message plus at least one assistant reply — i.e. right
    /// after the first turn. A later turn (user message count >= 2) is skipped, so
    /// the title is not re-summarized on every round.
    pub async fn generate_session_title(
        &self,
        provider: &dyn Provider,
        session_id: &str,
        model: &str,
        cancel: CancelToken,
    ) -> Option<String> {
        if cancel.is_cancelled() {
            return None;
        }
        let history = self.store.get_messages(session_id);
        let user_count = history
            .iter()
            .filter(|m| m.role == ff_core::Role::User)
            .count();
        let has_reply = history.iter().any(|m| m.role == ff_core::Role::Assistant);
        // Run once, after the first turn: exactly one user message and a reply.
        if user_count != 1 || !has_reply {
            return None;
        }

        let transcript = render_title_transcript(&history);
        if transcript.trim().is_empty() {
            return None;
        }
        let req = ff_llm::ChatRequest {
            model: model.to_string(),
            messages: vec![
                ff_llm::ChatMessage::text("system", TITLE_SYSTEM_PROMPT),
                ff_llm::ChatMessage::text("user", transcript),
            ],
            tools: Vec::new(),
            thinking: false,
            max_tokens: Some(TITLE_MAX_TOKENS),
            cache_messages: false,
        };

        let raw =
            match tokio::time::timeout(TITLE_TIMEOUT, collect_stream_text(provider, req, &cancel))
                .await
            {
                Ok(Some(text)) => text,
                Ok(None) => return None,
                Err(_) => {
                    tracing::warn!(session = %session_id, "title generation timed out");
                    return None;
                }
            };
        sanitize_generated_title(&raw)
    }

    /// The working directory for `session_id`'s tools this turn. Returns the
    /// session's persisted cwd if one has been set (#200/#279), otherwise the
    /// default [`workspace_root`](Self::workspace_root). A stored path that no
    /// longer resolves to a directory (e.g. deleted between runs) also falls back,
    /// so a restored session never runs its tools against a missing root.
    pub fn session_root(&self, session_id: &str) -> PathBuf {
        self.store
            .session_workspace(session_id)
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .unwrap_or_else(|| self.workspace_root.clone())
    }

    /// Set `session_id`'s working directory, persisted in the session row so it
    /// survives a restart (#279). Called by the `set_session_workspace` command
    /// (#200) and exercised by tests. No-op for an unknown session.
    pub fn set_session_cwd(&self, session_id: &str, dir: PathBuf) {
        self.store
            .set_session_workspace(session_id, Some(dir.display().to_string()));
    }

    /// The shared durable-memory store (RFC 0006), used for per-turn ambient
    /// injection and shared with the registered `memory_*` tools.
    pub fn memory(&self) -> Arc<Memory> {
        self.memory.clone()
    }

    /// The shared recall index, so per-turn ambient injection can skip dormant
    /// chunks (RFC 0007 §M6.1). Same `Arc` the `memory_*` tools hold.
    pub fn index(&self) -> Arc<dyn MemoryIndex> {
        self.memory_index.clone()
    }

    /// The directory installed skills live in.
    pub fn skills_root(&self) -> PathBuf {
        skills_root()
    }

    /// The retained-version history tree for skill evolution (RFC 0001 §8). Kept
    /// outside `skills_root` so the registry never sees version copies as skills.
    pub fn skill_history_root(&self) -> PathBuf {
        skill_history_root()
    }

    /// Re-scan the skills directory into the shared registry. Called after an
    /// install/uninstall so the change is visible without waiting on the watcher.
    pub fn reload_skills(&self) {
        reload_registry(&skills_root(), &self.skills);
    }

    /// Build the per-turn tool registry: built-in tools + MCP-bridged tools from
    /// running servers (RFC 0003 §6). Snapshotted per turn so a hot-reload mid-turn
    /// never races an in-flight tool call — same discipline as skill snapshots.
    pub fn build_tool_registry(&self, session_root: &Path) -> ToolRegistry {
        self.build_tool_registry_with_touch_log(session_root, None)
    }

    /// Build the registry, optionally wiring a [`TouchLog`](ff_memory::TouchLog)
    /// into `memory_write` so a goal loop can settle the memory it touched
    /// against the run's verdict (#1292). Chat turns pass `None`; only the goal
    /// loop reinforces/suppresses on outcome.
    pub fn build_tool_registry_with_touch_log(
        &self,
        session_root: &Path,
        touch_log: Option<ff_memory::TouchLog>,
    ) -> ToolRegistry {
        let mut reg = ToolRegistry::with_defaults();
        reg.register(Box::new(ff_tools::WebSearchTool::with_keys(
            self.search_config.clone(),
            std::sync::Arc::new(KeychainSearchKeys),
        )));
        // #1012: PubMed biomedical search (keyless). Registered unconditionally -- the
        // registry carries every source and `advertised_tools` scopes them per phenotype
        // via `search_sources` (#1011 2b), so registration stays independent of which
        // persona is active. #1021: email injected from search config when set.
        reg.register(Box::new(ff_tools::SearchTool::new(std::sync::Arc::new(
            ff_tools::PubMedSource::with_user_info(std::sync::Arc::new(ConfigUserInfo {
                config: self.search_config.clone(),
            })),
        ))));
        reg.register(Box::new(crate::tools::InstallSkillTool::new(
            skills_root(),
            self.skills.clone(),
        )));
        reg.register(Box::new(crate::tools::UninstallSkillTool::new(
            skills_root(),
            self.skills.clone(),
        )));
        reg.register(Box::new(crate::tools::SearchSkillsTool::new(
            self.skills.clone(),
        )));
        reg.register(Box::new(crate::tools::SkillsTool::new(self.skills.clone())));
        // Durable-memory recall tools (RFC 0006 §6-7). Share the long-lived store
        // and index so a `memory_write` is searchable within the same turn.
        reg.register(Box::new(MemorySearchTool::new(
            self.memory.clone(),
            self.memory_index.clone(),
        )));
        reg.register(Box::new(MemoryGetTool::new(self.memory.clone())));
        let memory_write = MemoryWriteTool::new(self.memory.clone(), self.memory_index.clone());
        let memory_write = match &touch_log {
            Some(log) => memory_write.with_touch_log(log.clone()),
            None => memory_write,
        };
        reg.register(Box::new(memory_write));
        // Background-process control (#218). App-global supervisor injected here so
        // a process started in one turn survives into later turns.
        reg.register(Box::new(ProcessManagerTool::new(
            self.process_supervisor.clone(),
        )));
        // #1039: give test_runner the same supervisor so `background: true` can spawn
        // the suite as a background process and auto-attach a wake observer. Overrides
        // the unit `TestRunnerTool` that `with_defaults` registered (synchronous-only).
        reg.register(Box::new(ff_tools::TestRunnerTool::with_supervisor(
            self.process_supervisor.clone(),
        )));
        // Background observers (#891 Phase 1): session-scoped
        // file/dir watchers (Phase 2/3 add http/process). Registered
        // next to `process_manager` because the two share the
        // "long-lived background resource" shape — same supervisor
        // discipline, same reap path, same id/discriminator tool
        // surface.
        reg.register(Box::new(ObserverTool::new(
            self.observer_supervisor.clone(),
        )));
        // Stateful Python kernel (#859): variables persist across `run_cell`
        // calls; scoped to the session and reaped on session end.
        reg.register(Box::new(NotebookTool::new(self.kernel_supervisor.clone())));
        reg.register(Box::new(MemoryConsolidateTool::new(
            self.memory.clone(),
            self.memory_index.clone(),
        )));
        // Reversible tool-result compaction retrieve (M7.1a, RFC 0016 Tier 1).
        // Shares the live session store so it can read originals stashed at ingest.
        // Also carries the memory index so a successful retrieve records a durable,
        // cross-session reinforcement signal (RFC 0022 §4.3, #1291) and maps the
        // retrieved content to related chunks for search ranking (#1296).
        reg.register(Box::new(
            ff_tools::CompactionRetrieveTool::new(self.store.clone())
                .with_index(self.memory_index.clone()),
        ));
        // Goal-mode completion signal (RFC 0020 §7, #716): a ReadOnly tool the
        // agent calls when the objective is met. Always registered so a goal can
        // complete regardless of which session drives it; a no-op outside a loop.
        reg.register(Box::new(ff_tools::GoalCompleteTool));
        // Bookkeeping counterpart (#1225): records evidence-first steps into the
        // goal's durable ledger. Registered on the same unconditional path as
        // `goal_complete` so the two hosts cannot diverge.
        reg.register(Box::new(ff_tools::GoalStepTool));
        // Bridge MCP tools from the instances this session resolves to (M4.3): every
        // global instance plus the workspace instances rooted at `session_root` (RFC
        // 0018 §4.6). Routing is by instance key, so a concurrent turn on another
        // workspace binds the same tool name to its own instance.
        if let Some(handle) = self.mcp_handle() {
            for tool in ff_mcp::build_bridged_tools(&handle, session_root) {
                reg.register(tool);
            }
        }
        // RFC 0024 Layer 1, and necessarily last: the index snapshots whichever tools
        // opted into deferral, so it has to be built after every other registration.
        // `tool_search` is only registered when something is actually deferred —
        // otherwise it would be a tool that can never return a hit, paying prompt
        // tokens to advertise nothing.
        let index = ff_tools::ToolSearchIndex::from_registry(&reg);
        if !index.is_empty() {
            let mut tool = ff_tools::ToolSearchTool::new(self.tool_search.clone(), index);
            // Phase 2B (#1138): semantic recall, fused with BM25F. Opt-in with the
            // rest of embeddings, and fail-soft — an unreachable server leaves the
            // Phase 2A lexical ranking untouched.
            if let Some((base, model, key)) = local_embedder_from_env(&memory_config_from_env()) {
                let embedder = std::sync::Arc::new(ff_memory::OpenAiEmbedder::with_timeout(
                    base,
                    model.clone(),
                    key,
                    ff_memory::INTERACTIVE_EMBED_TIMEOUT,
                ));
                tool = tool.with_embedder(embedder, model);
            }
            reg.register(Box::new(tool));
        }
        reg
    }

    /// Align the MCP supervisor's live instance set for `session_id`, now rooted at
    /// `root` (RFC 0018 §4.3/§4.5). Each `Workspace`-scoped server (e.g. codegraph) gets
    /// a per-root instance, ref-counted by session; a referenced instance not currently
    /// `Running` is proactively (re)started for this turn -- the fix for a codegraph
    /// parked in `Failed` (#557). Best-effort: a no-op when MCP is uninitialized.
    pub async fn align_session_mcp(&self, session_id: &str, root: &Path) {
        if let Some(sup) = self.mcp_handle() {
            let servers = self.resolve_mcp_servers(session_id);
            sup.align_session(session_id, root.to_path_buf(), servers)
                .await;
        }
    }

    /// Admit the active phenotype's `preheat` tools for `session_id` before the turn
    /// starts (#1179 3B).
    ///
    /// Must run before `ff-agent` reads the unlocked set, or the names land outside
    /// the stable prompt region and cost a re-prefill instead of saving the
    /// `tool_search` round-trip they exist to skip. Mirrors
    /// [`align_session_mcp`](Self::align_session_mcp): per-turn, idempotent
    /// (`preheat` is set union), best-effort.
    ///
    /// Returns what was dropped so the caller can surface it; a silent drop would
    /// make a typo'd phenotype indistinguishable from one that declared nothing.
    pub fn align_session_preheat(
        &self,
        session_id: &str,
        registry: &ToolRegistry,
    ) -> Option<ff_core::events::PhenotypePreheatDroppedEvent> {
        let pheno = self.session_phenotype(session_id);
        if pheno.preheat.is_empty() {
            return None;
        }
        // Takes the caller's registry rather than building its own: both call sites
        // construct one immediately after, and a second build would re-walk the tool
        // set for nothing.
        // Only *deferred* tools can be preheated: a resident tool is already in the
        // block, so "preheating" it would spend budget for nothing and is reported
        // as unknown rather than silently accepted.
        let deferred = registry.deferred_tool_names();
        let plan = ff_tools::resolve_preheat(&pheno.preheat, |name| {
            if !deferred.contains(name) {
                return None;
            }
            let one = std::iter::once(name.to_string()).collect();
            // `scope: None` yields the *full* parameter schema, which
            // `scoped_parameters` can only ever shrink. That makes this an upper
            // bound on resident cost, so the budget errs toward admitting less --
            // never toward letting a preheat list overrun it.
            let schemas = registry.openai_tools_named(&one, None);
            schemas
                .first()
                .and_then(|s| serde_json::to_string(s).ok())
                .map(|s| s.len())
        });
        self.tool_search.preheat(session_id, plan.admit.clone());
        let (mut unknown, mut over_budget) = (Vec::new(), Vec::new());
        for r in &plan.rejected {
            match r {
                ff_tools::PreheatRejection::Unknown { name } => unknown.push(name.clone()),
                ff_tools::PreheatRejection::OverBudget { name, .. } => {
                    over_budget.push(name.clone())
                }
            }
        }
        tracing::debug!(
            session_id,
            phenotype = %pheno.name,
            admitted = plan.admit.len(),
            bytes = plan.bytes,
            unknown = unknown.len(),
            over_budget = over_budget.len(),
            "preheat resolved"
        );
        if unknown.is_empty() && over_budget.is_empty() {
            return None;
        }
        Some(ff_core::events::PhenotypePreheatDroppedEvent {
            phenotype: pheno.name,
            session_id: session_id.to_string(),
            unknown,
            over_budget,
            admitted_bytes: u32::try_from(plan.bytes).unwrap_or(u32::MAX),
        })
    }

    /// Release `session_id`'s references to per-workspace MCP instances, evicting any
    /// whose ref-list empties (RFC 0018 §4.3). Called on session close/delete so a
    /// per-workspace codegraph is reaped once no live session references its path.
    pub async fn release_session_mcp(&self, session_id: &str) {
        if let Some(sup) = self.mcp_handle() {
            sup.release_session(session_id).await;
        }
    }

    /// Start the git HEAD watcher (#561 BE half) and store it. Returns the receiver
    /// that yields a `SessionWorkspace` after each changed branch resolution so the
    /// caller (the Tauri `setup` hook) can forward it as `workspace:branch-changed`.
    /// `None` when the watcher could not start -- live branch sync then degrades to
    /// off rather than failing app startup. Mirrors [`init_mcp`](Self::init_mcp): the
    /// watcher is parked in `AppState` and re-pointed per turn by
    /// [`align_git_watcher`](Self::align_git_watcher).
    pub fn init_git_watcher(
        &self,
    ) -> Option<tokio::sync::mpsc::UnboundedReceiver<SessionWorkspace>> {
        match GitHeadWatcher::spawn() {
            Ok((watcher, rx)) => {
                *self._git_watcher.lock().unwrap() = Some(watcher);
                Some(rx)
            }
            Err(e) => {
                tracing::warn!(error = %e, "git head watcher unavailable");
                None
            }
        }
    }

    /// Aim the git HEAD watcher at the active session's `root`, mirroring
    /// [`align_session_mcp`](Self::align_session_mcp). Idempotent and
    /// best-effort: a no-op when the watcher could not start or `root` is unchanged.
    pub fn align_git_watcher(&self, root: &Path) {
        if let Some(w) = self._git_watcher.lock().unwrap().as_mut() {
            w.re_point(root);
        }
    }

    /// Stop and remove all background processes owned by `session_id` (#218).
    /// Fire-and-forget: the session is already gone from the store; process
    /// cleanup is best-effort and must not block the UI thread.
    pub fn reap_session_processes(&self, session_id: &str) {
        let sup = self.process_supervisor.clone();
        let id = session_id.to_owned();
        // `tauri::async_runtime::spawn` (not bare `tokio::spawn`) so this is safe to
        // call from the synchronous `delete_session` command, which Tauri runs off
        // the reactor on macOS (#117). A bare `tokio::spawn` there panics with "no
        // reactor running", and the unwind through the command FFI boundary takes
        // the whole app down (#471).
        tauri::async_runtime::spawn(async move {
            let n = sup.reap_session(&id).await;
            if n > 0 {
                tracing::info!(session_id = %id, reaped = n, "reaped session processes");
            }
        });
    }

    /// Stop and remove the persistent Python kernel owned by `session_id` (#859).
    /// Same fire-and-forget, off-reactor-safe pattern as
    /// [`reap_session_processes`](Self::reap_session_processes): a kernel is a
    /// long-lived child process, so it must be killed when its session ends.
    pub fn reap_session_kernels(&self, session_id: &str) {
        let sup = self.kernel_supervisor.clone();
        let id = session_id.to_owned();
        tauri::async_runtime::spawn(async move {
            let n = sup.reap_session(&id).await;
            if n > 0 {
                tracing::info!(session_id = %id, reaped = n, "reaped session kernel");
            }
        });
    }

    /// Snapshot a session's `notebook_runner` kernel for the desktop status
    /// panel (#871). Read-only; safe for the panel to poll while a kernel runs.
    pub async fn notebook_snapshot(&self, session_id: &str) -> NotebookKernelState {
        self.kernel_supervisor.snapshot(session_id).await
    }

    /// Stop every kernel in a session (the panel's Stop button, #871).
    /// Idempotent — a session with no kernel is a no-op. Unlike
    /// [`reap_session_kernels`](Self::reap_session_kernels), this is awaited
    /// inline: the caller is an async command already running on the reactor,
    /// and the FE refreshes the snapshot once it resolves.
    pub async fn notebook_stop(&self, session_id: &str, kernel_id: Option<&str>) {
        match kernel_id {
            // Session-wide teardown (FE-1 Stop, back-compat): reap every kernel.
            None => {
                let n = self.kernel_supervisor.reap_session(session_id).await;
                if n > 0 {
                    tracing::info!(session_id = %session_id, reaped = n, "stopped session kernels (panel)");
                }
            }
            // Per-tab Stop (#871 FE-2 / #923): remove just the named kernel.
            Some(id) => match self.kernel_supervisor.stop(session_id, Some(id)).await {
                Ok(msg) => {
                    tracing::info!(session_id = %session_id, kernel_id = %id, "{msg} (panel)")
                }
                // A stale tab (already-gone kernel) is not worth surfacing; the
                // next snapshot simply won't list it.
                Err(e) => {
                    tracing::debug!(session_id = %session_id, kernel_id = %id, error = %e, "notebook_stop(one)")
                }
            },
        }
    }

    /// Restart a session's kernel — stop it and spawn a fresh replacement,
    /// preserving the session mapping with a new kernel id (the panel's Restart
    /// button, #871 FE-2 / #922). `kernel_id` targets a specific kernel when
    /// given, else the session's representative one (forward-compat for Phase 3
    /// multi-kernel). Returns the post-restart snapshot directly so the FE can
    /// render the fresh kernel without a follow-up `notebook_status` round-trip.
    /// The fresh kernel is rooted at the session's workspace, mirroring how the
    /// `notebook_runner` tool resolves its working directory.
    pub async fn notebook_restart(
        &self,
        session_id: &str,
        kernel_id: Option<&str>,
    ) -> Result<NotebookKernelState, String> {
        let dir = self.session_root(session_id);
        self.kernel_supervisor
            .restart(session_id, kernel_id, &dir)
            .await?;
        tracing::info!(session_id = %session_id, "restarted session kernel (panel)");
        Ok(self.kernel_supervisor.snapshot(session_id).await)
    }

    /// Stop and remove all background observers owned by `session_id` (#891
    /// Phase 1). Same fire-and-forget, off-reactor-safe pattern as
    /// [`reap_session_kernels`](Self::reap_session_kernels): a watcher is a
    /// long-lived OS resource (kqueue/inotify fd), so it must be closed when
    /// its session ends. Called from `delete_session` next to
    /// `reap_session_kernels`.
    pub fn reap_session_observers(&self, session_id: &str) {
        let sup = self.observer_supervisor.clone();
        let id = session_id.to_owned();
        tauri::async_runtime::spawn(async move {
            let n = sup.reap_session(&id).await;
            if n > 0 {
                tracing::info!(session_id = %id, reaped = n, "reaped session observers");
            }
        });
    }

    /// Start the single long-lived pump that drains the observer
    /// event channel and turns each event into a wake for the
    /// owning session. Mirrors `start_process_reaper`'s
    /// runtime-enter pattern so it's safe to call from Tauri's
    /// `setup` (which runs off-reactor on macOS, #117).
    ///
    /// Idempotent: the receiver is held in a `Mutex<Option<…>>`
    /// and pulled out once; a second call is a logged no-op rather
    /// than a double-pump. The wake helper lives in lib.rs
    /// (next to `spawn_assistant_turn`).
    pub fn start_observer_pump(self: &Arc<Self>, app: &tauri::AppHandle) {
        let rx = match self.observer_events_rx.lock().unwrap().take() {
            Some(rx) => rx,
            None => {
                tracing::warn!("start_observer_pump called twice; ignoring");
                return;
            }
        };
        let state = self.clone();
        let app = app.clone();
        let rt = tauri::async_runtime::handle();
        let _guard = rt.inner().enter();
        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                crate::wake_session_for_observer(&state, event, &app).await;
            }
            tracing::info!("observer event channel closed; pump exiting");
        });
    }

    /// Start the process-output pump (#873): a single long-lived task that
    /// receives a [`ProcessLifecycle::Started`] for every `process_manager`
    /// start and spawns a per-process bridge
    /// ([`crate::spawn_process_output_bridge`]) forwarding that process's live
    /// output to the frontend as `process:output` events, independently of any
    /// turn. Mirrors [`start_observer_pump`](Self::start_observer_pump):
    /// take-once idempotent, enters the shared runtime itself so it is safe to
    /// call from Tauri's off-reactor `setup` (#117).
    pub fn start_process_output_pump(self: &Arc<Self>, app: &tauri::AppHandle) {
        let rx = match self.process_lifecycle_rx.lock().unwrap().take() {
            Some(rx) => rx,
            None => {
                tracing::warn!("start_process_output_pump called twice; ignoring");
                return;
            }
        };
        let sup = self.process_supervisor.clone();
        let app = app.clone();
        let rt = tauri::async_runtime::handle();
        let _guard = rt.inner().enter();
        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(ff_tools::process::ProcessLifecycle::Started {
                id,
                session_id,
                command,
                rx: chunks,
            }) = rx.recv().await
            {
                tracing::debug!(process_id = id, session_id = %session_id, command = %command, "bridging process output");
                crate::spawn_process_output_bridge(
                    app.clone(),
                    sup.clone(),
                    id,
                    session_id,
                    chunks,
                );
            }
            tracing::info!("process lifecycle channel closed; pump exiting");
        });
    }

    /// Append an observer event to `session_id`'s deferral queue.
    /// Public so the wake helper in `lib.rs` can push without
    /// reaching into the private `observer_supervisor` field.
    pub fn buffer_observer_event(&self, session_id: &str, event: ff_observer::ObserverEvent) {
        self.observer_supervisor.buffer_event(session_id, event);
    }

    /// The just-in-time tool-discovery allow-set (RFC 0024 Layer 1), for handing to a
    /// turn's `ToolContext`. Public for the same reason as
    /// [`buffer_observer_event`](Self::buffer_observer_event).
    pub fn tool_search(&self) -> &Arc<ff_tools::ToolSearchState> {
        &self.tool_search
    }

    /// Take and clear every buffered observer event for
    /// `session_id`. Public for the same reason as
    /// [`buffer_observer_event`](Self::buffer_observer_event).
    pub fn drain_observer_buffer(&self, session_id: &str) -> Vec<ff_observer::ObserverEvent> {
        self.observer_supervisor.drain_buffer(session_id)
    }

    /// Whether `session_id` has any buffered observer wakes — events deferred
    /// because they fired while a turn was in flight, *or* buffered while no
    /// turn was running (e.g. right after an observer registers). Read-only.
    /// The turn-completion tail checks this to decide whether to spawn a drain
    /// turn so buffered wakes are not stranded when no further input follows
    /// (#1095).
    pub fn has_buffered_observer_events(&self, session_id: &str) -> bool {
        self.observer_supervisor.has_buffered(session_id)
    }

    /// The active observers for `session_id`, oldest id first (#1038 M2).
    /// Backs the `list_observers` command / the observer panel.
    pub fn list_observers(&self, session_id: &str) -> Vec<ff_observer::ObserverInfo> {
        self.observer_supervisor.list(session_id)
    }

    /// Stop and remove observer `id` if it belongs to `session_id` (#1038 M2).
    /// Backs the `stop_observer` command (the panel's `[×]`).
    pub async fn stop_observer(
        &self,
        id: ff_observer::ObserverId,
        session_id: &str,
    ) -> Result<String, String> {
        self.observer_supervisor.stop(id, session_id).await
    }

    /// Attach a background observer a tool declared via its `ToolOutcome` (#1039,
    /// epic #954 M3). Tools can't call the supervisor directly (`ff-observer`
    /// depends on `ff-tools`), so `process_manager`/`test_runner` return an
    /// `ObserverIntent` and the host translates it here. Best-effort: an invalid
    /// spec or a hit per-session cap logs a warning and is dropped — it never fails
    /// the originating tool call.
    ///
    /// Attach-window caveat (C3): the process observer subscribes to the
    /// supervisor's broadcast channel, which only delivers output produced *after*
    /// the subscription. Output emitted between the tool returning the intent and
    /// this attach — and any match within it — is not replayed (the ring buffer
    /// backs `poll` snapshots, not late subscribers). This is acceptable for a
    /// best-effort wake source: it watches long-running output where the events of
    /// interest (build done, error, tests finished) land after attach, and a missed
    /// match is followed by the next one. Do not rely on it to capture a match in
    /// that startup window.
    pub fn attach_observer_intent(&self, session_id: &str, intent: ff_tools::ObserverIntent) {
        use ff_observer::source::{ObserverKind, ObserverSpec};
        let kind = match intent.kind {
            ff_tools::ObserverIntentKind::File => ObserverKind::File,
            ff_tools::ObserverIntentKind::Http => ObserverKind::Http,
            ff_tools::ObserverIntentKind::Process => ObserverKind::Process,
        };
        let spec = ObserverSpec {
            label: intent.label,
            kind,
            target: intent.target,
            filter: intent.filter,
            interval_secs: intent.interval_secs,
            // Host-declared observers (tool `wake_on`) are change/process
            // watchers; readiness mode is only reachable via the `observer`
            // tool's explicit `mode: "ready"`. #954 item 4.
            http_mode: ff_observer::source::HttpMode::Change,
        };
        match self.observer_supervisor.start(spec, session_id) {
            Ok(id) => {
                tracing::info!(
                    observer_id = id,
                    session = session_id,
                    "tool auto-attached observer"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, session = session_id, "failed to auto-attach tool observer");
            }
        }
    }

    /// Start a periodic background reaper that drives
    /// [`ProcessSupervisor::reap_idle`] to remove finished processes and stop
    /// abandoned ones (started by the agent but never polled again) across all
    /// sessions. Fire-and-forget: the task lives for the app's lifetime on the
    /// shared Tokio runtime. Like [`init_mcp`](Self::init_mcp), it enters the
    /// runtime itself, so it's safe to call from Tauri's `setup` — which runs
    /// off-reactor on macOS (issue #117).
    pub fn start_process_reaper(&self) {
        let sup = self.process_supervisor.clone();
        // `tokio::spawn` needs an entered reactor; `setup` may not have one.
        let rt = tauri::async_runtime::handle();
        let _guard = rt.inner().enter();
        tokio::spawn(async move {
            // Scan every minute; reap processes unpolled for ten minutes. A
            // finished process is reaped on the first tick regardless of the idle
            // budget; a running one is only stopped once it has gone untouched.
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            let idle = Duration::from_secs(10 * 60);
            loop {
                interval.tick().await;
                let n = sup.reap_idle(idle).await;
                if n > 0 {
                    tracing::info!(reaped = n, "periodic process reaper cleaned up");
                }
            }
        });
    }

    /// Start the MCP host: begin watching `~/.flowforge/mcp.json` and spawn the
    /// lifecycle supervisor. Idempotent — a second call is a no-op so a re-`setup`
    /// can't double-spawn. Best-effort: a missing config dir or watcher failure is
    /// logged and leaves MCP disabled rather than failing app start (RFC 0003 §3,5).
    /// Safe to call from any thread: it enters the shared Tokio runtime itself, so
    /// callers (e.g. Tauri's `setup`, which runs outside an entered reactor on
    /// macOS) need not establish a runtime context.
    pub fn init_mcp(&self) {
        let Some(path) = ff_mcp::config_path() else {
            tracing::warn!("no home dir; mcp host disabled");
            return;
        };
        self.init_mcp_at(path);
    }

    /// Path-injectable core of [`init_mcp`](Self::init_mcp). Owns the runtime-enter
    /// so the supervisor's `tokio::spawn` always has a live reactor regardless of
    /// the calling thread (see issue #117). Separated so tests can drive it with a
    /// tempdir config path from a non-runtime thread.
    fn init_mcp_at(&self, path: PathBuf) {
        if self.mcp.lock().unwrap().is_some() {
            return;
        }
        // The supervisor actor is `tokio::spawn`'d; guarantee an entered reactor
        // regardless of the calling thread's context.
        let rt = tauri::async_runtime::handle();
        let _guard = rt.inner().enter();
        let (watcher, shared, change_rx) = match McpConfigWatcher::spawn(path.clone()) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "mcp config watcher unavailable");
                return;
            }
        };
        let handle =
            ff_mcp::spawn_supervisor(shared, change_rx, ff_mcp::SupervisorConfig::default());
        *self._mcp_watcher.lock().unwrap() = Some(watcher);
        *self.mcp.lock().unwrap() = Some(handle);
        *self.mcp_config_path.lock().unwrap() = Some(path);
    }

    /// Path to the watched `mcp.json`, or `None` before [`init_mcp`](Self::init_mcp)
    /// has run. The MCP control commands write back to this file (its edits then ride
    /// the existing config watcher into a reconcile).
    pub fn mcp_config_path(&self) -> Option<PathBuf> {
        self.mcp_config_path.lock().unwrap().clone()
    }

    /// A clone of the supervisor handle if MCP was successfully initialized; `None`
    /// when [`init_mcp`](Self::init_mcp) hasn't run or the watcher couldn't start.
    pub fn mcp_handle(&self) -> Option<SupervisorHandle> {
        self.mcp.lock().unwrap().clone()
    }

    /// MCP `initialize` guidance for servers that currently have at least one tool
    /// admitted for `session_id`, capped for injection (#1173).
    ///
    /// Gated on *reachability*, because guidance for a server the model cannot call
    /// is pure cost. A server is reachable either way round:
    ///
    /// - **Deferred** (the default -- [`McpServerConfig::defer`](ff_core::McpServerConfig)
    ///   is `Option<bool>` where `None` means deferred): its tools only enter the
    ///   block once `tool_search` admits them, so admission is the signal.
    /// - **Not deferred** (`defer = false`): its tools stand in the block from turn
    ///   one and `admit` is never called for them -- [`ToolSearchState::admit`] has a
    ///   single caller, the `tool_search` hit path. Gating on admission alone would
    ///   therefore suppress the guidance of exactly those servers the operator opted
    ///   into keeping resident, permanently and silently.
    ///
    /// Matching is by [`InstanceKey`] id, never by splitting the bridged
    /// `mcp__<server>__<tool>` name: the bridge sanitises only the tool segment and
    /// leaves the server id verbatim, so an id containing `__` makes that split
    /// ambiguous. Here the server id is compared against the `mcp__<id>__` prefix,
    /// which is exact in the direction that matters.
    ///
    /// Sorted by server id so the stable prompt prefix stays byte-identical across
    /// turns (RFC 0024 §276) regardless of supervisor iteration order.
    pub fn mcp_guidance(&self, session_id: &str) -> Vec<ff_agent::McpGuidance> {
        let Some(handle) = self.mcp_handle() else {
            return Vec::new();
        };
        // Union, so a preheated MCP tool's server guidance is injected too: the
        // model holding a tool whose usage instructions were withheld is the
        // failure #1173 exists to prevent (#1179 3B).
        let admitted = self.tool_search.unlocked(session_id);
        // Servers with at least one standing (non-deferred) tool are reachable
        // without admission. Same snapshot the bridge builds the registry from, so
        // this cannot disagree with what the model actually sees.
        let standing: std::collections::HashSet<String> = handle
            .tools_snapshot()
            .into_iter()
            .filter(|t| !t.info.defer)
            .map(|t| t.key.id.clone())
            .collect();
        let mut guidance: Vec<ff_agent::McpGuidance> = handle
            .instructions_snapshot()
            .into_iter()
            .filter(|(key, _)| {
                ff_agent::server_guidance_is_reachable(&key.id, &standing, &admitted)
            })
            .map(|(key, text)| ff_agent::McpGuidance {
                server: key.id.clone(),
                text,
            })
            .collect();
        guidance.sort_by(|a, b| a.server.cmp(&b.server));
        let (fitted, dropped) = ff_agent::fit_mcp_guidance(&guidance);
        if dropped > 0 {
            tracing::warn!(
                dropped,
                "MCP server guidance exceeded the injection budget; some servers' \
                 guidance was omitted"
            );
        }
        fitted
    }

    /// The full connection registry (clone — callers never hold the lock).
    /// Resolve the fast compaction model for a connection (#756).
    /// Precedence: env `FF_COMPACTION_MODEL` > connection config > None.
    pub fn compaction_model(&self, connection_id: &str) -> Option<String> {
        compaction_model_for(
            self.registry
                .lock()
                .unwrap()
                .connections
                .iter()
                .find(|c| c.id == connection_id),
        )
    }

    /// Resolve the compaction budget for a connection (#756).
    /// Precedence: env `FF_COMPACTION_BUDGET` > connection config > None (= computed).
    pub fn compaction_budget(&self, connection_id: &str) -> Option<u64> {
        if let Some(v) = std::env::var("FF_COMPACTION_BUDGET")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
        {
            return Some(v);
        }
        self.registry
            .lock()
            .unwrap()
            .connections
            .iter()
            .find(|c| c.id == connection_id)
            .and_then(|c| c.compaction_budget)
    }

    /// Resolve the Near-layer verbatim-tail budget for a connection (#1045).
    /// Precedence: env `FF_NEAR_BUDGET` > connection config > None (= default).
    pub fn near_budget(&self, connection_id: &str) -> Option<u64> {
        // Clamp env/config to the floor here too (#1045 finding 3): the settings
        // UI can validate its own input, but env and stored config bypass the UI
        // entirely, so the seam is the only place that catches every path.
        let clamp = |v: u64| v.max(ff_agent::MIN_NEAR_BUDGET_TOKENS);
        if let Some(v) = std::env::var("FF_NEAR_BUDGET")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
        {
            return Some(clamp(v));
        }
        self.registry
            .lock()
            .unwrap()
            .connections
            .iter()
            .find(|c| c.id == connection_id)
            .and_then(|c| c.near_budget)
            .map(clamp)
    }

    pub fn provider_registry(&self) -> ProviderRegistry {
        let mut reg = self.registry.lock().unwrap().clone();
        // Cross-check `has_key` against the live OS keychain. An app rebuild can
        // change the code-signing identity, after which the keychain ACL denies
        // the new binary and secrets written by the old build read back as
        // `None` — so `has_key` still says "key present" while auth fails. Flag
        // those connections so the UI can prompt to re-enter the key instead of
        // failing silently. Read-only probe over the clone; the persisted
        // registry is untouched, so `secret_missing` is never saved as `true`.
        for conn in &mut reg.connections {
            if conn.has_key && crate::secrets::present(conn.id.as_str()).is_empty() {
                conn.secret_missing = true;
            }
        }
        reg
    }

    /// Select the active connection by id; persists. `Err` on an unknown id.
    pub fn set_active_connection(&self, id: &str) -> Result<(), String> {
        let snapshot = {
            let mut reg = self.registry.lock().unwrap();
            reg.set_active(id)?;
            reg.clone()
        };
        save_registry(&snapshot);
        Ok(())
    }

    /// Add or update a connection (keyed by id, deriving one when blank); persists.
    /// Returns the stored connection so the caller sees the resolved id.
    pub fn upsert_connection(&self, conn: ProviderConnection) -> ProviderConnection {
        let (stored, snapshot) = {
            let mut reg = self.registry.lock().unwrap();
            let stored = reg.upsert(conn);
            (stored, reg.clone())
        };
        save_registry(&snapshot);
        stored
    }

    /// Remove a connection by id; persists. `Err` when removing the last one.
    pub fn remove_connection(&self, id: &str) -> Result<(), String> {
        let snapshot = {
            let mut reg = self.registry.lock().unwrap();
            reg.remove(id)?;
            reg.clone()
        };
        save_registry(&snapshot);
        Ok(())
    }

    /// Store a provider secret of `kind` for connection `conn_id` in the OS keychain,
    /// then flip that connection's `has_key` flag and persist. The secret value never
    /// enters the registry — only the coarse flag does. `Err` (without writing the
    /// secret) on an unknown connection id.
    pub fn set_connection_secret(
        &self,
        conn_id: &str,
        kind: SecretKind,
        value: &str,
    ) -> Result<(), String> {
        // Validate the id before any keychain write, but never hold the registry
        // lock across keychain I/O (#202 follow-up).
        {
            let reg = self.registry.lock().unwrap();
            if !reg.connections.iter().any(|c| c.id == conn_id) {
                return Err(format!("unknown connection: {conn_id}"));
            }
        }
        crate::secrets::set(conn_id, kind, value)?;
        let snapshot = {
            let mut reg = self.registry.lock().unwrap();
            if let Some(conn) = reg.connections.iter_mut().find(|c| c.id == conn_id) {
                conn.has_key = true;
            }
            reg.clone()
        };
        save_registry(&snapshot);
        Ok(())
    }

    /// Remove the secret of `kind` for `conn_id`, recompute `has_key` from the
    /// remaining stored secrets, and persist. Idempotent. `Err` on an unknown id.
    pub fn clear_connection_secret(&self, conn_id: &str, kind: SecretKind) -> Result<(), String> {
        // Validate the id before touching the keychain (#202 follow-up, mirrors set).
        {
            let reg = self.registry.lock().unwrap();
            if !reg.connections.iter().any(|c| c.id == conn_id) {
                return Err(format!("unknown connection: {conn_id}"));
            }
        }
        crate::secrets::clear(conn_id, kind)?;
        let has_key = SecretKind::ALL
            .iter()
            .any(|k| crate::secrets::get(conn_id, *k).is_some());
        let snapshot = {
            let mut reg = self.registry.lock().unwrap();
            if let Some(conn) = reg.connections.iter_mut().find(|c| c.id == conn_id) {
                conn.has_key = has_key;
            }
            reg.clone()
        };
        save_registry(&snapshot);
        Ok(())
    }

    /// The secret kinds currently stored for a connection (#320), so the UI can
    /// render Stored/Clear per field instead of off the coarse `has_key` flag.
    /// Presence only — no secret value leaves the backend. `Err` on an unknown id.
    pub fn connection_secret_presence(&self, conn_id: &str) -> Result<Vec<SecretKind>, String> {
        {
            let reg = self.registry.lock().unwrap();
            if !reg.connections.iter().any(|c| c.id == conn_id) {
                return Err(format!("unknown connection: {conn_id}"));
            }
        }
        Ok(crate::secrets::present(conn_id))
    }

    /// The concrete Bedrock auth a connection resolves to right now (#320): the
    /// explicit mode if pinned, otherwise the `Auto` precedence winner against live
    /// keychain presence. `None` for non-Bedrock connections or an unknown id, so
    /// the UI can flag which credential field is actually "active".
    pub fn resolved_bedrock_auth(&self, conn_id: &str) -> Option<BedrockAuth> {
        let conn = {
            let reg = self.registry.lock().unwrap();
            reg.connections.iter().find(|c| c.id == conn_id).cloned()?
        };
        if conn.kind != ProviderKind::Bedrock {
            return None;
        }
        Some(match conn.auth_mode.unwrap_or(BedrockAuth::Auto) {
            BedrockAuth::Auto => resolve_bedrock_auth(&conn),
            other => other,
        })
    }

    /// Current provider settings projected from the active connection (clone —
    /// callers never hold the lock). Legacy shim kept during the FE cutover (#126).
    pub fn provider_config(&self) -> ProviderConfig {
        let reg = self.registry.lock().unwrap();
        connection_to_config(&active_connection_or_default(&reg))
    }

    /// Apply legacy provider settings onto the active connection in place, then
    /// persist. Legacy shim kept during the FE cutover (#126).
    pub fn set_provider_config(&self, config: ProviderConfig) {
        let snapshot = {
            let mut reg = self.registry.lock().unwrap();
            let active_id = reg.active.clone();
            if let Some(conn) = reg.connections.iter_mut().find(|c| c.id == active_id) {
                conn.kind = config.kind;
                conn.base_url = config.base_url;
                conn.model = config.model;
                conn.has_key = config.has_key;
                conn.thinking = config.thinking;
            }
            reg.clone()
        };
        save_registry(&snapshot);
    }

    /// Current web-search settings (clone — callers never hold the lock).
    pub fn search_config(&self) -> SearchConfig {
        let mut config = self.search_config.lock().unwrap().clone();
        // `has_key` is derived from the keychain, never persisted (#1010): the FE
        // shows key-set state without the secret ever entering the config blob.
        config.has_key = search_backend_has_key(config.backend);
        config
    }

    /// Replace and persist web-search settings. Visible to the `web_search` tool on
    /// its next call (they share the `Arc`).
    pub fn set_search_config(&self, config: SearchConfig) {
        save_search_config(&config);
        *self.search_config.lock().unwrap() = config;
    }

    /// Store an API key for a search backend in the OS keychain (#1010). The secret
    /// never enters the persisted `SearchConfig`; `has_key` is derived from presence.
    pub fn set_search_secret(&self, backend: SearchBackend, value: &str) -> Result<(), String> {
        crate::secrets::set(search_secret_conn_id(backend), SecretKind::ApiKey, value)
    }

    /// Clear a search backend's stored API key (#1010).
    pub fn clear_search_secret(&self, backend: SearchBackend) -> Result<(), String> {
        crate::secrets::clear(search_secret_conn_id(backend), SecretKind::ApiKey)
    }

    /// Per-backend key presence for the Settings Search key panel (#1015). Boolean
    /// only — the secret value never leaves the keychain.
    pub fn search_secret_presence(&self) -> Vec<ff_core::SearchSecretPresence> {
        SearchBackend::ALL
            .iter()
            .map(|&backend| ff_core::SearchSecretPresence {
                backend,
                present: search_backend_has_key(backend),
            })
            .collect()
    }

    /// The persisted Control-panel config blob (#147), or the factory default on
    /// first load. Opaque JSON: the frontend owns the shape (see
    /// [`load_control_config`]).
    pub fn control_config(&self) -> serde_json::Value {
        load_control_config()
    }

    /// Persist the Control-panel config blob and echo it back. Read straight from
    /// disk on the next `control_config()`, so no in-memory cache is needed for
    /// this low-frequency settings surface.
    pub fn set_control_config(&self, config: serde_json::Value) -> serde_json::Value {
        save_control_config(&config);
        config
    }

    /// Load the Control config once and resolve both prompt-injection values for a
    /// turn. Avoids the double disk-read of calling `inject_memory_enabled()` and
    /// `resolve_extra_instructions()` separately. `session_root` expands the
    /// `{workspace}` placeholder in any `promptFiles` path.
    pub fn turn_prompt_injection(&self, session_root: &Path) -> (bool, Option<String>) {
        let cfg = self.control_config();
        (
            inject_memory_enabled_from(&cfg),
            resolve_extra_instructions_from(&cfg, session_root),
        )
    }

    /// Build a provider + model snapshot from the active connection for one turn.
    pub fn build_provider(&self) -> (Box<dyn Provider>, String) {
        self.build_provider_for(None, None)
    }

    /// Build a provider + model snapshot for a specific connection (`None` = the
    /// active one) running `model` (`None` = the connection's own default model).
    /// `send_message` passes the *resolved* model so the provider's wire-strip
    /// capabilities match the model actually running (RFC 0005 §11.3); `list_models`
    /// probes a connection with `None`. Returns the provider + the model it runs.
    pub fn build_provider_for(
        &self,
        id: Option<&str>,
        model: Option<&str>,
    ) -> (Box<dyn Provider>, String) {
        let conn = {
            let reg = self.registry.lock().unwrap();
            match id {
                Some(id) => reg.connections.iter().find(|c| c.id == id).cloned(),
                None => reg.active_connection().cloned(),
            }
            .unwrap_or_else(|| active_connection_or_default(&reg))
        };
        let resolved_model = model.unwrap_or(&conn.model).to_string();
        (build_provider(&conn, &resolved_model), resolved_model)
    }

    /// Resolve the `(connection, model)` for a turn using RFC 0005 §11.2 three-tier
    /// precedence: session > phenotype > global. An explicit per-session selection
    /// (set via [`set_session_model`](Self::set_session_model)) wins outright -- it is
    /// already a coherent `(connection, model)` pair. Absent that, this resolves the
    /// phenotype binding over the globally active connection. `model` falls back to the
    /// *resolved connection's* own model, never a foreign tier's, so a phenotype model
    /// override can never ride the wrong endpoint -- the latent bug in RFC 0005 §11.1.
    pub fn resolve_model_selection(&self, session_id: &str) -> ResolvedModel {
        let (connection, model) = if let Some(sel) = self.store.session_model(session_id) {
            (sel.connection, sel.model)
        } else {
            let pheno = self.session_phenotype(session_id);
            let reg = self.registry.lock().unwrap();
            let connection = pheno.provider.clone().unwrap_or_else(|| reg.active.clone());
            let model = pheno.model.clone().unwrap_or_else(|| {
                reg.connections
                    .iter()
                    .find(|c| c.id == connection)
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| active_connection_or_default(&reg).model)
            });
            (connection, model)
        };
        // Derive attachment caps from the resolved `(kind, model)` (RFC 0005 §11.3),
        // single-sourcing them at the resolution point. Fail-closed when the
        // connection dangles (no kind -> no caps), matching the gate's fail-closed.
        let kind = {
            let reg = self.registry.lock().unwrap();
            reg.connections
                .iter()
                .find(|c| c.id == connection)
                .map(|c| c.kind)
        };
        // Override-aware lookup, matching `build_provider` above (#1137).
        let supports_vision = kind.is_some_and(|k| ff_llm::model_supports_vision(k, &model));
        let supports_documents = kind.is_some_and(|k| model_supports_documents(k, &model));
        // #1023: seed the context window from the model spec so the FE context gauge
        // has a real denominator for *every* provider — not just Ollama. The async
        // `served_window` IPC still overrides this with a probed value where it can
        // (Ollama's `/api/ps`); non-probed providers (Bedrock/Anthropic/global) keep
        // this spec default instead of `None` (which forced the gauge into its
        // misleading estimate path). `model_context_window` is a pure table lookup,
        // so the cheap sync path stays cheap.
        let spec_window = u32::try_from(model_context_window(&model)).unwrap_or(u32::MAX);
        ResolvedModel {
            connection,
            model,
            supports_vision,
            supports_documents,
            context_window: Some(spec_window),
            trained_context_window: None,
            context_window_source: Some(ContextWindowSource::Default),
        }
    }

    /// Probe the served context window for the model `session_id` will run on
    /// (#602), memoized for [`SERVED_WINDOW_TTL`] per `(connection, model)`. The
    /// sync `resolve_model_selection` cannot do this -- the probe is HTTP -- so the
    /// IPC command awaits this and folds the result onto the `ResolvedModel` it
    /// returns. Ollama only; other kinds (and a dangling connection) yield an empty
    /// probe so the chip simply hides the readout.
    pub async fn served_window(&self, session_id: &str) -> ServedWindowProbe {
        let resolved = self.resolve_model_selection(session_id);
        let (kind, base_url, num_ctx) = {
            let reg = self.registry.lock().unwrap();
            match reg.connections.iter().find(|c| c.id == resolved.connection) {
                Some(c) => (c.kind, c.resolved_base_url().to_string(), c.num_ctx),
                None => return ServedWindowProbe::default(),
            }
        };
        if kind != ProviderKind::Ollama {
            return ServedWindowProbe::default();
        }
        let key = (resolved.connection.clone(), resolved.model.clone());
        // Fresh cache hit short-circuits the HTTP round-trip.
        if let Some((at, probe)) = self.served_window_cache.lock().unwrap().get(&key) {
            if at.elapsed() < SERVED_WINDOW_TTL {
                return probe.clone();
            }
        }
        // Probe the same window the turn will request (#651): per-connection value
        // first, else the env override, so the gauge and the served turn agree.
        let provider = OllamaProvider::new(base_url)
            .with_num_ctx(num_ctx.map(u64::from).or_else(ollama_num_ctx_from_env));
        let probe = provider.served_window(&resolved.model).await;
        self.served_window_cache
            .lock()
            .unwrap()
            .insert(key, (Instant::now(), probe.clone()));
        probe
    }

    /// Resolve this session's MCP server set for a turn using RFC 0018 §3.2 three-tier
    /// precedence: session > phenotype > global. The sibling of
    /// [`resolve_model_selection`](Self::resolve_model_selection) -- one mental model
    /// for both. Composition is **whole-record override-by-id** (RFC §11.5): a later
    /// tier's entry replaces a lower tier's entry with the same id as a unit (command +
    /// args + env + scope), never a field-level merge. A tier may set `disabled: true`
    /// to suppress an inherited server for this turn; suppressed entries are filtered
    /// out of the resolved set. Insertion order is preserved (global first) for stable
    /// reconcile/bridge ordering.
    pub fn resolve_mcp_servers(&self, session_id: &str) -> Vec<McpServerConfig> {
        let mut resolved: Vec<McpServerConfig> = self
            .mcp_config_path()
            .and_then(|p| ff_mcp::load(&p).ok())
            .unwrap_or_default();

        let overlay = |resolved: &mut Vec<McpServerConfig>, srv: McpServerConfig| match resolved
            .iter_mut()
            .find(|e| e.id == srv.id)
        {
            Some(existing) => *existing = srv,
            None => resolved.push(srv),
        };

        for srv in self.session_phenotype(session_id).mcp_servers {
            overlay(&mut resolved, srv);
        }
        if let Some(session_servers) = self.store.session_mcp_servers(session_id) {
            for srv in session_servers {
                overlay(&mut resolved, srv);
            }
        }

        resolved.retain(|s| !s.disabled);
        resolved
    }

    /// Try to claim the goal-loop slot for a session (#716). Returns `true` if
    /// this caller acquired it (no loop was running), `false` if a loop is
    /// already active — the caller must NOT spawn a second one. Single-flight so
    /// overlapping loops can't race the transcript or double-checkpoint.
    pub fn try_start_goal_loop(&self, session_id: &str) -> bool {
        self.goal_loops
            .lock()
            .unwrap()
            .insert(session_id.to_string())
    }

    /// Release the goal-loop slot when a loop finishes (any terminal stop).
    pub fn end_goal_loop(&self, session_id: &str) {
        self.goal_loops.lock().unwrap().remove(session_id);
    }

    /// Whether a goal loop is currently running for the session.
    pub fn goal_loop_running(&self, session_id: &str) -> bool {
        self.goal_loops.lock().unwrap().contains(session_id)
    }

    /// Queue a mode-switch marker for `session_id` to be persisted once the
    /// in-flight turn settles (#1066). See `deferred_mode_markers`.
    pub fn defer_mode_marker(&self, session_id: &str, marker: String) {
        self.deferred_mode_markers
            .lock()
            .unwrap()
            .entry(session_id.to_string())
            .or_default()
            .push(marker);
    }

    /// Persist and clear any mode-switch markers deferred while a turn was in
    /// flight (#1066). Called at turn settle, after the tool batch is complete,
    /// so the marker lands as a well-formed trailing `Role::User` row rather than
    /// interposed in a `tool_use`/`tool_result` pair. In insertion order.
    /// See [`Self::defer_mode_marker`] and the `deferred_mode_markers` field.
    ///
    /// Gated on [`Self::has_active_turn`]: `edit_message` cancels the running
    /// turn and immediately spawns a successor (re-registering a fresh cancel
    /// token) *before* the cancelled turn reaches its flush. Draining here while
    /// that successor is in flight would re-interpose the marker into the new
    /// turn's `tool_use`/`tool_result` window — exactly the wedge this fix
    /// prevents (#1068). When a turn is active we leave the queue intact so the
    /// successor flushes it at its own settle; the marker is never lost.
    pub fn flush_deferred_mode_markers(&self, session_id: &str) {
        if self.has_active_turn(session_id) {
            return;
        }
        let markers = self
            .deferred_mode_markers
            .lock()
            .unwrap()
            .remove(session_id)
            .unwrap_or_default();
        for marker in markers {
            self.store
                .add_message(session_id, ff_core::Role::User, marker);
        }
    }

    pub fn register_cancel(&self, session_id: &str, token: CancelToken) {
        self.approvals
            .lock()
            .unwrap()
            .cancels
            .insert(session_id.to_string(), token);
    }

    pub fn take_cancel(&self, session_id: &str) -> Option<CancelToken> {
        self.approvals.lock().unwrap().cancels.remove(session_id)
    }

    /// Whether a turn is currently running for this session (a live cancel token
    /// is registered). Used to gate the orphaned-row reconciliation so it never
    /// touches a live turn's reserved tail row (#646).
    pub fn has_active_turn(&self, session_id: &str) -> bool {
        self.approvals
            .lock()
            .unwrap()
            .cancels
            .contains_key(session_id)
    }

    /// Remove this session's cancel token only when it is still `expected` — the
    /// token THIS turn registered. A successor turn (the re-run that
    /// `edit_message` spawns after cancelling the in-flight one) may have already
    /// replaced it via `register_cancel`; removing unconditionally would strip
    /// the live turn's token, killing its Stop button and auto-denying its tool
    /// approvals. Identity is by shared flag (`CancelToken::ptr_eq`), not value.
    pub fn take_cancel_if(&self, session_id: &str, expected: &CancelToken) -> Option<CancelToken> {
        let mut reg = self.approvals.lock().unwrap();
        match reg.cancels.get(session_id) {
            Some(tok) if tok.ptr_eq(expected) => reg.cancels.remove(session_id),
            _ => None,
        }
    }

    /// A cheap clone of the current skill set, taken at turn start.
    pub fn skills_snapshot(&self) -> SkillRegistry {
        self.skills.read().unwrap().clone()
    }

    /// The active skill names, name-sorted (BTreeSet order).
    pub fn active_skills(&self) -> Vec<String> {
        self.active_skills.lock().unwrap().iter().cloned().collect()
    }

    /// Record that `skill` was active at the start of a turn (RFC 0001 §8).
    pub fn record_skill_activated(&self, skill: &str) {
        self.signals.lock().unwrap().record_activated(skill);
    }

    /// Fold a finished turn's metrics into `skill`'s aggregate (RFC 0001 §8).
    pub fn record_skill_completed(&self, ev: &SkillCompleted) {
        self.signals.lock().unwrap().record_completed(ev);
    }

    /// The telemetry aggregate for one skill, if any signals have been recorded.
    pub fn skill_telemetry(&self, skill: &str) -> Option<SkillAggregate> {
        self.signals.lock().unwrap().aggregate(skill)
    }

    /// Persist the telemetry aggregates once, at turn end. Snapshots the serialized
    /// payload under the lock, then drops the lock before the synchronous file write
    /// so `get_skill_telemetry` and concurrent turns never block on I/O (addresses
    /// #77 review nit 1). Best-effort: a write failure is logged and ignored.
    pub fn persist_signals(&self) {
        let payload = self.signals.lock().unwrap().snapshot_payload();
        SignalStore::persist_payload(payload);
    }

    /// Add a skill to the active set. Errors if no installed skill has this name
    /// so the UI/agent can't activate a phantom. Idempotent for an already-active
    /// skill.
    pub fn activate_skill(&self, name: &str) -> Result<(), String> {
        if self.skills.read().unwrap().get(name).is_none() {
            return Err(format!("unknown skill: {name}"));
        }
        self.active_skills.lock().unwrap().insert(name.to_string());
        Ok(())
    }

    /// Remove a skill from the active set. Idempotent — deactivating a skill that
    /// isn't active is a no-op.
    pub fn deactivate_skill(&self, name: &str) {
        self.active_skills.lock().unwrap().remove(name);
    }

    /// Drop active entries that are no longer installed (e.g. after an uninstall),
    /// so the active set never names a missing skill. Called after a reload.
    pub fn prune_active_skills(&self) {
        let known: BTreeSet<String> = self
            .skills
            .read()
            .unwrap()
            .names()
            .into_iter()
            .map(str::to_string)
            .collect();
        self.active_skills
            .lock()
            .unwrap()
            .retain(|n| known.contains(n));
    }

    /// All selectable phenotypes: the built-in `default` plus every definition in
    /// `~/.flowforge/phenos/`, name-sorted. Re-scanned per call (few files).
    pub fn list_phenotypes(&self) -> Vec<Phenotype> {
        let (map, errors) = load_phenotypes(&phenotypes_root());
        for e in &errors {
            tracing::warn!(error = %e, "phenotype load");
        }
        let mut out = vec![default_phenotype()];
        out.extend(map.into_values().filter(|p| p.name != DEFAULT_PHENOTYPE));
        out
    }

    /// The active phenotype (clone — never hold the lock).
    pub fn active_phenotype(&self) -> Phenotype {
        self.active_phenotype.lock().unwrap().clone()
    }

    /// Switch to `name`: applies its skills and overrides, then persists the choice.
    /// Errors on an unknown phenotype. Returns the resolved phenotype so the caller
    /// can report what is now active.
    pub fn switch_phenotype(&self, name: &str) -> Result<Phenotype, String> {
        let pheno = resolve_phenotype(name).ok_or_else(|| format!("unknown phenotype: {name}"))?;
        let skills = self.apply_phenotype(pheno.clone());
        self.warn_missing_skill_mcp(&pheno.name, &skills);
        save_active_phenotype_name(&pheno.name);
        Ok(pheno)
    }

    /// Persist an edited phenotype and, when it is the one currently active, re-apply
    /// it so the change takes effect immediately (RFC 0005 Phase D / #525). The
    /// built-in `default` is immutable and rejected by [`save_phenotype`]. A
    /// `provider` binding is validated against the live registry up front -- mirroring
    /// [`set_session_model`](Self::set_session_model) -- so the editor cannot pin a
    /// phantom connection. Returns the saved phenotype.
    pub fn update_phenotype(&self, pheno: Phenotype) -> Result<Phenotype, String> {
        if let Some(ref provider) = pheno.provider {
            let reg = self.registry.lock().unwrap();
            if !reg.connections.iter().any(|c| &c.id == provider) {
                return Err(format!("unknown connection: {provider}"));
            }
        }
        save_phenotype(&phenotypes_root(), &pheno).map_err(|e| e.to_string())?;
        if pheno.name == self.active_phenotype().name {
            let skills = self.apply_phenotype(pheno.clone());
            self.warn_missing_skill_mcp(&pheno.name, &skills);
        }
        Ok(pheno)
    }

    /// The active phenotype's model override, if any (replaces the provider config's
    /// model for the turn).
    pub fn active_model_override(&self) -> Option<String> {
        self.active_phenotype.lock().unwrap().model.clone()
    }

    /// Resolve the phenotype a *session* runs as (#246). A session with an explicit
    /// binding (set via [`set_session_phenotype`](Self::set_session_phenotype)) gets
    /// that phenotype; an unbound session — or one whose bound name no longer exists
    /// — inherits the global active phenotype. This is the single source of truth a
    /// turn uses to derive persona / skills / model / `max_iterations`, so two
    /// panes can run different Phenos simultaneously.
    pub fn session_phenotype(&self, session_id: &str) -> Phenotype {
        match self.store.session_phenotype(session_id) {
            Some(name) => resolve_phenotype(&name).unwrap_or_else(|| {
                tracing::warn!(
                    phenotype = %name,
                    session = %session_id,
                    "session bound to unknown phenotype; inheriting global active"
                );
                self.active_phenotype()
            }),
            None => self.active_phenotype(),
        }
    }

    /// Active skills for a *turn* (#246). Resolution differs from the persona/model
    /// path: only a session with an **explicit** phenotype binding uses that
    /// phenotype's declared skills. An unbound session keeps the global active set
    /// (`active_skills`) so the command palette (`activate_skill` /
    /// `deactivate_skill`) still takes effect on the next turn — otherwise a manual
    /// activation would silently stop reaching the agent while the palette still
    /// showed the skill active.
    pub fn turn_active_skills(&self, session_id: &str) -> Vec<String> {
        if self.store.session_phenotype(session_id).is_some() {
            self.resolve_skills(&self.session_phenotype(session_id))
                .into_iter()
                .collect()
        } else {
            self.active_skills()
        }
    }

    /// Bind `session_id` to a phenotype by name, or clear it (`None`) so the session
    /// inherits the global active one again (#246). Validates the name against the
    /// phenotype registry up front so a stale UI cannot bind a phantom — unknown
    /// names error rather than silently falling back. No-op-safe for the binding
    /// itself: clearing always succeeds.
    pub fn set_session_phenotype(
        &self,
        session_id: &str,
        name: Option<String>,
    ) -> Result<(), String> {
        if let Some(ref n) = name {
            let pheno = resolve_phenotype(n).ok_or_else(|| format!("unknown phenotype: {n}"))?;
            let skills = self.resolve_skills(&pheno);
            self.warn_missing_skill_mcp(&pheno.name, &skills);
        }
        self.store.set_session_phenotype(session_id, name);
        Ok(())
    }

    /// The global default mode (#265). Factory value [`Mode::Auto`].
    pub fn default_mode(&self) -> Mode {
        *self.default_mode.lock().unwrap()
    }

    /// Set and persist the global default mode (#265).
    pub fn set_default_mode(&self, mode: Mode) {
        *self.default_mode.lock().unwrap() = mode;
        save_default_mode(mode);
    }

    /// A snapshot of the permission matrix (#699/#702). Cloned so callers can hold
    /// it across an async turn without pinning the lock.
    pub fn permission_matrix(&self) -> ff_core::PermissionMatrix {
        self.permission_matrix.lock().unwrap().clone()
    }

    /// The Control-panel view of the matrix (#702).
    pub fn permission_matrix_view(&self) -> ff_core::PermissionMatrixView {
        self.permission_matrix.lock().unwrap().view()
    }

    /// Edit and persist a single matrix cell (#702), returning the updated view.
    pub fn set_permission_cell(
        &self,
        mode: Mode,
        safety: ff_core::Safety,
        cell: ff_core::PermissionCell,
    ) -> ff_core::PermissionMatrixView {
        // Clone + drop the guard before the disk write so the hot read path
        // (`permission_matrix()` on the live approval gate) doesn't block on I/O.
        let (view, snapshot) = {
            let mut guard = self.permission_matrix.lock().unwrap();
            guard.set_cell(mode, safety, cell);
            (guard.view(), guard.clone())
        };
        save_permission_matrix(&snapshot);
        view
    }

    /// Set and persist a per-tool override (#700/#702), returning the updated view.
    pub fn set_tool_override(
        &self,
        tool: String,
        cell: ff_core::PermissionCell,
    ) -> ff_core::PermissionMatrixView {
        let (view, snapshot) = {
            let mut guard = self.permission_matrix.lock().unwrap();
            guard.set_override(tool, cell);
            (guard.view(), guard.clone())
        };
        save_permission_matrix(&snapshot);
        view
    }

    /// Remove and persist a per-tool override (#700/#702), returning the updated view.
    pub fn remove_tool_override(&self, tool: &str) -> ff_core::PermissionMatrixView {
        let (view, snapshot) = {
            let mut guard = self.permission_matrix.lock().unwrap();
            guard.remove_override(tool);
            (guard.view(), guard.clone())
        };
        save_permission_matrix(&snapshot);
        view
    }

    #[cfg(test)]
    pub fn set_permission_matrix_for_test(&self, matrix: ff_core::PermissionMatrix) {
        *self.permission_matrix.lock().unwrap() = matrix;
    }

    /// Resolve the mode a turn for `session_id` runs as (#265): an explicit per-pane
    /// binding, else the global [`default_mode`](Self::default_mode). Mirrors
    /// [`session_phenotype`](Self::session_phenotype).
    pub fn session_mode(&self, session_id: &str) -> Mode {
        self.store
            .session_mode(session_id)
            .unwrap_or_else(|| self.default_mode())
    }

    /// Bind `session_id` to a mode, or clear it (`None`) so the session inherits the
    /// global default again (#265).
    pub fn set_session_mode(&self, session_id: &str, mode: Option<Mode>) {
        self.store.set_session_mode(session_id, mode);
    }

    /// The session's explicit per-pane model selection, or `None` if it inherits the
    /// phenotype's model (#499). Raw passthrough; resolution lives in
    /// [`resolve_model_selection`](Self::resolve_model_selection).
    pub fn session_model(&self, session_id: &str) -> Option<ModelSelection> {
        self.store.session_model(session_id)
    }

    /// Bind `session_id` to an explicit `(connection, model)` selection, or clear it
    /// (`None`) so the session inherits its phenotype's model again (#499). Validates
    /// the connection id against the live registry up front so a stale UI cannot pin a
    /// phantom endpoint -- unknown ids error rather than silently riding the wrong
    /// connection. Clearing always succeeds.
    pub fn set_session_model(
        &self,
        session_id: &str,
        selection: Option<ModelSelection>,
    ) -> Result<(), String> {
        if let Some(ref sel) = selection {
            let reg = self.registry.lock().unwrap();
            if !reg.connections.iter().any(|c| c.id == sel.connection) {
                return Err(format!("unknown connection: {}", sel.connection));
            }
        }
        self.store.set_session_model(session_id, selection);
        Ok(())
    }
}

impl AppState {
    /// Reserve a slot for a UI approval prompt. The caller awaits the returned
    /// receiver; the matching `resolve_approval` (or `cancel_pending_approvals` on
    /// cancel) wakes it.
    ///
    /// If the session has no live cancel token — the turn was already cancelled, or
    /// is being torn down — the prompt is refused: the sender is dropped before
    /// returning, so `rx.await` yields `Err`, which the approver treats as a deny.
    /// This closes the TOCTOU where a cancel lands between the agent loop's
    /// cancellation check and this registration, which would otherwise orphan the
    /// sender and hang the awaiting approver.
    pub fn register_approval(&self, session_id: &str, call_id: &str) -> oneshot::Receiver<bool> {
        let (tx, rx) = oneshot::channel();
        let mut reg = self.approvals.lock().unwrap();
        if reg.cancels.contains_key(session_id) {
            reg.pending
                .insert((session_id.to_string(), call_id.to_string()), tx);
        }
        // else: `tx` drops here -> `rx` resolves to `Err` -> deny.
        rx
    }

    /// Deliver the user's decision. An unknown `(session_id, call_id)` is a no-op
    /// (race: cancel raced the click).
    pub fn resolve_approval(&self, session_id: &str, call_id: &str, approved: bool) {
        let key = (session_id.to_string(), call_id.to_string());
        if let Some(tx) = self.approvals.lock().unwrap().pending.remove(&key) {
            // Receiver may have been dropped (cancel raced); ignore the send error.
            let _ = tx.send(approved);
        }
    }

    /// Drop one pending approval whose awaiting approver has given up (#1270).
    ///
    /// `tokio::time::timeout` drops the *receiver*, which leaves this sender parked
    /// in `pending` until the next `cancel_pending_approvals`. Nothing would ever
    /// read it, but a long session that times out repeatedly would accumulate dead
    /// slots, so the approver retires its own entry on expiry.
    pub fn discard_approval(&self, session_id: &str, call_id: &str) {
        let key = (session_id.to_string(), call_id.to_string());
        self.approvals.lock().unwrap().pending.remove(&key);
    }

    /// The `ask_user` counterpart to [`AppState::discard_approval`].
    pub fn discard_ask(&self, session_id: &str, call_id: &str) {
        let key = (session_id.to_string(), call_id.to_string());
        self.approvals.lock().unwrap().pending_asks.remove(&key);
    }

    /// Drop every pending approval AND pending question for this session — dropping
    /// the sender resolves the awaiting future with `Err`, which the approver
    /// translates to a deny / a dismissed question. Called on turn cancel so neither
    /// an open approval nor an open `ask_user` can hang the turn.
    pub fn cancel_pending_approvals(&self, session_id: &str) {
        let mut reg = self.approvals.lock().unwrap();
        reg.pending.retain(|(sid, _), _| sid != session_id);
        reg.pending_asks.retain(|(sid, _), _| sid != session_id);
    }

    /// Clear a session's "Allow this session" allowlist (#229). Called ONLY on
    /// session delete -- never on turn cancel, which would silently revoke a
    /// grant the user expects to last until the session ends.
    pub fn clear_session_approvals(&self, session_id: &str) {
        self.approvals
            .lock()
            .unwrap()
            .session_approved
            .remove(session_id);
    }

    /// Reserve a slot for an `ask_user` question (#44). Mirrors `register_approval`:
    /// the same TOCTOU liveness guard refuses (drops the sender -> `Err` -> `None`
    /// dismissed) when the session has no live turn, so a cancel racing registration
    /// can never orphan the sender.
    pub fn register_ask(&self, session_id: &str, call_id: &str) -> oneshot::Receiver<String> {
        let (tx, rx) = oneshot::channel();
        let mut reg = self.approvals.lock().unwrap();
        if reg.cancels.contains_key(session_id) {
            reg.pending_asks
                .insert((session_id.to_string(), call_id.to_string()), tx);
        }
        // else: `tx` drops here -> `rx` resolves to `Err` -> the question is dismissed.
        rx
    }

    /// Deliver the user's answer to an awaiting `ask_user`. An unknown
    /// `(session_id, call_id)` is a no-op (race: cancel raced the submit).
    pub fn resolve_ask(&self, session_id: &str, call_id: &str, answer: String) {
        let key = (session_id.to_string(), call_id.to_string());
        if let Some(tx) = self.approvals.lock().unwrap().pending_asks.remove(&key) {
            let _ = tx.send(answer);
        }
    }

    // --- Four-option approval (#229) ---

    /// Check if a tool is always-approved (global persistent allowlist).
    pub fn is_always_approved(&self, tool: &str) -> bool {
        self.approvals.lock().unwrap().is_always_approved(tool)
    }

    /// Check if a tool is session-approved for the given session.
    pub fn is_session_approved(&self, session_id: &str, tool: &str) -> bool {
        self.approvals
            .lock()
            .unwrap()
            .is_session_approved(session_id, tool)
    }

    /// Mark a tool as approved for this session only.
    pub fn set_session_approve(&self, session_id: &str, tool: &str) {
        self.approvals
            .lock()
            .unwrap()
            .set_session_approve(session_id, tool);
    }

    /// Add a tool to the global always-approved set and persist.
    pub fn set_always_approve(&self, tool: &str) {
        let set = {
            let mut reg = self.approvals.lock().unwrap();
            reg.set_always_approve(tool);
            reg.always_approved.clone()
        };
        if let Some(path) = tool_permissions_path() {
            if let Err(e) = ApprovalRegistry::save_always_approved(&path, &set) {
                tracing::warn!(error = %e, "failed to persist tool_permissions.json");
            }
        }
    }

    /// Remove a tool from the global always-approved set and persist.
    pub fn remove_always_approve(&self, tool: &str) {
        let set = {
            let mut reg = self.approvals.lock().unwrap();
            reg.remove_always_approve(tool);
            reg.always_approved.clone()
        };
        if let Some(path) = tool_permissions_path() {
            if let Err(e) = ApprovalRegistry::save_always_approved(&path, &set) {
                tracing::warn!(error = %e, "failed to persist tool_permissions.json");
            }
        }
    }

    /// List all globally always-approved tools, sorted.
    pub fn list_always_approved(&self) -> Vec<String> {
        self.approvals.lock().unwrap().list_always_approved()
    }

    /// Whether the #229 allowlist pre-approves this call without a prompt. The
    /// allowlist is keyed by tool name only, so it never covers `Dangerous` or
    /// `Publish` invocations -- those always re-prompt regardless of any "Always
    /// Allow" or "Allow this session" grant on the tool. Publish (`git push`,
    /// `gh pr merge`) is a remote side-effect: Auto must confirm every one, and
    /// Act auto-allows it via the matrix — a session grant must never let a
    /// remote publish fire silently in Auto (#1051).
    /// The caller (`UiApprover`) MUST check `matrix.effective_cell(...).is_deny()`
    /// first — a Deny cell is absolute and cannot be overridden by the allowlist
    /// (#827). This function only answers "is the tool on the list?";
    /// mode-awareness lives in the caller.
    pub fn allowlist_covers(&self, session_id: &str, tool: &str, safety: Safety) -> bool {
        !matches!(safety, Safety::Dangerous | Safety::Publish)
            && (self.is_always_approved(tool) || self.is_session_approved(session_id, tool))
    }
}

/// The default workspace root for M2: the user's home directory, falling back to the
/// process CWD. Replaced by a per-session, user-chosen folder in M3.
/// `~/.flowforge/skills`, the directory the skill watcher loads and watches.
#[cfg(not(test))]
fn skills_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".flowforge")
        .join("skills")
}

/// Test-only: point at a per-process temp directory so test runtime does not
/// depend on the developer's real `~/.flowforge/skills` (#1178).
#[cfg(test)]
fn skills_root() -> PathBuf {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    DIR.get_or_init(|| tempfile::tempdir().expect("temp dir"))
        .path()
        .to_path_buf()
}

/// `~/.flowforge/skill_history`, the retained-version tree for skill evolution (RFC
/// 0001 §8). Deliberately a sibling of `skills/`, not a child: the registry scans
/// every top-level dir under `skills/`, so version copies must live elsewhere.
fn skill_history_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".flowforge")
        .join("skill_history")
}

/// `~/.flowforge/skill_signals.json`, the per-skill telemetry aggregate store (RFC
/// 0001 §8). Lives under `~/.flowforge/` with the skills it describes, not the
/// platform config dir (that's for user *settings* like the provider/phenotype).
fn signals_path() -> Option<PathBuf> {
    dirs::home_dir().map(|d| d.join(".flowforge").join("skill_signals.json"))
}

/// Load the persisted telemetry aggregates, or an in-memory-only store when there is
/// no home dir. Best-effort: a missing/corrupt file starts empty.
fn load_signals() -> SignalStore {
    match signals_path() {
        Some(path) => SignalStore::load(path),
        None => SignalStore::new(),
    }
}

/// Start the skill watcher, falling back to a one-shot load (no hot-reload) if the
/// OS watcher cannot start (e.g. the skills dir does not exist yet).
#[cfg(not(test))]
fn load_skills() -> (Option<SkillWatcher>, SharedRegistry) {
    let root = skills_root();
    match SkillWatcher::spawn(root.clone()) {
        Ok((watcher, shared, errors)) => {
            for e in &errors {
                tracing::warn!(error = %e, "skill load");
            }
            (Some(watcher), shared)
        }
        Err(e) => {
            tracing::warn!(error = %e, "skill watcher unavailable; loading once");
            let (reg, _) = SkillRegistry::load_dir(&root);
            (None, Arc::new(std::sync::RwLock::new(reg)))
        }
    }
}

/// Test-only: skip the ~1.3s FSEvents watcher startup (#1178).
#[cfg(test)]
fn load_skills() -> (Option<SkillWatcher>, SharedRegistry) {
    let root = skills_root();
    let (watcher, shared, errors) =
        SkillWatcher::spawn_without_watcher(root).expect("skill load must succeed in tests");
    for e in &errors {
        tracing::warn!(error = %e, "skill load");
    }
    (Some(watcher), shared)
}

/// How much the transcript must grow (in messages) before a session that is still
/// over budget is flushed again. Keeps a long over-budget session from flushing
/// every turn while still capturing durable facts stated much later (RFC 0006 §7.2).
const REFLUSH_INTERVAL_MESSAGES: u64 = 40;

/// Current wall-clock time as Unix epoch milliseconds.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Build the durable-memory store, its recall index, and a debounced reindex
/// watcher (RFC 0006). Best-effort: the index is a derived cache, so if it can't
/// open on disk we fall back to an in-memory index, and if even that fails the
/// Memory config sourced from the environment. Persisted Settings (the Memory
/// pane, issue #131) is not wired yet, so until it lands semantic recall is opt-in
/// via `FF_MEMORY_EMBEDDINGS=1` (truthy). Everything else keeps `MemoryConfig`
/// defaults: memory on, embeddings off -> pure FTS5/BM25.
fn memory_config_from_env() -> MemoryConfig {
    let mut config = MemoryConfig::default();
    if env_flag("FF_MEMORY_EMBEDDINGS") {
        config.embeddings.enabled = true;
        config.embeddings.provider = EmbeddingProvider::Local;
    }
    config
}

/// Tier-2 abstractive cold-tail summary config from the environment (RFC 0016
/// M7.0). Default-off; opt in with `FF_COMPACT_ABSTRACTIVE`. The summarizer model
/// defaults to the session model and is overridable (same provider/connection)
/// with `FF_COMPACT_ABSTRACTIVE_MODEL`; the fire fraction is tunable via
/// `FF_COMPACT_ABSTRACTIVE_AT` (a 0..1 budget fraction).
pub(crate) fn abstractive_config_from_env() -> AbstractiveConfig {
    let mut config = AbstractiveConfig {
        enabled: env_flag("FF_COMPACT_ABSTRACTIVE"),
        ..AbstractiveConfig::default()
    };
    if let Some(model) = std::env::var("FF_COMPACT_ABSTRACTIVE_MODEL")
        .ok()
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
    {
        config.model = Some(model);
    }
    if let Some(at) = std::env::var("FF_COMPACT_ABSTRACTIVE_AT")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| (0.0..=1.0).contains(v))
    {
        config.fire_at_fraction = at;
    }
    // #972: override the Tier-2 input cap (proxy tokens; 0 = unbounded).
    if let Some(cap) = std::env::var("FF_COMPACT_ABSTRACTIVE_INPUT_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        config.max_summary_input_tokens = cap;
    }
    config
}

/// Fast model for compaction/flush LLM calls (#756). Precedence:
/// 1. `FF_COMPACTION_MODEL` env var (highest — dev override)
/// 2. `compaction_model` on the active `ProviderConnection` (user config)
/// 3. `None` = use session model (legacy)
pub(crate) fn compaction_model_for(
    connection: Option<&ff_core::ProviderConnection>,
) -> Option<String> {
    if let Some(m) = std::env::var("FF_COMPACTION_MODEL")
        .ok()
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
    {
        return Some(m);
    }
    connection.and_then(|c| c.compaction_model.clone())
}

/// Default local embedding endpoint: Ollama's OpenAI-compatible route.
///
/// Ollama is the pragmatic default because it is the most common way a developer
/// already has a local embedding model running, and it needs no new dependency --
/// [`OpenAiEmbedder`](ff_memory::OpenAiEmbedder) is a protocol client, so the same
/// code path serves Ollama, a self-hosted server, or a cloud endpoint.
const DEFAULT_EMBEDDING_BASE_URL: &str = "http://localhost:11434/v1";

/// Default embedding model. See [`local_embedder_from_env`] for why the context
/// window, not the benchmark rank, decides this.
const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";

/// The `(base_url, model, api_key)` for a local embedder, or `None` to stay on the
/// BM25 floor. Returns `Some` only when embeddings are enabled, the provider is
/// `Local`. Base URL and model both default to Ollama (`nomic-embed-text` on
/// `localhost:11434`), which speaks the same OpenAI-compatible `/v1/embeddings`
/// route; the API key is reserved for the M5.3.2 cloud path.
///
/// The model used to have no default, so enabling embeddings without also setting
/// `FF_MEMORY_EMBEDDINGS_MODEL` left the feature silently inert (#1138). A default
/// makes "enabled" mean enabled; a wrong or absent server still degrades to BM25
/// through the [`Embedder`](ff_memory::Embedder) contract rather than failing.
///
/// `nomic-embed-text` is the default for its 8K context window: the longest tool
/// and memory texts we embed run past 700 tokens, and shorter-window models
/// (`mxbai-embed-large` at 512, `all-minilm` at 256) would truncate them
/// silently, dropping exactly the tail that carries the discriminating detail.
fn local_embedder_from_env(config: &MemoryConfig) -> Option<(String, String, Option<String>)> {
    if !config.embeddings.enabled || config.embeddings.provider != EmbeddingProvider::Local {
        return None;
    }
    let model = std::env::var("FF_MEMORY_EMBEDDINGS_MODEL")
        .ok()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string());
    let base = std::env::var("FF_MEMORY_EMBEDDINGS_BASE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_EMBEDDING_BASE_URL.to_string());
    let key = std::env::var("FF_MEMORY_EMBEDDINGS_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());
    tracing::info!(base_url = %base, model = %model, "memory semantic recall enabled (local embedder)");
    Some((base, model, key))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// `memory_*` tools degrade to no-ops rather than failing app start. The initial
/// full reindex happens here (not in the watcher) so a build error is logged once.
/// Opens the memory recall index for an already-constructed [`Memory`]. Split out
/// of the (now I/O-free) `Memory` construction so `with_registry` can open the
/// four independent boot stores in parallel (#599 item 2). Carries the on-disk ->
/// in-memory -> `NullIndex` fallback plus the initial reindex and watcher spawn.
fn open_memory_index(
    memory: &Arc<Memory>,
    embedder: Option<(String, String, Option<String>)>,
    decay: DecayConfig,
) -> (Arc<dyn MemoryIndex>, Option<MemoryWatcher>) {
    let wrap = |i: Fts5Index| -> Arc<dyn MemoryIndex> {
        match &embedder {
            Some((base, model, key)) => Arc::new(HybridIndex::new(
                i,
                OpenAiEmbedder::new(base, model.clone(), key.clone()),
            )),
            None => Arc::new(HybridIndex::new(i, NoopEmbedder)),
        }
    };
    let index: Arc<dyn MemoryIndex> = match Fts5Index::open(memory.index_path()) {
        Ok(i) => wrap(i.with_decay(decay.clone())),
        Err(e) => {
            tracing::warn!(error = %e, "memory index unavailable on disk; using in-memory");
            match Fts5Index::open_in_memory() {
                Ok(i) => wrap(i.with_decay(decay.clone())),
                Err(e) => {
                    tracing::warn!(error = %e, "memory index unavailable; recall disabled");
                    return (Arc::new(NullIndex), None);
                }
            }
        }
    };
    // Defer the full reindex to a background thread (#599 item 4). The on-disk
    // FTS5 DB persists across launches, so recall is immediately available from
    // stale-but-valid data while the refresh runs async. This matches the existing
    // embeddings-on path (PR #215) and removes reindex from the boot critical path.
    let bg_index = index.clone();
    let bg_memory = memory.clone();
    std::thread::spawn(move || {
        // #974 (B.1): opportunistically consolidate on startup when the curated
        // MEMORY.md has grown past its injection budget — merge near-dupes, promote
        // recurring daily facts, and demote low-salience curated entries back to the
        // daily log (RFC 0007). Gated by `needs_consolidation()` so it's a no-op when
        // already within budget; runs here (background, off the boot critical path)
        // because consolidate rewrites files and the reindex below is blocking. The
        // rewrite happens before `all_chunks()`, so the single reindex covers it.
        let mut superseded: Vec<(String, String)> = Vec::new();
        if bg_memory.needs_consolidation() {
            let salience = bg_memory.chunk_stats_salience(bg_index.as_ref(), now_ms());
            match bg_memory.consolidate(&salience) {
                Ok(report) if report.ran => {
                    tracing::info!(
                        merged = report.merged,
                        promoted = report.promoted,
                        demoted = report.demoted,
                        bytes_before = report.bytes_before,
                        bytes_after = report.bytes_after,
                        "startup memory consolidation complete"
                    );
                    superseded = report.superseded;
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "startup memory consolidation failed"),
            }
        }
        let chunks = bg_memory.all_chunks();
        match bg_index.reindex(&chunks) {
            Ok(()) => tracing::info!("memory reindex complete"),
            Err(e) => tracing::warn!(error = %e, "background memory reindex failed"),
        }
        // Record supersession edges after reindex, when both endpoint keys are
        // live in the rebuilt index (memory sifting C1, #1293).
        for (from, to) in &superseded {
            let _ = bg_index.record_link(from, to, "supersession");
        }
    });
    #[cfg(not(test))]
    let watcher = MemoryWatcher::spawn(memory.clone(), index.clone())
        .map_err(|e| tracing::warn!(error = %e, "memory watcher unavailable"))
        .ok();
    #[cfg(test)]
    let watcher = Some(MemoryWatcher::spawn_without_watcher());
    (index, watcher)
}

/// A do-nothing recall index used only when SQLite is entirely unavailable, so the
/// `memory_*` tools return empty results instead of erroring the turn.
struct NullIndex;

impl MemoryIndex for NullIndex {
    fn reindex(&self, _chunks: &[ff_memory::MemoryChunk]) -> ff_memory::Result<()> {
        Ok(())
    }
    fn reindex_path(
        &self,
        _path: &Path,
        _chunks: &[ff_memory::MemoryChunk],
    ) -> ff_memory::Result<()> {
        Ok(())
    }
    fn remove_path(&self, _path: &Path) -> ff_memory::Result<()> {
        Ok(())
    }
    fn search(&self, _query: &str, _k: usize) -> ff_memory::Result<Vec<ff_memory::ScoredChunk>> {
        Ok(Vec::new())
    }
}

/// The default working directory for a session whose cwd is unset: `~/.flowforge/
/// workspaces/`, created on first use. Falls back to the home directory if the
/// directory cannot be created, and to the process CWD when there is no home dir.
/// A user-chosen folder (the per-session picker, #200) overrides this.
fn default_workspace_root() -> PathBuf {
    default_workspace_root_in(dirs::home_dir())
}

/// Testable core of [`default_workspace_root`] with the home dir injected.
fn default_workspace_root_in(home: Option<PathBuf>) -> PathBuf {
    if let Some(home) = home {
        let workspaces = home.join(".flowforge").join("workspaces");
        if fs::create_dir_all(&workspaces).is_ok() {
            return workspaces;
        }
        return home;
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Re-scan root and swap the shared registry's contents. Shared by the installer
/// tools and the install/uninstall commands so a change is visible immediately,
/// independent of the filesystem watcher.
pub fn reload_registry(root: &Path, registry: &SharedRegistry) {
    let (next, errors) = SkillRegistry::load_dir(root);
    for e in &errors {
        tracing::warn!(error = %e, "skill reload after install");
    }
    if let Ok(mut guard) = registry.write() {
        *guard = next;
    }
}

#[cfg(test)]
mod tests;
