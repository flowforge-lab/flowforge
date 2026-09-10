//! Model-free edge extraction for the `chunk_links` graph layer (memory sifting
//! C2, #1294). Two deterministic edge kinds are derived purely from chunk text
//! — no model calls, no external index:
//!
//! - [`parse_wikilinks`] — explicit `[[target]]` references authored in the
//!   Markdown, yielding a `"wikilink"` edge to the chunk whose heading matches.
//! - [`extract_identifiers`] — code-like symbols (backtick/fenced spans,
//!   `CamelCase`, `snake_case`, `path::sep`), the mechanical notion of symbol
//!   identity that connects two chunks talking about the same thing. Two chunks
//!   sharing an identifier get a `"cooccur"` edge.
//!
//! This is deliberately in `ff-memory` (a synchronous leaf crate) rather than
//! reaching for the workspace codegraph MCP tool: codegraph indexes *workspace
//! source*, resolved by file path, and does not index the user's prose memory
//! notes. C2's "symbol identity" is the same mechanical idea applied to memory
//! text — extracted here, deterministically, so it stays unit-testable and
//! never blocks on an out-of-process index.

use std::collections::BTreeSet;

/// Extract `[[wiki-link]]` targets from chunk text, in first-seen order with
/// duplicates removed. The returned strings are the raw inner text, trimmed;
/// resolution to a chunk key is the caller's job (headings are the anchor).
///
/// Only the simple `[[target]]` form is recognised — the `[[target|alias]]`
/// piped form yields the `target` side (before the first `|`). Empty or
/// whitespace-only targets are skipped.
pub fn parse_wikilinks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            if let Some(close) = text[i + 2..].find("]]") {
                let inner = &text[i + 2..i + 2 + close];
                let target = inner.split('|').next().unwrap_or("").trim();
                if !target.is_empty() && seen.insert(target.to_string()) {
                    out.push(target.to_string());
                }
                i += 2 + close + 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Extract code-like identifiers from chunk text — the deterministic,
/// model-free notion of "symbol identity" (#1294 AC3). Recognises:
///
/// - spans inside single/multi backticks (`` `SignalStore::ingest` ``),
/// - `snake_case` and `SCREAMING_SNAKE` words,
/// - `CamelCase`/`PascalCase` words,
/// - `path::separated::symbols`,
///
/// Bare lowercase prose words are intentionally excluded — they are not
/// distinctive enough to imply two chunks are about the same thing and would
/// make co-occurrence edges near-complete (O(n²) noise). Returned lowercased
/// and de-duplicated so casing/order never affects edge identity.
pub fn extract_identifiers(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();

    // 1. Backtick spans: everything between a run of backticks and the next
    //    matching run. Split the span into candidate tokens too.
    for span in backtick_spans(text) {
        for tok in tokenize(span) {
            if is_identifier(tok) {
                out.insert(tok.to_ascii_lowercase());
            }
        }
    }

    // 2. Free-text identifiers (CamelCase, snake_case, path::sep).
    for tok in tokenize(text) {
        if is_identifier(tok) {
            out.insert(tok.to_ascii_lowercase());
        }
    }

    out
}

/// Yield the inner text of each backtick-delimited span. Handles single and
/// multi-backtick fences by matching the opening run length.
fn backtick_spans(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start_run = i;
            while i < bytes.len() && bytes[i] == b'`' {
                i += 1;
            }
            let fence = &text[start_run..i];
            if let Some(rel) = text[i..].find(fence) {
                spans.push(&text[i..i + rel]);
                i += rel + fence.len();
                continue;
            }
        }
        i += 1;
    }
    spans
}

/// Split on characters that never appear inside an identifier (whitespace and
/// most punctuation), while keeping `_` and `:` so `snake_case` and `a::b`
/// survive as single tokens.
fn tokenize(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
        .filter(|t| !t.is_empty())
}

/// A token counts as an identifier if it is a distinctive code-like symbol:
/// contains `_`, `::`, an internal case transition (CamelCase), or a digit
/// adjacent to letters. Pure lowercase/UPPERCASE prose words are rejected.
fn is_identifier(tok: &str) -> bool {
    let tok = tok.trim_matches(':');
    if tok.len() < 2 {
        return false;
    }
    if !tok
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
    {
        return false;
    }
    if tok.contains('_') || tok.contains("::") {
        return true;
    }
    // CamelCase: a lowercase→uppercase transition anywhere.
    let mut prev_lower = false;
    let mut has_case_shift = false;
    let mut has_digit = false;
    for c in tok.chars() {
        if c.is_ascii_digit() {
            has_digit = true;
        }
        if prev_lower && c.is_uppercase() {
            has_case_shift = true;
        }
        prev_lower = c.is_lowercase();
    }
    has_case_shift || has_digit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wikilinks_parsed_in_order_deduped() {
        let t = "See [[Signal Store]] and [[Outcome Gate]], again [[Signal Store]].";
        assert_eq!(parse_wikilinks(t), vec!["Signal Store", "Outcome Gate"]);
    }

    #[test]
    fn wikilink_piped_form_takes_target() {
        assert_eq!(parse_wikilinks("[[target|shown alias]]"), vec!["target"]);
    }

    #[test]
    fn wikilink_empty_and_unclosed_ignored() {
        assert!(parse_wikilinks("[[]] and [[unclosed").is_empty());
    }

    #[test]
    fn identifiers_from_backtick_span() {
        let ids = extract_identifiers("call `SignalStore::ingest` before flush");
        assert!(ids.contains("signalstore::ingest"));
    }

    #[test]
    fn identifiers_camel_snake_path() {
        let ids = extract_identifiers("MemoryIndex and record_link and a::b::c here");
        assert!(ids.contains("memoryindex"));
        assert!(ids.contains("record_link"));
        assert!(ids.contains("a::b::c"));
    }

    #[test]
    fn prose_words_are_not_identifiers() {
        let ids = extract_identifiers("the quick brown fox jumps over memory");
        assert!(
            ids.is_empty(),
            "bare lowercase prose must not become identifiers: {ids:?}"
        );
    }

    #[test]
    fn extraction_is_case_insensitive_and_deterministic() {
        assert_eq!(
            extract_identifiers("`FooBar`"),
            extract_identifiers("foobar mention of FooBar")
                .into_iter()
                .filter(|s| s == "foobar")
                .collect(),
        );
    }
}
