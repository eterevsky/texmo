//! Builder for `hexbpe` tokensets.
//!
//! One vocabulary schema for every size from 32 upwards:
//!
//! * 16 reserved nibble tokens ("hex digits"). Any byte without a
//!   token of its own is emitted as two of them, so every set is
//!   lossless at every size and no residual entropy is left on the
//!   table.
//! * the remaining slots hold selected single bytes and merged
//!   strings, chosen greedily.
//!
//! The point of the schema is that the two ways to spend a slot are
//! measured in the SAME currency -- tokens saved over the training
//! sample:
//!
//! * selecting byte `b` takes it from two nibbles to one token, so it
//!   saves `count(b)`;
//! * merging an adjacent pair `(A, B)` of already-selected tokens
//!   saves `count(AB)`.
//!
//! Comparing "split an interval" against "BPE merge" by entropy is not
//! honest, because which one wins depends on how strong the model is.
//! Counting emitted tokens sidesteps that: both operations reduce the
//! same quantity, so a single greedy ordering is well defined.
//!
//! Weight accounting (see the tokenset discussion): a byte selection
//! is one stored number and a merge is two, so a set costs
//! `selected_bytes + 2 * merges` weights on top of its IO width.

use std::collections::HashMap;

use crate::processing::process2;

/// Reserved nibble tokens; ids 0..16 in the emitted vocabulary.
pub const NIBBLES: usize = 16;

/// capswords2 word terminator; see `HexBpe::merge_allowed`.
const WORD_MARKER: u8 = 0x16;

/// True when `s` holds at most one run of newlines and that run touches
/// the start or the end -- so the token never has real content on both
/// sides of a newline.
fn newline_run_at_one_end(s: &[u8]) -> bool {
    let Some(first) = s.iter().position(|&b| b == b'\n') else {
        return true;
    };
    let last = s.iter().rposition(|&b| b == b'\n').unwrap();
    if s[first..=last].iter().any(|&b| b != b'\n') {
        return false; // two separate runs, so content sits between them
    }
    first == 0 || last == s.len() - 1
}

/// True when nothing after the last word marker is a letter.
///
/// A word begins immediately after a marker (the single inter-word
/// space is elided), so letters there are the START of a following
/// word: "about<mark>self" would tokenize "self" differently from
/// every other occurrence of it. Non-ASCII bytes count as letters --
/// conservative, since a multi-byte character may be one.
fn no_word_start_inside(s: &[u8]) -> bool {
    match s.iter().rposition(|&b| b == WORD_MARKER) {
        None => true,
        Some(i) => !s[i + 1..]
            .iter()
            .any(|&b| b.is_ascii_alphabetic() || b >= 0x80),
    }
}

pub struct HexBpe {
    /// Token string per internal id. Ids 0..256 are the single bytes.
    strings: Vec<Vec<u8>>,
    /// Whether an id owns a vocabulary slot. Unselected single bytes
    /// fall back to two nibbles and cannot take part in a merge.
    selected: Vec<bool>,
    /// The training sample as internal ids.
    units: Vec<u32>,
    /// Merges in application order, as (left, right) token strings.
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl HexBpe {
    pub fn new(corpus: &[u8]) -> Self {
        let strings: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        HexBpe {
            strings,
            selected: vec![false; 256],
            units: corpus.iter().map(|&b| b as u32).collect(),
            merges: Vec::new(),
        }
    }

    /// Tokens emitted for the sample: 1 per selected unit, 2 for a
    /// byte that has to fall back to nibbles.
    pub fn cost(&self) -> u64 {
        self.units
            .iter()
            .map(|&u| if self.selected[u as usize] { 1 } else { 2 })
            .sum()
    }

    fn byte_counts(&self) -> HashMap<u32, u64> {
        let mut counts = HashMap::new();
        for &u in &self.units {
            if !self.selected[u as usize] {
                *counts.entry(u).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Counts of adjacent pairs where BOTH sides already hold a slot.
    /// Overlapping repeats are counted non-greedily (aaa -> one aa),
    /// matching how the merge will actually apply.
    fn pair_counts(&self) -> HashMap<(u32, u32), u64> {
        let mut counts = HashMap::new();
        let mut i = 0;
        while i + 1 < self.units.len() {
            let (a, b) = (self.units[i], self.units[i + 1]);
            if self.selected[a as usize] && self.selected[b as usize] {
                *counts.entry((a, b)).or_insert(0) += 1;
                if a == b {
                    // Don't let the same unit serve as the right half
                    // of one pair and the left half of the next.
                    i += 1;
                }
            }
            i += 1;
        }
        counts
    }

    /// Structural constraints on what a merge may produce.
    ///
    /// 1. No token has real content on both sides of a newline, so none
    ///    spans a paragraph break. ".\n\n" and "\n\n<caps>i" stay legal
    ///    (each attaches to one side of the break); ".\n\n<caps>" does
    ///    not. This keeps paragraph starts usable as cut points, so a
    ///    prompt can be split from a continuation without handing the
    ///    model a tokenization it never saw in training.
    /// 2. Nothing after the last word marker is a letter, so a token
    ///    never contains the START of a following word. A leading
    ///    marker is fine on its own ("<mark>. " is word-final material
    ///    with no word start in it); what matters is that "self" is
    ///    never glued onto a preceding word's end, which would give it
    ///    a different token from every other occurrence.
    ///
    /// Sentence breaks are deliberately unconstrained -- they are not
    /// cut points, and ". <caps>" is real structure worth learning.
    fn merge_allowed(&self, a: u32, b: u32) -> bool {
        let mut joined = self.strings[a as usize].clone();
        joined.extend_from_slice(&self.strings[b as usize]);
        newline_run_at_one_end(&joined) && no_word_start_inside(&joined)
    }

    fn apply_merge(&mut self, a: u32, b: u32) -> u32 {
        let mut string = self.strings[a as usize].clone();
        string.extend_from_slice(&self.strings[b as usize]);
        let id = self.strings.len() as u32;
        self.merges.push((
            self.strings[a as usize].clone(),
            self.strings[b as usize].clone(),
        ));
        self.strings.push(string);
        self.selected.push(true);

        let mut out = Vec::with_capacity(self.units.len());
        let mut i = 0;
        while i < self.units.len() {
            if i + 1 < self.units.len() && self.units[i] == a && self.units[i + 1] == b {
                out.push(id);
                i += 2;
            } else {
                out.push(self.units[i]);
                i += 1;
            }
        }
        self.units = out;
        id
    }

    /// Grow the vocabulary to `ntokens` total (including the 16
    /// reserved nibbles), taking the best-scoring operation each step.
    /// Returns false when nothing can improve the cost any more.
    pub fn build(&mut self, ntokens: usize, verbose: bool) -> bool {
        assert!(ntokens > NIBBLES, "hexbpe needs more than {NIBBLES} tokens");
        let budget = ntokens - NIBBLES;

        while self.selected.iter().filter(|&&s| s).count() < budget {
            let byte_best = self
                .byte_counts()
                .into_iter()
                .max_by_key(|&(id, n)| (n, std::cmp::Reverse(id)));
            let pair_best = self
                .pair_counts()
                .into_iter()
                .filter(|&((a, b), _)| self.merge_allowed(a, b))
                .max_by_key(|&(ids, n)| (n, std::cmp::Reverse(ids)));

            let byte_gain = byte_best.map_or(0, |(_, n)| n);
            let pair_gain = pair_best.map_or(0, |(_, n)| n);

            if byte_gain == 0 && pair_gain == 0 {
                return false;
            }

            // Ties go to the byte: it costs one stored number against
            // a merge's two, so it is the cheaper way to buy the same
            // saving.
            if byte_gain >= pair_gain {
                let (id, n) = byte_best.unwrap();
                self.selected[id as usize] = true;
                if verbose {
                    println!(
                        "  +byte {:?} (saves {})",
                        format_token(&self.strings[id as usize]),
                        n
                    );
                }
            } else {
                let ((a, b), n) = pair_best.unwrap();
                let id = self.apply_merge(a, b);
                if verbose {
                    println!(
                        "  +merge {:?} (saves {})",
                        format_token(&self.strings[id as usize]),
                        n
                    );
                }
            }
        }
        true
    }

    /// (selected single bytes, merged strings) in vocabulary order.
    pub fn vocabulary(&self) -> (Vec<u8>, Vec<Vec<u8>>) {
        let mut bytes = Vec::new();
        let mut merged = Vec::new();
        for (id, string) in self.strings.iter().enumerate() {
            if !self.selected[id] {
                continue;
            }
            if id < 256 {
                bytes.push(string[0]);
            } else {
                merged.push(string.clone());
            }
        }
        (bytes, merged)
    }

    /// Weights the tokenset costs beyond its IO width: one stored
    /// number per selected byte, two per merge.
    pub fn extra_weights(&self) -> usize {
        let (bytes, _) = self.vocabulary();
        bytes.len() + 2 * self.merges.len()
    }
}

pub fn format_token(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

/// Read `nchunks` evenly spaced chunks of the raw corpus, trim each to
/// whole lines, and run capswords2 over it.
///
/// Processing the sample instead of the whole corpus is what makes
/// this practical: books3 is 108 GB, and a preprocessed copy of it is
/// neither cheap to build nor worth keeping around.
pub fn sample_processed(
    path: &str,
    chunk_size: usize,
    nchunks: usize,
) -> std::io::Result<(Vec<u8>, u64)> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut raw_bytes: u64 = 0;
    let mut out = Vec::with_capacity(chunk_size * nchunks);
    let stride = if nchunks > 1 {
        size / nchunks as u64
    } else {
        0
    };

    let mut buf = vec![0u8; chunk_size];
    for i in 0..nchunks {
        let offset = stride * i as u64;
        if offset >= size {
            break;
        }
        file.seek(SeekFrom::Start(offset))?;
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        // Whole lines only: capswords2 is line-oriented, and a partial
        // line would put a word boundary where the corpus has none.
        let slice = &buf[..n];
        let start = slice.iter().position(|&b| b == b'\n').map_or(n, |p| p + 1);
        let end = slice.iter().rposition(|&b| b == b'\n').map_or(start, |p| p + 1);
        if end <= start {
            continue;
        }
        if let Ok(text) = std::str::from_utf8(&slice[start..end]) {
            // Raw length of exactly the lines that made it into the
            // sample: bytes_per_token must charge tokens against RAW
            // corpus bytes (the invariant losses are compared on),
            // while the processed length only sizes the sample itself.
            raw_bytes += (end - start) as u64;
            for line in text.lines() {
                out.extend_from_slice(process2(line).as_bytes());
                out.push(b'\n');
            }
        }
    }
    Ok((out, raw_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_frequent_bytes_before_merging() {
        // 'a' and 'b' are far more common than anything else, so the
        // first slots go to them, and only then to the pair.
        let corpus = "abababababab".repeat(20).into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        assert!(bpe.build(NIBBLES + 2, false));
        let (bytes, merged) = bpe.vocabulary();
        assert_eq!(bytes, vec![b'a', b'b']);
        assert!(merged.is_empty());
        assert!(bpe.merges.is_empty());
    }

    #[test]
    fn merges_once_the_bytes_are_selected() {
        let corpus = "abababababab".repeat(20).into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        assert!(bpe.build(NIBBLES + 3, false));
        let (_, merged) = bpe.vocabulary();
        assert_eq!(merged, vec![b"ab".to_vec()]);
        assert_eq!(bpe.merges.len(), 1);
        assert_eq!(bpe.merges[0], (b"a".to_vec(), b"b".to_vec()));
    }

    #[test]
    fn merges_compose_into_longer_tokens() {
        let corpus = "the ".repeat(200).into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        bpe.build(NIBBLES + 8, false);
        let (_, merged) = bpe.vocabulary();
        // 't','h','e',' ' get slots, then pairs build up to "the ".
        assert!(
            merged.iter().any(|m| m.len() >= 3),
            "expected a multi-byte merge, got {:?}",
            merged.iter().map(|m| format_token(m)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cost_falls_monotonically() {
        let corpus = "hello world, hello there".repeat(50).into_bytes();
        let bpe = HexBpe::new(&corpus);
        let start = bpe.cost();
        assert_eq!(start, 2 * corpus.len() as u64); // all fallback
        let mut prev = start;
        for n in 1..24 {
            let mut b = HexBpe::new(&corpus);
            b.build(NIBBLES + n, false);
            let c = b.cost();
            assert!(c <= prev, "cost rose at {n}: {prev} -> {c}");
            prev = c;
        }
        assert!(prev < start / 2);
    }

    #[test]
    fn newline_rule_accepts_only_one_sided_runs() {
        assert!(newline_run_at_one_end(b".\n\n")); // run at the end
        assert!(newline_run_at_one_end(b"\n\nxi")); // run at the start
        assert!(newline_run_at_one_end(b"\n"));
        assert!(newline_run_at_one_end(b"\n\n"));
        assert!(newline_run_at_one_end(b"the ")); // no newline at all
        assert!(!newline_run_at_one_end(b".\n\nx")); // content both sides
        assert!(!newline_run_at_one_end(b"a\nb"));
        assert!(!newline_run_at_one_end(b"\na\n")); // two runs
    }

    #[test]
    fn merges_never_span_a_paragraph_break() {
        let corpus = "end.\n\nStart of the next one.\n\n".repeat(300).into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        bpe.build(NIBBLES + 40, false);
        let (_, merged) = bpe.vocabulary();
        assert!(!merged.is_empty());
        for m in &merged {
            assert!(
                newline_run_at_one_end(m),
                "token {:?} spans a paragraph break",
                format_token(m)
            );
        }
    }

    #[test]
    fn no_merged_token_contains_a_following_word_start() {
        // capswords2 shape: words terminated by the marker, the single
        // inter-word space elided.
        let corpus = "one\x16self\x16two\x16self\x16".repeat(300).into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        bpe.build(NIBBLES + 30, false);
        let (_, merged) = bpe.vocabulary();
        assert!(!merged.is_empty());
        for m in &merged {
            assert!(
                no_word_start_inside(m),
                "token {:?} contains the start of a following word",
                format_token(m)
            );
        }
    }

    #[test]
    fn extra_weights_counts_selections_and_merges() {
        let corpus = "abcabcabc".repeat(30).into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        bpe.build(NIBBLES + 5, false);
        let (bytes, _) = bpe.vocabulary();
        assert_eq!(bpe.extra_weights(), bytes.len() + 2 * bpe.merges.len());
    }

    #[test]
    fn merge_operands_are_always_already_selected() {
        let corpus = "the quick brown fox jumps over the lazy dog. "
            .repeat(60)
            .into_bytes();
        let mut bpe = HexBpe::new(&corpus);
        bpe.build(NIBBLES + 40, false);
        // Every merge's operands must exist as tokens at the time it
        // is applied, or the merge table cannot be replayed.
        let mut known: Vec<Vec<u8>> = Vec::new();
        let (bytes, _) = bpe.vocabulary();
        for b in &bytes {
            known.push(vec![*b]);
        }
        for (a, b) in &bpe.merges {
            assert!(known.contains(a), "merge left {:?} unknown", format_token(a));
            assert!(known.contains(b), "merge right {:?} unknown", format_token(b));
            let mut joined = a.clone();
            joined.extend_from_slice(b);
            known.push(joined);
        }
    }
}

// --- JSON output -------------------------------------------------------

/// A token as the Python loader wants it: a string when the bytes are
/// valid UTF-8, otherwise a list of byte values.
fn json_token(bytes: &[u8]) -> serde_json::Value {
    match std::str::from_utf8(bytes) {
        Ok(s) => serde_json::Value::String(s.to_string()),
        Err(_) => serde_json::Value::Array(
            bytes.iter().map(|&b| serde_json::json!(b)).collect(),
        ),
    }
}

impl HexBpe {
    pub fn to_json(
        &self,
        ntokens: usize,
        raw_bytes: u64,
        scanned_bytes: u64,
        initial_size: u64,
    ) -> serde_json::Value {
        let (bytes, merged) = self.vocabulary();

        // Ids 0..16 are the nibbles, emitted as numbered ext tokens.
        let mut tokens: Vec<serde_json::Value> =
            (0..NIBBLES).map(|i| serde_json::json!(i)).collect();
        // Everything past the nibbles is sorted lexicographically, so a
        // set is easy to scan by eye. Safe to reorder: the merge table
        // refers to tokens by string, and `sequences` refers to the
        // nibbles by their numbered ids, which keep positions 0..16.
        let mut rest: Vec<Vec<u8>> = bytes.iter().map(|b| vec![*b]).collect();
        rest.extend(merged.iter().cloned());
        rest.sort();
        for token in &rest {
            tokens.push(json_token(token));
        }

        // Every byte without a slot falls back to two nibbles, which
        // is what keeps the set lossless at every size.
        let selected: std::collections::HashSet<u8> = bytes.iter().copied().collect();
        let sequences: Vec<serde_json::Value> = (0..=255u8)
            .filter(|b| !selected.contains(b))
            .map(|b| {
                serde_json::json!({
                    "string": json_token(&[b]),
                    "tokens": [b >> 4, b & 0xf],
                })
            })
            .collect();

        let merges: Vec<serde_json::Value> = self
            .merges
            .iter()
            .map(|(a, b)| serde_json::json!([json_token(a), json_token(b)]))
            .collect();

        let total_tokens = self.cost();
        serde_json::json!({
            "type": "hexbpe",
            "processing": "capswords2",
            "algorithm": "bpe",
            "tokens": tokens,
            "merges": merges,
            "sequences": sequences,
            "stats": {
                "ntokens": ntokens,
                // RAW corpus bytes per token: byte_loss divides by
                // this, and losses are compared per raw byte. The
                // processed length (scanned_bytes) is ~6% longer
                // under capswords2 and only sizes processed chunks.
                "bytes_per_token": raw_bytes as f64 / total_tokens as f64,
                "extra_weights": self.extra_weights(),
                "residual_bits_per_byte": 0.0,
                "selected_bytes": bytes.len(),
                "merges": self.merges.len(),
                "raw_bytes": raw_bytes,
                "scanned_bytes": scanned_bytes,
                "total_tokens": total_tokens,
                "initial_size": initial_size,
            },
        })
    }
}
