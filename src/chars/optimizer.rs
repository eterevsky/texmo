use std::cmp::min;
use std::mem;

use super::token_stats::CharsTokenStats;
use super::tokenizer::CharsTokenizer;
use super::tokens::CharsTokenSet;
use crate::input::sample::Sampler;

fn count_chars<'a, S: Sampler<'a>>(sampler: &'a S) -> Vec<u64> {
    let mut counts = Vec::new();

    for sample in sampler.iter() {
        eprint!(".");
        for c in sample.as_str().chars() {
            let idx = c as usize;
            if idx >= counts.len() {
                counts.resize(idx + 1, 0);
            }
            counts[idx] += 1;
        }
    }
    eprintln!();

    counts
}

fn tokenize<'a, S: Sampler<'a>>(tokenizer: &CharsTokenizer, sampler: &'a S) -> CharsTokenStats {
    let mut stats = CharsTokenStats::new(tokenizer.token_set.clone());

    for sample in sampler.iter() {
        let sample_stats = tokenizer.process_slice(sample.as_bytes());
        stats.merge(&sample_stats);
    }

    stats
}

#[derive(Debug)]
struct CharsSplit {
    /// First character in the range
    lo: char,
    /// The character in the range with the highest count
    top: char,
    /// The count for the top character
    top_count: u64,
}

enum HuffmanNodeContent {
    Leaf(char),
    Internal(Box<HuffmanNode>, Box<HuffmanNode>),
}

struct HuffmanNode {
    count: u64,
    content: HuffmanNodeContent,
}

fn node_to_split(node: &HuffmanNode) -> CharsSplit {
    match &node.content {
        HuffmanNodeContent::Leaf(ch) => CharsSplit {
            lo: *ch,
            top: *ch,
            top_count: node.count,
        },
        HuffmanNodeContent::Internal(first, second) => {
            let first_split = node_to_split(first);
            let second_split = node_to_split(second);

            let (top, top_count) = if first_split.top_count > second_split.top_count {
                (first_split.top, first_split.top_count)
            } else {
                (second_split.top, second_split.top_count)
            };

            CharsSplit {
                lo: min(first_split.lo, second_split.lo),
                top,
                top_count,
            }
        }
    }
}

fn optimize_splits(counts: &[(char, u64)], parts: usize) -> Vec<CharsSplit> {
    let mut nodes = Vec::new();
    for (ch, count) in counts {
        nodes.push(HuffmanNode {
            count: *count,
            content: HuffmanNodeContent::Leaf(*ch),
        });
    }

    while nodes.len() > parts {
        let mut min_pair_count = None;
        let mut best_idx = None;
        for i in 0..(nodes.len() - 1) {
            let pair_count = nodes[i].count + nodes[i + 1].count;
            if min_pair_count.is_none() || pair_count < min_pair_count.unwrap() {
                min_pair_count = Some(pair_count);
                best_idx = Some(i);
            }
        }
        let best_idx = best_idx.unwrap();
        let second = nodes.remove(best_idx + 1);
        let dummy = HuffmanNode {
            count: 0,
            content: HuffmanNodeContent::Leaf('\0'),
        };
        let first = mem::replace(&mut nodes[best_idx], dummy);
        let new_node = HuffmanNode {
            count: first.count + second.count,
            content: HuffmanNodeContent::Internal(Box::new(first), Box::new(second)),
        };
        nodes[best_idx] = new_node;
    }

    let mut splits = Vec::new();
    for node in nodes.iter() {
        splits.push(node_to_split(node));
    }

    splits
}

fn optimize_ext_encoding(counts: &[(char, u64)], n_ext_tokens: usize) -> Vec<(char, Vec<u32>)> {
    // if counts[0].0 == '\n' {
    //     println!("optimize_ext_encoding: {} {:?}", counts.iter().map(|&(_, s)| s).sum::<u64>(), counts);
    // }

    if counts.len() <= n_ext_tokens {
        return counts
            .iter()
            .enumerate()
            .map(|(i, &(ch, _))| (ch, vec![i as u32]))
            .collect::<Vec<_>>();
    }

    let splits = optimize_splits(counts, n_ext_tokens);
    let mut encodings = Vec::new();

    for (i, split) in splits.iter().enumerate() {
        encodings.push((split.top, vec![i as u32]));

        let counts_lo = counts.binary_search(&(split.lo, 0)).unwrap_err();
        let counts_hi = if i == splits.len() - 1 {
            counts.len()
        } else {
            counts.binary_search(&(splits[i + 1].lo, 0)).unwrap_err()
        };

        if counts_hi == counts_lo + 1 {
            continue;
        }

        let mut sub_counts = counts[counts_lo..counts_hi].to_vec();

        let top_idx = sub_counts.binary_search(&(split.top, 0)).unwrap_err();
        sub_counts.remove(top_idx);

        let sub_encs = optimize_ext_encoding(sub_counts.as_slice(), n_ext_tokens);
        for (ch, mut enc) in sub_encs {
            enc.insert(0, i as u32);
            encodings.push((ch, enc));
        }
    }

    encodings
}

fn optimize_chars(
    counts: &[(char, u64)],
    n_char_tokens: usize,
    n_ext_tokens: usize,
) -> CharsTokenSet {
    let mut token_set = CharsTokenSet::new(n_ext_tokens);

    let top_splits = optimize_splits(counts, n_char_tokens);
    for (i, split) in top_splits.iter().enumerate() {
        let char_token_id = token_set.add_char_token(split.top);

        let counts_lo = counts.binary_search(&(split.lo, 0)).unwrap_err();
        let counts_hi = if i == top_splits.len() - 1 {
            counts.len()
        } else {
            counts
                .binary_search(&(top_splits[i + 1].lo, 0))
                .unwrap_err()
        };

        if counts_hi == counts_lo + 1 {
            continue;
        }

        let mut sub_counts = counts[counts_lo..counts_hi].to_vec();

        let top_idx = sub_counts.binary_search(&(split.top, 0)).unwrap_err();
        sub_counts.remove(top_idx);

        let encs = optimize_ext_encoding(sub_counts.as_slice(), n_ext_tokens);
        for (ch, mut enc) in encs {
            enc.insert(0, (n_ext_tokens + i) as u32);
            let tokens = enc
                .iter()
                .map(|idx| CharsTokenSet::ext_token(*idx))
                .collect::<Vec<_>>();
            token_set.add_encoding(ch, tokens);
        }
    }

    token_set
}

fn optimize_chars_by_ext(counts: &[(char, u64)], ntokens: usize) -> CharsTokenSet {
    let mut best_token_set = None;
    let mut best_total_tokens = None;

    let max_ext_tokens = min(ntokens - 1, 8);

    for n_ext_tokens in 2..=max_ext_tokens {
        let n_char_tokens = ntokens - n_ext_tokens;

        let token_set = optimize_chars(counts, n_char_tokens, n_ext_tokens);

        let mut total = 0;
        for (c, count) in counts.iter() {
            let enc = token_set.char_encoding(*c);
            total += enc.len() as u64 * count;
        }

        if best_total_tokens.is_none() || total < best_total_tokens.unwrap() {
            best_token_set = Some(token_set);
            best_total_tokens = Some(total);
        }
    }

    best_token_set.unwrap()
}

pub fn optimize_chars_tokens<'a, SS: Sampler<'a>, S: Sampler<'a>, FS: Sampler<'a>>(
    slow_sampler: &'a SS,
    sampler: &'a S,
    fast_sampler: &'a FS,
) {
    let counts = count_chars(slow_sampler);
    let counts: Vec<(char, u64)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(idx, &c)| (char::from_u32(idx as u32).unwrap(), c))
        .collect::<Vec<_>>();
    let total_chars = counts.iter().map(|&(_, c)| c).sum::<u64>();

    let token_set = optimize_chars_by_ext(counts.as_slice(), 64);
    println!("{}", token_set);

    let mut total = 0;
    for (c, count) in counts.iter() {
        let enc = token_set.char_encoding(*c);
        total += enc.len() as u64 * count;
    }

    println!(
        "total: {}, {} t/ch",
        total,
        total as f64 / total_chars as f64
    );

    let tokenizer = CharsTokenizer::new(token_set);
}

// 2/2 total: 25531435, 2.625872554735772 t/ch
// 1/3 total: 26807758, 2.7571405988812745 t/ch

// 6/2 total: 17389788, 1.788515492445821 t/ch
// 5/3 total: 16995225, 1.7479352370542143 t/ch
// 4/4 total: 17670461, 1.8173823198511492 t/ch
// 3/5 total: 19028020, 1.9570053735312318 t/ch
// 2/6 total: 19617895, 2.0176731962848202 t/ch
// 1/7 total: 22345268, 2.2981797133383024 t/ch
