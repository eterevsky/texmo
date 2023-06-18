use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::sampler::{Sampler, SelectionSampler};
use crate::stats::TokenStats;
use crate::tokenizer::tokenize_file;
use crate::tokens::{LiteralEncoding, TokenSet};

fn format_token(s: &[u8]) -> String {
    match String::from_utf8(s.to_vec()) {
        Ok(string) => format!("{:?}", string),
        Err(_) => format!("{:?}", s),
    }
}

struct TokenizerCache<'a, S: Sampler<'a>> {
    sampler: &'a S,
    cache: HashMap<Vec<u8>, TokenStats>,
}

impl<'a, S: Sampler<'a>> TokenizerCache<'a, S> {
    fn new(sampler: &'a S) -> Self {
        TokenizerCache {
            sampler,
            cache: HashMap::new(),
        }
    }

    fn get_stats(&mut self, token_set: &TokenSet) -> TokenStats {
        let mut tokens = token_set
            .tokens
            .iter()
            .map(|t| t.string.clone())
            .collect::<Vec<_>>();
        tokens.sort_unstable();

        let mut key = Vec::new();
        for token in tokens {
            key.extend(token);
            key.push(0);
        }

        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let stats = tokenize_file(token_set, self.sampler);
        self.cache.insert(key, stats.clone());

        stats
    }

    fn total(&self) -> usize {
        self.cache.len()
    }
}

fn add_tokens<'a, S: Sampler<'a>>(
    tokenizer: &mut TokenizerCache<'a, S>,
    token_set: &mut TokenSet,
    tokens_to_add: usize,
) -> Vec<Vec<u8>> {
    let stats = tokenizer.get_stats(token_set);

    let mut token_values = Vec::new();

    for i in 0..256 {
        if stats.literal_count[i] > 0 {
            token_values.push((
                vec![i as u8],
                stats.literal_count[i] * (token_set.literal_cost() - 1),
            ))
        }
    }

    let set_size = token_set.tokens.len();

    for ipair in 0..stats.pair_count.len() {
        if stats.pair_count[ipair] > 0 {
            let ifirst = ipair / set_size;
            let isecond = ipair % set_size;

            let mut token_str = token_set.tokens[ifirst].string.clone();
            token_str.extend(token_set.tokens[isecond].string.clone());
            token_values.push((token_str, stats.pair_count[ipair]))
        }
    }

    token_values.sort_unstable_by_key(|&(_, value)| -(value as i64));

    let mut added = Vec::new();

    for (token_str, _) in token_values.iter().take(tokens_to_add) {
        token_set.add_token(token_str.as_slice());
        added.push(token_str.clone());
    }

    added
}

fn add_and_remove_token<'a, S1: Sampler<'a>, S2: Sampler<'a>>(
    tokenizer: &mut TokenizerCache<'a, S1>,
    fast_tokenizer: &mut TokenizerCache<'a, S2>,
    token_set: &TokenSet,
) -> Option<TokenSet> {
    let mut token_set = token_set.clone();
    let initial_stats = tokenizer.get_stats(&token_set);

    add_tokens(tokenizer, &mut token_set, 1);

    let stats_before = tokenizer.get_stats(&token_set);
    let fast_stats_before = fast_tokenizer.get_stats(&token_set);

    let mut token_ids: Vec<usize> = (0..token_set.tokens.len()).collect();
    token_ids.sort_unstable_by_key(|&i| stats_before.token_count[i]);

    // We construct a list of tokens to remove because the ids will change when
    // we are removing and adding tokens.
    let mut token_strs: Vec<Vec<u8>> = Vec::new();

    for &token_id_to_remove in token_ids.iter() {
        let token_to_remove = &token_set.tokens[token_id_to_remove];
        if token_to_remove.is_mandatory {
            continue;
        }
        token_strs.push(token_to_remove.string.clone());
    }

    let add_token_limit = initial_stats.cost() - stats_before.cost();
    // Check if removing a token adds > 3/2 the cost that we can afford on a sparser sampler.
    let fast_add_token_limit =
        add_token_limit * fast_stats_before.scanned_bytes * 3 / (stats_before.scanned_bytes * 2);
    let fast_cost_before = fast_stats_before.cost();

    println!(
        "Can add up to {:.4}% cost by removing a token.",
        100.0 * add_token_limit as f64 / stats_before.cost() as f64
    );

    let mut tries = 0;

    for token_str in token_strs.iter() {
        let token_str = token_str.as_slice();
        tries += 1;

        token_set.remove_token(token_str);
        let fast_stats = fast_tokenizer.get_stats(&token_set);

        if fast_stats.cost() - fast_cost_before > fast_add_token_limit {
            println!("Not checking whether we can remove {} since removing it adds {:.4}% tokens on the smaller sample.",
                     format_token(token_str), 100.0 * (fast_stats.cost() - fast_cost_before) as f64 / fast_cost_before as f64);
            continue;
        }

        let stats = tokenizer.get_stats(&token_set);
        // token_set.update_stats(&stats);

        // if stats.cost < initial_stats.cost
        //     && stats.token_to_add(&new_token_set) == token_str
        // {
        //     println!("Cost after removing {} would be {}, but it would be added on the next iteration.", format_token(&token_str), stats.cost);
        // }

        if stats.cost() < initial_stats.cost() {
            // Found a token to remove.
            println!(
                "removing {} after {} tries",
                format_token(&token_str),
                tries
            );
            return Some(token_set);
        }

        token_set.add_token(&token_str);
    }

    None
}

fn remove_and_add_token<'a, S1: Sampler<'a>, S2: Sampler<'a>>(
    tokenizer: &mut TokenizerCache<'a, S1>,
    _fast_tokenizer: &mut TokenizerCache<'a, S2>,
    token_set: &TokenSet,
    last_token_check: &mut HashMap<Vec<u8>, usize>,
    step: usize,
) -> Option<TokenSet> {
    let initial_stats = tokenizer.get_stats(token_set);

    let mut token_ids: Vec<usize> = (0..token_set.tokens.len()).collect();
    token_ids.sort_unstable_by_key(|&i| {
        last_token_check
            .get(&token_set.tokens[i].string)
            .unwrap_or(&0)
    });

    // We construct a list of tokens to remove because the ids will change when
    // we are removing and adding tokens.
    let mut token_strs: Vec<Vec<u8>> = Vec::new();

    for &token_id_to_remove in token_ids.iter() {
        let token_to_remove = &token_set.tokens[token_id_to_remove];
        if token_to_remove.is_mandatory {
            continue;
        }
        token_strs.push(token_to_remove.string.clone());
    }

    let mut tries = 0;

    eprint!("Trying to remove:");

    // Try no more than the first 256 tokens.
    for token_str in token_strs.iter().take(256) {
        eprint!(" {} ", format_token(token_str));
        last_token_check.insert(token_str.clone(), step);
        let token_str = token_str.as_slice();
        tries += 1;

        let mut new_token_set = token_set.clone();

        new_token_set.remove_token(token_str);

        let added = add_tokens(tokenizer, &mut new_token_set, 1);

        if added[0] == token_str {
            continue;
        }

        let new_stats = tokenizer.get_stats(&new_token_set);

        if new_stats.cost() < initial_stats.cost() {
            eprintln!(
                "\nReplacing {} -> {} after {} tries",
                format_token(&token_str),
                format_token(added[0].as_slice()),
                tries
            );
            return Some(new_token_set);
        }
    }

    eprintln!("\nNo token to replace after {} tries", tries);

    None
}

fn add_tokens_bpe<'a, S: Sampler<'a>>(
    tokenizer: &mut TokenizerCache<'a, S>,
    token_set: &mut TokenSet,
    ntokens: usize,
    add_block: usize,
) {
    while token_set.ntokens() < ntokens {
        let tokens_to_add = min(add_block, ntokens - token_set.ntokens());
        let added = add_tokens(tokenizer, token_set, tokens_to_add);
        for token_str in added.iter() {
            println!("Added {}", format_token(token_str.as_slice()));
        }
        let stats = tokenizer.get_stats(&token_set);
        println!(
            "{} tokens, bytes/cost = {:.3}  literals/bytes = {:.5}",
            token_set.ntokens(),
            stats.scanned_bytes as f64 / stats.cost() as f64,
            stats.total_literals() as f64 / stats.scanned_bytes as f64,
        );
    }
}

pub fn optimize_bpe<'a, S: Sampler<'a>>(
    token_set: &TokenSet,
    ntokens: usize,
    sampler: &'a S,
    fast_sampler: &'a SelectionSampler,
    add_block: usize,
) -> (TokenSet, TokenStats) {
    let mut token_set = token_set.clone();
    let mut tokenizer = TokenizerCache::new(sampler);
    let mut fast_tokenizer = TokenizerCache::new(fast_sampler);

    add_tokens_bpe(&mut tokenizer, &mut token_set, ntokens, add_block);

    let mut last_token_check = HashMap::new();
    let mut step = 0;

    loop {
        step += 1;
        let stats = tokenizer.get_stats(&token_set);
        println!(
            "{} tokens, bytes/cost = {:.3}  literals/bytes = {:.5}",
            token_set.ntokens(),
            stats.scanned_bytes as f64 / stats.cost() as f64,
            stats.total_literals() as f64 / stats.scanned_bytes as f64,
        );
        // if let Some(new_token_set) =
        //     add_and_remove_token(&mut tokenizer, &mut fast_tokenizer, &token_set)
        // {
        //     token_set = new_token_set;
        //     continue;
        // }
        if let Some(new_token_set) = remove_and_add_token(
            &mut tokenizer,
            &mut fast_tokenizer,
            &token_set,
            &mut last_token_check,
            step,
        ) {
            token_set = new_token_set;
            continue;
        } else {
            break;
        }
    }

    println!("Number of tokenizations: {}", tokenizer.total());

    let stats = tokenizer.get_stats(&token_set);
    (token_set, stats)
}

fn save_token_set(
    token_set: &TokenSet,
    stats: &TokenStats,
    output_path: &Path,
    processing: &str,
    initial_size: u64,
) {
    let mut tokens_json = token_set.to_json(stats, initial_size);
    tokens_json["processing"] = processing.into();

    println!("Writing to {}", output_path.display());

    std::fs::write(&output_path, json::stringify_pretty(tokens_json, 2)).unwrap();
}

fn optimize_token_set<'a, S1: Sampler<'a>, S2: Sampler<'a>, S3: Sampler<'a>>(
    initial_size: u64,
    mut token_set: TokenSet,
    slow_sampler: &'a S1,
    sampler: &'a S2,
    fast_sampler: &'a S3,
    ntokens: usize,
    processing: &str,
    block: usize,
    output_path: &Path,
) {
    let mut tokenizer = TokenizerCache::new(sampler);
    let mut fast_tokenizer = TokenizerCache::new(fast_sampler);

    let mut best_cost = if token_set.ntokens() < ntokens {
        add_tokens_bpe(&mut tokenizer, &mut token_set, ntokens, block);
        let stats = tokenize_file(&token_set, slow_sampler);

        save_token_set(&token_set, &stats, output_path, processing, initial_size);

        stats.cost()
    } else {
        let stats = tokenize_file(&token_set, slow_sampler);
        stats.cost()
    };

    let mut last_update_time = Instant::now();

    let mut last_token_check = HashMap::new();
    let mut step = 0;

    loop {
        step += 1;
        let stats = tokenizer.get_stats(&token_set);
        println!(
            "bytes/cost = {:.3}  literals/bytes = {:.5}",
            stats.scanned_bytes as f64 / stats.cost() as f64,
            stats.total_literals() as f64 / stats.scanned_bytes as f64,
        );
        if let Some(new_token_set) = remove_and_add_token(
            &mut tokenizer,
            &mut fast_tokenizer,
            &token_set,
            &mut last_token_check,
            step,
        ) {
            token_set = new_token_set;

            if Instant::now() - last_update_time > Duration::from_secs(300) {
                let stats = tokenize_file(&token_set, slow_sampler);
                let cost = stats.cost();
                if cost < best_cost {
                    println!(
                        "Slow stats: bytes/cost = {:.3}  literals/bytes = {:.5}",
                        initial_size as f64 / stats.cost() as f64,
                        stats.total_literals() as f64 / initial_size as f64,
                    );
                    save_token_set(&token_set, &stats, output_path, processing, initial_size);
                    best_cost = cost;
                } else {
                    println!("Cost increased, not saving");
                }
                last_update_time = Instant::now();
            }

            continue;
        } else {
            break;
        }
    }

    let stats = tokenize_file(&token_set, slow_sampler);
    let cost = stats.cost();
    if cost < best_cost {
        println!(
            "Slow stats: bytes/cost = {:.3}  literals/bytes = {:.5}",
            stats.scanned_bytes as f64 / stats.cost() as f64,
            stats.total_literals() as f64 / stats.scanned_bytes as f64,
        );
        save_token_set(&token_set, &stats, output_path, processing, initial_size);
    } else {
        println!("Cost increased, not saving");
    }
}

fn load_prev_token_set(
    tokens_dir: &str,
    ntokens: usize,
    processing: &str,
    literal_encoding: LiteralEncoding,
) -> Option<TokenSet> {
    let tokens_filename = format!(
        "{}/tokens{}_{}_{}.json",
        tokens_dir, ntokens, processing, literal_encoding
    );
    if Path::new(&tokens_filename).exists() {
        Some(TokenSet::from_json(&tokens_filename))
    } else {
        if ntokens > 2 {
            load_prev_token_set(tokens_dir, ntokens / 2, processing, literal_encoding)
        } else {
            None
        }
    }
}

pub fn optimize_all<'a, S1: Sampler<'a>, S2: Sampler<'a>, S3: Sampler<'a>>(
    initial_size: u64,
    slow_sampler: &'a S1,
    sampler: &'a S2,
    fast_sampler: &'a S3,
    tokens_dir: &str,
    min_tokens: usize,
    max_tokens: usize,
    processing: &str,
) {
    let tokens_dir_path = std::path::Path::new(tokens_dir);

    for &literal_encoding in &[
        LiteralEncoding::Bits1,
        LiteralEncoding::Bits2,
        LiteralEncoding::Bits4,
        LiteralEncoding::All,
        // LiteralEncoding::Dist2,
        // LiteralEncoding::Dist4,
        // LiteralEncoding::Dist8,
    ] {
        let mut ntokens = min_tokens;
        while ntokens <= max_tokens {
            if literal_encoding.reserved_tokens() <= ntokens
                && !(ntokens >= 128 && literal_encoding == LiteralEncoding::Bits1)
                && !(ntokens > 256 && literal_encoding == LiteralEncoding::Bits2)
                && !(ntokens < 256 && literal_encoding == LiteralEncoding::All)
                && !(ntokens >= 1024 && processing == "raw")
            {
                let mut block = ntokens / 2;
                while block * block >= ntokens {
                    block /= 2;
                }

                println!(
                    "Optimizing tokens for processing '{}', literals {}, ntokens {}, block {}",
                    processing, literal_encoding, ntokens, block
                );

                let token_set = if let Some(prev_token_set) =
                    load_prev_token_set(tokens_dir, ntokens, processing, literal_encoding)
                {
                    prev_token_set
                } else {
                    TokenSet::new(literal_encoding)
                };

                let output_filename =
                    format!("tokens{}_{}_{}.json", ntokens, processing, literal_encoding);
                let output_path = tokens_dir_path.join(output_filename);

                optimize_token_set(
                    initial_size,
                    token_set,
                    slow_sampler,
                    sampler,
                    fast_sampler,
                    ntokens,
                    processing,
                    block,
                    &output_path,
                );
            }

            ntokens *= 2;
        }
    }
}
