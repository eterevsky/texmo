use std::cmp::min;
use std::collections::HashMap;

use crate::sampler::{Sampler, SelectionSampler};
use crate::stats::TokenStats;
use crate::tokenizer::tokenize_file;
use crate::tokens::TokenSet;

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

    for token_str in token_strs.iter() {
        println!("Trying to remove {}         ", format_token(token_str));
        last_token_check.insert(token_str.clone(), step);
        let token_str = token_str.as_slice();
        tries += 1;

        let mut new_token_set = token_set.clone();

        new_token_set.remove_token(token_str);

        let added = add_tokens(tokenizer, &mut new_token_set, 1);

        if added[0] == token_str {
            println!("Same token added again, skipping.");
            continue;
        }

        let new_stats = tokenizer.get_stats(&new_token_set);

        if new_stats.cost() < initial_stats.cost() {
            println!(
                "Replacing {} -> {} after {} tries",
                format_token(&token_str),
                format_token(added[0].as_slice()),
                tries
            );
            return Some(new_token_set);
        }
    }

    None
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

    while token_set.ntokens() < ntokens {
        let tokens_to_add = min(add_block, ntokens - token_set.ntokens());
        let added = add_tokens(&mut tokenizer, &mut token_set, tokens_to_add);
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
