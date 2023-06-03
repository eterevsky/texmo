use crate::sampler::Sampler;
use crate::stats::TokenStats;
use crate::tokenizer::tokenize_file;
use crate::tokens::TokenSet;

fn format_token(s: &[u8]) -> String {
    match String::from_utf8(s.to_vec()) {
        Ok(string) => format!("{:?}", string),
        Err(_) => format!("{:?}", s),
    }
}

fn tokens_to_add(token_set: &TokenSet, stats: &TokenStats, ntokens: usize, literal_cost: u64) -> Vec<Vec<u8>> {
    let mut token_values = Vec::new();

    for i in 0..256 {
        if stats.literal_count[i] > 0 {
            token_values.push((vec![i as u8], stats.literal_count[i] * (literal_cost - 1)))
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

    token_values
        .iter()
        .take(ntokens)
        .map(|&(ref t, _)| t.clone())
        .collect()
}

fn token_to_remove<'a, S: Sampler<'a>>(
    token_set: &TokenSet,
    initial_cost: u64,
    sampler: &'a S,
) -> Option<(Vec<u8>, TokenStats)> {
    let mut token_set = token_set.clone();
    let stats = tokenize_file(&token_set, sampler);
    token_set.update_stats(&stats);

    let mut token_ids: Vec<usize> = (0..token_set.tokens.len()).collect();
    token_ids.sort_unstable_by_key(|&i| stats.token_count[i]);

    // We construct a list of tokens to remove because the ids will change when we are removing and adding
    // tokens.
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
        let token_str = token_str.as_slice();
        tries += 1;

        token_set.remove_token(token_str);
        let stats = tokenize_file(&token_set, sampler);
        token_set.update_stats(&stats);

        // if stats.cost < initial_stats.cost
        //     && stats.token_to_add(&new_token_set) == token_str
        // {
        //     println!("Cost after removing {} would be {}, but it would be added on the next iteration.", format_token(&token_str), stats.cost);
        // }

        if stats.cost(token_set.literal_cost()) < initial_cost
        // && stats.token_to_add(&new_token_set) != token_str
        {
            // Found a token to remove.
            println!(
                "removing {} after {} tries",
                format_token(&token_str),
                tries
            );
            return Some((token_str.to_vec(), stats));
        }

        token_set.add_token(&token_str);
    }

    None
}

pub fn optimize_bpe<'a, S: Sampler<'a>>(
    token_set: &TokenSet,
    ntokens: usize,
    sampler: &'a S,
    add_block: usize,
) -> (TokenSet, TokenStats) {
    let mut token_set = token_set.clone();
    let mut prev_stats = None;

    loop {
        let initial_stats = match prev_stats {
            Some(s) => s,
            None => tokenize_file(&token_set, sampler),
        };
        token_set.update_stats(&initial_stats);

        println!(
            "{} tokens, bytes/(tokens+literals) = {:.3}  literals/bytes = {:.5}",
            token_set.ntokens(),
            initial_stats.scanned_bytes as f64
                / (initial_stats.total_tokens() + initial_stats.total_literals()) as f64,
            initial_stats.total_literals() as f64 / initial_stats.scanned_bytes as f64,
        );
        let new_tokens = if token_set.ntokens() + add_block > ntokens {
            tokens_to_add(&token_set, &initial_stats, 1, token_set.literal_cost())
        } else {
            tokens_to_add(&token_set, &initial_stats, add_block, token_set.literal_cost())
        };
        let mut new_token_set = token_set.clone();
        for token_str in new_tokens {
            println!("adding {}", format_token(&token_str));
            new_token_set.add_token(&token_str);
        }
        prev_stats = None;

        if new_token_set.ntokens() > ntokens {
            if let Some((to_remove, stats)) = token_to_remove(
                &new_token_set,
                initial_stats.cost(token_set.literal_cost()),
                sampler,
            ) {
                new_token_set.remove_token(&to_remove);
                token_set = new_token_set;
                prev_stats = Some(stats);
            } else {
                return (token_set, initial_stats);
            }
        } else {
            token_set = new_token_set;
        }
    }
}
