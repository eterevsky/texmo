use crate::stats::TokenStats;
use crate::tokenizer::tokenize_file;
use crate::tokens::TokenSet;
use crate::sampler::Sampler;

fn format_token(s: &[u8]) -> String {
    match String::from_utf8(s.to_vec()) {
        Ok(string) => format!("{:?}", string),
        Err(_) => format!("{:?}", s),
    }
}

fn token_to_add(token_set: &TokenSet, stats: &TokenStats) -> Vec<u8> {
    let mut top_literal = 0;
    let mut top_literal_count = 0;

    for i in 0..256 {
        if stats.literal_count[i] > top_literal_count {
            top_literal = i;
            top_literal_count = stats.literal_count[i];
        }
    }

    let mut top_pair = 0;
    let mut top_pair_count = 0;
    for ipair in 0..stats.pair_count.len() {
        if stats.pair_count[ipair] > top_pair_count {
            top_pair = ipair;
            top_pair_count = stats.pair_count[ipair];
        }
    }

    if top_literal_count as f64 * (token_set.literal_cost - 1.0) > top_pair_count as f64 {
        vec![top_literal as u8]
    } else {
        let ifirst = top_pair / token_set.tokens.len();
        let isecond = top_pair % token_set.tokens.len();

        let mut token_str = token_set.tokens[ifirst].string.clone();
        token_str.extend(token_set.tokens[isecond].string.clone());

        token_str
    }
}

pub fn optimize_bpe<'a, S: Sampler<'a>>(
    token_set: &TokenSet,
    ntokens: usize,
    sampler: &'a S,
) -> (TokenSet, TokenStats) {
    let mut token_set = token_set.clone();
    let mut prev_stats = None;

    loop {
        let initial_stats = match prev_stats {
            Some(s) => s,
            None => tokenize_file(&token_set, sampler),
        };
        token_set.update_stats(&initial_stats);

        let bytes_per_token = (initial_stats.scanned_bytes - initial_stats.total_literals()) as f64
            / (initial_stats.total_tokens() as f64 + 1.0);
        println!(
            "Literal entropy: {}, bytes per token: {}, cost: {}",
            token_set.dist_entropy(),
            bytes_per_token,
            token_set.literal_cost
        );

        let new_token_str = token_to_add(&token_set, &initial_stats);
        let mut new_token_set = token_set.clone();
        new_token_set.add_token(&new_token_str);
        prev_stats = None;

        println!(
            "{} Adding token {}",
            initial_stats.cost(token_set.literal_cost),
            format_token(&new_token_str)
        );

        if new_token_set.ntokens() > ntokens {
            let stats = tokenize_file(&new_token_set, sampler);
            new_token_set.update_stats(&stats);
            let mut token_ids: Vec<usize> = (0..new_token_set.tokens.len()).collect();
            token_ids.sort_unstable_by_key(|&i| stats.token_count[i]);

            // We construct a list of tokens to remove because the ids will change when we are removing and adding
            // tokens.
            let mut token_strs = Vec::new();

            for &token_id_to_remove in token_ids.iter() {
                let token_to_remove = &new_token_set.tokens[token_id_to_remove];
                if token_to_remove.is_mandatory {
                    continue;
                }
                token_strs.push(token_to_remove.string.clone());
            }

            let mut found = false;
            let mut tries = 0;

            for token_str in token_strs.iter() {
                let token_str = token_str.as_slice();
                tries += 1;

                new_token_set.remove_token(token_str);
                let stats = tokenize_file(&new_token_set, sampler);
                new_token_set.update_stats(&stats);

                // if stats.cost < initial_stats.cost
                //     && stats.token_to_add(&new_token_set) == token_str
                // {
                //     println!("Cost after removing {} would be {}, but it would be added on the next iteration.", format_token(&token_str), stats.cost);
                // }

                if stats.cost(new_token_set.literal_cost)
                    < initial_stats.cost(token_set.literal_cost)
                // && stats.token_to_add(&new_token_set) != token_str
                {
                    // Found a token to remove.
                    found = true;
                    prev_stats = Some(stats);
                    println!(
                        "Removing token {} after {} tries",
                        format_token(&token_str),
                        tries
                    );
                    break;
                }

                new_token_set.add_token(&token_str);
            }

            if !found {
                return (token_set, initial_stats);
            }
        }

        token_set = new_token_set;
    }
}
