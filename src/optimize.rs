use crate::batch_tokenize::tokenize_file;
use crate::input::sample::Sampler;
use crate::processing::Processing;
use crate::stats2::TokenStats;
use crate::tokenset::{Token, TokenSet, TokenType};

trait BytesOptimizer {
    fn optimize_bytes(token_stats: &TokenStats, n_byte_tokens: usize) -> TokenSet;
}

struct SimpleBytesOptimizer {}

impl BytesOptimizer for SimpleBytesOptimizer {
    fn optimize_bytes(stats: &TokenStats, n_byte_tokens: usize) -> TokenSet {
        let mut byte_counts: [i64; 256] = [0; 256];
        let mut single_byte_tokens = Vec::new();

        for (token_id, token) in stats.token_set.tokens.iter().enumerate() {
            if let Token::Str(s) = token {
                if s.len() == 1 {
                    byte_counts[s[0] as usize] = stats.token_counts[token_id] as i64;
                    single_byte_tokens.push(s.clone());
                }
            }
        }

        for (seq_id, seq) in stats.token_set.sequences.iter().enumerate() {
            if seq.string.len() == 1 {
                byte_counts[seq.string[0] as usize] = stats.seq_counts[seq_id] as i64;
            }
        }

        let mut bytes = (0..=255).collect::<Vec<u8>>();
        bytes.sort_by_key(|&i| -byte_counts[i as usize]);
        let selected_bytes = &bytes[..n_byte_tokens];

        let mut new_token_set = match stats.token_set.token_type {
            TokenType::Bits1 => {
                TokenSet::new_bits1(stats.token_set.processing, stats.token_set.split_paragraphs)
            }
            TokenType::Bits2 => {
                TokenSet::new_bits2(stats.token_set.processing, stats.token_set.split_paragraphs)
            }
            TokenType::Bits4 => {
                TokenSet::new_bits4(stats.token_set.processing, stats.token_set.split_paragraphs)
            }
            _ => unreachable!(),
        };

        for &byte in selected_bytes.iter() {
            new_token_set.add_token(&[byte]);
        }

        for token in stats.token_set.tokens.iter() {
            if let Token::Str(s) = token {
                if s.len() > 1 {
                    new_token_set.add_token(s);
                }
            }
        }

        new_token_set
    }
}

struct NoopBytesOptimizer {}

impl BytesOptimizer for NoopBytesOptimizer {
    fn optimize_bytes(token_stats: &TokenStats, _n_byte_tokens: usize) -> TokenSet {
        token_stats.token_set.clone()
    }
}

fn optimize_tokenset_impl<'a, S: Sampler<'a>, BO: BytesOptimizer>(
    token_set: TokenSet,
    ntokens: usize,
    sampler: &'a S,
    _bytes_optimizer: &BO,
    initial_size: Option<u64>,
) -> TokenStats {
    let stats = tokenize_file(&token_set, sampler, initial_size);
    let new_token_set = BO::optimize_bytes(&stats, ntokens - token_set.n_ext_tokens);
    tokenize_file(&new_token_set, sampler, initial_size)
}

pub fn optimize_tokenset<'a, S: Sampler<'a>>(
    ntokens: usize,
    sampler: &'a S,
    processing: Processing,
    token_type: TokenType,
    initial_size: Option<u64>,
) -> TokenStats {
    let bytes_optimizer = SimpleBytesOptimizer {};
    match token_type {
        TokenType::Bits1 => {
            let token_set = TokenSet::new_bits1(processing, true);
            optimize_tokenset_impl(token_set, ntokens, sampler, &bytes_optimizer, initial_size)
        }
        TokenType::Bits2 => {
            let token_set = TokenSet::new_bits2(processing, true);
            optimize_tokenset_impl(token_set, ntokens, sampler, &bytes_optimizer, initial_size)
        }
        TokenType::Bits4 => {
            let token_set = TokenSet::new_bits4(processing, true);
            optimize_tokenset_impl(token_set, ntokens, sampler, &bytes_optimizer, initial_size)
        }
        TokenType::Bytes => {
            let token_set = TokenSet::new_bits4(processing, true);
            let noop_bytes_optimizer = NoopBytesOptimizer {};
            optimize_tokenset_impl(
                token_set,
                ntokens,
                sampler,
                &noop_bytes_optimizer,
                initial_size,
            )
        }
        TokenType::BytesHuff => unimplemented!(),
        TokenType::CharsHuff => unimplemented!(),
    }
}
