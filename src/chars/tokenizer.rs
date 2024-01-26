use crate::tokens::Token;

use super::token_stats::CharsTokenStats;
use super::tokens::CharsTokenIdx;
use super::tokens::CharsTokenSet;

#[derive(Clone, Copy, Debug, PartialEq)]
enum TokenId {
    Token(u32),
    Literal(char),
    Invalid,
    Start,
}

#[derive(Clone, Copy, Debug)]
struct TokenizerState {
    cost: u64,
    token: TokenId,
}

pub struct CharsTokenizer {
    pub token_set: CharsTokenSet,
}

impl CharsTokenizer {
    pub fn new(token_set: CharsTokenSet) -> Self {
        CharsTokenizer { token_set }
    }

    pub fn process_slice(&self, bytes: &[u8]) -> CharsTokenStats {
        let mut state = vec![TokenizerState {
            cost: 0,
            token: TokenId::Start,
        }];
        let invalid = TokenizerState {
            cost: 0,
            token: TokenId::Invalid,
        };
        let max_bytes_in_token = self.token_set.max_bytes_in_token();

        for pos in 1..=bytes.len() {
            if pos < bytes.len() && bytes[pos] >= 128 && bytes[pos] < 192 {
                state.push(invalid);
                continue;
            }

            let mut best_cost = None;
            let mut best_token = TokenId::Invalid;

            dbg!(pos);
            for from in pos.saturating_sub(max_bytes_in_token)..=(pos - 1) {
                let prev_state = state[from];
                dbg!(from, prev_state);
                if let TokenId::Invalid = prev_state.token {
                    continue;
                }
                let s = &bytes[from..pos];
                let (cost, token) = if let Ok(string) = String::from_utf8(s.to_vec()) {
                    dbg!(&string);
                    if let Some(token_idx) = self.token_set.token_by_str(&string) {
                        dbg!(token_idx);
                        (prev_state.cost + 1, TokenId::Token(token_idx.id()))
                    } else {
                        if string.chars().count() == 1 {
                            let ch = string.chars().next().unwrap();
                            let ch_cost = self.token_set.char_cost(ch);
                            (prev_state.cost + ch_cost as u64, TokenId::Literal(ch))
                        } else {
                            continue;
                        }
                    }
                } else {
                    continue;
                };

                if best_cost.is_none() || cost < best_cost.unwrap() {
                    best_cost = Some(cost);
                    best_token = token;
                }
            }

            assert!(best_token != TokenId::Invalid);
            let new_state = TokenizerState {
                cost: best_cost.unwrap(),
                token: best_token,
            };
            dbg!(&new_state);
            state.push(new_state);

        }

        let mut stats = CharsTokenStats::new(self.token_set.clone());

        let mut pos = bytes.len();
        while pos > 0 {
            let s = state[pos];
            match s.token {
                TokenId::Token(idx) => {
                    stats.count_token(idx as usize);
                    pos -= self.token_set.tokens[idx as usize].bytes_len();
                }
                TokenId::Literal(ch) => {
                    stats.count_literal(ch);
                    pos -= ch.len_utf8();
                }
                TokenId::Invalid => unreachable!(),
                TokenId::Start => unreachable!(),
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chars::tokens::CharsTokenSet;

    #[test]
    fn test_tokenize() {
        let mut token_set = CharsTokenSet::new(2);
        let a_id = token_set.add_char_token('A');
        let b_id = token_set.add_char_token('B');
        token_set.add_encoding('C', vec![b_id, CharsTokenSet::ext_token(0)]);
        token_set.add_encoding(
            'D',
            vec![
                b_id,
                CharsTokenSet::ext_token(0),
                CharsTokenSet::ext_token(1),
            ],
        );
        token_set.add_string("DA");
        token_set.add_string("AAA");

        let tokenizer = CharsTokenizer::new(token_set);

        let stats = tokenizer.process_slice("DAAA".as_bytes());
        assert_eq!(stats.total_tokens(), 3);
        assert_eq!(stats.total_literals(), 2);
    }
}
