use std::{collections::HashMap, str::Chars};

#[derive(Clone, Debug)]
enum CharsToken {
    /// Used as extensions after an Ext token to specify characters for which
    /// there are no dedicated tokens.
    Ext(u8),
    /// A single character token. If followed by a Char or Str token, it
    /// indicates the `ch` character. Otherwise, it is combined with a number
    /// of following Ext tokens to indicate a character in the range from
    /// `from` to the `from` character of the next Char token.
    Char{ch: char, from: char},
    /// Tokens indicating a given string.
    Str(String),
}

struct CharsTokenIdx(u32);

const HI_CHAR_THRESHOLD: char = '\u{1000}';

struct CharsTokenSet {
    tokens: Vec<CharsToken>,
    lo_chars_enc: Vec<Vec<CharsTokenIdx>>,
    hi_chars_enc: HashMap<char, Vec<CharsTokenIdx>>,
}

impl CharsTokenSet {
    pub fn new(num_ext: usize) -> Self {
        let mut tokens = Vec::new();
        for i in 0..num_ext {
            tokens.push(CharsToken::Ext(i as u8));
        }

        CharsTokenSet {
            tokens,
            lo_chars_enc: Vec::new(),
            hi_chars_enc: HashMap::new(),
        }
    }
}