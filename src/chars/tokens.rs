use std::{collections::HashMap, fmt};

#[derive(Clone, Debug)]
enum CharsToken {
    /// Used as extensions after an Ext token to specify characters for which
    /// there are no dedicated tokens.
    Ext(u8),
    /// A single character token. If followed by a Char or Str token, it
    /// indicates the `ch` character. Otherwise, it is combined with a number
    /// of following Ext tokens to indicate a character in the range from
    /// `from` to the `from` character of the next Char token.
    Char(char),
    /// Tokens indicating a given string.
    Str(String),
}

impl fmt::Display for CharsToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CharsToken::Ext(idx) => write!(f, "{}", idx),
            CharsToken::Char(ch) => write!(f, "{:?}", *ch),
            CharsToken::Str(s) => write!(f, "{:?}", s),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CharsTokenIdx(u32);

const HI_CHAR_THRESHOLD: usize = 256;

#[derive(Clone, Debug)]
pub struct CharsTokenSet {
    tokens: Vec<CharsToken>,
    lo_chars_enc: [Vec<CharsTokenIdx>; HI_CHAR_THRESHOLD],
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
            lo_chars_enc: [(); HI_CHAR_THRESHOLD].map(|_| Vec::new()),
            hi_chars_enc: HashMap::new(),
        }
    }

    pub fn add_char_token(&mut self, ch: char) -> CharsTokenIdx {
        let idx = CharsTokenIdx(self.tokens.len() as u32);
        self.tokens.push(CharsToken::Char(ch));

        self.add_encoding(ch, vec![idx]);

        idx
    }

    pub fn add_encoding(&mut self, ch: char, enc: Vec<CharsTokenIdx>) {
        if (ch as usize) < HI_CHAR_THRESHOLD {
            self.lo_chars_enc[ch as usize] = enc;
        } else {
            self.hi_chars_enc.insert(ch, enc);
        }
    }

    pub fn ext_token(idx: u32) -> CharsTokenIdx { CharsTokenIdx(idx) }

    pub fn char_encoding<'a>(&'a self, ch: char) -> &'a [CharsTokenIdx] {
        // TODO: fix for missing chars
        if (ch as usize) < HI_CHAR_THRESHOLD {
            &self.lo_chars_enc[ch as usize]
        } else {
            self.hi_chars_enc.get(&ch).unwrap()
        }
    }
}

impl fmt::Display for CharsTokenSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, token) in self.tokens.iter().enumerate() {
            writeln!(f, "{}  {}", i, token)?;
        }

        writeln!(f);

        for (ch, enc) in self.lo_chars_enc.iter().enumerate() {
            if !enc.is_empty() {
                let ch = char::from_u32(ch as u32).unwrap();
                write!(f, "{:?} ", ch)?;
                for &CharsTokenIdx(idx) in enc {
                    write!(f, " {}", self.tokens[idx as usize])?;
                }
                writeln!(f)?;
            }
        }

        let mut keys: Vec<_> = self.hi_chars_enc.keys().collect();
        keys.sort();

        for ch in keys {
            let enc = self.hi_chars_enc.get(ch).unwrap();
            write!(f, "{:?} ", ch);
            for &CharsTokenIdx(idx) in enc {
                write!(f, " {}", self.tokens[idx as usize])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}