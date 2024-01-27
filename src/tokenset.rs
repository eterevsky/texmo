use super::processing::Processing;
use serde_json::{json, Value};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenType {
    /// All bytes have their own tokens.
    Bytes,
    /// Ext tokens 0 and 1 are used to encode bytes bit by bit.
    Bits1,
    /// Ext tokens 0..4 are used to encode bytes.
    Bits2,
    /// Ext tokens 0..16 are used to encode bytes.
    Bits4,
    /// Missing bytes are represented as sequences of ext tokens, based on
    /// their frequency.
    BytesHuff,
    /// All Unicode characters are represented as sequences
    CharsHuff,
}

/// A single token that will be part of the final tokenization.
enum Token {
    /// A token representing a string of bytes.
    Str(Vec<u8>),
    /// A token that doesn't correspond to a specific string.
    Ext(u8),
}

fn bytes_to_json(bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.into(),
        Err(_) => json!(bytes.clone()),
    }
}

impl Into<Value> for Token {
    fn into(self) -> Value {
        match self {
            Token::Ext(n) => n.into(),
            Token::Str(bytes) => bytes_to_json(&bytes),
        }
    }
}
impl Into<Value> for &Token {
    fn into(self) -> Value {
        match self {
            Token::Ext(n) => (*n).into(),
            Token::Str(bytes) => bytes_to_json(bytes),
        }
    }
}

/// A substring of text covered by one token or a sequence of tokens in such
/// a way that it couldn't be subdivided into smaller strings
struct Span {
    string: Vec<u8>,
    /// Sequence of one or more token indices that encode this string.
    tokens: Vec<usize>,
    /// The longest suffix of `string` which is itself a different Span.
    suffix: Option<SpanIdx>,
}

impl Span {
    fn to_json(&self, token_set: &TokenSet) -> Value {
        let seq = self.tokens.iter().map(
            |idx| (&token_set.tokens[*idx]).into()
        ).collect::<Vec<Value>>();
        json!({
            "string": bytes_to_json(self.string.as_slice()),
            "tokens": seq,
        })
    }
}

struct SpanIdx(usize);

struct TokenSet {
    /// The type of the token set, specifying how it encodes bytes or characters
    /// that don't have specific tokens associated with them.
    token_type: TokenType,
    /// Type of pre-processing that should be done to the text before tokenization.
    processing: Processing,
    /// If true, the tokens can span accross paragraphs, i.e. a token can't have
    /// any non '\n' characters after "\n\n".
    split_paragraphs: bool,

    tokens: Vec<Token>,
    spans: Vec<Span>,
}

impl TokenSet {
    pub fn new(
        n_ext_tokens: usize,
        processing: Processing,
        token_type: TokenType,
        split_paragraphs: bool,
    ) -> Self {
        assert!(n_ext_tokens < 256);
        let tokens = (0..n_ext_tokens)
            .map(|i| Token::Ext(i as u8))
            .collect::<Vec<_>>();

        TokenSet {
            token_type,
            processing,
            tokens,
            spans: Vec::new(),
            split_paragraphs,
        }
    }

    pub fn new_bits4(processing: Processing, split_paragraphs: bool) -> Self {
        let mut token_set = Self::new(16, processing, TokenType::Bits4, split_paragraphs);
        for c in 0..256 {
            token_set.add_span(vec![c as u8], vec![c >> 4, c & 15]);
        }

        token_set
    }

    pub fn add_span(&mut self, string: Vec<u8>, tokens: Vec<usize>) {
        let span = Span { string, tokens, suffix: None };
        self.spans.push(span);
    }

    pub fn add_token(&mut self, token: &[u8]) {
        self.spans.retain(|s| s.string != token);
        let token = Token::Str(token.to_vec());
        self.tokens.push(token);
    }

    pub fn ntokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "type": self.token_type,
            "processing": self.processing,
            "tokens": self.tokens.iter().map(|t| t.into()).collect::<Vec<Value>>(),
            "split_paragraphs": self.split_paragraphs,
        });
        let sequences = self
            .spans
            .iter()
            .filter(|s| s.tokens.len() > 1)
            .map(|s| s.to_json(self))
            .collect::<Vec<_>>();
        if !sequences.is_empty() {
            value["sequences"] = json!(sequences);
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn token_to_json() {
        let token_value: Value = Token::Ext(127).into();
        assert_eq!(token_value, Value::Number(127.into()));
        let token_value: Value = Token::Str(vec![0xd0, 0xb0, 0xd0, 0xb1]).into();
        assert_eq!(token_value, Value::String("аб".to_string()));
        let token_value: Value = Token::Str(vec![0xb0, 0xff]).into();
        assert_eq!(
            token_value,
            Value::Array(vec![Value::Number(0xb0.into()), Value::Number(0xff.into())])
        );
    }

    #[test]
    fn token_set_to_json() {
        let mut token_set = TokenSet::new_bits4(Processing::Raw, true);
        token_set.add_token("a".as_bytes());
        token_set.add_token("b".as_bytes());
        token_set.add_token("c".as_bytes());

        println!("{}", serde_json::to_string_pretty(&token_set.to_json()).unwrap());

    }
}
