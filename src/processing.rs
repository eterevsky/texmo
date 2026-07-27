use clap::ValueEnum;
use serde::Serialize;
use std::fmt;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Processing {
    Raw,
    CapsWords,
    /// capswords plus `\x17` before each mid-word capital, so no
    /// uppercase letter survives processing and a tokenset never has
    /// to spend slots on them. Used by the `hexbpe` sets.
    CapsWords2,
}

impl fmt::Display for Processing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Processing::Raw => "raw",
                Processing::CapsWords => "capswords",
                Processing::CapsWords2 => "capswords2",
            }
        )
    }
}

enum CharType {
    Letter,
    NonLetter,
    Space,
}

fn get_char_type(ch: char) -> CharType {
    if ch.is_alphabetic() {
        CharType::Letter
    } else if ch == ' ' {
        CharType::Space
    } else {
        CharType::NonLetter
    }
}

fn add_word(out: &mut String, word: &str) {
    assert!(!word.is_empty());

    let mut chars = word.chars();
    let first = chars.next().unwrap();
    let rest = chars.as_str();

    if first.is_uppercase() {
        if rest.chars().all(|ch| ch.is_lowercase()) {
            out.push('\x14');
            out.push(first.to_lowercase().next().unwrap());
            out.push_str(rest);
        } else if rest.chars().all(|ch| ch.is_uppercase()) {
            out.push('\x15');
            out.push_str(word.to_lowercase().as_str());
        } else {
            out.push_str(word);
        }
    } else {
        out.push_str(word);
    }

    out.push('\x16');
}

enum State {
    Word,
    SpaceAfterWord,
    NonWord,
}

/// Processes the text by making the following changes:
///
/// 1. Adds a character `\x16` at the end of each word (a sequence of letter characters, before either a non-letter character or the end of the text).
/// 2. Removes a single space between words. In the sequence <letter> `\x16` <space> <letter>, the space is removed.
/// 3. A capitalized word (a word starting with a capital letter, with remaining letters lowercase) is replaced by a `\x14` character followed by the lowercase version of the word.
/// 4. An all-uppercase word is replaced by a `\x15` character followed by the lowercase version of the word.
pub fn process(text: &str) -> String {
    let mut out = String::with_capacity(2 * text.len());
    let mut state = State::NonWord;
    let mut word = String::new();

    for ch in text.chars() {
        state = match (state, get_char_type(ch)) {
            (State::NonWord, CharType::Letter) => {
                word.push(ch);
                State::Word
            }
            (State::NonWord, CharType::Space | CharType::NonLetter) => {
                out.push(ch);
                State::NonWord
            }
            (State::Word, CharType::Letter) => {
                word.push(ch);
                State::Word
            }
            (State::Word, CharType::Space) => {
                add_word(&mut out, &word);
                word.clear();
                State::SpaceAfterWord
            }
            (State::Word, CharType::NonLetter) => {
                add_word(&mut out, &word);
                word.clear();
                out.push(ch);
                State::NonWord
            }
            (State::SpaceAfterWord, CharType::Letter) => {
                // Skipping the space character, since there was only one space between the words.
                word.push(ch);
                State::Word
            }
            (State::SpaceAfterWord, CharType::Space | CharType::NonLetter) => {
                out.push(' ');
                out.push(ch);
                State::NonWord
            }
        };
    }

    match state {
        State::Word => add_word(&mut out, &word),
        State::SpaceAfterWord => out.push(' '),
        State::NonWord => {}
    }

    out
}

pub fn process_file<R: Read, W: Write>(input: &mut R, output: &mut W) -> io::Result<()> {
    let reader = BufReader::new(input);
    let mut writer = BufWriter::new(output);

    for line in reader.lines() {
        let line = line?;
        let processed = process(&line);
        writer.write_all(processed.as_bytes())?;
        writer.write(b"\n")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_process() {
        assert_eq!(super::process("Hello, world!"), "\x14hello\x16, world\x16!");
        assert_eq!(super::process("hello, world!"), "hello\x16, world\x16!");
        assert_eq!(super::process("HELLO, world!"), "\x15hello\x16, world\x16!");
        assert_eq!(super::process("HeLLo, world!"), "HeLLo\x16, world\x16!");
        assert_eq!(super::process("Hello world!"), "\x14hello\x16world\x16!");
        assert_eq!(
            super::process("Hello , world!"),
            "\x14hello\x16 , world\x16!"
        );
        assert_eq!(super::process("Hello, world "), "\x14hello\x16, world\x16 ");
        assert_eq!(super::process("Hello, world"), "\x14hello\x16, world\x16");
        assert_eq!(
            super::process("Hello, World"),
            "\x14hello\x16, \x14world\x16"
        );
        assert_eq!(super::process("Hello World"), "\x14hello\x16\x14world\x16");
        assert_eq!(super::process("Hello WORLD"), "\x14hello\x16\x15world\x16");
    }
}

// --- capswords2 ---------------------------------------------------------
//
// Mirrors `texmo/tokens/processing.py::process2` exactly. The Rust side
// builds tokensets and the Python side tokenizes with them, so any
// divergence shows up only as silently worse compression.

fn add_word2(out: &mut String, word: &str) {
    assert!(!word.is_empty());

    let mut chars = word.chars();
    let first = chars.next().unwrap();
    let rest = chars.as_str();

    if first.is_uppercase() && rest.chars().all(|ch| ch.is_lowercase()) {
        out.push('\x14');
        out.push_str(word.to_lowercase().as_str());
    } else if first.is_uppercase()
        && !rest.is_empty()
        && rest.chars().all(|ch| ch.is_uppercase())
    {
        out.push('\x15');
        out.push_str(word.to_lowercase().as_str());
    } else if word.chars().any(|ch| ch.is_uppercase()) {
        // Mixed case: one marker per capital, so the word itself is
        // all lowercase. capswords left these verbatim.
        for ch in word.chars() {
            if ch.is_uppercase() {
                out.push('\x17');
                for lower in ch.to_lowercase() {
                    out.push(lower);
                }
            } else {
                out.push(ch);
            }
        }
    } else {
        out.push_str(word);
    }

    out.push('\x16');
}

/// capswords2: like `process`, plus `\x17` before every mid-word capital.
pub fn process2(text: &str) -> String {
    let mut out = String::with_capacity(2 * text.len());
    let mut state = State::NonWord;
    let mut word = String::new();

    for ch in text.chars() {
        state = match (state, get_char_type(ch)) {
            (State::NonWord | State::SpaceAfterWord, CharType::Letter) => {
                // A single space before a word is elided: the previous
                // word's `\x16` already implies the boundary.
                word.push(ch);
                State::Word
            }
            (State::NonWord, CharType::Space | CharType::NonLetter) => {
                out.push(ch);
                State::NonWord
            }
            (State::Word, CharType::Letter) => {
                word.push(ch);
                State::Word
            }
            (State::Word, CharType::Space) => {
                add_word2(&mut out, &word);
                word.clear();
                State::SpaceAfterWord
            }
            (State::Word, CharType::NonLetter) => {
                add_word2(&mut out, &word);
                word.clear();
                out.push(ch);
                State::NonWord
            }
            (State::SpaceAfterWord, CharType::Space | CharType::NonLetter) => {
                out.push(' ');
                out.push(ch);
                State::NonWord
            }
        };
    }

    match state {
        State::Word => add_word2(&mut out, &word),
        State::SpaceAfterWord => out.push(' '),
        State::NonWord => {}
    }

    out
}

pub fn process_file2<R: Read, W: Write>(input: &mut R, output: &mut W) -> io::Result<()> {
    let reader = BufReader::new(input);
    let mut writer = BufWriter::new(output);

    for line in reader.lines() {
        let line = line?;
        writer.write_all(process2(&line).as_bytes())?;
        writer.write(b"\n")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests2 {
    use super::process2;

    #[test]
    fn test_process2_matches_capswords_where_they_overlap() {
        assert_eq!(process2("Hello, world!"), "\x14hello\x16, world\x16!");
        assert_eq!(process2("hello, world!"), "hello\x16, world\x16!");
        assert_eq!(process2("HELLO, world!"), "\x15hello\x16, world\x16!");
        assert_eq!(process2("Hello World"), "\x14hello\x16\x14world\x16");
        assert_eq!(process2("a  b"), "a\x16  b\x16");
    }

    #[test]
    fn test_process2_marks_mid_word_capitals() {
        // capswords passed these through verbatim.
        assert_eq!(process2("HeLLo"), "\x17he\x17l\x17lo\x16");
        assert_eq!(process2("iPhone"), "i\x17phone\x16");
        assert_eq!(process2("aB"), "a\x17b\x16");
        // Nothing uppercase survives.
        assert!(!process2("McDonald eBay HeLLo")
            .chars()
            .any(|c| c.is_uppercase()));
    }
}

#[cfg(test)]
mod capswords2_golden {
    use super::process2;
    use serde::Deserialize;
    use std::fs;

    /// One entry of `texmo/tokens/capswords2_cases.json`, the fixture
    /// shared with `texmo/tokens/processing_test.py`. Both languages
    /// must reproduce `out` exactly: Rust builds the tokensets and
    /// Python tokenizes with them, so a divergence never raises, it
    /// just silently costs compression.
    #[derive(Deserialize)]
    struct Case {
        #[serde(rename = "in")]
        input: String,
        out: String,
        #[allow(dead_code)]
        roundtrip: bool,
    }

    fn cases() -> Vec<Case> {
        // cargo runs tests from the crate root.
        let text = fs::read_to_string("texmo/tokens/capswords2_cases.json")
            .expect("shared capswords2 fixture missing");
        serde_json::from_str(&text).expect("malformed capswords2 fixture")
    }

    #[test]
    fn matches_the_shared_python_fixture() {
        let cases = cases();
        assert!(cases.len() >= 80, "fixture shrank unexpectedly");
        for case in &cases {
            assert_eq!(
                process2(&case.input),
                case.out,
                "capswords2 diverged from Python on input {:?}",
                case.input
            );
        }
    }

    fn has_marker(s: &str) -> bool {
        s.chars().any(|c| ('\x14'..='\x17').contains(&c))
    }

    #[test]
    fn no_uppercase_survives() {
        for case in cases().iter().filter(|c| !has_marker(&c.input)) {
            assert!(
                !case.out.chars().any(|c| c.is_uppercase()),
                "uppercase survived processing of {:?}",
                case.input
            );
        }
    }

    #[test]
    fn one_word_marker_per_word() {
        for case in cases().iter().filter(|c| !has_marker(&c.input)) {
            let mut words = 0;
            let mut in_word = false;
            for ch in case.input.chars() {
                if ch.is_alphabetic() {
                    if !in_word {
                        words += 1;
                    }
                    in_word = true;
                } else {
                    in_word = false;
                }
            }
            assert_eq!(
                case.out.matches('\x16').count(),
                words,
                "word marker count wrong for {:?}",
                case.input
            );
        }
    }

    #[test]
    fn single_inter_word_space_is_elided() {
        assert_eq!(process2("a b"), "a\x16b\x16");
        assert_eq!(process2("a  b"), "a\x16  b\x16");
        assert_eq!(process2("a   b"), "a\x16   b\x16");
    }

    #[test]
    fn uncased_scripts_pass_through() {
        assert_eq!(process2("日本語"), "日本語\x16");
        assert_eq!(process2("עברית"), "עברית\x16");
    }
}
