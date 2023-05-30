use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use clap::{Parser, Subcommand};

mod optimizer;
mod stats;
mod tokenizer;
mod tokens;

use self::optimizer::optimize_bpe;
use self::tokens::TokenSet;
use self::tokenizer::{CHUNK_SIZE, tokenize_file};


fn optimize_tokens(
    data_filename: &str,
    input_tokens: &Option<String>,
    output_tokens: &Option<String>,
    ntokens: usize,
    fallback_dist: bool,
    fallback_bits: Option<usize>,
    initial_size: Option<u64>,
    processing: &Option<String>,
) {
    println!(
        "Using {} threads",
        std::thread::available_parallelism().unwrap().get()
    );

    let token_set = if let Some(tokens_file) = input_tokens {
        TokenSet::from_json(tokens_file.as_str())
    } else {
        if fallback_dist {
            TokenSet::build_with_dist_fallback([1; 256])
        } else {
            TokenSet::build_with_fallback_bits(fallback_bits.unwrap())
        }
        // if fallback.unwrap() == 2 {
        //     TokenSet::build_with_bin_literals()
        // } else if fallback.unwrap() == 4 {
        //     TokenSet::build_with_quad_literals()
        // } else {
        //     TokenSet::build_with_hex_literals()
        // }
    };

    let (token_set, token_stats) = optimize_bpe(&token_set, ntokens, data_filename);

    let initial_size = initial_size.unwrap_or(token_stats.scanned_bytes);
    let mut tokens_json = token_set.to_json(&token_stats, initial_size);

    let mut processing_json: Vec<json::JsonValue> = Vec::new();
    if let Some(processing) = processing {
        for stage in processing.split(",") {
            if !stage.is_empty() {
                processing_json.push(stage.into());
            }
        }
    }

    tokens_json["processing"] = processing_json.into();

    let tokens_json_str = json::stringify_pretty(tokens_json, 2);
    println!("{}", &tokens_json_str);

    if let Some(out) = output_tokens {
        std::fs::write(&out, &tokens_json_str).unwrap();
    }
}

fn filter_text(filename: &str, caps: bool, words: bool, output: &str) {
    let input = File::open(filename).unwrap();
    let mut output = File::create(output).unwrap();

    let reader = BufReader::new(input);
    let mut writer = BufWriter::new(&mut output);

    let mut word = Vec::new();

    for line in reader.lines() {
        // println!("{:?}", line);
        let line = line.unwrap();
        let mut out_line = Vec::new();
        let mut hanging_space = false;
        word.clear();

        for ch in line.chars() {
            // println!("{:?}", ch);
            if ch.is_alphabetic() {
                hanging_space = false;
                word.push(ch);
            } else {
                if !word.is_empty() {
                    if caps {
                        if word[0].is_uppercase() && word[1..].iter().all(|&c| c.is_lowercase()) {
                            out_line.push('\x14');
                            word[0] = word[0].to_lowercase().next().unwrap();
                        } else if word.iter().all(|&c| c.is_uppercase()) {
                            out_line.push('\x15');
                            for i in 0..word.len() {
                                word[i] = word[i].to_lowercase().next().unwrap();
                            }
                        }
                    }

                    out_line.extend(word.iter());
                    word.clear();

                    if words {
                        out_line.push('\x16');
                    }

                    if words && ch == ' ' {
                        hanging_space = true;
                    } else {
                        out_line.push(ch);
                        hanging_space = false;
                    }
                } else {
                    if hanging_space {
                        out_line.push(' ');
                    }
                    hanging_space = false;
                    out_line.push(ch);
                }
            }
        }

        if !word.is_empty() {
            if caps {
                if word[0].is_uppercase() && word[1..].iter().all(|&c| c.is_lowercase()) {
                    out_line.push('\x14');
                    word[0] = word[0].to_lowercase().next().unwrap();
                } else if word.iter().all(|&c| c.is_uppercase()) {
                    out_line.push('\x15');
                    for i in 0..word.len() {
                        word[i] = word[i].to_lowercase().next().unwrap();
                    }
                }
            }

            out_line.extend(word.iter());

            if words {
                out_line.push('\x16');
            }
        }

        if hanging_space {
            out_line.push(' ');
        }

        out_line.push('\n');
        let output_string: String = out_line.iter().collect();

        writer.write_all(output_string.as_bytes()).unwrap();
    }
}

fn count_hex_digits(filename: &str) {
    let input = File::open(filename).unwrap();
    let reader = BufReader::new(input);

    let mut counts: [usize; 16] = [0; 16];

    for byte in reader.bytes() {
        let byte = byte.unwrap();
        counts[(byte >> 4) as usize] += 1;
        counts[(byte & 15) as usize] += 1;
    }

    let mut digits = (0..16).collect::<Vec<usize>>();
    digits.sort_unstable_by_key(|&d| counts[d]);

    for &d in digits.iter() {
        println!("{:x}  {}", d, counts[d]);
    }
}

fn count_bytes(filename: &str) -> [u64; 256] {
    let mut file = File::open(filename).unwrap();
    let mut buffer = Vec::new();
    buffer.resize(CHUNK_SIZE, 0);

    let mut counts = [0; 256];

    let mut read_bytes = file.read(&mut buffer).unwrap();
    while read_bytes > 0 {
        for byte in buffer[..read_bytes].iter() {
            counts[*byte as usize] += 1;
        }
        read_bytes = file.read(&mut buffer).unwrap();
    }

    counts
}

fn count_bytes_command(filename: &str) {
    let counts = count_bytes(filename);

    for c in counts.iter() {
        print!(" {}", c);
    }
    println!();
}

fn tokenize(filename: &str, tokens_file: &str, initial_size: u64) {
    let token_set = TokenSet::from_json(tokens_file);
    let stats = tokenize_file(&token_set, filename);
    let token_set_json = token_set.to_json(&stats, initial_size);
    println!("{}", json::stringify_pretty(token_set_json, 2));
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    data: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Tokenize {
        #[arg(short, long)]
        tokens: String,

        /// If the input is pre-processed, this argument specifies the size
        /// of the input before pre-processing.
        #[arg(long)]
        initial_size:u64,
    },

    OptimizeTokens {
        #[arg(short, long)]
        input_tokens: Option<String>,

        #[arg(short, long)]
        output_tokens: Option<String>,

        #[arg(short, long)]
        ntokens: usize,

        #[arg(long)]
        fallback_dist: bool,

        /// Add tokens for 1, 2 or 4-bit blocks to encode bytes that aren't
        /// tokens.
        #[arg(short, long)]
        fallback_bits: Option<usize>,

        // /// Fallback encoding for bytes that aren't tokens. Possible values:
        // /// 2, 4, 16
        // #[arg(long)]
        // fallback: Option<usize>,

        /// If the input is pre-processed, this argument specifies the size
        /// of the input before pre-processing.
        #[arg(long)]
        initial_size: Option<u64>,

        /// A comma-separated list of the processing stages applied to the input
        /// for the purposes of including it into the output.
        #[arg(long)]
        processing: Option<String>,
    },

    FilterText {
        #[arg(short, long)]
        caps: bool,

        #[arg(short, long)]
        words: bool,

        #[arg(short, long)]
        output: String,
    },

    CountHexDigits,
    CountBytes,
}

fn main() {
    let args = Args::parse();
    let filename = args.data.as_str();

    match &args.command {
        Command::Tokenize { tokens, initial_size } => tokenize(filename, tokens, *initial_size),

        Command::OptimizeTokens {
            input_tokens,
            output_tokens,
            ntokens,
            fallback_dist,
            fallback_bits,
            initial_size,
            processing,
        } => optimize_tokens(
            filename,
            input_tokens,
            output_tokens,
            *ntokens,
            *fallback_dist,
            *fallback_bits,
            *initial_size,
            processing,
        ),

        Command::FilterText {
            caps,
            words,
            output,
        } => filter_text(filename, *caps, *words, output.as_str()),

        Command::CountHexDigits => count_hex_digits(filename),

        Command::CountBytes => count_bytes_command(filename),
    }
}
