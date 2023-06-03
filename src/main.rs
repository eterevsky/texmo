use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use clap::{Parser, Subcommand};

mod optimizer;
mod sampler;
mod stats;
mod tokenizer;
mod tokens;

use crate::sampler::SelectionSampler;

use self::optimizer::optimize_bpe;
use self::sampler::{FileSampler, MemorySampler};
use self::tokenizer::tokenize_file;
use self::tokens::{LiteralEncoding, TokenSet};

fn optimize_tokens(
    data_filename: &str,
    input_tokens: &Option<String>,
    output_tokens: &Option<String>,
    ntokens: usize,
    initial_size: Option<u64>,
    processing: &Option<String>,
    in_memory: bool,
    nchunks: Option<usize>,
    chunk_size: usize,
    add_block: usize,
    literal_encoding: LiteralEncoding,
) {
    println!(
        "Using {} threads",
        std::thread::available_parallelism().unwrap().get()
    );

    let token_set = if let Some(tokens_file) = input_tokens {
        TokenSet::from_json(tokens_file.as_str())
    } else {
        TokenSet::new(literal_encoding)
    };

    let file_size = std::fs::metadata(data_filename).unwrap().len();
    let initial_size = initial_size.unwrap_or(file_size);

    let fast_sampler = SelectionSampler::new(data_filename, 16384, 1024);

    let (token_set, token_stats) = if nchunks.is_some() {
        let nchunks = nchunks.unwrap();
        let sampler = SelectionSampler::new(data_filename, chunk_size, nchunks);
        let (token_set, _) = optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block);

        let full_sampler = FileSampler::new(data_filename, chunk_size);
        let stats = tokenize_file(&token_set, &full_sampler);

        (token_set, stats)
    } else if in_memory {
        let sampler = MemorySampler::new(data_filename, chunk_size);
        optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block)
    } else {
        let sampler = FileSampler::new(data_filename, chunk_size);
        optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block)
    };

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

pub const CHUNK_SIZE: usize = 16 * 1024 * 1024;

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

fn tokenize(
    filename: &str,
    tokens_file: &str,
    initial_size: u64,
    chunk_size: usize,
    in_memory: bool,
) {
    let token_set = TokenSet::from_json(tokens_file);

    let stats = if in_memory {
        let sampler = MemorySampler::new(filename, chunk_size);
        tokenize_file(&token_set, &sampler)
    } else {
        let sampler = FileSampler::new(filename, chunk_size);
        tokenize_file(&token_set, &sampler)
    };

    let stats_json = stats.to_json(
        initial_size,
        token_set.literal_cost(),
        token_set.dist_entropy(),
        token_set.reserved_tokens(),
    );
    println!("{}", json::stringify_pretty(stats_json, 2));
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
        initial_size: u64,

        #[arg(long, default_value_t = 1024 * 1024)]
        chunk_size: usize,

        #[arg(long)]
        in_memory: bool,
    },

    OptimizeTokens {
        #[arg(short, long)]
        input_tokens: Option<String>,

        #[arg(short, long)]
        output_tokens: Option<String>,

        #[arg(short, long)]
        ntokens: usize,

        #[arg(long, default_value_t=LiteralEncoding::Dist8)]
        literals: LiteralEncoding,

        /// If the input is pre-processed, this argument specifies the size
        /// of the input before pre-processing.
        #[arg(long)]
        initial_size: Option<u64>,

        /// A comma-separated list of the processing stages applied to the input
        /// for the purposes of including it into the output.
        #[arg(long)]
        processing: Option<String>,

        #[arg(long, default_value_t=16 * 1024 * 1024)]
        chunk_size: usize,

        /// Sample this number of training data chunks, keep them in memory
        /// and train the token set on them. Final stats will be calculated
        /// the whole dataset.
        #[arg(long)]
        nchunks: Option<usize>,

        /// Use in-memory sampler in case all training data fits in memory.
        /// Ignored when nchunks is specified.
        #[arg(long)]
        in_memory: bool,

        /// How many tokens will be added after each pass.
        #[arg(long, default_value_t = 1)]
        add_block: usize,
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
        Command::Tokenize {
            tokens,
            initial_size,
            chunk_size,
            in_memory,
        } => tokenize(filename, tokens, *initial_size, *chunk_size, *in_memory),

        Command::OptimizeTokens {
            input_tokens,
            output_tokens,
            ntokens,
            literals,
            initial_size,
            processing,
            in_memory,
            nchunks,
            chunk_size,
            add_block,
        } => optimize_tokens(
            filename,
            input_tokens,
            output_tokens,
            *ntokens,
            *initial_size,
            processing,
            *in_memory,
            *nchunks,
            *chunk_size,
            *add_block,
            *literals,
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
