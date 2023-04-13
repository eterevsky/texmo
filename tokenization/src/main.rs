use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::process;
use std::collections::HashMap;
use std::collections::hash_map::Entry::Occupied;
use smallvec::{SmallVec, smallvec};



fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        process::exit(1);
    }

    // Read file name from command line argument
    let filename = &args[1];

    // Read binary file content into a Vec<u8>
    let contents = match read_binary_file_into_byte_array(filename) {
        Ok(contents) => contents,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            process::exit(1);
        }
    };

    println!("File size: {:?}", contents.len());

    get_top_substrings(&contents, 65536);
}

fn read_binary_file_into_byte_array(filename: &str) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(filename)?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    Ok(buffer)
}

type Token = SmallVec<[u8; 16]>;

fn get_top_substrings(data: &[u8], n: usize) {
    let mut counts: HashMap<Token, u64> = HashMap::new();

    for c in 0..=255 {
        let k: Token = smallvec![c];
        counts.insert(k, 1);
    }

    for c in data.iter() {
        let k: Token = smallvec![*c];
        counts.entry(k).and_modify(|e| *e += 1).or_insert(1);
    }

    for length in 2..17 {
        let counts_len = counts.len();
        let min_count = if counts_len < n {
            1
        } else {
            let mut pairs: Vec<(Token, u64)> = counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            pairs[n - 1].1
        };

        for i in 0..(data.len() - length + 1) {
            let slice = &data[i..i+length];
            let slice_vec: Token = SmallVec::from_slice(slice);

            if let Occupied(mut entry) = counts.entry(slice_vec.clone()) {
                entry.insert(entry.get() + 1);
                continue;
            }

            let prefix: Token = SmallVec::from_slice(&slice[..length-1]);
            let suffix: Token = SmallVec::from_slice(&slice[1..]);

            let max_count = std::cmp::max(*counts.get(&prefix).unwrap_or(&0),
                                            *counts.get(&suffix).unwrap_or(&0));

            if max_count > min_count {
                counts.insert(slice_vec, 1);
            }
        }
    }

    let mut pairs: Vec<(SmallVec<[u8; 16]>, u64)> = counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    for (c, count) in pairs.iter().take(n) {
        match std::str::from_utf8(&c) {
            Ok(string) => println!("{:?}  {}", string, count),
            Err(_) => println!("{:?}  {}", c, count),
        }
    }
}