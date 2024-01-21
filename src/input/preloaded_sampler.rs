use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::iter::Iterator;

use crate::input::sample::{Sample, Sampler};

pub struct SelectionSampler {
    chunks: Vec<Vec<u8>>,
    _total_size: u64,
}

impl SelectionSampler {
    pub fn new(filename: &str, chunk_size: usize, nchunks: usize) -> Self {
        // Get the metadata of the file
        let data_len = std::fs::metadata(filename).unwrap().len() as usize;

        let (chunk_size, nchunks) = if data_len <= chunk_size {
            (data_len, 1)
        } else if data_len <= chunk_size * nchunks {
            (chunk_size, data_len / chunk_size)
        } else {
            (chunk_size, nchunks)
        };

        println!(
            "Preparing SelectionSampler with {} chunks x {} bytes",
            nchunks, chunk_size
        );

        let step = data_len / nchunks;

        let mut file = File::open(filename).unwrap();

        let mut chunks = Vec::new();

        for i in 0..nchunks as u64 {
            file.seek(SeekFrom::Start(i * step as u64)).unwrap();
            let mut chunk = Vec::new();
            chunk.resize(chunk_size, 0);
            let read_bytes = file.read(&mut chunk).unwrap();
            chunk.truncate(read_bytes);

            chunks.push(chunk);
        }

        let _total_size = chunks.iter().map(|c| c.len() as u64).sum();
        SelectionSampler {
            chunks,
            _total_size,
        }
    }
}

impl<'a> Sampler<'a> for SelectionSampler {
    type Iter = SelectionIterator<'a>;

    fn iter(&'a self) -> Self::Iter {
        SelectionIterator {
            sampler: self,
            position: 0,
        }
    }

    fn total_size(&'a self) -> u64 {
        self._total_size
    }
}

pub struct SelectionIterator<'a> {
    sampler: &'a SelectionSampler,
    position: usize,
}

impl<'a> Iterator for SelectionIterator<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Sample<'a>> {
        if self.position < self.sampler.chunks.len() {
            let chunk = &self.sampler.chunks[self.position];
            self.position += 1;
            Some(Sample::from_bytes(chunk))
        } else {
            None
        }
    }
}
