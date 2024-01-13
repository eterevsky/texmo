use rand::Rng;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::iter::Iterator;

pub enum Sample<'a> {
    Data(Vec<u8>),
    Ref(&'a [u8]),
}

impl Sample<'_> {
    pub fn len(&self) -> usize {
        match self {
            Sample::Data(data) => data.len(),
            Sample::Ref(data) => data.len(),
        }
    }
}

pub trait Sampler<'a> {
    type Iter: Iterator<Item = Sample<'a>>;

    fn iter(&'a self) -> Self::Iter;

    fn total_size(&'a self) -> u64;
}

pub struct FileSampler {
    filename: String,
    chunk_size: usize,
    _total_size: u64,
    chunks_selection: Option<usize>,
}

impl FileSampler {
    pub fn new(filename: &str, chunk_size: usize, chunks_selection: Option<usize>) -> Self {
        let _total_size = if let Some(cs) = chunks_selection {
            (chunk_size * cs) as u64
        } else {
            std::fs::metadata(filename).unwrap().len()
        };
        FileSampler {
            filename: filename.to_string(),
            chunk_size,
            _total_size,
            chunks_selection,
        }
    }
}

impl<'a> Sampler<'a> for FileSampler {
    type Iter = FileIterator<'a>;

    fn iter(&'a self) -> Self::Iter {
        let file = File::open(self.filename.as_str()).unwrap();

        if let Some(chunks_selection) = self.chunks_selection {
            FileIterator {
                _sampler: self,
                file,
                chunk_size: self.chunk_size,
                total_size: self._total_size,
                sample_chunks_left: Some(chunks_selection),
                read_bytes: 0,
            }
        } else {
            FileIterator {
                _sampler: self,
                file,
                chunk_size: self.chunk_size,
                total_size: self._total_size,
                sample_chunks_left: None,
                read_bytes: 0,
            }
        }
    }

    fn total_size(&self) -> u64 {
        self._total_size
    }
}

pub struct FileIterator<'a> {
    _sampler: &'a FileSampler,
    file: File,
    chunk_size: usize,
    total_size: u64,
    sample_chunks_left: Option<usize>,

    pub read_bytes: usize,
}

impl<'a> Iterator for FileIterator<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Sample<'a>> {
        let mut buffer = Vec::new();
        buffer.resize(self.chunk_size, 0);

        if let Some(chunks_left) = self.sample_chunks_left {
            if chunks_left == 0 {
                None
            } else {
                self.sample_chunks_left = Some(chunks_left - 1);

                let mut rng = rand::thread_rng();
                let max_seek = self.total_size - self.chunk_size as u64;
                let start = rng.gen_range(0..max_seek);

                self.file.seek(SeekFrom::Start(start)).unwrap();
                let read_bytes = self.file.read(&mut buffer).unwrap();

                buffer.truncate(read_bytes);
                self.read_bytes += read_bytes;
                Some(Sample::Data(buffer))
            }
        } else {
            let read_bytes = self.file.read(&mut buffer).unwrap();

            if read_bytes == 0 {
                None
            } else {
                buffer.truncate(read_bytes);
                self.read_bytes += read_bytes;
                // if self.read_bytes & ((1<<30) - 1) == 0 {
                //     dbg!(self.read_bytes);
                // }
                Some(Sample::Data(buffer))
            }
        }
    }
}

pub struct MemorySampler {
    data: Vec<u8>,
    chunk_size: usize,
}

impl MemorySampler {
    pub fn new(filename: &str, chunk_size: usize) -> Self {
        let data = std::fs::read(filename).unwrap();
        MemorySampler { data, chunk_size }
    }
}

impl<'a> Sampler<'a> for MemorySampler {
    type Iter = MemoryIterator<'a>;

    fn iter(&'a self) -> Self::Iter {
        MemoryIterator {
            sampler: self,
            position: 0,
        }
    }

    fn total_size(&'a self) -> u64 {
        self.data.len() as u64
    }
}

pub struct MemoryIterator<'a> {
    sampler: &'a MemorySampler,
    position: usize,
}

impl<'a> Iterator for MemoryIterator<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Sample<'a>> {
        if self.position < self.sampler.data.len() {
            let start = self.position;
            self.position = std::cmp::min(start + self.sampler.chunk_size, self.sampler.data.len());
            Some(Sample::Ref(&self.sampler.data[start..self.position]))
        } else {
            None
        }
    }
}

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
            Some(Sample::Ref(chunk))
        } else {
            None
        }
    }
}
