use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::iter::Iterator;

pub enum Sample<'a> {
    Data(Vec<u8>),
    Ref(&'a [u8]),
}


pub trait Sampler<'a> {
    type Iter: Iterator<Item=Sample<'a>>;

    fn iter(&'a self) -> Self::Iter;
}

pub struct FileSampler {
    filename: String,
    chunk_size: usize,
}

impl FileSampler {
    pub fn new(filename: &str, chunk_size: usize) -> Self {
        FileSampler { filename: filename.to_string(), chunk_size }
    }
}

impl<'a> Sampler<'a> for FileSampler {
    type Iter = FileIterator<'a>;

    fn iter(&'a self) -> Self::Iter {
        let file = File::open(self.filename.as_str()).unwrap();
        FileIterator {
            _sampler: self,
            file,
            chunk_size: self.chunk_size,
        }
    }
}

pub struct FileIterator<'a> {
    _sampler: &'a FileSampler,
    file: File,
    chunk_size: usize,
}

impl<'a> Iterator for FileIterator<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Sample<'a>> {
        let mut buffer = Vec::new();
        buffer.resize(self.chunk_size, 0);
        let read_bytes = self.file.read(&mut buffer).unwrap();

        if read_bytes == 0 {
            None
        } else {
            buffer.truncate(read_bytes);
            Some(Sample::Data(buffer))
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
        MemorySampler {
            data,
            chunk_size,
        }
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
        
        SelectionSampler { chunks }
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
