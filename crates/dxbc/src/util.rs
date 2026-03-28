//! Shared low-level helpers for reading and writing binary data.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Read a null-terminated C string starting at `offset`.
///
/// Returns an empty string if `offset` is out of bounds or the bytes
/// are not valid UTF-8. In practice, all strings emitted by the HLSL
/// compiler are pure ASCII.
pub fn read_cstring(data: &[u8], offset: usize) -> &str {
    if offset >= data.len() {
        return "";
    }

    let mut end = offset;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }

    core::str::from_utf8(&data[offset..end]).unwrap_or("")
}

/// A write-side string table that collects null-terminated strings,
/// deduplicates them, and produces a compact byte blob.
///
/// Strings are stored in insertion order. Duplicate strings reuse the
/// offset of the first occurrence.
///
/// # Usage
/// ```ignore
/// let mut st = StringTableWriter::new(base_offset);
/// let off_a = st.add("POSITION");
/// let off_b = st.add("TEXCOORD");
/// let off_a2 = st.add("POSITION"); // returns same offset as off_a
/// let blob = st.finish();
/// ```
pub struct StringTableWriter {
    /// Absolute byte offset where the string table will be placed
    /// in the final chunk payload.
    base: usize,
    /// Maps string content → relative offset within the table.
    map: BTreeMap<String, usize>,
    /// The raw bytes of the string table built so far.
    buf: Vec<u8>,
}

impl StringTableWriter {
    /// Create a new writer. `base_offset` is the absolute position in the
    /// chunk payload where the string table blob will be written.
    pub fn new(base_offset: usize) -> Self {
        Self {
            base: base_offset,
            map: BTreeMap::new(),
            buf: Vec::new(),
        }
    }

    /// Insert a string into the table and return its **absolute** offset
    /// (i.e. `base_offset + relative_position`).
    ///
    /// If the string was already added, the previous offset is returned.
    pub fn add(&mut self, s: &str) -> u32 {
        if let Some(&rel) = self.map.get(s) {
            return (self.base + rel) as u32;
        }
        let rel = self.buf.len();
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0); // null terminator
        self.map.insert(String::from(s), rel);
        (self.base + rel) as u32
    }

    /// Consume the writer and return the serialized string table bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Current size of the string table in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the string table is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_table_dedup() {
        let mut st = StringTableWriter::new(100);
        let a = st.add("POSITION");
        let b = st.add("TEXCOORD");
        let a2 = st.add("POSITION");
        assert_eq!(a, a2);
        assert_ne!(a, b);
        // "POSITION\0" = 9 bytes, starts at relative 0 → absolute 100
        assert_eq!(a, 100);
        // "TEXCOORD\0" = 9 bytes, starts at relative 9 → absolute 109
        assert_eq!(b, 109);
        let blob = st.finish();
        assert_eq!(blob.len(), 18); // 9 + 9
        assert_eq!(&blob[0..9], b"POSITION\0");
        assert_eq!(&blob[9..18], b"TEXCOORD\0");
    }

    #[test]
    fn string_table_empty() {
        let st = StringTableWriter::new(0);
        assert!(st.is_empty());
        assert_eq!(st.len(), 0);
        assert_eq!(st.finish().len(), 0);
    }

    #[test]
    fn read_cstring_basic() {
        let data = b"hello\0world\0";
        assert_eq!(read_cstring(data, 0), "hello");
        assert_eq!(read_cstring(data, 6), "world");
        assert_eq!(read_cstring(data, 100), "");
    }
}
