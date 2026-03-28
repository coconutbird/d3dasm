//! ILDN chunk parser — debug name.
//!
//! The ILDN chunk stores the original shader file name or debug identifier
//! as a length-prefixed string. Layout:
//!   0x00: u16 — flags / unknown
//!   0x02: u16 — name length in bytes (excluding null terminator)
//!   0x04: [u8; name_len] — UTF-8 name
//!
//! Some compilers write a simpler variant with just a null-terminated string.

use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed ILDN (debug name) chunk.
#[derive(Debug, Clone)]
pub struct DebugName {
    /// The debug name / original file path.
    pub name: alloc::string::String,
}

/// Parse an ILDN chunk.
pub fn parse_ildn(data: &[u8]) -> Option<DebugName> {
    if data.len() < 4 {
        return None;
    }

    // Try the length-prefixed format first: u16 flags, u16 length, then bytes.
    let name_len = (data[2] as usize) | ((data[3] as usize) << 8);
    if name_len > 0 && 4 + name_len <= data.len() {
        let name_bytes = &data[4..4 + name_len];
        if let Ok(s) = core::str::from_utf8(name_bytes) {
            return Some(DebugName {
                name: alloc::string::String::from(s.trim_end_matches('\0')),
            });
        }
    }

    // Fallback: treat entire chunk as a null-terminated string.
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    let s = core::str::from_utf8(&data[..end]).unwrap_or("");
    if s.is_empty() {
        None
    } else {
        Some(DebugName {
            name: alloc::string::String::from(s),
        })
    }
}

impl ChunkParser for DebugName {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_ildn(data)
    }
}

impl ChunkWriter for DebugName {
    fn fourcc(&self) -> [u8; 4] {
        *b"ILDN"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut buf = alloc::vec::Vec::with_capacity(4 + name_bytes.len() + 1);
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(0); // null terminator
        buf
    }
}

impl fmt::Display for DebugName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Debug Name: {}", self.name)
    }
}
