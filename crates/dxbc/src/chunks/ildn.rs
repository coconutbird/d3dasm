//! ILDN chunk parser — debug name.
//!
//! The ILDN chunk stores the original shader file name or debug identifier
//! as a length-prefixed string. Layout:
//!   0x00: u16 — reserved flags (must be zero)
//!   0x02: u16 — name length in bytes (excluding null terminator)
//!   0x04: [u8; `name_len`] — UTF-8 name
//!   ...: u8 — null terminator, then zero padding to a 4-byte boundary

use alloc::borrow::Cow;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed ILDN (debug name) chunk.
#[derive(Debug, Clone)]
pub struct DebugName<'a> {
    /// Reserved flags field. A conforming ILDN chunk always stores zero.
    pub flags: u16,
    /// The debug name / original file path.
    pub name: Cow<'a, str>,
}

/// Parse an ILDN chunk.
#[must_use]
pub fn parse_ildn(data: &[u8]) -> Option<DebugName<'_>> {
    if data.len() < 8 || data.len() % 4 != 0 {
        return None;
    }

    let flags = u16::from_le_bytes(data[..2].try_into().ok()?);
    if flags != 0 {
        return None;
    }
    let name_len = usize::from(u16::from_le_bytes(data[2..4].try_into().ok()?));
    let terminator = 4usize.checked_add(name_len)?;
    let unpadded_size = terminator.checked_add(1)?;
    let aligned_size = unpadded_size.checked_add(3)? & !3;
    if aligned_size != data.len() || data.get(terminator).copied()? != 0 {
        return None;
    }
    let name_bytes = data.get(4..terminator)?;
    if name_bytes.contains(&0) || data[unpadded_size..].iter().any(|&byte| byte != 0) {
        return None;
    }
    let name = core::str::from_utf8(name_bytes).ok()?;

    Some(DebugName {
        flags,
        name: Cow::Borrowed(name),
    })
}

impl<'a> ChunkParser<'a> for DebugName<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_ildn(data)
    }
}

impl ChunkWriter for DebugName<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"ILDN"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let name_bytes = self.name.as_bytes();
        assert_eq!(self.flags, 0, "ILDN reserved flags must be zero");
        assert!(
            !name_bytes.contains(&0),
            "ILDN debug name cannot contain a null byte"
        );
        let mut buf = alloc::vec::Vec::with_capacity(8 + name_bytes.len());
        buf.extend_from_slice(&self.flags.to_le_bytes());
        let name_len = u16::try_from(name_bytes.len()).expect("debug name length exceeds u16");
        buf.extend_from_slice(&name_len.to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(0); // null terminator
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        buf
    }
}

impl fmt::Display for DebugName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Debug Name: {}", self.name)
    }
}
