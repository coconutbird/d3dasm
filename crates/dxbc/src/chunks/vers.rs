//! VERS chunk — the DXC compiler version recorded in a DXIL container.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// The compiler binary was built with debug information.
pub const VERSION_FLAG_DEBUG: u32 = 1;
/// The compiler uses an internal validator and cannot sign shaders.
pub const VERSION_FLAG_INTERNAL: u32 = 2;

/// `DxilCompilerVersion` followed by up to two null-terminated strings.
#[derive(Debug, Clone)]
pub struct CompilerVersion<'a> {
    /// DXC major version.
    pub major: u16,
    /// DXC minor version.
    pub minor: u16,
    /// `DxcVersionInfoFlags` bitmask.
    pub flags: u32,
    /// Source-control commit count.
    pub commit_count: u32,
    /// Source-control commit identifier bytes, when present.
    ///
    /// The published structure promises a null-terminated byte string but
    /// does not require UTF-8, so the raw bytes are retained.
    pub commit_sha: Option<Cow<'a, [u8]>>,
    /// Custom compiler version bytes, when present.
    pub custom_version: Option<Cow<'a, [u8]>>,
}

/// Parse a VERS payload.
#[must_use]
pub fn parse_vers(data: &[u8]) -> Option<CompilerVersion<'_>> {
    const HEADER_SIZE: usize = 16;
    if data.len() < HEADER_SIZE || data.len() % 4 != 0 {
        return None;
    }

    let major = u16::from_le_bytes(data[0..2].try_into().ok()?);
    let minor = u16::from_le_bytes(data[2..4].try_into().ok()?);
    let flags = u32::from_le_bytes(data[4..8].try_into().ok()?);
    let commit_count = u32::from_le_bytes(data[8..12].try_into().ok()?);
    let list_size = usize::try_from(u32::from_le_bytes(data[12..16].try_into().ok()?)).ok()?;
    let list_end = HEADER_SIZE.checked_add(list_size)?;
    let padded_end = list_end.checked_add(3)? & !3;
    if padded_end != data.len() || data[list_end..].iter().any(|&byte| byte != 0) {
        return None;
    }

    let list = &data[HEADER_SIZE..list_end];
    let mut strings: [Option<Cow<'_, [u8]>>; 2] = [None, None];
    let mut position = 0usize;
    for slot in &mut strings {
        if position == list.len() {
            break;
        }
        let relative_end = list[position..].iter().position(|&byte| byte == 0)?;
        let end = position.checked_add(relative_end)?;
        *slot = Some(Cow::Borrowed(&list[position..end]));
        position = end.checked_add(1)?;
    }
    if position != list.len() {
        return None;
    }

    Some(CompilerVersion {
        major,
        minor,
        flags,
        commit_count,
        commit_sha: strings[0].take(),
        custom_version: strings[1].take(),
    })
}

impl<'a> ChunkParser<'a> for CompilerVersion<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_vers(data)
    }
}

impl ChunkWriter for CompilerVersion<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"VERS"
    }

    fn write_payload(&self) -> Vec<u8> {
        let mut strings = Vec::new();
        if let Some(commit_sha) = &self.commit_sha {
            assert!(
                !commit_sha.contains(&0),
                "VERS commit SHA cannot contain a null byte"
            );
            strings.extend_from_slice(commit_sha);
            strings.push(0);
        }
        if let Some(custom_version) = &self.custom_version {
            if self.commit_sha.is_none() {
                // The custom string is the second item in the ordered list.
                strings.push(0);
            }
            assert!(
                !custom_version.contains(&0),
                "VERS custom version cannot contain a null byte"
            );
            strings.extend_from_slice(custom_version);
            strings.push(0);
        }

        let list_size = u32::try_from(strings.len()).expect("VERS string list exceeds u32");
        let mut out = Vec::with_capacity(16 + strings.len() + 3);
        out.extend_from_slice(&self.major.to_le_bytes());
        out.extend_from_slice(&self.minor.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.commit_count.to_le_bytes());
        out.extend_from_slice(&list_size.to_le_bytes());
        out.extend_from_slice(&strings);
        while out.len() % 4 != 0 {
            out.push(0);
        }
        out
    }
}

impl fmt::Display for CompilerVersion<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "// DXC Version: {}.{} (commit count {}",
            self.major, self.minor, self.commit_count
        )?;
        if let Some(sha) = &self.commit_sha {
            write!(f, ", {}", StringBytes(sha))?;
        }
        if let Some(custom) = &self.custom_version {
            write!(f, ", {}", StringBytes(custom))?;
        }
        writeln!(f, ", flags=0x{:08X})", self.flags)
    }
}

struct StringBytes<'a>(&'a [u8]);

impl fmt::Display for StringBytes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(value) = core::str::from_utf8(self.0) {
            f.write_str(value)
        } else {
            for byte in self.0 {
                write!(f, "\\x{byte:02X}")?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_round_trips_current_dxc_layout() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&9u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&5402u32.to_le_bytes());
        bytes.extend_from_slice(&20u32.to_le_bytes());
        bytes.extend_from_slice(b"0d3ee6b5\0");
        bytes.extend_from_slice(b"1.9.0.5402\0");

        let version = parse_vers(&bytes).expect("valid VERS");
        assert_eq!(version.commit_sha.as_deref(), Some(b"0d3ee6b5".as_slice()));
        assert_eq!(
            version.custom_version.as_deref(),
            Some(b"1.9.0.5402".as_slice())
        );
        assert_eq!(version.write_payload(), bytes);
    }

    #[test]
    fn non_utf8_version_bytes_are_preserved() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&9u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0, 0x80, 0]);

        let version = parse_vers(&bytes).expect("byte strings are valid VERS data");
        assert_eq!(version.commit_sha.as_deref(), Some([0xff].as_slice()));
        assert_eq!(version.custom_version.as_deref(), Some([0x80].as_slice()));
        assert_eq!(version.write_payload(), bytes);
    }
}
