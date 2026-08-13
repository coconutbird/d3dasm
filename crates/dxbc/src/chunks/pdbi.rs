//! PDBI chunk — RDAT-formatted shader creation metadata embedded in a PDB.
//!
//! `DxilShaderPDBInfo` replaced SRCI for modern DXC PDBs. Its payload is an
//! RDAT stream, either stored directly or compressed with zlib. This module
//! validates the published header and preserves the stored bytes without
//! requiring a decompressor in the `no_std` parser.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

const HEADER_SIZE: usize = 12;

/// Published `DxilShaderPDBInfoVersion` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderPdbInfoVersion {
    /// Initial and currently latest PDB-info layout.
    V0,
}

impl ShaderPdbInfoVersion {
    fn from_u16(value: u16) -> Option<Self> {
        (value == 0).then_some(Self::V0)
    }

    const fn to_u16(self) -> u16 {
        match self {
            Self::V0 => 0,
        }
    }
}

/// Published `DxilShaderPDBInfoCompressionType` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderPdbCompression {
    /// The RDAT bytes are stored directly.
    Uncompressed,
    /// The RDAT bytes are stored as a zlib stream.
    Zlib,
}

impl ShaderPdbCompression {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Uncompressed),
            1 => Some(Self::Zlib),
            _ => None,
        }
    }

    const fn to_u16(self) -> u16 {
        match self {
            Self::Uncompressed => 0,
            Self::Zlib => 1,
        }
    }
}

/// Published `DxilShaderPDBInfo` header and its stored RDAT bytes.
#[derive(Debug, Clone)]
pub struct ShaderPdbInfo<'a> {
    /// PDB-info format version.
    pub version: ShaderPdbInfoVersion,
    /// Storage compression applied to `data`.
    pub compression: ShaderPdbCompression,
    /// Size of the RDAT stream after decompression.
    pub uncompressed_size_in_bytes: u32,
    /// Stored RDAT bytes (raw or zlib-compressed according to `compression`).
    pub data: Cow<'a, [u8]>,
}

/// Parse a published `DxilShaderPDBInfo` payload.
#[must_use]
pub fn parse_pdbi(data: &[u8]) -> Option<ShaderPdbInfo<'_>> {
    if data.len() < HEADER_SIZE {
        return None;
    }
    let version = ShaderPdbInfoVersion::from_u16(u16::from_le_bytes(data[0..2].try_into().ok()?))?;
    let compression =
        ShaderPdbCompression::from_u16(u16::from_le_bytes(data[2..4].try_into().ok()?))?;
    let stored_size = usize::try_from(u32::from_le_bytes(data[4..8].try_into().ok()?)).ok()?;
    let uncompressed_size_in_bytes = u32::from_le_bytes(data[8..12].try_into().ok()?);
    if HEADER_SIZE.checked_add(stored_size)? != data.len() {
        return None;
    }
    if compression == ShaderPdbCompression::Uncompressed
        && usize::try_from(uncompressed_size_in_bytes).ok()? != stored_size
    {
        return None;
    }

    Some(ShaderPdbInfo {
        version,
        compression,
        uncompressed_size_in_bytes,
        data: Cow::Borrowed(&data[HEADER_SIZE..]),
    })
}

impl<'a> ChunkParser<'a> for ShaderPdbInfo<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_pdbi(data)
    }
}

impl ChunkWriter for ShaderPdbInfo<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"PDBI"
    }

    fn write_payload(&self) -> Vec<u8> {
        if self.compression == ShaderPdbCompression::Uncompressed {
            assert_eq!(
                usize::try_from(self.uncompressed_size_in_bytes).ok(),
                Some(self.data.len()),
                "uncompressed PDBI sizes must match"
            );
        }
        let stored_size = u32::try_from(self.data.len()).expect("PDBI data exceeds u32");
        let mut output = Vec::with_capacity(HEADER_SIZE + self.data.len());
        output.extend_from_slice(&self.version.to_u16().to_le_bytes());
        output.extend_from_slice(&self.compression.to_u16().to_le_bytes());
        output.extend_from_slice(&stored_size.to_le_bytes());
        output.extend_from_slice(&self.uncompressed_size_in_bytes.to_le_bytes());
        output.extend_from_slice(&self.data);
        output
    }
}

impl fmt::Display for ShaderPdbInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let compression = match self.compression {
            ShaderPdbCompression::Uncompressed => "uncompressed",
            ShaderPdbCompression::Zlib => "zlib",
        };
        writeln!(
            f,
            "// Shader PDB Info: {compression}, stored={} bytes, uncompressed={} bytes",
            self.data.len(),
            self.uncompressed_size_in_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_header_roundtrips() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(b"RDAT");
        let parsed = parse_pdbi(&bytes).expect("valid PDBI");
        assert_eq!(parsed.write_payload(), bytes);
    }

    #[test]
    fn rejects_future_versions_and_inconsistent_sizes() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[..2].copy_from_slice(&1u16.to_le_bytes());
        assert!(parse_pdbi(&bytes).is_none());
        bytes[..2].copy_from_slice(&0u16.to_le_bytes());
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_pdbi(&bytes).is_none());
    }
}
