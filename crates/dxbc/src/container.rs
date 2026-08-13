use alloc::vec::Vec;
use core::fmt;

use crate::chunks::WritableChunk;

const DXBC_MAGIC: &[u8; 4] = b"DXBC";
const DXBC_HEADER_SIZE: usize = 0x20;
const MAX_CONTAINER_SIZE: usize = 0x8000_0000;

fn wire_u32(value: usize) -> u32 {
    u32::try_from(value).expect("DXBC size or offset exceeds u32")
}

/// The two 16-bit fields stored in a DXBC container header.
///
/// Microsoft containers currently use major version 1 and minor version 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerVersion {
    /// Container format major version.
    pub major: u16,
    /// Container format minor version.
    pub minor: u16,
}

impl ContainerVersion {
    /// The version emitted by current DXBC/DXIL tooling.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    /// Decode the little-endian packed `u32` used by older versions of this API.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        let bytes = raw.to_le_bytes();
        Self {
            major: u16::from_le_bytes([bytes[0], bytes[1]]),
            minor: u16::from_le_bytes([bytes[2], bytes[3]]),
        }
    }

    /// Return the exact packed representation stored in the header.
    #[must_use]
    pub fn to_raw(self) -> u32 {
        u32::from(self.major) | (u32::from(self.minor) << 16)
    }
}

impl From<u32> for ContainerVersion {
    fn from(value: u32) -> Self {
        Self::from_raw(value)
    }
}

impl fmt::Display for ContainerVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Preserve the historic `.dxbc version=<packed-u32>` text grammar.
        write!(f, "{}", self.to_raw())
    }
}

/// A parsed DXBC container with its chunk table.
#[derive(Debug)]
pub struct DxbcContainer<'a> {
    /// Byte offset of this container within the source data.
    pub offset_in_file: usize,
    /// Total size of the container in bytes (from the DXBC header).
    pub total_size: u32,
    /// The 16-byte container header digest (bytes 4..20), preserved so the
    /// container can be rebuilt byte-identically.
    pub header_hash: [u8; 16],
    /// Container format version (two 16-bit fields at header bytes 20..24).
    pub version: ContainerVersion,
    /// Parsed chunk entries.
    pub chunks: Vec<DxbcChunk<'a>>,
}

/// A single chunk within a DXBC container.
#[derive(Debug)]
pub struct DxbcChunk<'a> {
    /// Four-character code identifying the chunk type (e.g. `RDEF`, `SHEX`).
    pub fourcc: [u8; 4],
    /// Chunk payload size in bytes.
    pub size: u32,
    /// Raw chunk payload.
    pub data: &'a [u8],
}

impl<'a> DxbcChunk<'a> {
    /// Returns the `FourCC` as a UTF-8 string, or `"????"` if it contains
    /// invalid bytes.
    #[must_use]
    pub fn fourcc_str(&self) -> &str {
        core::str::from_utf8(&self.fourcc).unwrap_or("????")
    }

    /// Parse the raw chunk payload into a typed [`ChunkData`](crate::chunks::ChunkData).
    #[must_use]
    pub fn parse(&self) -> crate::chunks::ChunkData<'a> {
        crate::chunks::parse_chunk(self)
    }
}

impl fmt::Display for DxbcContainer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DXBC at 0x{:X}, size={}, chunks={}",
            self.offset_in_file,
            self.total_size,
            self.chunks.len()
        )
    }
}

/// Scan a byte slice for all DXBC containers and parse them.
#[must_use]
pub fn scan_dxbc(data: &[u8]) -> Vec<DxbcContainer<'_>> {
    let mut results = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let Some(offset) = find_magic(data, pos) else {
            break;
        };
        if let Some(container) = parse_dxbc(data, offset) {
            let end = offset + container.total_size as usize;
            results.push(container);
            pos = end;
        } else {
            // A random payload may contain the ASCII bytes "DXBC". Keep
            // searching instead of letting one malformed candidate hide a
            // valid container later in the file.
            pos = offset + DXBC_MAGIC.len();
        }
    }
    results
}

fn find_magic(data: &[u8], start: usize) -> Option<usize> {
    memchr::memmem::find(&data[start..], DXBC_MAGIC).map(|p| start + p)
}

fn parse_dxbc(data: &[u8], offset: usize) -> Option<DxbcContainer<'_>> {
    let header_end = offset.checked_add(DXBC_HEADER_SIZE)?;
    if header_end > data.len() || data.get(offset..offset + 4)? != DXBC_MAGIC {
        return None;
    }

    // Header: magic(4), hash(16), version major/minor(2+2), size(4), count(4).
    let mut header_hash = [0u8; 16];
    header_hash.copy_from_slice(&data[offset + 4..offset + 20]);
    let major = u16::from_le_bytes(data[offset + 20..offset + 22].try_into().ok()?);
    let minor = u16::from_le_bytes(data[offset + 22..offset + 24].try_into().ok()?);
    let version = ContainerVersion { major, minor };
    // `IsValidDxilContainer` rejects unknown major versions.
    if version.major != ContainerVersion::CURRENT.major {
        return None;
    }

    let total_size = u32::from_le_bytes(data[offset + 24..offset + 28].try_into().ok()?);
    let total_size_usize = usize::try_from(total_size).ok()?;
    if !(DXBC_HEADER_SIZE..=MAX_CONTAINER_SIZE).contains(&total_size_usize) {
        return None;
    }
    let container_end = offset.checked_add(total_size_usize)?;
    if container_end > data.len() {
        return None;
    }

    let chunk_count = usize::try_from(u32::from_le_bytes(
        data[offset + 28..offset + 32].try_into().ok()?,
    ))
    .ok()?;
    let table_size = chunk_count.checked_mul(4)?;
    let table_end_rel = DXBC_HEADER_SIZE.checked_add(table_size)?;
    if table_end_rel > total_size_usize {
        return None;
    }

    // Grow only after each untrusted table entry and chunk header has passed
    // its bounds checks.
    let mut chunks = Vec::new();
    let mut expected_chunk_offset = table_end_rel;
    for i in 0..chunk_count {
        let entry_rel = DXBC_HEADER_SIZE.checked_add(i.checked_mul(4)?)?;
        let entry_abs = offset.checked_add(entry_rel)?;
        let chunk_offset_rel = usize::try_from(u32::from_le_bytes(
            data[entry_abs..entry_abs + 4].try_into().ok()?,
        ))
        .ok()?;

        // The canonical format is a contiguous sequence of parts immediately
        // following the offset table. DXC's validator enforces this exactly.
        if chunk_offset_rel != expected_chunk_offset {
            return None;
        }
        let chunk_abs = offset.checked_add(chunk_offset_rel)?;
        let chunk_header_end = chunk_abs.checked_add(8)?;
        if chunk_header_end > container_end {
            return None;
        }
        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&data[chunk_abs..chunk_abs + 4]);
        let chunk_size = u32::from_le_bytes(data[chunk_abs + 4..chunk_abs + 8].try_into().ok()?);
        let chunk_size_usize = usize::try_from(chunk_size).ok()?;
        let chunk_data_start = chunk_header_end;
        let chunk_data_end = chunk_data_start.checked_add(chunk_size_usize)?;
        if chunk_data_end > container_end {
            return None;
        }

        chunks.push(DxbcChunk {
            fourcc,
            size: chunk_size,
            data: &data[chunk_data_start..chunk_data_end],
        });
        expected_chunk_offset = chunk_data_end.checked_sub(offset)?;
    }

    if expected_chunk_offset != total_size_usize {
        return None;
    }

    Some(DxbcContainer {
        offset_in_file: offset,
        total_size,
        header_hash,
        version,
        chunks,
    })
}

/// Build a DXBC container from a set of writable chunks.
///
/// Produces a valid DXBC byte stream with the standard header layout:
/// magic(4) + hash(16 zeros) + version(4) + `total_size(4)` + `chunk_count(4)`
/// followed by the chunk offset table and chunk entries.
///
/// The 16-byte hash field is left zeroed — computing the DXBC hash is a
/// separate concern (typically only the GPU runtime validates it). Use
/// [`build_dxbc_with_header`] to preserve an existing hash/version for
/// byte-identical reconstruction.
#[must_use]
pub fn build_dxbc(chunks: &[WritableChunk]) -> Vec<u8> {
    build_dxbc_with_header(chunks, ContainerVersion::CURRENT, &[0u8; 16])
}

/// Build a DXBC container, preserving the given format `version` and 16-byte
/// header `hash`. The inverse of [`scan_dxbc`] for a single container: feeding
/// a parsed container's chunks plus its `version`/`header_hash` reproduces the
/// original bytes (for the standard contiguous chunk layout).
#[must_use]
pub fn build_dxbc_with_header(
    chunks: &[WritableChunk],
    version: impl Into<ContainerVersion>,
    hash: &[u8; 16],
) -> Vec<u8> {
    let version = version.into();
    let header_size = DXBC_HEADER_SIZE
        .checked_add(
            chunks
                .len()
                .checked_mul(4)
                .expect("DXBC chunk table overflow"),
        )
        .expect("DXBC header size overflow");
    let chunks_total = chunks.iter().fold(0usize, |total, chunk| {
        total
            .checked_add(8)
            .and_then(|value| value.checked_add(chunk.data.len()))
            .expect("DXBC chunk data size overflow")
    });
    let total_size = header_size
        .checked_add(chunks_total)
        .expect("DXBC container size overflow");

    let mut buf = Vec::with_capacity(total_size);
    buf.extend_from_slice(b"DXBC");
    buf.extend_from_slice(hash);
    buf.extend_from_slice(&version.major.to_le_bytes());
    buf.extend_from_slice(&version.minor.to_le_bytes());
    buf.extend_from_slice(&wire_u32(total_size).to_le_bytes());
    buf.extend_from_slice(&wire_u32(chunks.len()).to_le_bytes());

    // Chunk offset table
    let mut offset = header_size;
    for c in chunks {
        buf.extend_from_slice(&wire_u32(offset).to_le_bytes());
        offset += 8 + c.data.len();
    }

    // Chunk entries: fourcc(4) + size(4) + payload
    for c in chunks {
        buf.extend_from_slice(&c.fourcc);
        buf.extend_from_slice(&wire_u32(c.data.len()).to_le_bytes());
        buf.extend_from_slice(&c.data);
    }

    buf
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use super::*;

    /// Build a minimal valid DXBC container with the given chunks.
    /// Each chunk is (fourcc, data).
    fn build_dxbc(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        // DXBC header: magic(4) + hash(16) + 1(4) + total_size(4) + chunk_count(4) = 32 bytes
        // Then chunk_count * 4 bytes for chunk offsets
        // Then each chunk: fourcc(4) + size(4) + data
        let header_size = 0x20 + chunks.len() * 4;
        let mut chunk_data_sections: Vec<Vec<u8>> = Vec::new();
        for &(_, data) in chunks {
            let mut section = Vec::new();
            // We'll fill fourcc+size later in the final assembly
            section.extend_from_slice(data);
            chunk_data_sections.push(section);
        }

        // Calculate total size
        let chunks_total: usize = chunks
            .iter()
            .map(|(_, data)| 8 + data.len()) // fourcc + size + data
            .sum();
        let total_size = header_size + chunks_total;

        let mut buf = Vec::with_capacity(total_size);
        // Magic
        buf.extend_from_slice(b"DXBC");
        // Hash (16 bytes of zeros — we don't validate it)
        buf.extend_from_slice(&[0u8; 16]);
        // Version = 1
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Total size
        buf.extend_from_slice(&wire_u32(total_size).to_le_bytes());
        // Chunk count
        buf.extend_from_slice(&wire_u32(chunks.len()).to_le_bytes());

        // Chunk offset table
        let mut offset = header_size;
        for &(_, data) in chunks {
            buf.extend_from_slice(&wire_u32(offset).to_le_bytes());
            offset += 8 + data.len();
        }

        // Chunk data
        for (i, &(fourcc, data)) in chunks.iter().enumerate() {
            buf.extend_from_slice(fourcc);
            buf.extend_from_slice(&wire_u32(data.len()).to_le_bytes());
            buf.extend_from_slice(data);
            let _ = i;
        }

        buf
    }

    #[test]
    fn parse_single_chunk() {
        let shex_data = [0u8; 16]; // dummy SHEX payload
        let dxbc = build_dxbc(&[(b"SHEX", &shex_data)]);
        let containers = scan_dxbc(&dxbc);
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0].chunks.len(), 1);
        assert_eq!(containers[0].chunks[0].fourcc_str(), "SHEX");
        assert_eq!(containers[0].chunks[0].size, 16);
        assert_eq!(containers[0].chunks[0].data.len(), 16);
    }

    #[test]
    fn parse_multiple_chunks() {
        let rdef_data = [1u8; 32];
        let isgn_data = [2u8; 24];
        let osgn_data = [3u8; 24];
        let shex_data = [4u8; 64];
        let dxbc = build_dxbc(&[
            (b"RDEF", &rdef_data),
            (b"ISGN", &isgn_data),
            (b"OSGN", &osgn_data),
            (b"SHEX", &shex_data),
        ]);
        let containers = scan_dxbc(&dxbc);
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0].chunks.len(), 4);
        assert_eq!(containers[0].chunks[0].fourcc_str(), "RDEF");
        assert_eq!(containers[0].chunks[1].fourcc_str(), "ISGN");
        assert_eq!(containers[0].chunks[2].fourcc_str(), "OSGN");
        assert_eq!(containers[0].chunks[3].fourcc_str(), "SHEX");
        assert_eq!(containers[0].chunks[3].data.len(), 64);
    }

    #[test]
    fn parse_back_to_back_containers() {
        let shex1 = [0u8; 8];
        let shex2 = [1u8; 8];
        let dxbc1 = build_dxbc(&[(b"SHEX", &shex1)]);
        let dxbc2 = build_dxbc(&[(b"SHDR", &shex2)]);
        let mut combined = dxbc1.clone();
        combined.extend_from_slice(&dxbc2);
        let containers = scan_dxbc(&combined);
        assert_eq!(containers.len(), 2);
        assert_eq!(containers[0].chunks[0].fourcc_str(), "SHEX");
        assert_eq!(containers[1].chunks[0].fourcc_str(), "SHDR");
    }

    #[test]
    fn empty_input() {
        assert!(scan_dxbc(&[]).is_empty());
    }

    #[test]
    fn no_dxbc_magic() {
        assert!(scan_dxbc(b"not a dxbc file at all").is_empty());
    }

    #[test]
    fn truncated_header() {
        // Just the magic, not enough for a full header
        assert!(scan_dxbc(b"DXBC").is_empty());
    }

    #[test]
    fn display_format() {
        let dxbc = build_dxbc(&[(b"SHEX", &[0u8; 8])]);
        let containers = scan_dxbc(&dxbc);
        let display = format!("{}", containers[0]);
        assert!(display.contains("DXBC at 0x0"), "got: {display}");
        assert!(display.contains("chunks=1"), "got: {display}");
    }

    #[test]
    fn chunk_data_integrity() {
        let payload: Vec<u8> = (0..128).collect();
        let dxbc = build_dxbc(&[(b"TEST", &payload)]);
        let containers = scan_dxbc(&dxbc);
        assert_eq!(containers[0].chunks[0].data, payload);
    }
}
