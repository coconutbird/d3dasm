//! Strict parser for the versioned RDAT runtime-reflection layout.

use alloc::vec::Vec;

use super::types::{PartType, RdatPart, RuntimeData, RuntimeDataVersion};

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    Some(u32::from_le_bytes(data.get(offset..end)?.try_into().ok()?))
}

/// Parse an RDAT payload.
#[must_use]
pub fn parse_rdat(data: &[u8]) -> Option<RuntimeData> {
    if data.len() < 8 || data.len() % 4 != 0 {
        return None;
    }
    let version = RuntimeDataVersion::from_u32(read_u32(data, 0)?)?;
    let part_count = usize::try_from(read_u32(data, 4)?).ok()?;
    let offsets_size = part_count.checked_mul(4)?;
    let header_end = 8usize.checked_add(offsets_size)?;
    if header_end > data.len() {
        return None;
    }

    let mut offsets = Vec::with_capacity(part_count);
    for index in 0..part_count {
        let entry = 8usize.checked_add(index.checked_mul(4)?)?;
        let offset = usize::try_from(read_u32(data, entry)?).ok()?;
        if offset < header_end || offset % 4 != 0 || offset.checked_add(8)? > data.len() {
            return None;
        }
        offsets.push(offset);
    }

    let mut parts = Vec::with_capacity(part_count);
    let mut previous_end = header_end;
    for part_offset in offsets {
        // DXC's CheckedReader advances monotonically through offsets in table
        // order. Gaps are allowed, overlaps and backwards offsets are not.
        if part_offset < previous_end {
            return None;
        }
        let part_type_value = read_u32(data, part_offset)?;
        let part_type = PartType::from_u32(part_type_value);
        if part_type == Some(PartType::Invalid) {
            return None;
        }
        let part_size = usize::try_from(read_u32(data, part_offset + 4)?).ok()?;
        if part_size % 4 != 0 {
            return None;
        }
        let part_data_start = part_offset.checked_add(8)?;
        let part_data_end = part_data_start.checked_add(part_size)?;
        if part_data_end > data.len() {
            return None;
        }
        previous_end = part_data_end;
        let part_data = &data[part_data_start..part_data_end];

        let part = match part_type {
            Some(PartType::StringBuffer) => {
                if !part_data.is_empty() && part_data.last() != Some(&0) {
                    return None;
                }
                RdatPart::StringBuffer(Vec::from(part_data))
            }
            Some(PartType::IndexArrays) => {
                let mut values = Vec::with_capacity(part_data.len() / 4);
                for bytes in part_data.chunks_exact(4) {
                    values.push(u32::from_le_bytes(bytes.try_into().ok()?));
                }
                RdatPart::IndexArrays(values)
            }
            Some(PartType::RawBytes) => RdatPart::RawBytes(Vec::from(part_data)),
            Some(kind) if kind.is_record_table() => parse_record_table(part_data, part_type_value)?,
            _ => RdatPart::Unknown {
                part_type: part_type_value,
                data: Vec::from(part_data),
            },
        };
        parts.push(part);
    }

    let mut runtime = RuntimeData {
        version,
        parts,
        original_data: None,
        original_canonical: None,
    };
    runtime.original_canonical = Some(super::write::write_rdat_canonical(&runtime));
    runtime.original_data = Some(data.to_vec());
    Some(runtime)
}

fn parse_record_table(data: &[u8], part_type: u32) -> Option<RdatPart> {
    if data.len() < 8 {
        return None;
    }
    let record_count = usize::try_from(read_u32(data, 0)?).ok()?;
    let record_stride = read_u32(data, 4)?;
    let stride = usize::try_from(record_stride).ok()?;
    if stride == 0 || stride % 4 != 0 {
        return None;
    }
    let records_size = record_count.checked_mul(stride)?;
    if 8usize.checked_add(records_size)? != data.len() {
        return None;
    }

    let mut records = Vec::with_capacity(record_count);
    for record in data[8..].chunks_exact(stride) {
        records.push(Vec::from(record));
    }
    Some(RdatPart::RecordTable {
        part_type,
        record_stride,
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunks::ChunkWriter;

    #[test]
    fn rejects_prerelease_and_unaligned_parts() {
        assert!(parse_rdat(&[1, 0, 0, 0, 0, 0, 0, 0]).is_none());
        let runtime = RuntimeData {
            version: RuntimeDataVersion::V1_0,
            parts: alloc::vec![RdatPart::RawBytes(alloc::vec![1, 2, 3])],
            original_data: None,
            original_canonical: None,
        };
        let bytes = runtime.write_payload();
        assert_eq!(read_u32(&bytes, 16), Some(4));
        assert_eq!(parse_rdat(&bytes).expect("aligned RDAT").parts.len(), 1);
    }

    #[test]
    fn accepts_additive_versions_and_non_utf8_string_bytes() {
        let runtime = RuntimeData {
            version: RuntimeDataVersion::Newer(0x11),
            parts: alloc::vec![RdatPart::StringBuffer(alloc::vec![0xff, 0])],
            original_data: None,
            original_canonical: None,
        };
        let bytes = runtime.write_payload();
        let parsed = parse_rdat(&bytes).expect("additive RDAT version");
        assert_eq!(parsed.version, RuntimeDataVersion::Newer(0x11));
        assert_eq!(parsed.write_payload(), bytes);
    }

    #[test]
    fn valid_noncanonical_offsets_are_preserved_until_edited() {
        let runtime = RuntimeData {
            version: RuntimeDataVersion::V1_0,
            parts: alloc::vec![RdatPart::RawBytes(alloc::vec![1, 2, 3, 4])],
            original_data: None,
            original_canonical: None,
        };
        let canonical = runtime.write_payload();
        let mut noncanonical = canonical.clone();
        noncanonical[8..12].copy_from_slice(&16u32.to_le_bytes());
        noncanonical.splice(12..12, [0xde, 0xad, 0xbe, 0xef]);

        let mut parsed = parse_rdat(&noncanonical).expect("valid offset-based RDAT");
        assert_eq!(parsed.write_payload(), noncanonical);
        let RdatPart::RawBytes(data) = &mut parsed.parts[0] else {
            panic!("raw bytes expected");
        };
        data[0] = 9;
        let edited = parsed.write_payload();
        assert_ne!(edited, noncanonical);
        assert_eq!(
            parse_rdat(&edited)
                .expect("canonical edited RDAT")
                .parts
                .len(),
            1
        );
    }
}
