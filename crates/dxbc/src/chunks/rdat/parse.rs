//! RDAT chunk parser.
//!
//! Binary layout:
//!   RuntimeDataHeader { u32 version, u32 part_count }
//!   u32 offsets\[part_count\]   (relative to header start)
//!   For each part at offset:
//!     RuntimeDataPartHeader { u32 type, u32 size }
//!     byte data[ALIGN4(size)]
//!   For record tables, data starts with:
//!     RuntimeDataTableHeader { u32 record_count, u32 record_stride }
//!     byte records[record_count * record_stride]

use alloc::vec::Vec;
use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::types::*;

fn is_record_table(part_type: u32) -> bool {
    match PartType::from_u32(part_type) {
        Some(pt) => pt.is_record_table(),
        // Unknown part types in the record-table range (future extensions).
        None => matches!(part_type, 13..=21),
    }
}

/// Parse an RDAT chunk from raw bytes.
pub fn parse_rdat(data: &[u8]) -> Option<RuntimeData> {
    if data.len() < 8 {
        return None;
    }
    let mut c = SliceCursor::new(data);

    let first_u32 = c.read_u32_le().ok()?;

    // Detect version: if first u32 >= 0x10, it's a versioned header.
    // Pre-release RDAT started directly with part_count (small number).
    let (version, part_count) = if first_u32 >= RDAT_VERSION_10 {
        let pc = c.read_u32_le().ok()? as usize;
        (first_u32, pc)
    } else {
        // Legacy: first u32 is part_count, no version field.
        (0u32, first_u32 as usize)
    };

    // Read offsets array.
    let mut offsets = Vec::with_capacity(part_count);
    for _ in 0..part_count {
        offsets.push(c.read_u32_le().ok()?);
    }

    // Header base: where offsets are relative to.
    // For versioned: offsets are relative to start of RuntimeDataHeader (byte 0).
    // For legacy: offsets don't exist — parts follow sequentially.
    let header_base = if version >= RDAT_VERSION_10 {
        0usize
    } else {
        4usize
    };

    let mut parts = Vec::with_capacity(part_count);

    for offset_entry in offsets.iter().take(part_count) {
        let part_offset = if version >= RDAT_VERSION_10 {
            header_base + *offset_entry as usize
        } else {
            c.position()
        };

        if part_offset + 8 > data.len() {
            break;
        }
        c.seek(SeekFrom::Start(part_offset as u64)).ok()?;

        let part_type = c.read_u32_le().ok()?;
        let part_size = c.read_u32_le().ok()? as usize;
        let part_data_start = c.position();

        if part_data_start + part_size > data.len() {
            break;
        }

        let part_data = &data[part_data_start..part_data_start + part_size];
        let known = PartType::from_u32(part_type);
        let part = match known {
            Some(PartType::StringBuffer) => RdatPart::StringBuffer(Vec::from(part_data)),
            Some(PartType::IndexArrays) => {
                let count = part_size / 4;
                let mut indices = Vec::with_capacity(count);
                for j in 0..count {
                    let off = part_data_start + j * 4;
                    let v = u32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]);
                    indices.push(v);
                }
                RdatPart::IndexArrays(indices)
            }
            Some(PartType::RawBytes) => RdatPart::RawBytes(Vec::from(part_data)),
            Some(pt) if pt.is_record_table() => parse_record_table(part_data, part_type)?,
            _ if is_record_table(part_type) => parse_record_table(part_data, part_type)?,
            _ => RdatPart::Unknown {
                part_type,
                data: Vec::from(part_data),
            },
        };

        parts.push(part);
    }

    Some(RuntimeData { version, parts })
}

fn parse_record_table(data: &[u8], part_type: u32) -> Option<RdatPart> {
    if data.len() < 8 {
        return None;
    }
    let record_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let record_stride = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let stride = record_stride as usize;

    let table_data = &data[8..];
    let mut records = Vec::with_capacity(record_count);
    for i in 0..record_count {
        let start = i * stride;
        let end = start + stride;
        if end > table_data.len() {
            break;
        }
        records.push(Vec::from(&table_data[start..end]));
    }

    Some(RdatPart::RecordTable {
        part_type,
        record_stride,
        records,
    })
}
