//! RDAT chunk writer — reconstruct RDAT binary from parsed fields.

use alloc::vec::Vec;

use super::types::*;

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn align4(size: usize) -> usize {
    (size + 3) & !3
}

/// Serialize a single part's data (excluding RuntimeDataPartHeader).
fn write_part_data(part: &RdatPart) -> Vec<u8> {
    match part {
        RdatPart::StringBuffer(buf) => buf.clone(),
        RdatPart::IndexArrays(indices) => {
            let mut out = Vec::with_capacity(indices.len() * 4);
            for &v in indices {
                write_u32(&mut out, v);
            }
            out
        }
        RdatPart::RawBytes(buf) => buf.clone(),
        RdatPart::RecordTable {
            record_stride,
            records,
            ..
        } => {
            let mut out = Vec::with_capacity(8 + records.len() * *record_stride as usize);
            write_u32(&mut out, records.len() as u32);
            write_u32(&mut out, *record_stride);
            for rec in records {
                out.extend_from_slice(rec);
                // Pad record to stride if shorter.
                if rec.len() < *record_stride as usize {
                    out.resize(out.len() + (*record_stride as usize - rec.len()), 0);
                }
            }
            out
        }
        RdatPart::Unknown { data, .. } => data.clone(),
    }
}

fn part_type_id(part: &RdatPart) -> u32 {
    match part {
        RdatPart::StringBuffer(_) => PartType::StringBuffer as u32,
        RdatPart::IndexArrays(_) => PartType::IndexArrays as u32,
        RdatPart::RawBytes(_) => PartType::RawBytes as u32,
        RdatPart::RecordTable { part_type, .. } => *part_type,
        RdatPart::Unknown { part_type, .. } => *part_type,
    }
}

/// Reconstruct an RDAT chunk payload from parsed fields.
pub fn write_rdat(rdat: &RuntimeData) -> Vec<u8> {
    let part_count = rdat.parts.len();

    // Serialize all part payloads first so we know sizes.
    let payloads: Vec<Vec<u8>> = rdat.parts.iter().map(write_part_data).collect();

    let versioned = rdat.version >= RDAT_VERSION_10;

    if versioned {
        // Header: version(4) + part_count(4) + offsets(4*part_count)
        let header_size = 8 + 4 * part_count;

        // Compute offsets relative to header start (byte 0).
        let mut offsets = Vec::with_capacity(part_count);
        let mut cursor = header_size;
        for payload in &payloads {
            offsets.push(cursor as u32);
            // PartHeader (8) + aligned payload
            cursor += 8 + align4(payload.len());
        }

        let mut out = Vec::with_capacity(cursor);
        write_u32(&mut out, rdat.version);
        write_u32(&mut out, part_count as u32);
        for &off in &offsets {
            write_u32(&mut out, off);
        }

        for (i, payload) in payloads.iter().enumerate() {
            write_u32(&mut out, part_type_id(&rdat.parts[i]));
            write_u32(&mut out, payload.len() as u32);
            out.extend_from_slice(payload);
            // Pad to 4-byte alignment.
            let padded = align4(payload.len());
            out.resize(out.len() + padded - payload.len(), 0);
        }

        out
    } else {
        // Legacy: part_count + sequential parts (type + size + data)
        let mut out = Vec::new();
        write_u32(&mut out, part_count as u32);
        for (i, payload) in payloads.iter().enumerate() {
            write_u32(&mut out, part_type_id(&rdat.parts[i]));
            write_u32(&mut out, payload.len() as u32);
            out.extend_from_slice(payload);
            let padded = align4(payload.len());
            out.resize(out.len() + padded - payload.len(), 0);
        }
        out
    }
}
