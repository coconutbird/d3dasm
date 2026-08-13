//! RDAT writer for the versioned 1.0 container layout.

use alloc::vec::Vec;

use super::types::{PartType, RdatPart, RuntimeData};

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn wire_u32(value: usize) -> u32 {
    u32::try_from(value).expect("RDAT size or offset exceeds u32")
}

fn align_part(mut data: Vec<u8>) -> Vec<u8> {
    let aligned = data.len().checked_add(3).expect("RDAT part size overflow") & !3;
    data.resize(aligned, 0);
    data
}

fn write_part_data(part: &RdatPart) -> Vec<u8> {
    let data = match part {
        RdatPart::StringBuffer(buffer) => {
            assert!(
                buffer.is_empty() || buffer.last() == Some(&0),
                "non-empty RDAT string buffer must end in null"
            );
            buffer.clone()
        }
        RdatPart::RawBytes(buffer) | RdatPart::Unknown { data: buffer, .. } => buffer.clone(),
        RdatPart::IndexArrays(indices) => {
            let capacity = indices
                .len()
                .checked_mul(4)
                .expect("RDAT index array size overflow");
            let mut out = Vec::with_capacity(capacity);
            for value in indices {
                write_u32(&mut out, *value);
            }
            out
        }
        RdatPart::RecordTable {
            record_stride,
            records,
            ..
        } => {
            let stride = usize::try_from(*record_stride).expect("u32 fits in usize");
            assert!(stride > 0 && stride % 4 == 0, "invalid RDAT record stride");
            let capacity = 8usize
                .checked_add(
                    records
                        .len()
                        .checked_mul(stride)
                        .expect("RDAT record table size overflow"),
                )
                .expect("RDAT record table size overflow");
            let mut out = Vec::with_capacity(capacity);
            write_u32(&mut out, wire_u32(records.len()));
            write_u32(&mut out, *record_stride);
            for record in records {
                assert_eq!(record.len(), stride, "RDAT record does not match stride");
                out.extend_from_slice(record);
            }
            out
        }
    };
    align_part(data)
}

fn part_type_id(part: &RdatPart) -> u32 {
    match part {
        RdatPart::StringBuffer(_) => PartType::StringBuffer as u32,
        RdatPart::IndexArrays(_) => PartType::IndexArrays as u32,
        RdatPart::RawBytes(_) => PartType::RawBytes as u32,
        RdatPart::RecordTable { part_type, .. } | RdatPart::Unknown { part_type, .. } => *part_type,
    }
}

/// Serialize an RDAT payload.
#[must_use]
pub(super) fn write_rdat_canonical(rdat: &RuntimeData) -> Vec<u8> {
    assert!(
        rdat.version.to_u32() >= 0x10,
        "RDAT version predates the versioned layout"
    );
    let payloads: Vec<Vec<u8>> = rdat.parts.iter().map(write_part_data).collect();
    let part_count = rdat.parts.len();
    let header_size = 8usize
        .checked_add(
            part_count
                .checked_mul(4)
                .expect("RDAT offset table overflow"),
        )
        .expect("RDAT header size overflow");
    let mut offsets = Vec::with_capacity(part_count);
    let mut total_size = header_size;
    for payload in &payloads {
        offsets.push(wire_u32(total_size));
        total_size = total_size
            .checked_add(8)
            .and_then(|size| size.checked_add(payload.len()))
            .expect("RDAT size overflow");
    }

    let mut out = Vec::with_capacity(total_size);
    write_u32(&mut out, rdat.version.to_u32());
    write_u32(&mut out, wire_u32(part_count));
    for offset in offsets {
        write_u32(&mut out, offset);
    }
    for (part, payload) in rdat.parts.iter().zip(payloads) {
        write_u32(&mut out, part_type_id(part));
        // The specification requires the declared size itself to be aligned.
        write_u32(&mut out, wire_u32(payload.len()));
        out.extend_from_slice(&payload);
    }
    out
}

/// Serialize an RDAT payload, preserving an accepted non-canonical source
/// layout until its semantic fields are edited.
#[must_use]
pub fn write_rdat(rdat: &RuntimeData) -> Vec<u8> {
    let canonical = write_rdat_canonical(rdat);
    if rdat.original_canonical.as_deref() == Some(canonical.as_slice()) {
        if let Some(original) = &rdat.original_data {
            return original.clone();
        }
    }
    canonical
}
