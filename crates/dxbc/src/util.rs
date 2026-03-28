//! Shared low-level helpers for reading binary data.

/// Read a little-endian `u32` from `data` at the given byte `offset`.
///
/// # Panics
///
/// Panics if `offset + 4 > data.len()`.
#[inline]
pub fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

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
