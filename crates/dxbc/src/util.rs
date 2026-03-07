//! Shared low-level helpers for reading binary data.

use alloc::string::String;

/// Read a little-endian `u32` from `data` at the given byte `offset`.
///
/// # Panics
///
/// Panics if `offset + 4 > data.len()`.
#[inline]
pub(crate) fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Read a null-terminated C string starting at `offset`.
///
/// Returns an empty string if `offset` is out of bounds.
/// Non-UTF-8 bytes are replaced with the Unicode replacement character.
pub(crate) fn read_cstring(data: &[u8], offset: usize) -> String {
    if offset >= data.len() {
        return String::new();
    }
    let mut end = offset;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    String::from_utf8_lossy(&data[offset..end]).into_owned()
}
