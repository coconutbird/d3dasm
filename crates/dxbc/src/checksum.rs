//! DXBC container checksum — the 16-byte digest fxc stores in the container
//! header (bytes 4..20). It is a *modified* MD5 over the container bytes that
//! follow the hash field (offset 20 to end): standard MD5 transform, but the
//! message length is placed at the **start** of the final block and a
//! `(numBits >> 2) | 1` marker at the **end**, rather than MD5's tail padding.
//!
//! Recomputing this lets reassembled (possibly edited) containers carry a
//! valid checksum instead of a stale one.

/// Per-round left-rotation amounts (standard MD5).
#[rustfmt::skip]
const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// Per-round additive constants (standard MD5: floor(2^32 * abs(sin(i+1)))).
#[rustfmt::skip]
const K: [u32; 64] = [
    0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee,
    0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501,
    0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
    0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821,
    0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa,
    0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
    0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
    0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a,
    0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
    0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70,
    0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05,
    0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
    0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039,
    0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
    0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
    0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
];

fn transform(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut message_words = [0u32; 16];
    for (word_index, word) in message_words.iter_mut().enumerate() {
        *word = u32::from_le_bytes([
            block[word_index * 4],
            block[word_index * 4 + 1],
            block[word_index * 4 + 2],
            block[word_index * 4 + 3],
        ]);
    }

    let (mut alpha, mut bravo, mut charlie, mut delta) = (state[0], state[1], state[2], state[3]);
    for round in 0..64 {
        let (mixed, word_index) = if round < 16 {
            ((bravo & charlie) | (!bravo & delta), round)
        } else if round < 32 {
            ((delta & bravo) | (!delta & charlie), (5 * round + 1) % 16)
        } else if round < 48 {
            (bravo ^ charlie ^ delta, (3 * round + 5) % 16)
        } else {
            (charlie ^ (bravo | !delta), (7 * round) % 16)
        };
        let mixed = mixed
            .wrapping_add(alpha)
            .wrapping_add(K[round])
            .wrapping_add(message_words[word_index]);
        alpha = delta;
        delta = charlie;
        charlie = bravo;
        bravo = bravo.wrapping_add(mixed.rotate_left(S[round]));
    }

    state[0] = state[0].wrapping_add(alpha);
    state[1] = state[1].wrapping_add(bravo);
    state[2] = state[2].wrapping_add(charlie);
    state[3] = state[3].wrapping_add(delta);
}

/// Compute the DXBC container checksum over `data` (the bytes after the header
/// hash field, i.e. container offset 20 to end).
///
/// # Panics
///
/// Panics when `data` is larger than the DXBC format's 32-bit length field.
#[must_use]
pub fn dxbc_checksum(data: &[u8]) -> [u8; 16] {
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let num_bytes = data.len();
    let num_bits = u32::try_from(num_bytes)
        .expect("DXBC data length exceeds u32")
        .wrapping_mul(8);

    let whole = num_bytes / 64;
    for i in 0..whole {
        let mut block = [0u8; 64];
        block.copy_from_slice(&data[i * 64..i * 64 + 64]);
        transform(&mut state, &block);
    }

    let left = num_bytes % 64;
    let tail = &data[whole * 64..];

    if left >= 56 {
        // Not enough room: a padded block, then a length/marker block.
        let mut block = [0u8; 64];
        block[..left].copy_from_slice(tail);
        block[left] = 0x80;
        transform(&mut state, &block);

        let mut last = [0u8; 64];
        last[0..4].copy_from_slice(&num_bits.to_le_bytes());
        last[60..64].copy_from_slice(&((num_bits >> 2) | 1).to_le_bytes());
        transform(&mut state, &last);
    } else {
        // One final block: length at the front, data, 0x80, marker at the end.
        let mut block = [0u8; 64];
        block[0..4].copy_from_slice(&num_bits.to_le_bytes());
        block[4..4 + left].copy_from_slice(tail);
        block[4 + left] = 0x80;
        block[60..64].copy_from_slice(&((num_bits >> 2) | 1).to_le_bytes());
        transform(&mut state, &block);
    }

    let mut out = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::dxbc_checksum;

    // Regression lock. Correctness of the algorithm itself is proven externally
    // by reproducing all 1,152 real container hashes (examples/hash_check.rs);
    // this guards against accidental changes to the transform.
    #[test]
    fn known_answer() {
        let input = b"the quick brown fox jumps over the lazy dxbc container hash";
        assert_eq!(
            dxbc_checksum(input),
            [
                0xa1, 0xb2, 0xd9, 0xd5, 0x15, 0x9d, 0x4d, 0xca, 0x96, 0x58, 0x58, 0x3e, 0x71, 0x02,
                0xff, 0x1e,
            ]
        );
    }

    #[test]
    fn deterministic_and_input_sensitive() {
        let a = dxbc_checksum(b"content-a");
        assert_eq!(a, dxbc_checksum(b"content-a"));
        assert_ne!(a, dxbc_checksum(b"content-b"));
    }

    #[test]
    fn handles_block_boundaries() {
        // Exercise the < 56, >= 56, and exact-multiple-of-64 tail paths.
        for len in [0usize, 55, 56, 63, 64, 119, 120, 128] {
            let data: alloc::vec::Vec<u8> = (0..len)
                .map(|i| u8::try_from(i).expect("test input length fits in u8"))
                .collect();
            let h = dxbc_checksum(&data);
            assert_eq!(h, dxbc_checksum(&data), "len {len} not deterministic");
        }
    }
}
