/// Decode all operands from an instruction's token stream starting at `start` index.
pub fn decode_operands(tokens: &[u32], start: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut pos = start;
    while pos < tokens.len() {
        let (s, consumed) = decode_one(tokens, pos);
        if consumed == 0 {
            break;
        }
        result.push(s);
        pos += consumed;
    }
    result
}

/// Decode a single operand, returning (formatted_string, tokens_consumed).
fn decode_one(tokens: &[u32], pos: usize) -> (String, usize) {
    if pos >= tokens.len() {
        return (String::new(), 0);
    }
    let token = tokens[pos];
    let num_components = token & 0x3;
    let op_type = (token >> 12) & 0xFF;
    let index_dim = (token >> 20) & 0x3;
    let is_extended = (token >> 31) & 1 != 0;

    let mut consumed = 1usize;

    // Handle extended operand modifier
    let (negate, abs) = if is_extended && pos + consumed < tokens.len() {
        let ext = tokens[pos + consumed];
        consumed += 1;
        let neg = (ext >> 6) & 1 != 0;
        let ab = (ext >> 7) & 1 != 0;
        (neg, ab)
    } else {
        (false, false)
    };

    // Decode swizzle/mask
    let swizzle = decode_swizzle(token, num_components);

    // Decode the register name and indices
    let (name, idx_consumed) = decode_register(tokens, pos + consumed, op_type, index_dim);
    consumed += idx_consumed;

    let mut s = name;
    if !swizzle.is_empty() {
        s.push('.');
        s.push_str(&swizzle);
    }

    if negate && abs {
        s = format!("-|{s}|");
    } else if negate {
        s = format!("-{s}");
    } else if abs {
        s = format!("|{s}|");
    }

    (s, consumed)
}

fn decode_swizzle(token: u32, num_components: u32) -> String {
    let sel_mode = (token >> 2) & 0x3;
    match num_components {
        0 => String::new(), // 0-component
        1 => {
            // 1-component (mask)
            let mask = (token >> 4) & 0xF;
            format_mask(mask as u8)
        }
        2 => {
            match sel_mode {
                0 => {
                    // mask
                    let mask = (token >> 4) & 0xF;
                    format_mask(mask as u8)
                }
                1 => {
                    // swizzle
                    let mut s = String::with_capacity(4);
                    let comps = ['x', 'y', 'z', 'w'];
                    for i in 0..4 {
                        let c = ((token >> (4 + i * 2)) & 0x3) as usize;
                        s.push(comps[c]);
                    }
                    // Trim trailing repeated components
                    trim_swizzle(&s)
                }
                2 => {
                    // scalar select
                    let c = ((token >> 4) & 0x3) as usize;
                    ['x', 'y', 'z', 'w'][c].to_string()
                }
                _ => String::new(),
            }
        }
        _ => String::new(),
    }
}

fn format_mask(mask: u8) -> String {
    let mut s = String::with_capacity(4);
    if mask & 1 != 0 {
        s.push('x');
    }
    if mask & 2 != 0 {
        s.push('y');
    }
    if mask & 4 != 0 {
        s.push('z');
    }
    if mask & 8 != 0 {
        s.push('w');
    }
    s
}

fn trim_swizzle(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 1 {
        return s.to_string();
    }
    // If all same, return just one
    if chars.iter().all(|&c| c == chars[0]) {
        return chars[0].to_string();
    }
    // If last N are same, trim
    let mut end = chars.len();
    while end > 1 && chars[end - 1] == chars[end - 2] {
        end -= 1;
    }
    chars[..end].iter().collect()
}

fn decode_register(tokens: &[u32], pos: usize, op_type: u32, index_dim: u32) -> (String, usize) {
    let prefix = register_prefix(op_type);
    let mut consumed = 0usize;

    match index_dim {
        0 => {
            // 0D - no index (e.g., immediate values or special)
            if op_type == 4 {
                // Immediate32
                if pos + 4 <= tokens.len() {
                    let vals: Vec<String> =
                        (0..4).map(|i| format_immediate(tokens[pos + i])).collect();
                    consumed = 4;
                    return (format!("l({})", vals.join(", ")), consumed);
                } else if pos < tokens.len() {
                    consumed = 1;
                    return (format!("l({})", format_immediate(tokens[pos])), consumed);
                }
            }
            (prefix.to_string(), consumed)
        }
        1 => {
            // 1D index
            let (idx, ic) = decode_index(
                tokens,
                pos,
                (tokens
                    .get(pos.wrapping_sub(consumed + 1))
                    .copied()
                    .unwrap_or(0)
                    >> 22)
                    & 0x3,
            );
            consumed += ic;
            (format!("{prefix}{idx}"), consumed)
        }
        2 => {
            // 2D index (e.g., cb0[1])
            let parent_token = if pos >= 1 {
                tokens[pos - 1 - consumed]
            } else {
                0
            };
            let idx0_repr = (parent_token >> 22) & 0x3;
            let idx1_repr = (parent_token >> 24) & 0x3;
            let (idx0, ic0) = decode_index(tokens, pos, idx0_repr);
            consumed += ic0;
            let (idx1, ic1) = decode_index(tokens, pos + ic0, idx1_repr);
            consumed += ic1;
            (format!("{prefix}{idx0}[{idx1}]"), consumed)
        }
        3 => {
            // 3D index
            let (idx0, ic0) = decode_index(tokens, pos, 0);
            consumed += ic0;
            let (idx1, ic1) = decode_index(tokens, pos + ic0, 0);
            consumed += ic1;
            let (idx2, ic2) = decode_index(tokens, pos + ic0 + ic1, 0);
            consumed += ic2;
            (format!("{prefix}{idx0}[{idx1}][{idx2}]"), consumed)
        }
        _ => (prefix.to_string(), consumed),
    }
}

fn decode_index(tokens: &[u32], pos: usize, repr: u32) -> (String, usize) {
    match repr {
        0 => {
            // Immediate32
            if pos < tokens.len() {
                (format!("{}", tokens[pos]), 1)
            } else {
                ("?".to_string(), 0)
            }
        }
        1 => {
            // Immediate64
            if pos + 1 < tokens.len() {
                let val = tokens[pos] as u64 | ((tokens[pos + 1] as u64) << 32);
                (format!("{val}"), 2)
            } else {
                ("?".to_string(), 0)
            }
        }
        2 => {
            // Relative (register + offset)
            // The next tokens encode a sub-operand
            let (sub, consumed) = decode_one(tokens, pos);
            (sub, consumed)
        }
        3 => {
            // Relative + immediate
            if pos < tokens.len() {
                let imm = tokens[pos];
                let (sub, consumed) = decode_one(tokens, pos + 1);
                (format!("{imm} + {sub}"), 1 + consumed)
            } else {
                ("?".to_string(), 0)
            }
        }
        _ => ("?".to_string(), 0),
    }
}

fn format_immediate(val: u32) -> String {
    let f = f32::from_bits(val);
    if f == 0.0 || f == 1.0 || f == -1.0 || f == 0.5 || f == -0.5 || f == 2.0 {
        format!("{f:.6}")
    } else if val == 0 {
        "0".to_string()
    } else if f.is_finite() && f.abs() > 0.0001 && f.abs() < 1_000_000.0 {
        format!("{f:.6}")
    } else {
        // Could be int or unusual float
        format!("0x{val:08X}")
    }
}

fn register_prefix(op_type: u32) -> &'static str {
    match op_type {
        0 => "r",       // temp
        1 => "v",       // input
        2 => "o",       // output
        3 => "x",       // indexable temp
        4 => "l",       // immediate32 (handled specially)
        5 => "d",       // immediate64
        6 => "s",       // sampler
        7 => "t",       // resource (texture)
        8 => "cb",      // constant buffer
        9 => "icb",     // immediate constant buffer
        10 => "label",  // label
        11 => "vPrim",  // input primitive id
        12 => "oDepth", // output depth
        13 => "null",   // null
        14 => "rasterizer",
        15 => "oMask", // output coverage mask
        16 => "stream",
        17 => "function_body",
        18 => "function_table",
        19 => "interface",
        20 => "function_input",
        21 => "function_output",
        22 => "vOutputControlPointID",
        23 => "vForkInstanceID",
        24 => "vJoinInstanceID",
        25 => "vicp", // input control point
        26 => "vocp", // output control point
        27 => "vpc",  // patch constant
        28 => "vDomain",
        29 => "thisPointer",
        30 => "u", // UAV
        31 => "g", // thread group shared memory
        32 => "vThreadID",
        33 => "vThreadGroupID",
        34 => "vThreadIDInGroup",
        35 => "vCoverage",
        36 => "vThreadIDInGroupFlattened",
        37 => "vGSInstanceID",
        38 => "oDepthGE",
        39 => "oDepthLE",
        40 => "vCycleCounter",
        _ => "?reg",
    }
}
