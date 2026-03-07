use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::util::{read_cstring, read_u32};

/// A parsed input or output signature element.
#[derive(Debug)]
pub struct SignatureElement {
    pub semantic_name: String,
    pub semantic_index: u32,
    pub system_value: u32,
    pub component_type: u32,
    pub register: u32,
    pub mask: u8,
    pub rw_mask: u8,
}

impl fmt::Display for SignatureElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let comp = match self.component_type {
            1 => "uint",
            2 => "int",
            3 => "float",
            _ => "unknown",
        };
        let mask_str = format_mask(self.mask);
        let idx = if self.semantic_index > 0 {
            format!("{}", self.semantic_index)
        } else {
            String::new()
        };
        write!(
            f,
            "{}{:<20} {} v{}.{}",
            self.semantic_name, idx, comp, self.register, mask_str
        )
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

/// Parse an ISGN, OSGN, ISG1, OSG1, or OSG5 chunk.
pub fn parse_signature(data: &[u8]) -> Vec<SignatureElement> {
    if data.len() < 8 {
        return Vec::new();
    }
    let count = read_u32(data, 0) as usize;
    // skip 8 bytes header (count + 8-byte constant)
    let _eight = read_u32(data, 4);
    let mut elements = Vec::with_capacity(count);

    for i in 0..count {
        let base = 8 + i * 24;
        if base + 24 > data.len() {
            break;
        }
        let name_offset = read_u32(data, base) as usize;
        let semantic_index = read_u32(data, base + 4);
        let system_value = read_u32(data, base + 8);
        let component_type = read_u32(data, base + 12);
        let register = read_u32(data, base + 16);
        let mask = data[base + 20];
        let rw_mask = data[base + 21];

        let name = read_cstring(data, name_offset);

        elements.push(SignatureElement {
            semantic_name: name,
            semantic_index,
            system_value,
            component_type,
            register,
            mask,
            rw_mask,
        });
    }
    elements
}
