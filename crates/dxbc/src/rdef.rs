use alloc::vec::Vec;
use core::fmt;

use crate::util::{read_cstring, read_u32};

#[derive(Debug)]
pub struct ResourceDef<'a> {
    pub constant_buffers: Vec<CBufferDef<'a>>,
    pub bindings: Vec<ResourceBinding<'a>>,
    pub creator: &'a str,
}

#[derive(Debug)]
pub struct CBufferDef<'a> {
    pub name: &'a str,
    pub variables: Vec<CBufferVariable<'a>>,
    pub size: u32,
}

#[derive(Debug)]
pub struct CBufferVariable<'a> {
    pub name: &'a str,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug)]
pub struct ResourceBinding<'a> {
    pub name: &'a str,
    pub input_type: u32,
    pub return_type: u32,
    pub dimension: u32,
    pub bind_point: u32,
    pub bind_count: u32,
    pub flags: u32,
}

impl ResourceBinding<'_> {
    fn type_name(&self) -> &'static str {
        match self.input_type {
            0 => "cbuffer",
            1 => "tbuffer",
            2 => "texture",
            3 => "sampler",
            4 => "uav_rwtyped",
            5 => "structured",
            6 => "uav_rwstructured",
            7 => "byteaddress",
            8 => "uav_rwbyteaddress",
            9 => "uav_append_structured",
            10 => "uav_consume_structured",
            11 => "uav_rwstructured_with_counter",
            _ => "unknown",
        }
    }

    fn dim_name(&self) -> &'static str {
        match self.dimension {
            1 => "buf",
            2 => "1d",
            3 => "2d",
            4 => "2dMS",
            5 => "3d",
            6 => "cube",
            7 => "1darray",
            8 => "2darray",
            9 => "2dMSarray",
            10 => "cubearray",
            _ => "NA",
        }
    }

    /// Format the type, dimension, and slot columns (everything after the name).
    pub fn format_columns(&self) -> alloc::string::String {
        alloc::format!(
            "{:<12} {:<8} {}",
            self.type_name(),
            self.dim_name(),
            self.bind_point
        )
    }
}

impl fmt::Display for ResourceBinding<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<30} {}", self.name, self.format_columns())
    }
}

/// Parse an RDEF chunk.
///
/// RDEF header layout (28 bytes):
///   0: u32 constant_buffer_count
///   4: u32 constant_buffer_offset
///   8: u32 bound_resource_count
///  12: u32 bound_resource_offset
///  16: u32 target_version  (minor | major<<8 | type<<16)
///  20: u32 flags
///  24: u32 creator_offset
pub fn parse_rdef(data: &[u8]) -> Option<ResourceDef<'_>> {
    if data.len() < 28 {
        return None;
    }

    let cb_count = read_u32(data, 0) as usize;
    let cb_offset = read_u32(data, 4) as usize;
    let binding_count = read_u32(data, 8) as usize;
    let binding_offset = read_u32(data, 12) as usize;
    let target_version = read_u32(data, 16);

    // SM5 uses 40-byte variable descriptors; SM4 uses 24-byte.
    let major_version = (target_version >> 8) & 0xFF;
    let var_stride: usize = if major_version >= 5 { 40 } else { 24 };

    // Parse resource bindings
    let mut bindings = Vec::with_capacity(binding_count);
    for i in 0..binding_count {
        let base = binding_offset + i * 32;
        if base + 32 > data.len() {
            break;
        }
        let name_off = read_u32(data, base) as usize;
        bindings.push(ResourceBinding {
            name: read_cstring(data, name_off),
            input_type: read_u32(data, base + 4),
            return_type: read_u32(data, base + 8),
            dimension: read_u32(data, base + 12),
            bind_point: read_u32(data, base + 20),
            bind_count: read_u32(data, base + 24),
            flags: read_u32(data, base + 28),
        });
    }

    // Parse constant buffers
    let mut constant_buffers = Vec::with_capacity(cb_count);
    for i in 0..cb_count {
        let base = cb_offset + i * 24;
        if base + 24 > data.len() {
            break;
        }
        let name_off = read_u32(data, base) as usize;
        let var_count = read_u32(data, base + 4) as usize;
        let var_offset = read_u32(data, base + 8) as usize;
        let cb_size = read_u32(data, base + 12);

        let mut variables = Vec::with_capacity(var_count);
        for j in 0..var_count {
            let vbase = var_offset + j * var_stride;
            if vbase + var_stride > data.len() {
                break;
            }
            let vname_off = read_u32(data, vbase) as usize;
            variables.push(CBufferVariable {
                name: read_cstring(data, vname_off),
                offset: read_u32(data, vbase + 4),
                size: read_u32(data, vbase + 8),
            });
        }

        constant_buffers.push(CBufferDef {
            name: read_cstring(data, name_off),
            variables,
            size: cb_size,
        });
    }

    // Creator string (at offset 24 in the RDEF header)
    let creator = if data.len() >= 28 {
        let creator_off = read_u32(data, 24) as usize;
        if creator_off < data.len() {
            read_cstring(data, creator_off)
        } else {
            ""
        }
    } else {
        ""
    };

    Some(ResourceDef {
        constant_buffers,
        bindings,
        creator,
    })
}
