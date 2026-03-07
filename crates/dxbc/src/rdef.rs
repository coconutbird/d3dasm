use std::fmt;

#[derive(Debug)]
pub struct ResourceDef {
    pub constant_buffers: Vec<CBufferDef>,
    pub bindings: Vec<ResourceBinding>,
    pub creator: String,
}

#[derive(Debug)]
pub struct CBufferDef {
    pub name: String,
    pub variables: Vec<CBufferVariable>,
    pub size: u32,
}

#[derive(Debug)]
pub struct CBufferVariable {
    pub name: String,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug)]
pub struct ResourceBinding {
    pub name: String,
    pub input_type: u32,
    pub return_type: u32,
    pub dimension: u32,
    pub bind_point: u32,
    pub bind_count: u32,
    pub flags: u32,
}

impl fmt::Display for ResourceBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = match self.input_type {
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
        };
        let dim = match self.dimension {
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
            _ => "",
        };
        write!(
            f,
            "{:<30} {:<12} {:<8} {}",
            self.name, ty, dim, self.bind_point
        )
    }
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn read_cstring(data: &[u8], offset: usize) -> String {
    if offset >= data.len() {
        return String::new();
    }
    let mut end = offset;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    String::from_utf8_lossy(&data[offset..end]).into_owned()
}

/// Parse an RDEF chunk.
pub fn parse_rdef(data: &[u8]) -> Option<ResourceDef> {
    if data.len() < 28 {
        return None;
    }

    let cb_count = read_u32(data, 0) as usize;
    let cb_offset = read_u32(data, 4) as usize;
    let binding_count = read_u32(data, 8) as usize;
    let binding_offset = read_u32(data, 12) as usize;

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
            let vbase = var_offset + j * 24;
            if vbase + 24 > data.len() {
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

    // Creator string
    let creator_off = read_u32(data, 16) as usize;
    let creator = if creator_off < data.len() {
        read_cstring(data, creator_off)
    } else {
        String::new()
    };

    Some(ResourceDef {
        constant_buffers,
        bindings,
        creator,
    })
}
