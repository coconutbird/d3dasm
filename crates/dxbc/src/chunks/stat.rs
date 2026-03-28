//! Parser for the STAT (statistics) chunk in DXBC shader bytecode.

use core::fmt;

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::{ChunkParser, ChunkWriter};

/// Shader statistics extracted from the STAT chunk.
#[derive(Debug, Clone, Default)]
pub struct ShaderStats {
    pub instruction_count: u32,
    pub temp_register_count: u32,
    pub define_count: u32,
    pub declaration_count: u32,
    pub float_instruction_count: u32,
    pub int_instruction_count: u32,
    pub uint_instruction_count: u32,
    pub static_flow_control_count: u32,
    pub dynamic_flow_control_count: u32,
    pub macro_instruction_count: u32,
    pub temp_array_count: u32,
    pub array_instruction_count: u32,
    pub cut_instruction_count: u32,
    pub emit_instruction_count: u32,
    pub texture_normal_instructions: u32,
    pub texture_load_instructions: u32,
    pub texture_comp_instructions: u32,
    pub texture_bias_instructions: u32,
    pub texture_gradient_instructions: u32,
    pub mov_instruction_count: u32,
    pub movc_instruction_count: u32,
    pub conversion_instruction_count: u32,
    pub gs_input_primitive: u32,
    pub gs_output_topology: u32,
    pub gs_max_output_vertex_count: u32,
    pub is_sample_frequency: bool,
}

/// Parse a STAT chunk into [`ShaderStats`].
///
/// Returns `None` if the data is too short to contain a valid STAT chunk.
pub fn parse_stat(data: &[u8]) -> Option<ShaderStats> {
    if data.len() < 116 {
        return None;
    }

    let mut c = SliceCursor::new(data);
    let r = &mut c;
    Some(ShaderStats {
        instruction_count: r.read_u32_le().ok()?,
        temp_register_count: r.read_u32_le().ok()?,
        define_count: r.read_u32_le().ok()?,
        declaration_count: r.read_u32_le().ok()?,
        float_instruction_count: r.read_u32_le().ok()?,
        int_instruction_count: r.read_u32_le().ok()?,
        uint_instruction_count: r.read_u32_le().ok()?,
        static_flow_control_count: r.read_u32_le().ok()?,
        dynamic_flow_control_count: r.read_u32_le().ok()?,
        macro_instruction_count: r.read_u32_le().ok()?,
        temp_array_count: r.read_u32_le().ok()?,
        array_instruction_count: r.read_u32_le().ok()?,
        cut_instruction_count: r.read_u32_le().ok()?,
        emit_instruction_count: r.read_u32_le().ok()?,
        texture_normal_instructions: r.read_u32_le().ok()?,
        texture_load_instructions: r.read_u32_le().ok()?,
        texture_comp_instructions: r.read_u32_le().ok()?,
        texture_bias_instructions: r.read_u32_le().ok()?,
        texture_gradient_instructions: r.read_u32_le().ok()?,
        mov_instruction_count: r.read_u32_le().ok()?,
        movc_instruction_count: r.read_u32_le().ok()?,
        conversion_instruction_count: r.read_u32_le().ok()?,
        // Offsets 88 and 92 are unknown fields — skip 2 u32s
        gs_input_primitive: {
            r.seek(SeekFrom::Current(8)).ok()?;
            r.read_u32_le().ok()?
        },
        gs_output_topology: r.read_u32_le().ok()?,
        gs_max_output_vertex_count: r.read_u32_le().ok()?,
        // Offsets 108 and 112 are unknown fields
        is_sample_frequency: {
            if data.len() >= 120 {
                r.seek(SeekFrom::Start(116)).ok()?;
                r.read_u32_le().ok()? != 0
            } else {
                false
            }
        },
    })
}

impl ChunkParser for ShaderStats {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_stat(data)
    }
}

impl ChunkWriter for ShaderStats {
    fn fourcc(&self) -> [u8; 4] {
        *b"STAT"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec::Vec::with_capacity(120);
        let w = |buf: &mut alloc::vec::Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        w(&mut buf, self.instruction_count);
        w(&mut buf, self.temp_register_count);
        w(&mut buf, self.define_count);
        w(&mut buf, self.declaration_count);
        w(&mut buf, self.float_instruction_count);
        w(&mut buf, self.int_instruction_count);
        w(&mut buf, self.uint_instruction_count);
        w(&mut buf, self.static_flow_control_count);
        w(&mut buf, self.dynamic_flow_control_count);
        w(&mut buf, self.macro_instruction_count);
        w(&mut buf, self.temp_array_count);
        w(&mut buf, self.array_instruction_count);
        w(&mut buf, self.cut_instruction_count);
        w(&mut buf, self.emit_instruction_count);
        w(&mut buf, self.texture_normal_instructions);
        w(&mut buf, self.texture_load_instructions);
        w(&mut buf, self.texture_comp_instructions);
        w(&mut buf, self.texture_bias_instructions);
        w(&mut buf, self.texture_gradient_instructions);
        w(&mut buf, self.mov_instruction_count);
        w(&mut buf, self.movc_instruction_count);
        w(&mut buf, self.conversion_instruction_count);
        w(&mut buf, 0); // unknown at offset 88
        w(&mut buf, 0); // unknown at offset 92
        w(&mut buf, self.gs_input_primitive);
        w(&mut buf, self.gs_output_topology);
        w(&mut buf, self.gs_max_output_vertex_count);
        w(&mut buf, 0); // unknown at offset 108
        w(&mut buf, 0); // unknown at offset 112
        w(&mut buf, self.is_sample_frequency as u32);
        buf
    }
}

impl fmt::Display for ShaderStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Statistics:")?;
        writeln!(f, "//   {} instruction(s)", self.instruction_count)?;
        writeln!(f, "//   {} temp register(s)", self.temp_register_count)?;
        if self.declaration_count > 0 {
            writeln!(f, "//   {} declaration(s)", self.declaration_count)?;
        }
        if self.float_instruction_count > 0 {
            writeln!(
                f,
                "//   {} float instruction(s)",
                self.float_instruction_count
            )?;
        }
        if self.int_instruction_count > 0 {
            writeln!(f, "//   {} int instruction(s)", self.int_instruction_count)?;
        }
        if self.uint_instruction_count > 0 {
            writeln!(
                f,
                "//   {} uint instruction(s)",
                self.uint_instruction_count
            )?;
        }
        if self.texture_normal_instructions > 0 {
            writeln!(
                f,
                "//   {} texture normal instruction(s)",
                self.texture_normal_instructions
            )?;
        }
        if self.texture_load_instructions > 0 {
            writeln!(
                f,
                "//   {} texture load instruction(s)",
                self.texture_load_instructions
            )?;
        }
        if self.static_flow_control_count > 0 {
            writeln!(
                f,
                "//   {} static flow control(s)",
                self.static_flow_control_count
            )?;
        }
        if self.dynamic_flow_control_count > 0 {
            writeln!(
                f,
                "//   {} dynamic flow control(s)",
                self.dynamic_flow_control_count
            )?;
        }
        if self.cut_instruction_count > 0 {
            writeln!(f, "//   {} cut instruction(s)", self.cut_instruction_count)?;
        }
        if self.emit_instruction_count > 0 {
            writeln!(
                f,
                "//   {} emit instruction(s)",
                self.emit_instruction_count
            )?;
        }
        if self.is_sample_frequency {
            writeln!(f, "//   sample-frequency execution")?;
        }
        Ok(())
    }
}
