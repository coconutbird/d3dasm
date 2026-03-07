//! Parser for the STAT (statistics) chunk in DXBC shader bytecode.

use core::fmt;

use crate::util::read_u32;

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
    // Minimum STAT chunk is 29 u32 fields = 116 bytes
    if data.len() < 116 {
        return None;
    }

    Some(ShaderStats {
        instruction_count: read_u32(data, 0),
        temp_register_count: read_u32(data, 4),
        define_count: read_u32(data, 8),
        declaration_count: read_u32(data, 12),
        float_instruction_count: read_u32(data, 16),
        int_instruction_count: read_u32(data, 20),
        uint_instruction_count: read_u32(data, 24),
        static_flow_control_count: read_u32(data, 28),
        dynamic_flow_control_count: read_u32(data, 32),
        macro_instruction_count: read_u32(data, 36),
        temp_array_count: read_u32(data, 40),
        array_instruction_count: read_u32(data, 44),
        cut_instruction_count: read_u32(data, 48),
        emit_instruction_count: read_u32(data, 52),
        texture_normal_instructions: read_u32(data, 56),
        texture_load_instructions: read_u32(data, 60),
        texture_comp_instructions: read_u32(data, 64),
        texture_bias_instructions: read_u32(data, 68),
        texture_gradient_instructions: read_u32(data, 72),
        mov_instruction_count: read_u32(data, 76),
        movc_instruction_count: read_u32(data, 80),
        conversion_instruction_count: read_u32(data, 84),
        // Offsets 88 and 92 are unknown fields
        gs_input_primitive: read_u32(data, 96),
        gs_output_topology: read_u32(data, 100),
        gs_max_output_vertex_count: read_u32(data, 104),
        // Offsets 108 and 112 are unknown fields
        is_sample_frequency: data.len() >= 120 && read_u32(data, 116) != 0,
    })
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
