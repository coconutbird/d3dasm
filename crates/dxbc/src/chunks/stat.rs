//! Legacy STAT shader statistics.
//!
//! This is the fixed FXC statistics record used by SM4/SM5 containers. A STAT
//! part beginning with `DxilProgramHeader` is SM6 reflection bitcode and is
//! dispatched separately by [`super::parse_chunk`].

use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// The two observed and structurally complete FXC STAT layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShaderStatsLayout {
    /// 29 dwords (116 bytes), ending at `IsSampleFrequency`.
    Legacy116,
    /// 37 dwords (148 bytes), with the D3D11 tessellation/compute counters.
    #[default]
    Extended148,
}

impl ShaderStatsLayout {
    /// Payload size in bytes.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        match self {
            Self::Legacy116 => 116,
            Self::Extended148 => 148,
        }
    }

    const fn dword_len(self) -> usize {
        self.byte_len() / 4
    }
}

/// Counters stored in an FXC STAT part, in their serialized order.
#[derive(Debug, Clone, Default)]
pub struct ShaderStats {
    /// Wire layout present in the part.
    pub layout: ShaderStatsLayout,
    /// Total number of instructions.
    pub instruction_count: u32,
    /// Number of temporary registers.
    pub temp_register_count: u32,
    /// Number of constant definitions.
    pub define_count: u32,
    /// Number of declaration instructions.
    pub declaration_count: u32,
    /// Number of floating-point arithmetic instructions.
    pub float_instruction_count: u32,
    /// Number of signed-integer arithmetic instructions.
    pub int_instruction_count: u32,
    /// Number of unsigned-integer arithmetic instructions.
    pub uint_instruction_count: u32,
    /// Number of static flow-control instructions.
    pub static_flow_control_count: u32,
    /// Number of dynamic flow-control instructions.
    pub dynamic_flow_control_count: u32,
    /// Number of macro instructions.
    pub macro_instruction_count: u32,
    /// Number of indexable temporary arrays.
    pub temp_array_count: u32,
    /// Number of array-indexed instructions.
    pub array_instruction_count: u32,
    /// Number of geometry-shader cut instructions.
    pub cut_instruction_count: u32,
    /// Number of geometry-shader emit instructions.
    pub emit_instruction_count: u32,
    /// Number of uncategorized texture instructions.
    pub texture_normal_instructions: u32,
    /// Number of texture load instructions.
    pub texture_load_instructions: u32,
    /// Number of texture comparison instructions.
    pub texture_comp_instructions: u32,
    /// Number of texture bias instructions.
    pub texture_bias_instructions: u32,
    /// Number of texture gradient instructions.
    pub texture_gradient_instructions: u32,
    /// Number of `mov` instructions.
    pub mov_instruction_count: u32,
    /// Number of conditional `movc` instructions.
    pub movc_instruction_count: u32,
    /// Number of conversion instructions.
    pub conversion_instruction_count: u32,
    /// Number of bitwise instructions.
    pub bitwise_instruction_count: u32,
    /// Geometry/hull-shader input primitive (`D3D_PRIMITIVE`).
    pub input_primitive: u32,
    /// Geometry-shader output topology (`D3D_PRIMITIVE_TOPOLOGY`).
    pub gs_output_topology: u32,
    /// Maximum vertices emitted by a geometry-shader invocation.
    pub gs_max_output_vertex_count: u32,
    /// Number of texture gather instructions.
    pub texture_gather_instructions: u32,
    /// Number of texture LOD-calculation instructions.
    pub texture_calculate_lod_instructions: u32,
    /// Whether a pixel shader executes at sample frequency.
    pub is_sample_frequency: bool,
    /// Geometry-shader instance count (extended layout only).
    pub gs_instance_count: u32,
    /// Hull/domain control-point count (extended layout only).
    pub control_points: u32,
    /// Tessellator output primitive (extended layout only).
    pub hs_output_primitive: u32,
    /// Tessellator partitioning mode (extended layout only).
    pub hs_partitioning: u32,
    /// Tessellator domain (extended layout only).
    pub tessellator_domain: u32,
    /// Number of barrier instructions (extended layout only).
    pub barrier_instructions: u32,
    /// Number of interlocked instructions (extended layout only).
    pub interlocked_instructions: u32,
    /// Number of texture store instructions (extended layout only).
    pub texture_store_instructions: u32,
}

impl ShaderStats {
    /// Exact serialized payload size.
    #[must_use]
    pub const fn raw_size(&self) -> usize {
        self.layout.byte_len()
    }
}

/// Parse a complete FXC STAT record.
#[must_use]
pub fn parse_stat(data: &[u8]) -> Option<ShaderStats> {
    let layout = match data.len() {
        116 => ShaderStatsLayout::Legacy116,
        148 => ShaderStatsLayout::Extended148,
        _ => return None,
    };
    let mut dwords = [0u32; 37];
    for (slot, bytes) in dwords.iter_mut().zip(data.chunks_exact(4)) {
        *slot = u32::from_le_bytes(bytes.try_into().ok()?);
    }
    if dwords[28] > 1 {
        return None;
    }

    Some(ShaderStats {
        layout,
        instruction_count: dwords[0],
        temp_register_count: dwords[1],
        define_count: dwords[2],
        declaration_count: dwords[3],
        float_instruction_count: dwords[4],
        int_instruction_count: dwords[5],
        uint_instruction_count: dwords[6],
        static_flow_control_count: dwords[7],
        dynamic_flow_control_count: dwords[8],
        macro_instruction_count: dwords[9],
        temp_array_count: dwords[10],
        array_instruction_count: dwords[11],
        cut_instruction_count: dwords[12],
        emit_instruction_count: dwords[13],
        texture_normal_instructions: dwords[14],
        texture_load_instructions: dwords[15],
        texture_comp_instructions: dwords[16],
        texture_bias_instructions: dwords[17],
        texture_gradient_instructions: dwords[18],
        mov_instruction_count: dwords[19],
        movc_instruction_count: dwords[20],
        conversion_instruction_count: dwords[21],
        bitwise_instruction_count: dwords[22],
        input_primitive: dwords[23],
        gs_output_topology: dwords[24],
        gs_max_output_vertex_count: dwords[25],
        texture_gather_instructions: dwords[26],
        texture_calculate_lod_instructions: dwords[27],
        is_sample_frequency: dwords[28] != 0,
        gs_instance_count: dwords[29],
        control_points: dwords[30],
        hs_output_primitive: dwords[31],
        hs_partitioning: dwords[32],
        tessellator_domain: dwords[33],
        barrier_instructions: dwords[34],
        interlocked_instructions: dwords[35],
        texture_store_instructions: dwords[36],
    })
}

impl ChunkParser<'_> for ShaderStats {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_stat(data)
    }
}

impl ChunkWriter for ShaderStats {
    fn fourcc(&self) -> [u8; 4] {
        *b"STAT"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let dwords = [
            self.instruction_count,
            self.temp_register_count,
            self.define_count,
            self.declaration_count,
            self.float_instruction_count,
            self.int_instruction_count,
            self.uint_instruction_count,
            self.static_flow_control_count,
            self.dynamic_flow_control_count,
            self.macro_instruction_count,
            self.temp_array_count,
            self.array_instruction_count,
            self.cut_instruction_count,
            self.emit_instruction_count,
            self.texture_normal_instructions,
            self.texture_load_instructions,
            self.texture_comp_instructions,
            self.texture_bias_instructions,
            self.texture_gradient_instructions,
            self.mov_instruction_count,
            self.movc_instruction_count,
            self.conversion_instruction_count,
            self.bitwise_instruction_count,
            self.input_primitive,
            self.gs_output_topology,
            self.gs_max_output_vertex_count,
            self.texture_gather_instructions,
            self.texture_calculate_lod_instructions,
            u32::from(self.is_sample_frequency),
            self.gs_instance_count,
            self.control_points,
            self.hs_output_primitive,
            self.hs_partitioning,
            self.tessellator_domain,
            self.barrier_instructions,
            self.interlocked_instructions,
            self.texture_store_instructions,
        ];

        let mut out = alloc::vec::Vec::with_capacity(self.layout.byte_len());
        for value in dwords.iter().take(self.layout.dword_len()) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
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
                "//   {} static flow control instruction(s)",
                self.static_flow_control_count
            )?;
        }
        if self.dynamic_flow_control_count > 0 {
            writeln!(
                f,
                "//   {} dynamic flow control instruction(s)",
                self.dynamic_flow_control_count
            )?;
        }
        if self.is_sample_frequency {
            writeln!(f, "//   sample-frequency execution")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_extended_fields_have_their_wire_index() {
        let mut bytes = alloc::vec::Vec::new();
        for value in 1u32..=37 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes[28 * 4..29 * 4].copy_from_slice(&1u32.to_le_bytes());
        let stats = parse_stat(&bytes).expect("valid extended STAT");
        assert_eq!(stats.float_instruction_count, 5);
        assert_eq!(stats.texture_normal_instructions, 15);
        assert_eq!(stats.bitwise_instruction_count, 23);
        assert_eq!(stats.input_primitive, 24);
        assert_eq!(stats.texture_gather_instructions, 27);
        assert_eq!(stats.texture_calculate_lod_instructions, 28);
        assert!(stats.is_sample_frequency);
        assert_eq!(stats.gs_instance_count, 30);
        assert_eq!(stats.texture_store_instructions, 37);
        assert_eq!(stats.write_payload(), bytes);
    }

    #[test]
    fn rejects_partial_or_dxil_stat_records() {
        assert!(parse_stat(&[0; 120]).is_none());
        assert!(parse_stat(&[0; 24]).is_none());
    }
}
