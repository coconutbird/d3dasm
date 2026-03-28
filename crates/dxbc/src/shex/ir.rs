//! Structured intermediate representation for SM4/SM5 shader bytecode.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use super::opcodes::Opcode;

/// A decoded SM4/SM5 shader program.
#[derive(Debug, Clone)]
pub struct Program {
    /// Shader stage name (`"vs"`, `"ps"`, `"gs"`, `"hs"`, `"ds"`, or `"cs"`).
    pub shader_type: &'static str,
    /// Shader model major version (4 or 5).
    pub major_version: u32,
    /// Shader model minor version.
    pub minor_version: u32,
    /// Decoded instruction stream.
    pub instructions: Vec<Instruction>,
    /// Warnings encountered during decoding (malformed tokens, truncated data, etc.).
    pub warnings: Vec<String>,
    /// Original FourCC (`SHEX` or `SHDR`) for round-trip serialization.
    pub fourcc: [u8; 4],
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// A single decoded shader instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// The instruction opcode.
    pub opcode: Opcode,
    /// Whether the `_sat` (saturate) modifier is present.
    pub saturate: bool,
    /// `resinfo` return type: 0 = float, 1 = rcpFloat, 2 = uint.
    pub resinfo_return_type: Option<u32>,
    /// Texture sample/ld offsets from extended opcode token (u, v, w as signed 4-bit).
    pub tex_offsets: Option<[i8; 3]>,
    /// SM5.0 extended opcode type 2: resource dimension + stride for structured buffers.
    ///
    /// Encoded as the raw extended token dword (bits \[6:10\] = dimension, \[11:31\] = stride).
    pub resource_dim: Option<u32>,
    /// SM5.0 extended opcode type 3: per-component resource return types.
    ///
    /// Encoded as the raw extended token dword (bits \[6:9\], \[10:13\], \[14:17\], \[18:21\]).
    pub resource_return_type: Option<u32>,
    /// Raw opcode-specific bits from token0 (bits 11-23), preserved for round-trip
    /// fidelity. This captures fields like sample count (dcl\_resource with MS types),
    /// test condition (`if_nz` vs `if_z`), and any other instruction-embedded data
    /// that isn't separately decoded into IR fields.
    pub token0_opdata: u32,
    /// Instruction-specific payload (operands, declaration data, etc.).
    pub kind: InstructionKind,
}

/// Instruction-specific payload, split by category (declarations vs. generic ALU/flow).
#[derive(Debug, Clone, PartialEq)]
pub enum InstructionKind {
    /// A non-declaration instruction (ALU, flow-control, memory, etc.).
    Generic {
        /// Instruction operands (destination first, then sources).
        operands: Vec<Operand>,
    },
    /// `dcl_globalFlags` — shader-wide capability flags.
    DclGlobalFlags {
        /// Active flag strings (e.g. `"refactoringAllowed"`, `"enableDoublePrecisionFloatOps"`).
        flags: Vec<&'static str>,
    },
    /// `dcl_input` / `dcl_input_ps` — input register declaration.
    DclInput {
        /// PS interpolation mode (e.g. `"linear"`, `"constant"`), if applicable.
        interpolation: Option<&'static str>,
        /// System value semantic (e.g. `"position"`, `"undefined"`), if present.
        system_value: Option<&'static str>,
        /// Input register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_output` / `dcl_output_sgv` — output register declaration.
    DclOutput {
        /// System value semantic, if present.
        system_value: Option<&'static str>,
        /// Output register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_resource` — SRV resource declaration (Texture1D/2D/3D/Cube/etc.).
    DclResource {
        /// Resource dimension name (e.g. `"texture2d"`, `"buffer"`).
        dimension: &'static str,
        /// Per-component return types (e.g. `[Float, Float, Float, Float]`).
        return_type: [ReturnType; 4],
        /// Resource register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_sampler` — sampler state declaration.
    DclSampler {
        /// Sampler mode (`"default"`, `"comparison"`, `"mono"`).
        mode: &'static str,
        /// Sampler register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_constantbuffer` — constant-buffer binding.
    DclConstantBuffer {
        /// Access pattern (`"immediateIndexed"` or `"dynamicIndexed"`).
        access: &'static str,
        /// Constant-buffer register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_temps` — number of temporary registers.
    DclTemps {
        /// Number of temporary registers to allocate.
        count: u32,
    },
    /// `dcl_indexableTemp` — an indexable temporary register array.
    DclIndexableTemp {
        /// Register array index.
        reg: u32,
        /// Number of elements in the array.
        size: u32,
        /// Number of components per element.
        components: u32,
    },
    /// `dcl_inputPrimitive` — geometry shader input primitive topology.
    DclGsInputPrimitive {
        /// Primitive name (e.g. `"triangle"`, `"line"`, `"point"`).
        primitive: &'static str,
    },
    /// `dcl_outputTopology` — geometry shader output topology.
    DclGsOutputTopology {
        /// Topology name (e.g. `"trianglestrip"`, `"linestrip"`, `"pointlist"`).
        topology: &'static str,
    },
    /// `dcl_maxOutputVertexCount` — GS max output vertex count.
    DclMaxOutputVertexCount {
        /// Maximum vertices emitted per invocation.
        count: u32,
    },
    /// `dcl_gs_instance_count` — number of GS instances.
    DclGsInstanceCount {
        /// Number of geometry shader instances.
        count: u32,
    },
    /// `dcl_output_control_point_count` — HS output control point count.
    DclOutputControlPointCount {
        /// Number of output control points.
        count: u32,
    },
    /// `dcl_input_control_point_count` — HS input control point count.
    DclInputControlPointCount {
        /// Number of input control points.
        count: u32,
    },
    /// `dcl_tess_domain` — tessellator domain (tri, quad, isoline).
    DclTessDomain {
        /// Domain name string.
        domain: &'static str,
    },
    /// `dcl_tess_partitioning` — tessellator partitioning mode.
    DclTessPartitioning {
        /// Partitioning mode string.
        partitioning: &'static str,
    },
    /// `dcl_tess_output_primitive` — tessellator output primitive type.
    DclTessOutputPrimitive {
        /// Primitive type string.
        primitive: &'static str,
    },
    /// `dcl_hs_max_tessfactor` — maximum tessellation factor.
    DclHsMaxTessFactor {
        /// Maximum tessellation factor value.
        value: f32,
    },
    /// `dcl_hs_fork_phase_instance_count` — HS fork/join phase instance count.
    DclHsForkPhaseInstanceCount {
        /// Number of phase instances.
        count: u32,
    },
    /// `dcl_thread_group` — compute shader thread group dimensions.
    DclThreadGroup {
        /// Thread group X dimension.
        x: u32,
        /// Thread group Y dimension.
        y: u32,
        /// Thread group Z dimension.
        z: u32,
    },
    /// `dcl_uav_typed` — typed UAV declaration.
    DclUavTyped {
        /// UAV resource dimension.
        dimension: &'static str,
        /// Per-component return types.
        return_type: [ReturnType; 4],
        /// UAV register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_uav_raw` — raw (byte-address) UAV declaration.
    DclUavRaw {
        /// UAV flags (globally coherent, etc.).
        flags: u32,
        /// UAV register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_uav_structured` — structured UAV declaration.
    DclUavStructured {
        /// UAV flags (globally coherent, etc.).
        flags: u32,
        /// Structure byte stride.
        stride: u32,
        /// UAV register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_resource_raw` — raw (byte-address) SRV declaration.
    DclResourceRaw {
        /// SRV register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_resource_structured` — structured SRV declaration.
    DclResourceStructured {
        /// Structure byte stride.
        stride: u32,
        /// SRV register operands.
        operands: Vec<Operand>,
    },
    /// `dcl_function_body` — declares a function body slot.
    DclFunctionBody {
        /// Function body index.
        index: u32,
    },
    /// `dcl_function_table` — declares a function table for interface dispatch.
    DclFunctionTable {
        /// Table index.
        table_index: u32,
        /// Function body indices in this table.
        body_indices: Vec<u32>,
    },
    /// `dcl_interface` — declares a class interface binding.
    DclInterface {
        /// Interface index.
        interface_index: u32,
        /// Number of call sites using this interface.
        num_call_sites: u32,
        /// Function table indices for each type implementing the interface.
        table_indices: Vec<u32>,
    },
    /// `dcl_index_range` — declares an indexable register range.
    DclIndexRange {
        /// Register operands defining the range start.
        operands: Vec<Operand>,
        /// Number of registers in the range.
        count: u32,
    },
    /// Hull-shader phase marker (`hs_control_point_phase`, `hs_fork_phase`, etc.).
    HsPhase,
    /// `customdata` — embedded constant buffer or opaque blob.
    CustomData {
        /// Custom data sub-type (ICB, comment, opaque, etc.).
        subtype: CustomDataType,
        /// ICB values as four-component float vectors.
        values: Vec<[f32; 4]>,
        /// Total dword count of the custom data block (including header).
        raw_dword_count: usize,
    },
}

impl Instruction {
    /// Returns the operands of this instruction, if any.
    ///
    /// Most instruction kinds carry operands; declarations that encode
    /// everything in the token itself (e.g. `dcl_temps`) return an empty
    /// slice.
    pub fn operands(&self) -> &[Operand] {
        self.kind.operands()
    }
}

impl InstructionKind {
    /// Returns the operands embedded in this variant, or an empty slice
    /// if the variant has none.
    pub fn operands(&self) -> &[Operand] {
        match self {
            Self::Generic { operands }
            | Self::DclInput { operands, .. }
            | Self::DclOutput { operands, .. }
            | Self::DclResource { operands, .. }
            | Self::DclSampler { operands, .. }
            | Self::DclConstantBuffer { operands, .. }
            | Self::DclUavTyped { operands, .. }
            | Self::DclUavRaw { operands, .. }
            | Self::DclUavStructured { operands, .. }
            | Self::DclResourceRaw { operands }
            | Self::DclResourceStructured { operands, .. }
            | Self::DclIndexRange { operands, .. } => operands,
            _ => &[],
        }
    }
}

/// Sub-type tag for `customdata` blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomDataType {
    /// Comment block (opaque to the shader runtime).
    Comment,
    /// Debug information.
    DebugInfo,
    /// Opaque / unrecognised custom data.
    Opaque,
    /// Immediate constant buffer (ICB) data.
    ImmediateConstantBuffer,
    /// Unknown sub-type with the raw tag value.
    Other(u32),
}

/// A single source or destination operand.
#[derive(Debug, Clone, PartialEq)]
pub struct Operand {
    /// Register file (temp, input, output, constant-buffer, etc.).
    pub reg_type: RegisterType,
    /// Component selection (mask, swizzle, or scalar select).
    pub components: ComponentSelect,
    /// Raw `num_components` field from the operand token (bits 0–1).
    ///
    /// This is preserved for round-trip fidelity: `0` = zero-component,
    /// `1` = 1-component (single scalar), `2` = N-component (4-channel).
    pub num_components: u32,
    /// Source modifier: negate.
    pub negate: bool,
    /// Source modifier: absolute value.
    pub abs: bool,
    /// Register index levels (immediate, relative, or both).
    pub indices: Vec<OperandIndex>,
    /// Inline immediate values (for `Immediate32` / `Immediate64` operands).
    pub immediate_values: Vec<u32>,
}

/// How components are selected on an operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentSelect {
    /// No component selection (scalar register or zero-component).
    None,
    /// Write mask (destination): bitmask of enabled `.xyzw` components.
    Mask(u8),
    /// Four-component swizzle (source): each element is a channel index 0–3.
    Swizzle([u8; 4]),
    /// Single component select (source): channel index 0–3.
    Scalar(u8),
}

/// SM4/SM5 operand register type (`D3D10_SB_OPERAND_TYPE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterType {
    /// General-purpose temporary register (`r#`).
    Temp,
    /// Shader input register (`v#`).
    Input,
    /// Shader output register (`o#`).
    Output,
    /// Indexable temporary register array (`x#`).
    IndexableTemp,
    /// 32-bit inline immediate value (`l(...)`).
    Immediate32,
    /// 64-bit inline immediate value (`d(...)`).
    Immediate64,
    /// Sampler state (`s#`).
    Sampler,
    /// Shader resource view (`t#`).
    Resource,
    /// Constant buffer (`cb#`).
    ConstantBuffer,
    /// Immediate constant buffer (ICB, from `customdata`).
    ImmConstBuffer,
    /// Subroutine label.
    Label,
    /// System-generated primitive ID input.
    InputPrimitiveID,
    /// Pixel shader depth output (`oDepth`).
    OutputDepth,
    /// Null register (discard writes / unused reads).
    Null,
    /// Rasterizer register.
    Rasterizer,
    /// Sample coverage mask output (`oMask`).
    OutputCoverageMask,
    /// GS output stream selector.
    Stream,
    /// Function body reference (for interface dispatch).
    FunctionBody,
    /// Function table reference (for interface dispatch).
    FunctionTable,
    /// Class interface pointer.
    Interface,
    /// Function input parameter.
    FunctionInput,
    /// Function output parameter.
    FunctionOutput,
    /// HS output control-point ID (`vOutputControlPointID`).
    OutputControlPointID,
    /// HS fork-phase instance ID.
    ForkInstanceID,
    /// HS join-phase instance ID.
    JoinInstanceID,
    /// HS/DS input control point (`vicp`).
    InputControlPoint,
    /// HS output control point (`vocp`).
    OutputControlPoint,
    /// Patch constant data register (`vpc`).
    PatchConstant,
    /// DS domain location input (`vDomain`).
    DomainLocation,
    /// Class `this` pointer.
    ThisPointer,
    /// Unordered access view (`u#`).
    Uav,
    /// Thread-group shared memory (`g#`).
    ThreadGroupSharedMemory,
    /// Compute shader thread ID (`vThreadID`).
    ThreadID,
    /// Compute shader thread-group ID (`vThreadGroupID`).
    ThreadGroupID,
    /// Thread ID within the group (`vThreadIDInGroup`).
    ThreadIDInGroup,
    /// Sample coverage input (`vCoverage`).
    Coverage,
    /// Flattened thread ID within the group.
    ThreadIDInGroupFlattened,
    /// GS instance ID (`vGSInstanceID`).
    GsInstanceID,
    /// Depth output with greater-or-equal constraint (`oDepthGE`).
    OutputDepthGE,
    /// Depth output with less-or-equal constraint (`oDepthLE`).
    OutputDepthLE,
    /// GPU cycle counter.
    CycleCounter,
    /// Unrecognised register type (raw value preserved).
    Unknown(u32),
}

/// An operand index level (register indices can be multi-dimensional).
#[derive(Debug, Clone, PartialEq)]
pub enum OperandIndex {
    /// 32-bit immediate index.
    Imm32(u32),
    /// 64-bit immediate index.
    Imm64(u64),
    /// Fully relative index (address computed by another operand).
    Relative(Box<Operand>),
    /// Base immediate offset plus a relative operand.
    RelativePlusImm(u32, Box<Operand>),
}

/// Resource return type for SRV / UAV declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnType {
    /// Unsigned normalised (`[0, 1]`).
    Unorm,
    /// Signed normalised (`[-1, 1]`).
    Snorm,
    /// Signed 32-bit integer.
    Sint,
    /// Unsigned 32-bit integer.
    Uint,
    /// 32-bit IEEE float.
    Float,
    /// Mixed return types across components.
    Mixed,
    /// 64-bit double-precision float.
    Double,
    /// Continuation of a previous component (unused in practice).
    Continued,
    /// Component is unused.
    Unused,
    /// Unrecognised return type (raw value preserved).
    Unknown(u32),
}

impl RegisterType {
    /// Converts a raw `D3D10_SB_OPERAND_TYPE` value to the corresponding variant.
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Temp,
            1 => Self::Input,
            2 => Self::Output,
            3 => Self::IndexableTemp,
            4 => Self::Immediate32,
            5 => Self::Immediate64,
            6 => Self::Sampler,
            7 => Self::Resource,
            8 => Self::ConstantBuffer,
            9 => Self::ImmConstBuffer,
            10 => Self::Label,
            11 => Self::InputPrimitiveID,
            12 => Self::OutputDepth,
            13 => Self::Null,
            14 => Self::Rasterizer,
            15 => Self::OutputCoverageMask,
            16 => Self::Stream,
            17 => Self::FunctionBody,
            18 => Self::FunctionTable,
            19 => Self::Interface,
            20 => Self::FunctionInput,
            21 => Self::FunctionOutput,
            22 => Self::OutputControlPointID,
            23 => Self::ForkInstanceID,
            24 => Self::JoinInstanceID,
            25 => Self::InputControlPoint,
            26 => Self::OutputControlPoint,
            27 => Self::PatchConstant,
            28 => Self::DomainLocation,
            29 => Self::ThisPointer,
            30 => Self::Uav,
            31 => Self::ThreadGroupSharedMemory,
            32 => Self::ThreadID,
            33 => Self::ThreadGroupID,
            34 => Self::ThreadIDInGroup,
            35 => Self::Coverage,
            36 => Self::ThreadIDInGroupFlattened,
            37 => Self::GsInstanceID,
            38 => Self::OutputDepthGE,
            39 => Self::OutputDepthLE,
            40 => Self::CycleCounter,
            v => Self::Unknown(v),
        }
    }

    /// Converts back to the raw `D3D10_SB_OPERAND_TYPE` value.
    pub fn to_u32(&self) -> u32 {
        match self {
            Self::Temp => 0,
            Self::Input => 1,
            Self::Output => 2,
            Self::IndexableTemp => 3,
            Self::Immediate32 => 4,
            Self::Immediate64 => 5,
            Self::Sampler => 6,
            Self::Resource => 7,
            Self::ConstantBuffer => 8,
            Self::ImmConstBuffer => 9,
            Self::Label => 10,
            Self::InputPrimitiveID => 11,
            Self::OutputDepth => 12,
            Self::Null => 13,
            Self::Rasterizer => 14,
            Self::OutputCoverageMask => 15,
            Self::Stream => 16,
            Self::FunctionBody => 17,
            Self::FunctionTable => 18,
            Self::Interface => 19,
            Self::FunctionInput => 20,
            Self::FunctionOutput => 21,
            Self::OutputControlPointID => 22,
            Self::ForkInstanceID => 23,
            Self::JoinInstanceID => 24,
            Self::InputControlPoint => 25,
            Self::OutputControlPoint => 26,
            Self::PatchConstant => 27,
            Self::DomainLocation => 28,
            Self::ThisPointer => 29,
            Self::Uav => 30,
            Self::ThreadGroupSharedMemory => 31,
            Self::ThreadID => 32,
            Self::ThreadGroupID => 33,
            Self::ThreadIDInGroup => 34,
            Self::Coverage => 35,
            Self::ThreadIDInGroupFlattened => 36,
            Self::GsInstanceID => 37,
            Self::OutputDepthGE => 38,
            Self::OutputDepthLE => 39,
            Self::CycleCounter => 40,
            Self::Unknown(v) => *v,
        }
    }

    /// Returns the disassembly prefix string for this register type (e.g. `"r"`, `"v"`, `"cb"`).
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Temp => "r",
            Self::Input => "v",
            Self::Output => "o",
            Self::IndexableTemp => "x",
            Self::Immediate32 => "l",
            Self::Immediate64 => "d",
            Self::Sampler => "s",
            Self::Resource => "t",
            Self::ConstantBuffer => "cb",
            Self::ImmConstBuffer => "icb",
            Self::Label => "label",
            Self::InputPrimitiveID => "vPrim",
            Self::OutputDepth => "oDepth",
            Self::Null => "null",
            Self::Rasterizer => "rasterizer",
            Self::OutputCoverageMask => "oMask",
            Self::Stream => "stream",
            Self::FunctionBody => "function_body",
            Self::FunctionTable => "function_table",
            Self::Interface => "interface",
            Self::FunctionInput => "function_input",
            Self::FunctionOutput => "function_output",
            Self::OutputControlPointID => "vOutputControlPointID",
            Self::ForkInstanceID => "vForkInstanceID",
            Self::JoinInstanceID => "vJoinInstanceID",
            Self::InputControlPoint => "vicp",
            Self::OutputControlPoint => "vocp",
            Self::PatchConstant => "vpc",
            Self::DomainLocation => "vDomain",
            Self::ThisPointer => "thisPointer",
            Self::Uav => "u",
            Self::ThreadGroupSharedMemory => "g",
            Self::ThreadID => "vThreadID",
            Self::ThreadGroupID => "vThreadGroupID",
            Self::ThreadIDInGroup => "vThreadIDInGroup",
            Self::Coverage => "vCoverage",
            Self::ThreadIDInGroupFlattened => "vThreadIDInGroupFlattened",
            Self::GsInstanceID => "vGSInstanceID",
            Self::OutputDepthGE => "oDepthGE",
            Self::OutputDepthLE => "oDepthLE",
            Self::CycleCounter => "vCycleCounter",
            Self::Unknown(_) => "?reg",
        }
    }
}

impl ReturnType {
    /// Converts a raw `D3D10_SB_RESOURCE_RETURN_TYPE` value to the corresponding variant.
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Unorm,
            2 => Self::Snorm,
            3 => Self::Sint,
            4 => Self::Uint,
            5 => Self::Float,
            6 => Self::Mixed,
            7 => Self::Double,
            8 => Self::Continued,
            9 => Self::Unused,
            v => Self::Unknown(v),
        }
    }

    /// Converts back to the raw `D3D10_SB_RESOURCE_RETURN_TYPE` value.
    pub fn to_u32(&self) -> u32 {
        match self {
            Self::Unorm => 1,
            Self::Snorm => 2,
            Self::Sint => 3,
            Self::Uint => 4,
            Self::Float => 5,
            Self::Mixed => 6,
            Self::Double => 7,
            Self::Continued => 8,
            Self::Unused => 9,
            Self::Unknown(v) => *v,
        }
    }

    /// Returns the lowercase name used in disassembly output.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unorm => "unorm",
            Self::Snorm => "snorm",
            Self::Sint => "sint",
            Self::Uint => "uint",
            Self::Float => "float",
            Self::Mixed => "mixed",
            Self::Double => "double",
            Self::Continued => "continued",
            Self::Unused => "unused",
            Self::Unknown(_) => "?",
        }
    }
}

impl CustomDataType {
    /// Converts a raw `D3D10_SB_CUSTOMDATA_CLASS` value to the corresponding variant.
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Comment,
            1 => Self::DebugInfo,
            2 => Self::Opaque,
            3 => Self::ImmediateConstantBuffer,
            v => Self::Other(v),
        }
    }
}

/// Decode a system value name from its encoded D3D10_SB_NAME / D3D11_SB_NAME value.
pub fn system_value_name(val: u32) -> &'static str {
    match val {
        0 => "undefined",
        1 => "position",
        2 => "clip_distance",
        3 => "cull_distance",
        4 => "render_target_array_index",
        5 => "viewport_array_index",
        6 => "vertex_id",
        7 => "primitive_id",
        8 => "instance_id",
        9 => "is_front_face",
        10 => "sample_index",
        // Quad tessellation factors (4 edges + 2 inside)
        11 => "finalQuadUeq0EdgeTessFactor",
        12 => "finalQuadVeq0EdgeTessFactor",
        13 => "finalQuadUeq1EdgeTessFactor",
        14 => "finalQuadVeq1EdgeTessFactor",
        15 => "finalQuadUInsideTessFactor",
        16 => "finalQuadVInsideTessFactor",
        // Triangle tessellation factors (3 edges + 1 inside)
        17 => "finalTriUeq0EdgeTessFactor",
        18 => "finalTriVeq0EdgeTessFactor",
        19 => "finalTriWeq0EdgeTessFactor",
        20 => "finalTriInsideTessFactor",
        // Line tessellation factors
        21 => "finalLineDetailTessFactor",
        22 => "finalLineDensityTessFactor",
        23 => "target",
        24 => "depth",
        25 => "coverage",
        26 => "depth_greater_equal",
        27 => "depth_less_equal",
        64 => "stencil_ref",
        65 => "inner_coverage",
        _ => "unknown_sv",
    }
}
