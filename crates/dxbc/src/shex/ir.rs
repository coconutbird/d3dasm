//! Structured intermediate representation for SM4/SM5 shader bytecode.

use super::opcodes::Opcode;

#[derive(Debug, Clone)]
pub struct Program {
    pub shader_type: &'static str,
    pub major_version: u32,
    pub minor_version: u32,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub opcode: Opcode,
    pub saturate: bool,
    /// resinfo return type: 0=float, 1=rcpFloat, 2=uint
    pub resinfo_return_type: Option<u32>,
    /// Texture sample/ld offsets from extended opcode token (u,v,w as signed 4-bit).
    pub tex_offsets: Option<[i8; 3]>,
    pub kind: InstructionKind,
}

#[derive(Debug, Clone)]
pub enum InstructionKind {
    Generic {
        operands: Vec<Operand>,
    },
    DclGlobalFlags {
        flags: Vec<&'static str>,
    },
    DclInput {
        interpolation: Option<&'static str>,
        system_value: Option<&'static str>,
        operands: Vec<Operand>,
    },
    DclOutput {
        system_value: Option<&'static str>,
        operands: Vec<Operand>,
    },
    DclResource {
        dimension: &'static str,
        return_type: [ReturnType; 4],
        operands: Vec<Operand>,
    },
    DclSampler {
        mode: &'static str,
        operands: Vec<Operand>,
    },
    DclConstantBuffer {
        access: &'static str,
        operands: Vec<Operand>,
    },
    DclTemps {
        count: u32,
    },
    DclIndexableTemp {
        reg: u32,
        size: u32,
        components: u32,
    },
    DclGsInputPrimitive {
        primitive: &'static str,
    },
    DclGsOutputTopology {
        topology: &'static str,
    },
    DclMaxOutputVertexCount {
        count: u32,
    },
    DclGsInstanceCount {
        count: u32,
    },
    DclOutputControlPointCount {
        count: u32,
    },
    DclInputControlPointCount {
        count: u32,
    },
    DclTessDomain {
        domain: &'static str,
    },
    DclTessPartitioning {
        partitioning: &'static str,
    },
    DclTessOutputPrimitive {
        primitive: &'static str,
    },
    DclHsMaxTessFactor {
        value: f32,
    },
    DclHsForkPhaseInstanceCount {
        count: u32,
    },
    DclThreadGroup {
        x: u32,
        y: u32,
        z: u32,
    },
    DclUavTyped {
        dimension: &'static str,
        return_type: [ReturnType; 4],
        operands: Vec<Operand>,
    },
    DclUavRaw {
        flags: u32,
        operands: Vec<Operand>,
    },
    DclUavStructured {
        flags: u32,
        stride: u32,
        operands: Vec<Operand>,
    },
    DclResourceRaw {
        operands: Vec<Operand>,
    },
    DclResourceStructured {
        stride: u32,
        operands: Vec<Operand>,
    },
    // DclStream is handled as Generic (operand carries the stream index)
    DclIndexRange {
        operands: Vec<Operand>,
        count: u32,
    },
    HsPhase,
    CustomData {
        subtype: CustomDataType,
        values: Vec<[f32; 4]>,
        raw_dword_count: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomDataType {
    Comment,
    DebugInfo,
    Opaque,
    ImmediateConstantBuffer,
    Other(u32),
}

#[derive(Debug, Clone)]
pub struct Operand {
    pub reg_type: RegisterType,
    pub components: ComponentSelect,
    pub negate: bool,
    pub abs: bool,
    pub indices: Vec<OperandIndex>,
    pub immediate_values: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentSelect {
    None,
    Mask(u8),
    Swizzle([u8; 4]),
    Scalar(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterType {
    Temp,
    Input,
    Output,
    IndexableTemp,
    Immediate32,
    Immediate64,
    Sampler,
    Resource,
    ConstantBuffer,
    ImmConstBuffer,
    Label,
    InputPrimitiveID,
    OutputDepth,
    Null,
    Rasterizer,
    OutputCoverageMask,
    Stream,
    FunctionBody,
    FunctionTable,
    Interface,
    FunctionInput,
    FunctionOutput,
    OutputControlPointID,
    ForkInstanceID,
    JoinInstanceID,
    InputControlPoint,
    OutputControlPoint,
    PatchConstant,
    DomainLocation,
    ThisPointer,
    Uav,
    ThreadGroupSharedMemory,
    ThreadID,
    ThreadGroupID,
    ThreadIDInGroup,
    Coverage,
    ThreadIDInGroupFlattened,
    GsInstanceID,
    OutputDepthGE,
    OutputDepthLE,
    CycleCounter,
    Unknown(u32),
}

#[derive(Debug, Clone)]
pub enum OperandIndex {
    Imm32(u32),
    Imm64(u64),
    Relative(Box<Operand>),
    RelativePlusImm(u32, Box<Operand>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnType {
    Unorm,
    Snorm,
    Sint,
    Uint,
    Float,
    Mixed,
    Double,
    Continued,
    Unused,
    Unknown(u32),
}

impl RegisterType {
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

/// Decode a system value name from its encoded value.
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
        11 => "finalQuadEdgeTessfactor",
        12 => "finalQuadInsideTessfactor",
        13 => "finalTriEdgeTessfactor",
        14 => "finalTriInsideTessfactor",
        15 => "finalLineDetailTessfactor",
        16 => "finalLineDensityTessfactor",
        23 => "target",
        24 => "depth",
        25 => "coverage",
        26 => "depth_greater_equal",
        27 => "depth_less_equal",
        _ => "unknown_sv",
    }
}
