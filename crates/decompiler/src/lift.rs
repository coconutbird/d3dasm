//! SM4/SM5 structured instruction stream → shader-dialect RTL.
//!
//! The walk mirrors the bytecode's structured markers (`if`/`else`/
//! `endif`, `loop`/`endloop`, conditional transfers) into explicit blocks
//! and edges, and expresses each computational instruction as one typed
//! RTL statement: parallel transfers over raw registers whose expression
//! trees carry immediates, constant-buffer reads, and source modifiers
//! inline. Source swizzles are mapped through the destination mask here,
//! so every read lists the exact storage lanes it consumes. Blocks are
//! created lazily when a statement lands, so no empty semantic block
//! survives into MLIL.
//!
//! Lane interpretations are assigned per opcode: float arithmetic reads
//! `F32`, integer forms `I32`/`U32`, comparisons produce `U32` masks,
//! and transports (`mov`, `movc` data) read `Bits` so cfglib's web
//! typing routes the surrounding interpretation through them.

use cfglib::ir::rtl::{Expr, FunctionBuilder, ScalarType, Statement};
use cfglib::{BlockId, StructuredEdge, StructuredSink, StructuredWalk, StructuredWalkIssue};
use dxbc::shex::{
    ComponentSelect, InstructionKind, Opcode, Operand, OperandIndex, Program, RegisterType,
};

use crate::dialect::{Edge, RegFile, Register, RtlOp, ShaderDialect, SystemInput};

/// One declared input register row.
#[derive(Debug, Clone)]
pub struct InputRow {
    /// The input register.
    pub register: Register,
    /// Declared component selection.
    pub components: ComponentSelect,
    /// System-value name, when declared.
    pub system_value: Option<&'static str>,
    /// Pixel-shader interpolation mode string, when declared.
    pub interpolation: Option<&'static str>,
}

/// A declaration retained for header rendering.
#[derive(Debug, Clone)]
pub struct Declarations {
    /// `dcl_sampler` register indices with their mode strings.
    pub samplers: Vec<(u32, &'static str)>,
    /// `dcl_resource` entries: register, dimension, return types.
    pub resources: Vec<(u32, &'static str, [dxbc::shex::ReturnType; 4])>,
    /// `dcl_constantbuffer` entries: register and vector count.
    pub constant_buffers: Vec<(u32, u32)>,
    /// Input register rows: register, components, system value,
    /// interpolation mode.
    pub inputs: Vec<InputRow>,
    /// Output register indices with any system-value name.
    pub outputs: Vec<(Register, ComponentSelect, Option<&'static str>)>,
    /// `dcl_indexableTemp` entries: register, length, component count.
    pub indexable_temps: Vec<(u32, u32, u32)>,
    /// Immediate constant buffer rows, when declared.
    pub immediate_constants: Vec<[f32; 4]>,
    /// `dcl_resource_structured` entries: register and byte stride.
    pub structured: Vec<(u32, u32)>,
    /// `dcl_uav_typed` entries: register, dimension, return types.
    pub uavs: Vec<(u32, &'static str, [dxbc::shex::ReturnType; 4])>,
    /// `dcl_thread_group` sizes for compute shaders.
    pub thread_group: Option<(u32, u32, u32)>,
    /// Geometry-shader stage layout, when the program is a GS.
    pub geometry: Option<GeometryLayout>,
    /// Tessellation stage layout, when the program is an HS or DS.
    pub tessellation: Option<TessellationLayout>,
}

/// The tessellation interface layout shared by hull and domain shaders.
#[derive(Debug, Clone, Default)]
pub struct TessellationLayout {
    /// Tessellator domain (`quad`, `tri`, `isoline`).
    pub domain: &'static str,
    /// Partitioning mode, when declared (HS).
    pub partitioning: Option<&'static str>,
    /// Tessellator output primitive, when declared (HS).
    pub output_primitive: Option<&'static str>,
    /// Input control points per patch.
    pub input_control_points: u32,
    /// Output control points per patch, when declared (HS).
    pub output_control_points: u32,
    /// Maximum tessellation factor, when declared (HS).
    pub max_factor: Option<f32>,
}

/// The geometry-shader interface layout.
#[derive(Debug, Clone)]
pub struct GeometryLayout {
    /// HLSL input-primitive keyword (`point`, `line`, `triangle`, …).
    pub primitive: &'static str,
    /// Vertices per input primitive.
    pub vertices: u32,
    /// Maximum vertices emitted per invocation.
    pub max_vertices: u32,
}

/// One lifted shader body plus its rendering metadata.
pub struct LiftedShader {
    /// The RTL function over raw registers (the control-point phase for
    /// hull shaders).
    pub function: cfglib::ir::rtl::Function<ShaderDialect>,
    /// Declarations for the HLSL header.
    pub declarations: Declarations,
    /// Hull-shader patch-constant phases, when the program is an HS.
    pub hull: Option<HullPhases>,
}

/// The patch-constant phases of one hull shader.
pub struct HullPhases {
    /// Fork and join phase instance counts and functions, in order.
    pub constants: Vec<(u32, cfglib::ir::rtl::Function<ShaderDialect>)>,
}

/// A shape the lift does not model yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl core::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unsupported shader shape: {}", self.0)
    }
}

impl std::error::Error for Unsupported {}

fn unsupported<T>(message: impl Into<String>) -> Result<T, Unsupported> {
    Err(Unsupported(message.into()))
}

impl From<StructuredWalkIssue> for Unsupported {
    fn from(issue: StructuredWalkIssue) -> Self {
        Self(issue.to_string())
    }
}

/// Block minting and edge wiring for the structured walk: labels count
/// up in materialization order, and the generic edge roles map onto the
/// dialect's edge vocabulary (the synthetic root keeps its entry edge).
struct Sink<'a> {
    builder: &'a mut FunctionBuilder<ShaderDialect>,
    blocks: &'a mut u32,
}

impl StructuredSink for Sink<'_> {
    type Block = BlockId;

    fn new_block(&mut self) -> BlockId {
        let count = *self.blocks;
        let block = self.builder.new_block(format!("b{count}"));
        *self.blocks += 1;
        block
    }

    fn connect(&mut self, source: BlockId, target: BlockId, edge: StructuredEdge) {
        let edge = match edge {
            StructuredEdge::True => Edge::True,
            StructuredEdge::False => Edge::False,
            StructuredEdge::Fallthrough if source == self.builder.entry() => Edge::Entry,
            StructuredEdge::Fallthrough => Edge::Fall,
        };
        self.builder
            .add_edge(source, target, edge)
            .expect("structured edges name live blocks");
    }
}

struct Walk {
    builder: FunctionBuilder<ShaderDialect>,
    declarations: Declarations,
    /// The open-block, pending-edge, and frame-stack state.
    structure: StructuredWalk<BlockId>,
    blocks: u32,
    ret_block: Option<BlockId>,
}

mod declarations;
mod walk;

use declarations::collect_declaration;

type LiftExpr = Expr<ShaderDialect>;

impl Walk {
    /// The frame-stack state alongside a sink over the builder.
    fn parts(&mut self) -> (&mut StructuredWalk<BlockId>, Sink<'_>) {
        (
            &mut self.structure,
            Sink {
                builder: &mut self.builder,
                blocks: &mut self.blocks,
            },
        )
    }

    /// The block statements land in, materializing it if needed.
    fn landing(&mut self) -> BlockId {
        let (walk, mut sink) = self.parts();
        walk.landing(&mut sink)
    }

    fn emit(&mut self, statement: Statement<ShaderDialect>, span: (u32, u32)) {
        let block = self.landing();
        self.builder
            .append(block, statement, Some(span))
            .expect("statement shapes are validated at construction");
    }
}

/// Applies the negate/abs source modifiers around one value.
fn modified(base: LiftExpr, operand: &Operand) -> LiftExpr {
    let mut value = base;
    if operand.abs {
        let shape = value.shape();
        value = Expr::Apply {
            operator: RtlOp::Abs,
            operands: vec![value],
            shape,
        };
    }
    if operand.negate {
        let shape = value.shape();
        value = Expr::Apply {
            operator: RtlOp::Neg,
            operands: vec![value],
            shape,
        };
    }
    value
}

/// The lanes a componentwise source consumes for the destination lanes.
fn map_lanes(components: &ComponentSelect, lanes: &[u8]) -> Result<Vec<u8>, Unsupported> {
    match components {
        ComponentSelect::Swizzle(swizzle) => Ok(lanes
            .iter()
            .map(|&lane| swizzle[usize::from(lane)])
            .collect()),
        ComponentSelect::Scalar(lane) => Ok(vec![*lane; lanes.len()]),
        ComponentSelect::OneComponent | ComponentSelect::ZeroComponent => Ok(vec![0]),
        ComponentSelect::Mask(_) => Err(Unsupported("mask on a source operand".into())),
    }
}

fn mask_lanes(mask: u8) -> Vec<u8> {
    (0u8..4).filter(|lane| mask & (1 << lane) != 0).collect()
}

/// The scalar lane a test/select component names.
fn select_lane(components: &ComponentSelect) -> u8 {
    match components {
        ComponentSelect::Scalar(lane) => *lane,
        ComponentSelect::Swizzle(swizzle) => swizzle[0],
        _ => 0,
    }
}

/// The lane interpretation an opcode imposes on its arithmetic sources.
fn operand_scalar(opcode: Opcode) -> ScalarType {
    match opcode {
        Opcode::Iadd
        | Opcode::IMad
        | Opcode::IMin
        | Opcode::IMax
        | Opcode::INeg
        | Opcode::Ishl
        | Opcode::Ishr
        | Opcode::ILt
        | Opcode::IGe
        | Opcode::IEq
        | Opcode::INe
        | Opcode::Itof => ScalarType::I32,
        Opcode::And
        | Opcode::Or
        | Opcode::Xor
        | Opcode::Not
        | Opcode::UMad
        | Opcode::UMin
        | Opcode::UMax
        | Opcode::Ushr
        | Opcode::ULt
        | Opcode::UGe
        | Opcode::UDiv
        | Opcode::Bfi
        | Opcode::Utof => ScalarType::U32,
        Opcode::Mov => ScalarType::Bits,
        _ => ScalarType::F32,
    }
}

/// The lane interpretation of an opcode's result.
fn result_scalar(opcode: Opcode, bindings: &[&Operand], declarations: &Declarations) -> ScalarType {
    match opcode {
        Opcode::Iadd
        | Opcode::IMad
        | Opcode::IMin
        | Opcode::IMax
        | Opcode::INeg
        | Opcode::Ishl
        | Opcode::Ishr
        | Opcode::Ftoi => ScalarType::I32,
        Opcode::And
        | Opcode::Or
        | Opcode::Xor
        | Opcode::Not
        | Opcode::UMad
        | Opcode::UMin
        | Opcode::UMax
        | Opcode::Ushr
        | Opcode::Ftou
        | Opcode::UDiv
        | Opcode::Lt
        | Opcode::Ge
        | Opcode::Eq
        | Opcode::Ne
        | Opcode::ILt
        | Opcode::IGe
        | Opcode::IEq
        | Opcode::INe
        | Opcode::ULt
        | Opcode::UGe
        | Opcode::Bfi
        | Opcode::LdStructured => ScalarType::U32,
        Opcode::Mov | Opcode::Movc => ScalarType::Bits,
        Opcode::Ld
        | Opcode::LdMs
        | Opcode::Gather4
        | Opcode::Sample
        | Opcode::SampleL
        | Opcode::SampleB => resource_scalar(bindings, declarations),
        _ => ScalarType::F32,
    }
}

/// The element interpretation of the resource a fetch reads.
fn resource_scalar(bindings: &[&Operand], declarations: &Declarations) -> ScalarType {
    let Some(index) = bindings.iter().find_map(|operand| {
        (operand.reg_type == RegisterType::Resource)
            .then(|| match operand.indices.first() {
                Some(OperandIndex::Imm32(index)) => Some(*index),
                _ => None,
            })
            .flatten()
    }) else {
        return ScalarType::F32;
    };
    declarations
        .resources
        .iter()
        .find(|(declared, ..)| *declared == index)
        .map_or(ScalarType::F32, |(_, _, return_type)| {
            match return_type[0] {
                dxbc::shex::ReturnType::Uint => ScalarType::U32,
                dxbc::shex::ReturnType::Sint => ScalarType::I32,
                _ => ScalarType::F32,
            }
        })
}

/// The coordinate lanes a sample or load consumes, by resource dimension.
fn coordinate_lanes(
    bindings: &[&Operand],
    declarations: &Declarations,
    load: bool,
) -> &'static [u8] {
    let dimension = bindings
        .iter()
        .find_map(|operand| {
            (operand.reg_type == RegisterType::Resource)
                .then(|| match operand.indices.first() {
                    Some(OperandIndex::Imm32(index)) => Some(*index),
                    _ => None,
                })
                .flatten()
        })
        .and_then(|index| {
            declarations
                .resources
                .iter()
                .find(|(declared, ..)| *declared == index)
                .map(|(_, dimension, _)| *dimension)
        })
        .unwrap_or("texture2d");
    if load {
        match dimension {
            "texture1d" => &[0, 3],
            "texture3d" | "texture2darray" => &[0, 1, 2, 3],
            "buffer" => &[0],
            _ => &[0, 1, 3],
        }
    } else {
        match dimension {
            "texture1d" => &[0],
            "texture3d" | "texturecube" | "texture2darray" => &[0, 1, 2],
            "texturecubearray" => &[0, 1, 2, 3],
            _ => &[0, 1],
        }
    }
}

/// Maps a variable-backed register file, or `None` for files that
/// materialize differently.
fn variable_file(register: RegisterType) -> Option<RegFile> {
    match register {
        RegisterType::Temp => Some(RegFile::Temp),
        RegisterType::Input => Some(RegFile::Input),
        RegisterType::Output => Some(RegFile::Output),
        RegisterType::OutputDepth => Some(RegFile::OutputDepth),
        RegisterType::OutputCoverageMask => Some(RegFile::OutputCoverage),
        RegisterType::ThreadID => Some(RegFile::System(SystemInput::ThreadId)),
        RegisterType::ThreadGroupID => Some(RegFile::System(SystemInput::ThreadGroupId)),
        RegisterType::ThreadIDInGroup => Some(RegFile::System(SystemInput::ThreadIdInGroup)),
        RegisterType::Coverage => Some(RegFile::System(SystemInput::Coverage)),
        RegisterType::InputPrimitiveID => Some(RegFile::System(SystemInput::PrimitiveId)),
        RegisterType::DomainLocation => Some(RegFile::System(SystemInput::DomainPoint)),
        RegisterType::OutputControlPointID => Some(RegFile::System(SystemInput::ControlPointId)),
        RegisterType::ForkInstanceID => Some(RegFile::System(SystemInput::ForkInstanceId)),
        RegisterType::PatchConstant => Some(RegFile::PatchConstant),
        _ => None,
    }
}

/// The register a variable-backed operand names.
fn operand_register(operand: &Operand) -> Result<Register, Unsupported> {
    let Some(file) = variable_file(operand.reg_type) else {
        return unsupported(format!("register file {:?}", operand.reg_type));
    };
    let index = match operand.indices.first() {
        None => 0,
        Some(OperandIndex::Imm32(index)) => *index,
        Some(other) => return unsupported(format!("register index {other:?}")),
    };
    Ok(Register { file, index })
}

/// Lifts one decoded program into an RTL function.
///
/// # Errors
///
/// Returns [`Unsupported`] for shader shapes outside the modeled subset
/// (switch flow, subroutines, exotic operand addressing) and for
/// construction failures.
pub fn lift_program(program: &Program) -> Result<LiftedShader, Unsupported> {
    let source = format!(
        "{}_{}_{}",
        program.shader_type, program.major_version, program.minor_version
    );
    let mut declarations = Declarations {
        samplers: Vec::new(),
        resources: Vec::new(),
        constant_buffers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        indexable_temps: Vec::new(),
        immediate_constants: Vec::new(),
        structured: Vec::new(),
        uavs: Vec::new(),
        thread_group: None,
        geometry: None,
        tessellation: None,
    };
    if program.shader_type == "hs" {
        return lift_hull(program, &source, declarations);
    }
    for instruction in &program.instructions {
        collect_declaration(instruction, &mut declarations, true);
    }
    let function = walk_range(
        program,
        0..program.instructions.len(),
        &mut declarations,
        &source,
    )?;
    Ok(LiftedShader {
        function,
        declarations,
        hull: None,
    })
}

/// Walks one instruction range into an RTL function.
fn walk_range(
    program: &Program,
    range: core::ops::Range<usize>,
    declarations: &mut Declarations,
    source: &str,
) -> Result<cfglib::ir::rtl::Function<ShaderDialect>, Unsupported> {
    let builder = FunctionBuilder::new(source.to_owned());
    let entry = builder.entry();
    let mut walk = Walk {
        builder,
        declarations: core::mem::replace(
            declarations,
            Declarations {
                samplers: Vec::new(),
                resources: Vec::new(),
                constant_buffers: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                indexable_temps: Vec::new(),
                immediate_constants: Vec::new(),
                structured: Vec::new(),
                uavs: Vec::new(),
                thread_group: None,
                geometry: None,
                tessellation: None,
            },
        ),
        structure: StructuredWalk::new(entry),
        blocks: 0,
        ret_block: None,
    };
    for index in range {
        let instruction = &program.instructions[index];
        let span = (
            u32::try_from(index).unwrap_or(u32::MAX),
            u32::try_from(index + 1).unwrap_or(u32::MAX),
        );
        if let InstructionKind::Generic { operands, .. } = &instruction.kind {
            walk.generic(instruction, operands, span)?;
        }
    }
    finish_walk(walk, declarations)
}

fn finish_walk(
    walk: Walk,
    declarations: &mut Declarations,
) -> Result<cfglib::ir::rtl::Function<ShaderDialect>, Unsupported> {
    walk.structure.finish()?;
    *declarations = walk.declarations;
    let function = walk
        .builder
        .finish()
        .map_err(|error| Unsupported(format!("construction: {error}")))?;
    // A walk bug that strands blocks must fail loudly, never render a
    // silently truncated body.
    let reachable = cfglib::reachable(
        function.cfg(),
        [function.cfg().entry()],
        cfglib::TraversalDirection::Outgoing,
    );
    if reachable.contains(&false) {
        return unsupported("stranded unreachable blocks");
    }
    Ok(function)
}

/// Lifts one hull shader: the control-point phase becomes the main
/// function, fork and join phases become the patch-constant functions.
fn lift_hull(
    program: &Program,
    source: &str,
    mut declarations: Declarations,
) -> Result<LiftedShader, Unsupported> {
    // Phase boundaries: (phase opcode, body start).
    let mut phases: Vec<(Opcode, usize)> = Vec::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        match instruction.opcode {
            Opcode::HsControlPointPhase | Opcode::HsForkPhase | Opcode::HsJoinPhase => {
                phases.push((instruction.opcode, index + 1));
            }
            _ => {}
        }
    }
    let end_of = |position: usize| -> usize {
        phases
            .get(position + 1)
            .map_or(program.instructions.len(), |&(_, start)| start - 1)
    };
    // Declarations: everything global, but output rows only from the
    // control-point phase (fork outputs are patch constants).
    let control_point_range = phases
        .iter()
        .position(|&(opcode, _)| opcode == Opcode::HsControlPointPhase)
        .map(|position| phases[position].1..end_of(position));
    for (index, instruction) in program.instructions.iter().enumerate() {
        let in_control_point = control_point_range
            .as_ref()
            .is_some_and(|range| range.contains(&index));
        collect_declaration(instruction, &mut declarations, in_control_point);
    }
    let mut control_point = None;
    let mut constants = Vec::new();
    for (position, &(opcode, start)) in phases.iter().enumerate() {
        let range = start..end_of(position);
        let instances = program.instructions[range.clone()]
            .iter()
            .find_map(|instruction| match instruction.kind {
                InstructionKind::DclHsForkPhaseInstanceCount { count }
                | InstructionKind::DclHsJoinPhaseInstanceCount { count } => Some(count.max(1)),
                _ => None,
            })
            .unwrap_or(1);
        let function = walk_range(program, range, &mut declarations, source)?;
        match opcode {
            Opcode::HsControlPointPhase => control_point = Some(function),
            _ => constants.push((instances, function)),
        }
    }
    let function =
        control_point.ok_or_else(|| Unsupported("hull without a control-point phase".into()))?;
    Ok(LiftedShader {
        function,
        declarations,
        hull: Some(HullPhases { constants }),
    })
}
