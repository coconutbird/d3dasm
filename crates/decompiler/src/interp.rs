//! Reference interpreter for the decompiler's SM4/SM5 subset.
//!
//! Executes a decoded [`Program`] directly — independently of the MLIL
//! lift, so lift and interpreter bugs cannot cancel — with all external
//! state (inputs, constant buffers, resources) derived from a seeded
//! hash. Running the original program and a recompilation of its
//! decompiled HLSL under the same seed and comparing outputs bit-for-bit
//! is the decompiler's semantic-equivalence gate: every approximate
//! operation (`exp`, `rsq`, samples) is the same deterministic function
//! on both sides, so any output difference is a translation defect.

use std::collections::BTreeMap;

use dxbc::shex::{CustomDataType, InstructionKind, Opcode, OperandIndex, Program};

/// One register value: four raw 32-bit lanes.
pub type Lanes = [u32; 4];

/// One traced destination write: opcode, mask, and written lanes.
pub type TraceEntry = (Opcode, u8, Lanes);

/// Why execution could not produce a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Halt {
    /// An opcode or operand shape outside the interpreted subset.
    Unsupported(String),
    /// The step budget ran out (an input-driven loop went long).
    Budget,
    /// Structurally malformed flow (unmatched markers).
    Malformed(String),
}

impl core::fmt::Display for Halt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "uninterpreted: {what}"),
            Self::Budget => write!(f, "step budget exhausted"),
            Self::Malformed(what) => write!(f, "malformed flow: {what}"),
        }
    }
}

/// Stable identities for input and output registers, so recompiled
/// programs with renumbered registers still compare: keys come from
/// signature semantics.
#[derive(Debug, Clone, Default)]
pub struct IoKeys {
    /// Input register → stable identity.
    pub inputs: BTreeMap<u32, u64>,
    /// Output register → stable identity.
    pub outputs: BTreeMap<u32, u64>,
    /// Patch-constant `(register, lane)` → `(row identity, lane in row)`.
    pub patch: BTreeMap<(u32, u8), (u64, u8)>,
}

/// The architectural results of one execution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outputs {
    /// Written output registers, by stable identity.
    pub outputs: BTreeMap<u64, Lanes>,
    /// `oDepth`, when written.
    pub depth: Option<u32>,
    /// `oMask`, when written.
    pub coverage: Option<u32>,
    /// Whether a `discard` fired.
    pub discarded: bool,
    /// Typed UAV stores in program order: slot, canonical coordinate,
    /// masked value.
    pub uav_stores: Vec<(u32, Lanes, Lanes)>,
    /// Geometry-shader emissions: the output-register snapshot at each
    /// emit, keyed by stable identity.
    pub vertices: Vec<BTreeMap<u64, Lanes>>,
    /// Strip cuts, as counts of vertices emitted before each cut.
    pub cuts: Vec<usize>,
    /// Hull-shader control-point outputs, one snapshot per point.
    pub control_points: Vec<BTreeMap<u64, Lanes>>,
    /// Hull-shader patch constants, keyed by signature-row identity.
    pub patch: BTreeMap<u64, Lanes>,
}

/// Deterministic external state: everything the shader reads that it
/// does not compute is a pure function of the seed.
#[derive(Debug, Clone, Copy)]
pub struct World {
    /// Seed mixed into every external read.
    pub seed: u64,
}

impl World {
    fn mix(self, tag: u64, a: u64, b: u64, c: u64) -> u32 {
        let mut state = self
            .seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(tag)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            .wrapping_add(a)
            .wrapping_mul(0x94d0_49bb_1331_11eb)
            .wrapping_add(b)
            .wrapping_mul(0x2545_f491_4f6c_dd1d)
            .wrapping_add(c);
        state ^= state >> 33;
        state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
        state ^= state >> 29;
        u32::try_from(state >> 32).unwrap_or(0)
    }

    /// A register-file value: lanes cycle through small floats, small
    /// integers, and simple constants so both float and integer
    /// consumers see bounded, well-behaved data.
    fn value(self, tag: u64, index: u64) -> Lanes {
        core::array::from_fn(|lane| {
            let raw = self.mix(tag, index, lane as u64, 0);
            match raw % 4 {
                0 => {
                    // Small quarter-step float in [-16, 16].
                    let steps = i64::from(raw >> 8) % 129 - 64;
                    #[expect(clippy::cast_precision_loss, reason = "|steps| <= 64")]
                    let value = steps as f32 / 4.0;
                    value.to_bits()
                }
                1 => raw % 8,          // small unsigned integer
                2 => 1.0f32.to_bits(), // common constant
                _ => 0,                // zero
            }
        })
    }

    /// A resource read: a pure function of the full access identity.
    fn fetch(self, resource: u64, address: &[u32]) -> Lanes {
        let mut a = 0u64;
        let mut b = 0u64;
        for (position, &word) in address.iter().enumerate() {
            if position % 2 == 0 {
                a = a.wrapping_mul(0x1_0000_0001).wrapping_add(u64::from(word));
            } else {
                b = b.wrapping_mul(0x1_0000_0001).wrapping_add(u64::from(word));
            }
        }
        core::array::from_fn(|lane| {
            let raw = self.mix(resource.wrapping_add(97), a, b, lane as u64);
            // Resource lanes read as moderate floats.
            #[expect(clippy::cast_precision_loss, reason = "modulus bounds the value")]
            let value = (raw % 512) as f32 / 128.0;
            value.to_bits()
        })
    }
}

const STEP_BUDGET: usize = 262_144;

/// The execution machine.
struct Machine<'a> {
    world: World,
    temps: BTreeMap<u32, Lanes>,
    indexable: BTreeMap<(u32, u32), Lanes>,
    immediate_constants: &'a [[f32; 4]],
    /// Declared SRV dimensions by register.
    dimensions: BTreeMap<u32, &'static str>,
    /// Declared UAV dimensions by register.
    uav_dimensions: BTreeMap<u32, &'static str>,
    keys: &'a IoKeys,
    outputs: Outputs,
    trace: Option<Vec<TraceEntry>>,
    trace_opcode: Opcode,
    /// The current hull control-point index.
    control_point_id: u32,
    /// The current hull fork/join instance index.
    fork_instance_id: u32,
    /// Whether output writes are hull patch constants.
    patch_phase: bool,
}

mod machine;

/// Jump targets recovered from the structured markers.
struct FlowMap {
    /// `if` index → target when the test fails.
    if_skip: BTreeMap<usize, usize>,
    /// `else` index → `endif` + 1.
    else_skip: BTreeMap<usize, usize>,
    /// `endloop` index → its `loop` index.
    loop_back: BTreeMap<usize, usize>,
    /// `loop` index → `endloop` + 1.
    loop_exit: BTreeMap<usize, usize>,
}

fn flow_map(program: &Program) -> Result<FlowMap, Halt> {
    enum Open {
        If { at: usize, else_at: Option<usize> },
        Loop { at: usize },
    }
    let mut map = FlowMap {
        if_skip: BTreeMap::new(),
        else_skip: BTreeMap::new(),
        loop_back: BTreeMap::new(),
        loop_exit: BTreeMap::new(),
    };
    let mut stack: Vec<Open> = Vec::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        match instruction.opcode {
            Opcode::If => stack.push(Open::If {
                at: index,
                else_at: None,
            }),
            Opcode::Else => match stack.last_mut() {
                Some(Open::If { at, else_at }) if else_at.is_none() => {
                    map.if_skip.insert(*at, index + 1);
                    *else_at = Some(index);
                }
                _ => return Err(Halt::Malformed("else outside if".into())),
            },
            Opcode::EndIf => match stack.pop() {
                Some(Open::If { at, else_at }) => {
                    if let Some(else_at) = else_at {
                        map.else_skip.insert(else_at, index + 1);
                    } else {
                        map.if_skip.insert(at, index + 1);
                    }
                }
                _ => return Err(Halt::Malformed("endif outside if".into())),
            },
            Opcode::Loop => stack.push(Open::Loop { at: index }),
            Opcode::EndLoop => match stack.pop() {
                Some(Open::Loop { at }) => {
                    map.loop_back.insert(index, at);
                    map.loop_exit.insert(at, index + 1);
                }
                _ => return Err(Halt::Malformed("endloop outside loop".into())),
            },
            Opcode::Switch => return Err(Halt::Unsupported("switch".into())),
            _ => {}
        }
    }
    if stack.is_empty() {
        Ok(map)
    } else {
        Err(Halt::Malformed("unclosed region".into()))
    }
}

/// Executes one decoded program against a seeded world.
///
/// # Errors
///
/// Returns [`Halt`] when the program leaves the interpreted subset, the
/// step budget runs out, or structured markers are unmatched.
pub fn execute(program: &Program, world: World, keys: &IoKeys) -> Result<Outputs, Halt> {
    run(program, world, keys, false).map(|(outputs, _)| outputs)
}

/// Executes with a per-write trace of `(mask, written lanes)` pairs, for
/// differential debugging.
///
/// # Errors
///
/// Returns [`Halt`] exactly as [`execute`] does.
pub fn execute_traced(
    program: &Program,
    world: World,
    keys: &IoKeys,
) -> Result<(Outputs, Vec<TraceEntry>), Halt> {
    run(program, world, keys, true)
}

fn run(
    program: &Program,
    world: World,
    keys: &IoKeys,
    traced: bool,
) -> Result<(Outputs, Vec<TraceEntry>), Halt> {
    let map = flow_map(program)?;
    let immediate_constants = program
        .instructions
        .iter()
        .find_map(|instruction| match &instruction.kind {
            InstructionKind::CustomData {
                subtype: CustomDataType::ImmediateConstantBuffer,
                values,
                ..
            } => Some(values.as_slice()),
            _ => None,
        })
        .unwrap_or(&[]);
    let mut dimensions = BTreeMap::new();
    let mut uav_dimensions = BTreeMap::new();
    for instruction in &program.instructions {
        if let InstructionKind::DclResource {
            dimension,
            operands,
            ..
        } = &instruction.kind
            && let Some(operand) = operands.first()
            && let Some(OperandIndex::Imm32(register)) = operand.indices.first()
        {
            dimensions.insert(*register, *dimension);
        }
        if let InstructionKind::DclUavTyped {
            dimension,
            operands,
            ..
        } = &instruction.kind
            && let Some(operand) = operands.first()
            && let Some(OperandIndex::Imm32(register)) = operand.indices.first()
        {
            uav_dimensions.insert(*register, *dimension);
        }
    }
    let mut machine = Machine {
        world,
        temps: BTreeMap::new(),
        indexable: BTreeMap::new(),
        immediate_constants,
        dimensions,
        uav_dimensions,
        keys,
        outputs: Outputs::default(),
        trace: traced.then(Vec::new),
        trace_opcode: Opcode::Nop,
        control_point_id: 0,
        fork_instance_id: 0,
        patch_phase: false,
    };
    if program.shader_type == "hs" {
        run_hull(&mut machine, program, &map)?;
        return Ok((machine.outputs, machine.trace.unwrap_or_default()));
    }
    run_span(&mut machine, program, &map, 0, program.instructions.len())?;
    Ok((machine.outputs, machine.trace.unwrap_or_default()))
}

/// Runs the hull-shader phases: the control-point phase once per output
/// control point, then each patch-constant phase.
fn run_hull(machine: &mut Machine<'_>, program: &Program, map: &FlowMap) -> Result<(), Halt> {
    let mut control_points = 1u32;
    let mut phases: Vec<(Opcode, usize)> = Vec::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        if let InstructionKind::DclOutputControlPointCount { count } = instruction.kind {
            control_points = count.max(1);
        }
        if matches!(
            instruction.opcode,
            Opcode::HsControlPointPhase | Opcode::HsForkPhase | Opcode::HsJoinPhase
        ) {
            phases.push((instruction.opcode, index + 1));
        }
    }
    let end_of = |position: usize| -> usize {
        phases
            .get(position + 1)
            .map_or(program.instructions.len(), |&(_, start)| start - 1)
    };
    for (position, &(opcode, start)) in phases.iter().enumerate() {
        let end = end_of(position);
        if opcode == Opcode::HsControlPointPhase {
            for id in 0..control_points {
                machine.temps.clear();
                machine.indexable.clear();
                machine.control_point_id = id;
                run_span(machine, program, map, start, end)?;
                let snapshot = core::mem::take(&mut machine.outputs.outputs);
                machine.outputs.control_points.push(snapshot);
            }
        } else {
            let instances = program.instructions[start..end]
                .iter()
                .find_map(|instruction| match instruction.kind {
                    InstructionKind::DclHsForkPhaseInstanceCount { count }
                    | InstructionKind::DclHsJoinPhaseInstanceCount { count } => Some(count.max(1)),
                    _ => None,
                })
                .unwrap_or(1);
            machine.patch_phase = true;
            for instance in 0..instances {
                machine.temps.clear();
                machine.indexable.clear();
                machine.fork_instance_id = instance;
                run_span(machine, program, map, start, end)?;
            }
        }
    }
    Ok(())
}

/// Executes the instructions in `[start, end)`.
fn run_span(
    machine: &mut Machine<'_>,
    program: &Program,
    map: &FlowMap,
    start: usize,
    end: usize,
) -> Result<(), Halt> {
    let mut pc = start;
    let mut steps = 0usize;
    let mut loop_stack: Vec<usize> = Vec::new();
    while pc < end {
        steps += 1;
        if steps > STEP_BUDGET {
            return Err(Halt::Budget);
        }
        let instruction = &program.instructions[pc];
        let InstructionKind::Generic { operands, .. } = &instruction.kind else {
            pc += 1;
            continue;
        };
        let operands = operands.as_slice();
        match instruction.opcode {
            Opcode::If => {
                if !machine.test(instruction, operands)? {
                    pc = *map
                        .if_skip
                        .get(&pc)
                        .ok_or_else(|| Halt::Malformed("if without skip target".into()))?;
                    continue;
                }
            }
            Opcode::Else => {
                pc = *map
                    .else_skip
                    .get(&pc)
                    .ok_or_else(|| Halt::Malformed("else without skip target".into()))?;
                continue;
            }
            Opcode::EndIf | Opcode::Nop | Opcode::DclStream => {}
            Opcode::Loop => loop_stack.push(pc),
            Opcode::EndLoop => {
                // The back edge re-enters the body just after `loop`; the
                // frame stays on the stack.
                pc = *map
                    .loop_back
                    .get(&pc)
                    .ok_or_else(|| Halt::Malformed("endloop without loop".into()))?;
            }
            Opcode::Break => {
                let start = loop_top(&loop_stack)?;
                loop_stack.pop();
                pc = exit_of(map, start)?;
                continue;
            }
            Opcode::Breakc => {
                if machine.test(instruction, operands)? {
                    let start = loop_top(&loop_stack)?;
                    loop_stack.pop();
                    pc = exit_of(map, start)?;
                    continue;
                }
            }
            Opcode::Continue => {
                pc = exit_of(map, loop_top(&loop_stack)?)? - 1;
                continue;
            }
            Opcode::Continuec => {
                if machine.test(instruction, operands)? {
                    pc = exit_of(map, loop_top(&loop_stack)?)? - 1;
                    continue;
                }
            }
            Opcode::Ret => return Ok(()),
            Opcode::Retc => {
                if machine.test(instruction, operands)? {
                    return Ok(());
                }
            }
            Opcode::Discard => {
                if machine.test(instruction, operands)? {
                    machine.outputs.discarded = true;
                }
            }
            Opcode::StoreUavTyped => machine.store_uav(operands)?,
            Opcode::Emit | Opcode::EmitStream => {
                let snapshot = machine.outputs.outputs.clone();
                machine.outputs.vertices.push(snapshot);
            }
            Opcode::Cut | Opcode::CutStream => {
                let count = machine.outputs.vertices.len();
                machine.outputs.cuts.push(count);
            }
            Opcode::EmitThenCut | Opcode::EmitThenCutStream => {
                let snapshot = machine.outputs.outputs.clone();
                machine.outputs.vertices.push(snapshot);
                let count = machine.outputs.vertices.len();
                machine.outputs.cuts.push(count);
            }
            _ => {
                machine.trace_opcode = instruction.opcode;
                machine.step(instruction, operands)?;
            }
        }
        pc += 1;
    }
    Ok(())
}

fn loop_top(stack: &[usize]) -> Result<usize, Halt> {
    stack
        .last()
        .copied()
        .ok_or_else(|| Halt::Malformed("transfer outside loop".into()))
}

fn exit_of(map: &FlowMap, start: usize) -> Result<usize, Halt> {
    map.loop_exit
        .get(&start)
        .copied()
        .ok_or_else(|| Halt::Malformed("loop without exit".into()))
}

fn missing(what: &str) -> Halt {
    Halt::Malformed(format!("missing {what}"))
}
