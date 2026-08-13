//! Formatter: IR → human-readable disassembly text.
//!
//! The primary API is the set of `write_*` functions that accept any
//! `core::fmt::Write` sink, avoiding intermediate `String` allocations.
//! The `format_*` helpers are thin wrappers that allocate a `String` for
//! callers that need an owned value.

use alloc::string::String;
use core::fmt::Write;

use super::ir::{
    ComponentSelect, CustomDataType, GsPrimitive, Instruction, InstructionKind, Operand,
    MinPrecision, OperandIndex, Program, RegisterType, ReturnType, SystemValue,
};
use super::opcodes::Opcode;

// Write functions: stream directly into a fmt::Write sink.

/// Write a decoded [`Program`] as disassembly text into `w`.
///
/// # Errors
///
/// Returns the sink's formatting error if any write fails.
pub fn write_program(w: &mut dyn Write, program: &Program) -> core::fmt::Result {
    writeln!(
        w,
        "{}_{}_{}",
        program.shader_type, program.major_version, program.minor_version
    )?;

    let mut indent = 0u32;

    for warning in &program.warnings {
        writeln!(w, "// WARNING: {warning}")?;
    }

    for instr in &program.instructions {
        match instr.opcode {
            Opcode::Else | Opcode::EndIf | Opcode::EndLoop | Opcode::EndSwitch => {
                indent = indent.saturating_sub(1);
            }
            _ => {}
        }

        for _ in 0..indent {
            w.write_str("  ")?;
        }
        write_instruction(w, instr)?;
        w.write_char('\n')?;

        match instr.opcode {
            Opcode::If | Opcode::Else | Opcode::Loop | Opcode::Switch => {
                indent += 1;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Write a single [`Instruction`] into `w`.
///
/// # Errors
///
/// Returns the sink's formatting error if any write fails.
pub fn write_instruction(w: &mut dyn Write, instr: &Instruction) -> core::fmt::Result {
    match &instr.kind {
        InstructionKind::Generic {
            operands,
            trailing_tokens,
        } => {
            write_generic(w, instr, operands)?;
            if !trailing_tokens.is_empty() {
                w.write_str(" /* trailing")?;
                for token in trailing_tokens {
                    write!(w, " 0x{token:08x}")?;
                }
                w.write_str(" */")?;
            }
            Ok(())
        }
        InstructionKind::HsPhase => write_mnemonic(w, instr),
        InstructionKind::CustomData {
            subtype,
            values,
            raw_dword_count,
            ..
        } => write_custom_data(w, *subtype, values, *raw_dword_count),
        _ => write_declaration(w, instr),
    }
}

/// Write the full instruction mnemonic (opcode + modifier suffixes) into `w`.
///
/// # Errors
///
/// Returns the sink's formatting error if any write fails.
pub fn write_mnemonic(w: &mut dyn Write, instr: &Instruction) -> core::fmt::Result {
    w.write_str(instr.opcode.name())?;
    match instr.resinfo_return_type {
        Some(1) => w.write_str("_rcpFloat")?,
        Some(2) => w.write_str("_uint")?,
        _ => {}
    }
    // Conditional test suffix (`if_z`/`if_nz`, `breakc_z`/`breakc_nz`, …). This
    // changes the branch semantics — `_z` takes the branch on zero, `_nz` on
    // non-zero — so it must always be shown for the conditional opcodes.
    if matches!(
        instr.opcode,
        Opcode::If
            | Opcode::Breakc
            | Opcode::Callc
            | Opcode::Continuec
            | Opcode::Discard
            | Opcode::Retc
    ) {
        w.write_str(if instr.test_nonzero { "_nz" } else { "_z" })?;
    }
    if instr.saturate {
        w.write_str("_sat")?;
    }
    if let Some([u, v, ww]) = instr
        .tex_offsets
        .filter(|offsets| offsets.iter().any(|offset| *offset != 0))
    {
        write!(w, "({u}, {v}, {ww})")?;
    }
    // Resource extended tokens on sample/ld/resinfo: fxc renders these as
    // `_indexable(<dim>[, stride=N])(<rt>,<rt>,<rt>,<rt>)`.
    if let Some(rd) = instr.resource_dim {
        let dim = (rd >> 6) & 0x1F;
        let stride = (rd >> 11) & 0xF_FFFF;
        write!(w, "_indexable({}", resource_dim_name(dim))?;
        if matches!(dim, 11 | 12) {
            write!(w, ", stride={stride}")?;
        }
        w.write_str(")")?;
    }
    if let Some(rr) = instr.resource_return_type {
        write!(
            w,
            "({},{},{},{})",
            ReturnType::from_u32((rr >> 6) & 0xF).name(),
            ReturnType::from_u32((rr >> 10) & 0xF).name(),
            ReturnType::from_u32((rr >> 14) & 0xF).name(),
            ReturnType::from_u32((rr >> 18) & 0xF).name(),
        )?;
    }
    Ok(())
}

/// SB resource-dimension name (`D3D10_SB_RESOURCE_DIMENSION`), as used in the
/// resource extended token on sample/ld/resinfo.
fn resource_dim_name(dim: u32) -> &'static str {
    match dim {
        1 => "buffer",
        2 => "texture1d",
        3 => "texture2d",
        4 => "texture2dms",
        5 => "texture3d",
        6 => "texturecube",
        7 => "texture1darray",
        8 => "texture2darray",
        9 => "texture2dmsarray",
        10 => "texturecubearray",
        11 => "raw_buffer",
        12 => "structured_buffer",
        _ => "unknown",
    }
}

/// Write a single [`Operand`] into `w` (float immediate formatting).
///
/// # Errors
///
/// Returns the sink's formatting error if any write fails.
pub fn write_operand(w: &mut dyn Write, op: &Operand) -> core::fmt::Result {
    write_operand_core(w, op, None, ImmediateType::Float)
}

// String-returning wrappers for backward compatibility.

/// Format a decoded [`Program`] into disassembly text.
#[must_use]
pub fn format_program(program: &Program) -> String {
    let mut out = String::new();
    let _ = write_program(&mut out, program);
    out
}

/// Format a single [`Instruction`] into its disassembly text.
#[must_use]
pub fn format_instruction(instr: &Instruction) -> String {
    let mut out = String::new();
    let _ = write_instruction(&mut out, instr);
    out
}

/// Format the bare opcode name.
#[must_use]
pub fn format_opcode(opcode: &Opcode) -> &'static str {
    opcode.name()
}

/// Format the full instruction mnemonic, including modifier suffixes.
#[must_use]
pub fn format_mnemonic(instr: &Instruction) -> String {
    let mut out = String::new();
    let _ = write_mnemonic(&mut out, instr);
    out
}

/// Format a single [`Operand`].
#[must_use]
pub fn format_operand(op: &Operand) -> String {
    let mut out = String::new();
    let _ = write_operand(&mut out, op);
    out
}

/// Operand value type, used to select the correct immediate formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmediateType {
    /// Display immediates as floating-point (e.g. `1.000000`).
    Float,
    /// Display immediates as signed decimal integers (e.g. `-3`).
    Int,
    /// Display immediates as unsigned hex integers (e.g. `0x000000FF`).
    Uint,
}

/// Classify an opcode's source operand type for immediate formatting.
#[must_use]
pub fn opcode_imm_type(op: Opcode) -> ImmediateType {
    match op {
        // Signed integer ALU
        Opcode::Iadd | Opcode::IEq | Opcode::IGe | Opcode::ILt
        | Opcode::IMad | Opcode::IMax | Opcode::IMin | Opcode::IMul
        | Opcode::INe | Opcode::INeg | Opcode::Ishl | Opcode::Ishr
        | Opcode::Ibfe | Opcode::FirstbitShi
        // Conversion: int sources
        | Opcode::Itof | Opcode::Itod
        // Signed atomics
        | Opcode::AtomicIAdd | Opcode::AtomicIMax | Opcode::AtomicIMin
        | Opcode::ImmAtomicIAdd | Opcode::ImmAtomicIMax | Opcode::ImmAtomicIMin
        // Raw/structured buffer loads: the immediate is a byte offset (integer)
        | Opcode::LdRaw | Opcode::LdStructured
        => ImmediateType::Int,

        // Unsigned / bitwise — render as hex
        | Opcode::And | Opcode::Or | Opcode::Xor | Opcode::Not
        | Opcode::UDiv | Opcode::ULt | Opcode::UGe | Opcode::UMul | Opcode::UMad
        | Opcode::UMax | Opcode::UMin | Opcode::Ushr
        | Opcode::Ubfe | Opcode::Bfi | Opcode::Bfrev | Opcode::Countbits
        | Opcode::FirstbitHi | Opcode::FirstbitLo
        | Opcode::Uaddc | Opcode::Usubb
        | Opcode::Utof | Opcode::Utod | Opcode::F16tof32
        | Opcode::AtomicAnd | Opcode::AtomicOr | Opcode::AtomicXor
        | Opcode::AtomicCmpStore | Opcode::AtomicUMax | Opcode::AtomicUMin
        | Opcode::ImmAtomicAnd | Opcode::ImmAtomicOr | Opcode::ImmAtomicXor
        | Opcode::ImmAtomicExch | Opcode::ImmAtomicCmpExch
        | Opcode::ImmAtomicUMax | Opcode::ImmAtomicUMin
        | Opcode::Msad
        => ImmediateType::Uint,

        // Floating-point, bit-copy, flow-control, and declaration opcodes.
        _ => ImmediateType::Float,
    }
}

/// Write a generic ALU / flow-control / sample instruction into `w`.
fn write_generic(
    w: &mut dyn Write,
    instr: &Instruction,
    operands: &[Operand],
) -> core::fmt::Result {
    write_mnemonic(w, instr)?;

    if operands.is_empty() {
        return Ok(());
    }

    let imm_type = opcode_imm_type(instr.opcode);
    let dest_width = dest_component_count(operands.first());
    let src_width = source_width_override(instr.opcode, dest_width);

    w.write_char(' ')?;
    for (i, op) in operands.iter().enumerate() {
        if i > 0 {
            w.write_str(", ")?;
        }
        if i == 0 {
            write_operand_core(w, op, None, imm_type)?;
        } else {
            write_operand_core(w, op, src_width.map(|v| v as usize), imm_type)?;
        }
    }
    Ok(())
}

/// Write a declaration instruction into `w`.
fn write_declaration(w: &mut dyn Write, instr: &Instruction) -> core::fmt::Result {
    let name = instr.opcode.name();
    match &instr.kind {
        InstructionKind::DclGlobalFlags { flags } => {
            w.write_str("dcl_globalFlags ")?;
            for (i, f) in flags.iter().enumerate() {
                if i > 0 {
                    w.write_char('|')?;
                }
                w.write_str(f)?;
            }
            Ok(())
        }
        InstructionKind::DclInput {
            interpolation,
            system_value,
            operands,
        } => {
            w.write_str(name)?;
            if let Some(interp) = interpolation {
                write!(w, " {interp}")?;
                if !operands.is_empty() {
                    w.write_str(", ")?;
                }
            } else if !operands.is_empty() {
                w.write_char(' ')?;
            }
            write_operand_list(w, operands)?;
            if let Some(sv) = system_value.filter(|value| *value != SystemValue::Undefined) {
                write!(w, ", {sv}")?;
            }
            Ok(())
        }
        InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            w.write_str(name)?;
            if !operands.is_empty() {
                w.write_char(' ')?;
                write_operand_list(w, operands)?;
            }
            if let Some(sv) = system_value.filter(|value| *value != SystemValue::Undefined) {
                write!(w, ", {sv}")?;
            }
            Ok(())
        }
        InstructionKind::DclResource {
            dimension,
            return_type,
            operands,
            ..
        } => {
            write!(w, "dcl_resource_{dimension} (")?;
            for (i, rt) in return_type.iter().enumerate() {
                if i > 0 {
                    w.write_char(',')?;
                }
                w.write_str(rt.name())?;
            }
            w.write_str(") ")?;
            write_operand_list(w, operands)
        }
        InstructionKind::DclSampler { mode, operands, .. } => {
            w.write_str("dcl_sampler ")?;
            write_operand_list(w, operands)?;
            write!(w, ", mode_{mode}")
        }
        InstructionKind::DclConstantBuffer {
            access, operands, ..
        } => {
            w.write_str("dcl_constantbuffer ")?;
            write_operand_list(w, operands)?;
            write!(w, ", {access}")
        }
        _ => write_special_declaration(w, instr),
    }
}

fn write_special_declaration(w: &mut dyn Write, instr: &Instruction) -> core::fmt::Result {
    match &instr.kind {
        InstructionKind::DclTemps { count } => write!(w, "dcl_temps {count}"),
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            write!(w, "dcl_indexableTemp x{reg}[{size}], {components}")
        }
        InstructionKind::DclGsInputPrimitive { primitive } => {
            if let GsPrimitive::ControlPointPatch(n) = primitive {
                write!(w, "dcl_inputPrimitive patchlist_{n}")
            } else {
                write!(w, "dcl_inputPrimitive {}", primitive.name())
            }
        }
        InstructionKind::DclGsOutputTopology { topology } => {
            write!(w, "dcl_outputTopology {}", topology.name())
        }
        InstructionKind::DclMaxOutputVertexCount { count } => {
            write!(w, "dcl_maxOutputVertexCount {count}")
        }
        InstructionKind::DclGsInstanceCount { count } => write!(w, "dcl_gsInstanceCount {count}"),
        InstructionKind::DclOutputControlPointCount { count } => {
            write!(w, "dcl_outputControlPointCount {count}")
        }
        InstructionKind::DclInputControlPointCount { count } => {
            write!(w, "dcl_inputControlPointCount {count}")
        }
        InstructionKind::DclTessDomain { domain } => write!(w, "dcl_tessDomain {domain}"),
        InstructionKind::DclTessPartitioning { partitioning } => {
            write!(w, "dcl_tessPartitioning {partitioning}")
        }
        InstructionKind::DclTessOutputPrimitive { primitive } => {
            write!(w, "dcl_tessOutputPrimitive {primitive}")
        }
        InstructionKind::DclHsMaxTessFactor { value } => write!(w, "dcl_hsMaxTessFactor {value}"),
        InstructionKind::DclHsForkPhaseInstanceCount { count } => {
            write!(w, "dcl_hsForkPhaseInstanceCount {count}")
        }
        InstructionKind::DclHsJoinPhaseInstanceCount { count } => {
            write!(w, "dcl_hsJoinPhaseInstanceCount {count}")
        }
        InstructionKind::DclThreadGroup { x, y, z } => write!(w, "dcl_thread_group {x}, {y}, {z}"),
        InstructionKind::DclUavTyped {
            dimension,
            return_type,
            operands,
            ..
        } => {
            write!(w, "dcl_uav_typed_{dimension} (")?;
            for (i, rt) in return_type.iter().enumerate() {
                if i > 0 {
                    w.write_char(',')?;
                }
                w.write_str(rt.name())?;
            }
            w.write_str(") ")?;
            write_operand_list(w, operands)
        }
        InstructionKind::DclUavRaw {
            flags, operands, ..
        } => {
            w.write_str("dcl_uav_raw")?;
            write_uav_flags(w, *flags)?;
            w.write_char(' ')?;
            write_operand_list(w, operands)
        }
        InstructionKind::DclUavStructured {
            flags,
            stride,
            operands,
            ..
        } => {
            w.write_str("dcl_uav_structured")?;
            write_uav_flags(w, *flags)?;
            w.write_char(' ')?;
            write_operand_list(w, operands)?;
            write!(w, ", {stride}")
        }
        InstructionKind::DclResourceRaw { operands, .. } => {
            w.write_str("dcl_resource_raw ")?;
            write_operand_list(w, operands)
        }
        InstructionKind::DclResourceStructured {
            stride, operands, ..
        } => {
            w.write_str("dcl_resource_structured ")?;
            write_operand_list(w, operands)?;
            write!(w, ", {stride}")
        }
        InstructionKind::DclThreadGroupSharedMemoryRaw {
            operands,
            byte_count,
        } => {
            w.write_str("dcl_tgsm_raw ")?;
            write_operand_list(w, operands)?;
            write!(w, ", {byte_count}")
        }
        InstructionKind::DclThreadGroupSharedMemoryStructured {
            operands,
            stride,
            count,
        } => {
            w.write_str("dcl_tgsm_structured ")?;
            write_operand_list(w, operands)?;
            write!(w, ", {stride}, {count}")
        }
        InstructionKind::DclIndexRange { operands, count } => {
            w.write_str("dcl_indexRange ")?;
            write_operand_list(w, operands)?;
            write!(w, ", {count}")
        }
        _ => write_function_declaration(w, instr),
    }
}

fn write_function_declaration(w: &mut dyn Write, instr: &Instruction) -> core::fmt::Result {
    let name = instr.opcode.name();
    match &instr.kind {
        InstructionKind::DclFunctionBody { index } => write!(w, "dcl_function_body fb{index}"),
        InstructionKind::DclFunctionTable {
            table_index,
            body_indices,
        } => {
            write!(w, "dcl_function_table ft{table_index} = {{")?;
            for (i, b) in body_indices.iter().enumerate() {
                if i > 0 {
                    w.write_str(", ")?;
                }
                write!(w, "fb{b}")?;
            }
            w.write_char('}')
        }
        InstructionKind::DclInterface {
            dynamically_indexed,
            interface_index,
            expected_table_size,
            array_length,
            table_indices,
        } => {
            write!(
                w,
                "dcl_interface{} fp{interface_index}[{expected_table_size}][{array_length}][{}] = {{",
                if *dynamically_indexed { "_dynamic" } else { "" },
                table_indices.len()
            )?;
            for (i, t) in table_indices.iter().enumerate() {
                if i > 0 {
                    w.write_str(", ")?;
                }
                write!(w, "ft{t}")?;
            }
            w.write_char('}')
        }
        InstructionKind::InterfaceCall {
            function_index,
            operands,
        } => {
            write!(w, "fcall {function_index}, ")?;
            write_operand_list(w, operands)
        }
        _ => w.write_str(name),
    }
}

/// Write a comma-separated list of operands.
fn write_operand_list(w: &mut dyn Write, operands: &[Operand]) -> core::fmt::Result {
    for (i, op) in operands.iter().enumerate() {
        if i > 0 {
            w.write_str(", ")?;
        }
        write_operand(w, op)?;
    }
    Ok(())
}

fn write_uav_flags(w: &mut dyn Write, flags: u32) -> core::fmt::Result {
    if flags & 0x1 != 0 {
        w.write_str("_glc")?;
    }
    if flags & 0x2 != 0 {
        w.write_str("_opc")?;
    }
    Ok(())
}

fn write_custom_data(
    w: &mut dyn Write,
    subtype: CustomDataType,
    values: &[[f32; 4]],
    raw_dword_count: usize,
) -> core::fmt::Result {
    let ty = match subtype {
        CustomDataType::Comment => "comment",
        CustomDataType::DebugInfo => "debuginfo",
        CustomDataType::Opaque => "opaque",
        CustomDataType::ImmediateConstantBuffer => "dcl_immediateConstantBuffer",
        CustomDataType::Other(v) => {
            return write!(w, "customdata // subtype={v}, {raw_dword_count} dwords");
        }
    };
    if subtype == CustomDataType::ImmediateConstantBuffer && !values.is_empty() {
        write!(w, "{ty} {{")?;
        for v in values {
            w.write_str("\n  { ")?;
            write_immediate(w, v[0].to_bits(), ImmediateType::Float)?;
            w.write_str(", ")?;
            write_immediate(w, v[1].to_bits(), ImmediateType::Float)?;
            w.write_str(", ")?;
            write_immediate(w, v[2].to_bits(), ImmediateType::Float)?;
            w.write_str(", ")?;
            write_immediate(w, v[3].to_bits(), ImmediateType::Float)?;
            w.write_str(" }")?;
        }
        w.write_str("\n}")
    } else {
        write!(w, "{ty} // {raw_dword_count} dwords")
    }
}

/// Determine the number of swizzle components needed for a destination mask.
///
/// This is the position of the highest set bit + 1, NOT the popcount.
/// For `.xy` (bits 0,1) → 2, for `.zw` (bits 2,3) → 4, for `.xw` (bits 0,3) → 4.
/// Source swizzles can be safely truncated to this length since higher
/// components are never read.
fn dest_component_count(op: Option<&Operand>) -> Option<u8> {
    let op = op?;
    match &op.components {
        ComponentSelect::Mask(mask) => {
            if *mask == 0 {
                None
            } else {
                // Highest set bit position + 1
                Some(
                    u8::try_from(8 - mask.leading_zeros())
                        .expect("an 8-bit mask has at most eight components"),
                )
            }
        }
        _ => None,
    }
}

/// Override source width for instructions that read more components than
/// the destination mask implies (e.g., dp3 always reads 3 source components,
/// texture sampling always reads 4-component swizzles).
fn source_width_override(opcode: Opcode, dest_width: Option<u8>) -> Option<u8> {
    match opcode {
        Opcode::Dp2 => Some(2),
        Opcode::Dp3 => Some(3),
        Opcode::Dp4 => Some(4),
        // Texture/resource instructions: source swizzles are independent of
        // the destination mask — don't truncate them.
        Opcode::Sample
        | Opcode::SampleB
        | Opcode::SampleC
        | Opcode::SampleCLz
        | Opcode::SampleD
        | Opcode::SampleL
        | Opcode::Ld
        | Opcode::LdMs
        | Opcode::LdUavTyped
        | Opcode::Gather4
        | Opcode::Gather4C
        | Opcode::Gather4Po
        | Opcode::Gather4PoC
        | Opcode::Resinfo
        | Opcode::Lod
        | Opcode::SamplePos
        | Opcode::SampleInfo
        | Opcode::BufInfo
        // Store/atomic instructions: operand widths are dictated by the
        // resource, not the destination mask.
        | Opcode::StoreUavTyped
        | Opcode::StoreRaw
        | Opcode::StoreStructured
        | Opcode::AtomicAnd
        | Opcode::AtomicOr
        | Opcode::AtomicXor
        | Opcode::AtomicCmpStore
        | Opcode::AtomicIAdd
        | Opcode::AtomicIMax
        | Opcode::AtomicIMin
        | Opcode::AtomicUMax
        | Opcode::AtomicUMin
        | Opcode::ImmAtomicAlloc
        | Opcode::ImmAtomicConsume
        | Opcode::ImmAtomicIAdd
        | Opcode::ImmAtomicAnd
        | Opcode::ImmAtomicOr
        | Opcode::ImmAtomicXor
        | Opcode::ImmAtomicExch
        | Opcode::ImmAtomicCmpExch
        | Opcode::ImmAtomicIMax
        | Opcode::ImmAtomicIMin
        | Opcode::ImmAtomicUMax
        | Opcode::ImmAtomicUMin
        | Opcode::LdRaw
        | Opcode::LdStructured => None,
        _ => dest_width,
    }
}

/// Write a source operand, truncating its swizzle to `width` components.
fn write_operand_core(
    w: &mut dyn Write,
    op: &Operand,
    swizzle_width: Option<usize>,
    imm_type: ImmediateType,
) -> core::fmt::Result {
    // Immediates
    if op.reg_type == RegisterType::Immediate32 {
        w.write_str("l(")?;
        for (i, &v) in op.immediate_values.iter().enumerate() {
            if i > 0 {
                w.write_str(", ")?;
            }
            write_immediate(w, v, imm_type)?;
        }
        w.write_char(')')?;
        return write_operand_annotations(w, op);
    }
    if op.reg_type == RegisterType::Immediate64 {
        w.write_str("d(")?;
        let mut i = 0;
        let mut first = true;
        while i + 1 < op.immediate_values.len() {
            if !first {
                w.write_str(", ")?;
            }
            first = false;
            let lo = u64::from(op.immediate_values[i]);
            let hi = u64::from(op.immediate_values[i + 1]);
            let f = f64::from_bits(lo | (hi << 32));
            write!(w, "{f:?}")?;
            i += 2;
        }
        w.write_char(')')?;
        return write_operand_annotations(w, op);
    }

    // Modifiers wrap the entire operand
    if op.negate {
        w.write_char('-')?;
    }
    if op.abs {
        w.write_char('|')?;
    }

    // Register prefix
    w.write_str(op.reg_type.prefix())?;

    // Indices
    match op.indices.len() {
        0 => {}
        1 => {
            if matches!(
                &op.indices[0],
                OperandIndex::Relative(_) | OperandIndex::RelativePlusImm(_, _)
            ) {
                w.write_char('[')?;
                write_index(w, &op.indices[0])?;
                w.write_char(']')?;
            } else {
                write_index(w, &op.indices[0])?;
            }
        }
        _ => {
            write_index(w, &op.indices[0])?;
            for idx in &op.indices[1..] {
                w.write_char('[')?;
                write_index(w, idx)?;
                w.write_char(']')?;
            }
        }
    }

    // Swizzle/mask
    write_components(w, &op.components, swizzle_width)?;

    if op.abs {
        w.write_char('|')?;
    }
    write_operand_annotations(w, op)
}

fn write_operand_annotations(w: &mut dyn Write, op: &Operand) -> core::fmt::Result {
    let precision = match op.min_precision {
        MinPrecision::Default => None,
        MinPrecision::Float16 => Some("min16f"),
        MinPrecision::Float2_8 => Some("min2_8f"),
        MinPrecision::Sint16 => Some("min16i"),
        MinPrecision::Uint16 => Some("min16u"),
        MinPrecision::Reserved(_) => None,
    };
    if let Some(precision) = precision {
        write!(w, " {{{precision}}}")?;
    } else if let MinPrecision::Reserved(value) = op.min_precision {
        write!(w, " {{minprec={}}}", value & 0x7)?;
    }
    if op.non_uniform {
        w.write_str(" {nonuniform}")?;
    }
    Ok(())
}

fn write_index(w: &mut dyn Write, idx: &OperandIndex) -> core::fmt::Result {
    match idx {
        OperandIndex::Imm32(v) => write!(w, "{v}"),
        OperandIndex::Imm64(v) => write!(w, "{v}"),
        OperandIndex::Relative(sub) => write_operand(w, sub),
        OperandIndex::RelativePlusImm(imm, sub) => {
            write_operand(w, sub)?;
            write!(w, " + {imm}")
        }
        OperandIndex::RelativePlusImm64(imm, sub) => {
            write_operand(w, sub)?;
            write!(w, " + {imm}")
        }
    }
}

fn write_components(
    w: &mut dyn Write,
    comp: &ComponentSelect,
    width: Option<usize>,
) -> core::fmt::Result {
    const COMPS: [char; 4] = ['x', 'y', 'z', 'w'];
    match comp {
        ComponentSelect::ZeroComponent | ComponentSelect::OneComponent => Ok(()),
        ComponentSelect::Mask(mask) => {
            if *mask == 0 {
                return Ok(());
            }
            w.write_char('.')?;
            if mask & 1 != 0 {
                w.write_char('x')?;
            }
            if mask & 2 != 0 {
                w.write_char('y')?;
            }
            if mask & 4 != 0 {
                w.write_char('z')?;
            }
            if mask & 8 != 0 {
                w.write_char('w')?;
            }
            Ok(())
        }
        ComponentSelect::Swizzle(s) => {
            let count = width.unwrap_or(4).min(4);
            // Build the (potentially truncated) swizzle into a small buffer
            let mut buf = [0u8; 4];
            for i in 0..count {
                buf[i] = COMPS[s[i] as usize] as u8;
            }
            // Trim trailing repeated components
            let chars = &buf[..count];
            let trimmed = trim_swizzle_buf(chars);
            if !trimmed.is_empty() {
                w.write_char('.')?;
                for &c in trimmed {
                    w.write_char(c as char)?;
                }
            }
            Ok(())
        }
        ComponentSelect::Scalar(c) => {
            w.write_char('.')?;
            w.write_char(COMPS[*c as usize])
        }
    }
}

/// Trim trailing repeated swizzle components from a byte buffer.
fn trim_swizzle_buf(chars: &[u8]) -> &[u8] {
    if chars.len() <= 1 {
        return chars;
    }
    // All same → single component
    if chars.iter().all(|&c| c == chars[0]) {
        return &chars[..1];
    }
    let mut end = chars.len();
    while end > 1 && chars[end - 1] == chars[end - 2] {
        end -= 1;
    }
    &chars[..end]
}

/// Write a single immediate value according to its type classification.
fn write_immediate(w: &mut dyn Write, val: u32, imm_type: ImmediateType) -> core::fmt::Result {
    match imm_type {
        ImmediateType::Float => {
            let f = f32::from_bits(val);
            if f.is_finite() {
                write!(w, "{f:.6}")
            } else if f.is_nan() {
                write!(w, "0x{val:08X}")
            } else {
                write!(w, "{f:.6}")
            }
        }
        ImmediateType::Int => write!(w, "{}", i32::from_ne_bytes(val.to_ne_bytes())),
        ImmediateType::Uint => {
            if val == 0 {
                w.write_char('0')
            } else {
                write!(w, "0x{val:08x}")
            }
        }
    }
}

// Display impls.

impl core::fmt::Display for Instruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_instruction(f, self)
    }
}

impl core::fmt::Display for Operand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_operand(f, self)
    }
}

impl core::fmt::Display for Opcode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

impl core::fmt::Display for Program {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_program(f, self)
    }
}
