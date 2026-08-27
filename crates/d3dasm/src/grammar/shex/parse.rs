//! Assembler: `.d3dasm` text -> [`Program`].
//!
//! Inverse of [`super::serialize::serialize`]. A byte-level cursor performs
//! recursive-descent parsing; the per-opcode dispatch mirrors
//! `dxbc::shex` decode so the resulting IR re-encodes byte-identically.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use smallvec::SmallVec;

use super::{
    CB_ACCESS, DIMENSIONS, GLOBAL_FLAGS, INTERPOLATIONS, SAMPLER_MODES, SHADER_TYPES, TESS_DOMAINS,
    TESS_OUTPUT_PRIMS, TESS_PARTITIONINGS, axis_index, intern, value_of,
};
use dxbc::shex::{
    ComponentSelect, CustomDataType, FlagNames, GsOutputTopology, GsPrimitive, Immediates, Indices,
    Instruction, InstructionKind, MinPrecision, Opcode, Operand, OperandIndex, Operands, Program,
    RegisterType, ReturnType, SmallU32Vec, SystemValue,
};

/// Error produced while parsing `.d3dasm` text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmError {
    /// Human-readable description of the failure.
    pub message: String,
}

impl core::fmt::Display for AsmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "d3dasm parse error: {}", self.message)
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, AsmError> {
    Err(AsmError {
        message: msg.into(),
    })
}

/// Parse lossless `.d3dasm` text into a [`Program`].
///
/// # Errors
///
/// Returns an error when the text contains an invalid profile, instruction,
/// operand, modifier, or declaration.
pub fn parse(text: &str) -> Result<Program, AsmError> {
    // Strip `//` comments (full-line or trailing) and blank lines.
    let mut lines = text
        .lines()
        .map(|l| l.split("//").next().unwrap_or(l).trim())
        .filter(|l| !l.is_empty());

    let profile = lines.next().ok_or_else(|| AsmError {
        message: String::from("empty input: missing profile line"),
    })?;
    let (shader_type, major_version, minor_version, fourcc) = parse_profile(profile)?;

    let mut instructions = Vec::new();
    for line in lines {
        instructions.push(parse_instruction(line)?);
    }

    Ok(Program {
        shader_type,
        major_version,
        minor_version,
        instructions,
        warnings: Vec::new(),
        fourcc,
    })
}

fn parse_profile(line: &str) -> Result<(&'static str, u32, u32, [u8; 4]), AsmError> {
    // profile=<st>_<major>_<minor> [fourcc=XXXX], e.g. `profile=ps_5_0`.
    let line = line.strip_prefix("profile=").ok_or_else(|| AsmError {
        message: format!("expected `profile=` line: {line:?}"),
    })?;
    let (profile, fourcc) = match line.split_once(' ') {
        Some((p, rest)) => {
            let f = rest
                .trim()
                .strip_prefix("fourcc=")
                .ok_or_else(|| AsmError {
                    message: format!("expected `fourcc=` tag: {rest:?}"),
                })?;
            let bytes = f.as_bytes();
            if bytes.len() != 4 {
                return err(format!("fourcc must be 4 bytes: {f:?}"));
            }
            (p, [bytes[0], bytes[1], bytes[2], bytes[3]])
        }
        None => (line, *b"SHEX"),
    };
    // Shader type names contain no '_'.
    let (st, rest) = profile.split_once('_').ok_or_else(|| AsmError {
        message: format!("malformed profile line: {line:?}"),
    })?;
    let (maj, min) = rest.split_once('_').ok_or_else(|| AsmError {
        message: format!("malformed profile line: {line:?}"),
    })?;
    let shader_type = intern(SHADER_TYPES, st).ok_or_else(|| AsmError {
        message: format!("unknown shader type: {st:?}"),
    })?;
    let major = maj.parse::<u32>().map_err(|_| AsmError {
        message: format!("bad major version: {maj:?}"),
    })?;
    let minor = min.parse::<u32>().map_err(|_| AsmError {
        message: format!("bad minor version: {min:?}"),
    })?;
    Ok((shader_type, major, minor, fourcc))
}

fn parse_instruction(line: &str) -> Result<Instruction, AsmError> {
    // Split mnemonic (up to first space) from the remainder.
    let (mnemonic, rest) = match line.find(' ') {
        Some(i) => (&line[..i], &line[i + 1..]),
        None => (line, ""),
    };

    let (opcode, modifiers) = opcode_from_mnemonic(mnemonic)?;

    let mut instr = Instruction {
        opcode,
        saturate: false,
        test_nonzero: false,
        precise_mask: 0,
        resinfo_return_type: None,
        sync_flags: 0,
        tex_offsets: None,
        resource_dim: None,
        resource_return_type: None,
        kind: InstructionKind::Generic {
            operands: SmallVec::new(),
            trailing_tokens: SmallVec::new(),
        },
    };
    apply_modifiers(&mut instr, modifiers)?;

    let mut c = Cursor::new(rest);
    instr.kind = parse_kind(opcode, &mut c)?;
    Ok(instr)
}

/// Longest-prefix match of `mnemonic` against opcode names; returns the opcode
/// and the trailing modifier string (empty or starting with '_').
fn opcode_from_mnemonic(mnemonic: &str) -> Result<(Opcode, &str), AsmError> {
    let mut best: Option<(Opcode, usize)> = None;
    for v in 0..=234u32 {
        let op = Opcode::from_u32(v);
        let name = op.name();
        if mnemonic.starts_with(name) && best.is_none_or(|(_, len)| name.len() > len) {
            best = Some((op, name.len()));
        }
    }
    if let Some((op, len)) = best {
        let rest = &mnemonic[len..];
        if !rest.is_empty() && !rest.starts_with('_') {
            return err(format!("unrecognized mnemonic: {mnemonic:?}"));
        }
        Ok((op, rest))
    } else {
        // Unknown opcode form `op<value>` (no known name is a prefix of it).
        if let Some(after) = mnemonic.strip_prefix("op") {
            let digits: usize = after.bytes().take_while(u8::is_ascii_digit).count();
            if digits > 0 {
                let value = parse_dec(&after[..digits])?;
                return Ok((Opcode::Unknown(value), &after[digits..]));
            }
        }
        err(format!("unknown opcode in mnemonic: {mnemonic:?}"))
    }
}

/// Split an opcode's modifier tail on `_`, but only at paren depth 0 so a
/// parenthesized argument may itself contain `_` (e.g. `res(structured_buffer,
/// stride=32)`).
fn split_modifiers(s: &str) -> alloc::vec::Vec<&str> {
    let mut segs = alloc::vec::Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'_' if depth == 0 => {
                if i > start {
                    segs.push(&s[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        segs.push(&s[start..]);
    }
    segs
}

fn apply_modifiers(instr: &mut Instruction, modifiers: &str) -> Result<(), AsmError> {
    for seg in split_modifiers(modifiers) {
        if seg == "sat" {
            instr.saturate = true;
        } else if seg == "nz" {
            instr.test_nonzero = true;
        } else if let Some(n) = seg.strip_prefix("ri") {
            instr.resinfo_return_type = Some(parse_dec(n)?);
        } else if let Some(h) = seg.strip_prefix("pm") {
            instr.precise_mask = u8::try_from(parse_hex(h)?).map_err(|_| AsmError {
                message: format!("precise mask exceeds u8: {h:?}"),
            })?;
        } else if let Some(h) = seg.strip_prefix("sf") {
            instr.sync_flags = u8::try_from(parse_hex(h)?).map_err(|_| AsmError {
                message: format!("sync flags exceed u8: {h:?}"),
            })?;
        } else if let Some(body) = seg.strip_prefix("off(").and_then(|s| s.strip_suffix(')')) {
            let mut it = body.split(',');
            let u = next_i8(&mut it)?;
            let v = next_i8(&mut it)?;
            let x = next_i8(&mut it)?;
            instr.tex_offsets = Some([u, v, x]);
        } else if let Some(inner) = seg.strip_prefix("res(").and_then(|s| s.strip_suffix(')')) {
            instr.resource_dim = Some(parse_resource_dim(inner)?);
        } else if let Some(inner) = seg.strip_prefix("rt(").and_then(|s| s.strip_suffix(')')) {
            instr.resource_return_type = Some(parse_resource_rt(inner)?);
        } else if let Some(h) = seg.strip_prefix("rd") {
            instr.resource_dim = Some(parse_hex(h)?);
        } else if let Some(h) = seg.strip_prefix("rr") {
            instr.resource_return_type = Some(parse_hex(h)?);
        } else {
            return err(format!("unknown modifier: _{seg}"));
        }
    }
    Ok(())
}

// Per-opcode kind dispatch (mirrors decode::decode_instruction).

fn parse_kind(op: Opcode, c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    use Opcode::{
        CustomData, DclConstantBuffer, DclGlobalFlags, DclIndexRange, DclInput, DclInputPs,
        DclInputPsSgv, DclInputPsSiv, DclInputSgv, DclInputSiv, DclOutput, DclOutputSgv,
        DclOutputSiv, DclResource, DclResourceRaw, DclResourceStructured, DclSampler,
        DclThreadGroupSharedMemoryRaw, DclThreadGroupSharedMemoryStructured,
        DclUnorderedAccessViewRaw, DclUnorderedAccessViewStructured, DclUnorderedAccessViewTyped,
    };
    match op {
        CustomData => parse_custom_data(c),
        DclGlobalFlags => parse_global_flags(c),
        DclInput | DclInputSgv | DclInputSiv | DclInputPs | DclInputPsSgv | DclInputPsSiv => {
            parse_dcl_input(c)
        }
        DclOutput | DclOutputSgv | DclOutputSiv => parse_dcl_output(c),
        DclResource => parse_dcl_resource(c),
        DclUnorderedAccessViewTyped => parse_dcl_uav_typed(c),
        DclUnorderedAccessViewRaw => {
            let op0 = parse_leading_operand(c)?;
            let flags = parse_tag_hex(c, "flags")?;
            let space = parse_tag_dec(c, "space")?;
            Ok(InstructionKind::DclUavRaw {
                flags,
                operands: one(op0),
                space,
            })
        }
        DclUnorderedAccessViewStructured => parse_dcl_uav_structured(c),
        DclResourceRaw => {
            let operand = parse_leading_operand(c)?;
            let space = parse_tag_dec(c, "space")?;
            Ok(InstructionKind::DclResourceRaw {
                operands: one(operand),
                space,
            })
        }
        DclResourceStructured => {
            let op0 = parse_leading_operand(c)?;
            let stride = parse_tag_dec(c, "stride")?;
            let space = parse_tag_dec(c, "space")?;
            Ok(InstructionKind::DclResourceStructured {
                stride,
                operands: one(op0),
                space,
            })
        }
        DclThreadGroupSharedMemoryRaw => {
            let operand = parse_leading_operand(c)?;
            let byte_count = parse_tag_dec(c, "bytes")?;
            Ok(InstructionKind::DclThreadGroupSharedMemoryRaw {
                operands: one(operand),
                byte_count,
            })
        }
        DclThreadGroupSharedMemoryStructured => {
            let operand = parse_leading_operand(c)?;
            let stride = parse_tag_dec(c, "stride")?;
            let count = parse_tag_dec(c, "count")?;
            Ok(InstructionKind::DclThreadGroupSharedMemoryStructured {
                operands: one(operand),
                stride,
                count,
            })
        }
        DclIndexRange => {
            let op0 = parse_leading_operand(c)?;
            c.skip_spaces();
            let count = parse_dec(c.take_rest())?;
            Ok(InstructionKind::DclIndexRange {
                operands: one(op0),
                count,
            })
        }
        DclSampler => {
            let op0 = parse_leading_operand(c)?;
            let mode = parse_tag_enum(c, "mode", SAMPLER_MODES)?;
            let space = parse_tag_dec(c, "space")?;
            Ok(InstructionKind::DclSampler {
                mode,
                operands: one(op0),
                space,
            })
        }
        DclConstantBuffer => {
            let op0 = parse_leading_operand(c)?;
            let access = parse_tag_enum(c, "access", CB_ACCESS)?;
            let size = parse_tag_dec(c, "size")?;
            let space = parse_tag_dec(c, "space")?;
            Ok(InstructionKind::DclConstantBuffer {
                access,
                operands: one(op0),
                size,
                space,
            })
        }
        _ => parse_scalar_kind(op, c),
    }
}

fn parse_scalar_kind(op: Opcode, c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    use Opcode::{
        DclGsInputPrimitive, DclGsInstanceCount, DclGsOutputPrimitiveTopology,
        DclHsForkPhaseInstanceCount, DclHsJoinPhaseInstanceCount, DclHsMaxTessFactor,
        DclIndexableTemp, DclInputControlPointCount, DclMaxOutputVertexCount,
        DclOutputControlPointCount, DclTemps, DclTessDomain, DclTessOutputPrimitive,
        DclTessPartitioning, DclThreadGroup,
    };
    match op {
        DclTemps => Ok(InstructionKind::DclTemps {
            count: parse_trailing_dec(c)?,
        }),
        DclIndexableTemp => {
            c.skip_spaces();
            let reg = parse_dec(c.take_token())?;
            c.skip_spaces();
            let size = parse_dec(c.take_token())?;
            c.skip_spaces();
            let components = parse_dec(c.take_token())?;
            Ok(InstructionKind::DclIndexableTemp {
                reg,
                size,
                components,
            })
        }
        DclGsInputPrimitive => parse_gs_input(c),
        DclGsOutputPrimitiveTopology => {
            c.skip_spaces();
            let topology = parse_gs_topology(c.take_rest())?;
            Ok(InstructionKind::DclGsOutputTopology { topology })
        }
        DclMaxOutputVertexCount => Ok(InstructionKind::DclMaxOutputVertexCount {
            count: parse_trailing_dec(c)?,
        }),
        DclGsInstanceCount => Ok(InstructionKind::DclGsInstanceCount {
            count: parse_trailing_dec(c)?,
        }),
        DclOutputControlPointCount => Ok(InstructionKind::DclOutputControlPointCount {
            count: parse_trailing_dec(c)?,
        }),
        DclInputControlPointCount => Ok(InstructionKind::DclInputControlPointCount {
            count: parse_trailing_dec(c)?,
        }),
        DclTessDomain => {
            c.skip_spaces();
            Ok(InstructionKind::DclTessDomain {
                domain: intern_enum(c.take_rest(), TESS_DOMAINS)?,
            })
        }
        DclTessPartitioning => {
            c.skip_spaces();
            Ok(InstructionKind::DclTessPartitioning {
                partitioning: intern_enum(c.take_rest(), TESS_PARTITIONINGS)?,
            })
        }
        DclTessOutputPrimitive => {
            c.skip_spaces();
            Ok(InstructionKind::DclTessOutputPrimitive {
                primitive: intern_enum(c.take_rest(), TESS_OUTPUT_PRIMS)?,
            })
        }
        DclHsMaxTessFactor => {
            c.skip_spaces();
            Ok(InstructionKind::DclHsMaxTessFactor {
                value: f32::from_bits(parse_hex0x(c.take_rest())?),
            })
        }
        DclHsForkPhaseInstanceCount => Ok(InstructionKind::DclHsForkPhaseInstanceCount {
            count: parse_trailing_dec(c)?,
        }),
        DclHsJoinPhaseInstanceCount => Ok(InstructionKind::DclHsJoinPhaseInstanceCount {
            count: parse_trailing_dec(c)?,
        }),
        DclThreadGroup => {
            c.skip_spaces();
            let x = parse_dec(c.take_token())?;
            c.skip_spaces();
            let y = parse_dec(c.take_token())?;
            c.skip_spaces();
            let z = parse_dec(c.take_token())?;
            Ok(InstructionKind::DclThreadGroup { x, y, z })
        }
        _ => parse_function_kind(op, c),
    }
}

fn parse_function_kind(op: Opcode, c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    use Opcode::{
        DclFunctionBody, DclFunctionTable, DclInterface, HsControlPointPhase, HsDecls, HsForkPhase,
        HsJoinPhase, InterfaceCall,
    };
    match op {
        DclFunctionBody => Ok(InstructionKind::DclFunctionBody {
            index: parse_trailing_dec(c)?,
        }),
        DclFunctionTable => {
            c.skip_spaces();
            let table_index = parse_dec(c.take_token())?;
            c.skip_spaces();
            let body_indices = parse_u32_list(c)?;
            Ok(InstructionKind::DclFunctionTable {
                table_index,
                body_indices,
            })
        }
        DclInterface => {
            c.skip_spaces();
            let interface_index = parse_dec(c.take_token())?;
            c.skip_spaces();
            let expected_table_size = parse_dec(c.take_token())?;
            c.skip_spaces();
            let array_length = parse_dec(c.take_token())?;
            c.skip_spaces();
            let dynamically_indexed = parse_dec::<u8>(c.take_token())? != 0;
            c.skip_spaces();
            let table_indices = parse_u32_list(c)?;
            Ok(InstructionKind::DclInterface {
                dynamically_indexed,
                interface_index,
                expected_table_size,
                array_length,
                table_indices,
            })
        }
        InterfaceCall => {
            c.skip_spaces();
            let function_index = parse_dec(c.take_token())?;
            Ok(InstructionKind::InterfaceCall {
                function_index,
                operands: parse_operand_list(c)?,
            })
        }
        HsDecls | HsControlPointPhase | HsForkPhase | HsJoinPhase => Ok(InstructionKind::HsPhase),
        _ => parse_generic(c),
    }
}

// Declaration payload helpers

fn parse_global_flags(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    c.skip_spaces();
    let mut flags: FlagNames = SmallVec::new();
    let rest = c.take_rest();
    if !rest.is_empty() {
        for name in rest.split('|') {
            let interned = GLOBAL_FLAGS
                .iter()
                .find(|f| **f == name)
                .ok_or_else(|| AsmError {
                    message: format!("unknown global flag: {name:?}"),
                })?;
            flags.push(*interned);
        }
    }
    Ok(InstructionKind::DclGlobalFlags { flags })
}

fn parse_dcl_input(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    let mut interpolation = None;
    let mut system_value = None;
    let mut operands: Operands = SmallVec::new();
    loop {
        c.skip_spaces();
        if c.eof() {
            break;
        }
        if let Some(inner) = c.try_tag("interp") {
            interpolation = Some(intern_enum(inner, INTERPOLATIONS)?);
        } else if let Some(inner) = c.try_tag("sv") {
            system_value = Some(intern_system_value(inner)?);
        } else {
            operands.push(parse_operand(c)?);
        }
    }
    Ok(InstructionKind::DclInput {
        interpolation,
        system_value,
        operands,
    })
}

fn parse_dcl_output(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    let mut system_value = None;
    let mut operands: Operands = SmallVec::new();
    loop {
        c.skip_spaces();
        if c.eof() {
            break;
        }
        if let Some(inner) = c.try_tag("sv") {
            system_value = Some(intern_system_value(inner)?);
        } else {
            operands.push(parse_operand(c)?);
        }
    }
    Ok(InstructionKind::DclOutput {
        system_value,
        operands,
    })
}

fn parse_dcl_resource(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    let op0 = parse_leading_operand(c)?;
    let dimension = parse_tag_enum(c, "dim", DIMENSIONS)?;
    let return_type = parse_rt_tag(c)?;
    let sample_count = parse_tag_dec(c, "samples")?;
    Ok(InstructionKind::DclResource {
        dimension,
        sample_count,
        return_type,
        operands: one(op0),
        space: parse_tag_dec(c, "space")?,
    })
}

fn parse_dcl_uav_typed(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    let op0 = parse_leading_operand(c)?;
    let dimension = parse_tag_enum(c, "dim", DIMENSIONS)?;
    let return_type = parse_rt_tag(c)?;
    let flags = parse_tag_hex(c, "flags")?;
    let space = parse_tag_dec(c, "space")?;
    Ok(InstructionKind::DclUavTyped {
        dimension,
        flags,
        return_type,
        operands: one(op0),
        space,
    })
}

fn parse_dcl_uav_structured(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    let op0 = parse_leading_operand(c)?;
    let stride = parse_tag_dec(c, "stride")?;
    let flags = parse_tag_hex(c, "flags")?;
    let space = parse_tag_dec(c, "space")?;
    Ok(InstructionKind::DclUavStructured {
        flags,
        stride,
        operands: one(op0),
        space,
    })
}

fn parse_gs_input(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    c.skip_spaces();
    let tok = c.take_rest();
    let primitive = if let Some(inner) = tok
        .strip_prefix("patchlist(")
        .and_then(|s| s.strip_suffix(')'))
    {
        GsPrimitive::ControlPointPatch(parse_dec::<u8>(inner)?)
    } else {
        match tok {
            "undefined" => GsPrimitive::Undefined,
            "point" => GsPrimitive::Point,
            "line" => GsPrimitive::Line,
            "triangle" => GsPrimitive::Triangle,
            "lineAdj" => GsPrimitive::LineAdj,
            "triangleAdj" => GsPrimitive::TriangleAdj,
            _ => return err(format!("unknown gs primitive: {tok:?}")),
        }
    };
    Ok(InstructionKind::DclGsInputPrimitive { primitive })
}

fn parse_gs_topology(tok: &str) -> Result<GsOutputTopology, AsmError> {
    use GsOutputTopology::{
        LineList, LineListAdj, LineStrip, LineStripAdj, PointList, TriangleList, TriangleListAdj,
        TriangleStrip, TriangleStripAdj, Undefined,
    };
    Ok(match tok {
        "undefined" => Undefined,
        "pointlist" => PointList,
        "linelist" => LineList,
        "linestrip" => LineStrip,
        "trianglelist" => TriangleList,
        "trianglestrip" => TriangleStrip,
        "linelistAdj" => LineListAdj,
        "linestripAdj" => LineStripAdj,
        "trianglelistAdj" => TriangleListAdj,
        "trianglestripAdj" => TriangleStripAdj,
        _ => return err(format!("unknown gs topology: {tok:?}")),
    })
}

/// Parse an `rt=float,float,float,float` tag into the four resource return types.
fn parse_rt_tag(c: &mut Cursor<'_>) -> Result<[ReturnType; 4], AsmError> {
    let inner = expect_tag(c, "rt")?;
    let mut rts = [ReturnType::Unknown(0); 4];
    let mut it = inner.split(',');
    for slot in &mut rts {
        let name = it.next().ok_or_else(|| AsmError {
            message: format!("rt= needs four types: {inner:?}"),
        })?;
        *slot = parse_return_type(name)?;
    }
    Ok(rts)
}

fn parse_return_type(name: &str) -> Result<ReturnType, AsmError> {
    use ReturnType::{Continued, Double, Float, Mixed, Sint, Snorm, Uint, Unknown, Unorm, Unused};
    Ok(match name {
        "unorm" => Unorm,
        "snorm" => Snorm,
        "sint" => Sint,
        "uint" => Uint,
        "float" => Float,
        "mixed" => Mixed,
        "double" => Double,
        "continued" => Continued,
        "unused" => Unused,
        other => match other.strip_prefix("unknown") {
            Some(v) => Unknown(parse_dec(v)?),
            None => return err(format!("unknown return type: {name:?}")),
        },
    })
}

/// Reconstruct a resource-dimension extended token (type 2) from
/// `<dim>` or `<dim>,stride=<n>`.
fn parse_resource_dim(inner: &str) -> Result<u32, AsmError> {
    let (dim_name, stride) = match inner.split_once(',') {
        Some((d, s)) => {
            let s = s.strip_prefix("stride=").ok_or_else(|| AsmError {
                message: format!("expected stride= in res(...): {inner:?}"),
            })?;
            (d, parse_dec::<u32>(s)?)
        }
        None => (inner, 0),
    };
    let dim = value_of(DIMENSIONS, dim_name).ok_or_else(|| AsmError {
        message: format!("unknown resource dimension: {dim_name:?}"),
    })?;
    Ok(2 | (dim << 6) | (stride << 11))
}

/// Reconstruct a resource-return-type extended token (type 3) from four names.
fn parse_resource_rt(inner: &str) -> Result<u32, AsmError> {
    let mut parts = inner.split(',');
    let mut vals = [0u32; 4];
    for (i, slot) in vals.iter_mut().enumerate() {
        let name = parts.next().ok_or_else(|| AsmError {
            message: format!("rt(...) needs 4 return types, component {i} missing"),
        })?;
        *slot = parse_return_type(name)?.to_u32();
    }
    Ok(3 | (vals[0] << 6) | (vals[1] << 10) | (vals[2] << 14) | (vals[3] << 18))
}

fn parse_custom_data(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    c.skip_spaces();
    let tag = c.take_token();
    match tag {
        "icb" => {
            c.skip_spaces();
            if !c.eat_byte(b'{') {
                return err("expected '{' for icb");
            }
            let mut values: Vec<[f32; 4]> = Vec::new();
            loop {
                c.skip_spaces();
                if c.eat_byte(b'}') {
                    break;
                }
                if !values.is_empty() {
                    if !c.eat_byte(b',') {
                        return err("expected ',' between icb rows");
                    }
                    c.skip_spaces();
                }
                let mut row = [0.0f32; 4];
                for slot in &mut row {
                    c.skip_spaces();
                    let tok = c.take_while(|b| b != b' ' && b != b',' && b != b'}');
                    *slot = f32::from_bits(parse_hex0x(tok)?);
                }
                values.push(row);
            }
            let raw_dword_count = 2 + values.len() * 4;
            Ok(InstructionKind::CustomData {
                subtype: CustomDataType::ImmediateConstantBuffer,
                values,
                raw: Vec::new(),
                raw_dword_count,
            })
        }
        other => {
            let subtype = match other {
                "comment" => CustomDataType::Comment,
                "debuginfo" => CustomDataType::DebugInfo,
                "opaque" => CustomDataType::Opaque,
                s => match s.strip_prefix("other(").and_then(|x| x.strip_suffix(')')) {
                    Some(v) => CustomDataType::Other(parse_dec(v)?),
                    None => return err(format!("unknown customdata subtype: {s:?}")),
                },
            };
            let raw = parse_raw_block(c)?;
            let raw_dword_count = 2 + raw.len();
            Ok(InstructionKind::CustomData {
                subtype,
                values: Vec::new(),
                raw,
                raw_dword_count,
            })
        }
    }
}

/// Parse a ` { 0x.. 0x.. ... }` raw-dword block (non-ICB customdata payload).
fn parse_raw_block(c: &mut Cursor<'_>) -> Result<Vec<u32>, AsmError> {
    c.skip_spaces();
    if !c.eat_byte(b'{') {
        return err("expected '{' for raw payload block");
    }
    let mut raw = Vec::new();
    loop {
        c.skip_spaces();
        if c.eat_byte(b'}') {
            break;
        }
        let tok = c.take_while(|b| b != b' ' && b != b'}');
        raw.push(parse_hex0x(tok)?);
    }
    Ok(raw)
}

// Operands

fn parse_operand_list(c: &mut Cursor<'_>) -> Result<Operands, AsmError> {
    // Operands are whitespace-separated and each is a single whitespace-free
    // token (immediates use `,` internally, relative indices use `+`).
    let mut operands: Operands = SmallVec::new();
    loop {
        c.skip_spaces();
        if c.eof() {
            break;
        }
        operands.push(parse_operand(c)?);
    }
    Ok(operands)
}

fn parse_generic(c: &mut Cursor<'_>) -> Result<InstructionKind, AsmError> {
    let mut operands: Operands = SmallVec::new();
    loop {
        c.skip_spaces();
        if c.eof() || c.looking_at("trailing=") {
            break;
        }
        operands.push(parse_operand(c)?);
    }

    let mut trailing_tokens: SmallU32Vec = SmallVec::new();
    c.skip_spaces();
    if let Some(raw) = c.try_tag("trailing") {
        if raw.is_empty() {
            return err("trailing= requires at least one DWORD");
        }
        for token in raw.split(',') {
            trailing_tokens.push(parse_hex0x(token)?);
        }
    }
    c.skip_spaces();
    if !c.eof() {
        return err(format!(
            "unexpected generic-instruction text: {:?}",
            c.rest()
        ));
    }

    Ok(InstructionKind::Generic {
        operands,
        trailing_tokens,
    })
}

/// Parse a single space-prefixed leading operand (declaration operand).
fn parse_leading_operand(c: &mut Cursor<'_>) -> Result<Operand, AsmError> {
    c.skip_spaces();
    parse_operand(c)
}

fn parse_operand(c: &mut Cursor<'_>) -> Result<Operand, AsmError> {
    let negate = c.eat_byte(b'-');
    let abs = c.eat_byte(b'|');

    let (reg_type, indices, immediate_values) = if c.looking_at("l(") {
        c.eat_str("l(");
        let imms = parse_imm_list(c)?;
        (RegisterType::Immediate32, SmallVec::new(), imms)
    } else if c.looking_at("d(") {
        c.eat_str("d(");
        let imms = parse_imm_list(c)?;
        (RegisterType::Immediate64, SmallVec::new(), imms)
    } else {
        let reg_type = parse_register_prefix(c)?;
        let indices = parse_indices(c)?;
        (reg_type, indices, SmallVec::new())
    };

    let mut components = parse_components(c)?;
    // A bare inline immediate carries no written component selection; restore the
    // implied one the decoder produced from the value count: a scalar literal is
    // 1-component, a vector literal is 4-component mask-mode with an empty mask.
    if !immediate_values.is_empty() && matches!(components, ComponentSelect::ZeroComponent) {
        components = if immediate_values.len() == 1 {
            ComponentSelect::OneComponent
        } else {
            ComponentSelect::Mask(0)
        };
    }
    if abs && !c.eat_byte(b'|') {
        return err("expected closing '|' for abs operand");
    }

    let mut min_precision = MinPrecision::Default;
    let mut has_min_precision = false;
    let mut non_uniform = false;
    loop {
        let precision = if c.eat_str("@min16f") {
            Some(MinPrecision::Float16)
        } else if c.eat_str("@min2_8f") {
            Some(MinPrecision::Float2_8)
        } else if c.eat_str("@min16i") {
            Some(MinPrecision::Sint16)
        } else if c.eat_str("@min16u") {
            Some(MinPrecision::Uint16)
        } else if c.eat_str("@minprec") {
            let raw = parse_dec(c.take_while(|byte| byte.is_ascii_digit()))?;
            if raw > 7 {
                return err("minimum-precision wire value must be 0..=7");
            }
            Some(MinPrecision::from_u32(raw))
        } else {
            None
        };
        if let Some(precision) = precision {
            if has_min_precision {
                return err("duplicate minimum-precision operand tag");
            }
            has_min_precision = true;
            min_precision = precision;
            continue;
        }
        if c.eat_str("@nonuniform") {
            if non_uniform {
                return err("duplicate nonuniform operand tag");
            }
            non_uniform = true;
            continue;
        }
        break;
    }

    Ok(Operand {
        reg_type,
        components,
        negate,
        abs,
        min_precision,
        non_uniform,
        indices,
        immediate_values,
    })
}

fn parse_imm_list(c: &mut Cursor<'_>) -> Result<Immediates, AsmError> {
    let mut imms: Immediates = SmallVec::new();
    if c.eat_byte(b')') {
        return Ok(imms);
    }
    loop {
        imms.push(parse_immediate_token(
            c.take_while(|b| b != b',' && b != b')'),
        )?);
        if c.eat_byte(b',') {
            continue;
        }
        if c.eat_byte(b')') {
            break;
        }
        return err("malformed immediate list");
    }
    Ok(imms)
}

/// Parse one immediate token to its raw u32 bits. Disambiguated by syntax:
/// `0x..` = raw hex, contains `.`/`e`/`inf`/`nan` = float bits, else signed int.
fn parse_immediate_token(tok: &str) -> Result<u32, AsmError> {
    if tok.starts_with("0x") {
        return parse_hex0x(tok);
    }
    let looks_float = tok.bytes().any(|b| matches!(b, b'.' | b'e' | b'E'))
        || tok.contains("inf")
        || tok.contains("nan")
        || tok.contains("NaN");
    if looks_float {
        return tok.parse::<f32>().map(f32::to_bits).map_err(|_| AsmError {
            message: format!("bad float immediate: {tok:?}"),
        });
    }
    tok.parse::<i32>()
        .map(|value| u32::from_ne_bytes(value.to_ne_bytes()))
        .map_err(|_| AsmError {
            message: format!("bad immediate: {tok:?}"),
        })
}

fn parse_register_prefix(c: &mut Cursor<'_>) -> Result<RegisterType, AsmError> {
    // Unknown register form `?reg(<value>)` (preserves the raw operand type).
    if c.eat_str("?reg(") {
        let digits = c.take_while(|b| b != b')');
        if !c.eat_byte(b')') {
            return err("expected ')' after ?reg(");
        }
        return Ok(RegisterType::Unknown(parse_dec(digits)?));
    }
    // Longest-prefix match against known register prefixes. Immediate prefixes
    // (`l`/`d`) are included so an indexed immediate like `l0` parses; the
    // `l(...)` / `d(...)` value forms are handled by the caller before this.
    let rest = c.rest();
    let mut best: Option<(RegisterType, usize)> = None;
    for v in 0..=40u32 {
        let rt = RegisterType::from_u32(v);
        let p = rt.prefix();
        if rest.starts_with(p) && best.is_none_or(|(_, l)| p.len() > l) {
            best = Some((rt, p.len()));
        }
    }
    match best {
        Some((rt, len)) => {
            c.advance(len);
            Ok(rt)
        }
        None => err(format!("unknown register prefix at {:?}", c.rest())),
    }
}

fn parse_indices(c: &mut Cursor<'_>) -> Result<Indices, AsmError> {
    let mut indices: Indices = SmallVec::new();
    // Optional unbracketed first immediate index (digits).
    if c.peek().is_some_and(|b| b.is_ascii_digit()) {
        indices.push(parse_number_index(c)?);
    }
    while c.eat_byte(b'[') {
        indices.push(parse_bracket_index(c)?);
        if !c.eat_byte(b']') {
            return err("expected ']' after index");
        }
    }
    Ok(indices)
}

/// Parse a bare numeric index (digits, optional trailing 'L' for Imm64).
fn parse_number_index(c: &mut Cursor<'_>) -> Result<OperandIndex, AsmError> {
    let digits = c.take_while(|b| b.is_ascii_digit());
    if c.eat_byte(b'L') {
        Ok(OperandIndex::Imm64(parse_dec(digits)?))
    } else {
        Ok(OperandIndex::Imm32(parse_dec(digits)?))
    }
}

fn parse_bracket_index(c: &mut Cursor<'_>) -> Result<OperandIndex, AsmError> {
    if c.peek().is_some_and(|b| b.is_ascii_digit()) {
        return parse_number_index(c);
    }
    // Relative: a nested operand, optionally followed by "+imm".
    let sub = parse_operand(c)?;
    if c.eat_byte(b'+') {
        let imm = parse_dec(c.take_while(|b| b != b']'))?;
        Ok(OperandIndex::RelativePlusImm(imm, Box::new(sub)))
    } else {
        Ok(OperandIndex::Relative(Box::new(sub)))
    }
}

fn parse_components(c: &mut Cursor<'_>) -> Result<ComponentSelect, AsmError> {
    // Write-mask: `:<letters>` (e.g. `:xyzw`, `:xy`, or `:` for an empty mask).
    if c.eat_byte(b':') {
        let letters = c.take_while(|b| matches!(b, b'x' | b'y' | b'z' | b'w'));
        let mut mask = 0u8;
        for ch in letters.chars() {
            mask |= 1 << axis_index(ch).unwrap();
        }
        return Ok(ComponentSelect::Mask(mask));
    }
    if !c.eat_byte(b'.') {
        return Ok(ComponentSelect::ZeroComponent);
    }
    if c.eat_byte(b'1') {
        return Ok(ComponentSelect::OneComponent);
    }
    // Scalar (1 letter) or swizzle (4 letters).
    let letters = c.take_while(|b| matches!(b, b'x' | b'y' | b'z' | b'w'));
    match letters.len() {
        1 => Ok(ComponentSelect::Scalar(
            axis_index(letters.as_bytes()[0] as char).unwrap(),
        )),
        4 => {
            let mut s = [0u8; 4];
            for (i, ch) in letters.chars().enumerate() {
                s[i] = axis_index(ch).unwrap();
            }
            Ok(ComponentSelect::Swizzle(s))
        }
        n => err(format!("expected 1 or 4 swizzle components, got {n}")),
    }
}

fn intern_system_value(s: &str) -> Result<SystemValue, AsmError> {
    if let Some(hex) = s.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16)
            .map(SystemValue::from_u32)
            .map_err(|_| AsmError {
                message: format!("invalid raw system value: {s:?}"),
            });
    }
    SystemValue::from_name(s).ok_or_else(|| AsmError {
        message: format!("unknown system value: {s:?}"),
    })
}

// Small helpers

fn one(op: Operand) -> Operands {
    let mut v: Operands = SmallVec::new();
    v.push(op);
    v
}

fn intern_enum(s: &str, table: &[(&'static str, u32)]) -> Result<&'static str, AsmError> {
    intern(table, s).ok_or_else(|| AsmError {
        message: format!("unknown enum value: {s:?}"),
    })
}

fn parse_tag_enum(
    c: &mut Cursor<'_>,
    tag: &str,
    table: &[(&'static str, u32)],
) -> Result<&'static str, AsmError> {
    let inner = expect_tag(c, tag)?;
    intern_enum(inner, table)
}

fn parse_tag_dec<T: core::str::FromStr>(c: &mut Cursor<'_>, tag: &str) -> Result<T, AsmError> {
    let inner = expect_tag(c, tag)?;
    parse_dec(inner)
}

fn parse_tag_hex(c: &mut Cursor<'_>, tag: &str) -> Result<u32, AsmError> {
    let inner = expect_tag(c, tag)?;
    parse_hex0x(inner)
}

/// Skip a leading space then require `tag=value`, returning the value text.
fn expect_tag<'a>(c: &mut Cursor<'a>, tag: &str) -> Result<&'a str, AsmError> {
    c.skip_spaces();
    match c.try_tag(tag) {
        Some(inner) => Ok(inner),
        None => err(format!("expected {tag}=… at {:?}", c.rest())),
    }
}

fn parse_u32_list(c: &mut Cursor<'_>) -> Result<SmallU32Vec, AsmError> {
    let mut list: SmallU32Vec = SmallVec::new();
    if !c.eat_byte(b'{') {
        return err("expected '{' for list");
    }
    if c.eat_byte(b'}') {
        return Ok(list);
    }
    loop {
        list.push(parse_dec(c.take_while(|b| b != b',' && b != b'}'))?);
        if c.eat_str(", ") {
            continue;
        }
        if c.eat_byte(b'}') {
            break;
        }
        return err("malformed list");
    }
    Ok(list)
}

fn parse_trailing_dec<T: core::str::FromStr>(c: &mut Cursor<'_>) -> Result<T, AsmError> {
    c.skip_spaces();
    parse_dec(c.take_rest())
}

fn parse_dec<T: core::str::FromStr>(s: &str) -> Result<T, AsmError> {
    s.parse::<T>().map_err(|_| AsmError {
        message: format!("bad number: {s:?}"),
    })
}

fn next_i8<'a>(it: &mut impl Iterator<Item = &'a str>) -> Result<i8, AsmError> {
    let s = it.next().ok_or_else(|| AsmError {
        message: String::from("missing tex offset component"),
    })?;
    parse_dec(s)
}

fn parse_hex(s: &str) -> Result<u32, AsmError> {
    u32::from_str_radix(s, 16).map_err(|_| AsmError {
        message: format!("bad hex: {s:?}"),
    })
}

fn parse_hex0x(s: &str) -> Result<u32, AsmError> {
    let body = s.strip_prefix("0x").unwrap_or(s);
    parse_hex(body)
}

// Cursor

struct Cursor<'a> {
    s: &'a [u8],
    src: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Cursor {
            s: s.as_bytes(),
            src: s,
            pos: 0,
        }
    }

    fn eof(&self) -> bool {
        self.pos >= self.s.len()
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn looking_at(&self, p: &str) -> bool {
        self.rest().starts_with(p)
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn eat_byte(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_str(&mut self, p: &str) -> bool {
        if self.looking_at(p) {
            self.pos += p.len();
            true
        } else {
            false
        }
    }

    fn skip_spaces(&mut self) {
        while self.peek() == Some(b' ') {
            self.pos += 1;
        }
    }

    fn take_while(&mut self, pred: impl Fn(u8) -> bool) -> &'a str {
        let start = self.pos;
        while self.peek().is_some_and(&pred) {
            self.pos += 1;
        }
        &self.src[start..self.pos]
    }

    /// Take up to the next space (or end).
    fn take_token(&mut self) -> &'a str {
        self.take_while(|b| b != b' ')
    }

    /// Take everything remaining.
    fn take_rest(&mut self) -> &'a str {
        let start = self.pos;
        self.pos = self.s.len();
        &self.src[start..]
    }

    /// If the cursor is at `tag=`, consume `tag=` and return the value token
    /// (everything up to the next whitespace). The value itself is never empty
    /// of meaning here — list values like `float,float,float,float` contain no
    /// spaces by construction.
    fn try_tag(&mut self, tag: &str) -> Option<&'a str> {
        let mut probe = String::from(tag);
        probe.push('=');
        if !self.looking_at(&probe) {
            return None;
        }
        self.advance(probe.len());
        let inner = self.take_while(|b| b != b' ' && b != b'\t');
        Some(inner)
    }
}
