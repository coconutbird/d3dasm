//! Bit-exact folding of integer and raw transport expressions.

use std::collections::BTreeMap;

use cfglib::ir::rtl::{LiftedStatement, ValueShape, VarExpr};
use dxbc::shex::Opcode;

use super::{AluHead, Instruction, Op, RtlOp, VariableId};

pub(super) fn operation_constant(operation: &Op) -> Option<Vec<u64>> {
    let LiftedStatement::Assign {
        positions,
        width,
        merges,
        value: VarExpr::Const { bits, shape },
        effects,
    } = operation
    else {
        return None;
    };
    (effects.is_empty()
        && full_assignment(positions, *width, *merges, *shape)
        && shape.holds_words(bits))
    .then(|| bits.clone())
}

pub(super) fn fold_instruction(
    instruction: &Instruction,
    known: &BTreeMap<VariableId, Vec<u64>>,
) -> Option<(VariableId, Vec<u64>)> {
    let [definition] = instruction.defs() else {
        return None;
    };
    let LiftedStatement::Assign {
        positions,
        width,
        merges,
        value,
        effects,
    } = instruction.operation()
    else {
        return None;
    };
    let shape = value.shape();
    if !effects.is_empty() || !full_assignment(positions, *width, *merges, shape) {
        return None;
    }

    // Reads resolve positionally against the instruction's pre-order
    // use list; a fold is only trusted when it consumed every use.
    let uses = instruction.uses();
    let mut cursor = 0usize;
    let mut next_read = || {
        let variable = *uses.get(cursor)?;
        cursor += 1;
        known.get(&variable).cloned()
    };
    let bits = value.fold_constant(&mut next_read, &mut |operator, operands, shape| {
        fold_operator(operator, operands, *shape)
    })?;
    if cursor != uses.len() || !shape.holds_words(&bits) {
        return None;
    }
    Some((*definition, bits))
}

pub(super) fn materialized_operation(instruction: &Instruction, constant: &[u64]) -> Option<Op> {
    let LiftedStatement::Assign {
        positions,
        width,
        merges,
        value,
        effects,
    } = instruction.operation()
    else {
        return None;
    };
    let shape = value.shape();
    if matches!(value, VarExpr::Const { .. })
        || !effects.is_empty()
        || !full_assignment(positions, *width, *merges, shape)
        || !shape.holds_words(constant)
    {
        return None;
    }
    Some(LiftedStatement::Assign {
        positions: positions.clone(),
        width: *width,
        merges: false,
        value: VarExpr::Const {
            bits: constant.to_vec(),
            shape,
        },
        effects: Vec::new(),
    })
}

fn full_assignment(positions: &[u8], width: u8, merges: bool, shape: ValueShape) -> bool {
    !merges
        && shape.lanes == width
        && positions.len() == usize::from(width)
        && positions
            .iter()
            .enumerate()
            .all(|(index, &position)| usize::from(position) == index)
}

fn fold_operator(operator: &RtlOp, operands: &[Vec<u64>], shape: ValueShape) -> Option<Vec<u64>> {
    let RtlOp::Alu(head) = operator else {
        return None;
    };
    fold_alu(head, operands, shape)
}

fn fold_alu(head: &AluHead, operands: &[Vec<u64>], shape: ValueShape) -> Option<Vec<u64>> {
    if head.saturate
        || head.dest_position != 0
        || !head.bindings.is_empty()
        || head.tex_offsets.is_some()
        || head.resinfo_return_type.is_some()
    {
        return None;
    }
    let lanes = usize::from(shape.lanes);
    match head.opcode {
        Opcode::Mov => exact_operand(operands, 0, shape),
        Opcode::Movc => select(operands, shape),
        Opcode::Iadd => binary32(operands, lanes, u32::wrapping_add),
        Opcode::IMad | Opcode::UMad => {
            ternary32(operands, lanes, |a, b, c| a.wrapping_mul(b).wrapping_add(c))
        }
        Opcode::IMin => binary32(operands, lanes, |a, b| unsigned(signed(a).min(signed(b)))),
        Opcode::IMax => binary32(operands, lanes, |a, b| unsigned(signed(a).max(signed(b)))),
        Opcode::UMin => binary32(operands, lanes, u32::min),
        Opcode::UMax => binary32(operands, lanes, u32::max),
        Opcode::INeg => unary32(operands, lanes, |value| 0u32.wrapping_sub(value)),
        Opcode::And => binary32(operands, lanes, |a, b| a & b),
        Opcode::Or => binary32(operands, lanes, |a, b| a | b),
        Opcode::Xor => binary32(operands, lanes, |a, b| a ^ b),
        Opcode::Not => unary32(operands, lanes, |value| !value),
        Opcode::Ishl => binary32(operands, lanes, |value, shift| {
            value.wrapping_shl(shift & 31)
        }),
        Opcode::Ishr => binary32(operands, lanes, |value, shift| {
            unsigned(signed(value) >> (shift & 31))
        }),
        Opcode::Ushr => binary32(operands, lanes, |value, shift| value >> (shift & 31)),
        Opcode::ILt => comparison32(operands, lanes, |a, b| signed(a) < signed(b)),
        Opcode::IGe => comparison32(operands, lanes, |a, b| signed(a) >= signed(b)),
        Opcode::IEq => comparison32(operands, lanes, |a, b| a == b),
        Opcode::INe => comparison32(operands, lanes, |a, b| a != b),
        Opcode::ULt => comparison32(operands, lanes, |a, b| a < b),
        Opcode::UGe => comparison32(operands, lanes, |a, b| a >= b),
        _ => None,
    }
}

fn exact_operand(operands: &[Vec<u64>], index: usize, shape: ValueShape) -> Option<Vec<u64>> {
    let value = operands.get(index)?;
    shape.holds_words(value).then(|| value.clone())
}

fn select(operands: &[Vec<u64>], shape: ValueShape) -> Option<Vec<u64>> {
    let [condition, when_true, when_false] = operands else {
        return None;
    };
    let lanes = usize::from(shape.lanes);
    let words = shape.scalar.word_count();
    if condition.len() != lanes || !shape.holds_words(when_true) || !shape.holds_words(when_false) {
        return None;
    }
    let mut selected = Vec::with_capacity(lanes.checked_mul(words)?);
    for (lane, &condition) in condition.iter().enumerate() {
        u32::try_from(condition).ok()?;
        let source = if condition != 0 {
            when_true
        } else {
            when_false
        };
        let start = lane.checked_mul(words)?;
        selected.extend_from_slice(source.get(start..start.checked_add(words)?)?);
    }
    Some(selected)
}

fn unary32(
    operands: &[Vec<u64>],
    lanes: usize,
    operation: impl Fn(u32) -> u32,
) -> Option<Vec<u64>> {
    let [value] = operands else {
        return None;
    };
    if value.len() != lanes {
        return None;
    }
    value
        .iter()
        .map(|&value| Some(u64::from(operation(u32::try_from(value).ok()?))))
        .collect()
}

fn binary32(
    operands: &[Vec<u64>],
    lanes: usize,
    operation: impl Fn(u32, u32) -> u32,
) -> Option<Vec<u64>> {
    let [left, right] = operands else {
        return None;
    };
    if left.len() != lanes || right.len() != lanes {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(&left, &right)| {
            Some(u64::from(operation(
                u32::try_from(left).ok()?,
                u32::try_from(right).ok()?,
            )))
        })
        .collect()
}

fn ternary32(
    operands: &[Vec<u64>],
    lanes: usize,
    operation: impl Fn(u32, u32, u32) -> u32,
) -> Option<Vec<u64>> {
    let [first, second, third] = operands else {
        return None;
    };
    if first.len() != lanes || second.len() != lanes || third.len() != lanes {
        return None;
    }
    first
        .iter()
        .zip(second)
        .zip(third)
        .map(|((&first, &second), &third)| {
            Some(u64::from(operation(
                u32::try_from(first).ok()?,
                u32::try_from(second).ok()?,
                u32::try_from(third).ok()?,
            )))
        })
        .collect()
}

fn comparison32(
    operands: &[Vec<u64>],
    lanes: usize,
    predicate: impl Fn(u32, u32) -> bool,
) -> Option<Vec<u64>> {
    binary32(operands, lanes, |left, right| {
        if predicate(left, right) { u32::MAX } else { 0 }
    })
}

fn signed(value: u32) -> i32 {
    i32::from_ne_bytes(value.to_ne_bytes())
}

fn unsigned(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

#[cfg(test)]
mod tests;
