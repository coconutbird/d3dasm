//! Bit-exact folding of integer and raw transport expressions.

use std::collections::BTreeMap;

use cfglib::ir::rtl::{LiftedStatement, ValueShape, VarExpr};
use dxbc::shex::Opcode;

use super::{AluHead, Expr, Instruction, Op, RtlOp, VariableId};

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
        && valid_bits(bits, *shape))
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

    let mut evaluator = Evaluator {
        instruction,
        known,
        use_cursor: 0,
    };
    let bits = evaluator.value(value)?;
    if evaluator.use_cursor != instruction.uses().len() || !valid_bits(&bits, shape) {
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
        || !valid_bits(constant, shape)
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

fn valid_bits(bits: &[u64], shape: ValueShape) -> bool {
    usize::from(shape.lanes).checked_mul(shape.scalar.word_count()) == Some(bits.len())
}

struct Evaluator<'a> {
    instruction: &'a Instruction,
    known: &'a BTreeMap<VariableId, Vec<u64>>,
    use_cursor: usize,
}

impl Evaluator<'_> {
    fn value(&mut self, expression: &Expr) -> Option<Vec<u64>> {
        match expression {
            VarExpr::Read { positions, scalar } => {
                let variable = *self.instruction.uses().get(self.use_cursor)?;
                self.use_cursor += 1;
                let constant = self.known.get(&variable)?;
                let words = scalar.word_count();
                let mut selected = Vec::with_capacity(positions.len().checked_mul(words)?);
                for &position in positions {
                    let start = usize::from(position).checked_mul(words)?;
                    selected.extend_from_slice(constant.get(start..start.checked_add(words)?)?);
                }
                Some(selected)
            }
            VarExpr::Const { bits, shape } => valid_bits(bits, *shape).then(|| bits.clone()),
            VarExpr::Apply {
                operator,
                operands,
                shape,
            } => {
                let operands = operands
                    .iter()
                    .map(|operand| self.value(operand))
                    .collect::<Option<Vec<_>>>()?;
                fold_operator(operator, &operands, *shape)
            }
            VarExpr::Reinterpret { operand, shape } => {
                let bits = self.value(operand)?;
                valid_bits(&bits, *shape).then_some(bits)
            }
            VarExpr::Compose { parts, shape } => {
                let mut bits = Vec::new();
                for part in parts {
                    bits.extend(self.value(part)?);
                }
                valid_bits(&bits, *shape).then_some(bits)
            }
        }
    }
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
    valid_bits(value, shape).then(|| value.clone())
}

fn select(operands: &[Vec<u64>], shape: ValueShape) -> Option<Vec<u64>> {
    let [condition, when_true, when_false] = operands else {
        return None;
    };
    let lanes = usize::from(shape.lanes);
    let words = shape.scalar.word_count();
    if condition.len() != lanes || !valid_bits(when_true, shape) || !valid_bits(when_false, shape) {
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
