use cfglib::ir::rtl::{ScalarType, ValueShape};
use dxbc::shex::Opcode;

use super::{AluHead, fold_alu};

fn head(opcode: Opcode) -> AluHead {
    AluHead {
        opcode,
        saturate: false,
        dest_position: 0,
        bindings: Vec::new(),
        tex_offsets: None,
        resinfo_return_type: None,
    }
}

fn scalar() -> ValueShape {
    ValueShape::scalar(ScalarType::U32)
}

#[test]
fn integer_addition_wraps_at_32_bits() {
    assert_eq!(
        fold_alu(
            &head(Opcode::Iadd),
            &[vec![u64::from(u32::MAX)], vec![1]],
            scalar()
        ),
        Some(vec![0])
    );
}

#[test]
fn signed_shift_masks_the_count_and_extends_the_sign() {
    assert_eq!(
        fold_alu(
            &head(Opcode::Ishr),
            &[vec![0x8000_0000], vec![33]],
            scalar()
        ),
        Some(vec![0xc000_0000])
    );
}

#[test]
fn signed_comparison_returns_the_dx11_all_bits_mask() {
    assert_eq!(
        fold_alu(&head(Opcode::ILt), &[vec![0xffff_ffff], vec![0]], scalar()),
        Some(vec![u64::from(u32::MAX)])
    );
}

#[test]
fn float_arithmetic_is_never_folded() {
    assert_eq!(
        fold_alu(&head(Opcode::Add), &[vec![0], vec![0]], scalar()),
        None
    );
}
