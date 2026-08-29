//! SM4/SM5 shader bytecode adapter for `cfglib`'s graph analyses.
//!
//! Provides control-flow and component-granular data-flow adapters for
//! [`dxbc`] shader instructions, enabling CFG construction, SSA
//! conversion, and the rest of `cfglib`'s analyses directly from decoded
//! shader programs. This is a raw analysis surface at instruction
//! granularity; the decompiler itself lifts through the shader-dialect
//! MLIL instead (see [`crate::lift_program`]).
//!
//! # Example
//!
//! ```ignore
//! let program = /* decoded dxbc::shex::ir::Program */;
//! let cfg = decompiler::cfg::build_cfg(&program)?;
//! let dominators = cfglib::DominatorTree::compute(&cfg);
//! let ssa = cfglib::SsaForm::compute(&cfg, &dominators);
//! let memory = cfglib::MemorySSA::compute(&cfg, &decompiler::cfg::Sm4MemoryAlias);
//! println!("{}", cfg.to_dot());
//! ```

use cfglib::{BuildError, Cfg, CfgBuilder};
use dxbc::shex::Program;

mod instruction;

pub use instruction::{
    Sm4Component, Sm4Effect, Sm4Fence, Sm4Index, Sm4Instruction, Sm4MemoryAlias, Sm4MemoryEvent,
    Sm4MemoryLocation, Sm4MemoryRegion, Sm4Register, Sm4RegisterType, Sm4Variable,
};

/// Build a control-flow graph from a decoded shader program.
///
/// # Errors
///
/// Returns an error if the shader contains mismatched structured
/// control-flow instructions (e.g. `else` without `if`).
pub fn build_cfg(program: &Program) -> Result<Cfg<Sm4Instruction>, BuildError> {
    CfgBuilder::build(
        program
            .instructions
            .iter()
            .cloned()
            .map(Sm4Instruction::new),
    )
}
