//! The `.d3dasm` text grammar.
//!
//! This module owns the **text codecs** for the `.d3dasm` format: the lossless
//! serialize/parse pair for each DXBC chunk. The split of responsibility is:
//!
//! * [`dxbc`] is the *disassembler* — it turns chunk bytes into a typed IR and
//!   back (`decode`/`encode`/`parse_*`), and formats that IR as human-readable
//!   `fxc`-style disassembly (`Display`).
//! * This module is the *grammar* — it renders that same IR as the editable,
//!   round-trippable `.d3dasm` text and parses it back, so a `.d3dasm` document
//!   re-encodes to byte-identical bytecode.
//!
//! The container document layer ([`crate::container_doc`]) stitches these
//! per-chunk codecs together into a whole-file document.
//!
//! See `docs/d3dasm-grammar.md` for the format specification.

pub mod rdef;
pub mod shex;
pub mod signature;
pub mod stat;
