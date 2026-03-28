//! Chunk parsers for every known DXBC chunk type.
//!
//! Each sub-module handles one (or a family of related) chunk FourCCs.
//! [`parse_chunk`] dispatches on the FourCC and returns a typed [`ChunkData`]
//! variant so callers never need to match on raw strings themselves.

pub mod dxil;
pub mod hash;
pub mod ildb;
pub mod ildn;
pub mod lfs0;
pub mod libf;
pub mod libh;
pub mod priv_;
pub mod psv0;
pub mod rdat;
pub mod rdef;
pub mod rts0;
pub mod sfi0;
pub mod signature;
pub mod stat;

use alloc::vec::Vec;
use core::fmt;

use crate::container::DxbcChunk;
use crate::shex::ir::Program;

/// Trait implemented by every chunk parser.
///
/// Provides a uniform `parse(data) -> Option<Self>` interface so all chunk
/// types can be handled identically in the dispatcher.  Returning `None`
/// signals that the data was too short or malformed to extract a valid chunk.
pub trait ChunkParser: Sized + fmt::Debug + fmt::Display {
    /// Attempt to parse a chunk payload.  Returns `None` on invalid data.
    fn parse(data: &[u8]) -> Option<Self>;
}

// Re-export the most commonly used types at the `chunks` level so users
// can write `dxbc::chunks::RootSignature` instead of `dxbc::chunks::rts0::RootSignature`.
pub use dxil::DxilData;
pub use hash::ShaderHash;
pub use ildb::DebugData;
pub use ildn::DebugName;
pub use lfs0::LibraryFunctionSignatures;
pub use libf::LibraryFunction;
pub use libh::LibraryHeader;
pub use priv_::PrivateData;
pub use psv0::PipelineStateValidation;
pub use rdat::RuntimeData;
pub use rdef::ResourceDef;
pub use rts0::RootSignature;
pub use sfi0::ShaderFeatureInfo;
pub use signature::SignatureElement;
pub use stat::ShaderStats;

/// Parsed payload for any known DXBC chunk.
///
/// Adding a new chunk type:
/// 1. Create `chunks/<name>.rs` — struct + `parse_<name>()` returning `Option<T>` + `Display`.
/// 2. Implement [`ChunkParser`] for the struct.
/// 3. Add a variant here.
/// 4. Add a `try_parse!` arm in [`parse_chunk`].
#[derive(Debug)]
pub enum ChunkData<'a> {
    /// RDEF — resource definitions (constant buffers, bindings, creator).
    Rdef(rdef::ResourceDef<'a>),
    /// ISGN / ISG1 — input signature.
    InputSignature(Vec<signature::SignatureElement<'a>>),
    /// OSGN / OSG1 / OSG5 — output signature.
    OutputSignature(Vec<signature::SignatureElement<'a>>),
    /// PCSG / PSG1 — patch constant signature (hull/domain shaders).
    PatchConstantSignature(Vec<signature::SignatureElement<'a>>),
    /// SHEX / SHDR — decoded shader program.
    Shader(Program),
    /// STAT — shader statistics.
    Stats(stat::ShaderStats),
    /// RTS0 — D3D12 root signature.
    RootSignature(rts0::RootSignature),
    /// SFI0 — shader feature info flags.
    FeatureInfo(sfi0::ShaderFeatureInfo),
    /// HASH / XHSH — shader hash.
    Hash(hash::ShaderHash),
    /// ILDN — debug name.
    DebugName(ildn::DebugName),
    /// ILDB — embedded debug data blob.
    DebugData(ildb::DebugData),
    /// PRIV — private / tool-specific data.
    PrivateData(priv_::PrivateData),
    /// DXIL — SM6.0+ shader bytecode (LLVM IR).
    Dxil(dxil::DxilData),
    /// PSV0 — pipeline state validation.
    PipelineStateValidation(psv0::PipelineStateValidation),
    /// RDAT — runtime data tables.
    RuntimeData(rdat::RuntimeData),
    /// LIBF — library function table.
    LibraryFunction(libf::LibraryFunction),
    /// LFS0 — library function signatures.
    LibraryFunctionSignatures(lfs0::LibraryFunctionSignatures),
    /// LIBH — library header.
    LibraryHeader(libh::LibraryHeader),
    /// Unrecognised chunk — preserved so nothing is silently dropped.
    Unknown { fourcc: [u8; 4], data: &'a [u8] },
}

/// Helper: try a `ChunkParser` impl and fall back to `Unknown` on `None`.
macro_rules! try_parse {
    ($data:expr, $fourcc:expr, $variant:ident, $ty:ty) => {
        match <$ty as ChunkParser>::parse($data) {
            Some(v) => ChunkData::$variant(v),
            None => ChunkData::Unknown {
                fourcc: $fourcc,
                data: $data,
            },
        }
    };
}

/// Parse a raw [`DxbcChunk`] into a typed [`ChunkData`].
pub fn parse_chunk<'a>(chunk: &DxbcChunk<'a>) -> ChunkData<'a> {
    let d = chunk.data;
    let cc = chunk.fourcc;
    match chunk.fourcc_str() {
        "RDEF" => match rdef::parse_rdef(d) {
            Some(rd) => ChunkData::Rdef(rd),
            None => ChunkData::Unknown {
                fourcc: cc,
                data: d,
            },
        },
        cc_str @ ("ISGN" | "ISG1") => {
            ChunkData::InputSignature(signature::parse_signature(cc_str, d))
        }
        cc_str @ ("OSGN" | "OSG1" | "OSG5") => {
            ChunkData::OutputSignature(signature::parse_signature(cc_str, d))
        }
        cc_str @ ("PCSG" | "PSG1") => {
            ChunkData::PatchConstantSignature(signature::parse_signature(cc_str, d))
        }
        "SHEX" | "SHDR" => match crate::shex::decode::decode(d) {
            Ok(prog) => ChunkData::Shader(prog),
            Err(_) => ChunkData::Unknown {
                fourcc: cc,
                data: d,
            },
        },

        "RTS0" => try_parse!(d, cc, RootSignature, rts0::RootSignature),
        "STAT" => try_parse!(d, cc, Stats, stat::ShaderStats),
        "SFI0" => try_parse!(d, cc, FeatureInfo, sfi0::ShaderFeatureInfo),
        "HASH" | "XHSH" => try_parse!(d, cc, Hash, hash::ShaderHash),
        "ILDN" => try_parse!(d, cc, DebugName, ildn::DebugName),
        "ILDB" => try_parse!(d, cc, DebugData, ildb::DebugData),
        "PRIV" => try_parse!(d, cc, PrivateData, priv_::PrivateData),
        "DXIL" => try_parse!(d, cc, Dxil, dxil::DxilData),
        "PSV0" => try_parse!(
            d,
            cc,
            PipelineStateValidation,
            psv0::PipelineStateValidation
        ),
        "RDAT" => try_parse!(d, cc, RuntimeData, rdat::RuntimeData),
        "LIBF" => try_parse!(d, cc, LibraryFunction, libf::LibraryFunction),
        "LFS0" => try_parse!(
            d,
            cc,
            LibraryFunctionSignatures,
            lfs0::LibraryFunctionSignatures
        ),
        "LIBH" => try_parse!(d, cc, LibraryHeader, libh::LibraryHeader),

        _ => ChunkData::Unknown {
            fourcc: cc,
            data: d,
        },
    }
}
