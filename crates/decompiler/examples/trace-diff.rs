//! Differential trace debugging: runs the original shader and a
//! recompiled object under the same seeded world and prints both write
//! traces so the first divergence localizes a translation defect.

use std::fmt::Write as _;

fn semantic_key(name: &str, index: u32) -> u64 {
    let mut key = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.to_ascii_uppercase().bytes() {
        key = (key ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    key.wrapping_add(u64::from(index)) | (1 << 63)
}

fn io_keys(shader: &d3dasm::Shader<'_>) -> decompiler::interp::IoKeys {
    let normalize = |name: &str| -> String {
        let upper = name.to_ascii_uppercase();
        upper
            .strip_prefix("SV_")
            .map_or(upper.clone(), str::to_owned)
    };
    let mut keys = decompiler::interp::IoKeys::default();
    if let Some(signature) = shader.input_signature() {
        for element in &signature.elements {
            keys.inputs.entry(element.register).or_insert_with(|| {
                semantic_key(&normalize(&element.semantic_name), element.semantic_index)
            });
        }
    }
    if let Some(signature) = shader.output_signature() {
        for element in &signature.elements {
            keys.outputs.entry(element.register).or_insert_with(|| {
                semantic_key(&normalize(&element.semantic_name), element.semantic_index)
            });
        }
    }
    keys
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let original_path = arguments
        .next()
        .expect("usage: trace_diff <original> <index> <recompiled> <seed>");
    let index: usize = arguments.next().expect("index").parse().expect("index");
    let recompiled_path = arguments.next().expect("recompiled");
    let seed: u64 = arguments.next().expect("seed").parse().expect("seed");

    let original_bytes = std::fs::read(&original_path).expect("original");
    let original_shaders = d3dasm::parse(&original_bytes);
    let original = &original_shaders[index];
    let recompiled_bytes = std::fs::read(&recompiled_path).expect("recompiled");
    let recompiled_shaders = d3dasm::parse(&recompiled_bytes);
    let recompiled = &recompiled_shaders[0];

    let world = decompiler::interp::World { seed };
    let (_, first) = decompiler::interp::execute_traced(
        original.program().expect("program"),
        world,
        &io_keys(original),
    )
    .expect("original run");
    let (_, second) = decompiler::interp::execute_traced(
        recompiled.program().expect("program"),
        world,
        &io_keys(recompiled),
    )
    .expect("recompiled run");

    println!(
        "original {} writes, recompiled {} writes",
        first.len(),
        second.len()
    );
    let render = |(opcode, mask, lanes): &(dxbc::shex::Opcode, u8, [u32; 4])| -> String {
        let mut text = format!("{:12} ", opcode.name());
        for (lane, value) in lanes.iter().enumerate() {
            if mask & (1 << lane) != 0 {
                let _ = write!(text, "{value:08x} ");
            }
        }
        text
    };
    // Print both streams side by side; alignment drifts, so print all.
    let limit = first.len().max(second.len());
    for row in 0..limit {
        let left = first.get(row).map(&render).unwrap_or_default();
        let right = second.get(row).map(&render).unwrap_or_default();
        println!("{row:4}  {left:44}  | {right}");
    }
}
