//! Extract each raw DXBC container from input files into individual `.cso`
//! blobs, so an external tool (e.g. `fxc /dumpbin`) can consume them.
//!
//! Usage: cargo run --release --example `dump_containers` -- <`out_dir`> <files...>

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args
        .next()
        .expect("usage: dump_containers <out_dir> <files...>");
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    for arg in args {
        let Ok(data) = std::fs::read(&arg) else {
            eprintln!("skip {arg}");
            continue;
        };
        let stem = Path::new(&arg)
            .file_stem()
            .map_or_else(|| "shader".into(), |s| s.to_string_lossy().into_owned());

        for (idx, shader) in d3dasm::parse(&data).iter().enumerate() {
            let start = shader.offset();
            let bytes = &data[start..start + shader.size() as usize];
            let profile = shader.program().map_or_else(
                || "nocode".into(),
                |p| format!("{}_{}_{}", p.shader_type, p.major_version, p.minor_version),
            );
            let name = format!("{stem}.{idx:02}.{profile}.cso");
            let path = Path::new(&out_dir).join(&name);
            std::fs::write(&path, bytes).expect("write cso");
            println!("{} ({} bytes)", path.display(), bytes.len());
        }
    }
}
