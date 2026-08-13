//! Emit a full forensic `.d3dasm` document for every DXBC container in the
//! input files.
//!
//! Each `.bin` may be an archive of many containers; one `.d3dasm` file is
//! written per container, named `<stem>.<NN>.<profile>.d3dasm`. Every file is
//! verified on the way out (assemble must reproduce the original container bytes).
//!
//! Usage: cargo run --release --example `emit_d3dasm` -- <`out_dir`> <files...>

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args
        .next()
        .expect("usage: emit_d3dasm <out_dir> <files...>");
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let mut emitted = 0usize;
    let mut failed = 0usize;

    for arg in args {
        let data = match std::fs::read(&arg) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {arg}: {e}");
                continue;
            }
        };
        let stem = Path::new(&arg)
            .file_stem()
            .map_or_else(|| "shader".into(), |s| s.to_string_lossy().into_owned());

        for (idx, shader) in d3dasm::parse(&data).iter().enumerate() {
            let text = d3dasm::container_doc::serialize(shader);

            // Verify the document reassembles to the original container bytes.
            let start = shader.offset();
            let original = &data[start..start + shader.size() as usize];
            let verified = d3dasm::container_doc::assemble(&text).is_ok_and(|b| b == original);

            let profile = shader.program().map_or_else(
                || "nocode".into(),
                |p| format!("{}_{}_{}", p.shader_type, p.major_version, p.minor_version),
            );
            let name = format!("{stem}.{idx:02}.{profile}.d3dasm");
            let path = Path::new(&out_dir).join(&name);

            if let Err(e) = std::fs::write(&path, &text) {
                eprintln!("write {}: {e}", path.display());
                failed += 1;
            } else if !verified {
                eprintln!("VERIFY FAILED: {name}");
                failed += 1;
            } else {
                emitted += 1;
            }
        }
    }

    println!("{emitted} .d3dasm files written to {out_dir} ({failed} failures)");
    if failed > 0 {
        std::process::exit(1);
    }
}
