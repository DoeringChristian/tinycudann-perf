use std::time::{Duration, Instant};

use half::f16;
use hephaestus_jit::{tr, vulkan};
use serde::{Deserialize, Serialize};
use serde_pickle::{to_vec, to_writer};

#[derive(Debug, Serialize, Deserialize)]
struct PickleFile {
    batch_sizes: Vec<usize>,
    throughputs: Vec<f64>,
}

fn main() {
    let width = 64;
    let in_width = 64;
    let out_width = 64;
    let hidden_layers = 2;

    let mut throughputs = vec![];
    let mut batch_sizes = vec![];

    let n_iters = 1000;

    let device = vulkan(0);

    for i in (14..=21).rev() {
        let batch_size = 2usize.pow(i);
        println!("Pass: batch_size = {batch_size}");
        let weights = tr::sized_literal(f16::from_f32(1f32), width * width * (2 + hidden_layers));
        let input = tr::sized_literal(f16::from_f32(1f32), batch_size * in_width);

        input.schedule();
        weights.schedule();
        tr::compile().launch(&device);

        let output = tr::fused_mlp_inference(
            &input,
            &weights,
            width,
            in_width,
            out_width,
            hidden_layers,
            batch_size,
        );
        output.schedule();

        let graph = tr::compile();

        let mut duration = Duration::from_nanos(0);

        for i in 0..n_iters {
            let report = graph.launch(&device).unwrap();
            // duration += report.exec.cpu_duration;
            duration += report
                .exec
                .passes
                .iter()
                .find(|pass| pass.name == "Fused MLP")
                .unwrap()
                .duration;
        }

        // let duration = end - start;

        let throughput = (batch_size * n_iters) as f64 / duration.as_secs_f64();
        println!("\tthroughput = {throughput}");
        println!(
            "\tduration = {duration:?}",
            duration = (duration / n_iters as u32)
        );

        batch_sizes.push(batch_size);
        throughputs.push(throughput);

        // Sleep to cool off gpu
        // std::thread::sleep(Duration::from_secs_f64(10.));
    }

    let pickle = PickleFile {
        batch_sizes,
        throughputs,
    };

    if let Ok(mut file) = std::fs::File::options()
        .write(true)
        .create(true)
        .open("out/hep.pkl")
    {
        serde_pickle::to_writer(&mut file, &pickle, Default::default()).unwrap();
    }
}
