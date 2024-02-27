use std::time::Instant;

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

    let n_iters = 100;

    let device = vulkan(0);

    for i in 14..=21 {
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

        let start = Instant::now();
        for i in 0..n_iters {
            let report = graph.launch(&device).unwrap();
        }
        let end = Instant::now();

        let duration = end - start;

        let throughput = (batch_size * n_iters) as f64 / duration.as_secs_f64();
        println!("\tthroughput = {throughput}");
        println!(
            "\tduration = {duration:?}",
            duration = (duration / n_iters as u32)
        );

        batch_sizes.push(batch_size);
        throughputs.push(throughput);
    }

    let pickle = PickleFile {
        batch_sizes,
        throughputs,
    };

    let mut file = std::fs::File::options()
        .write(true)
        .create(true)
        .open("out/hep.pkl")
        .unwrap();

    serde_pickle::to_writer(&mut file, &pickle, Default::default()).unwrap();
}