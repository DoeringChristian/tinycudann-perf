import numpy as np
import torch
import tinycudann as tcnn
import time
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    device = "cuda"

    width = 64
    in_width = 64
    out_width = 64
    hidden_layers = 2

    batch_size = 2**20

    network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activations": "None",
        "n_neurons": width,
        "n_hidden_layers": hidden_layers,
    }

    model = tcnn.Network(in_width, out_width, network_config, seed=0)

    for param in model.parameters():
        print(f"{param.shape=}")

    N = 1000

    batch_sizes = [int(2**i) for i in range(14, 22)]

    durations = []
    throughputs = []

    for batch_size in batch_sizes:
        print(f"{batch_size=:_}")

        input = torch.ones([batch_size, in_width], device=device)

        start = time.perf_counter()
        with torch.no_grad():
            for i in range(N):
                output = model(input)

        end = time.perf_counter()
        duration = (end - start) / N
        throughput = batch_size / duration
        flops = width * width * batch_size * 2 / duration

        durations.append(duration)
        throughputs.append(throughput)

        print(f"time: {duration/N}s")
        print(f"throughput: {throughput:.2E}/s")
        print(f"flops: {flops:.2E}/s")

    data = {
        "batch_sizes": batch_sizes,
        "throughputs": throughputs,
    }

    with open("out/tcnn.pkl", "wb") as f:
        pickle.dump(data, f)

    # plt.plot(batch_sizes, throughputs)
    # plt.xscale("log", base=2)
    # plt.show()
