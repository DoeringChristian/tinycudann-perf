import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
import pickle
import json

if __name__ == "__main__":
    with open("out/hep.pkl", "rb") as f:
        hep = pickle.load(f)

    with open("out/tcnn.json", "rb") as f:
        # tcnn = pickle.load(f)
        tcnn = json.load(f)

    plt.title("Fully Fused MLP Inference")
    plt.plot(hep["batch_sizes"], hep["throughputs"], label="hephaestus")
    plt.plot(tcnn["batch_sizes"], tcnn["throughputs"], label="tiny-cuda-nn")
    plt.xscale("log", base=2)
    plt.xlabel("batch size")
    plt.ylabel("throughput")
    plt.grid(True)
    plt.legend()
    # plt.savefig("out/cmp.svg")
    plt.gcf().savefig("out/plot.svg")
    plt.gcf().savefig("out/plot.pgf", format="pgf")
    plt.show()
