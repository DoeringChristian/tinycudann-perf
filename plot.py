import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
import pickle

if __name__ == "__main__":
    with open("out/hep.pkl", "rb") as f:
        hep = pickle.load(f)
    with open("out/tcnn.pkl", "rb") as f:
        tcnn = pickle.load(f)

    plt.plot(hep["batch_sizes"], hep["throughputs"], label="hephaestus")
    plt.plot(tcnn["batch_sizes"], tcnn["throughputs"], label="tiny cuda nn")
    plt.xscale("log", base=2)
    plt.xlabel("batch size")
    plt.ylabel("throughput")
    plt.legend()
    plt.savefig("out/cmp.svg")
