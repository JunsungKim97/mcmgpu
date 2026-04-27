package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/dnn/relu"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// var numData = flag.Int("length", 4194304, "The number of samples to filter.")

var numData = flag.Int("length", 16777216, "The number of samples to filter.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := relu.NewBenchmark(runner.GPUDriver)
	benchmark.Length = *numData

	runner.AddBenchmark(benchmark)

	runner.Run()
}
