package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/amdappsdk/matrixtranspose"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// 32MB
// var dataWidth = flag.Int("width", 2048, "The dimension of the square matrix.")

var dataWidth = flag.Int("width", 8192, "The dimension of the square matrix.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := matrixtranspose.NewBenchmark(runner.GPUDriver)
	benchmark.Width = *dataWidth

	runner.AddBenchmark(benchmark)

	runner.Run()
}
