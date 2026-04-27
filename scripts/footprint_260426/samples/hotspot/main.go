package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/rodinia/hotspot"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var rows = flag.Int("rows", 1024, "Number of grid rows")
var cols = flag.Int("cols", 1024, "Number of grid cols")

// var rows = flag.Int("rows", 256, "Number of grid rows")
// var cols = flag.Int("cols", 256, "Number of grid cols")
var iterations = flag.Int("iterations", 60, "Total simulation iterations")
var pyramidHeight = flag.Int("pyramid-height", 1, "Iterations computed per kernel launch")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := hotspot.NewBenchmark(runner.GPUDriver)
	benchmark.GridRows = *rows
	benchmark.GridCols = *cols
	benchmark.Iterations = *iterations
	benchmark.PyramidHeight = *pyramidHeight

	runner.AddBenchmark(benchmark)
	runner.Run()
}
