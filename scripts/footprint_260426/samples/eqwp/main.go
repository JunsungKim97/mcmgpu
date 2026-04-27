package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/tartan/eqwp"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var nxFlag = flag.Int("nx", 96, "Grid X")
var nyFlag = flag.Int("ny", 96, "Grid Y")
var nzFlag = flag.Int("nz", 96, "Grid Z")

// var nxFlag = flag.Int("nx", 24, "Grid X")
// var nyFlag = flag.Int("ny", 24, "Grid Y")
// var nzFlag = flag.Int("nz", 24, "Grid Z")
var iterFlag = flag.Int("iterations", 3, "Stress-update iterations")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()
	benchmark := eqwp.NewBenchmark(runner.GPUDriver)
	benchmark.NX = *nxFlag
	benchmark.NY = *nyFlag
	benchmark.NZ = *nzFlag
	benchmark.Iterations = *iterFlag

	runner.AddBenchmark(benchmark)
	runner.Run()
}
