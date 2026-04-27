package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/tartan/hit"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var nxFlag = flag.Int("nx", 256, "Grid size in X")
var nyFlag = flag.Int("ny", 256, "Grid size in Y")
var nzFlag = flag.Int("nz", 256, "Grid size in Z")

// var nxFlag = flag.Int("nx", 32, "Grid size in X")
// var nyFlag = flag.Int("ny", 32, "Grid size in Y")
// var nzFlag = flag.Int("nz", 32, "Grid size in Z")
var iterFlag = flag.Int("iterations", 5, "Number of viscous sub-steps")
var dtFlag = flag.Float64("dt", 0.01, "Time step factor")
var nuFlag = flag.Float64("nu", 0.1, "Viscosity coefficient")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()
	benchmark := hit.NewBenchmark(runner.GPUDriver)
	benchmark.NX = *nxFlag
	benchmark.NY = *nyFlag
	benchmark.NZ = *nzFlag
	benchmark.Iterations = *iterFlag
	benchmark.Dt = float32(*dtFlag)
	benchmark.Nu = float32(*nuFlag)

	runner.AddBenchmark(benchmark)
	runner.Run()
}
