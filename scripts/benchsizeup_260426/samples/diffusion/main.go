package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/tartan/diffusion"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// 안끝남
// var nxFlag = flag.Int("nx", 512, "Grid size in X dimension")
// var nyFlag = flag.Int("ny", 512, "Grid size in Y dimension")
// var nzFlag = flag.Int("nz", 512, "Grid size in Z dimension")

var nxFlag = flag.Int("nx", 256, "Grid size in X dimension")
var nyFlag = flag.Int("ny", 256, "Grid size in Y dimension")
var nzFlag = flag.Int("nz", 256, "Grid size in Z dimension")

// var nxFlag = flag.Int("nx", 64, "Grid size in X dimension")
// var nyFlag = flag.Int("ny", 64, "Grid size in Y dimension")
// var nzFlag = flag.Int("nz", 64, "Grid size in Z dimension")
var iterFlag = flag.Int("iterations", 10, "Number of diffusion iterations")
var alphaFlag = flag.Float64("alpha", 0.1, "Diffusion coefficient")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()
	benchmark := diffusion.NewBenchmark(runner.GPUDriver)
	benchmark.NX = *nxFlag
	benchmark.NY = *nyFlag
	benchmark.NZ = *nzFlag
	benchmark.Iterations = *iterFlag
	benchmark.Alpha = float32(*alphaFlag)

	runner.AddBenchmark(benchmark)
	runner.Run()
}
