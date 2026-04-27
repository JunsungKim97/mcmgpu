package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/tartan/lulesh"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var numNodeFlag = flag.Int("num-node", 1<<20, "Number of nodes")

// var numNodeFlag = flag.Int("num-node", 1<<14, "Number of nodes")
var iterFlag = flag.Int("iterations", 4, "Nodal update iterations")
var dtFlag = flag.Float64("dt", 1.0e-7, "Delta time")
var uCutFlag = flag.Float64("u-cut", 1.0e-7, "Velocity cutoff")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()
	benchmark := lulesh.NewBenchmark(r.GPUDriver)
	benchmark.NumNode = *numNodeFlag
	benchmark.Iterations = *iterFlag
	benchmark.Deltatime = float32(*dtFlag)
	benchmark.UCut = float32(*uCutFlag)

	r.AddBenchmark(benchmark)
	r.Run()
}
