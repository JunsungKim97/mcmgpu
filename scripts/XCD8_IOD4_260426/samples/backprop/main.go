package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/rodinia/backprop"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var inputFlag = flag.Int("input", 8192, "Number of input neurons (multiple of 16)")
var iterFlag = flag.Int("iterations", 2, "Number of training iterations")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()
	b := backprop.NewBenchmark(r.GPUDriver)
	b.InputN = *inputFlag
	b.Iterations = *iterFlag

	r.AddBenchmark(b)
	r.Run()
}
