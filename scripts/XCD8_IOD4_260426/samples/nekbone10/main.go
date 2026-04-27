package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/coral/nekbone10"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var elements = flag.Int("elements", 256, "spectral elements")

// var elements = flag.Int("elements", 128, "spectral elements")
var iterations = flag.Int("iterations", 3, "operator applications")
var lambda = flag.Float64("lambda", 0.1, "mass term coefficient")

func main() {
	flag.Parse()
	r := new(runner.Runner).ParseFlag().Init()
	b := nekbone10.NewBenchmark(r.GPUDriver)
	b.Elements = *elements
	b.Iterations = *iterations
	b.Lambda = float32(*lambda)
	r.AddBenchmark(b)
	r.Run()
}
