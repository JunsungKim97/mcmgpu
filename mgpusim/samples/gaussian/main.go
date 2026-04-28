package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/rodinia/gaussian"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var dim = flag.Int("dim", 1536, "matrix dimension")

// 13MB
// var dim = flag.Int("dim", 512, "matrix dimension")

// var dim = flag.Int("dim", 128, "matrix dimension")

func main() {
	flag.Parse()
	r := new(runner.Runner).ParseFlag().Init()
	b := gaussian.NewBenchmark(r.GPUDriver)
	b.Dim = *dim
	r.AddBenchmark(b)
	r.Run()
}
