package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/rodinia/lud"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var dim = flag.Int("dim", 512, "matrix dimension")

// var dim = flag.Int("dim", 128, "matrix dimension")

func main() {
	flag.Parse()
	r := new(runner.Runner).ParseFlag().Init()
	b := lud.NewBenchmark(r.GPUDriver)
	b.Dim = *dim
	r.AddBenchmark(b)
	r.Run()
}
