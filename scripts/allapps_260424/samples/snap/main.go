package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/coral/snap"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// var nx = flag.Int("nx", 256, "cells in x")
// var ny = flag.Int("ny", 128, "cells in y")
// var nz = flag.Int("nz", 128, "cells in z")

var nx = flag.Int("nx", 512, "cells in x")
var ny = flag.Int("ny", 256, "cells in y")
var nz = flag.Int("nz", 256, "cells in z")

// var nx = flag.Int("nx", 64, "cells in x")
// var ny = flag.Int("ny", 32, "cells in y")
// var nz = flag.Int("nz", 32, "cells in z")
var iter = flag.Int("iterations", 4, "sweep iterations")

func main() {
	flag.Parse()
	r := new(runner.Runner).ParseFlag().Init()
	b := snap.NewBenchmark(r.GPUDriver)
	b.NX = *nx
	b.NY = *ny
	b.NZ = *nz
	b.Iterations = *iter
	r.AddBenchmark(b)
	r.Run()
}
