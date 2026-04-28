package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/rodinia/srad"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// 안끝남
// var rows = flag.Int("rows", 8192, "image rows")
// var cols = flag.Int("cols", 8192, "image cols")

var rows = flag.Int("rows", 4096, "image rows")
var cols = flag.Int("cols", 4096, "image cols")

// var rows = flag.Int("rows", 256, "image rows")
// var cols = flag.Int("cols", 256, "image cols")

// var rows = flag.Int("rows", 128, "image rows")
// var cols = flag.Int("cols", 128, "image cols")
var iter = flag.Int("iterations", 10, "srad iterations")
var lambda = flag.Float64("lambda", 0.5, "update step")

func main() {
	flag.Parse()
	r := new(runner.Runner).ParseFlag().Init()
	b := srad.NewBenchmark(r.GPUDriver)
	b.Rows = *rows
	b.Cols = *cols
	b.Iterations = *iter
	b.Lambda = float32(*lambda)
	r.AddBenchmark(b)
	r.Run()
}
