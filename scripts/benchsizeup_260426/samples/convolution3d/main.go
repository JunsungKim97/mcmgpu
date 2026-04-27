package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/polybench/convolution3d"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var niFlag = flag.Int("ni", 1024, "Dunno")
var njFlag = flag.Int("nj", 1024, "Dunno")
var nkFlag = flag.Int("nk", 1024, "Dunno")

// // 1GB
// var niFlag = flag.Int("ni", 512, "Dunno")
// var njFlag = flag.Int("nj", 512, "Dunno")
// var nkFlag = flag.Int("nk", 512, "Dunno")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := convolution3d.NewBenchmark(runner.GPUDriver)
	benchmark.NI = *niFlag
	benchmark.NJ = *njFlag
	benchmark.NK = *nkFlag

	runner.AddBenchmark(benchmark)

	runner.Run()
}
