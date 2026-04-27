package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/lonestar/mst"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var nodeFlag = flag.Int("nodes", 131072, "Number of graph nodes")
var extraFlag = flag.Int("extra-edges", 524288, "Additional random undirected edges")

// var nodeFlag = flag.Int("nodes", 4096, "Number of graph nodes")
// var extraFlag = flag.Int("extra-edges", 16384, "Additional random undirected edges")
var iterFlag = flag.Int("iterations", 20, "Max Boruvka iterations")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()
	b := mst.NewBenchmark(r.GPUDriver)
	b.NumNodes = *nodeFlag
	b.ExtraEdges = *extraFlag
	b.MaxBoruvkaIter = *iterFlag

	r.AddBenchmark(b)
	r.Run()
}
