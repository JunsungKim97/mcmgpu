package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/pannotia/mis"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// 16MB
// var NumNodes = flag.Int("numNodes", 512, "The number of rows in the input matrix.")
// var NumItems = flag.Int("numItems", 256, "The number of rows in the input matrix.")

var NumNodes = flag.Int("numNodes", 4096, "The number of rows in the input matrix.")
var NumItems = flag.Int("numItems", 2048, "The number of rows in the input matrix.")

// var NumNodes = flag.Int("numNodes", 256, "The number of rows in the input matrix.")
// var NumItems = flag.Int("numItems", 128, "The number of rows in the input matrix.")

// var NumNodes = flag.Int("numNodes", 128, "The number of rows in the input matrix.")
// var NumItems = flag.Int("numItems", 64, "The number of rows in the input matrix.")

// var NumNodes = flag.Int("numNodes", 64, "The number of rows in the input matrix.")
// var NumItems = flag.Int("numItems", 32, "The number of rows in the input matrix.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := mis.NewBenchmark(runner.GPUDriver)
	benchmark.NumNodes = *NumNodes
	benchmark.NumEdges = *NumItems

	runner.AddBenchmark(benchmark)

	runner.Run()
}
