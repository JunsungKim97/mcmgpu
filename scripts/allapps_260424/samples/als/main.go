package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/tartan/als"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var usersFlag = flag.Int("users", 8192, "Number of users")
var itemsFlag = flag.Int("items", 8192, "Number of items")

// var usersFlag = flag.Int("users", 1024, "Number of users")
// var itemsFlag = flag.Int("items", 1024, "Number of items")
var rankFlag = flag.Int("rank", 16, "Latent factor dimension")
var nnzPerUserFlag = flag.Int("nnz-per-user", 16, "Ratings per user")
var iterFlag = flag.Int("iterations", 4, "ALS alternating iterations")
var lambdaFlag = flag.Float64("lambda", 0.05, "L2 regularization")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()
	benchmark := als.NewBenchmark(r.GPUDriver)
	benchmark.NumUsers = *usersFlag
	benchmark.NumItems = *itemsFlag
	benchmark.Rank = *rankFlag
	benchmark.NNZPerUser = *nnzPerUserFlag
	benchmark.Iterations = *iterFlag
	benchmark.LambdaReg = float32(*lambdaFlag)

	r.AddBenchmark(benchmark)
	r.Run()
}
