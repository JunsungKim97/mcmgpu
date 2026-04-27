#!/bin/bash
cd samples
cd pagerank
./pagerank -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -node=8192 -sparsity=0.5 -iterations=1 