#!/bin/bash
cd samples
cd pagerank
./pagerank -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Xdiv -node=8192 -sparsity=0.5 -iterations=1 