#!/bin/bash
cd samples
cd spmv
./spmv -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -dim=2097152 -sparsity=0.00001 