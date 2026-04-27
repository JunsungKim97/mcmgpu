#!/bin/bash
cd samples
cd mis
./mis -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -numNodes=524288 -numItems=1048576 