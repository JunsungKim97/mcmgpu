#!/bin/bash
cd samples
cd shoc-reduction
./shoc-reduction -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -Size=67108864 -Iterations=2 