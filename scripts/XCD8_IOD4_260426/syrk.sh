#!/bin/bash
cd samples
cd syrk
./syrk -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -max-inst 10000000 -ni=2048 -nj=2048 