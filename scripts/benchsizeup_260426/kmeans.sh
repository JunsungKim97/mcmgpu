#!/bin/bash
cd samples
cd kmeans
./kmeans -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -points=524288 -features=32 -clusters=20 -max-iter=1 