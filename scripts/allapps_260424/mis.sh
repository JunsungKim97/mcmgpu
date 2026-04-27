#!/bin/bash
cd samples
cd mis
./mis -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Xdiv -numNodes=524288 -numItems=1048576 