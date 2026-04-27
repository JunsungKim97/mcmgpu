#!/bin/bash
cd samples
cd shoc-reduction
./shoc-reduction -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Xdiv -Size=67108864 -Iterations=2 