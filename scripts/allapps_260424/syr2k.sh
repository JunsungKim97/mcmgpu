#!/bin/bash
cd samples
cd syr2k
./syr2k -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Xdiv -max-inst 30000000 -ni=1024 -nj=1024 