#!/bin/bash
total=64
for (( i=1; i<=$total; i++ ))
do
  tend=`echo 0.25*$i/$total | bc -l`
  echo $tend
  ./ex1adj -pc_type lu -ts_event_tol 1e-10 -tend $tend
done

./ex1fwd -pc_type lu -ts_event_tol 1e-10 -tend 0.25
