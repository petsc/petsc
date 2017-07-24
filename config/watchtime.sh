#!/usr/bin/env bash
# Simple script for killing a process after a given time.
# No error checking done
pid=$1 
TIMEOUT=$2
sleep $TIMEOUT
if ps -p $pid > /dev/null; then
  kill -13 $pid && wait $pid 2>/dev/null  # Wait used here to capture the kill messagpe
fi
exit
