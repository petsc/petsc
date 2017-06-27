#!/usr/bin/env bash
# Simple script for killing a process after a given time.
# No error checking done
pid=$1 
TIMEOUT=$2
sleep $TIMEOUT
kill -s PIPE $pid  2>&1 >/dev/null 
kill -s PIPE $$    2>&1 >/dev/null # Script suicide
