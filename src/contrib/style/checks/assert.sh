#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use assert().'


# Steps:
# - find any line with assert


grep -n -H "assert *(" "$@"

