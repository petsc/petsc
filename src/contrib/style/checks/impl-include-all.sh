#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use private includes in the public interface. [Work in progress]'

# Steps:
# - Run over all files in src/ and run script
# 

find ./src -name "*.[hc]" | xargs $(dirname $0)/impl-include.sh

