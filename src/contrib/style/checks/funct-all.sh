#!/bin/bash

# Checks for compliance with 
# Rule: 'The following text should be before each function: #undef __FUNCT__; #define __FUNCT__ "Function Name" <br /> Reports accidental uses of __FUNC__ instead of __FUNCT__'
#
# Steps:
# - Check src/ folder and call checker script


find src/ -name *.[ch] \
 | xargs $(dirname $0)/funct.sh


