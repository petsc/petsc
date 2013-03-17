#!/bin/bash

# Checks for compliance with 
# Rule: 'The following text should be before each function: #undef __FUNCT__; #define __FUNCT__ "Function Name" <br /> Reports accidental uses of __FUNC__ instead of __FUNCT__'
#
# Steps:
# - Check for lines with __FUNC__


grep -n -H "__FUNC__"  "$@"


