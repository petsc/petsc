#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use private includes in the public interface. [Work in progress]'

# Steps:
# - find lines with 'isimpl.h>'
# 

grep -n -H "isimpl.h>" "$@"


