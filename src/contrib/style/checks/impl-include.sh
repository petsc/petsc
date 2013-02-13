#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use private includes in the public interface. [Work in progress]'

# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with 'isimpl.h>'
# 

find ./src -name "*.[hc]" | xargs grep "isimpl.h>"

