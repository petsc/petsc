#!/bin/bash

# Checks for compliance with 
# Rule: 'We want to get rid of Macros'


# Steps:
# - find any line with a #define ...(, where the dots denote arbitrary characters


grep -n -H "#define\s*[a-zA-Z_\-]*(" "$@"


