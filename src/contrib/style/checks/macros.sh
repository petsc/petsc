#!/bin/bash

# Checks for compliance with 
# Rule: 'We want to get rid of Macros'


# Steps:
# - exclude src/docs/ holding the documentation only
# - find any line with a #define ...(, where the dots denote arbitrary characters


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "#define\s*[a-zA-Z_\-]*(" \


