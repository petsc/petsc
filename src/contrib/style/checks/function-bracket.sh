#!/bin/bash

# Checks for compliance with 
# Rule: 'In function declaration the open bracket { should be on the next line, not on the same line as the function name and arguments'

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - get all lines with '{'
# - exclude all open brackets on their own line
# - exclude all lines with 'if (', 'else {', 'for (), etc.'

find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "{" \
 | grep -v ".*:\s*{\s*$" \
 | grep -v "if\s*(" \
 | grep -v "else\s*{" \
 | grep -v "do\s*{" \
 | grep -v "for\s*(" \
 | grep -v "switch\s*(" \
 | grep -v "while\s*(" \
 | grep -v "extern\s" \
 | grep -v "struct\s" \
 | grep -v "union\s" \
 | grep -v "typedef\s" \
 | grep -v "define\s" \
 | grep -v "[=+\\]" \
 | grep -v "try\s" \
 | grep -v "catch\s*(" \
 | grep -v "||" \
 | grep -v "{[0-9a-zA-Z{\.(\\-]"


