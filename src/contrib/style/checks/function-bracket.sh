#!/bin/bash

# Checks for compliance with 
# Rule: 'In function declaration the open bracket { should be on the next line, not on the same line as the function name and arguments'

# Steps:
# - get all lines with '{' following a closing parenthesis.
# - exclude all open brackets on their own line
# - exclude all lines with 'if (', 'else {', 'for (), etc.'
# - exclude all lines where some code follows after '{'
# - exclude lines with closing curly braces (e.g. simple one-liner functions)
# - exclude if-conditions that span over multiple lines. Identified by ')) {'

grep -n -H ")\s*{" "$@" \
 | grep -v ".*:\s*{\s*$" \
 | grep -v " if\s\s*(" \
 | grep -v " else\s*{" \
 | grep -v " do\s*{" \
 | grep -v " for\s*(" \
 | grep -v " switch\s*(" \
 | grep -v " while\s*(" \
 | grep -v "extern\s" \
 | grep -v "struct\s" \
 | grep -v "union\s" \
 | grep -v "typedef\s" \
 | grep -v "define\s" \
 | grep -v "[=+\\]" \
 | grep -v " try\s" \
 | grep -v " catch\s*(" \
 | grep -v "||" \
 | grep -v "{[0-9a-zA-Z{\.(\\-]" \
 | grep -v ") {;}" \
 | grep -v "}" \
 | grep -v ")) {" \


