#!/bin/bash

# Checks for compliance with 
# Rule: 'The closing bracket } should always be on its own line' (if, for, while, etc.)

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - get all lines with '}'
# - remove all lines having a '}' only
# - ignore lines with a comment following '}'
# - exclude all lines with 'else'
# - exclude line breaks '\' within macros
# - exclude lines ending with '};' due to array instantiation
# - exclude lines closing the typedef of a struct (e.g. '} MyStruct;')
# - exclude 'while' keyword from do { ... } while  (discuss this!)
# - exclude typedef and enum lines
# - exclude arrays of arrays instantiation

find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "}" \
 | grep -v ".*:\s*}\s*$" \
 | grep -v ".*:\s*}\s*/\*.*\*/" \
 | grep -v "else" \
 | grep -v '\\' \
 | grep -v '};' \
 | grep -v '.*:\s*}\s*[a-zA-Z_]*;' \
 | grep -v '}\s*while' \
 | grep -v 'typedef|enum' \
 | grep -v '{{' \
 | grep -v '},'
