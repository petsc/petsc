#!/bin/bash

# Checks for compliance with 
# Rule: 'The closing bracket } should always be on its own line' (if, for, while, etc.)

# Steps:
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
# - exclude lines using __LINE__ (this is certainly intentional at that spot)
# - exclude preprocessor definitions
# - exclude lines containing ${PETSC_DIR}

grep -n -H "}" "$@" \
 | grep -v ".*:\s*}\s*$" \
 | grep -v ".*:\s*}\s*/\*.*\*/" \
 | grep -v "else" \
 | grep -v '\\' \
 | grep -v '};' \
 | grep -v '.*:\s*}\s*[a-zA-Z_]*;' \
 | grep -v '}\s*while' \
 | grep -v 'typedef\|enum' \
 | grep -v '{{\|}}' \
 | grep -v '},' \
 | grep -v -F "__LINE__" \
 | grep -v -F "#define" \
 | grep -v -F '${PETSC_DIR}' \
 | grep -v -F 'if (' \
 | grep -v -F 'for (' \
 | grep -v -F 'while (' \
 | grep -v -F 'struct {' \
 | grep -v '{\s*;\s*}' \
 | grep -v ':\s*/\*' \
 | grep -v '}\s*\*/' \
 | grep -v -F ' catch' \
 | grep -v -F ':$' \
 | grep -v ') {\s*return.*}' \
 | grep -v '}\s*//' \
 | grep -v '{.*}' \
 | grep -v ':}.*;' \
 | grep -v '//\s*}' \
 | grep -v 'break;}' \
 | grep -v -F '"}"' \


