#!/usr/bin/env python

# Checks for compliance with the rule 'Do not put a blank line immediately before PetscFunctionReturn;'

# Steps:
# - Read each file argument to a single string, then run a simple multi-line regular expression on it.
#
# Note: Only file name is printed on match, no details
#


from __future__ import print_function
import sys
import re



for arg in sys.argv[1:]:
    inputfile = open(arg, "r")
    filestring = inputfile.read()
    inputfile.close()

    if re.search("\n\s*\n\s*PetscFunctionReturn", filestring):
        print(arg)




