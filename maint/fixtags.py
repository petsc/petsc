#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $ 
#
#
#   Adds file names to list of tags in a TAGS file
#   Also remove the #define somefunction_ somefunction from the Fortran custom files
#      from the tags list
#
#
# 
#
#
import os
import re
from exceptions import *
import sys
from string import *

#
#  Copies structs from filename to filename.tmp
    
def addFileNameTags(filename):
	removedefines = 0
	f = open(filename)
	g = open('TAGS','w')
	line = f.readline()
	while line:
	  if not (removedefines and line.startswith('#define ')): g.write(line)
	  if line.startswith('\f'):
	    line = f.readline()
	    g.write(line)
	    line = line[0:line.index(',')]
	    if os.path.dirname(line).endswith('custom') and not line.endswith('.h'):
	      print line
	      removedefines = 1
	    else: removedefines = 0
	    line = os.path.basename(line)
	    g.write(line+':^?'+line+'^A,1\n')
	  line = f.readline()
	f.close()
	g.close()

def main():
    arg_len = len(sys.argv)
    if not arg_len == 2: 
        print 'Usage:', sys.argv[0], 'tags file name'
        sys.exit()

    addFileNameTags(sys.argv[1])
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

