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
from sys import *
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
	    if os.path.dirname(line).endswith('custom'):
	      print line
	      removedefines = 1
	    else: removedefines = 0
	    line = os.path.basename(line)
	    g.write(line+':^?'+line+'^A,1\n')
	  line = f.readline()
	f.close()
	g.close()
#
#  Appends function functionname from filename to filename.tmp

def getfunctionC(g,filename,functionname):
        import re
	f = open(filename)
        g.write("/* Function "+functionname+"*/\n\n")
	line = f.readline()
	while line:
		for i in split('int double PetscReal PetscScalar PassiveReal PassiveScalar'," "):
                  reg = re.compile('^[ ]*'+i+'[ ]*'+functionname+'[ ]*\(')
                  fl = reg.search(line)
                  if fl:
                        print 'Extracting function', functionname
			while line:
				g.write(line)
				# this is dangerous, have no way to find end of function
				if line[0] == '}':
                                  break
 		                line = f.readline()
 		        line = f.readline()
                        continue
                line = f.readline()
	f.close()

def getfunctionF(filename,functionname):
        functionname = lower(functionname)
	newfile = filename + ".f"
	f = open(filename)
	g = open(newfile,"w")
	line = f.readline()
        line = lower(line)
	while line:
                sline = lstrip(line)
                if sline:
                  if len(sline) >= 11 + len(functionname): 
                     if sline[0:11+len(functionname)] == "subroutine "+functionname:
			while line:
                                sline = lstrip(line)
                                if sline:
				  g.write(line)
                                  if sline[0:4] == "end\n":
	 	                     	 break
 		                line = f.readline()
                                line = lower(line)
		line = f.readline()
                line = lower(line)
	f.close()
        g.close()

def main():

    arg_len = len(argv)
    if not arg_len == 2: 
        print 'Usage:', argv[0], 'tags file name'
        sys.exit()

    addFileNameTags(argv[1])
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

