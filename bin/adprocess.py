#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: adprocess.py,v 1.2 2001/05/02 21:11:01 bsmith Exp bsmith $ 
#
# change python1.5 to whatever is needed on your system to invoke python
#
#  Processes .c or .f files looking for a particular function.
#  Copies that function as well as an struct definitions 
#  into a new file to be processed with adiC
# 
#  Calling sequence: 
#      adprocess.py file1.[cf] functionname
#
import urllib
import os
import ftplib
import httplib
from exceptions import *
from sys import *
from string import *

#
#  Copies functionname from first filename to filename.tmp
    
def getfunctionC(filename,functionname):
	newfile = filename + ".tmp"
	f = open(filename)
	g = open(newfile,"w")
        g.write("#include <math.h>\n")
	line = f.readline()
	while line:
                line = lstrip(line)+" "
                if line[0:14] == "typedef struct":
			while line:
				g.write(line)
                                if line[0] == "}":
	 	                	 break
 		                line = f.readline()
                if len(line) >= 4 + len(functionname): 
                   if line[0:4+len(functionname)] == "int "+functionname:
			while line:
				g.write(line)
                                if line[0] == "}":
	 	                	 break
 		                line = f.readline()
		line = f.readline()
	f.close()
        g.close()

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
    from parseargs import *

    arg_len = len(argv)
    if arg_len < 2: 
        print 'Error! Insufficient arguments.'
        print 'Usage:', argv[0], 'file.[cf] functionname1 functionname2 ...' 
        sys.exit()

    ext = split(argv[1],'.')[-1]
    if ext == "c":
      getfunctionC(argv[1],argv[2])
    else:
      getfunctionF(argv[1],argv[2])

#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

