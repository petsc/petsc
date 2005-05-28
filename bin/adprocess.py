#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $ 
#
# change python to whatever is needed on your system to invoke python
#
#  Processes .c or .f files looking for a particular function.
#  Copies that function as well as an struct definitions 
#  into a new file to be processed with adiC
# 
#  Calling sequence: 
#      adprocess.py file1.[cf] functionname functionname2
#
#
#   Bugs: Final line of C function must begin with } on first line
#         Application context must be called AppCtx
#         structs possible format is not completely general
#
import urllib
import os
import ftplib
import httplib
import re
from exceptions import *
from sys import *
from string import *
from parseargs import *

#
#  Copies structs from filename to filename.tmp1
    
def setupfunctionC(filename,g = None):
        import re
        regtypedef  = re.compile('typedef [ ]*struct')
        reginclude  = re.compile('#include [ ]*"([a-zA-Z_0-9]*.h)"')
        regdefine   = re.compile('#define')
        regdefine__ = re.compile('#define [ ]*__FUNCT__')
        regextern   = re.compile('extern')
        regEXTERN   = re.compile('EXTERN')
        regif       = re.compile('#if')
        regendif    = re.compile('#endif')
	f = open(filename)
	if not g:
		newfile = filename + ".tmp1"
		g = open(newfile,"w")
		g.write("#include <math.h>\n")
		g.write("#define PetscMin(a,b) (((a)<(b)) ?  (a) : (b))\n")
	line = f.readline()
	while line:
#                line = lstrip(line)+" "
                fl = regtypedef.search(line)
                if fl:
                        struct = line
			while line:
                                reg = re.compile('^[ ]*}')
                                fl = reg.search(line)
                                if fl:
                                  break
 		                line = f.readline()
                                struct = struct + line
#
#        if this is the AppCtx then replace double and Scalar with passive
#
                        reg = re.compile('^[ ]*}[ ]*AppCtx[ ]*;')
                        fl = reg.search(line)
                        if fl:
                                print "Extracting structure AppCtx"
                                reg = re.compile('\n[ ]*PetscScalar ')
                                struct = reg.sub('\nPassiveScalar ',struct)
                                reg = re.compile('\n[ ]*double ')
                                struct = reg.sub('\nPassiveReal ',struct)
                                reg = re.compile('\n[ ]*PetscReal ')
                                struct = reg.sub('\nPassiveReal ',struct)
                        else:
                                reg = re.compile('^[ ]*}[ ]*')
                                line = reg.sub('',line)
                                reg = re.compile('[ ]*;\n')
                                line = reg.sub('',line)
                                print "Extracting structure "+line
                        g.write(struct)
		# copy over all #define macros
                fl = regdefine.search(line)
                if fl:
		        fl = regdefine__.search(line)
			if not fl:
                                g.write(line)
		# copy over extern statements		
		if regextern.search(line) or regEXTERN.search(line):
                        g.write(line)
			
		if reginclude.search(line):
			fname = reginclude.match(line).group(1)
			if os.path.exists(fname):
				setupfunctionC(fname,g)

                fl = regif.search(line)
                if fl:
                        g.write(line)
                fl = regendif.search(line)
                if fl:
                        g.write(line)

		line = f.readline()
	f.close()
        return g

#
#  Appends function functionname from filename to filename.tmp1

def getfunctionC(g,filename,functionname):
        import re
	f = open(filename)
        g.write("/* Function "+functionname+"*/\n\n")
	line = f.readline()
	while line:
		for i in split('int double PetscReal PetscScalar PassiveReal PassiveScalar PetscErrorCode'," "):
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
    if arg_len < 2: 
        print 'Error! Insufficient arguments.'
        print 'Usage:', argv[0], 'file.[cf] functionname1 functionname2 ...' 
        sys.exit()

    ext = split(argv[1],'.')[-1]
    if ext == "c":
      g = setupfunctionC(argv[1])
      for i in range(2,arg_len):
        getfunctionC(g,argv[1],argv[i])
      g.close()
    else:
      getfunctionF(argv[1],argv[2])

#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

