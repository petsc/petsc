#!/usr/bin/env python

import os
import sys
import popen2
import re
  
def portabilityCheck(filename,includes):

  if filename.endswith('.o'):
    # Check for bad symbols in file
    bad = re.compile(r'creat|unlink|fprint|open|socket|pipe|fork|exec|kill|signal|longjmp|system|fscanf|_f_open|_system_|_do_l_out')
    pipe = os.popen("nm " + filename)
    for line in pipe.readlines():
      if bad.search(line):
        print 'For portability avoid direct read, write, or system commands in file ' + filename
        print 'Function: '+ line
        os.unlink(filename)
        return 1
    return 0
  
  else:
    # Check for use of system includes 
    include = re.compile(r"""#\s*include\s*('|"|<)""")
    ok = re.compile(r"""#\s*include\s*('|"|<)petsc""")
    okf = re.compile(r"""#\s*include\s*('|"|<)include/finclude/petsc""")    
    file = open(filename)
    for line in file.readlines():
      if include.search(line) and not ok.search(line) and not okf.search(line):
        found = 0
        for l in includes:
          oki = re.compile(r"""#\s*include\s*('|"|<)"""+l)
          if oki.search(line):
            if portabilityCheck(l,sys.argv[2:]): return 1
            found = 1
        if not found:
          print 'For portability avoid direct use of generic system #include files in ' + filename
          print 'Line: '+ line
          return 1
    file.close()
    return 0

if __name__=="__main__":
  sys.exit(portabilityCheck(sys.argv[1],sys.argv[2:]))

      
