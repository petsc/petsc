#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: urlget.py,v 1.20 1999/01/14 23:38:48 balay Exp $ 
#
# Creates the index for the manualpages, ordering
# the indices into sections based on the 'Level of Difficulty'
#
#  Usage:
#    wwwindex.py PETSC_DIR
#
import os
import posixpath
from exceptions import *
from sys import *
from string import *

def main():
      arg_len = len(argv)
      
      if arg_len < 2: 
            print 'Error! Insufficient arguments.'
            print 'Usage:', argv[0], 'PETSC_DIR'
            exit()

      PETSC_DIR = argv[1]
    
      fd = os.popen('ls -d '+ PETSC_DIR + '/docs/manualpages/man*')
      buf = fd.read()

      # eliminate man* files that are not dirs 
      mandirs = []
      for filename in split(strip(buf),'\n'):
            if posixpath.isdir(filename):
                  mandirs.append(filename)

      for dirname in mandirs:
            fd = os.popen('ls '+ dirname + '/*.html')
            buf = fd.read()
            for filename in split(strip(buf),'\n'):
                  print filename
    

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
    
