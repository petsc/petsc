#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $ 
#
# change python to whatever is needed on your system to invoke python
#
#  Reads classes.data and prints the information out nicely
#
#  Crude as all hack!
#
#  Calling sequence: 
#      prettyprint.py
##
import os
import re
from exceptions import *
import sys
from string import *
import pickle

# list of classes found
classes = {}
enums = {}


def main(args):
  file = open('classes.data')
  enums   = pickle.load(file)
  classes = pickle.load(file)

  for i in enums:
    print i
    for j in enums[i]:
      print "  "+j
  print " "
  for i in classes:
    print i
    for j in classes[i]:
      print "  "+j+"()"
      for k in classes[i][j]:
        print "    "+k
  
  
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  main(sys.argv[1:])

