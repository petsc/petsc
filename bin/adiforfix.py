#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: adiforfix.py,v 1.1 2001/05/03 14:43:46 bsmith Exp bsmith $ 
#
# change python1.5 to whatever is needed on your system to invoke python
#
#  Adds & continuation marker to the end of continued lines
# 
#  Calling sequence: 
#      | adiforfix.py
#
import urllib
import os
import ftplib
import httplib
import sys
from exceptions import *
from sys import *
from string import *

def main():
    line = sys.stdin.readline()

#   replace tab with 6 spaces
    if len(line) > 0:
      if (line[0] == '\t'):
        line = "      "+line[1:]+"\n"

    while line:

#     replace comment indicator with !
      if len(line) > 0:
        if (line[0] != ' ') & (line[0] != '#'):
          line = "!"+line[1:]+"\n"


      nline = sys.stdin.readline()

#     replace tab with 6 spaces
      if len(nline) > 0:
        if (nline[0] == '\t'):
          nline = "      "+nline[1:]+"\n"

#     replace continuation indicator with &
      if len(nline) > 6:
        if (nline[5] != ' ') & (nline[0] == ' '):
          nline = "     &"+nline[6:]+"\n"

      if len(nline) > 6:
          if (nline[5] == '&') & (nline[0] == ' '):
            line = rstrip(line)+"                                            "                      
            line = line[0:72]+"&\n"

      sys.stdout.write(line)
      line = nline
     
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

