#!/usr/bin/env python
#!/bin/env python
# $Id: adiforfix.py,v 1.3 2001/08/24 16:32:18 bsmith Exp $ 
#
# change python to whatever is needed on your system to invoke python
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
    lines = sys.stdin.readlines()

    n = len(lines)

    for i in range(0,n):
      lines[i]=replace(lines[i],"\t","      ")

    for i in range(0,n):

#     replace tab with 6 spaces
      if len(lines[i]) > 0:
        if (lines[i][0] == '\t'):
          lines[i] = "      "+lines[i][1:]

#     replace comment indicator with !
      if len(lines[i]) > 0:
        if (lines[i][0] == 'c') | (lines[i][0] == 'C') | (lines[i][0] == '*'):
          lines[i] = "!"+lines[i][1:]

#     move number right one position
#      if len(lines[i]) > 0:
#        if (lines[i][0] != ' ') & (lines[i][0] != '!') & (lines[i][0] != '#'):
#          lines[i] = " "+lines[i]

#     replace continuation indicator with &
      if len(lines[i]) > 6:
        if (lines[i][5] != ' ') & (lines[i][5] != '\t') & (lines[i][0] == ' '):
          lines[i] = "     &"+lines[i][6:]

    for i in range(0,n):

      if lines[i][0] == ' ':
#       is next line a continued line
        for j in range(i+1,n):
          if len(lines[j]) > 6:
            if (lines[j][5] == '&') & (lines[j][0] == ' '):
              lines[i] = rstrip(lines[i])+"                                                                       "                      
              lines[i] = lines[i][0:72]+"&\n"
              break
            elif (lines[j][0] == ' '):
              break

#     replace E - with E-
      lines[i]=replace(lines[i],"E - ","E-")

    sys.stdout.writelines(lines)
     
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

