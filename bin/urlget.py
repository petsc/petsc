#!/usr/bin/env python1.5
# $Id: urlget.py,v 1.1 1998/01/14 16:23:22 balay Exp balay $ 
#
#  Retrieves a single file specified as a url and copy it to the specified filename
# 
#  Calling sequence: 
#      urlget.py ftp://hostname/directoryname/file 
#
#
import urllib
import sys
import os

arg_len = len(sys.argv)
if arg_len < 2 : 
  print 'Error! Insufficient arguments.'
  print 'Usage:', sys.argv[0], 'urlfilename localfilename'
  sys.exit()

urlfilename   = sys.argv[1]
tmpfile = ()
try:
  tmpfile = urllib.urlretrieve(urlfilename)
except:
  print 'Error!', sys.exc_type, sys.exc_value
  print 'Incorrect url specified.'
  sys.exit()
urllib.urlcleanup()
print tmpfile[0]
sys.exit()