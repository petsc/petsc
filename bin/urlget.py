#!/usr/bin/env python1.5
# $Id: urlget.py,v 1.4 1998/01/14 21:26:35 balay Exp balay $ 
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
import re
import string
import tempfile

arg_len = len(sys.argv)
if arg_len < 2 : 
  print 'Error! Insufficient arguments.'
  print 'Usage:', sys.argv[0], 'urlfilename localfilename'
  sys.exit()

urlfilename   = sys.argv[1]
tmpfilename = ()
try:
  tmpfilename = urllib.urlretrieve(urlfilename)
except:
  print 'Error! Accessing url on the server'
  sys.exit()

tmpfile = open(tmpfilename[0],'r')
filesize = os.lseek(tmpfile.fileno(),0,2)
os.lseek(tmpfile.fileno(),0,0)

if filesize < 2000 :
  print 'Error! Accessing url on the server. bytes-received :',filesize
  sys.exit()

outfilename = tempfile.mktemp()
os.link(tmpfilename[0],outfilename)
os.chmod(outfilename,500)
print outfilename
sys.exit()

