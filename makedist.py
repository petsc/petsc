#!/usr/bin/env python
#!/bin/env python
#
#      Makes the distribution files for all of our components.
#  Run it from the directory above bs directory
#
import urllib
import os
import ftplib
import httplib
from exceptions import *
from sys import *
from string import *
from string import *
from time import *
from urlparse import *
from string   import *
import commands
import re
import sys

#==================================================================================
def main():
  packages = []
  dirs = os.listdir(os.getcwd())
  for d in dirs:
    if os.path.isdir(d):
      print "Tarring "+d

#  Update the source to the SIDL and make sure that NO source files are listed as compiled      
      try: os.unlink(d+"/bsSource.db")
      except: pass
      (status,output) = commands.getstatusoutput("cd "+d+"; make.py sidl")
      if status:
         print "Unable to run Babel in "+d
         print output
         raise RuntimeError,"Unable to run Babel in "+d

      (status,output) = commands.getstatusoutput("tar --exclude-from xclude -zcf /home/ftp/pub/petsc/sidl/"+d+".tar.gz "+d)
      if status:
         print "Unable to tar "+d
         print output
         raise RuntimeError,"Unable to tar "+d
      else:
         if not d == "bs" and not d == "SIDLRuntimeANL" and not d == "gui":
            packages.append("ftp://info.mcs.anl.gov/pub/petsc/sidl/"+d+".tar.gz\n")

  f = open("/home/ftp/pub/petsc/sidl/packages","w")
  f.writelines(packages)
  f.close()
         
            
# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
    main()

