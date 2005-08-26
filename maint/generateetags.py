#!/usr/bin/env python
#!/bin/env python
#
#    Generates etag files for PETSc
#    Adds file names to list of tags in a TAGS file
#    Also removes the #define somefunction_ somefunction from the tags list
#
#
# 
#   Walks through the PETSc tree generating the TAGS file
#
import os
import re
from exceptions import *
import sys
from string import *
import commands

#
#  Copies structs from filename to filename.tmp
    
def addFileNameTags(filename):
  removedefines = 0
  f = open(filename)
  g = open('TAGS','w')
  line = f.readline()
  while line:
    if not (removedefines and line.startswith('#define ')): g.write(line)
    if line.startswith('\f'):
      line = f.readline()
      g.write(line)
      line = line[0:line.index(',')]
      if os.path.dirname(line).endswith('custom') and not line.endswith('.h'):
        removedefines = 1
      else: removedefines = 0
      line = os.path.basename(line)
      g.write(line+':^?'+line+'^A,1\n')
    line = f.readline()
  f.close()
  g.close()
  return

def processDir(tagfile,dirname,names):
  newls = []
  for l in names:
    if l.endswith('.py') or l.endswith('.c') or l.endswith('.F') or l.endswith('.h') or l.endswith('.tex') or l == 'makefile':
      newls.append(l)
  if newls:
    (status,output) = commands.getstatusoutput('cd '+dirname+';etags -a -o '+tagfile+' '+' '.join(newls))
    if status:
      raise RuntimeError("Error running etags "+output)

  # One-level unique dirs
  for exname in ['SCCS', 'output', 'BitKeeper', 'externalpackages', 'bilinear', 'ftn-auto','lib']:
    if exname in names:
      names.remove(exname)
  #  Multi-level unique dirs - specify from toplevel
  for exname in ['src/python/PETSc','client/c++','client/c','client/python']:
    for name in names:
      filename=os.path.join(dirname,name)
      if filename.find(exname) >=0:
        names.remove(name)
  return

def main():
  try: os.unlink('TAGS')
  except: pass
  tagfile = os.path.join(os.getcwd(),'ETAGS')
  os.path.walk(os.getcwd(),processDir,tagfile)
  addFileNameTags(tagfile)
  try: os.unlink('ETAGS')
  except: pass
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

