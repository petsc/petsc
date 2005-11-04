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

def createTags(tagfile,dirname,files):
  # error check for each parameter?
  (status,output) = commands.getstatusoutput('cd '+dirname+';etags -a -o '+tagfile+' '+' '.join(files))
  if status:
    raise RuntimeError("Error running etags "+output)
  return

def endsWithSuffix(file,suffixes):
  # returns 1 if any of the suffixes match - else return 0
  for suffix in suffixes:
    if file.endswith(suffix):
      return 1
  return 0

def badWebIndex(dirname,file):
  # checks if the file is bad index.html document [i.e not generated]
  if file != 'index.html':
    return 0
  elif file == 'index.html' and dirname.find('docs/website') >=0:
    return 0
  else:
    return 1

def processDir(tagfile,dirname,names):
  newls = []
  gsfx = ['.py','.c','.F','.h','.tex','.cxx','.hh','makefile']
  hsfx = ['.html']
  bsfx = ['.py.html','.c.html','.F.html','.h.html','.tex.html','.cxx.html','.hh.html','makefile.html']
  for l in names:
    if endsWithSuffix(l,gsfx):
      newls.append(l)
    elif endsWithSuffix(l,hsfx)  and not endsWithSuffix(l,bsfx) and not badWebIndex(dirname,l):
      # if html - and not bad suffix - and not badWebIndex - then add to etags-list
      newls.append(l)
  if newls: createTags(tagfile,dirname,newls)

  # exclude 'docs' but not 'src/docs'
  if dir in names and dirname.find('src') >=0:
    names.remove(exname)
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

