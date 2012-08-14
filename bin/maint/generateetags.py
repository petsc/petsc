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

def createTags(etagfile,ctagfile,dirname,files):
  import glob
  # error check for each parameter?
  (status,output) = commands.getstatusoutput('cd '+dirname+';etags -a -o '+etagfile+' '+' '.join(files))
  if status:
    raise RuntimeError("Error running etags "+output)

  # linux can use '--tag-relative=yes --langmap=c:+.cu'. For others [Mac,bsd] try running ctags in root directory - with relative path to file
  (status,output) = commands.getstatusoutput('cd '+dirname+';ctags --tag-relative=yes --langmap=c:+.cu  -a -f '+ctagfile+' '+' '.join(files))
  if status:
    files= []
    gfiles = glob.glob(os.path.join(dirname,'*'))
    for file in gfiles:
      if file.endswith('.h') or file.endswith('.c') or file.endswith('.cu') or file.endswith('.F') or file.endswith('.cpp') or file.endswith('.F90'):
        files.append(os.path.realpath(os.path.realpath(file)))
    if files:
      (status,output) = commands.getstatusoutput('ctags -a -f '+ctagfile+' '+' '.join(files))
      if status:
        raise RuntimeError("Error running ctags "+output)
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

def processDir(tagfiles,dirname,names):
  etagfile = tagfiles[0]
  ctagfile = tagfiles[1]
  newls = []
  gsfx = ['.py','.c','.cu','.F','.F90','.h','.h90','.tex','.cxx','.hh','makefile','.bib','.jl']
  hsfx = ['.html']
  bsfx = ['.py.html','.c.html','.F.html','.h.html','.tex.html','.cxx.html','.hh.html','makefile.html','.gcov.html','.cu.html','.cache.html']
  for l in names:
    if endsWithSuffix(l,gsfx):
      newls.append(l)
    elif endsWithSuffix(l,hsfx)  and not endsWithSuffix(l,bsfx) and not badWebIndex(dirname,l):
      # if html - and not bad suffix - and not badWebIndex - then add to etags-list
      newls.append(l)
  if newls: createTags(etagfile,ctagfile,dirname,newls)

  # exclude 'docs' but not 'src/docs'
  for exname in ['docs']:
    if exname in names and dirname.find('src') <0:
      names.remove(exname)
  # One-level unique dirs
  for exname in ['.hg','SCCS', 'output', 'BitKeeper', 'externalpackages', 'bilinear', 'ftn-auto','lib','systems']:
    if exname in names:
      names.remove(exname)
  #  Multi-level unique dirs - specify from toplevel
  for exname in ['src/python/PETSc','client/c++','client/c','client/python','src/docs/website/documentation/changes']:
    for name in names:
      filename=os.path.join(dirname,name)
      if filename.find(exname) >=0:
        names.remove(name)
  # check for configure generated PETSC_ARCHes
  rmnames=[]
  for name in names:
    if os.path.isdir(os.path.join(name,'conf')):
      rmnames.append(name)
  for rmname in rmnames:
    names.remove(rmname)
  return

def processFiles(dirname,etagfile,ctagfile):
  # list files that can't be done with global match [as above] with complete paths
  import glob
  files= []
  lists=['conf/*','src/docs/website/documentation/changes/dev.html']

  for glist in lists:
    gfiles = glob.glob(glist)
    for file in gfiles:
      if not (file.endswith('pyc') or file.endswith('/SCCS') or file.endswith('~')):
        files.append(file)
  if files: createTags(etagfile,ctagfile,dirname,files)
  return

def main():
  try: os.unlink('TAGS')
  except: pass
  try: os.unlink('CTAGS')
  except: pass
  etagfile = os.path.join(os.getcwd(),'ETAGS')
  ctagfile = os.path.join(os.getcwd(),'CTAGS')  
  os.path.walk(os.getcwd(),processDir,[etagfile,ctagfile])
  processFiles(os.getcwd(),etagfile,ctagfile)
  addFileNameTags(etagfile)
  try: os.unlink('ETAGS')
  except: pass
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

