#!/usr/bin/env python
#!/bin/env python
#
#    Generates etag and ctag (use -noctags to skip generation of ctags) files for PETSc
#    Adds file names to list of tags in a TAGS file
#    Also removes the #define somefunction_ somefunction from the tags list
#
#
#
#   Walks through the PETSc tree generating the TAGS file
#
import os
import re
import sys
from string import *
import subprocess
try:
  from subprocess import check_output
except ImportError:
  def check_output(*popenargs, **kwargs):
    """Implementation from Python-2.7 subprocess.check_output for use with
    Python-2.6 which does not provide check_output.
    """
    if 'stdout' in kwargs:
      raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
      cmd = kwargs.get("args")
      if cmd is None:
        cmd = popenargs[0]
      raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output

DEVNULL = open(os.devnull, 'w')
#
#  Copies structs from filename to filename.tmp

def addFileNameTags(filename):
  removedefines = 0
  f = open(filename, 'rb')
  g = open('TAGS', 'wb')
  line = f.readline()
  while line:
    if not (removedefines and line.startswith(b'#define ')): g.write(line)
    if line.startswith(b'\f'):
      line = f.readline()
      g.write(line)
      line = line[0:line.index(b',')]
      if os.path.dirname(line).endswith(b'custom') and not line.endswith(b'.h'):
        removedefines = 1
      else: removedefines = 0
      line = os.path.basename(line)
      g.write(line + b':^?' + line + b'^A,1\n')
    line = f.readline()
  f.close()
  g.close()
  return

def createTags(flist,etagfile,ctagfile):
  # split up the flist into blocks of 1000 - and call etags on each chunk
  nfiles = len(flist)
  niter  = nfiles//1000
  nrem   = nfiles%1000
  blocks = [i*1000 for i in range(niter+1)]
  if nrem: blocks.append(nfiles)
  for i in range(len(blocks)-1):
    createTagsBlock(flist[blocks[i]:blocks[i+1]],etagfile,ctagfile)
  return

def createTagsBlock(flist,etagfile,ctagfile):
  # error check for each parameter?
  frlist = [os.path.relpath(path,os.getcwd()) for path in flist]

  subprocess.check_call('etags -a -o '+etagfile+' '+' '.join(frlist), shell=True)

  # linux can use '--tag-relative=yes --langmap=c:+.cu'. For others [Mac,bsd] try running ctags in root directory - with relative path to file
  if ctagfile:
    status = subprocess.call('ctags --fields=+l --tag-relative=yes --langmap=c:+.cu  -a -f '+ctagfile+' '+' '.join(frlist), shell=True, stdout=DEVNULL, stderr=subprocess.STDOUT)
    if status:
      status = subprocess.call('/usr/local/bin/ctags -a -f '+ctagfile+' '+' '.join(frlist), shell=True, stdout=DEVNULL, stderr=subprocess.STDOUT)
      if status:
        status = subprocess.call('ctags -a -f '+ctagfile+' '+' '.join(frlist), shell=True, stdout=DEVNULL, stderr=subprocess.STDOUT)
        if status:
          raise RuntimeError("Error running ctags")
  return

def endsWithSuffix(file,suffixes):
  # returns 1 if any of the suffixes match - else return 0
  for suffix in suffixes:
    if file.endswith(suffix):
      return 1
  return 0

def startsWithPrefix(file,prefixes):
  # returns 1 if any of the prefix match - else return 0
  for prefix in prefixes:
    if file.startswith(prefix):
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

def processDir(flist, dirpath, dirnames, filenames):
  newls = []
  gsfx = ['.py','.c','.cu','.F','.F90','.h','.h90','.tex','.cxx','.hh','makefile','.bib','.jl']
  bpfx = ['.#']
  hsfx = ['.html']
  bsfx = ['.py.html','.c.html','.F.html','.h.html','.tex.html','.cxx.html','.hh.html','makefile.html','.gcov.html','.cu.html','.cache.html']
  for l in filenames:
    if endsWithSuffix(l,gsfx) and not startsWithPrefix(l,bpfx):
      newls.append(l)
    elif endsWithSuffix(l,hsfx)  and not endsWithSuffix(l,bsfx) and not badWebIndex(dirpath,l):
      # if html - and not bad suffix - and not badWebIndex - then add to etags-list
      newls.append(l)
  if newls: flist.extend([os.path.join(dirpath,name) for name in newls])

  # exclude 'petsc/docs/' only (and not docs/ in other locations)
  for exname in ['docs']:
    if exname in dirnames and os.path.realpath(dirpath) == os.path.realpath(os.getcwd()):
      dirnames.remove(exname)

  # One-level unique dirs
  for exname in ['.git','.hg','SCCS', 'output', 'BitKeeper', 'externalpackages', 'bilinear', 'ftn-auto','lib','systems']:
    if exname in dirnames:
      dirnames.remove(exname)
  #  Multi-level unique dirs - specify from toplevel
  for exname in ['src/python/PETSc','client/c++','client/c','client/python']:
    for name in dirnames:
      filename=os.path.join(dirpath,name)
      if filename.find(exname) >=0:
        dirnames.remove(name)
  # check for configure generated PETSC_ARCHes
  rmnames=[]
  for name in dirnames:
    if os.path.isdir(os.path.join(dirpath,name,'petsc','conf')):
      rmnames.append(name)
  for rmname in rmnames:
    dirnames.remove(rmname)
  return

def processFiles(dirname,flist):
  # list files that can't be done with global match [as above] with complete paths
  import glob
  files= []
  lists=['petsc/conf/*']

  for glist in lists:
    gfiles = glob.glob(glist)
    for file in gfiles:
      if not (file.endswith('pyc') or file.endswith('/SCCS') or file.endswith('~')):
        files.append(file)
  if files: flist.extend([os.path.join(dirname,name) for name in files])
  return

def main(ctags):
  try: os.unlink('TAGS')
  except: pass
  etagfile = os.path.join(os.getcwd(),'ETAGS')
  if ctags:
    try: os.unlink('CTAGS')
    except: pass
    ctagfile = os.path.join(os.getcwd(),'CTAGS')
  else:
    ctagfile = None
  flist = []
  if os.path.isdir('.git'):
    output = check_output('git ls-files | egrep -v \(^\(systems/\|share/petsc/datafiles/\)\|/output/\|\.\(png\|pdf\|ps\|ppt\|jpg\)$\)', shell=True)
    flist = output.decode(sys.getfilesystemencoding()).splitlines()
  else:
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
      processDir(flist, dirpath, dirnames, filenames)
    processFiles(os.getcwd(),flist)
  createTags(flist,etagfile,ctagfile)
  addFileNameTags(etagfile)
  try: os.unlink('ETAGS')
  except: pass
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
    if (len(sys.argv) > 1 and sys.argv[1] == "-noctags"): ctags = 0
    else: ctags = 1
    main(ctags)

