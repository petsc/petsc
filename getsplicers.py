#!/usr/bin/env python
import user

import os
import sys
import re
import cPickle
import project
import RDict
import commands

def getSIDL(splicedimpls,dir,file):
    args     = ' -printer=[ANL.SIDL.PrettyPrinter] '+os.path.join(dir,file)
    
    argsDB   = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
    # would be nice to do this with import but PrettyPrinter only prints to stdout
    projects = argsDB['installedprojects']
    for p in projects:
      if p.getUrl() == 'bk://sidl.bkbits.net/Compiler':
        args = os.path.join(p.getRoot(),'driver','python','scandalDoc.py')+args
    (status,output) = commands.getstatusoutput(args)
    splicedimpls['.sidl'][file] = output

def getSplicersDir(splicedimpls,dir,names):
  reg = re.compile('splicer.begin\(([A-Za-z0-9._]*)\)')

  if 'SCCS' in names: del names[names.index('SCCS')]
  if 'BitKeeper' in names: del names[names.index('BitKeeper')]
  if 'docs' in names: del names[names.index('docs')]
  for f in names:
    ext = os.path.splitext(f)[1]
    if ext == '.sidl':
      getSIDL(splicedimpls,dir,f)
      continue
    if not ext in splicedimpls: continue
    if f == '__init__.py': continue
    if not os.path.isfile(os.path.join(dir,f)): continue
    fd = open(os.path.join(dir,f),'r')
    line = fd.readline()
    while line:
      if not line.find('splicer.begin') == -1:
        fl = reg.search(line)
        name = fl.group(1)

        line = fd.readline()
        body = ''
        while line.find('splicer.end') == -1:
          body = body + line
          line = fd.readline()
        splicedimpls[ext][name] = body

      line = fd.readline()
    fd.close()
  
def getSplicers(directory):
  splicedimpls = {'.c' : {}, '.h' : {}, '.cc' : {}, '.hh' : {}, '.py' : {}, '.m' : {}, '.sidl':{}}

  if not directory: directory = os.getcwd()
  os.path.walk(directory,getSplicersDir,splicedimpls)

  f    = open('splicerblocks', 'w')
  cPickle.dump(splicedimpls,f)
  f.close()
    
if __name__ ==  '__main__':
  if len(sys.argv) > 2: sys.exit('Usage: getsplicers.py <directory>')
  sys.argv.append(None)
  getSplicers(sys.argv[1])

