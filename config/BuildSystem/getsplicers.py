#!/usr/bin/env python
#
#   This is absolute crap; we really need to parse the impls and process them
#
import user

import os
import sys
import re
import cPickle
import project
import RDict
import commands


def getSplicersDir(splicedimpls,dir,names):
  reg = re.compile('splicer.begin\(([A-Za-z0-9._]*)\)')

  if 'SCCS' in names: del names[names.index('SCCS')]
  if 'BitKeeper' in names: del names[names.index('BitKeeper')]
  if 'docs' in names: del names[names.index('docs')]
  for f in names:
    ext = os.path.splitext(f)[1]
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
  
def getSplicers(directories):
  splicedimpls = {'.c' : {}, '.h' : {}, '.cc' : {}, '.hh' : {}, '.py' : {}, '.m' : {}}

  if not directories: directories = [os.getcwd()]
  for directory in directories:
    os.path.walk(directory,getSplicersDir,splicedimpls)

  f    = open('splicerblocks', 'w')
  cPickle.dump(splicedimpls,f)
  f.close()
    
if __name__ ==  '__main__':
  if len(sys.argv) > 2: sys.exit('Usage: getsplicers.py <directory>')
  getSplicers(sys.argv[1:-1])

