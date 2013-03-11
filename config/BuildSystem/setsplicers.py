#!/usr/bin/env python
#
#   This is absolute crap; we really need to parse the impls and process them
#
import user

import os
import sys
import re
import cPickle
import string

def setSplicersDir(splicedimpls,dir,names):

  reg        = re.compile('splicer.begin\(([A-Za-z0-9._]*)\)')
  reginclude = re.compile('#include [ ]*"([a-zA-Z_0-9/]*.[h]*)"')

  if 'SCCS' in names: del names[names.index('SCCS')]
  if 'BitKeeper' in names: del names[names.index('BitKeeper')]
  if 'docs' in names: del names[names.index('docs')]
  for f in names:
    ext = os.path.splitext(f)[1]
    if not ext in splicedimpls: continue
    if f == '__init__.py': continue
    if not os.path.isfile(os.path.join(dir,f)): continue
    fd = open(os.path.join(dir,f),'r')
    foundreplacement = 0
    text = ''
    line = fd.readline()
    while line:
      text = text+line
      if not line.find('splicer.begin') == -1:
        fl = reg.search(line)
        name = fl.group(1)

        line = fd.readline()
        body = ''
        while line.find('splicer.end') == -1:
          body = body + line
          line = fd.readline()

            
        # replace body with saved splicer block
        if name.endswith('._includes') and ext == '.cc':
          foundreplacement = 1
#          print 'handling includes for class '+name
          name = name[0:-10]
          len1  = len(name)
          body = '#include "SIDL.hh"\n'
          for n in splicedimpls[ext]:
            if n.startswith(name) and n.endswith('._includes') and n[len1+1:-10].find('.') == -1:
#              print '   '+n
              body = body + splicedimpls[ext][n]
        elif name in splicedimpls[ext]:
          foundreplacement = 1
#          print 'Replacing -------'+name
#          print body
#          print 'with ------------'
#          print splicedimpls[ext][name]
          body = splicedimpls[ext][name]
        else:
#          print 'Cannot find splicer block '+name+' '+f+' ext '+ext
          pass
          
#         convert ASE directory hierarchy of includes
        nb = ''
        for l in body.split('\n'):
          if reginclude.search(l):
            fname    = reginclude.match(l).group(1)
            (fn,extmp) = os.path.splitext(fname)
            fn = fn.split('/')
            if len(fn) > 1 and fn[-1] == fn[-2]:
              t = '#include "'+string.join(fn[0:-1],'_')+'.hh"'
              nb = nb + t + '\n'
            else:
              nb = nb + l + '\n'              
          else:
            nb = nb + l + '\n'              
          
        text = text+nb
        text = text+line
      line = fd.readline()
    fd.close()

    if foundreplacement:
#      print 'Replaced blocks in '+os.path.join(dir,f)
      fd = open(os.path.join(dir,f),'w')
      fd.write(text)
      fd.close()

#    print text
  
def setSplicers(directory):

  f    = open('splicerblocks', 'r')
  splicedimpls = cPickle.load(f)
  f.close()

  # change SIDL.Args and SIDL.ProjectState impl names
  replaces =  {'SIDL.Args':'SIDLASE.Args','SIDL.ProjectState':'SIDLASE.ProjectState'}
  for i in splicedimpls:
    sillytmp = splicedimpls[i]
    for j in sillytmp:
      for k in replaces:
        if not string.find(j,k) == -1:
          newname = j.replace(k,replaces[k])
#          print 'Converting '+j+' to '+newname+' ext '+i
          splicedimpls[i][newname] = splicedimpls[i][j]
          del splicedimpls[i][j]

  
  regset    = re.compile('\.set\(([->< a-zA-Z_0-9/.\(\)\[\]&+*]*),([->< a-zA-Z_0-9/.\(\)\[\]&+*]*)\)[ ]*;')
  regcreate = re.compile('\.create\(([->< a-zA-Z_0-9/.\(\)\[\]&+*]*),([->< a-zA-Z_0-9/.\(\)\[\]&+*]*),([->< a-zA-Z_0-9/.\(\)\[\]&+*]*)\)[ ]*;')
  replaces =  {'SIDL/Args':'SIDLASE/Args',    'SIDL/ProjectState':'SIDLASE/ProjectState',
               'SIDL::Args':'SIDLASE::Args',  'SIDL::ProjectState':'SIDLASE::ProjectState',
               '.dim(':'.dimen(',             '.destroy(':'.deleteRef(',
               '.setMessage(':'.setNote(',    '.getMessage(':'getNote(',
               '.isInstanceOf(':'.isType(', ' IDENT':' MPIB::IDENT',
               ' SIMILAR':' MPIB::SIMILAR',    ' CONGRUENT':' MPIB::CONGRUENT',
               '__enum':''}
  for i in splicedimpls:
    for j in splicedimpls[i]:
      if regset.search(splicedimpls[i][j]):
        splicedimpls[i][j] = regset.sub('.set(\\2,\\1);',splicedimpls[i][j])
      if regcreate.search(splicedimpls[i][j]):
        splicedimpls[i][j] = regcreate.sub('.createRow(\\1,\\2,\\3);',splicedimpls[i][j])
      for k in replaces:    
        splicedimpls[i][j] = splicedimpls[i][j].replace(k,replaces[k])
  
  if not directory: directory = os.getcwd()
  os.path.walk(directory,setSplicersDir,splicedimpls)

    
if __name__ ==  '__main__':
  if len(sys.argv) > 2: sys.exit('Usage: getsplicers.py <directory>')
  sys.argv.append(None)
  setSplicers(sys.argv[1])

