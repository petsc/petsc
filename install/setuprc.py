#!/usr/bin/env python
'''Creates a .pythonrc.py file and puts the path for BuildSystem in it'''

import os
import sys
import os.path

def setupHostname():
  '''Check that hostname returns something BitKeeper is happy with, and returns any lines to be added to the RC file
     - Set BK_HOST if necessary'''
  import socket

  hostname = socket.gethostname()
  if len(hostname) > 8 and hostname[0:9] == 'localhost':
    return ['os.putenv("BK_HOST", "bkneedsname.org")']
  elif hostname[-1] == '.':
    return ['os.putenv("BK_HOST", "'+hostname+'org")']
  elif hostname.find('.') == -1:
    return ['os.putenv("BK_HOST", "'+hostname+'.org")']
  return []

def setupASESection(lines, path):
  '''Fill in the ASE section of the RC file'''
  top       = []
  bottom    = []
  ase       = []
  foundASE  = 0
  skipASE   = 0
  aseMarker = '###### ASE Section'

  for line in lines:
    if line.strip() == aseMarker:
      foundASE = 1
      skipASE  = not skipASE
      continue
    if skipASE:
      continue
    if foundASE:
      bottom.append(line)
    else:
      top.append(line)

  ase.append(aseMarker)
  ase.extend(['# Code added by sidl/BuildSystem/install/setuprc.py', 'import sys', 'sys.path.insert(0,"'+path+'")'])
  ase.extend(setupHostname())
  ase.append(aseMarker)

  return top+ase+bottom

def setupRC(path):
  filename = os.path.join(os.getenv('HOME'),'.pythonrc.py')
  if os.path.isfile(filename):
    f     = open(filename)
    lines = f.readlines()
    f.close()
  else:
    lines = []

  f = open(filename,'w')
  f.write('\n'.join(setupASESection(lines, path)))
  f.close()
  sys.path.insert(0, path)
  return

if __name__ ==  '__main__':
  import sys
  if len(sys.argv) < 2:
    sys.exit('Usage: setupRC.py <BuildSystem path>')
  setupRC(sys.argv[1])
  
