#!/usr/bin/env python
'''Creates a .pythonrc.py file and puts the path for BuildSystem in it'''

import os
import sys
import os.path

def setupRC(path):
  import user
  if path in sys.path:
    sys.path.remove(path)
  else:
    fname = os.path.join(os.getenv('HOME'),'.pythonrc.py')
    if os.path.isfile(fname):
      f = open(fname)
      contents = f.read()
      f.close()
    else:
      contents = ''

    contents = contents + '# Code added by BuildSystem/setuprc.py\n'
    contents = contents + 'import sys\n'
    contents = contents + 'sys.path.insert(0,"'+path+'")\n'
    f = open(fname,'w')
    f.write(contents)
    f.close()
  sys.path.insert(0, path)
  
if __name__ ==  '__main__':
  setupRC('hi')
  
