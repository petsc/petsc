#!/usr/bin/env python
'''Creates a .pythonrc.py file and puts the path for BuildSystem in it'''

import os
import sys
import os.path

def setupRC(path):
  fname = os.path.join(os.getenv('HOME'),'.pythonrc.py')
  if os.path.isfile(fname):
    f = open(fname)
    contents = f.read()
    f.close()
  else:
    contents = ''

  contents = contents + '# Code added by BuildSystem/install/setuprc.py\n'
  contents = contents + 'import sys\n'
  contents = contents + 'sys.path.insert(0,"'+path+'")\n'
  # Check that hostname returns something BitKeeper is happy with
  import socket
  hostname = socket.gethostname()
  if len(hostname) > 8 and hostname[0:9] == 'localhost':
    contents = contents + 'os.putenv("BK_HOST","bkneedsname.org")\n'
  elif hostname[-1] == '.':
    contents = contents + 'os.putenv("BK_HOST","'+hostname+'org")\n'
  elif hostname.find('.') == -1:
    contents = contents + 'os.putenv("BK_HOST","'+hostname+'.org")\n'

  f = open(fname,'w')
  f.write(contents)
  f.close()
  sys.path.insert(0, path)
  
if __name__ ==  '__main__':
  setupRC('hi')
  
