#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
  sys.path.insert(0, os.path.abspath('python'))
  import config.framework
  framework = config.framework.Framework(sys.argv[1:])
  framework.argDB['CPPFLAGS'] = ''
  framework.argDB['LIBS'] = ''
  framework.configure()
  framework.dumpSubstitutions()
