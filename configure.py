#!/usr/bin/env python
import os
import sys


def petsc_configure(configure_options):
  sys.path.insert(0, os.path.join(os.path.abspath('python'), 'BuildSystem'))
  sys.path.insert(0, os.path.abspath('python'))
  try:
    import config.framework
  except ImportError:
    sys.exit('''Could not locate BuildSystem in $PETSC_DIR/python.
    You can download this package using "bk clone bk://sidl.bkbits.net/BuildSystem $PETSC_DIR/python/BuildSystem"''')
  framework = config.framework.Framework(sys.argv[1:]+['-configModules=PETSc.Configure']+configure_options)
  framework.argDB['CPPFLAGS'] = ''
  framework.argDB['LIBS'] = ''
  framework.configure()
  #framework.dumpSubstitutions()

if __name__ == '__main__':
  petsc_configure([])
