#!/usr/bin/env python
import os
import sys
import commands

if not hasattr(sys, 'version_info'):
  raise RuntimeError('You must have Python version 2.2 or higher to run configure')

def petsc_configure(configure_options):
  # Should be run from the toplevel or from ./config
  pythonDir = os.path.abspath(os.path.join('..', 'python'))
  if not os.path.exists(pythonDir):
    pythonDir = os.path.abspath(os.path.join('python'))
    if not os.path.exists(pythonDir):
      raise RuntimeError('Run configure from $PETSC_DIR, not '+os.path.abspath('.'))

  if not os.path.isdir('python/BuildSystem'):
    print '''Could not locate BuildSystem in $PETSC_DIR/python.
    Downloading it using "bk clone bk://sidl.bkbits.net/BuildSystem $PETSC_DIR/python/BuildSystem"'''
    (status,output) = commands.getstatusoutput('bk clone bk://sidl.bkbits.net/BuildSystem python/BuildSystem')

  sys.path.insert(0, os.path.join(pythonDir, 'BuildSystem'))
  sys.path.insert(0, pythonDir)
  import config.framework

  framework = config.framework.Framework(sys.argv[1:]+['-configModules=PETSc.Configure']+configure_options, loadArgDB = 0)
  framework.argDB['CPPFLAGS'] = ''
  framework.argDB['LIBS'] = ''
  try:
    framework.configure(out = sys.stdout)
  except Exception, e:
    import traceback

    msg = 'CONFIGURATION FAILURE:\n'+str(e)+'\n'
    print msg
    framework.log.write(msg)
    traceback.print_tb(sys.exc_info()[2], file = framework.log)
    sys.exit(1)
  framework.storeSubstitutions(framework.argDB)

if __name__ == '__main__':
  petsc_configure([])
