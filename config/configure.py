#!/usr/bin/env python
import os
import sys
import commands


if not hasattr(sys, 'version_info'):
  raise RuntimeError('You must have Python version 2.2 or higher to run configure')

def getarch():
  if os.path.basename(sys.argv[0]).startswith('configure'): return ''
  else: return os.path.basename(sys.argv[0])[:-3]

def petsc_configure(configure_options):
  # use the name of the config/configure_arch.py to determine the arch
  if getarch(): configure_options.append('-PETSC_ARCH='+getarch())
  
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

    msg = 'CONFIGURATION FAILURE (see configure.log for full details):\n'+str(e)+'\n'
    print msg
    if hasattr(framework, 'log'):
      framework.log.write(msg)
      traceback.print_tb(sys.exc_info()[2], file = framework.log)
    sys.exit(1)
  framework.storeSubstitutions(framework.argDB)
  return 0

if __name__ == '__main__':
  for opt in sys.argv[1:]:
    if opt.startswith('--prefix') or opt.startswith('-prefix'):
      print '=====================================================================\nPETSc does NOT support the --prefix options. All installs are done in-place.\nMove your petsc directory to the location you wish it installed, before running configure\n'
      sys.exit(1)
  petsc_configure([])

