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
    if status:
      if output.find('ommand not found') >= 0:
        print '''Unable to locate bk (Bitkeeper) to download BuildSystem; make sure bk is in your path\nor manually copy BuildSystem to $PETSC_DIR/python/BuildSystem from a machine where you do have bk installed and can clone BuildSystem.'''
      elif output.find('Cannot resolve host') >= 0:
        print '''Unable to download BuildSystem. You must be off the network. Connect to the internet and run config/configure.py again'''
      else:
        print '''Unable to download BuildSystem. Please send this message to petsc-maint@mcs.anl.gov'''
      print output
      sys.exit(3);
      
  sys.path.insert(0, os.path.join(pythonDir, 'BuildSystem'))
  sys.path.insert(0, pythonDir)
  import config.framework

  
  framework = config.framework.Framework(sys.argv[1:]+['-configModules=PETSc.Configure']+configure_options, loadArgDB = 0)
  framework.argDB['CPPFLAGS'] = ''
  framework.argDB['LIBS'] = ''
  try:
    framework.configure(out = sys.stdout)
    framework.storeSubstitutions(framework.argDB)
    return 0
  except RuntimeError, e:
    msg = '******* Unable to configure with given options ******* (see configure.log for full details):\n'+str(e)+'\n******************************************************\n'
    se = ''
  except Exception, e:
    msg = '******* CONFIGURATION CRASH **** Please send configure.log to petsc-maint@mcs.anl.gov\n'
    se  = str(e)
    
  print msg
  if hasattr(framework, 'log'):
    import traceback
    framework.log.write(msg+se)
    traceback.print_tb(sys.exc_info()[2], file = framework.log)
    sys.exit(1)

if __name__ == '__main__':
  petsc_configure([])

