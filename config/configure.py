#!/usr/bin/env python
import os
import sys
import commands


if not hasattr(sys, 'version_info') or not sys.version_info[1] >= 2:
  print '********* You must have Python version 2.2 or higher to run configure ***********'
  print '* Python is easy to install for end users or sys-admin. We urge you to upgrade  *'
  print '*                   http://www.python.org/download/                             *'
  print '*                                                                               *'
  print '* You can configure PETSc manually BUT please, please consider upgrading python *'
  print '* http://www.mcs.anl.gov/petsc/petsc-2/documentation/installation.html#Manual   *'
  print '*********************************************************************************'
  sys.exit(4)
  
def getarch():
  if os.path.basename(sys.argv[0]).startswith('configure'): return ''
  else: return os.path.basename(sys.argv[0])[:-3]

def rhl9():
  try:
    file = open('/etc/redhat-release','r')
  except:
    return 0
  try:
    buf = file.read()
    file.close()
  except:
    # can't read file - assume dangerous RHL9
    return 1
  if buf.find('Shrike') > -1:
    return 1
  else:
    return 0

def petsc_configure(configure_options):
  # use the name of the config/configure_arch.py to determine the arch
  if getarch(): configure_options.append('-PETSC_ARCH='+getarch())

  # Disable threads on RHL9
  if rhl9():
    configure_options.append('--useThreads=0')
    print ' *** RHL9 detected. Disabling threads in configure *****'
  
  # Should be run from the toplevel
  pythonDir = os.path.abspath(os.path.join('python'))
  bsDir     = os.path.join(pythonDir, 'BuildSystem')
  if not os.path.isdir(pythonDir):
    raise RuntimeError('Run configure from $PETSC_DIR, not '+os.path.abspath('.'))
  if not os.path.isdir(bsDir):
    print '''++ Could not locate BuildSystem in $PETSC_DIR/python.'''
    print '''++ Downloading it using "bk clone bk://sidl.bkbits.net/BuildSystem $PETSC_DIR/python/BuildSystem"'''
    (status,output) = commands.getstatusoutput('bk clone bk://sidl.bkbits.net/BuildSystem python/BuildSystem')
    if status:
      if output.find('ommand not found') >= 0:
        print '''** Unable to locate bk (Bitkeeper) to download BuildSystem; make sure bk is in your path'''
        print '''** or manually copy BuildSystem to $PETSC_DIR/python/BuildSystem from a machine where'''
        print '''** you do have bk installed and can clone BuildSystem. '''
      elif output.find('Cannot resolve host') >= 0:
        print '''** Unable to download BuildSystem. You must be off the network.'''
        print '''** Connect to the internet and run config/configure.py again.'''
      else:
        print '''** Unable to download BuildSystem. Please send this message to petsc-maint@mcs.anl.gov'''
      print output
      sys.exit(3)
      
  sys.path.insert(0, bsDir)
  sys.path.insert(0, pythonDir)
  import config.framework
  import cPickle
  
  framework = config.framework.Framework(sys.argv[1:]+['-configModules=PETSc.Configure']+configure_options, loadArgDB = 0)
  try:
    framework.configure(out = sys.stdout)
    framework.storeSubstitutions(framework.argDB)
    framework.argDB['configureCache'] = cPickle.dumps(framework)
    return 0
  except RuntimeError, e:
    msg = '***** Unable to configure with given options ***** (see configure.log for full details):\n' \
    +str(e)+'\n******************************************************\n'
    se = ''
  except TypeError, e:
    msg = '***** Error in command line argument to configure.py *****\n' \
    +str(e)+'\n******************************************************\n'
    se = ''
  except ImportError, e:
    msg = '******* Unable to find module for configure.py *******\n' \
    +str(e)+'\n******************************************************\n'
    se = ''
  except SystemExit, e:
    if e.code is None or e.code == 0:
      return
    msg = '*** CONFIGURATION CRASH **** Please send configure.log to petsc-maint@mcs.anl.gov\n'
    se  = str(e)
  except Exception, e:
    msg = '*** CONFIGURATION CRASH **** Please send configure.log to petsc-maint@mcs.anl.gov\n'
    se  = str(e)
    
  print msg
  if hasattr(framework, 'log'):
    import traceback
    framework.log.write(msg+se)
    traceback.print_tb(sys.exc_info()[2], file = framework.log)
    sys.exit(1)

if __name__ == '__main__':
  petsc_configure([])

