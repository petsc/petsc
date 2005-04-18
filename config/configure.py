#!/usr/bin/env python
import os
import sys
import commands


if not hasattr(sys, 'version_info') or not sys.version_info[1] >= 2:
  print '**** You must have Python version 2.2 or higher to run config/configure.py ******'
  print '*           Python is easy to install for end users or sys-admin.               *'
  print '*                   http://www.python.org/download/                             *'
  print '*                                                                               *'
  print '*            You CANNOT configure PETSc without Python                          *'
  print '*    http://www.mcs.anl.gov/petsc/petsc-as/documentation/installation.html      *'
  print '*********************************************************************************'
  sys.exit(4)
  
def check_petsc_arch(opts):
  # Check for PETSC_ARCH in the following order:
  # 1. command-line (first occurance)
  # 2. specified in configure_options(in script)
  # 3. script name (if not configure.py)

  useName = ''
  for name in opts:
    if name.startswith('-PETSC_ARCH'):
      useName = name
      break
  # look for duplicates - and remove them
  dupnames = []
  if useName:
    for name in opts:
      if name.startswith('-PETSC_ARCH') and name != useName:
        opts.remove(name)
        dupnames.append(name) 
  # print warning for duplicates
  if dupnames:
    print '*********************************************************************************'
    print 'Warning: The following duplicate PETSC_ARCH options are removed:', dupnames
    print 'Warning: Using the option:', useName
    print '*********************************************************************************'
  # If not yet specified - use the filename of script
  if not useName:
      filename = os.path.basename(sys.argv[0])
      if not filename.startswith('configure'):
        useName = '-PETSC_ARCH='+os.path.splitext(os.path.basename(sys.argv[0]))[0]
        opts.append(useName)
  return

def chkcygwin():
  if os.path.exists('/usr/bin/cygcheck.exe'):
    buf = os.popen('/usr/bin/cygcheck.exe -c cygwin').read()
    if buf.find('1.5.11-1') > -1:
      return 1
    else:
      return 0
  return 0
  
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
  print '================================================================================='
  print '             Configuring PETSc to compile on your system                         '
  print '================================================================================='  

  sys.argv += configure_options
  # check PETSC_ARCH
  check_petsc_arch(sys.argv)

  # support a few standard configure option types 
  for l in range(0,len(sys.argv)):
    name = sys.argv[l]
    if name.startswith('--download'):
      sys.argv[l] = name.lower()
    if name.startswith('--enable'):
      sys.argv[l] = name.replace('--enable','--with')
      if name.find('=') == -1: sys.argv[l] += '=1'
    if name.startswith('--disable'):
      sys.argv[l] = name.replace('--disable','--with')
      if name.find('=') == -1: sys.argv[l] += '=0'
      elif name.endswith('=1'): sys.argv[l].replace('=1','=0')
    if name.startswith('--without'):
      sys.argv[l] = name.replace('--without','--with')
      if name.find('=') == -1: sys.argv[l] += '=0'
      elif name.endswith('=1'): sys.argv[l].replace('=1','=0')

  # Disable threads on RHL9
  if rhl9():
    sys.argv.append('--useThreads=0')
    print '================================================================================='
    print ' *** RHL9 detected. Threads do not work correctly with this distribution ***'
    print ' ******** Disabling thread usage for this run of config/configure.py *****'
    print '================================================================================='

  # Check for broken cygwin
  if chkcygwin():
    print '================================================================================='
    print ' *** cygwin-1.5.11-1 detected. config/configure.py fails with this version   ***'
    print ' *** Please upgrade to cygwin-1.5.12-1 or newer version. This can  ***'
    print ' *** be done by running cygwin-setup, selecting "next" all the way.***'
    print '================================================================================='
    sys.exit(3)
          
  # Should be run from the toplevel
  pythonDir = os.path.abspath(os.path.join('python'))
  bsDir     = os.path.join(pythonDir, 'BuildSystem')
  if not os.path.isdir(pythonDir):
    raise RuntimeError('Run configure from $PETSC_DIR, not '+os.path.abspath('.'))
  if not os.path.isdir(bsDir):
    print '================================================================================='
    print '''++ Could not locate BuildSystem in %s/python.''' % os.getcwd()
    print '''++ Downloading it using "bk clone http://sidl.bkbits.net/BuildSystem %s/python/BuildSystem"''' % os.getcwd()
    print '================================================================================='
    (status,output) = commands.getstatusoutput('bk clone http://sidl.bkbits.net/BuildSystem python/BuildSystem')
    if status:
      if output.find('ommand not found') >= 0:
        print '================================================================================='
        print '''** Unable to locate bk (Bitkeeper) to download BuildSystem; make sure bk is in your path'''
        print '''** or manually copy BuildSystem to $PETSC_DIR/python/BuildSystem from a machine where'''
        print '''** you do have bk installed and can clone BuildSystem. '''
        print '================================================================================='
      elif output.find('Cannot resolve host') >= 0:
        print '================================================================================='
        print '''** Unable to download BuildSystem. You must be off the network.'''
        print '''** Connect to the internet and run config/configure.py again.'''
        print '================================================================================='
      else:
        print '================================================================================='
        print '''** Unable to download BuildSystem. Please send this message to petsc-maint@mcs.anl.gov'''
        print '================================================================================='
      print output
      sys.exit(3)
      
  sys.path.insert(0, bsDir)
  sys.path.insert(0, pythonDir)
  import config.framework
  import cPickle

  # Disable shared libraries by default
  import nargs
  if nargs.Arg.findArgument('with-shared', sys.argv[1:]) is None:
    sys.argv.append('--with-shared=0')

  framework = config.framework.Framework(sys.argv[1:]+['-configModules=PETSc.Configure'], loadArgDB = 0)
  try:
    framework.configure(out = sys.stdout)
    framework.storeSubstitutions(framework.argDB)
    framework.argDB['configureCache'] = cPickle.dumps(framework)
    import PETSc.packages
    for i in framework.packages:
      if hasattr(i,'postProcess'):
        i.postProcess()
    framework.logClear()
    return 0
  except RuntimeError, e:
    emsg = str(e)
    if not emsg.endswith('\n'): emsg += '\n'
    msg ='*********************************************************************************\n'\
    +'         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):\n' \
    +'---------------------------------------------------------------------------------------\n'  \
    +emsg+'*********************************************************************************\n'
    se = ''
  except TypeError, e:
    emsg = str(e)
    if not emsg.endswith('\n'): emsg += '\n'
    msg ='*********************************************************************************\n'\
    +'                ERROR in COMMAND LINE ARGUMENT to config/configure.py \n' \
    +'---------------------------------------------------------------------------------------\n'  \
    +emsg+'*********************************************************************************\n'
    se = ''
  except ImportError, e :
    emsg = str(e)
    if not emsg.endswith('\n'): emsg += '\n'
    msg ='*********************************************************************************\n'\
    +'                     UNABLE to FIND MODULE for config/configure.py \n' \
    +'---------------------------------------------------------------------------------------\n'  \
    +emsg+'*********************************************************************************\n'
    se = ''
  except SystemExit, e:
    if e.code is None or e.code == 0:
      return
    msg ='*********************************************************************************\n'\
    +'           CONFIGURATION CRASH  (Please send configure.log to petsc-maint@mcs.anl.gov)\n' \
    +'*********************************************************************************\n'
    se  = str(e)
  except Exception, e:
    msg ='*********************************************************************************\n'\
    +'          CONFIGURATION CRASH  (Please send configure.log to petsc-maint@mcs.anl.gov)\n' \
    +'*********************************************************************************\n'
    se  = str(e)

  framework.logClear()
  print msg
  if hasattr(framework, 'log'):
    import traceback
    framework.log.write(msg+se)
    traceback.print_tb(sys.exc_info()[2], file = framework.log)
    if os.path.isfile(framework.logName+'.bkp'):
      framework.logPrintDivider()
      framework.logPrintBox('Previous configure logs below', debugSection = None)
      f = file(framework.logName+'.bkp')
      framework.log.write(f.read())
      f.close()
    sys.exit(1)

if __name__ == '__main__':
  petsc_configure([])

