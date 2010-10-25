#!/usr/bin/env python
import os
import sys
import commands
# to load ~/.pythonrc.py before inserting correct BuildSystem to path
import user
extraLogs = []
petsc_arch = ''

# Use en_US as language so that BuildSystem parses compiler messages in english
if 'LC_LOCAL' in os.environ and os.environ['LC_LOCAL'] != '' and os.environ['LC_LOCAL'] != 'en_US' and os.environ['LC_LOCAL']!= 'en_US.UTF-8': os.environ['LC_LOCAL'] = 'en_US.UTF-8'
if 'LANG' in os.environ and os.environ['LANG'] != '' and os.environ['LANG'] != 'en_US' and os.environ['LANG'] != 'en_US.UTF-8': os.environ['LANG'] = 'en_US.UTF-8'

if not hasattr(sys, 'version_info') or not sys.version_info[1] >= 2 or not sys.version_info[0] >= 2:
  print '*** You must have Python version 2.2 or higher to run config/configure.py *****'
  print '*          Python is easy to install for end users or sys-admin.              *'
  print '*                  http://www.python.org/download/                            *'
  print '*                                                                             *'
  print '*           You CANNOT configure PETSc without Python                         *'
  print '*   http://www.mcs.anl.gov/petsc/petsc-as/documentation/installation.html     *'
  print '*******************************************************************************'
  sys.exit(4)

if sys.platform == 'win32':
  print '**** Windows python detected. ****'
  print sys.version,'on',sys.platform
  print ''
  print '** You must use cygwin python, but not windows python with PETSc configure. ***'
  sys.exit(4)

def check_for_option_mistakes(opts):
  for opt in opts[1:]:
    name = opt.split('=')[0]
    if name.find('_') >= 0:
      exception = False
      for exc in ['superlu_dist', 'PETSC_ARCH', 'PETSC_DIR', 'CXX_CXXFLAGS', 'LD_SHARED', 'CC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'AR_FLAGS', 'C_VERSION', 'CXX_VERSION', 'FC_VERSION', 'size_t', 'MPI_Comm','MPI_Fint']:
        if name.find(exc) >= 0:
          exception = True
      if not exception:
        raise ValueError('The option '+name+' should probably be '+name.replace('_', '-'));
  return

def check_petsc_arch(opts):
  # If PETSC_ARCH not specified - use script name (if not configure.py)
  global petsc_arch
  found = 0
  for name in opts:
    if name.find('PETSC_ARCH=') >= 0:
      petsc_arch=name.split('=')[1]
      found = 1
      break
  # If not yet specified - use the filename of script
  if not found:
      filename = os.path.basename(sys.argv[0])
      if not filename.startswith('configure') and not filename.startswith('reconfigure'):
        petsc_arch=os.path.splitext(os.path.basename(sys.argv[0]))[0]
        useName = 'PETSC_ARCH='+petsc_arch
        opts.append(useName)
  return 0

def chkwinf90():
  for arg in sys.argv:
    if (arg.find('win32fe') >= 0 and (arg.find('f90') >=0 or arg.find('ifort') >=0)):
      return 1
  return 0

def chkcygwinlink():
  if os.path.exists('/usr/bin/cygcheck.exe') and os.path.exists('/usr/bin/link.exe') and chkwinf90():
      if '--ignore-cygwin-link' in sys.argv: return 0
      print '==============================================================================='
      print ' *** Cygwin /usr/bin/link detected! Compiles with CVF/Intel f90 can break!  **'
      print ' *** To workarround do: "mv /usr/bin/link.exe /usr/bin/link-cygwin.exe"     **'
      print ' *** Or to ignore this check, use configure option: --ignore-cygwin-link    **'
      print '==============================================================================='
      sys.exit(3)
  return 0

def chkbrokencygwin():
  if os.path.exists('/usr/bin/cygcheck.exe'):
    buf = os.popen('/usr/bin/cygcheck.exe -c cygwin').read()
    if buf.find('1.5.11-1') > -1:
      print '==============================================================================='
      print ' *** cygwin-1.5.11-1 detected. config/configure.py fails with this version ***'
      print ' *** Please upgrade to cygwin-1.5.12-1 or newer version. This can  ***'
      print ' *** be done by running cygwin-setup, selecting "next" all the way.***'
      print '==============================================================================='
      sys.exit(3)
  return 0

def chkusingwindowspython():
  if os.path.exists('/usr/bin/cygcheck.exe') and sys.platform != 'cygwin':
    print '==============================================================================='
    print ' *** Non-cygwin python detected. Please rerun config/configure.py **'
    print ' *** with cygwin-python. ***'
    print '==============================================================================='
    sys.exit(3)
  return 0

def chkcygwinpythonver():
  if os.path.exists('/usr/bin/cygcheck.exe'):
    buf = os.popen('/usr/bin/cygcheck.exe -c python').read()
    if (buf.find('2.4') > -1) or (buf.find('2.5') > -1) or (buf.find('2.6') > -1):
      sys.argv.append('--useThreads=0')
      extraLogs.append('''\
===============================================================================
** Cygwin-python-2.4/2.5/2.6 detected. Threads do not work correctly with this
** version. Disabling thread usage for this run of config/configure.py *******
===============================================================================''')
  return 0

def chkrhl9():
  if os.path.exists('/etc/redhat-release'):
    try:
      file = open('/etc/redhat-release','r')
      buf = file.read()
      file.close()
    except:
      # can't read file - assume dangerous RHL9
      buf = 'Shrike'
    if buf.find('Shrike') > -1: 
      sys.argv.append('--useThreads=0')
      extraLogs.append('''\
==============================================================================
   *** RHL9 detected. Threads do not work correctly with this distribution ***
   ****** Disabling thread usage for this run of config/configure.py *********
===============================================================================''')
  return 0

def check_broken_configure_log_links():
  '''Sometime symlinks can get broken if the original files are deleted. Delete such broken links'''
  import os
  for logfile in ['configure.log','configure.log.bkp']:
    if os.path.islink(logfile) and not os.path.isfile(logfile): os.remove(logfile)
  return

def move_configure_log(framework):
  '''Move configure.log to PETSC_ARCH/conf - and update configure.log.bkp in both locations appropriately'''
  global petsc_arch

  if hasattr(framework,'arch'): petsc_arch = framework.arch
  if hasattr(framework,'logName'): curr_file = framework.logName
  else: curr_file = 'configure.log'

  if petsc_arch:
    import shutil
    import os

    # Just in case - confdir is not created
    conf_dir = os.path.join(petsc_arch,'conf')
    if not os.path.isdir(petsc_arch): os.mkdir(petsc_arch)
    if not os.path.isdir(conf_dir): os.mkdir(conf_dir)

    curr_bkp  = curr_file + '.bkp'
    new_file  = os.path.join(conf_dir,curr_file)
    new_bkp   = new_file + '.bkp'

    # Keep backup in $PETSC_ARCH/conf location
    if os.path.isfile(new_bkp): os.remove(new_bkp)
    if os.path.isfile(new_file): os.rename(new_file,new_bkp)
    if os.path.isfile(curr_file):
      shutil.copyfile(curr_file,new_file)
      os.remove(curr_file)
    if os.path.isfile(new_file): os.symlink(new_file,curr_file)
    # If the old bkp is using the same PETSC_ARCH/conf - then update bkp link
    if os.path.realpath(curr_bkp) == os.path.realpath(new_file):
      if os.path.isfile(curr_bkp): os.remove(curr_bkp)
      if os.path.isfile(new_bkp): os.symlink(new_bkp,curr_bkp)
  return

def petsc_configure(configure_options): 
  print '==============================================================================='
  print '             Configuring PETSc to compile on your system                       '
  print '==============================================================================='  

  # Command line arguments take precedence (but don't destroy argv[0])
  sys.argv = sys.argv[:1] + configure_options + sys.argv[1:]
  check_for_option_mistakes(sys.argv)
  # check PETSC_ARCH
  check_petsc_arch(sys.argv)
  check_broken_configure_log_links()

  # support a few standard configure option types
  for l in range(0,len(sys.argv)):
    name = sys.argv[l]
    if name.find('enable-') >= 0:
      if name.find('=') == -1:
        sys.argv[l] = name.replace('enable-','with-')+'=1'
      else:
        head, tail = name.split('=', 1)
        sys.argv[l] = head.replace('enable-','with-')+'='+tail
    if name.find('disable-') >= 0:
      if name.find('=') == -1:
        sys.argv[l] = name.replace('disable-','with-')+'=0'
      else:
        head, tail = name.split('=', 1)
        if tail == '1': tail = '0'
        sys.argv[l] = head.replace('disable-','with-')+'='+tail
    if name.find('without-') >= 0:
      if name.find('=') == -1:
        sys.argv[l] = name.replace('without-','with-')+'=0'
      else:
        head, tail = name.split('=', 1)
        if tail == '1': tail = '0'
        sys.argv[l] = head.replace('without-','with-')+'='+tail

  # Check for broken cygwin
  chkbrokencygwin()
  # Disable threads on RHL9
  chkrhl9()
  # Make sure cygwin-python is used on windows
  chkusingwindowspython()
  # Threads don't work for cygwin & python-2.4, 2.5 etc..
  chkcygwinpythonver()
  chkcygwinlink()

  # Should be run from the toplevel
  configDir = os.path.abspath('config')
  bsDir     = os.path.join(configDir, 'BuildSystem')
  if not os.path.isdir(configDir):
    raise RuntimeError('Run configure from $PETSC_DIR, not '+os.path.abspath('.'))
  if not os.path.isdir(bsDir):
    print '==============================================================================='
    print '''++ Could not locate BuildSystem in %s.''' % configDir
    print '''++ Downloading it using "hg clone http://hg.mcs.anl.gov/petsc/BuildSystem %s"''' % bsDir
    print '==============================================================================='
    (status,output) = commands.getstatusoutput('hg clone http://petsc.cs.iit.edu/petsc/BuildSystem '+ bsDir)
    if status:
      if output.find('ommand not found') >= 0:
        print '==============================================================================='
        print '''** Unable to locate hg (Mercurial) to download BuildSystem; make sure hg is'''
        print '''** in your path or manually copy BuildSystem to $PETSC_DIR/config/BuildSystem'''
        print '''**  from a machine where you do have hg installed and can clone BuildSystem. '''
        print '==============================================================================='
      elif output.find('Cannot resolve host') >= 0:
        print '==============================================================================='
        print '''** Unable to download BuildSystem. You must be off the network.'''
        print '''** Connect to the internet and run config/configure.py again.'''
        print '==============================================================================='
      else:
        print '==============================================================================='
        print '''** Unable to download BuildSystem. Please send this message to petsc-maint@mcs.anl.gov'''
        print '==============================================================================='
      print output
      sys.exit(3)
      
  sys.path.insert(0, bsDir)
  sys.path.insert(0, configDir)
  import config.base
  import config.framework
  import cPickle

  framework = None
  try:
    framework = config.framework.Framework(['--configModules=PETSc.Configure','--optionsModule=PETSc.compilerOptions']+sys.argv[1:], loadArgDB = 0)
    framework.setup()
    framework.logPrint('\n'.join(extraLogs))
    framework.configure(out = sys.stdout)
    framework.storeSubstitutions(framework.argDB)
    framework.argDB['configureCache'] = cPickle.dumps(framework)
    import PETSc.packages
    for i in framework.packages:
      if hasattr(i,'postProcess'):
        i.postProcess()
    framework.printSummary()
    framework.logClear()
    framework.closeLog()
    try:
      move_configure_log(framework)
    except:
      # perhaps print an error about unable to shuffle logs?
      pass
    return 0
  except (RuntimeError, config.base.ConfigureSetupError), e:
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):\n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except (TypeError, ValueError), e:
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                ERROR in COMMAND LINE ARGUMENT to config/configure.py \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except ImportError, e :
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                     UNABLE to FIND MODULE for config/configure.py \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except OSError, e :
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                    UNABLE to EXECUTE BINARIES for config/configure.py \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except SystemExit, e:
    if e.code is None or e.code == 0:
      return
    msg ='*******************************************************************************\n'\
    +'         CONFIGURATION FAILURE  (Please send configure.log to petsc-maint@mcs.anl.gov)\n' \
    +'*******************************************************************************\n'
    se  = str(e)
  except Exception, e:
    msg ='*******************************************************************************\n'\
    +'        CONFIGURATION CRASH  (Please send configure.log to petsc-maint@mcs.anl.gov)\n' \
    +'*******************************************************************************\n'
    se  = str(e)

  print msg
  if not framework is None:
    framework.logClear()
    if hasattr(framework, 'log'):
      import traceback
      try:
        framework.log.write(msg+se)
        traceback.print_tb(sys.exc_info()[2], file = framework.log)
        if hasattr(framework,'log'): close(framework.log)
        move_configure_log(framework)
      except:
        pass
      sys.exit(1)
  else:
    print se
    import traceback
    traceback.print_tb(sys.exc_info()[2])
  if hasattr(framework,'log'): close(framework.log)
  move_configure_log(framework)

if __name__ == '__main__':
  petsc_configure([])

