#!/usr/bin/env python
import os, sys
import commands
# to load ~/.pythonrc.py before inserting correct BuildSystem to path
import user
extraLogs = []
petsc_arch = ''

import urllib
import tarfile

def untar(tar, path = '.', leading = ''):
  if leading:
    entries = [t.name for t in tar.getmembers()]
    prefix = os.path.commonprefix(entries)
    if prefix:
      for tarinfo in tar.getmembers():
        tail = tarinfo.name.split(prefix, 1)[1]
        tarinfo.name = os.path.join(leading, tail)
  for tarinfo in tar.getmembers():
    tar.extract(tarinfo, path)
  return

def downloadPackage(url, filename, targetDirname):
  '''Download the tarball for a package at url, save it as filename, and untar it into targetDirname'''
  filename, headers = urllib.urlretrieve(url, filename)
  tar = tarfile.open(filename, 'r:gz')
  untar(tar, targetDirname, leading = filename.split('.')[0])
  return

def getBuildSystem(configDir,bsDir):
  print '==============================================================================='
  print '''++ Could not locate BuildSystem in %s.''' % configDir
  (status,output) = commands.getstatusoutput('hg showconfig paths.default')
  if status or not output:
    print '++ Mercurial clone not found. Downloading it from http://petsc.cs.iit.edu/petsc/BuildSystem/archive/tip.tar.gz'
    downloadPackage('http://petsc.cs.iit.edu/petsc/BuildSystem/archive/tip.tar.gz', 'BuildSystem.tar.gz', configDir)
  else:
    print '++ Mercurial clone found. URL : ' + output
    bsurl = output.replace('petsc-dev','BuildSystem').replace('releases/petsc-','releases/BuildSystem-')
    if bsurl.find('bitbucket.org') >=0: bsurl = bsurl.lower()
    print '++ Using: hg clone '+ bsurl +' '+ bsDir
    (status,output) = commands.getstatusoutput('hg clone '+ bsurl +' '+ bsDir)
    if status:
      print '++ Unable to clone BuildSystem. Please clone manually'
      print '==============================================================================='
      sys.exit(3)
  print '==============================================================================='
  return


# Use en_US as language so that BuildSystem parses compiler messages in english
if 'LC_LOCAL' in os.environ and os.environ['LC_LOCAL'] != '' and os.environ['LC_LOCAL'] != 'en_US' and os.environ['LC_LOCAL']!= 'en_US.UTF-8': os.environ['LC_LOCAL'] = 'en_US.UTF-8'
if 'LANG' in os.environ and os.environ['LANG'] != '' and os.environ['LANG'] != 'en_US' and os.environ['LANG'] != 'en_US.UTF-8': os.environ['LANG'] = 'en_US.UTF-8'

if not hasattr(sys, 'version_info') or not sys.version_info[0] == 2 or not sys.version_info[1] >= 4:
  print '*** You must have Python2 version 2.4 or higher to run ./configure        *****'
  print '*          Python is easy to install for end users or sys-admin.              *'
  print '*                  http://www.python.org/download/                            *'
  print '*                                                                             *'
  print '*           You CANNOT configure PETSc without Python                         *'
  print '*   http://www.mcs.anl.gov/petsc/documentation/installation.html     *'
  print '*******************************************************************************'
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
    if opt.find('=') >=0:
      optval = opt.split('=')[1]
      if optval == 'ifneeded':
        raise ValueError('The option '+opt+' should probably be '+opt.replace('ifneeded', '1'));
  return

def check_for_option_changed(opts):
# Document changes in command line options here.
  optMap = [('c-blas-lapack','f2cblaslapack')]
  for opt in opts[1:]:
    optname = opt.split('=')[0].strip('-')
    for oldname,newname in optMap:
      if optname.find(oldname) >=0:
        raise ValueError('The option '+opt+' should probably be '+opt.replace(oldname,newname))
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
      if not filename.startswith('configure') and not filename.startswith('reconfigure') and not filename.startswith('setup'):
        petsc_arch=os.path.splitext(os.path.basename(sys.argv[0]))[0]
        useName = 'PETSC_ARCH='+petsc_arch
        opts.append(useName)
  return 0

def chkwinf90():
  for arg in sys.argv:
    if (arg.find('win32fe') >= 0 and (arg.find('f90') >=0 or arg.find('ifort') >=0)):
      return 1
  return 0

def chkdosfiles():
  if not os.path.exists('/usr/bin/cygcheck.exe'): return
  (status,output) = commands.getstatusoutput('hg showconfig paths.default')
  if not status and output: return
  # cygwin - but not a hg clone - so check files in bin dir
  (status,output) = commands.getstatusoutput('file bin/*')
  if status:
    print '==============================================================================='
    print ' *** Incomplete cygwin install? command "file" not found!                    **'
    print '==============================================================================='
    return
  if output.find('with CRLF line terminators') >= 0:
    print '==============================================================================='
    print ' *** Scripts are in DOS mode. Was winzip used instead of tar? Converting.......'
    print '==============================================================================='
    (status,output) = commands.getstatusoutput('dos2unix bin/*')
    if status:
      print '==============================================================================='
      print ' *** Incomplete cygwin install? command "dos2unix" not found!                **'
      print '==============================================================================='
  return

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
      print ' *** cygwin-1.5.11-1 detected. ./configure fails with this version ***'
      print ' *** Please upgrade to cygwin-1.5.12-1 or newer version. This can  ***'
      print ' *** be done by running cygwin-setup, selecting "next" all the way.***'
      print '==============================================================================='
      sys.exit(3)
  return 0

def chkcygwinpython():
  if os.path.exists('/usr/bin/cygcheck.exe') and sys.platform == 'cygwin' :
    sys.argv.append('--useThreads=0')
    extraLogs.append('''\
===============================================================================
** Cygwin-python detected. Threads do not work correctly. ***
** Disabling thread usage for this run of ./configure *******
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
   ****** Disabling thread usage for this run of ./configure *********
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
  try:
    petscdir = os.environ['PETSC_DIR']
    sys.path.append(os.path.join(petscdir,'bin'))
    import petscnagupgrade
    file     = os.path.join(petscdir,'.nagged')
    if not petscnagupgrade.naggedtoday(file):
      petscnagupgrade.currentversion(petscdir)  
  except:
    pass
  print '==============================================================================='
  print '             Configuring PETSc to compile on your system                       '
  print '==============================================================================='  

  try:
    # Command line arguments take precedence (but don't destroy argv[0])
    sys.argv = sys.argv[:1] + configure_options + sys.argv[1:]
    check_for_option_mistakes(sys.argv)
    check_for_option_changed(sys.argv)
  except (TypeError, ValueError), e:
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                ERROR in COMMAND LINE ARGUMENT to ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    sys.exit(msg)
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
  # Threads don't work for cygwin & python...
  chkcygwinpython()
  chkcygwinlink()
  chkdosfiles()

  # Should be run from the toplevel
  configDir = os.path.abspath('config')
  bsDir     = os.path.join(configDir, 'BuildSystem')
  if not os.path.isdir(configDir):
    raise RuntimeError('Run configure from $PETSC_DIR, not '+os.path.abspath('.'))
  if not os.path.isdir(bsDir): getBuildSystem(configDir,bsDir)
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
    framework.printSummary()
    framework.argDB.save(force = True)
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
    +'                ERROR in COMMAND LINE ARGUMENT to ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except ImportError, e :
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                     UNABLE to FIND MODULE for ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except OSError, e :
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                    UNABLE to EXECUTE BINARIES for ./configure \n' \
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
      try:
        framework.log.write('**** Configure header conftest.h ****\n')
        framework.outputHeader(framework.log)
        framework.log.write('**** C specific Configure header conffix.h ****\n')
        framework.outputCHeader(framework.log)
      except Exception, e:
        framework.log.write('Problem writing headers to log: '+str(e))
      import traceback
      try:
        framework.log.write(msg+se)
        traceback.print_tb(sys.exc_info()[2], file = framework.log)
        if hasattr(framework,'log'): framework.log.close()
        move_configure_log(framework)
      except:
        pass
      sys.exit(1)
  else:
    print se
    import traceback
    traceback.print_tb(sys.exc_info()[2])
  if hasattr(framework,'log'): framework.log.close()

if __name__ == '__main__':
  petsc_configure([])

