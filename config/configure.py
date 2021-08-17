#!/usr/bin/env python
from __future__ import print_function
import os, sys

extraLogs = []
petsc_arch = ''

# Use en_US as language so that BuildSystem parses compiler messages in english
def fixLang(lang):
  if lang in os.environ and os.environ[lang] != '':
    lv = os.environ[lang]
    enc = ''
    try: lv,enc = lv.split('.')
    except: pass
    if lv not in ['en_US','C']: lv = 'en_US'
    if enc: lv = lv+'.'+enc
    os.environ[lang] = lv

fixLang('LC_LOCAL')
fixLang('LANG')


def check_for_option_mistakes(opts):
  for opt in opts[1:]:
    name = opt.split('=')[0]
    if name.find(' ') >= 0:
      raise ValueError('The option "'+name+'" has a space character in the name - this is likely incorrect usage.');
    if name.find('_') >= 0:
      exception = False
      for exc in ['mkl_sparse', 'mkl_sparse_optimize', 'mkl_cpardiso', 'mkl_pardiso', 'superlu_dist', 'PETSC_ARCH', 'PETSC_DIR', 'CXX_CXXFLAGS', 'LD_SHARED', 'CC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'AR_FLAGS', 'C_VERSION', 'CXX_VERSION', 'FC_VERSION', 'size_t', 'MPI_Comm','MPI_Fint','int64_t']:
        if name.find(exc) >= 0:
          exception = True
      if not exception:
        raise ValueError('The option '+name+' should probably be '+name.replace('_', '-'));
    if opt.find('=') >=0:
      optval = opt.split('=')[1]
      if optval == 'ifneeded':
        raise ValueError('The option '+opt+' should probably be '+opt.replace('ifneeded', '1'));
    for exc in ['mkl_sparse', 'mkl_sparse_optimize', 'mkl_cpardiso', 'mkl_pardiso', 'superlu_dist']:
      if name.find(exc.replace('_','-')) > -1:
        raise ValueError('The option '+opt+' should be '+opt.replace(exc.replace('_','-'),exc));
  return

def check_for_unsupported_combinations(opts):
  if '--with-precision=single' in opts and '--with-clanguage=cxx' in opts and '--with-scalar-type=complex' in opts:
    sys.exit(ValueError('PETSc does not support single precision complex with C++ clanguage, run with --with-clanguage=c'))

def check_for_option_changed(opts):
# Document changes in command line options here.
  optMap = [('with-64bit-indices','with-64-bit-indices'),
            ('with-mpi-exec','with-mpiexec'),
            ('c-blas-lapack','f2cblaslapack'),
            ('cholmod','suitesparse'),
            ('umfpack','suitesparse'),
            ('matlabengine','matlab-engine'),
            ('sundials','sundials2'),
            ('f-blas-lapack','fblaslapack'),
            ('with-cuda-arch',
             'CUDAFLAGS=-arch'),
            ('with-packages-dir','with-packages-download-dir'),
            ('with-external-packages-dir','with-packages-build-dir'),
            ('package-dirs','with-packages-search-path'),
            ('download-petsc4py-python','with-python-exec'),
            ('search-dirs','with-executables-search-path')]
  for opt in opts[1:]:
    optname = opt.split('=')[0].strip('-')
    for oldname,newname in optMap:
      if optname.find(oldname) >=0 and not optname.find(newname):
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

def chkenable():
  #Replace all 'enable-'/'disable-' with 'with-'=0/1/tail
  #enable-fortran is a special case, the resulting --with-fortran is ambiguous.
  #Would it mean --with-fc=
  en_dash = u'\N{EN DASH}'
  if sys.version_info < (3, 0):
    en_dash = en_dash.encode('utf-8')
  for l in range(0,len(sys.argv)):
    name = sys.argv[l]

    if name.find(en_dash)  >= 0:
      sys.argv[l] = name.replace(en_dash,'-')
    if name.find('enable-cxx') >= 0:
      if name.find('=') == -1:
        sys.argv[l] = name.replace('enable-cxx','with-clanguage=C++')
      else:
        head, tail = name.split('=', 1)
        if tail=='0':
          sys.argv[l] = head.replace('enable-cxx','with-clanguage=C')
        else:
          sys.argv[l] = head.replace('enable-cxx','with-clanguage=C++')
      continue
    if name.find('disable-cxx') >= 0:
      if name.find('=') == -1:
        sys.argv[l] = name.replace('disable-cxx','with-clanguage=C')
      else:
        head, tail = name.split('=', 1)
        if tail == '0':
          sys.argv[l] = head.replace('disable-cxx','with-clanguage=C++')
        else:
          sys.argv[l] = head.replace('disable-cxx','with-clanguage=C')
      continue


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

def chksynonyms():
  #replace common configure options with ones that PETSc BuildSystem recognizes
  simplereplacements = {'F77' : 'FC', 'F90' : 'FC'}
  for l in range(0,len(sys.argv)):
    name = sys.argv[l]

    name = name.replace('download-petsc4py','with-petsc4py')
    name = name.replace('with-openmpi','with-mpi')
    name = name.replace('with-mpich','with-mpi')
    name = name.replace('with-blas-lapack','with-blaslapack')

    if name.find('with-debug=') >= 0 or name.endswith('with-debug'):
      if name.find('=') == -1:
        name = name.replace('with-debug','with-debugging')+'=1'
      else:
        head, tail = name.split('=', 1)
        name = head.replace('with-debug','with-debugging')+'='+tail

    if name.find('with-shared=') >= 0 or name.endswith('with-shared'):
      if name.find('=') == -1:
        name = name.replace('with-shared','with-shared-libraries')+'=1'
      else:
        head, tail = name.split('=', 1)
        name = head.replace('with-shared','with-shared-libraries')+'='+tail

    if name.find('with-index-size=') >=0:
      head,tail = name.split('=',1)
      if int(tail)==32:
        name = '--with-64-bit-indices=0'
      elif int(tail)==64:
        name = '--with-64-bit-indices=1'
      else:
        raise RuntimeError('--with-index-size= must be 32 or 64')

    if name.find('with-precision=') >=0:
      head,tail = name.split('=',1)
      if tail.find('quad')>=0:
        name='--with-precision=__float128'

    for i,j in simplereplacements.items():
      if name.find(i+'=') >= 0:
        name = name.replace(i+'=',j+'=')
      elif name.find('with-'+i.lower()+'=') >= 0:
        name = name.replace(i.lower()+'=',j.lower()+'=')

    # restore 'sys.argv[l]' from the intermediate var 'name'
    sys.argv[l] = name

def chkwincompilerusinglink():
  for arg in sys.argv:
    if (arg.find('win32fe') >= 0 and (arg.find('f90') >=0 or arg.find('ifort') >=0 or arg.find('icl') >=0)):
      return 1
  return 0

def chkdosfiles():
  # cygwin - but not a hg clone - so check one of files in bin dir
  if b"\r\n" in open(os.path.join('lib','petsc','bin','petscmpiexec'),"rb").read():
    print('===============================================================================')
    print(' *** Scripts are in DOS mode. Was winzip used to extract petsc sources?    ****')
    print(' *** Please restart with a fresh tarball and use "tar -xzf petsc.tar.gz"   ****')
    print('===============================================================================')
    sys.exit(3)
  return

def chkcygwinlink():
  if os.path.exists('/usr/bin/cygcheck.exe') and os.path.exists('/usr/bin/link.exe') and chkwincompilerusinglink():
      if '--ignore-cygwin-link' in sys.argv: return 0
      print('===============================================================================')
      print(' *** Cygwin /usr/bin/link detected! Compiles with Intel icl/ifort can break!  **')
      print(' *** To workarround do: "mv /usr/bin/link.exe /usr/bin/link-cygwin.exe"     **')
      print(' *** Or to ignore this check, use configure option: --ignore-cygwin-link. But compiles can fail. **')
      print('===============================================================================')
      sys.exit(3)
  return 0

def chkbrokencygwin():
  if os.path.exists('/usr/bin/cygcheck.exe'):
    buf = os.popen('/usr/bin/cygcheck.exe -c cygwin').read()
    if buf.find('1.5.11-1') > -1:
      print('===============================================================================')
      print(' *** cygwin-1.5.11-1 detected. ./configure fails with this version ***')
      print(' *** Please upgrade to cygwin-1.5.12-1 or newer version. This can  ***')
      print(' *** be done by running cygwin-setup, selecting "next" all the way.***')
      print('===============================================================================')
      sys.exit(3)
  return 0

def chkusingwindowspython():
  if sys.platform == 'win32':
    print('===============================================================================')
    print(' *** Windows python detected. Please rerun ./configure with cygwin-python. ***')
    print('===============================================================================')
    sys.exit(3)
  return 0

def chkcygwinpython():
  if sys.platform == 'cygwin' :
    import platform
    import re
    r=re.compile("([0-9]+).([0-9]+).([0-9]+)")
    m=r.match(platform.release())
    major=int(m.group(1))
    minor=int(m.group(2))
    subminor=int(m.group(3))
    if ((major < 1) or (major == 1 and minor < 7) or (major == 1 and minor == 7 and subminor < 34)):
      sys.argv.append('--useThreads=0')
      extraLogs.append('''\
===============================================================================
** Cygwin version is older than 1.7.34. Python threads do not work correctly. ***
** Disabling thread usage for this run of ./configure *******
===============================================================================''')
  return 0

def chkcygwinwindowscompilers():
  '''Adds win32fe for Microsoft/Intel compilers'''
  if os.path.exists('/usr/bin/cygcheck.exe'):
    for l in range(1,len(sys.argv)):
      option = sys.argv[l]
      for i in ['cl','icl','ifort']:
        if option.startswith(i):
          sys.argv[l] = 'win32fe '+option
          break
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

def chktmpnoexec():
  if not hasattr(os,'ST_NOEXEC'): return # novermin
  if 'TMPDIR' in os.environ: tmpDir = os.environ['TMPDIR']
  else: tmpDir = '/tmp'
  if os.statvfs(tmpDir).f_flag & os.ST_NOEXEC: # novermin
    if os.statvfs(os.path.abspath('.')).f_flag & os.ST_NOEXEC: # novermin
      print('************************************************************************')
      print('* TMPDIR '+tmpDir+' has noexec attribute. Same with '+os.path.abspath('.')+' where petsc is built.')
      print('* Suggest building PETSc in a location without this restriction!')
      print('* Alternatively, set env variable TMPDIR to a location that is not restricted to run binaries.')
      print('************************************************************************')
      sys.exit(4)
    else:
      newTmp = os.path.abspath('tmp-petsc')
      print('************************************************************************')
      print('* TMPDIR '+tmpDir+' has noexec attribute. Using '+newTmp+' instead.')
      print('************************************************************************')
      if not os.path.isdir(newTmp): os.mkdir(os.path.abspath(newTmp))
      os.environ['TMPDIR'] = newTmp
  return

def check_cray_modules():
  import script
  '''For Cray systems check if the cc, CC, ftn compiler suite modules have been set'''
  cray = os.getenv('CRAY_SITE_LIST_DIR')
  if not cray: return
  cray = os.getenv('CRAYPE_DIR')
  if not cray:
   print('************************************************************************')
   print('* You are on a Cray system but no programming environments have been loaded')
   print('* Perhaps you need:')
   print('*       module load intel ; module load PrgEnv-intel')
   print('*   or  module load PrgEnv-cray')
   print('*   or  module load PrgEnv-gnu')
   print('* See https://petsc.org/release/install/install/#installing-on-large-scale-doe-systems')
   print('************************************************************************')
   sys.exit(4)

def check_broken_configure_log_links():
  '''Sometime symlinks can get broken if the original files are deleted. Delete such broken links'''
  import os
  for logfile in ['configure.log','configure.log.bkp']:
    if os.path.islink(logfile) and not os.path.isfile(logfile): os.remove(logfile)
  return

def move_configure_log(framework):
  '''Move configure.log to PETSC_ARCH/lib/petsc/conf - and update configure.log.bkp in both locations appropriately'''
  global petsc_arch

  if hasattr(framework,'arch'): petsc_arch = framework.arch
  if hasattr(framework,'logName'): curr_file = framework.logName
  else: curr_file = 'configure.log'

  if petsc_arch:
    import shutil
    import os

    # Just in case - confdir is not created
    lib_dir = os.path.join(petsc_arch,'lib')
    petsc_dir = os.path.join(petsc_arch,'lib','petsc')
    conf_dir = os.path.join(petsc_arch,'lib','petsc','conf')
    if not os.path.isdir(petsc_arch): os.mkdir(petsc_arch)
    if not os.path.isdir(lib_dir): os.mkdir(lib_dir)
    if not os.path.isdir(petsc_dir): os.mkdir(petsc_dir)
    if not os.path.isdir(conf_dir): os.mkdir(conf_dir)

    curr_bkp  = curr_file + '.bkp'
    new_file  = os.path.join(conf_dir,curr_file)
    new_bkp   = new_file + '.bkp'

    # Keep backup in $PETSC_ARCH/lib/petsc/conf location
    if os.path.isfile(new_bkp): os.remove(new_bkp)
    if os.path.isfile(new_file): os.rename(new_file,new_bkp)
    if os.path.isfile(curr_file):
      shutil.copyfile(curr_file,new_file)
      os.remove(curr_file)
    if os.path.isfile(new_file): os.symlink(new_file,curr_file)
    # If the old bkp is using the same PETSC_ARCH/lib/petsc/conf - then update bkp link
    if os.path.realpath(curr_bkp) == os.path.realpath(new_file):
      if os.path.isfile(curr_bkp): os.remove(curr_bkp)
      if os.path.isfile(new_bkp): os.symlink(new_bkp,curr_bkp)
  return

def print_final_timestamp(framework):
  import time
  framework.log.write(('='*80)+'\n')
  framework.log.write('Finishing configure run at '+time.strftime('%a, %d %b %Y %H:%M:%S %z')+'\n')
  framework.log.write(('='*80)+'\n')
  return

def petsc_configure(configure_options):
  if 'PETSC_DIR' in os.environ:
    petscdir = os.environ['PETSC_DIR']
    if petscdir.find(' ') > -1:
      raise RuntimeError('Your PETSC_DIR '+petscdir+' has spaces in it; this is not allowed.\n Change the directory with PETSc to not have spaces in it')
    if not os.path.isabs(petscdir):
      raise RuntimeError('PETSC_DIR ("'+petscdir+'") is set as a relative path. It must be set as an absolute path.')

    try:
      sys.path.append(os.path.join(petscdir,'lib','petsc','bin'))
      import petscnagupgrade
      file     = os.path.join(petscdir,'.nagged')
      if not petscnagupgrade.naggedtoday(file):
        petscnagupgrade.currentversion(petscdir)
    except:
      pass
  print('=============================================================================================')
  print('                      Configuring PETSc to compile on your system                            ')
  print('=============================================================================================')

  try:
    # Command line arguments take precedence (but don't destroy argv[0])
    sys.argv = sys.argv[:1] + configure_options + sys.argv[1:]
    check_for_option_mistakes(sys.argv)
    check_for_option_changed(sys.argv)
  except (TypeError, ValueError) as e:
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                ERROR in COMMAND LINE ARGUMENT to ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    sys.exit(msg)
  # check PETSC_ARCH
  check_for_unsupported_combinations(sys.argv)
  check_petsc_arch(sys.argv)
  check_broken_configure_log_links()

  #rename '--enable-' to '--with-'
  chkenable()
  # support a few standard configure option types
  chksynonyms()
  # Check for broken cygwin
  chkbrokencygwin()
  # Disable threads on RHL9
  chkrhl9()
  # Make sure cygwin-python is used on windows
  chkusingwindowspython()
  # Threads don't work for cygwin & python...
  chkcygwinpython()
  chkcygwinlink()
  chkdosfiles()
  chkcygwinwindowscompilers()
  chktmpnoexec()

  for l in range(1,len(sys.argv)):
    if sys.argv[l].startswith('--with-fc=') and sys.argv[l].endswith('nagfor'):
      # need a way to save this value and later CC so that petscnagfor may use them
      name = sys.argv[l].split('=')[1]
      sys.argv[l] = '--with-fc='+os.path.join(os.path.abspath('.'),'lib','petsc','bin','petscnagfor')
      break


  # Should be run from the toplevel
  configDir = os.path.abspath('config')
  bsDir     = os.path.join(configDir, 'BuildSystem')
  if not os.path.isdir(configDir):
    raise RuntimeError('Run configure from $PETSC_DIR, not '+os.path.abspath('.'))
  sys.path.insert(0, bsDir)
  sys.path.insert(0, configDir)
  import config.base
  import config.framework
  import pickle
  import traceback

  # Check Cray without modules
  check_cray_modules()

  tbo = None
  framework = None
  try:
    framework = config.framework.Framework(['--configModules=PETSc.Configure','--optionsModule=config.compilerOptions']+sys.argv[1:], loadArgDB = 0)
    framework.setup()
    framework.logPrint('\n'.join(extraLogs))
    framework.configure(out = sys.stdout)
    framework.storeSubstitutions(framework.argDB)
    framework.argDB['configureCache'] = pickle.dumps(framework)
    framework.printSummary()
    framework.argDB.save(force = True)
    framework.logClear()
    print_final_timestamp(framework)
    framework.closeLog()
    try:
      move_configure_log(framework)
    except:
      # perhaps print an error about unable to shuffle logs?
      pass
    return 0
  except (RuntimeError, config.base.ConfigureSetupError) as e:
    tbo = sys.exc_info()[2]
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):\n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except (TypeError, ValueError) as e:
    # this exception is automatically deleted by Python so we need to save it to print below
    tbo = sys.exc_info()[2]
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'    TypeError or ValueError possibly related to ERROR in COMMAND LINE ARGUMENT while running ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except ImportError as e :
    # this exception is automatically deleted by Python so we need to save it to print below
    tbo = sys.exc_info()[2]
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                     ImportError while runing ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except OSError as e :
    tbo = sys.exc_info()[2]
    emsg = str(e)
    if not emsg.endswith('\n'): emsg = emsg+'\n'
    msg ='*******************************************************************************\n'\
    +'                    OSError while running ./configure \n' \
    +'-------------------------------------------------------------------------------\n'  \
    +emsg+'*******************************************************************************\n'
    se = ''
  except SystemExit as e:
    tbo = sys.exc_info()[2]
    if e.code is None or e.code == 0:
      return
    if e.code == 10:
      sys.exit(10)
    msg ='*******************************************************************************\n'\
    +'         CONFIGURATION FAILURE  (Please send configure.log to petsc-maint@mcs.anl.gov)\n' \
    +'*******************************************************************************\n'
    se  = str(e)
  except Exception as e:
    tbo = sys.exc_info()[2]
    msg ='*******************************************************************************\n'\
    +'        CONFIGURATION CRASH  (Please send configure.log to petsc-maint@mcs.anl.gov)\n' \
    +'*******************************************************************************\n'
    se  = str(e)

  print(msg)
  if not framework is None:
    framework.logClear()
    if hasattr(framework, 'log'):
      try:
        if hasattr(framework,'compilerDefines'):
          framework.log.write('**** Configure header '+framework.compilerDefines+' ****\n')
          framework.outputHeader(framework.log)
        if hasattr(framework,'compilerFixes'):
          framework.log.write('**** C specific Configure header '+framework.compilerFixes+' ****\n')
          framework.outputCHeader(framework.log)
      except Exception as e:
        framework.log.write('Problem writing headers to log: '+str(e))
      try:
        framework.log.write(msg+se)
        traceback.print_tb(tbo, file = framework.log)
        print_final_timestamp(framework)
        if hasattr(framework,'log'): framework.log.close()
        move_configure_log(framework)
      except Exception as e:
        print('Error printing error message from exception or printing the traceback:'+str(e))
        traceback.print_tb(sys.exc_info()[2])
      sys.exit(1)
    else:
      print(se)
      traceback.print_tb(tbo)
  else:
    print(se)
    traceback.print_tb(tbo)
  if hasattr(framework,'log'): framework.log.close()

if __name__ == '__main__':
  petsc_configure([])
