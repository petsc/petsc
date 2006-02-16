#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package
from stat import *

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download_lam     = ['http://www.lam-mpi.org/download/files/lam-7.1.1.tar.gz']
    self.download_mpich   = ['ftp://ftp.mcs.anl.gov/pub/mpi/mpich2-1.0.3.tar.gz']
    self.download         = ['redefine']
    self.functions        = ['MPI_Init', 'MPI_Comm_create']
    self.includes         = ['mpi.h']
    self.liblist_mpich    = [['libmpich.a', 'libpmpich.a'],
                             ['libfmpich.a','libmpich.a', 'libpmpich.a'],
                             ['libmpich.a'],
                             ['libfmpich.a','libmpich.a', 'libpmpich.a', 'libmpich.a', 'libpmpich.a', 'libpmpich.a'],
                             ['libmpich.a', 'libpmpich.a', 'libmpich.a', 'libpmpich.a', 'libpmpich.a'],
                             ['libmpich.a','libssl.a','libuuid.a','libpthread.a','librt.a','libdl.a'],
                             ['mpich2.lib'],
                             ['libmpich.a','libgm.a','libpthread.a'],
                             ['mpich.lib']]
    self.liblist_lam      = [['liblamf77mpi.a','libmpi++.a','libmpi.a','liblam.a'],
                             ['liblammpi++.a','libmpi.a','liblam.a'],
                             ['libmpi.a','libmpi++.a'],['libmpi.a'],
                             ['liblammpio.a','libpmpi.a','liblamf77mpi.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','liblamf90mpi.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','libmpi.a','liblam.a'],
                             ['liblammpi++.a','libmpi.a','liblam.a'],
                             ['libmpi.a','liblam.a']]
    self.liblist          = [[]] + self.liblist_lam + self.liblist_mpich
    # defaults to --with-mpi=yes
    self.required         = 1
    self.double           = 0
    self.complex          = 1
    self.isPOE            = 0
    self.usingMPIUni      = 0
    self.requires32bitint = 0
    self.shared           = 0
    return

  def setupHelp(self, help):
    PETSc.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('MPI', '-download-lam=<no,yes,ifneeded,filename>',    PETSc.package.ArgDownload(None, 0, 'Download and install LAM/MPI'))
    help.addArgument('MPI', '-download-mpich=<no,yes,ifneeded,filename>',  PETSc.package.ArgDownload(None, 0, 'Download and install MPICH-2'))
    help.addArgument('MPI', '-with-mpirun=<prog>',                nargs.Arg(None, None, 'The utility used to launch MPI jobs'))
    help.addArgument('MPI', '-with-mpi-compilers=<bool>',         nargs.ArgBool(None, 1, 'Try to use the MPI compilers, e.g. mpicc'))
    help.addArgument('MPI', '-download-mpich-machines=[machine1,machine2...]',  nargs.Arg(None, ['localhost','localhost'], 'Machines for MPI to use'))
    help.addArgument('MPI', '-download-mpich-pm=gforker or mpd',  nargs.Arg(None, 'gforker', 'Launcher for MPI processes')) 
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.types = framework.require('config.types', self)
    return

  # search many obscure locations for MPI
  def getSearchDirectories(self):
    import re
    yield ''
    # Try configure package directories
    dirExp = re.compile(r'mpi(ch)?(-.*)?')
    for packageDir in self.framework.argDB['package-dirs']:
      packageDir = os.path.abspath(packageDir)
      if not os.path.isdir(packageDir):
        raise RuntimeError('Invalid package directory: '+packageDir)
      for f in os.listdir(packageDir):
        dir = os.path.join(packageDir, f)
        if not os.path.isdir(dir):
          continue
        if not dirExp.match(f):
          continue
        yield (dir)
    # Try SUSE location
    yield (os.path.abspath(os.path.join('/opt', 'mpich')))
    # Try IBM
    self.isPOE = 1
    dir = os.path.abspath(os.path.join('/usr', 'lpp', 'ppe.poe'))
    yield (os.path.abspath(os.path.join('/usr', 'lpp', 'ppe.poe')))
    self.isPOE = 0
    # Try /usr/local
    yield (os.path.abspath(os.path.join('/usr', 'local')))
    # Try /usr/local/*mpich*
    if os.path.isdir(dir):
      ls = os.listdir(dir)
      for dir in ls:
        if dir.find('mpich') >= 0:
          dir = os.path.join('/usr','local',dir)
          if os.path.isdir(dir):
            yield (dir)
    # Try ~/mpich*
    homedir = os.getenv('HOME')
    if homedir:
      ls = os.listdir(homedir)
      for dir in ls:
        if dir.find('mpich') >= 0:
          dir = os.path.join(homedir,dir)
          if os.path.isdir(dir):
            yield (dir)
    # Try MPICH install locations under Windows
    yield(os.path.join('/cygdrive','c','Program\\ Files','MPICH2'))
    yield(os.path.join('/cygdrive','c','Program\\ Files','MPICH'))
    yield(os.path.join('/cygdrive','c','Program\\ Files','MPICH','SDK.gcc'))
    yield(os.path.join('/cygdrive','c','Program\\ Files','MPICH','SDK'))
    return

  def checkSharedLibrary(self):
    '''Check that the libraries for MPI are shared libraries'''
    self.executeTest(self.configureMPIRUN)
    try:
      self.shared = self.libraries.checkShared('#include <mpi.h>\n','MPI_Init','MPI_Initialized','MPI_Finalize',checkLink = self.checkPackageLink,libraries = self.lib, executor = self.mpirun)
    except RuntimeError, e:
      if self.framework.argDB['with-shared']:
        raise RuntimeError('PETSc shared libraries cannot be built using MPI provided.\nEither rebuild PETSc with --with-shared=0 or rebuild MPI with shared library support')
      self.framework.logPrint('MPI libraries cannot be used with shared libraries')
      self.shared = 0
    return

  def configureMPIRUN(self):
    '''Checking for mpirun'''
    if 'with-mpirun' in self.framework.argDB:
      self.framework.argDB['with-mpirun'] = os.path.expanduser(self.framework.argDB['with-mpirun'])
      if not self.getExecutable(self.framework.argDB['with-mpirun'], resultName = 'mpirun'):
        raise RuntimeError('Invalid mpirun specified: '+str(self.framework.argDB['with-mpirun']))
      return
    if self.isPOE:
      self.mpirun = os.path.join(self.petscdir.dir, 'bin', 'mpirun.poe')
      return
    mpiruns = ['mpiexec -np 1', 'mpirun -np 1', 'mpiexec', 'mpirun']
    path    = []
    if 'with-mpi-dir' in self.framework.argDB:
      path.append(os.path.join(os.path.abspath(self.framework.argDB['with-mpi-dir']), 'bin'))
      # MPICH-NT-1.2.5 installs MPIRun.exe in mpich/mpd/bin
      path.append(os.path.join(os.path.abspath(self.framework.argDB['with-mpi-dir']), 'mpd','bin'))
    for inc in self.include:
      path.append(os.path.join(os.path.dirname(inc), 'bin'))
      # MPICH-NT-1.2.5 installs MPIRun.exe in mpich/SDK/include/../../mpd/bin
      path.append(os.path.join(os.path.dirname(os.path.dirname(inc)),'mpd','bin'))
    for lib in self.lib:
      path.append(os.path.join(os.path.dirname(os.path.dirname(lib)), 'bin'))
    self.pushLanguage('C')
    if os.path.basename(self.getCompiler()) == 'mpicc' and os.path.dirname(self.getCompiler()):
      path.append(os.path.dirname(self.getCompiler()))
    self.popLanguage()
    self.getExecutable(mpiruns, path = path, useDefaultPath = 1, resultName = 'mpirun',setMakeMacro=0)
    self.addMakeMacro('MPIRUN',self.mpirun.replace('-np 1',''))
    return
        
  def configureConversion(self):
    '''Check for the functions which convert communicators between C and Fortran
       - Define HAVE_MPI_COMM_F2C and HAVE_MPI_COMM_C2F if they are present
       - Some older MPI 1 implementations are missing these'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS

    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Comm_f2c(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_F2C', 1)
    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Comm_c2f(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_C2F', 1)
    if self.checkLink('#include <mpi.h>\n', 'MPI_Fint a;\n'):
      self.addDefine('HAVE_MPI_FINT', 1)

    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return

  def configureTypes(self):
    '''Checking for MPI types'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.framework.batchIncludeDirs.extend([self.headers.getIncludeArgument(inc) for inc in self.include])
    self.types.checkSizeof('MPI_Comm', 'mpi.h')
    if 'HAVE_MPI_FINT' in self.defines:
      self.types.checkSizeof('MPI_Fint', 'mpi.h')
    self.compilers.CPPFLAGS = oldFlags
    return

  def alternateConfigureLibrary(self):
    '''Setup MPIUNI, our uniprocessor version of MPI'''
    self.addDefine('HAVE_MPIUNI', 1)
    self.include = [os.path.join(self.petscdir.dir,'include','mpiuni')]
    if 'STDCALL' in self.compilers.defines:
      self.framework.addDefine('MPIUNI_USE_STDCALL')
    self.lib = [os.path.join(self.petscdir.dir,'lib',self.arch.arch,'libmpiuni')]
    self.mpirun = '${PETSC_DIR}/bin/mpirun.uni'
    self.addMakeMacro('MPIRUN','${PETSC_DIR}/bin/mpirun.uni')
    self.addDefine('HAVE_MPI_COMM_F2C', 1)
    self.addDefine('HAVE_MPI_COMM_C2F', 1)
    self.addDefine('HAVE_MPI_FINT', 1)
    self.framework.packages.append(self)
    self.usingMPIUni = 1
    return

  def configureMissingPrototypes(self):
    '''Checks for missing prototypes, which it adds to petscfix.h'''
    if not 'HAVE_MPI_FINT' in self.defines:
      self.addPrototype('typedef int MPI_Fint;')
    if not 'HAVE_MPI_COMM_F2C' in self.defines:
      self.addPrototype('#define MPI_Comm_f2c(a) (a)')
    if not 'HAVE_MPI_COMM_C2F' in self.defines:
      self.addPrototype('#define MPI_Comm_c2f(a) (a)')
    return

  def configureMPICHShared(self):
    '''MPICH cannot be used with shared libraries on the Mac, reject if trying'''
    if config.setCompilers.Configure.isDarwin():
      if not self.setCompilers.staticLibraries:
        for lib in self.lib:
          if lib.find('mpich') >= 0:
            raise RuntimeError('Sorry, we have not been able to figure out how to use shared libraries on the \n \
              Mac with MPICH. Either run config/configure.py with --with-shared=0 or use LAM instead of MPICH; \n\
              for instance with --download-lam=1')
    return

  def checkDownload(self,preOrPost):
    '''Check if we should download LAM or MPICH'''

    if self.framework.argDB['download-lam'] and self.framework.argDB['download-mpich']:
      raise RuntimeError('Sorry, cannot install both LAM and MPICH. Install any one of the two')

    # check for LAM
    if self.framework.argDB['download-lam']:
      if config.setCompilers.Configure.isCygwin():
        raise RuntimeError('Sorry, cannot download-install LAM on Windows. Sugest installing windows version of MPICH manually')
      self.liblist      = self.liblist_lam   # only generate LAM MPI guesses
      self.download     = self.download_lam
      self.downloadname = 'lam'
      return PETSc.package.Package.checkDownload(self,preOrPost)
        
    # Check for MPICH
    if self.framework.argDB['download-mpich']:
      if config.setCompilers.Configure.isCygwin():
        raise RuntimeError('Sorry, cannot download-install MPICH on Windows. Sugest installing windows version of MPICH manually')
      self.liblist      = self.liblist_mpich   # only generate MPICH guesses
      self.download     = self.download_mpich
      self.downloadname = 'mpich'
      return PETSc.package.Package.checkDownload(self,preOrPost)
    return None

  def Install(self):
    if self.framework.argDB['download-lam']:
      return self.InstallLAM()
    elif self.framework.argDB['download-mpich']:
      return self.InstallMPICH()
    else:
      raise RuntimeError('Internal Error!')
    
  def InstallLAM(self):
    lamDir = self.getDir()

    # Get the LAM directories
    installDir = os.path.join(lamDir, self.arch.arch)
    # Configure and Build LAM
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-rsh=ssh','CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    if self.framework.argDB['with-shared']:
      if self.setCompilers.staticLibraries:
        raise RuntimeError('Configuring PETSc with shared libraries - but the system/compilers do not support this')
      args.append('--enable-shared')
    self.framework.popLanguage()
    # c++ can't be disabled with LAM
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    # no separate F90 options for LAM
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('FC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      args.append('--without-fc')
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild LAM oldargs = '+oldargs+'\n new args = '+args+'\n')
      try:
        self.logPrintBox('Configuring LAM/MPI; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+lamDir+';CXX='';export CXX; ./configure '+args, timeout=1500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on LAM/MPI: '+str(e))
      try:
        self.logPrintBox('Compiling LAM/MPI; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+lamDir+';LAM_INSTALL_DIR='+installDir+';export LAM_INSTALL_DIR; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on LAM/MPI: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on LAM/MPI   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on LAM follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on LAM *******\n')
        raise RuntimeError('Error running make on LAM, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      #need to run ranlib on the libraries using the full path
      try:
        output  = config.base.Configure.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on LAM/MPI libraries: '+str(e))
      # start up LAM demon; note lamboot does not close stdout, so call will ALWAYS timeout.
      try:
        output  = config.base.Configure.executeShellCommand('PATH=${PATH}:'+os.path.join(installDir,'bin')+' '+os.path.join(installDir,'bin','lamboot'), timeout=10, log = self.framework.log)[0]
      except:
        pass
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed LAM/MPI into '+installDir)
    return self.getDir()

  def InstallMPICH(self):
    mpichDir = self.getDir()
    installDir = os.path.join(mpichDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
      
    # Configure and Build MPICH
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir]
    args.append('CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      args.append('--disable-cxx')
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')      
      fc = self.framework.getCompiler()
      if self.compilers.fortranIsF90:
        try:
          output, error, status = self.executeShellCommand(fc+' -v')
          output += error
        except:
          output = ''
        if output.find('IBM') >= 0:
          fc = os.path.join(os.path.dirname(fc), 'xlf')
          self.framework.log.write('Using IBM f90 compiler for PETSc, switching to xlf for compiling MPICH\n')
        # now set F90
        args.append('F90="'+fc+' '+self.framework.getCompilerFlags().replace('-Mfree','')+'"')
      else:
        args.append('--disable-f90')
      args.append('F77="'+fc+' '+self.framework.getCompilerFlags().replace('-Mfree','')+'"')
      self.framework.popLanguage()
    else:
      args.append('--disable-f77')
      args.append('--disable-f90')
    if self.framework.argDB['with-shared']:
      if self.setCompilers.staticLibraries:
        raise RuntimeError('Configuring PETSc with shared libraries - but the system/compilers do not support this')
      if self.compilers.isGCC:
        if config.setCompilers.Configure.isDarwin():
          args.append('--enable-sharedlibs=gcc-osx')
        else:        
          args.append('--enable-sharedlibs=gcc')
      else:
        args.append('--enable-sharedlibs=libtool')
    args.append('--without-mpe')
    args.append('--with-pm='+self.argDB['download-mpich-pm'])
    args = ' '.join(args)
    configArgsFilename = os.path.join(installDir,'config.args')
    try:
      fd      = file(configArgsFilename)
      oldargs = fd.readline()
      fd.close()
    except:
      self.framework.logPrint('Unable to find old configure arguments in '+configArgsFilename)
      oldargs = ''
    if not oldargs == args:
      self.framework.logPrint('Have to rebuild MPICH oldargs = '+oldargs+'\n new args = '+args)
      try:
        self.logPrintBox('Running configure on MPICH; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+mpichDir+';./configure '+args, timeout=2000, log = self.framework.log)[0]
      except RuntimeError, e:
        if self.arch.hostOsBase.startswith('cygwin'):
          raise RuntimeError('Error running configure on MPICH. \n \
  On Microsoft Windows systems, please obtain and install the binary distribution from \n \
    http://www.mcs.anl.gov/mpi/mpich/mpich-nt \n \
  then rerun PETSc\'s configure.  \n \
  If you choose to install MPICH to a location other than the default, use \n \
    --with-mpi-dir=<directory> \n \
  to specify the location of the installation when you rerun configure.')
        raise RuntimeError('Error running configure on MPICH: '+str(e))
      try:
        self.logPrintBox('Running make on MPICH; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+mpichDir+';make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        if self.arch.hostOsBase.startswith('cygwin'):
          raise RuntimeError('Error running make; make install on MPICH. \n \
  On Microsoft Windows systems, please obtain and install the binary distribution from \n \
    http://www.mcs.anl.gov/mpi/mpich/mpich-nt \n \
  then rerun PETSc\'s configure.  \n \
  If you choose to install MPICH to a location other than the default, use \n \
    --with-mpi-dir=<directory> \n \
  to specify the location of the installation when you rerun configure.')
        raise RuntimeError('Error running make; make install on MPICH: '+str(e))

      try:
        fd = file(configArgsFilename, 'w')
        fd.write(args)
        fd.close()
      except:
        self.framework.logPrint('Unable to output configure arguments into '+configArgsFilename)
      if self.argDB['download-mpich-pm'] == 'mpd':
        homedir = os.getenv('HOME')
        if homedir:
          if not os.path.isfile(os.path.join(homedir,'.mpd.conf')):
            fd = open(os.path.join(homedir,'.mpd.conf'),'w')
            fd.write('secretword=mr45-j9z\n')
            fd.close()
            os.chmod(os.path.join(homedir,'.mpd.conf'),S_IRWXU)
        else:
          self.logPrint('No HOME env var, so could not check for or create .mpd.conf')

        # start up MPICH's demon
        self.framework.logPrint('Starting up MPICH mpd demon needed for mpirun')
        try:
          output = self.executeShellCommand('cd '+installDir+'; bin/mpdboot',timeout=25)
          self.framework.logPrint('Output from trying to run mpdboot:'+str(output))
          self.framework.logPrint('Started up MPICH mpd demon needed for mpirun')
        except RuntimeError, e:
          self.framework.logPrint('Error trying to run mpdboot:'+str(e))
      self.framework.actions.addArgument('MPI', 'Install', 'Installed MPICH into '+installDir)
    return self.getDir()

  def addExtraLibraries(self):
    '''Check for various auxiliary libraries we may need'''
    extraLib = []
    if not self.setCompilers.usedMPICompilers:
      if self.executeTest(self.libraries.check, [['rt'], 'timer_create', None, extraLib]):
        extraLib.append('librt.a')
      if self.executeTest(self.libraries.check, [['aio'], 'aio_read', None, extraLib]):
        extraLib.insert(0, 'libaio.a')
      if self.executeTest(self.libraries.check, [['nsl'], 'exit', None, extraLib]):
        extraLib.insert(0, 'libnsl.a')
      self.extraLib.extend(extraLib)
    return

  def SGIMPICheck(self):
    '''Returns true if SGI MPI is used'''
    if self.libraries.check(self.lib, 'MPI_SGI_barrier') :
      self.logPrint('SGI MPI detected - defining MISSING_SIGTERM')
      self.addDefine('MISSING_SIGTERM', 1)
      return 1
    else:
      self.logPrint('SGI MPI test failure')
      return 0

  def FortranMPICheck(self):
    '''Make sure fortran include [mpif.h] and library symbols are found'''
    if not hasattr(self.compilers, 'FC'):
      return 0
    # Fortran compiler is being used - so make sure mpif.h exists
    self.libraries.pushLanguage('FC')
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.framework.log.write('Checking for header mpif.h\n')
    if not self.libraries.check(self.lib,'', call = '       include \'mpif.h\''):
        raise RuntimeError('Fortran error! mpif.h could not be located at: '+str(self.include))
    # check if mpi_init form fortran works
    self.framework.log.write('Checking for fortran mpi_init()\n')
    if not self.libraries.check(self.lib,'', call = '       include \'mpif.h\'\n       integer ierr\n       call mpi_init(ierr)'):
      raise RuntimeError('Fortran error! mpi_init() could not be located!')
    self.compilers.CPPFLAGS = oldFlags
    self.libraries.popLanguage()
    return 0

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by MPI'''
    self.addExtraLibraries()
    PETSc.package.Package.configureLibrary(self)
    # Satish check here if the self.directory is truly the MPI root directory with mpicc underneath it
    # if not then set it to None

    #self.executeTest(self.configureMPICHShared)
    self.executeTest(self.configureConversion)
    self.executeTest(self.configureTypes)
    self.executeTest(self.configureMissingPrototypes)
    self.executeTest(self.SGIMPICheck)
    self.executeTest(self.FortranMPICheck)

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
