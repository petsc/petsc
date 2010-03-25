

#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import config.package
import os
from stat import *

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download_openmpi   = ['http://www.open-mpi.org/software/ompi/v1.4/downloads/openmpi-1.4.1.tar.gz']
    self.download_mpich     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich2-1.0.8.tar.gz']
    self.download           = ['redefine']
    self.functions          = ['MPI_Init', 'MPI_Comm_create']
    self.includes           = ['mpi.h']
    liblist_mpich         = [['fmpich2.lib','fmpich2g.lib','fmpich2s.lib','mpi.lib'],
                             ['fmpich2.lib','fmpich2g.lib','mpi.lib'],['fmpich2.lib','mpich2.lib'],
                             ['libfmpich2g.a','libmpi.a'],['libfmpich.a','libmpich.a', 'libpmpich.a'],
                             ['libmpich.a', 'libpmpich.a'],
                             ['libfmpich.a','libmpich.a', 'libpmpich.a', 'libmpich.a', 'libpmpich.a', 'libpmpich.a'],
                             ['libmpich.a', 'libpmpich.a', 'libmpich.a', 'libpmpich.a', 'libpmpich.a'],
                             ['libmpich.a','libssl.a','libuuid.a','libpthread.a','librt.a','libdl.a'],
                             ['libmpich.a','libnsl.a','libsocket.a','librt.a','libnsl.a','libsocket.a'],
                             ['libmpich.a','libgm.a','libpthread.a']]
    liblist_lam           = [['liblamf77mpi.a','libmpi++.a','libmpi.a','liblam.a'],
                             ['liblammpi++.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','liblamf77mpi.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','liblamf90mpi.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','libmpi.a','liblam.a'],
                             ['liblammpi++.a','libmpi.a','liblam.a'],
                             ['libmpi.a','liblam.a']]
    liblist_msmpi         = [[os.path.join('amd64','msmpifec.lib'),os.path.join('amd64','msmpi.lib')],
                             [os.path.join('i386','msmpifec.lib'),os.path.join('i386','msmpi.lib')]]
    liblist_other         = [['libmpich.a','libpthread.a'],['libmpi++.a','libmpi.a']]
    liblist_single        = [['libmpi.a'],['libmpich.a'],['mpi.lib'],['mpich2.lib'],['mpich.lib'],
                             [os.path.join('amd64','msmpi.lib')],[os.path.join('i386','msmpi.lib')]]
    self.liblist          = [[]] + liblist_mpich + liblist_lam + liblist_msmpi + liblist_other + liblist_single
    # defaults to --with-mpi=yes
    self.required         = 1
    self.double           = 0
    self.complex          = 1
    self.isPOE            = 0
    self.usingMPIUni      = 0
    self.requires32bitint = 0
    self.shared           = 0
    # local state
    self.commf2c          = 0
    self.commc2f          = 0
    self.needBatchMPI     = 1
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('MPI', '-download-mpich=<no,yes,ifneeded,filename>',        nargs.ArgDownload(None, 0, 'Download and install MPICH-2'))
    help.addArgument('MPI', '-download-openmpi=<no,yes,ifneeded,filename>',      nargs.ArgDownload(None, 0, 'Download and install OpenMPI'))
    help.addArgument('MPI', '-with-mpiexec=<prog>',                              nargs.Arg(None, None, 'The utility used to launch MPI jobs'))
    help.addArgument('MPI', '-with-mpi-compilers=<bool>',                        nargs.ArgBool(None, 1, 'Try to use the MPI compilers, e.g. mpicc'))
    help.addArgument('MPI', '-known-mpi-shared=<bool>',                          nargs.ArgBool(None, None, 'Indicates the MPI libraries are shared (the usual test will be skipped)'))
    help.addArgument('MPI', '-download-mpich-pm=hydra, gforker or mpd',          nargs.Arg(None, 'gforker', 'Launcher for MPI processes'))
    help.addArgument('MPI', '-download-mpich-device=ch3:nemesis or see mpich2 docs', nargs.Arg(None, 'ch3:sock', 'Communicator for MPI processes'))
    help.addArgument('MPI', '-download-mpich-mpe',                               nargs.ArgBool(None, 0, 'Install MPE with MPICH'))
    help.addArgument('MPI', '-download-mpich-shared',                            nargs.ArgBool(None, 0, 'Install MPICH with shared libraries'))    
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
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
    yield(os.path.join('/cygdrive','c','Program Files','Microsoft HPC Pack 2008 SDK'))
    yield(os.path.join('/cygdrive','c','Program Files','Microsoft Compute Cluster Pack'))
    yield(os.path.join('/cygdrive','c','Program Files','MPICH2'))
    yield(os.path.join('/cygdrive','c','Program Files (x86)','MPICH2'))
    yield(os.path.join('/cygdrive','c','Program Files','MPICH'))
    yield(os.path.join('/cygdrive','c','Program Files','MPICH','SDK.gcc'))
    yield(os.path.join('/cygdrive','c','Program Files','MPICH','SDK'))
    return

  def checkSharedLibrary(self):
    '''Sets flag indicating if MPI libraries are shared or not and
    determines if MPI libraries CANNOT be used by shared libraries'''
    self.executeTest(self.configureMPIEXEC)
    try:
      self.shared = self.libraries.checkShared('#include <mpi.h>\n','MPI_Init','MPI_Initialized','MPI_Finalize',checkLink = self.checkPackageLink,libraries = self.lib, defaultArg = 'known-mpi-shared', executor = self.mpiexec)
    except RuntimeError, e:
      if self.framework.argDB['with-shared']:
        raise RuntimeError('Shared libraries cannot be built using MPI provided.\nEither rebuild with --with-shared=0 or rebuild MPI with shared library support')
      self.framework.logPrint('MPI libraries cannot be used with shared libraries')
      self.shared = 0
    return

  def configureMPIEXEC(self):
    '''Checking for mpiexec'''
    if 'with-mpiexec' in self.framework.argDB:
      self.framework.argDB['with-mpiexec'] = os.path.expanduser(self.framework.argDB['with-mpiexec'])
      if not self.getExecutable(self.framework.argDB['with-mpiexec'], resultName = 'mpiexec'):
        raise RuntimeError('Invalid mpiexec specified: '+str(self.framework.argDB['with-mpiexec']))
      return
    if self.isPOE:
      self.mpiexec = os.path.abspath(os.path.join('bin', 'mpiexec.poe'))
      return
    if self.framework.argDB['with-batch']:
      self.mpiexec = 'Not_appropriate_for_batch_systems'
      self.addMakeMacro('MPIEXEC',self.mpiexec)
      return   
    mpiexecs = ['mpiexec -n 1', 'mpirun -n 1', 'mprun -n 1', 'mpiexec', 'mpirun', 'mprun']
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
    if not self.getExecutable(mpiexecs, path = path, useDefaultPath = 1, resultName = 'mpiexec',setMakeMacro=0):
      raise RuntimeError('Could not locate MPIEXEC - please specify --with-mpiexec option')
    self.addMakeMacro('MPIEXEC',self.mpiexec.replace(' -n 1','').replace(' ', '\\ '))
    return

  def configureMPI2(self):
    '''Check for functions added to the interface in MPI-2'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    if self.checkLink('#include <mpi.h>\n', 'int flag;if (MPI_Finalized(&flag));\n'):
      self.haveFinalized = 1
      self.addDefine('HAVE_MPI_FINALIZED', 1)
    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Allreduce(0, MPI_IN_PLACE, 1, MPI_INT, MPI_SUM, MPI_COMM_SELF));\n'):
      self.haveInPlace = 1
      self.addDefine('HAVE_MPI_IN_PLACE', 1)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
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
      self.commf2c = 1
      self.addDefine('HAVE_MPI_COMM_F2C', 1)
    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Comm_c2f(MPI_COMM_WORLD));\n'):
      self.commc2f = 1
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
    self.framework.addBatchLib(self.lib)
    self.types.checkSizeof('MPI_Comm', 'mpi.h')
    if 'HAVE_MPI_FINT' in self.defines:
      self.types.checkSizeof('MPI_Fint', 'mpi.h')
    self.compilers.CPPFLAGS = oldFlags
    return

  def configureMPITypes(self):
    '''Checking for MPI Datatype handles'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    #for datatype, name in [('MPI_LONG_DOUBLE', 'long-double'), ('MPI_ENORMOUS_DOUBLE', 'enormous-double'), ('MPI_UNBELIEVABLE_DOUBLE', 'unbelievable-double')]:
    for datatype, name in [('MPI_LONG_DOUBLE', 'long-double'),('MPI_C_DOUBLE_COMPLEX', 'c-double-complex')]:
      includes = '#ifdef PETSC_HAVE_STDLIB_H\n  #include <stdlib.h>\n#endif\n#include <mpi.h>\n'
      body     = 'MPI_Aint size;\nint ierr;\nMPI_Init(0,0);\nierr = MPI_Type_extent('+datatype+', &size);\nif(ierr || (size == 0)) exit(1);\nMPI_Finalize();\n'
      if self.checkCompile(includes, body):
        if 'known-mpi-'+name in self.argDB:
          if int(self.argDB['known-mpi-'+name]):
            self.addDefine('HAVE_'+datatype, 1)
        elif not self.argDB['with-batch']:
          self.pushLanguage('C')
          if self.checkRun(includes, body, defaultArg = 'known-mpi-'+name):
            self.addDefine('HAVE_'+datatype, 1)
          self.popLanguage()
        else:
          if self.needBatchMPI:
            self.framework.addBatchSetup('if (MPI_Init(&argc, &argv));')
            self.framework.addBatchCleanup('if (MPI_Finalize());')
            self.needBatchMPI = 0
          self.framework.addBatchInclude(['#include <stdlib.h>', '#define MPICH_IGNORE_CXX_SEEK', '#define MPICH_SKIP_MPICXX 1', '#define OMPI_SKIP_MPICXX 1', '#include <mpi.h>'])
          self.framework.addBatchBody('''
{
  MPI_Aint size=0;
  int ierr=0;
  if (MPI_LONG_DOUBLE != MPI_DATATYPE_NULL) {
    ierr = MPI_Type_extent(%s, &size);
  }
  if(!ierr && (size != 0)) {
    fprintf(output, "  \'--known-mpi-%s=1\',\\n");
  } else {
    fprintf(output, "  \'--known-mpi-%s=0\',\\n");
  }
}''' % (datatype, name, name))
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return

  def alternateConfigureLibrary(self):
    '''Setup MPIUNI, our uniprocessor version of MPI'''
    self.addDefine('HAVE_MPIUNI', 1)
    #
    #  Even though MPI-Uni is not an external package (it is in PETSc source) we need to stick the
    #  include path for its mpi.h and mpif.h so that external packages that are built with PETSc to
    #  use MPI-Uni can find them.
    self.include = [os.path.abspath(os.path.join('include', 'mpiuni'))]
    self.framework.packages.append(self)    
    self.mpiexec = '${PETSC_DIR}/bin/mpiexec.uni'
    self.addMakeMacro('MPIEXEC','${PETSC_DIR}/bin/mpiexec.uni')
    self.addDefine('HAVE_MPI_COMM_F2C', 1)
    self.addDefine('HAVE_MPI_COMM_C2F', 1)
    self.addDefine('HAVE_MPI_FINT', 1)
    self.addDefine('HAVE_MPI_C_DOUBLE_COMPLEX', 1)    
    self.commf2c = 1
    self.commc2f = 1
    self.usingMPIUni = 1
    self.version = 'PETSc MPIUNI uniprocessor MPI replacement'
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

  def checkDownload(self, requireDownload = 1):
    '''Check if we should download MPICH or OpenMPI'''
    if 'download-mpi' in self.framework.argDB and self.framework.argDB['download-mpi']:
      raise RuntimeError('Option --download-mpi does not exist! Use --download-mpich or --download-openmpi instead.')
    
    if self.framework.argDB['download-mpich'] and self.framework.argDB['download-openmpi']:
      raise RuntimeError('Cannot install more than one of OpenMPI or  MPICH-2 for a single configuration. \nUse different PETSC_ARCH if you want to be able to switch between two')

    if self.framework.argDB['download-openmpi'] and self.framework.argDB['prefix']:
      raise RuntimeError('Currently --download-openmpi option does not work with --prefix install.\nSuggest installing OpenMPI separately, and then configuring PETSc with --with-mpi-dir option.')

    # Check for MPICH
    if self.framework.argDB['download-mpich']:
      if config.setCompilers.Configure.isCygwin() and not config.setCompilers.Configure.isGNU(self.setCompilers.CC):
        raise RuntimeError('Sorry, cannot download-install MPICH on Windows. Sugest installing windows version of MPICH manually')
      self.liblist      = [[]]
      self.download     = self.download_mpich
      self.downloadname = 'mpich'
      return config.package.Package.checkDownload(self, requireDownload)

    # Check for OpenMPI
    if self.framework.argDB['download-openmpi']:
      if config.setCompilers.Configure.isCygwin() and not config.setCompilers.Configure.isGNU(self.setCompilers.CC):
        raise RuntimeError('Sorry, cannot download-install OpenMPI on Windows. Sugest installing windows version of MPICH manually')
      self.liblist      = [[]]
      self.download     = self.download_openmpi
      self.downloadname = 'openmpi'
      return config.package.Package.checkDownload(self, requireDownload)
    return None

  def Install(self):
    if self.framework.argDB['download-mpich']:
      return self.InstallMPICH()
    elif self.framework.argDB['download-openmpi']:
      return self.InstallOpenMPI()
    else:
      raise RuntimeError('Internal Error!')
    
  def InstallOpenMPI(self):
    openmpiDir = self.getDir()

    # Get the OPENMPI directories
    installDir = os.path.join(self.defaultInstallDir,self.arch)
    confDir = os.path.join(self.defaultInstallDir,self.arch,'conf')
    args = ['--prefix='+installDir,'--with-rsh=ssh']
    # Configure and Build OPENMPI
    self.pushLanguage('C')
    flags = self.getCompilerFlags()
    if config.setCompilers.Configure.isDarwin():
      # OpenMPI configure crashes on Apple if -g or -g3 flag is passed in here 
      flags = flags.replace('-g3','')
      flags = flags.replace('-g','')
    args.append('CC="'+self.getCompiler()+'"')
    args.append('CFLAGS="'+flags+'"')
    if self.framework.argDB['with-shared']:
      if self.setCompilers.staticLibraries:
        raise RuntimeError('Configuring with shared libraries - but the system/compilers do not support this')
      args.append('--enable-shared')
    self.popLanguage()
    # c++ can't be disabled with OPENMPI
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      flags = self.getCompilerFlags()
      if config.setCompilers.Configure.isDarwin():
        flags = flags.replace('-g3','')
        flags = flags.replace('-g','')
      args.append('CXX="'+self.getCompiler()+'"')
      args.append('CXXFLAGS="'+flags+'"')
      self.popLanguage()
    else:
      raise RuntimeError('Error: OpenMPI requires C++ compiler. None specified')
    # no separate F90 options for OPENMPI
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      args.append('F77="'+self.getCompiler()+'"')
      args.append('FFLAGS="'+self.getCompilerFlags()+'"')
      if self.compilers.fortranIsF90:
        args.append('FC="'+self.getCompiler()+'"')
        args.append('FCFLAGS="'+self.getCompilerFlags()+'"')
      else:
        args.append('--disable-mpi-f90')
        args.append('FC=""')
      self.popLanguage()
    else:
      args.append('--disable-mpi-f77')
      args.append('--disable-mpi-f90')
      args.append('F77=""')
      args.append('FC=""')
    if not self.framework.argDB['with-shared']:
      args.append('--enable-shared=no')
      args.append('--enable-static=yes')
        
    args = ' '.join(args)
    configArgsFilename = os.path.join(confDir,self.downloadname)
    try:
      fd      = file(configArgsFilename)
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild OPENMPI oldargs = '+oldargs+'\n new args = '+args+'\n')
      try:
        self.logPrintBox('Configuring OPENMPI/MPI; this may take several minutes')
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+openmpiDir+';./configure '+args, timeout=1500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on OPENMPI/MPI: '+str(e))
      try:
        self.logPrintBox('Compiling OPENMPI/MPI; this may take several minutes')
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+openmpiDir+'; make install', timeout=6000, log = self.framework.log)
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+openmpiDir+'; make clean', timeout=200, log = self.framework.log)        
      except RuntimeError, e:
        raise RuntimeError('Error running make on OPENMPI/MPI: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on OPENMPI/MPI   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on OPENMPI follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on OPENMPI *******\n')
        raise RuntimeError('Error running make on OPENMPI, libraries not installed')
      try:
        # OpenMPI puts Fortran 90 modules into lib instead of include like we want
        output,err,ret  = config.base.Configure.executeShellCommand('cp '+os.path.join(installDir,'lib','*.mod ')+os.path.join(installDir,'include'), timeout=30, log = self.framework.log)
      except RuntimeError, e:
        pass
    
      fd = file(configArgsFilename, 'w')
      fd.write(args)
      fd.close()
      #need to run ranlib on the libraries using the full path
      try:
        if not self.framework.argDB['with-shared']:
          output,err,ret  = config.base.Configure.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on OPENMPI/MPI libraries: '+str(e))
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed OPENMPI/MPI into '+installDir)

    self.updateCompilers(installDir,'mpicc','mpic++','mpif77','mpif90')
    return installDir

  def InstallMPICH(self):
    mpichDir = self.getDir()
    installDir = os.path.join(self.defaultInstallDir,self.arch)
    confDir = os.path.join(self.defaultInstallDir,self.arch,'conf')
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
      
    # Configure and Build MPICH
    self.pushLanguage('C')
    args = ['--prefix='+installDir]
    compiler = self.getCompiler()
    args.append('CC="'+self.getCompiler()+'"')
    args.append('CFLAGS="'+self.getCompilerFlags()+'"')
    self.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('CXX="'+self.getCompiler()+'"')
      args.append('CXXFLAGS="'+self.getCompilerFlags()+'"')
      self.popLanguage()
    else:
      args.append('--disable-cxx')
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')      
      fc = self.getCompiler()
      if self.compilers.fortranIsF90:
        try:
          output, error, status = self.executeShellCommand(fc+' -v')
          output += error
        except:
          output = ''
        if output.find('IBM') >= 0:
          fc = os.path.join(os.path.dirname(fc), 'xlf')
          self.framework.log.write('Using IBM f90 compiler, switching to xlf for compiling MPICH\n')
        # now set F90
        args.append('F90="'+fc+'"')
        args.append('F90FLAGS="'+self.getCompilerFlags().replace('-Mfree','')+'"')
      else:
        args.append('--disable-f90')
      args.append('F77="'+fc+'"')
      args.append('FFLAGS="'+self.getCompilerFlags().replace('-Mfree','')+'"')
      self.popLanguage()
    else:
      args.append('--disable-f77')
      args.append('--disable-f90')
    if self.framework.argDB['with-shared'] or self.framework.argDB['download-mpich-shared']:
      if self.compilers.isGCC or config.setCompilers.Configure.isIntel(compiler):
        if config.setCompilers.Configure.isDarwin():
          args.append('--enable-sharedlibs=gcc-osx')
        else:
          args.append('--enable-sharedlibs=gcc')
      elif config.setCompilers.Configure.isSun(compiler):
        args.append('--enable-sharedlibs=solaris-cc')
      else:
        args.append('--enable-sharedlibs=libtool')
    if 'download-mpich-device' in self.argDB:
      args.append('--with-device='+self.argDB['download-mpich-device'])
    if self.argDB['download-mpich-mpe']:
      args.append('--with-mpe')
    else:
      args.append('--without-mpe')
    args.append('--with-pm='+self.argDB['download-mpich-pm'])
    #if self.argDB['download-mpich-pm'] == 'hydra':
    #  args.append('--with-hydra-bss=fork,ssh')
    # make MPICH behave properly for valgrind
    args.append('--enable-g=meminit')    
    args.append('--enable-fast')    
    args = ' '.join(args)
    configArgsFilename = os.path.join(confDir,self.downloadname)
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
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+mpichDir+';./configure '+args, timeout=2000, log = self.framework.log)
      except RuntimeError, e:
        import sys
        if sys.platform.startswith('cygwin'):
          raise RuntimeError('Error running configure on MPICH. \n \
  On Microsoft Windows systems, please obtain and install the binary distribution from \n \
    http://www.mcs.anl.gov/mpi/mpich/mpich-nt \n \
  then rerun configure.  \n \
  If you choose to install MPICH to a location other than the default, use \n \
    --with-mpi-dir=<directory> \n \
  to specify the location of the installation when you rerun configure.')
        raise RuntimeError('Error running configure on MPICH: '+str(e))
      try:
        self.logPrintBox('Running make on MPICH; this may take several minutes')
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+mpichDir+';make; make install', timeout=6000, log = self.framework.log)
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+mpichDir+';make clean', timeout=200, log = self.framework.log)
      except RuntimeError, e:
        import sys
        if sys.platform.startswith('cygwin'):
          raise RuntimeError('Error running make; make install on MPICH. \n \
  On Microsoft Windows systems, please obtain and install the binary distribution from \n \
    http://www.mcs.anl.gov/mpi/mpich/mpich-nt \n \
  then rerun configure.  \n \
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

        self.logPrintBox('You have requested the mpd version of MPICH to be installed \nYou must start the demon in order to run MPI jobs, run the following:\n'+os.path.join(installDir,'bin','mpdboot')+'--file=hostsfile\nhostsfile should contain a list of machines where you wish to run MPICH jobs\nRun with --help for more options')
      self.framework.actions.addArgument('MPI', 'Install', 'Installed MPICH into '+installDir)

    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

  def updateCompilers(self, installDir, mpiccName, mpicxxName, mpif77Name, mpif90Name):
    '''Check if mpicc, mpicxx etc binaries exist - and update setCompilers() database.
    The input arguments are the names of the binaries specified by the respective pacakges MPICH/LAM.'''

    # Initialize to empty
    mpicc=''
    mpicxx=''
    mpifc=''

    mpicc = os.path.join(installDir,"bin",mpiccName)
    if not os.path.isfile(mpicc): raise RuntimeError('Could not locate installed MPI compiler: '+mpicc)
    if hasattr(self.compilers, 'CXX'):
      mpicxx = os.path.join(installDir,"bin",mpicxxName)
      if not os.path.isfile(mpicxx): raise RuntimeError('Could not locate installed MPI compiler: '+mpicxx)
    if hasattr(self.compilers, 'FC'):
      if self.compilers.fortranIsF90:
        mpifc = os.path.join(installDir,"bin",mpif90Name)
      else:
        mpifc = os.path.join(installDir,"bin",mpif77Name)
      if not os.path.isfile(mpifc): raise RuntimeError('Could not locate installed MPI compiler: '+mpifc)
    # redo compiler detection
    self.setCompilers.updateMPICompilers(mpicc,mpicxx,mpifc)
    self.compilers.__init__(self.framework)
    self.compilers.headerPrefix = self.headerPrefix
    self.compilers.configure()
    self.compilerFlags.configure()
    return

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

  def CxxMPICheck(self):
    '''Make sure C++ can compile and link'''
    if not hasattr(self.compilers, 'CXX'):
      return 0
    self.libraries.pushLanguage('Cxx')
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.framework.log.write('Checking for header mpi.h\n')
    if not self.libraries.checkCompile(includes = '#include <mpi.h>\n'):
      raise RuntimeError('C++ error! mpi.h could not be located at: '+str(self.include))
    # check if MPI_Finalize from c++ exists
    self.framework.log.write('Checking for C++ MPI_Finalize()\n')
    if not self.libraries.check(self.lib, 'MPI_Finalize', prototype = '#include <mpi.h>', call = 'int ierr;\nierr = MPI_Finalize();', cxxMangle = 1):
      raise RuntimeError('C++ error! MPI_Finalize() could not be located!')
    self.compilers.CPPFLAGS = oldFlags
    self.libraries.popLanguage()
    return

  def FortranMPICheck(self):
    '''Make sure fortran include [mpif.h] and library symbols are found'''
    if not hasattr(self.compilers, 'FC'):
      return 0
    # Fortran compiler is being used - so make sure mpif.h exists
    self.libraries.pushLanguage('FC')
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.framework.log.write('Checking for header mpif.h\n')
    if not self.libraries.checkCompile(body = '       include \'mpif.h\''):
      raise RuntimeError('Fortran error! mpif.h could not be located at: '+str(self.include))
    # check if mpi_init form fortran works
    self.framework.log.write('Checking for fortran mpi_init()\n')
    if not self.libraries.check(self.lib,'', call = '       include \'mpif.h\'\n       integer ierr\n       call mpi_init(ierr)'):
      raise RuntimeError('Fortran error! mpi_init() could not be located!')
    # check if mpi.mod exists
    if self.compilers.fortranIsF90:
      self.framework.log.write('Checking for mpi.mod\n')
      if self.libraries.check(self.lib,'', call = '       use mpi\n       integer ierr\n       call mpi_init(ierr)'):
        self.havef90module = 1
        self.addDefine('HAVE_MPI_F90MODULE', 1)
    self.compilers.CPPFLAGS = oldFlags
    self.libraries.popLanguage()
    return 0

  def configureIO(self):
    '''Check for the functions in MPI/IO
       - Define HAVE_MPIIO if they are present
       - Some older MPI 1 implementations are missing these'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    if not self.checkLink('#include <mpi.h>\n', 'MPI_Aint lb, extent;\nif (MPI_Type_get_extent(MPI_INT, &lb, &extent));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nvoid *buf;\nMPI_Status status;\nif (MPI_File_write_all(fh, buf, 1, MPI_INT, &status));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nvoid *buf;\nMPI_Status status;\nif (MPI_File_read_all(fh, buf, 1, MPI_INT, &status));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nMPI_Offset disp;\nMPI_Info info;\nif (MPI_File_set_view(fh, disp, MPI_INT, MPI_INT, "", info));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nMPI_Info info;\nif (MPI_File_open(MPI_COMM_SELF, "", 0, info, &fh));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nMPI_Info info;\nif (MPI_File_close(&fh));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    self.addDefine('HAVE_MPIIO', 1)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by MPI'''
    if 'with-'+self.package+'-shared' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
    self.addExtraLibraries()
    config.package.Package.configureLibrary(self)
    # Satish check here if the self.directory is truly the MPI root directory with mpicc underneath it
    # if not then set it to None

    self.executeTest(self.configureConversion)
    self.executeTest(self.configureMPI2)
    self.executeTest(self.configureTypes)
    self.executeTest(self.configureMPITypes)
    self.executeTest(self.configureMissingPrototypes)
    self.executeTest(self.SGIMPICheck)
    self.executeTest(self.CxxMPICheck)
    self.executeTest(self.FortranMPICheck)
    self.executeTest(self.configureIO)
    if self.libraries.check(self.dlib, "MPI_Alltoallw") and self.libraries.check(self.dlib, "MPI_Type_create_indexed_block"):
      self.addDefine('HAVE_MPI_ALLTOALLW',1)
    if self.libraries.check(self.dlib, "MPI_Comm_spawn"):
      self.addDefine('HAVE_MPI_COMM_SPAWN',1)
    if self.libraries.check(self.dlib, "MPI_Win_create"):
      self.addDefine('HAVE_MPI_WIN_CREATE',1)
    if self.libraries.check(self.dlib, "MPI_Finalized"):
      self.addDefine('HAVE_MPI_FINALIZED',1)

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
