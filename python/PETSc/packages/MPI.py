from __future__ import generators
import config.base

import os
import os.path
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.foundMPI     = 0
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.argDB        = framework.argDB
    self.foundLib     = 0
    self.foundInclude = 0
    self.isPOE        = 0
    self.compilers    = self.framework.require('config.compilers', self)
    self.setcompilers = self.framework.require('config.setCompilers', self)    
    self.types        = self.framework.require('config.types',     self)
    self.libraries    = self.framework.require('config.libraries', self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    self.name         = 'MPI'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    self.usingMPIUni  = 0
    return

  def __str__(self):
    if self.foundMPI:
      desc = ['MPI:']	
      desc.append('  Type: '+self.name)
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('MPI', '-with-mpi=<bool>',                   nargs.ArgBool(None, 1, 'Activate MPI'))
    help.addArgument('MPI', '-with-mpi-dir=<root dir>',           nargs.ArgDir(None, None, 'Specify the root directory of the MPI installation'))
    help.addArgument('MPI', '-with-mpi-include=<dir>',            nargs.ArgDir(None, None, 'The directory containing mpi.h'))
    help.addArgument('MPI', '-with-mpi-lib=<lib>',                nargs.Arg(None, None, 'The MPI library or list of libraries'))
    help.addArgument('MPI', '-with-mpirun=<prog>',                nargs.Arg(None, None, 'The utility used to launch MPI jobs'))
    help.addArgument('MPI', '-with-mpi-shared=<bool>',            nargs.ArgBool(None, 0, 'Require that the MPI library be shared'))
    help.addArgument('MPI', '-with-mpi-compilers=<bool>',         nargs.ArgBool(None, 1, 'Try to use the MPI compilers, e.g. mpicc'))
    help.addArgument('MPI', '-download-mpich=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install MPICH to provide MPI'))
    help.addArgument('MPI', '-download-mpich-machines=[machine1,machine2...]',  nargs.Arg(None, ['localhost','localhost'], 'Machines for MPI to use'))
    help.addArgument('MPI', '-download-mpich-device=ch_p4 or ch_shmem',         nargs.Arg(None, 'ch_p4', 'Device for MPI'))    
    return

  def checkLib(self, libraries):
    '''Check for MPI_Init and MPI_Comm_create in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = self.libraries.check(libraries, 'MPI_Init', otherLibs = self.compilers.flibs)
    if found:
      found = self.libraries.check(libraries, 'MPI_Comm_create',otherLibs = self.compilers.flibs)
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that mpi.h is present'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in includeDir])
    # Should we also look for mpif.h?
    found = self.checkPreprocess('#include <mpi.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return found

  def checkMPILink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None):
    '''Analogous to checkLink(), but the MPI includes and libraries are automatically provided'''
    success  = 0
    oldFlags = self.framework.argDB['CPPFLAGS']
    oldLibs  = self.framework.argDB['LIBS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in self.include])
    self.framework.argDB['LIBS'] = self.libraries.toString(self.lib+self.compilers.flibs)+' '+self.framework.argDB['LIBS']
    if self.checkLink(includes, body, cleanup, codeBegin, codeEnd):
      success = 1
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return success

  def outputMPIRun(self, includes, body, cleanup = 1, defaultOutputArg = ''):
    '''Analogous to outputRun(), but the MPI includes and libraries are automatically provided'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    oldLibs  = self.framework.argDB['LIBS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in self.include])
    self.framework.argDB['LIBS'] = self.libraries.toString(self.lib+self.compilers.flibs)+' '+self.framework.argDB['LIBS']
    output, status = self.outputRun(includes, body, cleanup, defaultOutputArg)
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return (output, status)

  def checkWorkingLink(self):
    '''Checking that we can link an MPI executable'''
    if not self.checkMPILink('#include <mpi.h>\n', 'MPI_Comm comm = MPI_COMM_WORLD;\nint size;\n\nMPI_Comm_size(comm, &size);\n'):
      self.framework.log.write('MPI cannot link, which indicates a problem with the MPI installation\n')
      return 0
    self.framework.log.write('MPI can link with C\n')
      
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('C++')
      self.sourceExtension = '.C'
      if not self.checkMPILink('#include <mpi.h>\n', 'MPI_Comm comm = MPI_COMM_WORLD;\nint size;\n\nMPI_Comm_size(comm, &size);\n'):
        self.framework.log.write('MPI cannot link C++ but can link C, which indicates a problem with the MPI installation\n')
        self.popLanguage()
        return 0
      self.popLanguage()
      self.framework.log.write('MPI can link with C++\n')
    
    if 'FC' in self.framework.argDB:
      self.pushLanguage('FC')
      self.sourceExtension = '.F'
      if not self.checkMPILink('', '          integer comm,size,ierr\n          call MPI_Comm_size(comm, size, ierr)\n'):
        if not self.checkMPILink('', '          include \'mpif.h\'\n          integer comm,size,ierr\n          call MPI_Comm_size(comm, size, ierr)\n'):
          self.framework.log.write('MPI cannot link Fortran using MPI_Comm_size(), but can link C, which indicates a problem with the MPI installation\nRun with -with-fc=0 if you do not wish to use Fortran')
          self.popLanguage()
          return 0
      if not self.checkMPILink('', '          call MPI_Init(ierr)\n'):
        if not self.checkMPILink('', '          include \'mpif.h\'\n          call MPI_Init(ierr)\n'):
          self.framework.log.write('MPI cannot link Fortran using MPI_Init((), but can link C, which indicates a problem with the MPI installation\nRun with -with-fc=0 if you do not wish to use Fortran')
          self.popLanguage()
          return 0
      if not self.checkMPILink('', '          include \'mpif.h\'\n          call MPI_Init(ierr)\n'):
        self.framework.log.write('MPI cannot locate Fortran includes, but can find the MPI libraries\n')
        if not self.include:
          # get around bug in g77 on SGI IA64 systems
          self.include.append(os.path.join('/usr', 'include'))
          if not self.checkMPILink('', '          include \'mpif.h\'\n          call MPI_Init(ierr)\n'):
            self.include.pop()
            self.popLanguage()
            self.framework.log.write('Adding /usr/include to search path did not fix MPI includes from Fortran. Failed\n')
            return 0
        else:
          self.popLanguage()
          return 0
        self.framework.log.write('Added /usr/include to search path to find MPI includes from Fortran. Succeeded\n')
      self.popLanguage()
      self.framework.log.write('MPI can link with Fortran\n')
    return 1

  def checkSharedLibrary(self):
    '''Check that the libraries for MPI are shared libraries'''
    return self.libraries.checkShared('#include <mpi.h>\n', 'MPI_Init', 'MPI_Initialized', 'MPI_Finalize', checkLink = self.checkMPILink, libraries = self.lib)

  def configureVersion(self):
    '''Determine the MPI version'''
    if self.framework.argDB['can-execute']:
      output, status = self.outputMPIRun('#include <stdio.h>\n#include <mpi.h>\n', 'int ver, subver;\n if (MPI_Get_version(&ver, &subver));\nprintf("%d.%d\\n", ver, subver)\n')
      if not status:
        # need to strip out information from batch system
        f = re.match('([0-9]*.[0-9]*)',output)
        if not f: return 'Unknown'
        return f.group()
    return 'Unknown'

  def includeGuesses(self, path):
    '''Return all include directories present in path or its ancestors'''
    while path:
      dir = os.path.join(path, 'include')
      if os.path.isdir(dir):
        yield [dir]
      path = os.path.dirname(path)
    return

  def libraryGuesses(self, root = None):
    '''Return standard library name guesses for a given installation root'''
    if root:
      yield [os.path.join(root, 'lib', 'shared', 'libfmpich.a'),os.path.join(root,'lib','shared','libmpich.a'),os.path.join(root,'lib','libmpichfarg.a')]
      yield [os.path.join(root, 'lib', 'shared', 'libmpich.a')]
      yield [os.path.join(root, 'lib', 'shared', 'libmpi.a')]
      yield [os.path.join(root, 'lib', 'shared', 'libmpich.a'), os.path.join(root, 'lib', 'shared', 'libpmpich.a')]
      yield [os.path.join(root, 'lib', 'libfmpich.a'), os.path.join(root,'lib','libmpich.a')]
      yield [os.path.join(root, 'lib', 'libmpich.a')]
      # LAM
      libs = [os.path.join(root, 'lib', 'liblammpio.a'),os.path.join(root, 'lib', 'libpmpi.a'),os.path.join(root, 'lib', 'liblamf77mpi.a'),os.path.join(root, 'lib', 'libmpi.a'),os.path.join(root, 'lib', 'liblam.a')]
      libs = [os.path.join(root, 'lib', 'liblammpio.a'),os.path.join(root, 'lib', 'libpmpi.a'),os.path.join(root, 'lib', 'liblamf90mpi.a'),os.path.join(root, 'lib', 'libmpi.a'),os.path.join(root, 'lib', 'liblam.a')]      
      libs = [os.path.join(root, 'lib', 'liblammpio.a'),os.path.join(root, 'lib', 'libpmpi.a'),os.path.join(root, 'lib', 'libmpi.a'),os.path.join(root, 'lib', 'liblam.a')]
      yield libs
      #  SGI 
      yield [os.path.join(root, 'lib', 'libmpi.a'),os.path.join(root, 'lib', 'libmpi++.a')]
      yield [os.path.join(root, 'lib', 'libmpi.a')]      
      yield [os.path.join(root, 'lib', 'libmpich.a'), os.path.join(root, 'lib', 'libpmpich.a')]
      yield [os.path.join(root, 'lib', 'mpich.lib'),'ws2_32.lib']
      # MS Windows
      yield [os.path.join(root, 'lib','mpich.lib'),'ws2_32.lib']
    else:
      yield ['']
      yield ['fmpich','mpich']
      yield ['mpich']
      yield ['mpi','mpi++']
      yield ['mpi']      
      yield ['mpich', 'pmpich']
      yield ['lammpio','pmpi','lamf77mpi','mpi','lam']
      yield ['lammpio','pmpi','lamf90mpi','mpi','lam']
      yield ['lammpio','pmpi','mpi','lam']
    return

  def generateGuesses(self):
    if 'with-mpi-lib' in self.framework.argDB and 'with-mpi-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH MPI library with --with-mpi-lib=<lib> and search directory with --with-mpi-dir=<dir>')
    if self.framework.argDB['download-mpich'] == 1:
      (name, lib, include) = self.downLoadMPICH()
      yield (name, lib, include)
      raise RuntimeError('Downloaded MPICH could not be used.\n')
    # Try specified library and include
    if 'with-mpi-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-mpi-lib']
      if not isinstance(libs, list): libs = [libs]
      if 'with-mpi-include' in self.framework.argDB:
        includes = [[self.framework.argDB['with-mpi-include']]]
      else:
        includes = self.includeGuesses('\n'.join(map(lambda inc: os.path.dirname(os.path.dirname(inc)), libs)))
      yield ('User specified library and includes', [libs], includes)
      raise RuntimeError('You set a value for --with-mpi-lib, but '+str(self.framework.argDB['with-mpi-lib'])+' cannot be used.\n It could be the MPI located is not working for all the languages, you can try running\n configure again with --with-fc=0 or --with-cxx=0\n')
    # Try specified installation root
    if 'with-mpi-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-mpi-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      yield ('User specified installation root', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
      yield ('User specified installation root for cygwin', self.libraryGuesses(os.path.join(dir,'SDK.gcc')),[[os.path.join(dir,'SDK.gcc','include')]])
      yield ('User specified installation root for MS Windows', self.libraryGuesses(os.path.join(dir,'SDK')),[[os.path.join(dir,'SDK','include')]])
      raise RuntimeError('You set a value for --with-mpi-dir, but '+self.framework.argDB['with-mpi-dir']+' cannot be used.\n It could be the MPI located is not working for all the languages, you can try running\n configure again with --with-fc=0 or --with-cxx=0\n')
    # May not need to list anything
    yield ('Default compiler locations', self.libraryGuesses(), [[]])
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
        yield ('Package directory installation root', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try SUSE location
    dir = os.path.abspath(os.path.join('/opt', 'mpich'))
    yield ('Default SUSE location', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try IBM
    self.isPOE = 1
    dir = os.path.abspath(os.path.join('/usr', 'lpp', 'ppe.poe'))
    yield ('IBM location (/usr/lpp/ppe.poe)', [[os.path.join(dir, 'lib', 'libmpi_r.a')]], [[os.path.join(dir, 'include')]])
    self.isPOE = 0
    # Try /usr/local
    dir = os.path.abspath(os.path.join('/usr', 'local'))
    yield ('Frequent user install location (/usr/local)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try /usr/local/*mpich*
    ls = os.listdir(os.path.join('/usr','local'))
    for dir in ls:
      if dir.find('mpich') >= 0:
        dir = os.path.join('/usr','local',dir)
        if os.path.isdir(dir):
          yield ('Frequent user install location (/usr/local/*mpich*)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try ~/mpich*
    ls = os.listdir(os.getenv('HOME'))
    for dir in ls:
      if dir.find('mpich') >= 0:
        dir = os.path.join(os.getenv('HOME'),dir)
        if os.path.isdir(dir):
          yield ('Frequent user install location (~/*mpich*)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try MPICH install locations under Windows
    dir = os.path.join('/cygdrive','c','Program\\ Files','MPICH')
    yield('Default MPICH install location (C:\Program Files\MPICH with MS compatible SDK',self.libraryGuesses(os.path.join(dir,'SDK')),[[os.path.join(dir,'SDK','include')]])
    yield('Default MPICH install location (C:\Program Files\MPICH with SDK.gcc',self.libraryGuesses(os.path.join(dir,'SDK.gcc')),[[os.path.join(dir,'SDK.gcc','include')]])
    dir = os.path.join('c:','Program\\ Files','MPICH')
    yield('Default MPICH install location (C:\Program Files\MPICH with MS compatible SDK',self.libraryGuesses(os.path.join(dir,'SDK')),[[os.path.join(dir,'SDK','include')]])
    yield('Default MPICH install location (C:\Program Files\MPICH with SDK.gcc',self.libraryGuesses(os.path.join(dir,'SDK.gcc')),[[os.path.join(dir,'SDK.gcc','include')]])
    
    # If necessary, download MPICH
    if not self.foundMPI and self.framework.argDB['download-mpich'] == 2:
      (name, lib, include) = self.downLoadMPICH()
      yield (name, lib, include)
      raise RuntimeError('Downloaded MPICH could not be used. Please check in install in '+os.path.dirname(include)+'\n')
    return

  def getDir(self):
    '''Find the directory containing MPICH'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    mpichDir = None
    for dir in os.listdir(packages):
      if dir.startswith('mpich') and os.path.isdir(os.path.join(packages, dir)):
        mpichDir = dir
    if mpichDir is None:
      self.framework.log.write('Could not locate already downloaded MPICH\n')
      raise RuntimeError('Error locating MPICH directory')
    return os.path.join(packages, mpichDir)

  def downLoadMPICH(self):
    self.framework.log.write('Downloading MPICH\n')
    try:
      mpichDir = self.getDir()
      self.framework.log.write('MPICH already downloaded, no need to ftp\n')
    except RuntimeError:
      import urllib

      packages = self.framework.argDB['with-external-packages-dir']
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/mpi/mpich.tar.gz', os.path.join(packages, 'mpich.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading MPICH: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip mpich.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping mpich.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf mpich.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf mpich.tar: '+str(e))
      os.unlink(os.path.join(packages, 'mpich.tar'))
      self.framework.actions.addArgument('MPI', 'Download', 'Downloaded MPICH into '+self.getDir())
    # Get the MPICH directories
    mpichDir = self.getDir()
    installDir = os.path.join(mpichDir, self.framework.argDB['PETSC_ARCH'])
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
      
    # Configure and Build MPICH
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    if 'CXX' in self.framework.argDB:
      self.framework.pushLanguage('Cxx')
      args.append('-c++="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      args.append('--disable-c++')
    if 'FC' in self.framework.argDB:
      self.framework.pushLanguage('FC')      
      fc = self.framework.getCompiler()
      if fc.find('f90') >= 0:
        import commands
        output  = commands.getoutput(fc+' -v')
        if output.find('IBM') >= 0:
          fc = os.path.join(os.path.dirname(fc),'xlf')
          self.framework.log.write('Using IBM f90 compiler for PETSc, switching to xlf for compiling MPICH\n')      
      args.append('-fc="'+fc+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      args.append('--disable-f77 --disable-f90')
    args.append('--without-mpe')
    args.append('-rsh=ssh')
    args.append('--with-device='+self.argDB['download-mpich-device'])
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild MPICH oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Running configure on MPICH; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+mpichDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on MPICH: '+str(e))
      try:
        self.logPrint("Running make on MPICH; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+mpichDir+';make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on MPICH: '+str(e))

      # put the list of machines in 
      ls = os.listdir(os.path.join(installDir,'share'))
      for l in ls:
        if l.startswith('machines.') and not l.endswith('.sample'):
          fd = file(os.path.join(installDir,'share',l),'w')
          for i in self.argDB['download-mpich-machines']:
            fd.write(i+'\n')
          fd.close()
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      self.framework.actions.addArgument('MPI', 'Install', 'Installed MPICH into '+installDir)
    
    # 
    lib     = [[os.path.join(installDir, 'lib', 'libmpich.a'), os.path.join(installDir, 'lib', 'libpmpich.a')]]
    include = [[os.path.join(installDir, 'include')]]
    return ('Downloaded MPICH', lib, include)

  def fixSolaris(self):
    '''I hate this. MPI should report this somehow.'''
    if self.framework.argDB['PETSC_ARCH_BASE'].startswith('solaris'):
      self.executeTest(self.libraries.check, [['rt', 'nsl', 'aio'], 'exit'])
    return

  def configureLibrary(self):
    '''Find all working MPI libraries and then choose one'''
    functionalMPI = []
    nonsharedMPI  = []

    self.fixSolaris()
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.log.write('================================================================================\n')
      self.framework.log.write('Checking for a functional MPI in '+name+'\n')
      self.lib     = None
      self.include = None
      found        = 0
      for libraries in libraryGuesses:
        if self.checkLib(libraries):
          self.lib = libraries
          for includeDir in includeGuesses:
            if self.checkInclude(includeDir):
              self.include = includeDir
              if self.executeTest(self.checkWorkingLink):
                found = 1
                break
          if found:
            break
      if not found: continue

      version = self.executeTest(self.configureVersion)
      if self.framework.argDB['with-mpi-shared']:
        if not self.executeTest(self.checkSharedLibrary):
          nonsharedMPI.append((name, self.lib, self.include, version))
          continue
      self.foundMPI = 1
      functionalMPI.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.foundMPI:
      self.name, self.lib, self.include, self.version = functionalMPI[0]
      self.dlib = self.lib + self.compilers.flibs
      if hasattr(self.framework,'packages'):
        self.framework.packages.append(self)
      self.framework.log.write('Choose MPI '+self.version+' in '+self.name+'\n')
    elif len(nonsharedMPI):
      raise RuntimeError('Could not locate any MPI with shared libraries')
    else:
      raise RuntimeError('Could not locate any functional MPI\n Rerun config/configure.py with --with-mpi=0 to install without MPI\n or -with-mpi-dir=directory to use a MPICH installation\n or -with-mpi-include=directory -with-mpi-lib=library (or list of libraries) -with-mpirun=mpiruncommand for other MPIs.\n Or run with --download-mpich=1 to have PETSc download and compile MPICH for you.\nIt could be the MPI located is not working for all the languages, you can try running\n configure again with --with-fc=0 or --with-cxx=0\n')
    return

  def configureConversion(self):
    '''Check for the functions which convert communicators between C and Fortran
       - Define HAVE_MPI_COMM_F2C and HAVE_MPI_COMM_C2F if they are present
       - Some older MPI 1 implementations are missing these'''
    if self.checkMPILink('#include <mpi.h>\n', 'if (MPI_Comm_f2c(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_F2C', 1)
    if self.checkMPILink('#include <mpi.h>\n', 'if (MPI_Comm_c2f(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_C2F', 1)
    if self.checkMPILink('#include <mpi.h>\n', 'MPI_Fint a;\n'):
      self.addDefine('HAVE_MPI_FINT', 1)
    return

  def configureTypes(self):
    '''Checking for MPI types'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in self.include])
    self.types.checkSizeof('MPI_Comm', 'mpi.h')
    if 'HAVE_MPI_FINT' in self.defines:
      self.types.checkSizeof('MPI_Fint', 'mpi.h')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return

  def configureMPIRUN(self):
    '''Checking for mpirun'''
    if self.isPOE:
      self.mpirun = os.path.join(self.framework.argDB['PETSC_DIR'], 'bin', 'mpirun.poe')
      return
    if 'with-mpirun' in self.framework.argDB:
      self.framework.argDB['with-mpirun'] = os.path.expanduser(self.framework.argDB['with-mpirun'])
      if not self.getExecutable(self.framework.argDB['with-mpirun'], resultName = 'mpirun'):
        raise RuntimeError('Invalid mpirun specified: '+str(self.framework.argDB['with-mpirun']))
      return
    mpiruns = ['mpirun', 'mpiexec']
    if 'with-mpi-dir' in self.framework.argDB:
      if self.getExecutables(mpiruns, path = [os.path.join(os.path.abspath(self.framework.argDB['with-mpi-dir']), 'bin')], resultName = 'mpirun'):
        return
    path = []
    for inc in self.include:
      path.append(os.path.join(os.path.dirname(inc), 'bin'))
    for lib in self.lib:
      path.append(os.path.join(os.path.dirname(os.path.dirname(lib)), 'bin'))
    self.pushLanguage('C')
    if os.path.basename(self.getCompiler()) == 'mpicc' and os.path.dirname(self.getCompiler()):
      path.append(os.path.dirname(self.getCompiler()))
    self.popLanguage()
    self.getExecutables(mpiruns, path = path, useDefaultPath = 1, resultName = 'mpirun')
    return

  def configureMPIUNI(self):
    '''Setup MPIUNI, our uniprocessor version of MPI'''
    self.framework.addDefine('HAVE_MPI', 1)
    self.include = ['-I${PETSC_DIR}/include/mpiuni']
    if 'STDCALL' in self.compilers.defines:
      self.include.append(' -DMPIUNI_USE_STDCALL')
    self.lib = ['-L${PETSC_DIR}/lib/${PETSC_ARCH}','-lmpiuni']
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


  def configure(self):
    if not self.framework.argDB['with-mpi']:
      self.configureMPIUNI()
    else:
      self.executeTest(self.configureLibrary)
      self.executeTest(self.configureConversion)
      self.executeTest(self.configureTypes)
      self.executeTest(self.configureMPIRUN)
      self.executeTest(self.configureMissingPrototypes)      
    return
