#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import PETSc.package

import re
import os

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.headerPrefix  = ''
    self.substPrefix   = ''
    self.found         = 0
    self.compilers     = self.framework.require('config.compilers', self)
    self.libraries     = self.framework.require('config.libraries', self)
    self.sourceControl = self.framework.require('config.sourceControl', self)
    self.arch          = self.framework.require('PETSc.utilities.arch', self)
    self.mpi           = self.framework.require('PETSc.packages.MPI', self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.name         = 'Spai'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    if self.found:
      desc = ['Spai:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Spai', '-with-spai=<bool>',                nargs.ArgBool(None, 0, 'Activate Spai'))
    help.addArgument('Spai', '-with-spai-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the Spai installation'))
    help.addArgument('Spai', '-with-spai-include=<dir>',         nargs.ArgDir(None, None, 'The directory containing spai.h'))
    help.addArgument('Spai', '-with-spai-lib=<lib>',             nargs.Arg(None, None, 'The Spai library or list of libraries'))
    help.addArgument('Spai', '-download-spai=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 2, 'Install MPICH to provide Spai'))
    return

  def checkLib(self, libraries):
    '''Check for bspai libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = self.libraries.check(libraries, 'bspai', otherLibs = self.blasLapack.dlib)
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that spai.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in includeDir+self.mpi.include])
    found = self.checkPreprocess('#include <spai.h>\n')
    self.compilers.CPPFLAGS = oldFlags
    return found

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
      yield [os.path.join(root, 'lib', 'libspai.a')]
    else:
      yield ['']
      yield ['spai']
    return

  def generateGuesses(self):
    if 'with-spai-lib' in self.framework.argDB and 'with-spai-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Spai library with --with-spai-lib=<lib> and search directory with --with-spai-dir=<dir>')
    if self.framework.argDB['download-spai'] == 1:
      (name, lib, include) = self.downloadSpai()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Spai could not be used. Please check install in '+os.path.dirname(include[0][0])+'\n')
    # Try specified library and include
    if 'with-spai-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-spai-lib']
      if not isinstance(libs, list): libs = [libs]
      if 'with-spai-include' in self.framework.argDB:
        includes = [[self.framework.argDB['with-spai-include']]]
      else:
        includes = self.includeGuesses('\n'.join(map(lambda inc: os.path.dirname(os.path.dirname(inc)), libs)))
      yield ('User specified library and includes', [libs], includes)
      raise RuntimeError('You set a value for --with-spai-lib, but '+str(self.framework.argDB['with-spai-lib'])+' cannot be used.\n')
    # Try specified installation root
    if 'with-spai-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-spai-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      yield ('User specified installation root', self.libraryGuesses(dir), [[os.path.join(dir, 'lib')]])
      raise RuntimeError('You set a value for --with-spai-dir, but '+self.framework.argDB['with-spai-dir']+' cannot be used.\n')
    # May not need to list anything
    yield ('Default compiler locations', self.libraryGuesses(), [[]])
    # Try configure package directories
    dirExp = re.compile(r'((p|P)ar)?(m|M)etis(-.*)?')
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
    # Try /usr/local
    dir = os.path.abspath(os.path.join('/usr', 'local'))
    yield ('Frequent user install location (/usr/local)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try /usr/local/*spai*
    ls = os.listdir(os.path.join('/usr','local'))
    for dir in ls:
      if not dirExp.match(f):
        continue
      dir = os.path.join('/usr','local',dir)
      if os.path.isdir(dir):
        yield ('Frequent user install location (/usr/local/spai*)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try ~/spai*
    ls = os.listdir(os.getenv('HOME'))
    for dir in ls:
      if not dirExp.match(f):
        continue
      dir = os.path.join(os.getenv('HOME'),dir)
      if os.path.isdir(dir):
        yield ('Frequent user install location (~/spai*)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try PETSc location
    if self.arch.dir and self.arch.arch:
      pass
##      try:
##        libArgs = config.base.Configure.executeShellCommand('cd '+PETSC_DIR+'; make BOPT=g_c++ getmpilinklibs', log = self.framework.log)[0].strip()
##        incArgs = config.base.Configure.executeShellCommand('cd '+PETSC_DIR+'; make BOPT=g_c++ getmpiincludedirs', log = self.framework.log)[0].strip()
##        libArgs = self.splitLibs(libArgs)
##        incArgs = self.splitIncludes(incArgs)
##        if libArgs and incArgs:
##          yield ('PETSc location', [libArgs], [incArgs])
##      except RuntimeError:
##        # This happens with older Petsc versions which are missing those targets
##        pass
    # If necessary, download Spai
    if not self.found and self.framework.argDB['download-spai'] == 2:
      (name, lib, include) = self.downloadSpai()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Spai could not be used. Please check in install in '+os.path.dirname(include[0][0])+'\n')
    return

  def getDir(self):
    '''Find the directory containing Spai'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    spaiDir = None
    for dir in os.listdir(packages):
      if dir.startswith('Spai') and os.path.isdir(os.path.join(packages, dir)):
        spaiDir = dir
    if spaiDir is None:
      self.framework.logPrint('Could not locate already downloaded Spai')
      raise RuntimeError('Error locating Spai directory')
    return os.path.join(packages, spaiDir)

  def downloadSpai(self):
    self.framework.logPrint('Downloading Spai')
    try:
      spaiDir = self.getDir()
      self.framework.logPrint('Spai already downloaded, no need to ftp')
    except RuntimeError:
      import urllib

      packages = self.framework.argDB['with-external-packages-dir']
      if hasattr(self.sourceControl, 'bk'):
        self.logPrint("Retrieving Spai; this may take several minutes\n", debugSection='screen')
        config.base.Configure.executeShellCommand('bk clone bk://spai.bkbits.net/Spai-dev '+os.path.join(packages,'Spai'), log = self.framework.log, timeout= 600.0)
      else:
        try:
          self.logPrint("Retrieving Spai; this may take several minutes\n", debugSection='screen')
          urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/spai.tar.gz', os.path.join(packages, 'spai.tar.gz'))
        except Exception, e:
          raise RuntimeError('Error downloading Spai: '+str(e))
        try:
          config.base.Configure.executeShellCommand('cd '+packages+'; gunzip spai.tar.gz', log = self.framework.log)
        except RuntimeError, e:
          raise RuntimeError('Error unzipping spai.tar.gz: '+str(e))
        try:
          config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf spai.tar', log = self.framework.log)
        except RuntimeError, e:
          raise RuntimeError('Error doing tar -xf spai.tar: '+str(e))
        os.unlink(os.path.join(packages, 'spai.tar'))
      self.framework.actions.addArgument('Spai', 'Download', 'Downloaded Spai into '+self.getDir())
    # Get the Spai directories
    spaiDir = self.getDir()
    installDir = os.path.join(spaiDir, self.framework.argDB['PETSC_ARCH'])
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build Spai
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"', '-PETSC_DIR='+self.arch.dir]
    self.framework.popLanguage()
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild Spai oldargs = '+oldargs+' new args '+args+'\n')
      self.logPrint("Configuring and compiling Spai; this may take several minutes\n", debugSection='screen')
      try:
        import logging
        # Split Graphs into its own repository
        oldDir = os.getcwd()
        os.chdir(spaiDir)
        oldLog = logging.Logger.defaultLog
        logging.Logger.defaultLog = file(os.path.join(spaiDir, 'build.log'), 'w')
        oldLevel = self.argDB['debugLevel']
        #self.argDB['debugLevel'] = 0
        oldIgnore = self.argDB['ignoreCompileOutput']
        #self.argDB['ignoreCompileOutput'] = 1
        if os.path.exists('RDict.db'):
          os.remove('RDict.db')
        if os.path.exists('bsSource.db'):
          os.remove('bsSource.db')
        make = self.getModule(spaiDir, 'make').Make()
        make.prefix = installDir
        make.run()
        self.argDB['ignoreCompileOutput'] = oldIgnore
        self.argDB['debugLevel'] = oldLevel
        logging.Logger.defaultLog = oldLog
        os.chdir(oldDir)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Spai: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      self.framework.actions.addArgument('Spai', 'Install', 'Installed Spai into '+installDir)
    lib     = [[os.path.join(installDir, 'lib', 'libspai.a'), os.path.join(installDir, 'lib', 'libmetis.a')]]
    include = [[os.path.join(installDir, 'include')]]
    return ('Downloaded Spai', lib, include)

  def configureVersion(self):
    '''Determine the Spai version, but there is no reliable way right now'''
    return 'Unknown'

  def configureLibrary(self):
    '''Find all working Spai installations and then choose one'''
    functionalSpai = []
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.logPrint('================================================================================')
      self.framework.logPrint('Checking for a functional Spai in '+name)
      self.lib     = None
      self.include = None
      found        = 0
      for libraries in libraryGuesses:
        if self.checkLib(libraries):
          self.lib = libraries
          for includeDir in includeGuesses:
            if self.checkInclude(includeDir):
              self.include = includeDir
              found = 1
              break
          if found:
            break
      if not found: continue
      version = self.executeTest(self.configureVersion)
      self.found = 1
      functionalSpai.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalSpai[0]
      self.framework.logPrint('Choose Spai '+self.version+' in '+self.name)
    else:
      self.framework.logPrint('Could not locate any functional Spai')
    return

  def configure(self):
    if (self.framework.argDB['with-spai'] or self.framework.argDB['download-spai'] == 1):
      if self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.framework.argDB['with-64-bit-ints']:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')
      self.executeTest(self.configureLibrary)
      self.framework.packages.append(self)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
