#!/usr/bin/env python
from __future__ import generators
import user
import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix  = ''
    self.substPrefix   = ''
    self.found         = 0
    self.compilers     = self.framework.require('config.compilers', self)
    self.libraries     = self.framework.require('config.libraries', self)
    self.sourceControl = self.framework.require('config.sourceControl', self)
    self.arch          = self.framework.require('PETSc.utilities.arch', self)
    self.mpi           = self.framework.require('PETSc.packages.MPI', self)
    self.name         = 'ParMetis'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    if self.found:
      desc = ['ParMetis:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('ParMetis', '-with-parmetis=<bool>',                nargs.ArgBool(None, 0, 'Activate ParMetis'))
    help.addArgument('ParMetis', '-with-parmetis-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the ParMetis installation'))
    help.addArgument('ParMetis', '-with-parmetis-include=<dir>',         nargs.ArgDir(None, None, 'The directory containing metis.h'))
    help.addArgument('ParMetis', '-with-parmetis-lib=<lib>',             nargs.Arg(None, None, 'The ParMetis library or list of libraries'))
    help.addArgument('ParMetis', '-download-parmetis=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 2, 'Install MPICH to provide ParMetis'))
    return

  def checkLib(self, libraries):
    '''Check for ParMetis_Init in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = self.libraries.check(libraries, 'ParMETIS_V3_PartKway', otherLibs = self.mpi.lib+self.libraries.math)
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that parmetis.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in includeDir+self.mpi.include])
    found = self.checkPreprocess('#include <parmetis.h>\n')
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
      yield [os.path.join(root, 'lib', 'libparmetis.a'), os.path.join(root, 'lib', 'libmetis.a')]
    else:
      yield ['']
      yield ['parmetis', 'metis']
    return

  def generateGuesses(self):
    if 'with-parmetis-lib' in self.framework.argDB and 'with-parmetis-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH ParMetis library with --with-parmetis-lib=<lib> and search directory with --with-parmetis-dir=<dir>')
    if self.framework.argDB['download-parmetis'] == 1:
      (name, lib, include) = self.downloadParMetis()
      yield (name, lib, include)
      raise RuntimeError('Downloaded ParMetis could not be used. Please check install in '+os.path.dirname(include[0][0])+'\n')
    # Try specified library and include
    if 'with-parmetis-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-parmetis-lib']
      if not isinstance(libs, list): libs = [libs]
      if 'with-parmetis-include' in self.framework.argDB:
        includes = [[self.framework.argDB['with-parmetis-include']]]
      else:
        includes = self.includeGuesses('\n'.join(map(lambda inc: os.path.dirname(os.path.dirname(inc)), libs)))
      yield ('User specified library and includes', [libs], includes)
      raise RuntimeError('You set a value for --with-parmetis-lib, but '+str(self.framework.argDB['with-parmetis-lib'])+' cannot be used.\n')
    # Try specified installation root
    if 'with-parmetis-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-parmetis-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      yield ('User specified installation root', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
      raise RuntimeError('You set a value for --with-parmetis-dir, but '+self.framework.argDB['with-parmetis-dir']+' cannot be used.\n')
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
    # Try /usr/local/*parmetis*
    ls = os.listdir(os.path.join('/usr','local'))
    for dir in ls:
      if not dirExp.match(f):
        continue
      dir = os.path.join('/usr','local',dir)
      if os.path.isdir(dir):
        yield ('Frequent user install location (/usr/local/parmetis*)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try ~/parmetis*
    ls = os.listdir(os.getenv('HOME'))
    for dir in ls:
      if not dirExp.match(f):
        continue
      dir = os.path.join(os.getenv('HOME'),dir)
      if os.path.isdir(dir):
        yield ('Frequent user install location (~/parmetis*)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try PETSc location
    if self.arch.dir and self.arch.arch:
      pass
    # If necessary, download ParMetis
    if not self.found and self.framework.argDB['download-parmetis'] == 2:
      (name, lib, include) = self.downloadParMetis()
      yield (name, lib, include)
      raise RuntimeError('Downloaded ParMetis could not be used. Please check in install in '+os.path.dirname(include[0][0])+'\n')
    return

  def getDir(self):
    '''Find the directory containing ParMetis'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    parmetisDir = None
    for dir in os.listdir(packages):
      if dir.startswith('ParMetis') and os.path.isdir(os.path.join(packages, dir)):
        parmetisDir = dir
    if parmetisDir is None:
      self.framework.logPrint('Could not locate already downloaded ParMetis')
      raise RuntimeError('Error locating ParMetis directory')
    return os.path.join(packages, parmetisDir)

  def downloadParMetis(self):
    self.framework.logPrint('Downloading ParMetis')
    try:
      parmetisDir = self.getDir()
      self.framework.logPrint('ParMetis already downloaded, no need to ftp')
    except RuntimeError:
      import urllib

      packages = self.framework.argDB['with-external-packages-dir']
      if hasattr(self.sourceControl, 'bk'):
        self.logPrint("Retrieving ParMetis; this may take several minutes\n", debugSection='screen')
        config.base.Configure.executeShellCommand('bk clone bk://parmetis.bkbits.net/ParMetis-dev '+os.path.join(packages,'ParMetis'), log = self.framework.log, timeout= 600.0)
      else:
        try:
          self.logPrint("Retrieving ParMetis; this may take several minutes\n", debugSection='screen')
          urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis.tar.gz', os.path.join(packages, 'parmetis.tar.gz'))
        except Exception, e:
          raise RuntimeError('Error downloading ParMetis: '+str(e))
        try:
          config.base.Configure.executeShellCommand('cd '+packages+'; gunzip parmetis.tar.gz', log = self.framework.log)
        except RuntimeError, e:
          raise RuntimeError('Error unzipping parmetis.tar.gz: '+str(e))
        try:
          config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf parmetis.tar', log = self.framework.log)
        except RuntimeError, e:
          raise RuntimeError('Error doing tar -xf parmetis.tar: '+str(e))
        os.unlink(os.path.join(packages, 'parmetis.tar'))
      self.framework.actions.addArgument('ParMetis', 'Download', 'Downloaded ParMetis into '+self.getDir())
    # Get the ParMetis directories
    parmetisDir = self.getDir()
    installDir = os.path.join(parmetisDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build ParMetis
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"', '-PETSC_DIR='+self.arch.dir]
    self.framework.popLanguage()
    if not 'FC' in self.framework.argDB:
      args.append('--with-fc=0')
    if not self.framework.argDB['with-shared']:
      args.append('--with-shared=0')
    args.extend(['--with-mpi-include='+str(self.mpi.include).replace(' ',''), '--with-mpi-lib='+str(self.mpi.lib).replace(' ','')])
    if self.framework.argDB['with-mpi-shared']:
      args.append('--with-mpi-shared')
    argsStr = ' '.join(args)
    try:
      fd         = file(os.path.join(installDir,'config.args'))
      oldArgsStr = fd.readline()
      fd.close()
    except:
      oldArgsStr = ''
    if not oldArgsStr == argsStr:
      self.framework.log.write('Have to rebuild ParMetis oldargs = '+oldArgsStr+' new args '+argsStr+'\n')
      self.logPrint("Configuring and compiling ParMetis; this may take several minutes\n", debugSection='screen')
      try:
        import logging
        # Split Graphs into its own repository
        oldDir = os.getcwd()
        os.chdir(parmetisDir)
        oldLog = logging.Logger.defaultLog
        logging.Logger.defaultLog = file(os.path.join(parmetisDir, 'build.log'), 'w')
        oldLevel = self.argDB['debugLevel']
        #self.argDB['debugLevel'] = 0
        oldIgnore = self.argDB['ignoreCompileOutput']
        #self.argDB['ignoreCompileOutput'] = 1
        if os.path.exists('RDict.db'):
          os.remove('RDict.db')
        if os.path.exists('bsSource.db'):
          os.remove('bsSource.db')
        make = self.getModule(parmetisDir, 'make').Make()
        make.prefix = installDir
        make.clArgs = map(lambda s: s.replace('"', ''), args)
        make.run()
        self.argDB['ignoreCompileOutput'] = oldIgnore
        self.argDB['debugLevel'] = oldLevel
        logging.Logger.defaultLog = oldLog
        os.chdir(oldDir)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on ParMetis: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(argsStr)
      fd.close()
      self.framework.actions.addArgument('ParMetis', 'Install', 'Installed ParMetis into '+installDir)
    lib     = [[os.path.join(installDir, 'lib', 'libparmetis.a'), os.path.join(installDir, 'lib', 'libmetis.a')]]
    include = [[os.path.join(installDir, 'include')]]
    return ('Downloaded ParMetis', lib, include)

  def configureVersion(self):
    '''Determine the ParMetis version, but there is no reliable way right now'''
    return 'Unknown'

  def configureLibrary(self):
    '''Find all working ParMetis installations and then choose one'''
    functionalParMetis = []
    if self.libraries.math is None:
      raise RuntimeError('ParMetis requires a math library')
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.logPrint('================================================================================')
      self.framework.logPrint('Checking for a functional ParMetis in '+name)
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
      functionalParMetis.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalParMetis[0]
      self.framework.logPrint('Choose ParMetis '+self.version+' in '+self.name)
    else:
      self.framework.logPrint('Could not locate any functional ParMetis')
    return

  def configure(self):
    if (self.framework.argDB['with-parmetis'] or self.framework.argDB['download-parmetis'] == 1):
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
