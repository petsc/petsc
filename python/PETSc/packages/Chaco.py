#!/usr/bin/env python
from __future__ import generators
import user
import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.found        = 0
    self.compilers    = self.framework.require('config.compilers', self)
    self.libraries    = self.framework.require('config.libraries', self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    self.mpi          = self.framework.require('PETSc.packages.MPI', self)
    self.name         = 'Chaco'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    if self.found:
      desc = ['Chaco:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Chaco', '-with-chaco=<bool>',                nargs.ArgBool(None, 0, 'Activate Chaco'))
    help.addArgument('Chaco', '-with-chaco-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the Chaco installation'))
    help.addArgument('Chaco', '-with-chaco-lib=<lib>',             nargs.Arg(None, None, 'The Chaco library or list of libraries'))
    help.addArgument('Chaco', '-download-chaco=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 2, 'Install MPICH to provide Chaco'))
    return

  def checkLib(self, libraries):
    '''Check for interface in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = self.libraries.check(libraries, 'interface', otherLibs = self.mpi.lib)
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def libraryGuesses(self, root = None):
    '''Return standard library name guesses for a given installation root'''
    if root:
      yield [os.path.join(root, 'lib', 'libchaco.a')]
    else:
      yield ['']
      yield ['chaco']
    return

  def generateGuesses(self):
    if 'with-chaco-lib' in self.framework.argDB and 'with-chaco-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Chaco library with --with-chaco-lib=<lib> and search directory with --with-chaco-dir=<dir>')
    if self.framework.argDB['download-chaco'] == 1:
      (name, lib) = self.downloadChaco()
      yield (name, lib)
      raise RuntimeError('Downloaded Chaco could not be used. Please check install in '+os.path.dirname(include[0][0])+'\n')
    # Try specified library 
    if 'with-chaco-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-chaco-lib']
      if not isinstance(libs, list): libs = [libs]
      yield ('User specified library', [libs])
      raise RuntimeError('You set a value for --with-chaco-lib, but '+str(self.framework.argDB['with-chaco-lib'])+' cannot be used.\n')
    # Try specified installation root
    if 'with-chaco-dir' in self.framework.argDB:   #~Chaco-2.2/$PETSC_ARCH
      dir = self.framework.argDB['with-chaco-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      yield ('User specified installation root', self.libraryGuesses(dir))
      raise RuntimeError('You set a value for --with-chaco-dir, but '+self.framework.argDB['with-chaco-dir']+' cannot be used.\n')
    # May not need to list anything  ???
    yield ('Default compiler locations', self.libraryGuesses(), [[]])
    # If necessary, download Chaco
    if not self.found and self.framework.argDB['download-chaco'] == 2:
      (name, lib) = self.downloadChaco()
      yield (name, lib)
      raise RuntimeError('Downloaded Chaco could not be used. Please check in install in '+os.path.dirname(include[0][0])+'\n')
    return

  def getDir(self):
    '''Find the directory containing Chaco'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    chacoDir = None
    for dir in os.listdir(packages):
      if dir.startswith('Chaco') and os.path.isdir(os.path.join(packages, dir)):
        chacoDir = dir
    if chacoDir is None:
      self.framework.logPrint('Could not locate already downloaded Chaco')
      raise RuntimeError('Error locating Chaco directory')
    return os.path.join(packages, chacoDir)

  def downloadChaco(self):
    self.framework.logPrint('Downloading Chaco')
    try:
      chacoDir = self.getDir()  #~Chaco-2.2
      self.framework.logPrint('Chaco already downloaded, no need to ftp')
    except RuntimeError:
      import urllib
      packages = self.framework.argDB['with-external-packages-dir']
      try:
        self.logPrint("Retrieving Chaco; this may take several minutes\n", debugSection='screen')
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/Chaco-2.2.tar.gz', os.path.join(packages, 'Chaco-2.2.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Chaco: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip Chaco-2.2.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping Chaco-2.2.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf Chaco-2.2.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf Chaco-2.2.tar: '+str(e))
      os.unlink(os.path.join(packages, 'Chaco-2.2.tar'))
      self.framework.actions.addArgument('Chaco', 'Download', 'Downloaded Chaco into '+self.getDir())
    # Get the Chaco directories
    chacoDir = self.getDir()  #~Chaco-2.2
    installDir = os.path.join(chacoDir, self.framework.argDB['PETSC_ARCH']) 
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build Chaco
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"', '-PETSC_DIR='+self.arch.dir]
    self.framework.popLanguage()
    if not 'FC' in self.framework.argDB:
      args.append('--with-fc=0')
    argsStr = ' '.join(args)
    try:
      fd         = file(os.path.join(installDir,'config.args'))
      oldArgsStr = fd.readline()
      fd.close()
    except:
      oldArgsStr = ''
    if not oldArgsStr == argsStr:
      self.framework.log.write('Have to rebuild Chaco oldargs = '+oldArgsStr+' new args '+argsStr+'\n')
      self.logPrint("Configuring and compiling Chaco; this may take several minutes\n", debugSection='screen')
      try:
        import logging
        # Split Graphs into its own repository
        oldDir = os.getcwd()
        os.chdir(chacoDir)
        oldLog = logging.Logger.defaultLog
        logging.Logger.defaultLog = file(os.path.join(chacoDir, 'build.log'), 'w')
        oldLevel = self.argDB['debugLevel']
        #self.argDB['debugLevel'] = 0
        oldIgnore = self.argDB['ignoreCompileOutput']
        #self.argDB['ignoreCompileOutput'] = 1
        if os.path.exists('RDict.db'):
          os.remove('RDict.db')
        if os.path.exists('bsSource.db'):
          os.remove('bsSource.db')
        self.executeShellCommand('cd code; make')
        os.chdir(installDir)
        self.executeShellCommand('mkdir lib; ar cr lib/libchaco.a `find ../code -name "*.o"`')
        self.executeShellCommand('cd lib; ar d libchaco.a main.o')
        self.argDB['ignoreCompileOutput'] = oldIgnore
        self.argDB['debugLevel'] = oldLevel
        logging.Logger.defaultLog = oldLog
        os.chdir(oldDir)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Chaco: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(argsStr)
      fd.close()
      self.framework.actions.addArgument('Chaco', 'Install', 'Installed Chaco into '+installDir)
    lib     = [[os.path.join(installDir, 'lib', 'libchaco.a')]]
    return ('Downloaded Chaco', lib)

  def configureVersion(self):
    '''Determine the Chaco version, but there is no reliable way right now'''
    return 'Unknown'

  def configureLibrary(self):
    '''Find all working Chaco installations and then choose one'''
    functionalChaco = []
    for (name, libraryGuesses) in self.generateGuesses():
      self.framework.logPrint('================================================================================')
      self.framework.logPrint('Checking for a functional Chaco in '+name)
      self.lib     = None
      self.include = None
      found        = 0
      for libraries in libraryGuesses:
        if self.checkLib(libraries):
          self.lib = libraries
          found = 1  
          break
      if not found: continue
      version = self.executeTest(self.configureVersion)
      self.found = 1
      functionalChaco.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalChaco[0]
      self.framework.logPrint('Choose Chaco '+self.version+' in '+self.name)
    else:
      self.framework.logPrint('Could not locate any functional Chaco')
    return

  def configure(self):
    if (self.framework.argDB['with-chaco'] or self.framework.argDB['download-chaco'] == 1):
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
