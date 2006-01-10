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
    return

  def __str__(self):
    if self.found:
      desc = ['Scotch:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Scotch', '-with-scotch=<bool>',                nargs.ArgBool(None, 0, 'Activate Scotch'))
    help.addArgument('Scotch', '-with-scotch-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the Scotch installation'))
    help.addArgument('Scotch', '-download-scotch=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Automatically install Scotch'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.petscdir       = framework.require('PETSc.utilities.petscdir', self)
    self.arch           = framework.require('PETSc.utilities.arch', self)
    self.compilers      = framework.require('config.compilers', self)
    self.headers        = framework.require('config.headers', self)
    self.libraries      = framework.require('config.libraries', self)
    self.mpi            = framework.require('PETSc.packages.MPI', self)
    self.libraryOptions = framework.require('PETSc.utilities.libraryOptions', self)
    return

  def checkLib(self, libraries):
    '''Check for SCOTCH_archBuild in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.compilers.LIBS
    found   = self.libraries.check(libraries, 'SCOTCH_archBuild', otherLibs = self.mpi.lib)
    self.compilers.LIBS = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that scotch.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(includeDir+self.mpi.include)
    found = self.checkPreprocess('#include <scotch.h>\n')
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
      yield [os.path.join(root,'libscotch.a'), os.path.join(root, 'libscotcherr.a'), os.path.join(root, 'libscotcherrcom.a'), os.path.join(root,'libcommon.a'), ]
    else:
      yield ['']
      yield ['scotch', 'scotcherr', 'scotcherrcom', 'common']
    return

  def generateGuesses(self):
    if self.framework.argDB['download-scotch'] == 1:
      (name, lib, include) = self.downloadScotch()
      yield (name, lib, include) 
      raise RuntimeError('Downloaded Scotch could not be used. Please check install in '+os.path.dirname(include[0][0])+'\n')
    # Try specified installation root
    if 'with-scotch-dir' in self.framework.argDB:  #~scotch_3.4
      dir = self.framework.argDB['with-scotch-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
        dir = os.path.join(dir, 'bin/i586_pc_linux2')
      yield ('User specified installation root', self.libraryGuesses(dir), [[dir]])
      raise RuntimeError('You set a value for --with-scotch-dir, but '+self.framework.argDB['with-scotch-dir']+' cannot be used.\n')
    # If necessary, download Scotch
    if not self.found and self.framework.argDB['download-scotch'] == 2:
      (name, lib, include) = self.downloadScotch()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Scotch could not be used. Please check in install in '+os.path.dirname(include[0][0])+'\n')
    return

  def getDir(self):
    '''Find the directory containing Scotch'''
    packages = self.petscdir.externalPackagesDir 
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    scotchDir = None
    for dir in os.listdir(packages):
      if dir.startswith('scotch_3.4') and os.path.isdir(os.path.join(packages, dir)):
        scotchDir = dir
    if scotchDir is None:
      self.framework.logPrint('Could not locate already downloaded Scotch')
      raise RuntimeError('Could not locate already downloaded Scotch')
    return os.path.join(packages, scotchDir)

  def downloadScotch(self):
    self.framework.logPrint('Downloading Scotch')
    try:
      scotchDir = self.getDir()
      self.framework.logPrint('Scotch already downloaded, no need to ftp')
    except RuntimeError:
      import urllib

      packages = self.petscdir.externalPackagesDir 
      try:
        self.logPrintBox('Retrieving Scotch; this may take several minutes')
        urllib.urlretrieve('http://www.labri.fr/Perso/~pelegrin/scotch/distrib/scotch_3.4.1A_i586_pc_linux2.tar.gz', os.path.join(packages, 'scotch_3.4.1A_i586_pc_linux2.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Scotch: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip scotch_3.4.1A_i586_pc_linux2.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping scotch_3.4.1A_i586_pc_linux2.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf scotch_3.4.1A_i586_pc_linux2.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf scotch_3.4.1A_i586_pc_linux2.tar: '+str(e))
      os.unlink(os.path.join(packages, 'scotch_3.4.1A_i586_pc_linux2.tar'))
    self.framework.actions.addArgument('Scotch', 'Download', 'Downloaded Scotch into '+self.getDir())
    # Get the Scotch directories
    scotchDir = self.getDir()  #~scotch_3.4
    installDir = os.path.join(scotchDir, 'bin/i586_pc_linux2') #~scotch_3.4/bin/i586_pc_linux2
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    lib = self.libraryGuesses(installDir) 
    include = [[installDir]]
    return ('Downloaded Scotch', lib, include)

  def configureVersion(self):
    '''Determine the Scotch version, but there is no reliable way right now'''
    return 'Unknown'

  def configureLibrary(self):
    '''Find all working Scotch installations and then choose one'''
    functionalScotch = []
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.logPrint('================================================================================')
      self.framework.logPrint('Checking for a functional Scotch in '+name)
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
      functionalScotch.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalScotch[0]
      self.framework.logPrint('Choose Scotch '+self.version+' in '+self.name)
    else:
      self.framework.logPrint('Could not locate any functional Scotch')
    return

  def configure(self):
    if (self.framework.argDB['with-scotch'] or self.framework.argDB['download-scotch'] == 1):
      if self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.libraryOptions.integerSize == 64:
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
