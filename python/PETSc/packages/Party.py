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
    self.name         = 'Party'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    if self.found:
      desc = ['Party:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Party', '-with-party=<bool>',                nargs.ArgBool(None, 0, 'Activate Party'))
    help.addArgument('Party', '-with-party-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the Party installation'))
    help.addArgument('Party', '-with-party-include=<dir>',         nargs.ArgDir(None, None, 'The directory containing party_lib.h'))
    help.addArgument('Party', '-with-party-lib=<lib>',             nargs.Arg(None, None, 'The Party library or list of libraries'))
    help.addArgument('Party', '-download-party=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Automatically install Party'))
    return

  def checkLib(self, libraries):
    '''Check for party_lib in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = self.libraries.check(libraries, 'party_lib')
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that party_lib.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in [includeDir]])
    found = self.checkPreprocess('#include <party_lib.h>\n')
    self.compilers.CPPFLAGS = oldFlags
    return found

  def generateGuesses(self):
    if 'with-party-lib' in self.framework.argDB and 'with-party-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Party library with --with-party-lib=<lib> and search directory with --with-party-dir=<dir>')
    if self.framework.argDB['download-party'] == 1:
      (name, lib, include) = self.downloadParty()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Party could not be used. Please check install in '+os.path.dirname(include[0][0])+'\n')
    # Try specified library (and include)
    if 'with-party-lib' in self.framework.argDB: #~PARTY_1.99/libparty.a
      libs = self.framework.argDB['with-party-lib'] #='~PARTY_1.99/libparty.a'
      if 'with-party-include' in self.framework.argDB: #=~PARTY_1.99
        includes = [self.framework.argDB['with-party-include']] 
      else:
        (includes,dummy) = os.path.split(libs)
      if not isinstance(libs, list): libs = [libs]
      if not isinstance(includes, list): includes = [includes]
      yield ('User specified library and includes', [libs], includes)
      raise RuntimeError('You set a value for --with-party-lib, but '+str(self.framework.argDB['with-party-lib'])+' cannot be used.\n')
    # Try specified directory of header files
    if 'with-party-include' in self.framework.argDB: #~PARTY_1.99
      dir = self.framework.argDB['with-party-include']
      libs = os.path.join(dir,'libparty.a')
      includes = dir
      yield ('User specified directory of header files', [[libs]], [includes])
      raise RuntimeError('You set a value for --with-party-include, but '+self.framework.argDB['with-party-include']+' cannot be used.\n')
    # Try specified installation root
    if 'with-party-dir' in self.framework.argDB: #dir=~PARTY_1.99
      dir = self.framework.argDB['with-party-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      libs = os.path.join(dir,'libparty.a')
      includes = dir
      yield ('User specified installation root', [[libs]], [includes])
      raise RuntimeError('You set a value for --with-party-dir, but '+self.framework.argDB['with-party-dir']+' cannot be used.\n')
    # If necessary, download Party
    if not self.found and self.framework.argDB['download-party'] == 2:
      (name, lib, include) = self.downloadParty()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Party could not be used. Please check in install in '+os.path.dirname(include[0][0])+'\n')
    return
  
  def getDir(self):
    '''Find the directory containing Party'''
    packages  = self.framework.argDB['with-external-packages-dir'] 
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    PartyDir = None
    for dir in os.listdir(packages):
      if dir.startswith('PARTY_1.99') and os.path.isdir(os.path.join(packages, dir)):
        PartyDir = dir
    if PartyDir is None:
      self.framework.logPrint('Could not locate already downloaded Party')
      raise RuntimeError('Error locating Party directory')
    return os.path.join(packages, PartyDir)

  def downloadParty(self):
    self.framework.logPrint('Downloading Party')
    try:
      PartyDir = self.getDir()  
      self.framework.logPrint('Party already downloaded, no need to ftp')
    except RuntimeError:
      import urllib
      packages = self.framework.argDB['with-external-packages-dir']
      try:
        self.logPrint("Retrieving Party; this may take several minutes\n", debugSection='screen')
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/PARTY_1.99.tar.gz', os.path.join(packages, 'PARTY_1.99.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Party: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip PARTY_1.99.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping PARTY_1.99.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf PARTY_1.99.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf PARTY_1.99.tar: '+str(e))
      os.unlink(os.path.join(packages, 'PARTY_1.99.tar'))
      self.framework.actions.addArgument('Party', 'Download', 'Downloaded Party into '+self.getDir())
    
    PartyDir = self.getDir()
    lib     = [[os.path.join(PartyDir, 'libparty.a')]] 
    include = [PartyDir]
    return ('Downloaded Party', lib, include)
  
  def configureVersion(self):
    '''Determine the Party version, but there is no reliable way right now'''
    return 'Unknown'

  def configureLibrary(self):
    '''Find all working Party installations and then choose one'''
    functionalParty = []
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.logPrint('================================================================================')
      self.framework.logPrint('Checking for a functional Party in '+name)
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
      functionalParty.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalParty[0]
      self.framework.logPrint('Choose Party '+self.version+' in '+self.name)
    else:
      self.framework.logPrint('Could not locate any functional Party')
    return

  def configure(self):
    if (self.framework.argDB['with-party'] or self.framework.argDB['download-party'] == 1):
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
