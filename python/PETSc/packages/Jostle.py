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
    self.mpi          = self.framework.require('PETSc.packages.MPI', self)
    self.name         = 'Jostle'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    if self.found:
      desc = ['Jostle:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Jostle', '-with-jostle=<bool>',                nargs.ArgBool(None, 0, 'Activate Jostle'))
    help.addArgument('Jostle', '-with-jostle-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the Jostle installation'))
    help.addArgument('Jostle', '-with-jostle-include=<dir>',         nargs.ArgDir(None, None, 'The directory containing jostle_lib.h'))
    help.addArgument('Jostle', '-with-jostle-lib=<lib>',             nargs.Arg(None, None, 'The Jostle library or list of libraries'))
    help.addArgument('Jostle', '-download-jostle=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Automatically install Jostle'))
    return

  def checkLib(self, libraries):
    '''Check for pjostle in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(libraries, 'pjostle', otherLibs = self.mpi.lib + ['libm.a'])
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that jostle.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in [includeDir]+self.mpi.include])
    found = self.checkPreprocess('#include <jostle.h>\n')
    self.compilers.CPPFLAGS = oldFlags
    return found

  def generateGuesses(self):
    if 'with-jostle-lib' in self.framework.argDB and 'with-jostle-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Jostle library with --with-jostle-lib=<lib> and search directory with --with-jostle-dir=<dir>')
    if self.framework.argDB['download-jostle'] == 1:
      (name, lib, include) = self.downloadJostle()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Jostle could not be used. Please check install in '+os.path.dirname(include[0][0])+'\n')
    # Try specified library (and include)
    if 'with-jostle-lib' in self.framework.argDB: #~JOSTLE/libjostle.lnx.a
      libs = self.framework.argDB['with-jostle-lib'] #='~JOSTLE/libjostle.lnx.a'
      if 'with-jostle-include' in self.framework.argDB: 
        includes = [self.framework.argDB['with-jostle-include']] 
      else:
        (includes,dummy) = os.path.split(libs)
      if not isinstance(libs, list): libs = [libs]
      if not isinstance(includes, list): includes = [includes]
      yield ('User specified library and includes', [libs], includes)
      raise RuntimeError('You set a value for --with-jostle-lib, but '+str(self.framework.argDB['with-jostle-lib'])+' cannot be used.\n')
    # Try specified directory of header files
    if 'with-jostle-include' in self.framework.argDB: 
      dir = self.framework.argDB['with-jostle-include']
      libs = os.path.join(dir,'libjostle.lnx.a')
      includes = dir
      yield ('User specified directory of header files', [[libs]], [includes])
      raise RuntimeError('You set a value for --with-jostle-include, but '+self.framework.argDB['with-jostle-include']+' cannot be used.\n')
    # Try specified installation root
    if 'with-jostle-dir' in self.framework.argDB: 
      dir = self.framework.argDB['with-jostle-dir']
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      libs = os.path.join(dir,'libjostle.lnx.a')
      includes = dir
      yield ('User specified installation root', [[libs]], [includes])
      raise RuntimeError('You set a value for --with-jostle-dir, but '+self.framework.argDB['with-jostle-dir']+' cannot be used.\n')
    # If necessary, download Jostle
    if not self.found and self.framework.argDB['download-jostle'] == 2:
      (name, lib, include) = self.downloadJostle()
      yield (name, lib, include)
      raise RuntimeError('Downloaded Jostle could not be used. Please check in install in '+os.path.dirname(include[0][0])+'\n')
    return

  def getDir(self):
    '''Find the directory containing Jostle'''
    packages  = self.framework.argDB['with-external-packages-dir'] 
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    jostleDir = None
    for dir in os.listdir(packages):
      if dir.startswith('jostle') and os.path.isdir(os.path.join(packages, dir)):
        jostleDir = dir
    if jostleDir is None:
      self.framework.logPrint('Could not locate already downloaded Jostle')
      raise RuntimeError('Error locating Jostle directory')
    return os.path.join(packages, jostleDir)

  def downloadJostle(self):
    self.framework.logPrint('Downloading Jostle')
    try:
      jostleDir = self.getDir()  
      self.framework.logPrint('Jostle already downloaded, no need to ftp')
    except RuntimeError:
      import urllib
      packages = self.framework.argDB['with-external-packages-dir']
      try:
        self.logPrint("Retrieving Jostle; this may take several minutes\n", debugSection='screen')
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/jostle.tar.gz', os.path.join(packages, 'jostle.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Jostle: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip jostle.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping jostle.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf jostle.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf jostle.tar: '+str(e))
      os.unlink(os.path.join(packages, 'jostle.tar'))
      self.framework.actions.addArgument('Jostle', 'Download', 'Downloaded Jostle into '+self.getDir())
    # Configure and Build Jostle ? Jostle is already configured and build with gcc!
    jostleDir = self.getDir()
    lib     = [[os.path.join(jostleDir, 'libjostle.lnx.a')]] 
    include = [jostleDir]
    return ('Downloaded Jostle', lib, include)

  def configureVersion(self):
    '''Determine the Jostle version, but there is no reliable way right now'''
    return 'Unknown'

  def configureLibrary(self):
    '''Find all working Jostle installations and then choose one'''
    functionalJostle = []
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.logPrint('================================================================================')
      self.framework.logPrint('Checking for a functional Jostle in '+name)
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
      functionalJostle.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalJostle[0]
      self.framework.logPrint('Choose Jostle '+self.version+' in '+self.name)
    else:
      self.framework.logPrint('Could not locate any functional Jostle')
    return

  def configure(self):
    if (self.framework.argDB['with-jostle'] or self.framework.argDB['download-jostle'] == 1):
    #if (self.framework.argDB['with-jostle']):
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
