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
    return

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
    if (self.framework.argDB['with-jostle']):
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
