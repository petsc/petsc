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
    oldFlags = self.framework.argDB['CPPFLAGS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in [includeDir]])
    found = self.checkPreprocess('#include <party_lib.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return found

  def generateGuesses(self):
    if 'with-party-lib' in self.framework.argDB and 'with-party-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Party library with --with-party-lib=<lib> and search directory with --with-party-dir=<dir>')
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
    return

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
    if (self.framework.argDB['with-party']):
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
