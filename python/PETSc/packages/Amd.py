#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for the AMD version 1.1: permutations for sparse matrices -- required by Umfpack.

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.found        = 0
    self.lib          = ''
    self.include      = ''
    return

  def __str__(self):
    output=''
    if self.found:
      output  = 'Amd:\n'
      output += '  Includes: '+ str(self.include)+'\n'
      output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def configureHelp(self,help):
    import nargs
    help.addArgument('AMD','-with-amd=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for Amd'))
    help.addArgument('AMD','-with-amd-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing Amd'))
    help.addArgument('AMD','-with-amd-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of header files for Amd'))
    help.addArgument('AMD','-with-amd-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the Amd installation'))
    return

  def generateIncludeGuesses(self):
    if 'with-amd' in self.framework.argDB:
      if 'with-amd-include' in self.framework.argDB:
        yield('User specified AMD header location',self.framework.argDB['with-amd-include'])
      if 'with-amd-lib' in self.framework.argDB:
        incl = self.lib[0]
        # We have ~AMD/Lib/libamd.a so remove the last 2 elements from the path
        for i in 1,2:
          (incl,dummy) = os.path.split(incl)
        yield('based on found library location',os.path.join(incl,'Include'))
      if 'with-amd-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-amd-dir']
        yield('based on found library location',os.path.join(dir,'Include'))

  def checkInclude(self,incl):
    '''Check that amd.h is present'''
    if not isinstance(incl,list):incl = [incl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in incl:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    found = self.checkPreprocess('#include <amd.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.include = incl
      self.framework.log.write('Found Amd header file amd.h: '+str(self.include)+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-amd' in self.framework.argDB:
      if 'with-amd-lib' in self.framework.argDB:
        yield ('User specified AMD library',self.framework.argDB['with-amd-lib'])
      if 'with-amd-include' in self.framework.argDB:
        dir = self.framework.argDB['with-amd-include']
         # We have ~AMD/Include and ~AMD/Lib, so remove 'Include', then add 'Lib'
        (dir,dummy) = os.path.split(dir)
        dir = os.path.join(dir,'Lib')
        yield('User specified AMD installation',os.path.join(dir,'libamd.a'))
      if 'with-amd-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-amd-dir']
        dir = os.path.join(dir,'Lib')
        yield('User specified AMD installation',os.path.join(dir,'libamd.a'))

  def checkLib(self,lib):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']  
    found = self.libraries.check(lib,'amd_defaults')
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.lib = lib
      self.framework.log.write('Found functional Amd: '+str(self.lib)+'\n')
    return found

  def configureLibrary(self):
    '''Find a Amd installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found = 0
    for (configstr,lib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional Amd in '+configstr+'\n')
      found = self.executeTest(self.checkLib,lib)
      if found: break
    if found:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for Amd headers in '+inclstr+': '+incl + '\n')
        if self.executeTest(self.checkInclude,incl):
          self.include = incl
          self.found   = 1
          self.setFoundOutput()
          break
    else:
      self.framework.log.write('Could not find a functional Amd\n')
      self.setEmptyOutput()
    return

  def setFoundOutput(self):
    self.addSubstitution('AMD_INCLUDE','-I'+self.include)
    self.addSubstitution('AMD_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_AMD',1)
    
  def setEmptyOutput(self):
    self.addSubstitution('AMD_INCLUDE', '')
    self.addSubstitution('AMD_LIB', '')
    return

  def configure(self):
    if not 'with-amd' in self.framework.argDB:
      self.setEmptyOutput()
      return
    self.executeTest(self.configureLibrary)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
