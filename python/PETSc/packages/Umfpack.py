#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

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
      output  = 'Umfpack:\n'
      output += '  Includes: '+ str(self.include)+'\n'
      output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def configureHelp(self,help):
    import nargs
    help.addArgument('UMFPACK','-with-umfpack=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for Umfpack'))
    help.addArgument('UMFPACK','-with-umfpack-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing Umfpack'))
    help.addArgument('UMFPACK','-with-umfpack-include=<lib>',nargs.ArgDir(None,None,'Indicate the header files for Umfpack'))
    help.addArgument('UMFPACK','-with-umfpack-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the Umfpack installation'))
    return

  def generateIncludeGuesses(self):
    if 'with-umfpack' in self.framework.argDB:
      if 'with-umfpack-lib' in self.framework.argDB:
        incl = self.lib[0]
        # We have ~umfpack/umfpack.a so remove the last element from the path
        (incl,dummy) = os.path.split(incl)
        yield('based on found library location',incl)
      if 'with-umfpack-include' in self.framework.argDB:
        yield('User specified UMFPACK header location',self.framework.argDB['with-umfpack-include'])
      if 'with-umfpack-dir' in self.framework.argDB:
        yield('User specified UMFPACK header location',self.framework.argDB['with-umfpack-dir'])

  def checkInclude(self,incl):
    '''Check that umfpack.h is present'''
    if not isinstance(incl,list):incl = [incl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in incl:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    found = self.checkPreprocess('#include <umfpack.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.include = incl
      self.framework.log.write('Found Umfpack header file umfpack.h: '+str(self.include)+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-umfpack' in self.framework.argDB:
      if 'with-umfpack-lib' in self.framework.argDB:
        yield ('User specified UMFPACK library',self.framework.argDB['with-umfpack-lib'])
      if 'with-umfpack-include' in self.framework.argDB:
        dir = self.framework.argDB['with-umfpack-include']
        yield('User specified UMFPACK installation',os.path.join(dir,'umfpack.a'))
      if 'with-umfpack-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-umfpack-dir']
        yield('User specified UMFPACK installation',os.path.join(dir,'umfpack.a'))

  def checkLib(self,lib):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']  
    found = self.libraries.check(lib,'umfpack_di_report_info')
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.lib = lib
      self.framework.log.write('Found functional Umfpack: '+str(self.lib)+'\n')
    return found

  def configureLibrary(self):
    '''Find a Umfpack installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found = 0
    for (configstr,lib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional Umfpack in '+configstr+'\n')
      found = self.executeTest(self.checkLib,lib)
      if found: break
    if found:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for Umfpack headers in '+inclstr+': '+incl + '\n')
        if self.executeTest(self.checkInclude,incl):
          self.include = incl
          self.found   = 1
          self.setFoundOutput()
          break
    else:
      self.framework.log.write('Could not find a functional Umfpack\n')
      self.setEmptyOutput()
    return

  def setFoundOutput(self):
    self.addSubstitution('UMFPACK_INCLUDE','-I'+self.include)
    self.addSubstitution('UMFPACK_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_UMFPACK',1)
    
  def setEmptyOutput(self):
    self.addSubstitution('UMFPACK_INCLUDE', '')
    self.addSubstitution('UMFPACK_LIB', '')
    return

  def configure(self):
    if not 'with-umfpack' in self.framework.argDB:
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
