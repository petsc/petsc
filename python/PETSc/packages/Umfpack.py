#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for the UMFPACK-4.3 and AMD-1.1

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.found        = 0
    self.lib          = ''
    self.lib_amd      = ''
    self.include      = []
    self.hfile        = ''
    return

  def __str__(self):
    output=''
    if self.found:
      output  = 'Umfpack:\n'
      output += '  Includes: '+ str(self.include[0])+'\n'
      output += '  Library: '+str(self.lib[0])+'\n'
    return output
  
  def configureHelp(self,help):
    import nargs
    help.addArgument('UMFPACK','-with-umfpack=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for Umfpack'))
    help.addArgument('UMFPACK','-with-umfpack-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing Umfpack'))
    help.addArgument('UMFPACK','-with-umfpack-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of header files for Umfpack'))
    help.addArgument('UMFPACK','-with-umfpack-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the Umfpack installation'))
    return

  def generateIncludeGuesses(self):
    if 'with-umfpack' in self.framework.argDB:
      if 'with-umfpack-include' in self.framework.argDB:
        yield('User specified UMFPACK header location',self.framework.argDB['with-umfpack-include'])
      elif 'with-umfpack-lib' in self.framework.argDB:
        incl     = self.lib[0]
        incl_amd = self.lib_amd[0]
        # We have ~UMFPACK/Lib/libumfpack.a so remove the last 2 elements from the path
        for i in 1,2:
          (incl,dummy) = os.path.split(incl)
          (incl_amd,dummy) = os.path.split(incl_amd)
        yield('based on found library location',os.path.join(incl,'Include'),os.path.join(incl_amd,'Include'),)
      elif 'with-umfpack-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-umfpack-dir']
        yield('based on found library location',os.path.join(dir,'Include'))

  def checkInclude(self,incl):
    if not isinstance(incl,list):incl = [incl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in incl:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    found = self.checkPreprocess('#include <' +self.hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +self.hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-umfpack' in self.framework.argDB:
      if 'with-umfpack-lib' in self.framework.argDB:
        # guess the default AMD lib
        lib_amd = self.framework.argDB['with-umfpack-lib']
        for i in 1,2,3:
          (lib_amd,dummy) = os.path.split(lib_amd)
        lib_amd = os.path.join(lib_amd,'AMD/Lib/libamd.a') 
        yield ('User specified UMFPACK library',self.framework.argDB['with-umfpack-lib'],lib_amd)
      elif 'with-umfpack-include' in self.framework.argDB:
        dir = self.framework.argDB['with-umfpack-include']
         # We have ~UMFPACK/Include and ~UMFPACK/Lib, so remove 'Include', then add 'Lib'
        (dir,dummy) = os.path.split(dir)
        dir = os.path.join(dir,'Lib')
        yield('User specified UMFPACK installation',os.path.join(dir,'libumfpack.a'))
      elif 'with-umfpack-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-umfpack-dir']
        dir = os.path.join(dir,'Lib')
        yield('User specified UMFPACK installation',os.path.join(dir,'libumfpack.a'))
      else:
        self.framework.log.write('Must specify either a library or installation root directory for UMFPACK\n')
        
  def checkLib(self,lib):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']  
    found = self.libraries.check(lib,'umfpack_di_report_info')
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.lib = lib  # a list now!
      self.framework.log.write('Found functional Umfpack: '+str(lib)+'\n')
    return found
  def checkLib_amd(self,lib):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']  #' -ldl -lm'
    found = self.libraries.check(lib,'amd_defaults')
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.lib_amd = lib
      self.framework.log.write('Found functional Amd: '+str(lib)+'\n')
    return found

  def configureLibrary(self):
    '''Find a Umfpack and AMD installation and check if they can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found = 0
    found_amd = 0
    foundh = 0
    foundh_amd = 0
    for (configstr,lib,lib_amd) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional Umfpack in '+configstr+'\n')
      found = self.executeTest(self.checkLib,lib)
      found_amd = self.executeTest(self.checkLib_amd,lib_amd)
      if found and found_amd: break
    if found and found_amd:
      for (inclstr,incl,incl_amd) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+ ' and '+incl_amd+'\n')
        self.hfile = 'umfpack.h'
        foundh = self.executeTest(self.checkInclude,incl)
        self.hfile = 'amd.h'
        foundh_amd = self.executeTest(self.checkInclude,incl_amd)
        if foundh and foundh_amd:
          self.include = [incl, incl_amd]
          self.found       = 1
          self.setFoundOutput()
          break
    else:
      self.framework.log.write('Could not find a functional Umfpack or AMD \n')
      self.setEmptyOutput()
    return

  def setFoundOutput(self):
    self.addSubstitution('UMFPACK_INCLUDE','-I'+self.include[0])
    self.addSubstitution('UMFPACK_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_UMFPACK',1)
    self.addSubstitution('AMD_INCLUDE','-I'+self.include[1])
    self.addSubstitution('AMD_LIB',' '.join(map(self.libraries.getLibArgument,self.lib_amd)))
    self.addDefine('HAVE_AMD',1)
    
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
