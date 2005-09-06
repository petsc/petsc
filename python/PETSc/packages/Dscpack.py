#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

#Developed for DSCPACK1.0

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.found        = 0
    self.lib          = []
    self.include      = []
    return

  def __str__(self):
    output=''
    if self.found:
      output  = self.name+':\n'
      output += '  Includes: '+self.include[0]+'\n'
      output += '  Library: '+self.lib[0]+'\n'
    return output
  
  def setupHelp(self,help):
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,0,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of header files for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi = framework.require('PETSc.packages.MPI',self)
    return

  def generateIncludeGuesses(self):
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-include' in self.framework.argDB:
        incl = self.framework.argDB['with-'+self.package+'-include']
        yield('User specified '+self.PACKAGE+' header location',incl)
      elif 'with-'+self.package+'-lib' in self.framework.argDB:
        incl     = self.lib[0]
        (incl,dummy) = os.path.split(incl)
        yield('based on found library location',incl)
      elif 'with-'+self.package+'-dir' in self.framework.argDB:
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('based on found root directory',os.path.join(dir,'DSC_LIB'))

  def checkInclude(self,incl,hfile):
    incl.extend(self.mpi.include)
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += self.headers.toString(incl)
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.compilers.CPPFLAGS = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-lib' in self.framework.argDB: #~DSCPACK1.0/DSC_LIB/dsclibdbl.a
        yield ('User specified '+self.PACKAGE+' library',self.framework.argDB['with-'+self.package+'-lib'])
      elif 'with-'+self.package+'-include' in self.framework.argDB: #~DSCPACK1.0/DSC_LIB
        dir = self.framework.argDB['with-'+self.package+'-include'] 
        yield('User specified '+self.PACKAGE+'/Include',os.path.join(dir,'dsclibdbl.a'))
      elif 'with-'+self.package+'-dir' in self.framework.argDB:  #DSCPACK1.0
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('User specified '+self.PACKAGE+' root directory',os.path.join(dir,'DSC_LIB/dsclibdbl.a'))
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+self.PACKAGE+'\n')
        
  def checkLib(self,lib,func):
    '''We may need the MPI libraries here'''
    oldLibs = self.compilers.LIBS
    found = self.libraries.check(lib,func,otherLibs=self.mpi.lib)
    self.compilers.LIBS=oldLibs  
    if found:
      self.framework.log.write('Found function '+func+' in '+str(lib)+'\n')
    return found
  
  def configureLibrary(self):
    '''Find a installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')
    foundLibrary = 0
    foundHeader  = 0
    for configstr, lib in self.generateLibGuesses():
      if not isinstance(lib, list): lib = [lib]
      self.framework.log.write('Checking for library '+configstr+': '+str(lib)+'\n')
      foundLibrary = self.executeTest(self.checkLib,[lib,'DSC_ErrorDisplay'])  
      if foundLibrary:
        self.lib = lib
        break
    for inclstr,incl in self.generateIncludeGuesses():
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for headers '+inclstr+': '+str(incl)+'\n')
      foundHeader = self.executeTest(self.checkInclude,[incl,'dscmain.h'])
      if foundHeader:
        self.include = incl
        break
    if foundLibrary and foundHeader:
      self.setFoundOutput()
      self.found = 1
    else:
      raise RuntimeError('Could not find a functional '+self.name+'\n')
    return

  def setFoundOutput(self):
    self.addDefine('HAVE_'+self.PACKAGE,1)
    self.framework.packages.append(self)
    
  def configure(self):
    if self.framework.argDB['with-'+self.package]:
      if self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.libraryOptions.integerSize == 64:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')
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
