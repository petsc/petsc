#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for the SuperLU_DIST_2.0

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = 'SuperLU_DIST'
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
    PACKAGE = self.name.upper()
    package = self.name.lower()
    help.addArgument(PACKAGE,'-with-'+package+'=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for '+self.name))
    help.addArgument(PACKAGE,'-with-'+package+'-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing '+self.name))
    help.addArgument(PACKAGE,'-with-'+package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of header files for '+self.name))
    help.addArgument(PACKAGE,'-with-'+package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    return

  def generateIncludeGuesses(self):
    PACKAGE = self.name.upper()
    package = self.name.lower()
    if 'with-'+package in self.framework.argDB:
      if 'with-'+package+'-include' in self.framework.argDB:
        incl = self.framework.argDB['with-'+package+'-include']
        yield('User specified '+PACKAGE+' header location',incl)
      elif 'with-'+package+'-lib' in self.framework.argDB:
        incl     = self.lib[0]
        (incl,dummy) = os.path.split(incl)
        yield('based on found library location',os.path.join(incl,'SRC'))
      elif 'with-'+package+'-dir' in self.framework.argDB:
        dir = os.path.abspath(self.framework.argDB['with-'+package+'-dir'])
        yield('based on found root directory',os.path.join(dir,'SRC'))
    return

  def checkInclude(self,incl,hfile):
    incl.extend(self.mpi.include)
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in incl:
      if not self.mpi.include is None:
        mpiincl = ' -I' + ' -I'.join(self.mpi.include)
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc+mpiincl
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    PACKAGE = self.name.upper()
    package = self.name.lower()
    if 'with-'+package in self.framework.argDB:
      if 'with-'+package+'-lib' in self.framework.argDB: #~SuperLU_DIST_2.0/superlu_linux.a
        yield ('User specified '+PACKAGE+' library', self.framework.argDB['with-'+package+'-lib'])
      elif 'with-'+package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-include'] #~SuperLU_DIST_2.0/SRC
        (dir,dummy) = os.path.split(dir)
        yield('User specified '+PACKAGE+'/Include', os.path.join(dir,'superlu_linux.a'))
      elif 'with-'+package+'-dir' in self.framework.argDB: 
        dir = os.path.abspath(self.framework.argDB['with-'+package+'-dir'])
        yield('User specified '+PACKAGE+' root directory', os.path.join(dir,'superlu_linux.a'))
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+PACKAGE+'\n')
    return
        
  def checkLib(self,lib,func):
    '''We may need the MPI libraries here'''
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(lib,func,otherLibs=' '.join(map(self.libraries.getLibArgument, self.mpi.lib)))
    self.framework.argDB['LIBS'] = oldLibs
    if found:
      self.framework.log.write('Found function '+func+' in '+str(lib)+'\n')
    return found
  
  def configureLibrary(self):
    '''Find an installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')
    foundLibrary = 0
    foundHeader  = 0
    for configstr, lib in self.generateLibGuesses():
      if not isinstance(lib, list): lib = [lib]
      self.framework.log.write('Checking for library '+configstr+': '+str(lib)+'\n')
      foundLibrary = self.executeTest(self.checkLib, [lib, 'set_default_options_dist']) 
      if foundLibrary:
        self.lib = lib
        break
    for inclstr, incl in self.generateIncludeGuesses():
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for headers '+inclstr+': '+str(incl)+'\n')
      foundHeader = self.executeTest(self.checkInclude, [incl, 'superlu_ddefs.h'])
      if foundHeader:
        self.include = incl
        break
    if foundLibrary and foundHeader:
      self.setFoundOutput()
      self.found = 1
    else:
      self.framework.log.write('Could not find a functional '+self.name+'\n')
      self.setEmptyOutput()
    return

  def setFoundOutput(self):
    PACKAGE = self.name.upper()
    self.addSubstitution(PACKAGE+'_INCLUDE','-I'+self.include[0])
    self.addSubstitution(PACKAGE+'_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_'+PACKAGE,1)
    
  def setEmptyOutput(self):
    PACKAGE = self.name.upper()
    self.addSubstitution(PACKAGE+'_INCLUDE', '')
    self.addSubstitution(PACKAGE+'_LIB', '')
    return

  def configure(self):
    package = self.name.lower()
    if not 'with-'+package in self.framework.argDB or not self.mpi.foundMPI or self.framework.argDB['with-64-bit-ints']:
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
