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
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
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

  def generateIncludeGuesses(self):
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-include' in self.framework.argDB:
        incl = self.framework.argDB['with-'+self.package+'-include']
        yield('User specified '+self.PACKAGE+' header location',incl)
      elif 'with-'+self.package+'-lib' in self.framework.argDB:
        incl     = self.lib[0]
        (incl,dummy) = os.path.split(incl)
        yield('based on found library location',os.path.join(incl,'SRC'))
      elif 'with-'+self.package+'-dir' in self.framework.argDB:
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('based on found root directory',os.path.join(dir,'SRC'))
    return

  def checkInclude(self,incl,hfile):
    incl.extend(self.mpi.include)
    oldFlags = self.framework.argDB['CPPFLAGS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in incl])
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-lib' in self.framework.argDB: #~SuperLU_DIST_2.0/superlu_linux.a
        yield ('User specified '+self.PACKAGE+' library', self.framework.argDB['with-'+self.package+'-lib'])
      elif 'with-'+self.package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+self.package+'-include'] #~SuperLU_DIST_2.0/SRC
        (dir,dummy) = os.path.split(dir)
        yield('User specified '+self.PACKAGE+'/Include', os.path.join(dir,'superlu_linux.a'))
      elif 'with-'+self.package+'-dir' in self.framework.argDB: 
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('User specified '+self.PACKAGE+' root directory', os.path.join(dir,'superlu_linux.a'))
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+self.PACKAGE+'\n')
    return
        
  def checkLib(self,lib,func):
    '''We may need the MPI libraries here'''
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(lib,func,otherLibs=self.libraries.toString(self.mpi.lib))
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
      raise RuntimeError('Could not find a functional '+self.name+'\n')
    return

  def setFoundOutput(self):
    self.framework.packages.append(self)
    

  def configure(self):
    if self.framework.argDB['with-'+self.package]:
      if self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.framework.argDB['with-64-bit-ints']:
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
