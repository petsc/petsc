#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for Mumps-4.3.1

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
    self.name         = 'Mumps'
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
    help.addArgument(PACKAGE,'-with-scalapack=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for SCALAPACK'))
    help.addArgument(PACKAGE,'-with-scalapack-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the SCALAPACK installation'))
    help.addArgument(PACKAGE,'-with-blacs=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for BLACS'))
    help.addArgument(PACKAGE,'-with-blacs-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the BLACS installation'))
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
        for i in 1,2:
          (incl,dummy) = os.path.split(incl)
        yield('based on found library location',os.path.join(incl,'include'))
      elif 'with-'+package+'-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-dir']
        yield('based on found root directory',os.path.join(dir,'include'))

  def checkInclude(self,incl,hfile):
    if not isinstance(incl,list): incl = [incl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in incl:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    PACKAGE = self.name.upper()
    package = self.name.lower()
    if 'with-'+package in self.framework.argDB:
      if 'with-'+package+'-lib' in self.framework.argDB: #~MUMPS_4.3.1/lib/libdmumps.a
        yield ('User specified '+PACKAGE+' library',self.framework.argDB['with-'+package+'-lib'])
      elif 'with-'+package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-include'] #~MUMPS_4.3.1/include
        (dir,dummy) = os.path.split(dir)
        yield('User specified '+PACKAGE+'/Include',os.path.join(dir,'lib/libdmumps.a'))
      elif 'with-'+package+'-dir' in self.framework.argDB: 
        dir = self.framework.argDB['with-'+package+'-dir']
        dir = os.path.join(dir,'lib')
        libs = []
        libs.append(os.path.join(dir,'libdmumps.a'))
        libs.append(os.path.join(dir,'libzmumps.a'))
        libs.append(os.path.join(dir,'libpord.a'))
        yield('User specified '+PACKAGE+' root directory',libs)
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+PACKAGE+'\n')

  def generateScalapackLibGuesses(self):
    if 'with-scalapack' in self.framework.argDB:
      if 'with-scalapack-lib' in self.framework.argDB: 
        yield ('User specified SCALAPACK library',self.framework.argDB['with-scalapack-lib'])
      elif 'with-scalapack-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-scalapack-dir']
        libs = []
        libs.append(os.path.join(dir,'libscalapack.a'))
        yield('User specified SCALAPACK root directory',libs)
      else:
        self.framework.log.write('Must specify either a library or installation root directory for SCALAPACK\n')
  
  def generateBlacsLibGuesses(self):
    if 'with-blacs' in self.framework.argDB:
      if 'with-blacs-lib' in self.framework.argDB: 
        yield ('User specified BLACS library',self.framework.argDB['with-blacs-lib'])
      elif 'with-blacs-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-blacs-dir']
        dir = os.path.join(dir,'LIB')
        libs = []
        libs.append(os.path.join(dir,'libblacs_MPI-LINUX-0.a'))
        libs.append(os.path.join(dir,'libblacsF77init_MPI-LINUX-0.a'))
        libs.append(os.path.join(dir,'libblacs_MPI-LINUX-0.a'))
        libs.append(os.path.join(dir,'libblacsF77init_MPI-LINUX-0.a'))
        yield('User specified BLACS root directory',libs)
      else:
        self.framework.log.write('Must specify either a library or installation root directory for BLACS\n')
        
  def checkLib(self,lib,libfile):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']
    mangleFunc  = 'FC' in self.framework.argDB
    #found = self.libraries.check(lib,libfile)
    found = self.libraries.check(lib,libfile,otherLibs = self.mpi.lib+self.compilers.flibs)
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.framework.log.write('Found functional '+libfile+' in '+lib[0]+'\n')
    return found
  
  def configureLibrary(self):
    '''Find a installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found  = 0
    foundlibs = 0
    foundh = 0
    for (configstr,libs) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      for lib in libs:
        #found = self.executeTest(self.checkLib,[libs[0],'dmumps_c',self.mpi.lib+self.compiler.flibs])
        found = self.executeTest(self.checkLib,[libs[0],'dmumps_c'])
        #found = 1
        foundlibs = foundlibs or found
        if found:
          self.lib.append(lib)
      break
    if foundlibs:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+'\n')
        foundh = self.executeTest(self.checkInclude,[incl,'dmumps_c.h'])
        if foundh:
          self.include = [incl]
          self.found   = 1
          break
    else:
      self.framework.log.write('Could not find a functional '+self.name+'\n')
      self.setEmptyOutput()
      return
    
    self.framework.log.write('Find a installation of SCALAPACK\n')
    found  = 0
    for (configstr,libs) in self.generateScalapackLibGuesses():
      self.framework.log.write('Checking for a functional SCALAPACK in '+configstr+'\n')
      lib = libs[0]
      #found = self.executeTest(self.checkLib,lib,'???']) #???
      found = 1  #???
      if found:
        self.lib.append(lib)
        break
      else:
        self.framework.log.write('Could not find a functional SCALAPACK\n')
        self.setEmptyOutput()
        return
      
    self.framework.log.write('Find a installation of BLACS\n')
    found  = 0
    for (configstr,libs) in self.generateBlacsLibGuesses():
      self.framework.log.write('Checking for a functional BLACS in '+configstr+'\n')
      for lib in libs: 
        #found = self.executeTest(self.checkLib,lib,'???']) #???
        found = 1  #???
        if found:
          self.lib.append(lib)
    
    if not found:
        self.framework.log.write('Could not find a functional BLACS\n')
        self.setEmptyOutput()
        return
    self.setFoundOutput()
    return

  def setFoundOutput(self):
    PACKAGE = self.name.upper()
    incl_str = ''
    for i in range(len(self.include)):
      incl_str += self.include[i]+ ' '
    self.addSubstitution(PACKAGE+'_INCLUDE','-I' +incl_str)
    self.addSubstitution(PACKAGE+'_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_'+PACKAGE,1)
    
  def setEmptyOutput(self):
    PACKAGE = self.name.upper()
    self.addSubstitution(PACKAGE+'_INCLUDE', '')
    self.addSubstitution(PACKAGE+'_LIB', '')
    return

  def configure(self):
    package = self.name.lower()
    if not 'with-'+package in self.framework.argDB:
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
