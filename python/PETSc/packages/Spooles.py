#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for the Spooles-2.2

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = 'Spooles'
    return

  def __str__(self):
    output=''
    if self.found:
      output  = self.name+':\n'
      output += '  Includes: '+self.include[0]+'\n'
      output += '  Library: '+self.lib[0]+'\n'
    return output
  
  def configureHelp(self,help):
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
        incl         = self.lib[1]  #=spooles-2.2/spooles.a
        (incl,dummy) = os.path.split(incl)
        yield('based on found library location',incl)
      elif 'with-'+package+'-dir' in self.framework.argDB:
        incl = self.framework.argDB['with-'+package+'-dir']
        yield('based on found root directory',incl)

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
      if 'with-'+package+'-lib' in self.framework.argDB: #~spooles-2.2/MPI/src/spoolesMPI.a ~spooles-2.2/spooles.a
        lib = self.framework.argDB['with-'+package+'-lib']
        (lib_mpi,dummy) = os.path.split(lib)
        lib_mpi = os.path.join(lib_mpi,'MPI/src/spoolesMPI.a')
        yield ('User specified '+PACKAGE+' library',lib_mpi,lib)
      elif 'with-'+package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-include'] #~spooles-2.2
        lib = os.path.join(dir,'spooles.a')
        lib_mpi = os.path.join(dir,'MPI/src/spoolesMPI.a')
        yield('User specified '+PACKAGE+' directory of header files',lib_mpi,lib)
      elif 'with-'+package+'-dir' in self.framework.argDB: 
        dir = self.framework.argDB['with-'+package+'-dir'] #~spooles-2.2
        lib = os.path.join(dir,'spooles.a')
        lib_mpi = os.path.join(dir,'MPI/src/spoolesMPI.a')
        yield('User specified '+PACKAGE+' root directory',lib_mpi,lib)
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+PACKAGE+'\n')
        
  def checkLib(self,lib,libfile):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']  
    found = self.libraries.check(lib,libfile)
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.framework.log.write('Found functional '+libfile+' in '+lib[0]+'\n')
    return found
  
  def configureLibrary(self):
    '''Find a installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found     = 0
    foundh    = 0
    for (configstr,lib_mpi,lib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      found = self.executeTest(self.checkLib,[lib,'InpMtx_init'])  
      if found:
        self.lib = [lib_mpi,lib]
        break
    if found:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+'\n')
        foundh = self.executeTest(self.checkInclude,[incl,'MPI/spoolesMPI.h'])
        if foundh:
          self.include = [incl]
          self.found   = 1
          self.setFoundOutput()
          break
    else:
      self.framework.log.write('Could not find a functional '+self.name+'\n')
      self.setEmptyOutput()
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
