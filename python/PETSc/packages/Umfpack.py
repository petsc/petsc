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
    self.lib          = []
    self.include      = []
    self.name         = 'Umfpack'
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
        (incl_amd,dummy) = os.path.split(incl)
        (incl_amd,dummy) = os.path.split(incl_amd)
        yield('User specified '+PACKAGE+' header location',incl,os.path.join(incl_amd,'AMD/Include'))
      elif 'with-'+package+'-lib' in self.framework.argDB:
        incl     = self.lib[0]
        incl_amd = self.lib[1]
        # We have ~UMFPACK/Lib/libumfpack.a so remove the last 2 elements from the path
        for i in 1,2:
          (incl,dummy) = os.path.split(incl)
          (incl_amd,dummy) = os.path.split(incl_amd)
        yield('based on found library location',os.path.join(incl,'Include'),os.path.join(incl_amd,'Include'))
      elif 'with-'+package+'-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-dir']
        (dir_amd,dummy) = os.path.split(dir)
        yield('based on found root directory',os.path.join(dir,'Include'),os.path.join(dir_amd,'AMD/Include'))

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
      if 'with-'+package+'-lib' in self.framework.argDB:
        # guess the default AMD lib
        lib_amd = self.framework.argDB['with-'+package+'-lib']
        for i in 1,2,3:
          (lib_amd,dummy) = os.path.split(lib_amd)
        lib_amd = os.path.join(lib_amd,'AMD/Lib/libamd.a') 
        yield ('User specified '+PACKAGE+' library',self.framework.argDB['with-'+package+'-lib'],lib_amd)
      elif 'with-'+package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-include']
         # We have ~UMFPACK/Include and ~UMFPACK/Lib, so remove 'Include', then add 'Lib'
        (dir,dummy) = os.path.split(dir)
        (dir_amd,dummy) = os.path.split(dir)
        dir_amd = os.path.join(dir_amd,'AMD/Lib')
        dir = os.path.join(dir,'Lib')
        yield('User specified '+PACKAGE+'/Include',os.path.join(dir,'libumfpack.a'),os.path.join(dir_amd,'libamd.a'))
      elif 'with-'+package+'-dir' in self.framework.argDB:
        dir = self.framework.argDB['with-'+package+'-dir']
        (dir_amd,dummy) = os.path.split(dir)
        dir_amd = os.path.join(dir_amd,'AMD/Lib')
        dir = os.path.join(dir,'Lib')
        yield('User specified '+PACKAGE+' root directory',os.path.join(dir,'libumfpack.a'),os.path.join(dir_amd,'libamd.a'))
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
    found      = 0
    found_amd  = 0
    foundh     = 0
    foundh_amd = 0
    for (configstr,lib,lib_amd) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      found = self.executeTest(self.checkLib,[lib,'umfpack_di_report_info'])
      found_amd = self.executeTest(self.checkLib,[lib_amd,'amd_defaults'])
      if found and found_amd:
        self.lib = [lib, lib_amd]
        break
    if found and found_amd:
      for (inclstr,incl,incl_amd) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+ ' and '+incl_amd+'\n')
        foundh = self.executeTest(self.checkInclude,[incl,'umfpack.h'])
        foundh_amd = self.executeTest(self.checkInclude,[incl_amd,'amd.h'])
        if foundh and foundh_amd:
          self.include = [incl, incl_amd]
          self.found   = 1
          self.setFoundOutput()
          break
    else:
      self.framework.log.write('Could not find a functional '+self.name+' or AMD \n')
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
