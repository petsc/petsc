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
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.foundUMF = 0
    self.lib          = ''
    self.include      = ''
    return

  def __str__(self):
    output=''
    if self.foundUMF:
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
      if 'with-umfpack-include' in self.framework.argDB:
        yield('User specified UMFPACK header location',self.framework.argDB['with-umfpack-include'])
      bsroot = self.lib[0]
      # We have /home/user/Umfpack/lib/libO/bsarch/libUMF.a so remove the last 4 elements from the path
      for i in 1,2,3,4:
        (bsroot,dummy) = os.path.split(bsroot)
      yield('based on found library location',os.path.join(bsroot,'include'))

  def checkInclude(self,umfincl):
    '''Check that umfpack.h is present'''
    if not isinstance(umfincl,list):umfincl = [umfincl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in umfincl:
      if not self.mpi.include is None:
        mpiincl = ' -I' + ' -I'.join(self.mpi.include)
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc+mpiincl
    found = self.checkPreprocess('#include <umfpack.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.include = umfincl
      self.framework.log.write('Found Umfpack header file umfpack.h: '+str(self.include)+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-umfpack' in self.framework.argDB:
      if 'with-umfpack-lib' in self.framework.argDB:
        yield ('User specified UMFPACK library',self.framework.argDB['with-umfpack-lib'])
      if 'with-umfpack-dir' in self.framework.argDB and 'with-umfpack-bopt' in self.framework.argDB and 'with-umfpack-arch' in self.framework.argDB:
        dir    = self.framework.argDB['with-umfpack-dir']
        bopt   = self.framework.argDB['with-umfpack-bopt']
        bsarch = self.framework.argDB['with-umfpack-arch']
        yield('User specified UMFPACK installation',os.path.join(dir,'lib','lib'+bopt,bsarch,'umfpack.a'))
      # Perhaps we could also check all possible umfpack installations based on just with-umfpack-dir trying all bopt and bsarch available....

  def checkLib(self,umflib):
    if not isinstance(umflib,list): umflib = [umflib]
    oldLibs = self.framework.argDB['LIBS']  
    # This next line is really ugly
    # ' '.join(map(self.libraries.getLibArgument',self.mpi.lib)
    # takes the location of the MPI library and separates it into a string that looks like:
    # -L<MPI_DIR> -l<mpi library>
    found = self.libraries.check(umflib,'umfpack_di_report_info',otherLibs=' '.join(map(self.libraries.getLibArgument, self.mpi.lib)))
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.lib    = umflib
      self.framework.log.write('Found functional Umfpack: '+str(self.lib)+'\n')
    return found

  def configureLibrary(self):
    '''Find a Umfpack installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found = 0
    for (configstr,umflib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional Umfpack in '+configstr+'\n')
      found = self.executeTest(self.checkLib,umflib)
      if found: break
    if found:
      for (inclstr,umfincl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for Umfpack headers in '+inclstr+': '+umfincl + '\n')
        if self.executeTest(self.checkInclude,umfincl):
          self.include = umfincl
          self.foundUMF = 1
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
    #self.addDefine('HAVE_UMFPACK', 0)
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
