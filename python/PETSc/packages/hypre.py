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
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = 'hypre'
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
    help.addArgument(PACKAGE,'-with-'+package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    return

  def generateIncludeGuesses(self):
    PACKAGE = self.name.upper()
    package = self.name.lower()
    if 'with-'+package in self.framework.argDB:
      if 'with-'+package+'-dir' in self.framework.argDB:
        dir = os.path.abspath(self.framework.argDB['with-'+package+'-dir'])
        yield('based on found root directory',os.path.join(dir,'include'))
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
      if 'with-'+package+'-dir' in self.framework.argDB:     
        dir = os.path.abspath(self.framework.argDB['with-'+package+'-dir'])
        dir = os.path.join(dir,'lib')
        libs = ['DistributedMatrix',
                'DistributedMatrixPilutSolver',
                'Euclid',
                'IJ_mv',
                'LSI',
                'MatrixMatrix',
                'ParaSails',
                'krylov',
                'lobpcg',
                'mli',
                'parcsr_ls',
                'parcsr_mv',
                'seq_mv',
                'sstruct_ls',
                'sstruct_mv',
                'struct_ls',
                'struct_mv',
                'utilities'
               ]
        alllibs = []
        for l in libs:
          alllibs.append(os.path.join(dir,'libHYPRE_'+l+'.a'))
        yield('User specified '+PACKAGE+' root directory', alllibs)
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+PACKAGE+'\n')
    return
        
  def checkLib(self,lib,func):
    '''We need the BLAS/Lapack libraries here plus (possibly) Fortran, and may need the MPI libraries'''
    oldLibs = self.framework.argDB['LIBS']
    otherLibs = self.blasLapack.lapackLibrary
    if not None in self.blasLapack.blasLibrary:
      otherLibs = otherLibs+self.blasLapack.blasLibrary
    otherLibs = ' '.join([self.libraries.getLibArgument(lib1) for lib1 in otherLibs])
    self.framework.log.write('Otherlibs '+otherLibs+'\n')
    otherLibs += ' '+' '.join(map(self.libraries.getLibArgument, self.mpi.lib))
    if hasattr(self.compilers,'flibs'): otherLibs += ' '+self.compilers.flibs
    self.framework.log.write('Otherlibs '+otherLibs+'\n')
    found = self.libraries.check(lib,func, otherLibs = otherLibs)
    self.framework.argDB['LIBS']=oldLibs
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
      foundLibrary = self.executeTest(self.checkLib, [lib, 'HYPRE_IJMatrixCreate'])  
      if foundLibrary:
        self.lib = lib
        break
    for inclstr, incl in self.generateIncludeGuesses():
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for headers '+inclstr+': '+str(incl)+'\n')
      foundHeader = self.executeTest(self.checkInclude, [incl, 'HYPRE.h'])
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

  def download(self):
    configure = 'configure --with-babel=0 --with-mli=0 --with-FEI=0 --with-mpi-include=/usr/local/mpich-1.2.5.2-gnu/include/ --with-mpi-lib-dirs=/usr/local/mpich-1.2.5.2-gnu/lib/ --with-mpi-libs="mpich pmpich" --with-CC=gcc --with-CXX=g++ --with-F77=g77 --with-blas=0'
    
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
