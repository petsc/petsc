#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/tmp/hypre.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['HYPRE_IJMatrixCreate']
    self.includes     = ['HYPRE.h']
    return

      raise RuntimeError('Downloaded hypre could not be used (missing include directory). Please check install in '+dir+'\n')
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-dir' in self.framework.argDB:
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('based on found root directory',os.path.join(dir,'include'))
    return

  def checkInclude(self,incl,hfile):
    incl.extend(self.mpi.include)
    oldFlags = self.framework.argDB['CPPFLAGS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in incl])
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
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Download LLNL hypre preconditioners'))
    return

  def generateIncludeGuesses(self):
    if self.framework.argDB['download-'+self.package] == 1 or self.framework.argDB['download-'+self.package] == 2:
      dir = self.downLoadhypre()
      yield('based on downloaded directory',os.path.join(dir,self.framework.argDB['PETSC_ARCH'],'include'))      
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but hypre requires the extra C++ library'''
    libs = ['DistributedMatrix',
            'DistributedMatrixPilutSolver',
            'Euclid',
            'IJ_mv',
            'LSI',
            'MatrixMatrix',
            'ParaSails',
            'krylov',
            'lobpcg',
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
    import config.setCompilers
    self.framework.pushLanguage('C')
    if config.setCompilers.Configure.isGNU(self.framework.getCompiler()):
      alllibs.append('-lstdc++')
    self.framework.popLanguage()    
    return alllibs
          
      alllibs = self.generateLibList(dir)
      yield('Download '+self.PACKAGE, alllibs)
      raise RuntimeError('Downloaded hypre could not be used. Please check install in '+dir+'\n')
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-dir' in self.framework.argDB:     
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        alllibs = self.generateLibList(dir)
        yield('User specified '+self.PACKAGE+' root directory', alllibs)
      else:
        self.framework.log.write('Must specify an installation root directory for '+self.PACKAGE+'\n')
    if self.framework.argDB['download-hypre'] == 2:
      dir = os.path.join(self.downLoadhypre(),self.framework.argDB['PETSC_ARCH'])
      alllibs = self.generateLibList(dir)
      yield('Download '+self.PACKAGE, alllibs)
      raise RuntimeError('Downloaded hypre could not be used. Please check install in '+dir+'\n')
    return
  def generateLibGuesses(self):
    if self.framework.argDB['download-'+self.package] == 1:
      dir = os.path.join(self.downLoadhypre(),self.framework.argDB['PETSC_ARCH'])
        
  def Install(self):
    if not os.path.isfile(os.path.expanduser(os.path.join('~','.hypre_license'))):
      print "**************************************************************************************************"
      print "You must register to use hypre at http://www.llnl.gov/CASC/hypre/download/hyprebeta_cur_agree.html"
      print "    Once you have registered, configure will continue and download and install hypre for you      "
      print "**************************************************************************************************"
      fd = open(os.path.expanduser(os.path.join('~','.hypre_license')),'w')
      fd.close()
    hypreDir = self.getDir()
    installDir = os.path.join(hypreDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build HYPRE
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    if 'CXX' in self.framework.argDB:
      self.framework.pushLanguage('Cxx')
      args.append('--with-CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if 'FC' in self.framework.argDB:
      self.framework.pushLanguage('FC')
      args.append('--with-F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if self.mpi.include:
      if len(self.mpi.include) > 1:
        raise RuntimeError("hypre assumes there is a single MPI include directory")
      args.append('--with-mpi-include="'+self.mpi.include[0].replace('-I','')+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-mpi-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-mpi-libs="'+libs+'"')
    args.append('--with-babel=0')
    args.append('--with-mli=0')    
    args.append('--with-FEI=0')    
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"')        
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild HYPRE oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Configuring hypre; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(hypreDir,'src')+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HYPRE: '+str(e))
      try:
        self.logPrint("Compiling hypre; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(hypreDir,'src')+';HYPRE_INSTALL_DIR='+installDir+';export HYPRE_INSTALL_DIR; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on HYPRE: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on HYPRE   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on HYPRE follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on HYPRE *******\n')
        raise RuntimeError('Error running make on HYPRE, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      #need to run ranlib on the libraries using the full path
      try:
        output  = config.base.Configure.executeShellCommand('ranlib '+os.path.join(installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on HYPRE libraries: '+str(e))
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed HYPRE into '+installDir)
    return self.getDir()
  
  def configureLibrary(self):
    '''Find an installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs         = []
    incls        = []
    for l in self.deps:
      if hasattr(l,'dlib'):    libs  += l.dlib
      if hasattr(l,'include'): incls += l.include
      
    for location, lib,incl in self.generateGuesses():
      if not isinstance(lib, list): lib = [lib]
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for library '+location+': '+str(lib)+'\n')
      if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs}):      
        self.lib = lib
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.libraries.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          self.found   = 1
          self.dlib    = self.lib+libs
          self.framework.packages.append(self)
          break
    if not self.found:
      raise RuntimeError('Could not find a functional '+self.name+'\n')

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by hypre'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self)
    # hypre requires LAPACK routine dgels()
    if not self.blasLapack.checkForRoutine('dgels'):
      raise RuntimeError('hypre requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
    self.framework.log.write('Found dgels() in Lapack library as needed by hypre\n')
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
