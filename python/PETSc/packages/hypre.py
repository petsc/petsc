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
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Download LLNL hypre preconditioners'))
    return

  def generateIncludeGuesses(self):
    if self.framework.argDB['download-'+self.package] == 1 or self.framework.argDB['download-'+self.package] == 2:
      dir = self.downLoadhypre()
      yield('based on downloaded directory',os.path.join(dir,self.framework.argDB['PETSC_ARCH'],'include'))      
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
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibList(self,dir):
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
    return alllibs
          
  def generateLibGuesses(self):
    if self.framework.argDB['download-'+self.package] == 1:
      dir = os.path.join(self.downLoadhypre(),self.framework.argDB['PETSC_ARCH'])
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

  def getDir(self):
    '''Find the directory containing HYPRE'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    hypreDir = None
    for dir in os.listdir(packages):
      if dir.startswith('hypre') and os.path.isdir(os.path.join(packages, dir)):
        hypreDir = dir
    if hypreDir is None:
      self.framework.log.write('Did not located already downloaded HYPRE\n')
      raise RuntimeError('Error locating HYPRE directory')
    return os.path.join(packages, hypreDir)

  def downLoadhypre(self):
    self.framework.log.write('Downloading hypre\n')
    try:
      hypreDir = self.getDir()
      self.framework.log.write('HYPRE already downloaded, no need to ftp\n')
    except RuntimeError:
      import urllib

      packages = self.framework.argDB['with-external-packages-dir']
      if not os.path.isfile(os.path.expanduser(os.path.join('~','.hypre_license'))):
        print "**************************************************************************************************"
        print "You must register to use hypre at http://www.llnl.gov/CASC/hypre/download/hyprebeta_cur_agree.html"
        print "       Once you have registered, configure will download and install hypre for you                 "
        print "**************************************************************************************************"
        fd = open(os.path.expanduser(os.path.join('~','.hypre_license')),'w')
        fd.close()
      
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/tmp/hypre.tar.gz', os.path.join(packages, 'hypre.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading HYPRE: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip hypre.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping hypre.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf hypre.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf hypre.tar: '+str(e))
      os.unlink(os.path.join(packages, 'hypre.tar'))
      self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+' into '+self.getDir())
    # Get the HYPRE directories
    hypreDir = self.getDir()
    installDir = os.path.join(hypreDir, self.framework.argDB['PETSC_ARCH'])
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
    blas = self.blasLapack.lapackLibrary
    if not None in self.blasLapack.blasLibrary:
      blas = blas+self.blasLapack.blasLibrary
    blas = ' '.join([self.libraries.getLibArgument(lib) for lib in blas])
    if hasattr(self.compilers,'flibs'): blas += ' '+self.compilers.flibs
    args.append('--with-blas="'+blas+'"')        
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
    return

  def setFoundOutput(self):
    self.framework.packages.append(self)
    
  def configure(self):
    if 'download-'+self.package in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
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
