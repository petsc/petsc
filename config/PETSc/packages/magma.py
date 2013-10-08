import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-1.4.0.tar.gz']
    self.functions    = ['magma_zgetrf','magma_dgetrf']
    self.includes     = ['magma.h']
    self.liblist      = [['libmagma.a'],['libmagma.a','libmagmablas.a']]
    self.double       = 0
    self.complex      = 1

    self.worksonWindows   = 0
    self.downloadonWindows= 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    # NOTE: CUDA dependency is only here if using CUBLAS
    self.cuda       = self.framework.require('PETSc.packages.cuda',self)
    self.deps       = [self.blasLapack, self.cuda]
    return

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('MAGMA', '-with-magma-device=<HAVE_CUBLAS or HAVE_clAmdBlas or HAVE_MIC>', nargs.ArgString(None, "HAVE_CUBLAS", 'Type of device to use'))
    return

  def Install(self):
    import os
    import re
    import sys

    # set blas name mangling
    if self.blasLapack.mangling == 'underscore':
      self.mangling   = '-DADD_'
    elif self.blasLapack.mangling == 'caps':
      self.mangling   = '-DUPCASE'
    else:
      self.mangling   = '-DNOCHANGE'
    
    self.isshared     = self.framework.argDB['with-shared-libraries']

    
    g = open(os.path.join(self.packageDir,'make.inc'),'w')

    # NOTE: LIB has to be set differently depending on -with-magma-device
    g.write('LIB          = '+self.libraries.toString(self.cuda.lib)+' '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')

    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    self.CCFLAGS_MAGMA    = self.setCompilers.getCompilerFlags()
    if self.isshared and '-fPIC' not in self.CCFLAGS_MAGMA:
        self.CCFLAGS_MAGMA = self.CCFLAGS_MAGMA + ' -fPIC'
    
    if self.setCompilers.isLinux():
      g.write('OPTS         = '+self.CCFLAGS_MAGMA+' '+self.mangling+' -DMAGMA_SETAFFINITY\n')
    else:
      g.write('OPTS         = '+self.CCFLAGS_MAGMA+' '+self.mangling+'\n')
    
    self.setCompilers.popLanguage()
    self.setCompilers.pushLanguage('FC')
    g.write('FORT         = '+self.setCompilers.getCompiler()+'\n')
    self.FCFLAGS_MAGMA    = self.setCompilers.getCompilerFlags()
    if self.isshared and '-fPIC' not in self.FCFLAGS_MAGMA:
        self.FCFLAGS_MAGMA = self.FCFLAGS_MAGMA + ' -fPIC'

    g.write('F77OPTS      = '+self.FCFLAGS_MAGMA+' '+self.mangling+' \n')
    g.write('FOPTS        = '+self.FCFLAGS_MAGMA+' '+self.mangling+' -x f95-cpp-input\n')

    self.setCompilers.popLanguage()
    # NOTE: next bunch of stuff only set for CUBLAS
    self.setCompilers.pushLanguage('CUDA')
    g.write('NVCC         = '+self.setCompilers.getCompiler()+'\n')
    #  Set the GPU_TARGET for MAGMA depending on the GPU detected by PETSc
    self.CUDAFLAGS=self.setCompilers.getCompilerFlags()
    if '-arch=sm_13' in self.CUDAFLAGS:
      g.write('GPU_TARGET   = Tesla\n')
      self.CUDAFLAGS=self.CUDAFLAGS.replace('-arch=sm_13','')
    elif '-arch=sm_20' in self.CUDAFLAGS:
      g.write('GPU_TARGET   = Fermi\n')
      self.CUDAFLAGS=self.CUDAFLAGS.replace('-arch=sm_20','')
    elif  '-arch=sm_30' in self.CUDAFLAGS:
      g.write('GPU_TARGET   = Kepler\n')
      self.CUDAFLAGS=self.CUDAFLAGS.replace('-arch=sm_30','')
    elif  '-arch=sm_35' in self.CUDAFLAGS:
      g.write('GPU_TARGET   = Kepler\n')
      self.CUDAFLAGS=self.CUDAFLAGS.replace('-arch=sm_35','')
    else:
      raise RuntimeError('MAGMA error: GPU_TARGET must be one of Tesla, Fermi, or Kepler.')
    if self.isshared:
        g.write('NVOPTS       = '+self.CUDAFLAGS+' '+self.mangling+' -Xcompiler \"-fno-strict-aliasing -fPIC\"\n')
    else:
        g.write('NVOPTS       = '+self.CUDAFLAGS+' '+self.mangling+' -Xcompiler -fno-strict-aliasing\n')
    
    g.write('LDOPTS       = -fopenmp\n')
    self.setCompilers.popLanguage()
    g.write('CUDADIR      = '+self.cuda.directory+'\n')
    g.write('INC          = -I'+self.cuda.directory+'/include\n')
    g.write('LIBDIR       = -L'+self.cuda.directory+'/lib\n')

    g.close()
    if self.installNeeded('make.inc'):

      try:
        self.logPrintBox('Compiling MAGMA; this may take several minutes')

        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make'+' lib', timeout=2500, log = self.framework.log)
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && mv -f lib/*.* '+libDir+'/. && cp -f include/*.* '+includeDir+'/.', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on MAGMA: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def configureLibrary(self):
    ''' Add the magma needed blas flag'''
    flagsArg = self.setCompilers.getPreprocessorFlagsArg()
    oldFlags = getattr(self.setCompilers, flagsArg)
    setattr(self.setCompilers, flagsArg, oldFlags+' -D'+self.framework.argDB['with-magma-device'])
    PETSc.package.NewPackage.configureLibrary(self)
    setattr(self.setCompilers, flagsArg, oldFlags)
    return
