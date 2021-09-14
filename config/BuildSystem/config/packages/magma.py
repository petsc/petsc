import config.package
import os
import sys

#class Configure(config.package.CMakePackage):
#  def __init__(self, framework):
#    config.package.CMakePackage.__init__(self, framework)
class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    # disable version check
    self.version          = '2.6.1'
    #self.minversion       = '2.6.0'
    #self.versionname      = ???
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://bitbucket.org/icl/magma']
    self.functions        = ['magma_init']
    self.includes         = ['magma_config.h']
    self.liblist          = [['libmagma_sparse.a','libmagma.a'],
                             ['libmagma_sparse.a','libmagma.a','libpthread.a'],
                             ['libmagma.a'],
                             ['libmagma.a','libpthread.a']]
    self.hastests         = 0
    self.hastestsdatafiles= 0
    self.requirec99flag   = 1 #From CMakeLists.txt -> some code may not compile
    self.precisions       = ['single','double']
    self.cxx              = 1
    self.minCxxVersion    = 'c++11' #From CMakeLists.txt -> some code may not compile
    self.makerulename     = ' lib ' #make sparse-lib is broken in many ways
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('MAGMA', '-with-magma-gputarget=<string>', nargs.ArgString(None, '', 'GPU_TARGET make variable'))
    help.addArgument('MAGMA', '-with-magma-fortran-bindings=<bool>', nargs.ArgBool(None, 0, 'Compile MAGMA Fortran bindings'))
    return

  def setupDependencies(self, framework):
    #config.package.CMakePackage.setupDependencies(self, framework)
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.cuda       = framework.require('config.packages.cuda',self)
    self.hip        = framework.require('config.packages.hip',self)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.pthread    = framework.require('config.packages.pthread',self)
    self.odeps      = [self.openmp,self.pthread,self.cuda,self.hip]
    self.deps       = [self.blasLapack]
    return

  def Install(self):
    import os

    if not self.cuda.found and not self.hip.found:
      raise RuntimeError('Need CUDA or HIP')

    usehip = False
    usecuda = False
    if self.hip.found:
      usehip = True
    else:
      usecuda = True

    if self.blasLapack.has64bitindices:
      raise RuntimeError('Not coded for 64bit BlasLapack')

    if self.openmp.found:
      self.usesopenmp = 'yes'

    fcbindings = self.argDB['with-magma-fortran-bindings']
    if fcbindings and not hasattr(self.compilers, 'FC'):
      raise RuntimeError('Missing Fortran compiler for MAGMA Fortran bindings')

    self.pushLanguage('C')
    cc = self.getCompiler()
    cflags = self.getCompilerFlags()
    self.popLanguage()

    self.pushLanguage('Cxx')
    cxx = self.getCompiler()
    cxxflags = self.getCompilerFlags()
    cxxflags = cxxflags.replace('-fvisibility=hidden','')
    self.popLanguage()

    fc = ''
    fcflags = ''
    if fcbindings:
      self.pushLanguage('FC')
      fc = self.getCompiler()
      fcflags = self.getCompilerFlags()
      self.popLanguage()

    nvccflags = ''
    if usecuda:
      self.pushLanguage('CUDA')
      nvcc = self.getCompiler()
      nvccflags = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(nvcc,getFullPath=1,resultName='systemNvcc')
      if hasattr(self,'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
      else:
        raise RuntimeError('Unable to locate CUDA NVCC compiler')

    hipccflags = ''
    if usehip:
      self.pushLanguage('HIP')
      hipcc = self.getCompiler()
      hipccflags = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(hipcc,getFullPath=1,resultName='systemHipc')
      if hasattr(self,'systemHipc'):
        hipccDir = os.path.dirname(self.systemHipc)
        hipDir = os.path.split(hipccDir)[0]
      else:
        raise RuntimeError('Unable to locate HIP compiler')

    cflags += ' -DNDEBUG'
    if self.blasLapack.mkl:
      cflags += ' -DMAGMA_WITH_MKL'

    # blas/lapack name mangling
    # it seems MAGMA (as of v2.5.2) does not support double underscores
    if self.blasLapack.mangling == 'underscore':
      mangle = ' -DADD_'
    elif self.blasLapack.mangling == 'caps':
      mangle = ' -DUPCASE'
    else:
      mangle = ' -DNOCHANGE'
    cflags += mangle
    cxxflags += mangle
    fcflags += mangle
    nvccflags += mangle
    hipccflags += mangle

    ldflags = self.setCompilers.LDFLAGS
    if self.openmp.found:
      ldflags += ' ' + self.openmp.ompflag

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      gputarget = ''
      if self.argDB['with-magma-gputarget']:
        gputarget = self.argDB['with-magma-gputarget']
      elif self.cuda.found and hasattr(self.cuda,'gencodearch') and self.cuda.gencodearch:
        gputarget = 'sm_'+self.cuda.gencodearch
      g.write('CC = '+cc+'\n')
      g.write('CFLAGS = '+cflags+'\n')
      g.write('CXX = '+cxx+'\n')
      g.write('CXXFLAGS = '+cxxflags+'\n')
      if usecuda:
        g.write('BACKEND = cuda\n')
        g.write('NVCC = '+nvcc+'\n')
        g.write('DEVCC = '+nvcc+'\n')
        #g.write('NVCCFLAGS = '+nvccflags+'\n')
        g.write('DEVCCFLAGS = '+nvccflags+'\n')
      if usehip:
        g.write('BACKEND = hip\n')
        g.write('HIPCC = '+hipcc+'\n')
        g.write('DEVCC = '+hipcc+'\n')
        g.write('HIPCCFLAGS = '+hipccflags+'\n')
        g.write('DEVCCFLAGS = '+hipccflags+'\n')
      if fcbindings:
        g.write('FORT = '+fc+'\n')
        g.write('FFLAGS = '+fcflags+'\n')
        g.write('F90LAGS = '+fcflags+'\n')
      if gputarget:
        g.write('GPU_TARGET = '+gputarget+'\n')
      if self.cuda.found and hasattr(self.cuda,'gencodearch') and self.cuda.gencodearch:
        # g.write('NVCCFLAGS += -gencode arch=compute_'+self.cuda.gencodearch+',code=sm_'+self.cuda.gencodearch+'\n')
        g.write('MIN_ARCH = '+self.cuda.gencodearch+'0\n')

      g.write('ARCH = '+self.setCompilers.AR+'\n')
      g.write('ARCHFLAGS = '+self.setCompilers.AR_FLAGS+'\n')
      g.write('RANLIB = '+self.setCompilers.RANLIB+'\n')
      g.write('LDFLAGS = '+ldflags+'\n')
      g.write('INC = '+self.headers.toString(self.blasLapack.include)+'\n')
      g.write('LIB = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
      if usecuda:
        g.write('INC += '+self.headers.toString(self.cuda.include)+'\n')
        g.write('LIB += '+self.libraries.toString(self.cuda.lib)+'\n')
      if usehip:
        g.write('INC += '+self.headers.toString(self.hip.include)+'\n')
        g.write('LIB += '+self.libraries.toString(self.hip.lib)+'\n')

      # blasfix
      if self.setCompilers.isDarwin(self.log):
        g.write('blas_fix = 1\n')
      g.write('prefix = '+self.installDir+'\n')

    if self.installNeeded('make.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('make clean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make clean on MAGMA: '+str(e))
        raise RuntimeError('Error running make clean on MAGMA')
      try:
        self.logPrintBox('Compiling MAGMA; this may take several minutes')
        codegen = ' codegen="' + sys.executable + ' tools/codegen.py"' # as of 2.6.1 they use /usr/bin/env python inside tools/codegen.py
        output2,err2,ret2 = config.package.Package.executeShellCommand(self.make.make_jnp + self.makerulename + codegen, cwd=self.packageDir, timeout=2500, log = self.log)
        # magma install is broken when fortran bindings are not requested
        dummymod = os.path.join(self.packageDir,'include','magma_petsc_dummy.mod')
        if not fcbindings and not os.path.isfile(dummymod):
          self.executeShellCommand('echo "!dummy mod" > '+dummymod,cwd=self.packageDir,log=self.log)
        self.logPrintBox('Installing MAGMA; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        # make install is broken if we are not building the sparse library
        # copy files directly instead of invoking the rule
        if 'sparse-lib' not in self.makerulename:
          incDir = os.path.join(self.installDir,'include')
          libDir = os.path.join(self.installDir,'lib')
          output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' '+self.make.make + ' install_dirs', cwd=self.packageDir, timeout=2500, log = self.log)
          output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' '+self.make.make + ' pkgconfig', cwd=self.packageDir, timeout=2500, log = self.log)
          output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' cp '+os.path.join(self.packageDir,'include','*.h')+' '+incDir, timeout=100, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' cp '+os.path.join(self.packageDir,'include','*.mod')+' '+incDir, timeout=100, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' cp '+os.path.join(self.packageDir,'lib','libmagma.*')+' '+libDir, timeout=100, log=self.log)
        else:
          output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' '+self.make.make + ' install', cwd=self.packageDir, timeout=2500, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make on MAGMA: '+str(e))
        raise RuntimeError('Error running make on MAGMA')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir

  def configureLibrary(self):
    d = None
    if 'with-'+self.package+'-include' in self.argDB:
      inc = self.argDB['with-'+self.package+'-include']
      if inc:
        d = os.path.dirname(inc[0])
    elif 'with-'+self.package+'-dir' in self.argDB:
      d = os.path.join(self.argDB['with-'+self.package+'-dir'],'include')
    if d:
      usecuda = False
      usehip  = False
      with open(os.path.join(d,self.includes[0])) as f:
        magmaconfig = f.read()
        if '#define MAGMA_HAVE_CUDA' in magmaconfig: usecuda = True
        if '#define MAGMA_HAVE_HIP'  in magmaconfig: usehip  = True
      if self.cuda.found and not usecuda:
        raise RuntimeError('Must enable CUDA to use MAGMA built with CUDA')
      if self.hip.found and not usehip:
        raise RuntimeError('Must enable HIP to use MAGMA built with HIP')
    config.package.Package.configureLibrary(self)
