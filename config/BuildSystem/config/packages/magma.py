import config.package
import os

#class Configure(config.package.CMakePackage):
#  def __init__(self, framework):
#    config.package.CMakePackage.__init__(self, framework)
class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    # disable version check
    #self.version          = '2.5.2'
    #self.minversion       = '2.5.2'
    #self.versionname      = ???
    #self.gitcommit        = 'v'+self.version
    version               = '2.5.4'
    self.gitcommit        = 'v'+version
    # hg stashing mechanism seems broken
    #self.download         = ['https://bitbucket.org/icl/magma/get/'+self.gitcommit+'.tar.gz','hg://https://bitbucket.org/icl/magma']
    self.download         = ['https://bitbucket.org/icl/magma/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['icl-magma']
    self.functions        = ['magma_init']
    self.includes         = ['magma.h']
    self.liblist          = [['libmagma_sparse.a','libmagma.a'],
                             ['libmagma_sparse.a','libmagma.a','libpthread.a']]
    self.hastests         = 0
    self.hastestsdatafiles= 0
    self.requirec99flag   = 1 #From CMakeLists.txt -> some code may not compile
    self.precisions       = ['double']
    self.cxx              = 1
    self.requirescxx11    = 1 #From CMakeLists.txt -> some code may not compile
    self.makerulename     = 'lib sparse-lib'
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('MAGMA', '-with-magma-gputarget', nargs.ArgString(None, '', 'GPU_TARGET make variable'))
    help.addArgument('MAGMA', '-with-magma-fortran-bindings', nargs.ArgBool(None, 1, 'Compile MAGMA Fortran bindings'))
    return

  def setupDependencies(self, framework):
    #config.package.CMakePackage.setupDependencies(self, framework)
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.cuda       = framework.require('config.packages.cuda',self)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.pthread    = framework.require('config.packages.pthread',self)
    self.odeps      = [self.openmp,self.pthread]
    self.deps       = [self.cuda,self.blasLapack]
    return

# CMAKE disabled (issues on SUMMIT)
#  def formCMakeConfigureArgs(self):
#    import os
#    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
#    args.append('-DLAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
#    if not hasattr(self.compilers, 'FC'):
#      args.append('-DUSE_FORTRAN=OFF')
#
#    #TODO Pass NVCC down?, OpenMP?
#    #TODO BuildType? None Debug Release RelWithDebInfo
#    #TODO FortranMangling?
#
#    if self.openmp.found:
#      self.usesopenmp = 'yes'
#
#    # need to generate the list of files if not present (TODO not the proper place. Only if commit changes?)
#    if not os.path.isfile(os.path.join(self.packageDir,'CMake.src')):
#      self.logPrintBox('Configuring '+self.PACKAGE+' with cmake, create CMake.src; this may take several minutes')
#      try:
#        self.executeShellCommand('echo "FORT = true" > make.inc && make generate',cwd=self.packageDir,log=self.log)
#      except RuntimeError as e:
#        raise RuntimeError('Could not create CMake.src\nError: '+str(e))
#    return args

  def Install(self):
    import os

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

    cflags += ' -DNDEBUG'
    if self.blasLapack.mkl:
      cflags += ' -DMAGMA_WITH_MKL'

    # blas/lapack name mangling
    # it seems MAGMA (as of v2.5.2) does not support double underscores
    if self.blasLapack.mangling == 'underscore':
      mangle = ' -DADD_'
    elif self.blasLapack.mangling == 'caps':
      mangle = ' -DUPCASE_'
    else:
      mangle = ' -DNOCHANGE_'
    cflags += mangle
    cxxflags += mangle
    fcflags += mangle
    nvccflags += mangle

    ldflags = self.setCompilers.LDFLAGS
    if self.openmp.found:
      ldflags += ' ' + self.openmp.ompflag

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      g.write('CC = '+cc+'\n')
      g.write('CFLAGS = '+cflags+'\n')
      g.write('CXX = '+cxx+'\n')
      g.write('CXXFLAGS = '+cxxflags+'\n')
      g.write('NVCC = '+nvcc+'\n')
      g.write('NVCCFLAGS = '+nvccflags+'\n')
      if fcbindings:
        g.write('FORT = '+fc+'\n')
        g.write('FFLAGS = '+fcflags+'\n')
        g.write('F90LAGS = '+fcflags+'\n')
      if self.argDB['with-magma-gputarget']:
        g.write('GPU_TARGET = '+self.argDB['with-magma-gputarget']+'\n')
      if hasattr(self.cuda,'gencodearch') and self.cuda.gencodearch:
        g.write('NVCCFLAGS += -gencode arch=compute_'+self.cuda.gencodearch+',code=sm_'+self.cuda.gencodearch+'\n')
        g.write('MIN_ARCH = '+self.cuda.gencodearch+'0\n')

      g.write('ARCH = '+self.setCompilers.AR+'\n')
      g.write('ARCHFLAGS = '+self.setCompilers.AR_FLAGS+'\n')
      g.write('RANLIB = '+self.setCompilers.RANLIB+'\n')
      g.write('LDFLAGS = '+ldflags+'\n')
      g.write('INC = '+self.headers.toString(self.blasLapack.include)+'\n')
      g.write('INC += '+self.headers.toString(self.cuda.include)+'\n')
      g.write('LIB = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
      g.write('LIB += '+self.libraries.toString(self.cuda.lib)+'\n')
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
        # make -j seems broken
        output2,err2,ret2 = config.package.Package.executeShellCommand(self.make.make_jnp + ' ' + self.makerulename, cwd=self.packageDir, timeout=2500, log = self.log)
        #output2,err2,ret2 = config.package.Package.executeShellCommand(self.make.make + ' ' + self.makerulename, cwd=self.packageDir, timeout=2500, log = self.log)
        # magma install (2.5.2) is broken when fortran bindings are not requested
        dummymod = os.path.join(self.packageDir,'include','magma_petsc_dummy.mod')
        if not fcbindings and not os.path.isfile(dummymod):
          self.executeShellCommand('echo "!dummy mod" > '+dummymod,cwd=self.packageDir,log=self.log)
        self.logPrintBox('Installing MAGMA; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+' '+self.make.make + ' install', cwd=self.packageDir, timeout=2500, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make on MAGMA: '+str(e))
        raise RuntimeError('Error running make on MAGMA')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
