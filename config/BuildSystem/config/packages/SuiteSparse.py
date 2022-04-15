import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self,framework)
    self.minversion        = '5.6.0'
    self.version           = '5.12.0'
    self.versioninclude    = 'SuiteSparse_config.h'
    self.versionname       = 'SUITESPARSE_MAIN_VERSION.SUITESPARSE_SUB_VERSION.SUITESPARSE_SUBSUB_VERSION'
    self.gitcommit         = 'v'+self.version
    self.download          = ['git://https://github.com/DrTimothyAldenDavis/SuiteSparse','https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/'+self.gitcommit+'.tar.gz']
    self.download_solaris  = ['https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.6.0.tar.gz']
    self.liblist           = [['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a'],
                             ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a','librt.a'],
                             ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a'],
                             ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a','librt.a']]
    self.functions         = ['umfpack_dl_wsolve','cholmod_l_solve','klu_l_solve','SuiteSparseQR_C_solve']
    self.includes          = ['umfpack.h','cholmod.h','klu.h','SuiteSparseQR_C.h']
    self.hastests          = 1
    self.buildLanguages    = ['Cxx']
    self.hastestsdatafiles = 1
    self.precisions        = ['double']
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    # This is set to 1 since CHOLMOD is broken with GPU support (does not even compile with icc on my workstation)
    # see https://github.com/DrTimothyAldenDavis/SuiteSparse/issues/5
    help.addArgument('SUITESPARSE', '-download-suitesparse-disablegpu=<bool>',    nargs.ArgBool(None, 1, 'Force disabling SuiteSparse/CHOLMOD use of GPUs'))

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.blasLapack,self.mathlib]
    self.cuda       = framework.require('config.packages.cuda',self)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.metis      = framework.require('config.packages.metis',self)
    self.odeps      = [self.openmp,self.cuda,self.metis]
    return

  def Install(self):
    import os
    self.log.write('SuiteSparseDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    if not self.make.haveGNUMake:
      raise RuntimeError('SuiteSparse buildtools require GNUMake. Use --with-make=gmake or --download-make')

    # Use CHOLMOD_OMP_NUM_THREADS to control the number of threads
    if self.openmp.found:
      self.usesopenmp = 'yes'

    # From v4.5.0, SuiteSparse_config/SuiteSparse_config.mk is not modifiable anymore. Instead, we must override make variables
    args=[]

    self.pushLanguage('C')
    args.append('CC="'+self.getCompiler()+'"')
    cflags=self.updatePackageCFlags(self.getCompilerFlags())
    if self.checkSharedLibrariesEnabled():
      ldflags=self.getDynamicLinkerFlags()
    else:
      ldflags=''
    ldflags += ' '+self.setCompilers.LDFLAGS
    # SuiteSparse 5.6.0 makefile has a bug in how it treats LDFLAGS (not using the override directive)
    ldflags+=" -L\$(INSTALL_LIB)"
    self.popLanguage()

    # CHOLMOD may build the shared library with CXX
    with self.Language('Cxx'):
      args.append(self.getCompiler().join(('CXX="','"')))

    args.append('MAKE="'+self.make.make+'"')
    args.append('RANLIB="'+self.setCompilers.RANLIB+'"')
    args.append('ARCHIVE="'+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'"')
    args.append('RM="'+self.programs.RM+'"')
    args.append('MV="'+self.programs.mv+'"')
    args.append('CP="'+self.programs.cp+'"')
    args.append('LDFLAGS="'+ldflags+'"')
    args.append('INSTALL_LIB='+self.libDir)
    args.append('INSTALL_INCLUDE='+self.includeDir)
    args.append('INSTALL_DOC='+self.installDir+'/share/doc/suitesparse')
    args.append('BLAS="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('LAPACK="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    # fix for bug in SuiteSparse
    if self.setCompilers.isDarwin(self.log):
      args.append('LDLIBS=""')
    if self.blasLapack.mangling == 'underscore':
      flg = ''
    elif self.blasLapack.mangling == 'caps':
      flg = '-DBLAS_CAPS_DOES_NOT_WORK'
    else:
      flg = '-DBLAS_NO_UNDERSCORE'
    args.append('UMFPACK_CONFIG='+flg)

    if self.metis.found:
      # '-I' is added automatically inside SuiteSparse_config.mk
      metisinc = self.headers.toString(self.metis.include).replace('-I','',1)
      args.append('MY_METIS_INC="'+metisinc+'"')
      args.append('MY_METIS_LIB="'+self.libraries.toString(self.metis.dlib)+'"')
    else:
      flg+=' -DNPARTITION'

    # CUDA support for 64bit indices installations only
    if self.cuda.found and self.defaultIndexSize == 64 and not self.argDB['download-suitesparse-disablegpu']:
      self.logPrintBox('SuiteSparse: Enabling support for CHOLMOD on GPUs (it can be disabled with --download-suitesparse-disablegpu=1)')
      args.append('CF="'+cflags+' -D_GNU_SOURCE"') # The GPU code branches use feenableexcept including fenv.h only
      self.pushLanguage('CUDA')
      petscNvcc = self.getCompiler()
      cudaFlags = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(petscNvcc,getFullPath=1,resultName='systemNvcc')
      if hasattr(self,'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
      else:
        raise RuntimeError('Unable to locate CUDA NVCC compiler')
      args.append('CUDA_ROOT='+cudaDir)
      args.append('GPU_BLAS_PATH='+cudaDir)
      args.append('CUDA_PATH='+cudaDir)
      args.append('CUDART_LIB='+cudaDir+'/lib64/libcudart.so')
      args.append('CUBLAS_LIB='+cudaDir+'/lib64/libcublas.so')
      args.append('CUDA_INC_PATH='+cudaDir+'/include')
      args.append('NVCCFLAGS="'+cudaFlags+' -Xcompiler -fPIC"')
      args.append('CHOLMOD_CONFIG="'+flg+' -DGPU_BLAS"')
      self.addDefine('USE_SUITESPARSE_GPU',1)
    else:
      if self.cuda.found and not self.argDB['download-suitesparse-disablegpu']:
        self.logPrintBox('SuiteSparse: Cannot enable support for GPUs. SuiteSparse only uses GPUs with --with-64-bit-indices')
      args.append('CF="'+cflags+'"')
      args.append('CHOLMOD_CONFIG="'+flg+'"')
      args.append('CUDA=no')
      args.append('CUDA_PATH=')

    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded(conffile):
      try:
        self.logPrintBox('Compiling and installing SuiteSparse; this may take several minutes')
        makewithargs=self.make.make+' '+args
        # SuiteSparse install may not create missing directories, hence we need to create them first
        output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
        if self.checkSharedLibrariesEnabled():
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/SuiteSparse_config && '+makewithargs+' clean && '+makewithargs+' && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/AMD                && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/COLAMD             && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/BTF                && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/CAMD               && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/CCOLAMD            && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/CHOLMOD            && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/UMFPACK            && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/KLU                && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/SPQR               && '+makewithargs+' clean && '+makewithargs+' library && '+makewithargs+' install && '+makewithargs+' clean', timeout=2500, log=self.log)
        else:
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/SuiteSparse_config && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' *h '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' libsuitesparseconfig.* '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/AMD                && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','amd.h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libamd.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/COLAMD             && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','*h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libcolamd.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/BTF                && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','btf.h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libbtf.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/CAMD               && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','camd.h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libcamd.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/CCOLAMD            && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','*h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libccolamd.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/CHOLMOD            && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','*h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libcholmod.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/UMFPACK            && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','*h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libumfpack.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/KLU                && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','*h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libklu.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)
          output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+'/SPQR               && '+makewithargs+' clean && '+makewithargs+' static && '+self.programs.cp+' '+os.path.join('Include','*h')+' '+os.path.join(self.installDir,'include')+' && '+self.programs.cp+' '+os.path.join('Lib','libspqr.*')+' '+os.path.join(self.installDir,'lib')+' && '+makewithargs+' clean', timeout=2500, log=self.log)

        self.addDefine('HAVE_SUITESPARSE',1)
      except RuntimeError as e:
        raise RuntimeError('Error running make on SuiteSparse: '+str(e))
      self.postInstall(output+err, conffile)
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package] and self.defaultIndexSize == 64 and self.types.sizes['void-p'] == 4:
      raise RuntimeError('SuiteSparse does not support 64bit indices in 32bit (pointer) mode.')
    return
