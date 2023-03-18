import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'd086b42987308d3f7585a8cbf1070065d766cfaa' # v0.10.0 on Mar 21, 2022
    self.download               = ['git://https://github.com/CEED/libceed.git']
    self.functions              = ['CeedRegister']
    self.includes               = ['ceed.h']
    self.liblist                = [['libceed.a']]
    self.requires32bitint       = 1;   # TODO: fix for 64 bit integers
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.make            = framework.require('config.packages.make',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.odeps           = [self.cuda,self.hip]
    return

  def Install(self):
    import os
    # TODO: maybe add support for various backends, CUDA, libXSMM, OCCA, MAGMA?
    args = ['prefix={0}'.format(self.installDir), 'V=1']
    with self.Language('C'):
      args += [
        'CC=' + self.getCompiler(),
        'CFLAGS=' + self.getCompilerFlags(),
      ]
    with self.Language('Cxx'):
      args += [
        'CXX=' + self.getCompiler(),
        'CXXFLAGS=' + self.getCompilerFlags(),
      ]
    if self.cuda.found:
      with self.Language('CUDA'):
        args += [
          'CUDA_DIR=' + self.cuda.cudaDir,
          'NVCC=' + self.getCompiler(),
          'NVCCFLAGS=' + self.getCompilerFlags(),
          'CUDA_ARCH=sm_' + self.cuda.cudaArchSingle()
        ]
    if self.hip.found:
      with self.Language('HIP'):
        args += [
          'HIP_DIR={0}'.format(self.hip.hipDir),
          'HIPCC={0}'.format(self.getCompiler()),
          'HIPFLAGS={0}'.format(self.getCompilerFlags()),
          'HIP_ARCH={0}'.format(self.hip.hipArch),
        ]
    if self.setCompilers.LDFLAGS: args += ['LDFLAGS={0}'.format(self.setCompilers.LDFLAGS)]
    try:
      self.logPrintBox('Compiling libceed; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand(self.make.make_jnp_list + args + ['-B'], cwd=self.packageDir, timeout=250, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on libceed: '+str(e))
    try:
      self.logPrintBox('Installing libceed; this may take several seconds')
      output,err,ret  = config.package.Package.executeShellCommand(self.make.make_jnp_list + args + ['install'], cwd=self.packageDir, timeout=60, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running install on libceed: '+str(e))
    return self.installDir
