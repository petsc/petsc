import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'v5.0.0'
    self.download         = ['http://crd-legacy.lbl.gov/~xiaoye/SuperLU/superlu_dist_5.0.0.tar.gz','git://https://github.com/xiaoyeli/superlu_dist']
    self.functions        = ['set_default_options_dist']
    self.includes         = ['superlu_ddefs.h']
    self.liblist          = [['libsuperlu_dist.a']]
    # SuperLU_Dist does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.downloadonWindows= 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    self.requirec99flag   = 1 # SuperLU_Dist uses C99 features
    return

  def setupHelp(self, help):
    import nargs
    config.package.CMakePackage.setupHelp(self, help)
    help.addArgument('SUPERLU_DIST', '-download-superlu_dist-gpu=<bool>',    nargs.ArgBool(None, 0, 'Install Superlu_DIST to use GPUs'))

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.metis          = framework.require('config.packages.metis',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    if self.framework.argDB['download-superlu_dist-gpu']:
      self.cuda           = framework.require('config.packages.cuda',self)
      self.openmp         = framework.require('config.packages.openmp',self)
      self.deps           = [self.mpi,self.blasLapack,self.parmetis,self.metis,self.cuda,self.openmp]
    else:
      self.deps           = [self.mpi,self.blasLapack,self.parmetis,self.metis]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')

    metis_inc = self.headers.toStringNoDupes(self.metis.include)[2:]
    parmetis_inc = self.headers.toStringNoDupes(self.parmetis.include)[2:]
    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_PARMETIS_INCLUDE_DIRS="'+metis_inc+';'+parmetis_inc+'"')
    args.append('-DTPL_PARMETIS_LIBRARIES="'+self.libraries.toString(self.parmetis.lib+self.metis.lib)+'"')

    if self.indexTypes.integerSize == 64:
      args.append('-DXSDK_INDEX_SIZE=64')

    args.append('-Denable_tests=0')
    args.append('-Denable_examples=0')
    #  CMake in SuperLU should set this; but like many other packages it does not
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')
    args.append('-DMPI_C_COMPILER:STRING="'+self.framework.getCompiler()+'"')
    args.append('-DMPI_C_COMPILE_FLAGS:STRING=""')
    args.append('-DMPI_C_INCLUDE_PATH:STRING=""')
    args.append('-DMPI_C_LIBRARIES:STRING=""')

    # Remove -DAdd_ from superlu cflags
    try:
      import shutil
      shutil.move(os.path.join(self.packageDir,'CMakeLists.txt'),
                  os.path.join(self.packageDir,'CMakeLists.txt.orig'))
      output,err,ret  = config.package.CMakePackage.executeShellCommand("sed -e 's/-DAdd_ //' %s > %s" % (os.path.join(self.packageDir,'CMakeLists.txt.orig'),
                          os.path.join(self.packageDir,'CMakeLists.txt')))
      output = output+err
      self.log.write(output)
    except RuntimeError, e:
       raise RuntimeError('Error running sed on SuperLU_DIST CMakeLists.txt: '+str(e))

    # Add in the correct flag
    if self.blasLapack.mangling == 'underscore':
      mangledef = '-DAdd_'
    elif self.blasLapack.mangling == 'caps':
      mangledef = '-DUpCase'
    else:
      mangledef = '-DNoChange'
    for place,item in enumerate(args):
      if item.find('CMAKE_C_FLAGS') >= 0 or item.find('CMAKE_CXX_FLAGS') >= 0:
        args[place]=item[:-1]+' '+mangledef+'"'


    return args


 #   if self.framework.argDB['download-superlu_dist-gpu']:
 #     g.write('ACC          = GPU\n')
 #     g.write('CUDAFLAGS    = -DGPU_ACC '+self.headers.toString(self.cuda.include)+'\n')
 #     g.write('CUDALIB      = '+self.libraries.toString(self.cuda.lib)+'\n')
 #   else:
 #     g.write('ACC          = \n')
 #     g.write('CUDAFLAGS    = \n')
 #     g.write('CUDALIB      = \n')



