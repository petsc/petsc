import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion       = '2020'
    self.versionname      = '__SYCL_COMPILER_VERSION'
    self.versioninclude  = 'CL/sycl/version.hpp'
    self.requiresversion = 2200
    # CL/sycl.h is dpcpp.  Other SYCL impls may use SYCL/sycl.hpp -- defer
    self.includes         = ['CL/sycl.hpp']
    self.includedir       = 'include/sycl'
    self.functionsCxx     = [1,'namespace sycl = cl;','sycl::device::get_devices()']
    # Unlike CUDA or HIP, the blas issues are just part of MKL and handled as such.
    self.liblist          = [['libsycl.a'],
                             ['sycl.lib'],]
    self.precisions       = ['single','double']
    self.cxx              = 1
    self.complex          = 1
    self.hastests         = 0
    self.hastestsdatafiles= 0

    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    return

  def getSearchDirectories(self):
    import os
    self.pushLanguage('SYCL')
    petscSycl = self.getCompiler()
    self.popLanguage()
    self.getExecutable(petscSycl,getFullPath=1,resultName='systemDpcpp')
    if hasattr(self,'systemDpcpp'):
      dpcppDir = os.path.dirname(self.systemSyclcxx)
      dpcDir = os.path.split(dpcppDir)[0]
      yield dpcDir
    return

  def checkSizeofVoidP(self):
    '''Checks if the SYCLCXX compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in SYCL is the same as with regular compiler\n')
    size = self.types.checkSizeof('void *', (8, 4), lang='SYCL', save=False)
    if size != self.types.sizes['void-p']:
      raise RuntimeError('SYCL Error: sizeof(void*) with SYCL compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    return

  def configureTypes(self):
    import config.setCompilers
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with SYCL')
    self.checkSizeofVoidP()
    return

  def checkSYCLCXXDoubleAlign(self):
    if 'known-sycl-align-double' in self.argDB:
      if not self.argDB['known-sycl-align-double']:
        raise RuntimeError('SYCL error: PETSC currently requires that SYCL double alignment match the C compiler')
    else:
      typedef = 'typedef struct {double a; int b;} teststruct;\n'
      sycl_size = self.types.checkSizeof('teststruct', (16, 12), lang='SYCL', codeBegin=typedef, save=False)
      c_size = self.types.checkSizeof('teststruct', (16, 12), lang='C', codeBegin=typedef, save=False)
      if c_size != sycl_size:
        raise RuntimeError('SYCL compiler error: memory alignment doesn\'t match C compiler (try adding -malign-double to compiler options)')
    return

  def configureLibrary(self):
    self.libraries.pushLanguage('SYCL')
    self.addDefine('HAVE_SYCL','1')
    config.package.Package.configureLibrary(self)
    #self.checkSYCLCXXDoubleAlign()
    self.configureTypes()
    self.libraries.popLanguage()
    return
