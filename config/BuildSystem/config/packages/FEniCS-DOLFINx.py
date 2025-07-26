import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version         = '0.9.0.post1'
    self.gitcommit       = 'v'+self.version
    self.download        = ['git://https://github.com/FEniCS/dolfinx/']
    self.functions       = []
    self.includes        = ['dolfinx/common/dolfinx_common.h']
    self.liblist         = [['libdolfinx.a']]
    self.buildLanguages  = ['Cxx']
    self.pkgname         = 'dolfinx'
    self.cmakelistsdir   = 'cpp'
    self.builtafterpetsc = 1
    self.useddirectly    = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags     = framework.require('config.compilerFlags', self)
    self.mpi4py            = framework.require('config.packages.mpi4py',self)
    self.petsc4py          = framework.require('config.packages.petsc4py',self)
    self.boost             = framework.require('config.packages.boost',self)
    self.basix             = framework.require('config.packages.fenics-basix',self)
    self.ffcx              = framework.require('config.packages.fenics_ffcx',self)
    self.scikit_build_core = framework.require('config.packages.scikit_build_core',self)
    self.pathspec          = framework.require('config.packages.pathspec',self)
    self.nanobind          = framework.require('config.packages.nanobind',self)
    self.hdf5              = framework.require('config.packages.HDF5',self)
    self.parmetis          = framework.require('config.packages.ParMETIS',self)
    self.ptscotch          = framework.require('config.packages.PTSCOTCH',self)
    self.python            = framework.require('config.packages.Python',self)
    self.pugixml           = framework.require('config.packages.pugixml',self)
    self.spdlog            = framework.require('config.packages.spdlog',self)
    self.slepc             = framework.require('config.packages.SLEPc',self)
    self.deps              = [self.mpi4py,self.petsc4py,self.boost,self.basix,self.ffcx,self.hdf5,self.pugixml,self.spdlog,self.scikit_build_core,self.nanobind,self.slepc]
    self.odeps             = [self.parmetis,self.ptscotch]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DDOLFINX_ENABLE_PETSC=ON')
    args.append('-DDOLFINX_ENABLE_SLEPC=ON')
    args.append('-DHDF5_DIR=' + self.hdf5.include[0])
    if self.parmetis.found:
      args.append('-DDOLFINX_ENABLE_PARMETIS=ON')
    if self.ptscotch.found:
      args.append('-DDOLFINX_ENABLE_SCOTCH=ON')
    if not self.parmetis.found and not self.ptscotch.found:
      raise RuntimeError('PETSc must provide either ParMETIS or PTSCOTCH, suggest --download-parmetis --download-metis')

    found_hdf5_cpp_binding = False
    for l in self.hdf5.lib:
      if 'libhdf5_cpp' in l:
        found_hdf5_cpp_binding = True
        break
    if not found_hdf5_cpp_binding:
      raise RuntimeError("FEniCS-DOLFINx requires HDF5 with C++ bindings, ensure you use --with-hdf5-cxx-bindings")
    return args

  def Install(self):
    # To avoid having the Python FEniCS-DOLFINx build its dependencies --no-deps is passed to pip (the dependencies are all managed with setupDependencies() above)
    output,err,ret  = config.package.Package.executeShellCommand('git describe --abbrev=12 --dirty --always --tags', cwd=self.packageDir)
    if not err and not ret:
      self.foundversion = output

    args = self.formCMakeConfigureArgs()
    if self.download and self.argDB['download-'+self.downloadname.lower()+'-cmake-arguments']:
       args.append(self.argDB['download-'+self.downloadname.lower()+'-cmake-arguments'])
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()

    if not self.installNeeded(conffile):
      return self.installDir
    if not self.cmake.found:
      raise RuntimeError('CMake not found, needed to build '+self.PACKAGE+'. Rerun configure with --download-cmake.')

    # effectively, this is 'make clean'
    folder = os.path.join(self.packageDir, self.cmakelistsdir, 'petsc-build')
    if os.path.isdir(folder):
      import shutil
      shutil.rmtree(folder)
    os.mkdir(folder)

    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

    # these checks are usually done in configureLibrary
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.directory = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      self.include_a = '-I'+os.path.join(os.path.abspath(os.path.expanduser(self.argDB['prefix'])),'include')
      self.lib_a = [os.path.join(os.path.abspath(os.path.expanduser(self.argDB['prefix'])),'lib',self.liblist[0][0])]
    else:
      self.directory = self.petscdir.dir
      self.include_a = '-I'+os.path.join(self.petscdir.dir,self.arch,'include')
      self.lib_a = [os.path.join(self.petscdir.dir,self.arch,'lib',self.liblist[0][0])]
    self.found_a     = 1
    self.addDefine('HAVE_DOLFINX', 1)
    self.addMakeMacro('DOLFINX_LIB',' '.join(map(self.libraries.getLibArgument, self.lib_a)))
    self.addMakeMacro('DOLFINX_INCLUDE',self.include_a)

    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
       carg = 'PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH="" SLEPC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))
       prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
       prefix = os.path.join(self.petscdir.dir,self.arch)
       carg = ''

    # provide access to mpi4py, petsc4py and FEnicS/ffcx.py to Python
    ppath = 'PYTHONPATH=' + os.path.join(self.installDir,'lib')
    dpath = 'DOLFINX_DIR=' + self.installDir + ' HDF5_ROOT="'+self.installDir+'" HDF5_ENABLE_PARALLEL=on CMAKE_PREFIX_PATH="' + self.pugixml.installDir + ':' + self.spdlog.installDir + '"'

    self.addDefine('HAVE_DOLFINX',1)
    self.addMakeMacro('DOLFINX','yes')

    ccarg = 'CC=' + self.compilers.CC
    if 'Cxx' in self.buildLanguages:
      self.pushLanguage('C++')
      ccarg += ' CXX=' + self.compilers.CXX
      ccarg += ' CXXFLAGS="' +  self.updatePackageCxxFlags(self.getCompilerFlags()) + '"'
      self.popLanguage()

    self.addPost(folder, [carg + ' ' + ppath + ' ' + self.cmake.cmake + ' .. ' + args,
                          self.make.make_jnp + ' ' + self.makerulename,
                          '${OMAKE} install',
                          'cd ../../python && ' + ccarg + ' ' + ppath + ' ' + dpath + ' ' + self.python.pyexe +  ' -m  pip install --no-build-isolation --no-deps --upgrade-strategy only-if-needed --upgrade --target=' + os.path.join(self.installDir,'lib') + ' .'])
    self.python.path.add(os.path.join(self.installDir,'lib'))
    return self.installDir
