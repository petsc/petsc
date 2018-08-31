import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['git://https://github.com/libMesh/libmesh.git', 'https://github.com/libMesh/libmesh/releases/download/v1.2.1/libmesh-1.2.1.tar.gz']
    self.gitcommit = '262c8c4d4c9669b6116084260baa39633743f4c9'
    self.functions = []
    self.includes  = ['libmesh/libmesh.h', 'libmesh/libmesh_config.h']
    self.liblist   = [['libmesh.la']]
    self.pkgname   = 'libmesh-1.2.1'
    self.cxx             = 1
    self.useddirectly    = 0
    self.linkedbypetsc   = 0
    self.builtafterpetsc = 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('LIBMESH', '-with-libmesh-method=<method>', nargs.Arg(None, 'dbg', 'User may provide which "METHOD" to compile against. Valid options: dbg, opt'))

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.mpi             = framework.require('config.packages.MPI',  self)
    self.boost           = framework.require('config.packages.boost',self)
    self.deps            = [self.mpi, self.boost]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if not self.argDB['download-libmesh']: #using User specified libmesh
      if self.argDB['with-libmesh-method'].lower() == 'dbg':
        args.append('--with-methods=dbg')
        args.append('--disable-glibcxx-debugging')
      elif self.argDB['with-libmesh-method'].lower() == 'opt':
        args.append('--with-methods=opt')
      else:
        raise RuntimeError('Valid options for --with-libmesh-method=<string> are OPT or DBG')
    elif self.compilerFlags.debugging:
      args.append('--with-methods=dbg')
      args.append('--disable-glibcxx-debugging')
    else:
      args.append('--with-methods=opt')
    args.append('--disable-cxx11')
    args.append('--disable-openmp')
    args.append('--disable-perflog')
    args.append('--disable-pthreads')
    args.append('--disable-cppthreads')
    args.append('--disable-unique-ptr')
    args.append('PETSC_DIR='+self.petscdir.dir)
    args.append('PETSC_ARCH='+os.environ['PETSC_ARCH'])
    args.append('--with-boost='+os.path.join(self.boost.directory,self.boost.includedir))
    args.append('--with-boost-libdir='+os.path.join(self.boost.directory,self.boost.libdir))
    return args
