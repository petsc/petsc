import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version                = '1.8.1'
    self.gitcommit              = 'v' + self.version
    self.download               = ['git://https://github.com/libMesh/libmesh.git','https://github.com/libMesh/libmesh/archive/'+self.gitcommit+'.tar.gz']
    self.functions              = []
    self.includes               = []
    self.skippackagewithoptions = 1
    self.gitsubmodules          = ['.']
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    return

  def Install(self):
    import os
    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix']:
       newdir = 'PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' '
       prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
       newdir = ' '
       prefix = os.path.join(self.petscdir.dir,self.arch)

    self.addDefine('HAVE_LIBMESH',1)
    self.addMakeMacro('LIBMESH','yes')
    self.addPost(self.packageDir, [newdir + ' ./configure --prefix=' + prefix,
                                   newdir + ' ' + self.make.make_jnp,
                                   newdir + ' make install'])
    self.logPrintBox('libMesh examples are available at '+os.path.join(self.packageDir,'examples'))
    return self.installDir

