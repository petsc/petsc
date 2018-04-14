import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'master'  #master+
    self.download               = ['git://https://github.com/libMesh/libmesh.git']
    self.functions              = []
    self.includes               = []
    self.skippackagewithoptions = 1
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

    #  if installing as Superuser than want to return to regular user for clean and build
    if self.installSudo:
       newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
       newuser = ''

    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix']:
       newdir = 'PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' '
       prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
       newdir = ' '
       prefix = os.path.join(self.petscdir.dir,self.arch)

    self.addDefine('HAVE_LIBMESH',1)
    self.addMakeMacro('LIBMESH','yes')
    self.addMakeRule('libmeshbuild','', \
                       ['@echo "*** Building libmesh ***"',\
                          '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/libmesh.errorflg',\
                          '@(cd '+self.packageDir+' && \\\n\
           '+newdir+' ./configure --prefix='+prefix+' && \\\n\
           '+newdir+' '+self.make.make_jnp+' ) > ${PETSC_ARCH}/lib/petsc/conf/libmesh.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building libmesh. Check ${PETSC_ARCH}/lib/petsc/conf/libmesh.log" && \\\n\
             echo "********************************************************************" && \\\n\
             touch ${PETSC_ARCH}/lib/petsc/conf/libmesh.errorflg && \\\n\
             exit 1)'])
    self.addMakeRule('libmeshinstall','', \
                       ['@echo "*** Installing libmesh ***"',\
                          '@(cd '+self.packageDir+' && \\\n\
           '+newuser+newdir+' make install ) >> ${PETSC_ARCH}/lib/petsc/conf/libmesh.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building libmesh. Check ${PETSC_ARCH}/lib/petsc/conf/libmesh.log" && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)'])
    if self.argDB['prefix']:
      self.addMakeRule('libmesh-build','')
      # the build must be done at install time because PETSc shared libraries must be in final location before building libmesh
      self.addMakeRule('libmesh-install','libmeshbuild libmeshinstall')
    else:
      self.addMakeRule('libmesh-build','libmeshbuild libmeshinstall')
      self.addMakeRule('libmesh-install','')

    return self.installDir

  def alternateConfigureLibrary(self):
    self.addMakeRule('libmesh-build','')
    self.addMakeRule('libmesh-install','')

