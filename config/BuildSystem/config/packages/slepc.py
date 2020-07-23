import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'bf3875a'
    self.download               = ['git://https://gitlab.com/slepc/slepc.git','https://gitlab.com/slepc/slepc/-/archive/'+self.gitcommit+'/slepc-'+self.gitcommit+'.tar.gz']
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
    self.python          = framework.require('config.packages.python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.parch           = framework.require('PETSc.options.arch',self)
    self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
    return

  def Install(self):
    import os

    #  if installing as Superuser than want to return to regular user for clean and build
    if self.installSudo:
       newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
       newuser = ''

    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
       iarch = 'installed-'+self.parch.nativeArch.replace('linux-','linux2-')
       if self.scalartypes.scalartype != 'real':
         iarch += '-' + self.scalartypes.scalartype
       carg = 'SLEPC_DIR='+self.packageDir+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH="" '
       barg = 'SLEPC_DIR='+self.packageDir+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH='+iarch+' '
       prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
       carg = ' SLEPC_DIR='+self.packageDir+' '
       barg = ' SLEPC_DIR='+self.packageDir+' '
       prefix = os.path.join(self.petscdir.dir,self.arch)

    self.addDefine('HAVE_SLEPC',1)
    self.addMakeMacro('SLEPC','yes')
    self.addMakeRule('slepcbuild','', \
                       ['@echo "*** Building slepc ***"',\
                          '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/slepc.errorflg',\
                          '@(cd '+self.packageDir+' && \\\n\
           '+carg+self.python.pyexe+' ./configure --with-clean --prefix='+prefix+' && \\\n\
           '+barg+'${OMAKE} '+barg+') > ${PETSC_ARCH}/lib/petsc/conf/slepc.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building slepc. Check ${PETSC_ARCH}/lib/petsc/conf/slepc.log" && \\\n\
             echo "********************************************************************" && \\\n\
             touch ${PETSC_ARCH}/lib/petsc/conf/slepc.errorflg && \\\n\
             exit 1)'])
    self.addMakeRule('slepcinstall','', \
                       ['@echo "*** Installing slepc ***"',\
                          '@(cd '+self.packageDir+' && \\\n\
           '+newuser+barg+'${OMAKE} install '+barg+') >> ${PETSC_ARCH}/lib/petsc/conf/slepc.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building slepc. Check ${PETSC_ARCH}/lib/petsc/conf/slepc.log" && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)'])
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.addMakeRule('slepc-build','')
      # the build must be done at install time because PETSc shared libraries must be in final location before building slepc
      self.addMakeRule('slepc-install','slepcbuild slepcinstall')
    else:
      self.addMakeRule('slepc-build','slepcbuild slepcinstall')
      self.addMakeRule('slepc-install','')

    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.logPrintBox('Slepc examples are available at '+os.path.join('${PETSC_DIR}',self.arch,'externalpackages','git.slepc')+'\nexport SLEPC_DIR='+prefix)
    else:
      self.logPrintBox('Slepc examples are available at '+os.path.join('${PETSC_DIR}',self.arch,'externalpackages','git.slepc')+'\nexport SLEPC_DIR='+os.path.join('${PETSC_DIR}',self.arch))

    return self.installDir

  def alternateConfigureLibrary(self):
    self.addMakeRule('slepc-build','')
    self.addMakeRule('slepc-install','')

