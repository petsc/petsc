import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit        = 'v1.3-p2'
    self.download         = ['git://https://bitbucket.org/petsc/pkg-triangle','https://bitbucket.org/petsc/pkg-triangle/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['petsc-pkg-triangle','Triangle']
    self.functions        = ['triangulate']
    self.includes         = ['triangle.h']
    self.liblist          = [['libtriangle.a']]
    self.precisions       = ['double']
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.x               = framework.require('config.packages.X', self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mathlib]
    return

  def Install(self):
    import os, sys
    import config.base

    libDir         = os.path.join(self.installDir, 'lib')
    includeDir     = os.path.join(self.installDir, 'include')
    makeinc        = os.path.join(self.packageDir, 'make.inc')
    installmakeinc = os.path.join(self.installDir, 'make.inc')

    g = open(makeinc,'w')
    g.write('SHELL            = '+self.programs.SHELL+'\n')
    g.write('CP               = '+self.programs.cp+'\n')
    g.write('RM               = '+self.programs.RM+'\n')
    g.write('MKDIR            = '+self.programs.mkdir+'\n')
    g.write('OMAKE            = '+self.make.make+' '+self.make.noprintdirflag+'\n')

    g.write('CLINKER          = '+self.getLinker()+'\n')
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS          = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX    = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB           = '+self.setCompilers.RANLIB+'\n')
    g.write('SL_LINKER_SUFFIX = '+self.setCompilers.sharedLibraryExt+'\n')

    g.write('TRIANGLE_ROOT    = '+self.packageDir+'\n')
    g.write('PREFIX           = '+self.installDir+'\n')
    g.write('LIBDIR           = '+libDir+'\n')
    g.write('INSTALL_LIB_DIR  = '+libDir+'\n')
    g.write('TRIANGLELIB      = libtriangle.$(AR_LIB_SUFFIX)\n')
    g.write('SHLIB            = libtriangle\n')

    self.pushLanguage('C')
    cflags = self.updatePackageCFlags(self.getCompilerFlags())
    cflags += ' '+self.headers.toString('.')
    cflags += ' -fPIC'

    g.write('CC             = '+self.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags+'\n')
    self.popLanguage()

    if self.checkSharedLibrariesEnabled():
      import config.setCompilers

      g.write('BUILDSHAREDLIB = yes\n')
      if config.setCompilers.Configure.isSolaris(self.log) and config.setCompilers.Configure.isGNU(self.getCompiler(), self.log):
        g.write('shared_arch: shared_'+sys.platform+'gnu\n')
      else:
        g.write('shared_arch: shared_'+sys.platform+'\n')
        g.write('''
triangle_shared:
	-@if [ "${BUILDSHAREDLIB}" = "no" ]; then \\
	    echo "Shared libraries disabled"; \\
	  else \
	    echo "making shared libraries in ${INSTALL_LIB_DIR}"; \\
	    ${RM} -rf ${INSTALL_LIB_DIR}/tmp-triangle-shlib; \\
	    mkdir ${INSTALL_LIB_DIR}/tmp-triangle-shlib; \\
            cwd=`pwd`; \\
	    for LIBNAME in ${SHLIB}; \\
	    do \\
	      if test -f ${INSTALL_LIB_DIR}/$$LIBNAME.${AR_LIB_SUFFIX} -o -f ${INSTALL_LIB_DIR}/lt_$$LIBNAME.${AR_LIB_SUFFIX}; then \\
	        if test -f ${INSTALL_LIB_DIR}/$$LIBNAME.${SL_LINKER_SUFFIX}; then \\
	          flag=`find ${INSTALL_LIB_DIR} -type f -name $$LIBNAME.${AR_LIB_SUFFIX} -newer ${INSTALL_LIB_DIR}/$$LIBNAME.${SL_LINKER_SUFFIX} -print`; \\
	          if [ "$$flag" = "" ]; then \\
	            flag=`find ${INSTALL_LIB_DIR} -type f -name lt_$$LIBNAME.${AR_LIB_SUFFIX} -newer ${INSTALL_LIB_DIR}/$$LIBNAME.${SL_LINKER_SUFFIX} -print`; \\
	          fi; \\
	        else \\
	          flag="build"; \\
	        fi; \\
	        if [ "$$flag" != "" ]; then \\
                echo "building $$LIBNAME.${SL_LINKER_SUFFIX}"; \\
                  ${RM} -f ${INSTALL_LIB_DIR}/tmp-triangle-shlib/*; \\
	          cd  ${INSTALL_LIB_DIR}/tmp-triangle-shlib; \\
	          ${AR} x ${INSTALL_LIB_DIR}/$$LIBNAME.${AR_LIB_SUFFIX}; \\
	          ${RANLIB} ${INSTALL_LIB_DIR}/$$LIBNAME.${AR_LIB_SUFFIX}; \\
                  cd $$cwd;\\
	          ${OMAKE} LIBNAME=$$LIBNAME SHARED_LIBRARY_TMPDIR=${INSTALL_LIB_DIR}/tmp-triangle-shlib shared_arch; \\
	        fi; \\
	      fi; \\
	    done; \\
	    ${RM} -rf ${INSTALL_LIB_DIR}/tmp-triangle-shlib; \\
	  fi\n''')
    else:
      g.write('BUILDSHAREDLIB = no\n')
      g.write('shared_arch:\n')
      g.write('shared:\n')
    g.close()

    # Now compile & install
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling Triangle; this may take several minutes')
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && make libtriangle.'+self.setCompilers.AR_LIB_SUFFIX+' && make clean', timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on Triangle: '+str(e))
      self.logPrintBox('Installing Triangle; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage()
      output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
      output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand(self.installSudo+'cp -f '+os.path.join(self.packageDir,'libtriangle.'+self.setCompilers.AR_LIB_SUFFIX)+' '+os.path.join(self.installDir,'lib'), timeout=60, log = self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand(self.installSudo+'cp -f '+os.path.join(self.packageDir, 'src', 'triangle.h')+' '+includeDir, timeout=60, log = self.log)
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    return
