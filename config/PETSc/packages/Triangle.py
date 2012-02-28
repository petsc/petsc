import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/Triangle.tar.gz']
    self.functions = ['triangulate']
    self.includes  = ['triangle.h']
    self.liblist   = [['libtriangle.a']]
    self.needsMath = 1
    self.complex   = 1
    self.double    = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.make            = framework.require('config.programs', self)
    self.x               = framework.require('PETSc.packages.X', self)
    self.deps = []
    return

  def Install(self):
    import os, sys
    import config.base

    libDir         = os.path.join(self.installDir, 'lib')
    includeDir     = os.path.join(self.installDir, 'include')
    makeinc        = os.path.join(self.packageDir, 'make.inc')
    installmakeinc = os.path.join(self.installDir, 'make.inc')

    g = open(makeinc,'w')
    ##g.write('include '+os.path.join(self.petscdir.dir, 'conf', 'rules')+'\n')
    g.write('SHELL            = '+self.programs.SHELL+'\n')
    g.write('CP               = '+self.programs.cp+'\n')
    g.write('RM               = '+self.programs.RM+'\n')
    g.write('MKDIR            = '+self.programs.mkdir+'\n')
    g.write('OMAKE            = '+self.make.make+' '+self.make.flags+'\n')

    g.write('CLINKER          = '+self.setCompilers.getLinker()+'\n')
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS          = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX    = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB           = '+self.setCompilers.RANLIB+'\n')
    g.write('SL_LINKER_SUFFIX = '+self.setCompilers.sharedLibraryExt+'\n')

    g.write('TRIANGLE_ROOT    = '+self.packageDir+'\n')
    g.write('PREFIX           = '+self.installDir+'\n')
    g.write('LIBDIR           = '+libDir+'\n')
    g.write('INSTALL_LIB_DIR  = '+libDir+'\n')
    g.write('TRIANGLELIB      = $(LIBDIR)/libtriangle.$(AR_LIB_SUFFIX)\n')
    g.write('SHLIB            = libtriangle\n')

    self.setCompilers.pushLanguage('C')
    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' '+self.headers.toString('.')

    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags+'\n')
    self.setCompilers.popLanguage()

    if self.sharedLibraries.useShared:
      import config.setCompilers

      g.write('BUILDSHAREDLIB = yes\n')
      if config.setCompilers.Configure.isSolaris() and config.setCompilers.Configure.isGNU(self.framework.getCompiler()):
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
        self.logPrintBox('Compiling & installing Triangle; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make '+os.path.join(libDir, 'libtriangle.'+self.setCompilers.AR_LIB_SUFFIX)+' && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Triangle: '+str(e))
      output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cp -f '+os.path.join(self.packageDir, 'src', 'triangle.h')+' '+includeDir, timeout=5, log = self.framework.log)
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    if self.found:
      self.framework.addDefine('ANSI_DECLARATORS', 1)
    return
