import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self,framework)
    self.download = ['http://www.cise.ufl.edu/research/sparse/SuiteSparse/SuiteSparse-4.2.1.tar.gz',
                     'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/SuiteSparse-4.2.1.tar.gz']
    self.liblist  = [['libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a'],
                     ['libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a','librt.a'],
                     ['libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a'],
                     ['libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a','librt.a']]
    self.functions = ['umfpack_dl_wsolve','cholmod_l_solve','klu_l_solve']
    self.includes  = ['umfpack.h','cholmod.h','klu.h']
    self.needsMath = 1
    self.complex   = 1
    self.requires32bitint = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def Install(self):
    import os
    self.framework.log.write('SuiteSparseDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    if not self.make.haveGNUMake:
      raise RuntimeError('SuiteSparse buildtools require GNUMake. Use --with-make=gmake or --download-make')

    mkfile = 'SuiteSparse_config/SuiteSparse_config.mk'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    if self.checkCompile('#ifdef PETSC_HAVE_LIMITS_H\n  #include <limits.h>\n#endif\n', 'long long i=LONG_MAX;\n\nif (i);\n'):
      long_max = 'LONG_MAX'
    else:
      long_max = '9223372036854775807LL'
    g.write('CF       = '+self.setCompilers.getCompilerFlags()+''' -DSuiteSparse_long="long long" -DSuiteSparse_long_max=''' + long_max + ''' -DSuiteSparse_long_id='"%lld"'\n''')
    self.setCompilers.popLanguage()
    g.write('MAKE         ='+self.make.make+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('ARCHIVE      = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RM           = '+self.programs.RM+'\n')
    g.write('MV           = '+self.programs.mv+'\n')
    g.write('CP           = '+self.programs.cp+'\n')
    g.write('CLEAN             = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d\n')
    g.write('INSTALL_LIB       = ' + self.libDir + '\n')
    g.write('INSTALL_INCLUDE   = ' + self.includeDir + '\n')
    if self.blasLapack.mangling == 'underscore':
      flg = ''
    elif self.blasLapack.mangling == 'caps':
      flg = '-DBLAS_CAPS_DOES_NOT_WORK'
    else:
      flg = '-DBLAS_NO_UNDERSCORE'
    g.write('UMFPACK_CONFIG    = '+flg+'\n')
    g.write('CHOLMOD_CONFIG    = '+flg+' -DNPARTITION\n')
    g.close()

    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling SuiteSparse; this may take several minutes')
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/SuiteSparse_config && '+self.make.make+' && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/AMD && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/COLAMD && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/BTF && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/CAMD && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/CCOLAMD && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/CHOLMOD && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/UMFPACK && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/KLU && '+self.make.make+' library && '+self.make.make+' install && '+self.make.make+' clean', timeout=2500, log=self.framework.log)

        self.addDefine('HAVE_SUITESPARSE',1)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SuiteSparse: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir

