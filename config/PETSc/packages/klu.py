import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download = ['http://www.cise.ufl.edu/research/sparse/SuiteSparse/SuiteSparse-4.2.1.tar.gz']
    self.downloadfilename = 'SuiteSparse'
    self.liblist   = [['libklu.a','libamd.a','libsuitesparseconfig.a','libbtf.a','libcolamd.a']]
    self.functions = ['klu_l_defaults','klu_l_analyze','klu_l_solve','klu_l_factor']
    self.includes  = ['klu.h']
    self.complex   = 1
    self.requires32bitint = 0
    return

  def Install(self):
    import os
    self.framework.log.write('kluDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'SuiteSparse_config/SuiteSparse_config.mk'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    if self.checkCompile('#ifdef PETSC_HAVE_LIMITS_H\n  #include <limits.h>\n#endif\n', 'long long i=ULONG_MAX;\n\nif (i);\n'):
      ulong_max = 'ULONG_MAX'
    else:
      ulong_max = '9223372036854775807LL'
    g.write('CF       = '+self.setCompilers.getCompilerFlags()+''' -DSuiteSparse_long="long long" -DSuiteSparse_long_max=''' + ulong_max + ''' -DSuiteSparse_long_id='"%lld"'\n''')
    self.setCompilers.popLanguage()
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('ARCHIVE      = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RM           = '+self.programs.RM+'\n')
    g.write('MV           = '+self.programs.mv+'\n')
    g.write('CP           = '+self.programs.cp+'\n')
    g.write('CLEAN             = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d\n')
    g.write('INSTALL_LIB       = ' + self.libDir + '\n')
    g.write('INSTALL_INCLUDE   = ' + self.includeDir + '\n')
    g.close()

    # Build KLU
    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling KLU; this may take several minutes')
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/AMD && make lib && make install ', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/COLAMD && make lib && make install', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/BTF && make library && make install',timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/KLU && make library && make install && make clean', timeout=2500, log=self.framework.log)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cp '+self.packageDir+'/SuiteSparse_config/*.h '+self.includeDir, timeout=2500, log=self.framework.log)
        self.addDefine('HAVE_KLU',1)
      except RuntimeError, e:
        raise RuntimeError('Error running make on KLU: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir
