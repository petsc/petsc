from __future__ import generators
import user
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/UMFPACK-5.5.1.tar.gz']
    self.liblist   = [['libumfpack.a','libamd.a'],
                      ['libumfpack.a','libcholmod.a','libcamd.a','libccolamd.a','libcolamd.a','libamd.a'],
                      ['libumfpack.a','libcholmod.a','libcamd.a','libccolamd.a','libcolamd.a','libamd.a','libsuitesparseconfig.a'],
                      ['libumfpack.a','libcholmod.a','libcamd.a','libccolamd.a','libcolamd.a','libamd.a','libsuitesparseconfig.a','libmetis.a']]
    self.functions = ['umfpack_di_report_info','umfpack_dl_symbolic','umfpack_dl_numeric','umfpack_dl_wsolve']
    self.includes  = ['umfpack.h']
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return
          
  def Install(self):
    self.framework.log.write('umfpackDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'UFconfig/UFconfig.mk'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    if self.checkCompile('#ifdef PETSC_HAVE_LIMITS_H\n  #include <limits.h>\n#endif\n', 'long long i=ULONG_MAX;\n\nif (i);\n'):
      ulong_max = 'ULONG_MAX'
    else:
      ulong_max = '9223372036854775807LL'
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+''' -DUF_long="long long" -DUF_long_max=''' + ulong_max + ''' -DUF_long_id='"%lld"' -DNCHOLMOD \n''')
    self.setCompilers.popLanguage()
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('AR           = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RM           = '+self.programs.RM+'\n')
    g.write('MV           = '+self.programs.mv+'\n')
    g.write('CP           = '+self.programs.cp+'\n')
    g.write('BLAS         = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    if self.blasLapack.mangling == 'underscore':
      flg = ''
    elif self.blasLapack.mangling == 'caps':
      flg = '-DBLAS_CAPS_DOES_NOT_WORK'
    else:
      flg = '-DBLAS_NO_UNDERSCORE'
    g.write('UMFPACK_CONFIG    = '+flg+'\n')
    g.write('CLEAN             = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d\n')
    g.write('INSTALL_LIB       = ' + self.libDir + '\n')
    g.write('INSTALL_INCLUDE   = ' + self.includeDir + '\n')
    g.close()
    
    # Build UMFPACK
    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling UMFPACK; this may take several minutes')
        output,err,ret = config.base.Configure.executeShellCommand('cd '+self.packageDir+'/UMFPACK && make && make install && make clean && cd ../AMD && make install && cd .. && cp UFconfig/*.h '+self.includeDir, timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on UMFPACK: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir
