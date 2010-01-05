import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['Not available for download: use --download-Suggar=Suggar.tar.gz']
    self.functions = ['ctkSortAllDonorsInGrid']
    self.liblist   = [['libsuggar_3d_opt_petsc.a'],['libsuggar_3d_opt_linux.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.expat      = framework.require('PETSc.packages.expat',self)
    self.cproto     = framework.require('PETSc.packages.cproto',self)
    self.p3dlib     = framework.require('PETSc.packages.P3Dlib',self)
    self.deps       = [self.p3dlib,self.expat,self.mpi]
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    g = open(os.path.join(self.packageDir,'src','FLAGS.local'),'w')
    g.write('CC ='+self.framework.getCompiler()+'\n')
    g.write('CFLAGS ='+self.framework.getCompilerFlags()+'\n')
    g.write('CPROTO = '+self.cproto.cproto+' -D__THROW= -D_STDARG_H ${TRACEMEM} -I..\n')
    g.write('P3DLIB_DIR = '+self.libraries.toString(self.p3dlib.lib)+'\n')
    g.write('P3DINC_DIR = '+self.headers.toString(self.p3dlib.include)+'\n')
    g.write('EXPATLIB_DIR = '+self.libraries.toString(self.expat.lib)+'\n')
    g.write('EXPATINC_DIR = '+self.headers.toString(self.expat.include)+'\n')
    g.write('MACHINE = '+'petsc\n')
    # from FLAGS.machine
    g.write('LD ='+self.framework.getCompiler()+'\n')    
    g.write('CFLAGS_G   ='+self.framework.getCompilerFlags()+'\n')
    g.write('CFLAGS_O   ='+self.framework.getCompilerFlags()+'\n')
    g.write('CFLAGS_EXP = -I./Proto ${P3DINC_DIR} ${EXPATINC_DIR} -I/usr/include/malloc\n')
    if self.compilers.fortranManglingDoubleUnderscore:
      cdefs = '-DF77_APPEND__'
    elif self.compilers.fortranMangling == 'underscore':
      cdefs = '-DF77_APPEND_'
    else:
      cdefs = ''
    import config.setCompilers
    if config.setCompilers.Configure.isDarwin():
      ddefs = ' -Dmacosx '
#    elif config.setCompilers.isIBM():
#      ddefs = ' -Dibm '
    else:  # need to test for Linux
      ddefs = ' -Dlinux '
    g.write('DEFINES   =  -DDUMP_FLEX '+ddefs+cdefs+'\n')
    g.write('DEFINES2D =  -DTWOD   $(DEFINES) \n')
    g.write('DEFINES3D =  -DTHREED $(DEFINES) \n')
    g.close()
    # this is a dummy file because Suggar expects it, all variables are set in FLAGS.local
    g = open(os.path.join(self.packageDir,'src','FLAGS.petsc'),'w')
    g.close()
    self.framework.popLanguage()

    if self.installNeeded(os.path.join('src','FLAGS.local')):
      try:
        self.logPrintBox('Compiling SUGGAR; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/src; rm -f ../petsc/*/*.o ;make makedirs libsuggar_3d_opt', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUGGAR: '+str(e))
      output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('mv -f '+os.path.join(self.packageDir,'bin','libsuggar_3d_opt_petsc.a')+' '+os.path.join(self.installDir,'lib'), timeout=5, log = self.framework.log)
                          
      self.postInstall(output1+err1+output2+err2,os.path.join('src','FLAGS.local'))
    return self.installDir
