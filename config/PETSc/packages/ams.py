import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.gitcommit = '3c24ac8df967279c3ceafa5c39fc230af30c63c4'
    self.giturls   = ['https://bitbucket.org/petsc/ams.git']
    self.download  = ['https://bitbucket.org/petsc/ams/get/master.tar.gz',
                      'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ams-dev.tar.gz']
    self.functions = ['AMS_Memory_create']
    self.includes  = ['ams.h']
    self.liblist   = [['libamspub.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.java       = framework.require('PETSc.packages.java',self)
    self.deps       = [self.mpi]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'makeinc'),'w')
    g.write('AR           = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    # should use the BuildSystem defined RM, MV
    g.write('RM           = rm -f\n')
    g.write('MV           = mv -f\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+' '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('CLINKER      = ${CC}\n')
    if self.setCompilers.isDarwin():
      g.write('LINKSHARED   = ${CC} -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup\n')
    else:
      g.write('LINKSHARED   = ${CC} -dynamiclib\n')
    if hasattr(self.java,'javac'):
      if self.setCompilers.isDarwin():
        g.write('JAVA_INCLUDES   =  -I/System/Library/Frameworks/JavaVM.framework/Headers/../../CurrentJDK/Headers\n')
      else:
        g.write('JAVA_INCLUDES   =  \n')
      g.write('JAVAC           = '+getattr(self.java, 'javac'))
    g.close()
    self.setCompilers.popLanguage()

    if self.installNeeded('makeinc'):
      try:
        self.logPrintBox('Compiling ams; this may take several minutes')
        if not os.path.isdir(os.path.join(self.installDir,'java')):
          os.mkdir(os.path.join(self.installDir,'java'))
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' &&  make all && cp lib/* '+os.path.join(self.installDir,'lib')+' && cp -r java/gov '+os.path.join(self.installDir,'java')+' &&  cp -f include/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on ams: '+str(e))
      self.postInstall(output+err,'makeinc')
    return self.installDir


  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.addDefine('AMS_DIR', '"'+os.path.dirname(self.include[0])+'"')
    self.addMakeMacro('AMS_DIR', '"'+os.path.dirname(self.include[0])+'"')
