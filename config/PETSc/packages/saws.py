import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
#    self.gitcommit = '3c24ac8df967279c3ceafa5c39fc230af30c63c4'
    self.giturls   = ['https://bitbucket.org/saws/saws.git']
    self.download  = ['https://bitbucket.org/saws/saws/get/master.tar.gz']
    self.functions = ['SAWs_Register']
    self.includes  = ['SAWs.h']
    self.liblist   = [['libSAWs.a']]
    self.libdir           = 'lib' # location of libraries in the package directory tree
    self.includedir       = 'include' # location of includes in the package directory tree    return
    self.needsMath        = 1;

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
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
    g.close()
    self.setCompilers.popLanguage()

    if self.installNeeded('makeinc'):
      try:
        self.logPrintBox('Compiling SAWs; this may take several minutes')
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' &&  make all && cp lib/* '+os.path.join(self.installDir,'lib')+' && cp -r java/gov '+os.path.join(self.installDir,'java')+' &&  cp -f include/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SAWs: '+str(e))
      self.postInstall(output+err,'makeinc')
    return self.installDir

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)

