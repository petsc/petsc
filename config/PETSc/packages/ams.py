import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://petsc.cs.iit.edu/petsc/petsc-dev/archive/tip.tar.gz']
#                      ssh://petsc@petsc.cs.iit.edu//hg/petsc/ams-dev 
#    Does not currently support automatic download and install
    self.functions = ['AMS_Memory_create']
    self.includes  = ['ams.h']
    self.liblist   = [['libamspub.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps       = [self.mpi]  
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'makeinc'),'w')
    g.write('AR           = '+self.setCompilers.AR+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('RM           = rm -f\n')
    g.write('CP           = cp -f\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CLINKER      = ${CC}\n')
    if self.setCompilers.isDarwin():    
      g.write('LINKSHARED   = ${CC} -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup\n')
    else:
      g.write('LINKSHARED   = ${CC} -dynamiclib\n')      
    self.setCompilers.popLanguage()

    if self.installNeeded('makeinc'):
      try:
        self.logPrintBox('Compiling ams; this may take several minutes')
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make all && cp lib/* '+os.path.join(self.installDir,'lib')+' &&  cp -f include/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on ams: '+str(e))
      self.postInstall(output+err,'makeinc')
    return self.installDir

  
  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.addDefine('AMS_DIR', '"'+os.path.dirname(self.include[0])+'"')
    self.addMakeMacro('AMS_DIR', '"'+os.path.dirname(self.include[0])+'"')    
