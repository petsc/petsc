import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/PARTY_1.99.tar.gz']
    self.functions    = ['party_lib']
    self.includes     = ['party_lib.h']
    self.liblist      = [['libparty.a']]
    self.license      = 'http://wwwcs.upb.de/fachbereich/AG/monien/RESEARCH/PART/party.html'
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+' '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.close()
    
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling party; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+'; PARTY_INSTALL_DIR='+self.installDir+';export PARTY_INSTALL_DIR; make clean; make all; cd ..; mv -f *.a '+os.path.join(self.installDir,self.libdir,'/,')+'; cp -f party_lib.h '+os.path.join(self.installDir,self.includedir,'.'), timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on PARTY: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir
