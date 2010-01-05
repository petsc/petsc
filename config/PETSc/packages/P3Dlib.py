import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['Not available for download: use --download-P3Dlib=P3Dlib.tar.gz']
    self.functions = ['p3d_ReadStructGridFileHeader']
    self.liblist   = [['libp3d.a']]
    self.includes  = ['p3dlib.h']
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    g = open(os.path.join(self.packageDir,'src','makefile.inc'),'w')
    g.write('CC='+self.framework.getCompiler()+'\n')
    g.write('CFLAGS='+self.framework.getCompilerFlags()+'\n')
    g.close()
    self.framework.popLanguage()

    if self.installNeeded(os.path.join('src','makefile.inc')):
      try:
        self.logPrintBox('Compiling P3DLIB; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/src; make libp3d.a', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on P3DLIB: '+str(e))
      output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('mv -f '+os.path.join(self.packageDir,'src','libp3d.a')+' '+os.path.join(self.installDir,'lib'), timeout=5, log = self.framework.log)
      output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand('cp -f '+os.path.join(self.packageDir,'src','*.h')+' '+os.path.join(self.installDir,'include'), timeout=5, log = self.framework.log)      
                          
      self.postInstall(output1+err1+output2+err2+output3+err3,os.path.join('src','makefile.inc'))
    return self.installDir
