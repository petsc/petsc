import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['http://software.lanl.gov/ascem/tpls/ascem-io-2.2.tar.gz']
    self.downloaddirnames = ['ascem-io']
    self.includes         = ['ascemio_util.h']
    self.liblist          = [['libparallelio.a']]
    self.functions        = []
    self.requirescxx11    = 0
    self.downloadonWindows= 0
    self.hastests         = 1
    self.useddirectly     = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.hdf5            = framework.require('config.packages.hdf5',self)
    self.deps            = [self.mpi,self.hdf5]
    return

  def Install(self):
    import os
    self.setCompilers.pushLanguage('C')
    if self.hdf5.include:
      MAKEARGS = 'MACHINE="" CC="'+self.setCompilers.getCompiler()+' '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'" HDF5_INCLUDE_DIR="'+self.hdf5.include[0]+'"'
    else:
      MAKEARGS = 'MACHINE="" CC="'+self.setCompilers.getCompiler()+' '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'"'
    self.setCompilers.popLanguage()
    INSTALLARGS = 'ASCEMIO_INSTALL_DIR="'+self.installDir+'"'
    g = open(os.path.join(self.packageDir,'compiledata'),'w')
    g.write(MAKEARGS+INSTALLARGS)
    g.close()

    if self.installNeeded('compiledata'):
      try:
        self.logPrintBox('Compiling and installing ascem-io; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && '+self.make.make+' '+MAKEARGS+' && '+self.installSudo+' '+self.make.make+' '+INSTALLARGS+' install', timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on ascem-io: '+str(e))
      self.postInstall(output+err,'compiledata')
    return self.installDir


