import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download  = ['http://www.zlib.net/zlib-1.2.11.tar.gz']
    self.functions = ['compress', 'uncompress']
    self.includes  = ['zlib.h']
    self.liblist   = [['libz.a'],['zlib.lib']]
    return

  def Install(self):
    import os

    args = []
    self.framework.pushLanguage('C')
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.removeWarningFlags(self.framework.getCompilerFlags())+'"')
    args.append('prefix="'+self.installDir+'"')
    self.framework.popLanguage()
    args=' '.join(args)

    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()

    if not self.installNeeded(conffile): return self.installDir
    self.log.write('zlibDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    self.logPrintBox('Building and installing zlib, this may take many minutes')
    self.installDirProvider.printSudoPasswordMessage()
    try:
      output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ' + args + ' ./configure && '+self.make.make_jnp+' && '+self.installSudo+' ' +self.make.make+' install', timeout=600, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error building/install zlib files from '+os.path.join(self.packageDir, 'zlib')+' to '+self.packageDir)
    self.postInstall(output+err,conffile)
    return self.installDir
