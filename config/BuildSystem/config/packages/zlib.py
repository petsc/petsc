import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://www.zlib.net/zlib-1.2.11.tar.gz',
                         'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/zlib-1.2.11.tar.gz']
    self.functions    = ['compress', 'uncompress']
    self.includes     = ['zlib.h']
    self.liblist      = [['libz.a'],['zlib.lib']]
    self.useddirectly = 0
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('ZLIB', '-download-zlib-static=<bool>',                 nargs.ArgBool(None, 0, 'Build libz as a static library'))

  def Install(self):
    import os

    args = []
    self.framework.pushLanguage('C')
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.removeWarningFlags(self.framework.getCompilerFlags())+'"')
    args.append('prefix="'+self.installDir+'"')
    self.framework.popLanguage()
    args=' '.join(args)

    cargs=[]
    if (not self.checkSharedLibrariesEnabled()) or self.argDB['download-zlib-static']:
      cargs.append('--static')
    cargs=' '.join(cargs)

    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write('args: '+args+'\n')
    fd.write('cargs: '+cargs+'\n')
    fd.close()

    if not self.installNeeded(conffile): return self.installDir
    self.log.write('zlibDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    self.logPrintBox('Building and installing zlib; this may take several minutes')
    self.installDirProvider.printSudoPasswordMessage()
    try:
      output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ' + args + ' ./configure '+cargs+' && '+self.make.make_jnp+' && '+self.installSudo+' ' +self.make.make+' install', timeout=600, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error building/install zlib files from '+os.path.join(self.packageDir, 'zlib')+' to '+self.packageDir)
    self.postInstall(output+err,conffile)
    return self.installDir
