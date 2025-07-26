import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit              = 'f38dd83e30136b4e25eb2343813ee4fbd7c16681'
    #self.gitcommit              = '5.2.0'
    self.download               = ['git://https://github.com/flame/libflame.git','https://github.com/flame/libflame/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['libflame']
    self.functions              = ['FLA_Cntl_gemv_obj_create']
    self.includes               = ['FLAME.h']
    self.liblist                = [['libflame.a','libblis.a'],['libflame.a','libblis-mt.a']]
    self.precisions             = ['single','double']
    self.buildLanguages         = ['C','FC']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.blis          = framework.require('config.packages.BLIS',self)
    self.deps          = [self.blis]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-lapack2flame')
    args.append('--disable-warnings')
    args.append('--enable-max-arg-list-hack')
    if self.argDB['with-shared-libraries']:
      args.append('--enable-dynamic-build')
    return [arg for arg in args if not arg in ['--enable-shared']]

  def Install(self):
    import os
    args = self.formGNUConfigureArgs()
    if self.download and self.argDB['download-'+self.downloadname.lower()+'-configure-arguments']:
       args.append(self.argDB['download-'+self.downloadname.lower()+'-configure-arguments'])
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()
    ### Use conffile to check whether a reconfigure/rebuild is required
    if not self.installNeeded(conffile):
      return self.installDir
    self.preInstall()

    ### Configure and Build package
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand(os.path.join('.',self.configureName)+' '+args, cwd=self.packageDir, timeout=2000, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running configure on ' + self.PACKAGE+': '+str(e))
      try:
        with open(os.path.join(self.packageDir,'config.log')) as fd:
          conf = fd.read()
          fd.close()
          self.logPrint('Output in config.log for ' + self.PACKAGE+': '+conf)
      except:
        pass
      raise RuntimeError('Error running configure on ' + self.PACKAGE)

    # Patch main Makefile so that FFLAGS is taken into account
    fname = os.path.join(self.packageDir,'Makefile')
    oldcode = r'''$(FC) -cpp -c $$< -o $$@'''
    newcode = r'''$(FC) $(FFLAGS) -cpp -c $$< -o $$@'''
    with open(fname,'r') as file:
      sourcecode = file.read()
    sourcecode = sourcecode.replace(oldcode,newcode)
    with open(fname,'w') as file:
      file.write(sourcecode)

    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      if self.parallelMake: pmake = self.make.make_jnp+' '+self.makerulename+' '
      else: pmake = self.make.make+' '+self.makerulename+' '

      output2,err2,ret2  = config.base.Configure.executeShellCommand(self.make.make+' clean', cwd=self.packageDir, timeout=200, log = self.log)
      output3,err3,ret3  = config.base.Configure.executeShellCommand(pmake+' V=1 all', cwd=self.packageDir, timeout=6000, log = self.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')
      output4,err4,ret4  = config.base.Configure.executeShellCommand(self.make.make+' install', cwd=self.packageDir, timeout=1000, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running make; make install on '+self.PACKAGE+': '+str(e))
      raise RuntimeError('Error running make; make install on '+self.PACKAGE)
    self.postInstall(output1+err1+output2+err2+output3+err3+output4+err4, conffile)
    return self.installDir
