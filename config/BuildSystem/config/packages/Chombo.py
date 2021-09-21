import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit        = '2469eee'
    self.download         = ['git://https://bitbucket.org/petsc/pkg-chombo-3.2.git','https://bitbucket.org/petsc/pkg-chombo-3.2/get/'+self.gitcommit+'.tar.gz']
    self.functionsCxx     = [1,'namespace Box {class Box{public: Box();};}','Box::Box *nb = new Box::Box()'] 
    self.includedir       = 'include'
    self.includes         = ['CH_config.H']
    self.downloadonWindows= 0
    self.hastestsdatafiles= 1
    self.downloaddirnames  = ['petsc-pkg-chombo-3.2']
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('CHOMBO', '-download-chombo-dimension=<1,2,3>',    nargs.ArgInt(None, 2, 'Install Chombo to work in this space dimension'))

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.hdf5 = self.framework.require('config.packages.hdf5',self)
    self.mpi = self.framework.require('config.packages.MPI',self)
    self.make = self.framework.require('config.packages.make',self)
    self.deps       = [self.mpi,self.blasLapack,self.hdf5]
    return

  def Install(self):
    import os
    self.getExecutable('csh',path='/bin')
    if not hasattr(self, 'csh'):
      raise RuntimeError('Cannot build Chombo. It requires /bin/csh. Please install csh and retry.\n')
    if not hasattr(self.compilers, 'FC'):
      raise RuntimeError('Cannot install '+self.name+' without Fortran, make sure you do NOT have --with-fc=0')
    if not self.make.haveGNUMake:
      raise RuntimeError('Cannot install '+self.name+' without GNUMake, suggest --download-make')

    dim = self.argDB['download-chombo-dimension']
    g = open(os.path.join(self.packageDir,'lib','mk','Make.defs.local'),'w')
    g.write('\n#begin\n')
    g.write('#DIM='+str(dim)+'\n')
    g.write('#DEBUG='+'\n')
    g.write('#OPT='+'\n')
    g.write('#PRECISION='+'\n')
    g.write('#PROFILE='+'\n')
    self.pushLanguage('Cxx')
    g.write('CXX='+self.getCompiler()+'\n')
    g.write('MPICXX='+self.getCompiler()+'\n')
    self.popLanguage()
    self.pushLanguage('FC')
    g.write('FC='+self.getCompiler()+'\n')
    self.popLanguage()
    g.write('#OBJMODEL='+'\n')
    g.write('#XTRACONFIG='+'\n')
    g.write('#USE_64='+'\n')
    g.write('#USE_COMPLEX='+'\n')
    g.write('#USE_EB='+'\n')
    g.write('#USE_CCSE='+'\n')
    g.write('USE_HDF=TRUE\n')
    g.write('HDFINCFLAGS='+self.headers.toString(self.hdf5.include)+'\n')
    g.write('HDFLIBFLAGS='+self.libraries.toString(self.hdf5.lib)+'\n')
    g.write('HDFMPIINCFLAGS='+self.headers.toString(self.hdf5.include)+'\n')
    g.write('HDFMPILIBFLAGS='+self.libraries.toString(self.hdf5.lib)+'\n')
    g.write('#USE_MF='+'\n')
    g.write('#USE_MT='+'\n')
    g.write('#USE_SETVAL='+'\n')
    g.write('#CH_AR='+self.setCompilers.AR+'\n')
    g.write('#CH_CPP='+'\n')
    g.write('#DOXYGEN='+'\n')
    g.write('#LD='+'\n')
    g.write('#PERL='+'\n')
    g.write('RANLIB='+self.setCompilers.RANLIB+'\n')
    g.write('#cppdbgflags='+'\n')
    g.write('#cppoptflags='+'\n')
    g.write('#cxxcppflags='+'\n')
    g.write('#cxxdbgflags='+'\n')
    g.write('#cxxoptflags='+'\n')
    g.write('#cxxprofflags='+'\n')
    g.write('#fcppflags='+'\n')
    g.write('#fdbgflags='+'\n')
    g.write('#foptflags='+'\n')
    g.write('#fprofflags='+'\n')
    g.write('#flibflags='+'\n')
    g.write('#lddbgflags='+'\n')
    g.write('#ldoptflags='+'\n')
    g.write('#ldprofflags='+'\n')
    g.write('syslibflags='+self.libraries.toString(self.blasLapack.lib)+'\n')
    g.write('\n#end\n')

    # write these into petscconf.h so user code that includes PETSc doesn't need to manually set them
    # these must be set before Chombo C++ include files are included
    self.framework.addDefine('CH_LANG_CC',1)
    self.framework.addDefine('CH_SPACEDIM',dim)

    g.close()
    if True: #self.installNeeded(os.path.join('lib','mk','Make.defs.local')):
      try:
        self.logPrintBox('Compiling and installing chombo; this may take several minutes')
        output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)

        #run make -p to get library (config) namen
        poutput,perr,pret = config.package.Package.executeShellCommand('make vars', cwd=os.path.join(self.packageDir,'lib'), timeout=2500, log = self.log)
        config_value=None
        for line in poutput.splitlines():
          if line.startswith('config='):
            config_value = line.split('=')[1]
            break
        if config_value is None:
          raise RuntimeError('Error running make on Chombo: config value not found')
        self.logPrint('Chombo installed using config=%s\n'%config_value)
        import glob
        output,err,ret = config.package.Package.executeShellCommandSeq(
          ['make clean',
           'make lib',
           'cp -f lib*.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,''),
           'cp -f include/*.H '+os.path.join(self.installDir,self.includedir,'')
          ], cwd=os.path.join(self.packageDir,'lib'), timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on Chombo: '+str(e))


      self.libdir = 'lib'
      self.liblist = [['libbasetools%s.a' % config_value,'libamrelliptic%s.a' % config_value,'libamrtimedependent%s.a' % config_value,'libamrtools%s.a' % config_value,'libboxtools%s.a' % config_value]]
      self.postInstall(output+err,os.path.join('lib','mk','Make.defs.local'))
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      pass
    return
