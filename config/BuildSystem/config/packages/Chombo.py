import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://www.mcs.anl.gov/~sarich/Chombo_3.2.tar.gz']
    self.functionsCxx = [1,'namespace Box {class Box{public: Box();};}','Box::Box *nb = new Box::Box()'] 
    self.includedir = 'include'
    self.includes     = ['CH_config.H']

    self.downloadonWindows= 0
    return


  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.hdf5 = self.framework.require('config.packages.hdf5',self)
    self.mpi = self.framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi,self.blasLapack,self.hdf5]
    return

  def Install(self):
    import os
    self.getExecutable('csh',path='/bin')
    if not hasattr(self, 'csh'):
      raise RuntimeError('Cannot build Chombo. It requires /bin/csh. Please install csh and retry.\n')

    self.framework.pushLanguage('Cxx')

    g = open(os.path.join(self.packageDir,'lib','mk','Make.defs.local'),'w')
    g.write('\n#begin\n')
    g.write('#DIM='+'\n')
    g.write('#DEBUG='+'\n')
    g.write('#OPT='+'\n')
    g.write('#PRECISION='+'\n')
    g.write('#PROFILE='+'\n')
    g.write('CXX='+self.framework.getCompiler()+'\n')
    g.write('MPICXX='+self.framework.getCompiler()+'\n')
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



    g.close()
    if True: #self.installNeeded(os.path.join('lib','mk','Make.defs.local')):
      try:
        self.logPrintBox('Compiling and installing chombo; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.framework.log)
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.framework.log)

        #run make -p to get library (config) namen
        poutput,perr,pret = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'lib') +' && make vars | egrep ^config', timeout=2500, log = self.framework.log)
        config_value=None
        ind = poutput.find('config=')
        if ind != 0:
          raise RuntimeError('Error running make on Chombo: config value not found')
        config_value=poutput.split('=')[1]
        self.logPrint('Chombo installed using config=%s\n'%config_value)
        output,err,ret = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'lib') +' && make clean && make all', timeout=2500, log = self.framework.log)
        output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+self.installSudo+'&& cp -f lib/lib*.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' &&  '+self.installSudo+'cp -f lib/include/*.H '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Chombo: '+str(e))


      self.libdir = 'lib'
      self.liblist = [['libbasetools%s.a' % config_value,'libamrelliptic%s' % config_value,'libamrtimedependent%s.a' % config_value,'libamrtools%s.a' % config_value,'libboxtools%s.a' % config_value]]
      self.postInstall(output+err,os.path.join('lib','mk','Make.defs.local'))
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      pass
    return
