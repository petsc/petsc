import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    return

  def __str__(self):
    return ''

  def setupHelp(self,help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('GMSH', '-with-gmsh-exec=<executable>', nargs.Arg(None, None, 'Gmsh executable to use'))
    return

  def configure(self):
    '''determine gmsh binary to use'''
    if ('with-gmsh-exec' not in self.argDB) and not self.argDB['with-gmsh']:
      return
    if 'with-gmsh-exec' in self.argDB:
      gmsh = self.argDB['with-gmsh-exec']
      self.log.write('Looking for specified Gmsh executable '+gmsh+'\n')
    else:
      gmsh = 'gmsh'
      self.log.write('Looking for default Gmsh executable\n')
    if self.getExecutable(gmsh, getFullPath=1, resultName='gmsh', setMakeMacro = 0):
      try:
        out,err,ret  = config.package.Package.executeShellCommand(self.gmsh + ' -info', timeout=60, log = self.log)
        self.addDefine('GMSH_EXE','"'+self.gmsh+'"')
      except RuntimeError as e:
        self.log.write('Unable to run Gmsh executable '+self.gmsh+'\n'+str(e)+'\n')
    return
