import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers', self)
    self.found        = 0
    return

  def __str__(self):
    if self.found:
      desc = ['ParMetis:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('ParMetis', '-with-parmetis=<bool>',                nargs.ArgBool(None, 0, 'Activate ParMetis'))
    help.addArgument('ParMetis', '-with-parmetis-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the ParMetis installation'))
    help.addArgument('ParMetis', '-with-parmetis-include=<dir>',         nargs.ArgDir(None, None, 'The directory containing metis.h'))
    help.addArgument('ParMetis', '-with-parmetis-lib=<lib>',             nargs.Arg(None, None, 'The ParMetis library or list of libraries'))
    help.addArgument('ParMetis', '-download-parmetis=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 2, 'Install MPICH to provide ParMetis'))
    return

  def configureLibrary(self):
    '''Find a ParMetis installation and check if it can work with PETSc'''
    return

  def setOutput(self):
    '''Add defines and substitutions
       - HAVE_PARMETIS is defined if a working ParMetis is found
       - PARMETIS_INCLUDE and PARMETIS are command line arguments for the compile and link
       - PARMETIS_INCLUDE_DIR is the directory containing metis.h
       - PARMETIS_LIBRARY is the list of ParMetis libraries'''
    if self.found:
      self.addDefine('HAVE_PARMETIS', 1)
    else:
      self.addSubstitution('PARMETIS_INCLUDE', '')
      self.addSubstitution('PARMETIS_LIB', '')
#    self.framework.packages.append(self)
    return

  def configure(self):
    if not self.framework.argDB['with-parmetis']:
      return
    self.executeTest(self.configureLibrary)
    self.setOutput()
    return
