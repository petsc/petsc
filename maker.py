import script

class Make(script.Script):
  '''Template for individual project makefiles. All project makes start with a local RDict.'''
  def __init__(self, builder = None):
    import RDict
    import sys

    script.Script.__init__(self, sys.argv[1:], RDict.RDict())
    if builder is None:
      self.builder = __import__('builder').Builder()
    else:
      self.builder = builder
    self.builder.pushLanguage('C')
    return

  def setupHelp(self, help):
    import nargs

    help = script.Script.setupHelp(self, help)
    help.addArgument('Make', 'forceConfigure', nargs.ArgBool(None, 0, 'Force a reconfiguration', isTemporary = 1))
    return help

  def setupDependencies(self, sourceDB):
    '''Override this method to setup dependencies between source files'''
    return

  def setup(self):
    script.Script.setup(self)
    self.builder.setup()
    self.setupDependencies(self.builder.shouldCompile.sourceDB)
    return

  def shouldConfigure(self, builder, framework):
    '''Determine whether we should reconfigure
       - If the configure header or substitution files are missing
       - If -forceConfigure is given
       - If configure.py has changed
       - If the database does not contain a cached configure'''
    import os

    if not os.path.isfile(framework.header) or not reduce(lambda x,y: x and y, [os.path.isfile(pair[1]) for pair in framework.substFiles], True):
      self.logPrint('Reconfiguring due to absence of configure generated files')
      return 1
    if self.argDB['forceConfigure']:
      self.logPrint('Reconfiguring forced')
      return 1
    if (not 'configure.py' in self.builder.shouldCompile.sourceDB or
        not self.builder.shouldCompile.sourceDB['configure.py'][0] == self.builder.shouldCompile.sourceDB.getChecksum('configure.py')):
      self.logPrint('Reconfiguring due to changed configure.py')
      return 1
    if not 'configureCache' in self.argDB:
      self.logPrint('Reconfiguring due to absence of configure cache')
      return 1
    return 0

  def configure(self, builder):
    '''Run configure if necessary and return the configuration Framework'''
    import config.framework
    import cPickle
    import os

    framework        = config.framework.Framework(self.clArgs+['-noOutput'], self.argDB)
    framework.header = os.path.join('include', 'config.h')
    try:
      framework.addChild(self.getModule(self.getRoot(), 'configure').Configure(framework))
    except ImportError, e:
      self.logPrint('Configure module not present: '+str(e))
      return
    doConfigure      = self.shouldConfigure(builder, framework)
    if not doConfigure:
      try:
        framework       = cPickle.loads(self.argDB['configureCache'])
        framework.argDB = self.argDB
      except cPickle.UnpicklingError, e:
        doConfigure     = 1
        self.logPrint('Invalid cached configure: '+str(e))
    if doConfigure:
      self.logPrint('Starting new configuration')
      framework.configure()
      self.builder.shouldCompile.sourceDB.updateSource('configure.py')
      self.argDB['configureCache'] = cPickle.dumps(framework)
    else:
      self.logPrint('Using cached configure')
      framework.cleanup()
    return framework

  def updateDependencies(self, sourceDB):
    '''Override this method to update dependencies between source files. This method saves the database'''
    sourceDB.save()
    return

  def build(self, builder):
    '''Override this method to execute all build operations. This method does nothing.'''
    return

  def run(self):
    self.setup()
    self.logPrint('Starting Build', debugSection = 'build')
    self.configure(self.builder)
    self.build(self.builder)
    self.updateDependencies(self.builder.shouldCompile.sourceDB)
    self.logPrint('Ending Build', debugSection = 'build')
    return 1
