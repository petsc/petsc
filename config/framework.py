'''
  The Framework object serves as the central control for a configure run. It
maintains a graph of all the configure modules involved, which is also used to
track dependencies between them. It initiates the run, compiles the results, and
handles the final output. It maintains the help list for all options available
in the run.

  The setup() method preforms generic Script setup and then is called recursively
on all the child modules. The cleanup() method performs the final output and
logging actions
    - Subtitute files
    - Output configure header
    - Log filesystem actions

  Children may be added to the Framework using addChild() or getChild(), but the
far more frequent method is to use require(). Here a module is requested, as in
getChild(), but it is also required to run before another module, usually the one
executing the require(). This provides a simple local interface to establish
dependencies between the child modules, and provides a partial order on the
children to the Framework.

  A backwards compatibility mode is provided for which the user specifies a
configure header and set of files to experience substitution, mirroring the
common usage of Autoconf. Slight improvements have been made in that all
defines are now guarded, various prefixes are allowed for defines and
substitutions, and C specific constructs such as function prototypes and
typedefs are removed to a separate header. However, this is not the intended
future usage.

  The use of configure modules by other modules in the same run provides a model
for the suggested interaction of a new build system with the Framework. If a
module requires another, it merely executes a require(). For instance, the PETSc
configure module for HYPRE requires information about MPI, and thus contains

      self.mpi = self.framework.require("config.packages.MPI", self)

Notice that passing self for the last arguments means that the MPI module will
run before the HYPRE module. Furthermore, we save the resulting object as
self.mpi so that we may interogate it later. HYPRE can initially test whether
MPI was indeed found using self.mpi.foundMPI. When HYPRE requires the list of
MPI libraries in order to link a test object, the module can use self.mpi.lib.
'''
import user
import script
import config.base
import time

import os
import re
# workarround for python2.2 which does not have pathsep
if not hasattr(os.path,'pathsep'): os.path.pathsep=':'

import cPickle

try:
  from hashlib import md5 as new_md5
except ImportError:
  from md5 import new as new_md5


try:
  enumerate([0, 1])
except NameError:
  def enumerate(l):
    return zip(range(len(l)), l)

class Framework(config.base.Configure, script.LanguageProcessor):
  '''This needs to manage configure information in itself just as Builder manages it for configurations'''
  def __init__(self, clArgs = None, argDB = None, loadArgDB = 1):
    import graph
    import nargs

    if argDB is None:
      import RDict

      argDB = RDict.RDict(load = loadArgDB)
    script.LanguageProcessor.__init__(self, clArgs, argDB)
    config.base.Configure.__init__(self, self)
    self.childGraph      = graph.DirectedGraph()
    self.substRE         = re.compile(r'@(?P<name>[^@]+)@')
    self.substFiles      = {}
    self.logName         = 'configure.log'
    self.header          = 'matt_config.h'
    self.makeMacroHeader = ''
    self.makeRuleHeader  = ''
    self.cHeader         = 'matt_fix.h'
    self.headerPrefix    = ''
    self.substPrefix     = ''
    self.warningRE       = re.compile('warning', re.I)
    if not nargs.Arg.findArgument('debugSections', self.clArgs):
      self.argDB['debugSections'] = ['screen']
    # Perhaps these initializations should just be local temporary arguments
    self.argDB['CPPFLAGS']  = ''
    if not 'LDFLAGS' in self.argDB:
      self.argDB['LDFLAGS'] = ''
    self.batchSetup         = []
    self.batchIncludes      = []
    self.batchBodies        = []
    self.batchCleanup       = []
    self.batchIncludeDirs   = []
    self.batchLibs          = []
    self.dependencies       = {}
    self.configureParent    = None
    # List of packages actually found
    self.packages           = []
    self.createChildren()
    return

  def __getstate__(self):
    '''We do not want to pickle the default log stream'''
    d = config.base.Configure.__getstate__(self)
    d = script.LanguageProcessor.__getstate__(self, d)
    if 'configureParent' in d:
      del d['configureParent']
    return d

  def __setstate__(self, d):
    '''We must create the default log stream'''
    config.base.Configure.__setstate__(self, d)
    script.LanguageProcessor.__setstate__(self, d)
    self.__dict__.update(d)
    return

  def listDirs(self, base, variable):
    '''Returns a list of all directories of the form base/variable where variable can be regular expression syntax'''
    if not variable: return [base]
    dirs     = []
    nextDirs = variable.split(os.sep)
    if os.path.isdir(base):
      files = os.listdir(base)
      files.sort()
      for dir in files:
        if re.match(nextDirs[0], dir):
          if nextDirs[1:]:
            rest = apply(os.path.join, nextDirs[1:])
          else:
            rest = None
          dirs.extend(self.listDirs(os.path.join(base, dir),rest ))            
    return dirs

  def getArchitecture(self):
    import sys

    auxDir = None
    searchDirs = [os.path.join(self.root, 'packages')] + sys.path
    for d in searchDirs:
      if os.path.isfile(os.path.join(d, 'config.sub')):
        auxDir      = d
        configSub   = os.path.join(auxDir, 'config.sub')
        configGuess = os.path.join(auxDir, 'config.guess')
        break
    if auxDir is None:
      self.logPrintBox('Unable to locate config.sub in '+str(searchDirs)+'.\nYour BuildSystem installation is incomplete.\n Get BuildSystem again', 'screen')
      return ('Unknown', 'Unknown', sys.platform)
    try:
      host   = config.base.Configure.executeShellCommand(self.shell+' '+configGuess, log = self.log)[0]
      output = config.base.Configure.executeShellCommand(self.shell+' '+configSub+' '+host, log = self.log)[0]
    except RuntimeError, e:
      fd = open(configGuess)
      data = fd.read()
      fd.close()
      if data.find('\r\n') >= 0:
        raise RuntimeError('''It appears BuildSystem.tar.gz is uncompressed on Windows (perhaps with Winzip)
          and files copied over to Unix/Linux. Windows introduces LF characters which are
          inappropriate on other systems. Please use gunzip/tar on the install machine.\n''')
      raise RuntimeError('Unable to determine host type using '+configSub+': '+str(e))
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m:
      raise RuntimeError('Unable to parse output of '+configSub+': '+output)
    return (m.group('cpu'), m.group('vendor'), m.group('os'))

  def getHostCPU(self):
    if not hasattr(self, '_host_cpu'):
      return self.argDB['with-host-cpu']
    return self._host_cpu
  def setHostCPU(self, cpu):
    self._host_cpu = cpu
  host_cpu = property(getHostCPU, setHostCPU, doc = 'Machine CPU')
  def getHostVendor(self):
    if not hasattr(self, '_host_vendor'):
      return self.argDB['with-host-vendor']
    return self._host_vendor
  def setHostVendor(self, vendor):
    self._host_vendor = vendor
  host_vendor = property(getHostVendor, setHostVendor, doc = 'Machine Vendor')
  def getHostOS(self):
    if not hasattr(self, '_host_os'):
      return self.argDB['with-host-os']
    return self._host_os
  def setHostOS(self, os):
    self._host_os = os
  host_os = property(getHostOS, setHostOS, doc = 'Machine OS')
  def getFileCreatePause(self):
    if not hasattr(self, '_file_create_pause'):
      return self.argDB['with-file-create-pause']
    return self._file_create_pause
  def setFileCreatePause(self, file_create_pause):
    self.file_create_pause = file_create_pause
  file_create_pause = property(getFileCreatePause, setFileCreatePause, doc = 'Add 1 sec pause between config temp file delete/recreate')

  def setupHelp(self, help):
    import nargs

    help        = config.base.Configure.setupHelp(self, help)
    searchdirs  = []
    packagedirs = []
    home = os.getenv('HOME')
    if home and os.path.isdir(home):
      packagedirs.append(home)
      searchdirs.append(home)
    list = self.listDirs('/opt/ibmcmp/vacpp/','[0-9.]*/bin')
    if list: searchdirs.append(list[-1])
    list = self.listDirs('/opt/ibmcmp/xlf/','[0-9.]*/bin')
    if list: searchdirs.append(list[-1])
    list = self.listDirs('/opt/','intel_cc_[0-9.]*/bin')
    if list: searchdirs.append(list[-1])
    list = self.listDirs('/opt/','intel_fc_[0-9.]*/bin')
    if list: searchdirs.append(list[-1])

    host_cpu, host_vendor, host_os = self.getArchitecture()

    help.addArgument('Framework', '-configModules',       nargs.Arg(None, None, 'A list of Python modules with a Configure class'))
    help.addArgument('Framework', '-ignoreCompileOutput', nargs.ArgBool(None, 1, 'Ignore compiler output'))
    help.addArgument('Framework', '-ignoreLinkOutput',    nargs.ArgBool(None, 1, 'Ignore linker output'))
    help.addArgument('Framework', '-ignoreWarnings',      nargs.ArgBool(None, 0, 'Ignore compiler and linker warnings'))
    help.addArgument('Framework', '-doCleanup',           nargs.ArgBool(None, 1, 'Delete any configure generated files (turn off for debugging)'))
    help.addArgument('Framework', '-with-alternatives',   nargs.ArgBool(None, 0, 'Provide a choice among alternative package installations'))
    help.addArgument('Framework', '-search-dirs',         nargs.Arg(None, searchdirs, 'A list of directories used to search for executables'))
    help.addArgument('Framework', '-package-dirs',        nargs.Arg(None, packagedirs, 'A list of directories used to search for packages'))
    help.addArgument('Framework', '-with-external-packages-dir=<dir>', nargs.Arg(None, None, 'Location to install downloaded packages'))
    help.addArgument('Framework', '-with-batch',          nargs.ArgBool(None, 0, 'Machine using cross-compilers or a batch system to submit jobs'))
    help.addArgument('Framework', '-with-host-cpu',       nargs.Arg(None, host_cpu,    'Machine CPU'))
    help.addArgument('Framework', '-with-host-vendor',    nargs.Arg(None, host_vendor, 'Machine vendor'))
    help.addArgument('Framework', '-with-host-os',        nargs.Arg(None, host_os,     'Machine OS'))
    help.addArgument('Framework', '-with-file-create-pause', nargs.ArgBool(None, 0, 'Add 1 sec pause between config temp file delete/recreate'))
    return help

  def getCleanup(self):
    if not hasattr(self, '_doCleanup'):
      return self.argDB['doCleanup']
    return self._doCleanup
  def setCleanup(self, doCleanup):
    self._doCleanup = doCleanup
    return
  doCleanup = property(getCleanup, setCleanup, doc = 'Flag for deleting generated files')

  def setupArguments(self, argDB):
    '''Change titles and setup all children'''
    argDB = script.Script.setupArguments(self, argDB)

    self.help.title = 'Python Configure Help\n   Comma seperated lists should be given between [] (use \[ \] in tcsh/csh)\n    For example: --with-mpi-lib=\[/usr/local/lib/libmpich.a,/usr/local/lib/libpmpich.a\]'
    self.actions.title = 'Python Configure Actions\n   These are the actions performed by configure on the filesystem'

    for child in self.childGraph.vertices:
      if hasattr(child, 'setupHelp'): child.setupHelp(self.help)
    return argDB

  def cleanup(self):
    '''Performs cleanup actions
       - Subtitute files
       - Output configure header
       - Log actions'''
    self.substitute()
    if self.makeMacroHeader:
      self.outputMakeMacroHeader(self.makeMacroHeader)
      self.log.write('**** ' + self.makeMacroHeader + ' ****\n')
      self.outputMakeMacroHeader(self.log)
      self.actions.addArgument('Framework', 'File creation', 'Created makefile configure header '+self.makeMacroHeader)
    if self.makeRuleHeader:
      self.outputMakeRuleHeader(self.makeRuleHeader)
      self.log.write('**** ' + self.makeRuleHeader + ' ****\n')
      self.outputMakeRuleHeader(self.log)
      self.actions.addArgument('Framework', 'File creation', 'Created makefile configure header '+self.makeMacroHeader)
    if self.header:
      self.outputHeader(self.header)
      self.log.write('**** ' + self.header + ' ****\n')
      self.outputHeader(self.log)
      self.actions.addArgument('Framework', 'File creation', 'Created configure header '+self.header)
    if self.cHeader:
      self.outputCHeader(self.cHeader)
      self.log.write('**** ' + self.cHeader + ' ****\n')
      self.outputCHeader(self.log)
      self.actions.addArgument('Framework', 'File creation', 'Created C specific configure header '+self.cHeader)
    self.log.write('\n')
    self.actions.output(self.log)
    return

  def printSummary(self):
    # __str__(), __str1__(), __str2__() are used to crate 3 different groups of summary outputs.
    for child in self.childGraph.vertices:
      self.logWrite(str(child), debugSection = 'screen', forceScroll = 1)
    for child in self.childGraph.vertices:
      if hasattr(child,'__str1__'):
        self.logWrite(child.__str1__(), debugSection = 'screen', forceScroll = 1)
    for child in self.childGraph.vertices:
      if hasattr(child,'__str2__'):
        self.logWrite(child.__str2__(), debugSection = 'screen', forceScroll = 1)
    return

  def addChild(self, config):
    '''Add a configure module to the framework'''
    self.childGraph.addVertex(config)
    return

  def getChild(self, moduleName, keywordArgs = {}):
    '''Returns the child matching the given module if present, and otherwise creates and appends it'''
    type   = __import__(moduleName, globals(), locals(), ['Configure']).Configure
    config = None
    for child in self.childGraph.vertices:
      if isinstance(child, type):
        config = child
        break
    if config is None and not self.configureParent is None:
      for child in self.configureParent.childGraph.vertices:
        if isinstance(child, type):
          config = child
          self.addChild(config)
          config.showHelp = 0
          config.logName  = self.logName
          # If it was pickled with a nonstandard RDict, supply one
          if config.argDB is None:
            config.argDB = self.argDB
          config.setup()
          config.setupPackageDependencies(self)
          config.setupDependencies(self)
          break
    if config is None:
      config = apply(type, [self], keywordArgs)
      self.addChild(config)
      config.showHelp = 0
      config.logName  = self.logName
      config.setup()
      config.setupPackageDependencies(self)
      config.setupDependencies(self)
    return config

  def createChildren(self):
    '''Create all children specified by --configModules'''
    import nargs

    modules = nargs.Arg.findArgument('configModules', self.clArgs)
    if modules is None:
      self.argDB['configModules'] = []
    elif not isinstance(modules, list):
      self.argDB['configModules'] = [modules]
    else:
      self.argDB['configModules'] = modules
    for moduleName in self.argDB['configModules']:
      self.getChild(moduleName)
    return

  def require(self, moduleName, depChild, keywordArgs = {}):
    '''Return a child from moduleName, creating it if necessary and making sure it runs before depChild'''
    config = self.getChild(moduleName, keywordArgs)
    self.childGraph.addEdges(depChild, [config])
    return config

  ###############################################
  # Dependency Mechanisms
  def loadFramework(self, path):
    import RDict
    oldDir = os.getcwd()
    os.chdir(path)
    argDB = RDict.RDict()
    os.chdir(oldDir)
    framework = self.loadConfigure(argDB)
    if framework is None:
      self.logPrint('Failed to load cached configuration from '+path)
    else:
      self.logPrint('Loaded cached configuration from '+path)
    return framework

  def addPackageDependency(self, dependency, depPath = None):
    if not dependency:
      return
    if isinstance(dependency, str):
      framework = self.loadFramework(dependency)
      if not framework:
        return
    else:
      framework = dependency
      if depPath:
        dependency = depPath
      else:
        dependency = os.path.dirname(dependency.__file__)
    self.dependencies[dependency] = new_md5(cPickle.dumps(framework)).hexdigest()
    self.logPrint('Added configure dependency from '+dependency+'('+str(self.dependencies[dependency])+')')
    for child in framework.childGraph.vertices:
      child.argDB = self.argDB
      child.showHelp = 0
      child.logName  = self.logName
      child.setup()
      self.childGraph.replaceVertex(self.require(child.__module__, None), child)
    return

  def updatePackageDependencies(self):
    for dependency, digest in self.dependencies.items():
      framework = self.loadFramework(dependency)
      if digest == new_md5(cPickle.dumps(framework)).hexdigest():
        continue
      self.logPrint('Configure dependency from '+dependency+' has changed. Reloading...')
      for child in framework.childGraph.vertices:
        self.childGraph.replaceVertex(self.require(child.__module__, None), child)
        self.logPrint('  Reloaded '+child.__module__)
      self.updateDependencies()
      for child in framework.childGraph.vertices:
        for depChild in self.childGraph.depthFirstVisit(child, outEdges = 0):
          if hasattr(depChild, '_configured'):
            del depChild._configured
        self.logPrint('  Will reconfigure subtree for '+child.__module__)
    return

  def updateDependencies(self):
    for child in self.childGraph.vertices:
      child.setupDependencies(self)
    return

  def setConfigureParent(self, parent):
    self.configureParent = parent
    return

  ###############################################
  # Filtering Mechanisms

  def filterPreprocessOutput(self,output):
    # Another PGI license warning, multiline so have to discard all
    if output.find('your evaluation license will expire') > -1 and output.lower().find('error') == -1:
      output = ''
    lines = output.splitlines()
    # IBM: 
    lines = filter(lambda s: not s.startswith('cc_r:'), lines)
    # PGI: Ignore warning about temporary license
    lines = filter(lambda s: s.find('license.dat') < 0, lines)
    # Cray XT3
    lines = filter(lambda s: s.find('INFO: catamount target') < 0, lines)
    lines = filter(lambda s: s.find('INFO: linux target') < 0, lines)    
    # Lahey/Fujitsu
    lines = filter(lambda s: s.find('Encountered 0 errors') < 0, lines)
    output = reduce(lambda s, t: s+t, lines, '')
    return output
  
  def filterCompileOutput(self, output):
    if self.argDB['ignoreCompileOutput']:
      output = ''
    elif output:
      lines = output.splitlines()
      if self.argDB['ignoreWarnings']:
        lines = filter(lambda s: not self.warningRE.search(s), lines)
        lines = filter(lambda s: s.find('In file included from') < 0, lines)
        lines = filter(lambda s: s.find('from ') < 0, lines)
      # GCC: Ignore headers to toplevel
      lines = filter(lambda s: s.find('At the top level') < 0, lines)
      # GCC: Ignore headers to functions
      lines = filter(lambda s: s.find(': In function') < 0, lines)
      # GCC: Ignore stupid warning about builtins
      lines = filter(lambda s: s.find('warning: conflicting types for built-in function') < 0, lines)
      # GCC: Ignore stupid warning about unused variables
      lines = filter(lambda s: s.find('warning: unused variable') < 0, lines)
      # PGI: Ignore warning about temporary license
      lines = filter(lambda s: s.find('license.dat') < 0, lines)
      # Cray XT3
      lines = filter(lambda s: s.find('INFO: catamount target') < 0, lines)
      lines = filter(lambda s: s.find('INFO: linux target') < 0, lines)
      # Lahey/Fujitsu
      lines = filter(lambda s: s.find('Encountered 0 errors') < 0, lines)
      output = reduce(lambda s, t: s+t, lines, '')
    return output

  def filterLinkOutput(self, output):
    if self.argDB['ignoreLinkOutput']:
      output = ''
    elif output:
      lines = output.splitlines()
      if self.argDB['ignoreWarnings']:
        lines = filter(lambda s: not self.warningRE.search(s), lines)
      # PGI: Ignore warning about temporary license
      lines = filter(lambda s: s.find('license.dat') < 0, lines)
      # Cray XT3
      lines = filter(lambda s: s.find('INFO: catamount target') < 0, lines)
      lines = filter(lambda s: s.find('INFO: linux target') < 0, lines)
      # Lahey/Fujitsu
      lines = filter(lambda s: s.find('Encountered 0 errors') < 0, lines)
      output = reduce(lambda s, t: s+t, lines, '')
    return output
        
  ###############################################
  # Output Mechanisms
  def addSubstitutionFile(self, inName, outName = ''):
    '''Designate that file should experience substitution
      - If outName is given, inName --> outName
      - If inName == foo.in, foo.in --> foo
      - If inName == foo,    foo.in --> foo
    '''
    if outName:
      if inName == outName:
        raise RuntimeError('Input and output substitution files identical: '+inName)
    else:
      if inName[-3:] == '.in':
        root  = inName[-3:]
      else:
        root  = inName
      inName  = root+'.in'
      outName = root
    if not os.path.exists(inName):
      raise RuntimeError('Nonexistent substitution file: '+inName)
    self.substFiles[inName] = outName
    return

  def getPrefix(self, child):
    '''Get the default prefix for a given child Configure'''
    mod = child.__class__.__module__
    if not mod == '__main__':
      prefix = mod.replace('.', '_')
    else:
      prefix = ''
    return prefix

  def getHeaderPrefix(self, child):
    '''Get the prefix for variables in the configuration header for a given child'''
    if hasattr(child, 'headerPrefix'):
      prefix = child.headerPrefix
    else:
      prefix = self.getPrefix(child)
    return prefix

  def getSubstitutionPrefix(self, child):
    '''Get the prefix for variables during substitution for a given child'''
    if hasattr(child, 'substPrefix'):
      prefix = child.substPrefix
    else:
      prefix = self.getPrefix(child)
    return prefix

  def substituteName(self, match, prefix = None):
    '''Return the substitution value for a given name, or return "@name_UNKNOWN@"'''
    name = match.group('name')
    if self.subst.has_key(name):
      return self.subst[name]
    elif self.argSubst.has_key(name):
      return self.argDB[self.argSubst[name]]
    else:
      for child in self.childGraph.vertices:
        if not hasattr(child, 'subst') or not isinstance(child.subst, dict):
          continue
        if prefix is None:
          substPrefix = self.getSubstitutionPrefix(child)
        else:
          substPrefix = prefix
        if substPrefix:
          substPrefix = substPrefix+'_'
          if name.startswith(substPrefix):
            childName = name.replace(substPrefix, '', 1)
          else:
            continue
        else:
          childName = name
        if child.subst.has_key(childName):
          return child.subst[childName]
        elif child.argSubst.has_key(childName):
          return self.argDB[child.argSubst[childName]]
    return '@'+name+'_UNKNOWN@'

  def substituteFile(self, inName, outName):
    '''Carry out substitution on the file "inName", creating "outName"'''
    inFile  = file(inName)
    if os.path.dirname(outName):
      if not os.path.exists(os.path.dirname(outName)):
        os.makedirs(os.path.dirname(outName))
    if self.file_create_pause: time.sleep(1)
    outFile = file(outName, 'w')
    for line in inFile.xreadlines():
      outFile.write(self.substRE.sub(self.substituteName, line))
    outFile.close()
    inFile.close()
    self.actions.addArgument('Framework', 'Substitution', inName+' was substituted to produce '+outName)
    return

  def substitute(self):
    '''Preform all substitution'''
    for pair in self.substFiles.items():
      self.substituteFile(pair[0], pair[1])
    return

  def dumpSubstitutions(self):
    for pair in self.subst.items():
      print pair[0]+'  --->  '+pair[1]
    for pair in self.argSubst.items():
      print pair[0]+'  --->  '+self.argDB[pair[1]]
    for child in self.childGraph.vertices:
      if not hasattr(child, 'subst') or not isinstance(child.subst, dict): continue
      substPrefix = self.getSubstitutionPrefix(child)
      for pair in child.subst.items():
        if substPrefix:
          print substPrefix+'_'+pair[0]+'  --->  '+str(pair[1])
        else:
          print pair[0]+'  --->  '+str(pair[1])
      for pair in child.argSubst.items():
        if substPrefix:
          print substPrefix+'_'+pair[0]+'  --->  '+str(self.argDB[pair[1]])
        else:
          print pair[0]+'  --->  '+str(self.argDB[pair[1]])
    return
  

  def storeSubstitutions(self, argDB):
    '''Store all the substitutions in the argument database'''
    argDB.update(self.subst)
    argDB.update(dict(map(lambda k: (k, self.argDB[self.argSubst[k]]), self.argSubst)))
    for child in self.childGraph.vertices:
      if not hasattr(child, 'subst') or not isinstance(child.subst, dict): continue
      substPrefix = self.getSubstitutionPrefix(child)
      if substPrefix:
        argDB.update(dict(map(lambda k: (substPrefix+'_'+k, child.subst[k]), child.subst)))
        argDB.update(dict(map(lambda k: (substPrefix+'_'+k, self.argDB[child.argSubst[k]]), child.argSubst)))
      else:
        argDB.update(child.subst)
        argDB.update(dict(map(lambda k: (k, self.argDB[child.argSubst[k]]), child.argSubst)))
    self.actions.addArgument('Framework', 'RDict update', 'Substitutions were stored in RDict with parent '+str(argDB.parentDirectory))
    return

  def outputDefine(self, f, name, value = None, comment = ''):
    '''Define "name" to "value" in the configuration header'''
    guard = re.match(r'^(\w+)(\([\w,]+\))?', name).group(1)
    if comment:
      for line in comment.split('\n'):
        if line: f.write('/* '+line+' */\n')
    f.write('#ifndef '+guard+'\n')
    if value:
      f.write('#define '+name+' '+str(value)+'\n')
    else:
      f.write('/* #undef '+name+' */\n')
    f.write('#endif\n\n')
    return

  def outputMakeMacro(self, f, name, value):
    f.write(name+' = '+str(value)+'\n')
    return

  def outputMakeRule(self, f, name, dependencies,rule):
    if rule:
      f.write(name+': '+dependencies+'\n')
      for r in rule:
        f.write('\t'+r+'\n')
      f.write('\n')            
    else:
      f.write(name+': '+dependencies+'\n\n')
    return

  def outputMakeRules(self, f, child, prefix = None):
    '''If the child contains a dictionary named "makerules", the entries are output in the makefile config header.
    - No prefix is used
    '''
    if not hasattr(child, 'makeRules') or not isinstance(child.makeRules, dict): return
    for pair in child.makeRules.items():
      if not pair[1]: continue
      self.outputMakeRule(f, pair[0], pair[1][0],pair[1][1])
    return

  def outputMakeMacros(self, f, child, prefix = None):
    '''If the child contains a dictionary named "makemacros", the entries are output in the makefile config header.
    - No prefix is used
    '''
    if not hasattr(child, 'makeMacros') or not isinstance(child.makeMacros, dict): return
    for pair in child.makeMacros.items():
      if not pair[1]:
        self.outputMakeMacro(f, pair[0], '')
      else:
        self.outputMakeMacro(f, pair[0], pair[1])
    return

  def outputDefines(self, f, child, prefix = None):
    '''If the child contains a dictionary named "defines", the entries are output as defines in the config header.
    The prefix to each define is calculated as follows:
    - If the prefix argument is given, this is used, otherwise
    - If the child contains "headerPrefix", this is used, otherwise
    - If the module containing the child class is not "__main__", this is used, otherwise
    - No prefix is used
    If the child contains a dictionary named "help", then a help string will be added before the define
    '''
    if not hasattr(child, 'defines') or not isinstance(child.defines, dict): return
    if hasattr(child, 'help') and isinstance(child.help, dict):
      help = child.help
    else:
      help = {}
    if prefix is None: prefix = self.getHeaderPrefix(child)
    if prefix:         prefix = prefix+'_'
    for pair in child.defines.items():
      if not pair[1]: continue
      if help.has_key(pair[0]):
        self.outputDefine(f, prefix+pair[0], pair[1], help[pair[0]])
      else:
        self.outputDefine(f, prefix+pair[0], pair[1])
    return

  def outputTypedefs(self, f, child):
    '''If the child contains a dictionary named "typedefs", the entries are output as typedefs in the config header.'''
    if not hasattr(child, 'typedefs') or not isinstance(child.typedefs, dict): return
    for newType, oldType in child.typedefs.items():
      f.write('typedef ')
      f.write(oldType)
      f.write(' ')
      f.write(newType)
      f.write(';\n')
    return

  def outputPrototypes(self, f, child, language = 'All'):
    '''If the child contains a dictionary named "prototypes", the entries for the given language are output as function prototypes in the C config header.'''
    if not hasattr(child, 'prototypes') or not isinstance(child.prototypes, dict): return
    if language in child.prototypes:
      for prototype in child.prototypes[language]:
        f.write(prototype)
        f.write('\n')
    return

  def outputMakeMacroHeader(self, name):
    '''Write the make configuration header (bmake file)'''
    if isinstance(name, file):
      f = name
      filename = 'Unknown'
    else:
      dir = os.path.dirname(name)
      if dir and not os.path.exists(dir):
        os.makedirs(dir)
      if self.file_create_pause: time.sleep(1)
      f = file(name, 'w')
      filename = os.path.basename(name)
    self.outputMakeMacros(f, self)
    for child in self.childGraph.vertices:
      self.outputMakeMacros(f, child)
    if not isinstance(name, file):
      f.close()
    return

  def outputMakeRuleHeader(self, name):
    '''Write the make configuration header (bmake file)'''
    if isinstance(name, file):
      f = name
      filename = 'Unknown'
    else:
      dir = os.path.dirname(name)
      if dir and not os.path.exists(dir):
        os.makedirs(dir)
      if self.file_create_pause: time.sleep(1)
      f = file(name, 'w')
      filename = os.path.basename(name)
    self.outputMakeRules(f, self)
    for child in self.childGraph.vertices:
      self.outputMakeRules(f, child)
    if not isinstance(name, file):
      f.close()
    return

  def outputHeader(self, name, prefix = None):
    '''Write the configuration header'''
    if isinstance(name, file):
      f = name
      filename = 'Unknown'
    else:
      dir = os.path.dirname(name)
      if dir and not os.path.exists(dir):
        os.makedirs(dir)
      if self.file_create_pause: time.sleep(1)
      f = file(name, 'w')
      filename = os.path.basename(name)
    guard = 'INCLUDED_'+filename.upper().replace('.', '_')
    f.write('#if !defined('+guard+')\n')
    f.write('#define '+guard+'\n\n')
    if hasattr(self, 'headerTop'):
      f.write(str(self.headerTop)+'\n')
    self.outputDefines(f, self, prefix)
    for child in self.childGraph.vertices:
      self.outputDefines(f, child, prefix)
    if hasattr(self, 'headerBottom'):
      f.write(str(self.headerBottom)+'\n')
    f.write('#endif\n')
    if not isinstance(name, file):
      f.close()
    return

  def outputCHeader(self, name):
    '''Write the C specific configuration header'''
    if isinstance(name, file):
      f = name
      filename = 'Unknown'
    else:
      dir = os.path.dirname(name)
      if dir and not os.path.exists(dir):
        os.makedirs(dir)
      if self.file_create_pause: time.sleep(1)
      f = file(name, 'w')
      filename = os.path.basename(name)
    guard = 'INCLUDED_'+filename.upper().replace('.', '_')
    f.write('#if !defined('+guard+')\n')
    f.write('#define '+guard+'\n\n')
    self.outputTypedefs(f, self)
    for child in self.childGraph.vertices:
      self.outputTypedefs(f, child)
    self.outputPrototypes(f, self)
    for child in self.childGraph.vertices:
      self.outputPrototypes(f, child)
    f.write('#if defined(__cplusplus)\n')
    self.outputPrototypes(f, self, 'Cxx')
    for child in self.childGraph.vertices:
      self.outputPrototypes(f, child, 'Cxx')
    f.write('extern "C" {\n')
    self.outputPrototypes(f, self, 'extern C')
    for child in self.childGraph.vertices:
      self.outputPrototypes(f, child, 'extern C')
    f.write('}\n')
    f.write('#else\n')
    self.outputPrototypes(f, self, 'C')
    for child in self.childGraph.vertices:
      self.outputPrototypes(f, child, 'C')
    f.write('#endif\n')
    f.write('#endif\n')
    if not isinstance(name, file):
      f.close()
    return

  def getOptionsString(self, omitArgs = []):
    import nargs
    args = self.clArgs[:]
    for arg in omitArgs:
      args = filter(lambda a: not nargs.Arg.parseArgument(a)[0] == arg, args)
    for a, arg in enumerate(args):
      parts = arg.split('=',1)
      if len(parts) == 2 and (' ' in parts[1] or '[' in parts[1]):
        args[a] = parts[0]+'=\"'+parts[1]+'\"'
    return ' '.join(args)

  def outputBanner(self):
    import time, sys
    self.log.write(('='*80)+'\n')
    self.log.write(('='*80)+'\n')
    self.log.write('Starting Configure Run at '+time.ctime(time.time())+'\n')
    self.log.write('Configure Options: '+self.getOptionsString()+'\n')
    self.log.write('Working directory: '+os.getcwd()+'\n')
    self.log.write('Machine uname:\n' + str(os.uname())+'\n')
    self.log.write('Python version:\n' + sys.version+'\n')
    self.log.write(('='*80)+'\n')
    return

  def configureExternalPackagesDir(self):
    if 'with-external-packages-dir' in self.argDB:
      self.externalPackagesDir = self.argDB['with-external-packages-dir']
    else:
      self.externalPackagesDir = None
    return

  def addBatchSetup(self, setup):
    '''Add code to be run before batch tests execute'''
    if not isinstance(setup, list):
      setup = [setup]
    self.batchSetup.extend(setup)
    return

  def addBatchInclude(self, includes):
    '''Add an include or a list of includes to the batch run'''
    if not isinstance(includes, list):
      includes = [includes]
    self.batchIncludes.extend(includes)
    return

  def addBatchLib(self, libs):
    '''Add a library or a list of libraries to the batch run'''
    if not isinstance(libs, list):
      libs = [libs]
    self.batchLibs.extend(libs)
    return

  def addBatchBody(self, statements):
    '''Add a statement or a list of statements to the batch run'''
    if not isinstance(statements, list):
      statements = [statements]
    self.batchBodies.extend(statements)
    return

  def addBatchCleanup(self, cleanup):
    '''Add code to be run after batch tests execute'''
    if not isinstance(cleanup, list):
      cleanup = [cleanup]
    self.batchCleanup.extend(cleanup)
    return

  def configureBatch(self):
    '''F'''
    if self.batchBodies:
      import nargs
      import sys

      args = self.clArgs[:]
      body = ['FILE *output = fopen("reconfigure.py","w");']
      body.append('fprintf(output, "#!'+sys.executable+'\\n");')
      body.append('fprintf(output, "\\nconfigure_options = [\\n");')
      body.extend(self.batchSetup)
      body.extend(self.batchBodies)
      body.extend(self.batchCleanup)
      # pretty print repr(args.values())
      for itm in args:
        if (itm != '--configModules=PETSc.Configure') and (itm != '--optionsModule=PETSc.compilerOptions'):
          body.append('fprintf(output,"  \''+str(itm)+'\',\\n");')
      body.append('fprintf(output,"]");')
      driver = ['fprintf(output, "\\nif __name__ == \'__main__\':',
                '  import os',
                '  import sys',
                '  sys.path.insert(0, os.path.abspath(\'config\'))',
                '  import configure',
                '  configure.petsc_configure(configure_options)\\n");']
      body.append('\\n'.join(driver))
      body.append('\nfclose(output);\n')
      body.append('chmod("reconfigure.py",0744);')

      oldFlags = self.compilers.CPPFLAGS
      oldLibs  = self.compilers.LIBS
      self.compilers.CPPFLAGS += ' ' + ' '.join(self.batchIncludeDirs)
      self.compilers.LIBS = self.libraries.toString(self.batchLibs)+' '+self.compilers.LIBS
      self.batchIncludes.insert(0, '#include <stdio.h>\n#include <sys/types.h>\n#include <sys/stat.h>')
      if not self.checkLink('\n'.join(self.batchIncludes)+'\n', '\n'.join(body), cleanup = 0, codeBegin = '\nint main(int argc, char **argv) {\n'):
        sys.exit('Unable to generate test file for cross-compilers/batch-system\n')
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      self.logClear()
      print '=================================================================================\r'
      print '    Since your compute nodes require use of a batch system or mpiexec you must:  \r'
      print ' 1) Submit ./conftest to 1 processor of your batch system or system you are      \r'
      print '    cross-compiling for; this will generate the file reconfigure.py              \r'
      print ' 2) Run ./reconfigure.py (to complete the configure process).                    \r'
      print '=================================================================================\r'
      sys.exit(0)
    return

  def configure(self, out = None):
    '''Configure the system
       - Must delay database initialization until children have contributed variable types
       - Any child with the "_configured" attribute will not be configured'''
    import graph

    self.setup()
    self.outputBanner()
    self.updateDependencies()
    self.executeTest(self.configureExternalPackagesDir)
    for child in graph.DirectedGraph.topologicalSort(self.childGraph):
      if not hasattr(child, '_configured'):
        child.configure()
      else:
        child.no_configure()
      child._configured = 1
    if self.argDB['with-batch']:
      self.configureBatch()
    self.cleanup()
    return 1
