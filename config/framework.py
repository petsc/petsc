import script
import config.base

import os
import re

class Framework(config.base.Configure, script.LanguageProcessor):
  def __init__(self, clArgs = None, argDB = None, loadArgDB = 1):
    import graph
    import nargs

    if argDB is None:
      import RDict

      argDB = RDict.RDict(load = loadArgDB)
    self.argDB  = argDB
    self.clArgs = clArgs
    if not nargs.Arg.findArgument('debugSections', self.clArgs):
      self.argDB['debugSections'] = ['screen']
    config.base.Configure.__init__(self, self)
    script.LanguageProcessor.__init__(self)
    self.childGraph   = graph.DirectedGraph()
    self.substRE      = re.compile(r'@(?P<name>[^@]+)@')
    self.substFiles   = {}
    self.logName      = 'configure.log'
    self.header       = 'matt_config.h'
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.warningRE    = re.compile('warning', re.I)
    self.createChildren()
    # Perhaps these initializations should just be local temporary arguments
    self.argDB['CPPFLAGS']   = ''
    self.argDB['LIBS']       = ''
    if not 'LDFLAGS' in self.argDB:
      self.argDB['LDFLAGS']  = ''
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
    
    help.addArgument('Framework', '-configModules',       nargs.Arg(None, None, 'A list of Python modules with a Configure class'))
    help.addArgument('Framework', '-ignoreCompileOutput', nargs.ArgBool(None, 1, 'Ignore compiler output'))
    help.addArgument('Framework', '-ignoreLinkOutput',    nargs.ArgBool(None, 1, 'Ignore linker output'))
    help.addArgument('Framework', '-ignoreWarnings',      nargs.ArgBool(None, 0, 'Ignore compiler and linker warnings'))
    help.addArgument('Framework', '-with-alternatives',   nargs.ArgBool(None, 0, 'Provide a choice among alternative package installations'))
    help.addArgument('Framework', '-search-dirs',         nargs.Arg(None, searchdirs, 'A list of directories used to search for executables'))
    help.addArgument('Framework', '-package-dirs',        nargs.Arg(None, packagedirs, 'A list of directories used to search for packages'))
    help.addArgument('Framework', '-can-execute',         nargs.ArgBool(None, 1, 'Disable this option on a batch system'))
    return help

  def setupArguments(self, argDB):
    '''Change titles and setup all children'''
    argDB = script.Script.setupArguments(self, argDB)

    self.help.title = 'Python Configure Help\n   Comma seperated lists should be given between [] (use \[ \] in tcsh/csh)\n    For example: --with-mpi-lib=\[/usr/local/lib/libmpich.a,/usr/local/lib/libpmpich.a\]'
    self.actions.title = 'Python Configure Actions\n   These are the actions performed by configure on the filesystem'

    for child in self.childGraph.vertices:
      if hasattr(child, 'setupHelp'): child.setupHelp(self.help)
    return argDB

  def setup(self):
    config.base.Configure.setup(self)
    for child in self.childGraph.vertices:
      child.setup()
    return

  def cleanup(self):
    '''Performs cleanup actions
       - Log all child string methods
       - Subtitute files
       - Output configure header
       - Log actions'''
    for child in self.childGraph.vertices:
      self.logWrite(str(child), debugSection = 'screen', forceScroll = 1)
    self.substitute()
    self.outputHeader(self.header)
    self.actions.addArgument('Framework', 'File creation', 'Created configure header '+self.header)
    self.log.write('\n')
    self.actions.output(self.log)
    return

  def addChild(self, config):
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
    if config is None:
      config = apply(type, [self], keywordArgs)
      self.addChild(config)
    return config

  def createChildren(self):
    '''Create all children specified by --configModules'''
    import nargs

    self.argDB['configModules'] = nargs.Arg.findArgument('configModules', self.clArgs)
    if self.argDB['configModules'] is None:
      self.argDB['configModules'] = []
    elif not isinstance(self.argDB['configModules'], list):
      self.argDB['configModules'] = [self.argDB['configModules']]
    for moduleName in self.argDB['configModules']:
      try:
        self.getChild(moduleName)
      except ImportError, e:
        self.logPrint('Could not import config module '+moduleName+': '+str(e))
    return

  def require(self, moduleName, depChild, keywordArgs = {}):
    '''Return a child from moduleName, creating it if necessary and making sure it runs before depChild'''
    config = self.getChild(moduleName, keywordArgs)
    self.childGraph.addEdges(depChild, [config])
    return config

  ###############################################
  # Filtering Mechanisms
  def filterCompileOutput(self, output):
    if self.argDB['ignoreCompileOutput']:
      output = ''
    elif output:
      lines = output.splitlines()
      if self.framework.argDB['ignoreWarnings']:
        lines = filter(lambda s: not self.warningRE.search(s), lines)
      # Ignore stupid warning from gcc about builtins
      lines = filter(lambda s: s.find('warning: conflicting types for built-in function') < 0, lines)
      output = reduce(lambda s, t: s+t, lines, '')
    return output

  def filterLinkOutput(self, output):
    if self.argDB['ignoreLinkOutput']:
      output = ''
    elif output:
      lines = output.splitlines()
      if self.framework.argDB['ignoreWarnings']:
        lines = filter(lambda s: not self.warningRE.search(s), lines)
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
    if not os.path.exists(os.path.dirname(outName)):
      os.makedirs(os.path.dirname(outName))
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

  def outputHeader(self, name):
    '''Write the configuration header'''
    if isinstance(name, file):
      f = name
      filename = 'Unknown'
    else:
      dir = os.path.dirname(name)
      if dir and not os.path.exists(dir):
        os.makedirs(dir)
      f = file(name, 'w')
      filename = os.path.basename(name)
    guard = 'INCLUDED_'+filename.upper().replace('.', '_')
    f.write('#if !defined('+guard+')\n')
    f.write('#define '+guard+'\n\n')
    if hasattr(self, 'headerTop'):
      f.write(str(self.headerTop)+'\n')
    self.outputDefines(f, self)
    for child in self.childGraph.vertices:
      self.outputDefines(f, child)
    if hasattr(self, 'headerBottom'):
      f.write(str(self.headerBottom)+'\n')
    f.write('#endif\n')
    if not isinstance(name, file):
      f.close()
    return

  def outputBanner(self):
    import time
    self.log.write(('='*80)+'\n')
    self.log.write(('='*80)+'\n')
    self.log.write('Starting Configure Run at '+time.ctime(time.time())+'\n')
    self.log.write('Configure Options: '+str(self.clArgs)+'\n')
    self.log.write('Working directory: '+os.getcwd()+'\n')
    self.log.write(('='*80)+'\n')
    return

  def configure(self, out = None):
    '''Configure the system
       - Must delay database initialization until children have contributed variable types'''
    import graph

    self.setup()
    self.outputBanner()
    for child in graph.DirectedGraph.topologicalSort(self.childGraph):
      child.configure()
    self.cleanup()
    return 1
