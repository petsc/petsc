import nargs
import config.base

import os
import re

class Help:
  def __init__(self, framework):
    self.framework = framework
    self.options   = {}
    self.sections  = []
    return

  def setTitle(self, title):
    self.title = title

  def addOption(self, section, name, comment, argType = nargs.ArgString, **kwargs):
    if self.options.has_key(section):
      if self.options[section].has_key(name):
        raise RuntimeError('Duplicate configure option '+name+' in section '+section)
    else:
      self.sections.append(section)
      self.options[section] = {}
    self.options[section][name] = (comment, len(self.options[section]))
    varName = name.split('=')[0]
    self.framework.argDB.setLocalType(varName, argType('Print help message', **kwargs))
    return

  def output(self):
    import sys

    print self.title
    for i in range(len(self.title)): sys.stdout.write('-')
    print
    nameLen = 1
    descLen = 1
    for section in self.sections:
      nameLen = max([nameLen, max(map(len, self.options[section].keys()))+1])
      descLen = max([descLen, max(map(lambda t: len(t[0]), self.options[section].values()))+1])
    format    = '  -%-'+str(nameLen)+'s: %s'
    formatDef = '  -%-'+str(nameLen)+'s: %-'+str(descLen)+'s  current: %s'
    for section in self.sections:
      print section+':'
      items = self.options[section].items()
      items.sort(lambda a, b: a[1][1].__cmp__(b[1][1]))
      for item in items:
        varName = item[0].split('=')[0].strip('-')

        if self.framework.argDB.has_key(varName):
          print formatDef % (item[0], item[1][0], str(self.framework.argDB[varName]))
        else:
          print format % (item[0], item[1][0])
    return

class Framework(config.base.Configure):
  def __init__(self, clArgs = None, argDB = None):
    self.argDB = self.setupArgDB(clArgs, argDB)
    config.base.Configure.__init__(self, self)
    self.children   = []
    self.substRE    = re.compile(r'@(?P<name>[^@]+)@')
    self.substFiles = {}
    self.header     = 'matt_config.h'
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.warningRE    = re.compile('warning', re.I)
    self.setupChildren()
    return

  def setupArgDB(self, clArgs, initDB):
    self.clArgs = clArgs
    if initDB is None:
      argDB = nargs.ArgDict('ArgDict', localDict = 1)
    else:
      argDB = initDB
    return argDB

  def setupLogging(self):
    if not hasattr(self, 'log'):
      self.logName   = self.argDB['log']
      self.logExists = os.path.exists(self.logName)
      if self.logExists:
        self.log     = file(self.logName, 'a')
      else:
        self.log     = file(self.logName, 'w')
    return self.log

  def setupArguments(self, clArgs = None):
    '''Set initial arguments into the database, and setup initial types'''
    self.help = Help(self)
    self.help.setTitle('Python Configure Help')

    self.configureHelp(self.help)
    for child in self.children:
      if hasattr(child, 'configureHelp'): child.configureHelp(self.help)

    self.argDB.insertArgs(os.environ)
    self.argDB.insertArgs(clArgs)
    return

  def setupChildren(self):
    self.argDB['configModules'] = nargs.findArgument('configModules', self.clArgs)
    if self.argDB['configModules'] is None:
      self.argDB['configModules'] = []
    elif not isinstance(self.argDB['configModules'], list):
      self.argDB['configModules'] = [self.argDB['configModules']]
    for moduleName in self.argDB['configModules']:
      try:
        self.children.append(__import__(moduleName, globals(), locals(), ['Configure']).Configure(self))
      except ImportError, e:
        print 'Could not import config module '+moduleName+': '+str(e)
    return

  def require(self, moduleName, depChild = None, keywordArgs = {}):
    type   = __import__(moduleName, globals(), locals(), ['Configure']).Configure
    config = None
    for child in self.children:
      if isinstance(child, type):
        config = child
    if not config:
      config = apply(type, [self], keywordArgs)
      self.children.append(config)
    if depChild in self.children and self.children.index(config) > self.children.index(depChild):
      self.children.remove(config)
      self.children.insert(self.children.index(depChild), config)
    return config
        
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
      for child in self.children:
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
    for child in self.children:
      if not hasattr(child, 'subst') or not isinstance(child.subst, dict): continue
      substPrefix = self.getSubstitutionPrefix(child)
      for pair in child.subst.items():
        if substPrefix:
          print substPrefix+'_'+pair[0]+'  --->  '+pair[1]
        else:
          print pair[0]+'  --->  '+pair[1]
      for pair in child.argSubst.items():
        if substPrefix:
          print substPrefix+'_'+pair[0]+'  --->  '+self.argDB[pair[1]]
        else:
          print pair[0]+'  --->  '+self.argDB[pair[1]]
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
    If the child contains a dictinary name "help", then a help string will be added before the define
    '''
    if not hasattr(child, 'defines') or not isinstance(child.defines, dict): return
    if hasattr(child, 'help') and isinstance(child.help, dict):
      help = child.help
    else:
      help = {}
    if prefix is None: prefix = self.getHeaderPrefix(child)
    if prefix:         prefix = prefix+'_'
    for pair in child.defines.items():
      if help.has_key(pair[0]):
        self.outputDefine(f, prefix+pair[0], pair[1], help[pair[0]])
      else:
        self.outputDefine(f, prefix+pair[0], pair[1])
    return

  def outputHeader(self, name):
    '''Write the configuration header'''
    f = file(name, 'w')
    guard = 'INCLUDED_'+os.path.basename(name).upper().replace('.', '_')
    f.write('#if !defined('+guard+')\n')
    f.write('#define '+guard+'\n\n')
    if hasattr(self, 'headerTop'):
      f.write(str(self.headerTop)+'\n')
    self.outputDefines(f, self)
    for child in self.children:
      self.outputDefines(f, child)
    if hasattr(self, 'headerBottom'):
      f.write(str(self.headerBottom)+'\n')
    f.write('#endif /* '+guard+' */\n')
    f.close()
    return

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

  def outputBanner(self):
    import time
    self.log.write(('='*80)+'\n')
    self.log.write(('='*80)+'\n')
    self.log.write('Starting Configure Run at '+time.ctime(time.time())+'\n')
    self.log.write('Configure Options: '+str(self.clArgs)+'\n')
    self.log.write('Working directory: '+os.getcwd()+'\n')
    self.log.write(('='*80)+'\n')
    return

  def configureHelp(self, help):
    help.addOption('Framework', 'configModules', 'A list of Python modules with a Configure class')
    help.addOption('Framework', 'help', 'Print this help message', nargs.ArgBool)
    help.addOption('Framework', 'h', 'Print this help message', nargs.ArgBool)
    help.addOption('Framework', 'log', 'The filename for the configure log', nargs.ArgString)
    help.addOption('Framework', 'ignoreCompileOutput', 'Ignore compiler output', nargs.ArgBool)
    help.addOption('Framework', 'ignoreLinkOutput', 'Ignore linker output', nargs.ArgBool)
    help.addOption('Framework', 'ignoreWarnings', 'Ignore compiler and linker warnings', nargs.ArgBool)

    self.argDB['h']                   = 0
    self.argDB['help']                = 0
    self.argDB['log']                 = 'configure.log'
    self.argDB['ignoreCompileOutput'] = 1
    self.argDB['ignoreLinkOutput']    = 1
    self.argDB['ignoreWarnings']      = 0
    return

  def configure(self):
    '''Configure the system'''
    self.checkPython()
    # Delay database initialization until children have contributed variable types
    self.setupArguments(self.clArgs)
    self.setupLogging()
    if self.argDB['help'] or self.argDB['h']:
      self.help.output()
      return
    self.outputBanner()
    for child in self.children:
      print 'Configuring '+child.__module__
      child.configure()
    self.substitute()
    self.outputHeader(self.header)
    return
