import bk
import bs
import fileset
import maker
import target
import transform
import BSTemplates.compileDefaults as compileDefaults
import BSTemplates.sidlDefaults as sidlDefaults

import os

class Defaults(maker.Maker):
  def __init__(self, project, sourceDB, argDB = None, sources = None, bootstrapPackages = []):
    maker.Maker.__init__(self, argDB)
    self.project    = project
    self.sourceDB   = sourceDB
    self.sources    = sources
    self.usingSIDL  = sidlDefaults.UsingSIDL(project, self.getPackages(), bootstrapPackages = bootstrapPackages)
    self.compileExt = []
    self.masterCompiler = self.usingSIDL.compilerDefaults.getCompilerModule().__name__
    # Give source database to usingSIDL
    self.usingSIDL.sourceDB = self.sourceDB
    self.setupLanguages()
    return

  def setupLanguages(self):
    for language in self.argDB['clientLanguages']:
      self.addClientLanguage(language)
    # Add C for the IOR
    self.addLanguage('C')
    return

  def getUsing(self, lang):
    return getattr(self, 'using'+lang.replace('+', 'x'))

  def addLanguage(self, lang):
    try:
      self.getUsing(lang)
    except AttributeError:
      lang = lang.replace('+', 'x')
      opt  = getattr(compileDefaults, 'Using'+lang)(self.usingSIDL)
      setattr(self, 'using'+lang, opt)
      self.compileExt.extend(opt.getCompileSuffixes())
    return

  def addClientLanguage(self, lang):
    if lang in self.argDB['installedLanguages']:
      if not lang in self.usingSIDL.clientLanguages:
        self.usingSIDL.clientLanguages.append(lang)
        self.addLanguage(lang)
    else:
      self.debugPrint('Language '+lang+' not installed', 2, 'compile')
    return

  def addServerLanguage(self, lang):
    if lang in self.argDB['installedLanguages']:
      if not lang in self.usingSIDL.serverLanguages:
        self.usingSIDL.serverLanguages.append(lang)
        self.addLanguage(lang)
    else:
      self.debugPrint('Language '+lang+' not installed', 2, 'compile')
    return

  def isImpl(self, source):
    if not self.usingSIDL.compilerDefaults.getCompilerModule().__name__ == self.masterCompiler:
      return 0
    if os.path.splitext(source)[1] == '.pyc':
      return 0
    if self.usingSIDL.compilerDefaults.getImplRE().match(os.path.dirname(source)):
      return 1
    return 0

  def isNewSidl(self, sources):
    if isinstance(sources, fileset.FileSet):
      if sources.tag == 'sidl' and len(sources) > 0:
        return 1
      else:
        return 0
    elif isinstance(sources, list):
      isNew = 0
      for source in sources:
        isNew = isNew or self.isNewSidl(source)
      return isNew
    else:
      raise RuntimeError('Invalid type for sources: '+type(sources))

  def getPackages(self):
    if self.sources:
      sources = self.sources
    else:
      sources = []
    return map(lambda file: os.path.splitext(os.path.split(file)[1])[0], sources)

  def getSIDLServerCompiler(self, lang, rootDir, generatedRoots, flags = None):
    if not flags: flags = self.usingSIDL.getServerCompilerFlags(lang)
    compiler            = self.usingSIDL.compilerDefaults.getCompilerModule().CompileSIDLServer(self.sourceDB,
                                                                                                fileset.ExtensionFileSet(generatedRoots, self.compileExt),
                                                                                                compilerFlags = flags)
    compiler.language   = lang
    compiler.outputDir  = rootDir
    self.usingSIDL.compilerDefaults.setupIncludes(compiler)
    return compiler

  def getSIDLClientCompiler(self, lang, rootDir, flags = None):
    if not flags: flags = self.usingSIDL.getClientCompilerFlags(lang)
    compiler            = self.usingSIDL.compilerDefaults.getCompilerModule().CompileSIDLClient(self.sourceDB,
                                                                                                fileset.ExtensionFileSet(rootDir, self.compileExt),
                                                                                                compilerFlags = flags)
    compiler.language   = lang
    compiler.outputDir  = rootDir
    self.usingSIDL.compilerDefaults.setupIncludes(compiler)
    return compiler

  def getSIDLPrintCompiler(self, outputDir = None, printer = None):
    compiler = self.usingSIDL.compilerDefaults.getCompilerModule().CompileSIDLPrint(self.sourceDB)
    if outputDir: compiler.outputDir = outputDir
    if printer:   compiler.printer   = printer
    self.usingSIDL.compilerDefaults.setupIncludes(compiler)
    return compiler

  def getRepositoryTargets(self):
    action = self.usingSIDL.compilerDefaults.getCompilerModule().CompileSIDLRepository(self.sourceDB, compilerFlags = self.usingSIDL.getClientCompilerFlags(self.usingSIDL.getBaseLanguage()))
    action.outputDir = os.path.join(self.usingSIDL.repositoryDir, 'xml')
    action.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return [target.Target(None, [self.usingSIDL.compilerDefaults.getTagger('repository'), action])]

  def getSIDLServerTargets(self):
    targets = []
    for lang in self.usingSIDL.serverLanguages:
      serverSourceRoots = fileset.FileSet(map(lambda package, lang=lang, self=self: self.usingSIDL.getServerRootDir(lang, package), self.getPackages()))
      for rootDir in serverSourceRoots:
        if not os.path.isdir(rootDir):
          os.makedirs(rootDir)

      genActions = [bk.TagBKOpen(roots = serverSourceRoots),
                    bk.BKOpen(),
                    # CompileSIDLServer() will add the package automatically to the output directory
                    self.getSIDLServerCompiler(lang, self.usingSIDL.getServerRootDir(lang), serverSourceRoots),
                    bk.TagBKClose(roots = serverSourceRoots),
                    transform.FileFilter(self.isImpl, tags = 'bkadd'),
                    bk.BKClose()]

      defActions = transform.Transform(fileset.ExtensionFileSet(serverSourceRoots, self.compileExt))

      targets.append(target.Target(None, [self.usingSIDL.compilerDefaults.getTagger('server'), target.If(self.isNewSidl, genActions, defActions)]))
    return targets

  def getSIDLClientTargets(self):
    targets = []
    for lang in self.usingSIDL.clientLanguages:
      targets.append(target.Target(None, [self.usingSIDL.compilerDefaults.getTagger('client'),
                                          self.getSIDLClientCompiler(lang, self.usingSIDL.getClientRootDir(lang))]))
    # Some clients have to be linked with the corresponding server (like the Bable bootstrap)
    for package in self.getPackages():
      for lang in self.usingSIDL.internalClientLanguages[package]:
        targets.append(target.Target(None, [self.usingSIDL.compilerDefaults.getTagger('client'),
                                            self.getSIDLClientCompiler(lang, self.usingSIDL.getServerRootDir(lang, package))]))
    extras = self.usingSIDL.compilerDefaults.getExtraClientTargets()
    if len(extras): targets.extend(extras)
    return targets

  def getSIDLTarget(self):
    return target.Target(self.sources, [tuple(self.getRepositoryTargets()+self.getSIDLServerTargets()+self.getSIDLClientTargets()),
                                        transform.Update(self.sourceDB),
                                        transform.SetFilter('old sidl')])

  def getSIDLPrintTarget(self):
    return target.Target(self.sources, [sidlDefaults.TagAllSIDL(self.sourceDB, force = 1), self.getSIDLPrintCompiler()])
