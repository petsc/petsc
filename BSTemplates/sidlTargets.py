import bk
import bs
import fileset
import logging
import target
import transform
import BSTemplates.compileDefaults as compileDefaults
import BSTemplates.sidlDefaults as sidlDefaults

import os

class Defaults(logging.Logger):
  def __init__(self, project, sources = None, bootstrapPackages = []):
    self.project    = project
    self.sources    = sources
    self.usingSIDL  = sidlDefaults.UsingSIDL(project, self.getPackages(), bootstrapPackages = bootstrapPackages)
    self.compileExt = []
    # Add C for the IOR
    self.addLanguage('C')
    # Setup compiler specific defaults
    if bs.argDB.has_key('babelCrap') and int(bs.argDB['babelCrap']):
      self.debugPrint('Compiling SIDL with Babel', 3, 'sidl')
      import BSTemplates.babelTargets
      self.compilerDefaults = BSTemplates.babelTargets.Defaults(self)
    else:
      self.debugPrint('Compiling SIDL with Scandal', 3, 'sidl')
      import BSTemplates.scandalTargets
      self.compilerDefaults = BSTemplates.scandalTargets.Defaults(self)
    return

  def getUsing(self, lang):
    return getattr(self, 'using'+lang.replace('+', 'x'))

  def addLanguage(self, lang):
    try:
      self.getUsing(lang.replace('+', 'x'))
    except AttributeError:
      lang = lang.replace('+', 'x')
      opt  = getattr(compileDefaults, 'Using'+lang)(self.usingSIDL)
      setattr(self, 'using'+lang, opt)
      self.compileExt.extend(opt.getCompileSuffixes())
    return

  def addClientLanguage(self, lang):
    self.usingSIDL.clientLanguages.append(lang)
    self.addLanguage(lang)

  def addServerLanguage(self, lang):
    self.usingSIDL.serverLanguages.append(lang)
    self.addLanguage(lang)

  def isImpl(self, source):
    if os.path.splitext(source)[1] == '.pyc':
      return 0
    if self.compilerDefaults.getImplRE().match(os.path.dirname(source)):
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

  def getSIDLServerCompiler(self, lang, rootDir, generatedRoots):
    compiler           = self.compilerDefaults.getCompilerModule().CompileSIDLServer(fileset.ExtensionFileSet(generatedRoots, self.compileExt),
                                                                                   compilerFlags = self.usingSIDL.getCompilerFlags())
    compiler.language  = lang
    compiler.outputDir = rootDir
    self.compilerDefaults.setupIncludes(compiler)
    return compiler

  def getSIDLClientCompiler(self, lang, rootDir):
    compiler           = self.compilerDefaults.getCompilerModule().CompileSIDLClient(fileset.ExtensionFileSet(rootDir, self.compileExt),
                                                                                     compilerFlags = self.usingSIDL.getCompilerFlags())
    compiler.language  = lang
    compiler.outputDir = rootDir
    self.compilerDefaults.setupIncludes(compiler)
    return compiler

  def getSIDLPrintCompiler(self, outputDir = None, printer = None):
    compiler = self.compilerDefaults.getCompilerModule().CompileSIDLPrint(compilerFlags = self.usingSIDL.getCompilerFlags())
    if outputDir: compiler.outputDir = outputDir
    if printer:   compiler.printer   = printer
    self.compilerDefaults.setupIncludes(compiler)
    return compiler

  def getRepositoryTargets(self):
    action = self.compilerDefaults.getCompilerModule().CompileSIDLRepository(compilerFlags = self.usingSIDL.getCompilerFlags())
    action.outputDir = os.path.join(self.usingSIDL.repositoryDir, 'xml')
    action.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return [target.Target(None, [self.compilerDefaults.getTagger('repository'), action])]

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

      targets.append(target.Target(None, [self.compilerDefaults.getTagger('server'), target.If(self.isNewSidl, genActions, defActions)]))
    return targets

  def getSIDLClientTargets(self):
    targets = []
    for lang in self.usingSIDL.clientLanguages:
      targets.append(target.Target(None, [self.compilerDefaults.getTagger('client'),
                                          self.getSIDLClientCompiler(lang, self.usingSIDL.getClientRootDir(lang))]))
    # Some clients have to be linked with the corresponding server (like the Bable bootstrap)
    for package in self.getPackages():
      for lang in self.usingSIDL.internalClientLanguages[package]:
        targets.append(target.Target(None, [self.compilerDefaults.getTagger('client'),
                                            self.getSIDLClientCompiler(lang, self.usingSIDL.getServerRootDir(lang, package))]))
    return targets

  def getSIDLTarget(self):
    return target.Target(self.sources, [tuple(self.getRepositoryTargets()+self.getSIDLServerTargets()+self.getSIDLClientTargets()),
                                        transform.Update(),
                                        transform.SetFilter('old sidl')])

  def getSIDLPrintTarget(self):
    return target.Target(self.sources, [sidlDefaults.TagAllSIDL(force = 1), self.getSIDLPrintCompiler()])
