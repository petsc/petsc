import compile
import fileset
import link
import target
import transform

class Defaults:
  def __init__(self, sidlTargets, etagsFile = None):
    self.sidlTargets = sidlTargets
    self.etagsFile    = etagsFile

  def doLibraryCheck(self):
    return not self.sidlTargets.project.getName() == 'bs' and not self.sidlTargets.project.getName() == 'sidlruntime'

  def getClientCompileTargets(self, lang, doCompile = 1, doLink = 1):
    compiler = []
    linker   = []
    try:
      if doCompile: compiler = self.sidlTargets.getUsing(lang).getClientCompileTarget(self.sidlTargets.project)
      if doLink:    linker   = self.sidlTargets.getUsing(lang).getClientLinkTarget(self.sidlTargets.project, self.doLibraryCheck())
    except AttributeError, e:
      print e
      raise RuntimeError('Unknown client language: '+lang)
    return compiler+linker+[transform.Update(self.sidlTargets.sourceDB)]

  def getServerCompileTargets(self, lang, package, doCompile = 1, doLink = 1):
    compiler = []
    linker   = []
    try:
      if doCompile: compiler = self.sidlTargets.getUsing(lang).getServerCompileTarget(self.sidlTargets.project, package)
      if doLink:    linker   = self.sidlTargets.getUsing(lang).getServerLinkTarget(self.sidlTargets.project, package, self.doLibraryCheck())
    except AttributeError, e:
      print e
      raise RuntimeError('Unknown server language: '+lang)
    return compiler+linker+[transform.Update(self.sidlTargets.sourceDB)]

  def getEmacsTagsTargets(self):
    compiler = compile.CompileEtags(self.sidlTargets.sourceDB, self.etagsFile)
    if compiler.checkCompiler():
      return [transform.FileFilter(self.sidlTargets.isImpl), compile.TagEtags(self.sidlTargets.sourceDB), compiler]
    return []

  def getCompileTargets(self, doCompile = 1, doLink = 1):
    bootstrapTargets = []
    clientTargets    = []
    serverTargets    = []

    for lang in self.sidlTargets.usingSIDL.clientLanguages:
      clientTargets.append(self.getClientCompileTargets(lang, doCompile, doLink))
    for lang in self.sidlTargets.usingSIDL.serverLanguages:
      for package in self.sidlTargets.getPackages():
        t = self.getServerCompileTargets(lang, package, doCompile, doLink)

        if package in self.sidlTargets.usingSIDL.bootstrapPackages:
          bootstrapTargets.append(t)
        else:
          serverTargets.append(t)

    compileTargets = bootstrapTargets+clientTargets+serverTargets

    if self.etagsFile:
      return [(compileTargets, self.getEmacsTagsTargets()), transform.Update(self.sidlTargets.sourceDB)]
    else:
      return compileTargets+[transform.Update(self.sidlTargets.sourceDB)]

  def getCompileTarget(self):
    return target.Target(None, [self.sidlTargets.getSIDLTarget()]+self.getCompileTargets(1, 1))

  def getExecutableDriverTargets(self, sources, lang, executable):
    try:
      compiler = self.sidlTargets.getUsing(lang).getExecutableCompileTarget(self.sidlTargets.project, sources, executable)
      linker   = self.sidlTargets.getUsing(lang).getExecutableLinkTarget(self.sidlTargets.project)
    except AttributeError, e:
      import sys
      import traceback
        
      print str(e)
      print traceback.print_tb(sys.exc_info()[2])
      raise RuntimeError('Unknown executable language: '+lang)
    return compiler+linker+[transform.Update(self.sidlTargets.sourceDB)]

  def getExecutableTarget(self, lang, sources, executable, noClient = 0):
    # TODO: Of course this should be determined from configure
    libraries = fileset.FileSet(['libdl.so'])

    # It is the repsonsibility of the user to make sure the implementation is built prior to use
    t = []
    if not noClient:
      t += self.getClientCompileTargets(lang)
    t += self.getExecutableDriverTargets(sources, lang, executable)
    if not lang == 'Java':
      t += [link.TagShared(self.sidlTargets.sourceDB), link.LinkExecutable(executable, extraLibraries = libraries)]
    return target.Target(sources, t)
