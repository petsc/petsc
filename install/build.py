import install.retrieval
import install.urlMapping

import os
import sys
import cPickle

class Builder(install.urlMapping.UrlMapping):
  def __init__(self, stamp = None):
    install.urlMapping.UrlMapping.__init__(self, stamp = stamp)
    self.retriever = install.retrieval.Retriever(stamp)
    return

  def getMakeModule(self, root, name = 'make'):
    import imp

    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()

  def loadLocalDict(self, root):
    '''Load any existing local RDict into the one we are currently using'''
    dictFilename = os.path.join(root, 'RDict.db')
    data         = None
    if os.path.exists(dictFilename):
      try:
        dbFile = file(dictFilename)
        data   = cPickle.load(dbFile)
        self.debugPrint('Loaded argument database from '+dictFilename, 2, 'install')
        keys   = self.argDB.keys()
        for k in filter(lambda k: not k in keys, data.keys()):
          if data[k].isValueSet():
            self.argDB.setType(k, data[k])
          self.debugPrint('Set key "'+str(k)+'" in argument database', 4, 'install')
        dbFile.close()
      except Exception, e:
        self.debugPrint('Problem loading dictionary from '+dictFilename+'\n--> '+str(e), 2, 'install')
        raise e
    else: keys = None
    return (data, keys)

  def unloadLocalDict(self, data, oldKeys):
    '''Remove keys from any existing local RDict from our dict'''
    if not data is None:
      for k in filter(lambda k: not k in oldKeys, data.keys()):
        if data[k].isValueSet():
          del self.argDB[k]
    return

  def purgeCheckpoint(self, proj):
    '''Remove all checkpoints from the local dictionaires'''
    import RDict
    import build.buildGraph

    oldDir = os.getcwd()
    self.debugPrint('Purging checkpoints from '+proj.getUrl()+' and dependencies', 3, 'install')
    for p in build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, proj):
      os.chdir(p.getRoot())
      d = RDict.RDict()
      if 'checkpoint' in d:
        self.debugPrint('Purging checkpoint in '+p.getUrl(), 4, 'install')
        del d['checkpoint']
        d.save(force = 1)
    os.chdir(oldDir)
    return

  def buildDependenceGraph(self, maker, seen = []):
    '''Retrieve all dependencies and construct the dependence graph'''
    for url in maker.executeTarget('getDependencies'):
      if url in seen: continue
      seen.append(url)
      self.debugPrint('Retrieving dependency '+url, 2, 'install')
      try:
        m = self.getMakeModule(self.retriever.retrieve(url)).PetscMake(sys.argv[1:], self.argDB)
      except ImportError:
        self.debugPrint('  No make module present in '+root, 2, 'install')
        continue
      self.buildDependenceGraph(m)
    self.debugPrint('Activating '+maker.project.getUrl(), 2, 'install')
    maker.mainBuild(['activate'])
    return maker.setupDependencies()

  def executeOverDependencies(self, proj, target):
    '''Execute a set of targets over the project dependencies'''
    import build.buildGraph

    self.debugPrint('Executing '+str(target)+' for '+proj.getUrl()+' and dependencies', 3, 'install')
    for p in build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, proj):
      try:
        maker = self.getMakeModule(p.getRoot()).PetscMake(sys.argv[1:], self.argDB)
      except ImportError:
        self.debugPrint('  No make module present in '+proj.getRoot(), 2, 'install')
        continue
      maker.mainBuild(target)
    return

  def build(self, root, target = ['default'], ignoreDependencies = 0):
    self.debugPrint('Building '+str(target)+' in '+root, 1, 'install')
    try:
      maker = self.getMakeModule(root).PetscMake(sys.argv[1:], self.argDB)
    except ImportError:
      self.debugPrint('  No make module present in '+root, 2, 'install')
      return
    self.dependenceGraph = self.buildDependenceGraph(maker)
    if not ignoreDependencies:
      self.executeOverDependencies(maker.project, ['configure'])
    self.debugPrint('Compiling in '+root, 2, 'install')
    root               = maker.getRoot()
    localDict, oldKeys = self.loadLocalDict(root)
    ret                = maker.mainBuild(target)
    self.unloadLocalDict(localDict, oldKeys)
    if not ignoreDependencies:
      # We must install project dependencies since the "install" target is purely local
      self.executeOverDependencies(maker.project, ['install'])
      self.purgeCheckpoint(maker.project)
    return ret
