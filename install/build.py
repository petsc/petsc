import install.retrieval
import install.urlMapping

import os
import sys

class Builder(install.urlMapping.UrlMapping):
  def __init__(self):
    install.urlMapping.UrlMapping.__init__(self)
    self.retriever = install.retrieval.Retriever()
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
        import cPickle
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
    return data

  def unloadLocalDict(self, data):
    '''Remove keys from any existing local RDict from our dict'''
    if not data is None:
      for k in filter(lambda k: not k in keys, data.keys()):
        if data[k].isValueSet():
          del self.argDB[k]
    return

  def buildDependenceGraph(self, maker):
    '''Retrieve all dependencies and construct the dependence graph'''
    seen = []
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
      m.setupDependencies()
    maker.setupDependencies()
    return

  def executeOverDependencies(self, proj, target):
    '''Execute a set of targets over the project dependencies'''
    import build.buildGraph

    self.debugPrint('Executing '+str(target)+' for '+proj.getUrl()+' and dependencies', 3, 'install')
    for p in build.buildGraph.BuildGraph.depthFirstVisit(self.argDB['projectDependenceGraph'], proj):
      try:
        maker = self.getMakeModule(p.getRoot()).PetscMake(sys.argv[1:], self.argDB)
      except ImportError:
        self.debugPrint('  No make module present in '+root, 2, 'install')
        continue
      maker.main(target)
    return

  def build(self, root, target = ['default'], ignoreDependencies = 0):
    self.debugPrint('Building '+str(target)+' in '+root, 1, 'install')
    try:
      maker = self.getMakeModule(root).PetscMake(sys.argv[1:], self.argDB)
    except ImportError:
      self.debugPrint('  No make module present in '+root, 2, 'install')
      return
    if not ignoreDependencies:
      self.executeOverDependencies(maker.project, ['activate', 'configure'])
    self.buildDependenceGraph(maker)
    self.debugPrint('Compiling in '+root, 2, 'install')
    root      = maker.getRoot()
    localDict = self.loadLocalDict(root)
    ret       = maker.main(target)
    self.unloadLocalDict(localDict)
    if not ignoreDependencies:
      # We must install project dependencies since the "install" target is purely local
      self.executeOverDependencies(maker.project, ['install'])
    return ret

  def build_old(self, root, target = ['default'], setupTarget = None, ignoreDependencies = 0):
    self.debugPrint('Building '+str(target)+' in '+root, 1, 'install')
    try:
      maker = self.getMakeModule(root).PetscMake(sys.argv[1:], self.argDB)
    except ImportError:
      self.debugPrint('  No make module present in '+root, 2, 'install')
      return
    root = maker.getRoot()
    if not ignoreDependencies:
      for url in maker.executeTarget('getDependencies'):
        self.debugPrint('  Retrieving and activating dependency '+url, 2, 'install')
        self.build(self.retriever.retrieve(url), target = ['activate', 'configure'])
    # Load any existing local RDict
    dictFilename = os.path.join(root, 'RDict.db')
    loadedRDict  = 0
    if os.path.exists(dictFilename):
      try:
        import cPickle
        dbFile = file(dictFilename)
        data   = cPickle.load(dbFile)
        self.debugPrint('Loaded argument database from '+dictFilename, 2, 'install')
        keys   = self.argDB.keys()
        for k in filter(lambda k: not k in keys, data.keys()):
          if data[k].isValueSet():
            self.argDB.setType(k, data[k])
          self.debugPrint('Set key "'+str(k)+'" in argument database', 4, 'install')
        dbFile.close()
        loadedRDict = 1
      except Exception, e:
        self.debugPrint('Problem loading dictionary from '+dictFilename+'\n--> '+str(e), 2, 'install')
        raise e
    self.debugPrint('Compiling in '+root, 2, 'install')
    if setupTarget is None:                 setupTarget = []
    elif not isinstance(setupTarget, list): setupTarget = [setupTarget]
    for t in setupTarget:
      maker.executeTarget(t)
    ret = maker.main(target)
    if loadedRDict:
      for k in filter(lambda k: not k in keys, data.keys()):
        if data[k].isValueSet():
          del self.argDB[k]
    if not ignoreDependencies:
      for url in maker.executeTarget('getDependencies'):
        self.debugPrint('  Installing dependency '+url, 2, 'install')
        self.build(self.getInstallRoot(url), target = ['install'])
      maker.executeTarget('install')
    return ret
