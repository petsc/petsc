import install.base
import install.retrieval

import os
import sys

class Builder(install.base.Base):
  def __init__(self, argDB):
    install.base.Base.__init__(self, argDB)
    self.retriever = install.retrieval.Retriever(argDB)
    return

  def build(self, root, target = ['activate', 'default'], setupTarget = None, ignoreDependencies = 0):
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
        keys   = self.argDB.keys()
        for k in filter(lambda k: not k in keys, data.keys()):
          if data[k].isValueSet():
            self.argDB.setType(k, data[k])
        dbFile.close()
        loadedRDict = 1
        self.debugPrint('Loaded dictionary from '+dictFilename, 2, 'install')
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
    # Save source database (since atexit() functions might not be called before another build)
    maker.saveSourceDB()
    return ret
