import install.base
import install.retrieval

import sys

class Builder(install.base.Base):
  def __init__(self, argDB):
    install.base.Base.__init__(self, argDB)
    self.retriever = install.retrieval.Retriever(argDB)
    return

  def build(self, root, target = 'default', setupTarget = ''):
    self.debugPrint('Building '+str(target)+' in '+root, 1, 'install')
    try:
      maker = self.getMakeModule(root).PetscMake(sys.argv[1:], self.argDB)
    except ImportError:
      self.debugPrint('  No make module present in '+root, 2, 'install')
      return
    root  = maker.getRoot()
    for url in maker.executeTarget('getDependencies'):
      self.debugPrint('  Building dependency '+url, 2, 'install')
      self.build(self.retriever.retrieve(url), target, setupTarget)
    self.debugPrint('Compiling in '+root, 2, 'install')
    if not isinstance(setupTarget, list): setupTarget = [setupTarget]
    for t in setupTarget:
      maker.executeTarget(t)
    ret = maker.main(target)
    # Python filters nonexistent paths, so we must add it again
    #   TODO: Should use getClientDir() from maker here
    import os
    path = os.path.join(root, 'python-scandal')
    if not path in sys.path:
      sys.path.append(path)
    return ret
