import install.base
import install.retrieval

import sys

class Builder(install.base.Base):
  def __init__(self, argDB):
    install.base.Base.__init__(self, argDB)
    self.argDB     = argDB
    self.retriever = install.retrieval.Retriever(argDB)
    return

  def build(self, root, target = 'default'):
    self.debugPrint('Building in '+root, 1, 'install')
    try:
      maker = self.getMakeModule(root).PetscMake(sys.argv[1:])
    except ImportError:
      self.debugPrint('  No make module present in '+root, 2, 'install')
      return
    root  = maker.getRoot()
    for url in maker.executeTarget('getDependencies'):
      self.debugPrint('  Building dependency '+url, 2, 'install')
      self.build(self.retriever.retrieve(url), target = target)
    self.debugPrint('Compiling in '+root, 2, 'install')
    return maker.main(target)
