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
    self.debugPrint('Building in '+root, 3, 'install')
    maker = self.getMakeModule(root).PetscMake(sys.argv[1:])
    for url in maker.executeTarget('getDependencies'):
      self.debugPrint('  Building dependency '+url, 3, 'install')
      self.build(self.retriever.retrieve(url))
    self.debugPrint('Compiling in '+maker.getRoot(), 3, 'install')
    return maker.main(target)
