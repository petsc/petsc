import install.base
import install.retrieval

import imp
import sys

class Builder(install.base.Base):
  def __init__(self, argDB):
    install.base.Base.__init__(self, argDB)
    self.argDB     = argDB
    self.retriever = install.retrieval.Retriever(argDB)
    return

  def getMakeModule(self, root):
    name = 'make'
    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()

  def build(self, root, target = 'default'):
    self.debugPrint('Building in '+root, 3, 'install')
    mod   = self.getMakeModule(root)
    maker = mod.PetscMake(sys.argv[1:])
    for url in maker.executeTarget('getDependencies'):
      self.debugPrint('  Building dependency '+url, 3, 'install')
      if not self.getInstalledProject(url) is None: continue
      self.build(self.retriever.retrieve(url))
    self.debugPrint('Compiling '+mod.__file__, 3, 'install')
    return maker.main(target)
