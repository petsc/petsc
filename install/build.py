import logging
import install.retrieval

import imp
import sys

class Builder(logging.Logger):
  def __init__(self, argDB):
    logging.Logger.__init__(self, argDB)
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

  def build(self, root):
    self.debugPrint('Building in '+root, 3, 'install')
    maker = self.getMakeModule(root).PetscMake(sys.argv[1:])
    for dep in maker.executeTarget('getDependencies'):
      self.debugPrint('  Building dependency '+dep, 3, 'install')
      depRoot = self.retriever.retrieve(dep);
      self.build(depRoot)
    self.debugPrint('Compiling '+sys.modules[maker.__module__].__file__, 3, 'install')
    return maker.main()
