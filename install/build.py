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
    root  = maker.getRoot()
    for url in maker.executeTarget('getDependencies'):
      self.debugPrint('  Building dependency '+url, 3, 'install')
      self.build(self.retriever.retrieve(url), target = target)
    # Check for remote compile
    try:
      import SIDL.Loader
      import SIDLLanguage.Parser
    except ImportError:
      # We cannot execute splicing correctly, so we just revert the files
      self.debugPrint('Reverting in '+root, 3, 'install')
      output = self.executeShellCommand('cd '+root+'; bk sfiles -lg | bk unedit -')
      output = self.executeShellCommand('cd '+root+'; bk -r co -q')
    self.debugPrint('Compiling in '+root, 3, 'install')
    return maker.main(target)
