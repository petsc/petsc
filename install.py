#!/usr/bin/env python

class Install (Transform):
  def __init__(self, rootDir, installDir, tags = [], sources = None, doLink = 0):
    Transform.__init__(self, sources)
    self.tags = tags
    if self.tags and not type(self.tags) == types.ListType:
      self.tags = [self.tags]
    self.tags = map(lambda tag: 'install '+tag, self.tags)
    self.rootDir    = os.path.abspath(rootDir)
    self.installDir = os.path.abspath(installDir)
    self.doLink     = doLink
    self.copyFunc   = os.link
    self.linkFunc   = os.symlink
    self.products   = []

  def getDest(self, source):
    if not os.path.commonprefix([source, self.rootDir]) == self.rootDir:
      raise RuntimeError(source+' is not under '+self.rootDir)
    return os.path.join(self.installDir, source[len(rootDir):])

  def fileExecute(self, source):
    dest = self.getDest(source)
    if self.doLink:
      self.linkFunc(source, dest)
    else:
      self.copyFunc(source, dest)

  def setExecute(self, set):
    if self.tags and set.tag in self.tags:
      Transform.setExecute(self, set)
    elif set.tag and set.tag[:6] == 'install':
      Transform.setExecute(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
