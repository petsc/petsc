import base

import os

class UsingC (base.Base):
  def __init__(self, sourceDB, project, usingSIDL):
    base.Base.__init__(self)
    self.sourceDB  = sourceDB
    self.project   = project
    self.usingSIDL = usingSIDL
    self.language  = 'C'
    return
