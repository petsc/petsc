import base

import os

class UsingC (base.Base):
  def __init__(self, sourceDB, project):
    base.Base.__init__(self)
    self.sourceDB = sourceDB
    self.project  = project
    return
