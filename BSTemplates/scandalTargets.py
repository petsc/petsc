import scandal
import BSTemplates.sidlDefaults

import os
import re

class Defaults:
  def __init__(self, defaults):
    self.defaults = defaults
    # Get runtime AST
    defaults.usingSIDL.repositoryDirs.append(self.defaults.usingSIDL.getRootDir())
    return

  def getCompilerModule(self):
    return scandal

  def getImplRE(self):
    return re.compile(r'^(.*)_impl$')

  def getTagger(self, type):
    return BSTemplates.sidlDefaults.TagSIDL()

  def setupIncludes(self, action):
    action.repositoryDirs.extend(self.defaults.usingSIDL.repositoryDirs)
    return
