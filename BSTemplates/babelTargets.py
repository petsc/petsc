import babel
import BSTemplates.sidlDefaults

import re

class Defaults:
  def __init__(self, defaults):
    self.defaults = defaults
    return

  def getCompilerModule(self):
    return babel

  def getImplRE(self):
    return re.compile(r'^(.*)_Impl$')

  def getTagger(self, type):
    if type == 'server':
      return BSTemplates.sidlDefaults.TagSIDL()
    elif type == 'client' or type == 'repository':
      return BSTemplates.sidlDefaults.TagAllSIDL()
    raise RuntimeError('Unknown build type: '+type)

  def setupIncludes(self, action):
    action.repositoryDirs.append(self.defaults.usingSIDL.repositoryDir)
    action.repositoryDirs.extend(self.defaults.usingSIDL.repositoryDirs)
    return
