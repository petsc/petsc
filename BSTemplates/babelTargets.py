import babel
import BSTemplates.sidlDefaults

import os
import re

class Defaults:
  def __init__(self, usingSIDL):
    self.usingSIDL = usingSIDL
    self.IORRE     = re.compile(r'^(.*)_IOR$')
    self.implRE    = re.compile(r'^(.*)_Impl$')
    self.serverRE  = re.compile(r'^(.*)_(IOR|skel|impl)')
    return

  def getCompilerModule(self):
    return babel

  def getServerRootDir(self, lang, package = None, base = None):
    if base:
      path = base+'-'
    else:
      path = ''
    path += lang.lower()
    if package:
      path += '-'+package
    return path

  def getClientRootDir(self, lang):
    return lang.lower()

  def generatesAllStubs(self):
    '''Babel generates stubs for all interfaces and classes involved in a compile'''
    return 1

  def getImplRE(self):
    return self.implRE

  def isIOR(self, source):
    # Just need to exclude Scandal IOR, not pick out Babel's
    if self.IORRE.match(os.path.dirname(source)):
      return 0
    return 1

  def isClient(self, source, root):
    dir = os.path.dirname(source)
    # Anything in the root or below root/src
    pattern = '^'+os.path.join(root, 'src').replace('+', r'\+')+r'.*$'
    if dir == root or re.match(pattern, dir):
      return 1
    return 0

  def isServer(self, source, root):
    # The best we can do is exclude the IORs, skels and impls from Scandal, but internal clients are still there
    if self.serverRE.match(os.path.dirname(source)):
      return 0
    return 1

  def getTagger(self, type):
    if type == 'server':
      return BSTemplates.sidlDefaults.TagSIDL()
    elif type == 'client' or type == 'repository':
      return BSTemplates.sidlDefaults.TagAllSIDL()
    raise RuntimeError('Unknown build type: '+type)

  def setupIncludes(self, action):
    action.repositoryDirs.append(self.usingSIDL.repositoryDir)
    action.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return

  def removeNullIORExceptionStub(self):
    root = self.usingSIDL.getClientRootDir('C++')
    stub = os.path.join(root, 'src', 'SIDL', 'NullIORException')
    if os.path.exists(stub): os.system('rm -rf '+stub)
    return

  def getExtraClientTargets(self):
    import transform
    return [transform.SimpleFunction(self.removeNullIORExceptionStub)]
