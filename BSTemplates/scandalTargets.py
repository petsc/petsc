import scandal
import BSTemplates.sidlDefaults

import os
import re

class Defaults:
  def __init__(self, usingSIDL):
    self.usingSIDL = usingSIDL
    self.IORRE     = re.compile(r'^(.*)_IOR$')
    self.implRE    = re.compile(r'^(.*)_impl$')
    self.serverRE  = re.compile(r'^(.*)_(Skel|Impl)')
    # Get runtime AST
    self.usingSIDL.repositoryDirs.append(self.usingSIDL.getRuntimeProject().getRoot())
    return

  def getCompilerModule(self):
    return scandal

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
    return lang.lower()+'-scandal'

  def generatesAllStubs(self):
    '''Scandal only generates stubs for the packages being compiled'''
    return 0

  def getImplRE(self):
    return self.implRE

  def isIOR(self, source):
    if self.IORRE.match(os.path.dirname(source)):
      return 1
    return 0

  def isClient(self, source, root):
    dir = os.path.dirname(source)
    # Anything not in the root or below root/src
    pattern = '^'+os.path.join(root, 'src').replace('+', r'\+')+r'.*$'
    if dir == root or re.match(pattern, dir):
      return 0
    return 1

  def isServer(self, source, root):
    dir = os.path.dirname(source)
    # The best we can do is exclude the IORS, skels and impls from Babel, but internal clients are still there
    if dir == root or self.serverRE.match(os.path.dirname(source)):
      return 0
    return 1

  def getTagger(self, type):
    return BSTemplates.sidlDefaults.TagSIDL()

  def setupIncludes(self, action):
    print 'dir:  '+self.usingSIDL.repositoryDir
    action.repositoryDirs.append(self.usingSIDL.repositoryDir)
    print 'dirs: '+str(self.usingSIDL.repositoryDirs)
    action.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return

  def getExtraClientTargets(self):
    return []
