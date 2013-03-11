import os

class Project(object):
  '''This class represents a SIDL project, and is the only BuildSystem class allowed in an RDict'''
  def __init__(self, url, root = None, web = None):
    import os
    if root is None: root = os.path.abspath(os.getcwd())
    # Read-only variables
    self.url          = url
    self.root         = root
    self.webdirectory = 'petsc@terra.mcs.anl.gov://mcs/www-unix/sidl/'
    self.web          = web
    # Updated variables
    self.paths           = {}
    self.packages        = []
    self.implementations = {}
    return

  def __str__(self):
    return self.url

  def __hash__(self):
    return self.url.__hash__()

  def __lt__(self, other):
    return self.url.__lt__(other.getUrl())

  def __le__(self, other):
    return self.url.__le__(other.getUrl())

  def __eq__(self, other):
    return self.url.__eq__(other.getUrl())

  def __ne__(self, other):
    return self.url.__ne__(other.getUrl())

  def __gt__(self, other):
    return self.url.__gt__(other.getUrl())

  def __ge__(self, other):
    return self.url.__ge__(other.getUrl())

  def getName(self):
    import urlparse
    # Fix parsing for nonstandard schemes
    urlparse.uses_netloc.extend(['hg', 'ssh'])
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(self.getUrl())
    return path.lower().replace('/', '-')

  def getUrl(self):
    '''Return the project URL, e.g. http://petsc.cs.iit.edu/petsc/petsc-dev'''
    return self.url

  def setWebDirectory(self,webdirectory):
    '''Sets the project Website directory path petsc@terra.mcs.anl.gov://mcs/www-unix/sidl/'''
    self.webdirectory = webdirectory

  def getWebDirectory(self):
    '''Returns the project Website directory path petsc@terra.mcs.anl.gov://mcs/www-unix/sidl/'''
    return self.webdirectory

  def getRoot(self):
    '''Return the root directory of the local installation'''
    return self.root

  def appendPath(self, lang, dir):
    '''Append a directory to the list of paths which must be given to this language in order for this project to function'''
    import os

    if not lang in self.paths:
      self.paths[lang] = []
    d = os.path.abspath(dir)
    if os.path.exists(d) and not d in self.paths[lang]:
      self.paths[lang].append(d)
    return self.paths[lang]

  def getPath(self, lang):
    '''Return the list of paths which must be given to this language in order for this project to function'''
    if lang in self.paths:
      return self.paths[lang]
    return []

  def appendPackages(self, packages):
    '''Appends package names'''
    self.packages += packages
    return

  def getPackages(self):
    '''Gets package names'''
    return self.packages

  def addImplementation(self, cls, library, lang):
    '''Specify the location and language for an implementation of the given class'''
    if not cls in self.implementations:
      self.implementations[cls] = []
    self.implementations[cls].append((library, lang))
    return

  def getImplementations(self):
    if hasattr(self, 'implementations'):
      return self.implementations
    return {}
