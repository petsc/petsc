class SIDLConstants:
  def getLanguages():
    return ['C', 'C++', 'Python', 'F77', 'F90', 'Java', 'Mathematica']
  getLanguages = staticmethod(getLanguages)

  def checkLanguage(language):
    if not language in SIDLConstants.getLanguages():
      raise ValueError('Invalid SIDL language: '+language)
  checkLanguage = staticmethod(checkLanguage)

class SIDLLanguageList (list):
  def __setitem__(self, key, value):
    SIDLConstants.checkLanguage(value)
    self.data[key] = value

class SIDLPackages:
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    self.defaults = defaults

  def getPackages(self):
    return self.defaults.getPackages()

  def checkPackage(self, package):
    if not package in self.getPackages():
      if package in SIDLConstants.getLanguages(): return
      if package == 'executable': return
      raise KeyError('Invalid SIDL package: '+package)

class SIDLPackageList (list, SIDLPackages):
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    list.__init__(self)
    SIDLPackages.__init__(self, defaults)

  def __setitem__(self, key, value):
    self.checkPackage(value)
    self.data[key] = value

class SIDLLanguageDict (dict, SIDLPackages):
  '''We allow only SIDL languages as keys'''
  def __init__(self, defaults):
    dict.__init__(self)
    SIDLPackages.__init__(self, defaults)

  def __getitem__(self, key):
    SIDLConstants.checkLanguage(key)
    if not self.has_key(key): dict.__setitem__(self, key, '')
    return dict.__getitem__(self, key)

  def __setitem__(self, key, value):
    SIDLConstants.checkLanguage(key)
    dict.__setitem__(self, key, value)

class SIDLPackageDict (dict, SIDLPackages):
  '''We now allow packages or languages as keys, and the values must be lists'''
  def __init__(self, defaults):
    dict.__init__(self)
    SIDLPackages.__init__(self, defaults)

  def __getitem__(self, key):
    self.checkPackage(key)
    if not self.has_key(key): dict.__setitem__(self, key, [])
    return dict.__getitem__(self, key)

  def __setitem__(self, key, value):
    self.checkPackage(key)
    if not isinstance(value, list): raise ValueError('Entries must be lists')
    dict.__setitem__(self, key, value)
