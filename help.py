'''This module is meant to provide support for information and help systems based upon RDict.'''
import logger

class Info(logger.Logger):
  '''This basic class provides information independent of RDict'''
  def __init__(self, argDB = None):
    '''Creates a dictionary "sections" whose keys are section names, and values are a tuple of (ordinal, nameList)'''
    logger.Logger.__init__(self, None, argDB)
    self.sections = {}
    return

  def getTitle(self):
    return self._title

  def setTitle(self, title):
    self._title = str(title)
  title = property(getTitle, setTitle, None, 'Title of the Information Menu')

  def getDescription(self, section, name):
    return self._desc[(section, name)]

  def setDescription(self, section, name, desc):
    if not hasattr(self, '_desc'):
      self._desc = {}
    self._desc[(section, name)] = desc
    return

  def addArgument(self, section, name, desc):
    '''Add an argument with given name and string to an information section'''
    if not section in self.sections:
      self.sections[section] = (len(self.sections), [])
    if name in self.sections[section][1]:
      name += '@'+str(len(filter(lambda n: name == n.split('@')[0], self.sections[section][1]))+1)
    self.sections[section][1].append(name)
    self.setDescription(section, name, desc)
    return

  def printBanner(self, f):
    '''Print a banner for the information screen'''
    f.write(self.title+'\n')
    for i in range(max(map(len, self.title.split('\n')))): f.write('-')
    f.write('\n')
    return

  def getTextSizes(self):
    '''Returns the maximum name and description lengths'''
    nameLen = 1
    descLen = 1
    for section in self.sections:
      nameLen = max([nameLen, max(map(lambda n: len(n.split('@')[0]), self.sections[section][1]))+1])
      descLen = max([descLen, max(map(lambda name: len(self.getDescription(section, name)), self.sections[section][1]))+1])
    return (nameLen, descLen)

  def output(self, f = None):
    '''Print a help screen with all the argument information.'''
    if f is  None:
      import sys
      f = sys.stdout
    self.printBanner(f)
    (nameLen, descLen) = self.getTextSizes()
    format = '  %-'+str(nameLen)+'s: %s\n'
    items  = self.sections.items()
    items.sort(lambda a, b: a[1][0].__cmp__(b[1][0]))
    for section, names in items:
      f.write(section+':\n')
      for name in names[1]:
        f.write(format % (name.split('@')[0], self.getDescription(section, name)))
    return
  
class Help(Info):
  '''Help provides a simple help system for RDict'''
  def __init__(self, argDB):
    '''Creates a dictionary "sections" whose keys are section names, and values are a tuple of (ordinal, nameList). Also provide the RDict upon which this will be based.'''
    Info.__init__(self, argDB)
    self.title = 'Help'
    return

  def getDescription(self, section, name):
    return self.argDB.getType(self.getArgName(name)).help

  def setDescription(self, section, name, desc):
    return

  def getArgName(self, name):
    '''Return the RDict key corresponding to a more verbose help name. Right now, this means discard everything after "=".'''
    #return name.split('=')[0].strip('-')
    argName = name.split('=')[0]
    while argName[0] == '-': argName = argName[1:]
    return argName

  def addArgument(self, section, name, type, ignoreDuplicates = 0):
    '''Add an argument with given name and type to a help section. The type, which can also have an initializer and help string, will be put into RDict.'''
##  super(Info, self).addArgument(section, name, None)
    if section in self.sections:
      if name in self.sections[section][1]:
        if ignoreDuplicates:
          return
        raise RuntimeError('Duplicate configure option '+name+' in section '+section)
    else:
      self.sections[section] = (len(self.sections), [])
    self.sections[section][1].append(name)

    self.argDB.setType(self.getArgName(name), type, forceLocal = 1)
    return

  def output(self, f = None):
    '''Print a help screen with all the argument information.'''
    if f is  None:
      import sys
      f = sys.stdout
    self.printBanner(f)
    (nameLen, descLen) = self.getTextSizes()
    format    = '  -%-'+str(nameLen)+'s: %s\n'
    formatDef = '  -%-'+str(nameLen)+'s: %-'+str(descLen)+'s  current: %s\n'
    items = self.sections.items()
    items.sort(lambda a, b: a[1][0].__cmp__(b[1][0]))
    for section, names in items:
      f.write(section+':\n')
      for name in names[1]:
        argName = self.getArgName(name)
        type    = self.argDB.getType(argName)
        if argName in self.argDB:
          f.write(formatDef % (name, type.help, str(self.argDB.getType(argName))))
        else:
          f.write(format % (name, type.help))
    return
