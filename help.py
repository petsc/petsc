'''This module is meant to provide support for help systems based upon RDict.'''
class Help:
  '''Help provides a simple help system for RDict'''
  def __init__(self, argDB):
    '''Creates a dictionary "sections" whose keys are section names, and values are a tuple of (ordinal, nameList). Also provide the RDict upon which this will be based.'''
    self.argDB    = argDB
    self.sections = {}
    self.title    = 'Help'
    return

  #def setTitle(self, title): self.__title = title
  #def getTitle(self, title): return self.__title
  #def delTitle(self, title): del self.__title
  #title = property(title, getTitle, setTitle, delTitle, 'Title of the Help Menu')

  def getArgName(self, name):
    '''Return the RDict key corresponding to a more verbose help name. Right now, this means discard everything after "=".'''
    #return name.split('=')[0].strip('-')
    argName = name.split('=')[0]
    while argName[0] == '-': argName = argName[1:]
    return argName

  def addArgument(self, section, name, type):
    '''Add an argument with given name and type to a help section. The type, which can also have an initializer and help string, will be put into RDict.'''
    if section in self.sections:
      if name in self.sections[section][1]:
        raise RuntimeError('Duplicate configure option '+name+' in section '+section)
    else:
      self.sections[section] = (len(self.sections), [])
    self.sections[section][1].append(name)
    self.argDB.setType(self.getArgName(name), type, forceLocal = 1)
    return

  def printBanner(self):
    '''Print a banner for the help screen'''
    import sys

    print self.title
    for i in range(len(self.title)): sys.stdout.write('-')
    print
    return

  def getTextSizes(self):
    '''Returns the maximum name and description lengths'''
    nameLen = 1
    descLen = 1
    for section in self.sections:
      nameLen = max([nameLen, max(map(len, self.sections[section][1]))+1])
      descLen = max([descLen, max(map(lambda a: len(self.argDB.getType(self.getArgName(a)).help), self.sections[section][1]))+1])
    return (nameLen, descLen)

  def output(self):
    '''Print a help screen with all the argument information.'''
    self.printBanner()
    (nameLen, descLen) = self.getTextSizes()
    format    = '  -%-'+str(nameLen)+'s: %s'
    formatDef = '  -%-'+str(nameLen)+'s: %-'+str(descLen)+'s  current: %s'
    items = self.sections.items()
    items.sort(lambda a, b: a[1][0].__cmp__(b[1][0]))
    for item in items:
      print item[0]+':'
      for name in item[1][1]:
        argName = self.getArgName(name)
        type    = self.argDB.getType(argName)
        if argName in self.argDB:
          print formatDef % (name, type.help, str(self.argDB[argName]))
        else:
          print format % (name, type.help)
    return
