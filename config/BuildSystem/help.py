'''This module is meant to provide support for information and help systems based upon RDict.'''
from __future__ import print_function
from __future__ import absolute_import
import logger
import sys

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
      name += '@'+str(len([n for n in self.sections[section][1] if name == n.split('@')[0]])+1)
    self.sections[section][1].append(name)
    self.setDescription(section, name, desc)
    return

  def printBanner(self, f):
    '''Print a banner for the information screen'''
    title   = self.title
    divider = '-' * logger.get_global_divider_length()
    f.write('{}\n{}\n'.format(title, divider))
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
      f = sys.stdout
    self.printBanner(f)
    (nameLen, descLen) = self.getTextSizes()
    format = '  %-'+str(nameLen)+'s: %s\n'
    items  = sorted(self.sections.items(), key=lambda a: a[1][0])
    for section, names in items:
      f.write(section+':\n')
      for name in names[1]:
        f.write(format % (name.split('@')[0], self.getDescription(section, name)))
    return

# I don't know how to not have this stupid global variable
_outputDownloadDone = 0

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

  def addArgument(self, section, name, argType, ignoreDuplicates = 0):
    '''Add an argument with given name and type to a help section. The type, which can also have an initializer and help string, will be put into RDict.'''
##  super(Info, self).addArgument(section, name, None)
    if section in self.sections:
      if name in self.sections[section][1]:
        if ignoreDuplicates:
          return
        raise RuntimeError('Duplicate configure option '+name+' in section '+section)
    else:
      self.sections[section] = (len(self.sections), [])
    if not argType.deprecated:
      self.sections[section][1].append(name)

    self.argDB.setType(self.getArgName(name), argType, forceLocal = 1)
    return

  def addDownload(self,name,dlist):
    if not hasattr(self.argDB,'dlist'):
      self.argDB.dlist = {}
    else:
      self.argDB.dlist[name] = dlist

  def output(self, f = None, sections = None):
    '''Print a help screen with all the argument information.'''
    def output_items(section_title, items):
      f.write(
        logger.build_multiline_message(
          '***** {} *****'.format(section_title), '', divider_char='-'
        ) + '\n'
      )

      for section, names in items:
        if sections and not section.casefold() in sections:
          continue

        f.write(section + ':\n')
        for name in names[1]:
          arg_name = self.getArgName(name)
          arg_type = self.argDB.getType(arg_name)
          arg_help = arg_type.help
          if arg_name in self.argDB:
            f.write('  -{}\n       {}  current: {}\n'.format(name, arg_help, arg_type))
          else:
            f.write('  -{}\n       {}\n'.format(name, arg_help))
      return

    if f is None:
      f = sys.stdout
    if sections:
      sections = {s.casefold() for s in sections}

    packages = []
    modules  = []
    for item in self.sections.items():
      # Packages all have -- for whatever reason -- an uppercase section name, so use this
      # to distinguish them. This is a vile hack.
      if item[0].isupper():
        packages.append(item)
      else:
        modules.append(item)

    self.printBanner(f)
    # sort the primary modules by their ordering, this happens to be nice and logical
    output_items('CORE OPTIONS', sorted(modules, key=lambda a: a[1][0]))
    # self.printBanner() will automatically append a '----' so we don't have to print a
    # divider above, but we do have to here
    f.write('-' * logger.get_global_divider_length() + '\n')
    # sort packages by name
    output_items('PACKAGE OPTIONS', sorted(packages, key=lambda a: a[0]))
    return


  def outputDownload(self):
    ''' Looks for downloaded packages in --with-packages-download-dir
        For any it finds it updates the --download-xxx= argument to point to this local copy
        If it does not find some needed packages then prints the packages that need to be downloaded and exits'''
    import nargs
    import os
    global _outputDownloadDone
    if _outputDownloadDone: return
    _outputDownloadDone = 1
    pkgdir = os.path.abspath(os.path.expanduser(nargs.Arg.findArgument('with-packages-download-dir', self.clArgs)))
    missing = 0
    for i in self.argDB.dlist.keys():
      if not nargs.Arg.findArgument('download-'+i, self.clArgs) == None and not nargs.Arg.findArgument('download-'+i, self.clArgs) == '0':
        dlist = self.argDB.dlist[i]
        found = 0
        for k in range(0,len(dlist)):
          fd = os.path.join(pkgdir,(os.path.basename(dlist[k])))
          if fd.endswith('.git'):
            fd = fd[:-4]
          if os.path.isdir(fd) or os.path.isfile(fd):
            found = 1
            break
        if not found:
          missing = 1
    if missing:
      print('Download the following packages to '+pkgdir+' \n')
    for i in self.argDB.dlist.keys():
      if not nargs.Arg.findArgument('download-'+i, self.clArgs) == None and not nargs.Arg.findArgument('download-'+i, self.clArgs) == '0':
        dlist = self.argDB.dlist[i]
        found = 0
        for k in range(0,len(dlist)):
          fd = os.path.join(pkgdir,(os.path.basename(dlist[k])))
          if fd.endswith('.git'):
            fd = fd[:-4]
          if os.path.isdir(fd) or os.path.isfile(fd):
            found = 1
            for k in range(0,len(self.clArgs)):
              if self.clArgs[k].startswith('--download-'+i):
                self.clArgs[k] = 'download-'+i+'='+fd
                self.argDB.insertArgs([self.clArgs[k]])
            break
        if not found:
          print(i + ' ' + str(self.argDB.dlist[i]).replace("git://","git clone "))
    if missing:
      print('\nThen run the script again\n')
      sys.exit(10)

