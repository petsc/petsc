#!/usr/bin/env python
import PETSc
import PETSc.Configure

import os
import re

## SECTION: Initialization
# Set default language to C

## SECTION: Installation

class Configure:
  def __init__(self):
    self.defines = {}
    self.subst   = {}
    self.help    = {}
    # Interaction with Autoconf
    self.m4         = '/usr/bin/m4'
    self.acMacroDir = '/usr/share/autoconf'
    self.acLocalDir = 'config'
    self.acReload   = '--reload'
    self.acMsgFD    = '2'
    self.acCCFD     = '/dev/null'
    # Interaction with the shell
    self.shell      = '/bin/sh'
    return

  def addDefine(self, name, value, comment = ''):
    self.defines[name] = value
    if comment: self.help[name] = comment
    return

  def addSubstitution(self, name, value):
    self.subst[name] = value
    return

  def getDefaultMacros(self):
    return 'AC_INIT_BINSH\nAC_CONFIG_AUX_DIR(config)\n'

  def getMacroVersion(self, macro):
    m = re.search(r'^dnl\s+Version:\s+(?P<version>\d+\.\d+)', macro, re.M)
    if m:
      return m.group('version')
    else:
      return ''

  def getMacroVariables(self, macro):
    varRE = re.compile(r'^dnl\s+Variable:\s+(?P<variable>\w+)', re.M)
    return varRE.findall(macro)

  def replaceDefaultDescriptors(self, codeStr):
    newCode = re.sub('AC_FD_MSG', self.acMsgFD, codeStr)
    newCode = re.sub('AC_FD_CC',  self.acCCFD,  newCode)
    return newCode

  def findUndefinedMacros(self, codeStr):
    matches = re.findall(r'AC_\w+', codeStr)
    if len(matches):
      msg = 'Undefined macros:\n'
      for m in matches: msg += '  '+m+'\n'
      raise RuntimeError(msg)
    return

  def macroToShell(self, macro):
    self.getMacroVersion(macro)
    command = self.m4
    if self.acMacroDir:
      command += ' -I'+self.acMacroDir
    if self.acLocalDir:
      command += ' -I'+self.acLocalDir+' -DAC_LOCALDIR='+self.acLocalDir
    if self.acReload and os.path.exists(os.path.join(self.acMacroDir, 'autoconf.m4f')):
      command += ' '+self.acReload+' autoconf.m4f'
    else:
      command += ' autoconf.m4'
    (input, output) = os.popen2(command)
    input.write(self.getDefaultMacros()+macro)
    input.close()
    out = output.read()
    shellCode = self.replaceDefaultDescriptors(out)
    self.findUndefinedMacros(shellCode)
    output.close()
    return (shellCode, self.getMacroVariables(macro))

  def getDefaultVariables(self):
    return 'host=NONE\nnonopt=NONE\nCONFIG_SHELL='+self.shell+'\n'

  def parseShellOutput(self, output):
    results = {}
    varRE   = re.compile(r'(?P<name>\w+)\s+=\s+(?P<value>.*)')
    for line in output.split('\n'):
      m = varRE.match(line)
      if m and m.group('value'):
        results[m.group('name')] = m.group('value')
    return results

  def executeShellCode(self, code):
    codeStr  = self.getDefaultVariables()
    codeStr += code[0]
    for var in code[1]:
      codeStr += 'echo "'+var+' = " ${'+var+'}\n'
    (input, output) = os.popen4(self.shell)
    input.write(codeStr)
    input.close()
    results = output.read()
    output.close()
    return self.parseShellOutput(results)

  def configure(self):
    pass

class Framework(Configure):
  def __init__(self):
    Configure.__init__(self)
    self.children   = []
    self.substRE    = re.compile(r'@(?P<name>[^@]+)@')
    self.substFiles = {}
    self.header     = 'matt_config.h'
    return

  def addSubstitutionFile(self, inName, outName = ''):
    if outName:
      if inName == outName:
        raise RuntimeError('Input and output substitution files identical: '+inName)
    else:
      if inName[-3:] == '.in':
        root  = inName[-3:]
      else:
        root  = inName
      inName  = root+'.in'
      outName = root
    if not os.path.exists(inName):
      raise RuntimeError('Nonexistent substitution file: '+inName)
    self.substFiles[inName] = outName
    return

  def getPrefix(self, child):
    mod = child.__class__.__module__
    if not mod == '__main__':
      prefix = mod.replace('.', '_')
    else:
      prefix = ''
    return prefix

  def getHeaderPrefix(self, child):
    if hasattr(child, 'headerPrefix'):
      prefix = child.headerPrefix
    else:
      prefix = self.getPrefix(child)
    return prefix

  def getSubstitutionPrefix(self, child):
    if hasattr(child, 'substPrefix'):
      prefix = child.substPrefix
    else:
      prefix = self.getPrefix(child)
    return prefix

  def substituteName(self, match, prefix = None):
    name = match.group('name')
    if self.subst.has_key(name):
      return self.subst[name]
    else:
      for child in self.children:
        if not hasattr(child, 'subst') or not isinstance(child.defines, dict): continue
        if prefix is None: prefix = self.getSubstitutionPrefix(child)
        if prefix:         prefix = prefix+'_'
        if name.startswith(prefix) and child.subst.has_key(name.replace(prefix, '', 1)):
          return child.subst[name.replace(prefix, '', 1)]
    return ''

  def substituteFile(self, inName, outName):
    inFile  = file(inName)
    outFile = file(outName, 'w')
    for line in inFile.xreadlines():
      outFile.write(self.substRE.sub(self.substituteName, line))
    outFile.close()
    inFile.close()

  def substitute(self):
    for pair in self.substFiles.items():
      self.substituteFile(pair[0], pair[1])
    return

  def outputDefine(self, f, name, value = None, comment = ''):
    name = name.upper()
    if comment:
      for line in comment.split('\n'):
        if line: f.write('/* '+line+' */\n')
    f.write('#ifndef '+name+'\n')
    if value:
      f.write('#define '+name+' '+value+'\n')
    else:
      f.write('/* #undef '+name+' */\n')
    f.write('#endif\n')

  def outputDefines(self, f, child, prefix = None):
    '''If the child contains a dictionary named "defines", the entries are output as defines in the config header.
    The prefix to each define is calculated as follows:
    - If the prefix argument is given, this is used, otherwise
    - If the child contains "headerPrefix", this is used, otherwise
    - If the module containing the child class is not "__main__", this is used, otherwise
    - No prefix is used
    If the child contains a dictinary name "help", then a help string will be added before the define
    '''
    if not hasattr(child, 'defines') or not isinstance(child.defines, dict): return
    if hasattr(child, 'help') and isinstance(child.help, dict):
      help = child.help
    else:
      help = {}
    if prefix is None: prefix = self.getHeaderPrefix(child)
    if prefix:         prefix = prefix+'_'
    for pair in child.defines.items():
      if help.has_key(pair[0]):
        self.outputDefine(f, prefix+pair[0], pair[1], help[pair[0]])
      else:
        self.outputDefine(f, prefix+pair[0], pair[1])
    return

  def outputHeader(self, name):
    f = file(name, 'w')
    self.outputDefines(f, self)
    for child in self.children:
      self.outputDefines(f, child)
    f.close()
    return

  def configure(self):
    for child in self.children:
      child.configure()
    self.substitute()
    self.outputHeader(self.header)
    return

if __name__ == '__main__':
  print 'Matt kicks ass'
  framework = Framework()
  framework.children.append(PETSc.Configure.Configure(framework))
  framework.addSubstitutionFile('matt')
  framework.configure()
  print 'Finished'
