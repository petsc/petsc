#!/usr/bin/env python3
#
#  Processes PETSc's (or SLEPc's) header and source files to determine
#  the PETSc enums, structs, functions, and classes
#
#  Calling sequence:
#      getAPI(directory, pkgname = 'petsc', verbose = False)
#
#  Notes:
#    const char *fill_array_of_strings[]              + fills up an array of strings; the array already exists in the calling routine
#    const char * const set_with_array_of_strings     + passes in an array of strings to be used by subroutine
#    const char * const * returns_an_array_of_strings + returns to the user an array of strings
##
import os
import re
import sys
import pickle
import pathlib
import subprocess
from subprocess import check_output

def mansecpath(mansec):
  '''Given a manual section, returns the path where it is located (it differs in some SLEPc classes)'''
  return os.path.join('sys','classes',mansec) if mansec in ['bv','ds','fn','rg','st'] else mansec

def verbosePrint(verbose, text):
  '''Prints the text if run with verbose option'''
  if verbose: print(text)

classes = {}
funcs = {}           # standalone functions like PetscInitialize()
allfuncs = set()     # both class and standalone functions, used to prevent duplicates
enums = {}
senums = {}          # like enums except strings instead of integer values for enumvalue
typedefs = {}
functiontypedefs = {}  # for example SNESFunctionFn
aliases = {}
structs = {}
includefiles = {}
mansecs = {}         # mansec[mansecname] = set(all submansecnames in mansecname)
submansecs = set()
manualpages = {}
defines = {}         # function-like macros
senumvalues = {}

regcomment   = re.compile(r'/\* [-A-Za-z _(),<>|^\*/0-9.:=\[\]\.;]* \*/')
regcomment2  = re.compile(r'// [-A-Za-z _(),<>|^\*/0-9.:=\[\]\.;]*')
regblank     = re.compile(r' [ ]*')

def displayIncludeMansec(obj):
    return '  include file: ' + str(obj.includefile)+'\n  manual page section (mansec): ' + str(obj.mansec) + '\n'

def displayFile(obj):
    return str(obj.dir) + '/' + str(obj.file) + '\n'

class Define:
    '''Represents a function-like macro'''
    def __init__(self, name, mansec, includefile, args):
        self.name        = name
        self.mansec      = mansec
        self.includefile = includefile
        self.args        = args

    def __str__(self):
        mstr = str(self.name) + '\n'
        mstr += displayIncludeMansec(self)
        mstr += '  Arguments\n'
        for i in self.args:
            mstr += '    ' + i + '\n'
        return mstr

class ManualPage:
    '''Represents a manual page'''
    def __init__(self, name, mansec, text, seealsos):
        self.name        = name
        self.mansec      = mansec
        self.text        = text
        self.seealsos    = seealsos

    def __str__(self):
        mstr = str(self.name) + '\n'
        mstr += str(self.seealsos)
        return mstr

class Typedef:
    '''Represents typedef oldtype newtype'''
    def __init__(self, name, mansec, includefile, value, *args, **kwargs):
        self.name        = name
        self.mansec      = mansec
        self.includefile = includefile
        self.value       = value
        self.opaque      = False

    def __str__(self):
        mstr = str(self.name) + ' ' + str(self.value)+'\n'
        mstr += displayIncludeMansec(self)
        return mstr

class Function:
    '''Represents a function in a class or standalone'''
    def __init__(self, name, *args, **kwargs):
        self.name        = name
        self.mansec      = None
        self.file        = None
        self.includefile = None
        self.dir         = None
        self.opaque      = False
        self.opaquestub  = False # only Fortran module interface is automatic, C stub is custom
        self.penss       = False # Function is labeled with PeNS or PeNSS
        self.arguments   = []

    def __str__(self):
        mstr = '  ' + str(self.name) + '()\n'
        mstr += '    source code location: ' + displayFile(self)
        if self.opaque:   mstr += '    opaque binding\n'
        elif self.opaque: mstr += '    opaque stub\n'
        if self.arguments:
          mstr += '    arguments:\n'
          for i in self.arguments:
            mstr += '  ' + str(i)
        return mstr

class FunctionTypedef:
    '''Represents a function typedef such as SNESFunctionFn'''
    def __init__(self, name, *args, **kwargs):
        self.name        = name
        self.mansec      = None
        self.file        = None
        self.includefile = None
        self.dir         = None
        self.arguments   = []

    def __str__(self):
        mstr = '  ' + str(self.name) + '()\n'
        mstr += displayIncludeMansec(self)
        if self.arguments:
          mstr += '    arguments:\n'
          for i in self.arguments:
            mstr += '  ' + str(i)
        return mstr

class Argument:
    '''Represents an argument in a Function'''
    def __init__(self, name = None, typename = None, stars = 0, array = False, const = False, *args, **kwargs):
        self.name       = name
        self.typename   = typename
        self.stars      = stars
        self.array      = array
        self.optional   = False
        self.const      = const
        self.isfunction = False
        self.fnctnptr   = None       # contains the signature if argument is a function pointer
        #  PETSc returns strings in two ways either
        #     with a pointer to an array: char *[]
        #     or by copying the string into a given array with a given length: char [], size_t len
        self.stringlen   = False  # if true the argument is the length of the previous argument which is a character string
        self.char_type   = None  # 'string' if it is a string, 'single' if it is a single character

    def __str__(self):
        mstr = '    ' + str(self.typename) + ' '
        stars = self.stars
        while stars:
          mstr += '*'
          stars = stars - 1
        mstr += str(self.name)
        if self.array: mstr += '[]'
        if self.optional: mstr += ' optional'
        mstr += '\n'
        return mstr

class Struct:
    '''Represents a C struct'''
    def __init__(self, name, mansec, includefile, opaque, records, *args, **kwargs):
        self.name        = name
        self.mansec      = mansec
        self.includefile = includefile
        self.opaque      = opaque
        self.records     = records

    def __str__(self):
        mstr = str(self.name) + '\n'
        mstr += displayIncludeMansec(self)
        if self.opaque:  mstr += '  opaque\n'
        mstr += '  Records:\n'
        for i in self.records:
          mstr += str(i)
        return mstr

class Record:
    '''Represents an entry in a struct'''
    def __init__(self, rawrecord, *args, **kwargs):
        self.name = None
        # name is currently unused and type contains the type followed by all the names with that type: e.g. PetscInt i,j,k
        self.type = rawrecord

    def __str__(self):
        mstr = '    ' + str(self.type)+'\n'
        return mstr

class Enum:
    '''Represents a C enum'''
    def __init__(self, name, mansec, includefile, values, *args, **kwargs):
        self.name        = name
        self.mansec      = mansec
        self.includefile = includefile
        self.values      = values

    def __str__(self):
        mstr = str(self.name) + '\n'
        mstr += displayIncludeMansec(self)
        mstr += '  values:\n'
        for i in self.values:
          mstr += '    ' + str(i) + '\n'
        return mstr

class Senum:
    '''Represents a PETSc string enum; a name and a set of string values'''
    def __init__(self, name, mansec, includefile, values, *args, **kwargs):
        self.name        = name
        self.mansec      = mansec
        self.includefile = includefile
        self.values      = values

    def __str__(self):
        mstr = str(self.name) + '\n'
        mstr += displayIncludeMansec(self)
        mstr += '  values:\n'
        for i in self.values.keys():
          mstr += '    ' + i + ' ' + self.values[i] + '\n'
        return mstr

class IncludeFile:
    '''Represents an include (interesting) file found and what interesting files it includes'''
    def __init__(self, mansec, includefile, included, *args, **kwargs):
        self.mansec      = mansec
        self.includefile = includefile
        self.included    = included # include files it includes

    def __str__(self):
        mstr = str(self.includefile) + '\n'
        mstr += '  manual page section (mansec): ' + str(self.mansec) + '\n'
        mstr += '  included files:\n'
        for i in self.included:
          mstr += '    ' + str(i) + '\n'
        return mstr

class Class:
    '''Represents a class (PetscObject and other _n_ opaque objects)'''
    def __init__(self, name, *args, **kwargs):
        self.name        = name
        self.mansec      = None
        self.includefile = None
        self.petscobject = True
        self.functions   = {}

    def __str__(self):
        mstr = str(self.name) + '\n'
        mstr += displayIncludeMansec(self)
        mstr += '  Subclass of PetscObject <' + str(self.petscobject) + '>\n\n'
        mstr += '  Methods:\n'
        for i in self.functions.keys():
          mstr += '  ' + str(self.functions[i]) + '\n'
        return mstr

def findmansec(line,mansec,submansec):
  '''Finds mansec and submansec in line from include/petsc*.h'''
  if line.find(' MANSEC') > -1:
    mansec = re.sub(r'[ ]*/\* [ ]*MANSEC[ ]*=[ ]*','',line).strip('\n').strip('*/').strip()
    if mansec == line[0].strip('\n'):
      mansec = re.sub('MANSEC[ ]*=[ ]*','',line.strip('\n').strip())
    mansec = mansec.lower()
  if line.find('SUBMANSEC') > -1:
    submansec = re.sub(r'[ ]*/\* [ ]*SUBMANSEC[ ]*=[ ]*','',line).strip('\n').strip('*/').strip()
    if submansec == line[0].strip('\n'):
      submansec = re.sub('SUBMANSEC[ ]*=[ ]*','',line.strip('\n').strip())
    submansec = submansec.lower()
    if not mansec: mansec = submansec
    submansecs.add(submansec)
    if not mansec in mansecs: mansecs[mansec] = set()
    mansecs[mansec].add(submansec)
  return mansec,submansec

def getIncludeFiles(filename,pkgname):
  import re

  file = os.path.basename(filename)
  mansec = None
  reginclude  = re.compile(r'^#include <[A-Za-z_0-9]*.h')
  f = open(filename)
  line = f.readline()
  included = []
  while line:
    mansec,submansec = findmansec(line,mansec,None)
    fl = reginclude.search(line)
    if fl and not line.find('deprecated') > -1:
      line = regcomment.sub("",line)
      line = regcomment2.sub("",line)
      line = line.replace('#include <','').replace('>','').strip()
      if not line == file and os.path.isfile(os.path.join('include',line)) or (pkgname == 'slepc' and line.startswith('petsc')):
        included.append(line)
    line = f.readline()
  includefiles[file] = IncludeFile(mansec,file,included)
  f.close()

badSeealso = False

def processManualPage(name, lines):
  '''Processes the manual page associated with a name'''
  global badSeealso
  cnt = 0
  lastline  = -1
  firstline = -1
  for i in lines:
    for flag in ['E', 'J', 'S', 'M', '@']:
      if lastline == -1 and i.find(flag + '*/') > -1:
        lastline = cnt + 1
        if lastline > 3:
           #print('It is unlikely ' +  name + ' has a manual page')
           return
      elif firstline == -1 and i.find('/*' + flag) > -1:
        firstline = cnt
        break
    cnt += 1
  if firstline == -1:
    #print('Unable to find /*' + 'X' + ' for ' + name)
    return
  if lastline == -1:
    #print('Unable to find ' + 'X' + '*/ for ' + name)
    return
  lines = lines[lastline:firstline]
  lines.reverse()
  text = lines
  top = lines[0].strip(' ')
  loc = top.find(name)
  if not loc == 0:
    print('First line of manual page does not start with ' + name)
    badSeealso = True
    return
  cnt = 0
  for i in lines:
    if i.startswith('.seealso:'): break
    cnt += 1
  if cnt == len(lines):
    print('.seealso missing in manual page for ' + name)
    badSeealso = True
    return
  seealso = lines[cnt][10:].strip().strip('\n')
  for i in range(cnt+1, len(lines)):
    seealso += ' ' + lines[i].strip().strip('\n')
  #print(seealso)
  old_seealso = seealso
  seealso = ''
  in_brack = False
  for i in old_seealso:
    if i == '[':
      if in_brack:
        print('See also for ' + name + ' has a double [')
        badSeealso = True
        return
      in_brack = True
    elif i == ']':
      if not in_brack:
        print('See also for ' + name + ' has a closing ] without and opening [')
        badSeealso = True
        return
      in_brack = False
    if i == ' ' and in_brack:
      seealso += '+'
    else:
      seealso += i
  # ensure there is a ", " after each token
  c_seealso = seealso.split(',')
  for i in range(1, len(c_seealso)):
    if not c_seealso[i].startswith(' '):
      print('See also for ' + name + ' is messed up before the ' + str(i-1) + ' token:' + c_seealso[i])
      badSeealso = True
      return
  s_seealso = seealso.split(' ')
  for i in range(0, len(s_seealso)-1):
    if not s_seealso[i].endswith(','):
      print('See also for ' + name + ' is messed up after the ' + str(i) + ' token:' + s_seealso[i])
      badSeealso = True
      return
  seealso = seealso.split(', ')
  # ensure each seealso is [zzz](xxxx_yyy) or `xxx[]`
  for i in seealso:
    if i.startswith('['):
      if not i.endswith(')'):
        print('See also for ' + name + ': ' + i + ' is misformed. Does not end with expected )')
        badSeealso = True
        return
    elif i.startswith('`'):
      if not i.endswith('`'):
        print('See also for ' + name + ': ' + i + ' is misformed. Does not end with expected `')
        badSeealso = True
        return
      if i.find(' ') > -1:
        print('See also for ' + name + ': ' + i + ' is misformed. Seems to contain a blank space')
        badSeealso = True
        return
      if i.endswith(')') and not i.endswith('()'):
        print('See also for ' + name + ': ' + i + ' is misformed. Seems to be missing a (')
        badSeealso = True
        return
    else:
      print('See also for ' + name + ': ' + i + ' is misformed')
      badSeealso = True
      return
  seealsos = []
  for i in seealso:
    if i.startswith('`'):
      see = i[1:-1]
      if see in seealsos:
        print('See also for ' + name + ': ' + see + ' is duplicate')
        badSeealso = True
        return
      seealsos.append(see)
  manualpages[name] = ManualPage(name, 'unknown', text, seealsos)
  #print(seealso)

def getEnums(filename):
  import re
  regtypedef  = re.compile(r'typedef [ ]*enum')
  reg         = re.compile(r'}')
  regname     = re.compile(r'}[ A-Za-z0-9]*')

  file = os.path.basename(filename).replace('types.h','.h')
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  submansec = None
  mansec = None
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regtypedef.search(line)
    if fl:
      struct = line
      while line:
        fl = reg.search(line)
        if fl:
          struct = regcomment.sub("",struct)
          struct = struct.replace("\\","")
          struct = struct.replace("\n","")
          struct = struct.replace(";","")
          struct = struct.replace("typedef enum","")
          struct = regblank.sub(" ",struct)

          name = regname.search(struct)
          name = name.group(0)
          name = name.replace("} ","")

          values = struct[struct.find("{") + 1:struct.find("}")]
          values = values.split(',')

          ivalues = []
          for i in values:
            if i:
              if i[0] == " ": i = i[1:]
              ivalues.append(i)

          enums[name] = Enum(name,mansec,file,ivalues)
          processManualPage(name, lines)
          lines = []
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
    lines.insert(0,line)
  f.close()

def getSenums(filename):
  import re
  regdefine   = re.compile(r'typedef const char \*[A-Za-z]*;')
  file = os.path.basename(filename).replace('types.h','.h')
  mansec = None
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    mansec,submansec = findmansec(line,mansec,None)
    fl = regdefine.search(line)
    if fl:
      senum = fl.group(0)[20:-1]
      line = regblank.sub(" ",f.readline().strip())
      d = {}
      while line:
        values = line.split(" ")
        d[values[1]] = values[2]
        line = regblank.sub(" ",f.readline().strip())
      senums[senum]             = Senum(senum,mansec,file,d)
      processManualPage(senum, lines)
      lines = []
    line = f.readline()
    lines.insert(0,line)
  f.close()

def getDefines(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regdefine   = re.compile(r'#define [A-Za-z0-9]*\([A-Za-z0-9_, ]*\) ')
  submansec = None
  mansec = None
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regdefine.search(line)
    if fl:
      name = fl.group(0).split('(')[0][8:]
      args = fl.group(0).split('(')[1][:-2]
      args = args.split(', ')
      defines[name] = Define(name,mansec,file,args)
      processManualPage(name, lines)
      lines = []
    line = f.readline()
    lines.insert(0,line)
  f.close()

def getTypedefs(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regdefine   = re.compile(r'typedef [A-Za-z0-9_]* [ ]*[A-Za-z0-9_]*;')
  submansec = None
  mansec = None
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regdefine.search(line)
    if fl:
      typedef = fl.group(0).split()[2][0:-1];
      if typedef in typedefs:
        typedefs[typedef].opaque = True # found more than once so cannot generate Fortran code from it
        pass
      else:
        typedefs[typedef] = Typedef(typedef,mansec,file,fl.group(0).split()[1])
      processManualPage(typedef, lines)
      lines = []
    line = f.readline()
    lines.insert(0,line)
  f.close()

def getFunctionTypedefs(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regdefine   = re.compile(r'PETSC_EXTERN_TYPEDEF typedef [a-zA-Z ]* ([a-zA-Z0-9]*\([ \[\]*()A-Za-z0-9_,]*\));')
  submansec = None
  mansec = None
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regdefine.search(line)
    if fl:
      fun             = parseFunction(regdefine.sub(r'\1',line))
      fun.mansec      = mansec
      fun.submansec   = submansec
      fun.includefile = os.path.basename(filename)
      functiontypedefs[fun.name] = fun
      processManualPage(fun.name, lines)
      lines = []
    line = f.readline()
    lines.insert(0,line)
  f.close()

def getStructs(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regtypedef  = re.compile(r'^typedef [ ]*struct {')
  reg         = re.compile(r'}')
  regname     = re.compile(r'}[ A-Za-z]*')
  submansec = None
  mansec = None
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regtypedef.search(line)
    opaque = False
    if fl:
      struct = line
      while line:
        fl = reg.search(line)
        if fl:
          struct = regcomment.sub("",struct)
          struct = regcomment2.sub("",struct)
          struct = struct.replace("\\","")
          struct = struct.replace("\n","")
          struct = struct.replace("typedef struct {","")
          struct = regblank.sub(" ",struct)
          struct = struct.replace("; ",";")

          name = regname.search(struct)
          name = name.group(0)
          name = name.replace("} ","")

          values = struct[struct.find("{") + 1:struct.find(";}")]
          if values.find('#') > -1 or values.find('*') > -1 or values.find('][') > -1: opaque = True
          if not values.find('#') == -1: opaque = True
          values = values.split(";")
          ivalues = []
          for i in values:
            ivalues.append(Record(i.strip()))
          structs[name] = Struct(name,mansec,file,opaque,ivalues)
          processManualPage(name, lines)
          lines = []
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
    lines.insert(0,line)
  f.close()

def getClasses(filename):
  import re
  regclass    = re.compile(r'typedef struct _[np]_[A-Za-z_]*[ ]*\*')
  regnclass    = re.compile(r'typedef struct _n_[A-Za-z_]*[ ]*\*')
  regsemi     = re.compile(r';')
  submansec = None
  mansec = None
  file = os.path.basename(filename).replace('types.h','.h')
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regclass.search(line)
    gl = regnclass.search(line)
    if fl:
      struct = line
      struct = regclass.sub("",struct)
      struct = regcomment.sub("",struct)
      struct = regblank.sub("",struct)
      struct = regsemi.sub("",struct)
      struct = struct.replace("\n","")
      classes[struct] = Class(struct)
      if not submansec: raise RuntimeError('No SUBMANSEC in file ' + filename)
      classes[struct].mansec = mansec
      classes[struct].includefile = file
      if gl: classes[struct].petscobject = False
      processManualPage(struct, lines)
      lines = []
    line = f.readline()
    lines.insert(0,line)
  f.close()

def findlmansec(dir):  # could use dir to determine mansec
    '''Finds mansec and submansec from a makefile'''
    file = os.path.join(dir,'makefile')
    mansec = None
    submansec = None
    with open(file) as mklines:
      submansecl = [line for line in mklines if line.find('BFORTSUBMANSEC') > -1]
      if submansecl:
        submansec = re.sub('BFORTSUBMANSEC[ ]*=[ ]*','',submansecl[0]).strip('\n').strip().lower()
    if not submansec:
      with open(file) as mklines:
        submansecl = [line for line in mklines if (line.find('SUBMANSEC') > -1 and line.find('BFORT') == -1)]
        if submansecl:
          submansec = re.sub('SUBMANSEC[ ]*=[ ]*','',submansecl[0]).strip('\n').strip().lower()
    with open(file) as mklines:
      mansecl = [line for line in mklines if line.startswith('MANSEC')]
      if mansecl:
        mansec = re.sub('MANSEC[ ]*=[ ]*','',mansecl[0]).strip('\n').strip().lower()
    if not submansec: submansec = mansec
    return mansec,submansec

def getpossiblefunctions(pkgname):
   '''Gets a list of all the functions in the include/ directory that may be used in the binding for other languages'''
   try:
     output = check_output('grep -F -e "' + pkgname.upper() + '_EXTERN PetscErrorCode" -e "static inline PetscErrorCode" include/*.h', shell=True).decode('utf-8')
   except subprocess.CalledProcessError as e:
     raise RuntimeError('Unable to find possible functions in the include files')
   funs = output.replace('' + pkgname.upper() + '_EXTERN','').replace('PetscErrorCode','').replace('static inline','')
   functiontoinclude = {}
   for i in funs.split('\n'):
     file = i[i.find('/') + 1:i.find('.') + 2]
     f = i[i.find(': ') + 2:i.find('(')].strip()
     functiontoinclude[f] = file.replace('types','')

   try:
     output = check_output('grep "' + pkgname.upper() + '_EXTERN [a-zA-Z]* *[a-zA-Z]*;" include/*.h', shell=True).decode('utf-8')
   except subprocess.CalledProcessError as e:
     raise RuntimeError('Unable to find possible functions in the include files')
   funs = output.replace('' + pkgname.upper() + '_EXTERN','')
   for i in funs.split('\n'):
     if not i: continue
     i = i.replace(';','').split()
     file = i[0][i[0].find('/') + 1:i[0].find('.') + 2]
     functiontoinclude[i[2]] = file.replace('types','')
   return functiontoinclude

def parseFunction(line):
  '''Parses a function declaration such as SNESFunctionFn(SNES snes, Vec u, Vec F, void *ctx)'''
  import re
  regfun      = re.compile(r'^[static inline]*PetscErrorCode ')
  regfunvoid  = re.compile(r'^[static inline]*void ')
  regarg      = re.compile(r'\([A-Za-z0-9*_\[\]]*[,\) ]')
  regerror    = re.compile(r'PetscErrorCode')
  reg         = re.compile(r' ([*])*[a-zA-Z0-9_]*([\[\]]*)')
  regname     = re.compile(r' [*]*([a-zA-Z0-9_]*)[\[\]]*')

  # for finding xxx (*yyy)([const] zzz, ...)
  regfncntnptrname  = re.compile(r'[A-Za-z0-9]* \(\*([A-Za-z0-9]*)\)\([_a-zA-Z0-9, *\[\]]*\)')
  regfncntnptr      = re.compile(r'[A-Za-z0-9]* \(\*[A-Za-z0-9]*\)\([_a-zA-Z0-9, *\[\]]*\)')
  regfncntnptrtype  = re.compile(r'([A-Za-z0-9]*) \(\*[A-Za-z0-9]*\)\([_a-zA-Z0-9, *\[\]]*\)')

  # for rejecting (**func), (*indices)[3], (*monitor[X]), and xxx (*)(yyy)
  regfncntnptrptr   = re.compile(r'\([*]*\*\*[ A-Za-z0-9]*\)')
  regfncntnptrarray = re.compile(r'\(\*[A-Za-z0-9]*\)\[[A-Za-z0-9_]*\]')
  regfncntnptrarrays = re.compile(r'\(\*[A-Za-z0-9]*\[[A-Za-z0-9]*\]\)')
  regfncntnptrnoname = re.compile(r'\(\*\)')

  rejects     = ['PetscErrorCode','...','<','(*)','(**)','off_t','MPI_Datatype','va_list','PetscStack','Ceed']
  #
  # search through list BACKWARDS to get the longest match
  #
  classlist = classes.keys()
  classlist = sorted(classlist)
  classlist.reverse()
  line = line.replace('PETSC_UNUSED ','')
  line = line.replace('PETSC_RESTRICT ','')
  line = line.strip()
  line = regfun.sub("",line)
  line = regfunvoid.sub("",line)
  line = regcomment.sub("",line)
  line = line.strip()
  name = line[:line.find("(")]

  # find arguments that return a function pointer (**xxx)
  fnctnptrptrs = regfncntnptrptr.findall(line)
  # find arguments such as PetscInt (*indices[XXX])
  fnctnptrarrays = regfncntnptrarrays.findall(line)
  # find arguments that are unnamed function pointers (*)
  fnctnptrnoname = regfncntnptrnoname.findall(line)
  # find all function pointers in the arguments xxx (*yyy)(zzz) and convert them to external yyy
  fnctnptrs     = regfncntnptr.findall(line)
  fnctnptrnames = regfncntnptrname.findall(line)
  for i in range(0,len(fnctnptrs)):
    line = line.replace(fnctnptrs[i], 'external ' + fnctnptrnames[i])

  fl = regarg.search(line)
  if not fl:
    raise RuntimeError('This cannot occur since the regarg was already found')
  fun = FunctionTypedef(name)
  arg = fl.group(0)
  arg = arg[1:-1]
  reject = 0
  for i in rejects:
    if line.find(i) > -1:
      reject = 1
  if  not reject:
    args = line[line.find("(") + 1:line.find(")")]
    if args != 'void':
      args = args.split(",")
      argnames = []
      for i in args:
        arg = Argument()
        if i.count("const "): arg.const = True
        i = i.replace("const ","")
        i = i.strip()
        if re.match(r'[a-zA-Z 0-9_]*\[[0-9]*\]$',i.replace('*','')):
          arg.array = True
          i = i[:i.find('[')]
        if i.find('*') > -1: arg.stars = 1
        if i.find('**') > -1: arg.stars = 2
        argname = re.findall(r' [*]*([a-zA-Z0-9_]*)[\[\]]*',i)
        if argname and argname[0]:
          arg.name = argname[0]
          if arg.name.lower() in argnames:
            arg.name = 'M_' + arg.name
          argnames.append(arg.name.lower())
        else:
          arg.name   = 'noname'
        i =  regblank.sub('',reg.sub(r'\1\2 ',i).strip()).replace('*','').replace('[]','')
        arg.typename = i
        # fix input character arrays that are written as *variable name
        if arg.typename == 'char' and not arg.array and arg.stars == 1:
          arg.array = 1
          arg.stars = 0
        if arg.typename == 'char' and not arg.array and arg.stars == 0:
          arg.char_type = 'single'
        if arg.typename.endswith('Fn'):
          arg.isfunction = True
        if arg.typename == 'external':
          arg.fnctnptr   = fnctnptrs[fnctnptrnames.index(arg.name)]
        if fun.arguments and not fun.arguments[-1].const and fun.arguments[-1].typename == 'char' and arg.typename == 'size_t':
          arg.stringlen = True
        fun.arguments.append(arg)
  return fun

def getFunctions(mansec, functiontoinclude, filename):
  '''Appends the functions found in filename to their associated class classes[i], or funcs[] if they are classless'''
  import re
  regfun      = re.compile(r'^[static inline]*PetscErrorCode ')
  regarg      = re.compile(r'\([A-Za-z0-9*_\[\]]*[,\) ]')
  regerror    = re.compile(r'PetscErrorCode')
  reg         = re.compile(r' ([*])*[a-zA-Z0-9_]*([\[\]]*)')
  regname     = re.compile(r' [*]*([a-zA-Z0-9_]*)[\[\]]*')

  # for finding xxx (*yyy)([const] zzz, ...)
  regfncntnptrname  = re.compile(r'[A-Za-z0-9]* \(\*([A-Za-z0-9]*)\)\([_a-zA-Z0-9, *\[\]]*\)')
  regfncntnptr      = re.compile(r'[A-Za-z0-9]* \(\*[A-Za-z0-9]*\)\([_a-zA-Z0-9, *\[\]]*\)')
  regfncntnptrtype  = re.compile(r'([A-Za-z0-9]*) \(\*[A-Za-z0-9]*\)\([_a-zA-Z0-9, *\[\]]*\)')

  # for rejecting (**func), (*indices)[3], (*monitor[X]), and xxx (*)(yyy)
  regfncntnptrptr   = re.compile(r'\([*]*\*\*[ A-Za-z0-9]*\)')
  regfncntnptrarray = re.compile(r'\(\*[A-Za-z0-9]*\)\[[A-Za-z0-9_]*\]')
  regfncntnptrarrays = re.compile(r'\(\*[A-Za-z0-9]*\[[A-Za-z0-9]*\]\)')
  regfncntnptrnoname = re.compile(r'\(\*\)')

  rejects     = ['...','<','(*)','(**)','off_t','MPI_Datatype','va_list','PetscStack','Ceed']
  #
  # search through list BACKWARDS to get the longest match
  #
  classlist = classes.keys()
  classlist = sorted(classlist)
  classlist.reverse()
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    fl = regfun.search(line)
    if fl:
      opaque = False
      opaquestub = False
      penss = False
      if  line[0:line.find('(')].find('_') > -1:
        line = f.readline()
        lines.insert(0,line)
        continue
      line = line.replace('PETSC_UNUSED ','')
      line = line.replace('PETSC_RESTRICT ','')
      line = line.strip()
      if line.endswith(' PeNS'):
        opaque = True
        penss  = True
        line = line[0:-5]
      if line.endswith(' PeNSS'):
        opaquestub = True
        penss      = True
        line = line[0:-6]
      if line.endswith(';'):
        line = f.readline()
        lines.insert(0,line)
        continue
      if not line.endswith(')'):
        line = f.readline()
        lines.insert(0,line)
        continue
      line = regfun.sub("",line)
      line = regcomment.sub("",line)
      line = line.replace("\n","")
      line = line.strip()
      name = line[:line.find("(")]
      if not name in functiontoinclude or name in allfuncs:
        line = f.readline()
        lines.insert(0,line)
        continue

      # find arguments that return a function pointer (**xxx)
      fnctnptrptrs = regfncntnptrptr.findall(line)
      if fnctnptrptrs:
        opaque = True
        #print('Opaque due to (**xxx) argument ' + line)

      # find arguments such as PetscInt (*indices)[3])
      fnctnptrarrays = regfncntnptrarray.findall(line)
      if fnctnptrarrays:
        opaque = True
        #print('Opaque due to (*xxx][n] argument ' + line)

      # find arguments such as PetscInt (*indices[XXX])
      fnctnptrarrays = regfncntnptrarrays.findall(line)
      if fnctnptrarrays:
        opaque = True
        #print('Opaque due to (*xxx[yyy]) argument ' + line)

      # find arguments that are unnamed function pointers (*)
      fnctnptrnoname = regfncntnptrnoname.findall(line)
      if fnctnptrnoname:
        opaque = True
        #print('Opaque due to (*) argument ' + line)

      # find all function pointers in the arguments xxx (*yyy)(zzz) and convert them to external yyy
      fnctnptrs     = regfncntnptr.findall(line)
      fnctnptrnames = regfncntnptrname.findall(line)
      #if len(fnctnptrs): print(line)
      for i in range(0,len(fnctnptrs)):
        line = line.replace(fnctnptrs[i], 'external ' + fnctnptrnames[i])
      #if len(fnctnptrs): print(line)

      fl = regarg.search(line)
      if fl:
        fun = Function(name)
        fun.opaque = opaque
        fun.opaquestub = opaquestub
        fun.penss = penss
        fun.file = os.path.basename(filename)
        fun.mansec = mansec
        fun.dir = os.path.dirname(filename)

        arg = fl.group(0)
        arg = arg[1:-1]
        reject = 0
        for i in rejects:
          if line.find(i) > -1:
            reject = 1
        if  not reject:
          fun.includefile = functiontoinclude[name]
          args = line[line.find("(") + 1:line.find(")")]
          if args != 'void':
            for i in ['FILE','hid_t','MPI_File','MPI_Offset','MPI_Info','PETSC_UINTPTR_T','LinkMode']:
              if args.find(i) > -1:
                fun.opaque = True
            args = args.split(",")
            argnames = []
            for i in args:
              arg = Argument()
              if i.find('**') > -1 and not i.strip().startswith('void'): fun.opaque = True
              if i.find('unsigned ') > -1: fun.opaque = True
              if i.count('const ') > 1: fun.opaque = True
              if i.count("const "): arg.const = True
              i = i.replace("const ","")
              if i.find('PeOp ') > -1: arg.optional = True
              i = i.replace('PeOp ','')
              i = i.strip()
              if re.match(r'[a-zA-Z 0-9_]*\[[0-9]*\]$',i.replace('*','')):
                arg.array = True
                i = i[:i.find('[')]
              if i.find('*') > -1: arg.stars = 1
              if i.find('**') > -1: arg.stars = 2
              argname = re.findall(r' [*]*([a-zA-Z0-9_]*)[\[\]]*',i)
              if argname and argname[0]:
                arg.name = argname[0]
                if arg.name.lower() in argnames:
                  arg.name = 'M_' + arg.name
                argnames.append(arg.name.lower())
              else:
                arg.name   = 'noname'
                fun.opaque = True
              i =  regblank.sub('',reg.sub(r'\1\2 ',i).strip()).replace('*','').replace('[]','')
              arg.typename = i
              # fix input character arrays that are written as *variable name
              if arg.typename == 'char' and not arg.array and arg.stars == 1:
                arg.array = 1
                arg.stars = 0
              if arg.typename == 'char' and not arg.array and arg.stars == 0:
                arg.char_type = 'single'
              #if arg.typename == 'char' and arg.array and arg.stars:
              #  arg.const = False
              if arg.typename.endswith('Fn'):
                arg.isfunction = True
                fun.opaquestub = True
              if arg.typename == 'external':
                fun.opaquestub = True
                arg.fnctnptr   = fnctnptrs[fnctnptrnames.index(arg.name)]
                arg.fun        = parseFunction(re.sub(r'\(\*([A-Za-z0-9]*)\)',r'\1',arg.fnctnptr))
                arg.name       = arg.fun.name
              if arg.typename.count('_') and not arg.typename in ['MPI_Comm', 'size_t']:
                fun.opaque = True
              if fun.arguments and not fun.arguments[-1].const and fun.arguments[-1].typename == 'char' and arg.typename == 'size_t':
                arg.stringlen = True
              fun.arguments.append(arg)

          #print('Opaqueness of function ' + fun.name + ' ' + str(fun.opaque) + ' ' + str(fun.opaquestub))
          # add function to appropriate class
          allfuncs.add(name)
          notfound = True
          for i in classlist:
            if name.lower().startswith(i.lower()):
              classes[i].functions[name] = fun
              notfound = False
              break
          if notfound:
            funcs[name] = fun
          processManualPage(name, lines)
          lines = []

    line = f.readline()
    lines.insert(0,line)
  f.close()

def getSenumValueManualPage(mansec, filename):
  '''Finds manual pages for SenumValues such as MATSEQ
     These are standalone and may not have code associated with them
  '''
  import re
  regdefine   = re.compile(r' [A-Za-z][A-Za-z0-9]* ')
  f = open(filename)
  lines = []
  line = f.readline()
  lines.insert(0,line)
  while line:
    if line.find('/*MC') > -1:
      line = f.readline()
      lines.insert(0,line)
      fl = regdefine.search(line)
      if fl:
        senumvalue = fl.group(0)[1:-1]
        if senumvalue in senumvalues:
          line = f.readline()
          lines.insert(0,line)
          while line:
            line = f.readline()
            lines.insert(0,line)
            if line.find('M*/') > -1:
              processManualPage(senumvalue, lines)
              lines = []
              line = None
    line = f.readline()
    lines.insert(0,line)
  f.close()

ForbiddenDirectories = ['tests', 'tutorials', 'doc', 'output', 'ftn-custom', 'ftn-auto', 'ftn-mod', 'binding', 'binding', 'config', 'lib', '.git', 'share', 'systems']

def getAPI(directory,pkgname = 'petsc',verbose = False):
  global typedefs
  args = [os.path.join('include',i) for i in os.listdir(os.path.join(directory,'include')) if i.endswith('.h') and not i.endswith('deprecated.h')]
  for i in args:
    getIncludeFiles(i,pkgname)
  verbosePrint(verbose, '# PETSc include files')
  for i in includefiles.keys():
    verbosePrint(verbose, includefiles[i])

  for i in args:
    getEnums(i)
  verbosePrint(verbose, '# PETSc integer represented enum types')
  for i in enums.keys():
    verbosePrint(verbose, enums[i])

  for i in args:
    getDefines(i)
  verbosePrint(verbose, 'Defines ---------------------------------------------')
  for i in defines.keys():
    verbosePrint(verbose, defines[i])

  for i in args:
    getSenums(i)
  verbosePrint(verbose, '# PETSc string represented enum types')
  for i in senums.keys():
    verbosePrint(verbose, senums[i])

  # make a list of all senum values to use to search for manual pages
  for i in senums.keys():
    for value in senums[i].values:
      senumvalues[value] = senums[i]

  for i in args:
    getStructs(i)
  verbosePrint(verbose, '# PETSc structs')
  for i in structs.keys():
    verbosePrint(verbose, structs[i])

  for i in args:
    getTypedefs(i)
  cp = {}
  for i in typedefs.keys():
    if typedefs[i].name: cp[i] = typedefs[i] # delete ones marked as having multiple definitions
  typedefs = cp
  verbosePrint(verbose, '# PETSc typedefs')
  for i in typedefs.keys():
    verbosePrint(verbose, typedefs[i])

  for i in args:
    getFunctionTypedefs(i)

  for i in args:
    getClasses(i)

  functiontoinclude = getpossiblefunctions(pkgname)
  for dirpath, dirnames, filenames in os.walk(os.path.join(directory,'src'),topdown=True):
    dirpath = dirpath.replace(directory + '/','')
    dirnames[:] = [d for d in dirnames if d not in ForbiddenDirectories]
    if not os.path.isfile(os.path.join(dirpath,'makefile')): continue
    mansec, submansec = findlmansec(dirpath)
    for i in os.listdir(dirpath):
      if i.startswith('.'): continue
      if i.endswith('.c') or i.endswith('.cxx') or i.endswith('.cu'):
        getFunctions(mansec, functiontoinclude, os.path.join(dirpath,i))
        getSenumValueManualPage(mansec, os.path.join(dirpath,i))
  for i in args:
    mansec = None
    with open(i) as fd:
      lines = fd.read().split('\n')
      for line in lines:
        if line.find(' MANSEC') > -1:
          mansec = re.sub(r'[ ]*/\* [ ]*MANSEC[ ]*=[ ]*','',line).strip('\n').strip('*/').strip()
    if not mansec:
      with open(i) as fd:
        lines = fd.read().split('\n')
        for line in lines:
          if line.find(' SUBMANSEC') > -1:
            mansec = re.sub(r'[ ]*/\* [ ]*SUBMANSEC[ ]*=[ ]*','',line).strip('\n').strip('*/').strip()
    if not mansec: raise RuntimeError(i + ' does not have a MANSEC or SUBMANSEC')
    getFunctions(mansec.lower(), functiontoinclude, i)

  if pkgname == 'petsc':
    # a few special cases that must be handled manually
    typedefs['PetscBool'] = Typedef('PetscBool','sys','petscsys.h','PetscBool')
    classes['PetscNull'] = Class('PetscNull')
    classes['PetscNull'].includefile = 'petscsys.h'
    classes['PetscNull'].mansec = 'sys'
    classes['PetscNull'].submansec = 'sys'
    classes['PetscNull'].petscobject = False
    classes['PetscObject'].petscobject = False
    classes['PetscObject'].includefile = 'petscsys.h'

    # these functions are funky macros in C and cannot be parsed directly
    funcs['PetscOptionsBegin']             = Function('PetscOptionsBegin')
    funcs['PetscOptionsBegin'].mansec      = 'sys'
    funcs['PetscOptionsBegin'].file        = 'aoptions.c';
    funcs['PetscOptionsBegin'].includefile = 'petscoptions.h'
    funcs['PetscOptionsBegin'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsBegin'].opaquestub  = True
    funcs['PetscOptionsBegin'].arguments   = [Argument('comm',   'MPI_Comm'),
                                              Argument('prefix', 'char', stars = 0, array = True, const = True),
                                              Argument('mess',   'char', stars = 0, array = True, const = True),
                                              Argument('sec',    'char', stars = 0, array = True, const = True)]
    funcs['PetscOptionsEnd']               = Function('PetscOptionsEnd')
    funcs['PetscOptionsEnd'].mansec        = 'sys'
    funcs['PetscOptionsEnd'].file          = 'aoptions.c';
    funcs['PetscOptionsEnd'].includefile   = 'petscoptions.h'
    funcs['PetscOptionsEnd'].dir           = 'src/sys/objects/'
    funcs['PetscOptionsEnd'].opaquestub    = True

    funcs['PetscOptionsBool']             = Function('PetscOptionsBool')
    funcs['PetscOptionsBool'].mansec      = 'sys'
    funcs['PetscOptionsBool'].file        = 'aoptions.c';
    funcs['PetscOptionsBool'].includefile = 'petscoptions.h'
    funcs['PetscOptionsBool'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsBool'].opaquestub  = True
    funcs['PetscOptionsBool'].arguments   = [Argument('opt',           'char',      stars = 0, array = True, const = True),
                                             Argument('text',          'char',      stars = 0, array = True, const = True),
                                             Argument('man',           'char',      stars = 0, array = True, const = True),
                                             Argument('currentvalue',  'PetscBool'),
                                             Argument('value',         'PetscBool', stars = 1),
                                             Argument('set',           'PetscBool', stars = 1)]
    funcs['PetscOptionsBool3']             = Function('PetscOptionsBool3')
    funcs['PetscOptionsBool3'].mansec      = 'sys'
    funcs['PetscOptionsBool3'].file        = 'aoptions.c';
    funcs['PetscOptionsBool3'].includefile = 'petscoptions.h'
    funcs['PetscOptionsBool3'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsBool3'].opaquestub  = True
    funcs['PetscOptionsBool3'].arguments   = [Argument('opt',           'char',      stars = 0, array = True, const = True),
                                              Argument('text',          'char',      stars = 0, array = True, const = True),
                                              Argument('man',           'char',      stars = 0, array = True, const = True),
                                              Argument('currentvalue',  'PetscBool3'),
                                              Argument('value',         'PetscBool3', stars = 1),
                                              Argument('set',           'PetscBool3', stars = 1)]
    funcs['PetscOptionsInt']             = Function('PetscOptionsInt')
    funcs['PetscOptionsInt'].mansec      = 'sys'
    funcs['PetscOptionsInt'].file        = 'aoptions.c';
    funcs['PetscOptionsInt'].includefile = 'petscoptions.h'
    funcs['PetscOptionsInt'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsInt'].opaquestub  = True
    funcs['PetscOptionsInt'].arguments   = [Argument('opt',           'char',      stars = 0, array = True, const = True),
                                            Argument('text',          'char',      stars = 0, array = True, const = True),
                                            Argument('man',           'char',      stars = 0, array = True, const = True),
                                            Argument('currentvalue',  'PetscInt'),
                                            Argument('value',         'PetscInt', stars = 1),
                                            Argument('set',           'PetscBool', stars = 1)]
    funcs['PetscOptionsReal']             = Function('PetscOptionsReal')
    funcs['PetscOptionsReal'].mansec      = 'sys'
    funcs['PetscOptionsReal'].file        = 'aoptions.c';
    funcs['PetscOptionsReal'].includefile = 'petscoptions.h'
    funcs['PetscOptionsReal'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsReal'].opaquestub  = True
    funcs['PetscOptionsReal'].arguments   = [Argument('opt',           'char',      stars = 0, array = True, const = True),
                                             Argument('text',          'char',      stars = 0, array = True, const = True),
                                             Argument('man',           'char',      stars = 0, array = True, const = True),
                                             Argument('currentvalue',  'PetscReal'),
                                             Argument('value',         'PetscReal', stars = 1),
                                             Argument('set',           'PetscBool', stars = 1)]
    funcs['PetscOptionsScalar']             = Function('PetscOptionsScalar')
    funcs['PetscOptionsScalar'].mansec      = 'sys'
    funcs['PetscOptionsScalar'].file        = 'aoptions.c';
    funcs['PetscOptionsScalar'].includefile = 'petscoptions.h'
    funcs['PetscOptionsScalar'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsScalar'].opaquestub  = True
    funcs['PetscOptionsScalar'].arguments   = [Argument('opt',           'char',      stars = 0, array = True, const = True),
                                               Argument('text',          'char',      stars = 0, array = True, const = True),
                                               Argument('man',           'char',      stars = 0, array = True, const = True),
                                               Argument('currentvalue',  'PetscScalar'),
                                               Argument('value',         'PetscScalar', stars = 1),
                                               Argument('set',           'PetscBool', stars = 1)]
    funcs['PetscOptionsScalarArray']             = Function('PetscOptionsScalarArray')
    funcs['PetscOptionsScalarArray'].mansec      = 'sys'
    funcs['PetscOptionsScalarArray'].file        = 'aoptions.c';
    funcs['PetscOptionsScalarArray'].includefile = 'petscoptions.h'
    funcs['PetscOptionsScalarArray'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsScalarArray'].opaquestub  = True
    funcs['PetscOptionsScalarArray'].arguments   = [Argument('opt',           'char',        stars = 0, array = True, const = True),
                                                    Argument('text',          'char',        stars = 0, array = True, const = True),
                                                    Argument('man',           'char',        stars = 0, array = True, const = True),
                                                    Argument('value',         'PetscScalar', array = 1),
                                                    Argument('n',             'PetscInt',    stars = 1),
                                                    Argument('set',           'PetscBool',   stars = 1)]
    funcs['PetscOptionsIntArray']             = Function('PetscOptionsIntArray')
    funcs['PetscOptionsIntArray'].mansec      = 'sys'
    funcs['PetscOptionsIntArray'].file        = 'aoptions.c';
    funcs['PetscOptionsIntArray'].includefile = 'petscoptions.h'
    funcs['PetscOptionsIntArray'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsIntArray'].opaquestub  = True
    funcs['PetscOptionsIntArray'].arguments   = [Argument('opt',           'char',        stars = 0, array = True, const = True),
                                                 Argument('text',          'char',        stars = 0, array = True, const = True),
                                                 Argument('man',           'char',        stars = 0, array = True, const = True),
                                                 Argument('value',         'PetscInt',    array = 1),
                                                 Argument('n',             'PetscInt',    stars = 1),
                                                 Argument('set',           'PetscBool',   stars = 1)]
    funcs['PetscOptionsRealArray']             = Function('PetscOptionsRealArray')
    funcs['PetscOptionsRealArray'].mansec      = 'sys'
    funcs['PetscOptionsRealArray'].file        = 'aoptions.c';
    funcs['PetscOptionsRealArray'].includefile = 'petscoptions.h'
    funcs['PetscOptionsRealArray'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsRealArray'].opaquestub  = True
    funcs['PetscOptionsRealArray'].arguments   = [Argument('opt',           'char',        stars = 0, array = True, const = True),
                                                  Argument('text',          'char',        stars = 0, array = True, const = True),
                                                  Argument('man',           'char',        stars = 0, array = True, const = True),
                                                  Argument('value',         'PetscReal',    array = 1),
                                                  Argument('n',             'PetscInt',    stars = 1),
                                                  Argument('set',           'PetscBool',   stars = 1)]
    funcs['PetscOptionsBoolArray']             = Function('PetscOptionsBoolArray')
    funcs['PetscOptionsBoolArray'].mansec      = 'sys'
    funcs['PetscOptionsBoolArray'].file        = 'aoptions.c';
    funcs['PetscOptionsBoolArray'].includefile = 'petscoptions.h'
    funcs['PetscOptionsBoolArray'].dir         = 'src/sys/objects/'
    funcs['PetscOptionsBoolArray'].opaquestub  = True
    funcs['PetscOptionsBoolArray'].arguments   = [Argument('opt',           'char',        stars = 0, array = True, const = True),
                                                  Argument('text',          'char',        stars = 0, array = True, const = True),
                                                  Argument('man',           'char',        stars = 0, array = True, const = True),
                                                  Argument('value',         'PetscBool',   array = 1),
                                                  Argument('n',             'PetscInt',    stars = 1),
                                                  Argument('set',           'PetscBool',   stars = 1)]

  verbosePrint(verbose, '# PETSc classes')
  for i in classes.keys():
    verbosePrint(verbose, classes[i])

  verbosePrint(verbose, '# PETSc standalone functions')
  for i in funcs.keys():
    verbosePrint(verbose, funcs[i])

  verbosePrint(verbose, '# PETSc typedefs for function prototypes')
  for i in functiontypedefs.keys():
    verbosePrint(verbose, functiontypedefs[i])

  verbosePrint(verbose, 'Function-like macros  --------------------------------')
  for i in defines.keys():
    verbosePrint(verbose, defines[i])

  # check seealso for manual pages that they actually point to a valid manual page
  # this will be turned on later
  #for i in manualpages.keys():
  #  man = manualpages[i]
  #  for j in man.seealsos:
  #    if j.endswith('()'): j = j[:-2]
  #    if not j in manualpages:
  #      print('Manual page ' + man.name + ' has incorrect seealso ' + j)

  verbosePrint(verbose, 'Manual pages  --------------------------------')
  for i in manualpages.keys():
    verbosePrint(verbose, manualpages[i])

  #file = open('classes.data','wb')
  #pickle.dump(enums,file)
  #pickle.dump(senums,file)
  #pickle.dump(structs,file)
  #pickle.dump(aliases,file)
  #pickle.dump(classes,file)
  #pickle.dump(typedefs,file)

  return classes, enums, senums, typedefs, functiontypedefs, structs, funcs, includefiles, mansecs, submansecs

#
if __name__ ==  '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Generate PETSc/SLEPc API', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--verbose', action='store_true', required=False, help='show generated API')
  parser.add_argument('--package', metavar='petsc/slepc', required=False, help='package name', default='petsc')
  parser.add_argument('directory', help='root directory, either PETSC_DIR or SLEPC_DIR')
  args = parser.parse_args()

  getAPI(args.directory, args.package, args.verbose)
  if badSeealso: exit(1)
