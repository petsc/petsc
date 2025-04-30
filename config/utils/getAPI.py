#!/usr/bin/env python3
#
#  Processes PETSc's include/petsc*.h and source files to determine
#  the PETSc enums, structs, functions and classes
#
#  Calling sequence: (must be called in the PETSC_DIR directory)
#      getAPI
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

verbose = False

def verbosePrint(text):
  '''Prints the text if run with verbose option'''
  if verbose: print(text)

classes = {}
funcs = {}           # standalone functions like PetscInitialize()
allfuncs = set()     # both class and standalone functions, used to prevent duplicates
enums = {}
senums = {}          # like enums except strings instead of integer values for enumvalue
typedefs = {}
aliases = {}
structs = {}
includefiles = {}
mansecs = {}         # mansec[mansecname] = set(all submansecnames in mansecname)
submansecs = set()

regcomment   = re.compile(r'/\* [-A-Za-z _(),<>|^\*/0-9.:=\[\]\.;]* \*/')
regcomment2  = re.compile(r'// [-A-Za-z _(),<>|^\*/0-9.:=\[\]\.;]*')
regblank     = re.compile(r' [ ]*')

def displayIncludeMansec(obj):
    return '  ' + str(obj.includefile)+' (' + str(obj.mansec) + ')\n'

def displayFile(obj):
    return '  ' + str(obj.dir) + '/' + str(obj.file) + '\n'

class Typedef:
    '''Represents typedef oldtype newtype'''
    def __init__(self, name, mansec, includefile, value, *args, **kwargs):
        self.name        = name
        self.mansec      = mansec
        self.includefile = includefile
        self.value       = value

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
        self.opaquestub  = False # only interface is automatic, C stub is custom
        self.arguments   = []

    def __str__(self):
        mstr = '  ' + str(self.name) + '()\n'
        mstr += '  ' + displayIncludeMansec(self)
        mstr += '  ' + displayFile(self)
        if self.opaque:   mstr += '    opaque binding\n'
        elif self.opaque: mstr += '    opaque stub\n'
        if self.arguments:
          mstr += '    Arguments\n'
          for i in self.arguments:
            mstr += '  ' + str(i)
        return mstr

class Argument:
    '''Represents an argument in a Function'''
    def __init__(self, *args, **kwargs):
        self.name      = None
        self.typename  = None
        self.stars     = 0
        self.array     = False
        self.optional  = False
        self.const     = False
        #  PETSc returns strings in two ways either
        #     with a pointer to an array: char *[]
        #     or by copying the string into a given array with a given length: char [], size_t len
        self.stringlen    = False  # if true the argument is the length of the previous argument which is a character string

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
        for i in self.values:
          mstr += '  ' + str(i) + '\n'
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
        for i in self.values.keys():
          mstr += '  ' + i + ' ' + self.values[i] + '\n'
        return mstr

class IncludeFile:
    '''Represents an include (interesting) file found and what interesting files it includes'''
    def __init__(self, mansec, includefile, included, *args, **kwargs):
        self.mansec      = mansec
        self.includefile = includefile
        self.included    = included # include files it includes

    def __str__(self):
        mstr = str(self.mansec) + ' ' + str(self.includefile) + '\n'
        for i in self.included:
          mstr += '  ' + str(i) + '\n'
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
        mstr += '  PetscObject <' + str(self.petscobject) + '>\n\n'
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

def getIncludeFiles(filename):
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
      if not line == file and os.path.isfile(os.path.join('include',line)):
        included.append(line)
    line = f.readline()
  includefiles[file] = IncludeFile(mansec,file,included)
  f.close()

def getEnums(filename):
  import re
  regtypedef  = re.compile(r'typedef [ ]*enum')
  reg         = re.compile(r'}')
  regname     = re.compile(r'}[ A-Za-z0-9]*')

  file = os.path.basename(filename).replace('types.h','.h')
  f = open(filename)
  line = f.readline()
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
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
  f.close()

def getSenums(filename):
  import re
  regdefine   = re.compile(r'typedef const char \*[A-Za-z]*;')
  file = os.path.basename(filename).replace('types.h','.h')
  mansec = None
  f = open(filename)
  line = f.readline()
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
    line = f.readline()
  f.close()

def getTypedefs(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regdefine   = re.compile(r'typedef [A-Za-z0-9_]* [ ]*[A-Za-z0-9_]*;')
  submansec = None
  mansec = None
  f = open(filename)
  line = f.readline()
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regdefine.search(line)
    if fl:
      typedef = fl.group(0).split()[2][0:-1];
      if typedef in typedefs:
        typedefs[typedef].name = None # mark to be deleted since it appears multiple times (with presumably different values)
      else:
        typedefs[typedef] = Typedef(typedef,mansec,file,fl.group(0).split()[1])
    line = f.readline()
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
  line = f.readline()
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
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
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
  line = f.readline()
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
    line = f.readline()
  f.close()

def findlmansec(dir):  # could use dir to determine mansec
    '''Finds mansec and submansec from a makefile'''
    file = os.path.join(dir,'makefile')
    mansec = None
    submansec = None
    with open(file) as mklines:
      #print(file)
      submansecl = [line for line in mklines if line.find('BFORTSUBMANSEC') > -1]
      if submansecl:
        submansec = re.sub('BFORTSUBMANSEC[ ]*=[ ]*','',submansecl[0]).strip('\n').strip().lower()
    if not submansec:
      with open(file) as mklines:
        #print(file)
        submansecl = [line for line in mklines if (line.find('SUBMANSEC') > -1 and line.find('BFORT') == -1)]
        if submansecl:
          submansec = re.sub('SUBMANSEC[ ]*=[ ]*','',submansecl[0]).strip('\n').strip().lower()
    with open(file) as mklines:
      mansecl = [line for line in mklines if line.startswith('MANSEC')]
      if mansecl:
        mansec = re.sub('MANSEC[ ]*=[ ]*','',mansecl[0]).strip('\n').strip().lower()
        #print(':MANSEC:' + mansec)
    if not submansec: submansec = mansec
    return mansec,submansec

def getpossiblefunctions():
   '''Gets a list of all the functions in the include/ directory that may be used in the binding for other languages'''
   try:
     output = check_output('grep -F -e "PETSC_EXTERN PetscErrorCode" -e "static inline PetscErrorCode" include/*.h', shell=True).decode('utf-8')
   except subprocess.CalledProcessError as e:
     raise RuntimeError('Unable to find possible functions in the include files')
   funs = output.replace('PETSC_EXTERN','').replace('PetscErrorCode','').replace('static inline','')
   functiontoinclude = {}
   for i in funs.split('\n'):
     file = i[i.find('/') + 1:i.find('.') + 2]
     f = i[i.find(': ') + 2:i.find('(')].strip()
     functiontoinclude[f] = file.replace('types','')
   return functiontoinclude

def getFunctions(mansec, functiontoinclude, filename):
  '''Appends the functions found in filename to their associated class classes[i], or funcs[] if they are classless'''
  import re
  regfun      = re.compile(r'^[static inline]*PetscErrorCode ')
  regarg      = re.compile(r'\([A-Za-z0-9*_\[\]]*[,\) ]')
  regerror    = re.compile(r'PetscErrorCode')
  reg         = re.compile(r' ([*])*[a-zA-Z0-9_]*([\[\]]*)')
  regname     = re.compile(r' [*]*([a-zA-Z0-9_]*)[\[\]]*')  

  rejects     = ['PetscErrorCode','...','<','(*)','(**)','off_t','MPI_Datatype','va_list','PetscStack','Ceed']
  #
  # search through list BACKWARDS to get the longest match
  #
  classlist = classes.keys()
  classlist = sorted(classlist)
  classlist.reverse()
  f = open(filename)
  line = f.readline()
  while line:
    fl = regfun.search(line)
    if fl:
      opaque = False
      opaquestub = False
      if  line[0:line.find('(')].find('_') > -1:
        line = f.readline()
        continue
      line = line.replace('PETSC_UNUSED','')
      line = line.strip()
      if line.endswith(' PeNS'):
        opaque = True
        line = line[0:-5]
      if line.endswith(' PeNSS'):
        opaquestub = True
        line = line[0:-6]
      if line.endswith(';') or line.find(')') < len(line)-1:
        line = f.readline()
        continue
      line = regfun.sub("",line)
      line = regcomment.sub("",line)
      line = line.replace("\n","")
      line = line.strip()
      name = line[:line.find("(")]
      if not name in functiontoinclude or name in allfuncs:
        line = f.readline()
        continue

      fl = regarg.search(line)
      if fl:
        fun = Function(name)
        fun.opaque = opaque
        fun.opaquestub = opaquestub
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
            for i in args:
              if i.find('**') > -1 and not i.strip().startswith('void'): fun.opaque = True
              if i.find('unsigned ') > -1: fun.opaque = True
              if i.replace('*','').endswith('Fn') or i.replace('*','').endswith('Func'): fun.opaque = True
              arg = Argument()
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
              if argname: arg.name = argname[0]
              else: arg.name = 'noname'
              i =  regblank.sub('',reg.sub(r'\1\2 ',i).strip()).replace('*','').replace('[]','')
              arg.typename = i
              # fix input character arrays that are written as *variable name
              if arg.typename == 'char' and not arg.array and arg.stars == 1:
                arg.array = 1
                arg.stars = 0
              if arg.typename == 'char' and not arg.array:
                fun.opaque = True
              #if arg.typename == 'char' and arg.array and arg.stars:
              #  arg.const = False
              if arg.typename.count('_') and not arg.typename in ['MPI_Comm', 'size_t']:
                fun.opaque = True
              if fun.arguments and not fun.arguments[-1].const and fun.arguments[-1].typename == 'char' and arg.typename == 'size_t':
                arg.stringlen = True
              fun.arguments.append(arg)

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

    line = f.readline()
  f.close()

ForbiddenDirectories = ['tests', 'tutorials', 'doc', 'output', 'ftn-custom', 'ftn-auto', 'ftn-mod', 'binding', 'binding', 'config', 'lib', '.git', 'share', 'systems']

def getAPI():
  global typedefs
  args = [os.path.join('include',i) for i in os.listdir('include') if i.endswith('.h') and not i.endswith('deprecated.h')]
  for i in args:
    getIncludeFiles(i)
  verbosePrint('Include files -------------------------------------')
  for i in includefiles.keys():
    verbosePrint(includefiles[i])

  for i in args:
    getEnums(i)
  verbosePrint('Enums ---------------------------------------------')
  for i in enums.keys():
    verbosePrint(enums[i])

  for i in args:
    getSenums(i)
  verbosePrint('String enums ---------------------------------------------')
  for i in senums.keys():
    verbosePrint(senums[i])

  for i in args:
    getStructs(i)
  verbosePrint('Structs ---------------------------------------------')
  for i in structs.keys():
    verbosePrint(structs[i])

  for i in args:
    getTypedefs(i)
  cp = {}
  for i in typedefs.keys():
    if typedefs[i].name: cp[i] = typedefs[i] # delete ones marked as having multiple definitions
  typedefs = cp
  verbosePrint('Typedefs ---------------------------------------------')
  for i in typedefs.keys():
    verbosePrint(typedefs[i])

  for i in args:
    getClasses(i)

  functiontoinclude = getpossiblefunctions()
  for dirpath, dirnames, filenames in os.walk('src',topdown=True):
    dirnames[:] = [d for d in dirnames if d not in ForbiddenDirectories]
    if not os.path.isfile(os.path.join(dirpath,'makefile')): continue
    mansec, submansec = findlmansec(dirpath)
    for i in os.listdir(dirpath):
      if i.endswith('.c') or i.endswith('.cxx'): getFunctions(mansec, functiontoinclude, os.path.join(dirpath,i))
  for i in args:
    getFunctions('sys', functiontoinclude, i)

  verbosePrint('Classes  ---------------------------------------------')
  for i in classes.keys():
    verbosePrint(classes[i])

  verbosePrint('Standalone functions  --------------------------------')
  for i in funcs.keys():
    verbosePrint(funcs[i])

  #file = open('classes.data','wb')
  #pickle.dump(enums,file)
  #pickle.dump(senums,file)
  #pickle.dump(structs,file)
  #pickle.dump(aliases,file)
  #pickle.dump(classes,file)
  #pickle.dump(typedefs,file)

  return classes, enums, senums, typedefs, structs, funcs, includefiles, mansecs, submansecs

#
if __name__ ==  '__main__':
  getAPI()

