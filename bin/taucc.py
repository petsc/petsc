#!/usr/bin/env python
#!/bin/env python

# Usage:
#  taucc -cc=gcc -pdt_parse=/../cxxparse -tau_instr=/.../tau_instrumentor COMPILE_OPTIONS
#  
def main():
  import sys
  import os
  import commands

  sourcefiles=[]
  arglist=''
  pdt_parse='cxxparse'
  tau_instr='tau_instrumentor'
  cc='gcc'
  verbose='false'
  compile='false'

  for arg in sys.argv[1:]:
    filename,ext = os.path.splitext(arg)
    argsplit =  arg.split('=')
    #look for sourcefiles, validate & add to a list
    if ext == '.c' or ext == '.C' or ext == '.cpp':
      if os.path.isfile(arg):
        sourcefiles.append(arg)
      else:
        print 'Source file not found - skipping: ' + arg
    elif argsplit[0] == '-cc':
      cc = argsplit[1]
    elif argsplit[0] == '-pdt_parse':
      pdt_parse = argsplit[1]
    elif argsplit[0] == '-tau_instr':
      tau_instr = argsplit[1]
    elif arg == '-c':
        compile = 'true'
    elif arg == '-v' or arg == '--verbose':
        verbose  = 'true'
        arglist += ' '+arg        
    else:
      # Now make sure quotes are escaped properly
      # Group the rest of the arguments into a different list
      arg=arg.replace('"','\\"')
      arglist += ' '+arg

  if compile == 'false' and sourcefiles != [] :
    raise RuntimeError('Incorrect invocation of tool. requires option -c')
  elif sourcefiles == []:
    raise RuntimeError('Incorrect invocation - no source files specified')    
  # Now Compile the sourcefiles
  for sourcefile in sourcefiles:
    root,ext = os.path.splitext(sourcefile)
    pdt_file = sourcefile+ '.pdb'
    tau_file = root +'.inst' + ext
    obj_file = root + '.o'
    cmd1  = pdt_parse + ' ' + sourcefile + arglist
    cmd2  = tau_instr + ' ' + pdt_file + ' ' + sourcefile +' -o '+ tau_file
    cmd2 += ' -c -rn PetscFunctionReturn -rv PetscFunctionReturnVoid\\(\\)'
    cmd3  = cc + ' -c ' + tau_file + ' -o ' + obj_file + arglist

    if verbose == 'true':
      print cmd1
    os.system(cmd1)
    if verbose == 'true':
      print cmd2
    os.system(cmd2)
    if verbose == 'true':
      print cmd3
    os.system(cmd3)
    os.remove(pdt_file)
    os.remove(tau_file)
    
if __name__ ==  '__main__': 
    main()

