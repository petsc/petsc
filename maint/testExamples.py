#!/usr/bin/env python
import os
import sys
import difflib
import TaoExamples    

def printHelp():
    sys.stdout.write("""
usage: testExamples.py [option] [-]tag1 [[-]tag2] ...
Use command line tags to match either the names of the TAO examples or any
of the examples's tags.  - means to exclude any examples with that tag.
If multiple tags with no dash(-) are given, the example must match all tags.
-h     : print this message and exit (also --help)
-v     : verbose, print make and execute commands (also --verbose)
-s     : verbose + show all output (also --showoutput)
-m     : only print example names that match tags (also --match)
-c     : compile only, do not execute (also --compile)
-l     : list all examples and their tags (alse --list)
-d     : compare results against output in $TAO_DIR/tests/ouptut directory (also --diff)

""")

if __name__=="__main__":
    verbose = False
    showoutput = False
    match = False
    showdiff = False        
    compileonly = False
    
    examples = TaoExamples.TaoExamples()
    args = sys.argv[1:]
    
    for arg in ['--verbose','-v']:
        if arg in args:
            args.remove(arg)
            verbose = True
    for arg in ['--match','-m']:
        if arg in args:
            args.remove(arg)
            match = True
            
    for arg in ['--help','-h']:
        if arg in args:
            args.remove(arg)
            printHelp()
            sys.exit(0)
    for arg in ['--diff','-d']:
        if arg in args:
            args.remove(arg)
            showdiff = True
    for arg in ['--showoutput','-s']:
        if arg in args:
            args.remove(arg)
            showoutput=True
    for arg in ['--compile','-c']:
        if arg in args:
            args.remove(arg)
            compileonly = True
    for arg in ['--list','-l']:
        if arg in args:
            args.remove(arg)
            for e in examples.list:
                sys.stdout.write("%25s:" %e.name)
                for t in e.tags:
                    sys.stdout.write(" %s"%t)
                sys.stdout.write("\n")
            sys.exit(0)
    
            
        
    examples.setWithTags(args)
    if examples is None:
        sys.stderr.write('No examples match arguments:\n%s\n' % str(args))
        sys.exit(0)
    #for e in examples.list:
    #    print(e.name)
    #sys.exit(0)
        
    for ex in examples.list: #.withTag("eptorsion"):
        #sys.stdout.write("\n\n*** Example %s ***\n" % ex.name)
        if match:
            sys.stdout.write(ex.name+"\n")
            continue

        #os.environ.update(TAO)
        #cwd = os.path.join(TAO['TAO_DIR'],"tests")
        cwd = os.path.join(os.environ['TAO_DIR'],"tests")
        (r,o,e) = examples.execute(['rm','-f',ex.executableName()])
        (r,o,e) = examples.execute(ex.buildCommand(),cwd=cwd,echo=verbose)
        if (showoutput):
            sys.stdout.write("\n")
            sys.stdout.write(o)
            sys.stderr.write(e)

        if (not showoutput and r != 0 or not os.access(os.path.join(cwd,ex.executableName()),os.X_OK)):
            sys.stdout.write("\n")
            sys.stdout.write(o)
            sys.stdout.write(e)
            sys.stdout.write("** Error compiling %s. **\n\n" % ex.name)
        elif (r==0 and compileonly):
            sys.stdout.write("%s compiled OK\n" % ex.name)
        elif (r==0 and not compileonly):
            (r,o,e) = examples.execute(ex.runCommand(),cwd=cwd,echo=verbose)
            if (showoutput):
                sys.stdout.write("\n")
                sys.stdout.write(o)
                sys.stderr.write(e)
                
            if (not showoutput and r != 0):
                sys.stdout.write("\n")
                sys.stdout.write(o)
                sys.stdout.write(e)
                sys.stdout.write("** Error running %s. **\n\n" % ex.name)
            elif (r==0):
                if (showdiff):
                    goodname = os.path.join(os.environ['TAO_DIR'],'tests','output',ex.name+".out" )
                    if (not os.access(goodname,os.R_OK)):
                        sys.stderr.write("Error: Could not find file %s\n" % goodname)
                    else:
                        good = open(goodname,'r')
                        goodtext = good.readlines()
                        good.close()
                        diff = list(difflib.context_diff(goodtext,o.splitlines(1),fromfile='TAO reference output',tofile='local output'))
                        if len(diff) != 0:
                            sys.stdout.write("\n")
                            for line in diff:
                                sys.stdout.write(line)
                                sys.stdout.write("** Possible error in %s. See diff above. **\n\n" % ex.name)
                        else:
                            sys.stdout.write("%s OK\n" % ex.name)

                         
