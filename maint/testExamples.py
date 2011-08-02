#!/usr/bin/env python
import os
import sys
import difflib
import TaoExamples    

if __name__=="__main__":
    
    examples = TaoExamples.TaoExamples()
    examples.setWithTags(sys.argv[1:])
    if examples is None:
        sys.stderr.write('No examples match arguments:\n%s\n' % str(sys.argv[1:]))
        sys.exit(0)
    #for e in examples.list:
    #    print(e.name)
    #sys.exit(0)
    for ex in examples.list: #.withTag("eptorsion"):
        #sys.stdout.write("\n\n*** Example %s ***\n" % ex.name)

        #os.environ.update(TAO)
        #cwd = os.path.join(TAO['TAO_DIR'],"tests")
        cwd = os.path.join(os.environ['TAO_DIR'],"tests")
        (r,o,e) = examples.execute(['rm','-f',ex.executableName()])
        (r,o,e) = examples.execute(ex.buildCommand(),cwd=cwd,echo=False)
        if (r != 0 or not os.access(os.path.join(cwd,ex.executableName()),os.X_OK)):
            sys.stdout.write("\n")
            sys.stdout.write(o)
            sys.stdout.write(e)
            sys.stdout.write("** Error compiling %s. **\n\n" % ex.name)
        else:
            (r,o,e) = examples.execute(ex.runCommand(),cwd=cwd,echo=False)
            if (r != 0):
                sys.stdout.write("\n")
                sys.stdout.write(o)
                sys.stdout.write(e)
                sys.stdout.write("** Error running %s. **\n\n" % ex.name)
            else:
                goodname = os.path.join('output',ex.name)
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

                         
