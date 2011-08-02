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
        if (r != 0):
            sys.stdout.write(o)
            sys.stdout.write(e)
        else:
            (r,o,e) = examples.execute(ex.runCommand(),cwd=cwd,echo=False)
            if (r != 0):
                sys.stdout.write(o)
                sys.stdout.write(e)
            else:
                ofname = os.path.join('output',ex.name+'.out')
                if (not os.access('output',os.X_OK)):
                    sys.stdout.write("Creating directory output\n")
                    os.mkdir('output')
                
                of = open(ofname,'w')
                sys.stdout.write("Writing %s\n" % ofname)
                of.write(o)
                of.close()

                         
