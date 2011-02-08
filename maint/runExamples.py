#!/usr/bin/env python
import os
import sys
import difflib
import TaoExamples    

TAO = {'TAO_DIR':'/home/sarich/working/tao_c',
        'PETSC_ARCH':'linux-gnu-cxx-debug',
        'PETSC_DIR':'/home/sarich/software/petsc-dev'}

if __name__=="__main__":
    examples = TaoExamples.TaoExamples()
    for ex in examples.list: #.withTag("eptorsion"):
        sys.stdout.write("\n\n*** Example %s ***\n" % ex.name)

        os.environ.update(TAO)
        cwd = os.path.join(TAO['TAO_DIR'],"tests")
        (r,o,e) = examples.execute(['rm','-f',ex.executableName()])
        (r,o,e) = examples.execute(ex.buildCommand(),cwd=cwd,echo=True)
        sys.stdout.write(o)

        (r,o1,e) = examples.execute(ex.runCommand(),cwd=cwd,echo=True)
        sys.stdout.write(o1)

                         
