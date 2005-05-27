.SUFFIXES: .so

PWD = /home/knepley/PETSc-dev/petsc-dev/python
WRAPPER = ${PWD}/wrappers
SOURCE = wrappers/PETSc/Base.c wrappers/PETSc/PetscLog.c wrappers/PETSc/PetscOptions.c wrappers/PETSc/PetscViewer.c wrappers/PETSc/PetscMap.c wrappers/PETSc/Vec.c wrappers/PETSc/Mat.c wrappers/PETSc/KSP.c wrappers/PETSc/PC.c wrappers/PETSc/SNES.c

build: ${SOURCE:.c=.so}

wrappers/PETSc/Base.c: sidl/snes.sidl
	/PETSc3/ase/Compiler/driver/python/scandal.py --client=Python -clientDirs={Python:${WRAPPER}} -ior=0 -defaultVisitors="{ASE.Compiler.StubGenerator:[PETSc.Python.StubGenerator],ASE.Compiler.CartilageGenerator:[],ASE.Compiler.IORGenerator:[],ASE.Compiler.SkeletonGenerator:[],ASE.Compiler.ImplementationGenerator:[],ASE.Compiler.Output:[ASE.Compiler.Cxx.Output,ASE.Compiler.Python.Output]}" -outputFiles=1 -packageDirs=[${PWD}/Generator] -packageServerDirs=[${PWD}/Generator/server-python-generator] -generatorPackage=PETSc -includes=[/PETSc3/ase/Runtime/sidl/ase.sidl] sidl/snes.sidl

.c.so:
	gcc -Wall -g -c -I/usr/include/python2.4 -I/usr/local/mpich/include -I${PETSC_DIR}/include -I${PETSC_DIR}/bmake/linux-gnu -o $*.o $<
	gcc -shared -o $@ $*.o -lpython2.4 -Wl,-rpath,/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu-test -L/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu-test -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetsc

clean:
	rm -f ${SOURCE:.c=.o} ${SOURCE:.c=.so}

realclean: clean
	rm -f ${SOURCE}

tarball:
	tar cvzf PETScPython.tar.gz -C wrappers `cd wrappers; find PETSc -name "*.c" -o -name "*.py"`

getTarball:
	wget --passive-ftp ftp://ftp.mcs.anl.gov/pub/petsc/PETScPython.tar.gz .

pushTarball:
	scp PETScPython.tar.gz petsc@terra.mcs.anl.gov:/mcs/ftp/pub/petsc

install:
	tar xvzf PETScPython.tar.gz -C ${PETSC_DIR}/src/python

vec: sidl/vec.sidl
	/PETSc3/ase/Compiler/driver/python/scandal.py --client=Python -clientDirs={Python:${WRAPPER}} -ior=0 -defaultVisitors="{ASE.Compiler.StubGenerator:[PETSc.Python.StubGenerator],ASE.Compiler.CartilageGenerator:[],ASE.Compiler.IORGenerator:[],ASE.Compiler.SkeletonGenerator:[],ASE.Compiler.ImplementationGenerator:[],ASE.Compiler.Output:[ASE.Compiler.Cxx.Output,ASE.Compiler.Python.Output]}" -outputFiles=1 -packageDirs=[${PWD}/Generator] -packageServerDirs=[${PWD}/Generator/server-python-generator] -generatorPackage=PETSc -includes=[/PETSc3/ase/Runtime/sidl/ase.sidl,sidl/base.sidl] sidl/vec.sidl
