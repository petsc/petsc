.SUFFIXES: .so

PWD = /home/knepley/PETSc-dev/petsc-dev/python
SOURCE = PETSc/Base.c PETSc/PetscViewer.c PETSc/PetscMap.c PETSc/Vec.c PETSc/Mat.c PETSc/KSP.c PETSc/PC.c PETSc/SNES.c

build: ${SOURCE:.c=.so}

PETSc/Base.c: sidl/snes.sidl
	/PETSc3/ase/Compiler/driver/python/scandal.py --client=Python -clientDirs={Python:${PWD}} -ior=0 -defaultVisitors="{ASE.Compiler.StubGenerator:[PETSc.Python.StubGenerator],ASE.Compiler.CartilageGenerator:[],ASE.Compiler.IORGenerator:[],ASE.Compiler.SkeletonGenerator:[],ASE.Compiler.ImplementationGenerator:[],ASE.Compiler.Output:[ASE.Compiler.Cxx.Output,ASE.Compiler.Python.Output]}" -outputFiles=1 -packageDirs=[${PWD}/Generator] -packageServerDirs=[${PWD}/Generator/server-python-generator] -generatorPackage=PETSc -includes=[/PETSc3/ase/Runtime/sidl/ase.sidl] sidl/snes.sidl

.c.so:
	gcc -Wall -g -c -I/usr/include/python2.4 -I/usr/local/mpich/include -I${PETSC_DIR}/include -I${PETSC_DIR}/bmake/linux-gnu -o $*.o $<
	gcc -shared -o $@ $*.o -lpython2.4 -Wl,-rpath,/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu-test -L/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu-test -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetsc

clean:
	rm -f ${SOURCE:.c=.o} ${SOURCE:.c=.so}

realclean: clean
	rm -f ${SOURCE}

tarball:
	tar cvzf PETScPython.tgz `find PETSc -name "*.c"`
