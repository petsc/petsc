PWD = /home/knepley/PETSc-dev/petsc-dev/python

gen:
	/PETSc3/ase/Compiler/driver/python/scandal.py --client=Python -clientDirs={Python:${PWD}} -ior=0 -defaultVisitors="{ASE.Compiler.StubGenerator:[PETSc.Python.StubGenerator],ASE.Compiler.CartilageGenerator:[],ASE.Compiler.IORGenerator:[],ASE.Compiler.SkeletonGenerator:[],ASE.Compiler.ImplementationGenerator:[],ASE.Compiler.Output:[ASE.Compiler.Cxx.Output,ASE.Compiler.Python.Output]}" -outputFiles=1 -packageDirs=[${PWD}/Generator] -packageServerDirs=[${PWD}/Generator/server-python-generator] -generatorPackage=PETSc -includes=[/PETSc3/ase/Runtime/sidl/ase.sidl] sidl/snes.sidl

PETSc/Base.so: PETSc/Base.c
	gcc -Wall -g -c -I/usr/include/python2.4 -I/usr/local/mpich/include -I${PETSC_DIR}/include -I${PETSC_DIR}/bmake/linux-gnu -o $*.o $<
	gcc -shared -o $@ $*.o -lpython2.4 -Wl,-rpath,/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu -L/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetsc

PETSc/SNES.so: PETSc/SNES.c
	gcc -Wall -g -c -I/usr/include/python2.4 -I/usr/local/mpich/include -I${PETSC_DIR}/include -I${PETSC_DIR}/bmake/linux-gnu -o $*.o $<
	gcc -shared -o $@ $*.o -lpython2.4 -Wl,-rpath,/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu -L/home/knepley/PETSc-dev/petsc-dev/lib/linux-gnu -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetsc

build: PETSc/SNES.so PETSc/Base.so

clean:
	rm -f PETSc/SNES.so PETSc/Base.so
