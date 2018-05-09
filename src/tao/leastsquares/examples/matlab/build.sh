
# Compile without mpi

mex -I${PETSC_DIR}/include -I${PETSC_DIR}/include/petsc/mpiuni -I${PETSC_DIR}/${PETSC_ARCH}/include taopounders.c -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc

# On linux; make sure LD_LIBRARY_PATH is set correctly
/soft/com/packages/MATLAB/R2017a/bin/mex -I${PETSC_DIR}/include -I${PETSC_DIR}/include/petsc/mpiuni -I${PETSC_DIR}/${PETSC_ARCH}/include taopounders.c -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc


