
static char help[] = "Tests binary I/O of vectors and illustrates the use of user-defined event logging.\n\n";

#include <petscvec.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

int main(int argc,char **args)
{
  PetscMPIInt    rank,size;
  PetscInt       i,m = 10,low,high,ldim,iglobal;
  PetscScalar    v;
  Vec            u;
  PetscViewer    viewer;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  VECTOR_GENERATE,VECTOR_READ;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* PART 1:  Generate vector, then write it in binary format */

  PetscCall(PetscLogEventRegister("Generate Vector",VEC_CLASSID,&VECTOR_GENERATE));
  PetscCall(PetscLogEventBegin(VECTOR_GENERATE,0,0,0,0));
  /* Generate vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,m));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecGetOwnershipRange(u,&low,&high));
  PetscCall(VecGetLocalSize(u,&ldim));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100*rank);
    PetscCall(VecSetValues(u,1,&iglobal,&v,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
  PetscCall(VecView(u,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscOptionsSetValue(NULL,"-viewer_binary_mpiio",""));

  PetscCall(PetscLogEventEnd(VECTOR_GENERATE,0,0,0,0));

  /* PART 2:  Read in vector in binary format */

  /* Read new vector in binary format */
  PetscCall(PetscLogEventRegister("Read Vector",VEC_CLASSID,&VECTOR_READ));
  PetscCall(PetscLogEventBegin(VECTOR_READ,0,0,0,0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecLoad(u,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscLogEventEnd(VECTOR_READ,0,0,0,0));
  PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 1
       requires: mpiio

     test:
       suffix: 2
       nsize: 2
       requires: mpiio

TEST*/
