
static char help[] = "Tests binary I/O of vectors and illustrates the use of user-defined event logging.\n\n";

#include <petscvec.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,m = 10,low,high,ldim,iglobal;
  PetscScalar    v;
  Vec            u;
  PetscViewer    viewer;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  VECTOR_GENERATE,VECTOR_READ;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* PART 1:  Generate vector, then write it in binary format */

  CHKERRQ(PetscLogEventRegister("Generate Vector",VEC_CLASSID,&VECTOR_GENERATE));
  CHKERRQ(PetscLogEventBegin(VECTOR_GENERATE,0,0,0,0));
  /* Generate vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecGetOwnershipRange(u,&low,&high));
  CHKERRQ(VecGetLocalSize(u,&ldim));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100*rank);
    CHKERRQ(VecSetValues(u,1,&iglobal,&v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
  CHKERRQ(VecView(u,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PetscOptionsSetValue(NULL,"-viewer_binary_mpiio",""));

  CHKERRQ(PetscLogEventEnd(VECTOR_GENERATE,0,0,0,0));

  /* PART 2:  Read in vector in binary format */

  /* Read new vector in binary format */
  CHKERRQ(PetscLogEventRegister("Read Vector",VEC_CLASSID,&VECTOR_READ));
  CHKERRQ(PetscLogEventBegin(VECTOR_READ,0,0,0,0));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecLoad(u,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscLogEventEnd(VECTOR_READ,0,0,0,0));
  CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  CHKERRQ(VecDestroy(&u));
  ierr = PetscFinalize();
  return ierr;
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
