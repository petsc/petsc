static char help[] = "Test device/host memory allocation in MatDenseSeqCUDA()\n\n";

/* Contributed by: Victor Eijkhout <eijkhout@tacc.utexas.edu> */

#include <petscmat.h>
int main(int argc, char** argv)
{
  PetscInt  global_size = 100;
  Mat       cuda_matrix;
  Vec       input,output;
  MPI_Comm  comm        = PETSC_COMM_SELF;
  PetscReal nrm         = 1;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(MatCreateDenseCUDA(comm,global_size,global_size,global_size,global_size,NULL,&cuda_matrix));
  PetscCall(MatAssemblyBegin(cuda_matrix,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(cuda_matrix,MAT_FINAL_ASSEMBLY));

  PetscCall(VecCreateSeqCUDA(comm,global_size,&input));
  PetscCall(VecDuplicate(input,&output));
  PetscCall(VecSet(input,1.));
  PetscCall(VecSet(output,2.));
  PetscCall(MatMult(cuda_matrix,input,output));
  PetscCall(VecNorm(output,NORM_2,&nrm));
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PETSc generated wrong result. Should be 0, but is %g",(double)nrm);
  PetscCall(VecDestroy(&input));
  PetscCall(VecDestroy(&output));
  PetscCall(MatDestroy(&cuda_matrix));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   build:
     requires: cuda

   test:
    nsize: 1

TEST*/
