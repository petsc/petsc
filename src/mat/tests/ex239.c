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

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(MatCreateDenseCUDA(comm,global_size,global_size,global_size,global_size,NULL,&cuda_matrix));
  CHKERRQ(MatAssemblyBegin(cuda_matrix,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(cuda_matrix,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecCreateSeqCUDA(comm,global_size,&input));
  CHKERRQ(VecDuplicate(input,&output));
  CHKERRQ(VecSet(input,1.));
  CHKERRQ(VecSet(output,2.));
  CHKERRQ(MatMult(cuda_matrix,input,output));
  CHKERRQ(VecNorm(output,NORM_2,&nrm));
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PETSc generated wrong result. Should be 0, but is %g",(double)nrm);
  CHKERRQ(VecDestroy(&input));
  CHKERRQ(VecDestroy(&output));
  CHKERRQ(MatDestroy(&cuda_matrix));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
   build:
     requires: cuda

   test:
    nsize: 1

TEST*/
