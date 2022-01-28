static char help[] = "Test device/host memory allocation in MatDenseSeqCUDA()\n\n";

/* Contributed by: Victor Eijkhout <eijkhout@tacc.utexas.edu> */

#include <petscmat.h>
int main(int argc, char** argv)
{
  PetscErrorCode ierr;
  PetscInt       global_size=100;
  Mat            cuda_matrix;
  Vec            input,output;
  MPI_Comm       comm = PETSC_COMM_SELF;
  PetscReal      nrm = 1;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MatCreateDenseCUDA(comm,global_size,global_size,global_size,global_size,NULL,&cuda_matrix);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(cuda_matrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(cuda_matrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecCreateSeqCUDA(comm,global_size,&input);CHKERRQ(ierr);
  ierr = VecDuplicate(input,&output);CHKERRQ(ierr);
  ierr = VecSet(input,1.);CHKERRQ(ierr);
  ierr = VecSet(output,2.);CHKERRQ(ierr);
  ierr = MatMult(cuda_matrix,input,output);CHKERRQ(ierr);
  ierr = VecNorm(output,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PETSc generated wrong result. Should be 0, but is %g",(double)nrm);
  ierr = VecDestroy(&input);CHKERRQ(ierr);
  ierr = VecDestroy(&output);CHKERRQ(ierr);
  ierr = MatDestroy(&cuda_matrix);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   build:
     requires: cuda

   test:
    nsize: 1

TEST*/
