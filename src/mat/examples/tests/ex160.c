
static char help[] = "Tests MatMPIBAIJ format in sequential run \n";

#include <petscmat.h> 
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat               A,B;
  PetscInt          i,rstart,rend;
  PetscMPIInt       rank,size;
  PetscErrorCode    ierr;
  PetscScalar       v;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Create a MPIBAIJ matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,32,32);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,2,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,2,2,PETSC_NULL,2,PETSC_NULL);CHKERRQ(ierr);
 
  v = 1.0;
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++){
    ierr = MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Convert A to AIJ format */
  ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr); 
  ierr = MatDestroy(&B);CHKERRQ(ierr); 
  ierr = PetscFinalize();
  return 0;
}
