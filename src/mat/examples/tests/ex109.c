static char help[] = "Test MatMatMult() for AIJ and Dense matrices.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscInt       i,M=10,N=5;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      equal=PETSC_FALSE;
  PetscReal      fill = 1.0;
  PetscInt       nza,am,an;
  PetscMPIInt    size;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-fill",&fill,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* create a aij matrix A */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,M);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  nza  = (PetscInt)(.3*M); /* num of nozeros in each row of A */
  ierr = MatSeqAIJSetPreallocation(A,nza,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,nza,PETSC_NULL,nza,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetRandom(A,r);CHKERRQ(ierr);

  /* create a dense matrix B */
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,an,PETSC_DECIDE,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = MatSetType(B,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetRandom(B,r);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test C = A*B (aij*dense) */
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);

  ierr = MatMatMultSymbolic(A,B,fill,&D);CHKERRQ(ierr);
  for (i=0; i<2; i++){
    ierr = MatMatMultNumeric(A,B,D);CHKERRQ(ierr);
  }
  ierr = MatEqual(C,D,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size == 1){
    /* Test D = C*A (dense*aij) */
    ierr = MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr); // error when size=2 - check later!
    ierr = MatMatMult(C,A,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);  
    ierr = MatDestroy(&D);CHKERRQ(ierr);

    /* Test D = A*C (aij*dense) */
    if (M == N){
      //printf(" Test D = A*C (aij*dense) ...\n");
      ierr = MatMatMult(A,C,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
      ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);
    }

    /* Test D = B*C (dense*dense) */
    //printf(" Test D = B*C (dense*dense) ...\n");
    ierr = MatMatMult(B,C,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
    ierr = MatMatMult(B,C,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}
