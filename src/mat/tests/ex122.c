static char help[] = "Test MatMatMult() for AIJ and Dense matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B,C;
  PetscInt       M=10,N=5;
  PetscRandom    r;
  PetscBool      equal=PETSC_FALSE;
  PetscReal      fill = 1.0;
  PetscInt       nza,am,an;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));

  /* create a aij matrix A */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,M));
  CHKERRQ(MatSetType(A,MATAIJ));
  nza  = (PetscInt)(.3*M); /* num of nozeros in each row of A */
  CHKERRQ(MatSeqAIJSetPreallocation(A,nza,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,nza,NULL,nza,NULL));
  CHKERRQ(MatSetRandom(A,r));

  /* create a dense matrix B */
  CHKERRQ(MatGetLocalSize(A,&am,&an));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,am,N,PETSC_DECIDE));
  CHKERRQ(MatSetType(B,MATDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(B,NULL));
  CHKERRQ(MatMPIDenseSetPreallocation(B,NULL));
  CHKERRQ(MatSetRandom(B,r));
  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test MatMatMult() */
  CHKERRQ(MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C));
  CHKERRQ(MatMatMult(B,A,MAT_REUSE_MATRIX,fill,&C));
  CHKERRQ(MatMatMultEqual(B,A,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != B*A");

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex122.out

TEST*/
