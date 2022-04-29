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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  PetscCall(PetscRandomSetFromOptions(r));

  /* create a aij matrix A */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,M));
  PetscCall(MatSetType(A,MATAIJ));
  nza  = (PetscInt)(.3*M); /* num of nozeros in each row of A */
  PetscCall(MatSeqAIJSetPreallocation(A,nza,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,nza,NULL,nza,NULL));
  PetscCall(MatSetRandom(A,r));

  /* create a dense matrix B */
  PetscCall(MatGetLocalSize(A,&am,&an));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,am,N,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATDENSE));
  PetscCall(MatSeqDenseSetPreallocation(B,NULL));
  PetscCall(MatMPIDenseSetPreallocation(B,NULL));
  PetscCall(MatSetRandom(B,r));
  PetscCall(PetscRandomDestroy(&r));

  /* Test MatMatMult() */
  PetscCall(MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C));
  PetscCall(MatMatMult(B,A,MAT_REUSE_MATRIX,fill,&C));
  PetscCall(MatMatMultEqual(B,A,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != B*A");

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex122.out

TEST*/
