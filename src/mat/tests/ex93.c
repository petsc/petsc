static char help[] = "Test MatMatMult() and MatPtAP() for AIJ matrices.\n\n";

#include <petscmat.h>

extern PetscErrorCode testPTAPRectangular(void);

int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscScalar    a[] ={1.,1.,0.,0.,1.,1.,0.,0.,1.};
  PetscInt       ij[]={0,1,2};
  PetscReal      fill=4.0;
  PetscMPIInt    size,rank;
  PetscBool      isequal;
#if defined(PETSC_HAVE_HYPRE)
  PetscBool      test_hypre=PETSC_FALSE;
#endif

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_hypre",&test_hypre,NULL));
#endif
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));

  if (rank == 0) {
    CHKERRQ(MatSetValues(A,3,ij,3,ij,a,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Test MatMatMult() */
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&B));      /* B = A^T */
  CHKERRQ(MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C)); /* C = B*A */
  CHKERRQ(MatMatMult(B,A,MAT_REUSE_MATRIX,fill,&C));   /* recompute C=B*A */
  CHKERRQ(MatSetOptionsPrefix(C,"C_"));
  CHKERRQ(MatMatMultEqual(B,A,C,10,&isequal));
  PetscCheck(isequal,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatMult: C != B*A");

  CHKERRQ(MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&D)); /* D = C*A = (A^T*A)*A */
  CHKERRQ(MatMatMult(C,A,MAT_REUSE_MATRIX,fill,&D));
  CHKERRQ(MatMatMultEqual(C,A,D,10,&isequal));
  PetscCheck(isequal,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatMult: D != C*A");

  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));

  /* Test MatPtAP */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));      /* B = A */
  CHKERRQ(MatPtAP(A,B,MAT_INITIAL_MATRIX,fill,&C)); /* C = B^T*A*B */
  CHKERRQ(MatPtAPMultEqual(A,B,C,10,&isequal));
  PetscCheck(isequal,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatPtAP: C != B^T*A*B");

  /* Repeat MatPtAP to test symbolic/numeric separation for reuse of the symbolic product */
  CHKERRQ(MatPtAP(A,B,MAT_REUSE_MATRIX,fill,&C));
  CHKERRQ(MatPtAPMultEqual(A,B,C,10,&isequal));
  PetscCheck(isequal,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatPtAP(reuse): C != B^T*A*B");

  CHKERRQ(MatDestroy(&C));

  /* Test MatPtAP with A as a dense matrix */
  {
    Mat Adense;
    CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
    CHKERRQ(MatPtAP(Adense,B,MAT_INITIAL_MATRIX,fill,&C));

    CHKERRQ(MatPtAPMultEqual(Adense,B,C,10,&isequal));
    PetscCheck(isequal,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatPtAP(reuse): C != B^T*Adense*B");
    CHKERRQ(MatDestroy(&Adense));
  }

  if (size == 1) {
    /* A test contributed by Tobias Neckel <neckel@in.tum.de> */
    CHKERRQ(testPTAPRectangular());

    /* test MatMatTransposeMult(): A*B^T */
    CHKERRQ(MatMatTransposeMult(A,A,MAT_INITIAL_MATRIX,fill,&D)); /* D = A*A^T */
    CHKERRQ(MatScale(A,2.0));
    CHKERRQ(MatMatTransposeMult(A,A,MAT_REUSE_MATRIX,fill,&D));
    CHKERRQ(MatMatTransposeMultEqual(A,A,D,10,&isequal));
    PetscCheck(isequal,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatTranspose: D != A*A^T");
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(PetscFinalize());
  return 0;
}

/* a test contributed by Tobias Neckel <neckel@in.tum.de>, 02 Jul 2008 */
PetscErrorCode testPTAPRectangular(void)
{
  const int      rows = 3,cols = 5;
  int            i;
  Mat            A,P,C;

  PetscFunctionBegin;
  /* set up A  */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD, rows, rows,1, NULL, &A));
  for (i=0; i<rows; i++) {
    CHKERRQ(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* set up P */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD, rows, cols,5, NULL, &P));
  CHKERRQ(MatSetValue(P, 0, 0,  1.0, INSERT_VALUES));
  CHKERRQ(MatSetValue(P, 0, 1,  2.0, INSERT_VALUES));
  CHKERRQ(MatSetValue(P, 0, 2,  0.0, INSERT_VALUES));

  CHKERRQ(MatSetValue(P, 0, 3, -1.0, INSERT_VALUES));

  CHKERRQ(MatSetValue(P, 1, 0,  0.0, INSERT_VALUES));
  CHKERRQ(MatSetValue(P, 1, 1, -1.0, INSERT_VALUES));
  CHKERRQ(MatSetValue(P, 1, 2,  1.0, INSERT_VALUES));

  CHKERRQ(MatSetValue(P, 2, 0,  3.0, INSERT_VALUES));
  CHKERRQ(MatSetValue(P, 2, 1,  0.0, INSERT_VALUES));
  CHKERRQ(MatSetValue(P, 2, 2, -3.0, INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

  /* compute C */
  CHKERRQ(MatPtAP(A, P, MAT_INITIAL_MATRIX, 1.0, &C));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* compare results */
  /*
  printf("C:\n");
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  blitz::Array<double,2> actualC(cols, cols);
  actualC = 0.0;
  for (int i=0; i<cols; i++) {
    for (int j=0; j<cols; j++) {
      CHKERRQ(MatGetValues(C, 1, &i, 1, &j, &actualC(i,j)));
    }
  }
  blitz::Array<double,2> expectedC(cols, cols);
  expectedC = 0.0;

  expectedC(0,0) = 10.0;
  expectedC(0,1) =  2.0;
  expectedC(0,2) = -9.0;
  expectedC(0,3) = -1.0;
  expectedC(1,0) =  2.0;
  expectedC(1,1) =  5.0;
  expectedC(1,2) = -1.0;
  expectedC(1,3) = -2.0;
  expectedC(2,0) = -9.0;
  expectedC(2,1) = -1.0;
  expectedC(2,2) = 10.0;
  expectedC(2,3) =  0.0;
  expectedC(3,0) = -1.0;
  expectedC(3,1) = -2.0;
  expectedC(3,2) =  0.0;
  expectedC(3,3) =  1.0;

  int check = areBlitzArrays2NumericallyEqual(actualC,expectedC);
  validateEqualsWithParams3(check, -1 , "testPTAPRectangular()", check, actualC(check), expectedC(check));
  */

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(MatDestroy(&C));
  PetscFunctionReturn(0);
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      args: -matmatmult_via nonscalable
      output_file: output/ex93_1.out

   test:
      suffix: 3
      nsize: 2
      output_file: output/ex93_1.out

   test:
      suffix: 4
      nsize: 2
      args: -matptap_via scalable
      output_file: output/ex93_1.out

   test:
      suffix: btheap
      args: -matmatmult_via btheap -matmattransmult_via color
      output_file: output/ex93_1.out

   test:
      suffix: heap
      args: -matmatmult_via heap
      output_file: output/ex93_1.out

   #HYPRE PtAP is broken for complex numbers
   test:
      suffix: hypre
      nsize: 3
      requires: hypre !complex
      args: -matmatmult_via hypre -matptap_via hypre -test_hypre
      output_file: output/ex93_hypre.out

   test:
      suffix: llcondensed
      args: -matmatmult_via llcondensed
      output_file: output/ex93_1.out

   test:
      suffix: scalable
      args: -matmatmult_via scalable
      output_file: output/ex93_1.out

   test:
      suffix: scalable_fast
      args: -matmatmult_via scalable_fast
      output_file: output/ex93_1.out

TEST*/
