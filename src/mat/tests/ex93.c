static char help[] = "Test MatMatMult() and MatPtAP() for AIJ matrices.\n\n";

#include <petscmat.h>

extern PetscErrorCode testPTAPRectangular(void);

int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscScalar    a[] ={1.,1.,0.,0.,1.,1.,0.,0.,1.};
  PetscInt       ij[]={0,1,2};
  PetscErrorCode ierr;
  PetscReal      fill=4.0;
  PetscMPIInt    size,rank;
  PetscBool      isequal;
#if defined(PETSC_HAVE_HYPRE)
  PetscBool      test_hypre=PETSC_FALSE;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_hypre",&test_hypre,NULL);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  if (rank == 0) {
    ierr = MatSetValues(A,3,ij,3,ij,a,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatMatMult() */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);      /* B = A^T */
  ierr = MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); /* C = B*A */
  ierr = MatMatMult(B,A,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);   /* recompute C=B*A */
  ierr = MatSetOptionsPrefix(C,"C_");CHKERRQ(ierr);
  ierr = MatMatMultEqual(B,A,C,10,&isequal);CHKERRQ(ierr);
  if (!isequal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatMult: C != B*A");

  ierr = MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr); /* D = C*A = (A^T*A)*A */
  ierr = MatMatMult(C,A,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMultEqual(C,A,D,10,&isequal);CHKERRQ(ierr);
  if (!isequal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatMult: D != C*A");

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test MatPtAP */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);      /* B = A */
  ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); /* C = B^T*A*B */
  ierr = MatPtAPMultEqual(A,B,C,10,&isequal);CHKERRQ(ierr);
  if (!isequal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatPtAP: C != B^T*A*B");

  /* Repeat MatPtAP to test symbolic/numeric separation for reuse of the symbolic product */
  ierr = MatPtAP(A,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatPtAPMultEqual(A,B,C,10,&isequal);CHKERRQ(ierr);
  if (!isequal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatPtAP(reuse): C != B^T*A*B");

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  if (size == 1) {
    /* A test contributed by Tobias Neckel <neckel@in.tum.de> */
    ierr = testPTAPRectangular();CHKERRQ(ierr);

    /* test MatMatTransposeMult(): A*B^T */
    ierr = MatMatTransposeMult(A,A,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr); /* D = A*A^T */
    ierr = MatScale(A,2.0);CHKERRQ(ierr);
    ierr = MatMatTransposeMult(A,A,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
    ierr = MatMatTransposeMultEqual(A,A,D,10,&isequal);CHKERRQ(ierr);
    if (!isequal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatTranspose: D != A*A^T");
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* a test contributed by Tobias Neckel <neckel@in.tum.de>, 02 Jul 2008 */
PetscErrorCode testPTAPRectangular(void)
{
  const int      rows = 3,cols = 5;
  PetscErrorCode ierr;
  int            i;
  Mat            A,P,C;

  PetscFunctionBegin;
  /* set up A  */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, rows, rows,1, NULL, &A);CHKERRQ(ierr);
  for (i=0; i<rows; i++) {
    ierr = MatSetValue(A, i, i, 1.0, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* set up P */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, rows, cols,5, NULL, &P);CHKERRQ(ierr);
  ierr = MatSetValue(P, 0, 0,  1.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(P, 0, 1,  2.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(P, 0, 2,  0.0, INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(P, 0, 3, -1.0, INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(P, 1, 0,  0.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(P, 1, 1, -1.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(P, 1, 2,  1.0, INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatSetValue(P, 2, 0,  3.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(P, 2, 1,  0.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(P, 2, 2, -3.0, INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* compute C */
  ierr = MatPtAP(A, P, MAT_INITIAL_MATRIX, 1.0, &C);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* compare results */
  /*
  printf("C:\n");
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  blitz::Array<double,2> actualC(cols, cols);
  actualC = 0.0;
  for (int i=0; i<cols; i++) {
    for (int j=0; j<cols; j++) {
      ierr = MatGetValues(C, 1, &i, 1, &j, &actualC(i,j));CHKERRQ(ierr);
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

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
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
