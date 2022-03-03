static char help[] = "Test MatMatMult(), MatTranspose(), MatTransposeMatMult() for Dense and Elemental matrices.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex104 -mat_type elemental
*/

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscInt       i,M=10,N=5,j,nrows,ncols,am,an,rstart,rend;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      equal,Aiselemental;
  PetscReal      fill = 1.0;
  IS             isrows,iscols;
  const PetscInt *rows,*cols;
  PetscScalar    *v,rval;
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool      Test_MatMatMult=PETSC_TRUE;
#else
  PetscBool      Test_MatMatMult=PETSC_FALSE;
#endif
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));

  /* Set local matrix entries */
  CHKERRQ(MatGetOwnershipIS(A,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      CHKERRQ(PetscRandomGetValue(r,&rval));
      v[i*ncols+j] = rval;
    }
  }
  CHKERRQ(MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));
  CHKERRQ(PetscRandomDestroy(&r));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATELEMENTAL,&Aiselemental));

  /* Test MatCreateTranspose() and MatTranspose() */
  CHKERRQ(MatCreateTranspose(A,&C));
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&B)); /* B = A^T */
  CHKERRQ(MatMultEqual(C,B,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A^T*x != (x^T*A)^T");
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  if (!Aiselemental) {
    CHKERRQ(MatTranspose(B,MAT_INPLACE_MATRIX,&B));
    CHKERRQ(MatMultEqual(C,B,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"C*x != B*x");
  }
  CHKERRQ(MatDestroy(&B));

  /* Test B = C*A for matrix type transpose and seqdense */
  if (size == 1 && !Aiselemental) {
    CHKERRQ(MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&B));
    CHKERRQ(MatMatMultEqual(C,A,B,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B != C*A for matrix type transpose and seqdense");
    CHKERRQ(MatDestroy(&B));
  }
  CHKERRQ(MatDestroy(&C));

  /* Test MatMatMult() */
  if (Test_MatMatMult) {
#if !defined(PETSC_HAVE_ELEMENTAL)
    PetscCheckFalse(size > 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"This test requires ELEMENTAL");
#endif
    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&B)); /* B = A^T */
    CHKERRQ(MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C)); /* C = B*A = A^T*A */
    CHKERRQ(MatMatMult(B,A,MAT_REUSE_MATRIX,fill,&C));
    CHKERRQ(MatMatMultEqual(B,A,C,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B*A*x != C*x");

    /* Test MatDuplicate for matrix product */
    CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&D));

    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatDestroy(&B));
  }

  /* Test MatTransposeMatMult() */
  if (!Aiselemental) {
    CHKERRQ(MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&D)); /* D = A^T*A */
    CHKERRQ(MatTransposeMatMult(A,A,MAT_REUSE_MATRIX,fill,&D));
    CHKERRQ(MatTransposeMatMultEqual(A,A,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"D*x != A^T*A*x");

    /* Test MatDuplicate for matrix product */
    CHKERRQ(MatDuplicate(D,MAT_COPY_VALUES,&C));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatDestroy(&D));

    /* Test D*x = A^T*C*A*x, where C is in AIJ format */
    CHKERRQ(MatGetLocalSize(A,&am,&an));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
    if (size == 1) {
      CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,am,am));
    } else {
      CHKERRQ(MatSetSizes(C,am,am,PETSC_DECIDE,PETSC_DECIDE));
    }
    CHKERRQ(MatSetFromOptions(C));
    CHKERRQ(MatSetUp(C));
    CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
    v[0] = 1.0;
    for (i=rstart; i<rend; i++) {
      CHKERRQ(MatSetValues(C,1,&i,1,&i,v,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

    /* B = C*A, D = A^T*B */
    CHKERRQ(MatMatMult(C,A,MAT_INITIAL_MATRIX,1.0,&B));
    CHKERRQ(MatTransposeMatMult(A,B,MAT_INITIAL_MATRIX,fill,&D));
    CHKERRQ(MatTransposeMatMultEqual(A,B,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"D*x != A^T*B*x");

    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatDestroy(&B));
  }

  /* Test MatMatTransposeMult() */
  if (!Aiselemental) {
    PetscReal diff, scale;
    PetscInt  am, an, aM, aN;

    CHKERRQ(MatGetLocalSize(A, &am, &an));
    CHKERRQ(MatGetSize(A, &aM, &aN));
    CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)A),PETSC_DECIDE, an, aM + 10, aN, NULL, &B));
    CHKERRQ(MatSetRandom(B, NULL));
    CHKERRQ(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatMatTransposeMult(A,B,MAT_INITIAL_MATRIX,fill,&D)); /* D = A*A^T */

    /* Test MatDuplicate for matrix product */
    CHKERRQ(MatDuplicate(D,MAT_COPY_VALUES,&C));

    CHKERRQ(MatMatTransposeMult(A,B,MAT_REUSE_MATRIX,fill,&D));
    CHKERRQ(MatAXPY(C, -1., D, SAME_NONZERO_PATTERN));

    CHKERRQ(MatNorm(C, NORM_FROBENIUS, &diff));
    CHKERRQ(MatNorm(D, NORM_FROBENIUS, &scale));
    PetscCheckFalse(diff > PETSC_SMALL * scale,PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "MatMatTransposeMult() differs between MAT_INITIAL_MATRIX and MAT_REUSE_MATRIX");
    CHKERRQ(MatDestroy(&C));

    CHKERRQ(MatMatTransposeMultEqual(A,B,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"D*x != A^T*A*x");
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&B));

  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree(v));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      output_file: output/ex104.out

    test:
      suffix: 2
      nsize: 2
      output_file: output/ex104.out

    test:
      suffix: 3
      nsize: 4
      output_file: output/ex104.out
      args: -M 23 -N 31

    test:
      suffix: 4
      nsize: 4
      output_file: output/ex104.out
      args: -M 23 -N 31 -matmattransmult_mpidense_mpidense_via cyclic

    test:
      suffix: 5
      nsize: 4
      output_file: output/ex104.out
      args: -M 23 -N 31 -matmattransmult_mpidense_mpidense_via allgatherv

    test:
      suffix: 6
      args: -mat_type elemental
      requires: elemental
      output_file: output/ex104.out

    test:
      suffix: 7
      nsize: 2
      args: -mat_type elemental
      requires: elemental
      output_file: output/ex104.out

TEST*/
