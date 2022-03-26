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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetType(A,MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  PetscCall(PetscRandomSetFromOptions(r));

  /* Set local matrix entries */
  PetscCall(MatGetOwnershipIS(A,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      PetscCall(PetscRandomGetValue(r,&rval));
      v[i*ncols+j] = rval;
    }
  }
  PetscCall(MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(PetscRandomDestroy(&r));

  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATELEMENTAL,&Aiselemental));

  /* Test MatCreateTranspose() and MatTranspose() */
  PetscCall(MatCreateTranspose(A,&C));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&B)); /* B = A^T */
  PetscCall(MatMultEqual(C,B,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A^T*x != (x^T*A)^T");
  PetscCall(MatDestroy(&B));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  if (!Aiselemental) {
    PetscCall(MatTranspose(B,MAT_INPLACE_MATRIX,&B));
    PetscCall(MatMultEqual(C,B,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"C*x != B*x");
  }
  PetscCall(MatDestroy(&B));

  /* Test B = C*A for matrix type transpose and seqdense */
  if (size == 1 && !Aiselemental) {
    PetscCall(MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&B));
    PetscCall(MatMatMultEqual(C,A,B,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B != C*A for matrix type transpose and seqdense");
    PetscCall(MatDestroy(&B));
  }
  PetscCall(MatDestroy(&C));

  /* Test MatMatMult() */
  if (Test_MatMatMult) {
#if !defined(PETSC_HAVE_ELEMENTAL)
    PetscCheckFalse(size > 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"This test requires ELEMENTAL");
#endif
    PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&B)); /* B = A^T */
    PetscCall(MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C)); /* C = B*A = A^T*A */
    PetscCall(MatMatMult(B,A,MAT_REUSE_MATRIX,fill,&C));
    PetscCall(MatMatMultEqual(B,A,C,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B*A*x != C*x");

    /* Test MatDuplicate for matrix product */
    PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&D));

    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&B));
  }

  /* Test MatTransposeMatMult() */
  if (!Aiselemental) {
    PetscCall(MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&D)); /* D = A^T*A */
    PetscCall(MatTransposeMatMult(A,A,MAT_REUSE_MATRIX,fill,&D));
    PetscCall(MatTransposeMatMultEqual(A,A,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"D*x != A^T*A*x");

    /* Test MatDuplicate for matrix product */
    PetscCall(MatDuplicate(D,MAT_COPY_VALUES,&C));
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&D));

    /* Test D*x = A^T*C*A*x, where C is in AIJ format */
    PetscCall(MatGetLocalSize(A,&am,&an));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
    if (size == 1) {
      PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,am,am));
    } else {
      PetscCall(MatSetSizes(C,am,am,PETSC_DECIDE,PETSC_DECIDE));
    }
    PetscCall(MatSetFromOptions(C));
    PetscCall(MatSetUp(C));
    PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
    v[0] = 1.0;
    for (i=rstart; i<rend; i++) {
      PetscCall(MatSetValues(C,1,&i,1,&i,v,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

    /* B = C*A, D = A^T*B */
    PetscCall(MatMatMult(C,A,MAT_INITIAL_MATRIX,1.0,&B));
    PetscCall(MatTransposeMatMult(A,B,MAT_INITIAL_MATRIX,fill,&D));
    PetscCall(MatTransposeMatMultEqual(A,B,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"D*x != A^T*B*x");

    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&B));
  }

  /* Test MatMatTransposeMult() */
  if (!Aiselemental) {
    PetscReal diff, scale;
    PetscInt  am, an, aM, aN;

    PetscCall(MatGetLocalSize(A, &am, &an));
    PetscCall(MatGetSize(A, &aM, &aN));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A),PETSC_DECIDE, an, aM + 10, aN, NULL, &B));
    PetscCall(MatSetRandom(B, NULL));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatMatTransposeMult(A,B,MAT_INITIAL_MATRIX,fill,&D)); /* D = A*A^T */

    /* Test MatDuplicate for matrix product */
    PetscCall(MatDuplicate(D,MAT_COPY_VALUES,&C));

    PetscCall(MatMatTransposeMult(A,B,MAT_REUSE_MATRIX,fill,&D));
    PetscCall(MatAXPY(C, -1., D, SAME_NONZERO_PATTERN));

    PetscCall(MatNorm(C, NORM_FROBENIUS, &diff));
    PetscCall(MatNorm(D, NORM_FROBENIUS, &scale));
    PetscCheckFalse(diff > PETSC_SMALL * scale,PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "MatMatTransposeMult() differs between MAT_INITIAL_MATRIX and MAT_REUSE_MATRIX");
    PetscCall(MatDestroy(&C));

    PetscCall(MatMatTransposeMultEqual(A,B,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"D*x != A^T*A*x");
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&B));

  }

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(v));
  PetscCall(PetscFinalize());
  return 0;
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
