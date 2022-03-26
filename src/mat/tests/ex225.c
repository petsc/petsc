
static char help[] = "Test Hypre matrix APIs\n";

#include <petscmathypre.h>

int main(int argc,char **args)
{
  Mat            A, B, C;
  PetscReal      err;
  PetscInt       i,j,M = 20;
  PetscMPIInt    NP;
  MPI_Comm       comm;
  PetscInt       *rows;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm,&NP));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCheckFalse(M < 6,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Matrix has to have more than 6 columns");
  /* Hypre matrix */
  PetscCall(MatCreate(comm,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,M));
  PetscCall(MatSetType(B,MATHYPRE));
  PetscCall(MatHYPRESetPreallocation(B,9,NULL,9,NULL));

  /* PETSc AIJ matrix */
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A,9,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,9,NULL,9,NULL));

  /*Set Values */
  for (i=0; i<M; i++) {
    PetscInt    cols[] = {0,1,2,3,4,5};
    PetscScalar vals[6] = {0};
    PetscScalar value[] = {100};
    for (j=0; j<6; j++)
      vals[j] = ((PetscReal)j)/NP;

    PetscCall(MatSetValues(B,1,&i,6,cols,vals,ADD_VALUES));
    PetscCall(MatSetValues(B,1,&i,1,&i,value,ADD_VALUES));
    PetscCall(MatSetValues(A,1,&i,6,cols,vals,ADD_VALUES));
    PetscCall(MatSetValues(A,1,&i,1,&i,value,ADD_VALUES));
  }

  /* MAT_FLUSH_ASSEMBLY currently not supported */
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Compare A and B */
  PetscCall(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatSetValues %g",err);
  PetscCall(MatDestroy(&C));

  /* MatZeroRows */
  PetscCall(PetscMalloc1(M, &rows));
  for (i=0; i<M; i++) rows[i] = i;
  PetscCall(MatZeroRows(B, M, rows, 10.0, NULL, NULL));
  PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatZeroRows(A, M, rows, 10.0,NULL, NULL));
  PetscCall(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatZeroRows %g",err);
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFree(rows));

  /* Test MatZeroEntries */
  PetscCall(MatZeroEntries(B));
  PetscCall(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatZeroEntries %g",err);
  PetscCall(MatDestroy(&C));

  /* Insert Values */
  for (i=0; i<M; i++) {
    PetscInt    cols[] = {0,1,2,3,4,5};
    PetscScalar vals[6] = {0};
    PetscScalar value[] = {100};

    for (j=0; j<6; j++)
      vals[j] = ((PetscReal)j)/NP;

    PetscCall(MatSetValues(B,1,&i,6,cols,vals,INSERT_VALUES));
    PetscCall(MatSetValues(B,1,&i,1,&i,value,INSERT_VALUES));
    PetscCall(MatSetValues(A,1,&i,6,cols,vals,INSERT_VALUES));
    PetscCall(MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES));
  }

  /* MAT_FLUSH_ASSEMBLY currently not supported */
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Rows are not sorted with HYPRE so we need an intermediate sort
     They use a temporary buffer, so we can sort inplace the const memory */
  {
    const PetscInt    *idxA,*idxB;
    const PetscScalar *vA, *vB;
    PetscInt          rstart, rend, nzA, nzB;
    PetscInt          cols[] = {0,1,2,3,4,-5};
    PetscInt          *rows;
    PetscScalar       *valuesA, *valuesB;
    PetscBool         flg;

    PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
    for (i=rstart; i<rend; i++) {
      PetscCall(MatGetRow(A,i,&nzA,&idxA,&vA));
      PetscCall(MatGetRow(B,i,&nzB,&idxB,&vB));
      PetscCheckFalse(nzA!=nzB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetRow %" PetscInt_FMT, nzA-nzB);
      PetscCall(PetscSortIntWithScalarArray(nzB,(PetscInt*)idxB,(PetscScalar*)vB));
      PetscCall(PetscArraycmp(idxA,idxB,nzA,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetRow %" PetscInt_FMT " (indices)",i);
      PetscCall(PetscArraycmp(vA,vB,nzA,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetRow %" PetscInt_FMT " (values)",i);
      PetscCall(MatRestoreRow(A,i,&nzA,&idxA,&vA));
      PetscCall(MatRestoreRow(B,i,&nzB,&idxB,&vB));
    }

    PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
    PetscCall(PetscCalloc3((rend-rstart)*6,&valuesA,(rend-rstart)*6,&valuesB,rend-rstart,&rows));
    for (i=rstart; i<rend; i++) rows[i-rstart] =i;

    PetscCall(MatGetValues(A,rend-rstart,rows,6,cols,valuesA));
    PetscCall(MatGetValues(B,rend-rstart,rows,6,cols,valuesB));

    for (i=0; i<(rend-rstart); i++) {
      PetscCall(PetscArraycmp(valuesA + 6*i,valuesB + 6*i,6,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetValues %" PetscInt_FMT,i + rstart);
    }
    PetscCall(PetscFree3(valuesA,valuesB,rows));
  }

  /* Compare A and B */
  PetscCall(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatSetValues with INSERT_VALUES %g",err);

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: hypre

   test:
      suffix: 1
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)

   test:
      suffix: 2
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      output_file: output/ex225_1.out
      nsize: 2

TEST*/
