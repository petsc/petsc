static const char help[] = "Tests MatGetSchurComplement\n";

#include <petscksp.h>

PetscErrorCode Create(MPI_Comm comm,Mat *inA,IS *is0,IS *is1)
{
  Mat            A;
  PetscInt       r,rend,M;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  *inA = 0;
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetSizes(A,4,4,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&r,&rend));
  PetscCall(MatGetSize(A,&M,NULL));

  PetscCall(ISCreateStride(comm,2,r,1,is0));
  PetscCall(ISCreateStride(comm,2,r+2,1,is1));

  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  {
    PetscInt    rows[4],cols0[5],cols1[5],cols2[3],cols3[3];
    PetscScalar RR = 1000.*rank,vals0[5],vals1[4],vals2[3],vals3[3];

    rows[0]            = r;
    rows[1]            = r+1;
    rows[2]            = r+2;
    rows[3]            = r+3;

    cols0[0]           = r+0;
    cols0[1]           = r+1;
    cols0[2]           = r+3;
    cols0[3]           = (r+4)%M;
    cols0[4]           = (r+M-4)%M;

    cols1[0]           = r+1;
    cols1[1]           = r+2;
    cols1[2]           = (r+4+1)%M;
    cols1[3]           = (r+M-4+1)%M;

    cols2[0]           = r;
    cols2[1]           = r+2;
    cols2[2]           = (r+4+2)%M;

    cols3[0]           = r+1;
    cols3[1]           = r+3;
    cols3[2]           = (r+4+3)%M;

    vals0[0] = RR+1.;
    vals0[1] = RR+2.;
    vals0[2] = RR+3.;
    vals0[3] = RR+4.;
    vals0[4] = RR+5.;

    vals1[0] = RR+6.;
    vals1[1] = RR+7.;
    vals1[2] = RR+8.;
    vals1[3] = RR+9.;

    vals2[0] = RR+10.;
    vals2[1] = RR+11.;
    vals2[2] = RR+12.;

    vals3[0] = RR+13.;
    vals3[1] = RR+14.;
    vals3[2] = RR+15.;
    PetscCall(MatSetValues(A,1,&rows[0],5,cols0,vals0,INSERT_VALUES));
    PetscCall(MatSetValues(A,1,&rows[1],4,cols1,vals1,INSERT_VALUES));
    PetscCall(MatSetValues(A,1,&rows[2],3,cols2,vals2,INSERT_VALUES));
    PetscCall(MatSetValues(A,1,&rows[3],3,cols3,vals3,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY));
  *inA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode Destroy(Mat *A,IS *is0,IS *is1)
{
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(A));
  PetscCall(ISDestroy(is0));
  PetscCall(ISDestroy(is1));
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  Mat                        A,S = NULL,Sexplicit = NULL;
  MatSchurComplementAinvType ainv_type = MAT_SCHUR_COMPLEMENT_AINV_DIAG;
  IS                         is0,is1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex21","KSP");
  PetscCall(PetscOptionsEnum("-mat_schur_complement_ainv_type","Type of approximation for inv(A00) used when assembling Sp = A11 - A10 inv(A00) A01","MatSchurComplementAinvType",MatSchurComplementAinvTypes,(PetscEnum)ainv_type,(PetscEnum*)&ainv_type,NULL));
  PetscOptionsEnd();

  /* Test the Schur complement one way */
  PetscCall(Create(PETSC_COMM_WORLD,&A,&is0,&is1));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISView(is0,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISView(is1,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatGetSchurComplement(A,is0,is0,is1,is1,MAT_INITIAL_MATRIX,&S,ainv_type,MAT_IGNORE_MATRIX,NULL));
  PetscCall(MatSetFromOptions(S));
  PetscCall(MatComputeOperator(S,MATAIJ,&Sexplicit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nExplicit Schur complement of (0,0) in (1,1)\n"));
  PetscCall(MatView(Sexplicit,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(Destroy(&A,&is0,&is1));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&Sexplicit));

  /* And the other */
  PetscCall(Create(PETSC_COMM_WORLD,&A,&is0,&is1));
  PetscCall(MatGetSchurComplement(A,is1,is1,is0,is0,MAT_INITIAL_MATRIX,&S,ainv_type,MAT_IGNORE_MATRIX,NULL));
  PetscCall(MatSetFromOptions(S));
  PetscCall(MatComputeOperator(S,MATAIJ,&Sexplicit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nExplicit Schur complement of (1,1) in (0,0)\n"));
  PetscCall(MatView(Sexplicit,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(Destroy(&A,&is0,&is1));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&Sexplicit));

  /* This time just the preconditioning matrix. */
  PetscCall(Create(PETSC_COMM_WORLD,&A,&is0,&is1));
  PetscCall(MatGetSchurComplement(A,is0,is0,is1,is1,MAT_IGNORE_MATRIX,NULL,ainv_type,MAT_INITIAL_MATRIX,&S));
  PetscCall(MatSetFromOptions(S));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nPreconditioning Schur complement of (0,0) in (1,1)\n"));
  PetscCall(MatView(S,PETSC_VIEWER_STDOUT_WORLD));
  /* Modify and refresh */
  PetscCall(MatShift(A,1.));
  PetscCall(MatGetSchurComplement(A,is0,is0,is1,is1,MAT_IGNORE_MATRIX,NULL,ainv_type,MAT_REUSE_MATRIX,&S));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nAfter update\n"));
  PetscCall(MatView(S,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(Destroy(&A,&is0,&is1));
  PetscCall(MatDestroy(&S));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: diag_1
    args: -mat_schur_complement_ainv_type diag
    nsize: 1
  test:
    suffix: blockdiag_1
    args: -mat_schur_complement_ainv_type blockdiag
    nsize: 1
  test:
    suffix: diag_2
    args: -mat_schur_complement_ainv_type diag
    nsize: 2
  test:
    suffix: blockdiag_2
    args: -mat_schur_complement_ainv_type blockdiag
    nsize: 2
  test:
    # does not work with single because residual norm computed by GMRES recurrence formula becomes invalid
    requires: !single
    suffix: diag_3
    args: -mat_schur_complement_ainv_type diag
    nsize: 3
  test:
    # does not work with single because residual norm computed by GMRES recurrence formula becomes invalid
    requires: !single
    suffix: blockdiag_3
    args: -mat_schur_complement_ainv_type blockdiag
    nsize: 3
TEST*/
