
static char help[] = "Tests ILU and ICC factorization with and without matrix ordering on seqsbaij format. Modified from ex30.c\n\
  Input parameters are:\n\
  -lf <level> : level of fill for ILU (default is 0)\n\
  -lu : use full LU or Cholesky factorization\n\
  -m <value>,-n <value> : grid dimensions\n\
Note that most users should employ the KSP interface to the\n\
linear solvers instead of using the factorization routines\n\
directly.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,sC,sA;
  PetscInt       i,j,m = 5,n = 5,Ii,J,lf = 0;
  PetscBool      CHOLESKY=PETSC_FALSE,TRIANGULAR=PETSC_FALSE,flg;
  PetscScalar    v;
  IS             row,col;
  MatFactorInfo  info;
  Vec            x,y,b,ytmp;
  PetscReal      norm2,tol = 100*PETSC_MACHINE_EPSILON;
  PetscRandom    rdm;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-lf",&lf,NULL));

  PetscCall(MatCreate(PETSC_COMM_SELF,&C));
  PetscCall(MatSetSizes(C,m*n,m*n,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  /* Create matrix C in seqaij format and sC in seqsbaij. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      J = Ii - n; if (J>=0)  PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + n; if (J<m*n) PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii - 1; if (J>=0)  PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + 1; if (J<m*n) PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  PetscCall(MatIsSymmetric(C,0.0,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"C is non-symmetric");
  PetscCall(MatConvert(C,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sC));

  /* Create vectors for error checking */
  PetscCall(MatCreateVecs(C,&x,&b));
  PetscCall(VecDuplicate(x,&y));
  PetscCall(VecDuplicate(x,&ytmp));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecSetRandom(x,rdm));
  PetscCall(MatMult(C,x,b));

  PetscCall(MatGetOrdering(C,MATORDERINGNATURAL,&row,&col));

  /* Compute CHOLESKY or ICC factor sA */
  PetscCall(MatFactorInfoInitialize(&info));

  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;

  PetscCall(PetscOptionsHasName(NULL,NULL,"-cholesky",&CHOLESKY));
  if (CHOLESKY) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Test CHOLESKY...\n"));
    PetscCall(MatGetFactor(sC,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sA));
    PetscCall(MatCholeskyFactorSymbolic(sA,sC,row,&info));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Test ICC...\n"));
    info.levels = lf;

    PetscCall(MatGetFactor(sC,MATSOLVERPETSC,MAT_FACTOR_ICC,&sA));
    PetscCall(MatICCFactorSymbolic(sA,sC,row,&info));
  }
  PetscCall(MatCholeskyFactorNumeric(sA,sC,&info));

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (CHOLESKY) {
    PetscCall(PetscOptionsHasName(NULL,NULL,"-triangular_solve",&TRIANGULAR));
    if (TRIANGULAR) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Test MatForwardSolve...\n"));
      PetscCall(MatForwardSolve(sA,b,ytmp));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Test MatBackwardSolve...\n"));
      PetscCall(MatBackwardSolve(sA,ytmp,y));
      PetscCall(VecAXPY(y,-1.0,x));
      PetscCall(VecNorm(y,NORM_2,&norm2));
      if (norm2 > tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g\n",(double)norm2));
      }
    }
  }

  PetscCall(MatSolve(sA,b,y));
  PetscCall(MatDestroy(&sC));
  PetscCall(MatDestroy(&sA));
  PetscCall(VecAXPY(y,-1.0,x));
  PetscCall(VecNorm(y,NORM_2,&norm2));
  if (lf == -1 && norm2 > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %" PetscInt_FMT ", residual %g\n",lf,(double)norm2));
  }

  /* Free data structures */
  PetscCall(MatDestroy(&C));
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ytmp));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex128.out

   test:
      suffix: 2
      args: -cholesky -triangular_solve

TEST*/
