
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
  PetscErrorCode ierr;
  PetscBool      CHOLESKY=PETSC_FALSE,TRIANGULAR=PETSC_FALSE,flg;
  PetscScalar    v;
  IS             row,col;
  MatFactorInfo  info;
  Vec            x,y,b,ytmp;
  PetscReal      norm2,tol = 100*PETSC_MACHINE_EPSILON;
  PetscRandom    rdm;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-lf",&lf,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,m*n,m*n,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  /* Create matrix C in seqaij format and sC in seqsbaij. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      J = Ii - n; if (J>=0)  CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + n; if (J<m*n) CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii - 1; if (J>=0)  CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + 1; if (J<m*n) CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatIsSymmetric(C,0.0,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"C is non-symmetric");
  CHKERRQ(MatConvert(C,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sC));

  /* Create vectors for error checking */
  CHKERRQ(MatCreateVecs(C,&x,&b));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecDuplicate(x,&ytmp));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecSetRandom(x,rdm));
  CHKERRQ(MatMult(C,x,b));

  CHKERRQ(MatGetOrdering(C,MATORDERINGNATURAL,&row,&col));

  /* Compute CHOLESKY or ICC factor sA */
  CHKERRQ(MatFactorInfoInitialize(&info));

  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-cholesky",&CHOLESKY));
  if (CHOLESKY) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Test CHOLESKY...\n"));
    CHKERRQ(MatGetFactor(sC,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sA));
    CHKERRQ(MatCholeskyFactorSymbolic(sA,sC,row,&info));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Test ICC...\n"));
    info.levels = lf;

    CHKERRQ(MatGetFactor(sC,MATSOLVERPETSC,MAT_FACTOR_ICC,&sA));
    CHKERRQ(MatICCFactorSymbolic(sA,sC,row,&info));
  }
  CHKERRQ(MatCholeskyFactorNumeric(sA,sC,&info));

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (CHOLESKY) {
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-triangular_solve",&TRIANGULAR));
    if (TRIANGULAR) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Test MatForwardSolve...\n"));
      CHKERRQ(MatForwardSolve(sA,b,ytmp));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Test MatBackwardSolve...\n"));
      CHKERRQ(MatBackwardSolve(sA,ytmp,y));
      CHKERRQ(VecAXPY(y,-1.0,x));
      CHKERRQ(VecNorm(y,NORM_2,&norm2));
      if (norm2 > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g\n",(double)norm2));
      }
    }
  }

  CHKERRQ(MatSolve(sA,b,y));
  CHKERRQ(MatDestroy(&sC));
  CHKERRQ(MatDestroy(&sA));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm2));
  if (lf == -1 && norm2 > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %" PetscInt_FMT ", residual %g\n",lf,(double)norm2));
  }

  /* Free data structures */
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(ISDestroy(&row));
  CHKERRQ(ISDestroy(&col));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&ytmp));
  CHKERRQ(VecDestroy(&b));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      output_file: output/ex128.out

   test:
      suffix: 2
      args: -cholesky -triangular_solve

TEST*/
