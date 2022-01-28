
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-lf",&lf,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m*n,m*n,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* Create matrix C in seqaij format and sC in seqsbaij. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      J = Ii - n; if (J>=0)  {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = Ii + n; if (J<m*n) {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = Ii - 1; if (J>=0)  {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = Ii + 1; if (J<m*n) {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatIsSymmetric(C,0.0,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"C is non-symmetric");
  ierr = MatConvert(C,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sC);CHKERRQ(ierr);

  /* Create vectors for error checking */
  ierr = MatCreateVecs(C,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ytmp);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
  ierr = MatMult(C,x,b);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERINGNATURAL,&row,&col);CHKERRQ(ierr);

  /* Compute CHOLESKY or ICC factor sA */
  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);

  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;

  ierr = PetscOptionsHasName(NULL,NULL,"-cholesky",&CHOLESKY);CHKERRQ(ierr);
  if (CHOLESKY) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Test CHOLESKY...\n");CHKERRQ(ierr);
    ierr = MatGetFactor(sC,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sA);CHKERRQ(ierr);
    ierr = MatCholeskyFactorSymbolic(sA,sC,row,&info);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Test ICC...\n");CHKERRQ(ierr);
    info.levels = lf;

    ierr = MatGetFactor(sC,MATSOLVERPETSC,MAT_FACTOR_ICC,&sA);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(sA,sC,row,&info);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(sA,sC,&info);CHKERRQ(ierr);

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (CHOLESKY) {
    ierr = PetscOptionsHasName(NULL,NULL,"-triangular_solve",&TRIANGULAR);CHKERRQ(ierr);
    if (TRIANGULAR) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Test MatForwardSolve...\n");CHKERRQ(ierr);
      ierr = MatForwardSolve(sA,b,ytmp);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Test MatBackwardSolve...\n");CHKERRQ(ierr);
      ierr = MatBackwardSolve(sA,ytmp,y);CHKERRQ(ierr);
      ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
      if (norm2 > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g\n",(double)norm2);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatSolve(sA,b,y);CHKERRQ(ierr);
  ierr = MatDestroy(&sC);CHKERRQ(ierr);
  ierr = MatDestroy(&sA);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  if (lf == -1 && norm2 > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %" PetscInt_FMT ", residual %g\n",lf,(double)norm2);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = ISDestroy(&row);CHKERRQ(ierr);
  ierr = ISDestroy(&col);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&ytmp);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
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
