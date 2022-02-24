static char help[] = "Tests MatCreateConstantDiagonal().\n"
"\n";

#include <petscmat.h>

/*T
    Concepts: Mat
T*/

int main(int argc, char **args)
{
  PetscErrorCode ierr;
  Vec            X, Y;
  Mat            A,B,Af;
  PetscBool      flg;
  PetscReal      xnorm,ynorm,anorm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,20,20,3.0,&A));
  CHKERRQ(MatCreateVecs(A,&X,&Y));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecSetRandom(X,NULL));
  CHKERRQ(VecNorm(X,NORM_2,&xnorm));
  CHKERRQ(MatMult(A,X,Y));
  CHKERRQ(VecNorm(Y,NORM_2,&ynorm));
  PetscCheckFalse(PetscAbsReal(ynorm - 3*xnorm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expected norm %g actual norm %g",(double)(3*xnorm),(double)ynorm);
  CHKERRQ(MatShift(A,5.0));
  CHKERRQ(MatScale(A,.5));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&anorm));
  PetscCheckFalse(PetscAbsReal(anorm - 4.0) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expected norm 4.0 actual norm %g",(double)anorm);

  /* Convert to AIJ (exercises MatGetRow/MatRestoreRow) */
  CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatMultEqual(A,B,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MatMult\n"));
  CHKERRQ(MatMultAddEqual(A,B,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MatMultAdd\n"));
  CHKERRQ(MatMultTransposeEqual(A,B,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MatMultTranspose\n"));
  CHKERRQ(MatMultTransposeAddEqual(A,B,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MatMultTransposeAdd\n"));

  CHKERRQ(MatGetDiagonal(A,Y));
  CHKERRQ(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&Af));
  CHKERRQ(MatLUFactorSymbolic(Af,A,NULL,NULL,NULL));
  CHKERRQ(MatLUFactorNumeric(Af,A,NULL));
  CHKERRQ(MatSolve(Af,X,Y));
  CHKERRQ(VecNorm(Y,NORM_2,&ynorm));
  PetscCheckFalse(PetscAbsReal(ynorm - xnorm/4) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expected norm %g actual norm %g",(double)(.25*xnorm),(double)ynorm);

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&Af));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    nsize: 2

TEST*/
