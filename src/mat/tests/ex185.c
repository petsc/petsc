static char help[] = "Tests MatCreateConstantDiagonal().\n"
"\n";

#include <petscmat.h>

/*T
    Concepts: Mat
T*/

int main(int argc, char **args)
{
  Vec            X, Y;
  Mat            A,B,Af;
  PetscBool      flg;
  PetscReal      xnorm,ynorm,anorm;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,20,20,3.0,&A));
  PetscCall(MatCreateVecs(A,&X,&Y));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecSetRandom(X,NULL));
  PetscCall(VecNorm(X,NORM_2,&xnorm));
  PetscCall(MatMult(A,X,Y));
  PetscCall(VecNorm(Y,NORM_2,&ynorm));
  PetscCheck(PetscAbsReal(ynorm - 3*xnorm) <= PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expected norm %g actual norm %g",(double)(3*xnorm),(double)ynorm);
  PetscCall(MatShift(A,5.0));
  PetscCall(MatScale(A,.5));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatNorm(A,NORM_FROBENIUS,&anorm));
  PetscCheck(PetscAbsReal(anorm - 4.0) <= PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expected norm 4.0 actual norm %g",(double)anorm);

  /* Convert to AIJ (exercises MatGetRow/MatRestoreRow) */
  PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));
  PetscCall(MatMultEqual(A,B,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MatMult\n"));
  PetscCall(MatMultAddEqual(A,B,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MatMultAdd\n"));
  PetscCall(MatMultTransposeEqual(A,B,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MatMultTranspose\n"));
  PetscCall(MatMultTransposeAddEqual(A,B,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MatMultTransposeAdd\n"));

  PetscCall(MatGetDiagonal(A,Y));
  PetscCall(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&Af));
  PetscCall(MatLUFactorSymbolic(Af,A,NULL,NULL,NULL));
  PetscCall(MatLUFactorNumeric(Af,A,NULL));
  PetscCall(MatSolve(Af,X,Y));
  PetscCall(VecNorm(Y,NORM_2,&ynorm));
  PetscCheck(PetscAbsReal(ynorm - xnorm/4) <= PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expected norm %g actual norm %g",(double)(.25*xnorm),(double)ynorm);

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Af));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2

TEST*/
