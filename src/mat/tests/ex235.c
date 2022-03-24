static char help[] = "Test combinations of scalings, shifts and get diagonal of MATSHELL\n\n";

#include <petscmat.h>

static PetscErrorCode myMult(Mat S,Vec x,Vec y)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMult(A,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode myGetDiagonal(Mat S,Vec d)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatGetDiagonal(A,d));
  PetscFunctionReturn(0);
}

static PetscErrorCode shiftandscale(Mat A,Vec *D)
{
  Vec            ll,d,rr;

  PetscFunctionBegin;
  CHKERRQ(MatCreateVecs(A,&ll,&rr));
  CHKERRQ(MatCreateVecs(A,&d,NULL));
  CHKERRQ(VecSetRandom(ll,NULL));
  CHKERRQ(VecSetRandom(rr,NULL));
  CHKERRQ(VecSetRandom(d,NULL));
  CHKERRQ(MatScale(A,3.0));
  CHKERRQ(MatShift(A,-4.0));
  CHKERRQ(MatScale(A,8.0));
  CHKERRQ(MatDiagonalSet(A,d,ADD_VALUES));
  CHKERRQ(MatShift(A,9.0));
  CHKERRQ(MatScale(A,8.0));
  CHKERRQ(VecSetRandom(ll,NULL));
  CHKERRQ(VecSetRandom(rr,NULL));
  CHKERRQ(MatDiagonalScale(A,ll,rr));
  CHKERRQ(MatShift(A,2.0));
  CHKERRQ(MatScale(A,11.0));
  CHKERRQ(VecSetRandom(d,NULL));
  CHKERRQ(MatDiagonalSet(A,d,ADD_VALUES));
  CHKERRQ(VecSetRandom(ll,NULL));
  CHKERRQ(VecSetRandom(rr,NULL));
  CHKERRQ(MatDiagonalScale(A,ll,rr));
  CHKERRQ(MatShift(A,5.0));
  CHKERRQ(MatScale(A,7.0));
  CHKERRQ(MatGetDiagonal(A,d));
  *D   = d;
  CHKERRQ(VecDestroy(&ll));
  CHKERRQ(VecDestroy(&rr));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,Aij,B;
  Vec            Adiag,Aijdiag;
  PetscInt       m = 3;
  PetscReal      Aijnorm,Aijdiagnorm,Bnorm,dnorm;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,m,7,NULL,6,NULL,&Aij));
  CHKERRQ(MatSetRandom(Aij,NULL));
  CHKERRQ(MatSetOption(Aij,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,m,Aij,&A));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void (*)(void)) myMult));
  CHKERRQ(MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void (*)(void)) myGetDiagonal));

  CHKERRQ(shiftandscale(A,&Adiag));
  CHKERRQ(MatComputeOperator(A,NULL,&B));
  CHKERRQ(shiftandscale(Aij,&Aijdiag));
  CHKERRQ(MatAXPY(Aij,-1.0,B,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatNorm(Aij,NORM_FROBENIUS,&Aijnorm));
  CHKERRQ(MatNorm(B,NORM_FROBENIUS,&Bnorm));
  PetscCheckFalse(Aijnorm/Bnorm > 100.0*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Altered matrices do not match, norm of difference %g",(double)(Aijnorm/Bnorm));
  CHKERRQ(VecAXPY(Aijdiag,-1.0,Adiag));
  CHKERRQ(VecNorm(Adiag,NORM_2,&dnorm));
  CHKERRQ(VecNorm(Aijdiag,NORM_2,&Aijdiagnorm));
  PetscCheckFalse(Aijdiagnorm/dnorm > 100.0*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Altered matrices diagonals do not match, norm of difference %g",(double)(Aijdiagnorm/dnorm));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Aij));
  CHKERRQ(VecDestroy(&Adiag));
  CHKERRQ(VecDestroy(&Aijdiag));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: {{1 2 3 4}}

TEST*/
