static char help[] = "Test combinations of scalings, shifts and get diagonal of MATSHELL\n\n";

#include <petscmat.h>

static PetscErrorCode myMult(Mat S,Vec x,Vec y)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode myGetDiagonal(Mat S,Vec d)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode shiftandscale(Mat A,Vec *D)
{
  PetscErrorCode ierr;
  Vec            ll,d,rr;

  PetscFunctionBegin;
  ierr = MatCreateVecs(A,&ll,&rr);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&d,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(ll,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(rr,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(d,NULL);CHKERRQ(ierr);
  ierr = MatScale(A,3.0);CHKERRQ(ierr);
  ierr = MatShift(A,-4.0);CHKERRQ(ierr);
  ierr = MatScale(A,8.0);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A,d,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatShift(A,9.0);CHKERRQ(ierr);
  ierr = MatScale(A,8.0);CHKERRQ(ierr);
  ierr = VecSetRandom(ll,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(rr,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,ll,rr);CHKERRQ(ierr);
  ierr = MatShift(A,2.0);CHKERRQ(ierr);
  ierr = MatScale(A,11.0);CHKERRQ(ierr);
  ierr = VecSetRandom(d,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A,d,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecSetRandom(ll,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(rr,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,ll,rr);CHKERRQ(ierr);
  ierr = MatShift(A,5.0);CHKERRQ(ierr);
  ierr = MatScale(A,7.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,d);CHKERRQ(ierr);
  *D   = d;
  ierr = VecDestroy(&ll);CHKERRQ(ierr);
  ierr = VecDestroy(&rr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,Aij,B;
  Vec            Adiag,Aijdiag;
  PetscErrorCode ierr;
  PetscInt       m = 3;
  PetscReal      Aijnorm,Aijdiagnorm,Bnorm,dnorm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,m,7,NULL,6,NULL,&Aij);CHKERRQ(ierr);
  ierr = MatSetRandom(Aij,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(Aij,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,m,Aij,&A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void)) myMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void (*)(void)) myGetDiagonal);CHKERRQ(ierr);

  ierr = shiftandscale(A,&Adiag);CHKERRQ(ierr);
  ierr = MatComputeOperator(A,NULL,&B);CHKERRQ(ierr);
  ierr = shiftandscale(Aij,&Aijdiag);CHKERRQ(ierr);
  ierr = MatAXPY(Aij,-1.0,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(Aij,NORM_FROBENIUS,&Aijnorm);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_FROBENIUS,&Bnorm);CHKERRQ(ierr);
  if (Aijnorm/Bnorm > 100.0*PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Altered matrices do not match, norm of difference %g",(double)(Aijnorm/Bnorm));
  ierr = VecAXPY(Aijdiag,-1.0,Adiag);CHKERRQ(ierr);
  ierr = VecNorm(Adiag,NORM_2,&dnorm);CHKERRQ(ierr);
  ierr = VecNorm(Aijdiag,NORM_2,&Aijdiagnorm);CHKERRQ(ierr);
  if (Aijdiagnorm/dnorm > 100.0*PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Altered matrices diagonals do not match, norm of difference %g",(double)(Aijdiagnorm/dnorm));
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Aij);CHKERRQ(ierr);
  ierr = VecDestroy(&Adiag);CHKERRQ(ierr);
  ierr = VecDestroy(&Aijdiag);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      nsize: {{1 2 3 4}}

TEST*/
