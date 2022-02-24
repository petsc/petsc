
static char help[] = "Tests incorrect use of MatDiagonalSet() for SHELL matrices\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatGetDiagonal(user->B,X));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  const PetscScalar xvals[] = {11,13};
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               A,S;
  Vec               X,Y;
  User              user;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetValues(A,2,inds,2,inds,avals,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,2,&X));
  CHKERRQ(VecSetValues(X,2,inds,xvals,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecDuplicate(X,&Y));

  CHKERRQ(PetscNew(&user));
  user->B = A;

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S));
  CHKERRQ(MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User));
  CHKERRQ(MatSetUp(S));

  CHKERRQ(MatShift(S,42));
  CHKERRQ(MatGetDiagonal(S,Y));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDiagonalSet(S,X,ADD_VALUES));
  CHKERRQ(MatGetDiagonal(S,Y));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatScale(S,42));
  CHKERRQ(MatGetDiagonal(S,Y));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(PetscFree(user));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
