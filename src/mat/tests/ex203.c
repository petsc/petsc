
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
  PetscCall(MatShellGetContext(A,&user));
  PetscCall(MatGetDiagonal(user->B,X));
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValues(A,2,inds,2,inds,avals,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,2,&X));
  PetscCall(VecSetValues(X,2,inds,xvals,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecDuplicate(X,&Y));

  PetscCall(PetscNew(&user));
  user->B = A;

  PetscCall(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S));
  PetscCall(MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User));
  PetscCall(MatSetUp(S));

  PetscCall(MatShift(S,42));
  PetscCall(MatGetDiagonal(S,Y));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDiagonalSet(S,X,ADD_VALUES));
  PetscCall(MatGetDiagonal(S,Y));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatScale(S,42));
  PetscCall(MatGetDiagonal(S,Y));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&S));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(PetscFree(user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
