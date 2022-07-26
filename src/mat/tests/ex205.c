static char help[] = "Tests MatCopy() for SHELL matrices\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat A;
};

static PetscErrorCode MatMult_User(Mat A,Vec X,Vec Y)
{
  User           user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&user));
  PetscCall(MatMult(user->A,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_User(Mat A,Mat B,MatStructure str)
{
  User           userA,userB;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&userA));
  if (userA) {
    PetscCall(PetscNew(&userB));
    PetscCall(MatDuplicate(userA->A,MAT_COPY_VALUES,&userB->A));
    PetscCall(MatShellSetContext(B, userB));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_User(Mat A)
{
  User           user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &user));
  if (user) {
    PetscCall(MatDestroy(&user->A));
    PetscCall(PetscFree(user));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  const PetscScalar xvals[] = {11,13},yvals[] = {17,19};
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               S1,S2;
  Vec               X,Y;
  User              user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  PetscCall(PetscNew(&user));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&user->A));
  PetscCall(MatSetUp(user->A));
  PetscCall(MatSetValues(user->A,2,inds,2,inds,avals,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(user->A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,2,&X));
  PetscCall(VecSetValues(X,2,inds,xvals,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecDuplicate(X,&Y));
  PetscCall(VecSetValues(Y,2,inds,yvals,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S1));
  PetscCall(MatSetUp(S1));
  PetscCall(MatShellSetOperation(S1,MATOP_MULT,(void (*)(void))MatMult_User));
  PetscCall(MatShellSetOperation(S1,MATOP_COPY,(void (*)(void))MatCopy_User));
  PetscCall(MatShellSetOperation(S1,MATOP_DESTROY,(void (*)(void))MatDestroy_User));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,NULL,&S2));
  PetscCall(MatSetUp(S2));
  PetscCall(MatShellSetOperation(S2,MATOP_MULT,(void (*)(void))MatMult_User));
  PetscCall(MatShellSetOperation(S2,MATOP_COPY,(void (*)(void))MatCopy_User));
  PetscCall(MatShellSetOperation(S2,MATOP_DESTROY,(void (*)(void))MatDestroy_User));

  PetscCall(MatScale(S1,31));
  PetscCall(MatShift(S1,37));
  PetscCall(MatDiagonalScale(S1,X,Y));
  PetscCall(MatCopy(S1,S2,SAME_NONZERO_PATTERN));
  PetscCall(MatMult(S1,X,Y));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatMult(S2,X,Y));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&S1));
  PetscCall(MatDestroy(&S2));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
