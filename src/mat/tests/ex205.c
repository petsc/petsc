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
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatMult(user->A,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_User(Mat A,Mat B,MatStructure str)
{
  User           userA,userB;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&userA));
  if (userA) {
    CHKERRQ(PetscNew(&userB));
    CHKERRQ(MatDuplicate(userA->A,MAT_COPY_VALUES,&userB->A));
    CHKERRQ(MatShellSetContext(B, userB));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_User(Mat A)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A, &user));
  if (user) {
    CHKERRQ(MatDestroy(&user->A));
    CHKERRQ(PetscFree(user));
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));

  CHKERRQ(PetscNew(&user));
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&user->A));
  CHKERRQ(MatSetUp(user->A));
  CHKERRQ(MatSetValues(user->A,2,inds,2,inds,avals,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(user->A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,2,&X));
  CHKERRQ(VecSetValues(X,2,inds,xvals,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecDuplicate(X,&Y));
  CHKERRQ(VecSetValues(Y,2,inds,yvals,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyEnd(Y));

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S1));
  CHKERRQ(MatSetUp(S1));
  CHKERRQ(MatShellSetOperation(S1,MATOP_MULT,(void (*)(void))MatMult_User));
  CHKERRQ(MatShellSetOperation(S1,MATOP_COPY,(void (*)(void))MatCopy_User));
  CHKERRQ(MatShellSetOperation(S1,MATOP_DESTROY,(void (*)(void))MatDestroy_User));
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,NULL,&S2));
  CHKERRQ(MatSetUp(S2));
  CHKERRQ(MatShellSetOperation(S2,MATOP_MULT,(void (*)(void))MatMult_User));
  CHKERRQ(MatShellSetOperation(S2,MATOP_COPY,(void (*)(void))MatCopy_User));
  CHKERRQ(MatShellSetOperation(S2,MATOP_DESTROY,(void (*)(void))MatDestroy_User));

  CHKERRQ(MatScale(S1,31));
  CHKERRQ(MatShift(S1,37));
  CHKERRQ(MatDiagonalScale(S1,X,Y));
  CHKERRQ(MatCopy(S1,S2,SAME_NONZERO_PATTERN));
  CHKERRQ(MatMult(S1,X,Y));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatMult(S2,X,Y));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&S1));
  CHKERRQ(MatDestroy(&S2));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
