
static char help[] = "Tests MatShellTestMult()\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatMult_User(Mat A,Vec X,Vec Y)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatMult(user->B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_User(Mat A,Vec X,Vec Y)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatMultTranspose(user->B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MyFunction(void *ctx,Vec x,Vec y)
{
  User           user = (User) ctx;

  PetscFunctionBegin;
  CHKERRQ(MatMult(user->B,x,y));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               S;
  User              user;
  Vec               base;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscNew(&user));
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&user->B));
  CHKERRQ(MatSetUp(user->B));
  CHKERRQ(MatSetValues(user->B,2,inds,2,inds,avals,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(user->B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(user->B,&base,NULL));
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S));
  CHKERRQ(MatSetUp(S));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User));

  CHKERRQ(MatShellTestMult(S,MyFunction,base,user,NULL));
  CHKERRQ(MatShellTestMultTranspose(S,MyFunction,base,user,NULL));

  CHKERRQ(VecDestroy(&base));
  CHKERRQ(MatDestroy(&user->B));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(PetscFree(user));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -mat_shell_test_mult_view -mat_shell_test_mult_transpose_view

TEST*/
