
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
  PetscCall(MatShellGetContext(A,&user));
  PetscCall(MatMult(user->B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_User(Mat A,Vec X,Vec Y)
{
  User           user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&user));
  PetscCall(MatMultTranspose(user->B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MyFunction(void *ctx,Vec x,Vec y)
{
  User           user = (User) ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(user->B,x,y));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               S;
  User              user;
  Vec               base;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscNew(&user));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&user->B));
  PetscCall(MatSetUp(user->B));
  PetscCall(MatSetValues(user->B,2,inds,2,inds,avals,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(user->B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(user->B,&base,NULL));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S));
  PetscCall(MatSetUp(S));
  PetscCall(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User));
  PetscCall(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User));

  PetscCall(MatShellTestMult(S,MyFunction,base,user,NULL));
  PetscCall(MatShellTestMultTranspose(S,MyFunction,base,user,NULL));

  PetscCall(VecDestroy(&base));
  PetscCall(MatDestroy(&user->B));
  PetscCall(MatDestroy(&S));
  PetscCall(PetscFree(user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -mat_shell_test_mult_view -mat_shell_test_mult_transpose_view

TEST*/
