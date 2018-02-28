
static char help[] = "Tests MatShellTestMult()\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatMult_User(Mat A,Vec X,Vec Y)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatMult(user->B,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_User(Mat A,Vec X,Vec Y)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->B,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyFunction(void *ctx,Vec x,Vec y)
{
  User           user = (User) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(user->B,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               S;
  User              user;
  PetscErrorCode    ierr;
  Vec               base;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&user->B);CHKERRQ(ierr);
  ierr = MatSetUp(user->B);CHKERRQ(ierr);
  ierr = MatSetValues(user->B,2,inds,2,inds,avals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(user->B,&base,NULL);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User);CHKERRQ(ierr);

  ierr = MatShellTestMult(S,MyFunction,base,user,NULL);CHKERRQ(ierr);
  ierr = MatShellTestMultTranspose(S,MyFunction,base,user,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&base);CHKERRQ(ierr);
  ierr = MatDestroy(&user->B);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
     args: -mat_shell_test_mult_view -mat_shell_test_mult_transpose_view

TEST*/
