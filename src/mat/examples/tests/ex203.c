
static char help[] = "Tests incorrect use of MatDiagonalSet() for SHELL matrices\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatGetDiagonal(user->B,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_User(Mat A,Vec D,InsertMode is)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatDiagonalSet(user->B,D,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ErrorHandler(MPI_Comm comm,int line,const char *func,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_SELF,"[ERROR] %s: %s\n",func,mess);
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
  PetscBool         flag;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetValues(A,2,inds,2,inds,avals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);
  ierr = VecSetValues(X,2,inds,xvals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);

  ierr    = PetscNew(&user);CHKERRQ(ierr);
  user->B = A;

  ierr = MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_DIAGONAL_SET,(void (*)(void))MatDiagonalSet_User);CHKERRQ(ierr);

  ierr = MatShift(S,42);CHKERRQ(ierr);
  ierr = MatGetDiagonal(S,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDiagonalSet(S,X,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatGetDiagonal(S,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecEqual(X,Y,&flag);CHKERRQ(ierr);
  if (!flag) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Diagonal not set correctly!\n");CHKERRQ(ierr);
  }

  /* NOTE: This test case will fail */
  ierr = PetscPushErrorHandler(ErrorHandler, NULL);CHKERRQ(ierr);
  ierr = MatScale(S,42);CHKERRQ(ierr);
  ierr = MatGetDiagonal(S,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDiagonalSet(S,X,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatGetDiagonal(S,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPopErrorHandler();CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
