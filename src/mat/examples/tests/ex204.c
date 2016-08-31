static char help[] = "Tests MatCopy() for SHELL matrices\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat A;
};

#undef __FUNCT__
#define __FUNCT__ "MatMult_User"
static PetscErrorCode MatMult_User(Mat A,Vec X,Vec Y)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatMult(user->A,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  const PetscScalar xvals[] = {11,13},yvals[] = {17,19};
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               S1,S2;
  Vec               X,Y;
  User              user1,user2;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscNew(&user1);CHKERRQ(ierr);
  ierr = PetscNew(&user2);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&user1->A);CHKERRQ(ierr);
  ierr = MatSetUp(user1->A);CHKERRQ(ierr);
  ierr = MatSetValues(user1->A,2,inds,2,inds,avals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user1->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user1->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDuplicate(user1->A,MAT_COPY_VALUES,&user2->A);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);
  ierr = VecSetValues(X,2,inds,xvals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = VecSetValues(Y,2,inds,yvals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user1,&S1);CHKERRQ(ierr);
  ierr = MatSetUp(S1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S1,MATOP_MULT,(void (*)(void))MatMult_User);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user2,&S2);CHKERRQ(ierr);
  ierr = MatSetUp(S2);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S2,MATOP_MULT,(void (*)(void))MatMult_User);CHKERRQ(ierr);

  ierr = MatScale(S1,31);CHKERRQ(ierr);
  ierr = MatShift(S1,37);CHKERRQ(ierr);
  ierr = MatDiagonalScale(S1,X,Y);CHKERRQ(ierr);
  ierr = MatCopy(S1,S2,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMult(S1,X,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatMult(S2,X,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&user1->A);CHKERRQ(ierr);
  ierr = MatDestroy(&user2->A);CHKERRQ(ierr);
  ierr = MatDestroy(&S1);CHKERRQ(ierr);
  ierr = MatDestroy(&S2);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = PetscFree(user1);CHKERRQ(ierr);
  ierr = PetscFree(user2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
