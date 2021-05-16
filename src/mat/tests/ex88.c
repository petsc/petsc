
static char help[] = "Tests MatShift(), MatScale(), and MatDiagonalScale() for SHELL and NEST matrices\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatView_User(Mat A,PetscViewer viewer)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatView(user->B,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatGetDiagonal(user->B,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestMatrix(Mat A,Vec X,Vec Y,Vec Z)
{
  PetscErrorCode ierr;
  Vec            W1,W2,diff;
  Mat            E;
  const char     *mattypename;
  PetscViewer    viewer = PETSC_VIEWER_STDOUT_WORLD;
  PetscScalar    diag[2]     = { 2.9678190300000000e+08, 1.4173141580000000e+09};
  PetscScalar    multadd[2]  = {-6.8966198500000000e+08,-2.0310609940000000e+09};
  PetscScalar    multtadd[2] = {-9.1052873900000000e+08,-1.8101942400000000e+09};
  PetscReal      nrm;

  PetscFunctionBegin;
  ierr = PetscObjectGetType((PetscObject)A,&mattypename);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\nMatrix of type: %s\n",mattypename);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&W1);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&W2);CHKERRQ(ierr);
  ierr = MatScale(A,31);CHKERRQ(ierr);
  ierr = MatShift(A,37);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,X,Y);CHKERRQ(ierr);
  ierr = MatScale(A,41);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,Y,Z);CHKERRQ(ierr);
  ierr = MatComputeOperator(A,MATDENSE,&E);CHKERRQ(ierr);

  ierr = MatView(E,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Testing MatMult + MatMultTranspose\n");CHKERRQ(ierr);
  ierr = MatMult(A,Z,W1);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,W1,W2);CHKERRQ(ierr);
  ierr = VecView(W2,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Testing MatMultAdd\n");CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,multadd,&diff);CHKERRQ(ierr);
  ierr = VecSet(W1,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(A,W1,W1,W2);CHKERRQ(ierr);
  ierr = VecView(W2,viewer);CHKERRQ(ierr);
  ierr = VecAXPY(W2,-1.0,diff);CHKERRQ(ierr);
  ierr = VecNorm(W2,NORM_2,&nrm);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  if (nrm > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultAdd(A,x,x,y) produces incorrect result");
#endif

  ierr = VecSet(W2,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(A,W1,W2,W2);CHKERRQ(ierr);
  ierr = VecView(W2,viewer);CHKERRQ(ierr);
  ierr = VecAXPY(W2,-1.0,diff);CHKERRQ(ierr);
  ierr = VecNorm(W2,NORM_2,&nrm);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  if (nrm > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultAdd(A,x,y,y) produces incorrect result");
#endif
  ierr = VecDestroy(&diff);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Testing MatMultTranposeAdd\n");CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,multtadd,&diff);CHKERRQ(ierr);

  ierr = VecSet(W1,-1.0);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(A,W1,W1,W2);CHKERRQ(ierr);
  ierr = VecView(W2,viewer);CHKERRQ(ierr);
  ierr = VecAXPY(W2,-1.0,diff);CHKERRQ(ierr);
  ierr = VecNorm(W2,NORM_2,&nrm);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  if (nrm > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultTranposeAdd(A,x,x,y) produces incorrect result");
#endif

  ierr = VecSet(W2,-1.0);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(A,W1,W2,W2);CHKERRQ(ierr);
  ierr = VecView(W2,viewer);CHKERRQ(ierr);
  ierr = VecAXPY(W2,-1.0,diff);CHKERRQ(ierr);
  ierr = VecNorm(W2,NORM_2,&nrm);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  if (nrm > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultTranposeAdd(A,x,y,y) produces incorrect result");
#endif
  ierr = VecDestroy(&diff);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Testing MatGetDiagonal\n");CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,W2);CHKERRQ(ierr);
  ierr = VecView(W2,viewer);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,diag,&diff);CHKERRQ(ierr);
  ierr = VecAXPY(diff,-1.0,W2);CHKERRQ(ierr);
  ierr = VecNorm(diff,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetDiagonal() produces incorrect result");
  ierr = VecDestroy(&diff);CHKERRQ(ierr);

  /* MATSHELL does not support MatDiagonalSet after MatScale */
  if (strncmp(mattypename, "shell", 5)) {
    ierr = MatDiagonalSet(A,X,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatGetDiagonal(A,W1);CHKERRQ(ierr);
    ierr = VecView(W1,viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"MatDiagonalSet not tested on MATSHELL\n");CHKERRQ(ierr);
  }

  ierr = MatDestroy(&E);CHKERRQ(ierr);
  ierr = VecDestroy(&W1);CHKERRQ(ierr);
  ierr = VecDestroy(&W2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  const PetscScalar xvals[] = {11,13},yvals[] = {17,19},zvals[] = {23,29};
  const PetscInt    inds[]  = {0,1};
  PetscScalar       avals[] = {2,3,5,7};
  Mat               A,S,D[4],N;
  Vec               X,Y,Z;
  User              user;
  PetscInt          i;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetValues(A,2,inds,2,inds,avals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Z);CHKERRQ(ierr);
  ierr = VecSetValues(X,2,inds,xvals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(Y,2,inds,yvals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(Z,2,inds,zvals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Z);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Z);CHKERRQ(ierr);

  ierr    = PetscNew(&user);CHKERRQ(ierr);
  user->B = A;

  ierr = MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_VIEW,(void (*)(void))MatView_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User);CHKERRQ(ierr);

  for (i=0; i<4; i++) {
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD,1,1,&avals[i],&D[i]);CHKERRQ(ierr);
  }
  ierr = MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,D,&N);CHKERRQ(ierr);
  ierr = MatSetUp(N);CHKERRQ(ierr);

  ierr = TestMatrix(S,X,Y,Z);CHKERRQ(ierr);
  ierr = TestMatrix(A,X,Y,Z);CHKERRQ(ierr);
  ierr = TestMatrix(N,X,Y,Z);CHKERRQ(ierr);

  for (i=0; i<4; i++) {ierr = MatDestroy(&D[i]);CHKERRQ(ierr);}
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = MatDestroy(&N);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Z);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
