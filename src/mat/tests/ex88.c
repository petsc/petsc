
static char help[] = "Tests MatShift(), MatScale(), and MatDiagonalScale() for SHELL and NEST matrices\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatView_User(Mat A,PetscViewer viewer)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatView(user->B,viewer));
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatGetDiagonal(user->B,X));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestMatrix(Mat A,Vec X,Vec Y,Vec Z)
{
  Vec            W1,W2,diff;
  Mat            E;
  const char     *mattypename;
  PetscViewer    viewer = PETSC_VIEWER_STDOUT_WORLD;
  PetscScalar    diag[2]     = { 2.9678190300000000e+08, 1.4173141580000000e+09};
  PetscScalar    multadd[2]  = {-6.8966198500000000e+08,-2.0310609940000000e+09};
  PetscScalar    multtadd[2] = {-9.1052873900000000e+08,-1.8101942400000000e+09};
  PetscReal      nrm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetType((PetscObject)A,&mattypename));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\nMatrix of type: %s\n",mattypename));
  CHKERRQ(VecDuplicate(X,&W1));
  CHKERRQ(VecDuplicate(X,&W2));
  CHKERRQ(MatScale(A,31));
  CHKERRQ(MatShift(A,37));
  CHKERRQ(MatDiagonalScale(A,X,Y));
  CHKERRQ(MatScale(A,41));
  CHKERRQ(MatDiagonalScale(A,Y,Z));
  CHKERRQ(MatComputeOperator(A,MATDENSE,&E));

  CHKERRQ(MatView(E,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing MatMult + MatMultTranspose\n"));
  CHKERRQ(MatMult(A,Z,W1));
  CHKERRQ(MatMultTranspose(A,W1,W2));
  CHKERRQ(VecView(W2,viewer));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing MatMultAdd\n"));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,multadd,&diff));
  CHKERRQ(VecSet(W1,-1.0));
  CHKERRQ(MatMultAdd(A,W1,W1,W2));
  CHKERRQ(VecView(W2,viewer));
  CHKERRQ(VecAXPY(W2,-1.0,diff));
  CHKERRQ(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultAdd(A,x,x,y) produces incorrect result");
#endif

  CHKERRQ(VecSet(W2,-1.0));
  CHKERRQ(MatMultAdd(A,W1,W2,W2));
  CHKERRQ(VecView(W2,viewer));
  CHKERRQ(VecAXPY(W2,-1.0,diff));
  CHKERRQ(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultAdd(A,x,y,y) produces incorrect result");
#endif
  CHKERRQ(VecDestroy(&diff));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing MatMultTransposeAdd\n"));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,multtadd,&diff));

  CHKERRQ(VecSet(W1,-1.0));
  CHKERRQ(MatMultTransposeAdd(A,W1,W1,W2));
  CHKERRQ(VecView(W2,viewer));
  CHKERRQ(VecAXPY(W2,-1.0,diff));
  CHKERRQ(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultTransposeAdd(A,x,x,y) produces incorrect result");
#endif

  CHKERRQ(VecSet(W2,-1.0));
  CHKERRQ(MatMultTransposeAdd(A,W1,W2,W2));
  CHKERRQ(VecView(W2,viewer));
  CHKERRQ(VecAXPY(W2,-1.0,diff));
  CHKERRQ(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultTransposeAdd(A,x,y,y) produces incorrect result");
#endif
  CHKERRQ(VecDestroy(&diff));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing MatGetDiagonal\n"));
  CHKERRQ(MatGetDiagonal(A,W2));
  CHKERRQ(VecView(W2,viewer));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,diag,&diff));
  CHKERRQ(VecAXPY(diff,-1.0,W2));
  CHKERRQ(VecNorm(diff,NORM_2,&nrm));
  PetscCheckFalse(nrm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetDiagonal() produces incorrect result");
  CHKERRQ(VecDestroy(&diff));

  /* MATSHELL does not support MatDiagonalSet after MatScale */
  if (strncmp(mattypename, "shell", 5)) {
    CHKERRQ(MatDiagonalSet(A,X,INSERT_VALUES));
    CHKERRQ(MatGetDiagonal(A,W1));
    CHKERRQ(VecView(W1,viewer));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"MatDiagonalSet not tested on MATSHELL\n"));
  }

  CHKERRQ(MatDestroy(&E));
  CHKERRQ(VecDestroy(&W1));
  CHKERRQ(VecDestroy(&W2));
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetValues(A,2,inds,2,inds,avals,INSERT_VALUES));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,2,&X));
  CHKERRQ(VecDuplicate(X,&Y));
  CHKERRQ(VecDuplicate(X,&Z));
  CHKERRQ(VecSetValues(X,2,inds,xvals,INSERT_VALUES));
  CHKERRQ(VecSetValues(Y,2,inds,yvals,INSERT_VALUES));
  CHKERRQ(VecSetValues(Z,2,inds,zvals,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyBegin(Z));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecAssemblyEnd(Y));
  CHKERRQ(VecAssemblyEnd(Z));

  CHKERRQ(PetscNew(&user));
  user->B = A;

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S));
  CHKERRQ(MatSetUp(S));
  CHKERRQ(MatShellSetOperation(S,MATOP_VIEW,(void (*)(void))MatView_User));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User));
  CHKERRQ(MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User));

  for (i=0; i<4; i++) {
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_WORLD,1,1,&avals[i],&D[i]));
  }
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,D,&N));
  CHKERRQ(MatSetUp(N));

  CHKERRQ(TestMatrix(S,X,Y,Z));
  CHKERRQ(TestMatrix(A,X,Y,Z));
  CHKERRQ(TestMatrix(N,X,Y,Z));

  for (i=0; i<4; i++) CHKERRQ(MatDestroy(&D[i]));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(MatDestroy(&N));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(VecDestroy(&Z));
  CHKERRQ(PetscFree(user));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
