
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
  PetscCall(MatShellGetContext(A,&user));
  PetscCall(MatView(user->B,viewer));
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&user));
  PetscCall(MatGetDiagonal(user->B,X));
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
  PetscCall(PetscObjectGetType((PetscObject)A,&mattypename));
  PetscCall(PetscViewerASCIIPrintf(viewer,"\nMatrix of type: %s\n",mattypename));
  PetscCall(VecDuplicate(X,&W1));
  PetscCall(VecDuplicate(X,&W2));
  PetscCall(MatScale(A,31));
  PetscCall(MatShift(A,37));
  PetscCall(MatDiagonalScale(A,X,Y));
  PetscCall(MatScale(A,41));
  PetscCall(MatDiagonalScale(A,Y,Z));
  PetscCall(MatComputeOperator(A,MATDENSE,&E));

  PetscCall(MatView(E,viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Testing MatMult + MatMultTranspose\n"));
  PetscCall(MatMult(A,Z,W1));
  PetscCall(MatMultTranspose(A,W1,W2));
  PetscCall(VecView(W2,viewer));

  PetscCall(PetscViewerASCIIPrintf(viewer,"Testing MatMultAdd\n"));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,multadd,&diff));
  PetscCall(VecSet(W1,-1.0));
  PetscCall(MatMultAdd(A,W1,W1,W2));
  PetscCall(VecView(W2,viewer));
  PetscCall(VecAXPY(W2,-1.0,diff));
  PetscCall(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheck(nrm <= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultAdd(A,x,x,y) produces incorrect result");
#endif

  PetscCall(VecSet(W2,-1.0));
  PetscCall(MatMultAdd(A,W1,W2,W2));
  PetscCall(VecView(W2,viewer));
  PetscCall(VecAXPY(W2,-1.0,diff));
  PetscCall(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheck(nrm <= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultAdd(A,x,y,y) produces incorrect result");
#endif
  PetscCall(VecDestroy(&diff));

  PetscCall(PetscViewerASCIIPrintf(viewer,"Testing MatMultTransposeAdd\n"));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,multtadd,&diff));

  PetscCall(VecSet(W1,-1.0));
  PetscCall(MatMultTransposeAdd(A,W1,W1,W2));
  PetscCall(VecView(W2,viewer));
  PetscCall(VecAXPY(W2,-1.0,diff));
  PetscCall(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheck(nrm <= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultTransposeAdd(A,x,x,y) produces incorrect result");
#endif

  PetscCall(VecSet(W2,-1.0));
  PetscCall(MatMultTransposeAdd(A,W1,W2,W2));
  PetscCall(VecView(W2,viewer));
  PetscCall(VecAXPY(W2,-1.0,diff));
  PetscCall(VecNorm(W2,NORM_2,&nrm));
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
  PetscCheck(nrm <= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatMultTransposeAdd(A,x,y,y) produces incorrect result");
#endif
  PetscCall(VecDestroy(&diff));

  PetscCall(PetscViewerASCIIPrintf(viewer,"Testing MatGetDiagonal\n"));
  PetscCall(MatGetDiagonal(A,W2));
  PetscCall(VecView(W2,viewer));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,diag,&diff));
  PetscCall(VecAXPY(diff,-1.0,W2));
  PetscCall(VecNorm(diff,NORM_2,&nrm));
  PetscCheck(nrm <= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetDiagonal() produces incorrect result");
  PetscCall(VecDestroy(&diff));

  /* MATSHELL does not support MatDiagonalSet after MatScale */
  if (strncmp(mattypename, "shell", 5)) {
    PetscCall(MatDiagonalSet(A,X,INSERT_VALUES));
    PetscCall(MatGetDiagonal(A,W1));
    PetscCall(VecView(W1,viewer));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer,"MatDiagonalSet not tested on MATSHELL\n"));
  }

  PetscCall(MatDestroy(&E));
  PetscCall(VecDestroy(&W1));
  PetscCall(VecDestroy(&W2));
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValues(A,2,inds,2,inds,avals,INSERT_VALUES));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,2,&X));
  PetscCall(VecDuplicate(X,&Y));
  PetscCall(VecDuplicate(X,&Z));
  PetscCall(VecSetValues(X,2,inds,xvals,INSERT_VALUES));
  PetscCall(VecSetValues(Y,2,inds,yvals,INSERT_VALUES));
  PetscCall(VecSetValues(Z,2,inds,zvals,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyBegin(Z));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecAssemblyEnd(Y));
  PetscCall(VecAssemblyEnd(Z));

  PetscCall(PetscNew(&user));
  user->B = A;

  PetscCall(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,user,&S));
  PetscCall(MatSetUp(S));
  PetscCall(MatShellSetOperation(S,MATOP_VIEW,(void (*)(void))MatView_User));
  PetscCall(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User));
  PetscCall(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User));
  PetscCall(MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User));

  for (i=0; i<4; i++) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,1,1,&avals[i],&D[i]));
  }
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,D,&N));
  PetscCall(MatSetUp(N));

  PetscCall(TestMatrix(S,X,Y,Z));
  PetscCall(TestMatrix(A,X,Y,Z));
  PetscCall(TestMatrix(N,X,Y,Z));

  for (i=0; i<4; i++) PetscCall(MatDestroy(&D[i]));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&N));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(VecDestroy(&Z));
  PetscCall(PetscFree(user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
