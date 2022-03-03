#include <petscmat.h>

static char help[] = "Tests MatMat operations with MAT_REUSE_MATRIX and already allocated dense result.\n\n";

static PetscScalar MAGIC_NUMBER = 12345;

static PetscErrorCode CheckLocal(Mat A, Mat B, PetscScalar *a, PetscScalar *b)
{
  PetscBool      wA = PETSC_FALSE, wB = PETSC_FALSE;
  PetscBool      wAv = PETSC_FALSE, wBv = PETSC_FALSE;
  PetscInt       lda,i,j,m,n;

  PetscFunctionBegin;
  if (a) {
    const PetscScalar *Aa;
    CHKERRQ(MatDenseGetArrayRead(A,&Aa));
    wA   = (PetscBool)(a != Aa);
    CHKERRQ(MatDenseGetLDA(A,&lda));
    CHKERRQ(MatGetLocalSize(A,&m,&n));
    for (j=0;j<n;j++) {
      for (i=m;i<lda;i++) {
        if (Aa[j*lda +i] != MAGIC_NUMBER) wAv = PETSC_TRUE;
      }
    }
    CHKERRQ(MatDenseRestoreArrayRead(A,&Aa));
  }
  if (b) {
    const PetscScalar *Bb;
    CHKERRQ(MatDenseGetArrayRead(B,&Bb));
    wB   = (PetscBool)(b != Bb);
    CHKERRQ(MatDenseGetLDA(B,&lda));
    CHKERRQ(MatGetLocalSize(B,&m,&n));
    for (j=0;j<n;j++) {
      for (i=m;i<lda;i++) {
        if (Bb[j*lda +i] != MAGIC_NUMBER) wBv = PETSC_TRUE;
      }
    }
    CHKERRQ(MatDenseRestoreArrayRead(B,&Bb));
  }
  PetscCheckFalse(wA || wB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array in first Mat? %d, Wrong array in second Mat? %d",wA,wB);
  PetscCheckFalse(wAv || wBv,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong data in first Mat? %d, Wrong data in second Mat? %d",wAv,wBv);
  PetscFunctionReturn(0);
}

typedef struct {
  Mat A;
  Mat P;
  Mat R;
} proj_data;

PetscErrorCode proj_destroy(void *ctx)
{
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  PetscCheck(userdata,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing userdata");
  CHKERRQ(MatDestroy(&userdata->A));
  CHKERRQ(MatDestroy(&userdata->P));
  CHKERRQ(MatDestroy(&userdata->R));
  CHKERRQ(PetscFree(userdata));
  PetscFunctionReturn(0);
}

PetscErrorCode proj_mult(Mat S, Vec X, Vec Y)
{
  Mat            A,R,P;
  Vec            Ax,Ay;
  Vec            Px,Py;
  proj_data      *userdata;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&userdata));
  PetscCheck(userdata,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing userdata");
  A = userdata->A;
  R = userdata->R;
  P = userdata->P;
  PetscCheck(A,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing matrix");
  PetscCheckFalse(!R && !P,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing projectors");
  PetscCheckFalse(R && P,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Both projectors");
  CHKERRQ(MatCreateVecs(A,&Ax,&Ay));
  if (R) {
    CHKERRQ(MatCreateVecs(R,&Py,&Px));
  } else {
    CHKERRQ(MatCreateVecs(P,&Px,&Py));
  }
  CHKERRQ(VecCopy(X,Px));
  if (P) {
    CHKERRQ(MatMult(P,Px,Py));
  } else {
    CHKERRQ(MatMultTranspose(R,Px,Py));
  }
  CHKERRQ(VecCopy(Py,Ax));
  CHKERRQ(MatMult(A,Ax,Ay));
  CHKERRQ(VecCopy(Ay,Py));
  if (P) {
    CHKERRQ(MatMultTranspose(P,Py,Px));
  } else {
    CHKERRQ(MatMult(R,Py,Px));
  }
  CHKERRQ(VecCopy(Px,Y));
  CHKERRQ(VecDestroy(&Px));
  CHKERRQ(VecDestroy(&Py));
  CHKERRQ(VecDestroy(&Ax));
  CHKERRQ(VecDestroy(&Ay));
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultSymbolic(Mat S, Mat P, Mat PtAP, void** ctx)
{
  proj_data      *userdata;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&userdata));
  CHKERRQ(MatShellSetContext(PtAP,userdata));
  *ctx = (void *)userdata;
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultNumeric(Mat S, Mat P, Mat PtAP, void *ctx)
{
  Mat            A;
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(PetscObjectReference((PetscObject)A));
  CHKERRQ(PetscObjectReference((PetscObject)P));
  CHKERRQ(MatDestroy(&userdata->A));
  CHKERRQ(MatDestroy(&userdata->P));
  CHKERRQ(MatDestroy(&userdata->R));
  userdata->A = A;
  userdata->P = P;
  CHKERRQ(MatShellSetOperation(PtAP,MATOP_MULT,(void (*)(void))proj_mult));
  CHKERRQ(MatSetUp(PtAP));
  CHKERRQ(MatAssemblyBegin(PtAP,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(PtAP,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultSymbolic(Mat S, Mat R, Mat RARt, void **ctx)
{
  proj_data      *userdata;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&userdata));
  CHKERRQ(MatShellSetContext(RARt,userdata));
  *ctx = (void *)userdata;
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultNumeric(Mat S, Mat R, Mat RARt, void *ctx)
{
  Mat            A;
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(PetscObjectReference((PetscObject)A));
  CHKERRQ(PetscObjectReference((PetscObject)R));
  CHKERRQ(MatDestroy(&userdata->A));
  CHKERRQ(MatDestroy(&userdata->P));
  CHKERRQ(MatDestroy(&userdata->R));
  userdata->A = A;
  userdata->R = R;
  CHKERRQ(MatShellSetOperation(RARt,MATOP_MULT,(void (*)(void))proj_mult));
  CHKERRQ(MatSetUp(RARt));
  CHKERRQ(MatAssemblyBegin(RARt,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(RARt,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMatMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatTransposeShellMatMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatTransposeMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMatTransposeMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMatTransposeMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            X,B,A,Bt,T,T2,PtAP = NULL,RARt = NULL, R = NULL;
  Vec            r,l,rs,ls;
  PetscInt       m,n,k,M = 10,N = 10,K = 5, ldx = 3, ldb = 5, ldr = 4;
  const char     *deft = MATAIJ;
  char           mattype[256];
  PetscBool      flg,symm = PETSC_FALSE,testtt = PETSC_TRUE, testnest = PETSC_TRUE, testtranspose = PETSC_TRUE, testcircular = PETSC_FALSE, local = PETSC_TRUE;
  PetscBool      testhtranspose = PETSC_TRUE;
  PetscBool      xgpu = PETSC_FALSE, bgpu = PETSC_FALSE, testshellops = PETSC_FALSE, testproj = PETSC_TRUE, testrart = PETSC_TRUE, testmatmatt = PETSC_TRUE, testmattmat = PETSC_TRUE;
  PetscScalar    *dataX = NULL,*dataB = NULL, *dataR = NULL, *dataBt = NULL;
  PetscScalar    *aX,*aB,*aBt;
  PetscReal      err;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-local",&local,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ldx",&ldx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ldb",&ldb,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ldr",&ldr,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testtranspose",&testtranspose,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testnest",&testnest,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testtt",&testtt,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testcircular",&testcircular,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testshellops",&testshellops,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testproj",&testproj,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testrart",&testrart,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testmatmatt",&testmatmatt,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testmattmat",&testmattmat,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-xgpu",&xgpu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-magic_number",&MAGIC_NUMBER,NULL));
  if (M != N) testproj = PETSC_FALSE;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetRandom(A,NULL));
  if (M==N && symm) {
    Mat AT;

    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
    CHKERRQ(MatAXPY(A,1.0,AT,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatDestroy(&AT));
    CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  }
  CHKERRQ(MatViewFromOptions(A,NULL,"-A_init_view"));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsFList("-A_mat_type","Matrix type","MatSetType",MatList,deft,mattype,256,&flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    Mat A2;

    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    CHKERRQ(MatConvert(A,mattype,MAT_INPLACE_MATRIX,&A));
    CHKERRQ(MatMultEqual(A,A2,10,&flg));
    if (!flg) {
      Mat AE,A2E;

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with convert\n"));
      CHKERRQ(MatComputeOperator(A,MATDENSE,&AE));
      CHKERRQ(MatComputeOperator(A2,MATDENSE,&A2E));
      CHKERRQ(MatView(AE,NULL));
      CHKERRQ(MatView(A2E,NULL));
      CHKERRQ(MatAXPY(A2E,-1.0,A,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(A2E,NULL));
      CHKERRQ(MatDestroy(&A2E));
      CHKERRQ(MatDestroy(&AE));
    }
    CHKERRQ(MatDestroy(&A2));
  }
  CHKERRQ(MatViewFromOptions(A,NULL,"-A_view"));

  CHKERRQ(MatGetLocalSize(A,&m,&n));
  if (local) {
    PetscInt i;

    CHKERRQ(PetscMalloc1((m+ldx)*K,&dataX));
    CHKERRQ(PetscMalloc1((n+ldb)*K,&dataB));
    for (i=0;i<(m+ldx)*K;i++) dataX[i] = MAGIC_NUMBER;
    for (i=0;i<(n+ldb)*K;i++) dataB[i] = MAGIC_NUMBER;
  }
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,K,dataB,&B));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,dataX,&X));
  if (local) {
    CHKERRQ(MatDenseSetLDA(X,m+ldx));
    CHKERRQ(MatDenseSetLDA(B,n+ldb));
  }
  CHKERRQ(MatGetLocalSize(B,NULL,&k));
  if (local) {
    PetscInt i;

    CHKERRQ(PetscMalloc1((k+ldr)*N,&dataBt));
    for (i=0;i<(k+ldr)*N;i++) dataBt[i] = MAGIC_NUMBER;
  }
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,k,n,K,N,dataBt,&Bt));
  if (local) {
    CHKERRQ(MatDenseSetLDA(Bt,k+ldr));
  }

  /* store pointer to dense data for testing */
  CHKERRQ(MatDenseGetArrayRead(B,(const PetscScalar**)&dataB));
  CHKERRQ(MatDenseGetArrayRead(X,(const PetscScalar**)&dataX));
  CHKERRQ(MatDenseGetArrayRead(Bt,(const PetscScalar**)&dataBt));
  aX   = dataX;
  aB   = dataB;
  aBt  = dataBt;
  CHKERRQ(MatDenseRestoreArrayRead(Bt,(const PetscScalar**)&dataBt));
  CHKERRQ(MatDenseRestoreArrayRead(B,(const PetscScalar**)&dataB));
  CHKERRQ(MatDenseRestoreArrayRead(X,(const PetscScalar**)&dataX));
  if (local) {
    dataX  = aX;
    dataB  = aB;
    dataBt = aBt;
  }

  CHKERRQ(MatSetRandom(X,NULL));
  CHKERRQ(MatSetRandom(B,NULL));
  CHKERRQ(MatSetRandom(Bt,NULL));
  CHKERRQ(CheckLocal(X,NULL,aX,NULL));
  CHKERRQ(CheckLocal(Bt,B,aBt,aB));

  /* convert to CUDA if needed */
  if (bgpu) {
    CHKERRQ(MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B));
    CHKERRQ(MatConvert(Bt,MATDENSECUDA,MAT_INPLACE_MATRIX,&Bt));
  }
  if (xgpu) {
    CHKERRQ(MatConvert(X,MATDENSECUDA,MAT_INPLACE_MATRIX,&X));
  }
  CHKERRQ(CheckLocal(B,X,aB,aX));

  /* Test MatDenseGetSubMatrix */
  {
    Mat B2,T3,T4;

    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    CHKERRQ(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&T4));
    CHKERRQ(MatSetRandom(T4,NULL));
    CHKERRQ(MatAXPY(B2,1.0,T4,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDenseGetSubMatrix(B,PetscMin(1,K),PetscMin(2,K),&T));
    CHKERRQ(MatDenseGetSubMatrix(T4,PetscMin(1,K),PetscMin(2,K),&T2));
    CHKERRQ(MatDenseGetSubMatrix(B2,PetscMin(1,K),PetscMin(2,K),&T3));
    CHKERRQ(MatAXPY(T,1.0,T2,SAME_NONZERO_PATTERN));
    CHKERRQ(MatAXPY(T3,-1.0,T,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(T3,NORM_FROBENIUS,&err));
    if (err > PETSC_SMALL) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetSubMatrix\n"));
      CHKERRQ(MatView(T3,NULL));
    }
    CHKERRQ(MatDenseRestoreSubMatrix(B,&T));
    CHKERRQ(MatDenseRestoreSubMatrix(T4,&T2));
    CHKERRQ(MatDenseRestoreSubMatrix(B2,&T3));
    CHKERRQ(CheckLocal(B,NULL,aB,NULL));
    CHKERRQ(MatDestroy(&B2));
    CHKERRQ(MatDestroy(&T4));
  }

  /* Test reusing a previously allocated dense buffer */
  CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
  CHKERRQ(CheckLocal(B,X,aB,aX));
  CHKERRQ(MatMatMultEqual(A,B,X,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage\n"));
    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
    CHKERRQ(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
    CHKERRQ(MatView(T,NULL));
    CHKERRQ(MatDestroy(&T));
  }

  /* Test MatTransposeMat and MatMatTranspose */
  if (testmattmat) {
    CHKERRQ(MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatTransposeMatMultEqual(A,X,B,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTransposeMat)\n"));
      CHKERRQ(MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B));
      CHKERRQ(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
  }
  if (testmatmatt) {
    CHKERRQ(MatMatTransposeMult(A,Bt,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    CHKERRQ(CheckLocal(Bt,X,aBt,aX));
    CHKERRQ(MatMatTransposeMultEqual(A,Bt,X,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatMatTranspose)\n"));
      CHKERRQ(MatMatTransposeMult(A,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
  }

  /* Test projection operations (PtAP and RARt) */
  if (testproj) {
    CHKERRQ(MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP));
    CHKERRQ(MatPtAPMultEqual(A,B,PtAP,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP\n"));
      CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatTransposeMatMult(B,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2));
      CHKERRQ(MatAXPY(T2,-1.0,PtAP,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T2,NULL));
      CHKERRQ(MatDestroy(&T2));
      CHKERRQ(MatDestroy(&T));
    }
    CHKERRQ(PetscMalloc1((k+ldr)*M,&dataR));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,m,K,M,dataR,&R));
    CHKERRQ(MatDenseSetLDA(R,k+ldr));
    CHKERRQ(MatSetRandom(R,NULL));
    if (testrart) { /* fails for AIJCUSPARSE because RA operation is not defined */
      CHKERRQ(MatRARt(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RARt));
      CHKERRQ(MatRARtMultEqual(A,R,RARt,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with RARt\n"));
        CHKERRQ(MatMatTransposeMult(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
        CHKERRQ(MatMatMult(R,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2));
        CHKERRQ(MatAXPY(T2,-1.0,RARt,SAME_NONZERO_PATTERN));
        CHKERRQ(MatView(T2,NULL));
        CHKERRQ(MatDestroy(&T2));
        CHKERRQ(MatDestroy(&T));
      }
    }
  }

  /* Test MatDenseGetColumnVec and friends */
  CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
  CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
  CHKERRQ(MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&T2));
  for (k=0;k<K;k++) {
    Vec Xv,Tv,T2v;

    CHKERRQ(MatDenseGetColumnVecRead(X,k,&Xv));
    CHKERRQ(MatDenseGetColumnVec(T,k,&Tv));
    CHKERRQ(MatDenseGetColumnVecWrite(T2,k,&T2v));
    CHKERRQ(VecCopy(Xv,T2v));
    CHKERRQ(VecAXPY(Tv,-1.,Xv));
    CHKERRQ(MatDenseRestoreColumnVecRead(X,k,&Xv));
    CHKERRQ(MatDenseRestoreColumnVec(T,k,&Tv));
    CHKERRQ(MatDenseRestoreColumnVecWrite(T2,k,&T2v));
  }
  CHKERRQ(MatNorm(T,NORM_FROBENIUS,&err));
  if (err > PETSC_SMALL) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVec\n"));
    CHKERRQ(MatView(T,NULL));
  }
  CHKERRQ(MatAXPY(T2,-1.,X,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(T2,NORM_FROBENIUS,&err));
  if (err > PETSC_SMALL) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVecWrite\n"));
    CHKERRQ(MatView(T2,NULL));
  }
  CHKERRQ(MatDestroy(&T));
  CHKERRQ(MatDestroy(&T2));

  /* Test with MatShell */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&T));
  CHKERRQ(MatConvert(T,MATSHELL,MAT_INITIAL_MATRIX,&T2));
  CHKERRQ(MatDestroy(&T));

  /* scale matrix */
  CHKERRQ(MatScale(A,2.0));
  CHKERRQ(MatScale(T2,2.0));
  CHKERRQ(MatCreateVecs(A,&r,&l));
  CHKERRQ(VecSetRandom(r,NULL));
  CHKERRQ(VecSetRandom(l,NULL));
  CHKERRQ(MatCreateVecs(T2,&rs,&ls));
  CHKERRQ(VecCopy(r,rs));
  CHKERRQ(VecCopy(l,ls));
  if (testproj) {
    CHKERRQ(MatDiagonalScale(A,r,r));
    CHKERRQ(MatDiagonalScale(T2,rs,rs));
  } else {
    CHKERRQ(MatDiagonalScale(A,l,r));
    CHKERRQ(MatDiagonalScale(T2,ls,rs));
  }
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&T));
  CHKERRQ(MatAXPY(A,4.5,T,SAME_NONZERO_PATTERN));
  CHKERRQ(MatAXPY(T2,4.5,T,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatMultEqual(T2,A,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with MATSHELL (MatMult)\n"));
  }
  CHKERRQ(MatMultTransposeEqual(T2,A,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with MATSHELL (MatMultTranspose)\n"));
  }
  CHKERRQ(MatDestroy(&T));
  CHKERRQ(VecDestroy(&ls));
  CHKERRQ(VecDestroy(&rs));
  CHKERRQ(VecDestroy(&l));
  CHKERRQ(VecDestroy(&r));

  /* recompute projections, test reusage */
  if (PtAP) CHKERRQ(MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&PtAP));
  if (RARt) CHKERRQ(MatRARt(A,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&RARt));
  if (testshellops) { /* test callbacks for user defined MatProducts */
    CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,NULL,MATDENSE,MATDENSE));
    CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA));
    CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,NULL,MATDENSE,MATDENSE));
    CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA));
    CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_ABt,NULL,MyMatShellMatTransposeMultNumeric,NULL,MATDENSE,MATDENSE));
    CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_ABt,NULL,MyMatShellMatTransposeMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA));
    if (testproj) {
      CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,MyPtShellPMultSymbolic,MyPtShellPMultNumeric,proj_destroy,MATDENSE,MATSHELL));
      CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,MyPtShellPMultSymbolic,MyPtShellPMultNumeric,proj_destroy,MATDENSECUDA,MATSHELL));
      CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,MyRShellRtMultSymbolic,MyRShellRtMultNumeric,proj_destroy,MATDENSE,MATSHELL));
      CHKERRQ(MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,MyRShellRtMultSymbolic,MyRShellRtMultNumeric,proj_destroy,MATDENSECUDA,MATSHELL));
    }
  }
  CHKERRQ(CheckLocal(B,X,aB,aX));
  /* we either use the shell operations or the loop over columns code, applying the operator */
  CHKERRQ(MatMatMult(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
  CHKERRQ(CheckLocal(B,X,aB,aX));
  CHKERRQ(MatMatMultEqual(T2,B,X,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MATSHELL)\n"));
    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
    CHKERRQ(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
    CHKERRQ(MatView(T,NULL));
    CHKERRQ(MatDestroy(&T));
  }
  if (testproj) {
    CHKERRQ(MatPtAPMultEqual(T2,B,PtAP,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP (MATSHELL)\n"));
    }
    if (testshellops) { /* projections fail if the product operations are not specified */
      CHKERRQ(MatPtAP(T2,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatPtAP(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatPtAPMultEqual(T2,B,T,10,&flg));
      if (!flg) {
        Mat TE;

        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP (MATSHELL user defined)\n"));
        CHKERRQ(MatComputeOperator(T,MATDENSE,&TE));
        CHKERRQ(MatView(TE,NULL));
        CHKERRQ(MatView(PtAP,NULL));
        CHKERRQ(MatAXPY(TE,-1.0,PtAP,SAME_NONZERO_PATTERN));
        CHKERRQ(MatView(TE,NULL));
        CHKERRQ(MatDestroy(&TE));
      }
      CHKERRQ(MatDestroy(&T));
    }
    if (RARt) {
      CHKERRQ(MatRARtMultEqual(T2,R,RARt,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with RARt (MATSHELL)\n"));
      }
    }
    if (testshellops) {
      CHKERRQ(MatRARt(T2,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatRARt(T2,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatRARtMultEqual(T2,R,T,10,&flg));
      if (!flg) {
        Mat TE;

        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with RARt (MATSHELL user defined)\n"));
        CHKERRQ(MatComputeOperator(T,MATDENSE,&TE));
        CHKERRQ(MatView(TE,NULL));
        if (RARt) {
          CHKERRQ(MatView(RARt,NULL));
          CHKERRQ(MatAXPY(TE,-1.0,RARt,SAME_NONZERO_PATTERN));
          CHKERRQ(MatView(TE,NULL));
        }
        CHKERRQ(MatDestroy(&TE));
      }
      CHKERRQ(MatDestroy(&T));
    }
  }

  if (testmattmat) { /* we either use the shell operations or the loop over columns code applying the transposed operator */
    CHKERRQ(MatTransposeMatMult(T2,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatTransposeMatMultEqual(T2,X,B,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTranspose, MATSHELL)\n"));
      CHKERRQ(MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
  }
  if (testmatmatt && testshellops) { /* only when shell operations are set */
    CHKERRQ(MatMatTransposeMult(T2,Bt,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    CHKERRQ(CheckLocal(Bt,X,aBt,aX));
    CHKERRQ(MatMatTransposeMultEqual(T2,Bt,X,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatMatTranspose, MATSHELL)\n"));
      CHKERRQ(MatMatTransposeMult(A,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
  }
  CHKERRQ(MatDestroy(&T2));

  if (testnest) { /* test with MatNest */
    Mat NA;

    CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,1,NULL,1,NULL,&A,&NA));
    CHKERRQ(MatViewFromOptions(NA,NULL,"-NA_view"));
    CHKERRQ(MatMatMult(NA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatMatMultEqual(NA,B,X,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with Nest\n"));
      CHKERRQ(MatMatMult(NA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
    CHKERRQ(MatDestroy(&NA));
  }

  if (testtranspose) { /* test with Transpose */
    Mat TA;

    CHKERRQ(MatCreateTranspose(A,&TA));
    CHKERRQ(MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatMatMultEqual(TA,X,B,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n"));
      CHKERRQ(MatMatMult(TA,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
    CHKERRQ(MatDestroy(&TA));
  }

  if (testhtranspose) { /* test with Hermitian Transpose */
    Mat TA;

    CHKERRQ(MatCreateHermitianTranspose(A,&TA));
    CHKERRQ(MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatMatMultEqual(TA,X,B,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n"));
      CHKERRQ(MatMatMult(TA,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
    CHKERRQ(MatDestroy(&TA));
  }

  if (testtt) { /* test with Transpose(Transpose) */
    Mat TA, TTA;

    CHKERRQ(MatCreateTranspose(A,&TA));
    CHKERRQ(MatCreateTranspose(TA,&TTA));
    CHKERRQ(MatMatMult(TTA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatMatMultEqual(TTA,B,X,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose(Transpose)\n"));
      CHKERRQ(MatMatMult(TTA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
    CHKERRQ(MatDestroy(&TA));
    CHKERRQ(MatDestroy(&TTA));
  }

  if (testcircular) { /* test circular */
    Mat AB;

    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AB));
    CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    CHKERRQ(CheckLocal(B,X,aB,aX));
    if (M == N && N == K) {
      CHKERRQ(MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    } else {
      CHKERRQ(MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    }
    CHKERRQ(CheckLocal(B,X,aB,aX));
    CHKERRQ(MatDestroy(&AB));
  }

  /* Test by Pierre Jolivet */
  {
    Mat C,D,D2,AtA;
    CHKERRQ(MatCreateNormal(A,&AtA));
    CHKERRQ(MatDuplicate(X,MAT_DO_NOT_COPY_VALUES,&C));
    CHKERRQ(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&D));
    CHKERRQ(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&D2));
    CHKERRQ(MatSetRandom(B,NULL));
    CHKERRQ(MatSetRandom(C,NULL));
    CHKERRQ(MatSetRandom(D,NULL));
    CHKERRQ(MatSetRandom(D2,NULL));
    CHKERRQ(MatProductCreateWithMat(A,B,NULL,C));
    CHKERRQ(MatProductSetType(C,MATPRODUCT_AB));
    CHKERRQ(MatProductSetFromOptions(C));
    CHKERRQ(MatProductSymbolic(C));
    CHKERRQ(MatProductCreateWithMat(A,C,NULL,D));
    CHKERRQ(MatProductSetType(D, MATPRODUCT_AtB));
    CHKERRQ(MatProductSetFromOptions(D));
    CHKERRQ(MatProductSymbolic(D));
    CHKERRQ(MatProductNumeric(C));
    CHKERRQ(MatProductNumeric(D));
    CHKERRQ(MatMatMultEqual(AtA,B,D,10,&flg));
    if (!flg) {
      CHKERRQ(MatMatMult(AtA,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatAXPY(T,-1.0,D,SAME_NONZERO_PATTERN));
      CHKERRQ(MatView(T,NULL));
      CHKERRQ(MatDestroy(&T));
    }
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&D2));
    CHKERRQ(MatDestroy(&AtA));
  }

  CHKERRQ(MatDestroy(&X));
  CHKERRQ(MatDestroy(&Bt));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(MatDestroy(&PtAP));
  CHKERRQ(MatDestroy(&RARt));
  CHKERRQ(PetscFree(dataX));
  CHKERRQ(PetscFree(dataB));
  CHKERRQ(PetscFree(dataR));
  CHKERRQ(PetscFree(dataBt));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1
    args: -local {{0 1}} -testshellops

  test:
    output_file: output/ex70_1.out
    requires: cuda
    suffix: 1_cuda
    args: -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type {{seqaijcusparse seqaij}} -testshellops {{0 1}}

  test:
    output_file: output/ex70_1.out
    nsize: 2
    suffix: 1_par
    args: -local {{0 1}} -testmatmatt 0

  test:
    output_file: output/ex70_1.out
    requires: cuda
    nsize: 2
    suffix: 1_par_cuda
    args: -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type {{mpiaijcusparse mpiaij}} -testnest 0 -testmatmatt 0 -matmatmult_Bbn 3

  test:
    output_file: output/ex70_1.out
    suffix: 2
    nsize: 1
    args: -M {{7 11}} -N {{12 9}} -K {{1 3}} -local {{0 1}}

  testset:
    requires: cuda
    output_file: output/ex70_1.out
    nsize: 1
    args: -M 7 -N 9 -K 2 -local {{0 1}} -testnest 0 -A_mat_type {{seqdensecuda seqdense}} -xgpu {{0 1}} -bgpu {{0 1}}
    test:
      requires: !complex
      suffix: 2_cuda_real
    test:
      # complex+single gives a little bigger error in the MatDenseGetColumnVec test
      requires: complex !single
      suffix: 2_cuda_complex

  test:
    output_file: output/ex70_1.out
    suffix: 2_par
    nsize: 2
    args: -M {{7 11}} -N {{12 9}} -K {{1 3}} -local {{0 1}} -testcircular -testmatmatt 0

  test:
    requires: cuda
    output_file: output/ex70_1.out
    suffix: 2_par_cuda
    nsize: 2
    args: -M 11 -N 9 -K 1 -local {{0 1}} -testcircular 0 -A_mat_type mpiaijcusparse -xgpu -bgpu -testnest 0 -testmatmatt 0

  test:
    output_file: output/ex70_1.out
    suffix: 3
    nsize: {{1 3}}
    args: -M 13 -N 13 -K {{1 3}} -local {{0 1}} -A_mat_type sbaij -symm -testproj 0 -testmatmatt 0

  test:
    output_file: output/ex70_1.out
    suffix: 4
    nsize: 1
    args: -M 3 -N 3 -K 3 -local {{0 1}} -testcircular

  test:
    output_file: output/ex70_1.out
    suffix: 5
    nsize: {{2 4}}
    args: -M 3 -N 3 -K 3 -local {{0 1}} -testcircular -testmatmatt 0

  test:
    output_file: output/ex70_1.out
    suffix: 6
    nsize: 1
    args: -M {{1 3}} -N {{2 5}} -K {{1 2}} -local {{0 1}} -testcircular

  test:
    output_file: output/ex70_1.out
    suffix: 7
    nsize: 1
    args: -M 13 -N 13 -K {{1 3}} -local {{0 1}} -A_mat_type dense -testnest -testcircular

TEST*/
