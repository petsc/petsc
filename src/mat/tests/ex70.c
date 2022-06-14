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
    PetscCall(MatDenseGetArrayRead(A,&Aa));
    wA   = (PetscBool)(a != Aa);
    PetscCall(MatDenseGetLDA(A,&lda));
    PetscCall(MatGetLocalSize(A,&m,&n));
    for (j=0;j<n;j++) {
      for (i=m;i<lda;i++) {
        if (Aa[j*lda +i] != MAGIC_NUMBER) wAv = PETSC_TRUE;
      }
    }
    PetscCall(MatDenseRestoreArrayRead(A,&Aa));
  }
  if (b) {
    const PetscScalar *Bb;
    PetscCall(MatDenseGetArrayRead(B,&Bb));
    wB   = (PetscBool)(b != Bb);
    PetscCall(MatDenseGetLDA(B,&lda));
    PetscCall(MatGetLocalSize(B,&m,&n));
    for (j=0;j<n;j++) {
      for (i=m;i<lda;i++) {
        if (Bb[j*lda +i] != MAGIC_NUMBER) wBv = PETSC_TRUE;
      }
    }
    PetscCall(MatDenseRestoreArrayRead(B,&Bb));
  }
  PetscCheck(!wA && !wB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array in first Mat? %d, Wrong array in second Mat? %d",wA,wB);
  PetscCheck(!wAv && !wBv,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong data in first Mat? %d, Wrong data in second Mat? %d",wAv,wBv);
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
  PetscCall(MatDestroy(&userdata->A));
  PetscCall(MatDestroy(&userdata->P));
  PetscCall(MatDestroy(&userdata->R));
  PetscCall(PetscFree(userdata));
  PetscFunctionReturn(0);
}

PetscErrorCode proj_mult(Mat S, Vec X, Vec Y)
{
  Mat            A,R,P;
  Vec            Ax,Ay;
  Vec            Px,Py;
  proj_data      *userdata;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S,&userdata));
  PetscCheck(userdata,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing userdata");
  A = userdata->A;
  R = userdata->R;
  P = userdata->P;
  PetscCheck(A,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing matrix");
  PetscCheck(R || P,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing projectors");
  PetscCheck(!R || !P,PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Both projectors");
  PetscCall(MatCreateVecs(A,&Ax,&Ay));
  if (R) {
    PetscCall(MatCreateVecs(R,&Py,&Px));
  } else {
    PetscCall(MatCreateVecs(P,&Px,&Py));
  }
  PetscCall(VecCopy(X,Px));
  if (P) {
    PetscCall(MatMult(P,Px,Py));
  } else {
    PetscCall(MatMultTranspose(R,Px,Py));
  }
  PetscCall(VecCopy(Py,Ax));
  PetscCall(MatMult(A,Ax,Ay));
  PetscCall(VecCopy(Ay,Py));
  if (P) {
    PetscCall(MatMultTranspose(P,Py,Px));
  } else {
    PetscCall(MatMult(R,Py,Px));
  }
  PetscCall(VecCopy(Px,Y));
  PetscCall(VecDestroy(&Px));
  PetscCall(VecDestroy(&Py));
  PetscCall(VecDestroy(&Ax));
  PetscCall(VecDestroy(&Ay));
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultSymbolic(Mat S, Mat P, Mat PtAP, void** ctx)
{
  proj_data      *userdata;

  PetscFunctionBegin;
  PetscCall(PetscNew(&userdata));
  PetscCall(MatShellSetContext(PtAP,userdata));
  *ctx = (void *)userdata;
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultNumeric(Mat S, Mat P, Mat PtAP, void *ctx)
{
  Mat            A;
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(PetscObjectReference((PetscObject)A));
  PetscCall(PetscObjectReference((PetscObject)P));
  PetscCall(MatDestroy(&userdata->A));
  PetscCall(MatDestroy(&userdata->P));
  PetscCall(MatDestroy(&userdata->R));
  userdata->A = A;
  userdata->P = P;
  PetscCall(MatShellSetOperation(PtAP,MATOP_MULT,(void (*)(void))proj_mult));
  PetscCall(MatSetUp(PtAP));
  PetscCall(MatAssemblyBegin(PtAP,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(PtAP,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultSymbolic(Mat S, Mat R, Mat RARt, void **ctx)
{
  proj_data      *userdata;

  PetscFunctionBegin;
  PetscCall(PetscNew(&userdata));
  PetscCall(MatShellSetContext(RARt,userdata));
  *ctx = (void *)userdata;
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultNumeric(Mat S, Mat R, Mat RARt, void *ctx)
{
  Mat            A;
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(PetscObjectReference((PetscObject)A));
  PetscCall(PetscObjectReference((PetscObject)R));
  PetscCall(MatDestroy(&userdata->A));
  PetscCall(MatDestroy(&userdata->P));
  PetscCall(MatDestroy(&userdata->R));
  userdata->A = A;
  userdata->R = R;
  PetscCall(MatShellSetOperation(RARt,MATOP_MULT,(void (*)(void))proj_mult));
  PetscCall(MatSetUp(RARt));
  PetscCall(MatAssemblyBegin(RARt,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RARt,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMatMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  Mat            A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatTransposeShellMatMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  Mat            A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(MatTransposeMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMatTransposeMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  Mat            A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(MatMatTransposeMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
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

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-local",&local,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ldx",&ldx,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ldb",&ldb,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ldr",&ldr,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testtranspose",&testtranspose,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testnest",&testnest,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testtt",&testtt,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testcircular",&testcircular,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testshellops",&testshellops,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testproj",&testproj,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testrart",&testrart,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testmatmatt",&testmatmatt,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testmattmat",&testmattmat,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-xgpu",&xgpu,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-magic_number",&MAGIC_NUMBER,NULL));
  if (M != N) testproj = PETSC_FALSE;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetRandom(A,NULL));
  if (M==N && symm) {
    Mat AT;

    PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
    PetscCall(MatAXPY(A,1.0,AT,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&AT));
    PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  }
  PetscCall(MatViewFromOptions(A,NULL,"-A_init_view"));
  PetscOptionsBegin(PETSC_COMM_WORLD,"","","");
  PetscCall(PetscOptionsFList("-A_mat_type","Matrix type","MatSetType",MatList,deft,mattype,256,&flg));
  PetscOptionsEnd();
  if (flg) {
    Mat A2;

    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    PetscCall(MatConvert(A,mattype,MAT_INPLACE_MATRIX,&A));
    PetscCall(MatMultEqual(A,A2,10,&flg));
    if (!flg) {
      Mat AE,A2E;

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with convert\n"));
      PetscCall(MatComputeOperator(A,MATDENSE,&AE));
      PetscCall(MatComputeOperator(A2,MATDENSE,&A2E));
      PetscCall(MatView(AE,NULL));
      PetscCall(MatView(A2E,NULL));
      PetscCall(MatAXPY(A2E,-1.0,A,SAME_NONZERO_PATTERN));
      PetscCall(MatView(A2E,NULL));
      PetscCall(MatDestroy(&A2E));
      PetscCall(MatDestroy(&AE));
    }
    PetscCall(MatDestroy(&A2));
  }
  PetscCall(MatViewFromOptions(A,NULL,"-A_view"));

  PetscCall(MatGetLocalSize(A,&m,&n));
  if (local) {
    PetscInt i;

    PetscCall(PetscMalloc1((m+ldx)*K,&dataX));
    PetscCall(PetscMalloc1((n+ldb)*K,&dataB));
    for (i=0;i<(m+ldx)*K;i++) dataX[i] = MAGIC_NUMBER;
    for (i=0;i<(n+ldb)*K;i++) dataB[i] = MAGIC_NUMBER;
  }
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,K,dataB,&B));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,dataX,&X));
  if (local) {
    PetscCall(MatDenseSetLDA(X,m+ldx));
    PetscCall(MatDenseSetLDA(B,n+ldb));
  }
  PetscCall(MatGetLocalSize(B,NULL,&k));
  if (local) {
    PetscInt i;

    PetscCall(PetscMalloc1((k+ldr)*N,&dataBt));
    for (i=0;i<(k+ldr)*N;i++) dataBt[i] = MAGIC_NUMBER;
  }
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,k,n,K,N,dataBt,&Bt));
  if (local) PetscCall(MatDenseSetLDA(Bt,k+ldr));

  /* store pointer to dense data for testing */
  PetscCall(MatDenseGetArrayRead(B,(const PetscScalar**)&dataB));
  PetscCall(MatDenseGetArrayRead(X,(const PetscScalar**)&dataX));
  PetscCall(MatDenseGetArrayRead(Bt,(const PetscScalar**)&dataBt));
  aX   = dataX;
  aB   = dataB;
  aBt  = dataBt;
  PetscCall(MatDenseRestoreArrayRead(Bt,(const PetscScalar**)&dataBt));
  PetscCall(MatDenseRestoreArrayRead(B,(const PetscScalar**)&dataB));
  PetscCall(MatDenseRestoreArrayRead(X,(const PetscScalar**)&dataX));
  if (local) {
    dataX  = aX;
    dataB  = aB;
    dataBt = aBt;
  }

  PetscCall(MatSetRandom(X,NULL));
  PetscCall(MatSetRandom(B,NULL));
  PetscCall(MatSetRandom(Bt,NULL));
  PetscCall(CheckLocal(X,NULL,aX,NULL));
  PetscCall(CheckLocal(Bt,B,aBt,aB));

  /* convert to CUDA if needed */
  if (bgpu) {
    PetscCall(MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B));
    PetscCall(MatConvert(Bt,MATDENSECUDA,MAT_INPLACE_MATRIX,&Bt));
  }
  if (xgpu) {
    PetscCall(MatConvert(X,MATDENSECUDA,MAT_INPLACE_MATRIX,&X));
  }
  PetscCall(CheckLocal(B,X,aB,aX));

  /* Test MatDenseGetSubMatrix */
  {
    Mat B2,T3,T4;

    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    PetscCall(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&T4));
    PetscCall(MatSetRandom(T4,NULL));
    PetscCall(MatAXPY(B2,1.0,T4,SAME_NONZERO_PATTERN));
    PetscCall(MatDenseGetSubMatrix(B,PETSC_DECIDE,PETSC_DECIDE,PetscMin(1,K-1),PetscMin(2,K),&T));
    PetscCall(MatDenseGetSubMatrix(T4,PETSC_DECIDE,PETSC_DECIDE,PetscMin(1,K-1),PetscMin(2,K),&T2));
    PetscCall(MatDenseGetSubMatrix(B2,PETSC_DECIDE,PETSC_DECIDE,PetscMin(1,K-1),PetscMin(2,K),&T3));
    PetscCall(MatAXPY(T,1.0,T2,SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(T3,-1.0,T,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(T3,NORM_FROBENIUS,&err));
    if (err > PETSC_SMALL) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetSubMatrix\n"));
      PetscCall(MatView(T3,NULL));
    }
    PetscCall(MatDenseRestoreSubMatrix(B,&T));
    PetscCall(MatDenseRestoreSubMatrix(T4,&T2));
    PetscCall(MatDenseRestoreSubMatrix(B2,&T3));
    PetscCall(CheckLocal(B,NULL,aB,NULL));
    PetscCall(MatDestroy(&B2));
    PetscCall(MatDestroy(&T4));
    if (N >= 2) {
      PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
      PetscCall(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&T4));
      PetscCall(MatSetRandom(T4,NULL));
      PetscCall(MatAXPY(B2,1.0,T4,SAME_NONZERO_PATTERN));
      PetscCall(MatDenseGetSubMatrix(B,N-2,PETSC_DECIDE,PetscMin(1,K-1),PetscMin(2,K),&T));
      PetscCall(MatDenseGetSubMatrix(T4,N-2,PETSC_DECIDE,PetscMin(1,K-1),PetscMin(2,K),&T2));
      PetscCall(MatDenseGetSubMatrix(B2,N-2,PETSC_DECIDE,PetscMin(1,K-1),PetscMin(2,K),&T3));
      PetscCall(MatAXPY(T,1.0,T2,SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(T3,-1.0,T,SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(T3,NORM_FROBENIUS,&err));
      if (err > PETSC_SMALL) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetSubMatrix\n"));
        PetscCall(MatView(T3,NULL));
      }
      PetscCall(MatDenseRestoreSubMatrix(B,&T));
      PetscCall(MatDenseRestoreSubMatrix(T4,&T2));
      PetscCall(MatDenseRestoreSubMatrix(B2,&T3));
      PetscCall(CheckLocal(B,NULL,aB,NULL));
      PetscCall(MatDestroy(&B2));
      PetscCall(MatDestroy(&T4));
      PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
      PetscCall(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&T4));
      PetscCall(MatSetRandom(T4,NULL));
      PetscCall(MatAXPY(B2,1.0,T4,SAME_NONZERO_PATTERN));
      PetscCall(MatDenseGetSubMatrix(B,PETSC_DECIDE,2,PetscMin(1,K-1),PetscMin(2,K),&T));
      PetscCall(MatDenseGetSubMatrix(T4,PETSC_DECIDE,2,PetscMin(1,K-1),PetscMin(2,K),&T2));
      PetscCall(MatDenseGetSubMatrix(B2,PETSC_DECIDE,2,PetscMin(1,K-1),PetscMin(2,K),&T3));
      PetscCall(MatAXPY(T,1.0,T2,SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(T3,-1.0,T,SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(T3,NORM_FROBENIUS,&err));
      if (err > PETSC_SMALL) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetSubMatrix\n"));
        PetscCall(MatView(T3,NULL));
      }
      PetscCall(MatDenseRestoreSubMatrix(B,&T));
      PetscCall(MatDenseRestoreSubMatrix(T4,&T2));
      PetscCall(MatDenseRestoreSubMatrix(B2,&T3));
      PetscCall(CheckLocal(B,NULL,aB,NULL));
      PetscCall(MatDestroy(&B2));
      PetscCall(MatDestroy(&T4));
    }
  }

  /* Test reusing a previously allocated dense buffer */
  PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
  PetscCall(CheckLocal(B,X,aB,aX));
  PetscCall(MatMatMultEqual(A,B,X,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage\n"));
    PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
    PetscCall(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
    PetscCall(MatView(T,NULL));
    PetscCall(MatDestroy(&T));
  }

  /* Test MatTransposeMat and MatMatTranspose */
  if (testmattmat) {
    PetscCall(MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatTransposeMatMultEqual(A,X,B,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTransposeMat)\n"));
      PetscCall(MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B));
      PetscCall(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
  }
  if (testmatmatt) {
    PetscCall(MatMatTransposeMult(A,Bt,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    PetscCall(CheckLocal(Bt,X,aBt,aX));
    PetscCall(MatMatTransposeMultEqual(A,Bt,X,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatMatTranspose)\n"));
      PetscCall(MatMatTransposeMult(A,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
  }

  /* Test projection operations (PtAP and RARt) */
  if (testproj) {
    PetscCall(MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP));
    PetscCall(MatPtAPMultEqual(A,B,PtAP,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP\n"));
      PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatTransposeMatMult(B,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2));
      PetscCall(MatAXPY(T2,-1.0,PtAP,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T2,NULL));
      PetscCall(MatDestroy(&T2));
      PetscCall(MatDestroy(&T));
    }
    PetscCall(PetscMalloc1((k+ldr)*M,&dataR));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,m,K,M,dataR,&R));
    PetscCall(MatDenseSetLDA(R,k+ldr));
    PetscCall(MatSetRandom(R,NULL));
    if (testrart) { /* fails for AIJCUSPARSE because RA operation is not defined */
      PetscCall(MatRARt(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RARt));
      PetscCall(MatRARtMultEqual(A,R,RARt,10,&flg));
      if (!flg) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with RARt\n"));
        PetscCall(MatMatTransposeMult(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
        PetscCall(MatMatMult(R,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2));
        PetscCall(MatAXPY(T2,-1.0,RARt,SAME_NONZERO_PATTERN));
        PetscCall(MatView(T2,NULL));
        PetscCall(MatDestroy(&T2));
        PetscCall(MatDestroy(&T));
      }
    }
  }

  /* Test MatDenseGetColumnVec and friends */
  PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
  PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
  PetscCall(MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&T2));
  for (k=0;k<K;k++) {
    Vec Xv,Tv,T2v;

    PetscCall(MatDenseGetColumnVecRead(X,k,&Xv));
    PetscCall(MatDenseGetColumnVec(T,k,&Tv));
    PetscCall(MatDenseGetColumnVecWrite(T2,k,&T2v));
    PetscCall(VecCopy(Xv,T2v));
    PetscCall(VecAXPY(Tv,-1.,Xv));
    PetscCall(MatDenseRestoreColumnVecRead(X,k,&Xv));
    PetscCall(MatDenseRestoreColumnVec(T,k,&Tv));
    PetscCall(MatDenseRestoreColumnVecWrite(T2,k,&T2v));
  }
  PetscCall(MatNorm(T,NORM_FROBENIUS,&err));
  if (err > PETSC_SMALL) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVec\n"));
    PetscCall(MatView(T,NULL));
  }
  PetscCall(MatAXPY(T2,-1.,X,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(T2,NORM_FROBENIUS,&err));
  if (err > PETSC_SMALL) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVecWrite\n"));
    PetscCall(MatView(T2,NULL));
  }
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&T2));

  /* Test with MatShell */
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&T));
  PetscCall(MatConvert(T,MATSHELL,MAT_INITIAL_MATRIX,&T2));
  PetscCall(MatDestroy(&T));

  /* scale matrix */
  PetscCall(MatScale(A,2.0));
  PetscCall(MatScale(T2,2.0));
  PetscCall(MatCreateVecs(A,&r,&l));
  PetscCall(VecSetRandom(r,NULL));
  PetscCall(VecSetRandom(l,NULL));
  PetscCall(MatCreateVecs(T2,&rs,&ls));
  PetscCall(VecCopy(r,rs));
  PetscCall(VecCopy(l,ls));
  if (testproj) {
    PetscCall(MatDiagonalScale(A,r,r));
    PetscCall(MatDiagonalScale(T2,rs,rs));
  } else {
    PetscCall(MatDiagonalScale(A,l,r));
    PetscCall(MatDiagonalScale(T2,ls,rs));
  }
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&T));
  PetscCall(MatAXPY(A,4.5,T,SAME_NONZERO_PATTERN));
  PetscCall(MatAXPY(T2,4.5,T,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMultEqual(T2,A,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MATSHELL (MatMult)\n"));
  }
  PetscCall(MatMultTransposeEqual(T2,A,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with MATSHELL (MatMultTranspose)\n"));
  }
  PetscCall(MatDestroy(&T));
  PetscCall(VecDestroy(&ls));
  PetscCall(VecDestroy(&rs));
  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&r));

  /* recompute projections, test reusage */
  if (PtAP) PetscCall(MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&PtAP));
  if (RARt) PetscCall(MatRARt(A,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&RARt));
  if (testshellops) { /* test callbacks for user defined MatProducts */
    PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,NULL,MATDENSE,MATDENSE));
    PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA));
    PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,NULL,MATDENSE,MATDENSE));
    PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA));
    PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_ABt,NULL,MyMatShellMatTransposeMultNumeric,NULL,MATDENSE,MATDENSE));
    PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_ABt,NULL,MyMatShellMatTransposeMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA));
    if (testproj) {
      PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,MyPtShellPMultSymbolic,MyPtShellPMultNumeric,proj_destroy,MATDENSE,MATSHELL));
      PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,MyPtShellPMultSymbolic,MyPtShellPMultNumeric,proj_destroy,MATDENSECUDA,MATSHELL));
      PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,MyRShellRtMultSymbolic,MyRShellRtMultNumeric,proj_destroy,MATDENSE,MATSHELL));
      PetscCall(MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,MyRShellRtMultSymbolic,MyRShellRtMultNumeric,proj_destroy,MATDENSECUDA,MATSHELL));
    }
  }
  PetscCall(CheckLocal(B,X,aB,aX));
  /* we either use the shell operations or the loop over columns code, applying the operator */
  PetscCall(MatMatMult(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
  PetscCall(CheckLocal(B,X,aB,aX));
  PetscCall(MatMatMultEqual(T2,B,X,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MATSHELL)\n"));
    PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
    PetscCall(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
    PetscCall(MatView(T,NULL));
    PetscCall(MatDestroy(&T));
  }
  if (testproj) {
    PetscCall(MatPtAPMultEqual(T2,B,PtAP,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP (MATSHELL)\n"));
    }
    if (testshellops) { /* projections fail if the product operations are not specified */
      PetscCall(MatPtAP(T2,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatPtAP(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatPtAPMultEqual(T2,B,T,10,&flg));
      if (!flg) {
        Mat TE;

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP (MATSHELL user defined)\n"));
        PetscCall(MatComputeOperator(T,MATDENSE,&TE));
        PetscCall(MatView(TE,NULL));
        PetscCall(MatView(PtAP,NULL));
        PetscCall(MatAXPY(TE,-1.0,PtAP,SAME_NONZERO_PATTERN));
        PetscCall(MatView(TE,NULL));
        PetscCall(MatDestroy(&TE));
      }
      PetscCall(MatDestroy(&T));
    }
    if (RARt) {
      PetscCall(MatRARtMultEqual(T2,R,RARt,10,&flg));
      if (!flg) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with RARt (MATSHELL)\n"));
      }
    }
    if (testshellops) {
      PetscCall(MatRARt(T2,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatRARt(T2,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatRARtMultEqual(T2,R,T,10,&flg));
      if (!flg) {
        Mat TE;

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with RARt (MATSHELL user defined)\n"));
        PetscCall(MatComputeOperator(T,MATDENSE,&TE));
        PetscCall(MatView(TE,NULL));
        if (RARt) {
          PetscCall(MatView(RARt,NULL));
          PetscCall(MatAXPY(TE,-1.0,RARt,SAME_NONZERO_PATTERN));
          PetscCall(MatView(TE,NULL));
        }
        PetscCall(MatDestroy(&TE));
      }
      PetscCall(MatDestroy(&T));
    }
  }

  if (testmattmat) { /* we either use the shell operations or the loop over columns code applying the transposed operator */
    PetscCall(MatTransposeMatMult(T2,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatTransposeMatMultEqual(T2,X,B,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTranspose, MATSHELL)\n"));
      PetscCall(MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
  }
  if (testmatmatt && testshellops) { /* only when shell operations are set */
    PetscCall(MatMatTransposeMult(T2,Bt,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    PetscCall(CheckLocal(Bt,X,aBt,aX));
    PetscCall(MatMatTransposeMultEqual(T2,Bt,X,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatMatTranspose, MATSHELL)\n"));
      PetscCall(MatMatTransposeMult(A,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
  }
  PetscCall(MatDestroy(&T2));

  if (testnest) { /* test with MatNest */
    Mat NA;

    PetscCall(MatCreateNest(PETSC_COMM_WORLD,1,NULL,1,NULL,&A,&NA));
    PetscCall(MatViewFromOptions(NA,NULL,"-NA_view"));
    PetscCall(MatMatMult(NA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatMatMultEqual(NA,B,X,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with Nest\n"));
      PetscCall(MatMatMult(NA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
    PetscCall(MatDestroy(&NA));
  }

  if (testtranspose) { /* test with Transpose */
    Mat TA;

    PetscCall(MatCreateTranspose(A,&TA));
    PetscCall(MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatMatMultEqual(TA,X,B,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n"));
      PetscCall(MatMatMult(TA,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
    PetscCall(MatDestroy(&TA));
  }

  if (testhtranspose) { /* test with Hermitian Transpose */
    Mat TA;

    PetscCall(MatCreateHermitianTranspose(A,&TA));
    PetscCall(MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatMatMultEqual(TA,X,B,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n"));
      PetscCall(MatMatMult(TA,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
    PetscCall(MatDestroy(&TA));
  }

  if (testtt) { /* test with Transpose(Transpose) */
    Mat TA, TTA;

    PetscCall(MatCreateTranspose(A,&TA));
    PetscCall(MatCreateTranspose(TA,&TTA));
    PetscCall(MatMatMult(TTA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatMatMultEqual(TTA,B,X,10,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose(Transpose)\n"));
      PetscCall(MatMatMult(TTA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
    PetscCall(MatDestroy(&TA));
    PetscCall(MatDestroy(&TTA));
  }

  if (testcircular) { /* test circular */
    Mat AB;

    PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AB));
    PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X));
    PetscCall(CheckLocal(B,X,aB,aX));
    if (M == N && N == K) {
      PetscCall(MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    } else {
      PetscCall(MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
    }
    PetscCall(CheckLocal(B,X,aB,aX));
    PetscCall(MatDestroy(&AB));
  }

  /* Test by Pierre Jolivet */
  {
    Mat C,D,D2,AtA;
    PetscCall(MatCreateNormal(A,&AtA));
    PetscCall(MatDuplicate(X,MAT_DO_NOT_COPY_VALUES,&C));
    PetscCall(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&D));
    PetscCall(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&D2));
    PetscCall(MatSetRandom(B,NULL));
    PetscCall(MatSetRandom(C,NULL));
    PetscCall(MatSetRandom(D,NULL));
    PetscCall(MatSetRandom(D2,NULL));
    PetscCall(MatProductCreateWithMat(A,B,NULL,C));
    PetscCall(MatProductSetType(C,MATPRODUCT_AB));
    PetscCall(MatProductSetFromOptions(C));
    PetscCall(MatProductSymbolic(C));
    PetscCall(MatProductCreateWithMat(A,C,NULL,D));
    PetscCall(MatProductSetType(D, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(D));
    PetscCall(MatProductSymbolic(D));
    PetscCall(MatProductNumeric(C));
    PetscCall(MatProductNumeric(D));
    PetscCall(MatMatMultEqual(AtA,B,D,10,&flg));
    if (!flg) {
      PetscCall(MatMatMult(AtA,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      PetscCall(MatAXPY(T,-1.0,D,SAME_NONZERO_PATTERN));
      PetscCall(MatView(T,NULL));
      PetscCall(MatDestroy(&T));
    }
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&D2));
    PetscCall(MatDestroy(&AtA));
  }

  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Bt));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&PtAP));
  PetscCall(MatDestroy(&RARt));
  PetscCall(PetscFree(dataX));
  PetscCall(PetscFree(dataB));
  PetscCall(PetscFree(dataR));
  PetscCall(PetscFree(dataBt));
  PetscCall(PetscFinalize());
  return 0;
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
