#include <petscmat.h>

static char help[] = "Tests MatMat operations with MAT_REUSE_MATRIX and already allocated dense result.\n\n";

static PetscScalar MAGIC_NUMBER = 12345;

static PetscErrorCode CheckLocal(Mat A, Mat B, PetscScalar *a, PetscScalar *b)
{
  PetscErrorCode ierr;
  PetscBool      wA = PETSC_FALSE, wB = PETSC_FALSE;
  PetscBool      wAv = PETSC_FALSE, wBv = PETSC_FALSE;
  PetscInt       lda,i,j,m,n;

  PetscFunctionBegin;
  if (a) {
    const PetscScalar *Aa;
    ierr = MatDenseGetArrayRead(A,&Aa);CHKERRQ(ierr);
    wA   = (PetscBool)(a != Aa);
    ierr = MatDenseGetLDA(A,&lda);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      for (i=m;i<lda;i++) {
        if (Aa[j*lda +i] != MAGIC_NUMBER) wAv = PETSC_TRUE;
      }
    }
    ierr = MatDenseRestoreArrayRead(A,&Aa);CHKERRQ(ierr);
  }
  if (b) {
    const PetscScalar *Bb;
    ierr = MatDenseGetArrayRead(B,&Bb);CHKERRQ(ierr);
    wB   = (PetscBool)(b != Bb);
    ierr = MatDenseGetLDA(B,&lda);CHKERRQ(ierr);
    ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      for (i=m;i<lda;i++) {
        if (Bb[j*lda +i] != MAGIC_NUMBER) wBv = PETSC_TRUE;
      }
    }
    ierr = MatDenseRestoreArrayRead(B,&Bb);CHKERRQ(ierr);
  }
  if (wA || wB) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array in first Mat? %d, Wrong array in second Mat? %d",wA,wB);
  if (wAv || wBv) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong data in first Mat? %d, Wrong data in second Mat? %d",wAv,wBv);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!userdata) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing userdata");
  ierr = MatDestroy(&userdata->A);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->P);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->R);CHKERRQ(ierr);
  ierr = PetscFree(userdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode proj_mult(Mat S, Vec X, Vec Y)
{
  Mat            A,R,P;
  Vec            Ax,Ay;
  Vec            Px,Py;
  proj_data      *userdata;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&userdata);CHKERRQ(ierr);
  if (!userdata) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing userdata");
  A = userdata->A;
  R = userdata->R;
  P = userdata->P;
  if (!A) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing matrix");
  if (!R && !P) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing projectors");
  if (R && P) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Both projectors");
  ierr = MatCreateVecs(A,&Ax,&Ay);CHKERRQ(ierr);
  if (R) {
    ierr = MatCreateVecs(R,&Py,&Px);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(P,&Px,&Py);CHKERRQ(ierr);
  }
  ierr = VecCopy(X,Px);CHKERRQ(ierr);
  if (P) {
    ierr = MatMult(P,Px,Py);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(R,Px,Py);CHKERRQ(ierr);
  }
  ierr = VecCopy(Py,Ax);CHKERRQ(ierr);
  ierr = MatMult(A,Ax,Ay);CHKERRQ(ierr);
  ierr = VecCopy(Ay,Py);CHKERRQ(ierr);
  if (P) {
    ierr = MatMultTranspose(P,Py,Px);CHKERRQ(ierr);
  } else {
    ierr = MatMult(R,Py,Px);CHKERRQ(ierr);
  }
  ierr = VecCopy(Px,Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Px);CHKERRQ(ierr);
  ierr = VecDestroy(&Py);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Ay);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultSymbolic(Mat S, Mat P, Mat PtAP, void** ctx)
{
  PetscErrorCode ierr;
  proj_data      *userdata;

  PetscFunctionBegin;
  ierr = PetscNew(&userdata);CHKERRQ(ierr);
  ierr = MatShellSetContext(PtAP,(void*)userdata);CHKERRQ(ierr);
  *ctx = (void *)userdata;
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultNumeric(Mat S, Mat P, Mat PtAP, void *ctx)
{
  Mat            A;
  PetscErrorCode ierr;
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->A);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->P);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->R);CHKERRQ(ierr);
  userdata->A = A;
  userdata->P = P;
  ierr = MatShellSetOperation(PtAP,MATOP_MULT,(void (*)(void))proj_mult);CHKERRQ(ierr);
  ierr = MatSetUp(PtAP);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(PtAP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(PtAP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultSymbolic(Mat S, Mat R, Mat RARt, void **ctx)
{
  PetscErrorCode ierr;
  proj_data      *userdata;

  PetscFunctionBegin;
  ierr = PetscNew(&userdata);CHKERRQ(ierr);
  ierr = MatShellSetContext(RARt,(void*)userdata);CHKERRQ(ierr);
  *ctx = (void *)userdata;
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultNumeric(Mat S, Mat R, Mat RARt, void *ctx)
{
  Mat            A;
  PetscErrorCode ierr;
  proj_data      *userdata = (proj_data*)ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)R);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->A);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->P);CHKERRQ(ierr);
  ierr = MatDestroy(&userdata->R);CHKERRQ(ierr);
  userdata->A = A;
  userdata->R = R;
  ierr = MatShellSetOperation(RARt,MATOP_MULT,(void (*)(void))proj_mult);CHKERRQ(ierr);
  ierr = MatSetUp(RARt);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(RARt,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RARt,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMatMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatTransposeShellMatMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMatTransposeMultNumeric(Mat S, Mat B, Mat C, void *ctx)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = MatMatTransposeMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-local",&local,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldx",&ldx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldb",&ldb,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldr",&ldr,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testtranspose",&testtranspose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testnest",&testnest,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testtt",&testtt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testcircular",&testcircular,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testshellops",&testshellops,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testproj",&testproj,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testrart",&testrart,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testmatmatt",&testmatmatt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testmattmat",&testmattmat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-xgpu",&xgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-magic_number",&MAGIC_NUMBER,NULL);CHKERRQ(ierr);
  if (M != N) testproj = PETSC_FALSE;

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);
  if (M==N && symm) {
    Mat AT;

    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&AT);CHKERRQ(ierr);
    ierr = MatAXPY(A,1.0,AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&AT);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(A,NULL,"-A_init_view");CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-A_mat_type","Matrix type","MatSetType",MatList,deft,mattype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    Mat A2;

    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatConvert(A,mattype,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatMultEqual(A,A2,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      Mat AE,A2E;

      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with convert\n");CHKERRQ(ierr);
      ierr = MatComputeOperator(A,MATDENSE,&AE);CHKERRQ(ierr);
      ierr = MatComputeOperator(A2,MATDENSE,&A2E);CHKERRQ(ierr);
      ierr = MatView(AE,NULL);CHKERRQ(ierr);
      ierr = MatView(A2E,NULL);CHKERRQ(ierr);
      ierr = MatAXPY(A2E,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(A2E,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&A2E);CHKERRQ(ierr);
      ierr = MatDestroy(&AE);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (local) {
    PetscInt i;

    ierr = PetscMalloc1((m+ldx)*K,&dataX);CHKERRQ(ierr);
    ierr = PetscMalloc1((n+ldb)*K,&dataB);CHKERRQ(ierr);
    for (i=0;i<(m+ldx)*K;i++) dataX[i] = MAGIC_NUMBER;
    for (i=0;i<(n+ldb)*K;i++) dataB[i] = MAGIC_NUMBER;
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,K,dataB,&B);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,dataX,&X);CHKERRQ(ierr);
  if (local) {
    ierr = MatDenseSetLDA(X,m+ldx);CHKERRQ(ierr);
    ierr = MatDenseSetLDA(B,n+ldb);CHKERRQ(ierr);
  }
  ierr = MatGetLocalSize(B,NULL,&k);CHKERRQ(ierr);
  if (local) {
    PetscInt i;

    ierr = PetscMalloc1((k+ldr)*N,&dataBt);CHKERRQ(ierr);
    for (i=0;i<(k+ldr)*N;i++) dataBt[i] = MAGIC_NUMBER;
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,k,n,K,N,dataBt,&Bt);CHKERRQ(ierr);
  if (local) {
    ierr = MatDenseSetLDA(Bt,k+ldr);CHKERRQ(ierr);
  }

  /* store pointer to dense data for testing */
  ierr = MatDenseGetArrayRead(B,(const PetscScalar**)&dataB);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(X,(const PetscScalar**)&dataX);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(Bt,(const PetscScalar**)&dataBt);CHKERRQ(ierr);
  aX   = dataX;
  aB   = dataB;
  aBt  = dataBt;
  ierr = MatDenseRestoreArrayRead(Bt,(const PetscScalar**)&dataBt);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,(const PetscScalar**)&dataB);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(X,(const PetscScalar**)&dataX);CHKERRQ(ierr);
  if (local) {
    dataX  = aX;
    dataB  = aB;
    dataBt = aBt;
  }

  ierr = MatSetRandom(X,NULL);CHKERRQ(ierr);
  ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
  ierr = MatSetRandom(Bt,NULL);CHKERRQ(ierr);
  ierr = CheckLocal(X,NULL,aX,NULL);CHKERRQ(ierr);
  ierr = CheckLocal(Bt,B,aBt,aB);CHKERRQ(ierr);

  /* convert to CUDA if needed */
  if (bgpu) {
    ierr = MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatConvert(Bt,MATDENSECUDA,MAT_INPLACE_MATRIX,&Bt);CHKERRQ(ierr);
  }
  if (xgpu) {
    ierr = MatConvert(X,MATDENSECUDA,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);

  /* Test MatDenseGetSubMatrix */
  {
    Mat B2,T3,T4;

    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&T4);CHKERRQ(ierr);
    ierr = MatSetRandom(T4,NULL);CHKERRQ(ierr);
    ierr = MatAXPY(B2,1.0,T4,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDenseGetSubMatrix(B,PetscMin(1,K),PetscMin(2,K),&T);CHKERRQ(ierr);
    ierr = MatDenseGetSubMatrix(T4,PetscMin(1,K),PetscMin(2,K),&T2);CHKERRQ(ierr);
    ierr = MatDenseGetSubMatrix(B2,PetscMin(1,K),PetscMin(2,K),&T3);CHKERRQ(ierr);
    ierr = MatAXPY(T,1.0,T2,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(T3,-1.0,T,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(T3,NORM_FROBENIUS,&err);CHKERRQ(ierr);
    if (err > PETSC_SMALL) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetSubMatrix\n");CHKERRQ(ierr);
      ierr = MatView(T3,NULL);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreSubMatrix(B,&T);CHKERRQ(ierr);
    ierr = MatDenseRestoreSubMatrix(T4,&T2);CHKERRQ(ierr);
    ierr = MatDenseRestoreSubMatrix(B2,&T3);CHKERRQ(ierr);
    ierr = CheckLocal(B,NULL,aB,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);
    ierr = MatDestroy(&T4);CHKERRQ(ierr);
  }

  /* Test reusing a previously allocated dense buffer */
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
  ierr = MatMatMultEqual(A,B,X,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage\n");CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
    ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(T,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }

  /* Test MatTransposeMat and MatMatTranspose */
  if (testmattmat) {
    ierr = MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatTransposeMatMultEqual(A,X,B,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTransposeMat)\n");CHKERRQ(ierr);
      ierr = MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }
  if (testmatmatt) {
    ierr = MatMatTransposeMult(A,Bt,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(Bt,X,aBt,aX);CHKERRQ(ierr);
    ierr = MatMatTransposeMultEqual(A,Bt,X,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatMatTranspose)\n");CHKERRQ(ierr);
      ierr = MatMatTransposeMult(A,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }

  /* Test projection operations (PtAP and RARt) */
  if (testproj) {
    ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP);CHKERRQ(ierr);
    ierr = MatPtAPMultEqual(A,B,PtAP,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP\n");CHKERRQ(ierr);
      ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(B,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2);CHKERRQ(ierr);
      ierr = MatAXPY(T2,-1.0,PtAP,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T2,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T2);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1((k+ldr)*M,&dataR);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,m,K,M,dataR,&R);CHKERRQ(ierr);
    ierr = MatDenseSetLDA(R,k+ldr);CHKERRQ(ierr);
    ierr = MatSetRandom(R,NULL);CHKERRQ(ierr);
    if (testrart) { /* fails for AIJCUSPARSE because RA operation is not defined */
      ierr = MatRARt(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RARt);CHKERRQ(ierr);
      ierr = MatRARtMultEqual(A,R,RARt,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with RARt\n");CHKERRQ(ierr);
        ierr = MatMatTransposeMult(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
        ierr = MatMatMult(R,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2);CHKERRQ(ierr);
        ierr = MatAXPY(T2,-1.0,RARt,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatView(T2,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&T2);CHKERRQ(ierr);
        ierr = MatDestroy(&T);CHKERRQ(ierr);
      }
    }
  }

  /* Test MatDenseGetColumnVec and friends */
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
  ierr = MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&T2);CHKERRQ(ierr);
  for (k=0;k<K;k++) {
    Vec Xv,Tv,T2v;

    ierr = MatDenseGetColumnVecRead(X,k,&Xv);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVec(T,k,&Tv);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVecWrite(T2,k,&T2v);CHKERRQ(ierr);
    ierr = VecCopy(Xv,T2v);CHKERRQ(ierr);
    ierr = VecAXPY(Tv,-1.,Xv);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecRead(X,k,&Xv);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVec(T,k,&Tv);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecWrite(T2,k,&T2v);CHKERRQ(ierr);
  }
  ierr = MatNorm(T,NORM_FROBENIUS,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVec\n");CHKERRQ(ierr);
    ierr = MatView(T,NULL);CHKERRQ(ierr);
  }
  ierr = MatAXPY(T2,-1.,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(T2,NORM_FROBENIUS,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVecWrite\n");CHKERRQ(ierr);
    ierr = MatView(T2,NULL);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = MatDestroy(&T2);CHKERRQ(ierr);

  /* Test with MatShell */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&T);CHKERRQ(ierr);
  ierr = MatConvert(T,MATSHELL,MAT_INITIAL_MATRIX,&T2);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);

  /* scale matrix */
  ierr = MatScale(A,2.0);CHKERRQ(ierr);
  ierr = MatScale(T2,2.0);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&r,&l);CHKERRQ(ierr);
  ierr = VecSetRandom(r,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(l,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(T2,&rs,&ls);CHKERRQ(ierr);
  ierr = VecCopy(r,rs);CHKERRQ(ierr);
  ierr = VecCopy(l,ls);CHKERRQ(ierr);
  if (testproj) {
    ierr = MatDiagonalScale(A,r,r);CHKERRQ(ierr);
    ierr = MatDiagonalScale(T2,rs,rs);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalScale(A,l,r);CHKERRQ(ierr);
    ierr = MatDiagonalScale(T2,ls,rs);CHKERRQ(ierr);
  }
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&T);CHKERRQ(ierr);
  ierr = MatAXPY(A,4.5,T,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(T2,4.5,T,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMultEqual(T2,A,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MATSHELL (MatMult)\n");CHKERRQ(ierr);
  }
  ierr = MatMultTransposeEqual(T2,A,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MATSHELL (MatMultTranspose)\n");CHKERRQ(ierr);
  }
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = VecDestroy(&ls);CHKERRQ(ierr);
  ierr = VecDestroy(&rs);CHKERRQ(ierr);
  ierr = VecDestroy(&l);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);

  /* recompute projections, test reusage */
  if (PtAP) { ierr = MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&PtAP);CHKERRQ(ierr); }
  if (RARt) { ierr = MatRARt(A,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&RARt);CHKERRQ(ierr); }
  if (testshellops) { /* test callbacks for user defined MatProducts */
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_ABt,NULL,MyMatShellMatTransposeMultNumeric,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_ABt,NULL,MyMatShellMatTransposeMultNumeric,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
    if (testproj) {
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,MyPtShellPMultSymbolic,MyPtShellPMultNumeric,proj_destroy,MATDENSE,MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,MyPtShellPMultSymbolic,MyPtShellPMultNumeric,proj_destroy,MATDENSECUDA,MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,MyRShellRtMultSymbolic,MyRShellRtMultNumeric,proj_destroy,MATDENSE,MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,MyRShellRtMultSymbolic,MyRShellRtMultNumeric,proj_destroy,MATDENSECUDA,MATSHELL);CHKERRQ(ierr);
    }
  }
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
  /* we either use the shell operations or the loop over columns code, applying the operator */
  ierr = MatMatMult(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
  ierr = MatMatMultEqual(T2,B,X,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MATSHELL)\n");CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
    ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(T,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }
  if (testproj) {
    ierr = MatPtAPMultEqual(T2,B,PtAP,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP\n");CHKERRQ(ierr);
    }
    if (testshellops) { /* projections fail if the product operations are not specified */
      ierr = MatPtAP(T2,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatPtAP(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatPtAPMultEqual(T2,B,T,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        Mat TE;

        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP (user defined)\n");CHKERRQ(ierr);
        ierr = MatComputeOperator(T,MATDENSE,&TE);CHKERRQ(ierr);
        ierr = MatView(TE,NULL);CHKERRQ(ierr);
        ierr = MatView(PtAP,NULL);CHKERRQ(ierr);
        ierr = MatAXPY(TE,-1.0,PtAP,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatView(TE,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&TE);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    if (RARt) {
      ierr = MatRARtMultEqual(T2,R,RARt,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with RARt\n");CHKERRQ(ierr);
      }
    }
    if (testshellops) {
      ierr = MatRARt(T2,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatRARt(T2,R,MAT_REUSE_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatRARtMultEqual(T2,R,T,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        Mat TE;

        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with RARt (user defined)\n");CHKERRQ(ierr);
        ierr = MatComputeOperator(T,MATDENSE,&TE);CHKERRQ(ierr);
        ierr = MatView(TE,NULL);CHKERRQ(ierr);
        if (RARt) {
          ierr = MatView(RARt,NULL);CHKERRQ(ierr);
          ierr = MatAXPY(TE,-1.0,RARt,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = MatView(TE,NULL);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&TE);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }

  if (testmattmat) { /* we either use the shell operations or the loop over columns code applying the transposed operator */
    ierr = MatTransposeMatMult(T2,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatTransposeMatMultEqual(T2,X,B,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTranspose, MATSHELL)\n");CHKERRQ(ierr);
      ierr = MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }
  if (testmatmatt && testshellops) { /* only when shell operations are set */
    ierr = MatMatTransposeMult(T2,Bt,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(Bt,X,aBt,aX);CHKERRQ(ierr);
    ierr = MatMatTransposeMultEqual(T2,Bt,X,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatMatTranspose, MATSHELL)\n");CHKERRQ(ierr);
      ierr = MatMatTransposeMult(A,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&T2);CHKERRQ(ierr);

  if (testnest) { /* test with MatNest */
    Mat        NA;
    const char *vtype;

    ierr = MatCreateNest(PETSC_COMM_WORLD,1,NULL,1,NULL,&A,&NA);CHKERRQ(ierr);
    /* needed to test against CUSPARSE matrices */
    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = MatSetVecType(NA,vtype);CHKERRQ(ierr);
    ierr = MatViewFromOptions(NA,NULL,"-NA_view");CHKERRQ(ierr);
    ierr = MatMatMult(NA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(NA,B,X,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Nest\n");CHKERRQ(ierr);
      ierr = MatMatMult(NA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&NA);CHKERRQ(ierr);
  }

  if (testtranspose) { /* test with Transpose */
    Mat TA;

    ierr = MatCreateTranspose(A,&TA);CHKERRQ(ierr);
    ierr = MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(TA,X,B,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n");CHKERRQ(ierr);
      ierr = MatMatMult(TA,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&TA);CHKERRQ(ierr);
  }

  if (testhtranspose) { /* test with Hermitian Transpose */
    Mat TA;

    ierr = MatCreateHermitianTranspose(A,&TA);CHKERRQ(ierr);
    ierr = MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(TA,X,B,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n");CHKERRQ(ierr);
      ierr = MatMatMult(TA,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&TA);CHKERRQ(ierr);
  }

  if (testtt) { /* test with Transpose(Transpose) */
    Mat TA, TTA;

    ierr = MatCreateTranspose(A,&TA);CHKERRQ(ierr);
    ierr = MatCreateTranspose(TA,&TTA);CHKERRQ(ierr);
    ierr = MatMatMult(TTA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(TTA,B,X,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose(Transpose)\n");CHKERRQ(ierr);
      ierr = MatMatMult(TTA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&TA);CHKERRQ(ierr);
    ierr = MatDestroy(&TTA);CHKERRQ(ierr);
  }

  if (testcircular) { /* test circular */
    Mat AB;

    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AB);CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    if (M == N && N == K) {
      ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    } else {
      ierr = MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    }
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatDestroy(&AB);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&Bt);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP);CHKERRQ(ierr);
  ierr = MatDestroy(&RARt);CHKERRQ(ierr);
  ierr = PetscFree(dataX);CHKERRQ(ierr);
  ierr = PetscFree(dataB);CHKERRQ(ierr);
  ierr = PetscFree(dataR);CHKERRQ(ierr);
  ierr = PetscFree(dataBt);CHKERRQ(ierr);
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
    args: -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type {{seqaijcusparse seqaij}} -testnest 0 -testshellops {{0 1}}

  test:
    TODO: VecGetSubVector seems broken with CUDA
    output_file: output/ex70_1.out
    requires: cuda
    suffix: 1_cuda_broken
    args: -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type seqaijcusparse -testnest

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

  test:
    requires: cuda
    output_file: output/ex70_1.out
    suffix: 2_cuda
    nsize: 1
    args: -M 7 -N 9 -K 2 -local {{0 1}} -testnest 0 -A_mat_type {{seqdensecuda seqdense}} -xgpu {{0 1}} -bgpu {{0 1}}

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
