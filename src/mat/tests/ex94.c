
static char help[] = "Tests sequential and parallel MatMatMult() and MatPtAP(), MatTransposeMatMult(), sequential MatMatTransposeMult(), MatRARt()\n\
Input arguments are:\n\
  -f0 <input_file> -f1 <input_file> -f2 <input_file> -f3 <input_file> : file to load\n\n";
/* Example of usage:
   ./ex94 -f0 <A_binary> -f1 <B_binary> -matmatmult_mat_view ascii::ascii_info -matmatmulttr_mat_view
   mpiexec -n 3 ./ex94 -f0 medium -f1 medium -f2 arco1 -f3 arco1 -matmatmult_mat_view
*/

#include <petscmat.h>

/*
     B = A - B
     norm = norm(B)
*/
PetscErrorCode MatNormDifference(Mat A,Mat B,PetscReal *norm)
{
  PetscFunctionBegin;
  CHKERRQ(MatAXPY(B,-1.0,A,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatNorm(B,NORM_FROBENIUS,norm));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,A_save,B,AT,ATT,BT,BTT,P,R,C,C1;
  Vec            x,v1,v2,v3,v4;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       i,m,n,j,*idxn,M,N,nzp,rstart,rend;
  PetscReal      norm,norm_abs,norm_tmp,fill=4.0;
  PetscRandom    rdm;
  char           file[4][128];
  PetscBool      flg,preload = PETSC_TRUE;
  PetscScalar    *a,rval,alpha,none = -1.0;
  PetscBool      Test_MatMatMult=PETSC_TRUE,Test_MatMatTr=PETSC_TRUE,Test_MatPtAP=PETSC_TRUE,Test_MatRARt=PETSC_TRUE,Test_MatMatMatMult=PETSC_TRUE;
  PetscBool      Test_MatAXPY=PETSC_FALSE,view=PETSC_FALSE;
  PetscInt       pm,pn,pM,pN;
  MatInfo        info;
  PetscBool      seqaij;
  MatType        mattype;
  Mat            Cdensetest,Pdense,Cdense,Adense;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-matops_view",&view,NULL));
  if (view) {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  }

  /*  Load the matrices A_save and B */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for small matrix A with the -f0 option.");
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for small matrix B with the -f1 option.");
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",file[2],sizeof(file[2]),&flg));
  if (!flg) {
    preload = PETSC_FALSE;
  } else {
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f3",file[3],sizeof(file[3]),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for test matrix B with the -f3 option.");
  }

  PetscPreLoadBegin(preload,"Load system");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt],FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A_save));
  CHKERRQ(MatSetFromOptions(A_save));
  CHKERRQ(MatLoad(A_save,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt+1],FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatLoad(B,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatGetType(B,&mattype));

  CHKERRQ(MatGetSize(B,&M,&N));
  nzp  = PetscMax((PetscInt)(0.1*M),5);
  CHKERRQ(PetscMalloc((nzp+1)*(sizeof(PetscInt)+sizeof(PetscScalar)),&idxn));
  a    = (PetscScalar*)(idxn + nzp);

  /* Create vectors v1 and v2 that are compatible with A_save */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v1));
  CHKERRQ(MatGetLocalSize(A_save,&m,NULL));
  CHKERRQ(VecSetSizes(v1,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(v1));
  CHKERRQ(VecDuplicate(v1,&v2));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));

  /* Test MatAXPY()    */
  /*-------------------*/
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_MatAXPY",&Test_MatAXPY));
  if (Test_MatAXPY) {
    Mat Btmp;
    CHKERRQ(MatDuplicate(A_save,MAT_COPY_VALUES,&A));
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&Btmp));
    CHKERRQ(MatAXPY(A,-1.0,B,DIFFERENT_NONZERO_PATTERN)); /* A = -B + A_save */

    CHKERRQ(MatScale(A,-1.0)); /* A = -A = B - A_save */
    CHKERRQ(MatAXPY(Btmp,-1.0,A,DIFFERENT_NONZERO_PATTERN)); /* Btmp = -A + B = A_save */
    CHKERRQ(MatMultEqual(A_save,Btmp,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatAXPY() is incorrect");
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&Btmp));

    Test_MatMatMult    = PETSC_FALSE;
    Test_MatMatTr      = PETSC_FALSE;
    Test_MatPtAP       = PETSC_FALSE;
    Test_MatRARt       = PETSC_FALSE;
    Test_MatMatMatMult = PETSC_FALSE;
  }

  /* 1) Test MatMatMult() */
  /* ---------------------*/
  if (Test_MatMatMult) {
    CHKERRQ(MatDuplicate(A_save,MAT_COPY_VALUES,&A));
    CHKERRQ(MatCreateTranspose(A,&AT));
    CHKERRQ(MatCreateTranspose(AT,&ATT));
    CHKERRQ(MatCreateTranspose(B,&BT));
    CHKERRQ(MatCreateTranspose(BT,&BTT));

    CHKERRQ(MatMatMult(AT,B,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatMatMultEqual(AT,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=AT*B");
    CHKERRQ(MatDestroy(&C));

    CHKERRQ(MatMatMult(ATT,B,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatMatMultEqual(ATT,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=ATT*B");
    CHKERRQ(MatDestroy(&C));

    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatMatMultEqual(A,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for reuse C=A*B");
    /* ATT has different matrix type as A (although they have same internal data structure),
       we cannot call MatProductReplaceMats(ATT,NULL,NULL,C) and MatMatMult(ATT,B,MAT_REUSE_MATRIX,fill,&C) */
    CHKERRQ(MatDestroy(&C));

    CHKERRQ(MatMatMult(A,BTT,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatMatMultEqual(A,BTT,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=A*BTT");
    CHKERRQ(MatDestroy(&C));

    CHKERRQ(MatMatMult(ATT,BTT,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatMatMultEqual(A,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()");
    CHKERRQ(MatDestroy(&C));

    CHKERRQ(MatDestroy(&BTT));
    CHKERRQ(MatDestroy(&BT));
    CHKERRQ(MatDestroy(&ATT));
    CHKERRQ(MatDestroy(&AT));

    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatSetOptionsPrefix(C,"matmatmult_")); /* enable option '-matmatmult_' for matrix C */
    CHKERRQ(MatGetInfo(C,MAT_GLOBAL_SUM,&info));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -=0.1;
      CHKERRQ(MatScale(A,alpha));
      CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C));
    }
    CHKERRQ(MatMatMultEqual(A,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()");
    CHKERRQ(MatDestroy(&A));

    /* Test MatDuplicate() of C=A*B */
    CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&C1));
    CHKERRQ(MatDestroy(&C1));
    CHKERRQ(MatDestroy(&C));
  } /* if (Test_MatMatMult) */

  /* 2) Test MatTransposeMatMult() and MatMatTransposeMult() */
  /* ------------------------------------------------------- */
  if (Test_MatMatTr) {
    /* Create P */
    PetscInt PN,rstart,rend;
    PN   = M/2;
    nzp  = 5; /* num of nonzeros in each row of P */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P));
    CHKERRQ(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,M,PN));
    CHKERRQ(MatSetType(P,mattype));
    CHKERRQ(MatSeqAIJSetPreallocation(P,nzp,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(P,nzp,NULL,nzp,NULL));
    CHKERRQ(MatGetOwnershipRange(P,&rstart,&rend));
    for (i=0; i<nzp; i++) {
      CHKERRQ(PetscRandomGetValue(rdm,&a[i]));
    }
    for (i=rstart; i<rend; i++) {
      for (j=0; j<nzp; j++) {
        CHKERRQ(PetscRandomGetValue(rdm,&rval));
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      CHKERRQ(MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

    /* Create R = P^T */
    CHKERRQ(MatTranspose(P,MAT_INITIAL_MATRIX,&R));

    { /* Test R = P^T, C1 = R*B */
      CHKERRQ(MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1));
      CHKERRQ(MatTranspose(P,MAT_REUSE_MATRIX,&R));
      CHKERRQ(MatMatMult(R,B,MAT_REUSE_MATRIX,fill,&C1));
      CHKERRQ(MatDestroy(&C1));
    }

    /* C = P^T*B */
    CHKERRQ(MatTransposeMatMult(P,B,MAT_INITIAL_MATRIX,fill,&C));
    CHKERRQ(MatGetInfo(C,MAT_GLOBAL_SUM,&info));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    CHKERRQ(MatTransposeMatMult(P,B,MAT_REUSE_MATRIX,fill,&C));
    if (view) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"C = P^T * B:\n"));
      CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    }
    CHKERRQ(MatProductClear(C));
    if (view) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * B after MatProductClear():\n"));
      CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    }

    /* Compare P^T*B and R*B */
    CHKERRQ(MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1));
    CHKERRQ(MatNormDifference(C,C1,&norm));
    PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatTransposeMatMult(): %g",(double)norm);
    CHKERRQ(MatDestroy(&C1));

    /* Test MatDuplicate() of C=P^T*B */
    CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&C1));
    CHKERRQ(MatDestroy(&C1));
    CHKERRQ(MatDestroy(&C));

    /* C = B*R^T */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)B,MATSEQAIJ,&seqaij));
    if (size == 1 && seqaij) {
      CHKERRQ(MatMatTransposeMult(B,R,MAT_INITIAL_MATRIX,fill,&C));
      CHKERRQ(MatSetOptionsPrefix(C,"matmatmulttr_")); /* enable '-matmatmulttr_' for matrix C */
      CHKERRQ(MatGetInfo(C,MAT_GLOBAL_SUM,&info));

      /* Test MAT_REUSE_MATRIX - reuse symbolic C */
      CHKERRQ(MatMatTransposeMult(B,R,MAT_REUSE_MATRIX,fill,&C));

      /* Check */
      CHKERRQ(MatMatMult(B,P,MAT_INITIAL_MATRIX,fill,&C1));
      CHKERRQ(MatNormDifference(C,C1,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatTransposeMult() %g",(double)norm);
      CHKERRQ(MatDestroy(&C1));
      CHKERRQ(MatDestroy(&C));
    }
    CHKERRQ(MatDestroy(&P));
    CHKERRQ(MatDestroy(&R));
  }

  /* 3) Test MatPtAP() */
  /*-------------------*/
  if (Test_MatPtAP) {
    PetscInt  PN;
    Mat       Cdup;

    CHKERRQ(MatDuplicate(A_save,MAT_COPY_VALUES,&A));
    CHKERRQ(MatGetSize(A,&M,&N));
    CHKERRQ(MatGetLocalSize(A,&m,&n));

    PN   = M/2;
    nzp  = (PetscInt)(0.1*PN+1); /* num of nozeros in each row of P */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P));
    CHKERRQ(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,N,PN));
    CHKERRQ(MatSetType(P,mattype));
    CHKERRQ(MatSeqAIJSetPreallocation(P,nzp,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(P,nzp,NULL,nzp,NULL));
    for (i=0; i<nzp; i++) {
      CHKERRQ(PetscRandomGetValue(rdm,&a[i]));
    }
    CHKERRQ(MatGetOwnershipRange(P,&rstart,&rend));
    for (i=rstart; i<rend; i++) {
      for (j=0; j<nzp; j++) {
        CHKERRQ(PetscRandomGetValue(rdm,&rval));
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      CHKERRQ(MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

    /* CHKERRQ(MatView(P,PETSC_VIEWER_STDOUT_WORLD)); */
    CHKERRQ(MatGetSize(P,&pM,&pN));
    CHKERRQ(MatGetLocalSize(P,&pm,&pn));
    CHKERRQ(MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -=0.1;
      CHKERRQ(MatScale(A,alpha));
      CHKERRQ(MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C));
    }

    /* Test PtAP ops with P Dense and A either AIJ or SeqDense (it assumes MatPtAP_XAIJ_XAIJ is fine) */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&seqaij));
    if (seqaij) {
      CHKERRQ(MatConvert(C,MATSEQDENSE,MAT_INITIAL_MATRIX,&Cdensetest));
      CHKERRQ(MatConvert(P,MATSEQDENSE,MAT_INITIAL_MATRIX,&Pdense));
    } else {
      CHKERRQ(MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdensetest));
      CHKERRQ(MatConvert(P,MATMPIDENSE,MAT_INITIAL_MATRIX,&Pdense));
    }

    /* test with A(AIJ), Pdense -- call MatPtAP_Basic() when np>1 */
    CHKERRQ(MatPtAP(A,Pdense,MAT_INITIAL_MATRIX,fill,&Cdense));
    CHKERRQ(MatPtAP(A,Pdense,MAT_REUSE_MATRIX,fill,&Cdense));
    CHKERRQ(MatPtAPMultEqual(A,Pdense,Cdense,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP with A AIJ and P Dense");
    CHKERRQ(MatDestroy(&Cdense));

    /* test with A SeqDense */
    if (seqaij) {
      CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&Adense));
      CHKERRQ(MatPtAP(Adense,Pdense,MAT_INITIAL_MATRIX,fill,&Cdense));
      CHKERRQ(MatPtAP(Adense,Pdense,MAT_REUSE_MATRIX,fill,&Cdense));
      CHKERRQ(MatPtAPMultEqual(Adense,Pdense,Cdense,10,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatPtAP with A SeqDense and P SeqDense");
      CHKERRQ(MatDestroy(&Cdense));
      CHKERRQ(MatDestroy(&Adense));
    }
    CHKERRQ(MatDestroy(&Cdensetest));
    CHKERRQ(MatDestroy(&Pdense));

    /* Test MatDuplicate() of C=PtAP and MatView(Cdup,...) */
    CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&Cdup));
    if (view) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * A * P:\n"));
      CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

      CHKERRQ(MatProductClear(C));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * A * P after MatProductClear():\n"));
      CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nCdup:\n"));
      CHKERRQ(MatView(Cdup,PETSC_VIEWER_STDOUT_WORLD));
    }
    CHKERRQ(MatDestroy(&Cdup));

    if (size>1 || !seqaij) Test_MatRARt = PETSC_FALSE;
    /* 4) Test MatRARt() */
    /* ----------------- */
    if (Test_MatRARt) {
      Mat R, RARt, Rdense, RARtdense;
      CHKERRQ(MatTranspose(P,MAT_INITIAL_MATRIX,&R));

      /* Test MatRARt_Basic(), MatMatMatMult_Basic() */
      CHKERRQ(MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&Rdense));
      CHKERRQ(MatRARt(A,Rdense,MAT_INITIAL_MATRIX,2.0,&RARtdense));
      CHKERRQ(MatRARt(A,Rdense,MAT_REUSE_MATRIX,2.0,&RARtdense));

      CHKERRQ(MatConvert(RARtdense,MATAIJ,MAT_INITIAL_MATRIX,&RARt));
      CHKERRQ(MatNormDifference(C,RARt,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"|PtAP - RARtdense| = %g",(double)norm);
      CHKERRQ(MatDestroy(&Rdense));
      CHKERRQ(MatDestroy(&RARtdense));
      CHKERRQ(MatDestroy(&RARt));

      /* Test MatRARt() for aij matrices */
      CHKERRQ(MatRARt(A,R,MAT_INITIAL_MATRIX,2.0,&RARt));
      CHKERRQ(MatRARt(A,R,MAT_REUSE_MATRIX,2.0,&RARt));
      CHKERRQ(MatNormDifference(C,RARt,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"|PtAP - RARt| = %g",(double)norm);
      CHKERRQ(MatDestroy(&R));
      CHKERRQ(MatDestroy(&RARt));
    }

    if (Test_MatMatMatMult && size == 1) {
      Mat       R, RAP;
      CHKERRQ(MatTranspose(P,MAT_INITIAL_MATRIX,&R));
      CHKERRQ(MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,2.0,&RAP));
      CHKERRQ(MatMatMatMult(R,A,P,MAT_REUSE_MATRIX,2.0,&RAP));
      CHKERRQ(MatNormDifference(C,RAP,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PtAP != RAP %g",(double)norm);
      CHKERRQ(MatDestroy(&R));
      CHKERRQ(MatDestroy(&RAP));
    }

    /* Create vector x that is compatible with P */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
    CHKERRQ(MatGetLocalSize(P,&m,&n));
    CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(x));

    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v3));
    CHKERRQ(VecSetSizes(v3,n,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(v3));
    CHKERRQ(VecDuplicate(v3,&v4));

    norm = 0.0;
    for (i=0; i<10; i++) {
      CHKERRQ(VecSetRandom(x,rdm));
      CHKERRQ(MatMult(P,x,v1));
      CHKERRQ(MatMult(A,v1,v2));  /* v2 = A*P*x */

      CHKERRQ(MatMultTranspose(P,v2,v3)); /* v3 = Pt*A*P*x */
      CHKERRQ(MatMult(C,x,v4));           /* v3 = C*x   */
      CHKERRQ(VecNorm(v4,NORM_2,&norm_abs));
      CHKERRQ(VecAXPY(v4,none,v3));
      CHKERRQ(VecNorm(v4,NORM_2,&norm_tmp));

      norm_tmp /= norm_abs;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    PetscCheckFalse(norm >= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatPtAP(), |v1 - v2|: %g",(double)norm);

    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&P));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(VecDestroy(&v3));
    CHKERRQ(VecDestroy(&v4));
    CHKERRQ(VecDestroy(&x));
  }

  /* Destroy objects */
  CHKERRQ(VecDestroy(&v1));
  CHKERRQ(VecDestroy(&v2));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(PetscFree(idxn));

  CHKERRQ(MatDestroy(&A_save));
  CHKERRQ(MatDestroy(&B));

  PetscPreLoadEnd();
  PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 2_mattransposematmult_matmatmult
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium -mattransposematmult_via at*b> ex94_2.tmp 2>&1

   test:
      suffix: 2_mattransposematmult_scalable
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium -mattransposematmult_via scalable> ex94_2.tmp 2>&1
      output_file: output/ex94_1.out

   test:
      suffix: axpy_mpiaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 8
      args: -f0 ${DATAFILESPATH}/matrices/poisson_2d5p -f1 ${DATAFILESPATH}/matrices/poisson_2d13p -test_MatAXPY
      output_file: output/ex94_1.out

   test:
      suffix: axpy_mpibaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 8
      args: -f0 ${DATAFILESPATH}/matrices/poisson_2d5p -f1 ${DATAFILESPATH}/matrices/poisson_2d13p -test_MatAXPY -mat_type baij
      output_file: output/ex94_1.out

   test:
      suffix: axpy_mpisbaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 8
      args: -f0 ${DATAFILESPATH}/matrices/poisson_2d5p -f1 ${DATAFILESPATH}/matrices/poisson_2d13p -test_MatAXPY -mat_type sbaij
      output_file: output/ex94_1.out

   test:
      suffix: matmatmult
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info
      output_file: output/ex94_1.out

   test:
      suffix: matmatmult_2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -mat_type mpiaij -viewer_binary_skip_info
      output_file: output/ex94_1.out

   test:
      suffix: matmatmult_scalable
      nsize: 4
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -matmatmult_via scalable
      output_file: output/ex94_1.out

   test:
      suffix: ptap
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium -matptap_via scalable
      output_file: output/ex94_1.out

   test:
      suffix: rap
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium
      output_file: output/ex94_1.out

   test:
      suffix: scalable0
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info
      output_file: output/ex94_1.out

   test:
      suffix: scalable1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info -matptap_via scalable
      output_file: output/ex94_1.out

   test:
      suffix: view
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/tiny -f1 ${DATAFILESPATH}/matrices/tiny -viewer_binary_skip_info -matops_view
      output_file: output/ex94_2.out

TEST*/
