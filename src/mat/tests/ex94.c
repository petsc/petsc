
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
  PetscCall(MatAXPY(B,-1.0,A,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(B,NORM_FROBENIUS,norm));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,A_save,B,AT,ATT,BT,BTT,P,R,C,C1;
  Vec            x,v1,v2,v3,v4;
  PetscViewer    viewer;
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-matops_view",&view,NULL));
  if (view) PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));

  /*  Load the matrices A_save and B */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for small matrix A with the -f0 option.");
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for small matrix B with the -f1 option.");
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f2",file[2],sizeof(file[2]),&flg));
  if (!flg) {
    preload = PETSC_FALSE;
  } else {
    PetscCall(PetscOptionsGetString(NULL,NULL,"-f3",file[3],sizeof(file[3]),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for test matrix B with the -f3 option.");
  }

  PetscPreLoadBegin(preload,"Load system");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt],FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A_save));
  PetscCall(MatSetFromOptions(A_save));
  PetscCall(MatLoad(A_save,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt+1],FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLoad(B,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatGetType(B,&mattype));

  PetscCall(MatGetSize(B,&M,&N));
  nzp  = PetscMax((PetscInt)(0.1*M),5);
  PetscCall(PetscMalloc((nzp+1)*(sizeof(PetscInt)+sizeof(PetscScalar)),&idxn));
  a    = (PetscScalar*)(idxn + nzp);

  /* Create vectors v1 and v2 that are compatible with A_save */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&v1));
  PetscCall(MatGetLocalSize(A_save,&m,NULL));
  PetscCall(VecSetSizes(v1,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(v1));
  PetscCall(VecDuplicate(v1,&v2));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));

  /* Test MatAXPY()    */
  /*-------------------*/
  PetscCall(PetscOptionsHasName(NULL,NULL,"-test_MatAXPY",&Test_MatAXPY));
  if (Test_MatAXPY) {
    Mat Btmp;
    PetscCall(MatDuplicate(A_save,MAT_COPY_VALUES,&A));
    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&Btmp));
    PetscCall(MatAXPY(A,-1.0,B,DIFFERENT_NONZERO_PATTERN)); /* A = -B + A_save */

    PetscCall(MatScale(A,-1.0)); /* A = -A = B - A_save */
    PetscCall(MatAXPY(Btmp,-1.0,A,DIFFERENT_NONZERO_PATTERN)); /* Btmp = -A + B = A_save */
    PetscCall(MatMultEqual(A_save,Btmp,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatAXPY() is incorrect");
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&Btmp));

    Test_MatMatMult    = PETSC_FALSE;
    Test_MatMatTr      = PETSC_FALSE;
    Test_MatPtAP       = PETSC_FALSE;
    Test_MatRARt       = PETSC_FALSE;
    Test_MatMatMatMult = PETSC_FALSE;
  }

  /* 1) Test MatMatMult() */
  /* ---------------------*/
  if (Test_MatMatMult) {
    PetscCall(MatDuplicate(A_save,MAT_COPY_VALUES,&A));
    PetscCall(MatCreateTranspose(A,&AT));
    PetscCall(MatCreateTranspose(AT,&ATT));
    PetscCall(MatCreateTranspose(B,&BT));
    PetscCall(MatCreateTranspose(BT,&BTT));

    PetscCall(MatMatMult(AT,B,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatMatMultEqual(AT,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=AT*B");
    PetscCall(MatDestroy(&C));

    PetscCall(MatMatMult(ATT,B,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatMatMultEqual(ATT,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=ATT*B");
    PetscCall(MatDestroy(&C));

    PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatMatMultEqual(A,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for reuse C=A*B");
    /* ATT has different matrix type as A (although they have same internal data structure),
       we cannot call MatProductReplaceMats(ATT,NULL,NULL,C) and MatMatMult(ATT,B,MAT_REUSE_MATRIX,fill,&C) */
    PetscCall(MatDestroy(&C));

    PetscCall(MatMatMult(A,BTT,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatMatMultEqual(A,BTT,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=A*BTT");
    PetscCall(MatDestroy(&C));

    PetscCall(MatMatMult(ATT,BTT,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatMatMultEqual(A,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()");
    PetscCall(MatDestroy(&C));

    PetscCall(MatDestroy(&BTT));
    PetscCall(MatDestroy(&BT));
    PetscCall(MatDestroy(&ATT));
    PetscCall(MatDestroy(&AT));

    PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatSetOptionsPrefix(C,"matmatmult_")); /* enable option '-matmatmult_' for matrix C */
    PetscCall(MatGetInfo(C,MAT_GLOBAL_SUM,&info));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -=0.1;
      PetscCall(MatScale(A,alpha));
      PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C));
    }
    PetscCall(MatMatMultEqual(A,B,C,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()");
    PetscCall(MatDestroy(&A));

    /* Test MatDuplicate() of C=A*B */
    PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&C1));
    PetscCall(MatDestroy(&C1));
    PetscCall(MatDestroy(&C));
  } /* if (Test_MatMatMult) */

  /* 2) Test MatTransposeMatMult() and MatMatTransposeMult() */
  /* ------------------------------------------------------- */
  if (Test_MatMatTr) {
    /* Create P */
    PetscInt PN,rstart,rend;
    PN   = M/2;
    nzp  = 5; /* num of nonzeros in each row of P */
    PetscCall(MatCreate(PETSC_COMM_WORLD,&P));
    PetscCall(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,M,PN));
    PetscCall(MatSetType(P,mattype));
    PetscCall(MatSeqAIJSetPreallocation(P,nzp,NULL));
    PetscCall(MatMPIAIJSetPreallocation(P,nzp,NULL,nzp,NULL));
    PetscCall(MatGetOwnershipRange(P,&rstart,&rend));
    for (i=0; i<nzp; i++) {
      PetscCall(PetscRandomGetValue(rdm,&a[i]));
    }
    for (i=rstart; i<rend; i++) {
      for (j=0; j<nzp; j++) {
        PetscCall(PetscRandomGetValue(rdm,&rval));
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      PetscCall(MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

    /* Create R = P^T */
    PetscCall(MatTranspose(P,MAT_INITIAL_MATRIX,&R));

    { /* Test R = P^T, C1 = R*B */
      PetscCall(MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1));
      PetscCall(MatTranspose(P,MAT_REUSE_MATRIX,&R));
      PetscCall(MatMatMult(R,B,MAT_REUSE_MATRIX,fill,&C1));
      PetscCall(MatDestroy(&C1));
    }

    /* C = P^T*B */
    PetscCall(MatTransposeMatMult(P,B,MAT_INITIAL_MATRIX,fill,&C));
    PetscCall(MatGetInfo(C,MAT_GLOBAL_SUM,&info));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    PetscCall(MatTransposeMatMult(P,B,MAT_REUSE_MATRIX,fill,&C));
    if (view) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"C = P^T * B:\n"));
      PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(MatProductClear(C));
    if (view) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * B after MatProductClear():\n"));
      PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    }

    /* Compare P^T*B and R*B */
    PetscCall(MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1));
    PetscCall(MatNormDifference(C,C1,&norm));
    PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatTransposeMatMult(): %g",(double)norm);
    PetscCall(MatDestroy(&C1));

    /* Test MatDuplicate() of C=P^T*B */
    PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&C1));
    PetscCall(MatDestroy(&C1));
    PetscCall(MatDestroy(&C));

    /* C = B*R^T */
    PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQAIJ,&seqaij));
    if (size == 1 && seqaij) {
      PetscCall(MatMatTransposeMult(B,R,MAT_INITIAL_MATRIX,fill,&C));
      PetscCall(MatSetOptionsPrefix(C,"matmatmulttr_")); /* enable '-matmatmulttr_' for matrix C */
      PetscCall(MatGetInfo(C,MAT_GLOBAL_SUM,&info));

      /* Test MAT_REUSE_MATRIX - reuse symbolic C */
      PetscCall(MatMatTransposeMult(B,R,MAT_REUSE_MATRIX,fill,&C));

      /* Check */
      PetscCall(MatMatMult(B,P,MAT_INITIAL_MATRIX,fill,&C1));
      PetscCall(MatNormDifference(C,C1,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatTransposeMult() %g",(double)norm);
      PetscCall(MatDestroy(&C1));
      PetscCall(MatDestroy(&C));
    }
    PetscCall(MatDestroy(&P));
    PetscCall(MatDestroy(&R));
  }

  /* 3) Test MatPtAP() */
  /*-------------------*/
  if (Test_MatPtAP) {
    PetscInt  PN;
    Mat       Cdup;

    PetscCall(MatDuplicate(A_save,MAT_COPY_VALUES,&A));
    PetscCall(MatGetSize(A,&M,&N));
    PetscCall(MatGetLocalSize(A,&m,&n));

    PN   = M/2;
    nzp  = (PetscInt)(0.1*PN+1); /* num of nozeros in each row of P */
    PetscCall(MatCreate(PETSC_COMM_WORLD,&P));
    PetscCall(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,N,PN));
    PetscCall(MatSetType(P,mattype));
    PetscCall(MatSeqAIJSetPreallocation(P,nzp,NULL));
    PetscCall(MatMPIAIJSetPreallocation(P,nzp,NULL,nzp,NULL));
    for (i=0; i<nzp; i++) {
      PetscCall(PetscRandomGetValue(rdm,&a[i]));
    }
    PetscCall(MatGetOwnershipRange(P,&rstart,&rend));
    for (i=rstart; i<rend; i++) {
      for (j=0; j<nzp; j++) {
        PetscCall(PetscRandomGetValue(rdm,&rval));
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      PetscCall(MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

    /* PetscCall(MatView(P,PETSC_VIEWER_STDOUT_WORLD)); */
    PetscCall(MatGetSize(P,&pM,&pN));
    PetscCall(MatGetLocalSize(P,&pm,&pn));
    PetscCall(MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -=0.1;
      PetscCall(MatScale(A,alpha));
      PetscCall(MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C));
    }

    /* Test PtAP ops with P Dense and A either AIJ or SeqDense (it assumes MatPtAP_XAIJ_XAIJ is fine) */
    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&seqaij));
    if (seqaij) {
      PetscCall(MatConvert(C,MATSEQDENSE,MAT_INITIAL_MATRIX,&Cdensetest));
      PetscCall(MatConvert(P,MATSEQDENSE,MAT_INITIAL_MATRIX,&Pdense));
    } else {
      PetscCall(MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdensetest));
      PetscCall(MatConvert(P,MATMPIDENSE,MAT_INITIAL_MATRIX,&Pdense));
    }

    /* test with A(AIJ), Pdense -- call MatPtAP_Basic() when np>1 */
    PetscCall(MatPtAP(A,Pdense,MAT_INITIAL_MATRIX,fill,&Cdense));
    PetscCall(MatPtAP(A,Pdense,MAT_REUSE_MATRIX,fill,&Cdense));
    PetscCall(MatPtAPMultEqual(A,Pdense,Cdense,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP with A AIJ and P Dense");
    PetscCall(MatDestroy(&Cdense));

    /* test with A SeqDense */
    if (seqaij) {
      PetscCall(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&Adense));
      PetscCall(MatPtAP(Adense,Pdense,MAT_INITIAL_MATRIX,fill,&Cdense));
      PetscCall(MatPtAP(Adense,Pdense,MAT_REUSE_MATRIX,fill,&Cdense));
      PetscCall(MatPtAPMultEqual(Adense,Pdense,Cdense,10,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatPtAP with A SeqDense and P SeqDense");
      PetscCall(MatDestroy(&Cdense));
      PetscCall(MatDestroy(&Adense));
    }
    PetscCall(MatDestroy(&Cdensetest));
    PetscCall(MatDestroy(&Pdense));

    /* Test MatDuplicate() of C=PtAP and MatView(Cdup,...) */
    PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&Cdup));
    if (view) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * A * P:\n"));
      PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

      PetscCall(MatProductClear(C));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * A * P after MatProductClear():\n"));
      PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nCdup:\n"));
      PetscCall(MatView(Cdup,PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(MatDestroy(&Cdup));

    if (size>1 || !seqaij) Test_MatRARt = PETSC_FALSE;
    /* 4) Test MatRARt() */
    /* ----------------- */
    if (Test_MatRARt) {
      Mat R, RARt, Rdense, RARtdense;
      PetscCall(MatTranspose(P,MAT_INITIAL_MATRIX,&R));

      /* Test MatRARt_Basic(), MatMatMatMult_Basic() */
      PetscCall(MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&Rdense));
      PetscCall(MatRARt(A,Rdense,MAT_INITIAL_MATRIX,2.0,&RARtdense));
      PetscCall(MatRARt(A,Rdense,MAT_REUSE_MATRIX,2.0,&RARtdense));

      PetscCall(MatConvert(RARtdense,MATAIJ,MAT_INITIAL_MATRIX,&RARt));
      PetscCall(MatNormDifference(C,RARt,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"|PtAP - RARtdense| = %g",(double)norm);
      PetscCall(MatDestroy(&Rdense));
      PetscCall(MatDestroy(&RARtdense));
      PetscCall(MatDestroy(&RARt));

      /* Test MatRARt() for aij matrices */
      PetscCall(MatRARt(A,R,MAT_INITIAL_MATRIX,2.0,&RARt));
      PetscCall(MatRARt(A,R,MAT_REUSE_MATRIX,2.0,&RARt));
      PetscCall(MatNormDifference(C,RARt,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"|PtAP - RARt| = %g",(double)norm);
      PetscCall(MatDestroy(&R));
      PetscCall(MatDestroy(&RARt));
    }

    if (Test_MatMatMatMult && size == 1) {
      Mat       R, RAP;
      PetscCall(MatTranspose(P,MAT_INITIAL_MATRIX,&R));
      PetscCall(MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,2.0,&RAP));
      PetscCall(MatMatMatMult(R,A,P,MAT_REUSE_MATRIX,2.0,&RAP));
      PetscCall(MatNormDifference(C,RAP,&norm));
      PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PtAP != RAP %g",(double)norm);
      PetscCall(MatDestroy(&R));
      PetscCall(MatDestroy(&RAP));
    }

    /* Create vector x that is compatible with P */
    PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
    PetscCall(MatGetLocalSize(P,&m,&n));
    PetscCall(VecSetSizes(x,n,PETSC_DECIDE));
    PetscCall(VecSetFromOptions(x));

    PetscCall(VecCreate(PETSC_COMM_WORLD,&v3));
    PetscCall(VecSetSizes(v3,n,PETSC_DECIDE));
    PetscCall(VecSetFromOptions(v3));
    PetscCall(VecDuplicate(v3,&v4));

    norm = 0.0;
    for (i=0; i<10; i++) {
      PetscCall(VecSetRandom(x,rdm));
      PetscCall(MatMult(P,x,v1));
      PetscCall(MatMult(A,v1,v2));  /* v2 = A*P*x */

      PetscCall(MatMultTranspose(P,v2,v3)); /* v3 = Pt*A*P*x */
      PetscCall(MatMult(C,x,v4));           /* v3 = C*x   */
      PetscCall(VecNorm(v4,NORM_2,&norm_abs));
      PetscCall(VecAXPY(v4,none,v3));
      PetscCall(VecNorm(v4,NORM_2,&norm_tmp));

      norm_tmp /= norm_abs;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    PetscCheckFalse(norm >= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatPtAP(), |v1 - v2|: %g",(double)norm);

    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&P));
    PetscCall(MatDestroy(&C));
    PetscCall(VecDestroy(&v3));
    PetscCall(VecDestroy(&v4));
    PetscCall(VecDestroy(&x));
  }

  /* Destroy objects */
  PetscCall(VecDestroy(&v1));
  PetscCall(VecDestroy(&v2));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFree(idxn));

  PetscCall(MatDestroy(&A_save));
  PetscCall(MatDestroy(&B));

  PetscPreLoadEnd();
  PetscCall(PetscFinalize());
  return 0;
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
