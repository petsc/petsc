
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAXPY(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_FROBENIUS,norm);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-matops_view",&view,NULL);CHKERRQ(ierr);
  if (view) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  }

  /*  Load the matrices A_save and B */
  ierr = PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for small matrix A with the -f0 option.");
  ierr = PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for small matrix B with the -f1 option.");
  ierr = PetscOptionsGetString(NULL,NULL,"-f2",file[2],sizeof(file[2]),&flg);CHKERRQ(ierr);
  if (!flg) {
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(NULL,NULL,"-f3",file[3],128,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for test matrix B with the -f3 option.");
  }

  PetscPreLoadBegin(preload,"Load system");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A_save);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A_save);CHKERRQ(ierr);
  ierr = MatLoad(A_save,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt+1],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatLoad(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatGetType(B,&mattype);CHKERRQ(ierr);

  ierr = MatGetSize(B,&M,&N);CHKERRQ(ierr);
  nzp  = PetscMax((PetscInt)(0.1*M),5);
  ierr = PetscMalloc((nzp+1)*(sizeof(PetscInt)+sizeof(PetscScalar)),&idxn);CHKERRQ(ierr);
  a    = (PetscScalar*)(idxn + nzp);

  /* Create vectors v1 and v2 that are compatible with A_save */
  ierr = VecCreate(PETSC_COMM_WORLD,&v1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A_save,&m,NULL);CHKERRQ(ierr);
  ierr = VecSetSizes(v1,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v1);CHKERRQ(ierr);
  ierr = VecDuplicate(v1,&v2);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL);CHKERRQ(ierr);

  /* Test MatAXPY()    */
  /*-------------------*/
  ierr = PetscOptionsHasName(NULL,NULL,"-test_MatAXPY",&Test_MatAXPY);CHKERRQ(ierr);
  if (Test_MatAXPY) {
    Mat Btmp;
    ierr = MatDuplicate(A_save,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&Btmp);CHKERRQ(ierr);
    ierr = MatAXPY(A,-1.0,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* A = -B + A_save */

    ierr = MatScale(A,-1.0);CHKERRQ(ierr); /* A = -A = B - A_save */
    ierr = MatAXPY(Btmp,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* Btmp = -A + B = A_save */
    ierr = MatMultEqual(A_save,Btmp,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatAXPY() is incorrect\n");
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&Btmp);CHKERRQ(ierr);

    Test_MatMatMult    = PETSC_FALSE;
    Test_MatMatTr      = PETSC_FALSE;
    Test_MatPtAP       = PETSC_FALSE;
    Test_MatRARt       = PETSC_FALSE;
    Test_MatMatMatMult = PETSC_FALSE;
  }

  /* 1) Test MatMatMult() */
  /* ---------------------*/
  if (Test_MatMatMult) {
    ierr = MatDuplicate(A_save,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatCreateTranspose(A,&AT);CHKERRQ(ierr);
    ierr = MatCreateTranspose(AT,&ATT);CHKERRQ(ierr);
    ierr = MatCreateTranspose(B,&BT);CHKERRQ(ierr);
    ierr = MatCreateTranspose(BT,&BTT);CHKERRQ(ierr);

    ierr = MatMatMult(AT,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatMatMultEqual(AT,B,C,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=AT*B");
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    ierr = MatMatMult(ATT,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatMatMultEqual(ATT,B,C,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=ATT*B");
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatMatMultEqual(A,B,C,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for reuse C=A*B");
    /* ATT has different matrix type as A (although they have same internal data structure),
       we cannot call MatProductReplaceMats(ATT,NULL,NULL,C) and MatMatMult(ATT,B,MAT_REUSE_MATRIX,fill,&C) */
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    ierr = MatMatMult(A,BTT,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatMatMultEqual(A,BTT,C,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult() for C=A*BTT");
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    ierr = MatMatMult(ATT,BTT,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatMatMultEqual(A,B,C,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()\n");
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    ierr = MatDestroy(&BTT);CHKERRQ(ierr);
    ierr = MatDestroy(&BT);CHKERRQ(ierr);
    ierr = MatDestroy(&ATT);CHKERRQ(ierr);
    ierr = MatDestroy(&AT);CHKERRQ(ierr);

    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(C,"matmatmult_");CHKERRQ(ierr); /* enable option '-matmatmult_' for matrix C */
    ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -=0.1;
      ierr   = MatScale(A,alpha);CHKERRQ(ierr);
      ierr   = MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }
    ierr = MatMatMultEqual(A,B,C,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()\n");
    ierr = MatDestroy(&A);CHKERRQ(ierr);

    /* Test MatDuplicate() of C=A*B */
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
  } /* if (Test_MatMatMult) */

  /* 2) Test MatTransposeMatMult() and MatMatTransposeMult() */
  /* ------------------------------------------------------- */
  if (Test_MatMatTr) {
    /* Create P */
    PetscInt PN,rstart,rend;
    PN   = M/2;
    nzp  = 5; /* num of nonzeros in each row of P */
    ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
    ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,M,PN);CHKERRQ(ierr);
    ierr = MatSetType(P,mattype);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(P,nzp,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(P,nzp,NULL,nzp,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(P,&rstart,&rend);CHKERRQ(ierr);
    for (i=0; i<nzp; i++) {
      ierr = PetscRandomGetValue(rdm,&a[i]);CHKERRQ(ierr);
    }
    for (i=rstart; i<rend; i++) {
      for (j=0; j<nzp; j++) {
        ierr    = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      ierr = MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Create R = P^T */
    ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);

    { /* Test R = P^T, C1 = R*B */
      ierr = MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1);CHKERRQ(ierr);
      ierr = MatTranspose(P,MAT_REUSE_MATRIX,&R);CHKERRQ(ierr);
      ierr = MatMatMult(R,B,MAT_REUSE_MATRIX,fill,&C1);CHKERRQ(ierr);
      ierr = MatDestroy(&C1);CHKERRQ(ierr);
    }

    /* C = P^T*B */
    ierr = MatTransposeMatMult(P,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    ierr = MatTransposeMatMult(P,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    if (view) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"C = P^T * B:\n");CHKERRQ(ierr);
      ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = MatFreeIntermediateDataStructures(C);CHKERRQ(ierr);
    if (view) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * B after MatFreeIntermediateDataStructures():\n");CHKERRQ(ierr);
      ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    /* Compare P^T*B and R*B */
    ierr = MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1);CHKERRQ(ierr);
    ierr = MatNormDifference(C,C1,&norm);CHKERRQ(ierr);
    if (norm > PETSC_SMALL) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatTransposeMatMult(): %g\n",(double)norm);
    ierr = MatDestroy(&C1);CHKERRQ(ierr);

    /* Test MatDuplicate() of C=P^T*B */
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    /* C = B*R^T */
    ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQAIJ,&seqaij);CHKERRQ(ierr);
    if (size == 1 && seqaij) {
      ierr = MatMatTransposeMult(B,R,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(C,"matmatmulttr_");CHKERRQ(ierr); /* enable '-matmatmulttr_' for matrix C */
      ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);

      /* Test MAT_REUSE_MATRIX - reuse symbolic C */
      ierr = MatMatTransposeMult(B,R,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);

      /* Check */
      ierr = MatMatMult(B,P,MAT_INITIAL_MATRIX,fill,&C1);CHKERRQ(ierr);
      ierr = MatNormDifference(C,C1,&norm);CHKERRQ(ierr);
      if (norm > PETSC_SMALL) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatTransposeMult() %g\n",(double)norm);
      ierr = MatDestroy(&C1);CHKERRQ(ierr);
      ierr = MatDestroy(&C);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatDestroy(&R);CHKERRQ(ierr);
  }

  /* 3) Test MatPtAP() */
  /*-------------------*/
  if (Test_MatPtAP) {
    PetscInt  PN;
    Mat       Cdup;

    ierr = MatDuplicate(A_save,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);

    PN   = M/2;
    nzp  = (PetscInt)(0.1*PN+1); /* num of nozeros in each row of P */
    ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
    ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,N,PN);CHKERRQ(ierr);
    ierr = MatSetType(P,mattype);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(P,nzp,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(P,nzp,NULL,nzp,NULL);CHKERRQ(ierr);
    for (i=0; i<nzp; i++) {
      ierr = PetscRandomGetValue(rdm,&a[i]);CHKERRQ(ierr);
    }
    ierr = MatGetOwnershipRange(P,&rstart,&rend);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      for (j=0; j<nzp; j++) {
        ierr    = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      ierr = MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* ierr = MatView(P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = MatGetSize(P,&pM,&pN);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,&pm,&pn);CHKERRQ(ierr);
    ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -=0.1;
      ierr   = MatScale(A,alpha);CHKERRQ(ierr);
      ierr   = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }

    /* Test PtAP ops with P Dense and A either AIJ or SeqDense (it assumes MatPtAP_XAIJ_XAIJ is fine) */
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&seqaij);CHKERRQ(ierr);
    if (seqaij) {
      ierr = MatConvert(C,MATSEQDENSE,MAT_INITIAL_MATRIX,&Cdensetest);CHKERRQ(ierr);
      ierr = MatConvert(P,MATSEQDENSE,MAT_INITIAL_MATRIX,&Pdense);CHKERRQ(ierr);
    } else {
      ierr = MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdensetest);CHKERRQ(ierr);
      ierr = MatConvert(P,MATMPIDENSE,MAT_INITIAL_MATRIX,&Pdense);CHKERRQ(ierr);
    }

    /* test with A(AIJ), Pdense -- call MatPtAP_Basic() when np>1 */
    ierr = MatPtAP(A,Pdense,MAT_INITIAL_MATRIX,fill,&Cdense);CHKERRQ(ierr);
    ierr = MatPtAP(A,Pdense,MAT_REUSE_MATRIX,fill,&Cdense);CHKERRQ(ierr);
    ierr = MatPtAPMultEqual(A,Pdense,Cdense,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP with A AIJ and P Dense");
    ierr = MatDestroy(&Cdense);CHKERRQ(ierr);

    /* test with A SeqDense */
    if (seqaij) {
      ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
      ierr = MatPtAP(Adense,Pdense,MAT_INITIAL_MATRIX,fill,&Cdense);CHKERRQ(ierr);
      ierr = MatPtAP(Adense,Pdense,MAT_REUSE_MATRIX,fill,&Cdense);CHKERRQ(ierr);
      ierr = MatPtAPMultEqual(Adense,Pdense,Cdense,10,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatPtAP with A SeqDense and P SeqDense");
      ierr = MatDestroy(&Cdense);CHKERRQ(ierr);
      ierr = MatDestroy(&Adense);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Cdensetest);CHKERRQ(ierr);
    ierr = MatDestroy(&Pdense);CHKERRQ(ierr);

    /* Test MatDuplicate() of C=PtAP and MatView(Cdup,...) */
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&Cdup);CHKERRQ(ierr);
    if (view) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * A * P:\n");CHKERRQ(ierr);
      ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

      ierr = MatFreeIntermediateDataStructures(C);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nC = P^T * A * P after MatFreeIntermediateDataStructures():\n");CHKERRQ(ierr);
      ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCdup:\n");CHKERRQ(ierr);
      ierr = MatView(Cdup,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Cdup);CHKERRQ(ierr);

    if (size>1 || !seqaij) Test_MatRARt = PETSC_FALSE;
    /* 4) Test MatRARt() */
    /* ----------------- */
    if (Test_MatRARt) {
      Mat R, RARt, Rdense, RARtdense;
      ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);

      /* Test MatRARt_Basic(), MatMatMatMult_Basic() */
      ierr = MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&Rdense);CHKERRQ(ierr);
      ierr = MatRARt(A,Rdense,MAT_INITIAL_MATRIX,2.0,&RARtdense);CHKERRQ(ierr);
      ierr = MatRARt(A,Rdense,MAT_REUSE_MATRIX,2.0,&RARtdense);CHKERRQ(ierr);

      ierr = MatConvert(RARtdense,MATAIJ,MAT_INITIAL_MATRIX,&RARt);CHKERRQ(ierr);
      ierr = MatNormDifference(C,RARt,&norm);CHKERRQ(ierr);
      if (norm > PETSC_SMALL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"|PtAP - RARtdense| = %g",(double)norm);
      ierr = MatDestroy(&Rdense);CHKERRQ(ierr);
      ierr = MatDestroy(&RARtdense);CHKERRQ(ierr);
      ierr = MatDestroy(&RARt);CHKERRQ(ierr);

      /* Test MatRARt() for aij matrices */
      ierr = MatRARt(A,R,MAT_INITIAL_MATRIX,2.0,&RARt);CHKERRQ(ierr);
      ierr = MatRARt(A,R,MAT_REUSE_MATRIX,2.0,&RARt);CHKERRQ(ierr);
      ierr = MatNormDifference(C,RARt,&norm);CHKERRQ(ierr);
      if (norm > PETSC_SMALL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"|PtAP - RARt| = %g",(double)norm);
      ierr = MatDestroy(&R);CHKERRQ(ierr);
      ierr = MatDestroy(&RARt);CHKERRQ(ierr);
    }

    if (Test_MatMatMatMult && size == 1) {
      Mat       R, RAP;
      ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);
      ierr = MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,2.0,&RAP);CHKERRQ(ierr);
      ierr = MatMatMatMult(R,A,P,MAT_REUSE_MATRIX,2.0,&RAP);CHKERRQ(ierr);
      ierr = MatNormDifference(C,RAP,&norm);CHKERRQ(ierr);
      if (norm > PETSC_SMALL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PtAP != RAP %g",(double)norm);
      ierr = MatDestroy(&R);CHKERRQ(ierr);
      ierr = MatDestroy(&RAP);CHKERRQ(ierr);
    }

    /* Create vector x that is compatible with P */
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,&m,&n);CHKERRQ(ierr);
    ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&v3);CHKERRQ(ierr);
    ierr = VecSetSizes(v3,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(v3);CHKERRQ(ierr);
    ierr = VecDuplicate(v3,&v4);CHKERRQ(ierr);

    norm = 0.0;
    for (i=0; i<10; i++) {
      ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
      ierr = MatMult(P,x,v1);CHKERRQ(ierr);
      ierr = MatMult(A,v1,v2);CHKERRQ(ierr);  /* v2 = A*P*x */

      ierr = MatMultTranspose(P,v2,v3);CHKERRQ(ierr); /* v3 = Pt*A*P*x */
      ierr = MatMult(C,x,v4);CHKERRQ(ierr);           /* v3 = C*x   */
      ierr = VecNorm(v4,NORM_2,&norm_abs);CHKERRQ(ierr);
      ierr = VecAXPY(v4,none,v3);CHKERRQ(ierr);
      ierr = VecNorm(v4,NORM_2,&norm_tmp);CHKERRQ(ierr);

      norm_tmp /= norm_abs;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= PETSC_SMALL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatPtAP(), |v1 - v2|: %g\n",(double)norm);

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = VecDestroy(&v3);CHKERRQ(ierr);
    ierr = VecDestroy(&v4);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }

  /* Destroy objects */
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);

  ierr = MatDestroy(&A_save);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  PetscPreLoadEnd();
  PetscFinalize();
  return ierr;
}



/*TEST

   test:
      suffix: 2_mattransposematmult_matmatmult
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium -mattransposematmult_via at*b> ex94_2.tmp 2>&1

   test:
      suffix: 2_mattransposematmult_scalable
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium -mattransposematmult_via scalable> ex94_2.tmp 2>&1
      output_file: output/ex94_1.out

   test:
      suffix: axpy_mpiaij
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      nsize: 8
      args: -f0 ${DATAFILESPATH}/matrices/poisson_2d5p -f1 ${DATAFILESPATH}/matrices/poisson_2d13p -test_MatAXPY
      output_file: output/ex94_1.out

   test:
      suffix: axpy_mpibaij
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      nsize: 8
      args: -f0 ${DATAFILESPATH}/matrices/poisson_2d5p -f1 ${DATAFILESPATH}/matrices/poisson_2d13p -test_MatAXPY -mat_type baij
      output_file: output/ex94_1.out

   test:
      suffix: axpy_mpisbaij
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      nsize: 8
      args: -f0 ${DATAFILESPATH}/matrices/poisson_2d5p -f1 ${DATAFILESPATH}/matrices/poisson_2d13p -test_MatAXPY -mat_type sbaij
      output_file: output/ex94_1.out

   test:
      suffix: matmatmult
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info
      output_file: output/ex94_1.out

   test:
      suffix: matmatmult_2
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -mat_type mpiaij -viewer_binary_skip_info
      output_file: output/ex94_1.out

   test:
      suffix: matmatmult_scalable
      nsize: 4
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -matmatmult_via scalable
      output_file: output/ex94_1.out

   test:
      suffix: ptap
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium -matptap_via scalable
      output_file: output/ex94_1.out

   test:
      suffix: rap
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/medium
      output_file: output/ex94_1.out

   test:
      suffix: scalable0
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info
      output_file: output/ex94_1.out

   test:
      suffix: scalable1
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/arco1 -f1 ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info -matptap_via scalable
      output_file: output/ex94_1.out

   test:
      suffix: view
      nsize: 2
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/tiny -f1 ${DATAFILESPATH}/matrices/tiny -viewer_binary_skip_info -matops_view
      output_file: output/ex94_2.out

TEST*/
