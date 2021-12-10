
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = R * A * R^T
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

PetscErrorCode MatDestroy_SeqAIJ_RARt(void *data)
{
  PetscErrorCode ierr;
  Mat_RARt       *rart = (Mat_RARt*)data;

  PetscFunctionBegin;
  ierr = MatTransposeColoringDestroy(&rart->matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->Rt);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->RARt);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->ARt);CHKERRQ(ierr);
  ierr = PetscFree(rart->work);CHKERRQ(ierr);
  if (rart->destroy) {
    ierr = (*rart->destroy)(rart->data);CHKERRQ(ierr);
  }
  ierr = PetscFree(rart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ_colorrart(Mat A,Mat R,PetscReal fill,Mat C)
{
  PetscErrorCode       ierr;
  Mat                  P;
  PetscInt             *rti,*rtj;
  Mat_RARt             *rart;
  MatColoring          coloring;
  MatTransposeColoring matcoloring;
  ISColoring           iscoloring;
  Mat                  Rt_dense,RARt_dense;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* create symbolic P=Rt */
  ierr = MatGetSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,R->cmap->n,R->rmap->n,rti,rtj,NULL,&P);CHKERRQ(ierr);

  /* get symbolic C=Pt*A*P */
  ierr = MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy(A,P,fill,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(C,PetscAbs(R->rmap->bs),PetscAbs(R->rmap->bs));CHKERRQ(ierr);
  C->ops->rartnumeric = MatRARtNumeric_SeqAIJ_SeqAIJ_colorrart;

  /* create a supporting struct */
  ierr = PetscNew(&rart);CHKERRQ(ierr);
  C->product->data    = rart;
  C->product->destroy = MatDestroy_SeqAIJ_RARt;

  /* ------ Use coloring ---------- */
  /* inode causes memory problem */
  ierr = MatSetOption(C,MAT_USE_INODES,PETSC_FALSE);CHKERRQ(ierr);

  /* Create MatTransposeColoring from symbolic C=R*A*R^T */
  ierr = MatColoringCreate(C,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(coloring,2);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatTransposeColoringCreate(C,iscoloring,&matcoloring);CHKERRQ(ierr);

  rart->matcoloring = matcoloring;
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

  /* Create Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&Rt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Rt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(Rt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Rt_dense,NULL);CHKERRQ(ierr);

  Rt_dense->assembled = PETSC_TRUE;
  rart->Rt            = Rt_dense;

  /* Create RARt_dense = R*A*Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&RARt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(RARt_dense,C->rmap->n,matcoloring->ncolors,C->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(RARt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(RARt_dense,NULL);CHKERRQ(ierr);

  rart->RARt = RARt_dense;

  /* Allocate work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  ierr = PetscMalloc1(A->rmap->n*4,&rart->work);CHKERRQ(ierr);

  /* clean up */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  {
    Mat_SeqAIJ *c = (Mat_SeqAIJ*)C->data;
    PetscReal density = (PetscReal)(c->nz)/(RARt_dense->rmap->n*RARt_dense->cmap->n);
    ierr = PetscInfo(C,"C=R*(A*Rt) via coloring C - use sparse-dense inner products\n");CHKERRQ(ierr);
    ierr = PetscInfo6(C,"RARt_den %" PetscInt_FMT " %" PetscInt_FMT "; Rt %" PetscInt_FMT " %" PetscInt_FMT " (RARt->nz %" PetscInt_FMT ")/(m*ncolors)=%g\n",RARt_dense->rmap->n,RARt_dense->cmap->n,R->cmap->n,R->rmap->n,c->nz,density);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*
 RAB = R * A * B, R and A in seqaij format, B in dense format;
*/
PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense(Mat R,Mat A,Mat B,Mat RAB,PetscScalar *work)
{
  Mat_SeqAIJ        *a=(Mat_SeqAIJ*)A->data,*r=(Mat_SeqAIJ*)R->data;
  PetscErrorCode    ierr;
  PetscScalar       r1,r2,r3,r4;
  const PetscScalar *b,*b1,*b2,*b3,*b4;
  MatScalar         *aa,*ra;
  PetscInt          cn =B->cmap->n,bm=B->rmap->n,col,i,j,n,*ai=a->i,*aj,am=A->rmap->n;
  PetscInt          am2=2*am,am3=3*am,bm4=4*bm;
  PetscScalar       *d,*c,*c2,*c3,*c4;
  PetscInt          *rj,rm=R->rmap->n,dm=RAB->rmap->n,dn=RAB->cmap->n;
  PetscInt         rm2=2*rm,rm3=3*rm,colrm;

  PetscFunctionBegin;
  if (!dm || !dn) PetscFunctionReturn(0);
  if (bm != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in A %" PetscInt_FMT " not equal rows in B %" PetscInt_FMT "",A->cmap->n,bm);
  if (am != R->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in R %" PetscInt_FMT " not equal rows in A %" PetscInt_FMT "",R->cmap->n,am);
  if (R->rmap->n != RAB->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows in RAB %" PetscInt_FMT " not equal rows in R %" PetscInt_FMT "",RAB->rmap->n,R->rmap->n);
  if (B->cmap->n != RAB->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in RAB %" PetscInt_FMT " not equal columns in B %" PetscInt_FMT "",RAB->cmap->n,B->cmap->n);

  { /*
     This approach is not as good as original ones (will be removed later), but it reveals that
     AB_den=A*B takes almost all execution time in R*A*B for src/ksp/ksp/tutorials/ex56.c
     */
    PetscBool via_matmatmult=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-matrart_via_matmatmult",&via_matmatmult,NULL);CHKERRQ(ierr);
    if (via_matmatmult) {
      Mat AB_den = NULL;
      ierr = MatCreate(PetscObjectComm((PetscObject)A),&AB_den);CHKERRQ(ierr);
      ierr = MatMatMultSymbolic_SeqAIJ_SeqDense(A,B,0.0,AB_den);CHKERRQ(ierr);
      ierr = MatMatMultNumeric_SeqAIJ_SeqDense(A,B,AB_den);CHKERRQ(ierr);
      ierr = MatMatMultNumeric_SeqAIJ_SeqDense(R,AB_den,RAB);CHKERRQ(ierr);
      ierr = MatDestroy(&AB_den);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
  ierr = MatDenseGetArray(RAB,&d);CHKERRQ(ierr);
  b1   = b; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;
  c    = work; c2 = c + am; c3 = c2 + am; c4 = c3 + am;
  for (col=0; col<cn-4; col += 4) {  /* over columns of C */
    for (i=0; i<am; i++) {        /* over rows of A in those columns */
      r1 = r2 = r3 = r4 = 0.0;
      n  = ai[i+1] - ai[i];
      aj = a->j + ai[i];
      aa = a->a + ai[i];
      for (j=0; j<n; j++) {
        r1 += (*aa)*b1[*aj];
        r2 += (*aa)*b2[*aj];
        r3 += (*aa)*b3[*aj];
        r4 += (*aa++)*b4[*aj++];
      }
      c[i]       = r1;
      c[am  + i] = r2;
      c[am2 + i] = r3;
      c[am3 + i] = r4;
    }
    b1 += bm4;
    b2 += bm4;
    b3 += bm4;
    b4 += bm4;

    /* RAB[:,col] = R*C[:,col] */
    colrm = col*rm;
    for (i=0; i<rm; i++) {        /* over rows of R in those columns */
      r1 = r2 = r3 = r4 = 0.0;
      n  = r->i[i+1] - r->i[i];
      rj = r->j + r->i[i];
      ra = r->a + r->i[i];
      for (j=0; j<n; j++) {
        r1 += (*ra)*c[*rj];
        r2 += (*ra)*c2[*rj];
        r3 += (*ra)*c3[*rj];
        r4 += (*ra++)*c4[*rj++];
      }
      d[colrm + i]       = r1;
      d[colrm + rm + i]  = r2;
      d[colrm + rm2 + i] = r3;
      d[colrm + rm3 + i] = r4;
    }
  }
  for (; col<cn; col++) {     /* over extra columns of C */
    for (i=0; i<am; i++) {  /* over rows of A in those columns */
      r1 = 0.0;
      n  = a->i[i+1] - a->i[i];
      aj = a->j + a->i[i];
      aa = a->a + a->i[i];
      for (j=0; j<n; j++) {
        r1 += (*aa++)*b1[*aj++];
      }
      c[i] = r1;
    }
    b1 += bm;

    for (i=0; i<rm; i++) {  /* over rows of R in those columns */
      r1 = 0.0;
      n  = r->i[i+1] - r->i[i];
      rj = r->j + r->i[i];
      ra = r->a + r->i[i];
      for (j=0; j<n; j++) {
        r1 += (*ra++)*c[*rj++];
      }
      d[col*rm + i] = r1;
    }
  }
  ierr = PetscLogFlops(cn*2.0*(a->nz + r->nz));CHKERRQ(ierr);

  ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(RAB,&d);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(RAB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RAB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ_colorrart(Mat A,Mat R,Mat C)
{
  PetscErrorCode       ierr;
  Mat_RARt             *rart;
  MatTransposeColoring matcoloring;
  Mat                  Rt,RARt;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;

  /* Get dense Rt by Apply MatTransposeColoring to R */
  matcoloring = rart->matcoloring;
  Rt          = rart->Rt;
  ierr  = MatTransColoringApplySpToDen(matcoloring,R,Rt);CHKERRQ(ierr);

  /* Get dense RARt = R*A*Rt -- dominates! */
  RARt = rart->RARt;
  ierr = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense(R,A,Rt,RARt,rart->work);CHKERRQ(ierr);

  /* Recover C from C_dense */
  ierr = MatTransColoringApplyDenToSp(matcoloring,RARt,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ_matmattransposemult(Mat A,Mat R,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;
  Mat            ARt;
  Mat_RARt       *rart;
  char           *alg;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* create symbolic ARt = A*R^T  */
  ierr = MatProductCreate(A,R,NULL,&ARt);CHKERRQ(ierr);
  ierr = MatProductSetType(ARt,MATPRODUCT_ABt);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(ARt,"sorted");CHKERRQ(ierr);
  ierr = MatProductSetFill(ARt,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(ARt);CHKERRQ(ierr);
  ierr = MatProductSymbolic(ARt);CHKERRQ(ierr);

  /* compute symbolic C = R*ARt */
  /* set algorithm for C = R*ARt */
  ierr = PetscStrallocpy(C->product->alg,&alg);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,"sorted");CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(R,ARt,fill,C);CHKERRQ(ierr);
  /* resume original algorithm for C */
  ierr = MatProductSetAlgorithm(C,alg);CHKERRQ(ierr);
  ierr = PetscFree(alg);CHKERRQ(ierr);

  C->ops->rartnumeric = MatRARtNumeric_SeqAIJ_SeqAIJ_matmattransposemult;

  ierr = PetscNew(&rart);CHKERRQ(ierr);
  rart->ARt = ARt;
  C->product->data    = rart;
  C->product->destroy = MatDestroy_SeqAIJ_RARt;
  ierr = PetscInfo(C,"Use ARt=A*R^T, C=R*ARt via MatMatTransposeMult(). Coloring can be applied to A*R^T.\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ_matmattransposemult(Mat A,Mat R,Mat C)
{
  PetscErrorCode ierr;
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;
  ierr = MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(A,R,rart->ARt);CHKERRQ(ierr); /* dominate! */
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ(R,rart->ARt,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat R,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;
  Mat            Rt;
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  ierr = MatTranspose_SeqAIJ(R,MAT_INITIAL_MATRIX,&Rt);CHKERRQ(ierr);
  ierr = MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(R,A,Rt,fill,C);CHKERRQ(ierr);

  ierr = PetscNew(&rart);CHKERRQ(ierr);
  rart->data = C->product->data;
  rart->destroy = C->product->destroy;
  rart->Rt = Rt;
  C->product->data    = rart;
  C->product->destroy = MatDestroy_SeqAIJ_RARt;
  C->ops->rartnumeric = MatRARtNumeric_SeqAIJ_SeqAIJ;
  ierr = PetscInfo(C,"Use Rt=R^T and C=R*A*Rt via MatMatMatMult() to avoid sparse inner products\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ(Mat A,Mat R,Mat C)
{
  PetscErrorCode ierr;
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;
  ierr = MatTranspose_SeqAIJ(R,MAT_REUSE_MATRIX,&rart->Rt);CHKERRQ(ierr);
  /* MatMatMatMultSymbolic used a different data */
  C->product->data = rart->data;
  ierr = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(R,A,rart->Rt,C);CHKERRQ(ierr);
  C->product->data = rart;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARt_SeqAIJ_SeqAIJ(Mat A,Mat R,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  const char     *algTypes[3] = {"matmatmatmult","matmattransposemult","coloring_rart"};
  PetscInt       alg=0; /* set default algorithm */

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatRARt","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matrart_via","Algorithmic approach","MatRARt",algTypes,3,algTypes[0],&alg,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    ierr = PetscLogEventBegin(MAT_RARtSymbolic,A,R,0,0);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,C);CHKERRQ(ierr);
    switch (alg) {
    case 1:
      /* via matmattransposemult: ARt=A*R^T, C=R*ARt - matrix coloring can be applied to A*R^T */
      ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ_matmattransposemult(A,R,fill,*C);CHKERRQ(ierr);
      break;
    case 2:
      /* via coloring_rart: apply coloring C = R*A*R^T                          */
      ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ_colorrart(A,R,fill,*C);CHKERRQ(ierr);
      break;
    default:
      /* via matmatmatmult: Rt=R^T, C=R*A*Rt - avoid inefficient sparse inner products */
      ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ(A,R,fill,*C);CHKERRQ(ierr);
      break;
    }
    ierr = PetscLogEventEnd(MAT_RARtSymbolic,A,R,0,0);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(MAT_RARtNumeric,A,R,0,0);CHKERRQ(ierr);
  ierr = ((*C)->ops->rartnumeric)(A,R,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_RARtNumeric,A,R,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------- */
PetscErrorCode MatProductSymbolic_RARt_SeqAIJ_SeqAIJ(Mat C)
{
  PetscErrorCode      ierr;
  Mat_Product         *product = C->product;
  Mat                 A=product->A,R=product->B;
  MatProductAlgorithm alg=product->alg;
  PetscReal           fill=product->fill;
  PetscBool           flg;

  PetscFunctionBegin;
  ierr = PetscStrcmp(alg,"r*a*rt",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ(A,R,fill,C);CHKERRQ(ierr);
    goto next;
  }

  ierr = PetscStrcmp(alg,"r*art",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ_matmattransposemult(A,R,fill,C);CHKERRQ(ierr);
    goto next;
  }

  ierr = PetscStrcmp(alg,"coloring_rart",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ_colorrart(A,R,fill,C);CHKERRQ(ierr);
    goto next;
  }

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProductAlgorithm is not supported");

next:
  C->ops->productnumeric = MatProductNumeric_RARt;
  PetscFunctionReturn(0);
}
