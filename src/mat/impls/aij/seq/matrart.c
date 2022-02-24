
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = R * A * R^T
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

PetscErrorCode MatDestroy_SeqAIJ_RARt(void *data)
{
  Mat_RARt       *rart = (Mat_RARt*)data;

  PetscFunctionBegin;
  CHKERRQ(MatTransposeColoringDestroy(&rart->matcoloring));
  CHKERRQ(MatDestroy(&rart->Rt));
  CHKERRQ(MatDestroy(&rart->RARt));
  CHKERRQ(MatDestroy(&rart->ARt));
  CHKERRQ(PetscFree(rart->work));
  if (rart->destroy) {
    CHKERRQ((*rart->destroy)(rart->data));
  }
  CHKERRQ(PetscFree(rart));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ_colorrart(Mat A,Mat R,PetscReal fill,Mat C)
{
  Mat                  P;
  PetscInt             *rti,*rtj;
  Mat_RARt             *rart;
  MatColoring          coloring;
  MatTransposeColoring matcoloring;
  ISColoring           iscoloring;
  Mat                  Rt_dense,RARt_dense;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* create symbolic P=Rt */
  CHKERRQ(MatGetSymbolicTranspose_SeqAIJ(R,&rti,&rtj));
  CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,R->cmap->n,R->rmap->n,rti,rtj,NULL,&P));

  /* get symbolic C=Pt*A*P */
  CHKERRQ(MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy(A,P,fill,C));
  CHKERRQ(MatSetBlockSizes(C,PetscAbs(R->rmap->bs),PetscAbs(R->rmap->bs)));
  C->ops->rartnumeric = MatRARtNumeric_SeqAIJ_SeqAIJ_colorrart;

  /* create a supporting struct */
  CHKERRQ(PetscNew(&rart));
  C->product->data    = rart;
  C->product->destroy = MatDestroy_SeqAIJ_RARt;

  /* ------ Use coloring ---------- */
  /* inode causes memory problem */
  CHKERRQ(MatSetOption(C,MAT_USE_INODES,PETSC_FALSE));

  /* Create MatTransposeColoring from symbolic C=R*A*R^T */
  CHKERRQ(MatColoringCreate(C,&coloring));
  CHKERRQ(MatColoringSetDistance(coloring,2));
  CHKERRQ(MatColoringSetType(coloring,MATCOLORINGSL));
  CHKERRQ(MatColoringSetFromOptions(coloring));
  CHKERRQ(MatColoringApply(coloring,&iscoloring));
  CHKERRQ(MatColoringDestroy(&coloring));
  CHKERRQ(MatTransposeColoringCreate(C,iscoloring,&matcoloring));

  rart->matcoloring = matcoloring;
  CHKERRQ(ISColoringDestroy(&iscoloring));

  /* Create Rt_dense */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&Rt_dense));
  CHKERRQ(MatSetSizes(Rt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors));
  CHKERRQ(MatSetType(Rt_dense,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(Rt_dense,NULL));

  Rt_dense->assembled = PETSC_TRUE;
  rart->Rt            = Rt_dense;

  /* Create RARt_dense = R*A*Rt_dense */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&RARt_dense));
  CHKERRQ(MatSetSizes(RARt_dense,C->rmap->n,matcoloring->ncolors,C->rmap->n,matcoloring->ncolors));
  CHKERRQ(MatSetType(RARt_dense,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(RARt_dense,NULL));

  rart->RARt = RARt_dense;

  /* Allocate work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  CHKERRQ(PetscMalloc1(A->rmap->n*4,&rart->work));

  /* clean up */
  CHKERRQ(MatRestoreSymbolicTranspose_SeqAIJ(R,&rti,&rtj));
  CHKERRQ(MatDestroy(&P));

#if defined(PETSC_USE_INFO)
  {
    Mat_SeqAIJ *c = (Mat_SeqAIJ*)C->data;
    PetscReal density = (PetscReal)(c->nz)/(RARt_dense->rmap->n*RARt_dense->cmap->n);
    CHKERRQ(PetscInfo(C,"C=R*(A*Rt) via coloring C - use sparse-dense inner products\n"));
    CHKERRQ(PetscInfo(C,"RARt_den %" PetscInt_FMT " %" PetscInt_FMT "; Rt %" PetscInt_FMT " %" PetscInt_FMT " (RARt->nz %" PetscInt_FMT ")/(m*ncolors)=%g\n",RARt_dense->rmap->n,RARt_dense->cmap->n,R->cmap->n,R->rmap->n,c->nz,density));
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
  PetscCheckFalse(bm != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in A %" PetscInt_FMT " not equal rows in B %" PetscInt_FMT,A->cmap->n,bm);
  PetscCheckFalse(am != R->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in R %" PetscInt_FMT " not equal rows in A %" PetscInt_FMT,R->cmap->n,am);
  PetscCheckFalse(R->rmap->n != RAB->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows in RAB %" PetscInt_FMT " not equal rows in R %" PetscInt_FMT,RAB->rmap->n,R->rmap->n);
  PetscCheckFalse(B->cmap->n != RAB->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in RAB %" PetscInt_FMT " not equal columns in B %" PetscInt_FMT,RAB->cmap->n,B->cmap->n);

  { /*
     This approach is not as good as original ones (will be removed later), but it reveals that
     AB_den=A*B takes almost all execution time in R*A*B for src/ksp/ksp/tutorials/ex56.c
     */
    PetscBool via_matmatmult=PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-matrart_via_matmatmult",&via_matmatmult,NULL));
    if (via_matmatmult) {
      Mat AB_den = NULL;
      CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&AB_den));
      CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqDense(A,B,0.0,AB_den));
      CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqDense(A,B,AB_den));
      CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqDense(R,AB_den,RAB));
      CHKERRQ(MatDestroy(&AB_den));
      PetscFunctionReturn(0);
    }
  }

  CHKERRQ(MatDenseGetArrayRead(B,&b));
  CHKERRQ(MatDenseGetArray(RAB,&d));
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
  CHKERRQ(PetscLogFlops(cn*2.0*(a->nz + r->nz)));

  CHKERRQ(MatDenseRestoreArrayRead(B,&b));
  CHKERRQ(MatDenseRestoreArray(RAB,&d));
  CHKERRQ(MatAssemblyBegin(RAB,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(RAB,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ_colorrart(Mat A,Mat R,Mat C)
{
  Mat_RARt             *rart;
  MatTransposeColoring matcoloring;
  Mat                  Rt,RARt;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheckFalse(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;

  /* Get dense Rt by Apply MatTransposeColoring to R */
  matcoloring = rart->matcoloring;
  Rt          = rart->Rt;
  CHKERRQ(MatTransColoringApplySpToDen(matcoloring,R,Rt));

  /* Get dense RARt = R*A*Rt -- dominates! */
  RARt = rart->RARt;
  CHKERRQ(MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense(R,A,Rt,RARt,rart->work));

  /* Recover C from C_dense */
  CHKERRQ(MatTransColoringApplyDenToSp(matcoloring,RARt,C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ_matmattransposemult(Mat A,Mat R,PetscReal fill,Mat C)
{
  Mat            ARt;
  Mat_RARt       *rart;
  char           *alg;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* create symbolic ARt = A*R^T  */
  CHKERRQ(MatProductCreate(A,R,NULL,&ARt));
  CHKERRQ(MatProductSetType(ARt,MATPRODUCT_ABt));
  CHKERRQ(MatProductSetAlgorithm(ARt,"sorted"));
  CHKERRQ(MatProductSetFill(ARt,fill));
  CHKERRQ(MatProductSetFromOptions(ARt));
  CHKERRQ(MatProductSymbolic(ARt));

  /* compute symbolic C = R*ARt */
  /* set algorithm for C = R*ARt */
  CHKERRQ(PetscStrallocpy(C->product->alg,&alg));
  CHKERRQ(MatProductSetAlgorithm(C,"sorted"));
  CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ(R,ARt,fill,C));
  /* resume original algorithm for C */
  CHKERRQ(MatProductSetAlgorithm(C,alg));
  CHKERRQ(PetscFree(alg));

  C->ops->rartnumeric = MatRARtNumeric_SeqAIJ_SeqAIJ_matmattransposemult;

  CHKERRQ(PetscNew(&rart));
  rart->ARt = ARt;
  C->product->data    = rart;
  C->product->destroy = MatDestroy_SeqAIJ_RARt;
  CHKERRQ(PetscInfo(C,"Use ARt=A*R^T, C=R*ARt via MatMatTransposeMult(). Coloring can be applied to A*R^T.\n"));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ_matmattransposemult(Mat A,Mat R,Mat C)
{
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheckFalse(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;
  CHKERRQ(MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(A,R,rart->ARt)); /* dominate! */
  CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqAIJ(R,rart->ARt,C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat R,PetscReal fill,Mat C)
{
  Mat            Rt;
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  CHKERRQ(MatTranspose_SeqAIJ(R,MAT_INITIAL_MATRIX,&Rt));
  CHKERRQ(MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(R,A,Rt,fill,C));

  CHKERRQ(PetscNew(&rart));
  rart->data = C->product->data;
  rart->destroy = C->product->destroy;
  rart->Rt = Rt;
  C->product->data    = rart;
  C->product->destroy = MatDestroy_SeqAIJ_RARt;
  C->ops->rartnumeric = MatRARtNumeric_SeqAIJ_SeqAIJ;
  CHKERRQ(PetscInfo(C,"Use Rt=R^T and C=R*A*Rt via MatMatMatMult() to avoid sparse inner products\n"));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ(Mat A,Mat R,Mat C)
{
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheckFalse(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;
  CHKERRQ(MatTranspose_SeqAIJ(R,MAT_REUSE_MATRIX,&rart->Rt));
  /* MatMatMatMultSymbolic used a different data */
  C->product->data = rart->data;
  CHKERRQ(MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(R,A,rart->Rt,C));
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
    CHKERRQ(PetscOptionsEList("-matrart_via","Algorithmic approach","MatRARt",algTypes,3,algTypes[0],&alg,NULL));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    CHKERRQ(PetscLogEventBegin(MAT_RARtSymbolic,A,R,0,0));
    CHKERRQ(MatCreate(PETSC_COMM_SELF,C));
    switch (alg) {
    case 1:
      /* via matmattransposemult: ARt=A*R^T, C=R*ARt - matrix coloring can be applied to A*R^T */
      CHKERRQ(MatRARtSymbolic_SeqAIJ_SeqAIJ_matmattransposemult(A,R,fill,*C));
      break;
    case 2:
      /* via coloring_rart: apply coloring C = R*A*R^T                          */
      CHKERRQ(MatRARtSymbolic_SeqAIJ_SeqAIJ_colorrart(A,R,fill,*C));
      break;
    default:
      /* via matmatmatmult: Rt=R^T, C=R*A*Rt - avoid inefficient sparse inner products */
      CHKERRQ(MatRARtSymbolic_SeqAIJ_SeqAIJ(A,R,fill,*C));
      break;
    }
    CHKERRQ(PetscLogEventEnd(MAT_RARtSymbolic,A,R,0,0));
  }

  CHKERRQ(PetscLogEventBegin(MAT_RARtNumeric,A,R,0,0));
  CHKERRQ(((*C)->ops->rartnumeric)(A,R,*C));
  CHKERRQ(PetscLogEventEnd(MAT_RARtNumeric,A,R,0,0));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------- */
PetscErrorCode MatProductSymbolic_RARt_SeqAIJ_SeqAIJ(Mat C)
{
  Mat_Product         *product = C->product;
  Mat                 A=product->A,R=product->B;
  MatProductAlgorithm alg=product->alg;
  PetscReal           fill=product->fill;
  PetscBool           flg;

  PetscFunctionBegin;
  CHKERRQ(PetscStrcmp(alg,"r*a*rt",&flg));
  if (flg) {
    CHKERRQ(MatRARtSymbolic_SeqAIJ_SeqAIJ(A,R,fill,C));
    goto next;
  }

  CHKERRQ(PetscStrcmp(alg,"r*art",&flg));
  if (flg) {
    CHKERRQ(MatRARtSymbolic_SeqAIJ_SeqAIJ_matmattransposemult(A,R,fill,C));
    goto next;
  }

  CHKERRQ(PetscStrcmp(alg,"coloring_rart",&flg));
  if (flg) {
    CHKERRQ(MatRARtSymbolic_SeqAIJ_SeqAIJ_colorrart(A,R,fill,C));
    goto next;
  }

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProductAlgorithm is not supported");

next:
  C->ops->productnumeric = MatProductNumeric_RARt;
  PetscFunctionReturn(0);
}
