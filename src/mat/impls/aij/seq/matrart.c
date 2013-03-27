
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsctime.h>

/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = R * A * R^T
*/

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJ_RARt"
PetscErrorCode MatDestroy_SeqAIJ_RARt(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a    = (Mat_SeqAIJ*)A->data;
  Mat_RARt       *rart = a->rart; 

  PetscFunctionBegin;
  ierr = MatTransposeColoringDestroy(&rart->matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->Rt);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->RARt);CHKERRQ(ierr);
  ierr = PetscFree(rart->work);CHKERRQ(ierr);

  A->ops->destroy = rart->destroy;
  if (A->ops->destroy) {
    ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscFree(rart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRARtSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat R,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  Mat                  P;
  PetscInt             *rti,*rtj;
  Mat_RARt             *rart;
  MatTransposeColoring matcoloring;
  ISColoring           iscoloring;
  Mat                  Rt_dense,RARt_dense;
  PetscLogDouble       GColor=0.0,MCCreate=0.0,MDenCreate=0.0,t0,tf,etime=0.0;
  Mat_SeqAIJ           *c;

  PetscFunctionBegin;
  ierr = PetscTime(&t0);CHKERRQ(ierr);
  /* create symbolic P=Rt */
  ierr = MatGetSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,R->cmap->n,R->rmap->n,rti,rtj,NULL,&P);CHKERRQ(ierr);

  /* get symbolic C=Pt*A*P */
  ierr           = MatPtAPSymbolic_SeqAIJ_SeqAIJ(A,P,fill,C);CHKERRQ(ierr);
  (*C)->rmap->bs = R->rmap->bs;
  (*C)->cmap->bs = R->rmap->bs;

  /* create a supporting struct */
  ierr = PetscNew(Mat_RARt,&rart);CHKERRQ(ierr);
  c       = (Mat_SeqAIJ*)(*C)->data;
  c->rart = rart;
  ierr   = PetscTime(&tf);CHKERRQ(ierr);
  etime += tf - t0;

  /* ------ Use coloring ---------- */
  /* Create MatTransposeColoring from symbolic C=R*A*R^T */
  ierr    = PetscTime(&t0);CHKERRQ(ierr);
  ierr    = MatGetColoring(*C,MATCOLORINGLF,&iscoloring);CHKERRQ(ierr);
  ierr    = PetscTime(&tf);CHKERRQ(ierr);
  GColor += tf - t0;

  ierr = PetscTime(&t0);CHKERRQ(ierr);
  ierr = MatTransposeColoringCreate(*C,iscoloring,&matcoloring);CHKERRQ(ierr);

  rart->matcoloring = matcoloring;

  ierr      = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr      = PetscTime(&tf);CHKERRQ(ierr);
  MCCreate += tf - t0;

  ierr = PetscTime(&t0);CHKERRQ(ierr);
  /* Create Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&Rt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Rt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(Rt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Rt_dense,NULL);CHKERRQ(ierr);

  Rt_dense->assembled = PETSC_TRUE;
  rart->Rt            = Rt_dense;

  /* Create RARt_dense = R*A*Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&RARt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(RARt_dense,(*C)->rmap->n,matcoloring->ncolors,(*C)->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(RARt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(RARt_dense,NULL);CHKERRQ(ierr);

  rart->RARt = RARt_dense;

  /* Allocate work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  ierr = PetscMalloc(A->rmap->n*4*sizeof(PetscScalar),&rart->work);CHKERRQ(ierr);

  ierr        = PetscTime(&tf);CHKERRQ(ierr);
  MDenCreate += tf - t0;

  rart->destroy      = (*C)->ops->destroy;
  (*C)->ops->destroy = MatDestroy_SeqAIJ_RARt;

  /* clean up */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  {
    PetscReal density= (PetscReal)(c->nz)/(RARt_dense->rmap->n*RARt_dense->cmap->n);
    ierr = PetscInfo6(*C,"RARt_den %D %D; Rt_den %D %D, (RARt->nz %D)/(m*ncolors)=%g\n",RARt_dense->rmap->n,RARt_dense->cmap->n,Rt_dense->rmap->n,Rt_dense->cmap->n,c->nz,density);CHKERRQ(ierr);
    ierr = PetscInfo5(*C,"Sym = GetColor %g + MColorCreate %g + MDenCreate %g + other %g = %g\n",GColor,MCCreate,MDenCreate,etime,GColor+MCCreate+MDenCreate+etime);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*
 RAB = R * A * B, R and A in seqaij format, B in dense format;
*/
#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense"
PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense(Mat R,Mat A,Mat B,Mat RAB,PetscScalar *work)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*r=(Mat_SeqAIJ*)R->data;
  PetscErrorCode ierr;
  PetscScalar    *b,r1,r2,r3,r4,*b1,*b2,*b3,*b4;
  MatScalar      *aa,*ra;
  PetscInt       cn =B->cmap->n,bm=B->rmap->n,col,i,j,n,*ai=a->i,*aj,am=A->rmap->n;
  PetscInt       am2=2*am,am3=3*am,bm4=4*bm;
  PetscScalar    *d,*c,*c2,*c3,*c4;
  PetscInt       *rj,rm=R->rmap->n,dm=RAB->rmap->n,dn=RAB->cmap->n;
  PetscInt       rm2=2*rm,rm3=3*rm,colrm;
  PetscLogDouble t0,tf,Mult_sp_den1=0.0,Mult_sp_den2=0.0;

  PetscFunctionBegin;
  if (!dm || !dn) PetscFunctionReturn(0);
  if (bm != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in A %D not equal rows in B %D\n",A->cmap->n,bm);
  if (am != R->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in R %D not equal rows in A %D\n",R->cmap->n,am);
  if (R->rmap->n != RAB->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows in RAB %D not equal rows in R %D\n",RAB->rmap->n,R->rmap->n);
  if (B->cmap->n != RAB->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in RAB %D not equal columns in B %D\n",RAB->cmap->n,B->cmap->n);

  { /* 
     This approach is not as good as original ones (will be removed later), but it reveals that
     AB_den=A*B takes almost all execution time in R*A*B for src/ksp/ksp/examples/tutorials/ex56.c
     */
    PetscBool via_matmatmult=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,"-matrart_via_matmatmult",&via_matmatmult,NULL);CHKERRQ(ierr);
    if (via_matmatmult) {
      Mat AB_den;
      ierr = MatMatMultSymbolic_SeqAIJ_SeqDense(A,B,0.0,&AB_den);CHKERRQ(ierr);

      ierr = PetscGetTime(&t0);CHKERRQ(ierr);
      ierr = MatMatMultNumeric_SeqAIJ_SeqDense(A,B,AB_den);CHKERRQ(ierr);
      ierr  = PetscGetTime(&tf);CHKERRQ(ierr);
      Mult_sp_den1 += tf - t0;
      ierr = PetscGetTime(&t0);CHKERRQ(ierr);
      ierr = MatMatMultNumeric_SeqAIJ_SeqDense(R,AB_den,RAB);CHKERRQ(ierr);
      ierr  = PetscGetTime(&tf);CHKERRQ(ierr);
      Mult_sp_den2 += tf - t0;
      printf(" MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense...time = %g + %g = %g; ncolor: %d\n",Mult_sp_den1, Mult_sp_den2, Mult_sp_den1+ Mult_sp_den2,B->cmap->n);
      ierr = MatDestroy(&AB_den);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  ierr = MatDenseGetArray(B,&b);CHKERRQ(ierr);
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

  ierr = MatDenseRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(RAB,&d);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(RAB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RAB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRARtNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ(Mat A,Mat R,Mat C)
{
  PetscErrorCode       ierr;
  Mat_SeqAIJ           *c = (Mat_SeqAIJ*)C->data;
  Mat_RARt             *rart=c->rart;
  MatTransposeColoring matcoloring;
  Mat                  Rt,RARt;
  PetscLogDouble       Mult_sp_den=0.0,app1=0.0,app2=0.0,t0,tf;

  PetscFunctionBegin;
  /* Get dense Rt by Apply MatTransposeColoring to R */
  matcoloring = rart->matcoloring;
  Rt          = rart->Rt;

  ierr  = PetscTime(&t0);CHKERRQ(ierr);
  ierr  = MatTransColoringApplySpToDen(matcoloring,R,Rt);CHKERRQ(ierr);
  ierr  = PetscTime(&tf);CHKERRQ(ierr);
  app1 += tf - t0;

  /* Get dense RARt = R*A*Rt -- dominates! */
  ierr = PetscTime(&t0);CHKERRQ(ierr);
  RARt = rart->RARt;
  ierr = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense(R,A,Rt,RARt,rart->work);CHKERRQ(ierr);
  ierr = PetscTime(&tf);CHKERRQ(ierr);

  Mult_sp_den += tf - t0;

  /* Recover C from C_dense */
  ierr = PetscTime(&t0);CHKERRQ(ierr);
  ierr = MatTransColoringApplyDenToSp(matcoloring,RARt,C);CHKERRQ(ierr);
  ierr = PetscTime(&tf);CHKERRQ(ierr);

  app2 += tf - t0;

#if defined(PETSC_USE_INFO)
  ierr = PetscInfo4(C,"Num = ColorApp %g + %g + Mult_sp_den %g  = %g\n",app1,app2,Mult_sp_den,app1+app2+Mult_sp_den);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRARt_SeqAIJ_SeqAIJ"
PetscErrorCode MatRARt_SeqAIJ_SeqAIJ(Mat A,Mat R,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscBool      usecoloring = PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,"-matrart_color",&usecoloring,NULL);CHKERRQ(ierr);

  if (!usecoloring) {
    Mat Rt;
    ierr = MatTranspose(R,MAT_INITIAL_MATRIX,&Rt);CHKERRQ(ierr); /* replace MAT_INITIAL_MATRIX with scall if !usecoloring is better */
    ierr = MatMatMatMult(R,A,Rt,scall,fill,C);CHKERRQ(ierr);
    ierr = MatDestroy(&Rt);CHKERRQ(ierr); 
    PetscFunctionReturn(0);
  }

  /* use coloring */
  PetscBool     newimpl=PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-matrart_new",&newimpl,NULL);CHKERRQ(ierr);
  if (newimpl) {
    Mat ARt,RARt;
    printf(" MatRARt_new ....\n");
    if (scall == MAT_INITIAL_MATRIX) {
      ierr = PetscLogEventBegin(MAT_RARtSymbolic,A,R,0,0);CHKERRQ(ierr);
      ierr = MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ(A,R,fill,&ARt);CHKERRQ(ierr);
      ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(R,ARt,fill,&RARt);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_RARtSymbolic,A,R,0,0);CHKERRQ(ierr);
    } 
    ierr = PetscLogEventBegin(MAT_RARtNumeric,A,R,0,0);CHKERRQ(ierr);
    ierr = MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(A,R,ARt);CHKERRQ(ierr);
    ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ(R,ARt,RARt);CHKERRQ(ierr);
    *C = RARt;
    ierr = PetscLogEventEnd(MAT_RARtNumeric,A,R,0,0);CHKERRQ(ierr);
    
    ierr = MatDestroy(&ARt);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_RARtSymbolic,A,R,0,0);CHKERRQ(ierr);
    ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ(A,R,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_RARtSymbolic,A,R,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_RARtNumeric,A,R,0,0);CHKERRQ(ierr);
  ierr = MatRARtNumeric_SeqAIJ_SeqAIJ(A,R,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_RARtNumeric,A,R,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
