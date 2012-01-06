
/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = P * A * P^T
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

/*
     MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ - Forms the symbolic product of two SeqAIJ matrices
           C = P * A * P^T;

     Note: C is assumed to be uncreated.
           If this is not the case, Destroy C before calling this routine.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) 
{
  /* Note: This code is virtually identical to that of MatApplyPtAP_SeqAIJ_Symbolic */
  /*        and MatMatMult_SeqAIJ_SeqAIJ_Symbolic.  Perhaps they could be merged nicely. */
  PetscErrorCode     ierr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  PetscInt           *ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pti,*ptj,*ptjj;
  PetscInt           *ci,*cj,*paj,*padenserow,*pasparserow,*denserow,*sparserow;
  PetscInt           an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N,pm=P->rmap->N;
  PetscInt           i,j,k,pnzi,arow,anzj,panzi,ptrow,ptnzj,cnzi;
  MatScalar          *ca;

  PetscFunctionBegin;
  /* some error checking which could be moved into interface layer */
  if (pn!=am) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pn,am);
  if (am!=an) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",am, an);

  /* Set up timers */
  ierr = PetscLogEventBegin(MAT_Applypapt_symbolic,A,P,0,0);CHKERRQ(ierr);

  /* Create ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((pm+1)*1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc4(an,PetscInt,&padenserow,an,PetscInt,&pasparserow,pm,PetscInt,&denserow,pm,PetscInt,&sparserow);CHKERRQ(ierr);
  ierr = PetscMemzero(padenserow,an*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(pasparserow,an*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(denserow,pm*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(sparserow,pm*sizeof(PetscInt));CHKERRQ(ierr);

  /* Set initial free space to be nnz(A) scaled by aspect ratio of Pt. */
  /* This should be reasonable if sparsity of PAPt is similar to that of A. */
  ierr          = PetscFreeSpaceGet((ai[am]/pn)*pm,&free_space);
  current_space = free_space;

  /* Determine fill for each row of C: */
  for (i=0;i<pm;i++) {
    pnzi  = pi[i+1] - pi[i];
    panzi = 0;
    /* Get symbolic sparse row of PA: */
    for (j=0;j<pnzi;j++) {
      arow = *pj++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      for (k=0;k<anzj;k++) {
        if (!padenserow[ajj[k]]) {
          padenserow[ajj[k]]   = -1;
          pasparserow[panzi++] = ajj[k];
        }
      }
    }
    /* Using symbolic row of PA, determine symbolic row of C: */
    paj    = pasparserow;
    cnzi   = 0;
    for (j=0;j<panzi;j++) {
      ptrow = *paj++;
      ptnzj = pti[ptrow+1] - pti[ptrow];
      ptjj  = ptj + pti[ptrow];
      for (k=0;k<ptnzj;k++) {
        if (!denserow[ptjj[k]]) {
          denserow[ptjj[k]] = -1;
          sparserow[cnzi++] = ptjj[k];
        }
      }
    }

    /* sort sparse representation */
    ierr = PetscSortInt(cnzi,sparserow);CHKERRQ(ierr);

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = PetscFreeSpaceGet(cnzi+current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out dense row */
    ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(PetscInt));CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    for (j=0;j<panzi;j++) {
      padenserow[pasparserow[j]] = 0;
    }
    for (j=0;j<cnzi;j++) {
      denserow[sparserow[j]] = 0;
    }
    ci[i+1] = ci[i] + cnzi;
  }
  /* column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pm]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree4(padenserow,pasparserow,denserow,sparserow);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pm]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pm]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,pm,pm,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(MAT_Applypapt_symbolic,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatApplyPAPt_Numeric_SeqAIJ - Forms the numeric product of two SeqAIJ matrices
           C = P * A * P^T;
     Note: C must have been created by calling MatApplyPAPt_Symbolic_SeqAIJ.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode ierr;
  PetscInt       flops=0;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ     *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ     *c  = (Mat_SeqAIJ *) C->data;
  PetscInt       *ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj=p->j,*paj,*pajdense,*ptj;
  PetscInt       *ci=c->i,*cj=c->j;
  PetscInt       an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N,pm=P->rmap->N,cn=C->cmap->N,cm=C->rmap->N;
  PetscInt       i,j,k,k1,k2,pnzi,anzj,panzj,arow,ptcol,ptnzj,cnzi;
  MatScalar      *aa=a->a,*pa=p->a,*pta=p->a,*ptaj,*paa,*aaj,*ca=c->a,sum;

  PetscFunctionBegin;
  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (pm!=cm) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pm,cm);
  if (pn!=am) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pn,am);
  if (am!=an) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",am, an);
  if (pm!=cn) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pm, cn);

  /* Set up timers */
  ierr = PetscLogEventBegin(MAT_Applypapt_numeric,A,P,C,0);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  ierr = PetscMalloc3(an,MatScalar,&paa,an,PetscInt,&paj,an,PetscInt,&pajdense);CHKERRQ(ierr);
  ierr = PetscMemzero(paa,an*(sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);
  
  for (i=0;i<pm;i++) {
    /* Form sparse row of P*A */
    pnzi  = pi[i+1] - pi[i];
    panzj = 0;
    for (j=0;j<pnzi;j++) {
      arow = *pj++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      aaj  = aa + ai[arow];
      for (k=0;k<anzj;k++) {
        if (!pajdense[ajj[k]]) {
          pajdense[ajj[k]] = -1;
          paj[panzj++]     = ajj[k];
        }
        paa[ajj[k]] += (*pa)*aaj[k];
      }
      flops += 2*anzj;
      pa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    ierr = PetscSortInt(panzj,paj);CHKERRQ(ierr);

    /* Compute P*A*P^T using sparse inner products. */
    /* Take advantage of pre-computed (i,j) of C for locations of non-zeros. */
    cnzi = ci[i+1] - ci[i];
    for (j=0;j<cnzi;j++) {
      /* Form sparse inner product of current row of P*A with (*cj++) col of P^T. */
      ptcol = *cj++;
      ptnzj = pi[ptcol+1] - pi[ptcol];
      ptj   = pjj + pi[ptcol];
      ptaj  = pta + pi[ptcol];
      sum   = 0.;
      k1    = 0;
      k2    = 0;
      while ((k1<panzj) && (k2<ptnzj)) {
        if (paj[k1]==ptj[k2]) {
          sum += paa[paj[k1++]]*ptaj[k2++];
        } else if (paj[k1] < ptj[k2]) {
          k1++;
        } else /* if (paj[k1] > ptj[k2]) */ {
          k2++;
        }
      }
      *ca++ = sum;
    }

    /* Zero the current row info for P*A */
    for (j=0;j<panzj;j++) {
      paa[paj[j]]      = 0.;
      pajdense[paj[j]] = 0;
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree3(paa,paj,pajdense);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Applypapt_numeric,A,P,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_SeqAIJ_SeqAIJ"
PetscErrorCode MatApplyPAPt_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_Applypapt,A,P,0,0);CHKERRQ(ierr);
  ierr = MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ(A,P,C);CHKERRQ(ierr);
  ierr = MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Applypapt,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------*/
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = R * A * R^T
*/

#undef __FUNCT__
#define __FUNCT__ "PetscContainerDestroy_Mat_RARt"
PetscErrorCode PetscContainerDestroy_Mat_RARt(void *ptr)
{
  PetscErrorCode ierr;
  Mat_RARt       *rart=(Mat_RARt*)ptr;

  PetscFunctionBegin;
  ierr = MatTransposeColoringDestroy(&rart->matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->Rt);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->RARt);CHKERRQ(ierr);
  ierr = PetscFree(rart->work);CHKERRQ(ierr);
  ierr = PetscFree(rart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJ_RARt"
PetscErrorCode MatDestroy_SeqAIJ_RARt(Mat A)
{
  PetscErrorCode ierr;
  PetscContainer container;
  Mat_RARt       *rart=PETSC_NULL;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_RARt",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Container does not exit");
  ierr = PetscContainerGetPointer(container,(void **)&rart);CHKERRQ(ierr);
  A->ops->destroy   = rart->destroy;
  if (A->ops->destroy) {
    ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)A,"Mat_RARt",0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRARtSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat R,PetscReal fill,Mat *C) 
{
  PetscErrorCode      ierr;
  Mat                 P;
  PetscInt            *rti,*rtj;
  Mat_RARt            *rart;
  PetscContainer      container;
  MatTransposeColoring matcoloring;
  ISColoring           iscoloring;
  Mat                  Rt_dense,RARt_dense;
  PetscLogDouble       GColor=0.0,MCCreate=0.0,MDenCreate=0.0,t0,tf,etime=0.0;
  Mat_SeqAIJ           *c;

  PetscFunctionBegin;
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  /* create symbolic P=Rt */
  ierr = MatGetSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,R->cmap->n,R->rmap->n,rti,rtj,PETSC_NULL,&P);CHKERRQ(ierr);

  /* get symbolic C=Pt*A*P */
  ierr = MatPtAPSymbolic_SeqAIJ_SeqAIJ(A,P,fill,C);CHKERRQ(ierr);

  /* create a supporting struct */
  ierr = PetscNew(Mat_RARt,&rart);CHKERRQ(ierr);

  /* attach the supporting struct to C */
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,rart);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_Mat_RARt);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*C),"Mat_RARt",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  etime += tf - t0;

  /* Create MatTransposeColoring from symbolic C=R*A*R^T */
  c=(Mat_SeqAIJ*)(*C)->data;
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ierr = MatGetColoring(*C,MATCOLORINGLF,&iscoloring);CHKERRQ(ierr); 
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  GColor += tf - t0;

  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ierr = MatTransposeColoringCreate(*C,iscoloring,&matcoloring);CHKERRQ(ierr);
  rart->matcoloring = matcoloring;
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  MCCreate += tf - t0;

  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  /* Create Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&Rt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Rt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(Rt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Rt_dense,PETSC_NULL);CHKERRQ(ierr);
  Rt_dense->assembled = PETSC_TRUE;
  rart->Rt            = Rt_dense;

  /* Create RARt_dense = R*A*Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&RARt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(RARt_dense,(*C)->rmap->n,matcoloring->ncolors,(*C)->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(RARt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(RARt_dense,PETSC_NULL);CHKERRQ(ierr);
  rart->RARt = RARt_dense;

  /* Allocate work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  ierr = PetscMalloc(A->rmap->n*4*sizeof(PetscScalar),&rart->work);CHKERRQ(ierr);

  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  MDenCreate += tf - t0;

  rart->destroy = (*C)->ops->destroy;
  (*C)->ops->destroy = MatDestroy_SeqAIJ_RARt;

  /* clean up */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  {
  PetscReal density= (PetscReal)(c->nz)/(RARt_dense->rmap->n*RARt_dense->cmap->n);
  ierr = PetscInfo6(*C,"RARt_den %D %D; Rt_den %D %D, (RARt->nz %D)/(m*ncolors)=%g\n",RARt_dense->rmap->n,RARt_dense->cmap->n,Rt_dense->rmap->n,Rt_dense->cmap->n,c->nz,density);
  ierr = PetscInfo5(*C,"Sym = GetColor %g + MColorCreate %g + MDenCreate %g + other %g = %g\n",GColor,MCCreate,MDenCreate,etime,GColor+MCCreate+MDenCreate+etime);
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
  PetscInt       cn=B->cmap->n,bm=B->rmap->n,col,i,j,n,*ai=a->i,*aj,am=A->rmap->n;
  PetscInt       am2=2*am,am3=3*am,bm4=4*bm;
  PetscScalar    *d,*c,*c2,*c3,*c4;
  PetscInt       *rj,rm=R->rmap->n,dm=RAB->rmap->n,dn=RAB->cmap->n;
  PetscInt       rm2=2*rm,rm3=3*rm,colrm;
 
  PetscFunctionBegin;
  if (!dm || !dn) PetscFunctionReturn(0);
  if (bm != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in A %D not equal rows in B %D\n",A->cmap->n,bm);
  if (am != R->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in R %D not equal rows in A %D\n",R->cmap->n,am);
  if (R->rmap->n != RAB->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows in RAB %D not equal rows in R %D\n",RAB->rmap->n,R->rmap->n);
  if (B->cmap->n != RAB->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in RAB %D not equal columns in B %D\n",RAB->cmap->n,B->cmap->n);

  ierr = MatGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatGetArray(RAB,&d);CHKERRQ(ierr);
  b1 = b; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;
  c = work; c2 = c + am; c3 = c2 + am; c4 = c3 + am;
  for (col=0; col<cn-4; col += 4){  /* over columns of C */
    for (i=0; i<am; i++) {        /* over rows of A in those columns */
      r1 = r2 = r3 = r4 = 0.0;
      n   = ai[i+1] - ai[i]; 
      aj  = a->j + ai[i];
      aa  = a->a + ai[i];
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
      n   = r->i[i+1] - r->i[i]; 
      rj  = r->j + r->i[i];
      ra  = r->a + r->i[i];
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
  for (;col<cn; col++){     /* over extra columns of C */
    for (i=0; i<am; i++) {  /* over rows of A in those columns */
      r1 = 0.0;
      n   = a->i[i+1] - a->i[i]; 
      aj  = a->j + a->i[i];
      aa  = a->a + a->i[i];
      for (j=0; j<n; j++) {
        r1 += (*aa++)*b1[*aj++]; 
      }
      c[i]     = r1;
    }
    b1 += bm;

    for (i=0; i<rm; i++) {  /* over rows of R in those columns */
      r1 = 0.0;
      n   = r->i[i+1] - r->i[i]; 
      rj  = r->j + r->i[i];
      ra  = r->a + r->i[i];
      for (j=0; j<n; j++) {
        r1 += (*ra++)*c[*rj++]; 
      }
      d[col*rm + i]     = r1;
    }
  }
  ierr = PetscLogFlops(cn*2.0*(a->nz + r->nz));CHKERRQ(ierr);

  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatRestoreArray(RAB,&d);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(RAB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RAB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRARtNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ(Mat A,Mat R,Mat C) 
{
  PetscErrorCode        ierr; 
  Mat_RARt              *rart;
  PetscContainer        container;
  MatTransposeColoring  matcoloring;
  Mat                   Rt,RARt;
  PetscLogDouble        Mult_sp_den=0.0,app1=0.0,app2=0.0,t0,tf;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"Mat_RARt",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Container does not exit");
  ierr  = PetscContainerGetPointer(container,(void **)&rart);CHKERRQ(ierr);

  /* Get dense Rt by Apply MatTransposeColoring to R */
  matcoloring = rart->matcoloring;
  Rt          = rart->Rt;
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ierr = MatTransColoringApplySpToDen(matcoloring,R,Rt);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  app1 += tf - t0;
  
  /* Get dense RARt = R*A*Rt */
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  RARt = rart->RARt;
  ierr = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense(R,A,Rt,RARt,rart->work);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  Mult_sp_den += tf - t0;

  /* Recover C from C_dense */
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ierr = MatTransColoringApplyDenToSp(matcoloring,RARt,C);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  app2 += tf - t0;

#if defined(PETSC_USE_INFO)
  ierr = PetscInfo4(C,"Num = ColorApp %g + %g + Mult_sp_den %g  = %g\n",app1,app2,Mult_sp_den,app1+app2+Mult_sp_den);
#endif
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatRARt_SeqAIJ_SeqAIJ"
PetscErrorCode MatRARt_SeqAIJ_SeqAIJ(Mat A,Mat R,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatRARtSymbolic_SeqAIJ_SeqAIJ(A,R,fill,C);CHKERRQ(ierr);
  }
  ierr = MatRARtNumeric_SeqAIJ_SeqAIJ(A,R,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
