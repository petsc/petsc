
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
  ierr = MatDestroy(&rart->ARt);CHKERRQ(ierr);
  ierr = MatDestroy(&rart->RARt);CHKERRQ(ierr);
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
  Mat                  Rt_dense,ARt_dense,RARt_dense;
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
  
  /* Create ARt_dense = A*Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&ARt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(ARt_dense,A->rmap->n,matcoloring->ncolors,A->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(ARt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(ARt_dense,PETSC_NULL);CHKERRQ(ierr);
  ARt_dense->assembled = PETSC_TRUE;
  rart->ARt            = ARt_dense;

  /* Create RARt_dense = R*A*Rt_dense */
  ierr = MatCreate(PETSC_COMM_SELF,&RARt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(RARt_dense,(*C)->rmap->n,matcoloring->ncolors,(*C)->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
  ierr = MatSetType(RARt_dense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(RARt_dense,PETSC_NULL);CHKERRQ(ierr);
   
  rart->RARt = RARt_dense;
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  MDenCreate += tf - t0;
  printf("RARt_den %d %d; ARt_den %d %d; Rt_den %d %d\n",RARt_dense->rmap->n,RARt_dense->cmap->n,ARt_dense->rmap->n,ARt_dense->cmap->n,Rt_dense->rmap->n,Rt_dense->cmap->n);
  printf("Sym = GetColor %g + MColorCreate %g + MDenCreate %g + other %g = %g\n",GColor,MCCreate,MDenCreate,etime,GColor+MCCreate+MDenCreate+etime);

  rart->destroy = (*C)->ops->destroy;
  (*C)->ops->destroy = MatDestroy_SeqAIJ_RARt;

  /* clean up */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(R,&rti,&rtj);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
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
  Mat                   Rt,ARt,RARt;
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
  
  /* Get dense ARt = A*Rt */
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ARt = rart->ARt;
  ierr = MatMatMultNumeric_SeqAIJ_SeqDense(A,Rt,ARt);CHKERRQ(ierr);

  /* Get dense RARt = R*ARt */
  RARt = rart->RARt;
  ierr = MatMatMultNumeric_SeqAIJ_SeqDense(R,ARt,RARt);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  Mult_sp_den += tf - t0;

  /* Recover C from C_dense */
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ierr = MatTransColoringApplyDenToSp(matcoloring,RARt,C);CHKERRQ(ierr);
  ierr = PetscGetTime(&tf);CHKERRQ(ierr);
  app2 += tf - t0;
  printf("Num = ColorApp %g + %g + Mult_sp_den %g  = %g\n",app1,app2,Mult_sp_den,app1+app2+Mult_sp_den);
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
