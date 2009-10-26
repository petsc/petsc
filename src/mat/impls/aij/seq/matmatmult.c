#define PETSCMAT_DLL

/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = A * B
*/

#include "../src/mat/impls/aij/seq/aij.h" /*I "petscmat.h" I*/
#include "../src/mat/utils/freespace.h"
#include "petscbt.h"
#include "../src/mat/impls/dense/seq/dense.h" /*I "petscmat.h" I*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMult_SeqAIJ_SeqAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  PetscInt           *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci,*cj;
  PetscInt           am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscInt           i,j,anzi,brow,bnzj,cnzi,nlnk,*lnk,nspacedouble=0;
  MatScalar          *ca;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  /* Set up */
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((am+1)+1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;
  
  /* create and initialize a linked list */
  nlnk = bn+1;
  ierr = PetscLLCreate(bn,bn,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+bi[bm])),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine symbolic info for each row of the product: */
  for (i=0;i<am;i++) {
    anzi = ai[i+1] - ai[i];
    cnzi = 0;
    j    = anzi;
    aj   = a->j + ai[i];
    while (j){/* assume cols are almost in increasing order, starting from its end saves computation */
      j--;
      brow = *(aj + j);
      bnzj = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      /* add non-zero cols of B into the sorted linked list lnk */
      ierr = PetscLLAdd(bnzj,bjj,bn,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      cnzi += nlnk;
    }

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = PetscFreeSpaceGet(cnzi+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLClean(bn,bn,cnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[am]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[am]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[am]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,am,bn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->free_a   = PETSC_TRUE;
  c->free_ij  = PETSC_TRUE;
  c->nonew    = 0;

#if defined(PETSC_USE_INFO)
  if (ci[am] != 0) {
    PetscReal afill = ((PetscReal)ci[am])/(ai[am]+bi[bm]);
    if (afill < 1.0) afill = 1.0;
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %G needed %G.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%G,&C) for best performance.;\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  PetscLogDouble flops=0.0;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ     *b = (Mat_SeqAIJ *)B->data;
  Mat_SeqAIJ     *c = (Mat_SeqAIJ *)C->data;
  PetscInt       *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  PetscInt       am=A->rmap->N,cm=C->rmap->N;
  PetscInt       i,j,k,anzi,bnzi,cnzi,brow,nextb;
  MatScalar      *aa=a->a,*ba=b->a,*baj,*ca=c->a; 

  PetscFunctionBegin;  
  /* clean old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);
  /* Traverse A row-wise. */
  /* Build the ith row in C by summing over nonzero columns in A, */
  /* the rows of B corresponding to nonzeros of A. */
  for (i=0;i<am;i++) {
    anzi = ai[i+1] - ai[i];
    for (j=0;j<anzi;j++) {
      brow = *aj++;
      bnzi = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      baj  = ba + bi[brow];
      nextb = 0;
      for (k=0; nextb<bnzi; k++) {
        if (cj[k] == bjj[nextb]){ /* ccol == bcol */
          ca[k] += (*aa)*baj[nextb++];
        }
      }
      flops += 2*bnzi;
      aa++;
    }
    cnzi = ci[i+1] - ci[i];
    ca += cnzi;
    cj += cnzi;
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);     

  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMatMultTranspose_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultTranspose_SeqAIJ_SeqAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultTransposeSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMultTransposeNumeric_SeqAIJ_SeqAIJ(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultTransposeSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultTransposeSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            At;
  PetscInt       *ati,*atj;

  PetscFunctionBegin;
  /* create symbolic At */
  ierr = MatGetSymbolicTranspose_SeqAIJ(A,&ati,&atj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,A->cmap->n,A->rmap->n,ati,atj,PETSC_NULL,&At);CHKERRQ(ierr);

  /* get symbolic C=At*B */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(At,B,fill,C);CHKERRQ(ierr);

  /* clean up */
  ierr = MatDestroy(At);CHKERRQ(ierr);
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(A,&ati,&atj);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultTransposeNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultTransposeNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr; 
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c=(Mat_SeqAIJ*)C->data;
  PetscInt       am=A->rmap->n,anzi,*ai=a->i,*aj=a->j,*bi=b->i,*bj,bnzi,nextb;
  PetscInt       cm=C->rmap->n,*ci=c->i,*cj=c->j,crow,*cjj,i,j,k;
  PetscLogDouble flops=0.0;
  MatScalar      *aa=a->a,*ba,*ca=c->a,*caj;
 
  PetscFunctionBegin;
  /* clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  /* compute A^T*B using outer product (A^T)[:,i]*B[i,:] */
  for (i=0;i<am;i++) {
    bj   = b->j + bi[i];
    ba   = b->a + bi[i];
    bnzi = bi[i+1] - bi[i];
    anzi = ai[i+1] - ai[i];
    for (j=0; j<anzi; j++) { 
      nextb = 0;
      crow  = *aj++;
      cjj   = cj + ci[crow];
      caj   = ca + ci[crow];
      /* perform sparse axpy operation.  Note cjj includes bj. */
      for (k=0; nextb<bnzi; k++) {
        if (cjj[k] == *(bj+nextb)) { /* ccol == bcol */
          caj[k] += (*aa)*(*(ba+nextb));
          nextb++;
        }
      }
      flops += 2*bnzi;
      aa++;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqDense"
PetscErrorCode MatMatMult_SeqAIJ_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_SeqAIJ_SeqDense(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMultNumeric_SeqAIJ_SeqDense(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_SeqAIJ_SeqDense"
PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultSymbolic_SeqDense_SeqDense(A,B,0.0,C);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_SeqAIJ_SeqDense"
PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *b,*c,r1,r2,r3,r4,*b1,*b2,*b3,*b4;
  MatScalar      *aa;
  PetscInt       cm=C->rmap->n, cn=B->cmap->n, bm=B->rmap->n, col, i,j,n,*aj, am = A->rmap->n;
  PetscInt       am2 = 2*am, am3 = 3*am,  bm4 = 4*bm,colam;

  PetscFunctionBegin;
  if (!cm || !cn) PetscFunctionReturn(0);
  if (bm != A->cmap->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Number columns in A %D not equal rows in B %D\n",A->cmap->n,bm);
  if (A->rmap->n != C->rmap->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Number rows in C %D not equal rows in A %D\n",C->rmap->n,A->rmap->n);
  if (B->cmap->n != C->cmap->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Number columns in B %D not equal columns in C %D\n",B->cmap->n,C->cmap->n);
  ierr = MatGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatGetArray(C,&c);CHKERRQ(ierr);
  b1 = b; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;
  for (col=0; col<cn-4; col += 4){  /* over columns of C */
    colam = col*am;
    for (i=0; i<am; i++) {        /* over rows of C in those columns */
      r1 = r2 = r3 = r4 = 0.0;
      n   = a->i[i+1] - a->i[i]; 
      aj  = a->j + a->i[i];
      aa  = a->a + a->i[i];
      for (j=0; j<n; j++) {
        r1 += (*aa)*b1[*aj]; 
        r2 += (*aa)*b2[*aj]; 
        r3 += (*aa)*b3[*aj]; 
        r4 += (*aa++)*b4[*aj++]; 
      }
      c[colam + i]       = r1;
      c[colam + am + i]  = r2;
      c[colam + am2 + i] = r3;
      c[colam + am3 + i] = r4;
    }
    b1 += bm4;
    b2 += bm4;
    b3 += bm4;
    b4 += bm4;
  }
  for (;col<cn; col++){     /* over extra columns of C */
    for (i=0; i<am; i++) {  /* over rows of C in those columns */
      r1 = 0.0;
      n   = a->i[i+1] - a->i[i]; 
      aj  = a->j + a->i[i];
      aa  = a->a + a->i[i];

      for (j=0; j<n; j++) {
        r1 += (*aa++)*b1[*aj++]; 
      }
      c[col*am + i]     = r1;
    }
    b1 += bm;
  }
  ierr = PetscLogFlops(cn*(2.0*a->nz));CHKERRQ(ierr);
  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatRestoreArray(C,&c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Note very similar to MatMult_SeqAIJ(), should generate both codes from same base
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumericAdd_SeqAIJ_SeqDense"
PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *b,*c,r1,r2,r3,r4,*b1,*b2,*b3,*b4;
  MatScalar      *aa;
  PetscInt       cm=C->rmap->n, cn=B->cmap->n, bm=B->rmap->n, col, i,j,n,*aj, am = A->rmap->n,*ii,arm;
  PetscInt       am2 = 2*am, am3 = 3*am,  bm4 = 4*bm,colam,*ridx;

  PetscFunctionBegin;
  if (!cm || !cn) PetscFunctionReturn(0);
  ierr = MatGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatGetArray(C,&c);CHKERRQ(ierr);
  b1 = b; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;

  if (a->compressedrow.use){ /* use compressed row format */
    for (col=0; col<cn-4; col += 4){  /* over columns of C */
      colam = col*am;
      arm   = a->compressedrow.nrows;
      ii    = a->compressedrow.i;
      ridx  = a->compressedrow.rindex;
      for (i=0; i<arm; i++) {        /* over rows of C in those columns */
	r1 = r2 = r3 = r4 = 0.0;
	n   = ii[i+1] - ii[i]; 
	aj  = a->j + ii[i];
	aa  = a->a + ii[i];
	for (j=0; j<n; j++) {
	  r1 += (*aa)*b1[*aj]; 
	  r2 += (*aa)*b2[*aj]; 
	  r3 += (*aa)*b3[*aj]; 
	  r4 += (*aa++)*b4[*aj++]; 
	}
	c[colam       + ridx[i]] += r1;
	c[colam + am  + ridx[i]] += r2;
	c[colam + am2 + ridx[i]] += r3;
	c[colam + am3 + ridx[i]] += r4;
      }
      b1 += bm4;
      b2 += bm4;
      b3 += bm4;
      b4 += bm4;
    }
    for (;col<cn; col++){     /* over extra columns of C */
      colam = col*am;
      arm   = a->compressedrow.nrows;
      ii    = a->compressedrow.i;
      ridx  = a->compressedrow.rindex;
      for (i=0; i<arm; i++) {  /* over rows of C in those columns */
	r1 = 0.0;
	n   = ii[i+1] - ii[i]; 
	aj  = a->j + ii[i];
	aa  = a->a + ii[i];

	for (j=0; j<n; j++) {
	  r1 += (*aa++)*b1[*aj++]; 
	}
	c[col*am + ridx[i]] += r1;
      }
      b1 += bm;
    }
  } else {
    for (col=0; col<cn-4; col += 4){  /* over columns of C */
      colam = col*am;
      for (i=0; i<am; i++) {        /* over rows of C in those columns */
	r1 = r2 = r3 = r4 = 0.0;
	n   = a->i[i+1] - a->i[i]; 
	aj  = a->j + a->i[i];
	aa  = a->a + a->i[i];
	for (j=0; j<n; j++) {
	  r1 += (*aa)*b1[*aj]; 
	  r2 += (*aa)*b2[*aj]; 
	  r3 += (*aa)*b3[*aj]; 
	  r4 += (*aa++)*b4[*aj++]; 
	}
	c[colam + i]       += r1;
	c[colam + am + i]  += r2;
	c[colam + am2 + i] += r3;
	c[colam + am3 + i] += r4;
      }
      b1 += bm4;
      b2 += bm4;
      b3 += bm4;
      b4 += bm4;
    }
    for (;col<cn; col++){     /* over extra columns of C */
      for (i=0; i<am; i++) {  /* over rows of C in those columns */
	r1 = 0.0;
	n   = a->i[i+1] - a->i[i]; 
	aj  = a->j + a->i[i];
	aa  = a->a + a->i[i];

	for (j=0; j<n; j++) {
	  r1 += (*aa++)*b1[*aj++]; 
	}
	c[col*am + i]     += r1;
      }
      b1 += bm;
    }
  }
  ierr = PetscLogFlops(cn*2.0*a->nz);CHKERRQ(ierr);
  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatRestoreArray(C,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
