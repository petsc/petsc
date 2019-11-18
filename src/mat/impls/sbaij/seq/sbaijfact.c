
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petscis.h>

PetscErrorCode MatGetInertia_SeqSBAIJ(Mat F,PetscInt *nneg,PetscInt *nzero,PetscInt *npos)
{
  Mat_SeqSBAIJ *fact=(Mat_SeqSBAIJ*)F->data;
  MatScalar    *dd=fact->a;
  PetscInt     mbs=fact->mbs,bs=F->rmap->bs,i,nneg_tmp,npos_tmp,*fi=fact->diag;

  PetscFunctionBegin;
  if (bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for bs: %D >1 yet",bs);
  if (F->factorerrortype==MAT_FACTOR_NUMERIC_ZEROPIVOT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatFactor fails with numeric zeropivot");

  nneg_tmp = 0; npos_tmp = 0;
  for (i=0; i<mbs; i++) {
    if (PetscRealPart(dd[*fi]) > 0.0) npos_tmp++;
    else if (PetscRealPart(dd[*fi]) < 0.0) nneg_tmp++;
    fi++;
  }
  if (nneg)  *nneg  = nneg_tmp;
  if (npos)  *npos  = npos_tmp;
  if (nzero) *nzero = mbs - nneg_tmp - npos_tmp;
  PetscFunctionReturn(0);
}

/*
  Symbolic U^T*D*U factorization for SBAIJ format. Modified from SSF of YSMP.
  Use Modified Sparse Row (MSR) storage for u and ju. See page 85, "Iterative Methods ..." by Saad.
*/
PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ_MSR(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b;
  PetscErrorCode ierr;
  const PetscInt *rip,*ai,*aj;
  PetscInt       i,mbs = a->mbs,*jutmp,bs = A->rmap->bs,bs2=a->bs2;
  PetscInt       m,reallocs = 0,prow;
  PetscInt       *jl,*q,jmin,jmax,juidx,nzk,qm,*iu,*ju,k,j,vj,umax,maxadd;
  PetscReal      f = info->fill;
  PetscBool      perm_identity;

  PetscFunctionBegin;
  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  if (perm_identity) { /* without permutation */
    a->permute = PETSC_FALSE;

    ai = a->i; aj = a->j;
  } else {            /* non-trivial permutation */
    a->permute = PETSC_TRUE;

    ierr = MatReorderingSeqSBAIJ(A,perm);CHKERRQ(ierr);

    ai = a->inew; aj = a->jnew;
  }

  /* initialization */
  ierr  = PetscMalloc1(mbs+1,&iu);CHKERRQ(ierr);
  umax  = (PetscInt)(f*ai[mbs] + 1); umax += mbs + 1;
  ierr  = PetscMalloc1(umax,&ju);CHKERRQ(ierr);
  iu[0] = mbs+1;
  juidx = mbs + 1; /* index for ju */
  /* jl linked list for pivot row -- linked list for col index */
  ierr = PetscMalloc2(mbs,&jl,mbs,&q);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    jl[i] = mbs;
    q[i]  = 0;
  }

  /* for each row k */
  for (k=0; k<mbs; k++) {
    for (i=0; i<mbs; i++) q[i] = 0;  /* to be removed! */
    nzk  = 0; /* num. of nz blocks in k-th block row with diagonal block excluded */
    q[k] = mbs;
    /* initialize nonzero structure of k-th row to row rip[k] of A */
    jmin = ai[rip[k]] +1; /* exclude diag[k] */
    jmax = ai[rip[k]+1];
    for (j=jmin; j<jmax; j++) {
      vj = rip[aj[j]]; /* col. value */
      if (vj > k) {
        qm = k;
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        if (qm == vj) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Duplicate entry in A\n");
        nzk++;
        q[m]  = vj;
        q[vj] = qm;
      } /* if (vj > k) */
    } /* for (j=jmin; j<jmax; j++) */

    /* modify nonzero structure of k-th row by computing fill-in
       for each row i to be merged in */
    prow = k;
    prow = jl[prow]; /* next pivot row (== mbs for symbolic factorization) */

    while (prow < k) {
      /* merge row prow into k-th row */
      jmin = iu[prow] + 1; jmax = iu[prow+1];
      qm   = k;
      for (j=jmin; j<jmax; j++) {
        vj = ju[j];
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        if (qm != vj) {
          nzk++; q[m] = vj; q[vj] = qm; qm = vj;
        }
      }
      prow = jl[prow]; /* next pivot row */
    }

    /* add k to row list for first nonzero element in k-th row */
    if (nzk > 0) {
      i     = q[k]; /* col value of first nonzero element in U(k, k+1:mbs-1) */
      jl[k] = jl[i]; jl[i] = k;
    }
    iu[k+1] = iu[k] + nzk;

    /* allocate more space to ju if needed */
    if (iu[k+1] > umax) {
      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      maxadd = umax;
      if (maxadd < nzk) maxadd = (mbs-k)*(nzk+1)/2;
      umax += maxadd;

      /* allocate a longer ju */
      ierr = PetscMalloc1(umax,&jutmp);CHKERRQ(ierr);
      ierr = PetscArraycpy(jutmp,ju,iu[k]);CHKERRQ(ierr);
      ierr = PetscFree(ju);CHKERRQ(ierr);
      ju   = jutmp;
      reallocs++; /* count how many times we realloc */
    }

    /* save nonzero structure of k-th row in ju */
    i=k;
    while (nzk--) {
      i           = q[i];
      ju[juidx++] = i;
    }
  }

#if defined(PETSC_USE_INFO)
  if (ai[mbs] != 0) {
    PetscReal af = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %g needed %g\n",reallocs,(double)f,(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %g or use \n",(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%g);\n",(double)af);CHKERRQ(ierr);
    ierr = PetscInfo(A,"for best performance.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
  }
#endif

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree2(jl,q);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqSBAIJSetPreallocation(F,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

  /* ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)iperm);CHKERRQ(ierr); */
  b                = (Mat_SeqSBAIJ*)(F)->data;
  b->singlemalloc  = PETSC_FALSE;
  b->free_a        = PETSC_TRUE;
  b->free_ij       = PETSC_TRUE;

  ierr    = PetscMalloc1((iu[mbs]+1)*bs2,&b->a);CHKERRQ(ierr);
  b->j    = ju;
  b->i    = iu;
  b->diag = 0;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;

  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);

  b->icol = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr    = PetscMalloc1(bs*mbs+bs,&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.
     Allocate idnew, solve_work, new a, new j */
  ierr     = PetscLogObjectMemory((PetscObject)F,(iu[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = iu[mbs];

  (F)->info.factor_mallocs   = reallocs;
  (F)->info.fill_ratio_given = f;
  if (ai[mbs] != 0) {
    (F)->info.fill_ratio_needed = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
  } else {
    (F)->info.fill_ratio_needed = 0.0;
  }
  ierr = MatSeqSBAIJSetNumericFactorization_inplace(F,perm_identity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
    Symbolic U^T*D*U factorization for SBAIJ format.
    See MatICCFactorSymbolic_SeqAIJ() for description of its data structure.
*/
#include <petscbt.h>
#include <../src/mat/utils/freespace.h>
PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscErrorCode     ierr;
  PetscBool          perm_identity,missing;
  PetscReal          fill = info->fill;
  const PetscInt     *rip,*ai=a->i,*aj=a->j;
  PetscInt           i,mbs=a->mbs,bs=A->rmap->bs,reallocs=0,prow;
  PetscInt           *jl,jmin,jmax,nzk,*ui,k,j,*il,nextprow;
  PetscInt           nlnk,*lnk,ncols,*cols,*uj,**ui_ptr,*uj_ptr,*udiag;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);
  if (bs > 1) {
    ierr = MatCholeskyFactorSymbolic_SeqSBAIJ_inplace(fact,A,perm,info);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported for sbaij matrix. Use aij format");
  a->permute = PETSC_FALSE;
  ierr       = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  /* initialization */
  ierr  = PetscMalloc1(mbs+1,&ui);CHKERRQ(ierr);
  ierr  = PetscMalloc1(mbs+1,&udiag);CHKERRQ(ierr);
  ui[0] = 0;

  /* jl: linked list for storing indices of the pivot rows
     il: il[i] points to the 1st nonzero entry of U(i,k:mbs-1) */
  ierr = PetscMalloc4(mbs,&ui_ptr,mbs,&il,mbs,&jl,mbs,&cols);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = mbs + 1;
  ierr = PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill*(ai[mbs]+1) */
  ierr          = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,ai[mbs]+1),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (k=0; k<mbs; k++) {  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk   = 0;
    ncols = ai[k+1] - ai[k];
    if (!ncols) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Empty row %D in matrix ",k);
    for (j=0; j<ncols; j++) {
      i       = *(aj + ai[k] + j);
      cols[j] = i;
    }
    ierr = PetscLLAdd(ncols,cols,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */

    while (prow < k) {
      nextprow = jl[prow];
      /* merge prow into k-th row */
      jmin   = il[prow] + 1; /* index of the 2nd nzero entry in U(prow,k:mbs-1) */
      jmax   = ui[prow+1];
      ncols  = jmax-jmin;
      uj_ptr = ui_ptr[prow] + jmin - ui[prow]; /* points to the 2nd nzero entry in U(prow,k:mbs-1) */
      ierr   = PetscLLAddSorted(ncols,uj_ptr,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      nzk   += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax) {
        il[prow] = jmin;
        j        = *uj_ptr; jl[prow] = jl[j]; jl[j] = prow;
      }
      prow = nextprow;
    }

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i    = mbs - k + 1; /* num of unfactored rows */
      i    = PetscIntMultTruncate(i,PetscMin(nzk, i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(mbs,mbs,nzk,lnk,current_space->array,lnkbt);CHKERRQ(ierr);

    /* add the k-th row into il and jl */
    if (nzk > 1) {
      i     = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:mbs-1) */
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    }
    ui_ptr[k] = current_space->array;

    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree4(ui_ptr,il,jl,cols);CHKERRQ(ierr);

  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ui[mbs]+1,&uj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous_Cholesky(&free_space,uj,mbs,ui,udiag);CHKERRQ(ierr); /* store matrix factor */
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */
  ierr = MatSeqSBAIJSetPreallocation(fact,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

  b               = (Mat_SeqSBAIJ*)fact->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;

  ierr = PetscMalloc1(ui[mbs]+1,&b->a);CHKERRQ(ierr);

  b->j         = uj;
  b->i         = ui;
  b->diag      = udiag;
  b->free_diag = PETSC_TRUE;
  b->ilen      = 0;
  b->imax      = 0;
  b->row       = perm;
  b->icol      = perm;

  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);

  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  ierr = PetscMalloc1(mbs+1,&b->solve_work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)fact,ui[mbs]*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);

  b->maxnz = b->nz = ui[mbs];

  fact->info.factor_mallocs   = reallocs;
  fact->info.fill_ratio_given = fill;
  if (ai[mbs] != 0) {
    fact->info.fill_ratio_needed = ((PetscReal)ui[mbs])/ai[mbs];
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[mbs] != 0) {
    PetscReal af = fact->info.fill_ratio_needed;
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %g or use \n",(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
  }
#endif
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ_inplace(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscErrorCode     ierr;
  PetscBool          perm_identity,missing;
  PetscReal          fill = info->fill;
  const PetscInt     *rip,*ai,*aj;
  PetscInt           i,mbs=a->mbs,bs=A->rmap->bs,reallocs=0,prow,d;
  PetscInt           *jl,jmin,jmax,nzk,*ui,k,j,*il,nextprow;
  PetscInt           nlnk,*lnk,ncols,*cols,*uj,**ui_ptr,*uj_ptr;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  ierr = MatMissingDiagonal(A,&missing,&d);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",d);

  /*
   This code originally uses Modified Sparse Row (MSR) storage
   (see page 85, "Iterative Methods ..." by Saad) for the output matrix B - bad choise!
   Then it is rewritten so the factor B takes seqsbaij format. However the associated
   MatCholeskyFactorNumeric_() have not been modified for the cases of bs>1 or !perm_identity,
   thus the original code in MSR format is still used for these cases.
   The code below should replace MatCholeskyFactorSymbolic_SeqSBAIJ_MSR() whenever
   MatCholeskyFactorNumeric_() is modified for using sbaij symbolic factor.
  */
  if (bs > 1) {
    ierr = MatCholeskyFactorSymbolic_SeqSBAIJ_MSR(fact,A,perm,info);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);

  if (perm_identity) {
    a->permute = PETSC_FALSE;

    ai = a->i; aj = a->j;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported for sbaij matrix. Use aij format");
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  /* initialization */
  ierr  = PetscMalloc1(mbs+1,&ui);CHKERRQ(ierr);
  ui[0] = 0;

  /* jl: linked list for storing indices of the pivot rows
     il: il[i] points to the 1st nonzero entry of U(i,k:mbs-1) */
  ierr = PetscMalloc4(mbs,&ui_ptr,mbs,&il,mbs,&jl,mbs,&cols);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = mbs + 1;
  ierr = PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill*(ai[mbs]+1) */
  ierr          = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,ai[mbs]+1),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (k=0; k<mbs; k++) {  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk   = 0;
    ncols = ai[rip[k]+1] - ai[rip[k]];
    for (j=0; j<ncols; j++) {
      i       = *(aj + ai[rip[k]] + j);
      cols[j] = rip[i];
    }
    ierr = PetscLLAdd(ncols,cols,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */

    while (prow < k) {
      nextprow = jl[prow];
      /* merge prow into k-th row */
      jmin   = il[prow] + 1; /* index of the 2nd nzero entry in U(prow,k:mbs-1) */
      jmax   = ui[prow+1];
      ncols  = jmax-jmin;
      uj_ptr = ui_ptr[prow] + jmin - ui[prow]; /* points to the 2nd nzero entry in U(prow,k:mbs-1) */
      ierr   = PetscLLAddSorted(ncols,uj_ptr,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      nzk   += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax) {
        il[prow] = jmin;

        j = *uj_ptr; jl[prow] = jl[j]; jl[j] = prow;
      }
      prow = nextprow;
    }

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i    = mbs - k + 1; /* num of unfactored rows */
      i    = PetscMin(PetscIntMultTruncate(i,nzk), PetscIntMultTruncate(i,i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(mbs,mbs,nzk,lnk,current_space->array,lnkbt);CHKERRQ(ierr);

    /* add the k-th row into il and jl */
    if (nzk-1 > 0) {
      i     = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:mbs-1) */
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    }
    ui_ptr[k] = current_space->array;

    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree4(ui_ptr,il,jl,cols);CHKERRQ(ierr);

  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ui[mbs]+1,&uj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,uj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */
  ierr = MatSeqSBAIJSetPreallocation(fact,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

  b               = (Mat_SeqSBAIJ*)fact->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;

  ierr = PetscMalloc1(ui[mbs]+1,&b->a);CHKERRQ(ierr);

  b->j    = uj;
  b->i    = ui;
  b->diag = 0;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;

  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  ierr     = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  b->icol  = perm;
  ierr     = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr     = PetscMalloc1(mbs+1,&b->solve_work);CHKERRQ(ierr);
  ierr     = PetscLogObjectMemory((PetscObject)fact,(ui[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = ui[mbs];

  fact->info.factor_mallocs   = reallocs;
  fact->info.fill_ratio_given = fill;
  if (ai[mbs] != 0) {
    fact->info.fill_ratio_needed = ((PetscReal)ui[mbs])/ai[mbs];
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[mbs] != 0) {
    PetscReal af = fact->info.fill_ratio_needed;
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %g or use \n",(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
  }
#endif
  ierr = MatSeqSBAIJSetNumericFactorization_inplace(fact,perm_identity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a   = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ*)C->data;
  IS             perm = b->row;
  PetscErrorCode ierr;
  const PetscInt *ai,*aj,*perm_ptr,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       i,j;
  PetscInt       *a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  PetscInt       bs  =A->rmap->bs,bs2 = a->bs2;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar      *work;
  PetscInt       *pivots;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  /* initialization */
  ierr = PetscCalloc1(bs2*mbs,&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,&il,mbs,&jl);CHKERRQ(ierr);
  allowzeropivot = PetscNot(A->erroriffailure);

  il[0] = 0;
  for (i=0; i<mbs; i++) jl[i] = mbs;

  ierr = PetscMalloc3(bs2,&dk,bs2,&uik,bs,&work);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&pivots);CHKERRQ(ierr);

  ierr = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute) {
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew;
    ierr = PetscMalloc1(bs2*ai[mbs],&aa);CHKERRQ(ierr);
    ierr = PetscArraycpy(aa,a->a,bs2*ai[mbs]);CHKERRQ(ierr);
    ierr = PetscMalloc1(ai[mbs],&a2anew);CHKERRQ(ierr);
    ierr = PetscArraycpy(a2anew,a->a2anew,ai[mbs]);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++) {
        while (a2anew[j] != j) {
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;
          for (k1=0; k1<bs2; k1++) {
            dk[k1]       = aa[k*bs2+k1];
            aa[k*bs2+k1] = aa[j*bs2+k1];
            aa[j*bs2+k1] = dk[k1];
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]) {
          ap = aa + j*bs2;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<bs2; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<bs; k++) {               /* j-th block of aa <- dk^T */
            for (k1=0; k1<bs; k1++) *ap++ = dk[k + bs*k1];
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRQ(ierr);
  }

  /* for each row k */
  for (k = 0; k<mbs; k++) {

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];

    ap = aa + jmin*bs2;
    for (j = jmin; j < jmax; j++) {
      vj       = perm_ptr[aj[j]];   /* block col. index */
      rtmp_ptr = rtmp + vj*bs2;
      for (i=0; i<bs2; i++) *rtmp_ptr++ = *ap++;
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscArraycpy(dk,rtmp+k*bs2,bs2);CHKERRQ(ierr);
    i    = jl[k]; /* first row to be added to k_th row  */

    while (i < k) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*bs2;
      u    = ba + ili*bs2;
      ierr = PetscArrayzero(uik,bs2);CHKERRQ(ierr);
      PetscKernel_A_gets_A_minus_B_times_C(bs,uik,diag,u);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      PetscKernel_A_gets_A_plus_Btranspose_times_C(bs,dk,uik,u);
      ierr = PetscLogFlops(4.0*bs*bs2);CHKERRQ(ierr);

      /* update -U(i,k) */
      ierr = PetscArraycpy(ba+ili*bs2,uik,bs2);CHKERRQ(ierr);

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*bs2;
          u        = ba + j*bs2;
          PetscKernel_A_gets_A_plus_Btranspose_times_C(bs,rtmp_ptr,uik,u);
        }
        ierr = PetscLogFlops(2.0*bs*bs2*(jmax-jmin));CHKERRQ(ierr);

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    diag = ba+k*bs2;
    ierr = PetscArraycpy(diag,dk,bs2);CHKERRQ(ierr);

    ierr = PetscKernel_A_gets_inverse_A(bs,diag,pivots,work,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++) {
        vj       = bj[j];      /* block col. index of U */
        u        = ba + j*bs2;
        rtmp_ptr = rtmp + vj*bs2;
        for (k1=0; k1<bs2; k1++) {
          *u++        = *rtmp_ptr;
          *rtmp_ptr++ = 0.0;
        }
      }

      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr);
  ierr = PetscFree3(dk,uik,work);CHKERRQ(ierr);
  ierr = PetscFree(pivots);CHKERRQ(ierr);
  if (a->permute) {
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_N_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_N_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_N_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_N_inplace;

  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;

  ierr = PetscLogFlops(1.3333*bs*bs2*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  PetscInt       bs  =A->rmap->bs,bs2 = a->bs2;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar      *work;
  PetscInt       *pivots;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  ierr = PetscCalloc1(bs2*mbs,&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,&il,mbs,&jl);CHKERRQ(ierr);
  il[0] = 0;
  for (i=0; i<mbs; i++) jl[i] = mbs;

  ierr = PetscMalloc3(bs2,&dk,bs2,&uik,bs,&work);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&pivots);CHKERRQ(ierr);
  allowzeropivot = PetscNot(A->erroriffailure);

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++) {

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    ap   = aa + jmin*bs2;
    for (j = jmin; j < jmax; j++) {
      vj       = aj[j];   /* block col. index */
      rtmp_ptr = rtmp + vj*bs2;
      for (i=0; i<bs2; i++) *rtmp_ptr++ = *ap++;
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscArraycpy(dk,rtmp+k*bs2,bs2);CHKERRQ(ierr);
    i    = jl[k]; /* first row to be added to k_th row  */

    while (i < k) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*bs2;
      u    = ba + ili*bs2;
      ierr = PetscArrayzero(uik,bs2);CHKERRQ(ierr);
      PetscKernel_A_gets_A_minus_B_times_C(bs,uik,diag,u);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      PetscKernel_A_gets_A_plus_Btranspose_times_C(bs,dk,uik,u);
      ierr = PetscLogFlops(2.0*bs*bs2);CHKERRQ(ierr);

      /* update -U(i,k) */
      ierr = PetscArraycpy(ba+ili*bs2,uik,bs2);CHKERRQ(ierr);

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*bs2;
          u        = ba + j*bs2;
          PetscKernel_A_gets_A_plus_Btranspose_times_C(bs,rtmp_ptr,uik,u);
        }
        ierr = PetscLogFlops(2.0*bs*bs2*(jmax-jmin));CHKERRQ(ierr);

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    diag = ba+k*bs2;
    ierr = PetscArraycpy(diag,dk,bs2);CHKERRQ(ierr);

    ierr = PetscKernel_A_gets_inverse_A(bs,diag,pivots,work,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++) {
        vj       = bj[j];      /* block col. index of U */
        u        = ba + j*bs2;
        rtmp_ptr = rtmp + vj*bs2;
        for (k1=0; k1<bs2; k1++) {
          *u++        = *rtmp_ptr;
          *rtmp_ptr++ = 0.0;
        }
      }

      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr);
  ierr = PetscFree3(dk,uik,work);CHKERRQ(ierr);
  ierr = PetscFree(pivots);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_N_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_N_NaturalOrdering_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  ierr = PetscLogFlops(1.3333*bs*bs2*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
    Numeric U^T*D*U factorization for SBAIJ format. Modified from SNF of YSMP.
    Version for blocks 2 by 2.
*/
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a   = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ*)C->data;
  IS             perm = b->row;
  PetscErrorCode ierr;
  const PetscInt *ai,*aj,*perm_ptr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr,dk[4],uik[4];
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);

  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list
     il(i): points to the first nonzero element in columns k,...,mbs-1 of
            row i of U */
  ierr = PetscCalloc1(4*mbs,&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,&il,mbs,&jl);CHKERRQ(ierr);
  il[0] = 0;
  for (i=0; i<mbs; i++) jl[i] = mbs;

  ierr = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute) {
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew;
    ierr = PetscMalloc1(4*ai[mbs],&aa);CHKERRQ(ierr);
    ierr = PetscArraycpy(aa,a->a,4*ai[mbs]);CHKERRQ(ierr);
    ierr = PetscMalloc1(ai[mbs],&a2anew);CHKERRQ(ierr);
    ierr = PetscArraycpy(a2anew,a->a2anew,ai[mbs]);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++) {
        while (a2anew[j] != j) {
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;
          for (k1=0; k1<4; k1++) {
            dk[k1]     = aa[k*4+k1];
            aa[k*4+k1] = aa[j*4+k1];
            aa[j*4+k1] = dk[k1];
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]) {
          ap    = aa + j*4;  /* ptr to the beginning of the block */
          dk[1] = ap[1];     /* swap ap[1] and ap[2] */
          ap[1] = ap[2];
          ap[2] = dk[1];
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRQ(ierr);
  }

  /* for each row k */
  for (k = 0; k<mbs; k++) {

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    ap   = aa + jmin*4;
    for (j = jmin; j < jmax; j++) {
      vj       = perm_ptr[aj[j]];   /* block col. index */
      rtmp_ptr = rtmp + vj*4;
      for (i=0; i<4; i++) *rtmp_ptr++ = *ap++;
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscArraycpy(dk,rtmp+k*4,4);CHKERRQ(ierr);
    i    = jl[k]; /* first row to be added to k_th row  */

    while (i < k) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k): - ba[ili]*ba[i] */
      diag   = ba + i*4;
      u      = ba + ili*4;
      uik[0] = -(diag[0]*u[0] + diag[2]*u[1]);
      uik[1] = -(diag[1]*u[0] + diag[3]*u[1]);
      uik[2] = -(diag[0]*u[2] + diag[2]*u[3]);
      uik[3] = -(diag[1]*u[2] + diag[3]*u[3]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k): dk += uik*ba[ili] */
      dk[0] += uik[0]*u[0] + uik[1]*u[1];
      dk[1] += uik[2]*u[0] + uik[3]*u[1];
      dk[2] += uik[0]*u[2] + uik[1]*u[3];
      dk[3] += uik[2]*u[2] + uik[3]*u[3];

      ierr = PetscLogFlops(16.0*2.0);CHKERRQ(ierr);

      /* update -U(i,k): ba[ili] = uik */
      ierr = PetscArraycpy(ba+ili*4,uik,4);CHKERRQ(ierr);

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j): rtmp[bj[j]] += uik*ba[j]; */
          rtmp_ptr     = rtmp + bj[j]*4;
          u            = ba + j*4;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1];
          rtmp_ptr[1] += uik[2]*u[0] + uik[3]*u[1];
          rtmp_ptr[2] += uik[0]*u[2] + uik[1]*u[3];
          rtmp_ptr[3] += uik[2]*u[2] + uik[3]*u[3];
        }
        ierr = PetscLogFlops(16.0*(jmax-jmin));CHKERRQ(ierr);

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    diag = ba+k*4;
    ierr = PetscArraycpy(diag,dk,4);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_2(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++) {
        vj       = bj[j];      /* block col. index of U */
        u        = ba + j*4;
        rtmp_ptr = rtmp + vj*4;
        for (k1=0; k1<4; k1++) {
          *u++        = *rtmp_ptr;
          *rtmp_ptr++ = 0.0;
        }
      }

      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr);
  if (a->permute) {
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_2_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_2_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  ierr = PetscLogFlops(1.3333*8*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 2 by 2 Using natural ordering
*/
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap,dk[8],uik[8];
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);

  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list
     il(i): points to the first nonzero element in columns k,...,mbs-1 of
            row i of U */
  ierr = PetscCalloc1(4*mbs,&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,&il,mbs,&jl);CHKERRQ(ierr);
  il[0] = 0;
  for (i=0; i<mbs; i++) jl[i] = mbs;

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++) {

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    ap   = aa + jmin*4;
    for (j = jmin; j < jmax; j++) {
      vj       = aj[j];   /* block col. index */
      rtmp_ptr = rtmp + vj*4;
      for (i=0; i<4; i++) *rtmp_ptr++ = *ap++;
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscArraycpy(dk,rtmp+k*4,4);CHKERRQ(ierr);
    i    = jl[k]; /* first row to be added to k_th row  */

    while (i < k) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k): - ba[ili]*ba[i] */
      diag   = ba + i*4;
      u      = ba + ili*4;
      uik[0] = -(diag[0]*u[0] + diag[2]*u[1]);
      uik[1] = -(diag[1]*u[0] + diag[3]*u[1]);
      uik[2] = -(diag[0]*u[2] + diag[2]*u[3]);
      uik[3] = -(diag[1]*u[2] + diag[3]*u[3]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k): dk += uik*ba[ili] */
      dk[0] += uik[0]*u[0] + uik[1]*u[1];
      dk[1] += uik[2]*u[0] + uik[3]*u[1];
      dk[2] += uik[0]*u[2] + uik[1]*u[3];
      dk[3] += uik[2]*u[2] + uik[3]*u[3];

      ierr = PetscLogFlops(16.0*2.0);CHKERRQ(ierr);

      /* update -U(i,k): ba[ili] = uik */
      ierr = PetscArraycpy(ba+ili*4,uik,4);CHKERRQ(ierr);

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j): rtmp[bj[j]] += uik*ba[j]; */
          rtmp_ptr     = rtmp + bj[j]*4;
          u            = ba + j*4;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1];
          rtmp_ptr[1] += uik[2]*u[0] + uik[3]*u[1];
          rtmp_ptr[2] += uik[0]*u[2] + uik[1]*u[3];
          rtmp_ptr[3] += uik[2]*u[2] + uik[3]*u[3];
        }
        ierr = PetscLogFlops(16.0*(jmax-jmin));CHKERRQ(ierr);

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    diag = ba+k*4;
    ierr = PetscArraycpy(diag,dk,4);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_2(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++) {
        vj       = bj[j];      /* block col. index of U */
        u        = ba + j*4;
        rtmp_ptr = rtmp + vj*4;
        for (k1=0; k1<4; k1++) {
          *u++        = *rtmp_ptr;
          *rtmp_ptr++ = 0.0;
        }
      }

      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_2_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_2_NaturalOrdering_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  ierr = PetscLogFlops(1.3333*8*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
    Numeric U^T*D*U factorization for SBAIJ format.
    Version for blocks are 1 by 1.
*/
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data,*b=(Mat_SeqSBAIJ*)C->data;
  IS             ip=b->row;
  PetscErrorCode ierr;
  const PetscInt *ai,*aj,*rip;
  PetscInt       *a2anew,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j,*bcol;
  PetscInt       k,jmin,jmax,*jl,*il,col,nexti,ili,nz;
  MatScalar      *rtmp,*ba=b->a,*bval,*aa,dk,uikdi;
  PetscReal      rs;
  FactorShiftCtx sctx;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  if (!a->permute) {
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai     = a->inew; aj = a->jnew;
    nz     = ai[mbs];
    ierr   = PetscMalloc1(nz,&aa);CHKERRQ(ierr);
    a2anew = a->a2anew;
    bval   = a->a;
    for (j=0; j<nz; j++) {
      aa[a2anew[j]] = *(bval++);
    }
  }

  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list
     il(i): points to the first nonzero element in columns k,...,mbs-1 of
            row i of U */
  ierr = PetscMalloc3(mbs,&rtmp,mbs,&il,mbs,&jl);CHKERRQ(ierr);

  do {
    sctx.newshift = PETSC_FALSE;
    il[0] = 0;
    for (i=0; i<mbs; i++) {
      rtmp[i] = 0.0; jl[i] = mbs;
    }

    for (k = 0; k<mbs; k++) {
      /*initialize k-th row by the perm[k]-th row of A */
      jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
      bval = ba + bi[k];
      for (j = jmin; j < jmax; j++) {
        col       = rip[aj[j]];
        rtmp[col] = aa[j];
        *bval++   = 0.0; /* for in-place factorization */
      }

      /* shift the diagonal of the matrix */
      if (sctx.nshift) rtmp[k] += sctx.shift_amount;

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = jl[k]; /* first row to be added to k_th row  */

      while (i < k) {
        nexti = jl[i]; /* next row to be added to k_th row */

        /* compute multiplier, update diag(k) and U(i,k) */
        ili     = il[i]; /* index of first nonzero element in U(i,k:bms-1) */
        uikdi   = -ba[ili]*ba[bi[i]]; /* diagonal(k) */
        dk     += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row */
        jmin = ili + 1; jmax = bi[i+1];
        if (jmin < jmax) {
          for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];
          ierr = PetscLogFlops(2.0*(jmax-jmin));CHKERRQ(ierr);

          /* update il and jl for row i */
          il[i] = jmin;
          j     = bj[jmin]; jl[i] = jl[j]; jl[j] = i;
        }
        i = nexti;
      }

      /* shift the diagonals when zero pivot is detected */
      /* compute rs=sum of abs(off-diagonal) */
      rs   = 0.0;
      jmin = bi[k]+1;
      nz   = bi[k+1] - jmin;
      if (nz) {
        bcol = bj + jmin;
        while (nz--) {
          rs += PetscAbsScalar(rtmp[*bcol]);
          bcol++;
        }
      }

      sctx.rs = rs;
      sctx.pv = dk;
      ierr    = MatPivotCheck(C,A,info,&sctx,k);CHKERRQ(ierr);
      if (sctx.newshift) break;    /* sctx.shift_amount is updated */
      dk = sctx.pv;

      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk; /* U(k,k) */
      jmin      = bi[k]+1; jmax = bi[k+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          col = bj[j]; ba[j] = rtmp[col]; rtmp[col] = 0.0;
        }
        /* add the k-th row into il and jl */
        il[k] = jmin;
        i     = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }
    }
  } while (sctx.newshift);
  ierr = PetscFree3(rtmp,il,jl);CHKERRQ(ierr);
  if (a->permute) {ierr = PetscFree(aa);CHKERRQ(ierr);}

  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_1_inplace;
  C->ops->solves         = MatSolves_SeqSBAIJ_1_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_1_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_1_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_1_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  ierr = PetscLogFlops(C->rmap->N);CHKERRQ(ierr);
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shiftnz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo2(A,"number of shiftpd tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  Version for when blocks are 1 by 1 Using natural ordering under new datastructure
  Modified from MatCholeskyFactorNumeric_SeqAIJ()
*/
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=A->rmap->n,*bi=b->i,*bj=b->j,*bdiag=b->diag,*bjtmp;
  PetscInt       *ai=a->i,*aj=a->j,*ajtmp;
  PetscInt       k,jmin,jmax,*c2r,*il,col,nexti,ili,nz;
  MatScalar      *rtmp,*ba=b->a,*bval,*aa=a->a,dk,uikdi;
  FactorShiftCtx sctx;
  PetscReal      rs;
  MatScalar      d,*v;

  PetscFunctionBegin;
  ierr = PetscMalloc3(mbs,&rtmp,mbs,&il,mbs,&c2r);CHKERRQ(ierr);

  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    sctx.shift_top = info->zeropivot;

    ierr = PetscArrayzero(rtmp,mbs);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d        = (aa)[a->diag[i]];
      rtmp[i] += -PetscRealPart(d);  /* diagonal entry */
      ajtmp    = aj + ai[i] + 1;     /* exclude diagonal */
      v        = aa + ai[i] + 1;
      nz       = ai[i+1] - ai[i] - 1;
      for (j=0; j<nz; j++) {
        rtmp[i]        += PetscAbsScalar(v[j]);
        rtmp[ajtmp[j]] += PetscAbsScalar(v[j]);
      }
      if (PetscRealPart(rtmp[i]) > sctx.shift_top) sctx.shift_top = PetscRealPart(rtmp[i]);
    }
    sctx.shift_top *= 1.1;
    sctx.nshift_max = 5;
    sctx.shift_lo   = 0.;
    sctx.shift_hi   = 1.;
  }

  /* allocate working arrays
     c2r: linked list, keep track of pivot rows for a given column. c2r[col]: head of the list for a given col
     il:  for active k row, il[i] gives the index of the 1st nonzero entry in U[i,k:n-1] in bj and ba arrays
  */
  do {
    sctx.newshift = PETSC_FALSE;

    for (i=0; i<mbs; i++) c2r[i] = mbs;
    if (mbs) il[0] = 0;

    for (k = 0; k<mbs; k++) {
      /* zero rtmp */
      nz    = bi[k+1] - bi[k];
      bjtmp = bj + bi[k];
      for (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;

      /* load in initial unfactored row */
      bval = ba + bi[k];
      jmin = ai[k]; jmax = ai[k+1];
      for (j = jmin; j < jmax; j++) {
        col       = aj[j];
        rtmp[col] = aa[j];
        *bval++   = 0.0; /* for in-place factorization */
      }
      /* shift the diagonal of the matrix: ZeropivotApply() */
      rtmp[k] += sctx.shift_amount;  /* shift the diagonal of the matrix */

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = c2r[k]; /* first row to be added to k_th row  */

      while (i < k) {
        nexti = c2r[i]; /* next row to be added to k_th row */

        /* compute multiplier, update diag(k) and U(i,k) */
        ili     = il[i]; /* index of first nonzero element in U(i,k:bms-1) */
        uikdi   = -ba[ili]*ba[bdiag[i]]; /* diagonal(k) */
        dk     += uikdi*ba[ili]; /* update diag[k] */
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row */
        jmin = ili + 1; jmax = bi[i+1];
        if (jmin < jmax) {
          for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];
          /* update il and c2r for row i */
          il[i] = jmin;
          j     = bj[jmin]; c2r[i] = c2r[j]; c2r[j] = i;
        }
        i = nexti;
      }

      /* copy data into U(k,:) */
      rs   = 0.0;
      jmin = bi[k]; jmax = bi[k+1]-1;
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          col = bj[j]; ba[j] = rtmp[col]; rs += PetscAbsScalar(ba[j]);
        }
        /* add the k-th row into il and c2r */
        il[k] = jmin;
        i     = bj[jmin]; c2r[k] = c2r[i]; c2r[i] = k;
      }

      sctx.rs = rs;
      sctx.pv = dk;
      ierr    = MatPivotCheck(B,A,info,&sctx,k);CHKERRQ(ierr);
      if (sctx.newshift) break;
      dk = sctx.pv;

      ba[bdiag[k]] = 1.0/dk; /* U(k,k) */
    }
  } while (sctx.newshift);

  ierr = PetscFree3(rtmp,il,c2r);CHKERRQ(ierr);

  B->ops->solve          = MatSolve_SeqSBAIJ_1_NaturalOrdering;
  B->ops->solves         = MatSolves_SeqSBAIJ_1;
  B->ops->solvetranspose = MatSolve_SeqSBAIJ_1_NaturalOrdering;
  B->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_1_NaturalOrdering;
  B->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering;

  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  ierr = PetscLogFlops(B->rmap->n);CHKERRQ(ierr);

  /* MatPivotView() */
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %g, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,(double)sctx.shift_amount,(double)sctx.shift_fraction,(double)sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS) {
      ierr = PetscInfo2(A,"number of shift_inblocks applied %D, each shift_amount %g\n",sctx.nshift,(double)info->shiftamount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data,*b=(Mat_SeqSBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs = a->mbs;
  PetscInt       *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscInt       k,jmin,*jl,*il,nexti,ili,*acol,*bcol,nz;
  MatScalar      *rtmp,*ba=b->a,*aa=a->a,dk,uikdi,*aval,*bval;
  PetscReal      rs;
  FactorShiftCtx sctx;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list
     il(i): points to the first nonzero element in U(i,k:mbs-1)
  */
  ierr = PetscMalloc1(mbs,&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,&il,mbs,&jl);CHKERRQ(ierr);

  do {
    sctx.newshift = PETSC_FALSE;
    il[0] = 0;
    for (i=0; i<mbs; i++) {
      rtmp[i] = 0.0; jl[i] = mbs;
    }

    for (k = 0; k<mbs; k++) {
      /*initialize k-th row with elements nonzero in row perm(k) of A */
      nz   = ai[k+1] - ai[k];
      acol = aj + ai[k];
      aval = aa + ai[k];
      bval = ba + bi[k];
      while (nz--) {
        rtmp[*acol++] = *aval++;
        *bval++       = 0.0; /* for in-place factorization */
      }

      /* shift the diagonal of the matrix */
      if (sctx.nshift) rtmp[k] += sctx.shift_amount;

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = jl[k]; /* first row to be added to k_th row  */

      while (i < k) {
        nexti = jl[i]; /* next row to be added to k_th row */
        /* compute multiplier, update D(k) and U(i,k) */
        ili     = il[i]; /* index of first nonzero element in U(i,k:bms-1) */
        uikdi   = -ba[ili]*ba[bi[i]];
        dk     += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row ... */
        jmin = ili + 1;
        nz   = bi[i+1] - jmin;
        if (nz > 0) {
          bcol = bj + jmin;
          bval = ba + jmin;
          ierr = PetscLogFlops(2.0*nz);CHKERRQ(ierr);
          while (nz--) rtmp[*bcol++] += uikdi*(*bval++);

          /* update il and jl for i-th row */
          il[i] = jmin;
          j     = bj[jmin]; jl[i] = jl[j]; jl[j] = i;
        }
        i = nexti;
      }

      /* shift the diagonals when zero pivot is detected */
      /* compute rs=sum of abs(off-diagonal) */
      rs   = 0.0;
      jmin = bi[k]+1;
      nz   = bi[k+1] - jmin;
      if (nz) {
        bcol = bj + jmin;
        while (nz--) {
          rs += PetscAbsScalar(rtmp[*bcol]);
          bcol++;
        }
      }

      sctx.rs = rs;
      sctx.pv = dk;
      ierr    = MatPivotCheck(C,A,info,&sctx,k);CHKERRQ(ierr);
      if (sctx.newshift) break;    /* sctx.shift_amount is updated */
      dk = sctx.pv;

      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk;
      jmin      = bi[k]+1;
      nz        = bi[k+1] - jmin;
      if (nz) {
        bcol = bj + jmin;
        bval = ba + jmin;
        while (nz--) {
          *bval++       = rtmp[*bcol];
          rtmp[*bcol++] = 0.0;
        }
        /* add k-th row into il and jl */
        il[k] = jmin;
        i     = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }
    } /* end of for (k = 0; k<mbs; k++) */
  } while (sctx.newshift);
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->ops->solves         = MatSolves_SeqSBAIJ_1_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace;

  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;

  ierr = PetscLogFlops(C->rmap->N);CHKERRQ(ierr);
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shiftnz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo2(A,"number of shiftpd tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactor_SeqSBAIJ(Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  Mat            C;

  PetscFunctionBegin;
  ierr = MatGetFactor(A,"petsc",MAT_FACTOR_CHOLESKY,&C);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(C,A,perm,info);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(C,A,info);CHKERRQ(ierr);

  A->ops->solve          = C->ops->solve;
  A->ops->solvetranspose = C->ops->solvetranspose;

  ierr = MatHeaderMerge(A,&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
