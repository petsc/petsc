
#include "src/mat/impls/baij/seq/baij.h" 
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/inline/ilu.h"
#include "include/petscis.h"

#if !defined(PETSC_USE_COMPLEX)
/* 
  input:
   F -- numeric factor 
  output:
   nneg, nzero, npos: matrix inertia 
*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia_SeqSBAIJ"
PetscErrorCode MatGetInertia_SeqSBAIJ(Mat F,PetscInt *nneig,PetscInt *nzero,PetscInt *npos)
{ 
  Mat_SeqSBAIJ *fact_ptr = (Mat_SeqSBAIJ*)F->data;
  PetscScalar  *dd = fact_ptr->a;
  PetscInt     mbs=fact_ptr->mbs,bs=F->bs,i,nneig_tmp,npos_tmp,*fi = fact_ptr->i;

  PetscFunctionBegin;
  if (bs != 1) SETERRQ1(PETSC_ERR_SUP,"No support for bs: %D >1 yet",bs);
  nneig_tmp = 0; npos_tmp = 0;
  for (i=0; i<mbs; i++){
    if (PetscRealPart(dd[*fi]) > 0.0){
      npos_tmp++;
    } else if (PetscRealPart(dd[*fi]) < 0.0){
      nneig_tmp++;
    }
    fi++;
  }
  if (nneig) *nneig = nneig_tmp;
  if (npos)  *npos  = npos_tmp;
  if (nzero) *nzero = mbs - nneig_tmp - npos_tmp;

  PetscFunctionReturn(0);
}
#endif /* !defined(PETSC_USE_COMPLEX) */

/* 
  Symbolic U^T*D*U factorization for SBAIJ format. Modified from SSF of YSMP.
  Use Modified Sparse Row (MSR) storage for u and ju. See page 85, "Iterative Methods ..." by Saad. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSBAIJ_MSR"
PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ_MSR(Mat A,IS perm,MatFactorInfo *info,Mat *B)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b;
  PetscErrorCode ierr;
  PetscInt       *rip,i,mbs = a->mbs,*ai,*aj;
  PetscInt       *jutmp,bs = A->bs,bs2=a->bs2;
  PetscInt       m,reallocs = 0,prow;
  PetscInt       *jl,*q,jmin,jmax,juidx,nzk,qm,*iu,*ju,k,j,vj,umax,maxadd;
  PetscReal      f = info->fill;
  PetscTruth     perm_identity;

  PetscFunctionBegin;
  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);   
  
  if (perm_identity){ /* without permutation */
    a->permute = PETSC_FALSE;
    ai = a->i; aj = a->j;
  } else {            /* non-trivial permutation */    
    a->permute = PETSC_TRUE;
    ierr = MatReorderingSeqSBAIJ(A,perm);CHKERRQ(ierr);   
    ai = a->inew; aj = a->jnew;
  }
  
  /* initialization */
  ierr  = PetscMalloc((mbs+1)*sizeof(PetscInt),&iu);CHKERRQ(ierr);
  umax  = (PetscInt)(f*ai[mbs] + 1); umax += mbs + 1; 
  ierr  = PetscMalloc(umax*sizeof(PetscInt),&ju);CHKERRQ(ierr);
  iu[0] = mbs+1; 
  juidx = mbs + 1; /* index for ju */
  ierr  = PetscMalloc(2*mbs*sizeof(PetscInt),&jl);CHKERRQ(ierr); /* linked list for pivot row */
  q     = jl + mbs;   /* linked list for col index */
  for (i=0; i<mbs; i++){
    jl[i] = mbs; 
    q[i] = 0;
  }

  /* for each row k */
  for (k=0; k<mbs; k++){   
    for (i=0; i<mbs; i++) q[i] = 0;  /* to be removed! */
    nzk  = 0; /* num. of nz blocks in k-th block row with diagonal block excluded */
    q[k] = mbs;
    /* initialize nonzero structure of k-th row to row rip[k] of A */
    jmin = ai[rip[k]] +1; /* exclude diag[k] */
    jmax = ai[rip[k]+1];
    for (j=jmin; j<jmax; j++){
      vj = rip[aj[j]]; /* col. value */
      if(vj > k){
        qm = k; 
        do {
          m  = qm; qm = q[m];
        } while(qm < vj);
        if (qm == vj) {
          SETERRQ(PETSC_ERR_PLIB,"Duplicate entry in A\n"); 
        }     
        nzk++;
        q[m]  = vj;
        q[vj] = qm;  
      } /* if(vj > k) */
    } /* for (j=jmin; j<jmax; j++) */

    /* modify nonzero structure of k-th row by computing fill-in
       for each row i to be merged in */
    prow = k; 
    prow = jl[prow]; /* next pivot row (== mbs for symbolic factorization) */
   
    while (prow < k){
      /* merge row prow into k-th row */
      jmin = iu[prow] + 1; jmax = iu[prow+1];
      qm = k;
      for (j=jmin; j<jmax; j++){
        vj = ju[j];
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        if (qm != vj){
         nzk++; q[m] = vj; q[vj] = qm; qm = vj;
        }
      } 
      prow = jl[prow]; /* next pivot row */     
    }  
   
    /* add k to row list for first nonzero element in k-th row */
    if (nzk > 0){
      i = q[k]; /* col value of first nonzero element in U(k, k+1:mbs-1) */    
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
      ierr = PetscMalloc(umax*sizeof(PetscInt),&jutmp);CHKERRQ(ierr);
      ierr = PetscMemcpy(jutmp,ju,iu[k]*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscFree(ju);CHKERRQ(ierr);       
      ju   = jutmp; 
      reallocs++; /* count how many times we realloc */
    }

    /* save nonzero structure of k-th row in ju */
    i=k;
    while (nzk --) {
      i           = q[i];
      ju[juidx++] = i;
    }     
  } 

  if (ai[mbs] != 0) {
    PetscReal af = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Reallocs %D Fill ratio:given %g needed %g\n",reallocs,f,af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Run with -pc_cholesky_fill %g or use \n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:PCCholeskySetFill(pc,%g);\n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:for best performance.\n");
  } else {
     PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Empty matrix.\n");
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree(jl);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatCreate(A->comm,bs*mbs,bs*mbs,bs*mbs,bs*mbs,B);CHKERRQ(ierr);
  ierr = MatSetType(*B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(*B,bs,0,PETSC_NULL);CHKERRQ(ierr);

  /* PetscLogObjectParent(*B,iperm); */
  b = (Mat_SeqSBAIJ*)(*B)->data;
  ierr = PetscFree(b->imax);CHKERRQ(ierr);
  b->singlemalloc = PETSC_FALSE;
  /* the next line frees the default space generated by the Create() */
  ierr = PetscFree(b->a);CHKERRQ(ierr);
  ierr = PetscFree(b->ilen);CHKERRQ(ierr);
  ierr = PetscMalloc((iu[mbs]+1)*sizeof(MatScalar)*bs2,&b->a);CHKERRQ(ierr);
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
  ierr    = PetscMalloc((bs*mbs+bs)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PetscLogObjectMemory(*B,(iu[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar)));
  b->maxnz = b->nz = iu[mbs];
  
  (*B)->factor                 = FACTOR_CHOLESKY;
  (*B)->info.factor_mallocs    = reallocs;
  (*B)->info.fill_ratio_given  = f;
  if (ai[mbs] != 0) {
    (*B)->info.fill_ratio_needed = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
  } else {
    (*B)->info.fill_ratio_needed = 0.0;
  }

  if (perm_identity){
    switch (bs) {
      case 1:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_1_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=1\n");
        break;
      case 2:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_2_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=2\n");
        break;
      case 3:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_3_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:sing special in-place natural ordering factor and solve BS=3\n");
        break; 
      case 4:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_4_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=4\n"); 
        break;
      case 5:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_5_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=5\n"); 
        break;
      case 6: 
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_6_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=6\n");
        break; 
      case 7:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_7_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=7\n");
      break; 
      default:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering; 
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_N_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS>7\n");
      break; 
    }
  } 
  PetscFunctionReturn(0); 
}
/*
    Symbolic U^T*D*U factorization for SBAIJ format. 
*/
#include "petscbt.h"
#include "src/mat/utils/freespace.h"
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSBAIJ"
PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ(Mat A,IS perm,MatFactorInfo *info,Mat *fact)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBAIJ   *b;
  Mat            B;
  PetscErrorCode ierr;
  PetscTruth     perm_identity;
  PetscReal      fill = info->fill;
  PetscInt       *rip,i,mbs=a->mbs,bs=A->bs,*ai,*aj,reallocs=0,prow;
  PetscInt       *jl,jmin,jmax,nzk,*ui,k,j,*il,nextprow;
  PetscInt       nlnk,*lnk,ncols,*cols,*uj,**ui_ptr,*uj_ptr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT        lnkbt;

  PetscFunctionBegin;
  /*  
   This code originally uses Modified Sparse Row (MSR) storage
   (see page 85, "Iterative Methods ..." by Saad) for the output matrix B - bad choise!
   Then it is rewritten so the factor B takes seqsbaij format. However the associated 
   MatCholeskyFactorNumeric_() have not been modified for the cases of bs>1 or !perm_identity, 
   thus the original code in MSR format is still used for these cases. 
   The code below should replace MatCholeskyFactorSymbolic_SeqSBAIJ_MSR() whenever 
   MatCholeskyFactorNumeric_() is modified for using sbaij symbolic factor.
  */
  if (bs > 1){  
    ierr = MatCholeskyFactorSymbolic_SeqSBAIJ_MSR(A,perm,info,fact);CHKERRQ(ierr);
    PetscFunctionReturn(0); 
  } 

  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);

  if (perm_identity){
    a->permute = PETSC_FALSE; 
    ai = a->i; aj = a->j;
  } else {
    a->permute = PETSC_TRUE;
    ierr = MatReorderingSeqSBAIJ(A,perm);CHKERRQ(ierr);   
    ai = a->inew; aj = a->jnew;
  }
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);   

  /* initialization */
  ierr  = PetscMalloc((mbs+1)*sizeof(PetscInt),&ui);CHKERRQ(ierr);
  ui[0] = 0; 

  /* jl: linked list for storing indices of the pivot rows 
     il: il[i] points to the 1st nonzero entry of U(i,k:mbs-1) */
  ierr = PetscMalloc((3*mbs+1)*sizeof(PetscInt)+mbs*sizeof(PetscInt*),&jl);CHKERRQ(ierr); 
  il     = jl + mbs;
  cols   = il + mbs;
  ui_ptr = (PetscInt**)(cols + mbs);
  
  for (i=0; i<mbs; i++){
    jl[i] = mbs; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = mbs + 1;
  ierr = PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill*(ai[mbs]+1) */
  ierr = GetMoreSpace((PetscInt)(fill*(ai[mbs]+1)),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (k=0; k<mbs; k++){  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk   = 0;
    ncols = ai[rip[k]+1] - ai[rip[k]]; 
    for (j=0; j<ncols; j++){
      i = *(aj + ai[rip[k]] + j);
      cols[j] = rip[i];
    }
    ierr = PetscLLAdd(ncols,cols,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */
   
    while (prow < k){
      nextprow = jl[prow];
      /* merge prow into k-th row */
      jmin = il[prow] + 1;  /* index of the 2nd nzero entry in U(prow,k:mbs-1) */
      jmax = ui[prow+1]; 
      ncols = jmax-jmin;
      uj_ptr = ui_ptr[prow] + jmin - ui[prow]; /* points to the 2nd nzero entry in U(prow,k:mbs-1) */
      ierr = PetscLLAdd(ncols,uj_ptr,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      nzk += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax){
        il[prow] = jmin;
        j = *uj_ptr; jl[prow] = jl[j]; jl[j] = prow;  
      } 
      prow = nextprow; 
    }  

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i = mbs - k + 1; /* num of unfactored rows */
      i = PetscMin(i*nzk, i*(i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = GetMoreSpace(i,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(mbs,mbs,nzk,lnk,current_space->array,lnkbt);CHKERRQ(ierr); 

    /* add the k-th row into il and jl */
    if (nzk-1 > 0){
      i = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:mbs-1) */    
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    } 
    ui_ptr[k] = current_space->array;
    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;  
  } 

  if (ai[mbs] != 0) {
    PetscReal af = ((PetscReal)ui[mbs])/((PetscReal)ai[mbs]);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Reallocs %D Fill ratio:given %g needed %g\n",reallocs,fill,af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Run with -pc_cholesky_fill %g or use \n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:PCCholeskySetFill(pc,%g) for best performance.\n",af);
  } else {
     PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Empty matrix.\n");
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree(jl);CHKERRQ(ierr);

  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ui[mbs]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,uj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */
  ierr = MatCreate(PETSC_COMM_SELF,mbs,mbs,mbs,mbs,fact);CHKERRQ(ierr);
  B    = *fact;
  ierr = MatSetType(B,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,bs,0,PETSC_NULL);CHKERRQ(ierr);

  b = (Mat_SeqSBAIJ*)B->data;
  ierr = PetscFree(b->imax);CHKERRQ(ierr);
  b->singlemalloc = PETSC_FALSE;
  /* the next line frees the default space generated by the Create() */
  ierr = PetscFree(b->a);CHKERRQ(ierr);
  ierr = PetscFree(b->ilen);CHKERRQ(ierr);
  ierr = PetscMalloc((ui[mbs]+1)*sizeof(MatScalar),&b->a);CHKERRQ(ierr);
  b->j    = uj;
  b->i    = ui;
  b->diag = 0;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  ierr    = PetscMalloc((mbs+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  PetscLogObjectMemory(B,(ui[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar)));
  b->maxnz = b->nz = ui[mbs];
  
  B->factor                 = FACTOR_CHOLESKY;
  B->info.factor_mallocs    = reallocs;
  B->info.fill_ratio_given  = fill;
  if (ai[mbs] != 0) {
    B->info.fill_ratio_needed = ((PetscReal)ui[mbs])/((PetscReal)ai[mbs]);
  } else {
    B->info.fill_ratio_needed = 0.0;
  }

  if (perm_identity){
    switch (bs) {
      case 1:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_1_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=1\n");
        break;
      case 2:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_2_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=2\n");
        break;
      case 3:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_3_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:sing special in-place natural ordering factor and solve BS=3\n");
        break; 
      case 4:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_4_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=4\n"); 
        break;
      case 5:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_5_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=5\n"); 
        break;
      case 6: 
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_6_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=6\n");
        break; 
      case 7:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering;
        B->ops->solve           = MatSolve_SeqSBAIJ_7_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=7\n");
      break; 
      default:
        B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering; 
        B->ops->solve           = MatSolve_SeqSBAIJ_N_NaturalOrdering;
        PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS>7\n");
      break; 
    }
  } 
  PetscFunctionReturn(0); 
}
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_N"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N(Mat A,Mat *B)
{
  Mat            C = *B;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS             perm = b->row;
  PetscErrorCode ierr;
  PetscInt       *perm_ptr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  PetscInt       bs=A->bs,bs2 = a->bs2;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar      *work;
  PetscInt       *pivots;

  PetscFunctionBegin;
  /* initialization */
  ierr = PetscMalloc(bs2*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(PetscInt),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc((2*bs2+bs)*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + bs2;
  work = uik + bs2;
  ierr = PetscMalloc(bs*sizeof(PetscInt),&pivots);CHKERRQ(ierr);
 
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);
  
  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew; 
    ierr = PetscMalloc(bs2*ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    ierr = PetscMemcpy(aa,a->a,bs2*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(PetscInt),&a2anew);CHKERRQ(ierr); 
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(PetscInt));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<bs2; k1++){
            dk[k1]       = aa[k*bs2+k1]; 
            aa[k*bs2+k1] = aa[j*bs2+k1]; 
            aa[j*bs2+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*bs2;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<bs2; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<bs; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<bs; k1++) *ap++ = dk[k + bs*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRQ(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
  
    ap = aa + jmin*bs2;
    for (j = jmin; j < jmax; j++){
      vj = perm_ptr[aj[j]];         /* block col. index */  
      rtmp_ptr = rtmp + vj*bs2;
      for (i=0; i<bs2; i++) *rtmp_ptr++ = *ap++;        
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*bs2;
      u    = ba + ili*bs2;
      ierr = PetscMemzero(uik,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      Kernel_A_gets_A_minus_B_times_C(bs,uik,diag,u);
      
      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      Kernel_A_gets_A_plus_Btranspose_times_C(bs,dk,uik,u);
 
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*bs2,uik,bs2*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*bs2;
          u = ba + j*bs2;
          Kernel_A_gets_A_plus_Btranspose_times_C(bs,rtmp_ptr,uik,u);
        }
      
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
    ierr = PetscMemcpy(diag,dk,bs2*sizeof(MatScalar));CHKERRQ(ierr);   
    ierr = Kernel_A_gets_inverse_A(bs,diag,pivots,work);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*bs2;
         rtmp_ptr = rtmp + vj*bs2;        
         for (k1=0; k1<bs2; k1++){
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
  ierr = PetscFree(il);CHKERRQ(ierr); 
  ierr = PetscFree(dk);CHKERRQ(ierr);
  ierr = PetscFree(pivots);CHKERRQ(ierr);
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor       = FACTOR_CHOLESKY;
  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(1.3333*bs*bs2*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat A,Mat *B)
{
  Mat            C = *B;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  PetscInt       bs=A->bs,bs2 = a->bs2;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar      *work;
  PetscInt       *pivots;

  PetscFunctionBegin;
  /* initialization */
  
  ierr = PetscMalloc(bs2*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(PetscInt),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc((2*bs2+bs)*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + bs2;
  work = uik + bs2;
  ierr = PetscMalloc(bs*sizeof(PetscInt),&pivots);CHKERRQ(ierr);
 
  ai = a->i; aj = a->j; aa = a->a;
   
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    ap = aa + jmin*bs2;
    for (j = jmin; j < jmax; j++){
      vj = aj[j];         /* block col. index */  
      rtmp_ptr = rtmp + vj*bs2;
      for (i=0; i<bs2; i++) *rtmp_ptr++ = *ap++;        
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*bs2;
      u    = ba + ili*bs2;
      ierr = PetscMemzero(uik,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      Kernel_A_gets_A_minus_B_times_C(bs,uik,diag,u);
      
      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      Kernel_A_gets_A_plus_Btranspose_times_C(bs,dk,uik,u);
 
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*bs2,uik,bs2*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*bs2;
          u = ba + j*bs2;
          Kernel_A_gets_A_plus_Btranspose_times_C(bs,rtmp_ptr,uik,u);
        }
      
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
    ierr = PetscMemcpy(diag,dk,bs2*sizeof(MatScalar));CHKERRQ(ierr);   
    ierr = Kernel_A_gets_inverse_A(bs,diag,pivots,work);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*bs2;
         rtmp_ptr = rtmp + vj*bs2;        
         for (k1=0; k1<bs2; k1++){
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
  ierr = PetscFree(il);CHKERRQ(ierr);
  ierr = PetscFree(dk);CHKERRQ(ierr);
  ierr = PetscFree(pivots);CHKERRQ(ierr);

  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(1.3333*bs*bs2*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
    Numeric U^T*D*U factorization for SBAIJ format. Modified from SNF of YSMP.
    Version for blocks 2 by 2.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_2"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2(Mat A,Mat *B)
{
  Mat            C = *B;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS             perm = b->row;
  PetscErrorCode ierr;
  PetscInt       *perm_ptr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;

  PetscFunctionBegin;
  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing 
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows 
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list                        
     il(i): points to the first nonzero element in columns k,...,mbs-1 of 
            row i of U */
  ierr = PetscMalloc(4*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,4*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(PetscInt),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc(8*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + 4;     
  ierr = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew; 
    ierr = PetscMalloc(4*ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    ierr = PetscMemcpy(aa,a->a,4*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(PetscInt),&a2anew);CHKERRQ(ierr); 
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(PetscInt));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<4; k1++){
            dk[k1]       = aa[k*4+k1]; 
            aa[k*4+k1] = aa[j*4+k1]; 
            aa[j*4+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*4;     /* ptr to the beginning of the block */
          dk[1] = ap[1];     /* swap ap[1] and ap[2] */
          ap[1] = ap[2];
          ap[2] = dk[1];
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRQ(ierr); 
  }

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];    
    ap = aa + jmin*4;
    for (j = jmin; j < jmax; j++){
      vj = perm_ptr[aj[j]];         /* block col. index */  
      rtmp_ptr = rtmp + vj*4;
      for (i=0; i<4; i++) *rtmp_ptr++ = *ap++;        
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*4,4*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k): - ba[ili]*ba[i] */
      diag = ba + i*4;
      u    = ba + ili*4;
      uik[0] = -(diag[0]*u[0] + diag[2]*u[1]);
      uik[1] = -(diag[1]*u[0] + diag[3]*u[1]);
      uik[2] = -(diag[0]*u[2] + diag[2]*u[3]);
      uik[3] = -(diag[1]*u[2] + diag[3]*u[3]);
  
      /* update D(k) += -U(i,k)^T * U_bar(i,k): dk += uik*ba[ili] */
      dk[0] += uik[0]*u[0] + uik[1]*u[1];
      dk[1] += uik[2]*u[0] + uik[3]*u[1];
      dk[2] += uik[0]*u[2] + uik[1]*u[3];
      dk[3] += uik[2]*u[2] + uik[3]*u[3];

      /* update -U(i,k): ba[ili] = uik */
      ierr = PetscMemcpy(ba+ili*4,uik,4*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j): rtmp[bj[j]] += uik*ba[j]; */
          rtmp_ptr = rtmp + bj[j]*4;
          u = ba + j*4;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1];
          rtmp_ptr[1] += uik[2]*u[0] + uik[3]*u[1];
          rtmp_ptr[2] += uik[0]*u[2] + uik[1]*u[3];
          rtmp_ptr[3] += uik[2]*u[2] + uik[3]*u[3];
        }
      
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
    ierr = PetscMemcpy(diag,dk,4*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_2(diag);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*4;
         rtmp_ptr = rtmp + vj*4;        
         for (k1=0; k1<4; k1++){
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
  ierr = PetscFree(il);CHKERRQ(ierr); 
  ierr = PetscFree(dk);CHKERRQ(ierr);
  if (a->permute) {
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(1.3333*8*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 2 by 2 Using natural ordering
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat A,Mat *B)
{
  Mat            C = *B;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;

  PetscFunctionBegin;
  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing 
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows 
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list                        
     il(i): points to the first nonzero element in columns k,...,mbs-1 of 
            row i of U */
  ierr = PetscMalloc(4*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,4*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(PetscInt),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc(8*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + 4;     
   
  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];   
    ap = aa + jmin*4;
    for (j = jmin; j < jmax; j++){
      vj = aj[j];         /* block col. index */  
      rtmp_ptr = rtmp + vj*4;
      for (i=0; i<4; i++) *rtmp_ptr++ = *ap++;        
    } 
    
    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*4,4*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k): - ba[ili]*ba[i] */
      diag = ba + i*4;
      u    = ba + ili*4;
      uik[0] = -(diag[0]*u[0] + diag[2]*u[1]);
      uik[1] = -(diag[1]*u[0] + diag[3]*u[1]);
      uik[2] = -(diag[0]*u[2] + diag[2]*u[3]);
      uik[3] = -(diag[1]*u[2] + diag[3]*u[3]);
  
      /* update D(k) += -U(i,k)^T * U_bar(i,k): dk += uik*ba[ili] */
      dk[0] += uik[0]*u[0] + uik[1]*u[1];
      dk[1] += uik[2]*u[0] + uik[3]*u[1];
      dk[2] += uik[0]*u[2] + uik[1]*u[3];
      dk[3] += uik[2]*u[2] + uik[3]*u[3];

      /* update -U(i,k): ba[ili] = uik */
      ierr = PetscMemcpy(ba+ili*4,uik,4*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j): rtmp[bj[j]] += uik*ba[j]; */
          rtmp_ptr = rtmp + bj[j]*4;
          u = ba + j*4;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1];
          rtmp_ptr[1] += uik[2]*u[0] + uik[3]*u[1];
          rtmp_ptr[2] += uik[0]*u[2] + uik[1]*u[3];
          rtmp_ptr[3] += uik[2]*u[2] + uik[3]*u[3];
        }
      
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
    ierr = PetscMemcpy(diag,dk,4*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_2(diag);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*4;
         rtmp_ptr = rtmp + vj*4;        
         for (k1=0; k1<4; k1++){
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
  ierr = PetscFree(il);CHKERRQ(ierr);
  ierr = PetscFree(dk);CHKERRQ(ierr);

  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(1.3333*8*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
    Numeric U^T*D*U factorization for SBAIJ format. 
    Version for blocks are 1 by 1.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_1"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1(Mat A,Mat *B)
{
  Mat            C = *B;
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data,*b=(Mat_SeqSBAIJ *)C->data;
  IS             ip=b->row;
  PetscErrorCode ierr;
  PetscInt       *rip,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j,*bcol;
  PetscInt       *ai,*aj,*a2anew;
  PetscInt       k,jmin,jmax,*jl,*il,col,nexti,ili,nz;
  MatScalar      *rtmp,*ba=b->a,*bval,*aa,dk,uikdi;
  PetscReal      damping=b->factor_damping,zeropivot=b->factor_zeropivot,shift_amount;
  PetscTruth     damp,chshift;
  PetscInt       nshift=0,ndamp=0;

  PetscFunctionBegin;
  ierr  = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    nz = ai[mbs];
    ierr = PetscMalloc(nz*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    a2anew = a->a2anew;
    bval   = a->a;
    for (j=0; j<nz; j++){
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
  nz   = (2*mbs+1)*sizeof(PetscInt)+mbs*sizeof(MatScalar);
  ierr = PetscMalloc(nz,&il);CHKERRQ(ierr);
  jl   = il + mbs;
  rtmp = (MatScalar*)(jl + mbs);

  shift_amount = 0;
  do {
    damp = PETSC_FALSE;
    chshift = PETSC_FALSE;
    for (i=0; i<mbs; i++) {
      rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
    } 
 
    for (k = 0; k<mbs; k++){
      /*initialize k-th row by the perm[k]-th row of A */
      jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
      PetscScalar *baval = ba + bi[k];
      for (j = jmin; j < jmax; j++){
        col = rip[aj[j]];
        rtmp[col] = aa[j];
        *baval++  = 0.0; /* for in-place factorization */
      } 
      /* damp the diagonal of the matrix */
      if (ndamp||nshift) rtmp[k] += damping+shift_amount; 

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i = jl[k]; /* first row to be added to k_th row  */  

      while (i < k){
        nexti = jl[i]; /* next row to be added to k_th row */

        /* compute multiplier, update diag(k) and U(i,k) */
        ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */
        uikdi = - ba[ili]*ba[bi[i]];  /* diagonal(k) */ 
        dk += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row */
        jmin = ili + 1; jmax = bi[i+1];
        if (jmin < jmax){
          for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];         
          /* update il and jl for row i */
          il[i] = jmin;             
          j = bj[jmin]; jl[i] = jl[j]; jl[j] = i; 
        }      
        i = nexti;         
      }

      if (PetscRealPart(dk) < zeropivot && b->factor_shift){
	/* calculate a shift that would make this row diagonally dominant */
	PetscReal rs = PetscAbs(PetscRealPart(dk));
	jmin      = bi[k]+1; 
	nz        = bi[k+1] - jmin; 
	if (nz){
	  bcol = bj + jmin;
	  bval = ba + jmin;
	  while (nz--){
	    rs += PetscAbsScalar(rtmp[*bcol++]);
	  }
	}
	/* if this shift is less than the previous, just up the previous
	   one by a bit */
	shift_amount = PetscMax(rs,1.1*shift_amount);
	chshift  = PETSC_TRUE;
	/* Unlike in the ILU case there is no exit condition on nshift:
	   we increase the shift until it converges. There is no guarantee that
	   this algorithm converges faster or slower, or is better or worse
	   than the ILU algorithm. */
	nshift++;
	break;
      }
      if (PetscRealPart(dk) < zeropivot){
        if (damping == (PetscReal) PETSC_DECIDE) damping = -PetscRealPart(dk)/(k+1);
        if (damping > 0.0) {      
          if (ndamp) damping *= 2.0;    
          damp = PETSC_TRUE;
          ndamp++;
          break; 
        } else if (PetscAbsScalar(dk) < zeropivot){
          SETERRQ3(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot row %D value %g tolerance %g",k,PetscRealPart(dk),zeropivot);  
        } else {
          PetscLogInfo((PetscObject)A,"Negative pivot %g in row %D of Cholesky factorization\n",PetscRealPart(dk),k);
        }
      }

      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk; /* U(k,k) */
      jmin = bi[k]+1; jmax = bi[k+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++){
          col = bj[j]; ba[j] = rtmp[col]; rtmp[col] = 0.0;
        }       
        /* add the k-th row into il and jl */
        il[k] = jmin;
        i = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }        
    } 
  } while (damp||chshift);
  ierr = PetscFree(il);CHKERRQ(ierr);
  if (a->permute){ierr = PetscFree(aa);CHKERRQ(ierr);}

  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  C->factor       = FACTOR_CHOLESKY; 
  C->assembled    = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(C->m);
  if (ndamp) {
    PetscLogInfo(0,"MatCholeskyFactorNumerical_SeqSBAIJ_1: number of damping tries %D damping value %g\n",ndamp,damping);
  }
  if (nshift) {
    PetscLogInfo(0,"MatCholeskyFactorNumeric_SeqSBAIJ_1 diagonal shifted %D shifts\n",nshift);
  }
  PetscFunctionReturn(0);
}

/*
  Version for when blocks are 1 by 1 Using natural ordering
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat A,Mat *B)
{
  Mat            C = *B;
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data,*b=(Mat_SeqSBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs = a->mbs;
  PetscInt       *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscInt       k,jmin,*jl,*il,nexti,ili,*acol,*bcol,nz,ndamp = 0;
  MatScalar      *rtmp,*ba=b->a,*aa=a->a,dk,uikdi,*aval,*bval;
  PetscReal      damping=b->factor_damping, zeropivot=b->factor_zeropivot,shift_amount;
  PetscTruth     damp,chshift;
  PetscInt       nshift=0;

  PetscFunctionBegin;
  /* initialization */
  /* il and jl record the first nonzero element in each row of the accessing 
     window U(0:k, k:mbs-1).
     jl:    list of rows to be added to uneliminated rows 
            i>= k: jl(i) is the first row to be added to row i
            i<  k: jl(i) is the row following row i in some list of rows
            jl(i) = mbs indicates the end of a list                        
     il(i): points to the first nonzero element in U(i,k:mbs-1) 
  */
  ierr = PetscMalloc(mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc(2*mbs*sizeof(PetscInt),&il);CHKERRQ(ierr);
  jl   = il + mbs;

  shift_amount = 0;
  do {
    damp = PETSC_FALSE;
    chshift = PETSC_FALSE;
    for (i=0; i<mbs; i++) {
      rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
    }

    for (k = 0; k<mbs; k++){ 
    /*initialize k-th row with elements nonzero in row perm(k) of A */
      nz   = ai[k+1] - ai[k];
      acol = aj + ai[k];
      aval = aa + ai[k];
      bval = ba + bi[k];
      while (nz -- ){
        rtmp[*acol++] = *aval++;
        *bval++       = 0.0; /* for in-place factorization */
      } 
      /* damp the diagonal of the matrix */
      if (ndamp||nshift) rtmp[k] += damping+shift_amount; 
    
      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = jl[k]; /* first row to be added to k_th row  */  

      while (i < k){
        nexti = jl[i]; /* next row to be added to k_th row */
        /* compute multiplier, update D(k) and U(i,k) */
        ili   = il[i];  /* index of first nonzero element in U(i,k:bms-1) */
        uikdi = - ba[ili]*ba[bi[i]];  
        dk   += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row ... */
        jmin = ili + 1; 
        nz   = bi[i+1] - jmin;
        if (nz > 0){
          bcol = bj + jmin;
          bval = ba + jmin; 
          while (nz --) rtmp[*bcol++] += uikdi*(*bval++);
          /* update il and jl for i-th row */
          il[i] = jmin;            
          j = bj[jmin]; jl[i] = jl[j]; jl[j] = i; 
        }      
        i = nexti;         
      }

      if (PetscRealPart(dk) < zeropivot && b->factor_shift){
	/* calculate a shift that would make this row diagonally dominant */
	PetscReal rs = PetscAbs(PetscRealPart(dk));
	jmin      = bi[k]+1; 
	nz        = bi[k+1] - jmin; 
	if (nz){
	  bcol = bj + jmin;
	  bval = ba + jmin;
	  while (nz--){
	    rs += PetscAbsScalar(rtmp[*bcol++]);
	  }
	}
	/* if this shift is less than the previous, just up the previous
	   one by a bit */
	shift_amount = PetscMax(rs,1.1*shift_amount);
	chshift  = PETSC_TRUE;
	/* Unlike in the ILU case there is no exit condition on nshift:
	   we increase the shift until it converges. There is no guarantee that
	   this algorithm converges faster or slower, or is better or worse
	   than the ILU algorithm. */
	nshift++;
	break;
      }
      if (PetscRealPart(dk) < zeropivot){
        if (damping == (PetscReal) PETSC_DECIDE) damping = -PetscRealPart(dk)/(k+1);
        if (damping > 0.0) {      
          if (ndamp) damping *= 2.0;    
          damp = PETSC_TRUE;
          ndamp++;
          break; 
        } else if (PetscAbsScalar(dk) < zeropivot){
          SETERRQ3(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot row %D value %g tolerance %g",k,PetscRealPart(dk),zeropivot);  
        } else {
          PetscLogInfo((PetscObject)A,"Negative pivot %g in row %D of Cholesky factorization\n",PetscRealPart(dk),k);
        }
      }
      
      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk;
      jmin      = bi[k]+1; 
      nz        = bi[k+1] - jmin; 
      if (nz){
        bcol = bj + jmin;
        bval = ba + jmin;
        while (nz--){
          *bval++       = rtmp[*bcol]; 
          rtmp[*bcol++] = 0.0; 
        }       
        /* add k-th row into il and jl */
        il[k] = jmin;
        i = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }        
    } /* end of for (k = 0; k<mbs; k++) */
  } while (damp||chshift);
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr);
  
  C->factor       = FACTOR_CHOLESKY; 
  C->assembled    = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(C->m);
  if (ndamp) {
    PetscLogInfo(0,"MatCholeskyFactorNumerical_SeqSBAIJ_1_NaturalOrdering: number of damping tries %D damping value %g\n",ndamp,damping);
  }
  if (nshift) {
    PetscLogInfo(0,"MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering diagonal shifted %D shifts\n",nshift);
  }
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactor_SeqSBAIJ"
PetscErrorCode MatCholeskyFactor_SeqSBAIJ(Mat A,IS perm,MatFactorInfo *info)
{
  PetscErrorCode ierr;
  Mat            C;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorSymbolic(A,perm,info,&C);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(A,&C);CHKERRQ(ierr);
  ierr = MatHeaderCopy(A,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


