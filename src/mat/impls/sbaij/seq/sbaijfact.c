/*$Id: sbaijfact.c,v 1.61 2001/08/06 21:15:47 bsmith Exp $*/

#include "src/mat/impls/baij/seq/baij.h" 
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/vec/vecimpl.h"
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
int MatGetInertia_SeqSBAIJ(Mat F,int *nneig,int *nzero,int *npos)
{ 
  Mat_SeqSBAIJ *fact_ptr = (Mat_SeqSBAIJ*)F->data;
  PetscScalar  *dd = fact_ptr->a;
  int          m = F->m,i;

  PetscFunctionBegin;
  if (nneig){
    *nneig = 0;
    for (i=0; i<m; i++){
      if (PetscRealPart(dd[i]) < 0.0) (*nneig)++;
    }
  }
  if (nzero){
    *nzero = 0;
    for (i=0; i<m; i++){
      if (PetscRealPart(dd[i]) == 0.0) (*nzero)++;
    }
  }
  if (npos){
    *npos = 0;
    for (i=0; i<m; i++){
      if (PetscRealPart(dd[i]) > 0.0) (*npos)++;
    }
  }
  PetscFunctionReturn(0);
}
#endif /* !defined(PETSC_USE_COMPLEX) */

/* Using Modified Sparse Row (MSR) storage.
See page 85, "Iterative Methods ..." by Saad. */
/*
    Symbolic U^T*D*U factorization for SBAIJ format. Modified from SSF of YSMP.
*/
/* Use Modified Sparse Row storage for u and ju, see Saad pp.85 */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSBAIJ"
int MatCholeskyFactorSymbolic_SeqSBAIJ(Mat A,IS perm,PetscReal f,Mat *B)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data,*b;
  int          *rip,ierr,i,mbs = a->mbs,*ai,*aj;
  int          *jutmp,bs = a->bs,bs2=a->bs2;
  int          m,realloc = 0,prow;
  int          *jl,*q,jmin,jmax,juidx,nzk,qm,*iu,*ju,k,j,vj,umax,maxadd;
  int          *il,ili,nextprow;
  PetscTruth   perm_identity;

  PetscFunctionBegin;
  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);

  /* -- inplace factorization, i.e., use sbaij for *B -- */
  if (perm_identity && bs==1 ){
    if (!perm_identity) a->permute = PETSC_TRUE; 
 
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);   
  
  if (perm_identity){ /* without permutation */
    ai = a->i; aj = a->j;
  } else {            /* non-trivial permutation */    
    ierr = MatReorderingSeqSBAIJ(A,perm);CHKERRQ(ierr);   
    ai = a->inew; aj = a->jnew;
  }
  
  /* initialization */
  ierr  = PetscMalloc((mbs+1)*sizeof(int),&iu);CHKERRQ(ierr);
  umax  = (int)(f*ai[mbs] + 1); 
  ierr  = PetscMalloc(umax*sizeof(int),&ju);CHKERRQ(ierr);
  iu[0] = 0; 
  juidx = 0; /* index for ju */
  ierr  = PetscMalloc((3*mbs+1)*sizeof(int),&jl);CHKERRQ(ierr); /* linked list for getting pivot row */
  q     = jl + mbs;   /* linked list for col index of active row */
  il    = q  + mbs;
  for (i=0; i<mbs; i++){
    jl[i] = mbs; 
    q[i]  = 0;
    il[i] = 0;
  }

  /* for each row k */
  for (k=0; k<mbs; k++){   
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
          SETERRQ(1," error: duplicate entry in A\n"); 
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
      nextprow = jl[prow];
      
      /* merge row prow into k-th row */
      ili = il[prow];
      jmin = ili + 1;  /* points to 2nd nzero entry in U(prow,k:mbs-1) */
      jmax = iu[prow+1]; 
      qm = k;
      for (j=jmin; j<jmax; j++){
        vj = ju[j];
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        if (qm != vj){  /* a fill */
          nzk++; q[m] = vj; q[vj] = qm; qm = vj;
        }
      } /* end of for (j=jmin; j<jmax; j++) */
      if (jmin < jmax){
        il[prow] = jmin;
        j = ju[jmin];
        jl[prow] = jl[j]; jl[j] = prow;  /* update jl */
      } 
      prow = nextprow; 
    }  
   
    /* update il and jl */
    if (nzk > 0){
      i = q[k]; /* col value of the first nonzero element in U(k, k+1:mbs-1) */    
      jl[k] = jl[i]; jl[i] = k;
      il[k] = iu[k] + 1;
    } 
    iu[k+1] = iu[k] + nzk + 1;  /* include diag[k] */

    /* allocate more space to ju if needed */
    if (iu[k+1] > umax) {
      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      maxadd = umax;      
      if (maxadd < nzk) maxadd = (mbs-k)*(nzk+1)/2;
      umax += maxadd;

      /* allocate a longer ju */
      ierr = PetscMalloc(umax*sizeof(int),&jutmp);CHKERRQ(ierr);
      ierr = PetscMemcpy(jutmp,ju,iu[k]*sizeof(int));CHKERRQ(ierr);
      ierr = PetscFree(ju);CHKERRQ(ierr);       
      ju   = jutmp; 
      realloc++; /* count how many times we realloc */
    }

    /* save nonzero structure of k-th row in ju */
    ju[juidx++] = k; /* diag[k] */
    i = k;
    while (nzk --) {
      i           = q[i]; 
      ju[juidx++] = i;
    }      
  } 

  if (ai[mbs] != 0) {
    PetscReal af = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Reallocs %d Fill ratio:given %g needed %g\n",realloc,f,af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Run with -pc_cholesky_fill %g or use \n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:PCCholeskySetFill(pc,%g);\n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:for best performance.\n");
  } else {
     PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Empty matrix.\n");
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  /* ierr = PetscFree(q);CHKERRQ(ierr); */
  ierr = PetscFree(jl);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatCreateSeqSBAIJ(A->comm,bs,bs*mbs,bs*mbs,0,PETSC_NULL,B);CHKERRQ(ierr);
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
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatCholeskyInfo */
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  ierr    = PetscMalloc((bs*mbs+bs)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PetscLogObjectMemory(*B,(iu[mbs]-mbs)*(sizeof(int)+sizeof(MatScalar)));
  b->s_maxnz = b->s_nz = iu[mbs];
  
  (*B)->factor                 = FACTOR_CHOLESKY;
  (*B)->info.factor_mallocs    = realloc;
  (*B)->info.fill_ratio_given  = f;
  if (ai[mbs] != 0) {
    (*B)->info.fill_ratio_needed = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
  } else {
    (*B)->info.fill_ratio_needed = 0.0;
  }


  (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace;
  (*B)->ops->solve           = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  PetscLogInfo(A,"MatICCFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=1\n");
  
  PetscFunctionReturn(0); 
  }
  /* -----------  end of new code --------------------*/


  if (!perm_identity) a->permute = PETSC_TRUE; 
 
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);   
  
  if (perm_identity){ /* without permutation */
    ai = a->i; aj = a->j;
  } else {            /* non-trivial permutation */    
    ierr = MatReorderingSeqSBAIJ(A,perm);CHKERRQ(ierr);   
    ai = a->inew; aj = a->jnew;
  }
  
  /* initialization */
  ierr  = PetscMalloc((mbs+1)*sizeof(int),&iu);CHKERRQ(ierr);
  umax  = (int)(f*ai[mbs] + 1); umax += mbs + 1; 
  ierr  = PetscMalloc(umax*sizeof(int),&ju);CHKERRQ(ierr);
  iu[0] = mbs+1; 
  juidx = mbs + 1; /* index for ju */
  ierr  = PetscMalloc(2*mbs*sizeof(int),&jl);CHKERRQ(ierr); /* linked list for pivot row */
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
          SETERRQ(1," error: duplicate entry in A\n"); 
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
      ierr = PetscMalloc(umax*sizeof(int),&jutmp);CHKERRQ(ierr);
      ierr = PetscMemcpy(jutmp,ju,iu[k]*sizeof(int));CHKERRQ(ierr);
      ierr = PetscFree(ju);CHKERRQ(ierr);       
      ju   = jutmp; 
      realloc++; /* count how many times we realloc */
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
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Reallocs %d Fill ratio:given %g needed %g\n",realloc,f,af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Run with -pc_cholesky_fill %g or use \n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:PCCholeskySetFill(pc,%g);\n",af);
    PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:for best performance.\n");
  } else {
     PetscLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Empty matrix.\n");
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  /* ierr = PetscFree(q);CHKERRQ(ierr); */
  ierr = PetscFree(jl);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatCreateSeqSBAIJ(A->comm,bs,bs*mbs,bs*mbs,0,PETSC_NULL,B);CHKERRQ(ierr);
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
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatCholeskyInfo */
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr    = PetscMalloc((bs*mbs+bs)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PetscLogObjectMemory(*B,(iu[mbs]-mbs)*(sizeof(int)+sizeof(MatScalar)));
  b->s_maxnz = b->s_nz = iu[mbs];
  
  (*B)->factor                 = FACTOR_CHOLESKY;
  (*B)->info.factor_mallocs    = realloc;
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


#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_N"
int MatCholeskyFactorNumeric_SeqSBAIJ_N(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 perm = b->row;
  int                *perm_ptr,ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  int                bs=a->bs,bs2 = a->bs2;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar          *work;
  int                *pivots;

  PetscFunctionBegin;

  /* initialization */
  ierr = PetscMalloc(bs2*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc((2*bs2+bs)*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + bs2;
  work = uik + bs2;
  ierr = PetscMalloc(bs*sizeof(int),&pivots);CHKERRQ(ierr);
 
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);
  
  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew; 
    ierr = PetscMalloc(bs2*ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    ierr = PetscMemcpy(aa,a->a,bs2*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(int),&a2anew);CHKERRQ(ierr); 
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

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
    Kernel_A_gets_inverse_A(bs,diag,pivots,work);
    
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
int MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  int                bs=a->bs,bs2 = a->bs2;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar          *work;
  int                *pivots;

  PetscFunctionBegin;

  /* initialization */
  
  ierr = PetscMalloc(bs2*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc((2*bs2+bs)*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + bs2;
  work = uik + bs2;
  ierr = PetscMalloc(bs*sizeof(int),&pivots);CHKERRQ(ierr);
 
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
    Kernel_A_gets_inverse_A(bs,diag,pivots,work);
    
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
int MatCholeskyFactorNumeric_SeqSBAIJ_2(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 perm = b->row;
  int                *perm_ptr,ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;

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
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
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
    ierr = PetscMalloc(ai[mbs]*sizeof(int),&a2anew);CHKERRQ(ierr); 
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

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
int MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;

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
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
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
    Numeric U^T*D*U factorization for SBAIJ format. Modified from SNF of YSMP.
    Version for blocks are 1 by 1.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_1"
int MatCholeskyFactorNumeric_SeqSBAIJ_1(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 ip = b->row;
  int                *rip,ierr,i,j,mbs = a->mbs,*bi = b->i,*bj = b->j;
  int                *ai,*aj,*r;
  int                k,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *rtmp;
  MatScalar          *ba = b->a,*aa,ak;
  MatScalar          dk,uikdi;

  PetscFunctionBegin;
  ierr  = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    ierr = PetscMalloc(ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    ierr = PetscMemcpy(aa,a->a,ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(int),&r);CHKERRQ(ierr); 
    ierr= PetscMemcpy(r,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    jmin = ai[0]; jmax = ai[mbs];
    for (j=jmin; j<jmax; j++){
      while (r[j] != j){  
        k = r[j]; r[j] = r[k]; r[k] = k;     
        ak = aa[k]; aa[k] = aa[j]; aa[j] = ak;         
      }
    }
    ierr = PetscFree(r);CHKERRQ(ierr); 
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
  ierr = PetscMalloc(mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
  }

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
    
    for (j = jmin; j < jmax; j++){
      vj = rip[aj[j]];
      rtmp[vj] = aa[j];
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    dk = rtmp[k];
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier, update D(k) and U(i,k) */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */
      uikdi = - ba[ili]*ba[i];  
      dk += uikdi*ba[ili];
      ba[ili] = uikdi; /* -U(i,k) */

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];         
        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }      
      i = nexti;         
    }

    /* check for zero pivot and save diagoanl element */
    if (dk == 0.0){
      SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot");    
      /*
    } else if (PetscRealPart(dk) < 0.0){
      SETERRQ2(PETSC_ERR_MAT_LU_ZRPVT,"Negative pivot: d[%d] = %g\n",k,dk);  
      */
    }                                               

    /* save nonzero entries in k-th row of U ... */
    ba[k] = 1.0/dk;
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j]; ba[j] = rtmp[vj]; rtmp[vj] = 0.0;
      }       
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }        
  } 
  
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr);
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY; 
  C->assembled = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(b->mbs);
  PetscFunctionReturn(0);
}

/*
  Version for when blocks are 1 by 1 Using natural ordering
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs = a->mbs,*bi = b->i,*bj = b->j;
  int                *ai,*aj;
  int                k,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *rtmp,*ba = b->a,*aa,dk,uikdi;

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
  ierr = PetscMalloc(mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
  }

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[k]; jmax = ai[k+1];
    
    for (j = jmin; j < jmax; j++){
      vj = aj[j];
      rtmp[vj] = aa[j];
    } 
    
    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    dk = rtmp[k];
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier, update D(k) and U(i,k) */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */
      uikdi = - ba[ili]*ba[i];  
      dk += uikdi*ba[ili];
      ba[ili] = uikdi; /* -U(i,k) */

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];         
        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }      
      i = nexti;         
    }

    /* check for zero pivot and save diagoanl element */
    if (dk == 0.0){
      SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot");
      /*
    } else if (PetscRealPart(dk) < 0){
      SETERRQ2(PETSC_ERR_MAT_LU_ZRPVT,"Negative pivot: d[%d] = %g\n",k,dk);  
      */
    }                                               

    /* save nonzero entries in k-th row of U ... */
    ba[k] = 1.0/dk;
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j]; ba[j] = rtmp[vj]; rtmp[vj] = 0.0;
      }       
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }        
  } 
  
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr);
  
  C->factor    = FACTOR_CHOLESKY; 
  C->assembled = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(b->mbs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace"
int MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a=(Mat_SeqSBAIJ*)A->data,*b=(Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs = a->mbs;
  int                *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  int                k,jmin,*jl,*il,nexti,ili,*acol,*bcol,nz;
  MatScalar          *rtmp,*ba=b->a,*aa=a->a,dk,uikdi,*aval,*bval;

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
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl   = il + mbs;
  for (i=0; i<mbs; i++) {
    rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
  }

  /* for each row k */
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
    
    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    dk = rtmp[k];
    i  = jl[k]; /* first row to be added to k_th row  */  

    while (i < k){
      nexti = jl[i]; /* next row to be added to k_th row */
      /* printf("factnum, k %d, i %d\n", k,i); */

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
        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }      
      i = nexti;         
    }

    /* check for zero pivot and save diagoanl element */
    if (dk == 0.0){
      SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot");  
    } else if( dk < 0.0){
      PetscLogInfo((PetscObject)A,"Negative pivot %g in Cholesky factorization\n",1./dk);
    }

    /* save nonzero entries in k-th row of U ... */
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
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }        
  } 
  
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr);
  
  C->factor       = FACTOR_CHOLESKY; 
  C->assembled    = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  PetscLogFlops(b->mbs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactor_SeqSBAIJ"
int MatCholeskyFactor_SeqSBAIJ(Mat A,IS perm,PetscReal f)
{
  int ierr;
  Mat C;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorSymbolic(A,perm,f,&C);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(A,&C);CHKERRQ(ierr);
  ierr = MatHeaderCopy(A,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


