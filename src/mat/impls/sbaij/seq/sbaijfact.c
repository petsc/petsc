/* Using Modified Sparse Row (MSR) storage.
See page 85, "Iterative Methods ..." by Saad. */

/*$Id: sbaijfact.c,v 1.48 2000/11/07 17:58:29 hzhang Exp hzhang $*/
/*
    Symbolic U^T*D*U factorization for SBAIJ format. Modified from SSF of YSMP.
*/
#include "sbaij.h"
#include "src/mat/impls/baij/seq/baij.h" 
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"
#include "include/petscis.h"

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorSymbolic_SeqSBAIJ"
int MatCholeskyFactorSymbolic_SeqSBAIJ(Mat A,IS perm,PetscReal f,Mat *B)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data,*b;
  int         *rip,ierr,i,mbs = a->mbs,*ai,*aj;
  int         *jutmp,bs = a->bs,bs2=a->bs2;
  int         m,nzi,realloc = 0;
  int         *jl,*q,jumin,jmin,jmax,juptr,nzk,qm,*iu,*ju,k,j,vj,umax,maxadd;
  PetscTruth  perm_identity;

  PetscFunctionBegin;
  if (!perm) {
    ierr = ISCreateStride(PETSC_COMM_SELF,mbs,0,1,&perm);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(perm,IS_COOKIE);

  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) a->permute = PETSC_TRUE; 
 
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);   
  
  if (perm_identity){ /* without permutation */
    ai = a->i; aj = a->j;
  } else {            /* non-trivial permutation */    
    ierr = MatReorderingSeqSBAIJ(A, perm);CHKERRA(ierr);   
    ai = a->inew; aj = a->jnew;
  }
  
  /* initialization */
  /* Don't know how many column pointers are needed so estimate. 
     Use Modified Sparse Row storage for u and ju, see Sasd pp.85 */
  iu   = (int*)PetscMalloc((mbs+1)*sizeof(int));CHKPTRQ(iu);
  umax = (int)(f*ai[mbs] + 1); umax += mbs + 1; 
  ju   = (int*)PetscMalloc(umax*sizeof(int));CHKPTRQ(ju);
  iu[0] = mbs+1; 
  juptr = mbs;
  jl =  (int*)PetscMalloc(mbs*sizeof(int));CHKPTRQ(jl);
  q  =  (int*)PetscMalloc(mbs*sizeof(int));CHKPTRQ(q);
  for (i=0; i<mbs; i++){
    jl[i] = mbs; q[i] = 0;
  }

  /* for each row k */
  for (k=0; k<mbs; k++){   
    nzk = 0; /* num. of nz blocks in k-th block row with diagonal block excluded */
    q[k] = mbs;
    /* initialize nonzero structure of k-th row to row rip[k] of A */
    jmin = ai[rip[k]];
    jmax = ai[rip[k]+1];
    for (j=jmin; j<jmax; j++){
      vj = rip[aj[j]]; /* col. value */
      if(vj > k){
        qm = k; 
        do {
          m  = qm; qm = q[m];
        } while(qm < vj);
        if (qm == vj) {
          printf(" error: duplicate entry in A\n"); break;
        }     
        nzk++;
        q[m] = vj;
        q[vj] = qm;  
      } /* if(vj > k) */
    } /* for (j=jmin; j<jmax; j++) */

    /* modify nonzero structure of k-th row by computing fill-in
       for each row i to be merged in */
    i = k; 
    i = jl[i]; /* next pivot row (== mbs for symbolic factorization) */
    /* printf(" next pivot row i=%d\n",i); */
    while (i < mbs){
      /* merge row i into k-th row */
      nzi = iu[i+1] - (iu[i]+1);
      jmin = iu[i] + 1; jmax = iu[i] + nzi;
      qm = k;
      for (j=jmin; j<jmax+1; j++){
        vj = ju[j];
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        if (qm != vj){
         nzk++; q[m] = vj; q[vj] = qm; qm = vj;
        }
      } 
      i = jl[i]; /* next pivot row */     
    }  
   
    /* add k to row list for first nonzero element in k-th row */
    if (nzk > 0){
      i = q[k]; /* col value of first nonzero element in U(k, k+1:mbs-1) */    
      jl[k] = jl[i]; jl[i] = k;
    } 
    iu[k+1] = iu[k] + nzk;   /* printf(" iu[%d]=%d, umax=%d\n", k+1, iu[k+1],umax);*/

    /* allocate more space to ju if needed */
    if (iu[k+1] > umax) { printf("allocate more space, iu[%d]=%d > umax=%d\n",k+1, iu[k+1],umax);
      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      maxadd = umax;      
      if (maxadd < nzk) maxadd = (mbs-k)*(nzk+1)/2;
      umax += maxadd;

      /* allocate a longer ju */
      jutmp = (int*)PetscMalloc(umax*sizeof(int));CHKPTRQ(jutmp);
      ierr  = PetscMemcpy(jutmp,ju,iu[k]*sizeof(int));CHKERRQ(ierr);
      ierr  = PetscFree(ju);CHKERRQ(ierr);       
      ju    = jutmp; 
      realloc++; /* count how many times we realloc */
    }

    /* save nonzero structure of k-th row in ju */
    i=k;
    jumin = juptr + 1; juptr += nzk; 
    for (j=jumin; j<juptr+1; j++){
      i=q[i];
      ju[j]=i;
    }     
  } 

  if (ai[mbs] != 0) {
    PetscReal af = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
    PLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Reallocs %d Fill ratio:given %g needed %g\n",realloc,f,af);
    PLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Run with -pc_lu_fill %g or use \n",af);
    PLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:PCLUSetFill(pc,%g);\n",af);
    PLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:for best performance.\n");
  } else {
     PLogInfo(A,"MatCholeskyFactorSymbolic_SeqSBAIJ:Empty matrix.\n");
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree(q);CHKERRQ(ierr);
  ierr = PetscFree(jl);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatCreateSeqSBAIJ(A->comm,bs,bs*mbs,bs*mbs,0,PETSC_NULL,B);CHKERRQ(ierr);
  /* PLogObjectParent(*B,iperm); */
  b = (Mat_SeqSBAIJ*)(*B)->data;
  ierr = PetscFree(b->imax);CHKERRQ(ierr);
  b->singlemalloc = PETSC_FALSE;
  /* the next line frees the default space generated by the Create() */
  ierr = PetscFree(b->a);CHKERRQ(ierr);
  ierr = PetscFree(b->ilen);CHKERRQ(ierr);
  b->a          = (MatScalar*)PetscMalloc((iu[mbs]+1)*sizeof(MatScalar)*bs2);CHKPTRQ(b->a);
  b->j          = ju;
  b->i          = iu;
  b->diag       = 0;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = perm;
  ierr          = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol       = perm;
  ierr          = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  b->solve_work = (Scalar*)PetscMalloc((bs*mbs+bs)*sizeof(Scalar));CHKPTRQ(b->solve_work);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PLogObjectMemory(*B,(iu[mbs]-mbs)*(sizeof(int)+sizeof(MatScalar)));
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
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=1\n");
        break;
      case 2:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_2_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=2\n");
        break;
      case 3:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_3_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:sing special in-place natural ordering factor and solve BS=3\n");
        break; 
      case 4:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_4_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=4\n"); 
        break;
      case 5:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_5_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=5\n"); 
        break;
      case 6: 
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_6_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=6\n");
        break; 
      case 7:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering;
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_7_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS=7\n");
      break; 
      default:
        (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering; 
        (*B)->ops->solve           = MatSolve_SeqSBAIJ_N_NaturalOrdering;
        PLogInfo(A,"MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ:Using special in-place natural ordering factor and solve BS>7\n");
      break; 
    }
  } 

  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_N"
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
  rtmp  = (MatScalar*)PetscMalloc(bs2*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr  = PetscMemzero(rtmp,bs2*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il    = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl    = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc((2*bs2+bs)*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + bs2;
  work  = uik + bs2;
  pivots= (int*)PetscMalloc(bs*sizeof(int));CHKPTRQ(pivots);
 
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);
  
  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(bs2*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,bs2*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

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
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*bs2;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        rtmp_ptr = rtmp + vj*bs2;
        for (i=0; i<bs2; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
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
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PLogFlops(1.3333*bs*bs2*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  int                bs=a->bs,bs2 = a->bs2;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;
  MatScalar          *work;
  int                *pivots;

  PetscFunctionBegin;

  /* initialization */
  
  rtmp  = (MatScalar*)PetscMalloc(bs2*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr  = PetscMemzero(rtmp,bs2*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il    = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl    = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc((2*bs2+bs)*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + bs2;
  work  = uik + bs2;
  pivots= (int*)PetscMalloc(bs*sizeof(int));CHKPTRQ(pivots);
 
  ai = a->i; aj = a->j; aa = a->a;
   
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*bs2;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        rtmp_ptr = rtmp + vj*bs2;
        for (i=0; i<bs2; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
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
  PLogFlops(1.3333*bs*bs2*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* Version for when blocks are 7 by 7 */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_7"
int MatCholeskyFactorNumeric_SeqSBAIJ_7(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 perm = b->row;
  int                *perm_ptr,ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*w,*wp;

  PetscFunctionBegin;
  /* initialization */
  w  = (MatScalar*)PetscMalloc(49*mbs*sizeof(MatScalar));CHKPTRQ(w);
  ierr = PetscMemzero(w,49*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(98*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 49;    
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(49*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,49*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<49; k1++){
            dk[k1]       = aa[k*49+k1]; 
            aa[k*49+k1] = aa[j*49+k1]; 
            aa[j*49+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*49;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<49; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<7; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<7; k1++) *ap++ = dk[k + 7*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*49;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        wp = w + vj*49;
        for (i=0; i<49; i++) *wp++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,w+k*49,49*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*49;
      u    = ba + ili*49;

      uik[0] = -(d[0]*u[0] + d[7]*u[1]+ d[14]*u[2]+ d[21]*u[3]+ d[28]*u[4]+ d[35]*u[5]+ d[42]*u[6]);
      uik[1] = -(d[1]*u[0] + d[8]*u[1]+ d[15]*u[2]+ d[22]*u[3]+ d[29]*u[4]+ d[36]*u[5]+ d[43]*u[6]);
      uik[2] = -(d[2]*u[0] + d[9]*u[1]+ d[16]*u[2]+ d[23]*u[3]+ d[30]*u[4]+ d[37]*u[5]+ d[44]*u[6]);
      uik[3] = -(d[3]*u[0]+ d[10]*u[1]+ d[17]*u[2]+ d[24]*u[3]+ d[31]*u[4]+ d[38]*u[5]+ d[45]*u[6]);
      uik[4] = -(d[4]*u[0]+ d[11]*u[1]+ d[18]*u[2]+ d[25]*u[3]+ d[32]*u[4]+ d[39]*u[5]+ d[46]*u[6]);
      uik[5] = -(d[5]*u[0]+ d[12]*u[1]+ d[19]*u[2]+ d[26]*u[3]+ d[33]*u[4]+ d[40]*u[5]+ d[47]*u[6]);
      uik[6] = -(d[6]*u[0]+ d[13]*u[1]+ d[20]*u[2]+ d[27]*u[3]+ d[34]*u[4]+ d[41]*u[5]+ d[48]*u[6]);

      uik[7] = -(d[0]*u[7] + d[7]*u[8]+ d[14]*u[9]+ d[21]*u[10]+ d[28]*u[11]+ d[35]*u[12]+ d[42]*u[13]);
      uik[8] = -(d[1]*u[7] + d[8]*u[8]+ d[15]*u[9]+ d[22]*u[10]+ d[29]*u[11]+ d[36]*u[12]+ d[43]*u[13]);
      uik[9] = -(d[2]*u[7] + d[9]*u[8]+ d[16]*u[9]+ d[23]*u[10]+ d[30]*u[11]+ d[37]*u[12]+ d[44]*u[13]);
      uik[10]= -(d[3]*u[7]+ d[10]*u[8]+ d[17]*u[9]+ d[24]*u[10]+ d[31]*u[11]+ d[38]*u[12]+ d[45]*u[13]);
      uik[11]= -(d[4]*u[7]+ d[11]*u[8]+ d[18]*u[9]+ d[25]*u[10]+ d[32]*u[11]+ d[39]*u[12]+ d[46]*u[13]);
      uik[12]= -(d[5]*u[7]+ d[12]*u[8]+ d[19]*u[9]+ d[26]*u[10]+ d[33]*u[11]+ d[40]*u[12]+ d[47]*u[13]);
      uik[13]= -(d[6]*u[7]+ d[13]*u[8]+ d[20]*u[9]+ d[27]*u[10]+ d[34]*u[11]+ d[41]*u[12]+ d[48]*u[13]);

      uik[14]= -(d[0]*u[14] + d[7]*u[15]+ d[14]*u[16]+ d[21]*u[17]+ d[28]*u[18]+ d[35]*u[19]+ d[42]*u[20]);
      uik[15]= -(d[1]*u[14] + d[8]*u[15]+ d[15]*u[16]+ d[22]*u[17]+ d[29]*u[18]+ d[36]*u[19]+ d[43]*u[20]);
      uik[16]= -(d[2]*u[14] + d[9]*u[15]+ d[16]*u[16]+ d[23]*u[17]+ d[30]*u[18]+ d[37]*u[19]+ d[44]*u[20]);
      uik[17]= -(d[3]*u[14]+ d[10]*u[15]+ d[17]*u[16]+ d[24]*u[17]+ d[31]*u[18]+ d[38]*u[19]+ d[45]*u[20]);
      uik[18]= -(d[4]*u[14]+ d[11]*u[15]+ d[18]*u[16]+ d[25]*u[17]+ d[32]*u[18]+ d[39]*u[19]+ d[46]*u[20]);
      uik[19]= -(d[5]*u[14]+ d[12]*u[15]+ d[19]*u[16]+ d[26]*u[17]+ d[33]*u[18]+ d[40]*u[19]+ d[47]*u[20]);
      uik[20]= -(d[6]*u[14]+ d[13]*u[15]+ d[20]*u[16]+ d[27]*u[17]+ d[34]*u[18]+ d[41]*u[19]+ d[48]*u[20]);

      uik[21]= -(d[0]*u[21] + d[7]*u[22]+ d[14]*u[23]+ d[21]*u[24]+ d[28]*u[25]+ d[35]*u[26]+ d[42]*u[27]);
      uik[22]= -(d[1]*u[21] + d[8]*u[22]+ d[15]*u[23]+ d[22]*u[24]+ d[29]*u[25]+ d[36]*u[26]+ d[43]*u[27]);
      uik[23]= -(d[2]*u[21] + d[9]*u[22]+ d[16]*u[23]+ d[23]*u[24]+ d[30]*u[25]+ d[37]*u[26]+ d[44]*u[27]);
      uik[24]= -(d[3]*u[21]+ d[10]*u[22]+ d[17]*u[23]+ d[24]*u[24]+ d[31]*u[25]+ d[38]*u[26]+ d[45]*u[27]);
      uik[25]= -(d[4]*u[21]+ d[11]*u[22]+ d[18]*u[23]+ d[25]*u[24]+ d[32]*u[25]+ d[39]*u[26]+ d[46]*u[27]);
      uik[26]= -(d[5]*u[21]+ d[12]*u[22]+ d[19]*u[23]+ d[26]*u[24]+ d[33]*u[25]+ d[40]*u[26]+ d[47]*u[27]);
      uik[27]= -(d[6]*u[21]+ d[13]*u[22]+ d[20]*u[23]+ d[27]*u[24]+ d[34]*u[25]+ d[41]*u[26]+ d[48]*u[27]);

      uik[28]= -(d[0]*u[28] + d[7]*u[29]+ d[14]*u[30]+ d[21]*u[31]+ d[28]*u[32]+ d[35]*u[33]+ d[42]*u[34]);
      uik[29]= -(d[1]*u[28] + d[8]*u[29]+ d[15]*u[30]+ d[22]*u[31]+ d[29]*u[32]+ d[36]*u[33]+ d[43]*u[34]);
      uik[30]= -(d[2]*u[28] + d[9]*u[29]+ d[16]*u[30]+ d[23]*u[31]+ d[30]*u[32]+ d[37]*u[33]+ d[44]*u[34]);
      uik[31]= -(d[3]*u[28]+ d[10]*u[29]+ d[17]*u[30]+ d[24]*u[31]+ d[31]*u[32]+ d[38]*u[33]+ d[45]*u[34]);
      uik[32]= -(d[4]*u[28]+ d[11]*u[29]+ d[18]*u[30]+ d[25]*u[31]+ d[32]*u[32]+ d[39]*u[33]+ d[46]*u[34]);
      uik[33]= -(d[5]*u[28]+ d[12]*u[29]+ d[19]*u[30]+ d[26]*u[31]+ d[33]*u[32]+ d[40]*u[33]+ d[47]*u[34]);
      uik[34]= -(d[6]*u[28]+ d[13]*u[29]+ d[20]*u[30]+ d[27]*u[31]+ d[34]*u[32]+ d[41]*u[33]+ d[48]*u[34]);

      uik[35]= -(d[0]*u[35] + d[7]*u[36]+ d[14]*u[37]+ d[21]*u[38]+ d[28]*u[39]+ d[35]*u[40]+ d[42]*u[41]);
      uik[36]= -(d[1]*u[35] + d[8]*u[36]+ d[15]*u[37]+ d[22]*u[38]+ d[29]*u[39]+ d[36]*u[40]+ d[43]*u[41]);
      uik[37]= -(d[2]*u[35] + d[9]*u[36]+ d[16]*u[37]+ d[23]*u[38]+ d[30]*u[39]+ d[37]*u[40]+ d[44]*u[41]);
      uik[38]= -(d[3]*u[35]+ d[10]*u[36]+ d[17]*u[37]+ d[24]*u[38]+ d[31]*u[39]+ d[38]*u[40]+ d[45]*u[41]);
      uik[39]= -(d[4]*u[35]+ d[11]*u[36]+ d[18]*u[37]+ d[25]*u[38]+ d[32]*u[39]+ d[39]*u[40]+ d[46]*u[41]);
      uik[40]= -(d[5]*u[35]+ d[12]*u[36]+ d[19]*u[37]+ d[26]*u[38]+ d[33]*u[39]+ d[40]*u[40]+ d[47]*u[41]);
      uik[41]= -(d[6]*u[35]+ d[13]*u[36]+ d[20]*u[37]+ d[27]*u[38]+ d[34]*u[39]+ d[41]*u[40]+ d[48]*u[41]);

      uik[42]= -(d[0]*u[42] + d[7]*u[43]+ d[14]*u[44]+ d[21]*u[45]+ d[28]*u[46]+ d[35]*u[47]+ d[42]*u[48]);
      uik[43]= -(d[1]*u[42] + d[8]*u[43]+ d[15]*u[44]+ d[22]*u[45]+ d[29]*u[46]+ d[36]*u[47]+ d[43]*u[48]);
      uik[44]= -(d[2]*u[42] + d[9]*u[43]+ d[16]*u[44]+ d[23]*u[45]+ d[30]*u[46]+ d[37]*u[47]+ d[44]*u[48]);
      uik[45]= -(d[3]*u[42]+ d[10]*u[43]+ d[17]*u[44]+ d[24]*u[45]+ d[31]*u[46]+ d[38]*u[47]+ d[45]*u[48]);
      uik[46]= -(d[4]*u[42]+ d[11]*u[43]+ d[18]*u[44]+ d[25]*u[45]+ d[32]*u[46]+ d[39]*u[47]+ d[46]*u[48]);
      uik[47]= -(d[5]*u[42]+ d[12]*u[43]+ d[19]*u[44]+ d[26]*u[45]+ d[33]*u[46]+ d[40]*u[47]+ d[47]*u[48]);
      uik[48]= -(d[6]*u[42]+ d[13]*u[43]+ d[20]*u[44]+ d[27]*u[45]+ d[34]*u[46]+ d[41]*u[47]+ d[48]*u[48]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0]+=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6];
      dk[1]+=  uik[7]*u[0] + uik[8]*u[1] + uik[9]*u[2]+ uik[10]*u[3]+ uik[11]*u[4]+ uik[12]*u[5]+ uik[13]*u[6];
      dk[2]+= uik[14]*u[0]+ uik[15]*u[1]+ uik[16]*u[2]+ uik[17]*u[3]+ uik[18]*u[4]+ uik[19]*u[5]+ uik[20]*u[6];
      dk[3]+= uik[21]*u[0]+ uik[22]*u[1]+ uik[23]*u[2]+ uik[24]*u[3]+ uik[25]*u[4]+ uik[26]*u[5]+ uik[27]*u[6];
      dk[4]+= uik[28]*u[0]+ uik[29]*u[1]+ uik[30]*u[2]+ uik[31]*u[3]+ uik[32]*u[4]+ uik[33]*u[5]+ uik[34]*u[6];
      dk[5]+= uik[35]*u[0]+ uik[36]*u[1]+ uik[37]*u[2]+ uik[38]*u[3]+ uik[39]*u[4]+ uik[40]*u[5]+ uik[41]*u[6];
      dk[6]+= uik[42]*u[0]+ uik[43]*u[1]+ uik[44]*u[2]+ uik[45]*u[3]+ uik[46]*u[4]+ uik[47]*u[5]+ uik[48]*u[6];

      dk[7]+=  uik[0]*u[7] + uik[1]*u[8] + uik[2]*u[9] + uik[3]*u[10] + uik[4]*u[11] + uik[5]*u[12] + uik[6]*u[13];
      dk[8]+=  uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13];
      dk[9]+= uik[14]*u[7]+ uik[15]*u[8]+ uik[16]*u[9]+ uik[17]*u[10]+ uik[18]*u[11]+ uik[19]*u[12]+ uik[20]*u[13];
      dk[10]+=uik[21]*u[7]+ uik[22]*u[8]+ uik[23]*u[9]+ uik[24]*u[10]+ uik[25]*u[11]+ uik[26]*u[12]+ uik[27]*u[13];
      dk[11]+=uik[28]*u[7]+ uik[29]*u[8]+ uik[30]*u[9]+ uik[31]*u[10]+ uik[32]*u[11]+ uik[33]*u[12]+ uik[34]*u[13];
      dk[12]+=uik[35]*u[7]+ uik[36]*u[8]+ uik[37]*u[9]+ uik[38]*u[10]+ uik[39]*u[11]+ uik[40]*u[12]+ uik[41]*u[13];
      dk[13]+=uik[42]*u[7]+ uik[43]*u[8]+ uik[44]*u[9]+ uik[45]*u[10]+ uik[46]*u[11]+ uik[47]*u[12]+ uik[48]*u[13];

      dk[14]+=  uik[0]*u[14] + uik[1]*u[15] + uik[2]*u[16] + uik[3]*u[17] + uik[4]*u[18] + uik[5]*u[19] + uik[6]*u[20];
      dk[15]+=  uik[7]*u[14] + uik[8]*u[15] + uik[9]*u[16]+ uik[10]*u[17]+ uik[11]*u[18]+ uik[12]*u[19]+ uik[13]*u[20];
      dk[16]+= uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20];
      dk[17]+= uik[21]*u[14]+ uik[22]*u[15]+ uik[23]*u[16]+ uik[24]*u[17]+ uik[25]*u[18]+ uik[26]*u[19]+ uik[27]*u[20];
      dk[18]+= uik[28]*u[14]+ uik[29]*u[15]+ uik[30]*u[16]+ uik[31]*u[17]+ uik[32]*u[18]+ uik[33]*u[19]+ uik[34]*u[20];
      dk[19]+= uik[35]*u[14]+ uik[36]*u[15]+ uik[37]*u[16]+ uik[38]*u[17]+ uik[39]*u[18]+ uik[40]*u[19]+ uik[41]*u[20];
      dk[20]+= uik[42]*u[14]+ uik[43]*u[15]+ uik[44]*u[16]+ uik[45]*u[17]+ uik[46]*u[18]+ uik[47]*u[19]+ uik[48]*u[20];

      dk[21]+=  uik[0]*u[21] + uik[1]*u[22] + uik[2]*u[23] + uik[3]*u[24] + uik[4]*u[25] + uik[5]*u[26] + uik[6]*u[27];
      dk[22]+=  uik[7]*u[21] + uik[8]*u[22] + uik[9]*u[23]+ uik[10]*u[24]+ uik[11]*u[25]+ uik[12]*u[26]+ uik[13]*u[27];
      dk[23]+= uik[14]*u[21]+ uik[15]*u[22]+ uik[16]*u[23]+ uik[17]*u[24]+ uik[18]*u[25]+ uik[19]*u[26]+ uik[20]*u[27];
      dk[24]+= uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27];
      dk[25]+= uik[28]*u[21]+ uik[29]*u[22]+ uik[30]*u[23]+ uik[31]*u[24]+ uik[32]*u[25]+ uik[33]*u[26]+ uik[34]*u[27];
      dk[26]+= uik[35]*u[21]+ uik[36]*u[22]+ uik[37]*u[23]+ uik[38]*u[24]+ uik[39]*u[25]+ uik[40]*u[26]+ uik[41]*u[27];
      dk[27]+= uik[42]*u[21]+ uik[43]*u[22]+ uik[44]*u[23]+ uik[45]*u[24]+ uik[46]*u[25]+ uik[47]*u[26]+ uik[48]*u[27];

      dk[28]+=  uik[0]*u[28] + uik[1]*u[29] + uik[2]*u[30] + uik[3]*u[31] + uik[4]*u[32] + uik[5]*u[33] + uik[6]*u[34];
      dk[29]+=  uik[7]*u[28] + uik[8]*u[29] + uik[9]*u[30]+ uik[10]*u[31]+ uik[11]*u[32]+ uik[12]*u[33]+ uik[13]*u[34];
      dk[30]+= uik[14]*u[28]+ uik[15]*u[29]+ uik[16]*u[30]+ uik[17]*u[31]+ uik[18]*u[32]+ uik[19]*u[33]+ uik[20]*u[34];
      dk[31]+= uik[21]*u[28]+ uik[22]*u[29]+ uik[23]*u[30]+ uik[24]*u[31]+ uik[25]*u[32]+ uik[26]*u[33]+ uik[27]*u[34];
      dk[32]+= uik[28]*u[28]+ uik[29]*u[29]+ uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34];
      dk[33]+= uik[35]*u[28]+ uik[36]*u[29]+ uik[37]*u[30]+ uik[38]*u[31]+ uik[39]*u[32]+ uik[40]*u[33]+ uik[41]*u[34];
      dk[34]+= uik[42]*u[28]+ uik[43]*u[29]+ uik[44]*u[30]+ uik[45]*u[31]+ uik[46]*u[32]+ uik[47]*u[33]+ uik[48]*u[34];

      dk[35]+=  uik[0]*u[35] + uik[1]*u[36] + uik[2]*u[37] + uik[3]*u[38] + uik[4]*u[39] + uik[5]*u[40] + uik[6]*u[41];
      dk[36]+=  uik[7]*u[35] + uik[8]*u[36] + uik[9]*u[37]+ uik[10]*u[38]+ uik[11]*u[39]+ uik[12]*u[40]+ uik[13]*u[41];
      dk[37]+= uik[14]*u[35]+ uik[15]*u[36]+ uik[16]*u[37]+ uik[17]*u[38]+ uik[18]*u[39]+ uik[19]*u[40]+ uik[20]*u[41];
      dk[38]+= uik[21]*u[35]+ uik[22]*u[36]+ uik[23]*u[37]+ uik[24]*u[38]+ uik[25]*u[39]+ uik[26]*u[40]+ uik[27]*u[41];
      dk[39]+= uik[28]*u[35]+ uik[29]*u[36]+ uik[30]*u[37]+ uik[31]*u[38]+ uik[32]*u[39]+ uik[33]*u[40]+ uik[34]*u[41];
      dk[40]+= uik[35]*u[35]+ uik[36]*u[36]+ uik[37]*u[37]+ uik[38]*u[38]+ uik[39]*u[39]+ uik[40]*u[40]+ uik[41]*u[41];
      dk[41]+= uik[42]*u[35]+ uik[43]*u[36]+ uik[44]*u[37]+ uik[45]*u[38]+ uik[46]*u[39]+ uik[47]*u[40]+ uik[48]*u[41];

      dk[42]+=  uik[0]*u[42] + uik[1]*u[43] + uik[2]*u[44] + uik[3]*u[45] + uik[4]*u[46] + uik[5]*u[47] + uik[6]*u[48];
      dk[43]+=  uik[7]*u[42] + uik[8]*u[43] + uik[9]*u[44]+ uik[10]*u[45]+ uik[11]*u[46]+ uik[12]*u[47]+ uik[13]*u[48];
      dk[44]+= uik[14]*u[42]+ uik[15]*u[43]+ uik[16]*u[44]+ uik[17]*u[45]+ uik[18]*u[46]+ uik[19]*u[47]+ uik[20]*u[48];
      dk[45]+= uik[21]*u[42]+ uik[22]*u[43]+ uik[23]*u[44]+ uik[24]*u[45]+ uik[25]*u[46]+ uik[26]*u[47]+ uik[27]*u[48];
      dk[46]+= uik[28]*u[42]+ uik[29]*u[43]+ uik[30]*u[44]+ uik[31]*u[45]+ uik[32]*u[46]+ uik[33]*u[47]+ uik[34]*u[48];
      dk[47]+= uik[35]*u[42]+ uik[36]*u[43]+ uik[37]*u[44]+ uik[38]*u[45]+ uik[39]*u[46]+ uik[40]*u[47]+ uik[41]*u[48];
      dk[48]+= uik[42]*u[42]+ uik[43]*u[43]+ uik[44]*u[44]+ uik[45]*u[45]+ uik[46]*u[46]+ uik[47]*u[47]+ uik[48]*u[48];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*49,uik,49*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*49;
          u = ba + j*49;

          wp[0]+=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6];
          wp[1]+=  uik[7]*u[0] + uik[8]*u[1] + uik[9]*u[2]+ uik[10]*u[3]+ uik[11]*u[4]+ uik[12]*u[5]+ uik[13]*u[6];
          wp[2]+= uik[14]*u[0]+ uik[15]*u[1]+ uik[16]*u[2]+ uik[17]*u[3]+ uik[18]*u[4]+ uik[19]*u[5]+ uik[20]*u[6];
          wp[3]+= uik[21]*u[0]+ uik[22]*u[1]+ uik[23]*u[2]+ uik[24]*u[3]+ uik[25]*u[4]+ uik[26]*u[5]+ uik[27]*u[6];
          wp[4]+= uik[28]*u[0]+ uik[29]*u[1]+ uik[30]*u[2]+ uik[31]*u[3]+ uik[32]*u[4]+ uik[33]*u[5]+ uik[34]*u[6];
          wp[5]+= uik[35]*u[0]+ uik[36]*u[1]+ uik[37]*u[2]+ uik[38]*u[3]+ uik[39]*u[4]+ uik[40]*u[5]+ uik[41]*u[6];
          wp[6]+= uik[42]*u[0]+ uik[43]*u[1]+ uik[44]*u[2]+ uik[45]*u[3]+ uik[46]*u[4]+ uik[47]*u[5]+ uik[48]*u[6];

          wp[7]+=  uik[0]*u[7] + uik[1]*u[8] + uik[2]*u[9] + uik[3]*u[10] + uik[4]*u[11] + uik[5]*u[12] + uik[6]*u[13];
          wp[8]+=  uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13];
          wp[9]+= uik[14]*u[7]+ uik[15]*u[8]+ uik[16]*u[9]+ uik[17]*u[10]+ uik[18]*u[11]+ uik[19]*u[12]+ uik[20]*u[13];
          wp[10]+=uik[21]*u[7]+ uik[22]*u[8]+ uik[23]*u[9]+ uik[24]*u[10]+ uik[25]*u[11]+ uik[26]*u[12]+ uik[27]*u[13];
          wp[11]+=uik[28]*u[7]+ uik[29]*u[8]+ uik[30]*u[9]+ uik[31]*u[10]+ uik[32]*u[11]+ uik[33]*u[12]+ uik[34]*u[13];
          wp[12]+=uik[35]*u[7]+ uik[36]*u[8]+ uik[37]*u[9]+ uik[38]*u[10]+ uik[39]*u[11]+ uik[40]*u[12]+ uik[41]*u[13];
          wp[13]+=uik[42]*u[7]+ uik[43]*u[8]+ uik[44]*u[9]+ uik[45]*u[10]+ uik[46]*u[11]+ uik[47]*u[12]+ uik[48]*u[13];

          wp[14]+=  uik[0]*u[14] + uik[1]*u[15] + uik[2]*u[16] + uik[3]*u[17] + uik[4]*u[18] + uik[5]*u[19] + uik[6]*u[20];
          wp[15]+=  uik[7]*u[14] + uik[8]*u[15] + uik[9]*u[16]+ uik[10]*u[17]+ uik[11]*u[18]+ uik[12]*u[19]+ uik[13]*u[20];
          wp[16]+= uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20];
          wp[17]+= uik[21]*u[14]+ uik[22]*u[15]+ uik[23]*u[16]+ uik[24]*u[17]+ uik[25]*u[18]+ uik[26]*u[19]+ uik[27]*u[20];
          wp[18]+= uik[28]*u[14]+ uik[29]*u[15]+ uik[30]*u[16]+ uik[31]*u[17]+ uik[32]*u[18]+ uik[33]*u[19]+ uik[34]*u[20];
          wp[19]+= uik[35]*u[14]+ uik[36]*u[15]+ uik[37]*u[16]+ uik[38]*u[17]+ uik[39]*u[18]+ uik[40]*u[19]+ uik[41]*u[20];
          wp[20]+= uik[42]*u[14]+ uik[43]*u[15]+ uik[44]*u[16]+ uik[45]*u[17]+ uik[46]*u[18]+ uik[47]*u[19]+ uik[48]*u[20];

          wp[21]+=  uik[0]*u[21] + uik[1]*u[22] + uik[2]*u[23] + uik[3]*u[24] + uik[4]*u[25] + uik[5]*u[26] + uik[6]*u[27];
          wp[22]+=  uik[7]*u[21] + uik[8]*u[22] + uik[9]*u[23]+ uik[10]*u[24]+ uik[11]*u[25]+ uik[12]*u[26]+ uik[13]*u[27];
          wp[23]+= uik[14]*u[21]+ uik[15]*u[22]+ uik[16]*u[23]+ uik[17]*u[24]+ uik[18]*u[25]+ uik[19]*u[26]+ uik[20]*u[27];
          wp[24]+= uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27];
          wp[25]+= uik[28]*u[21]+ uik[29]*u[22]+ uik[30]*u[23]+ uik[31]*u[24]+ uik[32]*u[25]+ uik[33]*u[26]+ uik[34]*u[27];
          wp[26]+= uik[35]*u[21]+ uik[36]*u[22]+ uik[37]*u[23]+ uik[38]*u[24]+ uik[39]*u[25]+ uik[40]*u[26]+ uik[41]*u[27];
          wp[27]+= uik[42]*u[21]+ uik[43]*u[22]+ uik[44]*u[23]+ uik[45]*u[24]+ uik[46]*u[25]+ uik[47]*u[26]+ uik[48]*u[27];

          wp[28]+=  uik[0]*u[28] + uik[1]*u[29] + uik[2]*u[30] + uik[3]*u[31] + uik[4]*u[32] + uik[5]*u[33] + uik[6]*u[34];
          wp[29]+=  uik[7]*u[28] + uik[8]*u[29] + uik[9]*u[30]+ uik[10]*u[31]+ uik[11]*u[32]+ uik[12]*u[33]+ uik[13]*u[34];
          wp[30]+= uik[14]*u[28]+ uik[15]*u[29]+ uik[16]*u[30]+ uik[17]*u[31]+ uik[18]*u[32]+ uik[19]*u[33]+ uik[20]*u[34];
          wp[31]+= uik[21]*u[28]+ uik[22]*u[29]+ uik[23]*u[30]+ uik[24]*u[31]+ uik[25]*u[32]+ uik[26]*u[33]+ uik[27]*u[34];
          wp[32]+= uik[28]*u[28]+ uik[29]*u[29]+ uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34];
          wp[33]+= uik[35]*u[28]+ uik[36]*u[29]+ uik[37]*u[30]+ uik[38]*u[31]+ uik[39]*u[32]+ uik[40]*u[33]+ uik[41]*u[34];
          wp[34]+= uik[42]*u[28]+ uik[43]*u[29]+ uik[44]*u[30]+ uik[45]*u[31]+ uik[46]*u[32]+ uik[47]*u[33]+ uik[48]*u[34];

          wp[35]+=  uik[0]*u[35] + uik[1]*u[36] + uik[2]*u[37] + uik[3]*u[38] + uik[4]*u[39] + uik[5]*u[40] + uik[6]*u[41];
          wp[36]+=  uik[7]*u[35] + uik[8]*u[36] + uik[9]*u[37]+ uik[10]*u[38]+ uik[11]*u[39]+ uik[12]*u[40]+ uik[13]*u[41];
          wp[37]+= uik[14]*u[35]+ uik[15]*u[36]+ uik[16]*u[37]+ uik[17]*u[38]+ uik[18]*u[39]+ uik[19]*u[40]+ uik[20]*u[41];
          wp[38]+= uik[21]*u[35]+ uik[22]*u[36]+ uik[23]*u[37]+ uik[24]*u[38]+ uik[25]*u[39]+ uik[26]*u[40]+ uik[27]*u[41];
          wp[39]+= uik[28]*u[35]+ uik[29]*u[36]+ uik[30]*u[37]+ uik[31]*u[38]+ uik[32]*u[39]+ uik[33]*u[40]+ uik[34]*u[41];
          wp[40]+= uik[35]*u[35]+ uik[36]*u[36]+ uik[37]*u[37]+ uik[38]*u[38]+ uik[39]*u[39]+ uik[40]*u[40]+ uik[41]*u[41];
          wp[41]+= uik[42]*u[35]+ uik[43]*u[36]+ uik[44]*u[37]+ uik[45]*u[38]+ uik[46]*u[39]+ uik[47]*u[40]+ uik[48]*u[41];

          wp[42]+=  uik[0]*u[42] + uik[1]*u[43] + uik[2]*u[44] + uik[3]*u[45] + uik[4]*u[46] + uik[5]*u[47] + uik[6]*u[48];
          wp[43]+=  uik[7]*u[42] + uik[8]*u[43] + uik[9]*u[44]+ uik[10]*u[45]+ uik[11]*u[46]+ uik[12]*u[47]+ uik[13]*u[48];
          wp[44]+= uik[14]*u[42]+ uik[15]*u[43]+ uik[16]*u[44]+ uik[17]*u[45]+ uik[18]*u[46]+ uik[19]*u[47]+ uik[20]*u[48];
          wp[45]+= uik[21]*u[42]+ uik[22]*u[43]+ uik[23]*u[44]+ uik[24]*u[45]+ uik[25]*u[46]+ uik[26]*u[47]+ uik[27]*u[48];
          wp[46]+= uik[28]*u[42]+ uik[29]*u[43]+ uik[30]*u[44]+ uik[31]*u[45]+ uik[32]*u[46]+ uik[33]*u[47]+ uik[34]*u[48];
          wp[47]+= uik[35]*u[42]+ uik[36]*u[43]+ uik[37]*u[44]+ uik[38]*u[45]+ uik[39]*u[46]+ uik[40]*u[47]+ uik[41]*u[48];
          wp[48]+= uik[42]*u[42]+ uik[43]*u[43]+ uik[44]*u[44]+ uik[45]*u[45]+ uik[46]*u[46]+ uik[47]*u[47]+ uik[48]*u[48];
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
    d = ba+k*49;
    ierr = PetscMemcpy(d,dk,49*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_7(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*49;
         wp = w + vj*49;        
         for (k1=0; k1<49; k1++){
           *u++        = *wp; 
           *wp++ = 0.0;
         }
      } 
      
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }    
  } 

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr);
  ierr = PetscFree(dk);CHKERRQ(ierr);
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;  
  PLogFlops(1.3333*343*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 7 by 7 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*w,*wp;

  PetscFunctionBegin;
  
  /* initialization */
  w  = (MatScalar*)PetscMalloc(49*mbs*sizeof(MatScalar));CHKPTRQ(w);
  ierr = PetscMemzero(w,49*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(98*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 49;    

  ai = a->i; aj = a->j; aa = a->a;
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*49;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        wp = w + vj*49;
        for (i=0; i<49; i++) *wp++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,w+k*49,49*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*49;
      u    = ba + ili*49;

      uik[0] = -(d[0]*u[0] + d[7]*u[1]+ d[14]*u[2]+ d[21]*u[3]+ d[28]*u[4]+ d[35]*u[5]+ d[42]*u[6]);
      uik[1] = -(d[1]*u[0] + d[8]*u[1]+ d[15]*u[2]+ d[22]*u[3]+ d[29]*u[4]+ d[36]*u[5]+ d[43]*u[6]);
      uik[2] = -(d[2]*u[0] + d[9]*u[1]+ d[16]*u[2]+ d[23]*u[3]+ d[30]*u[4]+ d[37]*u[5]+ d[44]*u[6]);
      uik[3] = -(d[3]*u[0]+ d[10]*u[1]+ d[17]*u[2]+ d[24]*u[3]+ d[31]*u[4]+ d[38]*u[5]+ d[45]*u[6]);
      uik[4] = -(d[4]*u[0]+ d[11]*u[1]+ d[18]*u[2]+ d[25]*u[3]+ d[32]*u[4]+ d[39]*u[5]+ d[46]*u[6]);
      uik[5] = -(d[5]*u[0]+ d[12]*u[1]+ d[19]*u[2]+ d[26]*u[3]+ d[33]*u[4]+ d[40]*u[5]+ d[47]*u[6]);
      uik[6] = -(d[6]*u[0]+ d[13]*u[1]+ d[20]*u[2]+ d[27]*u[3]+ d[34]*u[4]+ d[41]*u[5]+ d[48]*u[6]);

      uik[7] = -(d[0]*u[7] + d[7]*u[8]+ d[14]*u[9]+ d[21]*u[10]+ d[28]*u[11]+ d[35]*u[12]+ d[42]*u[13]);
      uik[8] = -(d[1]*u[7] + d[8]*u[8]+ d[15]*u[9]+ d[22]*u[10]+ d[29]*u[11]+ d[36]*u[12]+ d[43]*u[13]);
      uik[9] = -(d[2]*u[7] + d[9]*u[8]+ d[16]*u[9]+ d[23]*u[10]+ d[30]*u[11]+ d[37]*u[12]+ d[44]*u[13]);
      uik[10]= -(d[3]*u[7]+ d[10]*u[8]+ d[17]*u[9]+ d[24]*u[10]+ d[31]*u[11]+ d[38]*u[12]+ d[45]*u[13]);
      uik[11]= -(d[4]*u[7]+ d[11]*u[8]+ d[18]*u[9]+ d[25]*u[10]+ d[32]*u[11]+ d[39]*u[12]+ d[46]*u[13]);
      uik[12]= -(d[5]*u[7]+ d[12]*u[8]+ d[19]*u[9]+ d[26]*u[10]+ d[33]*u[11]+ d[40]*u[12]+ d[47]*u[13]);
      uik[13]= -(d[6]*u[7]+ d[13]*u[8]+ d[20]*u[9]+ d[27]*u[10]+ d[34]*u[11]+ d[41]*u[12]+ d[48]*u[13]);

      uik[14]= -(d[0]*u[14] + d[7]*u[15]+ d[14]*u[16]+ d[21]*u[17]+ d[28]*u[18]+ d[35]*u[19]+ d[42]*u[20]);
      uik[15]= -(d[1]*u[14] + d[8]*u[15]+ d[15]*u[16]+ d[22]*u[17]+ d[29]*u[18]+ d[36]*u[19]+ d[43]*u[20]);
      uik[16]= -(d[2]*u[14] + d[9]*u[15]+ d[16]*u[16]+ d[23]*u[17]+ d[30]*u[18]+ d[37]*u[19]+ d[44]*u[20]);
      uik[17]= -(d[3]*u[14]+ d[10]*u[15]+ d[17]*u[16]+ d[24]*u[17]+ d[31]*u[18]+ d[38]*u[19]+ d[45]*u[20]);
      uik[18]= -(d[4]*u[14]+ d[11]*u[15]+ d[18]*u[16]+ d[25]*u[17]+ d[32]*u[18]+ d[39]*u[19]+ d[46]*u[20]);
      uik[19]= -(d[5]*u[14]+ d[12]*u[15]+ d[19]*u[16]+ d[26]*u[17]+ d[33]*u[18]+ d[40]*u[19]+ d[47]*u[20]);
      uik[20]= -(d[6]*u[14]+ d[13]*u[15]+ d[20]*u[16]+ d[27]*u[17]+ d[34]*u[18]+ d[41]*u[19]+ d[48]*u[20]);

      uik[21]= -(d[0]*u[21] + d[7]*u[22]+ d[14]*u[23]+ d[21]*u[24]+ d[28]*u[25]+ d[35]*u[26]+ d[42]*u[27]);
      uik[22]= -(d[1]*u[21] + d[8]*u[22]+ d[15]*u[23]+ d[22]*u[24]+ d[29]*u[25]+ d[36]*u[26]+ d[43]*u[27]);
      uik[23]= -(d[2]*u[21] + d[9]*u[22]+ d[16]*u[23]+ d[23]*u[24]+ d[30]*u[25]+ d[37]*u[26]+ d[44]*u[27]);
      uik[24]= -(d[3]*u[21]+ d[10]*u[22]+ d[17]*u[23]+ d[24]*u[24]+ d[31]*u[25]+ d[38]*u[26]+ d[45]*u[27]);
      uik[25]= -(d[4]*u[21]+ d[11]*u[22]+ d[18]*u[23]+ d[25]*u[24]+ d[32]*u[25]+ d[39]*u[26]+ d[46]*u[27]);
      uik[26]= -(d[5]*u[21]+ d[12]*u[22]+ d[19]*u[23]+ d[26]*u[24]+ d[33]*u[25]+ d[40]*u[26]+ d[47]*u[27]);
      uik[27]= -(d[6]*u[21]+ d[13]*u[22]+ d[20]*u[23]+ d[27]*u[24]+ d[34]*u[25]+ d[41]*u[26]+ d[48]*u[27]);

      uik[28]= -(d[0]*u[28] + d[7]*u[29]+ d[14]*u[30]+ d[21]*u[31]+ d[28]*u[32]+ d[35]*u[33]+ d[42]*u[34]);
      uik[29]= -(d[1]*u[28] + d[8]*u[29]+ d[15]*u[30]+ d[22]*u[31]+ d[29]*u[32]+ d[36]*u[33]+ d[43]*u[34]);
      uik[30]= -(d[2]*u[28] + d[9]*u[29]+ d[16]*u[30]+ d[23]*u[31]+ d[30]*u[32]+ d[37]*u[33]+ d[44]*u[34]);
      uik[31]= -(d[3]*u[28]+ d[10]*u[29]+ d[17]*u[30]+ d[24]*u[31]+ d[31]*u[32]+ d[38]*u[33]+ d[45]*u[34]);
      uik[32]= -(d[4]*u[28]+ d[11]*u[29]+ d[18]*u[30]+ d[25]*u[31]+ d[32]*u[32]+ d[39]*u[33]+ d[46]*u[34]);
      uik[33]= -(d[5]*u[28]+ d[12]*u[29]+ d[19]*u[30]+ d[26]*u[31]+ d[33]*u[32]+ d[40]*u[33]+ d[47]*u[34]);
      uik[34]= -(d[6]*u[28]+ d[13]*u[29]+ d[20]*u[30]+ d[27]*u[31]+ d[34]*u[32]+ d[41]*u[33]+ d[48]*u[34]);

      uik[35]= -(d[0]*u[35] + d[7]*u[36]+ d[14]*u[37]+ d[21]*u[38]+ d[28]*u[39]+ d[35]*u[40]+ d[42]*u[41]);
      uik[36]= -(d[1]*u[35] + d[8]*u[36]+ d[15]*u[37]+ d[22]*u[38]+ d[29]*u[39]+ d[36]*u[40]+ d[43]*u[41]);
      uik[37]= -(d[2]*u[35] + d[9]*u[36]+ d[16]*u[37]+ d[23]*u[38]+ d[30]*u[39]+ d[37]*u[40]+ d[44]*u[41]);
      uik[38]= -(d[3]*u[35]+ d[10]*u[36]+ d[17]*u[37]+ d[24]*u[38]+ d[31]*u[39]+ d[38]*u[40]+ d[45]*u[41]);
      uik[39]= -(d[4]*u[35]+ d[11]*u[36]+ d[18]*u[37]+ d[25]*u[38]+ d[32]*u[39]+ d[39]*u[40]+ d[46]*u[41]);
      uik[40]= -(d[5]*u[35]+ d[12]*u[36]+ d[19]*u[37]+ d[26]*u[38]+ d[33]*u[39]+ d[40]*u[40]+ d[47]*u[41]);
      uik[41]= -(d[6]*u[35]+ d[13]*u[36]+ d[20]*u[37]+ d[27]*u[38]+ d[34]*u[39]+ d[41]*u[40]+ d[48]*u[41]);

      uik[42]= -(d[0]*u[42] + d[7]*u[43]+ d[14]*u[44]+ d[21]*u[45]+ d[28]*u[46]+ d[35]*u[47]+ d[42]*u[48]);
      uik[43]= -(d[1]*u[42] + d[8]*u[43]+ d[15]*u[44]+ d[22]*u[45]+ d[29]*u[46]+ d[36]*u[47]+ d[43]*u[48]);
      uik[44]= -(d[2]*u[42] + d[9]*u[43]+ d[16]*u[44]+ d[23]*u[45]+ d[30]*u[46]+ d[37]*u[47]+ d[44]*u[48]);
      uik[45]= -(d[3]*u[42]+ d[10]*u[43]+ d[17]*u[44]+ d[24]*u[45]+ d[31]*u[46]+ d[38]*u[47]+ d[45]*u[48]);
      uik[46]= -(d[4]*u[42]+ d[11]*u[43]+ d[18]*u[44]+ d[25]*u[45]+ d[32]*u[46]+ d[39]*u[47]+ d[46]*u[48]);
      uik[47]= -(d[5]*u[42]+ d[12]*u[43]+ d[19]*u[44]+ d[26]*u[45]+ d[33]*u[46]+ d[40]*u[47]+ d[47]*u[48]);
      uik[48]= -(d[6]*u[42]+ d[13]*u[43]+ d[20]*u[44]+ d[27]*u[45]+ d[34]*u[46]+ d[41]*u[47]+ d[48]*u[48]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0]+=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6];
      dk[1]+=  uik[7]*u[0] + uik[8]*u[1] + uik[9]*u[2]+ uik[10]*u[3]+ uik[11]*u[4]+ uik[12]*u[5]+ uik[13]*u[6];
      dk[2]+= uik[14]*u[0]+ uik[15]*u[1]+ uik[16]*u[2]+ uik[17]*u[3]+ uik[18]*u[4]+ uik[19]*u[5]+ uik[20]*u[6];
      dk[3]+= uik[21]*u[0]+ uik[22]*u[1]+ uik[23]*u[2]+ uik[24]*u[3]+ uik[25]*u[4]+ uik[26]*u[5]+ uik[27]*u[6];
      dk[4]+= uik[28]*u[0]+ uik[29]*u[1]+ uik[30]*u[2]+ uik[31]*u[3]+ uik[32]*u[4]+ uik[33]*u[5]+ uik[34]*u[6];
      dk[5]+= uik[35]*u[0]+ uik[36]*u[1]+ uik[37]*u[2]+ uik[38]*u[3]+ uik[39]*u[4]+ uik[40]*u[5]+ uik[41]*u[6];
      dk[6]+= uik[42]*u[0]+ uik[43]*u[1]+ uik[44]*u[2]+ uik[45]*u[3]+ uik[46]*u[4]+ uik[47]*u[5]+ uik[48]*u[6];

      dk[7]+=  uik[0]*u[7] + uik[1]*u[8] + uik[2]*u[9] + uik[3]*u[10] + uik[4]*u[11] + uik[5]*u[12] + uik[6]*u[13];
      dk[8]+=  uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13];
      dk[9]+= uik[14]*u[7]+ uik[15]*u[8]+ uik[16]*u[9]+ uik[17]*u[10]+ uik[18]*u[11]+ uik[19]*u[12]+ uik[20]*u[13];
      dk[10]+=uik[21]*u[7]+ uik[22]*u[8]+ uik[23]*u[9]+ uik[24]*u[10]+ uik[25]*u[11]+ uik[26]*u[12]+ uik[27]*u[13];
      dk[11]+=uik[28]*u[7]+ uik[29]*u[8]+ uik[30]*u[9]+ uik[31]*u[10]+ uik[32]*u[11]+ uik[33]*u[12]+ uik[34]*u[13];
      dk[12]+=uik[35]*u[7]+ uik[36]*u[8]+ uik[37]*u[9]+ uik[38]*u[10]+ uik[39]*u[11]+ uik[40]*u[12]+ uik[41]*u[13];
      dk[13]+=uik[42]*u[7]+ uik[43]*u[8]+ uik[44]*u[9]+ uik[45]*u[10]+ uik[46]*u[11]+ uik[47]*u[12]+ uik[48]*u[13];

      dk[14]+=  uik[0]*u[14] + uik[1]*u[15] + uik[2]*u[16] + uik[3]*u[17] + uik[4]*u[18] + uik[5]*u[19] + uik[6]*u[20];
      dk[15]+=  uik[7]*u[14] + uik[8]*u[15] + uik[9]*u[16]+ uik[10]*u[17]+ uik[11]*u[18]+ uik[12]*u[19]+ uik[13]*u[20];
      dk[16]+= uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20];
      dk[17]+= uik[21]*u[14]+ uik[22]*u[15]+ uik[23]*u[16]+ uik[24]*u[17]+ uik[25]*u[18]+ uik[26]*u[19]+ uik[27]*u[20];
      dk[18]+= uik[28]*u[14]+ uik[29]*u[15]+ uik[30]*u[16]+ uik[31]*u[17]+ uik[32]*u[18]+ uik[33]*u[19]+ uik[34]*u[20];
      dk[19]+= uik[35]*u[14]+ uik[36]*u[15]+ uik[37]*u[16]+ uik[38]*u[17]+ uik[39]*u[18]+ uik[40]*u[19]+ uik[41]*u[20];
      dk[20]+= uik[42]*u[14]+ uik[43]*u[15]+ uik[44]*u[16]+ uik[45]*u[17]+ uik[46]*u[18]+ uik[47]*u[19]+ uik[48]*u[20];

      dk[21]+=  uik[0]*u[21] + uik[1]*u[22] + uik[2]*u[23] + uik[3]*u[24] + uik[4]*u[25] + uik[5]*u[26] + uik[6]*u[27];
      dk[22]+=  uik[7]*u[21] + uik[8]*u[22] + uik[9]*u[23]+ uik[10]*u[24]+ uik[11]*u[25]+ uik[12]*u[26]+ uik[13]*u[27];
      dk[23]+= uik[14]*u[21]+ uik[15]*u[22]+ uik[16]*u[23]+ uik[17]*u[24]+ uik[18]*u[25]+ uik[19]*u[26]+ uik[20]*u[27];
      dk[24]+= uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27];
      dk[25]+= uik[28]*u[21]+ uik[29]*u[22]+ uik[30]*u[23]+ uik[31]*u[24]+ uik[32]*u[25]+ uik[33]*u[26]+ uik[34]*u[27];
      dk[26]+= uik[35]*u[21]+ uik[36]*u[22]+ uik[37]*u[23]+ uik[38]*u[24]+ uik[39]*u[25]+ uik[40]*u[26]+ uik[41]*u[27];
      dk[27]+= uik[42]*u[21]+ uik[43]*u[22]+ uik[44]*u[23]+ uik[45]*u[24]+ uik[46]*u[25]+ uik[47]*u[26]+ uik[48]*u[27];

      dk[28]+=  uik[0]*u[28] + uik[1]*u[29] + uik[2]*u[30] + uik[3]*u[31] + uik[4]*u[32] + uik[5]*u[33] + uik[6]*u[34];
      dk[29]+=  uik[7]*u[28] + uik[8]*u[29] + uik[9]*u[30]+ uik[10]*u[31]+ uik[11]*u[32]+ uik[12]*u[33]+ uik[13]*u[34];
      dk[30]+= uik[14]*u[28]+ uik[15]*u[29]+ uik[16]*u[30]+ uik[17]*u[31]+ uik[18]*u[32]+ uik[19]*u[33]+ uik[20]*u[34];
      dk[31]+= uik[21]*u[28]+ uik[22]*u[29]+ uik[23]*u[30]+ uik[24]*u[31]+ uik[25]*u[32]+ uik[26]*u[33]+ uik[27]*u[34];
      dk[32]+= uik[28]*u[28]+ uik[29]*u[29]+ uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34];
      dk[33]+= uik[35]*u[28]+ uik[36]*u[29]+ uik[37]*u[30]+ uik[38]*u[31]+ uik[39]*u[32]+ uik[40]*u[33]+ uik[41]*u[34];
      dk[34]+= uik[42]*u[28]+ uik[43]*u[29]+ uik[44]*u[30]+ uik[45]*u[31]+ uik[46]*u[32]+ uik[47]*u[33]+ uik[48]*u[34];

      dk[35]+=  uik[0]*u[35] + uik[1]*u[36] + uik[2]*u[37] + uik[3]*u[38] + uik[4]*u[39] + uik[5]*u[40] + uik[6]*u[41];
      dk[36]+=  uik[7]*u[35] + uik[8]*u[36] + uik[9]*u[37]+ uik[10]*u[38]+ uik[11]*u[39]+ uik[12]*u[40]+ uik[13]*u[41];
      dk[37]+= uik[14]*u[35]+ uik[15]*u[36]+ uik[16]*u[37]+ uik[17]*u[38]+ uik[18]*u[39]+ uik[19]*u[40]+ uik[20]*u[41];
      dk[38]+= uik[21]*u[35]+ uik[22]*u[36]+ uik[23]*u[37]+ uik[24]*u[38]+ uik[25]*u[39]+ uik[26]*u[40]+ uik[27]*u[41];
      dk[39]+= uik[28]*u[35]+ uik[29]*u[36]+ uik[30]*u[37]+ uik[31]*u[38]+ uik[32]*u[39]+ uik[33]*u[40]+ uik[34]*u[41];
      dk[40]+= uik[35]*u[35]+ uik[36]*u[36]+ uik[37]*u[37]+ uik[38]*u[38]+ uik[39]*u[39]+ uik[40]*u[40]+ uik[41]*u[41];
      dk[41]+= uik[42]*u[35]+ uik[43]*u[36]+ uik[44]*u[37]+ uik[45]*u[38]+ uik[46]*u[39]+ uik[47]*u[40]+ uik[48]*u[41];

      dk[42]+=  uik[0]*u[42] + uik[1]*u[43] + uik[2]*u[44] + uik[3]*u[45] + uik[4]*u[46] + uik[5]*u[47] + uik[6]*u[48];
      dk[43]+=  uik[7]*u[42] + uik[8]*u[43] + uik[9]*u[44]+ uik[10]*u[45]+ uik[11]*u[46]+ uik[12]*u[47]+ uik[13]*u[48];
      dk[44]+= uik[14]*u[42]+ uik[15]*u[43]+ uik[16]*u[44]+ uik[17]*u[45]+ uik[18]*u[46]+ uik[19]*u[47]+ uik[20]*u[48];
      dk[45]+= uik[21]*u[42]+ uik[22]*u[43]+ uik[23]*u[44]+ uik[24]*u[45]+ uik[25]*u[46]+ uik[26]*u[47]+ uik[27]*u[48];
      dk[46]+= uik[28]*u[42]+ uik[29]*u[43]+ uik[30]*u[44]+ uik[31]*u[45]+ uik[32]*u[46]+ uik[33]*u[47]+ uik[34]*u[48];
      dk[47]+= uik[35]*u[42]+ uik[36]*u[43]+ uik[37]*u[44]+ uik[38]*u[45]+ uik[39]*u[46]+ uik[40]*u[47]+ uik[41]*u[48];
      dk[48]+= uik[42]*u[42]+ uik[43]*u[43]+ uik[44]*u[44]+ uik[45]*u[45]+ uik[46]*u[46]+ uik[47]*u[47]+ uik[48]*u[48];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*49,uik,49*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*49;
          u = ba + j*49;

          wp[0]+=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6];
          wp[1]+=  uik[7]*u[0] + uik[8]*u[1] + uik[9]*u[2]+ uik[10]*u[3]+ uik[11]*u[4]+ uik[12]*u[5]+ uik[13]*u[6];
          wp[2]+= uik[14]*u[0]+ uik[15]*u[1]+ uik[16]*u[2]+ uik[17]*u[3]+ uik[18]*u[4]+ uik[19]*u[5]+ uik[20]*u[6];
          wp[3]+= uik[21]*u[0]+ uik[22]*u[1]+ uik[23]*u[2]+ uik[24]*u[3]+ uik[25]*u[4]+ uik[26]*u[5]+ uik[27]*u[6];
          wp[4]+= uik[28]*u[0]+ uik[29]*u[1]+ uik[30]*u[2]+ uik[31]*u[3]+ uik[32]*u[4]+ uik[33]*u[5]+ uik[34]*u[6];
          wp[5]+= uik[35]*u[0]+ uik[36]*u[1]+ uik[37]*u[2]+ uik[38]*u[3]+ uik[39]*u[4]+ uik[40]*u[5]+ uik[41]*u[6];
          wp[6]+= uik[42]*u[0]+ uik[43]*u[1]+ uik[44]*u[2]+ uik[45]*u[3]+ uik[46]*u[4]+ uik[47]*u[5]+ uik[48]*u[6];

          wp[7]+=  uik[0]*u[7] + uik[1]*u[8] + uik[2]*u[9] + uik[3]*u[10] + uik[4]*u[11] + uik[5]*u[12] + uik[6]*u[13];
          wp[8]+=  uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13];
          wp[9]+= uik[14]*u[7]+ uik[15]*u[8]+ uik[16]*u[9]+ uik[17]*u[10]+ uik[18]*u[11]+ uik[19]*u[12]+ uik[20]*u[13];
          wp[10]+=uik[21]*u[7]+ uik[22]*u[8]+ uik[23]*u[9]+ uik[24]*u[10]+ uik[25]*u[11]+ uik[26]*u[12]+ uik[27]*u[13];
          wp[11]+=uik[28]*u[7]+ uik[29]*u[8]+ uik[30]*u[9]+ uik[31]*u[10]+ uik[32]*u[11]+ uik[33]*u[12]+ uik[34]*u[13];
          wp[12]+=uik[35]*u[7]+ uik[36]*u[8]+ uik[37]*u[9]+ uik[38]*u[10]+ uik[39]*u[11]+ uik[40]*u[12]+ uik[41]*u[13];
          wp[13]+=uik[42]*u[7]+ uik[43]*u[8]+ uik[44]*u[9]+ uik[45]*u[10]+ uik[46]*u[11]+ uik[47]*u[12]+ uik[48]*u[13];

          wp[14]+=  uik[0]*u[14] + uik[1]*u[15] + uik[2]*u[16] + uik[3]*u[17] + uik[4]*u[18] + uik[5]*u[19] + uik[6]*u[20];
          wp[15]+=  uik[7]*u[14] + uik[8]*u[15] + uik[9]*u[16]+ uik[10]*u[17]+ uik[11]*u[18]+ uik[12]*u[19]+ uik[13]*u[20];
          wp[16]+= uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20];
          wp[17]+= uik[21]*u[14]+ uik[22]*u[15]+ uik[23]*u[16]+ uik[24]*u[17]+ uik[25]*u[18]+ uik[26]*u[19]+ uik[27]*u[20];
          wp[18]+= uik[28]*u[14]+ uik[29]*u[15]+ uik[30]*u[16]+ uik[31]*u[17]+ uik[32]*u[18]+ uik[33]*u[19]+ uik[34]*u[20];
          wp[19]+= uik[35]*u[14]+ uik[36]*u[15]+ uik[37]*u[16]+ uik[38]*u[17]+ uik[39]*u[18]+ uik[40]*u[19]+ uik[41]*u[20];
          wp[20]+= uik[42]*u[14]+ uik[43]*u[15]+ uik[44]*u[16]+ uik[45]*u[17]+ uik[46]*u[18]+ uik[47]*u[19]+ uik[48]*u[20];

          wp[21]+=  uik[0]*u[21] + uik[1]*u[22] + uik[2]*u[23] + uik[3]*u[24] + uik[4]*u[25] + uik[5]*u[26] + uik[6]*u[27];
          wp[22]+=  uik[7]*u[21] + uik[8]*u[22] + uik[9]*u[23]+ uik[10]*u[24]+ uik[11]*u[25]+ uik[12]*u[26]+ uik[13]*u[27];
          wp[23]+= uik[14]*u[21]+ uik[15]*u[22]+ uik[16]*u[23]+ uik[17]*u[24]+ uik[18]*u[25]+ uik[19]*u[26]+ uik[20]*u[27];
          wp[24]+= uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27];
          wp[25]+= uik[28]*u[21]+ uik[29]*u[22]+ uik[30]*u[23]+ uik[31]*u[24]+ uik[32]*u[25]+ uik[33]*u[26]+ uik[34]*u[27];
          wp[26]+= uik[35]*u[21]+ uik[36]*u[22]+ uik[37]*u[23]+ uik[38]*u[24]+ uik[39]*u[25]+ uik[40]*u[26]+ uik[41]*u[27];
          wp[27]+= uik[42]*u[21]+ uik[43]*u[22]+ uik[44]*u[23]+ uik[45]*u[24]+ uik[46]*u[25]+ uik[47]*u[26]+ uik[48]*u[27];

          wp[28]+=  uik[0]*u[28] + uik[1]*u[29] + uik[2]*u[30] + uik[3]*u[31] + uik[4]*u[32] + uik[5]*u[33] + uik[6]*u[34];
          wp[29]+=  uik[7]*u[28] + uik[8]*u[29] + uik[9]*u[30]+ uik[10]*u[31]+ uik[11]*u[32]+ uik[12]*u[33]+ uik[13]*u[34];
          wp[30]+= uik[14]*u[28]+ uik[15]*u[29]+ uik[16]*u[30]+ uik[17]*u[31]+ uik[18]*u[32]+ uik[19]*u[33]+ uik[20]*u[34];
          wp[31]+= uik[21]*u[28]+ uik[22]*u[29]+ uik[23]*u[30]+ uik[24]*u[31]+ uik[25]*u[32]+ uik[26]*u[33]+ uik[27]*u[34];
          wp[32]+= uik[28]*u[28]+ uik[29]*u[29]+ uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34];
          wp[33]+= uik[35]*u[28]+ uik[36]*u[29]+ uik[37]*u[30]+ uik[38]*u[31]+ uik[39]*u[32]+ uik[40]*u[33]+ uik[41]*u[34];
          wp[34]+= uik[42]*u[28]+ uik[43]*u[29]+ uik[44]*u[30]+ uik[45]*u[31]+ uik[46]*u[32]+ uik[47]*u[33]+ uik[48]*u[34];

          wp[35]+=  uik[0]*u[35] + uik[1]*u[36] + uik[2]*u[37] + uik[3]*u[38] + uik[4]*u[39] + uik[5]*u[40] + uik[6]*u[41];
          wp[36]+=  uik[7]*u[35] + uik[8]*u[36] + uik[9]*u[37]+ uik[10]*u[38]+ uik[11]*u[39]+ uik[12]*u[40]+ uik[13]*u[41];
          wp[37]+= uik[14]*u[35]+ uik[15]*u[36]+ uik[16]*u[37]+ uik[17]*u[38]+ uik[18]*u[39]+ uik[19]*u[40]+ uik[20]*u[41];
          wp[38]+= uik[21]*u[35]+ uik[22]*u[36]+ uik[23]*u[37]+ uik[24]*u[38]+ uik[25]*u[39]+ uik[26]*u[40]+ uik[27]*u[41];
          wp[39]+= uik[28]*u[35]+ uik[29]*u[36]+ uik[30]*u[37]+ uik[31]*u[38]+ uik[32]*u[39]+ uik[33]*u[40]+ uik[34]*u[41];
          wp[40]+= uik[35]*u[35]+ uik[36]*u[36]+ uik[37]*u[37]+ uik[38]*u[38]+ uik[39]*u[39]+ uik[40]*u[40]+ uik[41]*u[41];
          wp[41]+= uik[42]*u[35]+ uik[43]*u[36]+ uik[44]*u[37]+ uik[45]*u[38]+ uik[46]*u[39]+ uik[47]*u[40]+ uik[48]*u[41];

          wp[42]+=  uik[0]*u[42] + uik[1]*u[43] + uik[2]*u[44] + uik[3]*u[45] + uik[4]*u[46] + uik[5]*u[47] + uik[6]*u[48];
          wp[43]+=  uik[7]*u[42] + uik[8]*u[43] + uik[9]*u[44]+ uik[10]*u[45]+ uik[11]*u[46]+ uik[12]*u[47]+ uik[13]*u[48];
          wp[44]+= uik[14]*u[42]+ uik[15]*u[43]+ uik[16]*u[44]+ uik[17]*u[45]+ uik[18]*u[46]+ uik[19]*u[47]+ uik[20]*u[48];
          wp[45]+= uik[21]*u[42]+ uik[22]*u[43]+ uik[23]*u[44]+ uik[24]*u[45]+ uik[25]*u[46]+ uik[26]*u[47]+ uik[27]*u[48];
          wp[46]+= uik[28]*u[42]+ uik[29]*u[43]+ uik[30]*u[44]+ uik[31]*u[45]+ uik[32]*u[46]+ uik[33]*u[47]+ uik[34]*u[48];
          wp[47]+= uik[35]*u[42]+ uik[36]*u[43]+ uik[37]*u[44]+ uik[38]*u[45]+ uik[39]*u[46]+ uik[40]*u[47]+ uik[41]*u[48];
          wp[48]+= uik[42]*u[42]+ uik[43]*u[43]+ uik[44]*u[44]+ uik[45]*u[45]+ uik[46]*u[46]+ uik[47]*u[47]+ uik[48]*u[48];
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
    d = ba+k*49;
    ierr = PetscMemcpy(d,dk,49*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_7(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*49;
         wp = w + vj*49;        
         for (k1=0; k1<49; k1++){
           *u++        = *wp; 
           *wp++ = 0.0;
         }
      } 
      
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }    
  } 

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr);
  ierr = PetscFree(dk);CHKERRQ(ierr);

  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;  
  PLogFlops(1.3333*343*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* Version for when blocks are 6 by 6 */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_6"
int MatCholeskyFactorNumeric_SeqSBAIJ_6(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 perm = b->row;
  int                *perm_ptr,ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*w,*wp;

  PetscFunctionBegin;
  /* initialization */
  w  = (MatScalar*)PetscMalloc(36*mbs*sizeof(MatScalar));CHKPTRQ(w);
  ierr = PetscMemzero(w,36*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(72*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 36;     
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(36*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,36*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<36; k1++){
            dk[k1]       = aa[k*36+k1]; 
            aa[k*36+k1] = aa[j*36+k1]; 
            aa[j*36+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*36;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<36; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<6; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<6; k1++) *ap++ = dk[k + 6*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*36;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        wp = w + vj*36;
        for (i=0; i<36; i++) *wp++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,w+k*36,36*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*36;
      u    = ba + ili*36;

      uik[0] = -(d[0]*u[0] + d[6]*u[1] + d[12]*u[2] + d[18]*u[3] + d[24]*u[4] + d[30]*u[5]);
      uik[1] = -(d[1]*u[0] + d[7]*u[1] + d[13]*u[2] + d[19]*u[3] + d[25]*u[4] + d[31]*u[5]);
      uik[2] = -(d[2]*u[0] + d[8]*u[1] + d[14]*u[2] + d[20]*u[3] + d[26]*u[4] + d[32]*u[5]);
      uik[3] = -(d[3]*u[0] + d[9]*u[1] + d[15]*u[2] + d[21]*u[3] + d[27]*u[4] + d[33]*u[5]);
      uik[4] = -(d[4]*u[0]+ d[10]*u[1] + d[16]*u[2] + d[22]*u[3] + d[28]*u[4] + d[34]*u[5]);
      uik[5] = -(d[5]*u[0]+ d[11]*u[1] + d[17]*u[2] + d[23]*u[3] + d[29]*u[4] + d[35]*u[5]);

      uik[6] = -(d[0]*u[6] + d[6]*u[7] + d[12]*u[8] + d[18]*u[9] + d[24]*u[10] + d[30]*u[11]);
      uik[7] = -(d[1]*u[6] + d[7]*u[7] + d[13]*u[8] + d[19]*u[9] + d[25]*u[10] + d[31]*u[11]);
      uik[8] = -(d[2]*u[6] + d[8]*u[7] + d[14]*u[8] + d[20]*u[9] + d[26]*u[10] + d[32]*u[11]);
      uik[9] = -(d[3]*u[6] + d[9]*u[7] + d[15]*u[8] + d[21]*u[9] + d[27]*u[10] + d[33]*u[11]);
      uik[10]= -(d[4]*u[6]+ d[10]*u[7] + d[16]*u[8] + d[22]*u[9] + d[28]*u[10] + d[34]*u[11]);
      uik[11]= -(d[5]*u[6]+ d[11]*u[7] + d[17]*u[8] + d[23]*u[9] + d[29]*u[10] + d[35]*u[11]);

      uik[12] = -(d[0]*u[12] + d[6]*u[13] + d[12]*u[14] + d[18]*u[15] + d[24]*u[16] + d[30]*u[17]);
      uik[13] = -(d[1]*u[12] + d[7]*u[13] + d[13]*u[14] + d[19]*u[15] + d[25]*u[16] + d[31]*u[17]);
      uik[14] = -(d[2]*u[12] + d[8]*u[13] + d[14]*u[14] + d[20]*u[15] + d[26]*u[16] + d[32]*u[17]);
      uik[15] = -(d[3]*u[12] + d[9]*u[13] + d[15]*u[14] + d[21]*u[15] + d[27]*u[16] + d[33]*u[17]);
      uik[16] = -(d[4]*u[12]+ d[10]*u[13] + d[16]*u[14] + d[22]*u[15] + d[28]*u[16] + d[34]*u[17]);
      uik[17] = -(d[5]*u[12]+ d[11]*u[13] + d[17]*u[14] + d[23]*u[15] + d[29]*u[16] + d[35]*u[17]);

      uik[18] = -(d[0]*u[18] + d[6]*u[19] + d[12]*u[20] + d[18]*u[21] + d[24]*u[22] + d[30]*u[23]);
      uik[19] = -(d[1]*u[18] + d[7]*u[19] + d[13]*u[20] + d[19]*u[21] + d[25]*u[22] + d[31]*u[23]);
      uik[20] = -(d[2]*u[18] + d[8]*u[19] + d[14]*u[20] + d[20]*u[21] + d[26]*u[22] + d[32]*u[23]);
      uik[21] = -(d[3]*u[18] + d[9]*u[19] + d[15]*u[20] + d[21]*u[21] + d[27]*u[22] + d[33]*u[23]);
      uik[22] = -(d[4]*u[18]+ d[10]*u[19] + d[16]*u[20] + d[22]*u[21] + d[28]*u[22] + d[34]*u[23]);
      uik[23] = -(d[5]*u[18]+ d[11]*u[19] + d[17]*u[20] + d[23]*u[21] + d[29]*u[22] + d[35]*u[23]);

      uik[24] = -(d[0]*u[24] + d[6]*u[25] + d[12]*u[26] + d[18]*u[27] + d[24]*u[28] + d[30]*u[29]);
      uik[25] = -(d[1]*u[24] + d[7]*u[25] + d[13]*u[26] + d[19]*u[27] + d[25]*u[28] + d[31]*u[29]);
      uik[26] = -(d[2]*u[24] + d[8]*u[25] + d[14]*u[26] + d[20]*u[27] + d[26]*u[28] + d[32]*u[29]);
      uik[27] = -(d[3]*u[24] + d[9]*u[25] + d[15]*u[26] + d[21]*u[27] + d[27]*u[28] + d[33]*u[29]);
      uik[28] = -(d[4]*u[24]+ d[10]*u[25] + d[16]*u[26] + d[22]*u[27] + d[28]*u[28] + d[34]*u[29]);
      uik[29] = -(d[5]*u[24]+ d[11]*u[25] + d[17]*u[26] + d[23]*u[27] + d[29]*u[28] + d[35]*u[29]);

      uik[30] = -(d[0]*u[30] + d[6]*u[31] + d[12]*u[32] + d[18]*u[33] + d[24]*u[34] + d[30]*u[35]);
      uik[31] = -(d[1]*u[30] + d[7]*u[31] + d[13]*u[32] + d[19]*u[33] + d[25]*u[34] + d[31]*u[35]);
      uik[32] = -(d[2]*u[30] + d[8]*u[31] + d[14]*u[32] + d[20]*u[33] + d[26]*u[34] + d[32]*u[35]);
      uik[33] = -(d[3]*u[30] + d[9]*u[31] + d[15]*u[32] + d[21]*u[33] + d[27]*u[34] + d[33]*u[35]);
      uik[34] = -(d[4]*u[30]+ d[10]*u[31] + d[16]*u[32] + d[22]*u[33] + d[28]*u[34] + d[34]*u[35]);
      uik[35] = -(d[5]*u[30]+ d[11]*u[31] + d[17]*u[32] + d[23]*u[33] + d[29]*u[34] + d[35]*u[35]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
      dk[1] +=  uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2] + uik[9]*u[3]+ uik[10]*u[4]+ uik[11]*u[5];
      dk[2] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3]+ uik[16]*u[4]+ uik[17]*u[5];
      dk[3] += uik[18]*u[0]+ uik[19]*u[1]+ uik[20]*u[2]+ uik[21]*u[3]+ uik[22]*u[4]+ uik[23]*u[5];
      dk[4] += uik[24]*u[0]+ uik[25]*u[1]+ uik[26]*u[2]+ uik[27]*u[3]+ uik[28]*u[4]+ uik[29]*u[5];
      dk[5] += uik[30]*u[0]+ uik[31]*u[1]+ uik[32]*u[2]+ uik[33]*u[3]+ uik[34]*u[4]+ uik[35]*u[5];

      dk[6] +=  uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8] + uik[3]*u[9] + uik[4]*u[10] + uik[5]*u[11];
      dk[7] +=  uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11];
      dk[8] += uik[12]*u[6]+ uik[13]*u[7]+ uik[14]*u[8]+ uik[15]*u[9]+ uik[16]*u[10]+ uik[17]*u[11];
      dk[9] += uik[18]*u[6]+ uik[19]*u[7]+ uik[20]*u[8]+ uik[21]*u[9]+ uik[22]*u[10]+ uik[23]*u[11];
      dk[10]+= uik[24]*u[6]+ uik[25]*u[7]+ uik[26]*u[8]+ uik[27]*u[9]+ uik[28]*u[10]+ uik[29]*u[11];
      dk[11]+= uik[30]*u[6]+ uik[31]*u[7]+ uik[32]*u[8]+ uik[33]*u[9]+ uik[34]*u[10]+ uik[35]*u[11];

      dk[12]+=  uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15] + uik[4]*u[16] + uik[5]*u[17];
      dk[13]+=  uik[6]*u[12] + uik[7]*u[13] + uik[8]*u[14] + uik[9]*u[15]+ uik[10]*u[16]+ uik[11]*u[17];
      dk[14]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17];
      dk[15]+= uik[18]*u[12]+ uik[19]*u[13]+ uik[20]*u[14]+ uik[21]*u[15]+ uik[22]*u[16]+ uik[23]*u[17];
      dk[16]+= uik[24]*u[12]+ uik[25]*u[13]+ uik[26]*u[14]+ uik[27]*u[15]+ uik[28]*u[16]+ uik[29]*u[17];
      dk[17]+= uik[30]*u[12]+ uik[31]*u[13]+ uik[32]*u[14]+ uik[33]*u[15]+ uik[34]*u[16]+ uik[35]*u[17];

      dk[18]+=  uik[0]*u[18] + uik[1]*u[19] + uik[2]*u[20] + uik[3]*u[21] + uik[4]*u[22] + uik[5]*u[23];
      dk[19]+=  uik[6]*u[18] + uik[7]*u[19] + uik[8]*u[20] + uik[9]*u[21]+ uik[10]*u[22]+ uik[11]*u[23];
      dk[20]+= uik[12]*u[18]+ uik[13]*u[19]+ uik[14]*u[20]+ uik[15]*u[21]+ uik[16]*u[22]+ uik[17]*u[23];
      dk[21]+= uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23];
      dk[22]+= uik[24]*u[18]+ uik[25]*u[19]+ uik[26]*u[20]+ uik[27]*u[21]+ uik[28]*u[22]+ uik[29]*u[23];
      dk[23]+= uik[30]*u[18]+ uik[31]*u[19]+ uik[32]*u[20]+ uik[33]*u[21]+ uik[34]*u[22]+ uik[35]*u[23];

      dk[24]+=  uik[0]*u[24] + uik[1]*u[25] + uik[2]*u[26] + uik[3]*u[27] + uik[4]*u[28] + uik[5]*u[29];
      dk[25]+=  uik[6]*u[24] + uik[7]*u[25] + uik[8]*u[26] + uik[9]*u[27]+ uik[10]*u[28]+ uik[11]*u[29];
      dk[26]+= uik[12]*u[24]+ uik[13]*u[25]+ uik[14]*u[26]+ uik[15]*u[27]+ uik[16]*u[28]+ uik[17]*u[29];
      dk[27]+= uik[18]*u[24]+ uik[19]*u[25]+ uik[20]*u[26]+ uik[21]*u[27]+ uik[22]*u[28]+ uik[23]*u[29];
      dk[28]+= uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27]+ uik[28]*u[28]+ uik[29]*u[29];
      dk[29]+= uik[30]*u[24]+ uik[31]*u[25]+ uik[32]*u[26]+ uik[33]*u[27]+ uik[34]*u[28]+ uik[35]*u[29];

      dk[30]+=  uik[0]*u[30] + uik[1]*u[31] + uik[2]*u[32] + uik[3]*u[33] + uik[4]*u[34] + uik[5]*u[35];
      dk[31]+=  uik[6]*u[30] + uik[7]*u[31] + uik[8]*u[32] + uik[9]*u[33]+ uik[10]*u[34]+ uik[11]*u[35];
      dk[32]+= uik[12]*u[30]+ uik[13]*u[31]+ uik[14]*u[32]+ uik[15]*u[33]+ uik[16]*u[34]+ uik[17]*u[35];
      dk[33]+= uik[18]*u[30]+ uik[19]*u[31]+ uik[20]*u[32]+ uik[21]*u[33]+ uik[22]*u[34]+ uik[23]*u[35];
      dk[34]+= uik[24]*u[30]+ uik[25]*u[31]+ uik[26]*u[32]+ uik[27]*u[33]+ uik[28]*u[34]+ uik[29]*u[35];
      dk[35]+= uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34]+ uik[35]*u[35];
 
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*36,uik,36*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*36;
          u = ba + j*36;
          wp[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
          wp[1] +=  uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2] + uik[9]*u[3]+ uik[10]*u[4]+ uik[11]*u[5];
          wp[2] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3]+ uik[16]*u[4]+ uik[17]*u[5];
          wp[3] += uik[18]*u[0]+ uik[19]*u[1]+ uik[20]*u[2]+ uik[21]*u[3]+ uik[22]*u[4]+ uik[23]*u[5];
          wp[4] += uik[24]*u[0]+ uik[25]*u[1]+ uik[26]*u[2]+ uik[27]*u[3]+ uik[28]*u[4]+ uik[29]*u[5];
          wp[5] += uik[30]*u[0]+ uik[31]*u[1]+ uik[32]*u[2]+ uik[33]*u[3]+ uik[34]*u[4]+ uik[35]*u[5];

          wp[6] +=  uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8] + uik[3]*u[9] + uik[4]*u[10] + uik[5]*u[11];
          wp[7] +=  uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11];
          wp[8] += uik[12]*u[6]+ uik[13]*u[7]+ uik[14]*u[8]+ uik[15]*u[9]+ uik[16]*u[10]+ uik[17]*u[11];
          wp[9] += uik[18]*u[6]+ uik[19]*u[7]+ uik[20]*u[8]+ uik[21]*u[9]+ uik[22]*u[10]+ uik[23]*u[11];
          wp[10]+= uik[24]*u[6]+ uik[25]*u[7]+ uik[26]*u[8]+ uik[27]*u[9]+ uik[28]*u[10]+ uik[29]*u[11];
          wp[11]+= uik[30]*u[6]+ uik[31]*u[7]+ uik[32]*u[8]+ uik[33]*u[9]+ uik[34]*u[10]+ uik[35]*u[11];

          wp[12]+=  uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15] + uik[4]*u[16] + uik[5]*u[17];
          wp[13]+=  uik[6]*u[12] + uik[7]*u[13] + uik[8]*u[14] + uik[9]*u[15]+ uik[10]*u[16]+ uik[11]*u[17];
          wp[14]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17];
          wp[15]+= uik[18]*u[12]+ uik[19]*u[13]+ uik[20]*u[14]+ uik[21]*u[15]+ uik[22]*u[16]+ uik[23]*u[17];
          wp[16]+= uik[24]*u[12]+ uik[25]*u[13]+ uik[26]*u[14]+ uik[27]*u[15]+ uik[28]*u[16]+ uik[29]*u[17];
          wp[17]+= uik[30]*u[12]+ uik[31]*u[13]+ uik[32]*u[14]+ uik[33]*u[15]+ uik[34]*u[16]+ uik[35]*u[17];

          wp[18]+=  uik[0]*u[18] + uik[1]*u[19] + uik[2]*u[20] + uik[3]*u[21] + uik[4]*u[22] + uik[5]*u[23];
          wp[19]+=  uik[6]*u[18] + uik[7]*u[19] + uik[8]*u[20] + uik[9]*u[21]+ uik[10]*u[22]+ uik[11]*u[23];
          wp[20]+= uik[12]*u[18]+ uik[13]*u[19]+ uik[14]*u[20]+ uik[15]*u[21]+ uik[16]*u[22]+ uik[17]*u[23];
          wp[21]+= uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23];
          wp[22]+= uik[24]*u[18]+ uik[25]*u[19]+ uik[26]*u[20]+ uik[27]*u[21]+ uik[28]*u[22]+ uik[29]*u[23];
          wp[23]+= uik[30]*u[18]+ uik[31]*u[19]+ uik[32]*u[20]+ uik[33]*u[21]+ uik[34]*u[22]+ uik[35]*u[23];

          wp[24]+=  uik[0]*u[24] + uik[1]*u[25] + uik[2]*u[26] + uik[3]*u[27] + uik[4]*u[28] + uik[5]*u[29];
          wp[25]+=  uik[6]*u[24] + uik[7]*u[25] + uik[8]*u[26] + uik[9]*u[27]+ uik[10]*u[28]+ uik[11]*u[29];
          wp[26]+= uik[12]*u[24]+ uik[13]*u[25]+ uik[14]*u[26]+ uik[15]*u[27]+ uik[16]*u[28]+ uik[17]*u[29];
          wp[27]+= uik[18]*u[24]+ uik[19]*u[25]+ uik[20]*u[26]+ uik[21]*u[27]+ uik[22]*u[28]+ uik[23]*u[29];
          wp[28]+= uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27]+ uik[28]*u[28]+ uik[29]*u[29];
          wp[29]+= uik[30]*u[24]+ uik[31]*u[25]+ uik[32]*u[26]+ uik[33]*u[27]+ uik[34]*u[28]+ uik[35]*u[29];

          wp[30]+=  uik[0]*u[30] + uik[1]*u[31] + uik[2]*u[32] + uik[3]*u[33] + uik[4]*u[34] + uik[5]*u[35];
          wp[31]+=  uik[6]*u[30] + uik[7]*u[31] + uik[8]*u[32] + uik[9]*u[33]+ uik[10]*u[34]+ uik[11]*u[35];
          wp[32]+= uik[12]*u[30]+ uik[13]*u[31]+ uik[14]*u[32]+ uik[15]*u[33]+ uik[16]*u[34]+ uik[17]*u[35];
          wp[33]+= uik[18]*u[30]+ uik[19]*u[31]+ uik[20]*u[32]+ uik[21]*u[33]+ uik[22]*u[34]+ uik[23]*u[35];
          wp[34]+= uik[24]*u[30]+ uik[25]*u[31]+ uik[26]*u[32]+ uik[27]*u[33]+ uik[28]*u[34]+ uik[29]*u[35];
          wp[35]+= uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34]+ uik[35]*u[35];
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
    d = ba+k*36;
    ierr = PetscMemcpy(d,dk,36*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_6(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*36;
         wp = w + vj*36;        
         for (k1=0; k1<36; k1++){
           *u++        = *wp; 
           *wp++ = 0.0;
         }
      } 
      
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }    
  } 

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr); 
  ierr = PetscFree(dk);CHKERRQ(ierr);
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;  
  PLogFlops(1.3333*216*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 6 by 6 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*w,*wp;

  PetscFunctionBegin;
  
  /* initialization */
  w  = (MatScalar*)PetscMalloc(36*mbs*sizeof(MatScalar));CHKPTRQ(w);
  ierr = PetscMemzero(w,36*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(72*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 36;     

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*36;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        wp = w + vj*36;
        for (i=0; i<36; i++) *wp++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,w+k*36,36*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*36;
      u    = ba + ili*36;

      uik[0] = -(d[0]*u[0] + d[6]*u[1] + d[12]*u[2] + d[18]*u[3] + d[24]*u[4] + d[30]*u[5]);
      uik[1] = -(d[1]*u[0] + d[7]*u[1] + d[13]*u[2] + d[19]*u[3] + d[25]*u[4] + d[31]*u[5]);
      uik[2] = -(d[2]*u[0] + d[8]*u[1] + d[14]*u[2] + d[20]*u[3] + d[26]*u[4] + d[32]*u[5]);
      uik[3] = -(d[3]*u[0] + d[9]*u[1] + d[15]*u[2] + d[21]*u[3] + d[27]*u[4] + d[33]*u[5]);
      uik[4] = -(d[4]*u[0]+ d[10]*u[1] + d[16]*u[2] + d[22]*u[3] + d[28]*u[4] + d[34]*u[5]);
      uik[5] = -(d[5]*u[0]+ d[11]*u[1] + d[17]*u[2] + d[23]*u[3] + d[29]*u[4] + d[35]*u[5]);

      uik[6] = -(d[0]*u[6] + d[6]*u[7] + d[12]*u[8] + d[18]*u[9] + d[24]*u[10] + d[30]*u[11]);
      uik[7] = -(d[1]*u[6] + d[7]*u[7] + d[13]*u[8] + d[19]*u[9] + d[25]*u[10] + d[31]*u[11]);
      uik[8] = -(d[2]*u[6] + d[8]*u[7] + d[14]*u[8] + d[20]*u[9] + d[26]*u[10] + d[32]*u[11]);
      uik[9] = -(d[3]*u[6] + d[9]*u[7] + d[15]*u[8] + d[21]*u[9] + d[27]*u[10] + d[33]*u[11]);
      uik[10]= -(d[4]*u[6]+ d[10]*u[7] + d[16]*u[8] + d[22]*u[9] + d[28]*u[10] + d[34]*u[11]);
      uik[11]= -(d[5]*u[6]+ d[11]*u[7] + d[17]*u[8] + d[23]*u[9] + d[29]*u[10] + d[35]*u[11]);

      uik[12] = -(d[0]*u[12] + d[6]*u[13] + d[12]*u[14] + d[18]*u[15] + d[24]*u[16] + d[30]*u[17]);
      uik[13] = -(d[1]*u[12] + d[7]*u[13] + d[13]*u[14] + d[19]*u[15] + d[25]*u[16] + d[31]*u[17]);
      uik[14] = -(d[2]*u[12] + d[8]*u[13] + d[14]*u[14] + d[20]*u[15] + d[26]*u[16] + d[32]*u[17]);
      uik[15] = -(d[3]*u[12] + d[9]*u[13] + d[15]*u[14] + d[21]*u[15] + d[27]*u[16] + d[33]*u[17]);
      uik[16] = -(d[4]*u[12]+ d[10]*u[13] + d[16]*u[14] + d[22]*u[15] + d[28]*u[16] + d[34]*u[17]);
      uik[17] = -(d[5]*u[12]+ d[11]*u[13] + d[17]*u[14] + d[23]*u[15] + d[29]*u[16] + d[35]*u[17]);

      uik[18] = -(d[0]*u[18] + d[6]*u[19] + d[12]*u[20] + d[18]*u[21] + d[24]*u[22] + d[30]*u[23]);
      uik[19] = -(d[1]*u[18] + d[7]*u[19] + d[13]*u[20] + d[19]*u[21] + d[25]*u[22] + d[31]*u[23]);
      uik[20] = -(d[2]*u[18] + d[8]*u[19] + d[14]*u[20] + d[20]*u[21] + d[26]*u[22] + d[32]*u[23]);
      uik[21] = -(d[3]*u[18] + d[9]*u[19] + d[15]*u[20] + d[21]*u[21] + d[27]*u[22] + d[33]*u[23]);
      uik[22] = -(d[4]*u[18]+ d[10]*u[19] + d[16]*u[20] + d[22]*u[21] + d[28]*u[22] + d[34]*u[23]);
      uik[23] = -(d[5]*u[18]+ d[11]*u[19] + d[17]*u[20] + d[23]*u[21] + d[29]*u[22] + d[35]*u[23]);

      uik[24] = -(d[0]*u[24] + d[6]*u[25] + d[12]*u[26] + d[18]*u[27] + d[24]*u[28] + d[30]*u[29]);
      uik[25] = -(d[1]*u[24] + d[7]*u[25] + d[13]*u[26] + d[19]*u[27] + d[25]*u[28] + d[31]*u[29]);
      uik[26] = -(d[2]*u[24] + d[8]*u[25] + d[14]*u[26] + d[20]*u[27] + d[26]*u[28] + d[32]*u[29]);
      uik[27] = -(d[3]*u[24] + d[9]*u[25] + d[15]*u[26] + d[21]*u[27] + d[27]*u[28] + d[33]*u[29]);
      uik[28] = -(d[4]*u[24]+ d[10]*u[25] + d[16]*u[26] + d[22]*u[27] + d[28]*u[28] + d[34]*u[29]);
      uik[29] = -(d[5]*u[24]+ d[11]*u[25] + d[17]*u[26] + d[23]*u[27] + d[29]*u[28] + d[35]*u[29]);

      uik[30] = -(d[0]*u[30] + d[6]*u[31] + d[12]*u[32] + d[18]*u[33] + d[24]*u[34] + d[30]*u[35]);
      uik[31] = -(d[1]*u[30] + d[7]*u[31] + d[13]*u[32] + d[19]*u[33] + d[25]*u[34] + d[31]*u[35]);
      uik[32] = -(d[2]*u[30] + d[8]*u[31] + d[14]*u[32] + d[20]*u[33] + d[26]*u[34] + d[32]*u[35]);
      uik[33] = -(d[3]*u[30] + d[9]*u[31] + d[15]*u[32] + d[21]*u[33] + d[27]*u[34] + d[33]*u[35]);
      uik[34] = -(d[4]*u[30]+ d[10]*u[31] + d[16]*u[32] + d[22]*u[33] + d[28]*u[34] + d[34]*u[35]);
      uik[35] = -(d[5]*u[30]+ d[11]*u[31] + d[17]*u[32] + d[23]*u[33] + d[29]*u[34] + d[35]*u[35]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
      dk[1] +=  uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2] + uik[9]*u[3]+ uik[10]*u[4]+ uik[11]*u[5];
      dk[2] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3]+ uik[16]*u[4]+ uik[17]*u[5];
      dk[3] += uik[18]*u[0]+ uik[19]*u[1]+ uik[20]*u[2]+ uik[21]*u[3]+ uik[22]*u[4]+ uik[23]*u[5];
      dk[4] += uik[24]*u[0]+ uik[25]*u[1]+ uik[26]*u[2]+ uik[27]*u[3]+ uik[28]*u[4]+ uik[29]*u[5];
      dk[5] += uik[30]*u[0]+ uik[31]*u[1]+ uik[32]*u[2]+ uik[33]*u[3]+ uik[34]*u[4]+ uik[35]*u[5];

      dk[6] +=  uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8] + uik[3]*u[9] + uik[4]*u[10] + uik[5]*u[11];
      dk[7] +=  uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11];
      dk[8] += uik[12]*u[6]+ uik[13]*u[7]+ uik[14]*u[8]+ uik[15]*u[9]+ uik[16]*u[10]+ uik[17]*u[11];
      dk[9] += uik[18]*u[6]+ uik[19]*u[7]+ uik[20]*u[8]+ uik[21]*u[9]+ uik[22]*u[10]+ uik[23]*u[11];
      dk[10]+= uik[24]*u[6]+ uik[25]*u[7]+ uik[26]*u[8]+ uik[27]*u[9]+ uik[28]*u[10]+ uik[29]*u[11];
      dk[11]+= uik[30]*u[6]+ uik[31]*u[7]+ uik[32]*u[8]+ uik[33]*u[9]+ uik[34]*u[10]+ uik[35]*u[11];

      dk[12]+=  uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15] + uik[4]*u[16] + uik[5]*u[17];
      dk[13]+=  uik[6]*u[12] + uik[7]*u[13] + uik[8]*u[14] + uik[9]*u[15]+ uik[10]*u[16]+ uik[11]*u[17];
      dk[14]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17];
      dk[15]+= uik[18]*u[12]+ uik[19]*u[13]+ uik[20]*u[14]+ uik[21]*u[15]+ uik[22]*u[16]+ uik[23]*u[17];
      dk[16]+= uik[24]*u[12]+ uik[25]*u[13]+ uik[26]*u[14]+ uik[27]*u[15]+ uik[28]*u[16]+ uik[29]*u[17];
      dk[17]+= uik[30]*u[12]+ uik[31]*u[13]+ uik[32]*u[14]+ uik[33]*u[15]+ uik[34]*u[16]+ uik[35]*u[17];

      dk[18]+=  uik[0]*u[18] + uik[1]*u[19] + uik[2]*u[20] + uik[3]*u[21] + uik[4]*u[22] + uik[5]*u[23];
      dk[19]+=  uik[6]*u[18] + uik[7]*u[19] + uik[8]*u[20] + uik[9]*u[21]+ uik[10]*u[22]+ uik[11]*u[23];
      dk[20]+= uik[12]*u[18]+ uik[13]*u[19]+ uik[14]*u[20]+ uik[15]*u[21]+ uik[16]*u[22]+ uik[17]*u[23];
      dk[21]+= uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23];
      dk[22]+= uik[24]*u[18]+ uik[25]*u[19]+ uik[26]*u[20]+ uik[27]*u[21]+ uik[28]*u[22]+ uik[29]*u[23];
      dk[23]+= uik[30]*u[18]+ uik[31]*u[19]+ uik[32]*u[20]+ uik[33]*u[21]+ uik[34]*u[22]+ uik[35]*u[23];

      dk[24]+=  uik[0]*u[24] + uik[1]*u[25] + uik[2]*u[26] + uik[3]*u[27] + uik[4]*u[28] + uik[5]*u[29];
      dk[25]+=  uik[6]*u[24] + uik[7]*u[25] + uik[8]*u[26] + uik[9]*u[27]+ uik[10]*u[28]+ uik[11]*u[29];
      dk[26]+= uik[12]*u[24]+ uik[13]*u[25]+ uik[14]*u[26]+ uik[15]*u[27]+ uik[16]*u[28]+ uik[17]*u[29];
      dk[27]+= uik[18]*u[24]+ uik[19]*u[25]+ uik[20]*u[26]+ uik[21]*u[27]+ uik[22]*u[28]+ uik[23]*u[29];
      dk[28]+= uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27]+ uik[28]*u[28]+ uik[29]*u[29];
      dk[29]+= uik[30]*u[24]+ uik[31]*u[25]+ uik[32]*u[26]+ uik[33]*u[27]+ uik[34]*u[28]+ uik[35]*u[29];

      dk[30]+=  uik[0]*u[30] + uik[1]*u[31] + uik[2]*u[32] + uik[3]*u[33] + uik[4]*u[34] + uik[5]*u[35];
      dk[31]+=  uik[6]*u[30] + uik[7]*u[31] + uik[8]*u[32] + uik[9]*u[33]+ uik[10]*u[34]+ uik[11]*u[35];
      dk[32]+= uik[12]*u[30]+ uik[13]*u[31]+ uik[14]*u[32]+ uik[15]*u[33]+ uik[16]*u[34]+ uik[17]*u[35];
      dk[33]+= uik[18]*u[30]+ uik[19]*u[31]+ uik[20]*u[32]+ uik[21]*u[33]+ uik[22]*u[34]+ uik[23]*u[35];
      dk[34]+= uik[24]*u[30]+ uik[25]*u[31]+ uik[26]*u[32]+ uik[27]*u[33]+ uik[28]*u[34]+ uik[29]*u[35];
      dk[35]+= uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34]+ uik[35]*u[35];
 
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*36,uik,36*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*36;
          u = ba + j*36;
          wp[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
          wp[1] +=  uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2] + uik[9]*u[3]+ uik[10]*u[4]+ uik[11]*u[5];
          wp[2] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3]+ uik[16]*u[4]+ uik[17]*u[5];
          wp[3] += uik[18]*u[0]+ uik[19]*u[1]+ uik[20]*u[2]+ uik[21]*u[3]+ uik[22]*u[4]+ uik[23]*u[5];
          wp[4] += uik[24]*u[0]+ uik[25]*u[1]+ uik[26]*u[2]+ uik[27]*u[3]+ uik[28]*u[4]+ uik[29]*u[5];
          wp[5] += uik[30]*u[0]+ uik[31]*u[1]+ uik[32]*u[2]+ uik[33]*u[3]+ uik[34]*u[4]+ uik[35]*u[5];

          wp[6] +=  uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8] + uik[3]*u[9] + uik[4]*u[10] + uik[5]*u[11];
          wp[7] +=  uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11];
          wp[8] += uik[12]*u[6]+ uik[13]*u[7]+ uik[14]*u[8]+ uik[15]*u[9]+ uik[16]*u[10]+ uik[17]*u[11];
          wp[9] += uik[18]*u[6]+ uik[19]*u[7]+ uik[20]*u[8]+ uik[21]*u[9]+ uik[22]*u[10]+ uik[23]*u[11];
          wp[10]+= uik[24]*u[6]+ uik[25]*u[7]+ uik[26]*u[8]+ uik[27]*u[9]+ uik[28]*u[10]+ uik[29]*u[11];
          wp[11]+= uik[30]*u[6]+ uik[31]*u[7]+ uik[32]*u[8]+ uik[33]*u[9]+ uik[34]*u[10]+ uik[35]*u[11];

          wp[12]+=  uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15] + uik[4]*u[16] + uik[5]*u[17];
          wp[13]+=  uik[6]*u[12] + uik[7]*u[13] + uik[8]*u[14] + uik[9]*u[15]+ uik[10]*u[16]+ uik[11]*u[17];
          wp[14]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17];
          wp[15]+= uik[18]*u[12]+ uik[19]*u[13]+ uik[20]*u[14]+ uik[21]*u[15]+ uik[22]*u[16]+ uik[23]*u[17];
          wp[16]+= uik[24]*u[12]+ uik[25]*u[13]+ uik[26]*u[14]+ uik[27]*u[15]+ uik[28]*u[16]+ uik[29]*u[17];
          wp[17]+= uik[30]*u[12]+ uik[31]*u[13]+ uik[32]*u[14]+ uik[33]*u[15]+ uik[34]*u[16]+ uik[35]*u[17];

          wp[18]+=  uik[0]*u[18] + uik[1]*u[19] + uik[2]*u[20] + uik[3]*u[21] + uik[4]*u[22] + uik[5]*u[23];
          wp[19]+=  uik[6]*u[18] + uik[7]*u[19] + uik[8]*u[20] + uik[9]*u[21]+ uik[10]*u[22]+ uik[11]*u[23];
          wp[20]+= uik[12]*u[18]+ uik[13]*u[19]+ uik[14]*u[20]+ uik[15]*u[21]+ uik[16]*u[22]+ uik[17]*u[23];
          wp[21]+= uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23];
          wp[22]+= uik[24]*u[18]+ uik[25]*u[19]+ uik[26]*u[20]+ uik[27]*u[21]+ uik[28]*u[22]+ uik[29]*u[23];
          wp[23]+= uik[30]*u[18]+ uik[31]*u[19]+ uik[32]*u[20]+ uik[33]*u[21]+ uik[34]*u[22]+ uik[35]*u[23];

          wp[24]+=  uik[0]*u[24] + uik[1]*u[25] + uik[2]*u[26] + uik[3]*u[27] + uik[4]*u[28] + uik[5]*u[29];
          wp[25]+=  uik[6]*u[24] + uik[7]*u[25] + uik[8]*u[26] + uik[9]*u[27]+ uik[10]*u[28]+ uik[11]*u[29];
          wp[26]+= uik[12]*u[24]+ uik[13]*u[25]+ uik[14]*u[26]+ uik[15]*u[27]+ uik[16]*u[28]+ uik[17]*u[29];
          wp[27]+= uik[18]*u[24]+ uik[19]*u[25]+ uik[20]*u[26]+ uik[21]*u[27]+ uik[22]*u[28]+ uik[23]*u[29];
          wp[28]+= uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27]+ uik[28]*u[28]+ uik[29]*u[29];
          wp[29]+= uik[30]*u[24]+ uik[31]*u[25]+ uik[32]*u[26]+ uik[33]*u[27]+ uik[34]*u[28]+ uik[35]*u[29];

          wp[30]+=  uik[0]*u[30] + uik[1]*u[31] + uik[2]*u[32] + uik[3]*u[33] + uik[4]*u[34] + uik[5]*u[35];
          wp[31]+=  uik[6]*u[30] + uik[7]*u[31] + uik[8]*u[32] + uik[9]*u[33]+ uik[10]*u[34]+ uik[11]*u[35];
          wp[32]+= uik[12]*u[30]+ uik[13]*u[31]+ uik[14]*u[32]+ uik[15]*u[33]+ uik[16]*u[34]+ uik[17]*u[35];
          wp[33]+= uik[18]*u[30]+ uik[19]*u[31]+ uik[20]*u[32]+ uik[21]*u[33]+ uik[22]*u[34]+ uik[23]*u[35];
          wp[34]+= uik[24]*u[30]+ uik[25]*u[31]+ uik[26]*u[32]+ uik[27]*u[33]+ uik[28]*u[34]+ uik[29]*u[35];
          wp[35]+= uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34]+ uik[35]*u[35];
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
    d = ba+k*36;
    ierr = PetscMemcpy(d,dk,36*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_6(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*36;
         wp = w + vj*36;        
         for (k1=0; k1<36; k1++){
           *u++        = *wp; 
           *wp++ = 0.0;
         }
      } 
      
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }    
  } 

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree(il);CHKERRQ(ierr); 
  ierr = PetscFree(dk);CHKERRQ(ierr);

  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;  
  PLogFlops(1.3333*216*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* Version for when blocks are 5 by 5  */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_5"
int MatCholeskyFactorNumeric_SeqSBAIJ_5(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 perm = b->row;
  int                *perm_ptr,ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*rtmp,*rtmp_ptr;

  PetscFunctionBegin;
  /* initialization */
  rtmp  = (MatScalar*)PetscMalloc(25*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,25*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(50*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 25;    
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(25*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,25*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<25; k1++){
            dk[k1]       = aa[k*25+k1]; 
            aa[k*25+k1] = aa[j*25+k1]; 
            aa[j*25+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*25;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<25; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<5; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<5; k1++) *ap++ = dk[k + 5*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*25;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        rtmp_ptr = rtmp + vj*25;
        for (i=0; i<25; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*25,25*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*25;
      u    = ba + ili*25;

      uik[0] = -(d[0]*u[0] + d[5]*u[1] + d[10]*u[2] + d[15]*u[3] + d[20]*u[4]);
      uik[1] = -(d[1]*u[0] + d[6]*u[1] + d[11]*u[2] + d[16]*u[3] + d[21]*u[4]);
      uik[2] = -(d[2]*u[0] + d[7]*u[1] + d[12]*u[2] + d[17]*u[3] + d[22]*u[4]);
      uik[3] = -(d[3]*u[0] + d[8]*u[1] + d[13]*u[2] + d[18]*u[3] + d[23]*u[4]);
      uik[4] = -(d[4]*u[0] + d[9]*u[1] + d[14]*u[2] + d[19]*u[3] + d[24]*u[4]);

      uik[5] = -(d[0]*u[5] + d[5]*u[6] + d[10]*u[7] + d[15]*u[8] + d[20]*u[9]);
      uik[6] = -(d[1]*u[5] + d[6]*u[6] + d[11]*u[7] + d[16]*u[8] + d[21]*u[9]);
      uik[7] = -(d[2]*u[5] + d[7]*u[6] + d[12]*u[7] + d[17]*u[8] + d[22]*u[9]);
      uik[8] = -(d[3]*u[5] + d[8]*u[6] + d[13]*u[7] + d[18]*u[8] + d[23]*u[9]);
      uik[9] = -(d[4]*u[5] + d[9]*u[6] + d[14]*u[7] + d[19]*u[8] + d[24]*u[9]);

      uik[10]= -(d[0]*u[10] + d[5]*u[11] + d[10]*u[12] + d[15]*u[13] + d[20]*u[14]);
      uik[11]= -(d[1]*u[10] + d[6]*u[11] + d[11]*u[12] + d[16]*u[13] + d[21]*u[14]);
      uik[12]= -(d[2]*u[10] + d[7]*u[11] + d[12]*u[12] + d[17]*u[13] + d[22]*u[14]);
      uik[13]= -(d[3]*u[10] + d[8]*u[11] + d[13]*u[12] + d[18]*u[13] + d[23]*u[14]);
      uik[14]= -(d[4]*u[10] + d[9]*u[11] + d[14]*u[12] + d[19]*u[13] + d[24]*u[14]);

      uik[15]= -(d[0]*u[15] + d[5]*u[16] + d[10]*u[17] + d[15]*u[18] + d[20]*u[19]);
      uik[16]= -(d[1]*u[15] + d[6]*u[16] + d[11]*u[17] + d[16]*u[18] + d[21]*u[19]);
      uik[17]= -(d[2]*u[15] + d[7]*u[16] + d[12]*u[17] + d[17]*u[18] + d[22]*u[19]);
      uik[18]= -(d[3]*u[15] + d[8]*u[16] + d[13]*u[17] + d[18]*u[18] + d[23]*u[19]);
      uik[19]= -(d[4]*u[15] + d[9]*u[16] + d[14]*u[17] + d[19]*u[18] + d[24]*u[19]);

      uik[20]= -(d[0]*u[20] + d[5]*u[21] + d[10]*u[22] + d[15]*u[23] + d[20]*u[24]);
      uik[21]= -(d[1]*u[20] + d[6]*u[21] + d[11]*u[22] + d[16]*u[23] + d[21]*u[24]);
      uik[22]= -(d[2]*u[20] + d[7]*u[21] + d[12]*u[22] + d[17]*u[23] + d[22]*u[24]);
      uik[23]= -(d[3]*u[20] + d[8]*u[21] + d[13]*u[22] + d[18]*u[23] + d[23]*u[24]);
      uik[24]= -(d[4]*u[20] + d[9]*u[21] + d[14]*u[22] + d[19]*u[23] + d[24]*u[24]);


      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4];
      dk[1] +=  uik[5]*u[0] + uik[6]*u[1] + uik[7]*u[2] + uik[8]*u[3] + uik[9]*u[4];
      dk[2] += uik[10]*u[0]+ uik[11]*u[1]+ uik[12]*u[2]+ uik[13]*u[3]+ uik[14]*u[4];
      dk[3] += uik[15]*u[0]+ uik[16]*u[1]+ uik[17]*u[2]+ uik[18]*u[3]+ uik[19]*u[4];
      dk[4] += uik[20]*u[0]+ uik[21]*u[1]+ uik[22]*u[2]+ uik[23]*u[3]+ uik[24]*u[4];

      dk[5] +=  uik[0]*u[5] + uik[1]*u[6] + uik[2]*u[7] + uik[3]*u[8] + uik[4]*u[9];
      dk[6] +=  uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9];
      dk[7] += uik[10]*u[5]+ uik[11]*u[6]+ uik[12]*u[7]+ uik[13]*u[8]+ uik[14]*u[9];
      dk[8] += uik[15]*u[5]+ uik[16]*u[6]+ uik[17]*u[7]+ uik[18]*u[8]+ uik[19]*u[9];
      dk[9] += uik[20]*u[5]+ uik[21]*u[6]+ uik[22]*u[7]+ uik[23]*u[8]+ uik[24]*u[9];

      dk[10] +=  uik[0]*u[10] + uik[1]*u[11] + uik[2]*u[12] + uik[3]*u[13] + uik[4]*u[14];
      dk[11] +=  uik[5]*u[10] + uik[6]*u[11] + uik[7]*u[12] + uik[8]*u[13] + uik[9]*u[14];
      dk[12] += uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14];
      dk[13] += uik[15]*u[10]+ uik[16]*u[11]+ uik[17]*u[12]+ uik[18]*u[13]+ uik[19]*u[14];
      dk[14] += uik[20]*u[10]+ uik[21]*u[11]+ uik[22]*u[12]+ uik[23]*u[13]+ uik[24]*u[14];

      dk[15] +=  uik[0]*u[15] + uik[1]*u[16] + uik[2]*u[17] + uik[3]*u[18] + uik[4]*u[19];
      dk[16] +=  uik[5]*u[15] + uik[6]*u[16] + uik[7]*u[17] + uik[8]*u[18] + uik[9]*u[19];
      dk[17] += uik[10]*u[15]+ uik[11]*u[16]+ uik[12]*u[17]+ uik[13]*u[18]+ uik[14]*u[19];
      dk[18] += uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19];
      dk[19] += uik[20]*u[15]+ uik[21]*u[16]+ uik[22]*u[17]+ uik[23]*u[18]+ uik[24]*u[19];

      dk[20] +=  uik[0]*u[20] + uik[1]*u[21] + uik[2]*u[22] + uik[3]*u[23] + uik[4]*u[24];
      dk[21] +=  uik[5]*u[20] + uik[6]*u[21] + uik[7]*u[22] + uik[8]*u[23] + uik[9]*u[24];
      dk[22] += uik[10]*u[20]+ uik[11]*u[21]+ uik[12]*u[22]+ uik[13]*u[23]+ uik[14]*u[24];
      dk[23] += uik[15]*u[20]+ uik[16]*u[21]+ uik[17]*u[22]+ uik[18]*u[23]+ uik[19]*u[24];
      dk[24] += uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*25,uik,25*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*25;
          u = ba + j*25;
          rtmp_ptr[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4];
          rtmp_ptr[1] +=  uik[5]*u[0] + uik[6]*u[1] + uik[7]*u[2] + uik[8]*u[3] + uik[9]*u[4];
          rtmp_ptr[2] += uik[10]*u[0]+ uik[11]*u[1]+ uik[12]*u[2]+ uik[13]*u[3]+ uik[14]*u[4];
          rtmp_ptr[3] += uik[15]*u[0]+ uik[16]*u[1]+ uik[17]*u[2]+ uik[18]*u[3]+ uik[19]*u[4];
          rtmp_ptr[4] += uik[20]*u[0]+ uik[21]*u[1]+ uik[22]*u[2]+ uik[23]*u[3]+ uik[24]*u[4];

          rtmp_ptr[5] +=  uik[0]*u[5] + uik[1]*u[6] + uik[2]*u[7] + uik[3]*u[8] + uik[4]*u[9];
          rtmp_ptr[6] +=  uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9];
          rtmp_ptr[7] += uik[10]*u[5]+ uik[11]*u[6]+ uik[12]*u[7]+ uik[13]*u[8]+ uik[14]*u[9];
          rtmp_ptr[8] += uik[15]*u[5]+ uik[16]*u[6]+ uik[17]*u[7]+ uik[18]*u[8]+ uik[19]*u[9];
          rtmp_ptr[9] += uik[20]*u[5]+ uik[21]*u[6]+ uik[22]*u[7]+ uik[23]*u[8]+ uik[24]*u[9];

          rtmp_ptr[10] +=  uik[0]*u[10] + uik[1]*u[11] + uik[2]*u[12] + uik[3]*u[13] + uik[4]*u[14];
          rtmp_ptr[11] +=  uik[5]*u[10] + uik[6]*u[11] + uik[7]*u[12] + uik[8]*u[13] + uik[9]*u[14];
          rtmp_ptr[12] += uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14];
          rtmp_ptr[13] += uik[15]*u[10]+ uik[16]*u[11]+ uik[17]*u[12]+ uik[18]*u[13]+ uik[19]*u[14];
          rtmp_ptr[14] += uik[20]*u[10]+ uik[21]*u[11]+ uik[22]*u[12]+ uik[23]*u[13]+ uik[24]*u[14];

          rtmp_ptr[15] +=  uik[0]*u[15] + uik[1]*u[16] + uik[2]*u[17] + uik[3]*u[18] + uik[4]*u[19];
          rtmp_ptr[16] +=  uik[5]*u[15] + uik[6]*u[16] + uik[7]*u[17] + uik[8]*u[18] + uik[9]*u[19];
          rtmp_ptr[17] += uik[10]*u[15]+ uik[11]*u[16]+ uik[12]*u[17]+ uik[13]*u[18]+ uik[14]*u[19];
          rtmp_ptr[18] += uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19];
          rtmp_ptr[19] += uik[20]*u[15]+ uik[21]*u[16]+ uik[22]*u[17]+ uik[23]*u[18]+ uik[24]*u[19];

          rtmp_ptr[20] +=  uik[0]*u[20] + uik[1]*u[21] + uik[2]*u[22] + uik[3]*u[23] + uik[4]*u[24];
          rtmp_ptr[21] +=  uik[5]*u[20] + uik[6]*u[21] + uik[7]*u[22] + uik[8]*u[23] + uik[9]*u[24];
          rtmp_ptr[22] += uik[10]*u[20]+ uik[11]*u[21]+ uik[12]*u[22]+ uik[13]*u[23]+ uik[14]*u[24];
          rtmp_ptr[23] += uik[15]*u[20]+ uik[16]*u[21]+ uik[17]*u[22]+ uik[18]*u[23]+ uik[19]*u[24];
          rtmp_ptr[24] += uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24];
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
    d = ba+k*25;
    ierr = PetscMemcpy(d,dk,25*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_5(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*25;
         rtmp_ptr = rtmp + vj*25;        
         for (k1=0; k1<25; k1++){
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
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PLogFlops(1.3333*125*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 5 by 5 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*rtmp,*rtmp_ptr;

  PetscFunctionBegin;
  
  /* initialization */
  rtmp  = (MatScalar*)PetscMalloc(25*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,25*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(50*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 25;     

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*25;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        rtmp_ptr = rtmp + vj*25;
        for (i=0; i<25; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*25,25*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*25;
      u    = ba + ili*25;

      uik[0] = -(d[0]*u[0] + d[5]*u[1] + d[10]*u[2] + d[15]*u[3] + d[20]*u[4]);
      uik[1] = -(d[1]*u[0] + d[6]*u[1] + d[11]*u[2] + d[16]*u[3] + d[21]*u[4]);
      uik[2] = -(d[2]*u[0] + d[7]*u[1] + d[12]*u[2] + d[17]*u[3] + d[22]*u[4]);
      uik[3] = -(d[3]*u[0] + d[8]*u[1] + d[13]*u[2] + d[18]*u[3] + d[23]*u[4]);
      uik[4] = -(d[4]*u[0] + d[9]*u[1] + d[14]*u[2] + d[19]*u[3] + d[24]*u[4]);

      uik[5] = -(d[0]*u[5] + d[5]*u[6] + d[10]*u[7] + d[15]*u[8] + d[20]*u[9]);
      uik[6] = -(d[1]*u[5] + d[6]*u[6] + d[11]*u[7] + d[16]*u[8] + d[21]*u[9]);
      uik[7] = -(d[2]*u[5] + d[7]*u[6] + d[12]*u[7] + d[17]*u[8] + d[22]*u[9]);
      uik[8] = -(d[3]*u[5] + d[8]*u[6] + d[13]*u[7] + d[18]*u[8] + d[23]*u[9]);
      uik[9] = -(d[4]*u[5] + d[9]*u[6] + d[14]*u[7] + d[19]*u[8] + d[24]*u[9]);

      uik[10]= -(d[0]*u[10] + d[5]*u[11] + d[10]*u[12] + d[15]*u[13] + d[20]*u[14]);
      uik[11]= -(d[1]*u[10] + d[6]*u[11] + d[11]*u[12] + d[16]*u[13] + d[21]*u[14]);
      uik[12]= -(d[2]*u[10] + d[7]*u[11] + d[12]*u[12] + d[17]*u[13] + d[22]*u[14]);
      uik[13]= -(d[3]*u[10] + d[8]*u[11] + d[13]*u[12] + d[18]*u[13] + d[23]*u[14]);
      uik[14]= -(d[4]*u[10] + d[9]*u[11] + d[14]*u[12] + d[19]*u[13] + d[24]*u[14]);

      uik[15]= -(d[0]*u[15] + d[5]*u[16] + d[10]*u[17] + d[15]*u[18] + d[20]*u[19]);
      uik[16]= -(d[1]*u[15] + d[6]*u[16] + d[11]*u[17] + d[16]*u[18] + d[21]*u[19]);
      uik[17]= -(d[2]*u[15] + d[7]*u[16] + d[12]*u[17] + d[17]*u[18] + d[22]*u[19]);
      uik[18]= -(d[3]*u[15] + d[8]*u[16] + d[13]*u[17] + d[18]*u[18] + d[23]*u[19]);
      uik[19]= -(d[4]*u[15] + d[9]*u[16] + d[14]*u[17] + d[19]*u[18] + d[24]*u[19]);

      uik[20]= -(d[0]*u[20] + d[5]*u[21] + d[10]*u[22] + d[15]*u[23] + d[20]*u[24]);
      uik[21]= -(d[1]*u[20] + d[6]*u[21] + d[11]*u[22] + d[16]*u[23] + d[21]*u[24]);
      uik[22]= -(d[2]*u[20] + d[7]*u[21] + d[12]*u[22] + d[17]*u[23] + d[22]*u[24]);
      uik[23]= -(d[3]*u[20] + d[8]*u[21] + d[13]*u[22] + d[18]*u[23] + d[23]*u[24]);
      uik[24]= -(d[4]*u[20] + d[9]*u[21] + d[14]*u[22] + d[19]*u[23] + d[24]*u[24]);


      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4];
      dk[1] +=  uik[5]*u[0] + uik[6]*u[1] + uik[7]*u[2] + uik[8]*u[3] + uik[9]*u[4];
      dk[2] += uik[10]*u[0]+ uik[11]*u[1]+ uik[12]*u[2]+ uik[13]*u[3]+ uik[14]*u[4];
      dk[3] += uik[15]*u[0]+ uik[16]*u[1]+ uik[17]*u[2]+ uik[18]*u[3]+ uik[19]*u[4];
      dk[4] += uik[20]*u[0]+ uik[21]*u[1]+ uik[22]*u[2]+ uik[23]*u[3]+ uik[24]*u[4];

      dk[5] +=  uik[0]*u[5] + uik[1]*u[6] + uik[2]*u[7] + uik[3]*u[8] + uik[4]*u[9];
      dk[6] +=  uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9];
      dk[7] += uik[10]*u[5]+ uik[11]*u[6]+ uik[12]*u[7]+ uik[13]*u[8]+ uik[14]*u[9];
      dk[8] += uik[15]*u[5]+ uik[16]*u[6]+ uik[17]*u[7]+ uik[18]*u[8]+ uik[19]*u[9];
      dk[9] += uik[20]*u[5]+ uik[21]*u[6]+ uik[22]*u[7]+ uik[23]*u[8]+ uik[24]*u[9];

      dk[10] +=  uik[0]*u[10] + uik[1]*u[11] + uik[2]*u[12] + uik[3]*u[13] + uik[4]*u[14];
      dk[11] +=  uik[5]*u[10] + uik[6]*u[11] + uik[7]*u[12] + uik[8]*u[13] + uik[9]*u[14];
      dk[12] += uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14];
      dk[13] += uik[15]*u[10]+ uik[16]*u[11]+ uik[17]*u[12]+ uik[18]*u[13]+ uik[19]*u[14];
      dk[14] += uik[20]*u[10]+ uik[21]*u[11]+ uik[22]*u[12]+ uik[23]*u[13]+ uik[24]*u[14];

      dk[15] +=  uik[0]*u[15] + uik[1]*u[16] + uik[2]*u[17] + uik[3]*u[18] + uik[4]*u[19];
      dk[16] +=  uik[5]*u[15] + uik[6]*u[16] + uik[7]*u[17] + uik[8]*u[18] + uik[9]*u[19];
      dk[17] += uik[10]*u[15]+ uik[11]*u[16]+ uik[12]*u[17]+ uik[13]*u[18]+ uik[14]*u[19];
      dk[18] += uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19];
      dk[19] += uik[20]*u[15]+ uik[21]*u[16]+ uik[22]*u[17]+ uik[23]*u[18]+ uik[24]*u[19];

      dk[20] +=  uik[0]*u[20] + uik[1]*u[21] + uik[2]*u[22] + uik[3]*u[23] + uik[4]*u[24];
      dk[21] +=  uik[5]*u[20] + uik[6]*u[21] + uik[7]*u[22] + uik[8]*u[23] + uik[9]*u[24];
      dk[22] += uik[10]*u[20]+ uik[11]*u[21]+ uik[12]*u[22]+ uik[13]*u[23]+ uik[14]*u[24];
      dk[23] += uik[15]*u[20]+ uik[16]*u[21]+ uik[17]*u[22]+ uik[18]*u[23]+ uik[19]*u[24];
      dk[24] += uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*25,uik,25*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*25;
          u = ba + j*25;
          rtmp_ptr[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4];
          rtmp_ptr[1] +=  uik[5]*u[0] + uik[6]*u[1] + uik[7]*u[2] + uik[8]*u[3] + uik[9]*u[4];
          rtmp_ptr[2] += uik[10]*u[0]+ uik[11]*u[1]+ uik[12]*u[2]+ uik[13]*u[3]+ uik[14]*u[4];
          rtmp_ptr[3] += uik[15]*u[0]+ uik[16]*u[1]+ uik[17]*u[2]+ uik[18]*u[3]+ uik[19]*u[4];
          rtmp_ptr[4] += uik[20]*u[0]+ uik[21]*u[1]+ uik[22]*u[2]+ uik[23]*u[3]+ uik[24]*u[4];

          rtmp_ptr[5] +=  uik[0]*u[5] + uik[1]*u[6] + uik[2]*u[7] + uik[3]*u[8] + uik[4]*u[9];
          rtmp_ptr[6] +=  uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9];
          rtmp_ptr[7] += uik[10]*u[5]+ uik[11]*u[6]+ uik[12]*u[7]+ uik[13]*u[8]+ uik[14]*u[9];
          rtmp_ptr[8] += uik[15]*u[5]+ uik[16]*u[6]+ uik[17]*u[7]+ uik[18]*u[8]+ uik[19]*u[9];
          rtmp_ptr[9] += uik[20]*u[5]+ uik[21]*u[6]+ uik[22]*u[7]+ uik[23]*u[8]+ uik[24]*u[9];

          rtmp_ptr[10] +=  uik[0]*u[10] + uik[1]*u[11] + uik[2]*u[12] + uik[3]*u[13] + uik[4]*u[14];
          rtmp_ptr[11] +=  uik[5]*u[10] + uik[6]*u[11] + uik[7]*u[12] + uik[8]*u[13] + uik[9]*u[14];
          rtmp_ptr[12] += uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14];
          rtmp_ptr[13] += uik[15]*u[10]+ uik[16]*u[11]+ uik[17]*u[12]+ uik[18]*u[13]+ uik[19]*u[14];
          rtmp_ptr[14] += uik[20]*u[10]+ uik[21]*u[11]+ uik[22]*u[12]+ uik[23]*u[13]+ uik[24]*u[14];

          rtmp_ptr[15] +=  uik[0]*u[15] + uik[1]*u[16] + uik[2]*u[17] + uik[3]*u[18] + uik[4]*u[19];
          rtmp_ptr[16] +=  uik[5]*u[15] + uik[6]*u[16] + uik[7]*u[17] + uik[8]*u[18] + uik[9]*u[19];
          rtmp_ptr[17] += uik[10]*u[15]+ uik[11]*u[16]+ uik[12]*u[17]+ uik[13]*u[18]+ uik[14]*u[19];
          rtmp_ptr[18] += uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19];
          rtmp_ptr[19] += uik[20]*u[15]+ uik[21]*u[16]+ uik[22]*u[17]+ uik[23]*u[18]+ uik[24]*u[19];

          rtmp_ptr[20] +=  uik[0]*u[20] + uik[1]*u[21] + uik[2]*u[22] + uik[3]*u[23] + uik[4]*u[24];
          rtmp_ptr[21] +=  uik[5]*u[20] + uik[6]*u[21] + uik[7]*u[22] + uik[8]*u[23] + uik[9]*u[24];
          rtmp_ptr[22] += uik[10]*u[20]+ uik[11]*u[21]+ uik[12]*u[22]+ uik[13]*u[23]+ uik[14]*u[24];
          rtmp_ptr[23] += uik[15]*u[20]+ uik[16]*u[21]+ uik[17]*u[22]+ uik[18]*u[23]+ uik[19]*u[24];
          rtmp_ptr[24] += uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24];
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
    d = ba+k*25;
    ierr = PetscMemcpy(d,dk,25*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_5(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*25;
         rtmp_ptr = rtmp + vj*25;        
         for (k1=0; k1<25; k1++){
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
  PLogFlops(1.3333*125*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 4 by 4 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;

  PetscFunctionBegin;
  
  /* initialization */
  rtmp  = (MatScalar*)PetscMalloc(16*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,16*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(32*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 16;     

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*16;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        rtmp_ptr = rtmp + vj*16;
        for (i=0; i<16; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*16,16*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*16;
      u    = ba + ili*16;

      uik[0] = -(diag[0]*u[0] + diag[4]*u[1] + diag[8]*u[2] + diag[12]*u[3]);
      uik[1] = -(diag[1]*u[0] + diag[5]*u[1] + diag[9]*u[2] + diag[13]*u[3]);
      uik[2] = -(diag[2]*u[0] + diag[6]*u[1] + diag[10]*u[2]+ diag[14]*u[3]);
      uik[3] = -(diag[3]*u[0] + diag[7]*u[1] + diag[11]*u[2]+ diag[15]*u[3]);

      uik[4] = -(diag[0]*u[4] + diag[4]*u[5] + diag[8]*u[6] + diag[12]*u[7]);
      uik[5] = -(diag[1]*u[4] + diag[5]*u[5] + diag[9]*u[6] + diag[13]*u[7]);
      uik[6] = -(diag[2]*u[4] + diag[6]*u[5] + diag[10]*u[6]+ diag[14]*u[7]);
      uik[7] = -(diag[3]*u[4] + diag[7]*u[5] + diag[11]*u[6]+ diag[15]*u[7]);

      uik[8] = -(diag[0]*u[8] + diag[4]*u[9] + diag[8]*u[10] + diag[12]*u[11]);
      uik[9] = -(diag[1]*u[8] + diag[5]*u[9] + diag[9]*u[10] + diag[13]*u[11]);
      uik[10]= -(diag[2]*u[8] + diag[6]*u[9] + diag[10]*u[10]+ diag[14]*u[11]);
      uik[11]= -(diag[3]*u[8] + diag[7]*u[9] + diag[11]*u[10]+ diag[15]*u[11]);

      uik[12]= -(diag[0]*u[12] + diag[4]*u[13] + diag[8]*u[14] + diag[12]*u[15]);
      uik[13]= -(diag[1]*u[12] + diag[5]*u[13] + diag[9]*u[14] + diag[13]*u[15]);
      uik[14]= -(diag[2]*u[12] + diag[6]*u[13] + diag[10]*u[14]+ diag[14]*u[15]);
      uik[15]= -(diag[3]*u[12] + diag[7]*u[13] + diag[11]*u[14]+ diag[15]*u[15]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3];
      dk[1] += uik[4]*u[0] + uik[5]*u[1] + uik[6]*u[2] + uik[7]*u[3];
      dk[2] += uik[8]*u[0] + uik[9]*u[1] + uik[10]*u[2]+ uik[11]*u[3];
      dk[3] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3];

      dk[4] += uik[0]*u[4] + uik[1]*u[5] + uik[2]*u[6] + uik[3]*u[7];
      dk[5] += uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7];
      dk[6] += uik[8]*u[4] + uik[9]*u[5] + uik[10]*u[6]+ uik[11]*u[7];
      dk[7] += uik[12]*u[4]+ uik[13]*u[5]+ uik[14]*u[6]+ uik[15]*u[7];

      dk[8] += uik[0]*u[8] + uik[1]*u[9] + uik[2]*u[10] + uik[3]*u[11];
      dk[9] += uik[4]*u[8] + uik[5]*u[9] + uik[6]*u[10] + uik[7]*u[11];
      dk[10]+= uik[8]*u[8] + uik[9]*u[9] + uik[10]*u[10]+ uik[11]*u[11];
      dk[11]+= uik[12]*u[8]+ uik[13]*u[9]+ uik[14]*u[10]+ uik[15]*u[11];

      dk[12]+= uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15];
      dk[13]+= uik[4]*u[12] + uik[5]*u[13] + uik[6]*u[14] + uik[7]*u[15];
      dk[14]+= uik[8]*u[12] + uik[9]*u[13] + uik[10]*u[14]+ uik[11]*u[15];
      dk[15]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15];
      
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*16,uik,16*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*16;
          u = ba + j*16;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3];
          rtmp_ptr[1] += uik[4]*u[0] + uik[5]*u[1] + uik[6]*u[2] + uik[7]*u[3];
          rtmp_ptr[2] += uik[8]*u[0] + uik[9]*u[1] + uik[10]*u[2]+ uik[11]*u[3];
          rtmp_ptr[3] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3];

          rtmp_ptr[4] += uik[0]*u[4] + uik[1]*u[5] + uik[2]*u[6] + uik[3]*u[7];
          rtmp_ptr[5] += uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7];
          rtmp_ptr[6] += uik[8]*u[4] + uik[9]*u[5] + uik[10]*u[6]+ uik[11]*u[7];
          rtmp_ptr[7] += uik[12]*u[4]+ uik[13]*u[5]+ uik[14]*u[6]+ uik[15]*u[7];

          rtmp_ptr[8] += uik[0]*u[8] + uik[1]*u[9] + uik[2]*u[10] + uik[3]*u[11];
          rtmp_ptr[9] += uik[4]*u[8] + uik[5]*u[9] + uik[6]*u[10] + uik[7]*u[11];
          rtmp_ptr[10]+= uik[8]*u[8] + uik[9]*u[9] + uik[10]*u[10]+ uik[11]*u[11];
          rtmp_ptr[11]+= uik[12]*u[8]+ uik[13]*u[9]+ uik[14]*u[10]+ uik[15]*u[11];

          rtmp_ptr[12]+= uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15];
          rtmp_ptr[13]+= uik[4]*u[12] + uik[5]*u[13] + uik[6]*u[14] + uik[7]*u[15];
          rtmp_ptr[14]+= uik[8]*u[12] + uik[9]*u[13] + uik[10]*u[14]+ uik[11]*u[15];
          rtmp_ptr[15]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15];
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
    diag = ba+k*16;
    ierr = PetscMemcpy(diag,dk,16*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_4(diag);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*16;
         rtmp_ptr = rtmp + vj*16;        
         for (k1=0; k1<16; k1++){
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
  PLogFlops(1.3333*64*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* Version for when blocks are 4 by 4  */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_4"
int MatCholeskyFactorNumeric_SeqSBAIJ_4(Mat A,Mat *B)
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
  rtmp  = (MatScalar*)PetscMalloc(16*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,16*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk    = (MatScalar*)PetscMalloc(32*sizeof(MatScalar));CHKPTRQ(dk);
  uik   = dk + 16;     
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(16*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,16*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<16; k1++){
            dk[k1]       = aa[k*16+k1]; 
            aa[k*16+k1] = aa[j*16+k1]; 
            aa[j*16+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*16;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<16; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<4; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<4; k1++) *ap++ = dk[k + 4*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*16;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        rtmp_ptr = rtmp + vj*16;
        for (i=0; i<16; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*16,16*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*16;
      u    = ba + ili*16;

      uik[0] = -(diag[0]*u[0] + diag[4]*u[1] + diag[8]*u[2] + diag[12]*u[3]);
      uik[1] = -(diag[1]*u[0] + diag[5]*u[1] + diag[9]*u[2] + diag[13]*u[3]);
      uik[2] = -(diag[2]*u[0] + diag[6]*u[1] + diag[10]*u[2]+ diag[14]*u[3]);
      uik[3] = -(diag[3]*u[0] + diag[7]*u[1] + diag[11]*u[2]+ diag[15]*u[3]);

      uik[4] = -(diag[0]*u[4] + diag[4]*u[5] + diag[8]*u[6] + diag[12]*u[7]);
      uik[5] = -(diag[1]*u[4] + diag[5]*u[5] + diag[9]*u[6] + diag[13]*u[7]);
      uik[6] = -(diag[2]*u[4] + diag[6]*u[5] + diag[10]*u[6]+ diag[14]*u[7]);
      uik[7] = -(diag[3]*u[4] + diag[7]*u[5] + diag[11]*u[6]+ diag[15]*u[7]);

      uik[8] = -(diag[0]*u[8] + diag[4]*u[9] + diag[8]*u[10] + diag[12]*u[11]);
      uik[9] = -(diag[1]*u[8] + diag[5]*u[9] + diag[9]*u[10] + diag[13]*u[11]);
      uik[10]= -(diag[2]*u[8] + diag[6]*u[9] + diag[10]*u[10]+ diag[14]*u[11]);
      uik[11]= -(diag[3]*u[8] + diag[7]*u[9] + diag[11]*u[10]+ diag[15]*u[11]);

      uik[12]= -(diag[0]*u[12] + diag[4]*u[13] + diag[8]*u[14] + diag[12]*u[15]);
      uik[13]= -(diag[1]*u[12] + diag[5]*u[13] + diag[9]*u[14] + diag[13]*u[15]);
      uik[14]= -(diag[2]*u[12] + diag[6]*u[13] + diag[10]*u[14]+ diag[14]*u[15]);
      uik[15]= -(diag[3]*u[12] + diag[7]*u[13] + diag[11]*u[14]+ diag[15]*u[15]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3];
      dk[1] += uik[4]*u[0] + uik[5]*u[1] + uik[6]*u[2] + uik[7]*u[3];
      dk[2] += uik[8]*u[0] + uik[9]*u[1] + uik[10]*u[2]+ uik[11]*u[3];
      dk[3] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3];

      dk[4] += uik[0]*u[4] + uik[1]*u[5] + uik[2]*u[6] + uik[3]*u[7];
      dk[5] += uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7];
      dk[6] += uik[8]*u[4] + uik[9]*u[5] + uik[10]*u[6]+ uik[11]*u[7];
      dk[7] += uik[12]*u[4]+ uik[13]*u[5]+ uik[14]*u[6]+ uik[15]*u[7];

      dk[8] += uik[0]*u[8] + uik[1]*u[9] + uik[2]*u[10] + uik[3]*u[11];
      dk[9] += uik[4]*u[8] + uik[5]*u[9] + uik[6]*u[10] + uik[7]*u[11];
      dk[10]+= uik[8]*u[8] + uik[9]*u[9] + uik[10]*u[10]+ uik[11]*u[11];
      dk[11]+= uik[12]*u[8]+ uik[13]*u[9]+ uik[14]*u[10]+ uik[15]*u[11];

      dk[12]+= uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15];
      dk[13]+= uik[4]*u[12] + uik[5]*u[13] + uik[6]*u[14] + uik[7]*u[15];
      dk[14]+= uik[8]*u[12] + uik[9]*u[13] + uik[10]*u[14]+ uik[11]*u[15];
      dk[15]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15];
      
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*16,uik,16*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*16;
          u = ba + j*16;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3];
          rtmp_ptr[1] += uik[4]*u[0] + uik[5]*u[1] + uik[6]*u[2] + uik[7]*u[3];
          rtmp_ptr[2] += uik[8]*u[0] + uik[9]*u[1] + uik[10]*u[2]+ uik[11]*u[3];
          rtmp_ptr[3] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3];

          rtmp_ptr[4] += uik[0]*u[4] + uik[1]*u[5] + uik[2]*u[6] + uik[3]*u[7];
          rtmp_ptr[5] += uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7];
          rtmp_ptr[6] += uik[8]*u[4] + uik[9]*u[5] + uik[10]*u[6]+ uik[11]*u[7];
          rtmp_ptr[7] += uik[12]*u[4]+ uik[13]*u[5]+ uik[14]*u[6]+ uik[15]*u[7];

          rtmp_ptr[8] += uik[0]*u[8] + uik[1]*u[9] + uik[2]*u[10] + uik[3]*u[11];
          rtmp_ptr[9] += uik[4]*u[8] + uik[5]*u[9] + uik[6]*u[10] + uik[7]*u[11];
          rtmp_ptr[10]+= uik[8]*u[8] + uik[9]*u[9] + uik[10]*u[10]+ uik[11]*u[11];
          rtmp_ptr[11]+= uik[12]*u[8]+ uik[13]*u[9]+ uik[14]*u[10]+ uik[15]*u[11];

          rtmp_ptr[12]+= uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15];
          rtmp_ptr[13]+= uik[4]*u[12] + uik[5]*u[13] + uik[6]*u[14] + uik[7]*u[15];
          rtmp_ptr[14]+= uik[8]*u[12] + uik[9]*u[13] + uik[10]*u[14]+ uik[11]*u[15];
          rtmp_ptr[15]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15];
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
    diag = ba+k*16;
    ierr = PetscMemcpy(diag,dk,16*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_4(diag);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*16;
         rtmp_ptr = rtmp + vj*16;        
         for (k1=0; k1<16; k1++){
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
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PLogFlops(1.3333*64*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* Version for when blocks are 3 by 3  */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_3"
int MatCholeskyFactorNumeric_SeqSBAIJ_3(Mat A,Mat *B)
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
  rtmp  = (MatScalar*)PetscMalloc(9*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,9*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk  = (MatScalar*)PetscMalloc(18*sizeof(MatScalar));CHKPTRQ(dk);
  uik = dk + 9;     
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(9*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,9*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<9; k1++){
            dk[k1]       = aa[k*9+k1]; 
            aa[k*9+k1] = aa[j*9+k1]; 
            aa[j*9+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*9;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<9; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<3; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<3; k1++) *ap++ = dk[k + 3*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*9;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        rtmp_ptr = rtmp + vj*9;
        for (i=0; i<9; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*9,9*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*9;
      u    = ba + ili*9;

      uik[0] = -(diag[0]*u[0] + diag[3]*u[1] + diag[6]*u[2]);
      uik[1] = -(diag[1]*u[0] + diag[4]*u[1] + diag[7]*u[2]);
      uik[2] = -(diag[2]*u[0] + diag[5]*u[1] + diag[8]*u[2]);

      uik[3] = -(diag[0]*u[3] + diag[3]*u[4] + diag[6]*u[5]);
      uik[4] = -(diag[1]*u[3] + diag[4]*u[4] + diag[7]*u[5]);
      uik[5] = -(diag[2]*u[3] + diag[5]*u[4] + diag[8]*u[5]);

      uik[6] = -(diag[0]*u[6] + diag[3]*u[7] + diag[6]*u[8]);
      uik[7] = -(diag[1]*u[6] + diag[4]*u[7] + diag[7]*u[8]);
      uik[8] = -(diag[2]*u[6] + diag[5]*u[7] + diag[8]*u[8]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2];
      dk[1] += uik[3]*u[0] + uik[4]*u[1] + uik[5]*u[2];
      dk[2] += uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2];

      dk[3] += uik[0]*u[3] + uik[1]*u[4] + uik[2]*u[5];
      dk[4] += uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
      dk[5] += uik[6]*u[3] + uik[7]*u[4] + uik[8]*u[5];

      dk[6] += uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8];
      dk[7] += uik[3]*u[6] + uik[4]*u[7] + uik[5]*u[8];
      dk[8] += uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*9,uik,9*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*9;
          u = ba + j*9;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2];
          rtmp_ptr[1] += uik[3]*u[0] + uik[4]*u[1] + uik[5]*u[2];
          rtmp_ptr[2] += uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2];

          rtmp_ptr[3] += uik[0]*u[3] + uik[1]*u[4] + uik[2]*u[5];
          rtmp_ptr[4] += uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
          rtmp_ptr[5] += uik[6]*u[3] + uik[7]*u[4] + uik[8]*u[5];

          rtmp_ptr[6] += uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8];
          rtmp_ptr[7] += uik[3]*u[6] + uik[4]*u[7] + uik[5]*u[8];
          rtmp_ptr[8] += uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8];
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
    diag = ba+k*9;
    ierr = PetscMemcpy(diag,dk,9*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_3(diag);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*9;
         rtmp_ptr = rtmp + vj*9;        
         for (k1=0; k1<9; k1++){
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
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  PLogFlops(1.3333*27*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 3 by 3 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*diag,*rtmp,*rtmp_ptr;

  PetscFunctionBegin;
  
  /* initialization */
  rtmp  = (MatScalar*)PetscMalloc(9*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,9*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk  = (MatScalar*)PetscMalloc(18*sizeof(MatScalar));CHKPTRQ(dk);
  uik = dk + 9;   

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*9;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        rtmp_ptr = rtmp + vj*9;
        for (i=0; i<9; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*9,9*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*9;
      u    = ba + ili*9;

      uik[0] = -(diag[0]*u[0] + diag[3]*u[1] + diag[6]*u[2]);
      uik[1] = -(diag[1]*u[0] + diag[4]*u[1] + diag[7]*u[2]);
      uik[2] = -(diag[2]*u[0] + diag[5]*u[1] + diag[8]*u[2]);

      uik[3] = -(diag[0]*u[3] + diag[3]*u[4] + diag[6]*u[5]);
      uik[4] = -(diag[1]*u[3] + diag[4]*u[4] + diag[7]*u[5]);
      uik[5] = -(diag[2]*u[3] + diag[5]*u[4] + diag[8]*u[5]);

      uik[6] = -(diag[0]*u[6] + diag[3]*u[7] + diag[6]*u[8]);
      uik[7] = -(diag[1]*u[6] + diag[4]*u[7] + diag[7]*u[8]);
      uik[8] = -(diag[2]*u[6] + diag[5]*u[7] + diag[8]*u[8]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2];
      dk[1] += uik[3]*u[0] + uik[4]*u[1] + uik[5]*u[2];
      dk[2] += uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2];

      dk[3] += uik[0]*u[3] + uik[1]*u[4] + uik[2]*u[5];
      dk[4] += uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
      dk[5] += uik[6]*u[3] + uik[7]*u[4] + uik[8]*u[5];

      dk[6] += uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8];
      dk[7] += uik[3]*u[6] + uik[4]*u[7] + uik[5]*u[8];
      dk[8] += uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*9,uik,9*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*9;
          u = ba + j*9;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2];
          rtmp_ptr[1] += uik[3]*u[0] + uik[4]*u[1] + uik[5]*u[2];
          rtmp_ptr[2] += uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2];

          rtmp_ptr[3] += uik[0]*u[3] + uik[1]*u[4] + uik[2]*u[5];
          rtmp_ptr[4] += uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
          rtmp_ptr[5] += uik[6]*u[3] + uik[7]*u[4] + uik[8]*u[5];

          rtmp_ptr[6] += uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8];
          rtmp_ptr[7] += uik[3]*u[6] + uik[4]*u[7] + uik[5]*u[8];
          rtmp_ptr[8] += uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8];
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
    diag = ba+k*9;
    ierr = PetscMemcpy(diag,dk,9*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_3(diag);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*9;
         rtmp_ptr = rtmp + vj*9;        
         for (k1=0; k1<9; k1++){
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
  PLogFlops(1.3333*27*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
    Numeric U^T*D*U factorization for SBAIJ format. Modified from SNF of YSMP.
    Version for blocks 2 by 2.
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_2"
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
  rtmp  = (MatScalar*)PetscMalloc(4*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,4*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk  = (MatScalar*)PetscMalloc(8*sizeof(MatScalar));CHKPTRQ(dk);
  uik = dk + 4;     
  ierr  = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    aa = (MatScalar*)PetscMalloc(4*ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,4*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    a2anew  = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(a2anew); 
    ierr= PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

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
    ierr = PetscFree(a2anew);CHKERRA(ierr); 
  }

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*4;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        rtmp_ptr = rtmp + vj*4;
        for (i=0; i<4; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*4,4*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
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
  PLogFlops(1.3333*8*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 2 by 2 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering"
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
  rtmp  = (MatScalar*)PetscMalloc(4*mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  ierr = PetscMemzero(rtmp,4*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  dk  = (MatScalar*)PetscMalloc(8*sizeof(MatScalar));CHKPTRQ(dk);
  uik = dk + 4;     
   
  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*4;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        rtmp_ptr = rtmp + vj*4;
        for (i=0; i<4; i++) *rtmp_ptr++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*4,4*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
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
  PLogFlops(1.3333*8*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
    Numeric U^T*D*U factorization for SBAIJ format. Modified from SNF of YSMP.
    Version for blocks are 1 by 1.
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_1"
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
    aa = (MatScalar*)PetscMalloc(ai[mbs]*sizeof(MatScalar));CHKPTRQ(aa); 
    ierr = PetscMemcpy(aa,a->a,ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    r   = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(r); 
    ierr= PetscMemcpy(r,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    jmin = ai[0]; jmax = ai[mbs];
    for (j=jmin; j<jmax; j++){
      while (r[j] != j){  
        k = r[j]; r[j] = r[k]; r[k] = k;     
        ak = aa[k]; aa[k] = aa[j]; aa[j] = ak;         
      }
    }
    ierr = PetscFree(r);CHKERRA(ierr); 
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
  rtmp  = (MatScalar*)PetscMalloc(mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  il = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
  }

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
    if (jmin < jmax) {
      for (j = jmin; j < jmax; j++){
        vj = rip[aj[j]];
        rtmp[vj] = aa[j];
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    dk = rtmp[k];
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
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
    }else if (PetscRealPart(dk) < 0){
      ierr = PetscPrintf(PETSC_COMM_SELF,"Negative pivot: d[%d] = %g\n",k,dk);
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
  PLogFlops(b->mbs);
  PetscFunctionReturn(0);
}

/*
  Version for when blocks are 1 by 1 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering"
int MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  int                ierr,i,j,mbs = a->mbs,*bi = b->i,*bj = b->j;
  int                *ai,*aj,*r;
  int                k,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *rtmp;
  MatScalar          *ba = b->a,*aa,ak;
  MatScalar          dk,uikdi;

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
  rtmp  = (MatScalar*)PetscMalloc(mbs*sizeof(MatScalar));CHKPTRQ(rtmp);
  il    = (int*)PetscMalloc(2*mbs*sizeof(int));CHKPTRQ(il);
  jl    = il + mbs;
  for (i=0; i<mbs; i++) {
    rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
  }

  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      for (j = jmin; j < jmax; j++){
        vj = aj[j];
        rtmp[vj] = aa[j];
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    dk = rtmp[k];
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
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
    }else if (PetscRealPart(dk) < 0){
      ierr = PetscPrintf(PETSC_COMM_SELF,"Negative pivot: d[%d] = %g\n",k,dk);
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
  PLogFlops(b->mbs);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactor_SeqSBAIJ"
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


