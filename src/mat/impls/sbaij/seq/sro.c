/*$Id: sro.c,v 1.10 2000/09/11 19:42:15 balay Exp hzhang $*/

#include "petscsys.h"
#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "sbaij.h"   

/* 
This function is used before applying a 
symmetric reordering to matrix A that is 
in SBAIJ format. 

The permutation is assumed to be symmetric, i.e., 
P = P^T (= inv(P)),
so the permuted matrix P*A*inv(P)=P*A*P^T is ensured to be symmetric.

The function is modified from sro.f of YSMP. The description from YSMP:
C    THE NONZERO ENTRIES OF THE MATRIX M ARE ASSUMED TO BE STORED
C    SYMMETRICALLY IN (IA,JA,A) FORMAT (I.E., NOT BOTH M(I,J) AND M(J,I)
C    ARE STORED IF I NE J).
C
C    SRO DOES NOT REARRANGE THE ORDER OF THE ROWS, BUT DOES MOVE
C    NONZEROES FROM ONE ROW TO ANOTHER TO ENSURE THAT IF M(I,J) WILL BE
C    IN THE UPPER TRIANGLE OF M WITH RESPECT TO THE NEW ORDERING, THEN
C    M(I,J) IS STORED IN ROW I (AND THUS M(J,I) IS NOT STORED);  WHEREAS
C    IF M(I,J) WILL BE IN THE STRICT LOWER TRIANGLE OF M, THEN M(J,I) IS
C    STORED IN ROW J (AND THUS M(I,J) IS NOT STORED).   


  -- output: new index set (ai, aj, a) for A such that all 
             nonzero A_(p(i),isp(k)) will be stored in the upper triangle.
             Note: matrix A is not permuted by this function!
*/
#undef __FUNC__  
#define __FUNC__ "MatReorderingSeqSBAIJ"
int MatReorderingSeqSBAIJ(Mat A,IS perm)
{
  Mat_SeqSBAIJ     *a=(Mat_SeqSBAIJ *)A->data;
  int             *r,ierr,i,mbs=a->mbs,*ai=a->i,*aj=a->j,*rip,*riip;
  MatScalar       *aa=a->a;
  Scalar          ak;
  int             *nzr,nz,jmin,jmax,j,k,ajk;
  IS              iperm;  /* inverse of perm */

  PetscFunctionBegin;
  if (!mbs) PetscFunctionReturn(0);
  
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);
  
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr); 
  
  ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);

  for (i=0; i<mbs; i++) {
    if (rip[i] - riip[i] != 0) SETERRQ(1,1,"Non-symm. permutation, use symm. permutation or general matrix format");     
  }
  ierr = ISRestoreIndices(iperm,&riip);CHKERRA(ierr);
  ierr = ISDestroy(iperm);CHKERRA(ierr);
  

  /* Phase 1: find row in which to store each nonzero (r)
	      initialize count of nonzeros to be stored in each row (nzr) */

  nzr = (int*)PetscMalloc(mbs*sizeof(int));CHKPTRQ(nzr); 
  r   = (int*)PetscMalloc(ai[mbs]*sizeof(int));CHKPTRQ(r);
  for (i=0; i<mbs; i++) nzr[i] = 0;
  for (i=0; i<ai[mbs]; i++) r[i] = 0; 
                                                              
  /*  for each nonzero element */
  for (i=0; i<mbs; i++){
    nz = ai[i+1] - ai[i]; 
    j = ai[i];
    /* printf("nz = %d, j=%d\n",nz,j); */
    while (nz--){
      /*  --- find row (=r[j]) and column (=aj[j]) in which to store a[j] ...*/
      k = aj[j];
      /* printf("nz = %d, k=%d\n", nz,k); */

      if (rip[k] < rip[i]) aj[j] = i;    
      else k = i; 
      
      r[j] = k; j++;
      nzr[k] ++; /* increment count of nonzeros in that row */
    } 
  } 

  /* Phase 2: find new ai and permutation to apply to (aj,a)
              determine pointers (r) to delimit rows in permuted (aj,a) */
    for (i=0; i<mbs; i++){
      ai[i+1] = ai[i] + nzr[i]; 
      nzr[i]    = ai[i+1];
    }
                                                     
  /* determine where each (aj[j], a[j]) is stored in permuted (aj,a)
     for each nonzero element (in reverse order) */
  jmin = ai[0]; jmax = ai[mbs];
  nz = jmax - jmin;
  j = jmax-1;
  while (nz--){
    i = r[j];  /* row value */
    if (aj[j] == i) r[j] = ai[i]; /* put diagonal nonzero at beginning of row */
    else { /* put off-diagonal nonzero in last unused location in row */
      nzr[i]--; r[j] = nzr[i];
    }
    j--;
  }         
                                                
  /* Phase 3: permute (aj,a) to upper triangular form (wrt new ordering) */
  for (j=jmin; j<jmax; j++){
    while (r[j] != j){ 
      k = r[j]; r[j] = r[k]; r[k] = k;
      ajk = aj[k]; aj[k] = aj[j]; aj[j] = ajk;
      ak = aa[k]; aa[k] = aa[j]; aa[j] = ak;
    }
  }

  a->row  = perm;
  a->icol = perm;
  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);

  ierr= ISRestoreIndices(perm,&rip);CHKERRA(ierr);

  ierr = PetscFree(nzr);CHKERRA(ierr); 
  ierr = PetscFree(r);CHKERRA(ierr); 
  
  PetscFunctionReturn(0);
}

