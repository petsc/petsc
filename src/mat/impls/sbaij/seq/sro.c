#define PETSCMAT_DLL

#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/impls/sbaij/seq/sbaij.h"   

/* 
This function is used before applying a 
symmetric reordering to matrix A that is 
in SBAIJ format. 

The permutation is assumed to be symmetric, i.e., 
P = P^T (= inv(P)),
so the permuted matrix P*A*inv(P)=P*A*P^T is ensured to be symmetric. 
 - a wrong assumption! This code needs rework!  -- Hong

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


  -- output: new index set (inew, jnew) for A and a map a2anew that maps
             values a to anew, such that all 
             nonzero A_(perm(i),iperm(k)) will be stored in the upper triangle.
             Note: matrix A is not permuted by this function!
*/
#undef __FUNCT__  
#define __FUNCT__ "MatReorderingSeqSBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatReorderingSeqSBAIJ(Mat A,IS perm)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ *)A->data;
  PetscErrorCode ierr;
  const PetscInt mbs=a->mbs,*rip,*riip;
  PetscInt       *ai,*aj,*r;
  PetscInt       *nzr,nz,jmin,jmax,j,k,ajk,i;
  IS             iperm;  /* inverse of perm */

  PetscFunctionBegin;
  if (!mbs) PetscFunctionReturn(0); 
  SETERRQ(PETSC_ERR_SUP,"Matrix reordering is not supported for sbaij matrix. Use aij format");
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);  
  ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);

  for (i=0; i<mbs; i++) {
    if (rip[i] != riip[i]) SETERRQ(PETSC_ERR_ARG_INCOMP,"Non-symmetric permutation, use symmetric permutation for symmetric matrices");
  }
  ierr = ISRestoreIndices(iperm,&riip);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);

  if (!a->inew){ 
    ierr = PetscMalloc2(mbs+1,PetscInt,&ai, 2*a->i[mbs],PetscInt,&aj);CHKERRQ(ierr);
  } else {
    ai = a->inew; aj = a->jnew;
  }  
  ierr  = PetscMemcpy(ai,a->i,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr  = PetscMemcpy(aj,a->j,(a->i[mbs])*sizeof(PetscInt));CHKERRQ(ierr);
  
  /* 
     Phase 1: Find row index r in which to store each nonzero. 
	      Initialize count of nonzeros to be stored in each row (nzr).
              At the end of this phase, a nonzero a(*,*)=a(r(),aj())
              s.t. a(perm(r),perm(aj)) will fall into upper triangle part.
  */

  ierr = PetscMalloc(mbs*sizeof(PetscInt),&nzr);CHKERRQ(ierr); 
  ierr = PetscMalloc(ai[mbs]*sizeof(PetscInt),&r);CHKERRQ(ierr); 
  for (i=0; i<mbs; i++) nzr[i] = 0;
  for (i=0; i<ai[mbs]; i++) r[i] = 0; 
                                                              
  /*  for each nonzero element */
  for (i=0; i<mbs; i++){
    nz = ai[i+1] - ai[i]; 
    j = ai[i];
    /* printf("nz = %d, j=%d\n",nz,j); */
    while (nz--){
      /*  --- find row (=r[j]) and column (=aj[j]) in which to store a[j] ...*/
      k = aj[j];                          /* col. index */
      /* printf("nz = %d, k=%d\n", nz,k); */
      /* for entry that will be permuted into lower triangle, swap row and col. index */
      if (rip[k] < rip[i]) aj[j] = i; 
      else k = i; 
      
      r[j] = k; j++;
      nzr[k] ++; /* increment count of nonzeros in that row */
    } 
  } 

  /* Phase 2: Find new ai and permutation to apply to (aj,a).
              Determine pointers (r) to delimit rows in permuted (aj,a).
              Note: r is different from r used in phase 1.
              At the end of this phase, (aj[j],a[j]) will be stored in
              (aj[r(j)],a[r(j)]).
  */
    for (i=0; i<mbs; i++){
      ai[i+1] = ai[i] + nzr[i]; 
      nzr[i]    = ai[i+1];
    }
                                                     
  /* determine where each (aj[j], a[j]) is stored in new (aj,a)
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
  
  a->a2anew = aj + ai[mbs];
  ierr  = PetscMemcpy(a->a2anew,r,ai[mbs]*sizeof(PetscInt));CHKERRQ(ierr);
                                         
  /* Phase 3: permute (aj,a) to upper triangular form (wrt new ordering) */
  for (j=jmin; j<jmax; j++){
    while (r[j] != j){ 
      k = r[j]; r[j] = r[k]; r[k] = k;
      ajk = aj[k]; aj[k] = aj[j]; aj[j] = ajk;
      /* ak = aa[k]; aa[k] = aa[j]; aa[j] = ak; */
    }
  }
  ierr= ISRestoreIndices(perm,&rip);CHKERRQ(ierr);

  a->inew = ai;
  a->jnew = aj;

  if (a->row) {
    ierr = ISDestroy(a->row);CHKERRQ(ierr);
  }
  if (a->icol) {
    ierr = ISDestroy(a->icol);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  if (a->row) { ierr = ISDestroy(a->row);CHKERRQ(ierr); }
  a->row  = perm;
  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  if (a->icol) { ierr = ISDestroy(a->icol);CHKERRQ(ierr); }
  a->icol = perm;

  ierr = PetscFree(nzr);CHKERRQ(ierr); 
  ierr = PetscFree(r);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}


