/*$Id: aijfact.c,v 1.139 1999/12/18 00:44:57 bsmith Exp bsmith $*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/dot.h"

#undef __FUNC__  
#define __FUNC__ "MatOrdering_Flow_SeqAIJ"
int MatOrdering_Flow_SeqAIJ(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscFunctionBegin;

  SETERRQ(PETSC_ERR_SUP,0,"Code not written");
#if !defined(PETSC_USE_DEBUG)
  PetscFunctionReturn(0);
#endif
}


extern int MatMarkDiagonal_SeqAIJ(Mat);
extern int Mat_AIJ_CheckInode(Mat);

extern int SPARSEKIT2dperm(int*,Scalar*,int*,int*,Scalar*,int*,int*,int*,int*,int*);
extern int SPARSEKIT2ilutp(int*,Scalar*,int*,int*,int*,double*,double*,int*,Scalar*,int*,int*,int*,Scalar*,int*,int*,int*);
extern int SPARSEKIT2msrcsr(int*,Scalar*,int*,Scalar*,int*,int*,Scalar*,int*);

#undef __FUNC__  
#define __FUNC__ "MatILUDTFactor_SeqAIJ"
  /* ------------------------------------------------------------

          This interface was contribed by Tony Caola

     This routine is an interface to the pivoting drop-tolerance 
     ILU routine written by Yousef Saad (saad@cs.umn.edu) as part of 
     SPARSEKIT2.

     The SPARSEKIT2 routines used here are covered by the GNU 
     copyright; see the file gnu in this directory.

     Thanks to Prof. Saad, Dr. Hysom, and Dr. Smith for their
     help in getting this routine ironed out.

     The major drawback to this routine is that if info->fill is 
     not large enough it fails rather than allocating more space;
     this can be fixed by hacking/improving the f2c version of 
     Yousef Saad's code.

     ishift = 0, for indices start at 1
     ishift = 1, for indices starting at 0
     ------------------------------------------------------------
  */

int MatILUDTFactor_SeqAIJ(Mat A,MatILUInfo *info,IS isrow,IS iscol,Mat *fact)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data, *b;
  IS         iscolf, isicol, isirow;
  PetscTruth reorder;
  int        *c,*r,*ic,ierr, i, n = a->m;
  int        *old_i = a->i, *old_j = a->j, *new_i, *old_i2, *old_j2,*new_j;
  int        *ordcol, *iwk,*iperm, *jw;
  int        ishift = !a->indexshift;
  int        jmax,lfill,job,*o_i,*o_j;
  Scalar     *old_a = a->a, *w, *new_a, *old_a2, *wk,*o_a;
  double     permtol,af;

  PetscFunctionBegin;

  if (info->dt == PETSC_DEFAULT)      info->dt      = .005;
  if (info->dtcount == PETSC_DEFAULT) info->dtcount = (int) (1.5*a->rmax); 
  if (info->dtcol == PETSC_DEFAULT)   info->dtcol   = .01;
  if (info->fill == PETSC_DEFAULT)    info->fill    = ((double)(n*info->dtcount))/a->nz;
  lfill   = (int) (info->dtcount/2.0);
  jmax    = (int) (info->fill*a->nz);
  permtol = info->dtcol;


  /* ------------------------------------------------------------
     If reorder=.TRUE., then the original matrix has to be 
     reordered to reflect the user selected ordering scheme, and
     then de-reordered so it is in it's original format.  
     Because Saad's dperm() is NOT in place, we have to copy 
     the original matrix and allocate more storage. . . 
     ------------------------------------------------------------
  */

  /* set reorder to true if either isrow or iscol is not identity */
  ierr = ISIdentity(isrow,&reorder);CHKERRQ(ierr);
  if (reorder) {ierr = ISIdentity(iscol,&reorder);CHKERRQ(ierr);}
  reorder = PetscNot(reorder);

  
  /* storage for ilu factor */
  new_i = (int *)    PetscMalloc((n+1)*sizeof(int));   CHKPTRQ(new_i);
  new_j = (int *)    PetscMalloc(jmax*sizeof(int));    CHKPTRQ(new_j);
  new_a = (Scalar *) PetscMalloc(jmax*sizeof(Scalar)); CHKPTRQ(new_a);

  ordcol = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(ordcol);

  /* ------------------------------------------------------------
     Make sure that everything is Fortran formatted (1-Based)
     ------------------------------------------------------------
  */
  if (ishift) {
    for (i=old_i[0];i<old_i[n];i++) {
      old_j[i]++;
    }
    for(i=0;i<n+1;i++) {
      old_i[i]++;
    };
  }; 

  if (reorder) {
    ierr = ISGetIndices(iscol,&c);           CHKERRQ(ierr);
    ierr = ISGetIndices(isrow,&r);           CHKERRQ(ierr);
    for(i=0;i<n;i++) {
      r[i]  = r[i]+1;
      c[i]  = c[i]+1;
    }
    old_i2 = (int *) PetscMalloc((n+1)*sizeof(int)); CHKPTRQ(old_i2);
    old_j2 = (int *) PetscMalloc((old_i[n]-old_i[0]+1)*sizeof(int)); CHKPTRQ(old_j2);
    old_a2 = (Scalar *) PetscMalloc((old_i[n]-old_i[0]+1)*sizeof(Scalar));CHKPTRQ(old_a2);
    job = 3; SPARSEKIT2dperm(&n,old_a,old_j,old_i,old_a2,old_j2,old_i2,r,c,&job);
    for (i=0;i<n;i++) {
      r[i]  = r[i]-1;
      c[i]  = c[i]-1;
    }
    ierr = ISRestoreIndices(iscol,&c); CHKERRQ(ierr);
    ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
    o_a = old_a2;
    o_j = old_j2;
    o_i = old_i2;
  } else {
    o_a = old_a;
    o_j = old_j;
    o_i = old_i;
  }

  /* ------------------------------------------------------------
     Call Saad's ilutp() routine to generate the factorization
     ------------------------------------------------------------
  */

  iperm   = (int *)    PetscMalloc(2*n*sizeof(int)); CHKPTRQ(iperm);
  jw      = (int *)    PetscMalloc(2*n*sizeof(int)); CHKPTRQ(jw);
  w       = (Scalar *) PetscMalloc(n*sizeof(Scalar)); CHKPTRQ(w);

  SPARSEKIT2ilutp(&n,o_a,o_j,o_i,&lfill,&info->dt,&permtol,&n,new_a,new_j,new_i,&jmax,w,jw,iperm,&ierr); 
  if (ierr) {
    switch (ierr) {
      case -3: SETERRQ1(1,1,"ilutp(), matrix U overflows, need larger info->fill value %d",jmax);
      case -2: SETERRQ1(1,1,"ilutp(), matrix L overflows, need larger info->fill value %d",jmax);
      case -5: SETERRQ(1,1,"ilutp(), zero row encountered");
      case -1: SETERRQ(1,1,"ilutp(), input matrix may be wrong");
      case -4: SETERRQ1(1,1,"ilutp(), illegal info->fill value %d",jmax);
      default: SETERRQ1(1,1,"ilutp(), zero pivot detected on row %d",ierr);
    }
  }

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree(jw);CHKERRQ(ierr);

  /* ------------------------------------------------------------
     Saad's routine gives the result in Modified Sparse Row (msr)
     Convert to Compressed Sparse Row format (csr) 
     ------------------------------------------------------------
  */

  wk  = (Scalar *)    PetscMalloc(n*sizeof(Scalar)); CHKPTRQ(wk);   
  iwk = (int *) PetscMalloc((n+1)*sizeof(int)); CHKPTRQ(iwk);

  SPARSEKIT2msrcsr(&n,new_a,new_j,new_a,new_j,new_i,wk,iwk);

  ierr = PetscFree(iwk);CHKERRQ(ierr);
  ierr = PetscFree(wk);CHKERRQ(ierr);

  if (reorder) {
    ierr = PetscFree(old_a2);CHKERRQ(ierr);
    ierr = PetscFree(old_j2);CHKERRQ(ierr);
    ierr = PetscFree(old_i2);CHKERRQ(ierr);
  } else {
    /* fix permutation of old_j that the factorization introduced */
    for (i=old_i[0]; i<=old_i[n]; i++) {
      old_j[i-1] = iperm[old_j[i-1]-1]; 
    }
  }

  /* get rid of the shift to indices starting at 1 */
  if (ishift) {
    for (i=0; i<n+1; i++) {
      old_i[i]--;
    }
    for (i=old_i[0];i<old_i[n];i++) {
      old_j[i]--;
    }
  }

  /* Make the factored matrix 0-based */
  if (ishift) {
    for (i=0; i<n+1; i++) {
      new_i[i]--;
    }
    for (i=new_i[0];i<new_i[n];i++) {
      new_j[i]--;
    }
  }

  /*-- due to the pivoting, we need to reorder iscol to correctly --*/
  /*-- permute the right-hand-side and solution vectors           --*/
  ierr = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  ierr = ISInvertPermutation(isrow,&isirow); CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);          CHKERRQ(ierr);
  for(i=0; i<n; i++) {
    ordcol[i] = ic[iperm[i]-1];  
  };       
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISDestroy(isicol);CHKERRQ(ierr);

  ierr = PetscFree(iperm);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, ordcol, &iscolf); 
  ierr = PetscFree(ordcol);CHKERRQ(ierr);

  /*----- put together the new matrix -----*/

  ierr = MatCreateSeqAIJ(A->comm,n,n,0,PETSC_NULL,fact); CHKERRQ(ierr);
  (*fact)->factor    = FACTOR_LU;
  (*fact)->assembled = PETSC_TRUE;

  b = (Mat_SeqAIJ *) (*fact)->data;
  ierr = PetscFree(b->imax);CHKERRQ(ierr);
  b->sorted        = PETSC_FALSE;
  b->singlemalloc  = PETSC_FALSE;
  /* the next line frees the default space generated by the MatCreate() */
  ierr             = PetscFree(b->a);CHKERRQ(ierr);
  ierr             = PetscFree(b->ilen);CHKERRQ(ierr);
  b->a             = new_a;
  b->j             = new_j;
  b->i             = new_i;
  b->ilen          = 0;
  b->imax          = 0;
  /*  I am not sure why these are the inverses of the row and column permutations; but the other way is NO GOOD */
  b->row           = isirow;
  b->col           = iscolf;
  b->solve_work    =  (Scalar *) PetscMalloc( (n+1)*sizeof(Scalar));CHKPTRQ(b->solve_work);
  b->maxnz = b->nz = new_i[n];
  b->indexshift    = a->indexshift;
  ierr = MatMarkDiagonal_SeqAIJ(*fact);CHKERRQ(ierr);
  (*fact)->info.factor_mallocs = 0;

  ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);

  /* check out for identical nodes. If found, use inode functions */
  ierr = Mat_AIJ_CheckInode(*fact);CHKERRQ(ierr);

  af = ((double)b->nz)/((double)a->nz) + .001;
  PLogInfo(A,"MatILUDTFactor_SeqAIJ:Fill ratio:given %g needed %g\n",info->fill,af);
  PLogInfo(A,"MatILUDTFactor_SeqAIJ:Run with -pc_ilu_fill %g or use \n",af);
  PLogInfo(A,"MatILUDTFactor_SeqAIJ:PCILUSetFill(pc,%g);\n",af);
  PLogInfo(A,"MatILUDTFactor_SeqAIJ:for best performance.\n");

  PetscFunctionReturn(0);
}

/*
    Factorization code for AIJ format. 
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorSymbolic_SeqAIJ"
int MatLUFactorSymbolic_SeqAIJ(Mat A,IS isrow,IS iscol,double f,Mat *B)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data, *b;
  IS         isicol;
  int        *r,*ic, ierr, i, n = a->m, *ai = a->i, *aj = a->j;
  int        *ainew,*ajnew, jmax,*fill, *ajtmp, nz,shift = a->indexshift;
  int        *idnew, idx, row,m,fm, nnz, nzi, realloc = 0,nzbd,*im;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(isrow,IS_COOKIE);
  PetscValidHeaderSpecific(iscol,IS_COOKIE);
  if (A->M != A->N) SETERRQ(PETSC_ERR_ARG_WRONG,0,"matrix must be square");

  ierr = ISInvertPermutation(iscol,&isicol);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* get new row pointers */
  ainew    = (int *) PetscMalloc( (n+1)*sizeof(int) );CHKPTRQ(ainew);
  ainew[0] = -shift;
  /* don't know how many column pointers are needed so estimate */
  jmax  = (int) (f*ai[n]+(!shift));
  ajnew = (int *) PetscMalloc( (jmax)*sizeof(int) );CHKPTRQ(ajnew);
  /* fill is a linked list of nonzeros in active row */
  fill = (int *) PetscMalloc( (2*n+1)*sizeof(int));CHKPTRQ(fill);
  im   = fill + n + 1;
  /* idnew is location of diagonal in factor */
  idnew    = (int *) PetscMalloc( (n+1)*sizeof(int));CHKPTRQ(idnew);
  idnew[0] = -shift;

  for ( i=0; i<n; i++ ) {
    /* first copy previous fill into linked list */
    nnz     = nz    = ai[r[i]+1] - ai[r[i]];
    if (!nz) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,1,"Empty row in matrix");
    ajtmp   = aj + ai[r[i]] + shift;
    fill[n] = n;
    while (nz--) {
      fm  = n;
      idx = ic[*ajtmp++ + shift];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
    }
    row = fill[n];
    while ( row < i ) {
      ajtmp = ajnew + idnew[row] + (!shift);
      nzbd  = 1 + idnew[row] - ainew[row];
      nz    = im[row] - nzbd;
      fm    = row;
      while (nz-- > 0) {
        idx = *ajtmp++ + shift;
        nzbd++;
        if (idx == i) im[row] = nzbd;
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          fill[m]   = idx;
          fill[idx] = fm;
          fm        = idx;
          nnz++;
        }
      }
      row = fill[row];
    }
    /* copy new filled row into permanent storage */
    ainew[i+1] = ainew[i] + nnz;
    if (ainew[i+1] > jmax) {

      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      int maxadd = jmax;
      /* maxadd = (int) ((f*(ai[n]+(!shift))*(n-i+5))/n); */
      if (maxadd < nnz) maxadd = (n-i)*(nnz+1);
      jmax += maxadd;

      /* allocate a longer ajnew */
      ajtmp = (int *) PetscMalloc( jmax*sizeof(int) );CHKPTRQ(ajtmp);
      ierr  = PetscMemcpy(ajtmp,ajnew,(ainew[i]+shift)*sizeof(int));CHKERRQ(ierr);
      ierr  = PetscFree(ajnew);CHKERRQ(ierr);
      ajnew = ajtmp;
      realloc++; /* count how many times we realloc */
    }
    ajtmp = ajnew + ainew[i] + shift;
    fm    = fill[n];
    nzi   = 0;
    im[i] = nnz;
    while (nnz--) {
      if (fm < i) nzi++;
      *ajtmp++ = fm - shift;
      fm       = fill[fm];
    }
    idnew[i] = ainew[i] + nzi;
  }
  if (ai[n] != 0) {
    double af = ((double)ainew[n])/((double)ai[n]);
    PLogInfo(A,"MatLUFactorSymbolic_SeqAIJ:Reallocs %d Fill ratio:given %g needed %g\n",realloc,f,af);
    PLogInfo(A,"MatLUFactorSymbolic_SeqAIJ:Run with -pc_lu_fill %g or use \n",af);
    PLogInfo(A,"MatLUFactorSymbolic_SeqAIJ:PCLUSetFill(pc,%g);\n",af);
    PLogInfo(A,"MatLUFactorSymbolic_SeqAIJ:for best performance.\n");
  } else {
    PLogInfo(A,"MatLUFactorSymbolic_SeqAIJ: Empty matrix\n");
  }

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  ierr = PetscFree(fill);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatCreateSeqAIJ(A->comm,n,n,0,PETSC_NULL,B);CHKERRQ(ierr);
  PLogObjectParent(*B,isicol); 
  b = (Mat_SeqAIJ *) (*B)->data;
  ierr = PetscFree(b->imax);CHKERRQ(ierr);
  b->singlemalloc = PETSC_FALSE;
  /* the next line frees the default space generated by the Create() */
  ierr = PetscFree(b->a);CHKERRQ(ierr);
  ierr = PetscFree(b->ilen);CHKERRQ(ierr);
  b->a          = (Scalar *) PetscMalloc((ainew[n]+shift+1)*sizeof(Scalar));CHKPTRQ(b->a);
  b->j          = ajnew;
  b->i          = ainew;
  b->diag       = idnew;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  b->solve_work = (Scalar *) PetscMalloc( (n+1)*sizeof(Scalar));CHKPTRQ(b->solve_work);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PLogObjectMemory(*B,(ainew[n]+shift-n)*(sizeof(int)+sizeof(Scalar)));
  b->maxnz = b->nz = ainew[n] + shift;

  (*B)->factor                 =  FACTOR_LU;;
  (*B)->info.factor_mallocs    = realloc;
  (*B)->info.fill_ratio_given  = f;
  (*B)->ops->lufactornumeric   =  A->ops->lufactornumeric; /* Use Inode variant if A has inodes */

  if (ai[n] != 0) {
    (*B)->info.fill_ratio_needed = ((double)ainew[n])/((double)ai[n]);
  } else {
    (*B)->info.fill_ratio_needed = 0.0;
  }
  PetscFunctionReturn(0); 
}
/* ----------------------------------------------------------- */
extern int Mat_AIJ_CheckInode(Mat);

#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqAIJ"
int MatLUFactorNumeric_SeqAIJ(Mat A,Mat *B)
{
  Mat        C = *B;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data, *b = (Mat_SeqAIJ *)C->data;
  IS         isrow = b->row, isicol = b->icol;
  int        *r,*ic, ierr, i, j, n = a->m, *ai = b->i, *aj = b->j;
  int        *ajtmpold, *ajtmp, nz, row, *ics, shift = a->indexshift;
  int        *diag_offset = b->diag,diag,k;
  int        preserve_row_sums = (int) a->ilu_preserve_row_sums;
  register   int    *pj;
  Scalar     *rtmp,*v, *pc, multiplier,sum,inner_sum,*rowsums = 0;
  double     ssum; 
  register   Scalar *pv, *rtmps,*u_values;

  PetscFunctionBegin;

  ierr  = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc( (n+1)*sizeof(Scalar) );CHKPTRQ(rtmp);
  ierr  = PetscMemzero(rtmp,(n+1)*sizeof(Scalar));CHKERRQ(ierr);
  rtmps = rtmp + shift; ics = ic + shift;

  /* precalculate row sums */
  if (preserve_row_sums) {
    rowsums = (Scalar *) PetscMalloc( n*sizeof(Scalar) );CHKPTRQ(rowsums);
    for ( i=0; i<n; i++ ) {
      nz  = a->i[r[i]+1] - a->i[r[i]];
      v   = a->a + a->i[r[i]] + shift;
      sum = 0.0;
      for ( j=0; j<nz; j++ ) sum += v[j];
      rowsums[i] = sum;
    }
  }

  for ( i=0; i<n; i++ ) {
    nz    = ai[i+1] - ai[i];
    ajtmp = aj + ai[i] + shift;
    for  ( j=0; j<nz; j++ ) rtmps[ajtmp[j]] = 0.0;

    /* load in initial (unfactored row) */
    nz       = a->i[r[i]+1] - a->i[r[i]];
    ajtmpold = a->j + a->i[r[i]] + shift;
    v        = a->a + a->i[r[i]] + shift;
    for ( j=0; j<nz; j++ ) rtmp[ics[ajtmpold[j]]] =  v[j];

    row = *ajtmp++ + shift;
    while  (row < i ) {
      pc = rtmp + row;
      if (*pc != 0.0) {
        pv         = b->a + diag_offset[row] + shift;
        pj         = b->j + diag_offset[row] + (!shift);
        multiplier = *pc / *pv++;
        *pc        = multiplier;
        nz         = ai[row+1] - diag_offset[row] - 1;
        for (j=0; j<nz; j++) rtmps[pj[j]] -= multiplier * pv[j];
        PLogFlops(2*nz);
      }
      row = *ajtmp++ + shift;
    }
    /* finished row so stick it into b->a */
    pv = b->a + ai[i] + shift;
    pj = b->j + ai[i] + shift;
    nz = ai[i+1] - ai[i];
    for ( j=0; j<nz; j++ ) {pv[j] = rtmps[pj[j]];}
    diag = diag_offset[i] - ai[i];
    /*
          Possibly adjust diagonal entry on current row to force
        LU matrix to have same row sum as initial matrix. 
    */
    if (pv[diag] == 0.0) {
      SETERRQ1(PETSC_ERR_MAT_LU_ZRPVT,0,"Zero pivot row %d",i);
    }
    if (preserve_row_sums) {
      pj  = b->j + ai[i] + shift;
      sum = rowsums[i];
      for ( j=0; j<diag; j++ ) {
        u_values  = b->a + diag_offset[pj[j]] + shift;
        nz        = ai[pj[j]+1] - diag_offset[pj[j]];
        inner_sum = 0.0;
        for ( k=0; k<nz; k++ ) {
          inner_sum += u_values[k];
        }
        sum -= pv[j]*inner_sum;

      }
      nz       = ai[i+1] - diag_offset[i] - 1;
      u_values = b->a + diag_offset[i] + 1 + shift;
      for ( k=0; k<nz; k++ ) {
        sum -= u_values[k];
      }
      ssum = PetscAbsScalar(sum/pv[diag]);
      if (ssum < 1000. && ssum > .001) pv[diag] = sum; 
    }
    /* check pivot entry for current row */
  }

  /* invert diagonal entries for simplier triangular solves */
  for ( i=0; i<n; i++ ) {
    b->a[diag_offset[i]+shift] = 1.0/b->a[diag_offset[i]+shift];
  }

  if (preserve_row_sums) {ierr = PetscFree(rowsums);CHKERRQ(ierr);}
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  ierr = Mat_AIJ_CheckInode(C);CHKERRQ(ierr);
  C->assembled = PETSC_TRUE;
  PLogFlops(b->n);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatLUFactor_SeqAIJ"
int MatLUFactor_SeqAIJ(Mat A,IS row,IS col,double f)
{
  Mat_SeqAIJ     *mat = (Mat_SeqAIJ *) A->data;
  int            ierr,refct;
  Mat            C;
  PetscOps       *Abops;
  MatOps         Aops;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic(A,row,col,f,&C);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(A,&C);CHKERRQ(ierr);

  /* free all the data structures from mat */
  ierr = PetscFree(mat->a);CHKERRQ(ierr);
  if (!mat->singlemalloc) {
    ierr = PetscFree(mat->i);CHKERRQ(ierr);
    ierr = PetscFree(mat->j);CHKERRQ(ierr);
  }
  if (mat->diag) {ierr = PetscFree(mat->diag);CHKERRQ(ierr);}
  if (mat->ilen) {ierr = PetscFree(mat->ilen);CHKERRQ(ierr);}
  if (mat->imax) {ierr = PetscFree(mat->imax);CHKERRQ(ierr);}
  if (mat->solve_work) {ierr = PetscFree(mat->solve_work);CHKERRQ(ierr);}
  if (mat->inode.size) {ierr = PetscFree(mat->inode.size);CHKERRQ(ierr);}
  if (mat->icol) {ierr = ISDestroy(mat->icol);CHKERRQ(ierr);}
  ierr = PetscFree(mat);CHKERRQ(ierr);

  ierr = MapDestroy(A->rmap);CHKERRQ(ierr);
  ierr = MapDestroy(A->cmap);CHKERRQ(ierr);

  /*
       This is horrible, horrible code. We need to keep the 
    A pointers for the bops and ops but copy everything 
    else from C.
  */
  Abops = A->bops;
  Aops  = A->ops;
  refct = A->refct;
  ierr  = PetscMemcpy(A,C,sizeof(struct _p_Mat));CHKERRQ(ierr);
  mat   = (Mat_SeqAIJ *) A->data;
  PLogObjectParent(A,mat->icol); 
  
  A->bops  = Abops;
  A->ops   = Aops;
  A->qlist = 0;
  A->refct = refct;
  /* copy over the type_name and name */
  ierr     = PetscStrallocpy(C->type_name,&A->type_name);CHKERRQ(ierr);
  ierr     = PetscStrallocpy(C->name,&A->name);CHKERRQ(ierr);

  PetscHeaderDestroy(C);

  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqAIJ"
int MatSolve_SeqAIJ(Mat A,Vec bb, Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  IS         iscol = a->col, isrow = a->row;
  int        *r,*c, ierr, i,  n = a->m, *vi, *ai = a->i, *aj = a->j;
  int        nz,shift = a->indexshift,*rout,*cout;
  Scalar     *x,*b,*tmp, *tmps, *aa = a->a, sum, *v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  tmps   = tmp + shift;
  for ( i=1; i<n; i++ ) {
    v   = aa + ai[i] + shift;
    vi  = aj + ai[i] + shift;
    nz  = a->diag[i] - ai[i];
    sum = b[*r++];
    while (nz--) sum -= *v++ * tmps[*vi++];
    tmp[i] = sum;
  }

  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + a->diag[i] + (!shift);
    vi  = aj + a->diag[i] + (!shift);
    nz  = ai[i+1] - a->diag[i] - 1;
    sum = tmp[i];
    while (nz--) sum -= *v++ * tmps[*vi++];
    x[*c--] = tmp[i] = sum*aa[a->diag[i]+shift];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PLogFlops(2*a->nz - a->n);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqAIJ_NaturalOrdering"
int MatSolve_SeqAIJ_NaturalOrdering(Mat A,Vec bb, Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        n = a->m, *ai = a->i, *aj = a->j, *adiag = a->diag,ierr;
  Scalar     *x,*b, *aa = a->a, sum;
#if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
  int        adiag_i,i,*vi,nz,ai_i;
  Scalar     *v;
#endif

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (a->indexshift) {
     ierr = MatSolve_SeqAIJ(A,bb,xx);CHKERRQ(ierr);
     PetscFunctionReturn(0);
  }

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
  fortransolveaij_(&n,x,ai,aj,adiag,aa,b);
#else
  /* forward solve the lower triangular */
  x[0] = b[0];
  for ( i=1; i<n; i++ ) {
    ai_i = ai[i];
    v    = aa + ai_i;
    vi   = aj + ai_i;
    nz   = adiag[i] - ai_i;
    sum  = b[i];
    while (nz--) sum -= *v++ * x[*vi++];
    x[i] = sum;
  }

  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    adiag_i = adiag[i];
    v       = aa + adiag_i + 1;
    vi      = aj + adiag_i + 1;
    nz      = ai[i+1] - adiag_i - 1;
    sum     = x[i];
    while (nz--) sum -= *v++ * x[*vi++];
    x[i]    = sum*aa[adiag_i];
  }
#endif
  PLogFlops(2*a->nz - a->n);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveAdd_SeqAIJ"
int MatSolveAdd_SeqAIJ(Mat A,Vec bb, Vec yy, Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  IS         iscol = a->col, isrow = a->row;
  int        *r,*c, ierr, i,  n = a->m, *vi, *ai = a->i, *aj = a->j;
  int        nz, shift = a->indexshift,*rout,*cout;
  Scalar     *x,*b,*tmp, *aa = a->a, sum, *v;

  PetscFunctionBegin;
  if (yy != xx) {ierr = VecCopy(yy,xx);CHKERRQ(ierr);}

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  for ( i=1; i<n; i++ ) {
    v   = aa + ai[i] + shift;
    vi  = aj + ai[i] + shift;
    nz  = a->diag[i] - ai[i];
    sum = b[*r++];
    while (nz--) sum -= *v++ * tmp[*vi++ + shift];
    tmp[i] = sum;
  }

  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + a->diag[i] + (!shift);
    vi  = aj + a->diag[i] + (!shift);
    nz  = ai[i+1] - a->diag[i] - 1;
    sum = tmp[i];
    while (nz--) sum -= *v++ * tmp[*vi++ + shift];
    tmp[i] = sum*aa[a->diag[i]+shift];
    x[*c--] += tmp[i];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PLogFlops(2*a->nz);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatSolveTranspose_SeqAIJ"
int MatSolveTranspose_SeqAIJ(Mat A,Vec bb, Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  IS         iscol = a->col, isrow = a->row;
  int        *r,*c, ierr, i, n = a->m, *vi, *ai = a->i, *aj = a->j;
  int        nz,shift = a->indexshift,*rout,*cout, *diag = a->diag;
  Scalar     *x,*b,*tmp, *aa = a->a, *v, s1;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for ( i=0; i<n; i++ ) tmp[i] = b[c[i]]; 

  /* forward solve the U^T */
  for ( i=0; i<n; i++ ) {
    v   = aa + diag[i] + shift;
    vi  = aj + diag[i] + (!shift);
    nz  = ai[i+1] - diag[i] - 1;
    s1  = tmp[i];
    s1 *= *(v++);  /* multiply by inverse of diagonal entry */
    while (nz--) {
      tmp[*vi++ + shift] -= (*v++)*s1;
    }
    tmp[i] = s1;
  }

  /* backward solve the L^T */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + diag[i] - 1 + shift;
    vi  = aj + diag[i] - 1 + shift;
    nz  = diag[i] - ai[i];
    s1  = tmp[i];
    while (nz--) {
      tmp[*vi-- + shift] -= (*v--)*s1;
    }
  }

  /* copy tmp into x according to permutation */
  for ( i=0; i<n; i++ ) x[r[i]] = tmp[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  PLogFlops(2*a->nz-a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveTransposeAdd_SeqAIJ"
int MatSolveTransposeAdd_SeqAIJ(Mat A,Vec bb, Vec zz,Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  IS         iscol = a->col, isrow = a->row;
  int        *r,*c, ierr, i, n = a->m, *vi, *ai = a->i, *aj = a->j;
  int        nz,shift = a->indexshift, *rout, *cout, *diag = a->diag;
  Scalar     *x,*b,*tmp, *aa = a->a, *v;

  PetscFunctionBegin;
  if (zz != xx) VecCopy(zz,xx);

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for ( i=0; i<n; i++ ) tmp[i] = b[c[i]]; 

  /* forward solve the U^T */
  for ( i=0; i<n; i++ ) {
    v   = aa + diag[i] + shift;
    vi  = aj + diag[i] + (!shift);
    nz  = ai[i+1] - diag[i] - 1;
    tmp[i] *= *v++;
    while (nz--) {
      tmp[*vi++ + shift] -= (*v++)*tmp[i];
    }
  }

  /* backward solve the L^T */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + diag[i] - 1 + shift;
    vi  = aj + diag[i] - 1 + shift;
    nz  = diag[i] - ai[i];
    while (nz--) {
      tmp[*vi-- + shift] -= (*v--)*tmp[i];
    }
  }

  /* copy tmp into x according to permutation */
  for ( i=0; i<n; i++ ) x[r[i]] += tmp[i]; 

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  PLogFlops(2*a->nz);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/
extern int MatMissingDiagonal_SeqAIJ(Mat);

#undef __FUNC__  
#define __FUNC__ "MatILUFactorSymbolic_SeqAIJ"
int MatILUFactorSymbolic_SeqAIJ(Mat A,IS isrow,IS iscol,MatILUInfo *info,Mat *fact)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data, *b;
  IS         isicol;
  int        *r,*ic, ierr, prow, n = a->m, *ai = a->i, *aj = a->j;
  int        *ainew,*ajnew, jmax,*fill, *xi, nz, *im,*ajfill,*flev;
  int        *dloc, idx, row,m,fm, nzf, nzi,len,  realloc = 0, dcount = 0;
  int        incrlev,nnz,i,shift = a->indexshift,levels,diagonal_fill;
  PetscTruth col_identity, row_identity;
  double     f;
 
  PetscFunctionBegin;
  if (info) {
    f             = info->fill;
    levels        = (int) info->levels;
    diagonal_fill = (int) info->diagonal_fill;
  } else {
    f             = 1.0;
    levels        = 0;
    diagonal_fill = 0;
  }
  ierr = ISInvertPermutation(iscol,&isicol);CHKERRQ(ierr);

  /* special case that simply copies fill pattern */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!levels && row_identity && col_identity) {
    ierr = MatDuplicate_SeqAIJ(A,MAT_DO_NOT_COPY_VALUES,fact);CHKERRQ(ierr);
    (*fact)->factor = FACTOR_LU;
    b               = (Mat_SeqAIJ *) (*fact)->data;
    if (!b->diag) {
      ierr = MatMarkDiagonal_SeqAIJ(*fact);CHKERRQ(ierr);
    }
    ierr = MatMissingDiagonal_SeqAIJ(*fact);CHKERRQ(ierr);
    b->row              = isrow;
    b->col              = iscol;
    b->icol             = isicol;
    b->solve_work       = (Scalar *) PetscMalloc((b->m+1)*sizeof(Scalar));CHKPTRQ(b->solve_work);
    (*fact)->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
    ierr                = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
    ierr                = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* get new row pointers */
  ainew = (int *) PetscMalloc( (n+1)*sizeof(int) );CHKPTRQ(ainew);
  ainew[0] = -shift;
  /* don't know how many column pointers are needed so estimate */
  jmax = (int) (f*(ai[n]+!shift));
  ajnew = (int *) PetscMalloc( (jmax)*sizeof(int) );CHKPTRQ(ajnew);
  /* ajfill is level of fill for each fill entry */
  ajfill = (int *) PetscMalloc( (jmax)*sizeof(int) );CHKPTRQ(ajfill);
  /* fill is a linked list of nonzeros in active row */
  fill = (int *) PetscMalloc( (n+1)*sizeof(int));CHKPTRQ(fill);
  /* im is level for each filled value */
  im = (int *) PetscMalloc( (n+1)*sizeof(int));CHKPTRQ(im);
  /* dloc is location of diagonal in factor */
  dloc = (int *) PetscMalloc( (n+1)*sizeof(int));CHKPTRQ(dloc);
  dloc[0]  = 0;
  for ( prow=0; prow<n; prow++ ) {

    /* copy current row into linked list */
    nzf     = nz  = ai[r[prow]+1] - ai[r[prow]];
    if (!nz) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,1,"Empty row in matrix");
    xi      = aj + ai[r[prow]] + shift;
    fill[n]    = n;
    fill[prow] = -1; /* marker to indicate if diagonal exists */
    while (nz--) {
      fm  = n;
      idx = ic[*xi++ + shift];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
      im[idx]   = 0;
    }

    /* make sure diagonal entry is included */
    if (diagonal_fill && fill[prow] == -1) {
      fm = n;
      while (fill[fm] < prow) fm = fill[fm];
      fill[prow] = fill[fm]; /* insert diagonal into linked list */
      fill[fm]   = prow;
      im[prow]   = 0;
      nzf++;
      dcount++;
    }

    nzi = 0;
    row = fill[n];
    while ( row < prow ) {
      incrlev = im[row] + 1;
      nz      = dloc[row];
      xi      = ajnew  + ainew[row] + shift + nz + 1;
      flev    = ajfill + ainew[row] + shift + nz + 1;
      nnz     = ainew[row+1] - ainew[row] - nz - 1;
      fm      = row;
      while (nnz-- > 0) {
        idx = *xi++ + shift;
        if (*flev + incrlev > levels) {
          flev++;
          continue;
        }
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          im[idx]   = *flev + incrlev;
          fill[m]   = idx;
          fill[idx] = fm;
          fm        = idx;
          nzf++;
        } else {
          if (im[idx] > *flev + incrlev) im[idx] = *flev+incrlev;
        }
        flev++;
      }
      row = fill[row];
      nzi++;
    }
    /* copy new filled row into permanent storage */
    ainew[prow+1] = ainew[prow] + nzf;
    if (ainew[prow+1] > jmax-shift) {

      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      /*  maxadd = (int) ((f*(ai[n]+!shift)*(n-prow+5))/n); */
      int maxadd = jmax;
      if (maxadd < nzf) maxadd = (n-prow)*(nzf+1);
      jmax += maxadd;

      /* allocate a longer ajnew and ajfill */
      xi     = (int *) PetscMalloc( jmax*sizeof(int) );CHKPTRQ(xi);
      ierr   = PetscMemcpy(xi,ajnew,(ainew[prow]+shift)*sizeof(int));CHKERRQ(ierr);
      ierr = PetscFree(ajnew);CHKERRQ(ierr);
      ajnew  = xi;
      xi     = (int *) PetscMalloc( jmax*sizeof(int) );CHKPTRQ(xi);
      ierr   = PetscMemcpy(xi,ajfill,(ainew[prow]+shift)*sizeof(int));CHKERRQ(ierr);
      ierr = PetscFree(ajfill);CHKERRQ(ierr);
      ajfill = xi;
      realloc++; /* count how many times we realloc */
    }
    xi          = ajnew + ainew[prow] + shift;
    flev        = ajfill + ainew[prow] + shift;
    dloc[prow]  = nzi;
    fm          = fill[n];
    while (nzf--) {
      *xi++   = fm - shift;
      *flev++ = im[fm];
      fm      = fill[fm];
    }
    /* make sure row has diagonal entry */
    if (ajnew[ainew[prow]+shift+dloc[prow]]+shift != prow) {
      SETERRQ1(PETSC_ERR_MAT_LU_ZRPVT,1,"Row %d has missing diagonal in factored matrix\n\
    try running with -pc_ilu_nonzeros_along_diagonal or -pc_ilu_diagonal_fill",prow);
    }
  }
  ierr = PetscFree(ajfill); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscFree(fill);CHKERRQ(ierr);
  ierr = PetscFree(im);CHKERRQ(ierr);

  {
    double af = ((double)ainew[n])/((double)ai[n]);
    PLogInfo(A,"MatILUFactorSymbolic_SeqAIJ:Reallocs %d Fill ratio:given %g needed %g\n",realloc,f,af);
    PLogInfo(A,"MatILUFactorSymbolic_SeqAIJ:Run with -pc_ilu_fill %g or use \n",af);
    PLogInfo(A,"MatILUFactorSymbolic_SeqAIJ:PCILUSetFill(pc,%g);\n",af);
    PLogInfo(A,"MatILUFactorSymbolic_SeqAIJ:for best performance.\n");
    if (diagonal_fill) {
      PLogInfo(A,"MatILUFactorSymbolic_SeqAIJ:Detected and replace %d missing diagonals",dcount);
    }
  }

  /* put together the new matrix */
  ierr = MatCreateSeqAIJ(A->comm,n,n,0,PETSC_NULL,fact);CHKERRQ(ierr);
  PLogObjectParent(*fact,isicol);
  b = (Mat_SeqAIJ *) (*fact)->data;
  ierr = PetscFree(b->imax);CHKERRQ(ierr);
  b->singlemalloc = PETSC_FALSE;
  len = (ainew[n] + shift)*sizeof(Scalar);
  /* the next line frees the default space generated by the Create() */
  ierr = PetscFree(b->a);CHKERRQ(ierr);
  ierr = PetscFree(b->ilen);CHKERRQ(ierr);
  b->a          = (Scalar *) PetscMalloc( len+1 );CHKPTRQ(b->a);
  b->j          = ajnew;
  b->i          = ainew;
  for ( i=0; i<n; i++ ) dloc[i] += ainew[i];
  b->diag       = dloc;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  b->solve_work = (Scalar *) PetscMalloc( (n+1)*sizeof(Scalar));CHKPTRQ(b->solve_work);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate dloc, solve_work, new a, new j */
  PLogObjectMemory(*fact,(ainew[n]+shift-n) * (sizeof(int)+sizeof(Scalar)));
  b->maxnz          = b->nz = ainew[n] + shift;
  (*fact)->factor   = FACTOR_LU;

  (*fact)->info.factor_mallocs    = realloc;
  (*fact)->info.fill_ratio_given  = f;
  (*fact)->info.fill_ratio_needed = ((double)ainew[n])/((double)ai[prow]);
  (*fact)->factor                 =  FACTOR_LU;;

  PetscFunctionReturn(0); 
}




