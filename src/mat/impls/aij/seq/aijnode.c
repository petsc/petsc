#ifndef lint
static char vcid[] = "$Id: aijnode.c,v 1.37 1996/03/15 18:05:20 balay Exp bsmith $";
#endif
/*
  This file provides high performance routines for the AIJ (compressed row)
  format by taking advantage of rows with identical nonzero structure (I-nodes).
*/
#include "aij.h"                

int Mat_AIJ_CheckInode(Mat);
static int MatSolve_SeqAIJ_Inode(Mat ,Vec , Vec );
static int MatLUFactorNumeric_SeqAIJ_Inode(Mat ,Mat * );

/*
      This builds symmetric version of nonzero structure, 
*/
static int MatToSymmetricIJ_SeqAIJ_Inode( Mat_SeqAIJ *A, int **iia, int **jja,
                                          int ishift,int oshift)
{
  int *work,*ia,*ja,*j, nz, m ,n, row, col, *jmax;
  int *tns, *tvc, *ns = A->inode.size, nsz, i1, i2, *ai= A->i, *aj = A->j;

  m = A->inode.node_count;
  n = A->m;
  /* allocate space for reformated inode structure */
  tns = (int *) PetscMalloc((m +1 )*sizeof(int)); CHKPTRQ(tns);
  tvc = (int *) PetscMalloc((n +1 )*sizeof(int)); CHKPTRQ(tvc);
  for (i1=0, tns[0]=0; i1<m; ++i1) tns[i1+1] = tns[i1]+ ns[i1];

  for (i1=0, row=0; i1<m; ++i1){
    nsz = ns[i1];
    for ( i2=0; i2<nsz; ++i2, ++row)
      tvc[row] = i1;
  }
  /* allocate space for row pointers */
  *iia = ia = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(ia);
  PetscMemzero(ia,(m+1)*sizeof(int));
  work = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(work);

  /* determine the number of columns in each row */
  ia[0] = -oshift;
  for (i1=0 ; i1<m; ++i1) {
    row  = tns[i1];
    j    = aj + ai[row] + ishift;
    jmax = aj + ai[row+1] + ishift;
    i2   = 0;
    col  = *j + ishift;
    i2   = tvc[col];
    while (i2<i1 && j<jmax) { /* 1.[-xx-d-xx--] 2.[-xx-------], off-diagonal elemets */
      ia[i1+1]++;
      ia[i2+1]++;
      i2++;                     /* Start col of next node */
      while(((col=*j+ishift)<tns[i2]) && (j<jmax)) ++j;
      i2 = tvc[col];
    }
    if(i2 == i1) ia[i2+1]++;    /* now the diagonal element */
  }

  /* shift ia[i] to point to next row */
  for ( i1=1; i1<m+1; i1++ ) {
    row        = ia[i1-1];
    ia[i1]    += row;
    work[i1-1] = row - 1;
  }

  /* allocate space for column pointers */
  nz   = ia[m] + (!ishift);
  *jja = ja = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ja);

 /* loop over lower triangular part putting into ja */ 
  for (i1=0, row=0; i1<m; ++i1) {
    row  = tns[i1];
    j    = aj + ai[row] + ishift;
    jmax = aj + ai[row+1] + ishift;
    i2   = 0;                     /* Col inode index */
    col  = *j + ishift;
    i2   = tvc[col];
    while (i2<i1 && j<jmax) {
      ja[work[i2]++] = i1 - oshift;
      ja[work[i1]++] = i2 - oshift;
      ++i2;
      while(((col=*j+ishift)< tns[i2])&&(j<jmax)) ++j; /* Skip rest col indices in this node */
      i2 = tvc[col];
    }
    if (i2 == i1) ja[work[i1]++] = i2 - oshift;

  }
  PetscFree(work);
  PetscFree(tns);
  PetscFree(tvc);
  return 0;
}

/*
      This builds nonsymmetric version of nonzero structure, 
*/
static int MatToIJ_SeqAIJ_Inode( Mat_SeqAIJ *A, int **iia, int **jja,
                                          int ishift,int oshift)
{
  int *work,*ia,*ja,*j, nz, m ,n, row, col;
  int *tns, *tvc, *ns = A->inode.size, nsz, i1, i2, *ai= A->i, *aj = A->j;

  m = A->inode.node_count;
  n = A->m;
  /* allocate space for reformated inode structure */
  tns = (int *) PetscMalloc((m +1 )*sizeof(int)); CHKPTRQ(tns);
  tvc = (int *) PetscMalloc((n +1 )*sizeof(int)); CHKPTRQ(tvc);
  for (i1=0, tns[0]=0; i1<m; ++i1) tns[i1+1] = tns[i1]+ ns[i1];

  for (i1=0, row=0; i1<m; ++i1){
    nsz = ns[i1];
    for ( i2=0; i2<nsz; ++i2, ++row)
      tvc[row] = i1;
  }
  /* allocate space for row pointers */
  *iia = ia = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(ia);
  PetscMemzero(ia,(m+1)*sizeof(int));
  work = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(work);

  /* determine the number of columns in each row */
  ia[0] = -oshift;
  for (i1=0; i1<m; ++i1) {
    row = tns[i1];
    j   = aj + ai[row] + ishift;
    col = *j + ishift;
    i2  = tvc[col];
    nz  = ai[row+1] - ai[row]; 
    while (nz-- > 0) {           /* off-diagonal elemets */
      ia[i1+1]++;
      i2++;                     /* Start col of next node */
      while (((col = *j + ishift) < tns[i2]) && nz > 0) {++j;nz--;}
      i2 = tvc[col];
    }
  }

  /* shift ia[i] to point to next row */
  for ( i1=1; i1<m+1; i1++ ) {
    row        = ia[i1-1];
    ia[i1]    += row;
    work[i1-1] = row - 1;
  }

  /* allocate space for column pointers */
  nz   = ia[m] + (!ishift);
  *jja = ja = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ja);

 /* loop over matrix putting into ja */ 
  for (i1=0, row=0; i1<m; ++i1) {
    row = tns[i1];
    j   = aj + ai[row] + ishift;
    i2  = 0;                     /* Col inode index */
    col = *j + ishift;
    i2  = tvc[col];
    nz  = ai[row+1] - ai[row]; 
    while (nz-- > 0) {
      ja[work[i1]++] = i2 - oshift;
      ++i2;
      while(((col = *j + ishift)< tns[i2]) && nz > 0) {nz--; ++j;}
      i2 = tvc[col];
    }
  }
  PetscFree(work);
  PetscFree(tns);
  PetscFree(tvc);
  return 0;
}

static int MatGetReordering_SeqAIJ_Inode(Mat A,MatOrdering type,IS *rperm, IS *cperm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr, *ia, *ja,n = a->n,*idx,i,j,*ridx,*cidx,ishift,oshift;
  int        row,*permr, *permc,m ,*ns, *tns, start_val, end_val, indx;
  IS         ris= 0, cis = 0;

  if (type  == ORDER_NATURAL) {
    idx = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(idx);
    for ( i=0; i<n; i++ ) idx[i] = i;
    ierr = ISCreateSeq(MPI_COMM_SELF,n,idx,rperm); CHKERRQ(ierr);
    ierr = ISCreateSeq(MPI_COMM_SELF,n,idx,cperm); CHKERRQ(ierr);
    PetscFree(idx);
    ISSetPermutation(*rperm);
    ISSetPermutation(*cperm);
    ISSetIdentity(*rperm);
    ISSetIdentity(*cperm);
    return 0;
  }
  ns = a->inode.size;
  m  = a->inode.node_count;

  MatReorderingRegisterAll();
  ishift = a->indexshift;
  oshift = -MatReorderingIndexShift[(int)type];
  if (MatReorderingRequiresSymmetric[(int)type]) {
    ierr = MatToSymmetricIJ_SeqAIJ_Inode(a,&ia,&ja,ishift,oshift);CHKERRQ(ierr);
  } else {
    ierr = MatToIJ_SeqAIJ_Inode(a,&ia,&ja,ishift,oshift);CHKERRQ(ierr);
  }
  ierr = MatGetReordering_IJ(m,ia,ja,type,&ris,&cis); CHKERRQ(ierr);
  PetscFree(ia); PetscFree(ja);
  tns   = (int *) PetscMalloc((m +1 )*sizeof(int)); CHKPTRQ(tns);
  permr = (int *) PetscMalloc( (2*a->n+1)*sizeof(int) ); CHKPTRQ(permr);
  permc = permr + n;

  ierr  = ISGetIndices(ris,&ridx); CHKERRQ(ierr);
  ierr  = ISGetIndices(cis,&cidx); CHKERRQ(ierr);

  /* Form the inode structure for the rows of permuted matric using inv perm*/
  for ( i=0, tns[0]=0; i<m; ++i) tns[i+1] = tns[i] + ns[i];

  /* Construct the permutations for rows*/
  for ( i=0,row = 0; i<m; ++i){
    indx      = ridx[i];
    start_val = tns[indx];
    end_val   = tns[indx + 1];
    for ( j=start_val; j<end_val; ++j, ++row) permr[row]= j;
  }

 /*Construct permutations for columns*/
  for (i=0,row=0; i<m; ++i){
    indx      = cidx[i];
    start_val = tns[indx];
    end_val   = tns[indx + 1];
    for (j = start_val; j<end_val; ++j, ++row)
      permc[row]= j;
  }

  ierr = ISCreateSeq(MPI_COMM_SELF,n,permr,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateSeq(MPI_COMM_SELF,n,permc,cperm); CHKERRQ(ierr);
  ISSetPermutation(*cperm);
 
  ierr  = ISRestoreIndices(ris,&ridx); CHKERRQ(ierr);
  ierr  = ISRestoreIndices(cis,&cidx); CHKERRQ(ierr);

  PetscFree(permr);
  ISDestroy(cis); ISDestroy(ris); 
  PetscFree(tns);
  return 0; 
}


/* ----------------------------------------------------------- */

static int MatMult_SeqAIJ_Inode(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  Scalar     sum1, sum2, sum3, sum4, sum5, tmp0, tmp1;
  Scalar     *v1, *v2, *v3, *v4, *v5,*x, *y;
  int        *idx, i1, i2, n, i, row,node_max, *ns, *ii, nsz, sz;
  int        shift = a->indexshift;
  
  if (!a->inode.size)SETERRQ(1,"MatMult_SeqAIJ_Inode: Missing Inode Structure");
  node_max = a->inode.node_count;                
  ns       = a->inode.size;     /* Node Size array */
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  x    = x + shift;             /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v1   = a->a;
  ii   = a->i;

  for (i = 0, row = 0; i< node_max; ++i){
    nsz  = ns[i]; 
    n    = ii[1] - ii[0];
    ii  += nsz;
    sz   = n;                   /*No of non zeros in this row */
                                /* Switch on the size of Node */
    switch (nsz){               /* Each loop in 'case' is unrolled */
    case 1 :
      sum1  = 0;
      
      for( n = 0; n< sz-1; n+=2) {
        i1   = idx[0];          /* The instructions are ordered to */
        i2   = idx[1];          /* make the compiler's job easy */
        idx += 2;
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
       }
     
      if(n   == sz-1){          /* Take care of the last nonzero  */
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
      }
      y[row++]=sum1;
      break;
    case 2:
      sum1  = 0;
      sum2  = 0;
      v2    = v1 + n;
      
      for( n = 0; n< sz-1; n+=2) {
        i1   = idx[0];
        i2   = idx[1];
        idx += 2;
        tmp0 = x[i1];
        tmp1 = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
      }
      if(n   == sz-1){
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      v1      =v2;              /* Since the next block to be processed starts there*/
      idx    +=sz;
      break;
    case 3:
      sum1  = 0;
      sum2  = 0;
      sum3  = 0;
      v2    = v1 + n;
      v3    = v2 + n;
      
      for (n = 0; n< sz-1; n+=2) {
        i1   = idx[0];
        i2   = idx[1];
        idx += 2;
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if (n == sz-1){
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      v1       =v3;             /* Since the next block to be processed starts there*/
      idx     +=2*sz;
      break;
    case 4:
      sum1  = 0;
      sum2  = 0;
      sum3  = 0;
      sum4  = 0;
      v2    = v1 + n;
      v3    = v2 + n;
      v4    = v3 + n;
      
      for (n = 0; n< sz-1; n+=2) {
        i1   = idx[0];
        i2   = idx[1];
        idx += 2;
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
        sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
      }
      if (n == sz-1){
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      v1      =v4;              /* Since the next block to be processed starts there*/
      idx    +=3*sz;
      break;
    case 5:
      sum1  = 0;
      sum2  = 0;
      sum3  = 0;
      sum4  = 0;
      sum5  = 0;
      v2    = v1 + n;
      v3    = v2 + n;
      v4    = v3 + n;
      v5    = v4 + n;
      
      for (n = 0; n<sz-1; n+=2) {
        i1   = idx[0];
        i2   = idx[1];
        idx += 2;
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
        sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
        sum5 += v5[0] * tmp0 + v5[1] *tmp1; v5 += 2;
      }
      if(n   == sz-1){
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
        sum5 += *v5++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      y[row++]=sum5;
      v1      =v5;       /* Since the next block to be processed starts there */
      idx    +=4*sz;
      break;
    default :
      SETERRQ(1,"MatMult_SeqAIJ_Inode:Node size not yet supported");
    }
  }
  PLogFlops(2*a->nz - a->m);
  return 0;
}
/* ----------------------------------------------------------- */

int Mat_AIJ_CheckInode(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr, flg, i, j, m, nzx, nzy, *idx, *idy, *ns,*ii, node_count, blk_size;

  /* Notes: We set a->inode.limit=5 in MatCreateSeqAIJ(). */
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_no_inode", &flg); CHKERRQ(ierr);
  if (flg) return 0;
  ierr = OptionsGetInt(PETSC_NULL,"-mat_aij_inode_limit",&a->inode.limit, \
                       &flg);  CHKERRQ(ierr);
  if (a->inode.limit > a->inode.max_limit) a->inode.limit = a->inode.max_limit;
  m = a->m;    
  if (a->inode.size) {ns = a->inode.size;}
  else { ns = (int *)PetscMalloc((m+1)*sizeof(int));  CHKPTRQ(ns);}



  i          = 0;
  node_count = 0; 
  idx        = a->j;
  ii         = a->i;
  while ( i < m){                /* For each row */
    nzx = ii[i+1] - ii[i];       /* Number of nonzeros */
    /* Limits the number of elements in a node to 'a->inode.limit' */
    for (j=i+1, idy=idx, blk_size=1; j<m && blk_size <a->inode.limit; ++j,++blk_size) {
      nzy     = ii[j+1] - ii[j]; /* Same number of nonzeros */
      if(nzy != nzx) break;
      idy    += nzx;             /* Same nonzero pattern */
      if (PetscMemcmp(idx, idy, nzx*sizeof(int))) break;
    }
    ns[node_count++] = blk_size;
    /* printf("%3d \t %d\n", i, blk_size); */
    idx +=blk_size*nzx;
    i    = j;
  }

  A->ops.mult            = MatMult_SeqAIJ_Inode;
  A->ops.solve           = MatSolve_SeqAIJ_Inode;
  A->ops.getreordering   = MatGetReordering_SeqAIJ_Inode;
  A->ops.lufactornumeric = MatLUFactorNumeric_SeqAIJ_Inode;
  a->inode.node_count    = node_count;
  a->inode.size          = ns;
  PLogInfo(A,"Mat_AIJ_CheckInode: Found %d nodes. Limit used: %d. Using Inode routines\n",node_count,a->inode.limit);
  return 0;
}

/* ----------------------------------------------------------- */
static int MatSolve_SeqAIJ_Inode(Mat A,Vec bb, Vec xx)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  IS          iscol = a->col, isrow = a->row;
  int         *r,*c, ierr, i, j, n = a->m, *ai = a->i, nz,shift = a->indexshift, *a_j = a->j;
  int         node_max, *ns,row, nsz, aii,*vi,*ad, *aj, i0, i1;
  Scalar      *x,*b, *a_a = a->a,*tmp, *tmps, *aa, tmp0, tmp1;
  Scalar      sum1, sum2, sum3, sum4, sum5,*v1, *v2, *v3,*v4, *v5;

  if (A->factor!=FACTOR_LU) SETERRQ(1,"MatSolve_SeqAIJ_Inode: Not for unfactored matrix");
  if (!a->inode.size)SETERRQ(1,"MatSolve_SeqAIJ_Inode: Missing Inode Structure");
  node_max = a->inode.node_count;   
  ns       = a->inode.size;     /* Node Size array */

  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;
  
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr); c = c + (n-1);
  
  /* forward solve the lower triangular */
  tmps = tmp + shift;
  aa   = a_a +shift;
  aj   = a_j + shift;
  ad   = a->diag;

  for (i = 0, row = 0; i< node_max; ++i){
    nsz = ns[i];
    aii = ai[row];
    v1  = aa + aii;
    vi  = aj + aii;
    nz  = ad[row]- aii;
    
    switch (nsz){               /* Each loop in 'case' is unrolled */
    case 1 :
      sum1 = b[*r++];
      /*      while (nz--) sum1 -= *v1++ *tmps[*vi++];*/
      for( j=0; j<nz-1; j+=2){
        i0   = vi[0];
        i1   = vi[1];
        vi  +=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
      }
      if( j == nz-1){
        tmp0 = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
      }
      tmp[row ++]=sum1;
      break;
    case 2:
      sum1 = b[*r++];
      sum2 = b[*r++];
      v2   = aa + ai[row+1];

      for( j=0; j<nz-1; j+=2){
        i0   = vi[0];
        i1   = vi[1];
        vi  +=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
      } 
      if( j == nz-1){
        tmp0 = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
      }
      sum2 -= *v2++ * sum1;
      tmp[row ++]=sum1;
      tmp[row ++]=sum2;
      break;
    case 3:
      sum1 = b[*r++];
      sum2 = b[*r++];
      sum3 = b[*r++];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      
      for( j=0; j<nz-1; j+=2){
        i0   = vi[0];
        i1   = vi[1];
        vi  +=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];  
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if( j == nz-1){
        tmp0 = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
        sum3 -= *v3++ *tmp0;
      }
      sum2 -= *v2++ * sum1;
      sum3 -= *v3++ * sum1;
      sum3 -= *v3++ * sum2;
      tmp[row ++]=sum1;
      tmp[row ++]=sum2;
      tmp[row ++]=sum3;
      break;
      
    case 4:
      sum1 = b[*r++];
      sum2 = b[*r++];
      sum3 = b[*r++];
      sum4 = b[*r++];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      v4   = aa + ai[row+3];
      
      for( j=0; j<nz-1; j+=2){
        i0   = vi[0];
        i1   = vi[1];
        vi  +=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];   
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
        sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
      }
      if( j == nz-1){
        tmp0 = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
        sum3 -= *v3++ *tmp0;
        sum4 -= *v4++ *tmp0;
      }
      sum2 -= *v2++ * sum1;
      sum3 -= *v3++ * sum1;
      sum4 -= *v4++ * sum1;
      sum3 -= *v3++ * sum2;
      sum4 -= *v4++ * sum2;
      sum4 -= *v4++ * sum3;
      
      tmp[row ++]=sum1;
      tmp[row ++]=sum2;
      tmp[row ++]=sum3;
      tmp[row ++]=sum4;
      break;
    case 5:
      sum1 = b[*r++];
      sum2 = b[*r++];
      sum3 = b[*r++];
      sum4 = b[*r++];
      sum5 = b[*r++];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      v4   = aa + ai[row+3];
      v5   = aa + ai[row+4];
      
      for( j=0; j<nz-1; j+=2){
        i0   = vi[0];
        i1   = vi[1];
        vi  +=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];   
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
        sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
        sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
      }
      if( j == nz-1){
        tmp0 = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
        sum3 -= *v3++ *tmp0;
        sum4 -= *v4++ *tmp0;
        sum5 -= *v5++ *tmp0;
      }

      sum2 -= *v2++ * sum1;
      sum3 -= *v3++ * sum1;
      sum4 -= *v4++ * sum1;
      sum5 -= *v5++ * sum1;
      sum3 -= *v3++ * sum2;
      sum4 -= *v4++ * sum2;
      sum5 -= *v5++ * sum2;
      sum4 -= *v4++ * sum3;
      sum5 -= *v5++ * sum3;
      sum5 -= *v5++ * sum4;
      
      tmp[row ++]=sum1;
      tmp[row ++]=sum2;
      tmp[row ++]=sum3;
      tmp[row ++]=sum4;
      tmp[row ++]=sum5;
      break;
    default:
      SETERRQ(1,"MatSolve_SeqAIJ_Inode: Node size not yet supported \n");
    }
  }
  /* backward solve the upper triangular */
  for ( i=node_max -1 , row = n-1 ; i>=0; i-- ){
    nsz = ns[i];
    aii = ai[row+1] -1;
    v1  = aa + aii;
    vi  = aj + aii;
    nz  = aii- ad[row];
    switch (nsz){               /* Each loop in 'case' is unrolled */
    case 1 :
      sum1 = tmp[row];

      for( j=nz ; j>1; j-=2){
        i0   = vi[0];
        i1   = vi[-1];
        vi  -=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[-1] * tmp1; v1 -= 2;
      }
      if (j==1){
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
      }
      x[*c--] = tmp[row] = sum1*a_a[ad[row]+shift]; row--;
      break;
    case 2 :
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      v2   = aa + ai[row]-1;
      for( j=nz ; j>1; j-=2){
        i0   = vi[0];
        i1   = vi[-1];
        vi  -=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[-1] * tmp1; v1 -= 2;
        sum2 -= v2[0] * tmp0 + v2[-1] * tmp1; v2 -= 2;
      }
      if (j==1){
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
      }
      
      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]+shift]; row--;
      sum2   -= *v2-- * tmp0;
      x[*c--] = tmp[row] = sum2*a_a[ad[row]+shift]; row--;
      break;
    case 3 :
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      v2   = aa + ai[row]-1;
      v3   = aa + ai[row -1]-1;
      for( j=nz ; j>1; j-=2){
        i0   = vi[0];
        i1   = vi[-1];
        vi  -=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[-1] * tmp1; v1 -= 2;
        sum2 -= v2[0] * tmp0 + v2[-1] * tmp1; v2 -= 2;
        sum3 -= v3[0] * tmp0 + v3[-1] * tmp1; v3 -= 2;
      }
      if (j==1){
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
        sum3 -= *v3-- * tmp0;
      }
      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]+shift]; row--;
      sum2   -= *v2-- * tmp0;
      sum3   -= *v3-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum2*a_a[ad[row]+shift]; row--;
      sum3   -= *v3-- * tmp0;
      x[*c--] = tmp[row] = sum3*a_a[ad[row]+shift]; row--;
      
      break;
    case 4 :
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      sum4 = tmp[row -3];
      v2   = aa + ai[row]-1;
      v3   = aa + ai[row -1]-1;
      v4   = aa + ai[row -2]-1;

      for( j=nz ; j>1; j-=2){
        i0   = vi[0];
        i1   = vi[-1];
        vi  -=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[-1] * tmp1; v1 -= 2;
        sum2 -= v2[0] * tmp0 + v2[-1] * tmp1; v2 -= 2;
        sum3 -= v3[0] * tmp0 + v3[-1] * tmp1; v3 -= 2;
        sum4 -= v4[0] * tmp0 + v4[-1] * tmp1; v4 -= 2;
      }
      if (j==1){
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
        sum3 -= *v3-- * tmp0;
        sum4 -= *v4-- * tmp0;
      }

      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]+shift]; row--;
      sum2   -= *v2-- * tmp0;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum2*a_a[ad[row]+shift]; row--;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum3*a_a[ad[row]+shift]; row--;
      sum4   -= *v4-- * tmp0;
      x[*c--] = tmp[row] = sum4*a_a[ad[row]+shift]; row--;
      break;
    case 5 :
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      sum4 = tmp[row -3];
      sum5 = tmp[row -4];
      v2   = aa + ai[row]-1;
      v3   = aa + ai[row -1]-1;
      v4   = aa + ai[row -2]-1;
      v5   = aa + ai[row -3]-1;
      for( j=nz ; j>1; j-=2){
        i0   = vi[0];
        i1   = vi[-1];
        vi  -=2;
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[-1] * tmp1; v1 -= 2;
        sum2 -= v2[0] * tmp0 + v2[-1] * tmp1; v2 -= 2;
        sum3 -= v3[0] * tmp0 + v3[-1] * tmp1; v3 -= 2;
        sum4 -= v4[0] * tmp0 + v4[-1] * tmp1; v4 -= 2;
        sum5 -= v5[0] * tmp0 + v5[-1] * tmp1; v5 -= 2;
      }
      if (j==1){
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
        sum3 -= *v3-- * tmp0;
        sum4 -= *v4-- * tmp0;
        sum5 -= *v5-- * tmp0;
      }

      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]+shift]; row--;
      sum2   -= *v2-- * tmp0;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      sum5   -= *v5-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum2*a_a[ad[row]+shift]; row--;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      sum5   -= *v5-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum3*a_a[ad[row]+shift]; row--;
      sum4   -= *v4-- * tmp0;
      sum5   -= *v5-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum4*a_a[ad[row]+shift]; row--;
      sum5   -= *v5-- * tmp0;
      x[*c--] = tmp[row] = sum5*a_a[ad[row]+shift]; row--;
      break;
    default:
      SETERRQ(1,"MatSolve_SeqAIJ_Inode: Node size not yet supported \n");
    }
  }
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&c); CHKERRQ(ierr);
  PLogFlops(2*a->nz - a->n);
  return 0;
}


static int MatLUFactorNumeric_SeqAIJ_Inode(Mat A,Mat *B)
{
  Mat        C = *B;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data, *b = (Mat_SeqAIJ *)C->data;
  IS         iscol = b->col, isrow = b->row, isicol;
  int        shift = a->indexshift, *r,*ic,*c, ierr, n = a->m, *bi = b->i; 
  int        *bj = b->j+shift, *nbj=b->j +(!shift), *ajtmp, *bjtmp, nz, row, prow;
  int        *ics,i,j, idx, *ai = a->i, *aj = a->j+shift, *bd = b->diag, node_max, nsz;
  int        *ns, *nsa, *tmp_vec, *pj;
  Scalar     *rtmp1, *rtmp2, *rtmp3,*v1, *v2, *v3, *pc1, *pc2, *pc3, mul1, mul2, mul3;
  Scalar     tmp, *ba = b->a+shift, *aa = a->a+shift, *pv, *rtmps1, *rtmps2, *rtmps3;

  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr   = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr   = ISGetIndices(iscol,&c); CHKERRQ(ierr);
  ierr   = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp1  = (Scalar *) PetscMalloc( (3*n+1)*sizeof(Scalar) ); CHKPTRQ(rtmp1);
  ics    = ic + shift; rtmps1 = rtmp1 + shift; 
  rtmp2  = rtmp1 + n;  rtmps2 = rtmp2 + shift; 
  rtmp3  = rtmp2 + n;  rtmps3 = rtmp3 + shift; 
  
  node_max = a->inode.node_count; /* has to be same for both a,b */
  ns       = b->inode.size ;
  if (!ns){                      /* If mat_order!=natural, create inode info */
    nsa     = a->inode.size;
    ns      = (int *)PetscMalloc((n+1)* sizeof(int)); CHKPTRQ(ns);
    tmp_vec = (int *)PetscMalloc((n+1)* sizeof(int)); CHKPTRQ(tmp_vec);
    b->inode.size          = ns;
    b->inode.node_count    = node_max;
    b->inode.limit         = a->inode.limit;
    b->inode.max_limit     = a->inode.max_limit;
    C->ops.mult            = MatMult_SeqAIJ_Inode;
    C->ops.solve           = MatSolve_SeqAIJ_Inode;
    C->ops.getreordering   = MatGetReordering_SeqAIJ_Inode;
    C->ops.lufactornumeric = MatLUFactorNumeric_SeqAIJ_Inode;
    for(i = 0, row = 0; i< node_max; ++i){
      nsz = nsa[i];
      for( j = 0; j < nsz; ++j, ++row)
        tmp_vec[row] = i;
    }
    for( i = 0, row = 0; i < node_max ; ++i){
      ns[i] = nsa[tmp_vec[r[row]]];
      row  += ns[i];
    }
    PetscFree(tmp_vec);
  }
  /* If max inode size >3, split it into two inodes.*/
  tmp_vec       = (int *)PetscMalloc((n+1)* sizeof(int)); CHKPTRQ(tmp_vec);
  for(i=0, j=0; i< node_max; ++i, ++j){
    if(ns[i]>3) {
      tmp_vec[j] = ns[i]/2;++j; /* Assuming ns[i] < =5  */
      tmp_vec[j] = ns[i] -tmp_vec[j-1];
    } else tmp_vec[j] = ns[i];
  }

  /* Now use the new inode info created*/
  ns       = tmp_vec;
  node_max = j;

  for ( i=0,row=0; i<node_max; i++ ) { 
    nsz   = ns[i];
    nz    = bi[row+1] - bi[row];
    bjtmp = bj + bi[row];
    
    switch (nsz){
    case 1:
      for  ( j=0; j<nz; j++ ){
        idx         = bjtmp[j];
        rtmps1[idx] = 0.0;
      }
      
      /* load in initial (unfactored row) */
      idx   = r[row];
      nz    = ai[idx+1] - ai[idx];
      ajtmp = aj + ai[idx];
      v1    = aa + ai[idx];

      for ( j=0; j<nz; j++ ) {
        idx        = ics[ajtmp[j]];
        rtmp1[idx] = v1[j];
      }
      prow = *bjtmp++ + shift;
      while (prow < row) {
        pc1 = rtmp1 + prow;
        if (*pc1 != 0.0){
          pv   = ba + bd[prow];
          pj   = nbj + bd[prow];
          mul1 = *pc1 * *pv++;
          *pc1 = mul1;
          nz   = bi[prow+1] - bd[prow] - 1;
          PLogFlops(2*nz);
          for (j=0; j<nz; j++) {
            tmp = pv[j];
            idx = pj[j];
            rtmps1[idx] -= mul1 * tmp;
          }
        }
        prow = *bjtmp++ + shift;
      }
      nz  = bi[row+1] - bi[row];
      pj  = bj + bi[row];
      pc1 = ba + bi[row];
      if (rtmp1[row] == 0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
      rtmp1[row] = 1.0/rtmp1[row];
      for ( j=0; j<nz; j++ ) {
        idx    = pj[j];
        pc1[j] = rtmps1[idx];
      }
      break;
      
    case 2:
      for  ( j=0; j<nz; j++ ) {
        idx         = bjtmp[j];
        rtmps1[idx] = 0.0;
        rtmps2[idx] = 0.0;
      }
      
      /* load in initial (unfactored row) */
      idx   = r[row];
      nz    = ai[idx+1] - ai[idx];
      ajtmp = aj + ai[idx];
      v1    = aa + ai[idx];
      v2    = aa + ai[idx+1];
      
      for ( j=0; j<nz; j++ ) {
        idx        = ics[ajtmp[j]];
        rtmp1[idx] = v1[j];
        rtmp2[idx] = v2[j];
      }
      prow = *bjtmp++ + shift;
      while (prow < row) {
        pc1 = rtmp1 + prow;
        pc2 = rtmp2 + prow;
        if (*pc1 != 0.0 || *pc2 != 0.0){
          pv   = ba + bd[prow];
          pj   = nbj + bd[prow];
          mul1 = *pc1 * *pv;
          mul2 = *pc2 * *pv; ++pv;
          *pc1 = mul1;
          *pc2 = mul2;
          
          nz   = bi[prow+1] - bd[prow] - 1;
          PLogFlops(2*2*nz);
          for (j=0; j<nz; j++) {
            tmp = pv[j];
            idx = pj[j];
            rtmps1[idx] -= mul1 * tmp;
            rtmps2[idx] -= mul2 * tmp;
          }
        }
        prow = *bjtmp++ + shift;
      }
      /* Now take care of the odd element*/
      pc1 = rtmp1 + prow;
      pc2 = rtmp2 + prow;
      if (*pc2 != 0.0){
        pj   = nbj + bd[prow];
        if(*pc1 ==0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
        mul2 = (*pc2)/(*pc1); /* since diag is not yet inverted.*/
        *pc2 = mul2;
        nz   = bi[prow+1] - bd[prow] - 1;
        PLogFlops(2*nz);
        for (j=0; j<nz; j++) {
          idx = pj[j] + shift;
          tmp = rtmp1[idx];
          rtmp2[idx] -= mul2 * tmp;
        }
      }
 
      nz  = bi[row+1] - bi[row];
      pj  = bj + bi[row];
      pc1 = ba + bi[row];
      pc2 = ba + bi[row+1];
      if (rtmp1[row] == 0.0 || rtmp2[row+1] == 0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
      rtmp1[row]   = 1.0/rtmp1[row];
      rtmp2[row+1] = 1.0/rtmp2[row+1];
      for ( j=0; j<nz; j++ ) {
        idx    = pj[j];
        pc1[j] = rtmps1[idx];
        pc2[j] = rtmps2[idx];
      }
      break;
    case 3:
      for  ( j=0; j<nz; j++ ) {
        idx         = bjtmp[j];
        rtmps1[idx] = 0.0;
        rtmps2[idx] = 0.0;
        rtmps3[idx] = 0.0;
      }
      idx   = r[row];
      nz    = ai[idx+1] - ai[idx];
      ajtmp = aj + ai[idx];
      v1    = aa + ai[idx];
      v2    = aa + ai[idx+1];
      v3    = aa + ai[idx+2];
      for ( j=0; j<nz; j++ ) {
        idx        = ics[ajtmp[j]];
        rtmp1[idx] = v1[j];
        rtmp2[idx] = v2[j];
        rtmp3[idx] = v3[j];
      }
      prow = *bjtmp++ + shift;
      while (prow < row) {
        pc1 = rtmp1 + prow;
        pc2 = rtmp2 + prow;
        pc3 = rtmp3 + prow;
        if (*pc1 != 0.0 || *pc2 != 0.0 || *pc3 !=0.0 ){
          pv   = ba + bd[prow];
          pj   = nbj + bd[prow];
          mul1 = *pc1 * *pv;
          mul2 = *pc2 * *pv; 
          mul3 = *pc3 * *pv; ++pv;
          *pc1 = mul1;
          *pc2 = mul2;
          *pc3 = mul3;
          
          nz   = bi[prow+1] - bd[prow] - 1;
          PLogFlops(3*2*nz);
          for (j=0; j<nz; j++) {
            tmp = pv[j];
            idx = pj[j];
            rtmps1[idx] -= mul1 * tmp;
            rtmps2[idx] -= mul2 * tmp;
            rtmps3[idx] -= mul3 * tmp;
          }
        }
        prow = *bjtmp++ + shift;
      }
      /* Now take care of the odd elements*/
      pc1 = rtmp1 + prow;
      pc2 = rtmp2 + prow;
      pc3 = rtmp3 + prow;
      if (*pc2 != 0.0 || *pc3 != 0){
        pj   = nbj + bd[prow];
        if(*pc1 ==0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
        mul2 = (*pc2)/(*pc1);
        mul3 = (*pc3)/(*pc1);
        *pc2 = mul2;
        *pc3 = mul3;
        nz   = bi[prow+1] - bd[prow] - 1;
        PLogFlops(2*2*nz);
        for (j=0; j<nz; j++) {
          idx = pj[j] + shift;
          tmp = rtmp1[idx];
          rtmp2[idx] -= mul2 * tmp;
          rtmp3[idx] -= mul3 * tmp;
        }
      }
      ++prow;
      pc2 = rtmp2 + prow;
      pc3 = rtmp3 + prow;
      if (*pc3 != 0.0){
        pj   = nbj + bd[prow];
        if(*pc2 ==0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
        mul3 = (*pc3)/(*pc2);
        *pc3 = mul3;
        nz   = bi[prow+1] - bd[prow] - 1;
        PLogFlops(2*2*nz);
        for (j=0; j<nz; j++) {
          idx = pj[j] + shift;
          tmp = rtmp2[idx];
          rtmp3[idx] -= mul3 * tmp;
        }
      }
      nz  = bi[row+1] - bi[row];
      pj  = bj + bi[row];
      pc1 = ba + bi[row];
      pc2 = ba + bi[row+1];
      pc3 = ba + bi[row+2];
      if (rtmp1[row] == 0.0 || rtmp2[row+1] == 0.0 || rtmp3[row+2]==0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
      rtmp1[row]   = 1.0/rtmp1[row];
      rtmp2[row+1] = 1.0/rtmp2[row+1];
      rtmp3[row+2] = 1.0/rtmp3[row+2];
      for ( j=0; j<nz; j++ ) {
        idx    = pj[j];
        pc1[j] = rtmps1[idx];
        pc2[j] = rtmps2[idx];
        pc3[j] = rtmps3[idx];
      }
      break;
    default:
      SETERRQ(1,"MatLUFactorNumeric_SeqAIJ_Inode: Node size not yet supported \n");
    }
    row += nsz;                 /* Update the row */
  } 
  PetscFree(rtmp1);
  PetscFree(tmp_vec);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor      = FACTOR_LU;
  C->assembled   = PETSC_TRUE;
  PLogFlops(b->n);
  return 0;
}









