#ifndef lint
static char vcid[] = "$Id: aijnode.c,v 1.8 1995/11/19 00:47:05 bsmith Exp balay $";
#endif
/*
    Provides high performance routines for the AIJ (compressed row) storage 
  format by taking advantage of rows with identical non-zero structure (I-nodes).
*/
#include "aij.h"                

extern int Mat_AIJ_CheckInode(Mat);
extern int MatSolve_SeqAIJ_Inode(Mat ,Vec , Vec );

/*
   MatToSymmetricIJ_SeqAIJ_Inode - Convert a sparse AIJ matrix to IJ format.
           Creates a new block matrix using inode info (The dimension of
           new matrix is same as the no of Inodes).
           Uses only the lower triangular part of the matrix.

    Description:
    Take the data in the row-oriented sparse storage, and using the inode
    info, creates a block matrix, builds the IJ data for the Block Matrix.
    Return 0 on success. Produces the ij for a symmetric matrix by only 
    using the lower triangular part of the block matrix.

    Input Parameters:
.   Matrix - matrix to convert

    Output Parameters:
.   ia     - ia part of IJ representation of the block matrix (row information)
.   ja     - ja part of the block matrix(column indices)

    Notes:
$    Both ia and ja may be freed with PetscFree();
$    This routine is provided for ordering routines that require a 
$    symmetric structure.  It is used in SpOrder (and derivatives) since
$    those routines call SparsePak routines that expect a symmetric 
$    matrix.
*/
int MatToSymmetricIJ_SeqAIJ_Inode( Mat_SeqAIJ *A, int **iia, int **jja )
{
  int *work,*ia,*ja,*j, nz, m , row, wr, col, shift = A->indexshift;
  int *tns, *ns = A->inode.size, i1, i2;

  m = A->inode.node_count;
  /* allocate space for reformated inode structure */
  tns = (int *) PetscMalloc((m +1 )*sizeof(int)); CHKPTRQ(tns);
  for(i1 = 0, tns[0] =0; i1 < m; ++i1) tns[i1+1] = tns[i1]+ ns[i1];

  /* allocate space for row pointers */
  *iia = ia = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(ia);
  PetscMemzero(ia,(m+1)*sizeof(int));
  work = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(work);

  /* determine the number of columns in each row */
  ia[0] = 1;
  for (i1=0 ; i1 < m; ++i1) {
    row= tns[i1];
    j  = A->j + A->i[row] + shift;
    /* For each row,assume the colums to be of the same inode pattern
      of rows. Now identify the column indices of the *nonzero* inodes */
    i2 = 0;                     /* Col inode index */
    col = *j + shift;
    while (i2 <= i1) {
      while (col > tns[i2]) ++i2; /* skip until corresponding inode is found*/
      if(i2 >i1 ) {printf("goofy\n"); break;}
      if(i2 <i1 ) ia[i1+1]++;
      ia[i2+1]++;
      i2++;                     /* Start col of next node */
      while((col = *j + shift)< tns[i2]) ++j; /* goto the first col of this node*/
    }
  }

  /* shift ia[i] to point to next row */
  for ( i1=1; i1<m+1; i1++ ) {
    row        = ia[i1-1];
    ia[i1]     += row;
    work[i1-1] = row - 1;
  }

  /* allocate space for column pointers */
  nz = ia[m] + (!shift);
  *jja = ja = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ja);

 /* loop over lower triangular part putting into ja */ 
  for (i1=0, row = 0; i1 < m; ++i1) {
    row= tns[i1];
    j  = A->j + A->i[row] + shift;
    i2 = 0;                     /* Col inode index */
    col = *j + shift;
    while (i2 <= i1) {
      while (col > tns[i2]) ++i2; /* skip until corresponding inode is found*/
      if(i2 >i1 ) {printf("goofy\n"); break;}
      if(i2 <i1 ) {wr = work[i2]; work[i2] = wr +1; ja[wr] = i1 +1;}
      wr = work[i1]; work[i1] = wr + 1; ja[wr] = i2 + 1;
      ++i2;
      while((col = *j + shift)< tns[i2]) ++j; /* Skip all the col indices in this node */
    }
  }
  PetscFree(work);
  PetscFree(tns);
  return 0;
}

/*
   MatGetReordering_SeqAIJ_Inode - Convert a sparse AIJ matrix to IJ format.
           Creates a new block matrix using inode info (The dimension of
           new matrix is same as the no of Inodes).
           Uses only the lower triangular part of the matrix.

    Description:
    Take the data in the row-oriented sparse storage, and using the inode
    info, creates a block matrix, builds the IJ data for the Block Matrix.
    Return 0 on success. Produces the ij for a symmetric matrix by only 
    using the lower triangular part of the block matrix.

    Input Parameters:
.   Matrix - matrix to convert

    Output Parameters:
.   ia     - ia part of IJ representation of the block matrix (row information)
.   ja     - ja part of the block matrix(column indices)

    Notes:
$    Both ia and ja may be freed with PetscFree();
$    This routine is provided for ordering routines that require a 
$    symmetric structure.  It is used in SpOrder (and derivatives) since
$    those routines call SparsePak routines that expect a symmetric 
$    matrix.
*/

static int MatGetReordering_SeqAIJ_Inode(Mat A,MatOrdering type,IS *rperm, IS *cperm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr, *ia, *ja,n = a->n,*idx,i,j, *ridx, *cidx, *invridx, *invcidx;
  int        row,*permr, *permc,m ,*ns, *tns, start_val, end_val, indx;
  IS         ris= 0, cis = 0, invris, invcis;
  Viewer     V1, V2;
  if (!a->assembled) SETERRQ(1,"MatGetReordering_SeqAIJ_Inode:Not for unassembled matrix");

  /* 
     this is tacky: In the future when we have written special factorization
     and solve routines for the identity permutation we should use a 
     stride index set instead of the general one.
  */
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

  ierr  = MatToSymmetricIJ_SeqAIJ_Inode( a, &ia, &ja ); CHKERRQ(ierr);
  ierr  = MatGetReordering_IJ(m,ia,ja,type,&ris,&cis); CHKERRQ(ierr);

  tns   = (int *) PetscMalloc((m +1 )*sizeof(int)); CHKPTRQ(tns);
  permr = (int *) PetscMalloc( (2*a->n+1)*sizeof(int) ); CHKPTRQ(permr);
  permc = permr + n;

  ierr  = ISInvertPermutation(ris, &invris); CHKERRQ(ierr);
  ierr  = ISInvertPermutation(cis, &invcis); CHKERRQ(ierr);
  ierr  = ISGetIndices(ris,&ridx); CHKERRQ(ierr);
  ierr  = ISGetIndices(cis,&cidx); CHKERRQ(ierr);
  ierr  = ISGetIndices(invris,&invridx); CHKERRQ(ierr);
  ierr  = ISGetIndices(invcis,&invcidx); CHKERRQ(ierr);

  /* Form the inode structure for the rows of permuted matric using inv perm*/
  for(i = 0, tns[0] =0; i < m; ++i) tns[i+1] = tns[i]+ ns[invridx[i]];

  /* Consturct the permutations for rows*/
  for( i = 0,row = 0; i<m; ++i){
    indx      = ridx[i];
    start_val = tns[indx];
    end_val   = tns[indx + 1];
    for(j = start_val; j< end_val; ++j, ++row) permr[row]= j;
  }

   /* Form the inode structure for the cols of permuted matric using inv perm*/
  for(i = 0, tns[0] =0; i < m; ++i) tns[i+1] = tns[i]+ ns[invcidx[i]];

 /*Construct permutations for columns*/
  for( i = 0,row =0; i<m ; ++i){
    indx      = cidx[i];
    start_val = tns[indx];
    end_val   = tns[indx + 1];
    for(j = start_val; j< end_val; ++j, ++row)
      permc[row]= j;
  }

  ierr = ISCreateSeq(MPI_COMM_SELF,n,permr,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateSeq(MPI_COMM_SELF,n,permc,cperm); CHKERRQ(ierr);
  ISSetPermutation(*cperm);
 
  ViewerFileOpenASCII(MPI_COMM_SELF,"row_is", &V1);
  ViewerFileOpenASCII(MPI_COMM_SELF,"col_is", &V2);
  ISView(ris,V1);
  ISView(cis,V2);
  ViewerDestroy(V1);
  ViewerDestroy(V2);
 
  PetscFree(ia); PetscFree(ja); PetscFree(permr);
  ISDestroy(cis); ISDestroy(invcis);
  ISDestroy(ris); ISDestroy(invris);
  PetscFree(tns);
  return 0; 
}


/* ----------------------------------------------------------- */

int MatMult_SeqAIJ_Inode(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  Scalar     sum1, sum2, sum3, sum4, sum5, tmp0, tmp1;
  Scalar     *v1, *v2, *v3, *v4, *v5,*x, *y;
  int        *idx, i1, i2, n, i, row,node_max, *ns, *ii, nsz, sz;
  int        m = a->m, shift = a->indexshift;
  
  if (!a->assembled) SETERRQ(1,"MatMult_SeqAIJ_Inode: Not for unassembled matrix");
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
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        idx += 2;
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
        tmp0 = x[i1];
        tmp1 = x[i2];
        idx += 2;
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
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        idx += 2;
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
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        idx += 2;
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
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        idx += 2;
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
      v1      =v5;              /* Since the next block to be processed starts there*/
      idx    +=4*sz;
      break;
    default :
      SETERRQ(1,"MatMult_SeqAIJ_Inode:Node size not yet supported");
    }
  }
  PLogFlops(2*a->nz - m);
  return 0;
}
/* ----------------------------------------------------------- */

int Mat_AIJ_CheckInode(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        i, j, m, nzx, nzy, *idx, *idy, *ns,*ii, node_count, blk_size, limit;

  limit      = 5;               /* Mult/Solve Can't Handle more than 5 */
  if (OptionsHasName(0, "-mat_aij_no_inode")) return 0;
  OptionsGetInt(0, "-mat_aij_inode_limit", &limit);
  if(limit > 5) limit = 5;
  m = a->m;        
  if (!a->inode.size && m){
    ns = (int *)PetscMalloc(m*sizeof(int));  CHKPTRQ(ns);
  }
  else return 0;                /* Use the Already formed Inode info */

  i          = 0;
  node_count = 0; 
  idx        = a->j;
  ii         = a->i;
  while ( i < m){               /* For each row */
    nzx = ii[i+1] - ii[i];      /* No of non zeros*/
    /*Limits the no of elements in a node to 'limit'*/
    for(j=i+1, idy= idx, blk_size=1; j<m && blk_size <limit ;++j,++blk_size){
      nzy     = ii[j+1] - ii[j]; /* same no of nonzeros */
      if(nzy != nzx) break;
      idy    += nzx;            /* Same nonzero pattern */
      if (PetscMemcmp((char *)idx,(char *)idy,nzx*sizeof(int))) break;
    }
    ns[node_count++] = blk_size;
    /*printf("%3d \t %d\n", i, blk_size);*/
    idx +=blk_size*nzx;
    i    = j;
  }
  if (OptionsHasName(0, "-mat_aij_reorder_inode")){
    A->ops.getreordering   = MatGetReordering_SeqAIJ_Inode;
    /*A->ops.lufactornumeric = MatLUFactorNumeric_SeqAIJ_Inode;*/
  }
  /* Update  Mat with new info. Later make ops default? */
  A->ops.mult          = MatMult_SeqAIJ_Inode;
  A->ops.solve         = MatSolve_SeqAIJ_Inode;
  a->inode.node_count  = node_count;
  a->inode.size        = ns;
  PLogInfo((PetscObject)A,"Found %d nodes. Limit used:%d.Using Inode_Routines\n",node_count,limit);
  return 0;
}

/* ----------------------------------------------------------- */
int MatSolve_SeqAIJ_Inode(Mat A,Vec bb, Vec xx)
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  +=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  +=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];  
        vi  +=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];   
        vi  +=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];   
        vi  +=2;
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
      printf ("** Node size not yet supported \n");
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  -=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  -=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  -=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  -=2;
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
        tmp0 = tmps[i0];
        tmp1 = tmps[i1];
        vi  -=2;
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
      printf ("*** Node size not yet supported \n");
    }
  }

  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&c); CHKERRQ(ierr);
  PLogFlops(2*a->nz - a->n);
  return 0;
}


int MatLUFactorNumeric_SeqAIJ_Inode(Mat A,Mat *B)
{
  Mat        C = *B;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data, *b = (Mat_SeqAIJ *)C->data;
  IS         iscol = b->col, isrow = b->row, isicol;
  int        *r,*ic, ierr, n = a->m, *bi = b->i, *bj = b->j, *ai = a->i, *aj = a->j;
  int        *ajtmp, *bjtmp, nz, row, prow, *ics, shift = a->indexshift, i, j, idx;
  int        *bd = b->diag, node_max, nsz, *ns;
  Scalar     *rtmp,*v, *pc1, *pc2, tmp, mul1, mul2, *ba = b->a, *aa = a->a; 
  
  /* These declarations are for optimizations.  They reduce the number of
     memory references that are made by locally storing information; the
     word "register" used here with pointers can be viewed as "private" or 
     "known only to me"
   */
  register Scalar *pv, *rtmps;
  register int    *pj;

  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc( (2*n+1)*sizeof(Scalar) ); CHKPTRQ(rtmp);
  rtmps = rtmp + shift; ics = ic + shift;
  
  if (!a->inode.size)SETERRQ(1,"MatLUFactorNumeric_SeqAIJ_Inode: Missing Inode Structure");
  node_max = a->inode.node_count;                
  ns       = a->inode.size;     /* Node Size array */
  
  for ( i=0,row=0; i<node_max; i++ ) { /* make sure row is updated */
    nsz   = ns[i];
    nz    = bi[row+1] - bi[row];
    bjtmp = bj + bi[row] + shift;

    switch (nsz){
    case 1:
      for  ( j=0; j<nz; j++ ){
        idx        = bjtmp[j];
        rtmps[idx] = 0.0;
      }
      break;
    
    case 2:
      for  ( j=0; j<nz; j++ ) {
        idx          = 2 * bjtmp[j];
        rtmps[idx]   = 0.0;
        rtmps[idx+1] = 0.0;
      }
      break;
    default:
      printf("Not yet supported - inode size %d \n", nsz);
    }

    /* load in initial (unfactored row) */
    idx      = r[row];
    nz       = ai[idx+1] - ai[idx];
    ajtmp = aj + ai[idx] + shift;
    v        = aa + ai[idx] + shift;
    switch (nsz){
    case 1:
      for ( j=0; j<nz; j++ ) {
        idx       = ics[ajtmp[j]];
        rtmp[idx] =  v[j];
      }
      break;
    case 2:
      for ( j=0; j<nz; j++ ) {
        idx       = 2 * ics[ajtmp[j]];
        rtmp[idx] =  v[j];
        rtmp[idx] =  v[j+1];
      }
      break;
    default:
      printf("Not yet supported - inode size %d \n", nsz);
    }

    prow = *bjtmp + shift; bjtmp += nsz;
    switch (nsz){
    case 1    :
      while (prow < row) {
        pc1 = rtmp + prow;
        if (*pc1 != 0.0){
          pv   = ba + bd[prow] + shift;
          pj   = bj + bd[prow] + (!shift);
          mul1 = *pc1 * *pv++;
          *pc1 = mul1;
          nz   = bi[prow+1] - bd[prow] - 1;
          PLogFlops(2*nz);
          for (j=0; j<nz; j++) rtmps[pj[j]] -= mul1 * pv[j];
        }
        prow = *bjtmp++ + shift;
      }
      break;
    case 2  :
      while (prow < row) {
        pc1 = rtmp + 2 * prow;
        pc2 = pc1 + 1;
        if (*pc1 != 0.0){        /* ????????????????? */
          pv   = ba + bd[prow] + shift;
          pj   = bj + bd[prow] + (!shift);
          mul1 = *pc1 * *pv;
          mul2 = *pc2 * *pv; ++pv;
          *pc1 = mul1;
          *pc2 = mul2;
          
          nz   = bi[prow+1] - bd[prow] - 1;
          PLogFlops(2*2*nz);
          for (j=0; j<nz; j++) {
            idx = pj[j];
            tmp = pv[j];
            rtmps[idx]   -= mul1 * tmp;
            rtmps[idx+1] -= mul2 * tmp;
          }
        }
        prow = *bjtmp++ + shift;
      }
      /* Now take care of the odd element*/
      pc2 = rtmp + 2 * prow;
      if (*pc1 != 0.0){        /* ????????????????? */
        pv      = ba + bd[prow] + shift;
        pj      = bj + bd[prow] + (!shift);
        mul2    = *pc2 / *(pc2 -1);
        *pc2     = mul2;
            
        nz   = bi[prow+1] - bd[prow] - 1;
        PLogFlops(2*2*nz);
        for (j=0; j<nz; j++) {
          idx = pj[j];
          tmp = pv[j];
          rtmps[idx]   -= mul1 * tmp;
          rtmps[idx+1] -= mul2 * tmp;
        }
      }
      break;
    default:
      printf("error\n");
    }
  
    
    /* finished row so stick it into b->a */
    pv = ba + bi[row] + shift;
    pj = bj + bi[row] + shift;
    nz = bi[row+1] - bi[row];
    if (rtmp[row] == 0.0) {SETERRQ(1,"MatLUFactorNumeric_SeqAIJ:Zero pivot");}
    rtmp[row] = 1.0/rtmp[row];
    for ( j=0; j<nz; j++ ) {pv[j] = rtmps[pj[j]];}
  } 
  PetscFree(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor      = FACTOR_LU;
  ierr = Mat_AIJ_CheckInode(C); CHKERRQ(ierr);
  b->assembled = 1;
  PLogFlops(b->n);
  return 0;
}




