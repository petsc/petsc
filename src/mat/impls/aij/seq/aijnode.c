#include "aij.h"                

int Mat_AIJ_CheckInode(Mat);
int MatSolve_SeqAIJ_Inode(Mat ,Vec , Vec );

/* ----------------------------------------------------------- */

static int MatMult_SeqAIJ_Inode(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ       *a = (Mat_SeqAIJ *) A->data; 
  Scalar           *x, *y;
  register Scalar  sum1, sum2, sum3, sum4, sum5, tmp0, tmp1;
  register Scalar  *v1, *v2, *v3, *v4, *v5;
  register int     *idx, i1, i2, n, i, row;
  int              node_max, *ns, *ii, nsz, sz;
  int              m = a->m, shift = a->indexshift;
  
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
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
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
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
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
      
      for( n = 0; n< sz-1; n+=2) {
        i1   = idx[0];
        i2   = idx[1];
        tmp0 = x[i1];
        tmp1 = x[i2]; 
        idx += 2;
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
      }
      if(n   == sz-1){
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
      
      for( n = 0; n< sz-1; n+=2) {
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
      if(n   == sz-1){
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
      
      for( n = 0; n< sz-1; n+=2) {
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
  int        i, j, m, nzx, nzy, *idx, *idy;
  int        * ns,*ii, node_count, blk_size, limit;

  limit      = 5;               /* Mult/Solve Can't Handle more than 5 */
  OptionsGetInt(0, "-mat_aij_inode_limit", &limit);
  if(limit >5 ) limit = 5;
  m = a->m;        
  if(!a->inode.size){
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
      if(PetscMemcmp((char *)idx,(char *)idy,nzx*sizeof(int))) break;
    }
    ns[node_count++] = blk_size;
    idx +=blk_size*nzx;
    i    = j;
  }
  if( node_count < 1.1 * m){   /* .90 is chosen arbitarily. */
    A->ops.mult         = MatMult_SeqAIJ_Inode;
    A->ops.solve        = MatSolve_SeqAIJ_Inode;
    a->inode.node_count = node_count;
    a->inode.size       = ns;
    PLogInfo((PetscObject)A, "Found %d nodes. Limit used : %d. Using Inode_Routines\n", node_count, limit);
  } else {
    PetscFree (ns);
    a->inode.node_count=0;
    PLogInfo((PetscObject)A, "Found %d nodes.Limit used : %d. Not using Inode_routines\n",node_count, limit);
  }
  return 0;
}

/* ----------------------------------------------------------- */
int MatSolve_SeqAIJ_Inode(Mat A,Vec bb, Vec xx)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *) A->data;
  IS              iscol = a->col, isrow = a->row;
  int             *r,*c, ierr, i, j, n = a->m, *ai = a->i;
  int             nz,shift = a->indexshift, *a_j = a->j;
  Scalar          *x,*b, *a_a = a->a;
  register int    node_max, *ns,row, nsz, aii,*vi,*ad, *aj, i0, i1;
  register Scalar *tmp, *tmps, *aa, tmp0, tmp1;
  register Scalar sum1, sum2, sum3, sum4, sum5,*v1, *v2, *v3,*v4, *v5;
  
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








