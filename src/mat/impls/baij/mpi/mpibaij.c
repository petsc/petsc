#ifndef lint
static char vcid[] = "$Id: mbaijfact.c,v 1.1 1996/06/03 19:55:56 balay Exp $";
#endif

#include "mpibaij.h"


static int MatDestroy_MPIBAIJ(PetscObject obj)
{
  Mat         mat = (Mat) obj;
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  int         ierr;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",baij->M,baij->N);
#endif

  PetscFree(baij->rowners); 
  ierr = MatDestroy(baij->A); CHKERRQ(ierr);
  ierr = MatDestroy(baij->B); CHKERRQ(ierr);
  if (baij->colmap) PetscFree(baij->colmap);
  if (baij->garray) PetscFree(baij->garray);
  if (baij->lvec)   VecDestroy(baij->lvec);
  if (baij->Mvctx)  VecScatterDestroy(baij->Mvctx);
  if (baij->rowvalues) PetscFree(baij->rowvalues);
  PetscFree(baij); 
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,0,
  0};
                                

/*@C
   MatCreateMPIBAIJ - Creates a sparse parallel matrix in block AIJ format
   (block compressed row).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Input Parameters:
.  comm - MPI communicator
.  bs   - size of blockk
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated 
           if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated 
           if n is given)
.  d_nz  - number of block nonzeros per block row in diagonal portion of local 
           submatrix  (same for all local rows)
.  d_nzz - number of block nonzeros per block row in diagonal portion of local 
           submatrix or null (possibly different for each row).  You must leave 
           room for the diagonal entry even if it is zero.
.  o_nz  - number of block nonzeros per block row in off-diagonal portion of local
           submatrix (same for all local rows).
.  o_nzz - number of block nonzeros per block row in off-diagonal portion of local 
           submatrix or null (possibly different for each row).

   Output Parameter:
.  A - the matrix 

   Notes:
   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Storage Information:
   For a square global matrix we define each processor's diagonal portion 
   to be its local rows and the corresponding columns (a square submatrix);  
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix). 

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set 
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

$          0 1 2 3 4 5 6 7 8 9 10 11
$         -------------------
$  row 3  |  o o o d d d o o o o o o
$  row 4  |  o o o d d d o o o o o o
$  row 5  |  o o o d d d o o o o o o
$         -------------------
$ 

   Thus, any entries in the d locations are stored in the d (diagonal) 
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQAIJ format for compressed row storage.

   Now d_nz should indicate the number of nonzeros per row in the d matrix,
   and o_nz should indicate the number of nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For additional details, see the users manual
   chapter on matrices and the file $(PETSC_DIR)/Performance.

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues()
@*/
int MatCreateMPIBAIJ(MPI_Comm comm,int bs,int m,int n,int M,int N,
                    int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  Mat          B;
  Mat_MPIBAIJ  *b;
  int          ierr, i,sum[2],work[2],mbs,nbs,Mbs,Nbs;

  if (bs < 1) SETERRQ(1,"MatCreateMPIBAIJ: invalid block size specified");
  *A = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATMPIBAIJ,comm);
  PLogObjectCreate(B);
  B->data       = (void *) (b = PetscNew(Mat_MPIBAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_MPIBAIJ));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy    = MatDestroy_MPIBAIJ;
  /*
  B->view       = MatView_MPIBAIJ;
  */
  B->factor     = 0;
  B->assembled  = PETSC_FALSE;

  b->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&b->rank);
  MPI_Comm_size(comm,&b->size);

  if (m == PETSC_DECIDE && (d_nnz != PETSC_NULL || o_nnz != PETSC_NULL)) 
    SETERRQ(1,"MatCreateMPIAIJ:Cannot have PETSC_DECIDE rows but set d_nnz or o_nnz");

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    mbs = m/bs; nbs = n/bs;
    if (mbs*bs != m || nbs*n != nbs) SETERRQ(1,"MatCreateMPIBAIJ: No of local rows, cols must be divisible by blocksize");
    MPI_Allreduce( work, sum,2,MPI_INT,MPI_SUM,comm );
    if (M == PETSC_DECIDE) {M = sum[0]; Mbs = M/bs;}
    if (N == PETSC_DECIDE) {N = sum[1]; Nbs = N/bs;}
  }
  if (m == PETSC_DECIDE) {
    Mbs = M/bs;
    if (Mbs*bs != M) SETERRQ(1,"MatCreateMPIBAIJ: No of global rows must be divisible by blocksize");
    mbs = Mbs/b->size + ((Mbs % b->size) > b->rank);
    m   = mbs*bs;
  }
  if (n == PETSC_DECIDE) {
    Nbs = N/bs;
    if (Nbs*bs != N) SETERRQ(1,"MatCreateMPIBAIJ: No of global cols must be divisible by blocksize");
    nbs = Nbs/b->size + ((Nbs % b->size) > b->rank);
    n   = nbs*bs;
  }

  b->m = m; B->m = m;
  b->n = n; B->n = n;
  b->N = N; B->N = N;
  b->M = M; B->M = M;
  b->bs  = bs;
  b->bs2 = bs*bs;
  b->mbs = mbs;
  b->nbs = nbs;
  b->Mbs = Mbs;
  b->Nbs = Nbs;

  /* build local table of row and column ownerships */
  b->rowners = (int *) PetscMalloc(2*(b->size+2)*sizeof(int)); CHKPTRQ(b->rowners);
  PLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIBAIJ));
  b->cowners = b->rowners + b->size + 1;
  MPI_Allgather(&mbs,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart = b->rowners[b->rank]; 
  b->rend   = b->rowners[b->rank+1]; 
  MPI_Allgather(&nbs,1,MPI_INT,b->cowners+1,1,MPI_INT,comm);
  b->cowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart = b->cowners[b->rank]; 
  b->cend   = b->cowners[b->rank+1]; 

  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,bs,m,n,d_nz,d_nnz,&b->A); CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,bs,m,N,o_nz,o_nnz,&b->B); CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&b->stash); CHKERRQ(ierr);
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = 1;

  /* stuff used for matrix vector multiply */
  b->lvec      = 0;
  b->Mvctx     = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  *A = B;
  return 0;
}
