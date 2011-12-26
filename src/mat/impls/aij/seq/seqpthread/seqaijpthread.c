#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

#if defined(PETSC_HAVE_CPU_SET_T)
extern void DoCoreAffinity(void);
#else
#define DoCoreAffinity();
#endif

extern PetscMPIInt PetscMaxThreads;
extern PetscInt    PetscMainThreadShareWork;
static PetscInt    mats_created=0;
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt);

typedef struct {
  PetscInt   *rstart; /* starting row numbers */
  PetscInt   *nrows; /* Number of rows */
  PetscInt   nthreads; /* Number of threads */
}Mat_SeqAIJPThread;

typedef struct {
  MatScalar *aa;
  PetscInt  *ai;
  PetscInt  *aj;
  PetscInt  *adiag;
  PetscScalar *x,*y;
  PetscInt   nrows;
  PetscInt   nonzerorow;
}Mat_KernelData;

Mat_KernelData *mat_kerneldatap;
Mat_KernelData **mat_pdata;

void* MatMult_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*aj = (const PetscInt*)data->aj;
  const MatScalar   *aa = (const MatScalar*)data->aa;
  const PetscScalar *x = (const PetscScalar*)data->x; 
  PetscScalar       *y = data->y;
  PetscInt           nrows = data->nrows;
  PetscInt           nz,i;
  PetscScalar        sum;

  DoCoreAffinity();
  data->nonzerorow = 0;
  for(i=0;i<nrows;i++) {
    nz = ai[i+1] - ai[i];
    aj = data->aj + ai[i];
    aa = data->aa + ai[i];
    sum = 0.0;
    data->nonzerorow += (nz>0);
    PetscSparseDensePlusDot(sum,x,aa,aj,nz);
    y[i] = sum;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqAIJPThread"
PetscErrorCode MatMult_SeqAIJPThread(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJPThread *ap = (Mat_SeqAIJPThread*)A->spptr;
  PetscScalar       *x,*y;
  PetscErrorCode    ierr;
  MatScalar         *aa = a->a;
  PetscInt          *ai = a->i, *aj = a->j;
  PetscInt          i,nonzerorow=0;;

  PetscFunctionBegin;

  if(a->compressedrow.use) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Compressed row format not supported yet");
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for(i=0;i < ap->nthreads;i++) {
    mat_kerneldatap[i].ai    = ai + ap->rstart[i];
    mat_kerneldatap[i].aa    = aa + ai[ap->rstart[i]];
    mat_kerneldatap[i].aj    = aj + aj[ap->rstart[i]];
    mat_kerneldatap[i].nrows = ap->nrows[i];
    mat_kerneldatap[i].x     = x;
    mat_kerneldatap[i].y     = y + ap->rstart[i];
    mat_pdata[i]             = &mat_kerneldatap[i];
  }
  ierr = MainJob(MatMult_Kernel,(void**)mat_pdata,ap->nthreads);

  for(i=0;i< ap->nthreads;i++) nonzerorow += mat_kerneldatap[i].nonzerorow;

  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJPThread"
PetscErrorCode MatDestroy_SeqAIJPThread(Mat A)
{
  Mat_SeqAIJPThread  *s = (Mat_SeqAIJPThread*)A->spptr;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = PetscFree2(s->rstart,s->nrows);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);

  mats_created--;
  /* Free the kernel data structure on the destruction of the last matrix */
  if(!mats_created) {
    ierr = PetscFree(mat_kerneldatap);CHKERRQ(ierr);
    ierr = PetscFree(mat_pdata);CHKERRQ(ierr);
  }
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPThreadSetNThreads"
/*
   MatPThreadSetNThreads - Set the number of threads to be used for matrix operations.

   Input Parameters
+  A - the matrix
-  nthreads - number of threads

   Level: intermediate

   Concepts: matrix^setting number of threads

.seealso: MatCreateSeqAIJPThread()
*/
PetscErrorCode MatPThreadSetNThreads(Mat A,PetscInt nthreads)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJPThread  *s = (Mat_SeqAIJPThread*)A->spptr;
  PetscInt           Q,R;
  PetscBool          S;
  PetscInt           i,iIndex=0;

  PetscFunctionBegin;
  if(nthreads > PetscMaxThreads) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Mat A: threads requested %D, Max. threads initialized %D",nthreads,PetscMaxThreads);
  if(s->nthreads != 0) {
    ierr = PetscFree2(s->rstart,s->nrows);CHKERRQ(ierr);
  }
  s->nthreads = nthreads+PetscMainThreadShareWork;
  Q = A->rmap->n/s->nthreads;
  R = A->rmap->n-Q*s->nthreads;

  /* Set starting row and nrows each thread */
  ierr = PetscMalloc2(s->nthreads,PetscInt,&s->rstart,s->nthreads,PetscInt,&s->nrows);CHKERRQ(ierr);
  for (i=0; i< s->nthreads; i++) {
    s->rstart[i] = iIndex;
    S = (PetscBool)(i<R);
    s->nrows[i] = S?Q+1:Q;
    iIndex += s->nrows[i];
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJPThread"
PetscErrorCode MatCreate_SeqAIJPThread(Mat B)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJPThread  *s;
  PetscInt            nthreads = PetscMaxThreads;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);
  ierr = PetscNewLog(B,Mat_SeqAIJPThread,&s);CHKERRQ(ierr);
  B->spptr = s;
  s->nthreads = 0;
  B->ops->mult    = MatMult_SeqAIJPThread;
  B->ops->destroy = MatDestroy_SeqAIJPThread;

  if(mats_created == 0) {
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Mat_KernelData),&mat_kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Mat_KernelData*),&mat_pdata);CHKERRQ(ierr);
  }
  mats_created++;

  ierr = PetscOptionsInt("-mat_threads","Set number of threads to be used with the matrix","MatPThreadSetNThreads",nthreads,&nthreads,PETSC_NULL);
  ierr = MatPThreadSetNThreads(B,nthreads);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJPTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJPThread"
/*
   MatCreateSeqAIJPThread - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format) using posix threads.  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nthreads - number of threads for matrix operations
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   It is recommended that one use the MatCreate(), MatSetSizes(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_no_inode  - Do not use inodes
-  -mat_inode_limit <limit> - Sets inode limit (max limit=5)

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays()

*/
PetscErrorCode  MatCreateSeqAIJPThread(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nthreads,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatPThreadSetNThreads(*A,nthreads);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
