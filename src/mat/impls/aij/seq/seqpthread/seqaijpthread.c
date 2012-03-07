
#include <private/vecimpl.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>
#include <../src/mat/impls/aij/seq/seqpthread/seqaijpthread.h>

static PetscInt    mats_created=0;

void* MatMult_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*ajbase = (const PetscInt*)data->aj,*aj;
  const MatScalar   *aabase = (const MatScalar*)data->aa,*aa;
  const PetscScalar *x = (const PetscScalar*)data->x; 
  PetscScalar       *y = data->y;
  PetscInt           nrows = data->nrows;
  PetscInt           nz,i;
  PetscScalar        sum;

  DoCoreAffinity();
  data->nonzerorow = 0;
  for(i=0;i<nrows;i++) {
    nz = ai[i+1] - ai[i];
    aj = ajbase + ai[i];
    aa = aabase + ai[i];
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
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt          *trstarts=tmap->trstarts;
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

  for(i=0;i < tmap->nthreads;i++) {
    mat_kerneldatap[i].ai    = ai + trstarts[i];
    mat_kerneldatap[i].aa    = aa;
    mat_kerneldatap[i].aj    = aj;
    mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
    mat_kerneldatap[i].x     = x;
    mat_kerneldatap[i].y     = y + trstarts[i];
    mat_pdata[i]             = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatMult_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);

  for(i=0;i< tmap->nthreads;i++) nonzerorow += mat_kerneldatap[i].nonzerorow;

  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

void* MatMultAdd_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*aj = (const PetscInt*)data->aj;
  const MatScalar   *aa = (const MatScalar*)data->aa;
  const PetscScalar *x = (const PetscScalar*)data->x; 
  PetscScalar       *y = data->y;
  PetscScalar       *z = data->z;
  PetscInt           nrows = data->nrows;
  PetscInt           nz,i;
  PetscScalar        sum;

  DoCoreAffinity();
  data->nonzerorow = 0;
  for(i=0;i<nrows;i++) {
    nz = ai[i+1] - ai[i];
    aj = data->aj + ai[i];
    aa = data->aa + ai[i];
    sum = y[i];
    data->nonzerorow += (nz>0);
    PetscSparseDensePlusDot(sum,x,aa,aj,nz);
    z[i] = sum;
  }
  return(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqAIJPThread"
PetscErrorCode MatMultAdd_SeqAIJPThread(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt          *trstarts=tmap->trstarts;
  PetscScalar       *x,*y,*z;
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
  if(zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else z = y;

  for(i=0;i < tmap->nthreads;i++) {
    mat_kerneldatap[i].ai    = ai + trstarts[i];
    mat_kerneldatap[i].aa    = aa;
    mat_kerneldatap[i].aj    = aj;
    mat_kerneldatap[i].nrows = trstarts[i+1]-trstarts[i];
    mat_kerneldatap[i].x     = x;
    mat_kerneldatap[i].y     = y + trstarts[i];
    mat_kerneldatap[i].z     = z + trstarts[i];
    mat_pdata[i]             = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatMultAdd_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);

  for(i=0;i< tmap->nthreads;i++) nonzerorow += mat_kerneldatap[i].nonzerorow;

  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if(zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_SeqAIJPThread"
PetscErrorCode MatSetUp_SeqAIJPThread(Mat A)
{
  PetscErrorCode ierr;
  PetscThreadsLayout tmap=A->rmap->tmap;

  PetscFunctionBegin;
  ierr = MatSetUp_SeqAIJ(A);CHKERRQ(ierr);
  tmap->N = A->rmap->n;
  ierr = PetscThreadsLayoutSetUp(tmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJPThread"
PetscErrorCode MatDestroy_SeqAIJPThread(Mat A)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  if(!A->rmap->refcnt) {
    ierr = PetscThreadsLayoutDestroy(&A->rmap->tmap);CHKERRQ(ierr);
  }

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
#define __FUNCT__ "MatSetNThreads"
/*
   MatSetNThreads - Set the number of threads to be used for matrix operations.

   Input Parameters
+  A - the matrix
-  nthreads - number of threads

   Level: intermediate

   Concepts: matrix^setting number of threads

.seealso: MatCreateSeqAIJPThread()
*/
PetscErrorCode MatSetNThreads(Mat A,PetscInt nthreads)
{
  PetscErrorCode     ierr;
  PetscInt           nthr;
  PetscBool          flg;
  PetscThreadsLayout tmap=A->rmap->tmap;

  PetscFunctionBegin;
  
  if(!tmap) {
    ierr = PetscThreadsLayoutCreate(&tmap);CHKERRQ(ierr);
    A->rmap->tmap = tmap;
  }

  if(nthreads == PETSC_DECIDE) {
    ierr = PetscOptionsInt("-mat_threads","Set number of threads to be used for matrix operations","MatSetNThreads",PetscMaxThreads,&nthr,&flg);CHKERRQ(ierr);
    if(flg && nthr > PetscMaxThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Mat A: threads requested %D, Max. threads initialized %D",nthr,PetscMaxThreads);
    }
    if(!flg) nthr = PetscMaxThreads;
    tmap->nthreads = nthr+PetscMainThreadShareWork;
  } else {
    if(nthreads > PetscMaxThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Mat A: threads requested %D, Max. threads initialized %D",nthreads,PetscMaxThreads);
    }
    tmap->nthreads = nthreads + PetscMainThreadShareWork;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetThreadAffinities"
/*
   MatSetThreadAffinities - Sets the CPU affinities of matrix threads.

   Input Parameters
+  A - the matrix
-  affinities - list of cpu affinities for threads.

   Notes:
   Must set affinities for all the threads used with the matrix.
   size(affinities[]) = nthreads
   Use affinities[] = PETSC_NULL for PETSc to set the thread affinities.

   Options Database Keys:
+  -mat_thread_affinities - Comma seperated list of thread affinities

   Level: intermediate

   Concepts: matrices^thread cpu affinity

.seealso: MatPThreadGetThreadAffinities()
*/
PetscErrorCode MatSetThreadAffinities(Mat A,const PetscInt affinities[])
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt          nmax=PetscMaxThreads+PetscMainThreadShareWork;
  PetscBool         flg;

  PetscFunctionBegin;

  if(!tmap) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must set the number of threads before setting thread affinities");
  }

  ierr = PetscMalloc(tmap->nthreads*sizeof(PetscInt),&tmap->affinity);CHKERRQ(ierr);

  if(affinities == PETSC_NULL) {
    PetscInt *thread_affinities;
    ierr = PetscMalloc(nmax*sizeof(PetscInt),&thread_affinities);CHKERRQ(ierr);
    /* Check if run-time option is set */
    ierr = PetscOptionsIntArray("-mat_thread_affinities","Set CPU affinity for each thread","MatSetThreadAffinities",thread_affinities,&nmax,&flg);CHKERRQ(ierr);
    if(flg) {
      if(nmax != tmap->nthreads-PetscMainThreadShareWork) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, matrix A Threads = %D, CPU affinities set = %D",tmap->nthreads-PetscMainThreadShareWork,nmax);
      }
      ierr = PetscMemcpy(tmap->affinity+PetscMainThreadShareWork,thread_affinities,(tmap->nthreads-PetscMainThreadShareWork)*sizeof(PetscInt));
      ierr = PetscFree(thread_affinities);CHKERRQ(ierr);
    } else {
      /* Reuse the core affinities set for first s->nthreads */
      ierr = PetscMemcpy(tmap->affinity+PetscMainThreadShareWork,ThreadCoreAffinity,(tmap->nthreads-PetscMainThreadShareWork)*sizeof(PetscInt));
    }
  } else {
    /* Set user provided affinities */
    ierr = PetscMemcpy(tmap->affinity+PetscMainThreadShareWork,affinities,(tmap->nthreads-PetscMainThreadShareWork)*sizeof(PetscInt));
  }
  if(PetscMainThreadShareWork) tmap->affinity[0] = MainThreadCoreAffinity;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJPThread"
PetscErrorCode MatCreate_SeqAIJPThread(Mat B)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *b;
  PetscThreadsLayout tmap=B->rmap->tmap;

  PetscFunctionBegin;
  ierr = PetscThreadsInitialize(PetscMaxThreads);CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(B);
  b = (Mat_SeqAIJ*)B->data;
  /* Set inodes off */
  b->inode.use = PETSC_FALSE;

  if(!B->rmap->tmap) {
    ierr = PetscThreadsLayoutCreate(&B->rmap->tmap);CHKERRQ(ierr);
    tmap = B->rmap->tmap;
  }

  /* Set the number of threads */
  if(tmap->nthreads == PETSC_DECIDE) {
    ierr = MatSetNThreads(B,PETSC_DECIDE);CHKERRQ(ierr);
  }
  /* Set thread affinities */
  if(!tmap->affinity) {
    ierr = MatSetThreadAffinities(B,PETSC_NULL);CHKERRQ(ierr);
  }

  B->ops->mult    = MatMult_SeqAIJPThread;
  B->ops->destroy = MatDestroy_SeqAIJPThread;
  B->ops->multadd = MatMultAdd_SeqAIJPThread;
  B->ops->setup   = MatSetUp_SeqAIJPThread;

  if(mats_created == 0) {
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Mat_KernelData),&mat_kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Mat_KernelData*),&mat_pdata);CHKERRQ(ierr);
  }
  mats_created++;

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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays()

*/
PetscErrorCode  MatCreateSeqAIJPThread(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nthreads,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetNThreads(*A,nthreads);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
