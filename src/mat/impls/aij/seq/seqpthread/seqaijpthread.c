#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

#if defined(PETSC_HAVE_CPU_SET_T)
extern void DoCoreAffinity(void);
#else
#define DoCoreAffinity();
#endif

extern PetscMPIInt PetscMaxThreads;
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt);

typedef struct {
  const MatScalar* matdata;
  const PetscScalar* vecdata;
  PetscScalar* vecout;
  const PetscInt* colindnz;
  const PetscInt* rownumnz;
  PetscInt numrows;
  const PetscInt* specidx;
  PetscInt nzr;
} MatMult_KernelData;

void* MatMult_Kernel(void *arg)
{

  MatMult_KernelData *data = (MatMult_KernelData*)arg;
  PetscScalar       sum;
  const MatScalar   *aabase = data->matdata,*aa;
  const PetscScalar *x = data->vecdata;
  PetscScalar       *y = data->vecout;
  const PetscInt    *ajbase = data->colindnz,*aj;
  const PetscInt    *ii = data->rownumnz;
  PetscInt          m  = data->numrows;
  const PetscInt    *ridx = data->specidx;
  PetscInt          i,n,nonzerorow = 0;

  DoCoreAffinity();
  if(ridx!=NULL) {
    for (i=0; i<m; i++){
      n   = ii[i+1] - ii[i];
      aj  = ajbase + ii[i];
      aa  = aabase + ii[i];
      sum = 0.0;
      if(n>0) {
        PetscSparseDensePlusDot(sum,x,aa,aj,n);
        nonzerorow++;
      }
      y[*ridx++] = sum;
    }
  }
  else {
    PetscInt ibase = data->nzr;
    for (i=0; i<m; i++) {
      n   = ii[i+1] - ii[i];
      aj  = ajbase + ii[i];
      aa  = aabase + ii[i];
      sum  = 0.0;
      if(n>0) {
        PetscSparseDensePlusDot(sum,x,aa,aj,n);
        nonzerorow++;
      }
      y[i+ibase] = sum;
    }
  }
  data->nzr = nonzerorow;
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqAIJPThread"
PetscErrorCode MatMult_SeqAIJPThread(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n,nonzerorow=0;
  PetscBool         usecprow=a->compressedrow.use;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  if(usecprow) {
    PetscInt          NumPerThread,iindex;
    const MatScalar   *aa = a->a;
    const PetscInt    *aj = a->j,*ii = a->compressedrow.i,*ridx=a->compressedrow.rindex;
    PetscInt          i,iStartVal,iEndVal,iStartIndex,iEndIndex;
    const PetscInt    iNumThreads = PetscMaxThreads;  /* this number could be different */
    MatMult_KernelData* kerneldatap = (MatMult_KernelData*)malloc(iNumThreads*sizeof(MatMult_KernelData));
    MatMult_KernelData** pdata = (MatMult_KernelData**)malloc(iNumThreads*sizeof(MatMult_KernelData*));

    m    = a->compressedrow.nrows;
    NumPerThread = ii[m]/iNumThreads;
    iindex = 0;
    for(i=0; i<iNumThreads;i++) {
      iStartIndex = iindex;
      iStartVal = ii[iStartIndex];
      iEndVal = iStartVal;
      /* determine number of rows to process */
      while(iEndVal-iStartVal<NumPerThread) {
	iindex++;
	iEndVal = ii[iindex];
      }
      /* determine whether to go back 1 */
      if(iEndVal-iStartVal-NumPerThread>NumPerThread-(ii[iindex-1]-iStartVal)) {
	iindex--;
	iEndVal = ii[iindex];
      }
      iEndIndex = iindex;
      kerneldatap[i].matdata  = aa;
      kerneldatap[i].vecdata  = x;
      kerneldatap[i].vecout   = y;
      kerneldatap[i].colindnz = aj;
      kerneldatap[i].rownumnz = ii + iStartIndex;
      kerneldatap[i].numrows  = iEndIndex - iStartIndex + 1;
      kerneldatap[i].specidx  = ridx + iStartVal;
      kerneldatap[i].nzr      = 0;
      pdata[i] = &kerneldatap[i];
      iindex++;
    }
    ierr = MainJob(MatMult_Kernel,(void**)pdata,iNumThreads);
    /* collect results */
    for(i=0; i<iNumThreads; i++) {
      nonzerorow += kerneldatap[i].nzr;
    }
    free(kerneldatap);
    free(pdata);
  }
  else {
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  fortranmultaij_(&m,x,a->i,a->j,a->a,y);
#else
  PetscInt            i,iindex;
    const MatScalar   *aa = a->a;
    const PetscInt    *aj = a->j,*ii = a->i;
    const PetscInt    iNumThreads = PetscMaxThreads;  /* this number could be different */
    PetscInt          Q = m/iNumThreads;
    PetscInt          R = m-Q*iNumThreads;
    PetscBool         S;

    MatMult_KernelData* kerneldatap = (MatMult_KernelData*)malloc(iNumThreads*sizeof(MatMult_KernelData));
    MatMult_KernelData** pdata = (MatMult_KernelData**)malloc(iNumThreads*sizeof(MatMult_KernelData*));

    iindex = 0;
    for(i=0; i<iNumThreads;i++) {
      S = (PetscBool)(i<R);
      kerneldatap[i].matdata  = aa;
      kerneldatap[i].vecdata  = x;
      kerneldatap[i].vecout   = y;
      kerneldatap[i].colindnz = aj;
      kerneldatap[i].rownumnz = ii + iindex;
      kerneldatap[i].numrows  = S?Q+1:Q;
      kerneldatap[i].specidx  = PETSC_NULL;
      kerneldatap[i].nzr      = iindex; /* serves as the 'base' row (needed to access correctly into output vector y) */
      pdata[i] = &kerneldatap[i];
      iindex += kerneldatap[i].numrows;
    }
    MainJob(MatMult_Kernel,(void**)pdata,iNumThreads);
    /* collect results */
    for(i=0; i<iNumThreads; i++) {
      nonzerorow += kerneldatap[i].nzr;
    }
    free(kerneldatap);
    free(pdata);
#endif
  }

  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PTHREADCLASSES)
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJPThread"
PetscErrorCode  MatCreate_SeqAIJPThread(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);
  B->ops->mult = MatMult_SeqAIJPThread;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJPTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif
