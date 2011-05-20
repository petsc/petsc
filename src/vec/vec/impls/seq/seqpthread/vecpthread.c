
/*
   Implements the sequential pthread based vectors.
*/

#include <petscconf.h>
#include <private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <pthread.h>

typedef struct {
  const PetscScalar *x,*y;
  PetscInt          n;
  PetscScalar       result;
} VecDot_KernelData;

void *VecDot_Kernel(void *arg)
{
  VecDot_KernelData *data = arg;
  const PetscScalar *x, *y;
  PetscScalar LocalSum = 0;
  PetscInt    i,j,n;

  x = data->x;
  y = data->y;
  n = data->n;
  for (j=0; j<1000; j++) {
  for (i=0; i<n; i++) {
    LocalSum += x[i]*PetscConj(y[i]);
  }
  }
  //LocalSum = LocalSum/100.0;
  data->result = LocalSum;
  return NULL;
}
VecDot_KernelData kerneldatap[2]; //must match iNumThreads below

#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqPThread"
PetscErrorCode VecDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscErrorCode    ierr;
  PetscInt          i, iIndex = 0;
  void              *joinstatus;
  // VecDot_KernelData *kerneldatap;
  const PetscInt    iNumThreads = 2;
  pthread_t         aiThread[iNumThreads];
  PetscInt          Q = xin->map->n/(iNumThreads);
  PetscInt          R = xin->map->n-Q*(iNumThreads);
  PetscBool         S;

  //kerneldatap = (VecDot_KernelData*)malloc(iNumThreads*sizeof(VecDot_KernelData));

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<iNumThreads; i++) {
    S = i<R;
    kerneldatap[i].x = &xa[iIndex];
    kerneldatap[i].y = &ya[iIndex];
    kerneldatap[i].n = S?Q+1:Q;
    ierr = pthread_create(&aiThread[i], NULL, VecDot_Kernel, &kerneldatap[i]);CHKERRQ(ierr);
    iIndex += kerneldatap[i].n;
  }

  //code used if 'main' thread is to be a 'worker' too!
  //kerneldatap[iNumThreads].x = &xa[iIndex];
  //kerneldatap[iNumThreads].y = &ya[iIndex];
  //kerneldatap[iNumThreads].n = Q;
  //VecDot_Kernel(&kerneldatap[iNumThreads]);

  for (i=0; i<iNumThreads; i++) {
    pthread_join(aiThread[i], &joinstatus);
  }

  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  //*z = kerneldatap[iNumThreads].result;
  for(i=0; i<iNumThreads; i++) {
    *z += kerneldatap[i].result;
  }
  //free(kerneldatap);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_SeqPThread"
PetscErrorCode VecDestroy_SeqPThread(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy_Seq(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_SeqPThread"
PetscErrorCode  VecCreate_SeqPThread(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    *array;
  PetscInt       n = PetscMax(V->map->n,V->map->N);
  
  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if  (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQTHREAD on more than one process");
  ierr = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(array,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(V,array);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQPTHREAD);CHKERRQ(ierr);
  V->ops->dot             = VecDot_SeqPThread;
  PetscFunctionReturn(0);
}
EXTERN_C_END
