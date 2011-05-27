
/*
   Implements the sequential pthread based vectors.
*/

#include <petscconf.h>
#include <private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petscblaslapack.h>
#include <pthread.h>
void           MainWait(void);
void           MainJob(void* (*pFunc)(void*),void**,pthread_barrier_t*,PetscInt);
extern PetscBool    PetscUseThreadPool;
extern pthread_barrier_t* BarrPoint;
#define giNUM_THREADS 2
pthread_t apThread[giNUM_THREADS];

typedef struct {
  const PetscScalar *x,*y;
  PetscInt          n;
  PetscScalar       result;
} VecDot_KernelData;
VecDot_KernelData kerneldatap[giNUM_THREADS];

static inline void* PetscThreadRun(MPI_Comm Comm,void* funcp,int iNumThreads,void** data) {
  PetscInt    ierr;
  int i;
  for(i=0; i<iNumThreads; i++) {
    ierr = pthread_create(&apThread[i],NULL,funcp,data[i]);
    //CHKERRQ((PetscErrorCode)ierr);
  }
  return NULL;
}

static inline void* PetscThreadStop(MPI_Comm Comm,int iNumThreads) {
  int i;
  void* joinstatus;
  for (i=0; i<iNumThreads; i++) {
    pthread_join(apThread[i], &joinstatus);
  }
  return NULL;
}

void *VecDot_Kernel(void *arg)
{
  VecDot_KernelData *data = (VecDot_KernelData*)arg;
  const PetscScalar *x, *y;
  PetscBLASInt one = 1, bn;
  PetscInt    n;
  printf("You Are in Thread Kernel With Data Address %p!\n",arg);
  x = data->x;
  y = data->y;
  n = data->n;
  bn = PetscBLASIntCast(n);
  data->result = BLASdot_(&bn,x,&one,y,&one);
  printf("Data Result = %f\n",data->result);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqPThread"
PetscErrorCode VecDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscErrorCode    ierr;
  PetscInt          i, iIndex = 0;
  const PetscInt    iNumThreads = giNUM_THREADS;
  PetscInt          Q = xin->map->n/(iNumThreads);
  PetscInt          R = xin->map->n-Q*(iNumThreads);
  PetscBool         S;
  VecDot_KernelData* pdata[giNUM_THREADS];

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<iNumThreads; i++) {
    S = i<R;
    kerneldatap[i].x = &xa[iIndex];
    kerneldatap[i].y = &ya[iIndex];
    kerneldatap[i].n = S?Q+1:Q;
    iIndex += kerneldatap[i].n;
    pdata[i] = &kerneldatap[i];
    printf("Data Address %d = %p\n",i,pdata[i]);
  }

  if(PetscUseThreadPool) {
    printf("Main Got INTO Wait Function\n");
    MainWait();
    printf("Main Got OUT of Wait Function\n");
    MainJob(VecDot_Kernel,(void**)pdata,&BarrPoint[2],2);
    printf("Main Processed Job!\n");
  }
  else {
    PetscThreadRun(MPI_COMM_WORLD,VecDot_Kernel,giNUM_THREADS,(void**)pdata);
    PetscThreadStop(MPI_COMM_WORLD,giNUM_THREADS); //ensures that all threads are finished with the job
  }
  //do i need to have main stall until i know the results are in?  YES!
  MainWait();
  //gather result
  *z = 0.0;
  for(i=0; i<giNUM_THREADS; i++) {
    *z += kerneldatap[i].result;
  }
  printf("Results Have Been Gathered With Value %f\n",*z);
  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);

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
