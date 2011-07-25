
/*
   Implements the sequential pthread based vectors.
*/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <petscconf.h>
#include <private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petscblaslapack.h>
#include <private/petscaxpy.h>
#include <pthread.h>
#include <unistd.h>

extern void           (*MainWait)(void);
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt);
extern PetscBool    PetscUseThreadPool;
extern PetscMPIInt PetscMaxThreads;
extern pthread_t*   PetscThreadPoint;
extern int* ThreadCoreAffinity;
void* PetscThreadRun(MPI_Comm Comm,void* (*pFunc)(void*),int,pthread_t*,void**);
void* PetscThreadStop(MPI_Comm Comm,int,pthread_t*);
void* DoCoreAffinity(void);

typedef struct {
  const PetscScalar *x,*y;
  PetscInt          n;
  PetscScalar       result;
} VecDot_KernelData;

typedef struct {
  PetscScalar *x;
  PetscScalar alpha;
  PetscInt    n;
} VecScale_KernelData;

typedef struct {
  PetscScalar *y;
  const PetscScalar *x;
  PetscScalar alpha;
  PetscInt    n;
} VecAXPY_KernelData;

typedef struct {
  PetscScalar *yy;
  const PetscScalar *xx;
  PetscScalar alpha;
  PetscInt    n;
} VecAYPX_KernelData;

typedef struct {
  PetscScalar *ww;
  const PetscScalar *yy;
  const PetscScalar *xx;
  PetscScalar alpha;
  PetscInt    n;
} VecWAXPY_KernelData;

typedef struct {
  const PetscScalar *x;
  NormType typeUse;
  PetscInt    n;
  PetscScalar result;
} VecNorm_KernelData;

typedef struct {
  const PetscScalar* xvalin;
  Vec*               yavecin;
  PetscInt           nelem;
  PetscInt           ntoproc;
  PetscScalar*       result;
} VecMDot_KernelData;

typedef struct {
  const PetscScalar *x;
  PetscInt          gind;
  PetscInt          localn;
  PetscInt          localind;
  PetscReal         localmax;
} VecMax_KernelData;

typedef struct {
  const PetscScalar *x;
  PetscInt          gind;
  PetscInt          localn;
  PetscInt          localind;
  PetscReal         localmin;
} VecMin_KernelData;

typedef struct {
  PetscScalar *wpin,*xpin,*ypin;
  PetscInt          nlocal;
} VecPointwiseMult_KernelData;

typedef struct {
  PetscScalar *wpin,*xpin,*ypin;
  PetscInt          nlocal;
} VecPointwiseDivide_KernelData;

typedef struct {
  PetscScalar *xpin,*ypin;
  PetscInt nlocal;
} VecSwap_KernelData;

typedef struct {
  PetscScalar *xpin;
  PetscRandom   rin;
  PetscInt nlocal;
} VecSetRandom_KernelData;

typedef struct {
  const PetscScalar *xpin;
  PetscScalar   *ypin;
  PetscInt nlocal;
} VecCopy_KernelData;

typedef struct {
  PetscScalar*       xavalin; //vector out
  Vec*               yavecin; //array of data vectors
  const PetscScalar* amult;   //multipliers
  PetscInt           nelem;   //number of elements in vector to process
  PetscInt           ntoproc; //number of data vectors
  PetscInt           ibase;   //used to properly index into other vectors
} VecMAXPY_KernelData;

typedef struct {
  PetscScalar *xpin;
  PetscScalar alphain;
  PetscInt nelem;
} VecSet_KernelData;

void* PetscThreadRun(MPI_Comm Comm,void* (*funcp)(void*),int iTotThreads,pthread_t* ThreadId,void** data) {
  PetscInt    ierr;
  int i;
  for(i=0; i<iTotThreads; i++) {
    ierr = pthread_create(&ThreadId[i],NULL,funcp,data[i]);
  }
  return NULL;
}

void* PetscThreadStop(MPI_Comm Comm,int iTotThreads,pthread_t* ThreadId) {
  int i;
  void* joinstatus;
  for (i=0; i<iTotThreads; i++) {
    pthread_join(ThreadId[i], &joinstatus);
  }
  return NULL;
}

/* Change these macros so can be used in thread kernels */
#undef CHKERRQP
#define CHKERRQP(ierr) if (ierr) return (void*)(long int)ierr

void* VecDot_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecDot_KernelData *data = (VecDot_KernelData*)arg;
  const PetscScalar *x, *y;
  PetscBLASInt one = 1, bn;
  PetscInt    n;

  x = data->x;
  y = data->y;
  n = data->n;
  bn = PetscBLASIntCast(n);
  data->result = BLASdot_(&bn,x,&one,y,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqPThread"
PetscErrorCode VecDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscErrorCode    ierr;
  PetscInt          i, iIndex = 0;
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = xin->map->n/(iNumThreads);
  PetscInt          R = xin->map->n-Q*(iNumThreads);
  PetscBool         S;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);

  VecDot_KernelData* kerneldatap = (VecDot_KernelData*)malloc(iNumThreads*sizeof(VecDot_KernelData));
  VecDot_KernelData** pdata = (VecDot_KernelData**)malloc(iNumThreads*sizeof(VecDot_KernelData*));

  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].x = &xa[iIndex];
    kerneldatap[i].y = &ya[iIndex];
    kerneldatap[i].n = S?Q+1:Q;
    iIndex += kerneldatap[i].n;
    pdata[i] = &kerneldatap[i];
  }

  ierr = MainJob(VecDot_Kernel,(void**)pdata,iNumThreads);

  //gather result
  *z = 0.0;
  for(i=0; i<iNumThreads; i++) {
    *z += kerneldatap[i].result;
  }
  free(kerneldatap);
  free(pdata);

  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);

  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}


void* VecScale_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecScale_KernelData *data = (VecScale_KernelData*)arg;
  PetscScalar a,*x;
  PetscBLASInt one = 1, bn;
  PetscInt    n;

  x = data->x;
  a = data->alpha;
  n = data->n;
  bn = PetscBLASIntCast(n);
  BLASscal_(&bn,&a,x,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecScale_SeqPThread"
PetscErrorCode VecScale_SeqPThread(Vec xin, PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (alpha == 0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    PetscScalar a = alpha,*xarray,*xp;
    const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
    PetscInt          i,Q = xin->map->n/(iNumThreads);
    PetscInt          R = xin->map->n-Q*(iNumThreads);
    PetscBool         S;
    VecScale_KernelData* kerneldatap = (VecScale_KernelData*)malloc(iNumThreads*sizeof(VecScale_KernelData));
    VecScale_KernelData** pdata = (VecScale_KernelData**)malloc(iNumThreads*sizeof(VecScale_KernelData*));
    ierr = VecGetArray(xin,&xarray);CHKERRQ(ierr); //get address of first element in data array
    xp = xarray;
    for (i=0; i<iNumThreads; i++) {
      S = (PetscBool)(i<R);
      kerneldatap[i].x = xp;
      kerneldatap[i].alpha = a;
      kerneldatap[i].n = S?Q+1:Q;
      xp += kerneldatap[i].n; //pointer arithmetic
      pdata[i] = &kerneldatap[i];
    }
    ierr = MainJob(VecScale_Kernel,(void**)pdata,iNumThreads);
    free(kerneldatap);
    free(pdata);
    ierr = VecRestoreArray(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

void* VecAXPY_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecAXPY_KernelData *data = (VecAXPY_KernelData*)arg;
  PetscScalar a,*y;
  const PetscScalar *x;
  PetscBLASInt one = 1, bn;
  PetscInt    n;

  x = data->x;
  y = data->y;
  a = data->alpha;
  n = data->n;
  bn = PetscBLASIntCast(n);
  BLASaxpy_(&bn,&a,x,&one,y,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAXPY_SeqPThread"
PetscErrorCode VecAXPY_SeqPThread(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarray,*xp;
  PetscScalar       a=alpha,*yarray,*yp;
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          i,Q = xin->map->n/(iNumThreads);
  PetscInt          R = xin->map->n-Q*(iNumThreads);
  PetscBool         S;
  VecAXPY_KernelData* kerneldatap = (VecAXPY_KernelData*)malloc(iNumThreads*sizeof(VecAXPY_KernelData));
  VecAXPY_KernelData** pdata = (VecAXPY_KernelData**)malloc(iNumThreads*sizeof(VecAXPY_KernelData*));

  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != 0.0) {
    ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yarray);CHKERRQ(ierr);
    xp = xarray;
    yp = yarray;
    for (i=0; i<iNumThreads; i++) {
      S = (PetscBool)(i<R);
      kerneldatap[i].x = xp;
      kerneldatap[i].y = yp;
      kerneldatap[i].alpha = a;
      kerneldatap[i].n = S?Q+1:Q;
      xp += kerneldatap[i].n; //pointer arithmetic
      yp += kerneldatap[i].n;
      pdata[i] = &kerneldatap[i];
    }
    ierr = MainJob(VecAXPY_Kernel,(void**)pdata,iNumThreads);
    ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  free(kerneldatap);
  free(pdata);
  PetscFunctionReturn(0);
}

void* VecAYPX_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecAYPX_KernelData *data = (VecAYPX_KernelData*)arg;
  PetscScalar a,*yy;
  const PetscScalar *xx;
  PetscInt    i,n;

  xx = data->xx;
  yy = data->yy;
  a = data->alpha;
  n = data->n;
  if(a==-1.0) {
    for (i=0; i<n; i++) {
      yy[i] = xx[i] - yy[i];
    }
  }
  else {
    for (i=0; i<n; i++) {
      yy[i] = xx[i] + a*yy[i];
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAYPX_SeqPThread"
PetscErrorCode VecAYPX_SeqPThread(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  if (alpha == 0.0) {
    ierr = VecCopy(xin,yin);CHKERRQ(ierr);
  }
  else if (alpha == 1.0) {
    ierr = VecAXPY_SeqPThread(yin,alpha,xin);CHKERRQ(ierr);
  }
  else {
    PetscInt          n = yin->map->n;
    PetscScalar       *yy;
    const PetscScalar *xx;
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
    #if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
    {
      PetscScalar oalpha = alpha;
      fortranaypx_(&n,&oalpha,xx,yy);
    }
    #else
    {
      const PetscScalar *xp = xx;
      PetscScalar       a=alpha,*yp = yy;
      const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
      PetscInt          i,Q = xin->map->n/(iNumThreads);
      PetscInt          R = xin->map->n-Q*(iNumThreads);
      PetscBool         S;
      VecAYPX_KernelData* kerneldatap = (VecAYPX_KernelData*)malloc(iNumThreads*sizeof(VecAYPX_KernelData));
      VecAYPX_KernelData** pdata = (VecAYPX_KernelData**)malloc(iNumThreads*sizeof(VecAYPX_KernelData*));

      for (i=0; i<iNumThreads; i++) {
        S = (PetscBool)(i<R);
        kerneldatap[i].xx = xp;
        kerneldatap[i].yy = yp;
        kerneldatap[i].alpha = a;
        kerneldatap[i].n = S?Q+1:Q;
        xp += kerneldatap[i].n; //pointer arithmetic
        yp += kerneldatap[i].n;
        pdata[i] = &kerneldatap[i];
      }
      ierr = MainJob(VecAYPX_Kernel,(void**)pdata,iNumThreads);
      free(kerneldatap);
      free(pdata);
    }
    #endif
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    if(alpha==-1.0) {
      ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
    }
    else {
      ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

void* VecWAXPY_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecWAXPY_KernelData *data = (VecWAXPY_KernelData*)arg;
  PetscScalar a,*ww;
  const PetscScalar *xx,*yy;
  PetscInt    i,n;

  ww = data->ww;
  xx = data->xx;
  yy = data->yy;
  a = data->alpha;
  n = data->n;
  if(a==-1.0) {
    for (i=0; i<n; i++) {
      ww[i] = yy[i] - xx[i];
    }
  }
  else if(a==1.0) {
    for (i=0; i<n; i++) {
      ww[i] = yy[i] + xx[i];
    }
  }
  else {
    for (i=0; i<n; i++) {
      ww[i] = a*xx[i] + yy[i];
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_SeqPThread"
PetscErrorCode VecWAXPY_SeqPThread(Vec win, PetscScalar alpha,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           n = win->map->n;
  PetscScalar        *ww;
  const PetscScalar  *yy,*xx;

  PetscFunctionBegin;

  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  if (alpha == 0.0) {
    ierr = PetscMemcpy(ww,yy,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  else {
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    PetscScalar oalpha = alpha;
    fortranwaxpy_(&n,&oalpha,xx,yy,ww);
#else
    const PetscScalar *xp = xx,*yp = yy;
    PetscScalar       a=alpha,*wp = ww;
    const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
    PetscInt          i,Q = n/(iNumThreads);
    PetscInt          R = n-Q*(iNumThreads);
    PetscBool         S;
    VecWAXPY_KernelData* kerneldatap = (VecWAXPY_KernelData*)malloc(iNumThreads*sizeof(VecWAXPY_KernelData));
    VecWAXPY_KernelData** pdata = (VecWAXPY_KernelData**)malloc(iNumThreads*sizeof(VecWAXPY_KernelData*));

    for (i=0; i<iNumThreads; i++) {
      S = (PetscBool)(i<R);
      kerneldatap[i].ww = wp;
      kerneldatap[i].xx = xp;
      kerneldatap[i].yy = yp;
      kerneldatap[i].alpha = a;
      kerneldatap[i].n = S?Q+1:Q;
      wp += kerneldatap[i].n; //pointer arithmetic
      xp += kerneldatap[i].n;
      yp += kerneldatap[i].n;
      pdata[i] = &kerneldatap[i];
    }
    ierr = MainJob(VecWAXPY_Kernel,(void**)pdata,iNumThreads);
    free(kerneldatap);
    free(pdata);
#endif
    if (alpha == 1.0 || alpha == -1.0) {
      ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
    }
    else {
      ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecNorm_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecNorm_KernelData *data = (VecNorm_KernelData*)arg;
  const PetscScalar *x;
  NormType type;
  PetscInt    i,n;

  x = data->x;
  type = data->typeUse;
  n = data->n;
  data->result = 0.0;
  if(type==NORM_1) {
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
    data->result = BLASasum_(&bn,x,&one);
  }
  else if(type==NORM_INFINITY) {
    PetscReal    maxv = 0.0,tmp;
    for(i=0; i<n; i++) {
      tmp = PetscAbsScalar(x[i]);
      if(tmp>maxv) {
        maxv = tmp;
      }
    }
    data->result = maxv;
  }
  else {
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
    data->result = BLASnrm2_(&bn,x,&one);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNorm_SeqPThread"
PetscErrorCode VecNorm_SeqPThread(Vec xin,NormType type,PetscReal* z)
{
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;

  PetscFunctionBegin;
  if(type == NORM_1_AND_2) {
    ierr = VecNorm_SeqPThread(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqPThread(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  else {
   ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
   const PetscScalar *xp = xx;
   const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
   PetscInt          i,Q = n/(iNumThreads);
   PetscInt          R = n-Q*(iNumThreads);
   PetscBool         S;
   VecNorm_KernelData* kerneldatap = (VecNorm_KernelData*)malloc(iNumThreads*sizeof(VecNorm_KernelData));
   VecNorm_KernelData** pdata = (VecNorm_KernelData**)malloc(iNumThreads*sizeof(VecNorm_KernelData*));

   for (i=0; i<iNumThreads; i++) {
     S = (PetscBool)(i<R);
     kerneldatap[i].x = xp;
     kerneldatap[i].typeUse = type;
     kerneldatap[i].n = S?Q+1:Q;
     xp += kerneldatap[i].n; //pointer arithmetic
     pdata[i] = &kerneldatap[i];
   }
   ierr = MainJob(VecNorm_Kernel,(void**)pdata,iNumThreads);
   //collect results
   *z = 0.0;
   if(type == NORM_1) {
     for(i=0; i<iNumThreads; i++) {
       *z += kerneldatap[i].result;
     }
     ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
   }
   else if(type == NORM_2 || type == NORM_FROBENIUS) {
     for(i=0; i<iNumThreads; i++) {
       *z += kerneldatap[i].result*kerneldatap[i].result;
     }
     *z = sqrt(*z);
     ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
   }
   else {
     PetscReal    maxv = 0.0,tmp;
     for(i=0; i<iNumThreads; i++) {
       tmp = kerneldatap[i].result;
       if(tmp>maxv) {
         maxv = tmp;
       }
     }
     *z = maxv;
   }
   free(kerneldatap);
   free(pdata);
   ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecMDot_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecMDot_KernelData *data = (VecMDot_KernelData*)arg;
  const PetscScalar  *xbase = data->xvalin;
  Vec*               yin = data->yavecin;
  PetscInt           n = data->nelem;
  PetscInt           nv = data->ntoproc;
  PetscScalar*       z = data->result;
  PetscErrorCode     ierr;
  PetscInt           i,j,nv_rem,j_rem;
  PetscScalar        sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar  *yy0,*yy1,*yy2,*yy3,*x;
  Vec                *yy;

  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;
  i      = nv;
  nv_rem = nv&0x3;
  yy     = yin;
  j      = n;
  x      = xbase;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2 = x[2]; 
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]); 
      sum2 += x2*PetscConj(yy2[2]); 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]); 
      sum2 += x1*PetscConj(yy2[1]); 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]); 
      sum2 += x0*PetscConj(yy2[0]); 
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]); 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]); 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]); 
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
 
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; sum0 += x2*PetscConj(yy0[2]);
    case 2: 
      x1 = x[1]; sum0 += x1*PetscConj(yy0[1]);
    case 1: 
      x0 = x[0]; sum0 += x0*PetscConj(yy0[0]);
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*PetscConj(yy0[0]) + x[1]*PetscConj(yy0[1])
            + x[2]*PetscConj(yy0[2]) + x[3]*PetscConj(yy0[3]); 
      yy0+=4;
      j -= 4; x+=4;
    }
    z[0] = sum0;

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQP(ierr);

    j = n;
    x = xbase;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]); 
      sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
    case 2: 
      x1 = x[1]; 
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]); 
      sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
    case 1: 
      x0 = x[0]; 
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]); 
      sum2 += x0*PetscConj(yy2[0]); sum3 += x0*PetscConj(yy3[0]);
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      sum3 += x0*PetscConj(yy3[0]) + x1*PetscConj(yy3[1]) + x2*PetscConj(yy3[2]) + x3*PetscConj(yy3[3]); yy3+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQP(ierr);
    yy  += 4;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqPThread"
PetscErrorCode VecMDot_SeqPThread(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          i,j=0;
  Vec               *yy = (Vec *)yin;
  const PetscScalar *xbase;
  PetscFunctionBegin;

  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          n=xin->map->n,Q = nv/(iNumThreads);
  PetscInt          R = nv-Q*(iNumThreads);
  PetscBool         S;
  VecMDot_KernelData* kerneldatap = (VecMDot_KernelData*)malloc(iNumThreads*sizeof(VecMDot_KernelData));
  VecMDot_KernelData** pdata = (VecMDot_KernelData**)malloc(iNumThreads*sizeof(VecMDot_KernelData*));
  ierr   = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].xvalin = xbase;
    kerneldatap[i].yavecin = &yy[j];
    kerneldatap[i].nelem = n;
    kerneldatap[i].ntoproc = S?Q+1:Q;
    kerneldatap[i].result = &z[j];
    j += kerneldatap[i].ntoproc;
    pdata[i] = &kerneldatap[i];
  }
  ierr = MainJob(VecMDot_Kernel,(void**)pdata,iNumThreads);
  free(kerneldatap);
  free(pdata);
  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecMax_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecMax_KernelData *data = (VecMax_KernelData*)arg;
  const PetscScalar *xx = data->x;
  PetscInt          i,j,n = data->localn;
  PetscReal         lmax,tmp;

#if defined(PETSC_USE_COMPLEX)
  lmax = PetscRealPart(*xx++); j = 0;
#else
  lmax = *xx++; j = 0;
#endif
  for (i=1; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
    if ((tmp = PetscRealPart(*xx++)) > lmax) { j = i; lmax = tmp;}
#else
    if ((tmp = *xx++) > lmax) { j = i; lmax = tmp; }
#endif
  }

  data->localmax = lmax;
  data->localind = j;
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMax_SeqPThread"
PetscErrorCode VecMax_SeqPThread(Vec xin,PetscInt* idx,PetscReal * z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         max;
  const PetscScalar *xx,*xp;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  if (!n) {
    max = PETSC_MIN_REAL;
    j   = -1;
  } else {
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          gind,Q = n/(iNumThreads);
  PetscInt          R = n-Q*(iNumThreads);
  PetscBool         S;
  VecMax_KernelData* kerneldatap = (VecMax_KernelData*)malloc(iNumThreads*sizeof(VecMax_KernelData));
  VecMax_KernelData** pdata = (VecMax_KernelData**)malloc(iNumThreads*sizeof(VecMax_KernelData*));

  gind = 0;
  xp = xx;
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].x = xp;
    kerneldatap[i].gind = gind;
    kerneldatap[i].localn = S?Q+1:Q;
    xp += kerneldatap[i].localn; //pointer arithmetic
    gind += kerneldatap[i].localn;
    pdata[i] = &kerneldatap[i];
  }
  ierr = MainJob(VecMax_Kernel,(void**)pdata,iNumThreads);
  //collect results, determine global max, global index
  max = kerneldatap[0].localmax;
  j   = kerneldatap[0].localind;
  for(i=1; i<iNumThreads; i++) {
    if(kerneldatap[i].localmax>max) {
      max = kerneldatap[i].localmax;
      j   = kerneldatap[i].gind+kerneldatap[i].localind;
    }
  }
  free(kerneldatap);
  free(pdata);
  }
  *z   = max;
  if (idx) *idx = j;
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecMin_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecMin_KernelData *data = (VecMin_KernelData*)arg;
  const PetscScalar *xx = data->x;
  PetscInt          i,j,n = data->localn;
  PetscReal         lmin,tmp;

#if defined(PETSC_USE_COMPLEX)
  lmin = PetscRealPart(*xx++); j = 0;
#else
  lmin = *xx++; j = 0;
#endif
  for (i=1; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
    if ((tmp = PetscRealPart(*xx++)) < lmin) { j = i; lmin = tmp;}
#else
    if ((tmp = *xx++) < lmin) { j = i; lmin = tmp; }
#endif
  }

  data->localmin = lmin;
  data->localind = j;
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMin_SeqPThread"
PetscErrorCode VecMin_SeqPThread(Vec xin,PetscInt* idx,PetscReal * z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         min;
  const PetscScalar *xx,*xp;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  if (!n) {
    min = PETSC_MAX_REAL;
    j   = -1;
  } else {
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          gind,Q = n/(iNumThreads);
  PetscInt          R = n-Q*(iNumThreads);
  PetscBool         S;
  VecMin_KernelData* kerneldatap = (VecMin_KernelData*)malloc(iNumThreads*sizeof(VecMin_KernelData));
  VecMin_KernelData** pdata = (VecMin_KernelData**)malloc(iNumThreads*sizeof(VecMin_KernelData*));

  gind = 0;
  xp = xx;
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].x = xp;
    kerneldatap[i].gind = gind;
    kerneldatap[i].localn = S?Q+1:Q;
    xp += kerneldatap[i].localn; //pointer arithmetic
    gind += kerneldatap[i].localn;
    pdata[i] = &kerneldatap[i];
  }
  ierr = MainJob(VecMin_Kernel,(void**)pdata,iNumThreads);
  //collect results, determine global max, global index
  min = kerneldatap[0].localmin;
  j   = kerneldatap[0].localind;
  for(i=1; i<iNumThreads; i++) {
    if(kerneldatap[i].localmin<min) {
      min = kerneldatap[i].localmin;
      j   = kerneldatap[i].gind+kerneldatap[i].localind;
    }
  }
  free(kerneldatap);
  free(pdata);
  }
  *z   = min;
  if (idx) *idx = j;
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecPointwiseMult_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecPointwiseMult_KernelData *data = (VecPointwiseMult_KernelData*)arg;
  PetscScalar *ww = data->wpin,*xx = data->xpin,*yy = data->ypin;
  PetscInt    n = data->nlocal,i;

  if (ww == xx) {
    for (i=0; i<n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=0; i<n; i++) ww[i] *= xx[i];
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  return(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>
#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_SeqPThread"
static PetscErrorCode VecPointwiseMult_SeqPThread(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i,iIndex;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = n/(iNumThreads);
  PetscInt          R = n-Q*(iNumThreads);
  PetscBool         S;

  VecPointwiseMult_KernelData* kerneldatap = (VecPointwiseMult_KernelData*)malloc(iNumThreads*sizeof(VecPointwiseMult_KernelData));
  VecPointwiseMult_KernelData** pdata = (VecPointwiseMult_KernelData**)malloc(iNumThreads*sizeof(VecPointwiseMult_KernelData*));


  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);

  iIndex = 0;
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].wpin = ww+iIndex;
    kerneldatap[i].xpin = xx+iIndex;
    kerneldatap[i].ypin = yy+iIndex;
    kerneldatap[i].nlocal = S?Q+1:Q;
    iIndex += kerneldatap[i].nlocal;
    pdata[i] = &kerneldatap[i];
  }

  ierr  = MainJob(VecPointwiseMult_Kernel,(void**)pdata,iNumThreads);
  free(kerneldatap);
  free(pdata);

  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecPointwiseDivide_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecPointwiseDivide_KernelData *data = (VecPointwiseDivide_KernelData*)arg;
  PetscScalar *ww = data->wpin,*xx = data->xpin,*yy = data->ypin;
  PetscInt    n = data->nlocal,i;

  for (i=0; i<n; i++) {
    ww[i] = xx[i] / yy[i];
  }
  return(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_SeqPThread"
static PetscErrorCode VecPointwiseDivide_SeqPThread(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i,iIndex;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = n/(iNumThreads);
  PetscInt          R = n-Q*(iNumThreads);
  PetscBool         S;

  VecPointwiseDivide_KernelData* kerneldatap = (VecPointwiseDivide_KernelData*)malloc(iNumThreads*sizeof(VecPointwiseDivide_KernelData));
  VecPointwiseDivide_KernelData** pdata = (VecPointwiseDivide_KernelData**)malloc(iNumThreads*sizeof(VecPointwiseDivide_KernelData*));


  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);

  iIndex = 0;
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].wpin = ww+iIndex;
    kerneldatap[i].xpin = xx+iIndex;
    kerneldatap[i].ypin = yy+iIndex;
    kerneldatap[i].nlocal = S?Q+1:Q;
    iIndex += kerneldatap[i].nlocal;
    pdata[i] = &kerneldatap[i];
  }

  ierr = MainJob(VecPointwiseDivide_Kernel,(void**)pdata,iNumThreads);
  free(kerneldatap);
  free(pdata);

  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecSwap_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecSwap_KernelData *data = (VecSwap_KernelData*)arg;
  PetscScalar *xa = data->xpin,*ya = data->ypin;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(data->nlocal);

  BLASswap_(&bn,xa,&one,ya,&one);
  return(0);
}

#include <petscblaslapack.h>
#undef __FUNCT__
#define __FUNCT__ "VecSwap_SeqPThread"
static PetscErrorCode VecSwap_SeqPThread(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
    PetscInt          n = xin->map->n,Q = n/(iNumThreads),R = n-Q*(iNumThreads),i,iIndex;
    PetscBool         S;

    VecSwap_KernelData* kerneldatap = (VecSwap_KernelData*)malloc(iNumThreads*sizeof(VecSwap_KernelData));
    VecSwap_KernelData** pdata = (VecSwap_KernelData**)malloc(iNumThreads*sizeof(VecSwap_KernelData*));

    iIndex = 0;
    for (i=0; i<iNumThreads; i++) {
      S = (PetscBool)(i<R);
      kerneldatap[i].xpin = xa+iIndex;
      kerneldatap[i].ypin = ya+iIndex;
      kerneldatap[i].nlocal = S?Q+1:Q;
      iIndex += kerneldatap[i].nlocal;
      pdata[i] = &kerneldatap[i];
    }

    ierr = MainJob(VecSwap_Kernel,(void**)pdata,iNumThreads);
    free(kerneldatap);
    free(pdata);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecSetRandom_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecSetRandom_KernelData *data = (VecSetRandom_KernelData*)arg;
  PetscScalar  *xx = data->xpin;
  PetscRandom  r = data->rin;
  PetscInt     i,n = data->nlocal;
  PetscErrorCode ierr;

  for(i=0; i<n; i++) {
    ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQP(ierr);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_SeqPThread"
static PetscErrorCode VecSetRandom_SeqPThread(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = n/(iNumThreads),R = n-Q*(iNumThreads),iIndex;
  PetscBool         S;

  VecSetRandom_KernelData* kerneldatap = (VecSetRandom_KernelData*)malloc(iNumThreads*sizeof(VecSetRandom_KernelData));
  VecSetRandom_KernelData** pdata = (VecSetRandom_KernelData**)malloc(iNumThreads*sizeof(VecSetRandom_KernelData*));

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);

  iIndex = 0;
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].xpin   = xx+iIndex;
    kerneldatap[i].rin    = r;
    kerneldatap[i].nlocal = S?Q+1:Q;
    iIndex += kerneldatap[i].nlocal;
    pdata[i] = &kerneldatap[i];
   }

  ierr = MainJob(VecSetRandom_Kernel,(void**)pdata,iNumThreads);
  free(kerneldatap);
  free(pdata);
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecCopy_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecCopy_KernelData *data = (VecCopy_KernelData*)arg;
  const PetscScalar  *xa = data->xpin;
  PetscScalar        *ya = data->ypin;
  PetscInt           n = data->nlocal;
  PetscErrorCode ierr;

  ierr = PetscMemcpy(ya,xa,n*sizeof(PetscScalar));CHKERRQP(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqPThread"
static PetscErrorCode VecCopy_SeqPThread(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

  PetscInt       n = xin->map->n,i;
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = n/(iNumThreads),R = n-Q*(iNumThreads),iIndex;
  PetscBool         S;

  VecCopy_KernelData* kerneldatap = (VecCopy_KernelData*)malloc(iNumThreads*sizeof(VecCopy_KernelData));
  VecCopy_KernelData** pdata = (VecCopy_KernelData**)malloc(iNumThreads*sizeof(VecCopy_KernelData*));

    iIndex = 0;
    for (i=0; i<iNumThreads; i++) {
      S = (PetscBool)(i<R);
      kerneldatap[i].xpin   = xa+iIndex;
      kerneldatap[i].ypin   = ya+iIndex;
      kerneldatap[i].nlocal = S?Q+1:Q;
      iIndex += kerneldatap[i].nlocal;
      pdata[i] = &kerneldatap[i];
    }
    ierr = MainJob(VecCopy_Kernel,(void**)pdata,iNumThreads);
    free(kerneldatap);
    free(pdata);

    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecMAXPY_Kernel(void* arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecMAXPY_KernelData *data = (VecMAXPY_KernelData*)arg;
  PetscErrorCode    ierr;
  PetscInt          n = data->nelem,nv=data->ntoproc,ibase=data->ibase,j,j_rem;
  const PetscScalar *alpha=data->amult,*yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx = data->xavalin,alpha0,alpha1,alpha2,alpha3;
  Vec* y = data->yavecin;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  switch (j_rem=nv&0x3) {
  case 3: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQP(ierr);
    yy0 += ibase; yy1 += ibase; yy2 += ibase; //pointer arithmetic
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha2 = alpha[2]; 
    alpha += 3;
    PetscAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQP(ierr);
    y     += 3;
    break;
  case 2: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQP(ierr);
    yy0 += ibase; yy1 += ibase;
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha +=2;
    PetscAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQP(ierr);
    y     +=2;
    break;
  case 1: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQP(ierr);
    yy0 += ibase;
    alpha0 = *alpha++; 
    PetscAXPY(xx,alpha0,yy0,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQP(ierr);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[3],&yy3);CHKERRQP(ierr);
    yy0 += ibase; yy1 += ibase; yy2 += ibase; yy3 += ibase;
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;

    PetscAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQP(ierr);
    y      += 4;
  }
  return(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_SeqPThread"
PetscErrorCode VecMAXPY_SeqPThread(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i,j=0;
  PetscScalar       *xx;

  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = n/(iNumThreads),R = n-Q*(iNumThreads);
  //PetscInt          K = nv / 4; /*how many groups of 4 are present */
  //PetscInt          Q = K / iNumThreads; /* how many groups of 4 to give to each thread */
  //PetscInt          R = nv - Q*iNumThreads*4;
  PetscBool         S;
  VecMAXPY_KernelData* kerneldatap = (VecMAXPY_KernelData*)malloc(iNumThreads*sizeof(VecMAXPY_KernelData));
  VecMAXPY_KernelData** pdata = (VecMAXPY_KernelData**)malloc(iNumThreads*sizeof(VecMAXPY_KernelData*));

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for(i=0;i<nv;i++) {
    if(y[i]->petscnative!=PETSC_TRUE) {
      printf("Non PETSC Native Vector!\n");
    }
  }
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].xavalin = xx+j;
    kerneldatap[i].yavecin = &y[0];
    kerneldatap[i].amult   = &alpha[0];
    kerneldatap[i].nelem   = S?Q+1:Q;
    kerneldatap[i].ntoproc = nv;
    kerneldatap[i].ibase   = j;
    j += kerneldatap[i].nelem;
    pdata[i] = &kerneldatap[i];
  }
  ierr = MainJob(VecMAXPY_Kernel,(void**)pdata,iNumThreads);
  free(kerneldatap);
  free(pdata);

  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecSet_Kernel(void *arg)
{
  if(PetscUseThreadPool==PETSC_FALSE) {
    DoCoreAffinity();
  }
  VecSet_KernelData *data = (VecSet_KernelData*)arg;
  PetscScalar        *xx = data->xpin;
  PetscScalar        alpha = data->alphain;
  PetscInt           i,n = data->nelem;
  PetscErrorCode ierr;

  if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemzero(xx,n*sizeof(PetscScalar));CHKERRQP(ierr);
  } else {
    for (i=0; i<n; i++) xx[i] = alpha;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSet_SeqPThread"
PetscErrorCode VecSet_SeqPThread(Vec xin,PetscScalar alpha)
{
  PetscInt       i,n = xin->map->n;
  PetscScalar    *xx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  const PetscInt    iNumThreads = PetscMaxThreads;  //this number could be different
  PetscInt          Q = n/(iNumThreads),R = n-Q*(iNumThreads),iIndex;
  PetscBool         S;

  VecSet_KernelData* kerneldatap = (VecSet_KernelData*)malloc(iNumThreads*sizeof(VecSet_KernelData));
  VecSet_KernelData** pdata = (VecSet_KernelData**)malloc(iNumThreads*sizeof(VecSet_KernelData*));

  iIndex = 0;
  for (i=0; i<iNumThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].xpin   = xx+iIndex;
    kerneldatap[i].alphain   = alpha;
    kerneldatap[i].nelem = S?Q+1:Q;
    iIndex += kerneldatap[i].nelem;
    pdata[i] = &kerneldatap[i];
  }
  ierr = MainJob(VecSet_Kernel,(void**)pdata,iNumThreads);
  free(kerneldatap);
  free(pdata);
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
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

void* DoCoreAffinity(void)
{
  int i,icorr=0; cpu_set_t mset;
  pthread_t pThread = pthread_self();
  for(i=0; i<PetscMaxThreads; i++) {
    if(pthread_equal(pThread,PetscThreadPoint[i])) {
      icorr = ThreadCoreAffinity[i];
    }
  }
  CPU_ZERO(&mset);
  CPU_SET(icorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);
  return(0);
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
  V->ops->mdot            = VecMDot_SeqPThread;
  V->ops->scale           = VecScale_SeqPThread;
  V->ops->axpy            = VecAXPY_SeqPThread;
  V->ops->aypx            = VecAYPX_SeqPThread;
  V->ops->waxpy           = VecWAXPY_SeqPThread;
  V->ops->norm            = VecNorm_SeqPThread;
  V->ops->max             = VecMax_SeqPThread;
  V->ops->min             = VecMin_SeqPThread;
  V->ops->pointwisemult   = VecPointwiseMult_SeqPThread;
  V->ops->pointwisedivide = VecPointwiseDivide_SeqPThread;
  V->ops->swap            = VecSwap_SeqPThread;
  V->ops->setrandom       = VecSetRandom_SeqPThread;
  V->ops->copy            = VecCopy_SeqPThread;
  V->ops->maxpy           = VecMAXPY_SeqPThread;
  V->ops->set             = VecSet_SeqPThread;
  VecSet(V,0);
  PetscFunctionReturn(0);
}
EXTERN_C_END
