/*
   Implements the sequential pthread based vectors.
*/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <petscconf.h>
#include <../src/vec/vec/impls/dvecimpl.h>                          /*I  "petscvec.h" I*/
#include <../src/vec/vec/impls/seq/seqpthread/vecpthreadimpl.h>
#include <petscblaslapack.h>
#include <private/petscaxpy.h>
#include <pthread.h>
#include <unistd.h>

/* Global variables */
extern PetscMPIInt  PetscMaxThreads;
extern PetscInt     PetscMainThreadShareWork;
extern PetscInt*    ThreadCoreAffinity;
extern PetscInt     MainThreadCoreAffinity;

PetscInt vecs_created=0;
Kernel_Data *kerneldatap;
Kernel_Data **pdata;

/* Global function pointer */
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Change these macros so can be used in thread kernels */
#undef CHKERRQP
#define CHKERRQP(ierr) if (ierr) return (void*)(long int)ierr

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
extern void DoCoreAffinity(void);
#else
#define DoCoreAffinity()
#endif

void* VecDot_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  const PetscScalar *x, *y;
  PetscInt    n;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
  y = (const PetscScalar*)data->y;
  n = data->n;
#if defined(PETSC_USE_COMPLEX)
  PetscInt i;
  PetscScalar sum = 0.0;
  for(i=0;i < n; i++) {
    sum += x[i]*PetscConj(y[i]);
  }
  data->result = sum;
# else
  PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
  data->result = BLASdot_(&bn,x,&one,y,&one);
#endif
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqPThread"
PetscErrorCode VecDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x = &xa[ix[i]];
    kerneldatap[i].y = &ya[ix[i]];
    kerneldatap[i].n = nx[i];
    pdata[i]         = &kerneldatap[i];
  }

  ierr = MainJob(VecDot_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);

  /* gather result */
  *z = kerneldatap[0].result;
  for(i=1; i<x->nthreads; i++) {
    *z += kerneldatap[i].result;
  }

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);

  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1+x->nthreads-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecTDot_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  const PetscScalar *x, *y;
  PetscInt    n;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
  y = (const PetscScalar*)data->y;
  n = data->n;
#if defined(PETSC_USE_COMPLEX)
  PetscInt i;
  PetscScalar sum = 0.0;
  for(i=0;i < n; i++) {
    sum += x[i]*y[i];
  }
  data->result = sum;
# else
  PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
  data->result = BLASdot_(&bn,x,&one,y,&one);
#endif
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecTDot_SeqPThread"
PetscErrorCode VecTDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x = &xa[ix[i]];
    kerneldatap[i].y = &ya[ix[i]];
    kerneldatap[i].n = nx[i];
    pdata[i]         = &kerneldatap[i];
  }

  ierr = MainJob(VecTDot_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);

  /* gather result */
  *z = kerneldatap[0].result;
  for(i=1; i<x->nthreads; i++) {
    *z += kerneldatap[i].result;
  }

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);

  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1+x->nthreads-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

void* VecScale_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  PetscScalar a,*x;
  PetscBLASInt one = 1, bn;
  PetscInt    n;

  DoCoreAffinity();
  x = data->x;
  a = data->alpha;
  n = data->n;
  bn = PetscBLASIntCast(n);
  BLASscal_(&bn,&a,x,&one);
  return(0);
}

PetscErrorCode VecSet_SeqPThread(Vec,PetscScalar);

#undef __FUNCT__
#define __FUNCT__ "VecScale_SeqPThread"
PetscErrorCode VecScale_SeqPThread(Vec xin, PetscScalar alpha)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;

  PetscFunctionBegin;

  if (alpha == 0.0) {
    ierr = VecSet_SeqPThread(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    PetscScalar *xa;
    PetscInt    i;

    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x     = &xa[ix[i]];
      kerneldatap[i].alpha = alpha;
      kerneldatap[i].n     = nx[i];  
      pdata[i]             = &kerneldatap[i];
    }
    ierr = MainJob(VecScale_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);

    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

void* VecAXPY_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  PetscScalar a,*y;
  const PetscScalar *x;
  PetscBLASInt one = 1, bn;
  PetscInt    n;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
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
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (alpha != 0.0) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x = &xa[ix[i]];
      kerneldatap[i].y = &ya[ix[i]];
      kerneldatap[i].n = nx[i];
      kerneldatap[i].alpha = alpha;
      pdata[i] = &kerneldatap[i];
    }
    ierr = MainJob(VecAXPY_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecAYPX_Kernel(void *arg)
{
  Kernel_Data       *data = (Kernel_Data*)arg;
  PetscScalar       a,*y;
  const PetscScalar *x;
  PetscInt          n;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;
  n = data->n;

#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
  fortranaypx_(&n,&a,x,y);
#else
  PetscInt i;
  if(a==-1.0) {
    for (i=0; i<n; i++) {
      y[i] = x[i] - y[i];
    }
  }
  else {
    for (i=0; i<n; i++) {
      y[i] = x[i] + a*y[i];
    }
  }
#endif
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAYPX_SeqPThread"
PetscErrorCode VecAYPX_SeqPThread(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (alpha == 0.0) {
    ierr = VecCopy(xin,yin);CHKERRQ(ierr);
  }
  else if (alpha == 1.0) {
    ierr = VecAXPY_SeqPThread(yin,alpha,xin);CHKERRQ(ierr);
  }
  else {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x     = &xa[ix[i]];
      kerneldatap[i].y     = &ya[ix[i]];
      kerneldatap[i].n     = nx[i];
      kerneldatap[i].alpha = alpha;
      pdata[i]             = &kerneldatap[i];
    }
    ierr = MainJob(VecAYPX_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
    if(alpha==-1.0) {
      ierr = PetscLogFlops(1.0*xin->map->n);CHKERRQ(ierr);
    }
    else {
      ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

void* VecAX_Kernel(void *arg)
{
  Kernel_Data       *data = (Kernel_Data*)arg;
  PetscScalar        a,*y;
  const PetscScalar *x;
  PetscInt           n,i;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;
  n = data->n;
  for(i=0;i < n; i++) y[i] = a*x[i];
  return(0);
}

void* VecAXPBY_Kernel(void *arg)
{
  Kernel_Data       *data = (Kernel_Data*)arg;
  PetscScalar        a,b,*y;
  const PetscScalar *x;
  PetscInt           n,i;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;
  b = data->beta;
  n = data->n;
  for(i=0;i < n; i++) y[i] = a*x[i] + b*y[i];
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_SeqPThread"
PetscErrorCode VecAXPBY_SeqPThread(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i=0;

  PetscFunctionBegin;

  if(alpha == 0.0 && beta == 1.0) {
    PetscFunctionReturn(0);
  }

  if(alpha == (PetscScalar)0.0) {
    ierr = VecScale_SeqPThread(yin,beta);CHKERRQ(ierr);
  } else if (beta == (PetscScalar)1.0) {
    ierr = VecAXPY_SeqPThread(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)1.0) {
    ierr = VecAYPX_SeqPThread(yin,beta,xin);CHKERRQ(ierr);
  } else if (beta == (PetscScalar)0.0) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x = &xa[ix[i]];
      kerneldatap[i].y = &ya[ix[i]];
      kerneldatap[i].n = nx[i];
      kerneldatap[i].alpha = alpha;
      pdata[i] = &kerneldatap[i];
    }
    
    ierr = MainJob(VecAX_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
      
  } else {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x = &xa[ix[i]];
      kerneldatap[i].y = &ya[ix[i]];
      kerneldatap[i].n = nx[i];
      kerneldatap[i].alpha = alpha;
      kerneldatap[i].beta = beta;
      pdata[i] = &kerneldatap[i];
    }
    
    ierr = MainJob(VecAXPBY_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
    
  }
  PetscFunctionReturn(0);
}

void* VecWAXPY_Kernel(void *arg)
{
  Kernel_Data       *data = (Kernel_Data*)arg;
  PetscScalar       a,*ww;
  const PetscScalar *xx,*yy;
  PetscInt          i,n;
  PetscErrorCode    ierr;

  DoCoreAffinity();
  ww = data->w;
  xx = (const PetscScalar*)data->x;
  yy = (const PetscScalar*)data->y;
  a = data->alpha;
  n = data->n;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
  fortranwaxpy_(&n,&a,xx,yy,ww);
#else
  if (a == 0.0) {
    ierr = PetscMemcpy(ww,yy,n*sizeof(PetscScalar));CHKERRQP(ierr);
  }
  else if(a==-1.0) {
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
#endif
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_SeqPThread"
PetscErrorCode VecWAXPY_SeqPThread(Vec win, PetscScalar alpha,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *w =  (Vec_SeqPthread*)win->data;
  PetscInt          *iw = w->arrindex;
  PetscInt          *nw = w->nelem;
  PetscScalar       *ya,*xa,*wa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecGetArray(win,&wa);CHKERRQ(ierr);

  for (i=0; i<w->nthreads; i++) {
    kerneldatap[i].x = &xa[iw[i]];
    kerneldatap[i].y = &ya[iw[i]];
    kerneldatap[i].w = &wa[iw[i]];
    kerneldatap[i].alpha = alpha;
    kerneldatap[i].n = nw[i];
    pdata[i] = &kerneldatap[i];
  }
  ierr = MainJob(VecWAXPY_Kernel,(void**)pdata,w->nthreads,w->cpu_affinity);

  if (alpha == 1.0 || alpha == -1.0) {
    ierr = PetscLogFlops(1.0*win->map->n);CHKERRQ(ierr);
  }
  else {
    ierr = PetscLogFlops(2.0*win->map->n);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&wa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecNorm_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  const PetscScalar *x;
  NormType type;
  PetscInt    i,n;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
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
    data->result = BLASdot_(&bn,x,&one,x,&one);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNorm_SeqPThread"
PetscErrorCode VecNorm_SeqPThread(Vec xin,NormType type,PetscReal* z)
{

  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *xa;

  PetscFunctionBegin;

  if(type == NORM_1_AND_2) {
    ierr = VecNorm_SeqPThread(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqPThread(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  else {
    PetscInt i;

    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x = &xa[ix[i]];
      kerneldatap[i].typeUse = type;
      kerneldatap[i].n = nx[i];
      pdata[i] = &kerneldatap[i];
    }
    ierr = MainJob(VecNorm_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    /* collect results */
    *z = (PetscReal)kerneldatap[0].result;
    if(type == NORM_1) {
      for(i=1; i<x->nthreads; i++) {
        *z += (PetscReal)kerneldatap[i].result;
      }
      ierr = PetscLogFlops(PetscMax(xin->map->n-1.0+x->nthreads-1,0.0));CHKERRQ(ierr);
    }
    else if(type == NORM_2 || type == NORM_FROBENIUS) {
      *z = (PetscReal)kerneldatap[0].result;
      for(i=1; i<x->nthreads; i++) {
        *z += (PetscReal)kerneldatap[i].result;
      }
      *z = PetscSqrtReal(*z);
      ierr = PetscLogFlops(PetscMax(2.0*xin->map->n-1+x->nthreads-1,0.0));CHKERRQ(ierr);
    }
    else {
      PetscReal    maxv = 0.0,tmp;
      for(i=0; i<x->nthreads; i++) {
        tmp = (PetscReal)kerneldatap[i].result;
        if(tmp>maxv) {
          maxv = tmp;
        }
      }
      *z = maxv;
    }
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecMDot_Kernel4(void* arg)
{
  Kernel_Data        *data = (Kernel_Data*)arg;
  PetscInt           n = data->n;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;
  const PetscScalar  *y1 = (const PetscScalar*)data->y1;
  const PetscScalar  *y2 = (const PetscScalar*)data->y2;
  const PetscScalar  *y3 = (const PetscScalar*)data->y3;

  DoCoreAffinity();
#if defined(PETSC_USE_COMPLEX)
    PetscInt i;
    PetscScalar sum0,sum1,sum2,sum3;
    sum0 = sum1 = sum2 = sum3 = 0.0;
    for(i=0;i<n;i++) {
      sum0 += x[i]*PetscConj(y0[i]);
      sum1 += x[i]*PetscConj(y1[i]);
      sum2 += x[i]*PetscConj(y2[i]);
      sum3 += x[i]*PetscConj(y3[i]);
    }
    data->result0 = sum0; data->result1 = sum1; data->result2 = sum2; data->result3 = sum3;
# else
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
    data->result0 = BLASdot_(&bn,x,&one,y0,&one);
    data->result1 = BLASdot_(&bn,x,&one,y1,&one);
    data->result2 = BLASdot_(&bn,x,&one,y2,&one);
    data->result3 = BLASdot_(&bn,x,&one,y3,&one);
#endif
    return(0);
}

void* VecMDot_Kernel3(void* arg)
{
  Kernel_Data        *data = (Kernel_Data*)arg;
  PetscInt           n = data->n;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;
  const PetscScalar  *y1 = (const PetscScalar*)data->y1;
  const PetscScalar  *y2 = (const PetscScalar*)data->y2;

  DoCoreAffinity();
#if defined(PETSC_USE_COMPLEX)
    PetscInt i;
    PetscScalar sum0,sum1,sum2;
    sum0 = sum1 = sum2 = 0.0;
    for(i=0;i<n;i++) {
      sum0 += x[i]*PetscConj(y0[i]);
      sum1 += x[i]*PetscConj(y1[i]);
      sum2 += x[i]*PetscConj(y2[i]);
    }
    data->result0 = sum0; data->result1 = sum1; data->result2 = sum2;
# else
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
    data->result0 = BLASdot_(&bn,x,&one,y0,&one);
    data->result1 = BLASdot_(&bn,x,&one,y1,&one);
    data->result2 = BLASdot_(&bn,x,&one,y2,&one);
#endif
    return(0);
}

void* VecMDot_Kernel2(void* arg)
{
  Kernel_Data        *data = (Kernel_Data*)arg;
  PetscInt           n = data->n;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;
  const PetscScalar  *y1 = (const PetscScalar*)data->y1;

  DoCoreAffinity();
#if defined(PETSC_USE_COMPLEX)
    PetscInt i;
    PetscScalar sum0,sum1;
    sum0 = sum1 = 0.0;
    for(i=0;i<n;i++) {
      sum0 += x[i]*PetscConj(y0[i]);
      sum1 += x[i]*PetscConj(y1[i]);
    }
    data->result0 = sum0; data->result1 = sum1;
# else
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
    data->result0 = BLASdot_(&bn,x,&one,y0,&one);
    data->result1 = BLASdot_(&bn,x,&one,y1,&one);
#endif
    return(0);
}

void* VecMDot_Kernel1(void* arg)
{
  Kernel_Data        *data = (Kernel_Data*)arg;
  PetscInt           n = data->n;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;

  DoCoreAffinity();
#if defined(PETSC_USE_COMPLEX)
    PetscInt i;
    PetscScalar sum0;
    sum0 = 0.0;
    for(i=0;i<n;i++) {
      sum0 += x[i]*PetscConj(y0[i]);
    }
    data->result0 = sum0;
# else
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n);
    data->result0 = BLASdot_(&bn,x,&one,y0,&one);
#endif
    return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqPThread"
PetscErrorCode VecMDot_SeqPThread(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode ierr;
  Vec_SeqPthread *x = (Vec_SeqPthread*)xin->data;
  PetscInt       *ix = x->arrindex;
  PetscInt       *nx = x->nelem;
  Vec            *yy = (Vec*)yin;
  PetscScalar    *xa,*y0,*y1,*y2,*y3;
  PetscInt       i,j,j_rem;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  switch(j_rem = nv&0x3) {
  case 3:
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecGetArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecGetArray(yy[2],&y2);CHKERRQ(ierr);

    for(i=0;i<x->nthreads;i++) {
      kerneldatap[i].x      = xa + ix[i];
      kerneldatap[i].y0     = y0 + ix[i];
      kerneldatap[i].y1     = y1 + ix[i];
      kerneldatap[i].y2     = y2 + ix[i];
      kerneldatap[i].n      = nx[i];
      pdata[i]              = &kerneldatap[i];
    }
    ierr = MainJob(VecMDot_Kernel3,(void**)pdata,x->nthreads,x->cpu_affinity);

    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[2],&y2);CHKERRQ(ierr);

    z[0] = kerneldatap[0].result0;
    for(j=1;j<x->nthreads;j++) {
      z[0] += kerneldatap[j].result0;
    }
    z[1] = kerneldatap[0].result1;
    for(j=1;j<x->nthreads;j++) {
      z[1] += kerneldatap[j].result1;
    }
    z[2] = kerneldatap[0].result2;
    for(j=1;j<x->nthreads;j++) {
      z[2] += kerneldatap[j].result2;
    }
    yy += 3;
    z  += 3;
    break;
  case 2:
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecGetArray(yy[1],&y1);CHKERRQ(ierr);

    for(i=0;i<x->nthreads;i++) {
      kerneldatap[i].x      = xa + ix[i];
      kerneldatap[i].y0     = y0 + ix[i];
      kerneldatap[i].y1     = y1 + ix[i];
      kerneldatap[i].n      = nx[i];
      pdata[i]              = &kerneldatap[i];
    }
    ierr = MainJob(VecMDot_Kernel2,(void**)pdata,x->nthreads,x->cpu_affinity);

    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[1],&y1);CHKERRQ(ierr);

    z[0] = kerneldatap[0].result0;
    for(j=1;j<x->nthreads;j++) {
      z[0] += kerneldatap[j].result0;
    }
    z[1] = kerneldatap[0].result1;
    for(j=1;j<x->nthreads;j++) {
      z[1] += kerneldatap[j].result1;
    }
    yy += 2; z += 2;
    break;
  case 1:
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);

    for(i=0;i<x->nthreads;i++) {
      kerneldatap[i].x    = xa + ix[i];
      kerneldatap[i].y0   = y0 + ix[i];
      kerneldatap[i].n    = nx[i];
      pdata[i]            = &kerneldatap[i];
    }
    ierr = MainJob(VecMDot_Kernel1,(void**)pdata,x->nthreads,x->cpu_affinity);
    
    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);

    z[0] = kerneldatap[0].result0;
    for(j=1;j<x->nthreads;j++) {
      z[0] += kerneldatap[j].result0;
    }
    yy++; z++;
    break;
  }
  for(j=j_rem;j<nv;j+=4) {
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecGetArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecGetArray(yy[2],&y2);CHKERRQ(ierr);
    ierr = VecGetArray(yy[3],&y3);CHKERRQ(ierr);

    for(i=0;i<x->nthreads;i++) {
      kerneldatap[i].x      = xa + ix[i];
      kerneldatap[i].y0     = y0 + ix[i];
      kerneldatap[i].y1     = y1 + ix[i];
      kerneldatap[i].y2     = y2 + ix[i];
      kerneldatap[i].y3     = y3 + ix[i];
      kerneldatap[i].n      = nx[i];
      pdata[i]              = &kerneldatap[i];
    }
    ierr = MainJob(VecMDot_Kernel4,(void**)pdata,x->nthreads,x->cpu_affinity);

    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[2],&y2);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[3],&y3);CHKERRQ(ierr);

    z[0] = kerneldatap[0].result0;
    for(j=1;j<x->nthreads;j++) {
      z[0] += kerneldatap[j].result0;
    }
    z[1] = kerneldatap[0].result1;
    for(j=1;j<x->nthreads;j++) {
      z[1] += kerneldatap[j].result1;
    }
    z[2] = kerneldatap[0].result2;
    for(j=1;j<x->nthreads;j++) {
      z[2] += kerneldatap[j].result2;
    }
    z[3] = kerneldatap[0].result3;
    for(j=1;j<x->nthreads;j++) {
      z[3] += kerneldatap[j].result3;
    }
    yy += 4;
    z  += 4;
  }    
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);

  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1+x->nthreads-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMTDot_SeqPThread"
PetscErrorCode VecMTDot_SeqPThread(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode   ierr=0;
  PetscInt         j;

  PetscFunctionBegin;

  for(j=0;j<nv;j++) {
    ierr = VecTDot_SeqPThread(xin,yin[j],&z[j]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


void* VecMax_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  const PetscScalar *xx = (const PetscScalar*)data->x;
  PetscInt          i,j,n = data->n;
  PetscReal         lmax,tmp;

  DoCoreAffinity();
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
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscInt          i,j=0;
  PetscScalar       *xa;
  PetscReal         max;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  if (!xin->map->n) {
    max = PETSC_MIN_REAL;
    j   = -1;
  } else {
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x    = &xa[ix[i]];
      kerneldatap[i].gind = ix[i];
      kerneldatap[i].n    = nx[i];
      pdata[i]            = &kerneldatap[i];
    }
    ierr = MainJob(VecMax_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    /* collect results, determine global max, global index */
    max = kerneldatap[0].localmax;
    j   = kerneldatap[0].localind;
    for(i=1; i<x->nthreads; i++) {
      if(kerneldatap[i].localmax > max) {
        max = kerneldatap[i].localmax;
        j   = kerneldatap[i].gind+kerneldatap[i].localind;
      }
    }
  }
  *z   = max;
  if (idx) *idx = j;
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecMin_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  const PetscScalar *xx = (const PetscScalar*)data->x;
  PetscInt          i,j,n = data->n;
  PetscReal         lmin,tmp;

  DoCoreAffinity();
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
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscInt          i,j=0;
  PetscScalar       *xa;
  PetscReal         min;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  if (!xin->map->n) {
    min = PETSC_MAX_REAL;
    j   = -1;
  } else {
    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x    = &xa[ix[i]];
      kerneldatap[i].gind = ix[i];
      kerneldatap[i].n    = nx[i];
      pdata[i]            = &kerneldatap[i];
    }

    ierr = MainJob(VecMin_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    /* collect results, determine global max, global index */
    min = kerneldatap[0].localmin;
    j   = kerneldatap[0].localind;
    for(i=1; i<x->nthreads; i++) {
      if(kerneldatap[i].localmin < min) {
        min = kerneldatap[i].localmin;
        j   = kerneldatap[i].gind+kerneldatap[i].localind;
      }
    }
  }
  *z   = min;
  if (idx) *idx = j;
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>
void* VecPointwiseMult_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  PetscScalar *ww = data->w,*xx = data->x,*yy = data->y;
  PetscInt    n = data->n,i;

  DoCoreAffinity();
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

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_SeqPThread"
PetscErrorCode VecPointwiseMult_SeqPThread(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *w =  (Vec_SeqPthread*)win->data;
  PetscInt          *iw=w->arrindex;
  PetscInt          *nw = w->nelem;
  PetscScalar       *ya,*xa,*wa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecGetArray(win,&wa);CHKERRQ(ierr);

  for (i=0; i<w->nthreads; i++) {
    kerneldatap[i].w = &wa[iw[i]];
    kerneldatap[i].x = &xa[iw[i]];
    kerneldatap[i].y = &ya[iw[i]];
    kerneldatap[i].n = nw[i];
    pdata[i]         = &kerneldatap[i];
  }

  ierr  = MainJob(VecPointwiseMult_Kernel,(void**)pdata,w->nthreads,w->cpu_affinity);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&wa);CHKERRQ(ierr);
  ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecPointwiseDivide_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  PetscScalar *ww = data->w,*xx = data->x,*yy = data->y;
  PetscInt    n = data->n,i;

  DoCoreAffinity();
  for (i=0; i<n; i++) {
    ww[i] = xx[i] / yy[i];
  }
  return(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_SeqPThread"
PetscErrorCode VecPointwiseDivide_SeqPThread(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *w =  (Vec_SeqPthread*)win->data;
  PetscInt          *iw=w->arrindex;
  PetscInt          *nw = w->nelem;
  PetscScalar       *ya,*xa,*wa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecGetArray(win,&wa);CHKERRQ(ierr);

  for (i=0; i<w->nthreads; i++) {
    kerneldatap[i].w = &wa[iw[i]];
    kerneldatap[i].x = &xa[iw[i]];
    kerneldatap[i].y = &ya[iw[i]];
    kerneldatap[i].n = nw[i];
    pdata[i]         = &kerneldatap[i];
  }

  ierr  = MainJob(VecPointwiseDivide_Kernel,(void**)pdata,w->nthreads,w->cpu_affinity);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&wa);CHKERRQ(ierr);
  ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscblaslapack.h>
void* VecSwap_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  PetscScalar *xa = data->x,*ya = data->y;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(data->n);

  DoCoreAffinity();
  BLASswap_(&bn,xa,&one,ya,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSwap_SeqPThread"
PetscErrorCode VecSwap_SeqPThread(Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x = &xa[ix[i]];
      kerneldatap[i].y = &ya[ix[i]];
      kerneldatap[i].n = nx[i];
      pdata[i]         = &kerneldatap[i];
    }

    ierr = MainJob(VecSwap_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecSetRandom_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  PetscScalar  *xx = data->x;
  PetscRandom  r = data->rand;
  PetscInt     i,n = data->n;
  PetscErrorCode ierr;

  DoCoreAffinity();
  for(i=0; i<n; i++) {
    ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQP(ierr);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_SeqPThread"
PetscErrorCode VecSetRandom_SeqPThread(Vec xin,PetscRandom r)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          i,*nx = x->nelem;
  PetscScalar       *xa;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x    = xa+ix[i];
    kerneldatap[i].rand = r;
    kerneldatap[i].n    = nx[i];
    pdata[i]            = &kerneldatap[i];
   }

  ierr = MainJob(VecSetRandom_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecCopy_Kernel(void *arg)
{
  Kernel_Data        *data = (Kernel_Data*)arg;
  const PetscScalar  *xa = (const PetscScalar*)data->x;
  PetscScalar        *ya = data->y;
  PetscInt           n = data->n;
  PetscErrorCode     ierr;

  DoCoreAffinity();
  ierr = PetscMemcpy(ya,xa,n*sizeof(PetscScalar));CHKERRQP(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqPThread"
PetscErrorCode VecCopy_SeqPThread(Vec xin,Vec yin)
{

  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x   = xa+ix[i];
      kerneldatap[i].y   = ya+ix[i];
      kerneldatap[i].n   = nx[i];
      pdata[i]           = &kerneldatap[i];
    }
    ierr = MainJob(VecCopy_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);

    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

void* VecMAXPY_Kernel(void* arg)
{
  Kernel_Data       *data = (Kernel_Data*)arg;
  PetscErrorCode    ierr;
  PetscInt          n = data->n,nv=data->nvec,istart=data->istart,j,j_rem;
  const PetscScalar *alpha=data->amult,*yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx = data->x,alpha0,alpha1,alpha2,alpha3;
  Vec*              y = (Vec*)data->yvec;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  DoCoreAffinity();
  switch (j_rem=nv&0x3) {
  case 3: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQP(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQP(ierr);
    yy0 += istart; yy1 += istart; yy2 += istart;
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
    yy0 += istart; yy1 += istart;
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
    yy0 += istart; yy1 += istart;
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
    yy0 += istart; yy1 += istart; yy2 += istart; yy3 += istart;
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
PetscErrorCode VecMAXPY_SeqPThread(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *yin)
{
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          *nx = x->nelem;
  PetscInt          i;
  Vec               *yy = (Vec *)yin;
  PetscScalar       *xa;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x      = xa + ix[i];
    kerneldatap[i].yvec   = yy;
    kerneldatap[i].amult  = &alpha[0];
    kerneldatap[i].n      = nx[i];
    kerneldatap[i].nvec   = nv;
    kerneldatap[i].istart = ix[i];
    pdata[i]              = &kerneldatap[i];
  }
  ierr = MainJob(VecMAXPY_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = PetscLogFlops(nv*2.0*xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void* VecSet_Kernel(void *arg)
{
  Kernel_Data    *data = (Kernel_Data*)arg;
  PetscScalar    *xx = data->x;
  PetscScalar    alpha = data->alpha;
  PetscInt       i,n = data->n;
  PetscErrorCode ierr;

  DoCoreAffinity();
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
  PetscErrorCode    ierr;
  Vec_SeqPthread    *x =  (Vec_SeqPthread*)xin->data;
  PetscInt          *ix = x->arrindex;
  PetscInt          i,*nx = x->nelem;
  PetscScalar       *xa;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x       = xa+ix[i];
    kerneldatap[i].alpha   = alpha;
    kerneldatap[i].n       = nx[i];
    pdata[i]               = &kerneldatap[i];
  }
  ierr = MainJob(VecSet_Kernel,(void**)pdata,x->nthreads,x->cpu_affinity);
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_SeqPThread"
PetscErrorCode VecDestroy_SeqPThread(Vec v)
{
  Vec_SeqPthread        *vs = (Vec_SeqPthread*)v->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree2(vs->arrindex,vs->nelem);CHKERRQ(ierr);
  ierr = PetscFree(vs->cpu_affinity);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);

  vecs_created--;
  /* Free the kernel data structure on the destruction of the last vector */
  if (!vecs_created) {
    ierr = PetscFree(kerneldatap);CHKERRQ(ierr);
    ierr = PetscFree(pdata);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_SeqPThread"
PetscErrorCode VecDuplicate_SeqPThread(Vec win,Vec *V)
{
  PetscErrorCode ierr;
  Vec_SeqPthread *s = (Vec_SeqPthread*)win->data;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,V);CHKERRQ(ierr);
  ierr = PetscObjectSetPrecision((PetscObject)*V,((PetscObject)win)->precision);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,win->map->n,win->map->n);CHKERRQ(ierr);
  ierr = VecSetType(*V,((PetscObject)win)->type_name);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = VecPThreadSetNThreads(*V,s->nthreads-PetscMainThreadShareWork);CHKERRQ(ierr);
  ierr = VecPThreadSetThreadAffinities(*V,s->cpu_affinity+PetscMainThreadShareWork);CHKERRQ(ierr);
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPThreadSetNThreads"
/*@
   VecPThreadSetNThreads - Set the number of threads to be used for vector operations.

   Input Parameters
+  v - the vector
-  nthreads - number of threads

   Note:
   Use nthreads = PETSC_DECIDE for PETSc to determine the number of threads.

   Options Database keys:
   -vec_threads <nthreads> - Number of threads

   Level: intermediate

   Concepts: vectors^number of threads

.seealso: VecCreateSeqPThread(), VecPThreadGetNThreads()
@*/
PetscErrorCode VecPThreadSetNThreads(Vec v,PetscInt nthreads)
{
  PetscErrorCode ierr;
  Vec_SeqPthread *s = (Vec_SeqPthread*)v->data;
  PetscInt       Q,R;
  PetscBool      S;
  PetscInt       i,iIndex=0,nthr;
  PetscBool      flg;

  PetscFunctionBegin;

  if(s->nthreads != 0) {
    ierr = PetscFree2(s->arrindex,s->nelem);CHKERRQ(ierr);
  }

  if(nthreads == PETSC_DECIDE) {
    ierr = PetscOptionsInt("-vec_threads","Set number of threads to be used for vector operations","VecPThreadSetNThreads",PetscMaxThreads,&nthr,&flg);CHKERRQ(ierr);
    if(flg && nthr > PetscMaxThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Vec x: threads requested %D, Max. threads initialized %D",nthr,PetscMaxThreads);
    }
    if(!flg) nthr = PetscMaxThreads;
    s->nthreads = nthr+PetscMainThreadShareWork;
  } else {
    if(nthreads > PetscMaxThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Vec x: threads requested %D, Max. threads initialized %D",nthreads,PetscMaxThreads);
    }
    s->nthreads = nthreads + PetscMainThreadShareWork;
  }
  Q = v->map->n/s->nthreads;
  R = v->map->n-Q*s->nthreads;

  /* Set array portion for each thread */
  ierr = PetscMalloc2(s->nthreads,PetscInt,&s->arrindex,s->nthreads,PetscInt,&s->nelem);CHKERRQ(ierr);
  s->arrindex[0] = 0;
  for (i=0; i< s->nthreads; i++) {
    s->arrindex[i] = iIndex;
    S = (PetscBool)(i<R);
    s->nelem[i] = S?Q+1:Q;
    iIndex += s->nelem[i];
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPThreadGetNThreads"
/*@
   VecPThreadGetNThreads - Returns the number of threads used for vector operations.

   Input Parameter
.  v - the vector

   Output Parameter
.  nthreads - number of threads

   Level: intermediate

   Concepts: vectors^number of threads

.seealso: VecPThreadSetNThreads()
@*/
PetscErrorCode VecPThreadGetNThreads(Vec v,PetscInt *nthreads)
{
  Vec_SeqPthread *s = (Vec_SeqPthread*)v->data;
  
  PetscFunctionBegin;
  *nthreads = s->nthreads - PetscMainThreadShareWork;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPThreadSetThreadAffinities"
/*@
   VecPThreadSetThreadAffinities - Sets the CPU affinities of vector threads.

   Input Parameters
+  v - the vector
-  affinities - list of cpu affinities for threads.

   Notes:
   Must set affinities for all the threads used with the vector.
 
   Use affinities[] = PETSC_NULL for PETSc to decide the thread affinities.

   Options Database Keys:
   -vec_thread_affinities - Comma seperated list of thread affinities

   Level: intermediate

   Concepts: vectors^thread cpu affinity

.seealso: VecPThreadGetThreadAffinities()
@*/
PetscErrorCode VecPThreadSetThreadAffinities(Vec v,const PetscInt affinities[])
{
  PetscErrorCode  ierr;
  Vec_SeqPthread *s = (Vec_SeqPthread*)v->data;
  PetscInt        nmax=PetscMaxThreads+PetscMainThreadShareWork;
  PetscBool       flg;

  PetscFunctionBegin;

  if(s->cpu_affinity) {
    ierr = PetscFree(s->cpu_affinity);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(s->nthreads*sizeof(PetscInt),&s->cpu_affinity);CHKERRQ(ierr);

  if(affinities == PETSC_NULL) {
    /* PETSc decides affinities */
    PetscInt        *thread_affinities;
    ierr = PetscMalloc(nmax*sizeof(PetscInt),&thread_affinities);CHKERRQ(ierr);
    /* Check if run-time option is set */
    ierr = PetscOptionsIntArray("-vec_thread_affinities","Set CPU affinity for each thread","VecPThreadSetThreadAffinities",thread_affinities,&nmax,&flg);CHKERRQ(ierr);
    if(flg) {
      if(nmax != s->nthreads-PetscMainThreadShareWork) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, vector Threads = %D, CPU affinities set = %D",s->nthreads-PetscMainThreadShareWork,nmax);
      }
      ierr = PetscMemcpy(s->cpu_affinity+PetscMainThreadShareWork,thread_affinities,(s->nthreads-PetscMainThreadShareWork)*sizeof(PetscInt));
    } else {
      /* Reuse the core affinities set for the first nthreads */
      ierr = PetscMemcpy(s->cpu_affinity+PetscMainThreadShareWork,ThreadCoreAffinity,(s->nthreads-PetscMainThreadShareWork)*sizeof(PetscInt));
    }
    ierr = PetscFree(thread_affinities);CHKERRQ(ierr);
  } else {
    /* Set user provided affinities */
    ierr = PetscMemcpy(s->cpu_affinity+PetscMainThreadShareWork,affinities,(s->nthreads-PetscMainThreadShareWork)*sizeof(PetscInt));
  }
  if(PetscMainThreadShareWork) s->cpu_affinity[0] = MainThreadCoreAffinity;

  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {VecDuplicate_SeqPThread, /* 1 */
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_SeqPThread,
            VecMDot_SeqPThread,
            VecNorm_SeqPThread, 
            VecTDot_SeqPThread,
            VecMTDot_SeqPThread,
            VecScale_SeqPThread,
            VecCopy_SeqPThread, /* 10 */
            VecSet_SeqPThread,
            VecSwap_Seq,
            VecAXPY_SeqPThread,
            VecAXPBY_SeqPThread,
            VecMAXPY_SeqPThread,
            VecAYPX_SeqPThread,
            VecWAXPY_SeqPThread,
            VecAXPBYPCZ_Seq,
            VecPointwiseMult_SeqPThread,
            VecPointwiseDivide_SeqPThread, 
            VecSetValues_Seq, /* 20 */
            0,0,
            0,
            VecGetSize_Seq,
            VecGetSize_Seq,
            0,
            VecMax_SeqPThread,
            VecMin_SeqPThread,
            VecSetRandom_SeqPThread,
            VecSetOption_Seq, /* 30 */
            VecSetValuesBlocked_Seq,
            VecDestroy_SeqPThread,
            VecView_Seq,
            VecPlaceArray_Seq,
            VecReplaceArray_Seq,
            VecDot_SeqPThread,
            VecTDot_SeqPThread,
            VecNorm_SeqPThread,
            VecMDot_SeqPThread,
            VecMTDot_SeqPThread, /* 40 */
	    VecLoad_Default,		       
            VecReciprocal_Default,
            VecConjugate_Seq,
	    0,
	    0,
            VecResetArray_Seq,
            0,
            VecMaxPointwiseDivide_Seq,
            VecPointwiseMax_Seq,
            VecPointwiseMaxAbs_Seq,
            VecPointwiseMin_Seq,
            VecGetValues_Seq,
    	    0,
    	    0,
    	    0,
    	    0,
    	    0,
    	    0,
   	    VecStrideGather_Default,
   	    VecStrideScatter_Default
          };

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_SeqPThread_Private"
PetscErrorCode VecCreate_SeqPThread_Private(Vec v,const PetscScalar array[])
{
  Vec_SeqPthread *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  ierr = PetscNewLog(v,Vec_SeqPthread,&s);CHKERRQ(ierr);
  v->data            = (void*)s;

  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar *)array;
  s->array_allocated = 0;
  s->nthreads        = 0;
  s->arrindex        = 0;
  s->cpu_affinity    = 0;

  /* If this is the first vector being created then also create the common Kernel data structure */
  if(vecs_created == 0) {
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Kernel_Data),&kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Kernel_Data*),&pdata);CHKERRQ(ierr);
  }
  vecs_created++;

  if (v->map->bs == -1) v->map->bs = 1;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);

  /* Set the number of threads */
  ierr = VecPThreadSetNThreads(v,PETSC_DECIDE);CHKERRQ(ierr);
  /* Set thread affinities */
  ierr = VecPThreadSetThreadAffinities(v,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQPTHREAD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
   VECSEQPTHREAD - VECSEQPTHREAD = "seqpthread" - The basic sequential vector using posix threads

   Options Database Keys:
.  -vec_type seqpthread - sets the vector type to VECSEQPTHREAD during a call to VecSetFromOptions()

   Level: intermediate

.seealso: VecCreate(), VecCreateSeqPThread(), VecSetType(), VecSetFromOptions(), VECSEQ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_SeqPThread"
PetscErrorCode VecCreate_SeqPThread(Vec V)
{
  Vec_SeqPthread  *s;
  PetscScalar     *array;
  PetscErrorCode  ierr;
  PetscInt        n = PetscMax(V->map->n,V->map->N);
  PetscMPIInt     size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQPTHREAD on more than one process");
  ierr = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_SeqPThread_Private(V,array);CHKERRQ(ierr);
  ierr = VecSet_SeqPThread(V,0.0);CHKERRQ(ierr);
  s    = (Vec_SeqPthread*)V->data;
  s->array_allocated = (PetscScalar*)s->array;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecCreateSeqPThread"
/*@
   VecCreateSeqPThread - Creates a standard, sequential array-style vector using posix threads.

   Collective on MPI_Comm

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  n - the vector length 
.  nthreads - number of threads
-  affinities - thread affinities

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Use nthreads = PETSC_DECIDE for PETSc to decide the number of threads and
   affinities = PETSC_NULL to decide the thread affinities.

   Options Database Keys:
   -vec_threads <nthreads> - Sets number of threads to be used for vector operations
   -vec_thread_affinities  - Comma seperated list of thread affinities

   Level: intermediate

   Concepts: vectors^creating sequential with threads

.seealso: VecCreateSeq(), VecPThreadSetNThreads(), VecPThreadSetThreadAffinities(), VecDuplicate(), VecDuplicateVecs()
@*/
PetscErrorCode VecCreateSeqPThread(MPI_Comm comm,PetscInt n,PetscInt nthreads,PetscInt affinities[],Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQPTHREAD);CHKERRQ(ierr);
  ierr = VecPThreadSetNThreads(*v,nthreads);CHKERRQ(ierr);
  ierr = VecPThreadSetThreadAffinities(*v,affinities);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
