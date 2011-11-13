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
extern PetscBool    PetscUseThreadPool;
extern PetscMPIInt  PetscMaxThreads;
extern pthread_t*   PetscThreadPoint;
extern int*         ThreadCoreAffinity;

PetscInt vecs_created=0;
Kernel_Data *kerneldatap;
Kernel_Data **pdata;


/* Global function pointer */
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt);

/* Change these macros so can be used in thread kernels */
#undef CHKERRQP
#define CHKERRQP(ierr) if (ierr) return (void*)(long int)ierr

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
PETSC_STATIC_INLINE void DoCoreAffinity(void)
{
  if (!PetscUseThreadPool) return;
  else {
    int       i,icorr=0; 
    cpu_set_t mset;
    pthread_t pThread = pthread_self();

    for (i=0; i<PetscMaxThreads; i++) {
      if (pthread_equal(pThread,PetscThreadPoint[i])) {
        icorr = ThreadCoreAffinity[i];
      }
    }
    CPU_ZERO(&mset);
    CPU_SET(icorr,&mset);
    sched_setaffinity(0,sizeof(cpu_set_t),&mset);
  }
  return(0);
}
#else
#define DoCoreAffinity()
#endif

#undef __FUNCT__
#define __FUNCT__ "VecSeqPThreadCheckNThreads"
PETSC_STATIC_INLINE PetscErrorCode VecSeqPThreadCheckNThreads(Vec v)
{
  Vec_SeqPthread *vs = (Vec_SeqPthread*)v->data;

  PetscFunctionBegin;
  if(!vs->nthreads) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecSeqPThreadSetNThreads() must be called before doing any vector operations");
  PetscFunctionReturn(0);
}

void* VecDot_Kernel(void *arg)
{
  Kernel_Data *data = (Kernel_Data*)arg;
  const PetscScalar *x, *y;
  PetscBLASInt one = 1, bn;
  PetscInt    n;

  DoCoreAffinity();
  x = (const PetscScalar*)data->x;
  y = (const PetscScalar*)data->y;
  n = data->n;.
  bn = PetscBLASIntCast(n);
  data->result = BLASdot_(&bn,x,&one,y,&one);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x = &xa[ix[i]];
    kerneldatap[i].y = &ya[ix[i]];
    kerneldatap[i].n = nx[i];
    pdata[i]         = &kerneldatap[i];
  }

  ierr = MainJob(VecDot_Kernel,(void**)pdata,x->nthreads);

  /* gather result */
  *z = 0.0;
  for(i=0; i<x->nthreads; i++) {
    *z += kerneldatap[i].result;
  }

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);

  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

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
    ierr = MainJob(VecScale_Kernel,(void**)pdata,x->nthreads);

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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
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
    ierr = MainJob(VecAXPY_Kernel,(void**)pdata,x->nthreads);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

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
    ierr = MainJob(VecAYPX_Kernel,(void**)pdata,x->nthreads);
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
  ierr = VecSeqPThreadCheckNThreads(win);CHKERRQ(ierr);

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
  ierr = MainJob(VecWAXPY_Kernel,(void**)pdata,w->nthreads);

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
    data->result = BLASnrm2_(&bn,x,&one);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

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
    ierr = MainJob(VecNorm_Kernel,(void**)pdata,x->nthreads);
    /* collect results */
    *z = 0.0;
    if(type == NORM_1) {
      for(i=0; i<x->nthreads; i++) {
        *z += kerneldatap[i].result;
      }
      ierr = PetscLogFlops(PetscMax(xin->map->n-1.0,0.0));CHKERRQ(ierr);
    }
    else if(type == NORM_2 || type == NORM_FROBENIUS) {
      for(i=0; i<x->nthreads; i++) {
        *z += kerneldatap[i].result*kerneldatap[i].result;
      }
      *z = PetscSqrtReal(*z);
      ierr = PetscLogFlops(PetscMax(2.0*xin->map->n-1,0.0));CHKERRQ(ierr);
    }
    else {
      PetscReal    maxv = 0.0,tmp;
      for(i=0; i<x->nthreads; i++) {
        tmp = kerneldatap[i].result;
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


#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#include <../src/vec/vec/impls/seq/ftn-kernels/fmdot.h>
void* VecMDot_Kernel(void *arg)
{
  PetscErrorCode     ierr;
  Kernel_Data        *data = (Kernel_Data*)arg;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  Vec*               yin = (Vec*)data->yvec;
  PetscInt           nv = data->nvec;
  PetscInt           n = data->n;
  PetscScalar        *z = data->results;
  PetscInt           i,nv_rem;
  PetscScalar        sum0,sum1,sum2,sum3;
  const PetscScalar  *yy0,*yy1,*yy2,*yy3;
  Vec                *yy;

  DoCoreAffinity();
  sum0 = 0.0;
  sum1 = 0.0;
  sum2 = 0.0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    z[0] = sum0; 
    z[1] = sum1;
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    fortranmdot1_(x,yy0,&n,&sum0);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    z[0] = sum0;
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
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQP(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQP(ierr);
    yy  += 4;
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  return(0);
}
#else
void* VecMDot_Kernel(void *arg)
{
  Kernel_Data        *data = (Kernel_Data*)arg;
  const PetscScalar  *xbase = (const PetscScalar*)data->x;
  Vec*               yin = (Vec*)data->yvec;
  PetscInt           n = data->n;
  PetscInt           nv = data->nvec;
  PetscScalar*       z = data->results;
  PetscErrorCode     ierr;
  PetscInt           i,j,nv_rem,j_rem;
  PetscScalar        sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar  *yy0,*yy1,*yy2,*yy3,*x;
  Vec                *yy;

  DoCoreAffinity();
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
#endif

#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqPThread"
PetscErrorCode VecMDot_SeqPThread(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          i,j=0;
  Vec               *yy = (Vec *)yin;
  PetscScalar       *xa;
  PetscInt          n=xin->map->n,Q = nv/(PetscMaxThreads);
  PetscInt          R = nv-Q*(PetscMaxThreads);
  PetscBool         S;

  PetscFunctionBegin;
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  ierr   = VecGetArray(xin,&xa);CHKERRQ(ierr);
  for (i=0; i<PetscMaxThreads; i++) {
    S = (PetscBool)(i<R);
    kerneldatap[i].x       = xa;
    kerneldatap[i].yvec    = &yy[j];
    kerneldatap[i].n       = n;
    kerneldatap[i].nvec    = S?Q+1:Q;
    kerneldatap[i].results = &z[j];
    pdata[i]               = &kerneldatap[i];
    j += kerneldatap[i].nvec;
  }
  ierr = MainJob(VecMDot_Kernel,(void**)pdata,PetscMaxThreads);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

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
    ierr = MainJob(VecMax_Kernel,(void**)pdata,x->nthreads);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

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

    ierr = MainJob(VecMin_Kernel,(void**)pdata,x->nthreads);
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
  ierr = VecSeqPThreadCheckNThreads(win);CHKERRQ(ierr);

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

  ierr  = MainJob(VecPointwiseMult_Kernel,(void**)pdata,w->nthreads);

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
  ierr = VecSeqPThreadCheckNThreads(win);CHKERRQ(ierr);

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

  ierr  = MainJob(VecPointwiseDivide_Kernel,(void**)pdata,w->nthreads);

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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x = &xa[ix[i]];
      kerneldatap[i].y = &ya[ix[i]];
      kerneldatap[i].n = nx[i];
      pdata[i]         = &kerneldatap[i];
    }

    ierr = MainJob(VecSwap_Kernel,(void**)pdata,x->nthreads);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x    = xa+ix[i];
    kerneldatap[i].rand = r;
    kerneldatap[i].n    = nx[i];
    pdata[i]            = &kerneldatap[i];
   }

  ierr = MainJob(VecSetRandom_Kernel,(void**)pdata,x->nthreads);
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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

    for (i=0; i<x->nthreads; i++) {
      kerneldatap[i].x   = xa+ix[i];
      kerneldatap[i].y   = ya+ix[i];
      kerneldatap[i].n   = nx[i];
      pdata[i]           = &kerneldatap[i];
    }
    ierr = MainJob(VecCopy_Kernel,(void**)pdata,x->nthreads);

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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

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
  ierr = MainJob(VecMAXPY_Kernel,(void**)pdata,x->nthreads);

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
  ierr = VecSeqPThreadCheckNThreads(xin);CHKERRQ(ierr);

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

  for (i=0; i<x->nthreads; i++) {
    kerneldatap[i].x       = xa+ix[i];
    kerneldatap[i].alpha   = alpha;
    kerneldatap[i].n       = nx[i];
    pdata[i]               = &kerneldatap[i];
  }
  ierr = MainJob(VecSet_Kernel,(void**)pdata,x->nthreads);
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
  if(!s->arrindex) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must set the number of threads for the first vector before duplicating it");
  ierr = VecSeqPThreadSetNThreads(*V,s->nthreads);CHKERRQ(ierr);
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetFromOptions_SeqPThread"
PetscErrorCode VecSetFromOptions_SeqPThread(Vec v)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       nthreads=0;
  PetscFunctionBegin;
  ierr = PetscOptionsInt("-vec_threads","Set number of threads to be used with the vector","VecSeqPThreadSetNThreads",nthreads,&nthreads,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecSeqPThreadSetNThreads(v,nthreads);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSeqPThreadSetNThreads"
/*@
   VecSeqPThreadSetNThreads - Set the number of threads to be used for vector operations.

   Input Parameters
+  v - the vector
-  nthreads - number of threads

   Notes:
    This routine must be called before any vector operations are done.

   Level: intermediate

   Concepts: vectors^setting number of threads

.seealso: VecCreateSeqPThread()
@*/
PetscErrorCode VecSeqPThreadSetNThreads(Vec v,PetscInt nthreads)
{
  PetscErrorCode ierr;
  Vec_SeqPthread *s = (Vec_SeqPthread*)v->data;
  PetscInt       Q = v->map->n/nthreads;
  PetscInt       R = v->map->n-Q*nthreads;
  PetscBool      S;
  PetscInt       i,iIndex=0;


  PetscFunctionBegin;
  if(nthreads > PetscMaxThreads) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Vec x: threads requested %D, Max. threads initialized %D",nthreads,PetscMaxThreads);
  s->nthreads = nthreads;

  /* Set array portion for each thread */
  ierr = PetscMalloc2(nthreads,PetscInt,&s->arrindex,nthreads,PetscInt,&s->nelem);CHKERRQ(ierr);
  s->arrindex[0] = 0;
  for (i=0; i<nthreads; i++) {
    s->arrindex[i] = iIndex;
    S = (PetscBool)(i<R);
    s->nelem[i] = S?Q+1:Q;
    iIndex += s->nelem[i];
  }

  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {VecDuplicate_SeqPThread, /* 1 */
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_SeqPThread,
            VecMDot_SeqPThread,
            VecNorm_SeqPThread, 
            VecTDot_Seq,
            VecMTDot_Seq,
            VecScale_SeqPThread,
            VecCopy_SeqPThread, /* 10 */
            VecSet_SeqPThread,
            VecSwap_Seq,
            VecAXPY_SeqPThread,
            VecAXPBY_Seq,
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
            VecTDot_Seq,
            VecNorm_SeqPThread,
            VecMDot_SeqPThread,
            VecMTDot_Seq, /* 40 */
	    VecLoad_Default,		       
            VecReciprocal_Default,
            VecConjugate_Seq,
	    0,
	    0,
            VecResetArray_Seq,
            VecSetFromOptions_SeqPThread,
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

  /* If this is the first vector being created then also create the common Kernel data structure */
  if(vecs_created == 0) {
    ierr = PetscMalloc(PetscMaxThreads*sizeof(Kernel_Data),&kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc(PetscMaxThreads*sizeof(Kernel_Data*),&pdata);CHKERRQ(ierr);
  }
  vecs_created++;

  if (v->map->bs == -1) v->map->bs = 1;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQPTHREAD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
  ierr = PetscMemzero(array,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_SeqPThread_Private(V,array);CHKERRQ(ierr);
  s    = (Vec_SeqPthread*)V->data;
  s->array_allocated = (PetscScalar*)array;

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
-  nthreads - number of threads.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

   Concepts: vectors^creating sequential with threads

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode VecCreateSeqPThread(MPI_Comm comm,PetscInt n,PetscInt nthreads,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQPTHREAD);CHKERRQ(ierr);
  ierr = VecSeqPThreadSetNThreads(*v,nthreads);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
