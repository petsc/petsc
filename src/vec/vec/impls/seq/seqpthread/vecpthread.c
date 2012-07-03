/*
   Implements the sequential pthread based vectors.
*/
#include <petscconf.h>
#include <../src/vec/vec/impls/dvecimpl.h>                          /*I  "petscvec.h" I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>
#include <../src/vec/vec/impls/seq/seqpthread/vecpthreadimpl.h>
#include <petscblaslapack.h>
#include <petsc-private/petscaxpy.h>

PetscInt vecs_created=0;
Vec_KernelData *vec_kerneldatap;
Vec_KernelData **vec_pdata;

#undef __FUNCT__
#define __FUNCT__ "VecGetThreadOwnershipRange"
/*@
   VecGetThreadOwnershipRange - Returns the range of indices owned by
   this thread, assuming that the vectors are laid out with the first
   thread operating on the first n1 elements, next n2 elements by second,
   etc.

   Not thread collective
   Input Parameter:
+  X - the vector
-  thread_id - Thread number

   Output Parameters:
+  start - the first thread local element index, pass in PETSC_NULL if not interested
-  end   - one more than the last thread local element index, pass in PETSC_NULL if not interested

   Level: beginner

   Concepts: vector^ownership of elements on threads
@*/
PetscErrorCode VecGetThreadOwnershipRange(Vec X,PetscInt thread_id,PetscInt *start,PetscInt *end)
{
  PetscThreadsLayout tmap=X->map->tmap;
  PetscInt           *trstarts=tmap->trstarts;

  PetscFunctionBegin;
  if(start) *start = trstarts[thread_id];
  if(end)   *end   = trstarts[thread_id+1];
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_Kernel(void *arg)
{
  PetscErrorCode     ierr;
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  const PetscScalar  *x, *y;
  PetscInt           n,start,end;
  PetscBLASInt one = 1, bn,bstart;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  y = (const PetscScalar*)data->y;
  n = end-start;
  bn = PetscBLASIntCast(n);
  bstart = PetscBLASIntCast(start);
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc the second */
  data->result = BLASdot_(&bn,y+bstart,&one,x+bstart,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqPThread"
PetscErrorCode VecDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscScalar        *ya,*xa;
  PetscInt           i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X         = xin;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].x         = xa;
    vec_kerneldatap[i].y         = ya;
    vec_pdata[i]                 = &vec_kerneldatap[i];
  }

  ierr = PetscThreadsRunKernel(VecDot_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

  /* gather result */
  *z = vec_kerneldatap[0].result;
  for(i=1; i<tmap->nthreads; i++) {
    *z += vec_kerneldatap[i].result;
  }

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);

  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1+tmap->nthreads-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_Kernel(void *arg)
{
  PetscErrorCode ierr;
  Vec_KernelData *data = (Vec_KernelData*)arg;
  Vec             X = data->X;
  PetscInt        thread_id=data->thread_id;
  const PetscScalar *x, *y;
  PetscInt    n,start,end;
  PetscBLASInt one = 1, bn,bstart;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  y = (const PetscScalar*)data->y;
  n = end-start;
  bn = PetscBLASIntCast(n);
  bstart = PetscBLASIntCast(start);
  data->result = BLASdotu_(&bn,x+bstart,&one,y+bstart,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecTDot_SeqPThread"
PetscErrorCode VecTDot_SeqPThread(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap = xin->map->tmap;
  PetscScalar        *ya,*xa;
  PetscInt           i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X         = xin;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].x = xa;
    vec_kerneldatap[i].y = ya;
    vec_pdata[i]         = &vec_kerneldatap[i];
  }

  ierr = PetscThreadsRunKernel(VecTDot_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

  /* gather result */
  *z = vec_kerneldatap[0].result;
  for(i=1; i<tmap->nthreads; i++) {
    *z += vec_kerneldatap[i].result;
  }

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);

  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1+tmap->nthreads-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode VecScale_Kernel(void *arg)
{
  PetscErrorCode ierr;
  Vec_KernelData *data = (Vec_KernelData*)arg;
  Vec            X=data->X;
  PetscInt       thread_id=data->thread_id;
  PetscScalar    a,*x;
  PetscBLASInt   one = 1, bn,bstart;
  PetscInt       n,start,end;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = data->x;
  a = data->alpha;
  n = end-start;
  bn = PetscBLASIntCast(n);
  bstart = PetscBLASIntCast(start);
  BLASscal_(&bn,&a,x+bstart,&one);
  return(0);
}

PetscErrorCode VecSet_SeqPThread(Vec,PetscScalar);

#undef __FUNCT__
#define __FUNCT__ "VecScale_SeqPThread"
PetscErrorCode VecScale_SeqPThread(Vec xin, PetscScalar alpha)
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=xin->map->tmap;

  PetscFunctionBegin;

  if (alpha == 0.0) {
    ierr = VecSet_SeqPThread(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    PetscScalar *xa;
    PetscInt    i;

    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    for (i=0; i< tmap->nthreads; i++) {
      vec_kerneldatap[i].X     = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x     = xa;
      vec_kerneldatap[i].alpha = alpha;
      vec_pdata[i]             = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecScale_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode VecAXPY_Kernel(void *arg)
{
  PetscErrorCode    ierr;
  Vec_KernelData    *data = (Vec_KernelData*)arg;
  Vec               X=data->X;
  PetscInt          thread_id=data->thread_id;
  PetscScalar       a,*y;
  const PetscScalar *x;
  PetscBLASInt      one = 1, bn,bstart;
  PetscInt          n,start,end;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;
  n = end-start;
  bn = PetscBLASIntCast(n);
  bstart = PetscBLASIntCast(start);
  BLASaxpy_(&bn,&a,x+bstart,&one,y+bstart,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAXPY_SeqPThread"
PetscErrorCode VecAXPY_SeqPThread(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=yin->map->tmap;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (alpha != 0.0) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X = yin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x = xa;
      vec_kerneldatap[i].y = ya;
      vec_kerneldatap[i].alpha = alpha;
      vec_pdata[i] = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecAXPY_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAYPX_Kernel(void *arg)
{
  PetscErrorCode    ierr;
  Vec_KernelData    *data = (Vec_KernelData*)arg;
  Vec               X=data->X;
  PetscInt          thread_id=data->thread_id;
  PetscScalar       a,*y;
  const PetscScalar *x;
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
  PetscInt          n;
#endif
  PetscInt          i,start,end;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;

#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
  n = end-start;
  fortranaypx_(&n,&a,x+start,y+start);
#else
  if(a==-1.0) {
    for (i=start; i<end; i++) {
      y[i] = x[i] - y[i];
    }
  }
  else {
    for (i=start; i<end; i++) {
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
  PetscThreadsLayout tmap=yin->map->tmap;
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
    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X     = yin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x     = xa;
      vec_kerneldatap[i].y     = ya;
      vec_kerneldatap[i].alpha = alpha;
      vec_pdata[i]             = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecAYPX_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
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

PetscErrorCode VecAX_Kernel(void *arg)
{
  PetscErrorCode     ierr;
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscScalar        a,*y;
  const PetscScalar *x;
  PetscInt           i,start,end;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;
  for(i=start;i < end; i++) y[i] = a*x[i];
  return(0);
}

PetscErrorCode VecAXPBY_Kernel(void *arg)
{
  PetscErrorCode     ierr;
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscScalar        a,b,*y;
  const PetscScalar *x;
  PetscInt           i,start,end;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  y = data->y;
  a = data->alpha;
  b = data->beta;
  for(i=start;i < end; i++) y[i] = a*x[i] + b*y[i];
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_SeqPThread"
PetscErrorCode VecAXPBY_SeqPThread(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=yin->map->tmap;
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
    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X = yin;
      vec_kerneldatap[i].thread_id=i;
      vec_kerneldatap[i].x = xa;
      vec_kerneldatap[i].y = ya;
      vec_kerneldatap[i].alpha = alpha;
      vec_pdata[i] = &vec_kerneldatap[i];
    }
    
    ierr = PetscThreadsRunKernel(VecAX_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
      
  } else {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X = yin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x = xa;
      vec_kerneldatap[i].y = ya;
      vec_kerneldatap[i].alpha = alpha;
      vec_kerneldatap[i].beta = beta;
      vec_pdata[i] = &vec_kerneldatap[i];
    }
    
    ierr = PetscThreadsRunKernel(VecAXPBY_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
    
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_Kernel(void *arg)
{
  Vec_KernelData    *data = (Vec_KernelData*)arg;
  Vec               X=data->X;
  PetscInt          thread_id=data->thread_id;
  PetscScalar       a,*ww;
  const PetscScalar *xx,*yy;
  PetscInt          n;
  PetscInt          i,start,end;
  PetscErrorCode    ierr;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  ww = data->w;
  xx = (const PetscScalar*)data->x;
  yy = (const PetscScalar*)data->y;
  a = data->alpha;
  n = end-start;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
  fortranwaxpy_(&n,&a,xx,yy,ww);
#else
  if (a == 0.0) {
    ierr = PetscMemcpy(ww+start,yy+start,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  else if(a==-1.0) {
    for (i=start; i<end; i++) {
      ww[i] = yy[i] - xx[i];
    }
  }
  else if(a==1.0) {
    for (i=start; i<end; i++) {
      ww[i] = yy[i] + xx[i];
    }
  }
  else {
    for (i=start; i<end; i++) {
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
  PetscThreadsLayout tmap=win->map->tmap;
  PetscScalar       *ya,*xa,*wa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecGetArray(win,&wa);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X = win;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].x = xa;
    vec_kerneldatap[i].y = ya;
    vec_kerneldatap[i].w = wa;
    vec_kerneldatap[i].alpha = alpha;
    vec_pdata[i] = &vec_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(VecWAXPY_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

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

PetscErrorCode VecNorm_Kernel(void *arg)
{
  Vec_KernelData *data = (Vec_KernelData*)arg;
  PetscErrorCode ierr;
  Vec            X=data->X;
  PetscInt       thread_id=data->thread_id;
  const PetscScalar *x;
  NormType type;
  PetscInt    i,n,start,end;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  x = (const PetscScalar*)data->x;
  type = data->typeUse;
  n = end-start;
  data->result = 0.0;
  if(type==NORM_1) {
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n),bstart=PetscBLASIntCast(start);
    data->result = BLASasum_(&bn,x+bstart,&one);
  }
  else if(type==NORM_INFINITY) {
    PetscReal    maxv = 0.0,tmp;
    for(i=start; i<end; i++) {
      tmp = PetscAbsScalar(x[i]);
      if(tmp>maxv) {
        maxv = tmp;
      }
    }
    data->result = maxv;
  } else {
    PetscBLASInt one = 1, bn = PetscBLASIntCast(n),bstart=PetscBLASIntCast(start);
    data->result = BLASdot_(&bn,x+bstart,&one,x+bstart,&one);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNorm_SeqPThread"
PetscErrorCode VecNorm_SeqPThread(Vec xin,NormType type,PetscReal* z)
{

  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscScalar       *xa;

  PetscFunctionBegin;

  if(type == NORM_1_AND_2) {
    ierr = VecNorm_SeqPThread(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqPThread(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  else {
    PetscInt i;

    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x = xa;
      vec_kerneldatap[i].typeUse = type;
      vec_pdata[i] = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecNorm_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    /* collect results */
    *z = (PetscReal)vec_kerneldatap[0].result;
    if(type == NORM_1) {
      for(i=1; i<tmap->nthreads; i++) {
        *z += (PetscReal)vec_kerneldatap[i].result;
      }
      ierr = PetscLogFlops(PetscMax(xin->map->n-1.0+tmap->nthreads-1,0.0));CHKERRQ(ierr);
    }
    else if(type == NORM_2 || type == NORM_FROBENIUS) {
      *z = (PetscReal)vec_kerneldatap[0].result;
      for(i=1; i<tmap->nthreads; i++) {
        *z += (PetscReal)vec_kerneldatap[i].result;
      }
      *z = PetscSqrtReal(*z);
      ierr = PetscLogFlops(PetscMax(2.0*xin->map->n-1+tmap->nthreads-1,0.0));CHKERRQ(ierr);
    }
    else {
      PetscReal    maxv = 0.0,tmp;
      for(i=0; i<tmap->nthreads; i++) {
        tmp = (PetscReal)vec_kerneldatap[i].result;
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

PetscErrorCode VecMDot_Kernel4(void* arg)
{
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;
  const PetscScalar  *y1 = (const PetscScalar*)data->y1;
  const PetscScalar  *y2 = (const PetscScalar*)data->y2;
  const PetscScalar  *y3 = (const PetscScalar*)data->y3;
  PetscInt           i;
  PetscScalar        sum0,sum1,sum2,sum3;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);

  sum0 = sum1 = sum2 = sum3 = 0.0;
  for(i=start;i<end;i++) {
    sum0 += (x[i])*PetscConj(y0[i]);
    sum1 += (x[i])*PetscConj(y1[i]);
    sum2 += (x[i])*PetscConj(y2[i]);
    sum3 += (x[i])*PetscConj(y3[i]);
  }
  data->result0 = sum0; data->result1 = sum1; data->result2 = sum2; data->result3 = sum3;
  return(0);
}

PetscErrorCode VecMDot_Kernel3(void* arg)
{
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;
  const PetscScalar  *y1 = (const PetscScalar*)data->y1;
  const PetscScalar  *y2 = (const PetscScalar*)data->y2;
  PetscInt           i;
  PetscScalar        sum0,sum1,sum2;
  
  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  sum0 = sum1 = sum2 = 0.0;
  for(i=start;i<end;i++) {
    sum0 += (x[i])*PetscConj(y0[i]);
    sum1 += (x[i])*PetscConj(y1[i]);
    sum2 += (x[i])*PetscConj(y2[i]);
  }
  data->result0 = sum0; data->result1 = sum1; data->result2 = sum2;
  return(0);
}

PetscErrorCode VecMDot_Kernel2(void* arg)
{
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  const PetscScalar  *x = (const PetscScalar*)data->x;
  const PetscScalar  *y0 = (const PetscScalar*)data->y0;
  const PetscScalar  *y1 = (const PetscScalar*)data->y1;
  PetscInt           i;
  PetscScalar        sum0,sum1;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  sum0 = sum1 = 0.0;
  for(i=start;i<end;i++) {
    sum0 += (x[i])*PetscConj(y0[i]);
    sum1 += (x[i])*PetscConj(y1[i]);
  }
  data->result0 = sum0; data->result1 = sum1;
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqPThread"
PetscErrorCode VecMDot_SeqPThread(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  Vec                *yy = (Vec*)yin;
  PetscScalar        *xa,*y0,*y1,*y2,*y3;
  PetscInt           i,j,j_rem;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  switch(j_rem = nv&0x3) {
  case 3:
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecGetArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecGetArray(yy[2],&y2);CHKERRQ(ierr);

    for(i=0;i<tmap->nthreads;i++) {
      vec_kerneldatap[i].X      = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x      = xa;
      vec_kerneldatap[i].y0     = y0;
      vec_kerneldatap[i].y1     = y1;
      vec_kerneldatap[i].y2     = y2;
      vec_pdata[i]              = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecMDot_Kernel3,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[2],&y2);CHKERRQ(ierr);

    z[0] = vec_kerneldatap[0].result0;
    for(j=1;j<tmap->nthreads;j++) {
      z[0] += vec_kerneldatap[j].result0;
    }
    z[1] = vec_kerneldatap[0].result1;
    for(j=1;j<tmap->nthreads;j++) {
      z[1] += vec_kerneldatap[j].result1;
    }
    z[2] = vec_kerneldatap[0].result2;
    for(j=1;j<tmap->nthreads;j++) {
      z[2] += vec_kerneldatap[j].result2;
    }
    yy += 3;
    z  += 3;
    break;
  case 2:
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecGetArray(yy[1],&y1);CHKERRQ(ierr);

    for(i=0;i<tmap->nthreads;i++) {
      vec_kerneldatap[i].X      = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x      = xa;
      vec_kerneldatap[i].y0     = y0;
      vec_kerneldatap[i].y1     = y1;
      vec_pdata[i]              = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecMDot_Kernel2,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[1],&y1);CHKERRQ(ierr);

    z[0] = vec_kerneldatap[0].result0;
    for(j=1;j<tmap->nthreads;j++) {
      z[0] += vec_kerneldatap[j].result0;
    }
    z[1] = vec_kerneldatap[0].result1;
    for(j=1;j<tmap->nthreads;j++) {
      z[1] += vec_kerneldatap[j].result1;
    }
    yy += 2; z += 2;
    break;
  case 1:
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);

    for(i=0;i<tmap->nthreads;i++) {
      vec_kerneldatap[i].X    = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x    = xa;
      vec_kerneldatap[i].y    = y0;
      vec_pdata[i]            = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecDot_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    
    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);

    z[0] = vec_kerneldatap[0].result;
    for(j=1;j<tmap->nthreads;j++) {
      z[0] += vec_kerneldatap[j].result;
    }
    yy++; z++;
    break;
  }
  for(j=j_rem;j<nv;j+=4) {
    ierr = VecGetArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecGetArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecGetArray(yy[2],&y2);CHKERRQ(ierr);
    ierr = VecGetArray(yy[3],&y3);CHKERRQ(ierr);

    for(i=0;i<tmap->nthreads;i++) {
      vec_kerneldatap[i].X      = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x      = xa;
      vec_kerneldatap[i].y0     = y0;
      vec_kerneldatap[i].y1     = y1;
      vec_kerneldatap[i].y2     = y2;
      vec_kerneldatap[i].y3     = y3;
      vec_pdata[i]              = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecMDot_Kernel4,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

    ierr = VecRestoreArray(yy[0],&y0);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[1],&y1);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[2],&y2);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy[3],&y3);CHKERRQ(ierr);

    z[0] = vec_kerneldatap[0].result0;
    for(i=1;i<tmap->nthreads;i++) {
      z[0] += vec_kerneldatap[i].result0;
    }
    z[1] = vec_kerneldatap[0].result1;
    for(i=1;i<tmap->nthreads;i++) {
      z[1] += vec_kerneldatap[i].result1;
    }
    z[2] = vec_kerneldatap[0].result2;
    for(i=1;i<tmap->nthreads;i++) {
      z[2] += vec_kerneldatap[i].result2;
    }
    z[3] = vec_kerneldatap[0].result3;
    for(i=1;i<tmap->nthreads;i++) {
      z[3] += vec_kerneldatap[i].result3;
    }
    yy += 4;
    z  += 4;
  }    
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);

  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1+tmap->nthreads-1),0.0));CHKERRQ(ierr);
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


PetscErrorCode VecMax_Kernel(void *arg)
{
  Vec_KernelData *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  const PetscScalar *xx = (const PetscScalar*)data->x;
  PetscInt          i,j;
  PetscReal         lmax,tmp;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  lmax = PetscRealPart(xx[start]); j = 0;
#else
  lmax = xx[start]; j = 0;
#endif
  for (i=start+1; i<end; i++) {
#if defined(PETSC_USE_COMPLEX)
    if ((tmp = PetscRealPart(xx[i])) > lmax) { j = i; lmax = tmp;}
#else
    if ((tmp = xx[i]) > lmax) { j = i; lmax = tmp; }
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
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscInt          i,j=0;
  PetscScalar       *xa;
  PetscReal         max;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  if (!xin->map->n) {
    max = PETSC_MIN_REAL;
    j   = -1;
  } else {
    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X    = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x    = xa;
      vec_pdata[i]            = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecMax_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    /* collect results, determine global max, global index */
    max = vec_kerneldatap[0].localmax;
    j   = vec_kerneldatap[0].localind;
    for(i=1; i<tmap->nthreads; i++) {
      if(vec_kerneldatap[i].localmax > max) {
        max = vec_kerneldatap[i].localmax;
        j   = vec_kerneldatap[i].localind;
      }
    }
  }
  *z   = max;
  if (idx) *idx = j;
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_Kernel(void *arg)
{
  Vec_KernelData *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  const PetscScalar *xx = (const PetscScalar*)data->x;
  PetscInt          i,j;
  PetscReal         lmin,tmp;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  lmin = PetscRealPart(xx[start]); j = 0;
#else
  lmin = xx[start]; j = 0;
#endif
  for (i=start+1; i<end; i++) {
#if defined(PETSC_USE_COMPLEX)
    if ((tmp = PetscRealPart(xx[i])) < lmin) { j = i; lmin = tmp;}
#else
    if ((tmp = xx[i]) < lmin) { j = i; lmin = tmp; }
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
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscInt          i,j=0;
  PetscScalar       *xa;
  PetscReal         min;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  if (!xin->map->n) {
    min = PETSC_MAX_REAL;
    j   = -1;
  } else {
    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X    = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x    = xa;
      vec_pdata[i]            = &vec_kerneldatap[i];
    }

    ierr = PetscThreadsRunKernel(VecMin_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    /* collect results, determine global max, global index */
    min = vec_kerneldatap[0].localmin;
    j   = vec_kerneldatap[0].localind;
    for(i=1; i<tmap->nthreads; i++) {
      if(vec_kerneldatap[i].localmin < min) {
        min = vec_kerneldatap[i].localmin;
        j   = vec_kerneldatap[i].localind;
      }
    }
  }
  *z   = min;
  if (idx) *idx = j;
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>
PetscErrorCode VecPointwiseMult_Kernel(void *arg)
{
  Vec_KernelData *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
  PetscInt n;
#endif
  PetscScalar *ww = data->w,*xx = data->x,*yy = data->y;
  PetscInt    i;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  if (ww == xx) {
    for (i=start; i<end; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=start; i<end; i++) ww[i] *= xx[i];
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    n = end-start;
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=start; i<end; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  return(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_SeqPThread"
PetscErrorCode VecPointwiseMult_SeqPThread(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=win->map->tmap;
  PetscScalar       *ya,*xa,*wa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecGetArray(win,&wa);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X = win;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].w = wa;
    vec_kerneldatap[i].x = xa;
    vec_kerneldatap[i].y = ya;
    vec_pdata[i]         = &vec_kerneldatap[i];
  }

  ierr  = PetscThreadsRunKernel(VecPointwiseMult_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&wa);CHKERRQ(ierr);
  ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_Kernel(void *arg)
{
  Vec_KernelData *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  PetscScalar *ww = data->w,*xx = data->x,*yy = data->y;
  PetscInt    i;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    ww[i] = xx[i] / yy[i];
  }
  return(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_SeqPThread"
PetscErrorCode VecPointwiseDivide_SeqPThread(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=win->map->tmap;
  PetscScalar       *ya,*xa,*wa;
  PetscInt          i;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecGetArray(win,&wa);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X = win;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].w = wa;
    vec_kerneldatap[i].x = xa;
    vec_kerneldatap[i].y = ya;
    vec_pdata[i]         = &vec_kerneldatap[i];
  }

  ierr  = PetscThreadsRunKernel(VecPointwiseDivide_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&wa);CHKERRQ(ierr);
  ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscblaslapack.h>
PetscErrorCode VecSwap_Kernel(void *arg)
{
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           n,start,end;
  PetscScalar        *xa = data->x,*ya = data->y;
  PetscBLASInt       one = 1,bn,bstart;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  n = end-start;
  bn = PetscBLASIntCast(n);
  bstart = PetscBLASIntCast(start);
  BLASswap_(&bn,xa+bstart,&one,ya+bstart,&one);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSwap_SeqPThread"
PetscErrorCode VecSwap_SeqPThread(Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X = xin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x = xa;
      vec_kerneldatap[i].y = ya;
      vec_pdata[i]         = &vec_kerneldatap[i];
    }

    ierr = PetscThreadsRunKernel(VecSwap_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_Kernel(void *arg)
{
  Vec_KernelData *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  PetscScalar  *xx = data->x;
  PetscRandom  r = data->rand;
  PetscInt     i;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  for(i=start; i<end; i++) {
    ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_SeqPThread"
PetscErrorCode VecSetRandom_SeqPThread(Vec xin,PetscRandom r)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscInt          i;
  PetscScalar       *xa;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X = xin;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].x    = xa;
    vec_kerneldatap[i].rand = r;
    vec_pdata[i]            = &vec_kerneldatap[i];
   }

  ierr = PetscThreadsRunKernel(VecSetRandom_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_Kernel(void *arg)
{
  Vec_KernelData     *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  const PetscScalar  *xa = (const PetscScalar*)data->x;
  PetscScalar        *ya = data->y;
  PetscInt           n;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  n = end-start;
  ierr = PetscMemcpy(ya+start,xa+start,n*sizeof(PetscScalar));CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqPThread"
PetscErrorCode VecCopy_SeqPThread(Vec xin,Vec yin)
{

  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=yin->map->tmap;
  PetscScalar       *ya,*xa;
  PetscInt          i;

  PetscFunctionBegin;

  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);

    for (i=0; i<tmap->nthreads; i++) {
      vec_kerneldatap[i].X = yin;
      vec_kerneldatap[i].thread_id = i;
      vec_kerneldatap[i].x   = xa;
      vec_kerneldatap[i].y   = ya;
      vec_pdata[i]           = &vec_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(VecCopy_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_Kernel(void* arg)
{
  Vec_KernelData       *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  PetscInt           n,nv=data->nvec,j,j_rem;
  const PetscScalar *alpha=data->amult,*yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx,alpha0,alpha1,alpha2,alpha3;
  Vec*              y = (Vec*)data->yvec;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  xx = data->x+start;
  n = end-start;
  switch (j_rem=nv&0x3) {
  case 3: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    yy0 += start; yy1 += start; yy2 += start;
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha2 = alpha[2]; 
    alpha += 3;
    PetscAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    yy0 += start; yy1 += start;
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha +=2;
    PetscAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    y     +=2;
    break;
  case 1: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    yy0 += start; yy1 += start;
    alpha0 = *alpha++; 
    PetscAXPY(xx,alpha0,yy0,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
    yy0 += start; yy1 += start; yy2 += start; yy3 += start;
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;

    PetscAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  return(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_SeqPThread"
PetscErrorCode VecMAXPY_SeqPThread(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *yin)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscInt          i;
  Vec               *yy = (Vec *)yin;
  PetscScalar       *xa;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X = xin;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].x      = xa;
    vec_kerneldatap[i].yvec   = yy;
    vec_kerneldatap[i].amult  = &alpha[0];
    vec_kerneldatap[i].nvec   = nv;
    vec_pdata[i]              = &vec_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(VecMAXPY_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);

  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  ierr = PetscLogFlops(nv*2.0*xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSet_Kernel(void *arg)
{
  Vec_KernelData    *data = (Vec_KernelData*)arg;
  PetscErrorCode     ierr;
  Vec                X=data->X;
  PetscInt           thread_id=data->thread_id;
  PetscInt           start,end;
  PetscScalar        *xx = data->x;
  PetscScalar        alpha = data->alpha;
  PetscInt           i,n;

  ierr = VecGetThreadOwnershipRange(X,thread_id,&start,&end);CHKERRQ(ierr);
  n = end-start;
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemzero(xx+start,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    for (i=start; i<end; i++) xx[i] = alpha;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSet_SeqPThread"
PetscErrorCode VecSet_SeqPThread(Vec xin,PetscScalar alpha)
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=xin->map->tmap;
  PetscInt          i;
  PetscScalar       *xa;

  PetscFunctionBegin;

  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);

  for (i=0; i<tmap->nthreads; i++) {
    vec_kerneldatap[i].X = xin;
    vec_kerneldatap[i].thread_id = i;
    vec_kerneldatap[i].x       = xa;
    vec_kerneldatap[i].alpha   = alpha;
    vec_pdata[i]               = &vec_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(VecSet_Kernel,(void**)vec_pdata,tmap->nthreads,tmap->affinity);
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_SeqPThread"
PetscErrorCode VecDestroy_SeqPThread(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree(v->data);CHKERRQ(ierr);

  vecs_created--;
  /* Free the kernel data structure on the destruction of the last vector */
  if (!vecs_created) {
    ierr = PetscFree(vec_kerneldatap);CHKERRQ(ierr);
    ierr = PetscFree(vec_pdata);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_SeqPThread"
PetscErrorCode VecDuplicate_SeqPThread(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,V);CHKERRQ(ierr);
  ierr = PetscObjectSetPrecision((PetscObject)*V,((PetscObject)win)->precision);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,win->map->n,win->map->n);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = VecSetType(*V,((PetscObject)win)->type_name);CHKERRQ(ierr);
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetNThreads"
/*@
   VecSetNThreads - Set the number of threads to be used for vector operations.

   Input Parameters
+  v - the vector
-  nthreads - number of threads

   Note:
   Use nthreads = PETSC_DECIDE for PETSc to determine the number of threads.

   Options Database keys:
   -vec_threads <nthreads> - Number of threads

   Level: intermediate

   Concepts: vectors^number of threads

.seealso: VecCreateSeqPThread(), VecGetNThreads()
@*/
PetscErrorCode VecSetNThreads(Vec v,PetscInt nthreads)
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=v->map->tmap;
  PetscInt           nworkThreads=PetscMaxThreads+PetscMainThreadShareWork;

  PetscFunctionBegin;

  if(!tmap) {
    ierr = PetscThreadsLayoutCreate(&tmap);CHKERRQ(ierr);
    v->map->tmap = tmap;
  }

  if(nthreads == PETSC_DECIDE) {
    tmap->nthreads = nworkThreads;
    ierr = PetscOptionsInt("-vec_threads","Set number of threads to be used for vector operations","VecSetNThreads",nworkThreads,&tmap->nthreads,PETSC_NULL);CHKERRQ(ierr);
    if(tmap->nthreads > nworkThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Vec x: threads requested %D, Max. threads initialized %D",tmap->nthreads,nworkThreads);
    }
  } else {
    if(nthreads > nworkThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Vec x: threads requested %D, Max. threads initialized %D",nthreads,nworkThreads);
    }
    tmap->nthreads = nthreads;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetNThreads"
/*@
   VecGetNThreads - Returns the number of threads used for vector operations.

   Input Parameter
.  v - the vector

   Output Parameter
.  nthreads - number of threads

   Level: intermediate

   Concepts: vectors^number of threads

.seealso: VecSetNThreads()
@*/
PetscErrorCode VecGetNThreads(Vec v,PetscInt *nthreads)
{
  PetscThreadsLayout tmap=v->map->tmap;
  PetscFunctionBegin;
  *nthreads = tmap->nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetThreadAffinities"
/*@
   VecSetThreadAffinities - Sets the CPU affinities of vector threads.

   Input Parameters
+  v - the vector
-  affinities - list of cpu affinities for threads.

   Notes:
   Must set affinities for all the threads used with the vector (not including the main thread)
 
   Use affinities[] = PETSC_NULL for PETSc to decide the thread affinities.

   Options Database Keys:
   -vec_thread_affinities - Comma seperated list of thread affinities

   Level: intermediate

   Concepts: vectors^thread cpu affinity

.seealso: VecGetThreadAffinities()
@*/
PetscErrorCode VecSetThreadAffinities(Vec v,const PetscInt affinities[])
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap = v->map->tmap;
  PetscInt           nmax=PetscMaxThreads+PetscMainThreadShareWork;
  PetscBool          flg;

  PetscFunctionBegin;

  if(!tmap) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must set the number of threads before setting thread affinities");
  }

  ierr = PetscMalloc(tmap->nthreads*sizeof(PetscInt),&tmap->affinity);CHKERRQ(ierr);

  if(affinities == PETSC_NULL) {
    /* PETSc decides affinities */
    PetscInt        *thread_affinities;
    ierr = PetscMalloc(nmax*sizeof(PetscInt),&thread_affinities);CHKERRQ(ierr);
    /* Check if run-time option is set */
    ierr = PetscOptionsIntArray("-vec_thread_affinities","Set CPU affinities of vector threads","VecSetThreadAffinities",thread_affinities,&nmax,&flg);CHKERRQ(ierr);
    if(flg) {
      if(nmax != tmap->nthreads) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, vector Threads = %D, CPU affinities set = %D",tmap->nthreads,nmax);
      }
      ierr = PetscMemcpy(tmap->affinity,thread_affinities,tmap->nthreads*sizeof(PetscInt));
    } else {
      /* Reuse the core affinities set for the first nthreads */
      ierr = PetscMemcpy(tmap->affinity,PetscThreadsCoreAffinities,tmap->nthreads*sizeof(PetscInt));
    }
    ierr = PetscFree(thread_affinities);CHKERRQ(ierr);
  } else {
    /* Set user provided affinities */
    ierr = PetscMemcpy(tmap->affinity,affinities,tmap->nthreads*sizeof(PetscInt));
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_SeqPthread"
PetscErrorCode VecView_SeqPthread(Vec xin,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = VecView_Seq(xin,viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Number threads used=%D\n",xin->map->tmap->nthreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
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
            VecSwap_SeqPThread,
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
            VecView_SeqPthread,
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
  Vec_Seq            *s;
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=v->map->tmap;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,Vec_Seq,&s);CHKERRQ(ierr);
  ierr = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  v->data            = (void*)s;
  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar *)array;
  s->array_allocated = 0;

  if(!v->map->tmap) {
    ierr = PetscThreadsLayoutCreate(&v->map->tmap);CHKERRQ(ierr);
    tmap = v->map->tmap;
  }

  /* If this is the first vector being created then also create the common Kernel data structure */
  if(vecs_created == 0) {
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Vec_KernelData),&vec_kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Vec_KernelData*),&vec_pdata);CHKERRQ(ierr);
  }
  vecs_created++;

  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);

  tmap->N = v->map->n;
 
 /* Set the number of threads */
  if(tmap->nthreads == PETSC_DECIDE) {
    ierr = VecSetNThreads(v,PETSC_DECIDE);CHKERRQ(ierr);
  }
  /* Set thread affinities */
  if(!tmap->affinity) {
    ierr = VecSetThreadAffinities(v,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscThreadsLayoutSetUp(tmap);CHKERRQ(ierr);

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
  Vec_Seq         *s;
  PetscScalar     *array;
  PetscErrorCode  ierr;
  PetscInt        n = PetscMax(V->map->n,V->map->N);
  PetscMPIInt     size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQPTHREAD on more than one process");
  ierr = PetscThreadsInitialize(PetscMaxThreads);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_SeqPThread_Private(V,array);CHKERRQ(ierr);
  ierr = VecSet_SeqPThread(V,0.0);CHKERRQ(ierr);
  s    = (Vec_Seq*)V->data;
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

.seealso: VecCreateSeq(), VecSetNThreads(), VecSetThreadAffinities(), VecDuplicate(), VecDuplicateVecs()
@*/
PetscErrorCode VecCreateSeqPThread(MPI_Comm comm,PetscInt n,PetscInt nthreads,PetscInt affinities[],Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetNThreads(*v,nthreads);CHKERRQ(ierr);
  ierr = VecSetThreadAffinities(*v,affinities);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQPTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
