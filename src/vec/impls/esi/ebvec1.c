
/*$Id: ebvec1.c,v 1.3 2001/09/07 20:09:03 bsmith Exp bsmith $*/


#include "src/vec/vecimpl.h" 
#include "esi/ESI.h"
#include "esi/petsc/vector.h"

typedef struct { 
  esi::Vector<double,int> *evec;
} Vec_ESI;

#undef __FUNCT__  
#define __FUNCT__ "VecESISetVector"
int VecESISetVector(Vec xin,esi::Vector<double,int> *v)
{
  Vec_ESI    *x;
  PetscTruth tesi;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)xin,0,&tesi);CHKERRQ(ierr);
  if (tesi) {
    ierr = VecSetType(xin,VEC_ESI);CHKERRQ(ierr);
  }
  ierr = PetscTypeCompare((PetscObject)xin,VEC_ESI,&tesi);CHKERRQ(ierr);
  if (tesi) {
    int                    n,N;
    esi::MapPartition<int> *map;

    ierr = v->getGlobalSize(N);CHKERRQ(ierr);
    if (xin->N == -1) xin->N = N;
    else if (xin->N != N) SETERRQ2(1,"Global size of Vec %d not equal size of esi::Vector %d",xin->N,N);

    ierr = v->getMapPartition(map);CHKERRQ(ierr); 
    ierr = map->getLocalSize(n);CHKERRQ(ierr);
    if (xin->n == -1) xin->n = n;
    else if (xin->n != n) SETERRQ2(1,"Local size of Vec %d not equal size of esi::Vector %d",xin->n,n);

    x       = (Vec_ESI*)xin->data;
    x->evec = v;
    v->addReference();
    ierr = PetscMapCreateMPI(xin->comm,n,N,&xin->map);CHKERRQ(ierr);
    ierr = VecStashCreate_Private(xin->comm,1,&xin->stash);CHKERRQ(ierr);
    ierr = VecStashCreate_Private(xin->comm,1,&xin->bstash);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecSet_ESI"
int VecSet_ESI(const PetscScalar *alpha,Vec xin)
{
  Vec_ESI *x = (Vec_ESI*)xin->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = x->evec->put(*alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_ESI"
int VecDuplicate_ESI(Vec xin,Vec *xout)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *nevec;

  PetscFunctionBegin;
  ierr = VecCreate(xin->comm,xin->n,xin->N,xout);CHKERRQ(ierr);
  ierr = x->evec->clone(nevec);
  ierr = VecESISetVector(*xout,nevec);CHKERRQ(ierr);
  ierr = nevec->deleteReference();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDot_ESI"
int VecDot_ESI(Vec xin,Vec yin,PetscScalar *z)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ytmp = new esi::petsc::Vector<double,int>(yin);
  ierr = x->evec->dot(*ytmp,*z);CHKERRQ(ierr);
  ytmp->deleteReference();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_ESI"
int VecAXPY_ESI(const PetscScalar *a,Vec xin,Vec yin)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ytmp = new esi::petsc::Vector<double,int>(yin);
  ierr = ytmp->axpy(*x->evec,*a);CHKERRQ(ierr);
  ytmp->deleteReference();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_ESI"
int VecAYPX_ESI(const PetscScalar *a,Vec xin,Vec yin)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ytmp = new esi::petsc::Vector<double,int>(yin);
  ierr = x->evec->aypx(*a,*ytmp);CHKERRQ(ierr);
  ytmp->deleteReference();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY_ESI"
int VecWAXPY_ESI(const PetscScalar *a,Vec xin,Vec yin,Vec win)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *ytmp,*wtmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ytmp = new esi::petsc::Vector<double,int>(yin);
  wtmp = new esi::petsc::Vector<double,int>(win);
  ierr = x->evec->axpby(*a,*ytmp,1.0,*wtmp);CHKERRQ(ierr);
  ytmp->deleteReference();
  wtmp->deleteReference();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_ESI"
int VecCopy_ESI(Vec xin,Vec yin)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  if (xin != yin) {
    /* Make yin look like an esi:Vector */
    ytmp = new esi::petsc::Vector<double,int>(yin);
    ierr = x->evec->copy(*ytmp);CHKERRQ(ierr);
    ytmp->deleteReference();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_ESI"
int VecPointwiseMult_ESI(Vec xin,Vec yin,Vec zin)
{
  Vec_ESI                 *x = (Vec_ESI*)xin->data;
  int                     ierr;
  esi::Vector<double,int> *ztmp;

  PetscFunctionBegin;
  if (zin != yin) {
    ierr = VecCopy(yin,zin);CHKERRQ(ierr);
  }

  /* Make zin look like an esi:Vector */
  ztmp = new esi::petsc::Vector<double,int>(zin);
  ierr = ztmp->scaleDiagonal(*x->evec);CHKERRQ(ierr);
  ztmp->deleteReference();

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_ESI"
int VecPointwiseDivide_ESI(Vec xin,Vec yin,Vec win)
{
  int          n = xin->n,i,ierr;
  PetscScalar  *xx,*yy,*ww;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(yin,&yy);CHKERRQ(ierr);
  if (yin != xin) {ierr = VecGetArrayFast(xin,&xx);CHKERRQ(ierr);}
  else xx = yy;
  if (yin != win) {ierr = VecGetArrayFast(win,&ww);CHKERRQ(ierr);}
  else ww = yy;
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
  ierr = VecRestoreArrayFast(yin,&yy);CHKERRQ(ierr);
  if (yin != win) {ierr = VecRestoreArrayFast(win,&ww);CHKERRQ(ierr);}
  if (xin != win) {ierr = VecRestoreArrayFast(xin,&xx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
/*
    ESI does not provide a method for this 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecSwap_ESI"
int VecSwap_ESI(Vec xin,Vec yin)
{
  int                     ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    PetscScalar *ya,*xa;
    int         one = 1;

    ierr = VecGetArrayFast(yin,&ya);CHKERRQ(ierr);
    ierr = VecGetArrayFast(xin,&xa);CHKERRQ(ierr);
    BLswap_(&xin->n,xa,&one,ya,&one);
    ierr = VecRestoreArrayFast(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot_ESI"
int VecMDot_ESI(int nv,Vec xin,const Vec yin[],PetscScalar *z)
{
  Vec_ESI                 *x = (Vec_ESI *)xin->data;
  int                     ierr,i;
  esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  for (i=0; i<nv; i++) {
    /* Make yin look like an esi:Vector */
    ytmp = new esi::petsc::Vector<double,int>(yin[i]);
    ierr = x->evec->dot(*ytmp,z[i]);CHKERRQ(ierr);
    ytmp->deleteReference();
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_ESI"
int VecMAXPY_ESI(int nv,const PetscScalar *a,Vec xin,const Vec yin[])
{
  Vec_ESI                 *x = (Vec_ESI *)xin->data;
  int                     ierr,i;
  esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  for (i=0; i<nv; i++) {
    /* Make yin look like an esi:Vector */
    ytmp = new esi::petsc::Vector<double,int>(yin[i]);
    ierr = x->evec->axpy(*ytmp,a[i]);CHKERRQ(ierr);
    ytmp->deleteReference();
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecGetSize_ESI"
int VecGetSize_ESI(Vec vin,int *size)
{
  Vec_ESI                 *x = (Vec_ESI*)vin->data;
  int                     ierr;

  PetscFunctionBegin;
  ierr = x->evec->getGlobalSize(*size);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetLocalSize_ESI"
int VecGetLocalSize_ESI(Vec vin,int *size)
{
  Vec_ESI                *x = (Vec_ESI*)vin->data;
  int                    ierr;
  esi::MapPartition<int> *map;

  PetscFunctionBegin;
  ierr = x->evec->getMapPartition(map);CHKERRQ(ierr); 
  ierr = map->getLocalSize(*size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetArray_ESI"
int VecGetArray_ESI(Vec vin,PetscScalar **array)
{
  Vec_ESI                 *x = (Vec_ESI*)vin->data;
  int                     ierr;

  PetscFunctionBegin;
  ierr = x->evec->getCoefPtrReadWriteLock(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray_ESI"
int VecRestoreArray_ESI(Vec vin,PetscScalar **array)
{
  Vec_ESI                 *x = (Vec_ESI*)vin->data;
  int                     ierr;

  PetscFunctionBegin;
  ierr = x->evec->releaseCoefPtrLock(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale_ESI"
int VecScale_ESI(const PetscScalar *array,Vec vin)
{
  Vec_ESI                 *x = (Vec_ESI*)vin->data;
  int                     ierr;

  PetscFunctionBegin;
  ierr = x->evec->scale(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNorm_ESI"
int VecNorm_ESI(Vec vin,NormType ntype,PetscReal *norm)
{
  Vec_ESI                 *x = (Vec_ESI*)vin->data;
  int                     ierr;

  PetscFunctionBegin;
  if (ntype == NORM_2) {
    ierr = x->evec->norm2(*norm);CHKERRQ(ierr);
  } else if (ntype == NORM_1) {
    ierr = x->evec->norm1(*norm);CHKERRQ(ierr);
  } else if (ntype == NORM_INFINITY) {
    ierr = x->evec->normInfinity(*norm);CHKERRQ(ierr);
  } else SETERRQ1(1,"Unknown NormType %d",ntype);
  PetscFunctionReturn(0);
}

extern int VecSetValues_MPI(Vec,int,const int[],const PetscScalar[],InsertMode);
extern int VecAssemblyBegin_MPI(Vec);
extern int VecAssemblyEnd_MPI(Vec);
extern int VecView_MPI(Vec,PetscViewer);

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_ESI"
int VecDestroy_ESI(Vec v)
{
  Vec_ESI *vs = (Vec_ESI*)v->data;
  int     ierr;

  PetscFunctionBegin;
  if (vs->evec) {
    vs->evec->deleteReference();
  }
  ierr = VecStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  ierr = VecStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_PetscESI"
int VecCreate_PetscESI(Vec V)
{
  int                            ierr;
  Vec                            v;
  esi::petsc::Vector<double,int> *ve;

  PetscFunctionBegin;
  ierr = VecSetType(V,VEC_ESI);CHKERRQ(ierr);
  ierr = VecCreate(V->comm,V->n,V->N,&v);CHKERRQ(ierr);
  if (V->bs > 1) {ierr = VecSetBlockSize(v,V->bs);CHKERRQ(ierr);}
  ierr = VecSetType(v,VEC_MPI);CHKERRQ(ierr);
  ve   = new esi::petsc::Vector<double,int>(v);
  ierr = VecESISetVector(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

static struct _VecOps EvOps = {VecDuplicate_ESI,
			       VecDuplicateVecs_Default,
			       VecDestroyVecs_Default,
			       VecDot_ESI,
			       VecMDot_ESI,
			       VecNorm_ESI,
			       0,
			       0,
			       VecScale_ESI,
			       VecCopy_ESI,
			       VecSet_ESI,
			       VecSwap_ESI,
			       VecAXPY_ESI,
			       0,
			       VecMAXPY_ESI,
			       VecAYPX_ESI,
			       VecWAXPY_ESI,
			       VecPointwiseMult_ESI,
			       VecPointwiseDivide_ESI,
			       VecSetValues_MPI,
			       VecAssemblyBegin_MPI,
			       VecAssemblyEnd_MPI,
			       VecGetArray_ESI,
			       VecGetSize_ESI,
			       VecGetLocalSize_ESI,
			       VecRestoreArray_ESI,
			       0,
			       0,
			       0,
			       0,
			       0,
			       VecDestroy_ESI,
			       VecView_MPI,
			       0,
			       0,
			       0,
			       0,
                               0};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_ESI"
int VecCreate_ESI(Vec V)
{
  Vec_ESI      *s;
  int          ierr;

  PetscFunctionBegin;
  ierr    = PetscNew(Vec_ESI,&s);CHKERRQ(ierr);
  s->evec = 0;
  V->data = (void*)s;
  ierr    = PetscMemcpy(V->ops,&EvOps,sizeof(EvOps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
