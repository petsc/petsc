
/*$Id: ebvec1.c,v 1.1 2001/08/29 20:54:49 bsmith Exp bsmith $*/


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
  delete ytmp;
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
    delete ytmp;
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
#define __FUNCT__ "VecDestroy_ESI"
int VecDestroy_ESI(Vec v)
{
  Vec_ESI *vs = (Vec_ESI*)v->data;
  int     ierr;

  PetscFunctionBegin;
  if (vs->evec) {
    vs->evec->deleteReference();
  }
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_PetscESI"
int VecCreate_PetscESI(Vec V)
{
  int          ierr;
  Vec          v;

  PetscFunctionBegin;
  ierr = VecSetType(V,VEC_ESI);CHKERRQ(ierr);
  ierr = VecCreateMPI(V->comm,V->n,V->N,&v);CHKERRQ(ierr);
  ierr = VecESISetVector(V,new esi::petsc::Vector<double,int>(v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

static struct _VecOps EvOps = {VecDuplicate_ESI,
			       VecDuplicateVecs_Default,
			       VecDestroyVecs_Default,
			       VecDot_ESI,
			       VecMDot_ESI,
			       0,
			       0,
			       0,
			       0,
			       0,
			       VecSet_ESI,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       VecGetSize_ESI,
			       VecGetLocalSize_ESI,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
			       0,
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
