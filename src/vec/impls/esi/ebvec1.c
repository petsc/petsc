


/*$Id: ebvec1.c,v 1.9 2001/09/26 17:10:29 balay Exp $*/


#include "src/vec/vecimpl.h"                 /*I "petscvec.h" I*/
#include "src/vec/impls/dvecimpl.h" 
#include "esi/petsc/vector.h"

typedef struct { 
  ::esi::Vector<double,int> *evec;
} Vec_ESI;

/*
    Wraps a PETSc vector to look like an ESI vector and stashes the wrapper inside the
  PETSc vector. If PETSc vector already had wrapper uses that instead.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecESIWrap"
int VecESIWrap(Vec xin,::esi::Vector<double,int> **v)
{
  esi::petsc::Vector<double,int> *t;
  int                             ierr;

  PetscFunctionBegin;
  if (!xin->esivec) {
    t = new esi::petsc::Vector<double,int>(xin);
    ierr = t->getInterface("esi::Vector",xin->esivec);CHKERRQ(ierr);
  }
  *v = reinterpret_cast<esi::Vector<double,int>* >(xin->esivec);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecESISetVector"
/*@C
  VecESISetVector - Takes a PETSc vector sets it to type ESI and 
  provides the ESI vector that it wraps to look like a PETSc vector.

  Level: intermediate

@*/
 int VecESISetVector(Vec xin,::esi::Vector<double,int> *v)
{
  Vec_ESI    *x;
  PetscTruth tesi;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)xin,0,&tesi);CHKERRQ(ierr);
  if (tesi) {
    ierr = VecSetType(xin,VECESI);CHKERRQ(ierr);
  }
  ierr = PetscTypeCompare((PetscObject)xin,VECESI,&tesi);CHKERRQ(ierr);
  if (tesi) {
    int                      n,N;
    ::esi::IndexSpace<int>   *map;

    ierr = v->getGlobalSize(N);CHKERRQ(ierr);
    if (xin->N == -1) xin->N = N;
    else if (xin->N != N) SETERRQ2(1,"Global size of Vec %d not equal size of esi::Vector %d",xin->N,N);

    ierr = v->getIndexSpace(map);CHKERRQ(ierr); 
    ierr = map->getLocalSize(n);CHKERRQ(ierr);
    if (xin->n == -1) xin->n = n;
    else if (xin->n != n) SETERRQ2(1,"Local size of Vec %d not equal size of esi::Vector %d",xin->n,n);

    x       = (Vec_ESI*)xin->data;
    x->evec = v;
    v->addReference();
    if (!xin->map){
      ierr = PetscMapCreateMPI(xin->comm,n,N,&xin->map);CHKERRQ(ierr);
    }
    ierr = VecStashCreate_Private(xin->comm,1,&xin->stash);CHKERRQ(ierr);
    ierr = VecStashCreate_Private(xin->comm,xin->bs,&xin->bstash);CHKERRQ(ierr); 
    ierr = (v)->getInterface("esi::Vector",xin->esivec);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecPlaceArray_ESI"
int VecPlaceArray_ESI(Vec vin,const PetscScalar *a)
{
  Vec_ESI                                *v = (Vec_ESI *)vin->data;
  ::esi::VectorReplaceAccess<double,int> *vr;
  int                                    ierr;

  PetscFunctionBegin;
  ierr = v->evec->getInterface("esi::VectorReplaceAccess",reinterpret_cast<void *&>(vr));CHKERRQ(ierr);
  ierr = vr->setArrayPointer((PetscScalar*)a,vin->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *nevec;

  PetscFunctionBegin;
  ierr = VecCreate(xin->comm,xout);CHKERRQ(ierr);
  ierr = VecSetSizes(*xout,xin->n,xin->N);CHKERRQ(ierr);
  ierr = x->evec->clone(nevec);
  ierr = VecESISetVector(*xout,nevec);CHKERRQ(ierr);
  ierr = nevec->deleteReference();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDot_ESI"
int VecDot_ESI(Vec xin,Vec yin,PetscScalar *z)
{
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ierr = VecESIWrap(yin,&ytmp);CHKERRQ(ierr);
  ierr = x->evec->dot(*ytmp,*z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_ESI"
int VecAXPY_ESI(const PetscScalar *a,Vec xin,Vec yin)
{
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ierr = VecESIWrap(yin,&ytmp);CHKERRQ(ierr);
  ierr = ytmp->axpy(*x->evec,*a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_ESI"
int VecAYPX_ESI(const PetscScalar *a,Vec xin,Vec yin)
{
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ierr = VecESIWrap(yin,&ytmp);CHKERRQ(ierr);
  ierr = ytmp->aypx(*a,*x->evec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY_ESI"
int VecWAXPY_ESI(const PetscScalar *a,Vec xin,Vec yin,Vec win)
{
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *ytmp,*wtmp;

  PetscFunctionBegin;
  /* Make yin look like an esi:Vector */
  ierr = VecESIWrap(yin,&ytmp);CHKERRQ(ierr);
  ierr = VecESIWrap(win,&wtmp);CHKERRQ(ierr);
  ierr = wtmp->axpby(*a,*x->evec,1.0,*ytmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_ESI"
int VecCopy_ESI(Vec xin,Vec yin)
{
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  if (xin != yin) {
    /* Make yin look like an esi:Vector */
    ierr = VecESIWrap(yin,&ytmp);CHKERRQ(ierr);
    ierr = ytmp->copy(*x->evec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_ESI"
int VecPointwiseMult_ESI(Vec xin,Vec yin,Vec zin)
{
  Vec_ESI                   *x = (Vec_ESI*)xin->data;
  int                       ierr;
  ::esi::Vector<double,int> *ztmp;

  PetscFunctionBegin;
  if (zin != yin) {
    ierr = VecCopy(yin,zin);CHKERRQ(ierr);
  }

  /* Make zin look like an esi:Vector */
  ierr = VecESIWrap(zin,&ztmp);CHKERRQ(ierr);
  ierr = ztmp->scaleDiagonal(*x->evec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_ESI"
int VecPointwiseDivide_ESI(Vec xin,Vec yin,Vec win)
{
  int          n = xin->n,i,ierr;
  PetscScalar  *xx,*yy,*ww;

  PetscFunctionBegin;
  ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
  if (yin != xin) {ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);}
  else xx = yy;
  if (yin != win) {ierr = VecGetArray(win,&ww);CHKERRQ(ierr);}
  else ww = yy;
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
  ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
  if (yin != win) {ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);}
  if (xin != win) {ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);}
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
  int ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    PetscScalar *ya,*xa;
    int         one = 1;

    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    BLswap_(&xin->n,xa,&one,ya,&one);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot_ESI"
int VecMDot_ESI(int nv,Vec xin,const Vec yin[],PetscScalar *z)
{
  Vec_ESI                   *x = (Vec_ESI *)xin->data;
  int                       ierr,i;
  ::esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  for (i=0; i<nv; i++) {
    /* Make yin look like an esi:Vector */
    ierr = VecESIWrap(yin[i],&ytmp);CHKERRQ(ierr);
    ierr = x->evec->dot(*ytmp,z[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_ESI"
int VecMAXPY_ESI(int nv,const PetscScalar *a,Vec xin, Vec yin[])
{
  Vec_ESI                   *x = (Vec_ESI *)xin->data;
  int                       ierr,i;
  ::esi::Vector<double,int> *ytmp;

  PetscFunctionBegin;
  for (i=0; i<nv; i++) {
    /* Make yin look like an esi:Vector */
    ierr = VecESIWrap(yin[i],&ytmp);CHKERRQ(ierr);
    ierr = x->evec->axpy(*ytmp,a[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecGetSize_ESI"
int VecGetSize_ESI(Vec vin,int *size)
{
  Vec_ESI *x = (Vec_ESI*)vin->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = x->evec->getGlobalSize(*size);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetLocalSize_ESI"
int VecGetLocalSize_ESI(Vec vin,int *size)
{
  Vec_ESI                  *x = (Vec_ESI*)vin->data;
  int                      ierr;
  ::esi::IndexSpace<int>   *map;

  PetscFunctionBegin;
  ierr = x->evec->getIndexSpace(map);CHKERRQ(ierr); 
  ierr = map->getLocalSize(*size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetArray_ESI"
int VecGetArray_ESI(Vec vin,PetscScalar **array)
{
  Vec_ESI *x = (Vec_ESI*)vin->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = x->evec->getCoefPtrReadWriteLock(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray_ESI"
int VecRestoreArray_ESI(Vec vin,PetscScalar **array)
{
  Vec_ESI *x = (Vec_ESI*)vin->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = x->evec->releaseCoefPtrLock(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale_ESI"
int VecScale_ESI(const PetscScalar *array,Vec vin)
{
  Vec_ESI *x = (Vec_ESI*)vin->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = x->evec->scale(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNorm_ESI"
int VecNorm_ESI(Vec vin,NormType ntype,PetscReal *norm)
{
  Vec_ESI *x = (Vec_ESI*)vin->data;
  int     ierr;

  PetscFunctionBegin;
  if (ntype == NORM_2 || ntype == NORM_FROBENIUS) {
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
extern int VecReciprocal_Default(Vec);
extern int VecSetRandom_Seq(PetscRandom,Vec);
extern int VecSetValuesBlocked_MPI(Vec,int,const int[],const PetscScalar[],InsertMode);
extern int VecMax_MPI(Vec,int*,PetscReal*);
extern int VecMin_MPI(Vec,int*,PetscReal*);

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
  V->ops->destroy = 0;  /* since this is called from VecSetType() we have to make sure it doesn't get destroyed twice */
  ierr = VecSetType(V,VECESI);CHKERRQ(ierr);
  ierr = VecCreate(V->comm,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,V->n,V->N);CHKERRQ(ierr);
  if (V->bs > 1) {ierr = VecSetBlockSize(v,V->bs);CHKERRQ(ierr);}
  ierr = PetscObjectSetOptionsPrefix((PetscObject)v,"esi_");CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
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
			       VecMax_MPI,
			       VecMin_MPI,
			       VecSetRandom_Seq,
			       0,
			       VecSetValuesBlocked_MPI,
			       VecDestroy_ESI,
			       VecView_MPI,
			       VecPlaceArray_ESI,
			       0,
			       VecDot_Seq,
			       VecTDot_Seq,
			       VecNorm_Seq,
                               0,
                               VecReciprocal_Default};

#undef __FUNCT__  
#define __FUNCT__ "VecESISetFromOptions"
int VecESISetFromOptions(Vec V)
{
  char       string[1024];
  PetscTruth flg;
  int        ierr;
 
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)V,VECESI,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsGetString(V->prefix,"-vec_esi_type",string,1024,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecESISetType(V,string);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_ESI"
int VecCreate_ESI(Vec V)
{
  Vec_ESI *s;
  int     ierr;
 
  PetscFunctionBegin;
  ierr    = PetscNew(Vec_ESI,&s);CHKERRQ(ierr);
  ierr    = PetscMemzero(s,sizeof(Vec_ESI));CHKERRQ(ierr);

  s->evec        = 0;
  V->data        = (void*)s;
  V->petscnative = PETSC_FALSE;
  V->esivec      = 0;
  ierr           = PetscMemcpy(V->ops,&EvOps,sizeof(EvOps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

extern PetscFList CCAList;

#undef __FUNCT__  
#define __FUNCT__ "ESICreateIndexSpace"
/*@C
  ESICreateIndexSpace - Creates an esi::IndexSpace using the -is_esi_type type 

  Level: beginner
    
@*/
int ESICreateIndexSpace(const char * commname,void *comm,int m,::esi::IndexSpace<int>*&v)
{
  int                             ierr;
  ::esi::IndexSpace<int>::Factory *f;
  ::esi::IndexSpace<int>::Factory *(*r)(void);
  char                            name[PETSC_MAX_PATH_LEN];
  PetscTruth                      found;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(PETSC_NULL,"-is_esi_type",name,1024,&found);CHKERRQ(ierr);
  if (!found) {
    ierr = PetscStrcpy(name,"esi::petsc::IndexSpace");CHKERRQ(ierr);
  }
  ierr = PetscFListFind(*(MPI_Comm*)comm,CCAList,name,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to load esi::IndexSpace Factory constructor %s",name);
  f    = (*r)();
  ierr = f->create(commname,comm,m,PETSC_DECIDE,PETSC_DECIDE,v);CHKERRQ(ierr);
  delete f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecESISetType"
/*@C
    VecESISetType - Given a PETSc vector of type ESI loads the ESI constructor
          by name and wraps the ESI vector to look like a PETSc vector.

  Level: intermediate
@*/
int VecESISetType(Vec V,const char *name)
{
  int                                ierr;
  ::esi::Vector<double,int>          *ve;
  ::esi::Vector<double,int>::Factory *f,*(*r)(void);
  ::esi::IndexSpace<int>             *map;

  PetscFunctionBegin;
  ierr = PetscFListFind(V->comm,CCAList,name,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to load esi::VectorFactory constructor %s",name);
  f    = (*r)();
  if (V->n == PETSC_DECIDE) {
    ierr = PetscSplitOwnership(V->comm,&V->n,&V->N);CHKERRQ(ierr);
  }
  ierr = ESICreateIndexSpace("MPI",&V->comm,V->n,map);CHKERRQ(ierr);
  ierr = f->create(*map,ve);CHKERRQ(ierr);
  ierr = map->deleteReference();CHKERRQ(ierr);
  delete f;
  ierr = VecESISetVector(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ESILoadFactory"
/*@C
    ESILoadFactory - Creates an object for any ESI factory class. Generally does
          this by dynamicly loading the constructor.

     Collective on MPI_Comm

    Input Parameters:
+     commname - name of parallel context; currently only "MPI" is supported
.     comm - communicator for parallel computing model, currently only MPI_Comm's are supported
-     name - name of class the factory constructs, for example "esi::petsc::Vector"

    Output Parameter:
.     f - the factory object

    Notes: The name of the class is the name of the class the factory CONSTRUCTS, not the name
           of the factory class

  Level: intermediate
@*/
int ESILoadFactory(char *commname,void *comm,char *classname,void *&f)
{
  int           ierr;
  void          *(*r)();
  PetscTruth    flag;

  PetscFunctionBegin;
  ierr = PetscStrcmp(commname,"MPI",&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ1(1,"Parallel computing model %s not supported",commname);
  ierr = PetscFListFind(*(MPI_Comm*)comm,CCAList,classname,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to load constructor %s",classname);
  f    = (*r)();
  PetscFunctionReturn(0);
}

