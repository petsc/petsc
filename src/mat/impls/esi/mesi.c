/*$Id: mesi.c,v 1.1 2001/09/12 03:30:08 bsmith Exp bsmith $*/
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscsys.h"       
#include "esi/ESI.h"
#include "esi/petsc/matrix.h"
#include "src/mat/matimpl.h"   /*I "petscmat.h" I*/

typedef struct { 
  esi::Operator<double,int>   *eop;
  esi::MatrixData<int>        *emat;
} Mat_ESI;

#undef __FUNCT__  
#define __FUNCT__ "MatESISetOperator"
/*@C
     MatESISetOperator - Takes a PETSc matrix sets it to type ESI and 
       provides the ESI operator that it wraps to look like a PETSc matrix.

@*/
 int MatESISetOperator(Mat xin,esi::Operator<double,int> *v)
{
  Mat_ESI    *x = (Mat_ESI*)xin->data;
  PetscTruth tesi;
  int        ierr;

  PetscFunctionBegin;

  ierr = v->getInterface("esi::MatrixData",static_cast<void*>(x->emat));
  if (!x->emat) SETERRQ(1,"PETSc currently requires esi::Operator to support esi::MatrixData interface");

  ierr = PetscTypeCompare((PetscObject)xin,0,&tesi);CHKERRQ(ierr);
  if (tesi) {
    ierr = MatSetType(xin,MATESI);CHKERRQ(ierr);
  }
  ierr = PetscTypeCompare((PetscObject)xin,MATESI,&tesi);CHKERRQ(ierr);
  if (tesi) {
    int                    m,n,M,N;
    esi::IndexSpace<int>   *rmap,*cmap;

    ierr = x->emat->getIndexSpaces(rmap,cmap);CHKERRQ(ierr);

    ierr = rmap->getGlobalSize(M);CHKERRQ(ierr);
    if (xin->M == -1) xin->M = M;
    else if (xin->M != M) SETERRQ2(1,"Global rows of Mat %d not equal size of esi::MatrixData %d",xin->M,M);

    ierr = cmap->getGlobalSize(N);CHKERRQ(ierr);
    if (xin->N == -1) xin->N = N;
    else if (xin->N != N) SETERRQ2(1,"Global columns of Mat %d not equal size of esi::MatrixData %d",xin->N,N);

    ierr = rmap->getLocalSize(m);CHKERRQ(ierr);
    if (xin->m == -1) xin->m = m;
    else if (xin->m != m) SETERRQ2(1,"Local rows of Mat %d not equal size of esi::MatrixData %d",xin->n,n);

    ierr = cmap->getLocalSize(n);CHKERRQ(ierr);
    if (xin->n == -1) xin->n = n;
    else if (xin->n != n) SETERRQ2(1,"Local columns of Mat %d not equal size of esi::MatrixData %d",xin->n,n);

    x->eop  = v;
    v->addReference();
    if (!xin->rmap){
      ierr = PetscMapCreateMPI(xin->comm,m,M,&xin->rmap);CHKERRQ(ierr);
    }
    if (!xin->cmap){
      ierr = PetscMapCreateMPI(xin->comm,n,N,&xin->cmap);CHKERRQ(ierr);
    }
    ierr = MatStashCreate_Private(xin->comm,1,&xin->stash);CHKERRQ(ierr);
    //    ierr = (v)->getInterface("esi::Vector",xin->esivec);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {0,
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
       0,
       0,
       0,
       0,
				       0};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_ESI"
int MatCreate_ESI(Mat B)
{
  int        ierr;
  Mat_ESI    *b;

  PetscFunctionBegin;

  B->m = B->M = PetscMax(B->m,B->M);
  B->n = B->N = PetscMax(B->n,B->N);

  ierr                = PetscNew(Mat_ESI,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr                = PetscMemzero(b,sizeof(Mat_ESI));CHKERRQ(ierr);
  ierr                = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-mat_lu_pivotthreshold",&B->lupivotthreshold,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscMapCreateMPI(B->comm,B->m,B->m,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->n,B->n,&B->cmap);CHKERRQ(ierr);

  b->emat = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_PetscESI"
int MatCreate_PetscESI(Mat V)
{
  int                            ierr;
  Mat                            v;
  esi::petsc::Matrix<double,int> *ve;

  PetscFunctionBegin;
  V->ops->destroy = 0;  /* since this is called from MatSetType() we have to make sure it doesn't get destroyed twice */
  ierr = MatSetType(V,MATESI);CHKERRQ(ierr);
  ierr = MatCreate(V->comm,V->m,V->n,V->M,V->N,&v);CHKERRQ(ierr);
  ierr = MatSetType(v,MATMPIAIJ);CHKERRQ(ierr);
  ve   = new esi::petsc::Matrix<double,int>(v);
  ierr = MatESISetOperator(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

