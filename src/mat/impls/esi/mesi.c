/*$Id: mesi.c,v 1.1 2001/09/12 03:30:08 bsmith Exp bsmith $*/
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscsys.h"       
#include "esi/ESI.h"
#include "esi/petsc/vector.h"
#include "esi/petsc/matrix.h"
#include "src/mat/matimpl.h"   /*I "petscmat.h" I*/

typedef struct { 
  int                                   rstart,rend; /* range of local rows */
  esi::Operator<double,int>             *eop;
  esi::MatrixData<int>                  *emat;
  esi::MatrixRowWriteAccess<double,int> *rmat;
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
  ierr = v->getInterface("esi::MatrixRowWriteAccess",static_cast<void*>(x->rmat));CHKERRQ(ierr);
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
    ierr = PetscMapGetLocalRange(xin->rmap,&x->rstart,&x->rend);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(xin->comm,1,&xin->stash);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscFList CCAList;

#undef __FUNCT__  
#define __FUNCT__ "MatESISetType"
/*@
    MatESISetType - Given a PETSc matrix of type ESI loads the ESI constructor
          by name and wraps the ESI operator to look like a PETSc matrix.
@*/
int MatESISetType(Mat V,char *name)
{
  int                                ierr;
  ::esi::Operator<double,int>        *ve;
  ::esi::OperatorFactory<double,int> *f;
  void                               *(*r)(void);
  ::esi::IndexSpace<int>             *rmap,*cmap;

  PetscFunctionBegin;
  ierr = PetscFListFind(V->comm,CCAList,name,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to load esi::OperatorFactory constructor %s",name);
#if defined(PETSC_HAVE_CCA)
  gov::cca::Component *component = (gov::cca::Component *)(*r)();
  gov::cca::Port      *port      = dynamic_cast<gov::cca::Port*>(component);
  f    = dynamic_cast<esi::OperatorFactory<double,int>*>(port);
#else
  f    = (::esi::OperatorFactory<double,int> *)(*r)();
#endif
  ierr = ESICreateIndexSpace("MPI",&V->comm,V->m,rmap);CHKERRQ(ierr);
  ierr = ESICreateIndexSpace("MPI",&V->comm,V->n,cmap);CHKERRQ(ierr);
  ierr = f->getOperator(*rmap,*cmap,ve);CHKERRQ(ierr);
  ierr = rmap->deleteReference();CHKERRQ(ierr);
  ierr = cmap->deleteReference();CHKERRQ(ierr);
  delete f;
  ierr = MatESISetOperator(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatESISetFromOptions"
int MatESISetFromOptions(Mat V)
{
  Mat_ESI      *s;
  int          ierr;
  char         string[1024];
  PetscTruth   flg;
 
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)V,MATESI,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsGetString(V->prefix,"-mat_esi_type",string,1024,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatESISetType(V,string);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_ESI"
int MatSetValues_ESI(Mat mat,int m,int *im,int n,int *in,PetscScalar *v,InsertMode addv)
{
  Mat_ESI                               *esi = (Mat_ESI*)mat->data;
  PetscScalar                           value;
  int                                   ierr,i,j,rstart = esi->rstart,rend = esi->rend;
  int                                   row,col;
 
  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
    if (im[i] >= mat->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
          ierr = esi->rmat->copyInRow(im[i],&v[i+j*m],&in[j],1);CHKERRQ(ierr);
       }
    } else {
      ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_ESI"
int MatAssemblyBegin_ESI(Mat mat,MatAssemblyType mode)
{ 
  int         ierr,nstash,reallocs,*rowners;
  InsertMode  addv;

  PetscFunctionBegin;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,mat->comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = PetscMapGetGlobalRange(mat->rmap,&rowners);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(&mat->stash,rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PetscLogInfo(0,"MatAssemblyBegin_ESI:Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_ESI"
int MatAssemblyEnd_ESI(Mat mat,MatAssemblyType mode)
{ 
  Mat_ESI     *a = (Mat_ESI*)mat->data;
  int         i,j,rstart,ncols,n,ierr,flg;
  int         *row,*col,other_disassembled;
  PetscScalar *val;
  InsertMode  addv = mat->insertmode;

  PetscFunctionBegin;
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
    if (!flg) break;
     for (i=0; i<n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      ierr = MatSetValues_ESI(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
  a->rmat->loadComplete();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_ESI"
int MatMult_ESI(Mat A,Vec xx,Vec yy)
{
  Mat_ESI                 *a = (Mat_ESI*)A->data;
  int                     ierr;
  esi::Vector<double,int> *x,*y;

  PetscFunctionBegin;
  ierr = VecESIWrap(xx,&x);CHKERRQ(ierr);
  ierr = VecESIWrap(yy,&y);CHKERRQ(ierr);
  ierr = a->eop->apply(*x,*y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_ESI"
int MatDestroy_ESI(Mat v)
{
  Mat_ESI *vs = (Mat_ESI*)v->data;
  int     ierr;

  PetscFunctionBegin;
  if (vs->eop) {
    vs->eop->deleteReference();
  }
  ierr = MatStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_ESI,
       0,
       0,
       MatMult_ESI,
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
       MatAssemblyBegin_ESI,
       MatAssemblyEnd_ESI,
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
       0,
       0,
       0,
       0,
       0,
       MatDestroy_ESI,
       				       0};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_ESI"
int MatCreate_ESI(Mat B)
{
  int        ierr;
  Mat_ESI    *b;

  PetscFunctionBegin;

  ierr                = PetscNew(Mat_ESI,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr                = PetscMemzero(b,sizeof(Mat_ESI));CHKERRQ(ierr);
  ierr                = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-mat_lu_pivotthreshold",&B->lupivotthreshold,PETSC_NULL);CHKERRQ(ierr);

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

