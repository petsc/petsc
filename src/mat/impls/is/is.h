/*$Id: nn.c,v 1.2 2000/06/05 03:52:13 bsmith Exp $*/
/*
    Creates a matrix class for using the Neumann-Neumann type preconditioners.
   This stores the matrices in globally unassembled form. Each processor 
   assembles only its local Neumann problem and the parallel matrix vector 
   product is handled "implicitly".

     We provide:
         MatMult()

    Currently this allows for only one subdomain per processor.

*/

#include "src/mat/matimpl.h"

typedef struct {
  Mat                    A;             /* the local Neumann matrix */
  VecScatter             ctx;           /* update ghost points for matrix vector product */
  Vec                    x,y;           /* work space for ghost values for matrix vector product */
  ISLocalToGlobalMapping mapping;
  int                    rstart,rend;   /* local row ownership */
  int                    *zeroedrows,nzeroedrows;
} Mat_NN;

#undef __FUNC__  
#define __FUNC__ /*<a name="MatDestroy_NN"></a>*/"MatDestroy_NN" 
int MatDestroy_NN(Mat A)
{
  int    ierr;
  Mat_NN *b = (Mat_NN*)A->data;

  PetscFunctionBegin;
  if (b->A) {
    ierr = MatDestroy(b->A);CHKERRQ(ierr);
  }
  if (b->ctx) {
    ierr = VecScatterDestroy(b->ctx);CHKERRQ(ierr);
  }
  if (b->x) {
    ierr = VecDestroy(b->x);CHKERRQ(ierr);
  }
  if (b->y) {
    ierr = VecDestroy(b->y);CHKERRQ(ierr);
  }
  if (b->zeroedrows) {
    ierr = PetscFree(b->zeroedrows);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMult_NN"></a>*/"MatMult_NN" 
int MatMult_NN(Mat A,Vec x,Vec y)
{
  int    ierr,i;
  Mat_NN *nn = (Mat_NN*)A->data;
  Scalar zero = 0.0,*array;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(x,nn->x,INSERT_VALUES,SCATTER_FORWARD,nn->ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,nn->x,INSERT_VALUES,SCATTER_FORWARD,nn->ctx);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMult(nn->A,nn->x,nn->y);CHKERRQ(ierr);

  /* zero out redundant rows in matrix */
  ierr = VecGetArray(nn->y,&array);CHKERRQ(ierr);
  for (i=0; i<nn->nzeroedrows;i++) {
    array[nn->zeroedrows[i]] = 0.0;
  }
  ierr = VecRestoreArray(nn->y,&array);CHKERRQ(ierr);

  /* scatter product back into global memory */
  ierr = VecSet(&zero,y);CHKERRQ(ierr);
  ierr = VecScatterBegin(nn->y,y,ADD_VALUES,SCATTER_REVERSE,nn->ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(nn->y,y,ADD_VALUES,SCATTER_REVERSE,nn->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetLocalToGlobalMapping_NN"></a>*/"MatSetLocalToGlobalMapping_NN" 
int MatSetLocalToGlobalMapping_NN(Mat A,ISLocalToGlobalMapping mapping)
{
  int    ierr,n;
  Mat_NN *nn = (Mat_NN*)A->data;
  IS     from,to;
  Vec    global;

  PetscFunctionBegin;
  nn->mapping = mapping;

  /* Create the local matrix A */
  ierr = ISLocalToGlobalMappingGetSize(mapping,&n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,n,n,n,n,&nn->A);CHKERRQ(ierr);

  /* Create the local work vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&nn->x);CHKERRQ(ierr);
  ierr = VecDuplicate(nn->x,&nn->y);CHKERRQ(ierr);

  /* setup the global to local scatter */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&to);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(mapping,to,&from);CHKERRQ(ierr);
  ierr = VecCreateMPI(A->comm,A->n,A->N,&global);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,nn->x,to,&nn->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValuesLocal_NN"></a>*/"MatSetValuesLocal_NN" 
int MatSetValuesLocal_NN(Mat A,int m,int *rows,int n,int *cols,Scalar *values,InsertMode addv)
{
  int    ierr;
  Mat_NN *nn = (Mat_NN*)A->data;

  PetscFunctionBegin;
  ierr = MatSetValues(nn->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatZeroRowsLocal_NN"></a>*/"MatZeroRowsLocal_NN" 
int MatZeroRowsLocal_NN(Mat A,IS isrows,Scalar *diag)
{
  Mat_NN *nn = (Mat_NN*)A->data;
  int    ierr,i,rstart = nn->rstart,rend = nn->rend,*global,*nonlocal,n,cnt,*rows;
  IS     isglobal;

  PetscFunctionBegin;

  /* 
      zerorows[] contains the list of rows that have been zeroed but are not
    owned by this processor. We needs this so that in the matrix vector product
    we can zero the contribution from this processor (because the owner processor
    is already doing the calculation.) We don't just completely zero the local
    corresponding local rows because then our local submatrices will be screwed
    up (have zero rows and thus be singular).
  */
  ierr = ISGetLocalSize(isrows,&n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(nn->mapping,isrows,&isglobal);CHKERRQ(ierr);
  ierr = ISGetIndices(isglobal,&global);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<n; i++) {
    if (global[i] < rstart || global[i] >= rend) cnt++;
  }
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  nn->nzeroedrows = cnt;
  nn->zeroedrows  = (int*)PetscMalloc((cnt+1)*sizeof(int));CHKPTRQ(nn->zeroedrows);
  cnt = 0;
  for (i=0; i<n; i++) {
    if (global[i] < rstart || global[i] >= rend) nn->zeroedrows[cnt++] = rows[i];
  }
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isglobal,&global);CHKERRQ(ierr);
  ierr = ISDestroy(isglobal);CHKERRQ(ierr);
  ierr = MatZeroRows(nn->A,isrows,diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatAssemblyBegin_NN"></a>*/"MatAssemblyBegin_NN" 
int MatAssemblyBegin_NN(Mat A,MatAssemblyType type)
{
  Mat_NN *nn = (Mat_NN*)A->data;
  int    ierr;
  PetscFunctionBegin;
  ierr = MatAssemblyBegin(nn->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatAssemblyEnd_NN"></a>*/"MatAssemblyEnd_NN" 
int MatAssemblyEnd_NN(Mat A,MatAssemblyType type)
{
  Mat_NN *nn = (Mat_NN*)A->data;
  int    ierr;
  PetscFunctionBegin;
  ierr = MatAssemblyEnd(nn->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="MatCreate_NN"></a>*/"MatCreate_NN" 
int MatCreate_NN(Mat A)
{
  int    ierr;
  Mat_NN *b;

  PetscFunctionBegin;
  A->data             = (void*)(b = PetscNew(Mat_NN));CHKPTRQ(b);
  ierr = PetscMemzero(b,sizeof(Mat_NN));CHKERRQ(ierr);
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->factor           = 0;
  A->mapping          = 0;

  A->ops->mult                    = MatMult_NN;
  A->ops->destroy                 = MatDestroy_NN;
  A->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_NN;
  A->ops->setvalueslocal          = MatSetValuesLocal_NN;
  A->ops->zerorowslocal           = MatZeroRowsLocal_NN;
  A->ops->assemblybegin           = MatAssemblyBegin_NN;
  A->ops->assemblyend             = MatAssemblyEnd_NN;

  ierr = PetscSplitOwnership(A->comm,&A->m,&A->M);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(A->comm,&A->n,&A->N);CHKERRQ(ierr);
  ierr = MPI_Scan(&A->m,&b->rend,1,MPI_INT,MPI_SUM,A->comm);CHKERRQ(ierr);
  b->rstart = b->rend - A->m;
  b->A          = 0;
  b->ctx        = 0;
  b->x          = 0;  
  b->y          = 0;  
  b->zeroedrows = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END













