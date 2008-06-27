#define PETSCMAT_DLL

#include "src/mat/impls/is/matis.h"      /*I "petscmat.h" I*/


#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
#define VecScatterBegin(sct,x,y,im,sm) VecScatterBegin(x,y,im,sm,sct)
#define VecScatterEnd(sct,x,y,im,sm)   VecScatterEnd(x,y,im,sm,sct)
#endif

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_IS(Mat);
EXTERN_C_END

/* -------------------------------------------------------------------------- */

#define ISG2LMapSetUp(mapping)\
  if (!(mapping)->globals) {\
   PetscErrorCode _ierr = ISGlobalToLocalMappingApply((mapping),IS_GTOLM_MASK,0,0,0,0);CHKERRQ(_ierr);\
  }

#if defined(PETSC_USE_DEBUG)

#define ISG2LMapApply(mapping,n,in,out) 0;\
  ISG2LMapSetUp(mapping) \
  {\
    PetscInt _i,*_globals = (mapping)->globals,_start = (mapping)->globalstart,_end = (mapping)->globalend;\
    for (_i=0; _i<n; _i++) {\
      if (in[_i] < 0) out[_i] = in[_i];\
      else if ((in[_i] < _start) || (in[_i] > _end)) {SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Index out of range: idx %D min %D max %D",in[_i],_start,_end);}\
      else out[_i] = _globals[in[_i] - _start];\
    }\
  }

#else

#define ISG2LMapApply(mapping,n,in,out) 0;\
  ISG2LMapSetUp(mapping)\
  {\
    PetscInt _i,*_globals = (mapping)->globals,_start = (mapping)->globalstart;\
    for (_i=0; _i<n; _i++) {\
      if (in[_i] < 0) out[_i] = in[_i];\
      else            out[_i] = _globals[in[_i] - _start];\
    }\
  }

#endif

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_IS"
static PetscErrorCode MatSetValues_IS(Mat mat,
				      PetscInt m,const PetscInt *rows,
				      PetscInt n,const PetscInt *cols,
				      const PetscScalar *values,
				      InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       rows_l[2048],cols_l[2048];
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG) 
  if (m > 2048 || n > 2048) {
    SETERRQ2(PETSC_ERR_SUP,"Number row/column indices must be <= 2048: are %D %D",m,n);
  }
#endif
  ierr = ISG2LMapApply(is->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISG2LMapApply(is->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValues(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef ISG2LMapSetUp
#undef ISG2LMapApply

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_IS" 
static PetscErrorCode MatZeroRows_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscInt       n_l=0, *rows_l = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) {
    ierr = PetscMalloc(n*sizeof(PetscInt),&rows_l);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(is->mapping,IS_GTOLM_DROP,n,rows,&n_l,rows_l);CHKERRQ(ierr);
  }
  ierr = MatZeroRowsLocal(A,n_l,rows_l,diag);CHKERRQ(ierr);
  if (rows_l) { ierr = PetscFree(rows_l);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatScale_IS"
static PetscErrorCode MatScale_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(is->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_IS"
static PetscErrorCode MatGetDiagonal_IS(Mat A, Vec v)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get diagonal of the local matrix */
  ierr = MatGetDiagonal(is->A,is->x);CHKERRQ(ierr);
  /* scatter diagonal back into global vector */
  ierr = VecSet(v,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->x,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,is->x,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_IS"
static PetscErrorCode MatMultAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin; /*  v3 = v2 + A * v1.*/
  /*  scatter the global vector v1 into the local work vector */
  ierr = VecScatterBegin(is->ctx,v1,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,v1,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,v2,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,v2,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* multiply the local matrix */
  ierr = MatMultAdd(is->A,is->x,is->y,is->y);CHKERRQ(ierr);
  /* scatter result back into global vector */
  ierr = VecSet(v3,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,v3,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,is->y,v3,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_IS" 
static PetscErrorCode MatMultTranspose_IS(Mat A,Vec x,Vec y)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  y = A' * x */
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* multiply the local matrix */
  ierr = MatMultTranspose(is->A,is->x,is->y);CHKERRQ(ierr);
  /* scatter product back into global vector */
  ierr = VecSet(y,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_IS"
static PetscErrorCode MatMultTransposeAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  v3 = v2 + A' * v1.*/
  /*  scatter the global vector v1 into the local work vector */
  ierr = VecScatterBegin(is->ctx,v1,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,v1,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,v2,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,v2,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* multiply the local matrix */
  ierr = MatMultTransposeAdd(is->A,is->x,is->y,is->y);CHKERRQ(ierr);
  /* scatter result back into global vector */
  ierr = VecSet(v3,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,v3,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,is->y,v3,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatSetLocalToGlobalMapping_IS" 
static PetscErrorCode MatSetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;
  PetscInt       n;
  Mat_IS         *is = (Mat_IS*)A->data;
  IS             from,to;
  Vec            global;

  PetscFunctionBegin;
  if (is->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for matrix");
  }
  /* set the local to global mapping */
  PetscCheckSameComm(A,1,mapping,2);
  is->mapping = mapping;
  ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
  /* create the local matrix A */
  ierr = ISLocalToGlobalMappingGetSize(mapping,&n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&is->A);CHKERRQ(ierr);
  ierr = MatSetSizes(is->A,n,n,n,n);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)is->A,"is");CHKERRQ(ierr);
  ierr = MatSetFromOptions(is->A);CHKERRQ(ierr);
  /* create the local work vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&is->x);CHKERRQ(ierr);
  ierr = VecDuplicate(is->x,&is->y);CHKERRQ(ierr);
  /* setup the global to local scatter */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&to);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(mapping,to,&from);CHKERRQ(ierr);
  ierr = VecCreateMPI (((PetscObject)A)->comm,A->cmap.n,A->cmap.N,&global);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,is->x,to,&is->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_ISX"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_ISX(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create */
  if (A->rmap.bs <= 0) A->rmap.bs = 1;
  if (A->cmap.bs <= 0) A->cmap.bs = 1;
  ierr = MatCreate_IS(A);CHKERRQ(ierr);
  /* set local to global mapping */
  ierr = MatShellSetOperation(A,MATOP_SET_LOCAL_TO_GLOBAL_MAP,(void(*)(void))MatSetLocalToGlobalMapping_IS);CHKERRQ(ierr);
  /* set values, zero rows and scale */
  ierr = MatShellSetOperation(A,MATOP_SET_VALUES,(void(*)(void))MatSetValues_IS);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_ZERO_ROWS,(void(*)(void))MatZeroRows_IS);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_SCALE,(void(*)(void))MatScale_IS);CHKERRQ(ierr);
  /* get diagonal */
  ierr = MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_IS);CHKERRQ(ierr);
  /* mult operations */
  ierr = MatShellSetOperation(A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_IS);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_IS);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_IS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
