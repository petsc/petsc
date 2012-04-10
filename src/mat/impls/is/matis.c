
/*
    Creates a matrix class for using the Neumann-Neumann type preconditioners.
   This stores the matrices in globally unassembled form. Each processor 
   assembles only its local Neumann problem and the parallel matrix vector 
   product is handled "implicitly".

     We provide:
         MatMult()

    Currently this allows for only one subdomain per processor.

*/

#include <../src/mat/impls/is/matis.h>      /*I "petscmat.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_IS" 
PetscErrorCode MatDestroy_IS(Mat A)
{
  PetscErrorCode ierr;
  Mat_IS         *b = (Mat_IS*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->x);CHKERRQ(ierr);
  ierr = VecDestroy(&b->y);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&b->mapping);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_IS" 
PetscErrorCode MatMult_IS(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMult(is->A,is->x,is->y);CHKERRQ(ierr);

  /* scatter product back into global memory */
  ierr = VecSet(y,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
PetscErrorCode MatMultTranspose_IS(Mat A,Vec x,Vec y)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  y = A' * x */
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMultTranspose(is->A,is->x,is->y);CHKERRQ(ierr);

  /* scatter product back into global vector */
  ierr = VecSet(y,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_IS"
PetscErrorCode MatMultTransposeAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  v3 = v2 + A' * v1.*/
  /*  scatter the global vectors v1 and v2  into the local work vectors */
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

#undef __FUNCT__  
#define __FUNCT__ "MatView_IS"
PetscErrorCode MatView_IS(Mat A,PetscViewer viewer)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
  ierr = MatView(a->A,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetLocalToGlobalMapping_IS" 
PetscErrorCode MatSetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscErrorCode ierr;
  PetscInt       n;
  Mat_IS         *is = (Mat_IS*)A->data;
  IS             from,to;
  Vec            global;

  PetscFunctionBegin;
  if (is->mapping) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for matrix");
  PetscCheckSameComm(A,1,rmapping,2);
  if (rmapping != cmapping) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"MATIS requires the row and column mappings to be identical");
  ierr = PetscObjectReference((PetscObject)rmapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&is->mapping);CHKERRQ(ierr); 
  is->mapping = rmapping;

  /* Create the local matrix A */
  ierr = ISLocalToGlobalMappingGetSize(rmapping,&n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&is->A);CHKERRQ(ierr);
  ierr = MatSetSizes(is->A,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(is->A,"is");CHKERRQ(ierr);
  ierr = MatSetFromOptions(is->A);CHKERRQ(ierr);

  /* Create the local work vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&is->x);CHKERRQ(ierr);
  ierr = VecDuplicate(is->x,&is->y);CHKERRQ(ierr);

  /* setup the global to local scatter */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&to);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(rmapping,to,&from);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)A)->comm,1,A->cmap->n,A->cmap->N,PETSC_NULL,&global);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,is->x,to,&is->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define ISG2LMapApply(mapping,n,in,out) 0;\
  if (!(mapping)->globals) {\
   PetscErrorCode _ierr = ISGlobalToLocalMappingApply((mapping),IS_GTOLM_MASK,0,0,0,0);CHKERRQ(_ierr);\
  }\
  {\
    PetscInt _i,*_globals = (mapping)->globals,_start = (mapping)->globalstart,_end = (mapping)->globalend;\
    for (_i=0; _i<n; _i++) {\
      if (in[_i] < 0)           out[_i] = in[_i];\
      else if (in[_i] < _start) out[_i] = -1;\
      else if (in[_i] > _end)   out[_i] = -1;\
      else                      out[_i] = _globals[in[_i] - _start];\
    }\
  }

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_IS"
PetscErrorCode MatSetValues_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       rows_l[2048],cols_l[2048];
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG) 
  if (m > 2048 || n > 2048) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= 2048: they are %D %D",m,n);
  }
#endif
  ierr = ISG2LMapApply(is->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISG2LMapApply(is->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValues(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef ISG2LMapSetUp
#undef ISG2LMapApply

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesLocal_IS" 
PetscErrorCode MatSetValuesLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  ierr = MatSetValues(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_IS" 
PetscErrorCode MatZeroRows_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscInt       n_l=0, *rows_l = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x && b) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"No support");
  if (n) {
    ierr = PetscMalloc(n*sizeof(PetscInt),&rows_l);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(is->mapping,IS_GTOLM_DROP,n,rows,&n_l,rows_l);CHKERRQ(ierr);
  }
  ierr = MatZeroRowsLocal(A,n_l,rows_l,diag,x,b);CHKERRQ(ierr);
  ierr = PetscFree(rows_l);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRowsLocal_IS" 
PetscErrorCode MatZeroRowsLocal_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (x && b) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"No support");
  {
    /*
       Set up is->x as a "counting vector". This is in order to MatMult_IS
       work properly in the interface nodes.
    */
    Vec         counter;
    PetscScalar one=1.0, zero=0.0;
    ierr = VecCreateMPI(((PetscObject)A)->comm,A->cmap->n,A->cmap->N,&counter);CHKERRQ(ierr);
    ierr = VecSet(counter,zero);CHKERRQ(ierr);
    ierr = VecSet(is->x,one);CHKERRQ(ierr);
    ierr = VecScatterBegin(is->ctx,is->x,counter,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (is->ctx,is->x,counter,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(is->ctx,counter,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (is->ctx,counter,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecDestroy(&counter);CHKERRQ(ierr);
  }
  if (!n) {
    is->pure_neumann = PETSC_TRUE;
  } else {
    is->pure_neumann = PETSC_FALSE;
    ierr = VecGetArray(is->x,&array);CHKERRQ(ierr);
    ierr = MatZeroRows(is->A,n,rows,diag,0,0);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = MatSetValue(is->A,rows[i],rows[i],diag/(array[rows[i]]),INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(is->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (is->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecRestoreArray(is->x,&array);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_IS" 
PetscErrorCode MatAssemblyBegin_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyBegin(is->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_IS" 
PetscErrorCode MatAssemblyEnd_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd(is->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatISGetLocalMat_IS"
PetscErrorCode  MatISGetLocalMat_IS(Mat mat,Mat *local)
{
  Mat_IS *is = (Mat_IS *)mat->data;
  
  PetscFunctionBegin;
  *local = is->A;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatISGetLocalMat"
/*@
    MatISGetLocalMat - Gets the local matrix stored inside a MATIS matrix.

  Input Parameter:
.  mat - the matrix

  Output Parameter:
.  local - the local matrix usually MATSEQAIJ

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the 
  matrix and want to provide it to the inner matrix object to improve the performance
  of the MatSetValues() operation.

.seealso: MATIS
@*/ 
PetscErrorCode  MatISGetLocalMat(Mat mat,Mat *local)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  ierr = PetscUseMethod(mat,"MatISGetLocalMat_C",(Mat,Mat *),(mat,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatISSetLocalMat_IS"
PetscErrorCode  MatISSetLocalMat_IS(Mat mat,Mat local)
{
  Mat_IS *is = (Mat_IS *)mat->data;
  PetscInt nrows,ncols,orows,ocols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(is->A,&orows,&ocols);CHKERRQ(ierr);
  ierr = MatGetSize(local,&nrows,&ncols);CHKERRQ(ierr);
  if(orows != nrows || ocols != ncols ) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local MATIS matrix should be of size %dx%d (you passed a %dx%d matrix)\n",orows,ocols,nrows,ncols);
  }
  ierr = PetscObjectReference((PetscObject)local);CHKERRQ(ierr);
  ierr = MatDestroy(&is->A);CHKERRQ(ierr);
  is->A = local;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatISSetLocalMat"
/*@
    MatISSetLocalMat - Set the local matrix stored inside a MATIS matrix.

  Input Parameter:
.  mat - the matrix
.  local - the local matrix usually MATSEQAIJ

  Output Parameter:

  Level: advanced

  Notes:
    This can be called if you have precomputed the local matrix and 
  want to provide it to the matrix object MATIS.

.seealso: MATIS
@*/
PetscErrorCode  MatISSetLocalMat(Mat mat,Mat local)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatISSetLocalMat_C",(Mat,Mat),(mat,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_IS"
PetscErrorCode MatZeroEntries_IS(Mat A)
{
  Mat_IS         *a = (Mat_IS*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatZeroEntries(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_IS"
PetscErrorCode MatScale_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(is->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_IS"
PetscErrorCode MatGetDiagonal_IS(Mat A, Vec v)
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

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_IS"
PetscErrorCode MatSetOption_IS(Mat A,MatOption op,PetscBool  flg)
{
  Mat_IS         *a = (Mat_IS*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateIS" 
/*@
    MatCreateIS - Creates a "process" unassmembled matrix, it is assembled on each 
       process but not across processes.

   Input Parameters:
+     comm - MPI communicator that will share the matrix
.     m,n,M,N - local and/or global sizes of the the left and right vector used in matrix vector products
-     map - mapping that defines the global number for each local number

   Output Parameter:
.    A - the resulting matrix

   Level: advanced

   Notes: See MATIS for more details
          m and n are NOT related to the size of the map, they are the size of the part of the vector owned
          by that process. m + nghosts (or n + nghosts) is the length of map since map maps all local points 
          plus the ghost points to global indices.

.seealso: MATIS, MatSetLocalToGlobalMapping()
@*/
PetscErrorCode  MatCreateIS(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,ISLocalToGlobalMapping map,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATIS);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*A,map,map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATIS - MATIS = "is" - A matrix type to be used for using the Neumann-Neumann type preconditioners.
   This stores the matrices in globally unassembled form. Each processor 
   assembles only its local Neumann problem and the parallel matrix vector 
   product is handled "implicitly".

   Operations Provided:
+  MatMult()
.  MatMultAdd()
.  MatMultTranspose()
.  MatMultTransposeAdd()
.  MatZeroEntries()
.  MatSetOption()
.  MatZeroRows()
.  MatZeroRowsLocal()
.  MatSetValues()
.  MatSetValuesLocal()
.  MatScale()
.  MatGetDiagonal()
-  MatSetLocalToGlobalMapping()

   Options Database Keys:
. -mat_type is - sets the matrix type to "is" during a call to MatSetFromOptions()

   Notes: Options prefix for the inner matrix are given by -is_mat_xxx
    
          You must call MatSetLocalToGlobalMapping() before using this matrix type.

          You can do matrix preallocation on the local matrix after you obtain it with 
          MatISGetLocalMat()

  Level: advanced

.seealso: PC, MatISGetLocalMat(), MatSetLocalToGlobalMapping()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_IS" 
PetscErrorCode  MatCreate_IS(Mat A)
{
  PetscErrorCode ierr;
  Mat_IS         *b;

  PetscFunctionBegin;
  ierr                = PetscNewLog(A,Mat_IS,&b);CHKERRQ(ierr);
  A->data             = (void*)b;
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->mult                    = MatMult_IS;
  A->ops->multadd                 = MatMultAdd_IS;
  A->ops->multtranspose           = MatMultTranspose_IS;
  A->ops->multtransposeadd        = MatMultTransposeAdd_IS;
  A->ops->destroy                 = MatDestroy_IS;
  A->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_IS;
  A->ops->setvalues               = MatSetValues_IS;
  A->ops->setvalueslocal          = MatSetValuesLocal_IS;
  A->ops->zerorows                = MatZeroRows_IS;
  A->ops->zerorowslocal           = MatZeroRowsLocal_IS;
  A->ops->assemblybegin           = MatAssemblyBegin_IS;
  A->ops->assemblyend             = MatAssemblyEnd_IS;
  A->ops->view                    = MatView_IS;
  A->ops->zeroentries             = MatZeroEntries_IS;
  A->ops->scale                   = MatScale_IS;
  A->ops->getdiagonal             = MatGetDiagonal_IS;
  A->ops->setoption               = MatSetOption_IS;

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  b->A          = 0;
  b->ctx        = 0;
  b->x          = 0;  
  b->y          = 0;  
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatISGetLocalMat_C","MatISGetLocalMat_IS",MatISGetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatISSetLocalMat_C","MatISSetLocalMat_IS",MatISSetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATIS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
