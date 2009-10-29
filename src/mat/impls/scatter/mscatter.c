#define PETSCMAT_DLL

/*
   This provides a matrix that applies a VecScatter to a vector.
*/

#include "private/matimpl.h"        /*I "petscmat.h" I*/
#include "private/vecimpl.h"  

typedef struct {
  VecScatter scatter;
} Mat_Scatter;      

#undef __FUNCT__  
#define __FUNCT__ "MatScatterGetVecScatter"
/*@
    MatScatterGetVecScatter - Returns the user-provided scatter set with MatScatterSetVecScatter()

    Not Collective, but not cannot use scatter if not used collectively on Mat

    Input Parameter:
.   mat - the matrix, should have been created with MatCreateScatter() or have type MATSCATTER

    Output Parameter:
.   scatter - the scatter context

    Level: intermediate

.keywords: matrix, scatter, get

.seealso: MatCreateScatter(), MatScatterSetVecScatter(), MATSCATTER
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatScatterGetVecScatter(Mat mat,VecScatter *scatter)
{
  Mat_Scatter    *mscatter;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(scatter,2); 
  mscatter = (Mat_Scatter*)mat->data;
  *scatter = mscatter->scatter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Scatter"
PetscErrorCode MatDestroy_Scatter(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Scatter    *scatter = (Mat_Scatter*)mat->data;

  PetscFunctionBegin;
  if (scatter->scatter) {ierr = VecScatterDestroy(scatter->scatter);CHKERRQ(ierr);}
  ierr = PetscFree(scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Scatter"
PetscErrorCode MatMult_Scatter(Mat A,Vec x,Vec y)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!scatter->scatter) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter->scatter,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Scatter"
PetscErrorCode MatMultAdd_Scatter(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!scatter->scatter) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  if (z != y) {ierr = VecCopy(y,z);CHKERRQ(ierr);}
  ierr = VecScatterBegin(scatter->scatter,x,z,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,z,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Scatter"
PetscErrorCode MatMultTranspose_Scatter(Mat A,Vec x,Vec y)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!scatter->scatter) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter->scatter,x,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_Scatter"
PetscErrorCode MatMultTransposeAdd_Scatter(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!scatter->scatter) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  if (z != y) {ierr = VecCopy(y,z);CHKERRQ(ierr);}
  ierr = VecScatterBegin(scatter->scatter,x,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {0,
       0,
       0,
       MatMult_Scatter,
/* 4*/ MatMultAdd_Scatter,
       MatMultTranspose_Scatter,
       MatMultTransposeAdd_Scatter,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       0,
       0,
/*15*/ 0,
       0,
       0,
       0,
       0,
/*20*/ 0,
       0,
       0,
       0,
/*24*/ 0,
       0,
       0,
       0,
       0,
/*29*/ 0,
       0,
       0,
       0,
       0,
/*34*/ 0,
       0,
       0,
       0,
       0,
/*39*/ 0,
       0,
       0,
       0,
       0,
/*44*/ 0,
       0,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ 0,
       MatDestroy_Scatter,
       0,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       0,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
       0,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0,
       0,
       0,
       0,
       0,
/*94*/ 0,
       0,
       0,
       0};

/*MC
   MATSCATTER - MATSCATTER = "scatter" - A matrix type that simply applies a VecScatterBegin/End()

  Level: advanced

.seealso: MatCreateScatter(), MatScatterSetVecScatter(), MatScatterGetVecScatter()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Scatter"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Scatter(Mat A)
{
  Mat_Scatter    *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  ierr = PetscNewLog(A,Mat_Scatter,&b);CHKERRQ(ierr);

  A->data = (void*)b;

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->assembled     = PETSC_TRUE;
  A->preallocated  = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSCATTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateScatter"
/*@C
   MatCreateScatter - Creates a new matrix based on a VecScatter

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
-  scatter - a VecScatterContext

   Output Parameter:
.  A - the matrix

   Level: intermediate

   PETSc requires that matrices and vectors being used for certain
   operations are partitioned accordingly.  For example, when
   creating a scatter matrix, A, that supports parallel matrix-vector
   products using MatMult(A,x,y) the user should set the number
   of local matrix rows to be the number of local elements of the
   corresponding result vector, y. Note that this is information is
   required for use of the matrix interface routines, even though
   the scatter matrix may not actually be physically partitioned.
   For example,

.keywords: matrix, scatter, create

.seealso: MatScatterSetVecScatter(), MatScatterGetVecScatter(), MATSCATTER
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateScatter(MPI_Comm comm,VecScatter scatter,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,scatter->to_n,scatter->from_n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSCATTER);CHKERRQ(ierr);
  ierr = MatScatterSetVecScatter(*A,scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScatterSetVecScatter"
/*@
    MatScatterSetVecScatter - sets that scatter that the matrix is to apply as its linear operator

   Collective on Mat

    Input Parameters:
+   mat - the scatter matrix
-   scatter - the scatter context create with VecScatterCreate()

   Level: advanced


.seealso: MatCreateScatter(), MATSCATTER
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatScatterSetVecScatter(Mat mat,VecScatter scatter)
{
  Mat_Scatter    *mscatter = (Mat_Scatter*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(scatter,VEC_SCATTER_COOKIE,2);
  PetscCheckSameComm((PetscObject)scatter,1,(PetscObject)mat,2);
  if (mat->rmap->n != scatter->to_n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Number of local rows in matrix %D not equal local scatter size %D",mat->rmap->n,scatter->to_n);
  if (mat->cmap->n != scatter->from_n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Number of local columns in matrix %D not equal local scatter size %D",mat->cmap->n,scatter->from_n);

  ierr = PetscObjectReference((PetscObject)scatter);CHKERRQ(ierr);
  if (mscatter->scatter) {ierr = VecScatterDestroy(mscatter->scatter);CHKERRQ(ierr);}
  mscatter->scatter = scatter;
  PetscFunctionReturn(0);
}


