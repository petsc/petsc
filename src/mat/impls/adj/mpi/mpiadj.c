#define PETSCMAT_DLL

/*
    Defines the basic matrix operations for the ADJ adjacency list matrix data-structure. 
*/
#include "../src/mat/impls/adj/mpi/mpiadj.h"

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAdj_ASCII"
PetscErrorCode MatView_MPIAdj_ASCII(Mat A,PetscViewer viewer)
{
  Mat_MPIAdj        *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    SETERRQ(PETSC_ERR_SUP,"Matlab format not supported");
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"row %D:",i+A->rmap->rstart);CHKERRQ(ierr);
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %D ",a->j[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } 
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAdj"
PetscErrorCode MatView_MPIAdj(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_MPIAdj_ASCII(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by MPIAdj",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAdj"
PetscErrorCode MatDestroy_MPIAdj(Mat mat)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D, NZ=%D",mat->rmap->n,mat->cmap->n,a->nz);
#endif
  ierr = PetscFree(a->diag);CHKERRQ(ierr);
  if (a->freeaij) {
    if (a->freeaijwithfree) {
      if (a->i) free(a->i);
      if (a->j) free(a->j);
    } else {
      ierr = PetscFree(a->i);CHKERRQ(ierr);
      ierr = PetscFree(a->j);CHKERRQ(ierr);
      ierr = PetscFree(a->values);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIAdj"
PetscErrorCode MatSetOption_MPIAdj(Mat A,MatOption op,PetscTruth flg)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
    a->symmetric = flg;
    break;
  case MAT_SYMMETRY_ETERNAL:
    break;
  default:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}


/*
     Adds diagonal pointers to sparse matrix structure.
*/

#undef __FUNCT__  
#define __FUNCT__ "MatMarkDiagonal_MPIAdj"
PetscErrorCode MatMarkDiagonal_MPIAdj(Mat A)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data; 
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n;

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscInt),&a->diag);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(A,m*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIAdj"
PetscErrorCode MatGetRow_MPIAdj(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data;
  PetscInt   *itmp;

  PetscFunctionBegin;
  row -= A->rmap->rstart;

  if (row < 0 || row >= A->rmap->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row out of range");

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = PETSC_NULL;
  if (idx) {
    itmp = a->j + a->i[row];
    if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPIAdj"
PetscErrorCode MatRestoreRow_MPIAdj(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPIAdj"
PetscErrorCode MatEqual_MPIAdj(Mat A,Mat B,PetscTruth* flg)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj *)A->data,*b = (Mat_MPIAdj *)B->data;
  PetscErrorCode ierr;
  PetscTruth     flag;

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) ||(a->nz != b->nz)) {
    flag = PETSC_FALSE;
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(A->rmap->n+1)*sizeof(PetscInt),&flag);CHKERRQ(ierr);
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(PetscInt),&flag);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&flag,flg,1,MPI_INT,MPI_LAND,((PetscObject)A)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ_MPIAdj"
PetscErrorCode MatGetRowIJ_MPIAdj(Mat A,PetscInt oshift,PetscTruth symmetric,PetscTruth blockcompressed,PetscInt *m,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  PetscInt       i;
  Mat_MPIAdj     *a = (Mat_MPIAdj *)A->data;

  PetscFunctionBegin;
  *m    = A->rmap->n;
  *ia   = a->i;
  *ja   = a->j;
  *done = PETSC_TRUE;
  if (oshift) {
    for (i=0; i<(*ia)[*m]; i++) {
      (*ja)[i]++;
    }
    for (i=0; i<=(*m); i++) (*ia)[i]++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ_MPIAdj"
PetscErrorCode MatRestoreRowIJ_MPIAdj(Mat A,PetscInt oshift,PetscTruth symmetric,PetscTruth blockcompressed,PetscInt *m,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  PetscInt   i;
  Mat_MPIAdj *a = (Mat_MPIAdj *)A->data;

  PetscFunctionBegin;
  if (ia && a->i != *ia) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"ia passed back is not one obtained with MatGetRowIJ()");
  if (ja && a->j != *ja) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"ja passed back is not one obtained with MatGetRowIJ()");
  if (oshift) {
    for (i=0; i<=(*m); i++) (*ia)[i]--;
    for (i=0; i<(*ia)[*m]; i++) {
      (*ja)[i]--;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatConvertFrom_MPIAdj"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvertFrom_MPIAdj(Mat A,const MatType type,MatReuse reuse,Mat *newmat)
{
  Mat               B;
  PetscErrorCode    ierr;
  PetscInt          i,m,N,nzeros = 0,*ia,*ja,len,rstart,cnt,j,*a;
  const PetscInt    *rj;
  const PetscScalar *ra;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,PETSC_NULL);CHKERRQ(ierr);
  
  /* count the number of nonzeros per row */
  for (i=0; i<m; i++) {
    ierr   = MatGetRow(A,i+rstart,&len,&rj,PETSC_NULL);CHKERRQ(ierr);
    for (j=0; j<len; j++) {
      if (rj[j] == i+rstart) {len--; break;}    /* don't count diagonal */
    }
    ierr   = MatRestoreRow(A,i+rstart,&len,&rj,PETSC_NULL);CHKERRQ(ierr);
    nzeros += len;
  }

  /* malloc space for nonzeros */
  ierr = PetscMalloc((nzeros+1)*sizeof(PetscInt),&a);CHKERRQ(ierr);
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&ia);CHKERRQ(ierr);
  ierr = PetscMalloc((nzeros+1)*sizeof(PetscInt),&ja);CHKERRQ(ierr);

  nzeros = 0;
  ia[0]  = 0;
  for (i=0; i<m; i++) {
    ierr    = MatGetRow(A,i+rstart,&len,&rj,&ra);CHKERRQ(ierr);
    cnt     = 0;
    for (j=0; j<len; j++) {
      if (rj[j] != i+rstart) { /* if not diagonal */
        a[nzeros+cnt]    = (PetscInt) PetscAbsScalar(ra[j]);
        ja[nzeros+cnt++] = rj[j];
      } 
    }
    ierr    = MatRestoreRow(A,i+rstart,&len,&rj,&ra);CHKERRQ(ierr);
    nzeros += cnt;
    ia[i+1] = nzeros; 
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,PETSC_DETERMINE,PETSC_DETERMINE,N);CHKERRQ(ierr);
  ierr = MatSetType(B,type);CHKERRQ(ierr);
  ierr = MatMPIAdjSetPreallocation(B,ia,ja,a);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {0,
       MatGetRow_MPIAdj,
       MatRestoreRow_MPIAdj,
       0,
/* 4*/ 0,
       0,
       0,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       0,
       0,
/*15*/ 0,
       MatEqual_MPIAdj,
       0,
       0,
       0,
/*20*/ 0,
       0,
       MatSetOption_MPIAdj,
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
       MatGetRowIJ_MPIAdj,
       MatRestoreRowIJ_MPIAdj,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ 0,
       MatDestroy_MPIAdj,
       MatView_MPIAdj,
       MatConvertFrom_MPIAdj,
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPIAdjSetPreallocation_MPIAdj"
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAdjSetPreallocation_MPIAdj(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  Mat_MPIAdj     *b = (Mat_MPIAdj *)B->data;
  PetscErrorCode ierr;
#if defined(PETSC_USE_DEBUG)
  PetscInt       ii;
#endif

  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(B->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  if (i[0] != 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"First i[] index must be zero, instead it is %D\n",i[0]);
  for (ii=1; ii<B->rmap->n; ii++) {
    if (i[ii] < 0 || i[ii] < i[ii-1]) {
      SETERRQ4(PETSC_ERR_ARG_OUTOFRANGE,"i[%D]=%D index is out of range: i[%D]=%D",ii,i[ii],ii-1,i[ii-1]);
    }
  }
  for (ii=0; ii<i[B->rmap->n]; ii++) {
    if (j[ii] < 0 || j[ii] >= B->cmap->N) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column index %D out of range %D\n",ii,j[ii]);
    }
  } 
#endif
  B->preallocated = PETSC_TRUE;

  b->j      = j;
  b->i      = i;
  b->values = values;

  b->nz               = i[B->rmap->n];
  b->diag             = 0;
  b->symmetric        = PETSC_FALSE;
  b->freeaij          = PETSC_TRUE;

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATMPIADJ - MATMPIADJ = "mpiadj" - A matrix type to be used for distributed adjacency matrices,
   intended for use constructing orderings and partitionings.

  Level: beginner

.seealso: MatCreateMPIAdj
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIAdj"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAdj(Mat B)
{
  Mat_MPIAdj     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)B)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)B)->comm,&rank);CHKERRQ(ierr);

  ierr                = PetscNewLog(B,Mat_MPIAdj,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr                = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->mapping          = 0;
  B->assembled        = PETSC_FALSE;
  
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAdjSetPreallocation_C",
                                    "MatMPIAdjSetPreallocation_MPIAdj",
                                     MatMPIAdjSetPreallocation_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIADJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIAdjSetPreallocation"
/*@C
   MatMPIAdjSetPreallocation - Sets the array used for storing the matrix elements

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).
-  values - [optional] edge weights

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAdj(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAdjSetPreallocation(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt*,PetscInt*,PetscInt*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPIAdjSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,i,j,values);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIAdj"
/*@C
   MatCreateMPIAdj - Creates a sparse matrix representing an adjacency list.
   The matrix does not have numerical values associated with it, but is
   intended for ordering (to reduce bandwidth etc) and partitioning.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows
.  N - number of global columns
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).
-  values -[optional] edge weights

   Output Parameter:
.  A - the matrix 

   Level: intermediate

   Notes: This matrix object does not support most matrix operations, include
   MatSetValues().
   You must NOT free the ii, values and jj arrays yourself. PETSc will free them
   when the matrix is destroyed; you must allocate them with PetscMalloc(). If you 
    call from Fortran you need not create the arrays with PetscMalloc().
   Should not include the matrix diagonals.

   If you already have a matrix, you can create its adjacency matrix by a call
   to MatConvert, specifying a type of MATMPIADJ.

   Possible values for MatSetOption() - MAT_STRUCTURALLY_SYMMETRIC

.seealso: MatCreate(), MatConvert(), MatGetOrdering()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAdj(MPI_Comm comm,PetscInt m,PetscInt N,PetscInt *i,PetscInt *j,PetscInt *values,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,PETSC_DETERMINE,PETSC_DETERMINE,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATMPIADJ);CHKERRQ(ierr);
  ierr = MatMPIAdjSetPreallocation(*A,i,j,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







