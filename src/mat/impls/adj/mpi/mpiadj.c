
/*
    Defines the basic matrix operations for the ADJ adjacency list matrix data-structure.
*/
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/

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
    SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"MATLAB format not supported");
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"row %D:",i+A->rmap->rstart);CHKERRQ(ierr);
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %D ",a->j[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MPIAdj"
PetscErrorCode MatView_MPIAdj(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_MPIAdj_ASCII(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by MPIAdj",((PetscObject)viewer)->type_name);
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
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjCreateNonemptySubcommMat_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_MPIAdj"
PetscErrorCode MatSetOption_MPIAdj(Mat A,MatOption op,PetscBool  flg)
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

  if (row < 0 || row >= A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row out of range");

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
PetscErrorCode MatEqual_MPIAdj(Mat A,Mat B,PetscBool * flg)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj *)A->data,*b = (Mat_MPIAdj *)B->data;
  PetscErrorCode ierr;
  PetscBool      flag;

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
PetscErrorCode MatGetRowIJ_MPIAdj(Mat A,PetscInt oshift,PetscBool  symmetric,PetscBool  blockcompressed,PetscInt *m,const PetscInt *inia[],const PetscInt *inja[],PetscBool  *done)
{
  PetscInt       i;
  Mat_MPIAdj     *a = (Mat_MPIAdj *)A->data;
  PetscInt       **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

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
PetscErrorCode MatRestoreRowIJ_MPIAdj(Mat A,PetscInt oshift,PetscBool  symmetric,PetscBool  blockcompressed,PetscInt *m,const PetscInt *inia[],const PetscInt *inja[],PetscBool  *done)
{
  PetscInt   i;
  Mat_MPIAdj *a = (Mat_MPIAdj *)A->data;
  PetscInt   **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

  PetscFunctionBegin;
  if (ia && a->i != *ia) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"ia passed back is not one obtained with MatGetRowIJ()");
  if (ja && a->j != *ja) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"ja passed back is not one obtained with MatGetRowIJ()");
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
PetscErrorCode  MatConvertFrom_MPIAdj(Mat A,MatType type,MatReuse reuse,Mat *newmat)
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
    ierr   = MatGetRow(A,i+rstart,&len,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr   = MatRestoreRow(A,i+rstart,&len,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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
      a[nzeros+cnt]    = (PetscInt) PetscAbsScalar(ra[j]);
      ja[nzeros+cnt++] = rj[j];
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
PetscErrorCode  MatMPIAdjSetPreallocation_MPIAdj(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  Mat_MPIAdj     *b = (Mat_MPIAdj *)B->data;
  PetscErrorCode ierr;
#if defined(PETSC_USE_DEBUG)
  PetscInt       ii;
#endif

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  if (i[0] != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"First i[] index must be zero, instead it is %D\n",i[0]);
  for (ii=1; ii<B->rmap->n; ii++) {
    if (i[ii] < 0 || i[ii] < i[ii-1]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i[%D]=%D index is out of range: i[%D]=%D",ii,i[ii],ii-1,i[ii-1]);
  }
  for (ii=0; ii<i[B->rmap->n]; ii++) {
    if (j[ii] < 0 || j[ii] >= B->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index %D out of range %D\n",ii,j[ii]);
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

#undef __FUNCT__
#define __FUNCT__ "MatMPIAdjCreateNonemptySubcommMat_MPIAdj"
PETSC_EXTERN_C PetscErrorCode MatMPIAdjCreateNonemptySubcommMat_MPIAdj(Mat A,Mat *B)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode ierr;
  const PetscInt *ranges;
  MPI_Comm       acomm,bcomm;
  MPI_Group      agroup,bgroup;
  PetscMPIInt    i,rank,size,nranks,*ranks;

  PetscFunctionBegin;
  *B = PETSC_NULL;
  acomm = ((PetscObject)A)->comm;
  ierr = MPI_Comm_size(acomm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_size(acomm,&rank);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(A,&ranges);CHKERRQ(ierr);
  for (i=0,nranks=0; i<size; i++) {
    if (ranges[i+1] - ranges[i] > 0) nranks++;
  }
  if (nranks == size) {         /* All ranks have a positive number of rows, so we do not need to create a subcomm; */
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    *B = A;
    PetscFunctionReturn(0);
  }

  ierr = PetscMalloc(nranks*sizeof(PetscMPIInt),&ranks);CHKERRQ(ierr);
  for (i=0,nranks=0; i<size; i++) {
    if (ranges[i+1] - ranges[i] > 0) ranks[nranks++] = i;
  }
  ierr = MPI_Comm_group(acomm,&agroup);CHKERRQ(ierr);
  ierr = MPI_Group_incl(agroup,nranks,ranks,&bgroup);CHKERRQ(ierr);
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  ierr = MPI_Comm_create(acomm,bgroup,&bcomm);CHKERRQ(ierr);
  ierr = MPI_Group_free(&agroup);CHKERRQ(ierr);
  ierr = MPI_Group_free(&bgroup);CHKERRQ(ierr);
  if (bcomm != MPI_COMM_NULL) {
    PetscInt   m,N;
    Mat_MPIAdj *b;
    ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
    ierr = MatCreateMPIAdj(bcomm,m,N,a->i,a->j,a->values,B);CHKERRQ(ierr);
    b = (Mat_MPIAdj*)(*B)->data;
    b->freeaij = PETSC_FALSE;
    ierr = MPI_Comm_free(&bcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIAdjCreateNonemptySubcommMat"
/*@
   MatMPIAdjCreateNonemptySubcommMat - create the same MPIAdj matrix on a subcommunicator containing only processes owning a positive number of rows

   Collective

   Input Arguments:
.  A - original MPIAdj matrix

   Output Arguments:
.  B - matrix on subcommunicator, PETSC_NULL on ranks that owned zero rows of A

   Level: developer

   Note:
   This function is mostly useful for internal use by mesh partitioning packages that require that every process owns at least one row.

   The matrix B should be destroyed with MatDestroy(). The arrays are not copied, so B should be destroyed before A is destroyed.

.seealso: MatCreateMPIAdj()
@*/
PetscErrorCode MatMPIAdjCreateNonemptySubcommMat(Mat A,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatMPIAdjCreateNonemptySubcommMat_C",(Mat,Mat*),(A,B));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATMPIADJ - MATMPIADJ = "mpiadj" - A matrix type to be used for distributed adjacency matrices,
   intended for use constructing orderings and partitionings.

  Level: beginner

.seealso: MatCreateMPIAdj
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIAdj"
PetscErrorCode  MatCreate_MPIAdj(Mat B)
{
  Mat_MPIAdj     *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                = PetscNewLog(B,Mat_MPIAdj,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr                = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->assembled        = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAdjSetPreallocation_C",
                                    "MatMPIAdjSetPreallocation_MPIAdj",
                                     MatMPIAdjSetPreallocation_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAdjCreateNonemptySubcommMat_C",
                                    "MatMPIAdjCreateNonemptySubcommMat_MPIAdj",
                                     MatMPIAdjCreateNonemptySubcommMat_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIADJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatMPIAdjSetPreallocation"
/*@C
   MatMPIAdjSetPreallocation - Sets the array used for storing the matrix elements

   Logically Collective on MPI_Comm

   Input Parameters:
+  A - the matrix
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).
-  values - [optional] edge weights

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAdj(), MatSetValues()
@*/
PetscErrorCode  MatMPIAdjSetPreallocation(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(B,"MatMPIAdjSetPreallocation_C",(Mat,PetscInt*,PetscInt*,PetscInt*),(B,i,j,values));CHKERRQ(ierr);
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
PetscErrorCode  MatCreateMPIAdj(MPI_Comm comm,PetscInt m,PetscInt N,PetscInt *i,PetscInt *j,PetscInt *values,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,PETSC_DETERMINE,PETSC_DETERMINE,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATMPIADJ);CHKERRQ(ierr);
  ierr = MatMPIAdjSetPreallocation(*A,i,j,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







