
#include <petsc/private/matimpl.h>       /*I "petscmat.h"  I*/

PETSC_INTERN PetscErrorCode MatSetBlockSizes_Default(Mat mat,PetscInt rbs, PetscInt cbs)
{
  PetscFunctionBegin;
  if (!mat->preallocated) PetscFunctionReturn(0);
  if (mat->rmap->bs > 0 && mat->rmap->bs != rbs) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot change row block size %D to %D\n",mat->rmap->bs,rbs);
  if (mat->cmap->bs > 0 && mat->cmap->bs != cbs) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot change column block size %D to %D\n",mat->cmap->bs,cbs);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatShift_Basic(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  PetscInt       i,start,end;
  PetscScalar    alpha = a;
  PetscBool      prevoption;

  PetscFunctionBegin;
  ierr = MatGetOption(Y,MAT_NO_OFF_PROC_ENTRIES,&prevoption);CHKERRQ(ierr);
  ierr = MatSetOption(Y,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    if (i < Y->cmap->N) {
      ierr = MatSetValues(Y,1,&i,1,&i,&alpha,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(Y,MAT_NO_OFF_PROC_ENTRIES,prevoption);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreate - Creates a matrix where the type is determined
   from either a call to MatSetType() or from the options database
   with a call to MatSetFromOptions(). The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() or MatCreateAIJ()
   if you do not set a type in the options database. If you never
   call MatSetType() or MatSetFromOptions() it will generate an
   error when you try to use the matrix.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateAIJ()
.    -mat_type seqdense - dense type, uses MatCreateSeqDense()
.    -mat_type mpidense - dense type, uses MatCreateDense()
.    -mat_type seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_type mpibaij  - block AIJ type, uses MatCreateBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Level: beginner

.seealso: MatCreateSeqAIJ(), MatCreateAIJ(),
          MatCreateSeqDense(), MatCreateDense(),
          MatCreateSeqBAIJ(), MatCreateBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateSBAIJ(),
          MatConvert()
@*/
PetscErrorCode  MatCreate(MPI_Comm comm,Mat *A)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(A,2);

  *A = NULL;
  ierr = MatInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(B,MAT_CLASSID,"Mat","Matrix","Mat",comm,MatDestroy,MatView);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&B->cmap);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECSTANDARD,&B->defaultvectype);CHKERRQ(ierr);

  B->congruentlayouts = PETSC_DECIDE;
  B->preallocated     = PETSC_FALSE;
#if defined(PETSC_HAVE_DEVICE)
  B->boundtocpu       = PETSC_TRUE;
#endif
  *A                  = B;
  PetscFunctionReturn(0);
}

/*@
   MatSetErrorIfFailure - Causes Mat to generate an error, for example a zero pivot, is detected.

   Logically Collective on Mat

   Input Parameters:
+  mat -  matrix obtained from MatCreate()
-  flg - PETSC_TRUE indicates you want the error generated

   Level: advanced

.seealso: PCSetErrorIfFailure()
@*/
PetscErrorCode  MatSetErrorIfFailure(Mat mat,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(mat,flg,2);
  mat->erroriffailure = flg;
  PetscFunctionReturn(0);
}

/*@
  MatSetSizes - Sets the local and global sizes, and checks to determine compatibility

  Collective on Mat

  Input Parameters:
+  A - the matrix
.  m - number of local rows (or PETSC_DECIDE)
.  n - number of local columns (or PETSC_DECIDE)
.  M - number of global rows (or PETSC_DETERMINE)
-  N - number of global columns (or PETSC_DETERMINE)

   Notes:
   m (n) and M (N) cannot be both PETSC_DECIDE
   If one processor calls this with M (N) of PETSC_DECIDE then all processors must, otherwise the program will hang.

   If PETSC_DECIDE is not used for the arguments 'm' and 'n', then the
   user must ensure that they are chosen to be compatible with the
   vectors. To do this, one first considers the matrix-vector product
   'y = A x'. The 'm' that is used in the above routine must match the
   local size used in the vector creation routine VecCreateMPI() for 'y'.
   Likewise, the 'n' used must match that used as the local size in
   VecCreateMPI() for 'x'.

   You cannot change the sizes once they have been set.

   The sizes must be set before MatSetUp() or MatXXXSetPreallocation() is called.

  Level: beginner

.seealso: MatGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode  MatSetSizes(Mat A, PetscInt m, PetscInt n, PetscInt M, PetscInt N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(A,M,4);
  PetscValidLogicalCollectiveInt(A,N,5);
  if (M > 0 && m > M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local row size %D cannot be larger than global row size %D",m,M);
  if (N > 0 && n > N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local column size %D cannot be larger than global column size %D",n,N);
  if ((A->rmap->n >= 0 && A->rmap->N >= 0) && (A->rmap->n != m || (M > 0 && A->rmap->N != M))) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset row sizes to %D local %D global after previously setting them to %D local %D global",m,M,A->rmap->n,A->rmap->N);
  if ((A->cmap->n >= 0 && A->cmap->N >= 0) && (A->cmap->n != n || (N > 0 && A->cmap->N != N))) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset column sizes to %D local %D global after previously setting them to %D local %D global",n,N,A->cmap->n,A->cmap->N);
  A->rmap->n = m;
  A->cmap->n = n;
  A->rmap->N = M > -1 ? M : A->rmap->N;
  A->cmap->N = N > -1 ? N : A->cmap->N;
  PetscFunctionReturn(0);
}

/*@
   MatSetFromOptions - Creates a matrix where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.  The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() and MatCreateAIJ() if
   you do not select a type in the options database.

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateAIJ()
.    -mat_type seqdense - dense type, uses MatCreateSeqDense()
.    -mat_type mpidense - dense type, uses MatCreateDense()
.    -mat_type seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_type mpibaij  - block AIJ type, uses MatCreateBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Level: beginner

.seealso: MatCreateSeqAIJ((), MatCreateAIJ(),
          MatCreateSeqDense(), MatCreateDense(),
          MatCreateSeqBAIJ(), MatCreateBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateSBAIJ(),
          MatConvert()
@*/
PetscErrorCode  MatSetFromOptions(Mat B)
{
  PetscErrorCode ierr;
  const char     *deft = MATAIJ;
  char           type[256];
  PetscBool      flg,set;
  PetscInt       bind_below = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)B);CHKERRQ(ierr);

  if (B->rmap->bs < 0) {
    PetscInt newbs = -1;
    ierr = PetscOptionsInt("-mat_block_size","Set the blocksize used to store the matrix","MatSetBlockSize",newbs,&newbs,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscLayoutSetBlockSize(B->rmap,newbs);CHKERRQ(ierr);
      ierr = PetscLayoutSetBlockSize(B->cmap,newbs);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsFList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetType(B,type);CHKERRQ(ierr);
  } else if (!((PetscObject)B)->type_name) {
    ierr = MatSetType(B,deft);CHKERRQ(ierr);
  }

  ierr = PetscOptionsName("-mat_is_symmetric","Checks if mat is symmetric on MatAssemblyEnd()","MatIsSymmetric",&B->checksymmetryonassembly);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_is_symmetric","Checks if mat is symmetric on MatAssemblyEnd()","MatIsSymmetric",B->checksymmetrytol,&B->checksymmetrytol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_null_space_test","Checks if provided null space is correct in MatAssemblyEnd()","MatSetNullSpaceTest",B->checknullspaceonassembly,&B->checknullspaceonassembly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_error_if_failure","Generate an error if an error occurs when factoring the matrix","MatSetErrorIfFailure",B->erroriffailure,&B->erroriffailure,NULL);CHKERRQ(ierr);

  if (B->ops->setfromoptions) {
    ierr = (*B->ops->setfromoptions)(PetscOptionsObject,B);CHKERRQ(ierr);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_new_nonzero_location_err","Generate an error if new nonzeros are created in the matrix structure (useful to test preallocation)","MatSetOption",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,flg);CHKERRQ(ierr);}
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_new_nonzero_allocation_err","Generate an error if new nonzeros are allocated in the matrix structure (useful to test preallocation)","MatSetOption",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,flg);CHKERRQ(ierr);}
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_ignore_zero_entries","For AIJ/IS matrices this will stop zero values from creating a zero location in the matrix","MatSetOption",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = MatSetOption(B,MAT_IGNORE_ZERO_ENTRIES,flg);CHKERRQ(ierr);}

  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_form_explicit_transpose","Hint to form an explicit transpose for operations like MatMultTranspose","MatSetOption",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = MatSetOption(B,MAT_FORM_EXPLICIT_TRANSPOSE,flg);CHKERRQ(ierr);}

  /* Bind to CPU if below a user-specified size threshold.
   * This perhaps belongs in the options for the GPU Mat types, but MatBindToCPU() does nothing when called on non-GPU types,
   * and putting it here makes is more maintainable than duplicating this for all. */
  ierr = PetscOptionsInt("-mat_bind_below","Set the size threshold (in local rows) below which the Mat is bound to the CPU","MatBindToCPU",bind_below,&bind_below,&flg);CHKERRQ(ierr);
  if (flg && B->rmap->n < bind_below) {
    ierr = MatBindToCPU(B,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)B);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatXAIJSetPreallocation - set preallocation for serial and parallel AIJ, BAIJ, and SBAIJ matrices and their unassembled versions.

   Collective on Mat

   Input Parameters:
+  A - matrix being preallocated
.  bs - block size
.  dnnz - number of nonzero column blocks per block row of diagonal part of parallel matrix
.  onnz - number of nonzero column blocks per block row of off-diagonal part of parallel matrix
.  dnnzu - number of nonzero column blocks per block row of upper-triangular part of diagonal part of parallel matrix
-  onnzu - number of nonzero column blocks per block row of upper-triangular part of off-diagonal part of parallel matrix

   Level: beginner

.seealso: MatSeqAIJSetPreallocation(), MatMPIAIJSetPreallocation(), MatSeqBAIJSetPreallocation(), MatMPIBAIJSetPreallocation(), MatSeqSBAIJSetPreallocation(), MatMPISBAIJSetPreallocation(),
          PetscSplitOwnership()
@*/
PetscErrorCode MatXAIJSetPreallocation(Mat A,PetscInt bs,const PetscInt dnnz[],const PetscInt onnz[],const PetscInt dnnzu[],const PetscInt onnzu[])
{
  PetscErrorCode ierr;
  PetscInt       cbs;
  void           (*aij)(void);
  void           (*is)(void);
  void           (*hyp)(void) = NULL;

  PetscFunctionBegin;
  if (bs != PETSC_DECIDE) { /* don't mess with an already set block size */
    ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  }
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = MatGetBlockSizes(A,&bs,&cbs);CHKERRQ(ierr);
  /* these routines assumes bs == cbs, this should be checked somehow */
  ierr = MatSeqBAIJSetPreallocation(A,bs,0,dnnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,bs,0,dnnz,0,onnz);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,bs,0,dnnzu);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,bs,0,dnnzu,0,onnzu);CHKERRQ(ierr);
  /*
    In general, we have to do extra work to preallocate for scalar (AIJ) or unassembled (IS) matrices so we check whether it will do any
    good before going on with it.
  */
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatISSetPreallocation_C",&is);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatHYPRESetPreallocation_C",&hyp);CHKERRQ(ierr);
#endif
  if (!aij && !is && !hyp) {
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  }
  if (aij || is || hyp) {
    if (bs == cbs && bs == 1) {
      ierr = MatSeqAIJSetPreallocation(A,0,dnnz);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(A,0,dnnz,0,onnz);CHKERRQ(ierr);
      ierr = MatISSetPreallocation(A,0,dnnz,0,onnz);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
      ierr = MatHYPRESetPreallocation(A,0,dnnz,0,onnz);CHKERRQ(ierr);
#endif
    } else { /* Convert block-row precallocation to scalar-row */
      PetscInt i,m,*sdnnz,*sonnz;
      ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc2((!!dnnz)*m,&sdnnz,(!!onnz)*m,&sonnz);CHKERRQ(ierr);
      for (i=0; i<m; i++) {
        if (dnnz) sdnnz[i] = dnnz[i/bs] * cbs;
        if (onnz) sonnz[i] = onnz[i/bs] * cbs;
      }
      ierr = MatSeqAIJSetPreallocation(A,0,dnnz ? sdnnz : NULL);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(A,0,dnnz ? sdnnz : NULL,0,onnz ? sonnz : NULL);CHKERRQ(ierr);
      ierr = MatISSetPreallocation(A,0,dnnz ? sdnnz : NULL,0,onnz ? sonnz : NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
      ierr = MatHYPRESetPreallocation(A,0,dnnz ? sdnnz : NULL,0,onnz ? sonnz : NULL);CHKERRQ(ierr);
#endif
      ierr = PetscFree2(sdnnz,sonnz);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
        Merges some information from Cs header to A; the C object is then destroyed

        This is somewhat different from MatHeaderReplace() it would be nice to merge the code
*/
PetscErrorCode MatHeaderMerge(Mat A,Mat *C)
{
  PetscErrorCode   ierr;
  PetscInt         refct;
  PetscOps         Abops;
  struct _MatOps   Aops;
  char             *mtype,*mname,*mprefix;
  Mat_Product      *product;
  PetscObjectState state;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(*C,MAT_CLASSID,2);
  if (A == *C) PetscFunctionReturn(0);
  PetscCheckSameComm(A,1,*C,2);
  /* save the parts of A we need */
  Abops = ((PetscObject)A)->bops[0];
  Aops  = A->ops[0];
  refct = ((PetscObject)A)->refct;
  mtype = ((PetscObject)A)->type_name;
  mname = ((PetscObject)A)->name;
  state = ((PetscObject)A)->state;
  mprefix = ((PetscObject)A)->prefix;
  product = A->product;

  /* zero these so the destroy below does not free them */
  ((PetscObject)A)->type_name = NULL;
  ((PetscObject)A)->name      = NULL;

  /* free all the interior data structures from mat */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);

  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&A->cmap);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&((PetscObject)A)->qlist);CHKERRQ(ierr);
  ierr = PetscObjectListDestroy(&((PetscObject)A)->olist);CHKERRQ(ierr);
  ierr = PetscComposedQuantitiesDestroy((PetscObject)A);CHKERRQ(ierr);

  /* copy C over to A */
  ierr = PetscMemcpy(A,*C,sizeof(struct _p_Mat));CHKERRQ(ierr);

  /* return the parts of A we saved */
  ((PetscObject)A)->bops[0]   = Abops;
  A->ops[0]                   = Aops;
  ((PetscObject)A)->refct     = refct;
  ((PetscObject)A)->type_name = mtype;
  ((PetscObject)A)->name      = mname;
  ((PetscObject)A)->prefix    = mprefix;
  ((PetscObject)A)->state     = state + 1;
  A->product                  = product;

  /* since these two are copied into A we do not want them destroyed in C */
  ((PetscObject)*C)->qlist = NULL;
  ((PetscObject)*C)->olist = NULL;

  ierr = PetscHeaderDestroy(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
        Replace A's header with that of C; the C object is then destroyed

        This is essentially code moved from MatDestroy()

        This is somewhat different from MatHeaderMerge() it would be nice to merge the code

        Used in DM hence is declared PETSC_EXTERN
*/
PETSC_EXTERN PetscErrorCode MatHeaderReplace(Mat A,Mat *C)
{
  PetscErrorCode   ierr;
  PetscInt         refct;
  PetscObjectState state;
  struct _p_Mat    buffer;
  MatStencilInfo   stencil;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(*C,MAT_CLASSID,2);
  if (A == *C) PetscFunctionReturn(0);
  PetscCheckSameComm(A,1,*C,2);
  if (((PetscObject)*C)->refct != 1) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Object C has refct %D > 1, would leave hanging reference",((PetscObject)*C)->refct);

  /* swap C and A */
  refct   = ((PetscObject)A)->refct;
  state   = ((PetscObject)A)->state;
  stencil = A->stencil;
  ierr  = PetscMemcpy(&buffer,A,sizeof(struct _p_Mat));CHKERRQ(ierr);
  ierr  = PetscMemcpy(A,*C,sizeof(struct _p_Mat));CHKERRQ(ierr);
  ierr  = PetscMemcpy(*C,&buffer,sizeof(struct _p_Mat));CHKERRQ(ierr);
  ((PetscObject)A)->refct   = refct;
  ((PetscObject)A)->state   = state + 1;
  A->stencil                = stencil;

  ((PetscObject)*C)->refct = 1;
  ierr = MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))NULL);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     MatBindToCPU - marks a matrix to temporarily stay on the CPU and perform computations on the CPU

   Logically collective on Mat

   Input Parameters:
+   A - the matrix
-   flg - bind to the CPU if value of PETSC_TRUE

   Level: intermediate

.seealso: MatBoundToCPU()
@*/
PetscErrorCode MatBindToCPU(Mat A,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,flg,2);
#if defined(PETSC_HAVE_DEVICE)
  if (A->boundtocpu == flg) PetscFunctionReturn(0);
  A->boundtocpu = flg;
  if (A->ops->bindtocpu) {
    PetscErrorCode ierr;
    ierr = (*A->ops->bindtocpu)(A,flg);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*@
     MatBoundToCPU - query if a matrix is bound to the CPU

   Input Parameter:
.   A - the matrix

   Output Parameter:
.   flg - the logical flag

   Level: intermediate

.seealso: MatBindToCPU()
@*/
PetscErrorCode MatBoundToCPU(Mat A,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(flg,2);
#if defined(PETSC_HAVE_DEVICE)
  *flg = A->boundtocpu;
#else
  *flg = PETSC_TRUE;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValuesCOO_Basic(Mat A,const PetscScalar coo_v[],InsertMode imode)
{
  IS             is_coo_i,is_coo_j;
  const PetscInt *coo_i,*coo_j;
  PetscInt       n,n_i,n_j;
  PetscScalar    zero = 0.;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"__PETSc_coo_i",(PetscObject*)&is_coo_i);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"__PETSc_coo_j",(PetscObject*)&is_coo_j);CHKERRQ(ierr);
  if (!is_coo_i) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_COR,"Missing coo_i IS");
  if (!is_coo_j) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_COR,"Missing coo_j IS");
  ierr = ISGetLocalSize(is_coo_i,&n_i);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is_coo_j,&n_j);CHKERRQ(ierr);
  if (n_i != n_j)  SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"Wrong local size %D != %D",n_i,n_j);
  ierr = ISGetIndices(is_coo_i,&coo_i);CHKERRQ(ierr);
  ierr = ISGetIndices(is_coo_j,&coo_j);CHKERRQ(ierr);
  if (imode != ADD_VALUES) {
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
  }
  for (n = 0; n < n_i; n++) {
    ierr = MatSetValue(A,coo_i[n],coo_j[n],coo_v ? coo_v[n] : zero,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(is_coo_i,&coo_i);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_coo_j,&coo_j);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetPreallocationCOO_Basic(Mat A,PetscInt ncoo,const PetscInt coo_i[],const PetscInt coo_j[])
{
  Mat            preallocator;
  IS             is_coo_i,is_coo_j;
  PetscScalar    zero = 0.0;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&preallocator);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetLayouts(preallocator,A->rmap,A->cmap);CHKERRQ(ierr);
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);
  for (n = 0; n < ncoo; n++) {
    ierr = MatSetValue(preallocator,coo_i[n],coo_j[n],zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatPreallocatorPreallocate(preallocator,PETSC_TRUE,A);CHKERRQ(ierr);
  ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ncoo,coo_i,PETSC_COPY_VALUES,&is_coo_i);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ncoo,coo_j,PETSC_COPY_VALUES,&is_coo_j);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"__PETSc_coo_i",(PetscObject)is_coo_i);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"__PETSc_coo_j",(PetscObject)is_coo_j);CHKERRQ(ierr);
  ierr = ISDestroy(&is_coo_i);CHKERRQ(ierr);
  ierr = ISDestroy(&is_coo_j);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatSetPreallocationCOO - set preallocation for matrices using a coordinate format of the entries

   Collective on Mat

   Input Parameters:
+  A - matrix being preallocated
.  ncoo - number of entries in the locally owned part of the parallel matrix
.  coo_i - row indices
-  coo_j - column indices

   Level: beginner

   Notes: Entries can be repeated, see MatSetValuesCOO(). Currently optimized for cuSPARSE matrices only.

.seealso: MatSetValuesCOO(), MatSeqAIJSetPreallocation(), MatMPIAIJSetPreallocation(), MatSeqBAIJSetPreallocation(), MatMPIBAIJSetPreallocation(), MatSeqSBAIJSetPreallocation(), MatMPISBAIJSetPreallocation()
@*/
PetscErrorCode MatSetPreallocationCOO(Mat A,PetscInt ncoo,const PetscInt coo_i[],const PetscInt coo_j[])
{
  PetscErrorCode (*f)(Mat,PetscInt,const PetscInt[],const PetscInt[]) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (ncoo) PetscValidIntPointer(coo_i,3);
  if (ncoo) PetscValidIntPointer(coo_j,4);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt i;
    for (i = 0; i < ncoo; i++) {
      if (coo_i[i] < A->rmap->rstart || coo_i[i] >= A->rmap->rend) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid row index %D! Must be in [%D,%D)",coo_i[i],A->rmap->rstart,A->rmap->rend);
      if (coo_j[i] < 0 || coo_j[i] >= A->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid col index %D! Must be in [0,%D)",coo_j[i],A->cmap->N);
    }
  }
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSetPreallocationCOO_C",&f);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_PreallCOO,A,0,0,0);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,ncoo,coo_i,coo_j);CHKERRQ(ierr);
  } else { /* allow fallback, very slow */
    ierr = MatSetPreallocationCOO_Basic(A,ncoo,coo_i,coo_j);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_PreallCOO,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatSetValuesCOO - set values at once in a matrix preallocated using MatSetPreallocationCOO()

   Collective on Mat

   Input Parameters:
+  A - matrix being preallocated
.  coo_v - the matrix values (can be NULL)
-  imode - the insert mode

   Level: beginner

   Notes: The values must follow the order of the indices prescribed with MatSetPreallocationCOO().
          When repeated entries are specified in the COO indices the coo_v values are first properly summed.
          The imode flag indicates if coo_v must be added to the current values of the matrix (ADD_VALUES) or overwritten (INSERT_VALUES).
          Currently optimized for cuSPARSE matrices only.
          Passing coo_v == NULL is equivalent to passing an array of zeros.

.seealso: MatSetPreallocationCOO(), InsertMode, INSERT_VALUES, ADD_VALUES
@*/
PetscErrorCode MatSetValuesCOO(Mat A, const PetscScalar coo_v[], InsertMode imode)
{
  PetscErrorCode (*f)(Mat,const PetscScalar[],InsertMode) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  MatCheckPreallocated(A,1);
  PetscValidLogicalCollectiveEnum(A,imode,3);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSetValuesCOO_C",&f);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_SetVCOO,A,0,0,0);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,coo_v,imode);CHKERRQ(ierr);
  } else { /* allow fallback */
    ierr = MatSetValuesCOO_Basic(A,coo_v,imode);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_SetVCOO,A,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatSetBindingPropagates - Sets whether the state of being bound to the CPU for a GPU matrix type propagates to child and some other associated objects

   Input Parameters:
+  A - the matrix
-  flg - flag indicating whether the boundtocpu flag should be propagated

   Level: developer

   Notes:
   If the value of flg is set to true, the following will occur:

   MatCreateSubMatrices() and MatCreateRedundantMatrix() will bind created matrices to CPU if the input matrix is bound to the CPU.
   MatCreateVecs() will bind created vectors to CPU if the input matrix is bound to the CPU.
   The bindingpropagates flag itself is also propagated by the above routines.

   Developer Notes:
   If the fine-scale DMDA has the -dm_bind_below option set to true, then DMCreateInterpolationScale() calls MatSetBindingPropagates()
   on the restriction/interpolation operator to set the bindingpropagates flag to true.

.seealso: VecSetBindingPropagates(), MatGetBindingPropagates()
@*/
PetscErrorCode MatSetBindingPropagates(Mat A,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  A->bindingpropagates = flg;
#endif
  PetscFunctionReturn(0);
}

/*@
   MatGetBindingPropagates - Gets whether the state of being bound to the CPU for a GPU matrix type propagates to child and some other associated objects

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  flg - flag indicating whether the boundtocpu flag will be propagated

   Level: developer

.seealso: MatSetBindingPropagates()
@*/
PetscErrorCode MatGetBindingPropagates(Mat A,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(flg,2);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  *flg = A->bindingpropagates;
#else
  *flg = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}
