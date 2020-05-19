
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
  if (M > 0) PetscValidLogicalCollectiveInt(A,M,4);
  if (N > 0) PetscValidLogicalCollectiveInt(A,N,5);
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

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)B);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatXAIJSetPreallocation - set preallocation for serial and parallel AIJ, BAIJ, and SBAIJ matrices and their unassembled versions.

   Collective on Mat

   Input Arguments:
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
  PetscErrorCode ierr;
  PetscInt       refct;
  PetscOps       Abops;
  struct _MatOps Aops;
  char           *mtype,*mname,*mprefix;
  Mat_Product    *product;

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
  mprefix = ((PetscObject)A)->prefix;
  product = A->product;

  /* zero these so the destroy below does not free them */
  ((PetscObject)A)->type_name = 0;
  ((PetscObject)A)->name      = 0;

  /* free all the interior data structures from mat */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);

  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&A->cmap);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&((PetscObject)A)->qlist);CHKERRQ(ierr);
  ierr = PetscObjectListDestroy(&((PetscObject)A)->olist);CHKERRQ(ierr);

  /* copy C over to A */
  ierr = PetscMemcpy(A,*C,sizeof(struct _p_Mat));CHKERRQ(ierr);

  /* return the parts of A we saved */
  ((PetscObject)A)->bops[0]   = Abops;
  A->ops[0]                   = Aops;
  ((PetscObject)A)->refct     = refct;
  ((PetscObject)A)->type_name = mtype;
  ((PetscObject)A)->name      = mname;
  ((PetscObject)A)->prefix    = mprefix;
  A->product                  = product;

  /* since these two are copied into A we do not want them destroyed in C */
  ((PetscObject)*C)->qlist = 0;
  ((PetscObject)*C)->olist = 0;

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(*C,MAT_CLASSID,2);
  if (A == *C) PetscFunctionReturn(0);
  PetscCheckSameComm(A,1,*C,2);
  if (((PetscObject)*C)->refct != 1) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Object C has refct %D > 1, would leave hanging reference",((PetscObject)*C)->refct);

  /* swap C and A */
  refct = ((PetscObject)A)->refct;
  state = ((PetscObject)A)->state;
  ierr  = PetscMemcpy(&buffer,A,sizeof(struct _p_Mat));CHKERRQ(ierr);
  ierr  = PetscMemcpy(A,*C,sizeof(struct _p_Mat));CHKERRQ(ierr);
  ierr  = PetscMemcpy(*C,&buffer,sizeof(struct _p_Mat));CHKERRQ(ierr);
  ((PetscObject)A)->refct = refct;
  ((PetscObject)A)->state = state + 1;

  ((PetscObject)*C)->refct = 1;
  ierr = MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))NULL);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     MatBindToCPU - marks a matrix to temporarily stay on the CPU and perform computations on the CPU

   Input Parameters:
+   A - the matrix
-   flg - bind to the CPU if value of PETSC_TRUE

   Level: intermediate
@*/
PetscErrorCode MatBindToCPU(Mat A,PetscBool flg)
{
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,flg,2);
  if (A->boundtocpu == flg) PetscFunctionReturn(0);
  A->boundtocpu = flg;
  if (A->ops->bindtocpu) {
    ierr = (*A->ops->bindtocpu)(A,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,flg,2);
  PetscFunctionReturn(0);
#endif
}
