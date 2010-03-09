#define PETSCMAT_DLL

#include "private/matimpl.h"       /*I "petscmat.h"  I*/

#if 0
#undef __FUNCT__  
#define __FUNCT__ "MatPublish_Base"
static PetscErrorCode MatPublish_Base(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatCreate"
/*@
   MatCreate - Creates a matrix where the type is determined
   from either a call to MatSetType() or from the options database
   with a call to MatSetFromOptions(). The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() or MatCreateMPIAIJ()
   if you do not set a type in the options database. If you never
   call MatSetType() or MatSetFromOptions() it will generate an 
   error when you try to use the matrix.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator
 
   Output Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateMPIAIJ()
.    -mat_type seqdense - dense type, uses MatCreateSeqDense()
.    -mat_type mpidense - dense type, uses MatCreateMPIDense()
.    -mat_type seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_type mpibaij  - block AIJ type, uses MatCreateMPIBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Notes:

   Level: beginner

   User manual sections:
+   sec_matcreate
-   chapter_matrices

.keywords: matrix, create

.seealso: MatCreateSeqAIJ(), MatCreateMPIAIJ(), 
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateSeqBAIJ(), MatCreateMPIBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateMPISBAIJ(),
          MatConvert()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate(MPI_Comm comm,Mat *A)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(A,2);

  *A = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,0,"Mat",comm,MatDestroy,MatView);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&B->cmap);CHKERRQ(ierr);
  B->preallocated  = PETSC_FALSE;
  *A               = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetSizes"
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

  Level: beginner

.seealso: MatGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetSizes(Mat A, PetscInt m, PetscInt n, PetscInt M, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1); 
  if (M > 0 && m > M) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Local column size %D cannot be larger than global column size %D",m,M);
  if (N > 0 && n > N) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Local row size %D cannot be larger than global row size %D",n,N);
  if (A->ops->setsizes) {
    /* Since this will not be set until the type has been set, this will NOT be called on the initial
       call of MatSetSizes() (which must be called BEFORE MatSetType() */
    ierr = (*A->ops->setsizes)(A,m,n,M,N);CHKERRQ(ierr);
  } else {
    if ((A->rmap->n >= 0 || A->rmap->N >= 0) && (A->rmap->n != m || A->rmap->N != M)) SETERRQ4(PETSC_ERR_SUP,"Cannot change/reset row sizes to %D local %D global after previously setting them to %D local %D global",m,M,A->rmap->n,A->rmap->N);
    if ((A->cmap->n >= 0 || A->cmap->N >= 0) && (A->cmap->n != n || A->cmap->N != N)) SETERRQ4(PETSC_ERR_SUP,"Cannot change/reset column sizes to %D local %D global after previously setting them to %D local %D global",n,N,A->cmap->n,A->cmap->N);
  }
  A->rmap->n = m;
  A->cmap->n = n;
  A->rmap->N = M;
  A->cmap->N = N;
  if (A->ops->create) {
    ierr = (*A->ops->create)(A);CHKERRQ(ierr);
    A->ops->create = 0;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetFromOptions"
/*@
   MatSetFromOptions - Creates a matrix where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.  The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() and MatCreateMPIAIJ() if
   you do not select a type in the options database.

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateMPIAIJ()
.    -mat_type seqdense - dense type, uses MatCreateSeqDense()
.    -mat_type mpidense - dense type, uses MatCreateMPIDense()
.    -mat_type seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_type mpibaij  - block AIJ type, uses MatCreateMPIBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Level: beginner

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateSeqBAIJ(), MatCreateMPIBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateMPISBAIJ(),
          MatConvert()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetFromOptions(Mat B)
{
  PetscErrorCode ierr;
  const char     *deft = MATAIJ;
  char           type[256];
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_COOKIE,1);

  ierr = PetscOptionsBegin(((PetscObject)B)->comm,((PetscObject)B)->prefix,"Matrix options","Mat");CHKERRQ(ierr); 
    ierr = PetscOptionsList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSetType(B,type);CHKERRQ(ierr);
    } else if (!((PetscObject)B)->type_name) {
      ierr = MatSetType(B,deft);CHKERRQ(ierr);
    }

    if (B->ops->setfromoptions) {
      ierr = (*B->ops->setfromoptions)(B);CHKERRQ(ierr);
    }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation"
/*@
   MatSetUpPreallocation

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Level: beginner

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateSeqBAIJ(), MatCreateMPIBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateMPISBAIJ(),
          MatConvert()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetUpPreallocation(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!B->preallocated && B->ops->setuppreallocation) {
    ierr = PetscInfo(B,"Warning not preallocating matrix storage\n");CHKERRQ(ierr);
    ierr = (*B->ops->setuppreallocation)(B);CHKERRQ(ierr);
  }
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
        Copies from Cs header to A

        This is somewhat different from MatHeaderReplace() it would be nice to merge the code
*/
#undef __FUNCT__  
#define __FUNCT__ "MatHeaderCopy"
PetscErrorCode MatHeaderCopy(Mat A,Mat C)
{
  PetscErrorCode ierr;
  PetscInt       refct;
  PetscOps       *Abops;
  MatOps         Aops;
  char           *mtype,*mname;
  void           *spptr;

  PetscFunctionBegin;
  /* save the parts of A we need */
  Abops = ((PetscObject)A)->bops;
  Aops  = A->ops;
  refct = ((PetscObject)A)->refct;
  mtype = ((PetscObject)A)->type_name;
  mname = ((PetscObject)A)->name;
  spptr = A->spptr;

  /* zero these so the destroy below does not free them */
  ((PetscObject)A)->type_name = 0;
  ((PetscObject)A)->name      = 0;

  /* free all the interior data structures from mat */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);

  ierr = PetscFree(C->spptr);CHKERRQ(ierr);

  ierr = PetscLayoutDestroy(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(A->cmap);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&((PetscObject)A)->qlist);CHKERRQ(ierr);
  ierr = PetscOListDestroy(((PetscObject)A)->olist);CHKERRQ(ierr);

  /* copy C over to A */
  ierr  = PetscMemcpy(A,C,sizeof(struct _p_Mat));CHKERRQ(ierr);

  /* return the parts of A we saved */
  ((PetscObject)A)->bops      = Abops;
  A->ops                      = Aops;
  ((PetscObject)A)->refct     = refct;
  ((PetscObject)A)->type_name = mtype;
  ((PetscObject)A)->name      = mname;
  A->spptr                    = spptr;

  /* since these two are copied into A we do not want them destroyed in C */
  ((PetscObject)C)->qlist = 0;
  ((PetscObject)C)->olist = 0;
  ierr = PetscHeaderDestroy(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
        Replace A's header with that of C
        This is essentially code moved from MatDestroy

        This is somewhat different from MatHeaderCopy() it would be nice to merge the code
*/
#undef __FUNCT__  
#define __FUNCT__ "MatHeaderReplace"
PetscErrorCode MatHeaderReplace(Mat A,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A == C) PetscFunctionReturn(0);

  /* free all the interior data structures from mat */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy_Private((PetscObject)A);CHKERRQ(ierr);
  ierr = PetscFree(A->ops);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(A->cmap);CHKERRQ(ierr);
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  
  /* copy C over to A */
  if (C) {
    ierr = PetscMemcpy(A,C,sizeof(struct _p_Mat));CHKERRQ(ierr);
    ierr = PetscLogObjectDestroy((PetscObject)C);CHKERRQ(ierr);
    ierr = PetscFree(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
