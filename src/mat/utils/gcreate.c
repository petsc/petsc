/*$Id: gcreate.c,v 1.123 2000/09/25 19:57:36 curfman Exp bsmith $*/

#include "petscsys.h"
#include "src/mat/matimpl.h"       /*I "petscmat.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatCreate"
/*@C
   MatCreate - Creates a matrix where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.  The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() and MatCreateMPIAIJ(). 

   Collective on MPI_Comm

   Input Parameters:
+  m - number of local rows (or PETSC_DECIDE)
.  n - number of local columns (or PETSC_DECIDE)
.  M - number of global rows (or PETSC_DETERMINE)
.  N - number of global columns (or PETSC_DETERMINE)
-  comm - MPI communicator
 
   Output Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateMPIAIJ()
.    -mat_type seqbdiag - block diagonal type, uses MatCreateSeqBDiag()
.    -mat_type mpibdiag - block diagonal type, uses MatCreateMPIBDiag()
.    -mat_type mpirowbs - rowbs type, uses MatCreateMPIRowbs()
.    -mat_type seqdense - dense type, uses MatCreateSeqDense()
.    -mat_type mpidense - dense type, uses MatCreateMPIDense()
.    -mat_type seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_type mpibaij  - block AIJ type, uses MatCreateMPIBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Notes:
   If PETSC_DECIDE is not used for the arguments 'm' and 'n', then the
   user must ensure that they are chosen to be compatible with the
   vectors. To do this, one first considers the matrix-vector product 
   'y = A x'. The 'm' that is used in the above routine must match the 
   local size used in the vector creation routine VecCreateMPI() for 'y'.
   Likewise, the 'n' used must match that used as the local size in
   VecCreateMPI() for 'x'.

   Level: beginner

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateMPIRowbs(), MatCreateSeqBAIJ(), MatCreateMPIBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateMPISBAIJ(),
          MatConvert()
@*/
int MatCreate(MPI_Comm comm,int m,int n,int M,int N,Mat *A)
{
  Mat B;

  PetscFunctionBegin;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,0,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);

  B->m = m;
  B->n = n;
  B->M = M;
  B->N = N;

  B->preallocated = PETSC_FALSE;

  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetFromOptions"
/*@C
   MatSetFromOptions - Creates a matrix where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.  The default matrix type is
   AIJ, using the routines MatSetFromOptionsSeqAIJ() and MatSetFromOptionsMPIAIJ(). 

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateMPIAIJ()
.    -mat_type seqbdiag - block diagonal type, uses MatCreateSeqBDiag()
.    -mat_type mpibdiag - block diagonal type, uses MatCreateMPIBDiag()
.    -mat_type mpirowbs - rowbs type, uses MatCreateMPIRowbs()
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
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateMPIRowbs(), MatCreateSeqBAIJ(), MatCreateMPIBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateMPISBAIJ(),
          MatConvert()
@*/
int MatSetFromOptions(Mat B)
{
  int        ierr,size;
  char       mtype[256];
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsGetString(B->prefix,"-mat_type",mtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetType(B,mtype);CHKERRQ(ierr);
  }
  if (!B->type_name) {
    ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
    } else {
      ierr = MatSetType(B,MATMPIAIJ);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetUpPreallocation"
/*@C
   MatSetUpPreallocation

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Level: beginner

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateMPIRowbs(), MatCreateSeqBAIJ(), MatCreateMPIBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateMPISBAIJ(),
          MatConvert()
@*/
int MatSetUpPreallocation(Mat B)
{
  int        ierr;

  PetscFunctionBegin;
  if (B->ops->setuppreallocation) {
    PLogInfo(B,"MatSetTpPreallocation: Warning not preallocating matrix storage");
    ierr = (*B->ops->setuppreallocation)(B);CHKERRQ(ierr);
    B->ops->setuppreallocation = 0;
    B->preallocated            = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*
        Copies from Cs header to A
*/
#undef __FUNC__  
#define __FUNC__ /*<a name="MatHeaderCopy"></a>*/"MatHeaderCopy"
int MatHeaderCopy(Mat A,Mat C)
{
  int         ierr,refct;
  PetscOps    *Abops;
  MatOps      Aops;
  char        *mtype,*mname;

  PetscFunctionBegin;
  /* free all the interior data structures from mat */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);

  ierr = MapDestroy(A->rmap);CHKERRQ(ierr);
  ierr = MapDestroy(A->cmap);CHKERRQ(ierr);

  /* save the parts of A we need */
  Abops = A->bops;
  Aops  = A->ops;
  refct = A->refct;
  mtype = A->type_name;
  mname = A->name;

  /* copy C over to A */
  ierr  = PetscMemcpy(A,C,sizeof(struct _p_Mat));CHKERRQ(ierr);

  /* return the parts of A we saved */
  A->bops      = Abops;
  A->ops       = Aops;
  A->qlist     = 0;
  A->refct     = refct;
  A->type_name = mtype;
  A->name      = mname;

  PetscHeaderDestroy(C);
  PetscFunctionReturn(0);
}
