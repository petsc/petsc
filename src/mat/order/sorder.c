/*$Id: sorder.c,v 1.78 2000/09/13 15:20:10 balay Exp balay $*/
/*
     Provides the code that allows PETSc users to register their own
  sequential matrix Ordering routines.
*/
#include "src/mat/matimpl.h"
#include "petscsys.h"

FList      MatOrderingList = 0;
PetscTruth MatOrderingRegisterAllCalled = PETSC_FALSE;

EXTERN int MatOrdering_Flow_SeqAIJ(Mat,MatOrderingType,IS *,IS *);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatOrdering_Flow"
int MatOrdering_Flow(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  int ierr;

  PetscFunctionBegin;
  if (mat->type == MATSEQAIJ) {
    ierr = MatOrdering_Flow_SeqAIJ(mat,type,irow,icol);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  SETERRQ(PETSC_ERR_SUP,0,"Cannot do default flow ordering for matrix type");
#if !defined(PETSC_USE_DEBUG)
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatOrdering_Natural"
int MatOrdering_Natural(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  int        n,size,ierr,i,*ii;
  MPI_Comm   comm;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);

  if (mat->type == MATMPIROWBS || mat->type == MATSEQBDIAG || mat->type == MATMPIBDIAG) {
    int start,end;
    /*
        BlockSolve Format doesn't really require the Ordering,but PETSc wants
       to provide it to everyone.
    */
    ierr = MatGetOwnershipRange(mat,&start,&end);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,end-start,start,1,irow);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,end-start,start,1,icol);CHKERRQ(ierr);
    ierr = ISSetIdentity(*irow);CHKERRQ(ierr);
    ierr = ISSetIdentity(*icol);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
    
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    SETERRQ(PETSC_ERR_SUP,0,"Currently only for 1 processor matrices");
  }

  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,&n,PETSC_NULL,PETSC_NULL,&done);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,&n,PETSC_NULL,PETSC_NULL,&done);CHKERRQ(ierr);

  /*
    We actually create general index sets because this avoids mallocs to
    to obtain the indices in the MatSolve() routines.
    ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,irow);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,icol);CHKERRQ(ierr);
  */
  ii = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(ii);
  for (i=0; i<n; i++) ii[i] = i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,irow);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,icol);CHKERRQ(ierr);
  ierr = PetscFree(ii);CHKERRQ(ierr);

  ierr = ISSetIdentity(*irow);CHKERRQ(ierr);
  ierr = ISSetIdentity(*icol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/*
     Orders the rows (and columns) by the lengths of the rows. 
   This produces a symmetric Ordering but does not require a 
   matrix with symmetric non-zero structure.
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatOrdering_RowLength"
int MatOrdering_RowLength(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  int        ierr,n,*ia,*ja,*permr,*lens,i;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,&n,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,0,"Cannot get rows for matrix");

  lens  = (int*)PetscMalloc(2*n*sizeof(int));CHKPTRQ(lens);
  permr = lens + n;
  for (i=0; i<n; i++) { 
    lens[i]  = ia[i+1] - ia[i];
    permr[i] = i;
  }
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,&n,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = PetscSortIntWithPermutation(n,lens,permr);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,irow);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,icol);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MatOrderingRegisterDynamic - Adds a new sparse matrix ordering to the 
                               matrix package. 

   Synopsis:
   MatOrderingRegisterDynamic(char *name_ordering,char *path,char *name_create,int (*routine_create)(MatOrdering))

   Not Collective

   Input Parameters:
+  sname - name of ordering (for example MATORDERING_ND)
.  path - location of library where creation routine is 
.  name - name of function that creates the ordering type,a string
-  function - function pointer that creates the ordering

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatOrderingRegisterDynamic("my_order",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyOrder",MyOrder);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatOrderingSetType(part,"my_order)
   or at runtime via the option
$     -pc_ilu_mat_ordering_type my_order
$     -pc_lu_mat_ordering_type my_order

   $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

.keywords: matrix, ordering, register

.seealso: MatOrderingRegisterDestroy(), MatOrderingRegisterAll()
M*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatOrderingRegister" 
int MatOrderingRegister(char *sname,char *path,char *name,int (*function)(Mat,MatOrderingType,IS*,IS*))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&MatOrderingList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatOrderingRegisterDestroy"
/*@C
   MatOrderingRegisterDestroy - Frees the list of ordering routines.

   Not collective

   Level: developer
   
.keywords: matrix, register, destroy

.seealso: MatOrderingRegisterDynamic(), MatOrderingRegisterAll()
@*/
int MatOrderingRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatOrderingList) {
    ierr = FListDestroy(&MatOrderingList);CHKERRQ(ierr);
    MatOrderingList = 0;
  }
  PetscFunctionReturn(0);
}

EXTERN int MatAdjustForInodes(Mat,IS *,IS *);

#include "src/mat/impls/aij/mpi/mpiaij.h"
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetOrdering"
/*@C
   MatGetOrdering - Gets a reordering for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - type of reordering, one of the following:
$      MATORDERING_NATURAL - Natural
$      MATORDERING_ND - Nested Dissection
$      MATORDERING_1WD - One-way Dissection
$      MATORDERING_RCM - Reverse Cuthill-McKee
$      MATORDERING_QMD - Quotient Minimum Degree

   Output Parameters:
+  rperm - row permutation indices
-  cperm - column permutation indices


   Level: intermediate

   The user can define additional orderings; see MatOrderingRegisterDynamic().

.keywords: matrix, set, ordering, factorization, direct, ILU, LU,
           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McKee, 
           Quotient Minimum Degree

.seealso:   MatOrderingRegisterDynamic()
@*/
int MatGetOrdering(Mat mat,MatOrderingType type,IS *rperm,IS *cperm)
{
  int         ierr,mmat,nmat,mis;
  int         (*r)(Mat,MatOrderingType,IS*,IS*);
  PetscTruth  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  if (mat->type == MATSEQDENSE || mat->type == MATMPIDENSE) { 
    /*
       Dense matrices don't need the ordering
    */
    *rperm = *cperm = 0; PetscFunctionReturn(0);
  }

  if (!mat->M) {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,cperm);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,rperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*cperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!MatOrderingRegisterAllCalled) {
    ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PLogEventBegin(MAT_GetOrdering,mat,0,0,0);CHKERRQ(ierr);
  ierr =  FListFind(mat->comm,MatOrderingList,type,(int (**)(void *)) &r);CHKERRQ(ierr);
  if (!r) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown or unregistered type: %s",type);}

  ierr = (*r)(mat,type,rperm,cperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);

  /*
      Adjust for inode (reduced matrix ordering) only if row permutation
    is smaller then matrix size
  */
  ierr = MatGetLocalSize(mat,&mmat,&nmat);CHKERRQ(ierr);
  ierr = ISGetLocalSize(*rperm,&mis);CHKERRQ(ierr);
  if (mmat > mis) {  
    ierr = MatAdjustForInodes(mat,rperm,cperm);CHKERRQ(ierr);
  }

  ierr = PLogEventEnd(MAT_GetOrdering,mat,0,0,0);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-mat_view_ordering_draw",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat tmat;
    ierr = OptionsHasName(PETSC_NULL,"-mat_view_contour",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = ViewerPushFormat(VIEWER_DRAW_(mat->comm),VIEWER_FORMAT_DRAW_CONTOUR,0);CHKERRQ(ierr);
    }
    ierr = MatPermute(mat,*rperm,*cperm,&tmat);CHKERRQ(ierr);
    ierr = MatView(tmat,VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    if (flg) {
      ierr = ViewerPopFormat(VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    }
    ierr = MatDestroy(tmat);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetOrderingList"
int MatGetOrderingList(FList *list)
{
  PetscFunctionBegin;
  *list = MatOrderingList;
  PetscFunctionReturn(0);
}
