#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sorder.c,v 1.53 1999/01/13 21:41:17 bsmith Exp bsmith $";
#endif
/*
     Provides the code that allows PETSc users to register their own
  sequential matrix reordering routines.
*/
#include "src/mat/matimpl.h"
#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MatReorderingList = 0;
int  MatReorderingRegisterAllCalled = 0;

extern int MatOrder_Flow_SeqAIJ(Mat,MatReorderingType,IS *,IS *);

#undef __FUNC__  
#define __FUNC__ "MatOrder_Flow"
int MatOrder_Flow(Mat mat,MatReorderingType type,IS *irow,IS *icol)
{
  int ierr;

  PetscFunctionBegin;
  if (mat->type == MATSEQAIJ) {
    ierr = MatOrder_Flow_SeqAIJ(mat,type,irow,icol);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  SETERRQ(PETSC_ERR_SUP,0,"Cannot do default flow ordering for matrix type");
#if !defined(USE_PETSC_DEBUG)
  PetscFunctionReturn(0);
#endif
}

#undef __FUNC__  
#define __FUNC__ "MatOrder_Natural"
int MatOrder_Natural(Mat mat,MatReorderingType type,IS *irow,IS *icol)
{
  int        n, size,ierr,i,*ii;
  MPI_Comm   comm;
  PetscTruth done;

  PetscFunctionBegin;
  if (mat->type == MATMPIROWBS || mat->type == MATSEQBDIAG || mat->type == MATMPIBDIAG) {
    int start, end;
    /*
        BlockSolve Format doesn't really require the reordering, but PETSc wants
       to provide it to everyone.
    */
    ierr = MatGetOwnershipRange(mat,&start,&end); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,irow); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,icol); CHKERRQ(ierr);
    ierr = ISSetIdentity(*irow); CHKERRQ(ierr);
    ierr = ISSetIdentity(*icol); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
    
  ierr = PetscObjectGetComm((PetscObject)mat,&comm); CHKERRQ(ierr);
  MPI_Comm_size(comm,&size);

  if (size > 1) {
    SETERRQ(PETSC_ERR_SUP,0,"Currently only for 1 processor matrices");
  }

  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,&n,PETSC_NULL,PETSC_NULL,&done);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,&n,PETSC_NULL,PETSC_NULL,&done);CHKERRQ(ierr);

  /*
    We actually create general index sets because this avoids mallocs to
    to obtain the indices in the MatSolve() routines.
    ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,irow); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,icol); CHKERRQ(ierr);
  */
  ii = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) ii[i] = i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,irow); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,icol); CHKERRQ(ierr);
  PetscFree(ii);

  ierr = ISSetIdentity(*irow); CHKERRQ(ierr);
  ierr = ISSetIdentity(*icol); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Orders the rows (and columns) by the lengths of the rows. 
   This produces a symmetric reordering but does not require a 
   matrix with symmetric non-zero structure.
*/
#undef __FUNC__  
#define __FUNC__ "MatOrder_RowLength"
int MatOrder_RowLength(Mat mat,MatReorderingType type,IS *irow,IS *icol)
{
  int        ierr,n,*ia,*ja,*permr,*lens,i;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,&n,&ia,&ja,&done); CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,0,"Cannot get rows for matrix");

  lens  = (int *) PetscMalloc( 2*n*sizeof(int) ); CHKPTRQ(lens);
  permr = lens + n;
  for ( i=0; i<n; i++ ) { 
    lens[i]  = ia[i+1] - ia[i];
    permr[i] = i;
  }
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,&n,&ia,&ja,&done); CHKERRQ(ierr);

  ierr = PetscSortIntWithPermutation(n, lens, permr); CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,irow); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,icol); CHKERRQ(ierr);
  PetscFree(lens);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatReorderingRegister"
/*@C
   MatReorderingRegister - Adds a new sparse matrix reordering to the 
   matrix package. 

   Input Parameters:
.  name - name of ordering (if built-in) else ORDER_NEW
.  sname -  corresponding string for name
.  order - routine that does reordering

   Output Parameters:
.  out - number associated with the reordering 

   Not Collective

.keywords: matrix, reordering, register

.seealso: MatReorderingRegisterDestroy(), MatReorderingRegisterAll()
@*/
int  MatReorderingRegister(MatReorderingType name,MatReorderingType *out,char *sname,int (*order)(Mat,MatReorderingType,IS*,IS*))
{
  int         ierr;
  static int  numberregistered = 0;

  PetscFunctionBegin;
  if (name == ORDER_NEW) {
    name = (MatReorderingType) (ORDER_NEW + numberregistered++);
  }
  if (out) *out = name;

  if (!__MatReorderingList) {
    ierr = NRCreate(&__MatReorderingList); CHKERRQ(ierr);
  }
  ierr = NRRegister(__MatReorderingList,(int)name,sname,(int (*)(void*))order);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatReorderingRegisterDestroy"
/*@C
   MatReorderingRegisterDestroy - Frees the list of ordering routines.

   Notes:
   This routine is NOT collective.

.keywords: matrix, register, destroy

.seealso: MatReorderingRegister(), MatReorderingRegisterAll()
@*/
int MatReorderingRegisterDestroy(void)
{
  PetscFunctionBegin;
  if (__MatReorderingList) {
    NRDestroy( __MatReorderingList );
    __MatReorderingList = 0;
  }
  MatReorderingRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetReorderingTypeFromOptions"
/*@C
   MatGetReorderingTypeFromOptions - Gets matrix reordering method from the
   options database.

   Input Parameter:
.  prefix - optional database prefix

   Output Parameter:
.  type - reordering method

   Options Database Keys:
   To specify the ordering through the options database, use one of
   the following 
$    -mat_order natural, -mat_order nd, -mat_order 1wd, 
$    -mat_order rcm, -mat_order qmd

   Notes:
   This routine is NOT collective.

.keywords: matrix, set, ordering, factorization, direct, ILU, LU,
           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McKee, 
           Quotient Minimum Degree

.seealso: MatGetReordering()
@*/
int MatGetReorderingTypeFromOptions(char *prefix,MatReorderingType *type)
{
  char sbuf[50];
  int  ierr,flg;
  
  PetscFunctionBegin;
  ierr = OptionsGetString(prefix,"-mat_order", sbuf, 50,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!MatReorderingRegisterAllCalled) {ierr = MatReorderingRegisterAll();CHKERRQ(ierr);}
    *type = (MatReorderingType)NRFindID( __MatReorderingList, sbuf );
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatReorderingGetName"
/*@C
   MatReorderingGetName - Gets the name associated with a reordering.

   Input Parameter:
.  ordering - integer name of reordering

   Output Parameter:
.  name - name of reordering

   Not Collective

.keywords: PC, get, method, name, type
@*/
int MatReorderingGetName(MatReorderingType meth,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!MatReorderingRegisterAllCalled) {ierr = MatReorderingRegisterAll(); CHKERRQ(ierr);}
   *name = NRFindName( __MatReorderingList, (int)meth );
  PetscFunctionReturn(0);
}

extern int MatAdjustForInodes(Mat,IS *,IS *);

#include "src/mat/impls/aij/mpi/mpiaij.h"
#undef __FUNC__  
#define __FUNC__ "MatGetReordering"
/*@C
   MatGetReordering - Gets a reordering for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Input Parameters:
.  mat - the matrix
.  type - type of reordering, one of the following:
$      ORDER_NATURAL - Natural
$      ORDER_ND - Nested Dissection
$      ORDER_1WD - One-way Dissection
$      ORDER_RCM - Reverse Cuthill-McKee
$      ORDER_QMD - Quotient Minimum Degree

   Output Parameters:
.  rperm - row permutation indices
.  cperm - column permutation indices

   Collective on Mat

   Options Database Keys:
   To specify the ordering through the options database, use one of
   the following 
$    -mat_order natural, -mat_order nd, -mat_order 1wd, 
$    -mat_order rcm, -mat_order qmd

   The user can define additional orderings; see MatReorderingRegister().

.keywords: matrix, set, ordering, factorization, direct, ILU, LU,
           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McKee, 
           Quotient Minimum Degree

.seealso:  MatGetReorderingTypeFromOptions(), MatReorderingRegister()
@*/
int MatGetReordering(Mat mat,MatReorderingType type,IS *rperm,IS *cperm)
{
  int         ierr,flg,mmat,nmat,mis;
  int         (*r)(Mat,MatReorderingType,IS*,IS*);

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

  if (mat->M == 0) {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,cperm); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,rperm); CHKERRQ(ierr);
    ierr = ISSetIdentity(*cperm); CHKERRQ(ierr);
    ierr = ISSetIdentity(*rperm); CHKERRQ(ierr);
    ierr = ISSetPermutation(*rperm); CHKERRQ(ierr);
    ierr = ISSetPermutation(*cperm); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }


  if (!MatReorderingRegisterAllCalled) {
    ierr = MatReorderingRegisterAll();CHKERRQ(ierr);
  }

  ierr = MatGetReorderingTypeFromOptions(0,&type); CHKERRQ(ierr);
  PLogEventBegin(MAT_GetReordering,mat,0,0,0);
  r =  (int (*)(Mat,MatReorderingType,IS*,IS*))NRFindRoutine(__MatReorderingList,(int)type,(char *)0);
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown or unregistered type");}

  ierr = (*r)(mat,type,rperm,cperm); CHKERRQ(ierr);
  ierr = ISSetPermutation(*rperm); CHKERRQ(ierr);
  ierr = ISSetPermutation(*cperm); CHKERRQ(ierr);

  /*
      Adjust for inode (reduced matrix ordering) only if row permutation
    is smaller then matrix size
  */
  ierr = MatGetLocalSize(mat,&mmat,&nmat); CHKERRQ(ierr);
  ierr = ISGetSize(*rperm,&mis); CHKERRQ(ierr);
  if (mmat > mis) {  
    ierr = MatAdjustForInodes(mat,rperm,cperm); CHKERRQ(ierr);
  }

  PLogEventEnd(MAT_GetReordering,mat,0,0,0);

  ierr = OptionsHasName(0,"-mat_view_reordering_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    Mat tmat;
    ierr = OptionsHasName(0,"-mat_view_contour",&flg); CHKERRQ(ierr);
    if (flg) {
      ierr = ViewerPushFormat(VIEWER_DRAW_(mat->comm),VIEWER_FORMAT_DRAW_CONTOUR,0);CHKERRQ(ierr);
    }
    ierr = MatPermute(mat,*rperm,*cperm,&tmat); CHKERRQ(ierr);
    ierr = MatView(tmat,VIEWER_DRAW_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAW_(mat->comm)); CHKERRQ(ierr);
    if (flg) {
      ierr = ViewerPopFormat(VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    }
    ierr = MatDestroy(tmat);  CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}








