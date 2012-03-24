
/*
     Provides the code that allows PETSc users to register their own
  sequential matrix Ordering routines.
*/
#include <petsc-private/matimpl.h>
#include <petscmat.h>  /*I "petscmat.h" I*/

PetscFList      MatOrderingList = 0;
PetscBool  MatOrderingRegisterAllCalled = PETSC_FALSE;

extern PetscErrorCode MatGetOrdering_Flow_SeqAIJ(Mat,const MatOrderingType,IS *,IS *);

#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering_Flow"
PetscErrorCode MatGetOrdering_Flow(Mat mat,const MatOrderingType type,IS *irow,IS *icol)
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot do default flow ordering for matrix type");
#if !defined(PETSC_USE_DEBUG)
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering_Natural"
PetscErrorCode  MatGetOrdering_Natural(Mat mat,const MatOrderingType type,IS *irow,IS *icol)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*ii;
  PetscBool      done;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,PETSC_NULL,PETSC_NULL,&done);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,PETSC_NULL,PETSC_NULL,&done);CHKERRQ(ierr);
  if (done) { /* matrix may be "compressed" in symbolic factorization, due to i-nodes or block storage */
    /*
      We actually create general index sets because this avoids mallocs to
      to obtain the indices in the MatSolve() routines.
      ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,irow);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,icol);CHKERRQ(ierr);
    */
    ierr = PetscMalloc(n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) ii[i] = i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_COPY_VALUES,irow);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,icol);CHKERRQ(ierr);
  } else {
    PetscInt start,end;

    ierr = MatGetOwnershipRange(mat,&start,&end);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,end-start,start,1,irow);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,end-start,start,1,icol);CHKERRQ(ierr);
  }
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
#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering_RowLength"
PetscErrorCode  MatGetOrdering_RowLength(Mat mat,const MatOrderingType type,IS *irow,IS *icol)
{
  PetscErrorCode ierr;
  PetscInt       n,*ia,*ja,*permr,*lens,i;
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr  = PetscMalloc2(n,PetscInt,&lens,n,PetscInt,&permr);CHKERRQ(ierr);
  for (i=0; i<n; i++) { 
    lens[i]  = ia[i+1] - ia[i];
    permr[i] = i;
  }
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = PetscSortIntWithPermutation(n,lens,permr);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,irow);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,icol);CHKERRQ(ierr);
  ierr = PetscFree2(lens,permr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatOrderingRegister" 
PetscErrorCode  MatOrderingRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(Mat,const MatOrderingType,IS*,IS*))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatOrderingList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatOrderingRegisterDestroy"
/*@
   MatOrderingRegisterDestroy - Frees the list of ordering routines.

   Not collective

   Level: developer
   
.keywords: matrix, register, destroy

.seealso: MatOrderingRegisterDynamic(), MatOrderingRegisterAll()
@*/
PetscErrorCode  MatOrderingRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&MatOrderingList);CHKERRQ(ierr);
  MatOrderingRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering"
/*@C
   MatGetOrdering - Gets a reordering for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - type of reordering, one of the following:
$      MATORDERINGNATURAL - Natural
$      MATORDERINGND - Nested Dissection
$      MATORDERING1WD - One-way Dissection
$      MATORDERINGRCM - Reverse Cuthill-McKee
$      MATORDERINGQMD - Quotient Minimum Degree

   Output Parameters:
+  rperm - row permutation indices
-  cperm - column permutation indices


   Options Database Key:
. -mat_view_ordering_draw - plots matrix nonzero structure in new ordering

   Level: intermediate
   
   Notes:
      This DOES NOT actually reorder the matrix; it merely returns two index sets
   that define a reordering. This is usually not used directly, rather use the 
   options PCFactorSetMatOrderingType()

   The user can define additional orderings; see MatOrderingRegisterDynamic().

   These are generally only implemented for sequential sparse matrices.

   The external packages that PETSc can use for direct factorization such as SuperLU do not accept orderings provided by 
   this call.


.keywords: matrix, set, ordering, factorization, direct, ILU, LU,
           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McKee, 
           Quotient Minimum Degree

.seealso:   MatOrderingRegisterDynamic(), PCFactorSetMatOrderingType()
@*/
PetscErrorCode  MatGetOrdering(Mat mat,const MatOrderingType type,IS *rperm,IS *cperm)
{
  PetscErrorCode ierr;
  PetscInt       mmat,nmat,mis,m;
  PetscErrorCode (*r)(Mat,const MatOrderingType,IS*,IS*);
  PetscBool      flg = PETSC_FALSE,isseqdense,ismpidense,ismpiaij,ismpibaij,ismpisbaij,ismpiaijcusp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(rperm,2);
  PetscValidPointer(cperm,3);
  if (!mat->assembled) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factortype) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  /* this chunk of code is REALLY bad, should maybe get the ordering from the factor matrix,
     then those that don't support orderings will handle their cases themselfs. */
  ierr = PetscTypeCompare((PetscObject)mat,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIDENSE,&ismpidense);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIAIJCUSP,&ismpiaijcusp);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIBAIJ,&ismpibaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPISBAIJ,&ismpisbaij);CHKERRQ(ierr);
  if (isseqdense || ismpidense || ismpiaij || ismpibaij || ismpisbaij || ismpiaijcusp) {
    ierr = MatGetLocalSize(mat,&m,PETSC_NULL);CHKERRQ(ierr);
    /*
       Dense matrices only give natural ordering
    */
    ierr = ISCreateStride(PETSC_COMM_SELF,0,m,1,cperm);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,m,1,rperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*cperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!mat->rmap->N) { /* matrix has zero rows */
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,cperm);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,rperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*cperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  
  ierr = MatGetLocalSize(mat,&mmat,&nmat);CHKERRQ(ierr);
  if (mmat != nmat) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",mmat,nmat);

  if (!MatOrderingRegisterAllCalled) {ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscFListFind(MatOrderingList,((PetscObject)mat)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown or unregistered type: %s",type);

  ierr = PetscLogEventBegin(MAT_GetOrdering,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*r)(mat,type,rperm,cperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);
  /* Adjust for inode (reduced matrix ordering) only if row permutation is smaller the matrix size */
  ierr = ISGetLocalSize(*rperm,&mis);CHKERRQ(ierr);
  if (mmat > mis) {ierr = MatInodeAdjustForInodes(mat,rperm,cperm);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(MAT_GetOrdering,mat,0,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(((PetscObject)mat)->prefix,"-mat_view_ordering_draw",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    Mat tmat;
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(((PetscObject)mat)->prefix,"-mat_view_contour",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm),PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
    }
    ierr = MatPermute(mat,*rperm,*cperm,&tmat);CHKERRQ(ierr);
    ierr = MatView(tmat,PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPopFormat(PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    }
    ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOrderingList"
PetscErrorCode MatGetOrderingList(PetscFList *list)
{
  PetscFunctionBegin;
  *list = MatOrderingList;
  PetscFunctionReturn(0);
}
