
/*
     Provides the code that allows PETSc users to register their own
  sequential matrix Ordering routines.
*/
#include <petsc/private/matimpl.h>
#include <petscmat.h>  /*I "petscmat.h" I*/

PetscFunctionList MatOrderingList              = NULL;
PetscBool         MatOrderingRegisterAllCalled = PETSC_FALSE;

extern PetscErrorCode MatGetOrdering_Flow_SeqAIJ(Mat,MatOrderingType,IS*,IS*);

PetscErrorCode MatGetOrdering_Flow(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot do default flow ordering for matrix type");
}

PETSC_INTERN PetscErrorCode MatGetOrdering_Natural(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*ii;
  PetscBool      done;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,NULL,NULL,&done);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,NULL,NULL,&done);CHKERRQ(ierr);
  if (done) { /* matrix may be "compressed" in symbolic factorization, due to i-nodes or block storage */
    /*
      We actually create general index sets because this avoids mallocs to
      to obtain the indices in the MatSolve() routines.
      ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,irow);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,icol);CHKERRQ(ierr);
    */
    ierr = PetscMalloc1(n,&ii);CHKERRQ(ierr);
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

/*
     Orders the rows (and columns) by the lengths of the rows.
   This produces a symmetric Ordering but does not require a
   matrix with symmetric non-zero structure.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_RowLength(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscErrorCode ierr;
  PetscInt       n,*permr,*lens,i;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc2(n,&lens,n,&permr);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    lens[i]  = ia[i+1] - ia[i];
    permr[i] = i;
  }
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = PetscSortIntWithPermutation(n,lens,permr);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,irow);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,icol);CHKERRQ(ierr);
  ierr = PetscFree2(lens,permr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatOrderingRegister - Adds a new sparse matrix ordering to the matrix package.

   Not Collective

   Input Parameters:
+  sname - name of ordering (for example MATORDERINGND)
-  function - function pointer that creates the ordering

   Level: developer

   Sample usage:
.vb
   MatOrderingRegister("my_order", MyOrder);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatOrderingSetType(part,"my_order)
   or at runtime via the option
$     -pc_factor_mat_ordering_type my_order

.seealso: MatOrderingRegisterDestroy(), MatOrderingRegisterAll()
@*/
PetscErrorCode  MatOrderingRegister(const char sname[],PetscErrorCode (*function)(Mat,MatOrderingType,IS*,IS*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&MatOrderingList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>
/*@C
   MatGetOrdering - Gets a reordering for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - type of reordering, one of the following:
$      MATORDERINGNATURAL_OR_ND - Nested dissection unless matrix is SBAIJ then it is natural
$      MATORDERINGNATURAL - Natural
$      MATORDERINGND - Nested Dissection
$      MATORDERING1WD - One-way Dissection
$      MATORDERINGRCM - Reverse Cuthill-McKee
$      MATORDERINGQMD - Quotient Minimum Degree
$      MATORDERINGEXTERNAL - Use an ordering internal to the factorzation package and do not compute or use PETSc's

   Output Parameters:
+  rperm - row permutation indices
-  cperm - column permutation indices

   Options Database Key:
+ -mat_view_ordering draw - plots matrix nonzero structure in new ordering
- -pc_factor_mat_ordering_type <nd,natural,..> - ordering to use with PCs based on factorization, LU, ILU, Cholesky, ICC

   Level: intermediate

   Notes:
      This DOES NOT actually reorder the matrix; it merely returns two index sets
   that define a reordering. This is usually not used directly, rather use the
   options PCFactorSetMatOrderingType()

   The user can define additional orderings; see MatOrderingRegister().

   These are generally only implemented for sequential sparse matrices.

   Some external packages that PETSc can use for direct factorization such as SuperLU do not accept orderings provided by
   this call.

   If MATORDERINGEXTERNAL is used then PETSc does not compute an ordering and utilizes one built into the factorization package

           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McKee,
           Quotient Minimum Degree

.seealso:   MatOrderingRegister(), PCFactorSetMatOrderingType(), MatColoring, MatColoringCreate()
@*/
PetscErrorCode  MatGetOrdering(Mat mat,MatOrderingType type,IS *rperm,IS *cperm)
{
  PetscErrorCode ierr;
  PetscInt       mmat,nmat,mis,m;
  PetscErrorCode (*r)(Mat,MatOrderingType,IS*,IS*);
  PetscBool      flg = PETSC_FALSE,isseqdense,ismpidense,ismpiaij,ismpibaij,ismpisbaij,ismpiaijcusparse,iselemental,isscalapack,flg1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(rperm,3);
  PetscValidPointer(cperm,4);
  if (!mat->assembled) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factortype) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = PetscStrcmp(type,MATORDERINGEXTERNAL,&flg1);CHKERRQ(ierr);
  if (flg1) {
    *rperm = NULL;
    *cperm = NULL;
    PetscFunctionReturn(0);
  }

  ierr = PetscStrcmp(type,MATORDERINGNATURAL_OR_ND,&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscBool isseqsbaij;
    ierr = PetscObjectTypeCompareAny((PetscObject)mat,&isseqsbaij,MATSEQSBAIJ,MATSEQBAIJ,NULL);CHKERRQ(ierr);
    if (isseqsbaij) {
      type = MATORDERINGNATURAL;
    } else {
      type = MATORDERINGND;
    }
  }

  /* This code is terrible. MatGetOrdering() multiple dispatch should use matrix and this code should move to impls/aij/mpi. */
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (ismpiaij) {               /* Reorder using diagonal block */
    Mat            Ad,Ao;
    const PetscInt *colmap;
    IS             lrowperm,lcolperm;
    PetscInt       i,rstart,rend,*idx;
    const PetscInt *lidx;

    ierr = MatMPIAIJGetSeqAIJ(mat,&Ad,&Ao,&colmap);CHKERRQ(ierr);
    ierr = MatGetOrdering(Ad,type,&lrowperm,&lcolperm);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
    /* Remap row index set to global space */
    ierr = ISGetIndices(lrowperm,&lidx);CHKERRQ(ierr);
    ierr = PetscMalloc1(rend-rstart,&idx);CHKERRQ(ierr);
    for (i=0; i+rstart<rend; i++) idx[i] = rstart + lidx[i];
    ierr = ISRestoreIndices(lrowperm,&lidx);CHKERRQ(ierr);
    ierr = ISDestroy(&lrowperm);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mat),rend-rstart,idx,PETSC_OWN_POINTER,rperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
    /* Remap column index set to global space */
    ierr = ISGetIndices(lcolperm,&lidx);CHKERRQ(ierr);
    ierr = PetscMalloc1(rend-rstart,&idx);CHKERRQ(ierr);
    for (i=0; i+rstart<rend; i++) idx[i] = rstart + lidx[i];
    ierr = ISRestoreIndices(lcolperm,&lidx);CHKERRQ(ierr);
    ierr = ISDestroy(&lcolperm);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mat),rend-rstart,idx,PETSC_OWN_POINTER,cperm);CHKERRQ(ierr);
    ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* this chunk of code is REALLY bad, should maybe get the ordering from the factor matrix,
     then those that don't support orderings will handle their cases themselves. */
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSE,&ismpidense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIAIJCUSPARSE,&ismpiaijcusparse);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIBAIJ,&ismpibaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPISBAIJ,&ismpisbaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATELEMENTAL,&iselemental);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSCALAPACK,&isscalapack);CHKERRQ(ierr);
  if (isseqdense || ismpidense || ismpibaij || ismpisbaij || ismpiaijcusparse || iselemental || isscalapack) {
    ierr = MatGetLocalSize(mat,&m,NULL);CHKERRQ(ierr);
    /*
       These matrices only give natural ordering
    */
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,cperm);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,rperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*cperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*rperm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!mat->rmap->N) { /* matrix has zero rows */
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,cperm);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,rperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*cperm);CHKERRQ(ierr);
    ierr = ISSetIdentity(*rperm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = MatGetLocalSize(mat,&mmat,&nmat);CHKERRQ(ierr);
  if (mmat != nmat) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",mmat,nmat);

  ierr = MatOrderingRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(MatOrderingList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown or unregistered type: %s",type);

  ierr = PetscLogEventBegin(MAT_GetOrdering,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*r)(mat,type,rperm,cperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);
  /* Adjust for inode (reduced matrix ordering) only if row permutation is smaller the matrix size */
  ierr = ISGetLocalSize(*rperm,&mis);CHKERRQ(ierr);
  if (mmat > mis) {ierr = MatInodeAdjustForInodes(mat,rperm,cperm);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(MAT_GetOrdering,mat,0,0,0);CHKERRQ(ierr);


  ierr = PetscOptionsHasName(((PetscObject)mat)->options,((PetscObject)mat)->prefix,"-mat_view_ordering",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat tmat;
    ierr = MatPermute(mat,*rperm,*cperm,&tmat);CHKERRQ(ierr);
    ierr = MatViewFromOptions(tmat,(PetscObject)mat,"-mat_view_ordering");CHKERRQ(ierr);
    ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetOrderingList(PetscFunctionList *list)
{
  PetscFunctionBegin;
  *list = MatOrderingList;
  PetscFunctionReturn(0);
}
