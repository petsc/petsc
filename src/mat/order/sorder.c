
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
  PetscInt       n,i,*ii;
  PetscBool      done;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));
  CHKERRQ(MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,NULL,NULL,&done));
  CHKERRQ(MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,NULL,NULL,&done));
  if (done) { /* matrix may be "compressed" in symbolic factorization, due to i-nodes or block storage */
    /*
      We actually create general index sets because this avoids mallocs to
      to obtain the indices in the MatSolve() routines.
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,0,1,irow));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,0,1,icol));
    */
    CHKERRQ(PetscMalloc1(n,&ii));
    for (i=0; i<n; i++) ii[i] = i;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_COPY_VALUES,irow));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,icol));
  } else {
    PetscInt start,end;

    CHKERRQ(MatGetOwnershipRange(mat,&start,&end));
    CHKERRQ(ISCreateStride(comm,end-start,start,1,irow));
    CHKERRQ(ISCreateStride(comm,end-start,start,1,icol));
  }
  CHKERRQ(ISSetIdentity(*irow));
  CHKERRQ(ISSetIdentity(*icol));
  PetscFunctionReturn(0);
}

/*
     Orders the rows (and columns) by the lengths of the rows.
   This produces a symmetric Ordering but does not require a
   matrix with symmetric non-zero structure.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_RowLength(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscInt       n,*permr,*lens,i;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  CHKERRQ(MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,&ia,&ja,&done));
  PetscCheckFalse(!done,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");

  CHKERRQ(PetscMalloc2(n,&lens,n,&permr));
  for (i=0; i<n; i++) {
    lens[i]  = ia[i+1] - ia[i];
    permr[i] = i;
  }
  CHKERRQ(MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,&ia,&ja,&done));

  CHKERRQ(PetscSortIntWithPermutation(n,lens,permr));

  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,irow));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,icol));
  CHKERRQ(PetscFree2(lens,permr));
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

.seealso: MatOrderingRegisterAll()
@*/
PetscErrorCode  MatOrderingRegister(const char sname[],PetscErrorCode (*function)(Mat,MatOrderingType,IS*,IS*))
{
  PetscFunctionBegin;
  CHKERRQ(MatInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&MatOrderingList,sname,function));
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
  PetscInt       mmat,nmat,mis;
  PetscErrorCode (*r)(Mat,MatOrderingType,IS*,IS*);
  PetscBool      flg,ismpiaij;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(rperm,3);
  PetscValidPointer(cperm,4);
  PetscCheckFalse(!mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheckFalse(mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheckFalse(!type,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Ordering type cannot be null");

  CHKERRQ(PetscStrcmp(type,MATORDERINGEXTERNAL,&flg));
  if (flg) {
    *rperm = NULL;
    *cperm = NULL;
    PetscFunctionReturn(0);
  }

  /* This code is terrible. MatGetOrdering() multiple dispatch should use matrix and this code should move to impls/aij/mpi. */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATMPIAIJ,&ismpiaij));
  if (ismpiaij) {               /* Reorder using diagonal block */
    Mat            Ad,Ao;
    const PetscInt *colmap;
    IS             lrowperm,lcolperm;
    PetscInt       i,rstart,rend,*idx;
    const PetscInt *lidx;

    CHKERRQ(MatMPIAIJGetSeqAIJ(mat,&Ad,&Ao,&colmap));
    CHKERRQ(MatGetOrdering(Ad,type,&lrowperm,&lcolperm));
    CHKERRQ(MatGetOwnershipRange(mat,&rstart,&rend));
    /* Remap row index set to global space */
    CHKERRQ(ISGetIndices(lrowperm,&lidx));
    CHKERRQ(PetscMalloc1(rend-rstart,&idx));
    for (i=0; i+rstart<rend; i++) idx[i] = rstart + lidx[i];
    CHKERRQ(ISRestoreIndices(lrowperm,&lidx));
    CHKERRQ(ISDestroy(&lrowperm));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)mat),rend-rstart,idx,PETSC_OWN_POINTER,rperm));
    CHKERRQ(ISSetPermutation(*rperm));
    /* Remap column index set to global space */
    CHKERRQ(ISGetIndices(lcolperm,&lidx));
    CHKERRQ(PetscMalloc1(rend-rstart,&idx));
    for (i=0; i+rstart<rend; i++) idx[i] = rstart + lidx[i];
    CHKERRQ(ISRestoreIndices(lcolperm,&lidx));
    CHKERRQ(ISDestroy(&lcolperm));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)mat),rend-rstart,idx,PETSC_OWN_POINTER,cperm));
    CHKERRQ(ISSetPermutation(*cperm));
    PetscFunctionReturn(0);
  }

  if (!mat->rmap->N) { /* matrix has zero rows */
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,cperm));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,rperm));
    CHKERRQ(ISSetIdentity(*cperm));
    CHKERRQ(ISSetIdentity(*rperm));
    PetscFunctionReturn(0);
  }

  CHKERRQ(MatGetLocalSize(mat,&mmat,&nmat));
  PetscCheckFalse(mmat != nmat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT,mmat,nmat);

  CHKERRQ(MatOrderingRegisterAll());
  CHKERRQ(PetscFunctionListFind(MatOrderingList,type,&r));
  PetscCheckFalse(!r,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown or unregistered type: %s",type);

  CHKERRQ(PetscLogEventBegin(MAT_GetOrdering,mat,0,0,0));
  CHKERRQ((*r)(mat,type,rperm,cperm));
  CHKERRQ(ISSetPermutation(*rperm));
  CHKERRQ(ISSetPermutation(*cperm));
  /* Adjust for inode (reduced matrix ordering) only if row permutation is smaller the matrix size */
  CHKERRQ(ISGetLocalSize(*rperm,&mis));
  if (mmat > mis) CHKERRQ(MatInodeAdjustForInodes(mat,rperm,cperm));
  CHKERRQ(PetscLogEventEnd(MAT_GetOrdering,mat,0,0,0));

  CHKERRQ(PetscOptionsHasName(((PetscObject)mat)->options,((PetscObject)mat)->prefix,"-mat_view_ordering",&flg));
  if (flg) {
    Mat tmat;
    CHKERRQ(MatPermute(mat,*rperm,*cperm,&tmat));
    CHKERRQ(MatViewFromOptions(tmat,(PetscObject)mat,"-mat_view_ordering"));
    CHKERRQ(MatDestroy(&tmat));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetOrderingList(PetscFunctionList *list)
{
  PetscFunctionBegin;
  *list = MatOrderingList;
  PetscFunctionReturn(0);
}
