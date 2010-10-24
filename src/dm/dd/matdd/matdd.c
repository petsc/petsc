#define PETSCMAT_DLL

#include "../src/mat/impls/dd/matdd.h"          /*I "petscmat.h" I*/



/*
  MatDD represents a global matrix A composed of local submatrices or blocks A_{ij}.
  The blocks are combined using scatters S_j and gather G_i:
           A = \sum_{i,j=0}^{N-1} G_i A_{ij} S_j.
  Therefore, 
      scatters {S_i} define a "lifting" of the input Vec space into a direct sum \R^N = \osum_{0 \leq i < I} \R^{N_i}, 
      gathers  {G_i} define a "covering" of the output Vec space by a direct sum \R^M = \osum_{0 \leq i < J} \R^{M_i},
  while the block matrix \overline A = (A_{ij}) acts from \R^N to \R^M.
  Matrix A need not be assembled and its action can be defined via the action of the individual scatters, blocks
  and gathers.  However, it can be useful to "fuse" scatters and gathers to amortize the communication cost.
  Furthermore, A can also be fused ("assembly" has a well-established meaning already, so it is avoided here), 
  although that is a nontrivial operation to carry out efficiently.

  The block structure of A can be stored in various ways.  In particular, blocks can be logical blocks within a single
  Mat of a fixed type, resulting in MATDDXXX, where XXX is a matrix type (e.g., MATDDAIJ, MATDDDENSE, etc).
  Alternatively, blocks can be stored in a *metamatrix* of blocks. Currently, we support MATDDMETAAIJ, where an AIJ-like
  structure keeps track of the nonzero blocks. We anticipate supporting MATDDMETADENSE (easier, in many ways),
  and MATDDMETASHELL (even easier).

  MATDDMETAAIJ:
  Blocks are set by calling MatDDMetaAddBlock(A, i,j, B). This is analogous to declaring a nonzero element in an 
  AIJ matrix.  Although generally there will be only a few nonzero blocks, it might be benefitial to 
  preallocate the local (per rank) block nonzero structure by calling MatDDMetaAIJSetPreallocation(Mat A, PetscInt nz, PetscInt *nnz),
  where nz is the number of nonzero blocks per block row or nnz is an array of such numbers, and the usage 
  of nz and nnz is mutually exclusive.

  MATDDMETADENSE:
  This type will preallocate the I \times J array of block data structures.  

  MATDDMETASHELL:
  This type will require the user to supply a routine that applies the (i,j)-th block,
  with the type PetscErrorCode (*MatDDMetaShellBlockMult)(Mat A, PetscInt i, PetscInt j, Vec x, Vec y).


  MATDDXXX:
  This will use a single matrix of a given type (MATAIJ or MATDENSE) to store the blocks, essentially, as in \overline A.
  MATDD will manage the translation of block and in-block indices to global indices into \overline A.
  This can be useful, if many blocks are desired (all FEM or SEM blocks), and the overhead of maintaining the 
  individual matrix blocks becomes prohibitive.


Use cases that must work:
1. Efficient representation of a stiffness matrix for a 2D P1 triangular mesh using just the vertex array and an element Mat.
2. Assembly of a stiffness matrix for a geometrically conforming mortar element in 2D on a square doublet (e.g., as in P. Fischer et al).
 a) Elimination of the slaved degrees of freedom on the mortar.
 b) Adding of  Lagrange multipliers' degrees of freedom for the mortar.
3. Efficient representation of a bordered matrix.  For example, a single row/col border arising from parameter continuation.
4. Construction of a global pressure split scatter from a local scatter for Stokes on function spaces in 1 and 2.
5. MG

Scatter construction should be simplified with helper routines for the following:
1. P1 simplicial gather.
2. Mortar element mesh gather (with and without constraints).
3. Field split-off scatter from local splits.
4. easy combinations of gathers/scatters with certain types of blocks:
   - "tensor product" of a scatter with a block
*/

#undef __FUNCT__
#define __FUNCT__ "MatDDMultRow"
PetscErrorCode MatDDMultRow(Mat M, PetscInt i, Vec x, Vec y) {
  struct Mat_DD* dd = (struct Mat_DD*)M->data;
  PetscFunctionBegin;
  if(!dd->ops->multrow) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__ " not supported by this MatDD type");
  }
  if(i < 0 || i >= dd->rmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD row block %d: must be between 0 <= and < %d", i, dd->rmapdd->dcount);
  }
  ierr = dd->ops->multrow(M,i,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMultRow() */

#undef __FUNCT__
#define __FUNCT__ "MatDDMultAddCol"
PetscErrorCode MatDDMultAddCol(Mat M, PetscInt j, Vec x, Vec y) {
  struct Mat_DD* dd = (struct Mat_DD*)M->data;
  PetscFunctionBegin;
  if(!dd->ops->multaddcol) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__ " not supported by this MatDD type");
  }
  if(j < 0 || j >= dd->cmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD col block %d: must be between 0 <= and < %d", j, dd->cmapdd->dcount);
  }
  ierr = dd->ops->multaddcol(M,j,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMultAddCol() */

#undef __FUNCT__
#define __FUNCT__ "MatDDMultTransposeAddRow"
PetscErrorCode MatDDMultTransposeAddRow(Mat M, PetscInt i, Vec x, Vec y) {
  struct Mat_DD* dd = (struct Mat_DD*)M->data;
  PetscFunctionBegin;
  if(!dd->ops->multtransposeaddrow) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__ " not supported by this MatDD type");
  }
  if(i < 0 || i >= dd->rmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD row block %d: must be between 0 <= and < %d", i, dd->rmapdd->dcount);
  }
  ierr = dd->ops->multtransposeaddrow(M,i,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMultTransposeAddRow() */

#undef __FUNCT__
#define __FUNCT__ "MatDDMultTransposeCol"
PetscErrorCode MatDDMultTransposeCol(Mat M, PetscInt j, Vec x, Vec y) {
  struct Mat_DD* dd = (struct Mat_DD*)M->data;
  PetscFunctionBegin;
  if(!dd->ops->multtransposecol) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__ " not supported by this MatDD type");
  }
  if(j < 0 || j >= dd->cmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD col block %d: must be between 0 <= and < %d", j, dd->cmapdd->dcoount);
  }
  ierr = dd->ops->multtranposecol(M,j,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMultTransposeCol() */

#undef __FUNCT__
#define __FUNCT__ "MatDDMultAddBlock"
PetscErrorCode MatDDMultAddBlock(Mat M, PetscInt i, PetscInt j, Vec x, Vec y) {
  struct Mat_DD* dd = (struct Mat_DD*)M->data;
  PetscFunctionBegin;
  if(!dd->ops->multaddblock) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__ " not supported by this MatDD type");
  }
  if(i < 0 || i >= dd->rmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD row block %d: must be between 0 <= and < %d", i, dd->rmapdd->dcount);
  }
  if(j < 0 || j >= dd->cmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD col block %d: must be between 0 <= and < %d", j, dd->cmapdd->dcount);
  }
  ierr = dd->ops->multaddblock(M,i,j,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMultAddBlock() */

#undef __FUNCT__
#define __FUNCT__ "MatDDMultTransposeAddBlock"
PetscErrorCode MatDDMultTransposeAddBlock(MatDD M, PetscInt i, PetscInt j, Vec x, Vec y) {
  struct Mat_DD* dd = (struct Mat_DD*)M->data;
  PetscFunctionBegin;
  if(!dd->multaddblock) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__ " not supported by this MatDD type");
  }
  if(i < 0 || i >= dd->rmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD row block %d: must be between 0 <= and < %d", i, dd->rmapdd->dcount);
  }
  if(j < 0 || j >= dd->cmapdd->dcount) {
    SETERRQ2(PETSC_ERR_USER, "Invalid MatDD col block %d: must be between 0 <= and < %d", j, dd->cmapdd->dcount);
  }
  ierr = dd->ops->multtransposeaddblock(M,i,j,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMultTransposeAddBlock() */


/****************************************************************************/

#undef __FUNCT__  
#define __FUNCT__ "MatDDSetGather"
PetscErrorCode MatDDSetGather(Mat A, Mat G, PetscInt rowd_count, PetscInt **row_dN) { 
  PetscErrorCode ierr;
  Mat_DD *dd = (Mat_DD*) A->data;
  PetscInt m,M;
  PetscFunctionBegin;
  if(dd->gather) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cannot reset gather");
  }
  if(G){
    m = G->cmap->n; M = G->cmap->M;
  }
  else {
    m = A->rmap->n; M = A->rmap->N;
  }
  ierr = VecCreateDD(((PetscObject)A)->comm, m, M, rowdcount, row_dN, &dd->outvec); CHKERRQ(ierr);
  ierr = VecDDGetDDLayout(dd->outvec, &dd->rmapdd); CHKERRQ(ierr);
  dd->gather = G;
  PetscFunctionReturn(0);
}/* MatDDSetGather() */


#undef __FUNCT__  
#define __FUNCT__ "MatDDSetGatherLocal"
PetscErrorCode MatDDSetGatherLocal(Mat A, Mat G, PetscInt rowd_count, PetscInt **row_dn) { 
  PetscErrorCode ierr;
  Mat_DD *dd = (Mat_DD*) A->data;
  PetscInt m,M;
  PetscFunctionBegin;
  if(dd->gather) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cannot reset gather");
  }
  if(G){
    m = G->cmap->n; M = G->cmap->M;
  }
  else {
    m = A->rmap->n; M = A->rmap->N;
  }
  ierr = VecCreateDDLocal(((PetscObject)A)->comm, m, M, rowdcount, row_dn, &dd->outvec); CHKERRQ(ierr);
  ierr = VecDDGetDDLayout(dd->outvec, &dd->rmapdd); CHKERRQ(ierr);
  dd->gather = G;
  PetscFunctionReturn(0);
}/* MatDDSetGatherLocal() */


#undef __FUNCT__  
#define __FUNCT__ "MatDDSetScatter"
PetscErrorCode MatDDSetScatter(Mat A, Mat S, PetscInt col_dcount, PetscInt **col_dN) { 
  PetscErrorCode ierr;
  Mat_DD *dd = (Mat_DD*) A->data;
  PetscInt n,N;
  PetscFunctionBegin;
  if(dd->gather) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cannot reset scatter");
  }
  if(S) {
    n = S->rmap->n; N = S->rmap->N;
  }
  else {
    n = A->cmap->n; N = A->cmap->N;
  }
  ierr = VecCreateDD(((PetscObject)A)->comm, n, N, col_dcount, col_dN, &dd->invec); CHKERRQ(ierr);
  ierr = VecDDGetDDLayout(dd->invec, &dd->cmapdd); CHKERRQ(ierr);
  dd->scatter = S;
  PetscFunctionReturn(0);
}/* MatDDSetScatter() */


#undef __FUNCT__  
#define __FUNCT__ "MatDDSetScatterLocal"
PetscErrorCode MatDDSetScatterLocal(Mat A, Mat S, PetscInt col_dcount, PetscInt **col_dn) { 
  PetscErrorCode ierr;
  Mat_DD *dd = (Mat_DD*) A->data;
  PetscInt n,N;
  PetscFunctionBegin;
  if(dd->gather) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cannot reset scatter");
  }
  if(S) {
    n = S->rmap->n; N = S->rmap->N;
  }
  else {
    n = A->cmap->n; N = A->cmap->N;
  }
  ierr = VecCreateDDLocal(((PetscObject)A)->comm, n, N, col_dcount, col_dn, &dd->invec); CHKERRQ(ierr);
  ierr = VecDDGetDDLayout(dd->invec, &dd->cmapdd); CHKERRQ(ierr);
  dd->scatter = S;
  PetscFunctionReturn(0);
}/* MatDDSetScatterLocal() */


#undef  __FUNCT__
#define __FUNCT__ "MatCreate_DD_Private"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_DD_Private(Mat A, Mat_DD *dd) {
  /* This is a constructor that may be called by a derived constructor,
     so it does allocate data or reset the classname. */
  /* Assume that this is called after MatSetSizes() */
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  /* Zero out most of MatOps: MATDD is an abstract base class */
  ierr = PetscMemzero(A->ops, sizeof(struct _MatOps)); CHKERRQ(ierr);
  /* Some of these methods can be given default implementations, which rely on other,
     more basic methods: setsizes, etc. */ 
  /* The following methods are expected to be set by classes deriving from MATDD */
  A->ops->mult          = 0; 
  A->ops->multtranspose = 0;
  A->ops->setvalues     = 0;
  A->ops->assemblybegin = 0;
  A->ops->assemblyend   = 0;
  A->ops->destroy       = 0;

  A->assembled    = PETSC_FALSE;
  A->same_nonzero = PETSC_FALSE;

  A->data = (void*)dd;
  /* Zero out all MatDD ops: purely abstract base class. */
  ierr = PetscMemzero(dd->ops, sizeof(struct _MatDDOps)); CHKERRQ(ierr);

  /* Data fields set by MatDDSetGather/MatDDSetScatter */
  dd->scatter = PETSC_NULL;
  dd->gather  = PETSC_NULL;
  dd->rmapdd  = PETSC_NULL;
  dd->cmapdd  = PETSC_NULL;
  dd->invec   = PETSC_NULL;
  dd->outvec  = PETSC_NULL;

  dd->setup   = PETSC_FALSE;
  PetscFunctionReturn(0);
}/* MatCreate_DD_Private() */


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_DD_Private"
PetscErrorCode MatDestroy_DD_Private(Mat mat)
{
  /* This is a destructor that may be called by a derived destructor,
     so it does not free data or reset the classname. */
  PetscErrorCode        ierr;
  Mat_DD*               dd = (Mat_DD*)mat->data;
  PetscInt              k;

  PetscFunctionBegin;


  if(dd->invec) {
    ierr = DDLayoutDestroy(dd->cmapdd);  CHKERRQ(ierr);
    ierr = VecDestroy(dd->invec);        CHKERRQ(ierr);
  }
  if(dd->outvec) {
    ierr = DDLayoutDestroy(dd->rmapdd);  CHKERRQ(ierr);
    ierr = VecDestroy(dd->outvec);       CHKERRQ(ierr);
  }
  if(dd->scatter) {
    ierr = MatDestroy(dd->scatter);                                  CHKERRQ(ierr);
  }
  if(dd->gather) {
    ierr = MatDestroy(dd->gather);                                   CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* MatDestroy_DD_Private() */

