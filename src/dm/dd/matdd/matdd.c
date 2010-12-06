#define PETSCMAT_DLL

#include "../src/dm/dd/matdd/matdd.h"          /*I "petscmat.h" I*/



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
  Mat of a fixed type, resulting in MATXXXDD, where XXX is a matrix type (e.g., MATAIJDD, MATDENSEDD, etc).
  Alternatively, blocks can be stored in a *metamatrix* of blocks MATDDXXX. The XXX matrix format is used to organize
  the block matrices. Currently, we support MATDDAIJ, where an AIJ-like structure keeps track of the nonzero blocks. 
  We anticipate supporting MATDDDENSE (easier, in many ways), and MATDDSHELL (even easier).

  MATDDAIJ:
  Blocks are set by calling MatDDAddBlock(A, i,j, B). This is analogous to declaring a nonzero element in an 
  AIJ matrix.  Although generally there will be only a few nonzero blocks, it might be benefitial to 
  preallocate the local (per rank) block nonzero structure by calling MatDDAIJSetPreallocation(Mat A, PetscInt nz, PetscInt *nnz),
  where nz is the number of nonzero blocks per block row or nnz is an array of such numbers, and the usage 
  of nz and nnz is mutually exclusive.

  MATDDDENSE:
  This type will preallocate the I \times J array of block data structures.  

  MATDDSHELL:
  This type will require the user to supply a routine that applies the (i,j)-th block,
  with the type PetscErrorCode (*MatDDShellBlockMult)(Mat A, PetscInt i, PetscInt j, Vec x, Vec y).


  MATXXXDD:
  This will use a single matrix of a given type (MATAIJ or MATDENSE) to store the blocks, essentially, as in \overline A.
  MATXXXDD will manage the translation of block and in-block indices to global indices into \overline A.
  Each MATXXXDD will be an extension of the corresponding MATXXX and MATDD.
  MATXXXDD formats can be useful, if many blocks are desired (all FEM or SEM blocks), 
  and the overhead  of maintaining the individual matrix blocks as in MATDDXXX becomes prohibitive.


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


#undef  __FUNCT__
#define __FUNCT__ "MatDD_BlockInit"
PetscErrorCode  MatDD_BlockInit(Mat A, PetscInt i, PetscInt j, const MatType blockmattype, MatDDBlockCommType subcommtype, MatDD_Block *block) {
  PetscErrorCode        ierr;
  Mat_DD*               dd = (Mat_DD*)A->data;
  PetscInt              m,n;
  MPI_Comm              comm = ((PetscObject)A)->comm, subcomm;
  PetscMPIInt           commsize, commrank, subcommsize;
  PetscMPIInt           subcommcolor;
  
  PetscFunctionBegin;
  
  /**/
  m = dd->rmapdd->dn[2*i+1]-dd->rmapdd->dn[2*i];
  n = dd->cmapdd->dn[2*j+1]-dd->cmapdd->dn[2*j];
  switch(subcommtype) {
  case MATDDMETA_BLOCK_COMM_SELF:
    subcomm = PETSC_COMM_SELF;
    subcommsize = 1;
    break;
  case MATDDMETA_BLOCK_COMM_SAME:
    subcomm = comm;
    ierr = MPI_Comm_size(subcomm, &subcommsize); CHKERRQ(ierr);
    break;
  case MATDDMETA_BLOCK_COMM_DETERMINE:
    /* determine and construct a subcomm that includes all of the procs with nonzero m and n */
    ierr = MPI_Comm_size(comm, &commsize); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &commrank); CHKERRQ(ierr);
    if(m || n) {
      subcommcolor = 1;
    }
    else {
      subcommcolor = MPI_UNDEFINED;
    }
    ierr = MPI_Comm_split(comm, subcommcolor, commrank, &subcomm); CHKERRQ(ierr);
    ierr = MPI_Comm_size(subcomm, &subcommsize);                   CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_ERR_USER, "Unknown block comm type: %d", commsize);
    break;
  }/* switch(subcommtype) */
  if(subcomm != MPI_COMM_NULL) {
    ierr = MatCreate(subcomm, &(block->mat)); CHKERRQ(ierr);
    ierr = MatSetSizes(block->mat,m,n,PETSC_DETERMINE,PETSC_DETERMINE); CHKERRQ(ierr);
    /**/
    if(!blockmattype) {
      ierr = MatSetType(block->mat,meta->default_block_type); CHKERRQ(ierr);
    }
    else {
      ierr = MatSetType(block->mat,blockmattype); CHKERRQ(ierr);
    }
  }
  else {
    block->mat = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}/* MatDD_BlockInit() */

#undef  __FUNCT__
#define __FUNCT__ "MatDDAddBlockLocal"
PetscErrorCode  MatDDAddBlockLocal(Mat A, PetscInt i, PetscInt j, const MatType blockmattype, MatDDBlockCommType blockcommtype, Mat *_B) {
  PetscErrorCode   ierr;
  Mat_DD*          dd = (Mat_DD*)A->data;
  MatDD_Block     *_block;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(i < 0 || i > dd->rmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "row domain id %d is invalid; must be >= 0 and < %d", i, dd->rmapdd->dcount);
  }
  if(j < 0 || j > dd->cmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "col domain id %d is invalid; must be >= 0 and < %d", j, dd->cmapdd->dcount);
  }
  ierr = meta->ops->locateblock(A,i,j, PETSC_TRUE, &_block); CHKERRQ(ierr);
  ierr = MatDD_BlockInit(A,i,j,blockmattype, blockcommtype, _block); CHKERRQ(ierr);
  if(_B) {
    ierr = MatDD_BlockGetMat(A, i, j, _block, _B); CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)(*_B)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* MatDDAddBlockLocal() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDMetaSetBlock"
PetscErrorCode  MatDDMetaSetBlock(Mat A, PetscInt i, PetscInt j, Mat B) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  Mat_DDMeta*          meta = (Mat_DDMeta*)A->data;
  MatDDMeta_Block     *_block;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(i < 0 || i > dd->rmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "row domain id %d is invalid; must be >= 0 and < %d", i, dd->rmapdd->dcount);
  }
  if(j < 0 || j > dd->cmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "col domain id %d is invalid; must be >= 0 and < %d", j, dd->cmapdd->dcount);
  }
  ierr = meta->ops->locateblock(A,i, j, PETSC_TRUE, &_block); CHKERRQ(ierr);
  ierr = MatDDMeta_BlockSetMat(A,i,j,_block, B); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(B)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMetaSetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDMetaGetBlock"
PetscErrorCode  MatDDMetaGetBlock(Mat A, PetscInt i, PetscInt j, Mat *_B) {
  PetscErrorCode     ierr;
  Mat_DD*            dd = (Mat_DD*)A->data;
  Mat_DDmeta*         meta = (Mat_DDMeta*)A->data;
  MatDDMeta_Block     *_block;
  
  PetscFunctionBegin;
  if(i < 0 || i > dd->rmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "row domain id %d is invalid; must be >= 0 and < %d", i, dd->rmapdd->dcount);
  }
  if(j < 0 || j > dd->cmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "col domain id %d is invalid; must be >= 0 and < %d", j, dd->cmapdd->dcount);
  }
  ierr = meta->ops->locateblock(A, i, j, PETSC_FALSE, &_block); CHKERRQ(ierr);
  if(_block == PETSC_NULL) {
    SETERRQ2(PETSC_ERR_USER, "Domain block not found: row %d col %d", i, j);
  }
  ierr = MatDDMetaBlockGetMat(A,i,j,_block, _B); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(*_B)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMetaGetBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatDDMetaRestoreBlock"
PetscErrorCode  MatDDMetaRestoreBlock(Mat A, PetscInt i, PetscInt j, Mat *_B) {
  PetscErrorCode     ierr;
  Mat_DD*            dd = (Mat_DD*)A->data;
  Mat_DDmeta*        meta = (Mat_DDMeta*)A->data;
  MatDDMeta_Block   *_block;
  Mat               *_B2;
  PetscInt          refcnt;
  
  PetscFunctionBegin;
  if(i < 0 || i > dd->rmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "row domain id %d is invalid; must be >= 0 and < %d", i, dd->rmapdd->dcount);
  }
  if(j < 0 || j > dd->cmapdd->dcount){
    SETERRQ2(PETSC_ERR_USER, "col domain block id %d is invalid; must be >= 0 and < %d", j, dd->cmapdd->dcount);
  }
  ierr = meta->ops->locateblock(A, i, j, PETSC_FALSE, &_block); CHKERRQ(ierr);
  if(_block == PETSC_NULL) {
    SETERRQ2(PETSC_ERR_USER, "Domain block not found: row %d col %d", i, j);
  }
  ierr = MatDDMetaBlockGetMat(A,i,j,_block, _B2); CHKERRQ(ierr);
  if(*_B != *_B2) {
    SETERRQ2(PETSC_ERR_USER, "Domain block mat for row %d col %d being restored is not the same that was gotten", i, j);
  }
  ierr = PetscObjectGetReference((PetscObject)(*_B), &refcnt); CHKERRQ(ierr);
  if(refcnt <= 1) {
    SETERRQ2(PETSC_ERR_USER, "Restoring domain block mat for row %d col %d too low a reference count.\nPerhaps restoring a block that was not gotten?", i, j);
  }
  ierr = PetscObjectReference((PetscObject)(*_B)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDMetaRestoreBlock() */


#undef __FUNCT__  
#define __FUNCT__ "MatDDSetGather"
PetscErrorCode MatDDSetGather(Mat A, Mat G) { 
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
#define __FUNCT__ "MatDDSetScatter"
PetscErrorCode MatDDSetScatter(Mat A, Mat S) { 
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



#undef  __FUNCT__
#define __FUNCT__ "MatDD_BlockFinit"
PetscErrorCode  MatDD_BlockFinit(Mat A, MatDD_Block *block) {
  PetscErrorCode        ierr;
  MPI_Comm              comm = ((PetscObject)A)->comm, subcomm = ((PetscObject)(block->mat))->comm;
  PetscMPIInt           flag;
  
  PetscFunctionBegin;
  
  ierr = MatDestroy(block->mat); CHKERRQ(ierr);
  /**/
  ierr = MPI_Comm_compare(subcomm, comm, &flag); CHKERRQ(ierr);
  if(flag != MPI_IDENT) {
    ierr = MPI_Comm_compare(subcomm, PETSC_COMM_SELF, &flag); CHKERRQ(ierr);
    if(flag != MPI_IDENT) {
      ierr = MPI_Comm_free(&subcomm); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}/* MatDD_BlockFinit() */



#undef  __FUNCT__
#define __FUNCT__ "MatDDSetDefaultBlockType"
PetscErrorCode  MatDDSetDefaltBlockType(Mat A, const MatType type) {
  Mat_DD  *dd = (Mat_DDMeta*)A->data;
  PetscFunctionBegin;
  if(!type){
    SETERRQ(PETSC_ERR_USER, "Unknown default block type");
  }
  dd->default_block_type = type;
  PetscFunctionReturn(0);
}/* MatDDSetDefaultBlockType()*/

#undef  __FUNCT__
#define __FUNCT__ "MatDDGetDefaultBlockType"
PetscErrorCode  MatDDGetDefaltBlockType(Mat A, const MatType *type) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  PetscFunctionBegin;
  *type = dd->default_block_type;
  PetscFunctionReturn(0);
}/* MatDDSetDefaultBlockType()*/









#undef  __FUNCT__
#define __FUNCT__ "MatCreate_DD_Private"
PetscErrorCode  MatCreate_DD_Private(Mat A, Mat_DD *dd) {
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

