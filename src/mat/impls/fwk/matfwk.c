#define PETSCMAT_DLL

#include "private/matimpl.h"
#include <vector>


/*
  Creates a matrix out of blocks:
  M = \sum_{i=0}^{N-1} G_i M_i S_i
  where M_i are the blocks, 
  S_i is a scatter mapping from M's domain to M_i's domain,
  G_i are the gathers mapping from M_i's range to M's range.
  y = M*x proceeds by 
   - setting y to zero initially
   - scattering x to x_i = S_i*x
   - applying M_i: y_i = M_i*x_i
   - accumulating the partial result: y += G_i*y_i
   - x_i is referred to as the i-th "invec", and y_i as the i-th "outvec"
   - invecs and outvecs are allocated and held internally.
  This is called MAT_FRAMEWORK_ADDITIVE.

  An alternative is MAT_FRAMEWORK_MULTIPLICATIVE:
  M = \prod_{i=0}^{N-1} G_i M_i
  G_i, a gather, maps from M_i's range to M_{i+1}'s domain; G_{N-1} maps to the range of M.
  y = M*x proceeds by:
   - scattering x to x_0 = S_0*x
   - applying M_0: y_0 = M_0*x_0
   - gathering intermediate result to x_1 = G_0*y_0
   - continuing until x_{N-1}
   - gathering final result to y = G_{N-1}*y_{N-1}

  These are MAT_FRAMEWORK_DISASSEMBLED modes.
  Here individual blocks are added with MatFrameworkAddBlock by specifying G_i,S_i or G_i, as appropriate,
  and the added block is returned as a Mat.  This Mat is a "shell" that holds the scatter/gather pair 
  and can hold an internal Mat, specified by the user, implementing the action of the block.
  MatFrameworkBlockSetMat(Mat block, Mat action) or MatFrameworkBlockGetMat(Mat block, Mat *action)
  are used to set and retrieve the action Mat.
  
  An alternative is MAT_FRAMEWORK_ASSEMBLED
  Once the blocks are specified, the complete matrix is assembled into a format specified by
  MatFrameworkSetAssembledType.  
  The assembly is carried out by MatFrameworkAssemblyBegin/MatFrameworkAssemblyEnd.
  The assembly is essentially a collection of MatMatMultSymbolic calls, which calculate
  the matrix sparsity structure.
  The underlying assembled matrix can then be retrieved/set using MatFrameworkGetAssembledForm/MatFrameworkSetAssembledForm.
  The blocks are then set into the assembled matrix directly.
 */


/* 
Use cases that must work:

1. Efficient representation of a stiffness matrix for a 2D P1 triangular mesh using just the connectivity Map and an element Mat.
2. Assembly of a stiffness matrix for a geometrically conforming mortar element in 2D on a square doublet 
  (e.g., as in P. Fischer et al).
 a) Eliminate slaved degrees of freedom on the mortar.
 b) Add Lagrange multipliers on the mortar.
3. Efficient representation of a bordered matrix.  For example, a single row/col border arising from parameter
continuation.
4. Construction of a global pressure split scatter from a local scatter for Stokes on function spaces in 1 and 2.
5. Mesh connectivity presentation and (re)distribution using Map [speculative].
  - Allow Map composition, much like for Mats.
  - MatFrameworkAssemblyBegin/MatFrameworkAssemblyEnd and MapFrameworkAssemblyBegin/MapFrameworkAssemblyEnd
    will "flatten" out the Mat/Map objects respectively, by explicitly carrying out the composition.
    This can be used to (re)distribute connectivity graphs.
  - Alternatively, can use MatMatMult and MapMapMult.
6. MG
*/

/**/
/* Carry out MatMatMult on the factors, including preallocation, and put the result in a given MatType */
/*
EXTERN PetscErrorCode MatFrameworkAssemblyBegin(Mat framework, MatType assembledMatrixType, Mat *assembledForm);
EXTERN PetscErrorCode MatFrameworkAssemblyEnd(  Mat framework, MatType assembledMatrixType, Mat *assembledForm);
EXTERN PetscErrorCode MatFrameworkSetAssembledForm(Mat framework, Mat assembledForm);
EXTERN PetscErrorCode MatFrameworkGetAssembledForm(Mat framework, Mat *assembledForm);
EXTERN PetscErrorCode MatFrameworkSetMode(Mat framework, MatFrameworkAssembledMode mode);
EXTERN PetscErrorCode MatFrameworkGetMode(Mat framework, MatFrameworkAssembledMode *mode);
*/
/* Need to provide operations for 
  1. easy construction of gathers/scatters:
   - P1 simplicial gather
   - mortar element mesh gather (with and without constraints)
   - field split-off scatter from local splits
  2. easy combinations of gathers/scatters with certain types of blocks:
   - "tensor product" of a scatter with a block
*/

typedef struct _Mat_FwkSeqBlock {
  PetscInt                 rowblock, colblock;
  Mat                      mat;
  struct _Mat_FwkSeqBlock* next;
} Mat_FwkSeqBlock;

typedef struct {
  /* These defined the fundamental modes of operation of MatFwk */
  MatFwkType       composition_type;
  MatFwkMode       merged_mode; 
  /* These define the block structure of MatFwk */
  PetscInt         rowblockcount, colblockcount;
  PetscInt        *rowblocks, *colblocks;
  /* These are valid only if merged_mode is MATFWK_MODE_SPLIT */
  Vec            **rowvecs, **colvecs;
  Mat_FwkSeqBlock *blocks;
  PetscInt        *bi, *bj;
  /* These are valid only if merged_mode is MATFWK_MODE_MERGED */
  Mat              merged_mat;
  MatType          merged_mat_type;
} Mat_FwkSeq;


#undef  __FUNCT__
#define __FUNCT__ "MatCreate_FwkSeq"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_FwkSeq(PetscInt rowblockcount, PetscInt colblockcount, PetscInt m, PetscInt n, Mat *A) {
  Mat_FwkSeq  *fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n); CHKERRQ(ierr);
  ierr = PetscNewLog(*A,Mat_FwkSeq,&fwk);CHKERRQ(ierr);
  A->data = (void*)fwk;
  
  fwk->composition_type = MATFWK_ADDITIVE; /* FIX: for now this is the only composition type supported */
  fwk->merged_mode = MATFWK_MERGED;
  fwk->merged_mat_type = MATAIJ;

  fwk->rowblockcount = rowblockcount; /* FIX: check nonnegative */
  fwk->colblockcount = colblockcount; /* FIX: check nonnegative */
  ierr = PetscMalloc((rowblockcount*2)*sizeof(PetscInt),&fwk->rowblocks);CHKERRQ(ierr);
  ierr = PetscMemzero(fwk->rowblocks,(rowblockcount*2)*sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscMalloc((colblockcount*2)*sizeof(PetscInt),&fwk->colblocks);CHKERRQ(ierr);
  ierr = PetscMemzero(fwk->colblocks,(colblockcount*2)*sizeof(PetscInt)); CHKERRQ(ierr);

  ierr = MatFwk_SetupMergedMode_Private(*A, fwk->merged_mode); CHKERRQ(ierr);
  
  (*A)->ops->mult          = MatMult_FwkSeq;
  (*A)->ops->multtranspose = MatMultTranspose_FwkSeq;
  (*A)->assembled          = PETSC_FALSE;
  (*A)->ops->destroy       = MatDestroy_FwkSeq;
  PetscFunctionReturn(0);
}/* MatCreate_FwkSeq() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetCompositionType"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetCompositionType(Mat A, MatFwkType type) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fwk->composition_type = MATFWK_ADDITIVE;
  /* FIX: for now only MATFWK_ADDITIVE is supported */
  PetscFunctionReturn(0);
}/* MatFwkSetCompositionType()*/

#undef  __FUNCT__
#define __FUNCT__ "MatFwkGetCompositionType"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetCompositionType(Mat A, MatFwkType *type) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *type = fwk->composition_type;
  PetscFunctionReturn(0);
}/* MatFwkGetCompositionType() */


#undef  __FUNCT__
#define __FUNCT__ "MatFwk_SetupMergedMode_Private"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwk_SetupMergedMode_Private(Mat A, MatFwkMode mode) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(mode == MATFWK_MERGED) {
    /* Check if we need to clean up the split mode structures */
    /* CONTINUE: here */
  }
  else {
    /* Check if we need to clean up the merged mode structures */
    if(fwk->merged_mat) {
      ierr = MatDestroy(fwk->merged); CHKERRQ(ierr);
    }
    ierr = PetscMalloc((rowblockcount*2)*sizeof(Vec*),&fwk->rowvecs);CHKERRQ(ierr);
    ierr = PetscMalloc((colblockcount*2)*sizeof(Vec*),&fwk->colvecs);CHKERRQ(ierr);
    ierr = PetscMemzero(fwk->rowvecs,(rowblockcount*2)*sizeof(Vec*)); CHKERRQ(ierr);
    ierr = PetscMemzero(fwk->colvecs,(colblockcount*2)*sizeof(Vec*)); CHKERRQ(ierr);
  }

  fwk->blocks = PETSC_NULL;

  fwk->merged_mode = mode;

  PetscFunctionReturn(0);
}/* MatFwk_SetupMergedMode_Private()*/

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetMergedMode"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetMergedMode(Mat A, MatFwkMode mode) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(mode != fwk->merged_mode) {
    ierr = MatFwk_SetupMode_Private(A, mode); CHKERRQ(ierr);
  }
  fwk->merged_mode = mode;

  PetscFunctionReturn(0);
}/* MatFwkSetMergedMode()*/

#undef  __FUNCT__
#define __FUNCT__ "MatFwkGetMergedMode"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetMergedMode(Mat A, MatFwkMode *mode) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *mode = fwk->merged_mode;
  PetscFunctionReturn(0);
}/* MatFwkGetMergedMode() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetMergedMatType"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetMergedMatType(Mat A, const MatType type) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  Mat new_merged_mat;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(fwk->merged_mode == MATFWK_MERGED && fwk->merged_mat_type != type) {
    ierr = MatConvert(fwk->merged_mat, type, MAT_INITIAL_MATRIX, &new_merged_mat); CHKERRQ(ierr);
    ierr = MatDestroy(fwk->merged_mat); CHKERRQ(ierr);
    fwk->merged_mat = new_merged_mat;
  }
  fwk->merged_mat_type = type;
  PetscFunctionReturn(0);
}/* MatFwkSetMergedMatType()*/

#undef  __FUNCT__
#define __FUNCT__ "MatFwkGetMergedMatType"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetMergedMatType(Mat A, MatType *type) {
  Mat_FwkSeq  *fwk = (Mat_FwkSeq*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *type = fwk->merged_mat_type;
  PetscFunctionReturn(0);
}/* MatFwkGetMergedMatType() */


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_FwkSeq"
PetscErrorCode MatDestroy_FwkSeq(Mat mat)
{
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  PetscInt              i,j;

  PetscFunctionBegin;
  for(i = 0; i < fwk->rowblockcount; ++i) {
    if(!fwk->rowvecs[i]) {
      ierr = VecDestroy(*(fwk->rowvecs[i])); CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(fwk->rowvecs); CHKERRQ(ierr);
  for(j = 0; j < fwk->colblockcount; ++j) {
    if(!fwk->colvecs[j]) {
      ierr = VecDestroy(*(fwk->colvecs[j])); CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(fwk->colvecs); CHKERRQ(ierr);
  ierr = PetscFree(fwk->rowblocks);CHKERRQ(ierr);
  ierr = PetscFree(fwk->colblocks);CHKERRQ(ierr);

  
  while(!fwk->blocks) do {
    Mat_FwkSeqBlock *block = fwk->blocks;
    fwk->blocks = block->next;
    ierr = MatDestroy(block->mat); CHKERRQ(ierr);
    ierr = PetscFree(block); CHKERRQ(ierr);
  }
  ierr      = PetscFree(fwk);CHKERRQ(ierr);
  mat->data = 0;
  PetscFunctionReturn(0);
}/* MatDestroy_FwkSeq() */



#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqSetRowBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSeqSetRowBlock(Mat A, PetscInt rowblock, PetscInt offset, PetscInt size) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  Vec*                  v;
  
  PetscFunctionBegin;
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  fwk->rowblocks[2*rowblock] = offset;
  if(size != fwk->rowblocks[2*rowblock+1]) {
    if(fwk->rowvecs[rowblock]) {
      ierr = VecDestroy(*(fwk->rowvecs[rowblock])); CHKERRQ(ierr);
    }
    ierr = VecCreateSeq(size, fwk->rowvecs[rowblock]); CHKERRQ(ierr);
  }
  fwk->rowblocks[2*rowblock+1] = size;
  PetscFunctionReturn(0);
  
}/* MatFwkSeqSetRowBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqGetRowBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSeqGetRowBlock(Mat A, PetscInt rowblock, PetscInt *offset, PetscInt *size) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  
  PetscFunctionBegin;
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  *offset = fwk->rowblocks[2*rowblock];
  *size   = fwk->rowblocks[2*rowblock+1];
  PetscFunctionReturn(0);
}/* MatFwkSeqGetRowBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqSetColBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSeqSetColBlock(Mat A, PetscInt colblock, PetscInt offset, PetscInt size) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  Vec*                  v;
  
  PetscFunctionBegin;
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  fwk->colblocks[2*colblock] = offset;
  if(size != fwk->colblocks[2*colblock+1]) {
    if(fwk->colvecs[colblock]) {
      ierr = VecDestroy(*(fwk->colvecs[colblock])); CHKERRQ(ierr);
    }
    fwk->colvecs[colblock] = PETSC_NULL;
  }
  if(size) {
    ierr = VecCreateSeq(size, fwk->colvecs[colblock]); CHKERRQ(ierr);
  }
  fwk->colblocks[2*colblock+1] = size;
  PetscFunctionReturn(0);
  
}/* MatFwkSeqSetColBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqGetColBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSeqGetColBlock(Mat A, PetscInt colblock, PetscInt *offset, PetscInt *size) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  
  PetscFunctionBegin;
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  *offset = fwk->colblocks[2*colblock];
  *size   = fwk->colblocks[2*colblock+1];
  PetscFunctionReturn(0);
}/* MatFwkSeqGetColBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqSetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSeqSetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat blockmat, PetscTruth duplicate) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  Mat_FwkSeqBlock*      block;
  Mat                   _blockmat;
  PetscInt              blockm, blockn;
  
  PetscFunctionBegin;
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  ierr = MatGetSizes(blockmat, &blockm, &blockn); CHKERRQ(ierr);
  /* CONTINUE: check size compatibility */

  ierr = PetscMalloc(sizeof(Mat_FwkSeqBlock),&block);CHKERRQ(ierr);
  /* CONTINUE: duplicate blockmat --> _blockmat, if necessary, store it in the new block */
  block->mat = _blockmat;
  PetscFunctionReturn(0);
}/* MatFwkSeqSetBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqGetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSeqGetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat *block) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  
  PetscFunctionBegin;
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }

  PetscFunctionReturn(0);
}/* MatFwkSeqGetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatMult_FwkSeq"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_FwkSeq(Mat A, Vec x, Vec y) {
  Mat_FwkSeq  *fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fwk = (Mat_FwkSeq*)A->data;
  /* IMPL: check Vec sizes */
  for(unsigned int i = 0; i < fwk->block.size(); ++i) {
    Mat M = fwk->block[i];/* Matrix block */
    Mat S = fwk->scatter[i];/* Scatter: restriction op that defines the input for block matrix M */
    Vec xx = fwk->invec[i]; /* Restricted vec -- input for M */
    ierr = MatMult(S,x,xx); CHKERRQ(ierr); /* Restrict x --> xx */
    Vec yy = fwk->outvec[i];/* Output of M -- restricted output */
    ierr = MatMult(M,xx,yy); CHKERRQ(ierr); /* Apply block */
    Mat G = fwk->gather[i]; /* Gather: prolongation op; injects yy into y */
    ierr = MatMultAdd(G,yy,y,y); CHKERRQ(ierr);/* Gather a piece of y from yy*/
  }
  PetscFunctionReturn(0);
}// MatMult_FwkSeq()

#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_FwkSeq"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_FwkSeq(Mat A, Vec x, Vec y) {
  Mat_FwkSeq  *fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fwk = A->data;
  /* IMPL: check Vec sizes (transposed) */
  for(unsigned int i = 0; i < a->block.size(); ++i) {
    /* IMPL: check if some Mat (G,B, or S) does not support MatMultTranspose */
    Mat G = fwk->gather[i];
    Vec xx = fwk->invec[i]; 
    ierr = MatMultTranspose(G,x,xx); CHKERRQ(ierr); 
    Mat M = fwk->block[i];
    Vec yy = fwk->outvec[i];
    ierr = MatMultTranspose(M,xx,yy); CHKERRQ(ierr); 
    Mat S = fwk->scatter[i]; 
    ierr = MatMultTransposeAdd(S,yy,y,y); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}// MatMultTranspose_FwkSeq()

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSeqCreateScatterBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkCreateScatterBlock(Mat A, IS is, Mat *block) {
  PetscErrorCode        ierr;
  Mat_FwkSeq*           fwk = (Mat_FwkSeq*)mat->data;
  PetscFunctionBegin;
  /* CONTINUE: what are we doing here? */
  PetscFunctionReturn(0);
}/* MatFwkSeqCreateScatterBlock() */

#endif
