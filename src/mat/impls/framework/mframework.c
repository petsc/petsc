#define PETSCMAT_DLL

#include "private/matimpl.h"


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
  Here individual blocks are added with MatFrameworkAddBlock by specifying G_i,S_i or G_i, as appropriate.
  Block matricies are then set or retrieved by MatFrameworkSetBlock and MatFrameworkGetBlock.
  
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

typedef struct {
  Mat mat;
  VecScatter scatter, gather;
  Vec invec, outvec;      
} Mat_FrameworkBlock;

typedef struct {
  Mat_FrameworkBlock*    blocks;
  PetscInt           block_count;
  MatFrameworkType       composition_type;
  MatFrameworkMode       assembled_mode; 
  Mat                assembledForm;
} Mat_Framework;


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Framework"
PetscErrorCode MatDestroy_Framework(Mat mat)
{
  PetscErrorCode   ierr;
  Mat_Framework       *shell = (Mat_Framework*)mat->data;
  Mat_CompositeLink next = shell->head,oldnext;

  PetscFunctionBegin;
  while (next) {
    ierr = MatDestroy(next->mat);CHKERRQ(ierr);
    if (next->work && (!next->next || next->work != next->next->work)) {
      ierr = VecDestroy(next->work);CHKERRQ(ierr);
    }
    oldnext = next;
    next     = next->next;
    ierr     = PetscFree(oldnext);CHKERRQ(ierr);
  }
  if (shell->work) {ierr = VecDestroy(shell->work);CHKERRQ(ierr);}
  if (shell->left) {ierr = VecDestroy(shell->left);CHKERRQ(ierr);}
  if (shell->right) {ierr = VecDestroy(shell->right);CHKERRQ(ierr);}
  if (shell->leftwork) {ierr = VecDestroy(shell->leftwork);CHKERRQ(ierr);}
  if (shell->rightwork) {ierr = VecDestroy(shell->rightwork);CHKERRQ(ierr);}
  ierr      = PetscFree(shell);CHKERRQ(ierr);
  mat->data = 0;
  PetscFunctionReturn(0);
}/* MatDestroy_Framework() */


#undef  __FUNCT__
#define __FUNCT__ "MatCreate_Framework"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Framework(MPI_Comm comm, Mat A) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = new Mat_Framework;
  A->data = (void*)a;
  PetscFunctionReturn(0);
}/* MatCreate_Framework() */

#undef  __FUNCT__
#define __FUNCT__ "MatDestroy_Framework"
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_Framework(Mat A) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = A->data;
  A->data = (void*)PETSC_NULL;
  delete a;
  PetscFunctionReturn(0);
}// MatDestroy_Framework()


#undef  __FUNCT__
#define __FUNCT__ "MatMult_Framework"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_Framework(Mat A, Vec x, Vec y) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = A->data;
  /* IMPL: check Vec sizes */
  for(unsigned int i = 0; i < a->block.size(); ++i) {
    Mat M = a->block[i];/* Matrix block */
    Mat S = a->scatter[i];/* Scatter: restriction op that defines the input for block matrix M */
    Vec xx = a->invec[i]; /* Restricted vec -- input for M */
    ierr = MatMult(S,x,xx); CHKERRQ(ierr); /* Restrict x --> xx */
    Vec yy = a->outvec[i];/* Output of M -- restricted output */
    ierr = MatMult(M,xx,yy); CHKERRQ(ierr); /* Apply block */
    Mat G = a->gather[i]; /* Gather: prolongation op; injects yy into y */
    ierr = MatMultAdd(G,yy,y,y); CHKERRQ(ierr);/* Gather a piece of y from yy*/
  }
  PetscFunctionReturn(0);
}// MatMult_Framework()

#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_Framework"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_Framework(Mat A, Vec x, Vec y) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = A->data;
  /* IMPL: check Vec sizes (transposed) */
  for(unsigned int i = 0; i < a->block.size(); ++i) {
    /* IMPL: check if some Mat (G,B, or S) does not support MatMultTranspose */
    Mat G = a->gather[i];
    Vec xx = a->invec[i]; 
    ierr = MatMultTranspose(G,x,xx); CHKERRQ(ierr); 
    Mat M = a->block[i];
    Vec yy = a->outvec[i];
    ierr = MatMultTranspose(M,xx,yy); CHKERRQ(ierr); 
    Mat S = a->scatter[i]; 
    ierr = MatMultTransposeAdd(S,yy,y,y); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}// MatMultTranspose_Framework()

#undef  __FUNCT__
#define __FUNCT__ "MatFrameworkSetBlockEmbedding"
PetscErrorCode PETSCMAT_DLLEXPORT MatFrameworkSetBlockEmbedding(Mat A, PetscInt block_idx, IS S, IS G) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscInt       am,an,sm,sn,bm,bn,gm,gn;
  Vec            blockin, blockout;
  PetscFunctionBegin;
  a = A->data;
  ierr = MatGetLocalSize(A,&am,&an); CHKERRQ(ierr);
  ierr = MatGetLocalSize(S,&sm,&sn); CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn); CHKERRQ(ierr);
  ierr = MatGetLocalSize(G,&gm,&gn); CHKERRQ(ierr);
  if(an != sn) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Incompatible matrix column counts: framework (%d) and scatter (%d)", an,sn);
  }
  if(sm != bn) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Incompatible matrix sizes: scatter row count (%d) and block column count (%d)", sm,bn);
  }
  if(bm != gn) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Incompatible matrix sizes: block row count (%d) and gather column count (%d)", bm,gn);
  }
  if(an != sn) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Incompatible matrix row counts: framework (%d) and gather (%d)", am,gm);
  }
  a->scatter.append(S);
  a->block.append(B);
  a->block.append(G);
  ierr = VecCreateSeq(PETSC_COMM_SELF, bn,&blockin); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, bm,&blockout); CHKERRQ(ierr);
  a->invec.append(blockin);
  a->invec.append(blockout);
  PetscFunctionReturn(0);
}// MatFrameworkSetBlockEmbedding()


#undef  __FUNCT__
#define __FUNCT__ "MatFrameworkGetBlockEmbedding"
PetscErrorCode PETSCMAT_DLLEXPORT MatFrameworkGetBlockEmbedding(Mat A, PetscInt block_idx, IS* S, IS* G) {
  PetscErrorCode ierr;
  Mat_Framework *a;
  PetscFunctionBegin;
  a = A->data;
  // IMPL
  PetscFunctionReturn(0);
}// MatFrameworkGetBlockEmbedding()




#undef  __FUNCT__
#define __FUNCT__ "MatFrameworkGetBlockCount"
PetscErrorCode PETSCMAT_DLLEXPORT MatFrameworkGetBlockCount(Mat A, PetscInt *count) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = A->data;
  *count = a->block.size();
  PetscFunctionReturn(0);
}// MatFrameworkGetBlockCount()


#undef  __FUNCT__
#define __FUNCT__ "MatFrameworkGetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFrameworkGetBlock(Mat A, PetscInt i, Mat* B) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = A->data;
  if(i < 0 || i > a->block.size()) {
    /* IMPL: throw an error */
  }
  else {
    /* IMPL: branch on assembled_mode, and get a submatrix, is MAT_FRAMEWORK_ASSEMBLED */
    *B = a->block[i];
  }
  PetscFunctionReturn(0);
}// MatFrameworkGetBlock()

#undef  __FUNCT__
#define __FUNCT__ "MatFrameworkSetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFrameworkSetBlock(Mat A, PetscInt i, Mat B) {
  Mat_Framework  *a;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  a = A->data;
  if(i < 0 || i > a->block.size()) {
    /* IMPL: throw an error */
  }
  else {
    /* IMPL: branch on assembled_mode, and set a submatrix, is MAT_FRAMEWORK_ASSEMBLED */    
  }
  PetscFunctionReturn(0);
}// MatFrameworkSetBlock()





#endif
