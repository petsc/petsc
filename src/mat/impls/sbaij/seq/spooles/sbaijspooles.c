#define PETSCMAT_DLL

/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"

/* 
  input:
   F:                 numeric factor
  output:
   nneg, nzero, npos: matrix inertia 
*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia_SeqSBAIJSpooles"
PetscErrorCode MatGetInertia_SeqSBAIJSpooles(Mat F,int *nneg,int *nzero,int *npos)
{ 
  Mat_Spooles *lu = (Mat_Spooles*)F->spptr; 
  int         neg,zero,pos;

  PetscFunctionBegin;
  FrontMtx_inertia(lu->frontmtx, &neg, &zero, &pos);
  if(nneg)  *nneg  = neg;
  if(nzero) *nzero = zero;
  if(npos)  *npos  = pos;
  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSBAIJSpooles"
PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJSpooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{ 
  Mat            B = *F;
  PetscErrorCode ierr;

  PetscFunctionBegin;	
  B->factor = FACTOR_CHOLESKY;  
  PetscFunctionReturn(0); 
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqsbaij_spooles"
PetscErrorCode MatGetFactor_seqsbaij_spooles(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Spooles    *lu;   

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ(PETSC_ERR_SUP,"Only Cholesky factorization is support for Spooles from SBAIJ matrix");
  ierr = MatCreate(((PetscObject)A)->comm,&B);
  ierr = MatSetSizes(B,A->rmap.n,A->cmap.n,A->rmap.n,A->cmap.n);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_Spooles,&lu);CHKERRQ(ierr);
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->options.symflag       = SPOOLES_SYMMETRIC;   /* default */
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJSpooles;
  B->ops->choleskyfactornumeric  = MatFactorNumeric_SeqSpooles;
  B->ops->getinertia             = MatGetInertia_SeqSBAIJSpooles;
  *F = B;
  PetscFunctionReturn(0);
}

/*MC
  MATSEQSBAIJSPOOLES - MATSEQSBAIJSPOOLES = "seqsbaijspooles" - A matrix type providing direct solvers (Cholesky) for sequential symmetric
  matrices via the external package Spooles.

  If Spooles is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes Spooles solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSEQSBAIJSPOOLES), then 
  optionally call MatSeqSBAIJSetPreallocation() or MatMPISBAIJSetPreallocation() DO NOT
  call MatCreateSeqSBAIJ/MPISBAIJ() directly or the preallocation information will be LOST!

  This matrix inherits from MATSEQSBAIJ.  As a result, MatSeqSBAIJSetPreallocation() is 
  supported for this matrix type.  One can also call MatConvert() for an inplace conversion to or from 
  the MATSEQSBAIJ type without data copy, after the matrix values have been set.

  Options Database Keys:
+ -mat_type seqsbaijspooles - sets the matrix type to seqsbaijspooles during calls to MatSetFromOptions()
. -mat_spooles_tau <tau> - upper bound on the magnitude of the largest element in L or U
. -mat_spooles_seed <seed> - random number seed used for ordering
. -mat_spooles_msglvl <msglvl> - message output level
. -mat_spooles_ordering <BestOfNDandMS,MMD,MS,ND> - ordering used
. -mat_spooles_maxdomainsize <n> - maximum subgraph size used by Spooles orderings
. -mat_spooles_maxzeros <n> - maximum number of zeros inside a supernode
. -mat_spooles_maxsize <n> - maximum size of a supernode
. -mat_spooles_FrontMtxInfo <true,fase> - print Spooles information about the computed factorization
. -mat_spooles_symmetryflag <0,1,2> - 0: SPOOLES_SYMMETRIC, 1: SPOOLES_HERMITIAN, 2: SPOOLES_NONSYMMETRIC
. -mat_spooles_patchAndGoFlag <0,1,2> - 0: no patch, 1: use PatchAndGo strategy 1, 2: use PatchAndGo strategy 2
. -mat_spooles_toosmall <dt> - drop tolerance for PatchAndGo strategy 1
. -mat_spooles_storeids <bool integer> - if nonzero, stores row and col numbers where patches were applied in an IV object
. -mat_spooles_fudge <delta> - fudge factor for rescaling diagonals with PatchAndGo strategy 2
- -mat_spooles_storevalues <bool integer> - if nonzero and PatchAndGo strategy 2 is used, store change in diagonal value in a DV object

   Level: beginner

.seealso: MATMPISBAIJSPOOLES, MATSEQAIJSPOOLES, MATMPIAIJSPOOLES, PCCHOLESKY
M*/

/*MC
  MATSBAIJSPOOLES - MATSBAIJSPOOLES = "sbaijspooles" - A matrix type providing direct solvers (Cholesky) for sequential and parallel symmetric matrices via the external package Spooles.

  If Spooles is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes Spooles solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSBAIJSPOOLES), then 
  optionally call MatSeqSBAIJSetPreallocation() or MatMPISBAIJSetPreallocation() DO NOT
  call MatCreateSeqSBAIJ/MPISBAIJ() directly or the preallocation information will be LOST!

  This matrix inherits from MATSBAIJ.  As a result, MatSeqSBAIJSetPreallocation() and MatMPISBAIJSetPreallocation() are 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSBAIJ type without data copy after the matrix values have been set.

  Options Database Keys:
+ -mat_type sbaijspooles - sets the matrix type to sbaijspooles during calls to MatSetFromOptions()
. -mat_spooles_tau <tau> - upper bound on the magnitude of the largest element in L or U
. -mat_spooles_seed <seed> - random number seed used for ordering
. -mat_spooles_msglvl <msglvl> - message output level
. -mat_spooles_ordering <BestOfNDandMS,MMD,MS,ND> - ordering used
. -mat_spooles_maxdomainsize <n> - maximum subgraph size used by Spooles orderings
. -mat_spooles_maxzeros <n> - maximum number of zeros inside a supernode
. -mat_spooles_maxsize <n> - maximum size of a supernode
. -mat_spooles_FrontMtxInfo <true,fase> - print Spooles information about the computed factorization
. -mat_spooles_symmetryflag <0,1,2> - 0: SPOOLES_SYMMETRIC, 1: SPOOLES_HERMITIAN, 2: SPOOLES_NONSYMMETRIC
. -mat_spooles_patchAndGoFlag <0,1,2> - 0: no patch, 1: use PatchAndGo strategy 1, 2: use PatchAndGo strategy 2
. -mat_spooles_toosmall <dt> - drop tolerance for PatchAndGo strategy 1
. -mat_spooles_storeids <bool integer> - if nonzero, stores row and col numbers where patches were applied in an IV object
. -mat_spooles_fudge <delta> - fudge factor for rescaling diagonals with PatchAndGo strategy 2
- -mat_spooles_storevalues <bool integer> - if nonzero and PatchAndGo strategy 2 is used, store change in diagonal value in a DV object

   Level: beginner

.seealso: MATMPISBAIJSPOOLES, MATSEQAIJSPOOLES, MATMPIAIJSPOOLES, PCCHOLESKY
M*/

