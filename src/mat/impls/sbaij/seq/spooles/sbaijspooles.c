/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqSBAIJSpooles"
PetscErrorCode MatDestroy_SeqSBAIJSpooles(Mat A) 
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  /* SeqSBAIJ_Spooles isn't really the matrix that USES spooles, */
  /* rather it is a factory class for creating a symmetric matrix that can */
  /* invoke Spooles' sequential cholesky solver. */
  /* As a result, we don't have to clean up the stuff set by spooles */
  /* as in MatDestroy_SeqAIJ_Spooles. */
  ierr = MatConvert_Spooles_Base(A,MATSEQSBAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqSBAIJSpooles"
PetscErrorCode MatAssemblyEnd_SeqSBAIJSpooles(Mat A,MatAssemblyType mode) \
{
  PetscErrorCode ierr;
  PetscInt       bs;
  Mat_Spooles    *lu=(Mat_Spooles *)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (bs > 1) SETERRQ1(PETSC_ERR_SUP,"Block size %D not supported by Spooles",bs);
  lu->MatCholeskyFactorSymbolic  = A->ops->choleskyfactorsymbolic;
  A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJSpooles;  
  PetscFunctionReturn(0);
}

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
  Mat         B;
  Mat_Spooles *lu;   
  PetscErrorCode ierr;
  int m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,m,n,m,n,&B);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactornumeric  = MatFactorNumeric_SeqAIJSpooles;
  B->ops->getinertia             = MatGetInertia_SeqSBAIJSpooles;
  B->factor                      = FACTOR_CHOLESKY;  

  lu                        = (Mat_Spooles *)(B->spptr);
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->options.symflag       = SPOOLES_SYMMETRIC;   /* default */
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  *F = B;
  PetscFunctionReturn(0); 
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqSBAIJ_SeqSBAIJSpooles"
PetscErrorCode MatConvert_SeqSBAIJ_SeqSBAIJSpooles(Mat A,const MatType type,MatReuse reuse,Mat *newmat) {
  /* This routine is only called to convert a MATSEQSBAIJ matrix */
  /* to a MATSEQSBAIJSPOOLES matrix, so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat         B=*newmat;
  Mat_Spooles *lu;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    /* This routine is inherited, so we know the type is correct. */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  B->spptr                       = (void*)lu;

  lu->basetype                   = MATSEQSBAIJ;
  lu->CleanUpSpooles             = PETSC_FALSE;
  lu->MatDuplicate               = A->ops->duplicate;
  lu->MatCholeskyFactorSymbolic  = A->ops->choleskyfactorsymbolic;
  lu->MatLUFactorSymbolic        = A->ops->lufactorsymbolic; 
  lu->MatView                    = A->ops->view;
  lu->MatAssemblyEnd             = A->ops->assemblyend;
  lu->MatDestroy                 = A->ops->destroy;

  B->ops->duplicate              = MatDuplicate_Spooles;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJSpooles;
  B->ops->assemblyend            = MatAssemblyEnd_SeqSBAIJSpooles;
  B->ops->destroy                = MatDestroy_SeqSBAIJSpooles;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqsbaijspooles_seqsbaij_C",
                                           "MatConvert_Spooles_Base",MatConvert_Spooles_Base);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqsbaij_seqsbaijspooles_C",
                                           "MatConvert_SeqSBAIJ_SeqSBAIJSpooles",MatConvert_SeqSBAIJ_SeqSBAIJSpooles);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQSBAIJSPOOLES);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATSEQSBAIJSPOOLES - MATSEQSBAIJSPOOLES = "seqsbaijspooles" - A matrix type providing direct solvers (Cholesky) for sequential symmetric
  matrices via the external package Spooles.

  If Spooles is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes Spooles solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSEQSBAIJSPOOLES).

  This matrix inherits from MATSEQSBAIJ.  As a result, MatSeqSBAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSEQSBAIJ type without data copy.

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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqSBAIJSpooles"
PetscErrorCode MatCreate_SeqSBAIJSpooles(Mat A) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqSBAIJ */
  /*   and SeqSBAIJSpooles types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQSBAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqSBAIJ_SeqSBAIJSpooles(A,MATSEQSBAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATSBAIJSPOOLES - MATSBAIJSPOOLES = "sbaijspooles" - A matrix type providing direct solvers (Cholesky) for sequential and parallel symmetric matrices via the external package Spooles.

  If Spooles is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes Spooles solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSBAIJSPOOLES).

  This matrix inherits from MATSBAIJ.  As a result, MatSeqSBAIJSetPreallocation and MatMPISBAIJSetPreallocation are 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSBAIJ type without data copy.

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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SBAIJSpooles"
PetscErrorCode MatCreate_SBAIJSpooles(Mat A) 
{
  PetscErrorCode ierr;
  int size;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqSBAIJSpooles or MPISBAIJSpooles */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSBAIJSPOOLES);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatConvert_SeqSBAIJ_SeqSBAIJSpooles(A,MATSEQSBAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr   = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatConvert_MPISBAIJ_MPISBAIJSpooles(A,MATMPISBAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
EXTERN_C_END
