#define PETSCMAT_DLL

/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"

/* 
  input:
   F:                 numeric factor
  output:
   nneg, nzero, npos: global matrix inertia in all processors
*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia_MPISBAIJSpooles"
PetscErrorCode MatGetInertia_MPISBAIJSpooles(Mat F,int *nneg,int *nzero,int *npos)
{ 
  Mat_Spooles *lu = (Mat_Spooles*)F->spptr; 
  PetscErrorCode ierr;
  int neg,zero,pos,sbuf[3],rbuf[3];

  PetscFunctionBegin;
  FrontMtx_inertia(lu->frontmtx, &neg, &zero, &pos);
  sbuf[0] = neg; sbuf[1] = zero; sbuf[2] = pos;
  ierr = MPI_Allreduce(sbuf,rbuf,3,MPI_INT,MPI_SUM,((PetscObject)F)->comm);CHKERRQ(ierr);
  *nneg  = rbuf[0]; *nzero = rbuf[1]; *npos  = rbuf[2];
  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPISBAIJSpooles"
PetscErrorCode MatCholeskyFactorSymbolic_MPISBAIJSpooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{
  Mat           B = *F;
  Mat_Spooles   *lu;   
  PetscErrorCode ierr;
  
  PetscFunctionBegin;	
  B->factor                     = FACTOR_CHOLESKY;  

  lu                       = (Mat_Spooles*)(B->spptr);
  lu->options.pivotingflag = SPOOLES_NO_PIVOTING; 
  lu->flg                  = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR        = PETSC_FALSE;
  lu->options.symflag      = SPOOLES_SYMMETRIC;  /* default */

  ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(lu->comm_spooles));CHKERRQ(ierr);
  *F = B;
  PetscFunctionReturn(0); 
}


/*MC
  MATMPISBAIJSPOOLES - MATMPISBAIJSPOOLES = "mpisbaijspooles" - a matrix type providing direct solvers (Cholesky) for distributed symmetric
  matrices via the external package Spooles.

  If Spooles is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes Spooles solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATMPISBAIJSPOOLES), then 
  optionally call MatSeqSBAIJSetPreallocation() or MatMPISBAIJSetPreallocation() DO NOT
  call MatCreateSeqSBAIJ/MPISBAIJ() directly or the preallocation information will be LOST!

  This matrix inherits from MATMPISBAIJ.  As a result, MatMPISBAIJSetPreallocation() is 
  supported for this matrix type.  One can also call MatConvert() for an inplace conversion to or from 
  the MATMPISBAIJ type without data copy AFTER the matrix values have been set.

  Options Database Keys:
+ -mat_type mpisbaijspooles - sets the matrix type to mpisbaijspooles during a call to MatSetFromOptions()
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

.seealso: MATSEQSBAIJSPOOLES, MATSEQAIJSPOOLES, MATMPIAIJSPOOLES, PCCHOLESKY
M*/

