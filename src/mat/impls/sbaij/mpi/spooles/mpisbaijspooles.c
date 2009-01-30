#define PETSCMAT_DLL

/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/

#include "../src/mat/impls/aij/seq/spooles/spooles.h"
#include "../src/mat/impls/sbaij/mpi/mpisbaij.h"

#if !defined(PETSC_USE_COMPLEX)
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
  Mat_Spooles    *lu = (Mat_Spooles*)F->spptr; 
  PetscErrorCode ierr;
  int            neg,zero,pos,sbuf[3],rbuf[3];

  PetscFunctionBegin;
  FrontMtx_inertia(lu->frontmtx, &neg, &zero, &pos);
  sbuf[0] = neg; sbuf[1] = zero; sbuf[2] = pos;
  ierr = MPI_Allreduce(sbuf,rbuf,3,MPI_INT,MPI_SUM,((PetscObject)F)->comm);CHKERRQ(ierr);
  *nneg  = rbuf[0]; *nzero = rbuf[1]; *npos  = rbuf[2];
  PetscFunctionReturn(0);
}
#endif /* !defined(PETSC_USE_COMPLEX) */

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPISBAIJSpooles"
PetscErrorCode MatCholeskyFactorSymbolic_MPISBAIJSpooles(Mat B,Mat A,IS r,const MatFactorInfo *info)
{
  PetscFunctionBegin;	
  (B)->ops->choleskyfactornumeric  = MatFactorNumeric_MPISpooles;
  PetscFunctionReturn(0); 
}

extern PetscErrorCode MatDestroy_MPISBAIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPISBAIJSpooles"
PetscErrorCode MatDestroy_MPISBAIJSpooles(Mat A)
{
  Mat_Spooles   *lu = (Mat_Spooles*)A->spptr; 
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (lu->CleanUpSpooles) {
    FrontMtx_free(lu->frontmtx);        
    IV_free(lu->newToOldIV);            
    IV_free(lu->oldToNewIV); 
    IV_free(lu->vtxmapIV);
    InpMtx_free(lu->mtxA);             
    ETree_free(lu->frontETree);          
    IVL_free(lu->symbfacIVL);         
    SubMtxManager_free(lu->mtxmanager);    
    DenseMtx_free(lu->mtxX);
    DenseMtx_free(lu->mtxY);
    ierr = MPI_Comm_free(&(lu->comm_spooles));CHKERRQ(ierr);
    if ( lu->scat ){
      ierr = VecDestroy(lu->vec_spooles);CHKERRQ(ierr); 
      ierr = ISDestroy(lu->iden);CHKERRQ(ierr); 
      ierr = ISDestroy(lu->is_petsc);CHKERRQ(ierr);
      ierr = VecScatterDestroy(lu->scat);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy_MPISBAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_mpisbaij_spooles"
PetscErrorCode MatFactorGetSolverPackage_mpisbaij_spooles(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MAT_SOLVER_SPOOLES;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpisbaij_spooles"
PetscErrorCode MatGetFactor_mpisbaij_spooles(Mat A,MatFactorType ftype,Mat *F)
{
  Mat_Spooles    *lu;
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */  
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscNewLog(B,Mat_Spooles,&lu);CHKERRQ(ierr);
  B->spptr          = lu;
  lu->flg           = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR = PETSC_FALSE;

  if (ftype == MAT_FACTOR_CHOLESKY) {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MPISBAIJSpooles;
    B->ops->view                   = MatView_Spooles;
    B->ops->destroy                = MatDestroy_MPISBAIJSpooles;  
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mpisbaij_spooles",MatFactorGetSolverPackage_mpisbaij_spooles);CHKERRQ(ierr);

    lu->options.symflag      = SPOOLES_SYMMETRIC;
    lu->options.pivotingflag = SPOOLES_NO_PIVOTING; 
  } else SETERRQ(PETSC_ERR_SUP,"Only Cholesky for SBAIJ matrices, use AIJ for LU");

  B->factor = ftype;
  ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(lu->comm_spooles));CHKERRQ(ierr);
  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END

/*MC
  MAT_SOLVER_SPOOLES - "spooles" - a matrix type providing direct solvers (LU and Cholesky) for distributed symmetric
  and non-symmetric  matrices via the external package Spooles.

  If Spooles is installed (run config/configure.py with the option --download-spooles)

  Options Database Keys:
+ -mat_spooles_tau <tau> - upper bound on the magnitude of the largest element in L or U
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

.seealso: MAT_SOLVER_SUPERLU, MAT_SOLVER_MUMPS, MAT_SOLVER_SUPERLU_DIST, PCFactorSetMatSolverPackage(), MatSolverPackage 
M*/

