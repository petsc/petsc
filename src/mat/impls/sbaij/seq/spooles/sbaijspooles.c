
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include <../src/mat/impls/aij/seq/spooles/spooles.h>

extern PetscErrorCode MatDestroy_SeqSBAIJ(Mat);

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqSBAIJSpooles"
PetscErrorCode MatDestroy_SeqSBAIJSpooles(Mat A)
{
  Mat_Spooles    *lu = (Mat_Spooles*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lu && lu->CleanUpSpooles) {
    FrontMtx_free(lu->frontmtx);
    IV_free(lu->newToOldIV);
    IV_free(lu->oldToNewIV);
    InpMtx_free(lu->mtxA);
    ETree_free(lu->frontETree);
    IVL_free(lu->symbfacIVL);
    SubMtxManager_free(lu->mtxmanager);
    Graph_free(lu->graph);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  ierr = MatDestroy_SeqSBAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)
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
#endif /* !defined(PETSC_USE_COMPLEX) */

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSBAIJSpooles"
PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJSpooles(Mat B,Mat A,IS r,const MatFactorInfo *info)
{ 
  PetscFunctionBegin;	
  B->ops->choleskyfactornumeric  = MatFactorNumeric_SeqSpooles;
#if !defined(PETSC_USE_COMPLEX)
  B->ops->getinertia             = MatGetInertia_SeqSBAIJSpooles;
#endif

  PetscFunctionReturn(0); 
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqsbaij_spooles"
PetscErrorCode MatFactorGetSolverPackage_seqsbaij_spooles(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSPOOLES;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqsbaij_spooles"
PetscErrorCode MatGetFactor_seqsbaij_spooles(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Spooles    *lu;   

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only Cholesky factorization is support for Spooles from SBAIJ matrix");
  ierr = MatCreate(((PetscObject)A)->comm,&B);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_Spooles,&lu);CHKERRQ(ierr);
  B->spptr = lu;
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->options.symflag       = SPOOLES_SYMMETRIC;   /* default */
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJSpooles;
  B->ops->destroy                = MatDestroy_SeqSBAIJSpooles;  
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqsbaij_spooles",MatFactorGetSolverPackage_seqsbaij_spooles);CHKERRQ(ierr);
  B->factortype                  = ftype;
  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
