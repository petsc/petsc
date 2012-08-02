#include <../src/mat/impls/aij/mpi/clique/matcliqueimpl.h> /*I "petscmat.h" I*/
/*
 Provides an interface to the Clique sparse solver
*/

#undef __FUNCT__
#define __FUNCT__ "MatView_Clique"
PetscErrorCode MatView_Clique(Mat A,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Clique"
PetscErrorCode MatDestroy_Clique(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_Clique"
PetscErrorCode MatSolve_Clique(Mat A,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_Clique"
PetscErrorCode MatCholeskyFactorNumeric_Clique(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_Clique"
PetscErrorCode MatCholeskyFactorSymbolic_Clique(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_Clique"
PetscErrorCode MatFactorGetSolverPackage_Clique(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCLIQUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/*
    The seq and mpi versions of this function are the same
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_clique"
PetscErrorCode MatGetFactor_aij_clique(Mat A,MatFactorType ftype,Mat *F)
{
  PetscFunctionBegin;
  printf("MatGetFactor_seqaij_Clique ...\n");
  Mat            B;
  Mat_Clique     *Acliq;
  PetscErrorCode ierr;
  //PetscInt       indx,m=A->rmap->n,n=A->cmap->n;  
  //PetscBool      flg;
  //const char     *colperm[]={"NATURAL","MMD_ATA","MMD_AT_PLUS_A","COLAMD"}; /* MY_PERMC - not supported by the petsc interface yet */
  //const char     *iterrefine[]={"NOREFINE", "SINGLE", "DOUBLE", "EXTRA"};
  //const char     *rowperm[]={"NOROWPERM", "LargeDiag"}; /* MY_PERMC - not supported by the petsc interface yet */

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_CHOLESKY){
    B->ops->choleskyfactorsymbolic  = MatCholeskyFactorSymbolic_Clique;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  B->ops->destroy          = MatDestroy_Clique;
  B->ops->view             = MatView_Clique;
  B->factortype            = ftype; 
  B->assembled             = PETSC_TRUE;  /* required by -ksp_view */
  //B->preallocated          = PETSC_TRUE;
  
  ierr = PetscNewLog(B,Mat_Clique,&Acliq);CHKERRQ(ierr);
  
  /*if (ftype == MAT_FACTOR_LU){
    lu->options.Equil = NO;
    }*/
  B->spptr = Acliq;
  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
