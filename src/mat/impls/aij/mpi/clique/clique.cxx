#include <../src/mat/impls/aij/mpi/clique/matcliqueimpl.h> /*I "petscmat.h" I*/
/*
 Provides an interface to the Clique sparse solver
*/

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
  PetscFunctionReturn(0);
}
EXTERN_C_END
