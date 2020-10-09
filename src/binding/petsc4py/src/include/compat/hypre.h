#ifndef PETSC4PY_COMPAT_HYPRE_H
#define PETSC4PY_COMPAT_HYPRE_H

#if !defined(PETSC_HAVE_HYPRE)

#define PetscPCHYPREError do { \
    PetscFunctionBegin; \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HYPRE",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PCHYPREGetType(PETSC_UNUSED PC pc,PETSC_UNUSED const char *name[]){PetscPCHYPREError;}
PetscErrorCode PCHYPRESetType(PETSC_UNUSED PC pc,PETSC_UNUSED const char name[]){PetscPCHYPREError;}
PetscErrorCode PCHYPRESetDiscreteCurl(PETSC_UNUSED PC pc,PETSC_UNUSED Mat C){PetscPCHYPREError;}
PetscErrorCode PCHYPRESetDiscreteGradient(PETSC_UNUSED PC pc,PETSC_UNUSED Mat G){PetscPCHYPREError;}
PetscErrorCode PCHYPRESetAlphaPoissonMatrix(PETSC_UNUSED PC pc,PETSC_UNUSED Mat A){PetscPCHYPREError;}
PetscErrorCode PCHYPRESetBetaPoissonMatrix(PETSC_UNUSED PC pc,PETSC_UNUSED Mat B){PetscPCHYPREError;}
PetscErrorCode PCHYPRESetEdgeConstantVectors(PETSC_UNUSED PC pc,PETSC_UNUSED Vec ozz,PETSC_UNUSED Vec zoz,PETSC_UNUSED Vec zzo){PetscPCHYPREError;}

#undef PetscPCHYPREError

#endif

#endif/*PETSC4PY_COMPAT_HYPRE_H*/
