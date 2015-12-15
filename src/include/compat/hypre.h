#ifndef PETSC4PY_COMPAT_HYPRE_H
#define PETSC4PY_COMPAT_HYPRE_H

#if !defined(PETSC_HAVE_HYPRE)
#define PetscPCHYPREError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() requires HYPRE"); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)
#undef __FUNCT__
#define __FUNCT__ "PCHYPREGetType"
PetscErrorCode PCHYPREGetType(PETSC_UNUSED PC pc,PETSC_UNUSED const char *name[]){PetscPCHYPREError;}
#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetType"
PetscErrorCode PCHYPRESetType(PETSC_UNUSED PC pc,PETSC_UNUSED const char name[]){PetscPCHYPREError;}
#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetDiscreteCurl"
PetscErrorCode PCHYPRESetDiscreteCurl(PETSC_UNUSED PC pc,PETSC_UNUSED Mat C){PetscPCHYPREError;}
#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetDiscreteGradient"
PetscErrorCode PCHYPRESetDiscreteGradient(PETSC_UNUSED PC pc,PETSC_UNUSED Mat G){PetscPCHYPREError;}
#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetAlphaPoissonMatrix"
PetscErrorCode PCHYPRESetAlphaPoissonMatrix(PETSC_UNUSED PC pc,PETSC_UNUSED Mat A){PetscPCHYPREError;}
#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetBetaPoissonMatrix"
PetscErrorCode PCHYPRESetBetaPoissonMatrix(PETSC_UNUSED PC pc,PETSC_UNUSED Mat B){PetscPCHYPREError;}
#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetEdgeConstantVectors"
PetscErrorCode PCHYPRESetEdgeConstantVectors(PETSC_UNUSED PC pc,PETSC_UNUSED Vec ozz,PETSC_UNUSED Vec zoz,PETSC_UNUSED Vec zzo){PetscPCHYPREError;}
#undef PetscPCHYPREError
#endif

#endif/*PETSC4PY_COMPAT_HYPRE_H*/
