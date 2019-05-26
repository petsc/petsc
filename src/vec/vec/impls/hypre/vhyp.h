#if !defined(_VHYP_H)
#define _VHYP_H

#include <petscsys.h>
#include <HYPRE_IJ_mv.h>
#include <_hypre_IJ_mv.h>
/*
    Replaces the address where the HYPRE vector points to its data with the address of
  PETSc's data. Saves the old address so it can be reset when we are finished with it.
  Allows use to get the data into a HYPRE vector without the cost of memcopies
*/
#define VecHYPRE_ParVectorReplacePointer(b,newvalue,savedvalue) {                                 \
    hypre_ParVector *par_vector   = (hypre_ParVector*)hypre_IJVectorObject(((hypre_IJVector*)b)); \
    hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector);                       \
    savedvalue         = local_vector->data;                                                      \
    local_vector->data = newvalue;                                                                \
}

PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(Vec,HYPRE_IJVector*);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCopy(Vec,HYPRE_IJVector);

#endif
