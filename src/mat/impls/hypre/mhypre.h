
#if !defined(_MHYPRE_H)
#define _MHYPRE_H

#include <HYPRE.h>
#include <_hypre_IJ_mv.h>
#include <HYPRE_IJ_mv.h>

typedef struct {
  HYPRE_IJMatrix ij;
  HYPRE_IJVector x;
  HYPRE_IJVector b;
  MPI_Comm       comm;
  PetscBool      inner_free;
} Mat_HYPRE;

PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(Vec,HYPRE_IJVector*);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCopy(Vec,HYPRE_IJVector);

/*
    Replaces the address where the HYPRE vector points to its data with the address of
    PETSc's data. Saves the old address so it can be reset when we are finished with it.
    Allows users to get the data into a HYPRE vector without the cost of memcopies
*/
#define HYPREReplacePointer(b,newvalue,savedvalue) { \
    hypre_ParVector *par_vector   = (hypre_ParVector*)hypre_IJVectorObject(((hypre_IJVector*)b)); \
    hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector); \
    savedvalue         = local_vector->data; \
    local_vector->data = newvalue;          \
}


#endif
