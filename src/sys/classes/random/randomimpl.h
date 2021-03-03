
#if !defined(PETSCRANDOMIMPL_H)
#define PETSCRANDOMIMPL_H

#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool PetscRandomRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscRandomRegisterAll(void);

typedef struct _PetscRandomOps *PetscRandomOps;
struct _PetscRandomOps {
  /* 0 */
  PetscErrorCode (*seed)(PetscRandom);
  PetscErrorCode (*getvalue)(PetscRandom,PetscScalar*);
  PetscErrorCode (*getvaluereal)(PetscRandom,PetscReal*);
  PetscErrorCode (*getvalues)(PetscRandom,PetscInt,PetscScalar*);
  PetscErrorCode (*getvaluesreal)(PetscRandom,PetscInt,PetscReal*);
  PetscErrorCode (*destroy)(PetscRandom);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscRandom);
};

struct _p_PetscRandom {
  PETSCHEADER(struct _PetscRandomOps);
  void          *data;         /* implementation-specific data */
  unsigned long seed;
  PetscScalar   low,width;     /* lower bound and width of the interval over
                                  which the random numbers are distributed */
  PetscBool iset;              /* if true, indicates that the user has set the interval */
  /* array for shuffling ??? */
};

#endif

