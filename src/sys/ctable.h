/* Contributed by - Mark Adams */

#if !defined(__CTABLE_H)
#define __CTABLE_H

typedef struct _p_PetscTable {
  PetscInt *keytable;
  PetscInt *table;
  PetscInt count;
  PetscInt tablesize;
  PetscInt head;
};

#include "petscctable.h"
#endif
