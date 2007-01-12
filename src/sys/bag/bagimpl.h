#if !defined(__BAGIMPL)
#define __BAGIMPL
#include "petscbag.h"

#define PETSC_BAG_NAME_LENGTH 64
#define PETSC_BAG_HELP_LENGTH 128
#define PETSC_BAG_FILE_COOKIE 1211219

struct _n_PetscBagItem {
  PetscDataType dtype;
  PetscInt      offset;
  PetscInt      msize;
  char          name[PETSC_BAG_NAME_LENGTH],help[PETSC_BAG_HELP_LENGTH]; 
  const char    **list;
  PetscTruth    freelist;
  PetscBagItem  next;
};

struct _n_PetscBag {
  MPI_Comm     bagcomm;
  PetscInt     bagsize;
  PetscInt     count;
  char         bagname[PETSC_BAG_NAME_LENGTH];
  char         baghelp[PETSC_BAG_HELP_LENGTH];
  PetscBagItem bagitems;
};


#endif
