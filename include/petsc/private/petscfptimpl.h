#pragma once
#include <petscviewertypes.h>
#include <petscsys.h>
/*
    Function pointer table that maps from function pointers to their string representation

    Does not use the PetscFunctionBegin/Return() or PetscCall() because these routines are called within those macros
*/
#define PetscCallQ(A) \
  do { \
    PetscErrorCode ierr = A; \
    if (ierr) return ierr; \
  } while (0);

typedef struct _n_PetscFPT *PetscFPT;
struct _n_PetscFPT {
  void     **functionpointer;
  char     **functionname;
  PetscCount count;
  PetscCount tablesize;
};
PETSC_INTERN PetscFPT PetscFPTData;

static inline PetscErrorCode PetscFPTView(PETSC_UNUSED PetscViewer viewer)
{
  if (PetscFPTData) {
    for (PetscCount i = 0; i < PetscFPTData->tablesize; ++i) {
      if (PetscFPTData->functionpointer[i]) printf("%s()\n", PetscFPTData->functionname[i]);
    }
  }
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscFPTDestroy(void)
{
  PetscFPT data = PetscFPTData;

  PetscFPTData = NULL;
  if (!data) return PETSC_SUCCESS;
  PetscCallQ(PetscFree(data->functionpointer));
  PetscCallQ(PetscFree(data->functionname));
  PetscCallQ(PetscFree(data));
  return PETSC_SUCCESS;
}

/*
   PetscFPTCreate  Creates a PETSc look up table from function pointers to strings

   Input Parameter:
.     n - expected number of keys

*/
static inline PetscErrorCode PetscFPTCreate(PetscCount n)
{
  PetscFPT _PetscFPTData;

  PetscCheck(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n < 0");
  /* Cannot use PetscNew() here because it is not yet defined in the include file chain */
  PetscCallQ(PetscMalloc(sizeof(struct _n_PetscFPT), &_PetscFPTData));
  if (n > PETSC_COUNT_MAX - n / 2 - 17) _PetscFPTData->tablesize = PETSC_COUNT_MAX / 4;
  else _PetscFPTData->tablesize = n + n / 2 + 17;
  PetscCallQ(PetscCalloc1(_PetscFPTData->tablesize, &_PetscFPTData->functionpointer));
  PetscCallQ(PetscMalloc1(_PetscFPTData->tablesize, &_PetscFPTData->functionname));
  _PetscFPTData->count = 0;
  PetscFPTData         = _PetscFPTData;
  return PETSC_SUCCESS;
}

static inline PetscCount PetscFPTHashPointer(void *ptr)
{
#define PETSC_FPT_HASH_FACT 79943
  return (PetscCount)((PETSC_FPT_HASH_FACT * ((size_t)ptr)) % (size_t)PetscFPTData->tablesize);
}

static inline PetscErrorCode PetscFPTAdd(void *key, const char *data)
{
  PetscCheck(data, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Null function name");
  if (!PetscFPTData) return PETSC_SUCCESS;
  for (PetscCount i = 0, hash = PetscFPTHashPointer(key); i < PetscFPTData->tablesize; ++i) {
    if (PetscFPTData->functionpointer[hash] == key) {
      PetscFPTData->functionname[hash] = (char *)data;
      return PETSC_SUCCESS;
    } else if (!PetscFPTData->functionpointer[hash]) {
      PetscFPTData->count++;
      PetscFPTData->functionpointer[hash] = key;
      PetscFPTData->functionname[hash]    = (char *)data;
      return PETSC_SUCCESS;
    }
    hash = (hash == (PetscFPTData->tablesize - 1)) ? 0 : hash + 1;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Function pointer table is full");
}

/*
    PetscFPTFind - checks if a function pointer is in the table

    If data==0, then no entry exists

*/
static inline PetscErrorCode PetscFPTFind(void *key, char const **data)
{
  PetscCount hash, ii = 0;

  *data = NULL;
  if (!PetscFPTData) return PETSC_SUCCESS;
  hash = PetscFPTHashPointer(key);
  while (ii++ < PetscFPTData->tablesize) {
    if (!PetscFPTData->functionpointer[hash]) break;
    else if (PetscFPTData->functionpointer[hash] == key) {
      *data = PetscFPTData->functionname[hash];
      break;
    }
    hash = (hash == (PetscFPTData->tablesize - 1)) ? 0 : hash + 1;
  }
  return PETSC_SUCCESS;
}
