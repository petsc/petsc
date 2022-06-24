
#if !defined(PETSCFPIMPL_H)
#define PETSCFPIMPL_H
#include <petscviewertypes.h>
#include <petscsys.h>
/*
    Function pointer table that maps from function pointers to their string representation

    Does not use the PetscFunctionBegin/Return() or PetscCall() because these routines are called within those macros
*/
#define PetscCallQ(A) do {PetscErrorCode ierr = A; if (ierr) return(ierr);} while (0);

typedef struct _n_PetscFPT* PetscFPT;
struct _n_PetscFPT {
  void     **functionpointer;
  char     **functionname;
  PetscInt count;
  PetscInt tablesize;
};
PETSC_INTERN PetscFPT PetscFPTData;

static inline PetscErrorCode PetscFPTView(PetscViewer viewer)
{
  if (PetscFPTData) {
    for (PetscInt i = 0; i < PetscFPTData->tablesize; ++i) {
      if (PetscFPTData->functionpointer[i]) {
        printf("%s()\n",PetscFPTData->functionname[i]);
      }
    }
  }
  return 0;
}

static inline PetscErrorCode PetscFPTDestroy(void)
{
  PetscFPT data = PetscFPTData;

  PetscFPTData = NULL;
  if (!data) return 0;
  PetscCallQ(PetscFree(data->functionpointer));
  PetscCallQ(PetscFree(data->functionname));
  PetscCallQ(PetscFree(data));
  return 0;
}

/*
   PetscFPTCreate  Creates a PETSc look up table from function pointers to strings

   Input Parameters:
.     n - expected number of keys

*/
static inline PetscErrorCode  PetscFPTCreate(PetscInt n)
{
  PetscFPT _PetscFPTData;

  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n < 0");
  /* Cannot use PetscNew() here because it is not yet defined in the include file chain */
  PetscCallQ(PetscMalloc(sizeof(struct _n_PetscFPT),&_PetscFPTData));
  _PetscFPTData->tablesize = (3*n)/2 + 17;
  if (_PetscFPTData->tablesize < n) _PetscFPTData->tablesize = PETSC_MAX_INT/4; /* overflow */
  PetscCallQ(PetscCalloc(sizeof(void*)*_PetscFPTData->tablesize,&_PetscFPTData->functionpointer));
  PetscCallQ(PetscMalloc(sizeof(char**)*_PetscFPTData->tablesize,&_PetscFPTData->functionname));
  _PetscFPTData->count = 0;
  PetscFPTData = _PetscFPTData;
  return(0);
}

static inline unsigned long PetscFPTHashPointer(void *ptr)
{
#define PETSC_FPT_HASH_FACT 79943
  return((PETSC_FPT_HASH_FACT*((size_t)ptr))%PetscFPTData->tablesize);
}

static inline PetscErrorCode PetscFPTAdd(void* key,const char* data)
{
  PetscCheck(data,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null function name");
  if (!PetscFPTData) return 0;
  for (PetscInt i = 0, hash = (PetscInt)PetscFPTHashPointer(key); i < PetscFPTData->tablesize; ++i) {
    if (PetscFPTData->functionpointer[hash] == key) {
      PetscFPTData->functionname[hash] = (char*) data;
      return 0;
    } else if (!PetscFPTData->functionpointer[hash]) {
      PetscFPTData->count++;
      PetscFPTData->functionpointer[hash] = key;
      PetscFPTData->functionname[hash] = (char*) data;
      return 0;
    }
    hash = (hash == (PetscFPTData->tablesize-1)) ? 0 : hash+1;
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Function pointer table is full");
}

/*
    PetscFPTFind - checks if a function pointer is in the table

    If data==0, then no entry exists

*/
static inline PetscErrorCode  PetscFPTFind(void* key,char const **data)
{
  PetscInt hash,ii = 0;

  *data = NULL;
  if (!PetscFPTData) return(0);
  hash = PetscFPTHashPointer(key);
  while (ii++ < PetscFPTData->tablesize) {
    if (!PetscFPTData->functionpointer[hash]) break;
    else if (PetscFPTData->functionpointer[hash] == key) {
      *data = PetscFPTData->functionname[hash];
      break;
    }
    hash = (hash == (PetscFPTData->tablesize-1)) ? 0 : hash+1;
  }
  return 0;
}

#endif
