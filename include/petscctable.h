#ifndef PETSCCTABLE_H
#define PETSCCTABLE_H

#include <petscsys.h>

#if defined(PETSC_SKIP_PETSCTABLE_DEPRECATION_WARNING)
  #define PETSC_TABLE_DEPRECATION_WARNING(...)
#else
  #define PETSC_TABLE_DEPRECATION_WARNING(...) PETSC_DEPRECATED_FUNCTION(__VA_ARGS__ "(since version 3.19)")
#endif

struct _n_PetscTable {
  PetscInt *keytable;
  PetscInt *table;
  PetscInt  count;
  PetscInt  tablesize;
  PetscInt  head;
  PetscInt  maxkey; /* largest key allowed */
};

typedef struct _n_PetscTable *PetscTable;
typedef PetscInt             *PetscTablePosition;

#define PetscHashMacroImplToGetAroundDeprecationWarning_Private(ta, x) (((unsigned long)(x)) % ((unsigned long)((ta)->tablesize)))

PETSC_TABLE_DEPRECATION_WARNING("") static inline unsigned long PetscHash(PetscTable ta, unsigned long x)
{
  return PetscHashMacroImplToGetAroundDeprecationWarning_Private(ta, x);
}

#define PetscHashStepMacroImplToGetAroundDeprecationWarning_Private(ta, x) (1 + (((unsigned long)(x)) % ((unsigned long)((ta)->tablesize - 1))))

PETSC_TABLE_DEPRECATION_WARNING("") static inline unsigned long PetscHashStep(PetscTable ta, unsigned long x)
{
  return PetscHashStepMacroImplToGetAroundDeprecationWarning_Private(ta, x);
}

PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapICreateWithSize()") PETSC_EXTERN PetscErrorCode PetscTableCreate(PetscInt, PetscInt, PetscTable *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapIDuplicate()") PETSC_EXTERN PetscErrorCode PetscTableCreateCopy(PetscTable, PetscTable *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapIDestroy()") PETSC_EXTERN PetscErrorCode PetscTableDestroy(PetscTable *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapIGetSize()") PETSC_EXTERN PetscErrorCode PetscTableGetCount(PetscTable, PetscInt *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapIGetSize()") PETSC_EXTERN PetscErrorCode PetscTableIsEmpty(PetscTable, PetscInt *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapISetWithMode()") PETSC_EXTERN PetscErrorCode PetscTableAddExpand(PetscTable, PetscInt, PetscInt, InsertMode);
PETSC_TABLE_DEPRECATION_WARNING("") PETSC_EXTERN PetscErrorCode PetscTableAddCountExpand(PetscTable, PetscInt);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHashIterBegin()") PETSC_EXTERN PetscErrorCode PetscTableGetHeadPosition(PetscTable, PetscTablePosition *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHashIterNext(), PetscHashIterGetKey(), and PetscHashIterGetVal()") PETSC_EXTERN PetscErrorCode PetscTableGetNext(PetscTable, PetscTablePosition *, PetscInt *, PetscInt *);
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapIClear()") PETSC_EXTERN PetscErrorCode PetscTableRemoveAll(PetscTable);

PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapISetWithMode()") static inline PetscErrorCode PetscTableAdd(PetscTable ta, PetscInt key, PetscInt data, InsertMode imode)
{
  PetscInt i, hash = (PetscInt)PetscHashMacroImplToGetAroundDeprecationWarning_Private(ta, key);
  PetscInt hashstep = (PetscInt)PetscHashStepMacroImplToGetAroundDeprecationWarning_Private(ta, key);

  PetscFunctionBegin;
  PetscCheck(key > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "key (value %" PetscInt_FMT ") <= 0", key);
  PetscCheck(key <= ta->maxkey, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "key %" PetscInt_FMT " is greater than largest key allowed %" PetscInt_FMT, key, ta->maxkey);
  PetscCheck(data, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Null data");

  for (i = 0; i < ta->tablesize; i++) {
    if (ta->keytable[hash] == key) {
      switch (imode) {
      case INSERT_VALUES:
        ta->table[hash] = data; /* over write */
        break;
      case ADD_VALUES:
        ta->table[hash] += data;
        break;
      case MAX_VALUES:
        ta->table[hash] = PetscMax(ta->table[hash], data);
        break;
      case MIN_VALUES:
        ta->table[hash] = PetscMin(ta->table[hash], data);
        break;
      case NOT_SET_VALUES:
      case INSERT_ALL_VALUES:
      case ADD_ALL_VALUES:
      case INSERT_BC_VALUES:
      case ADD_BC_VALUES:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported InsertMode");
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (!ta->keytable[hash]) {
      if (ta->count < 5 * (ta->tablesize / 6) - 1) {
        ta->count++; /* add */
        ta->keytable[hash] = key;
        ta->table[hash]    = data;
      } else PetscCall(PetscTableAddExpand(ta, key, data, imode));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    hash = (hash + hashstep) % ta->tablesize;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_COR, "Full table");
  /* PetscFunctionReturn(PETSC_SUCCESS); */
}

PETSC_TABLE_DEPRECATION_WARNING("") static inline PetscErrorCode PetscTableAddCount(PetscTable ta, PetscInt key)
{
  PetscInt i, hash = (PetscInt)PetscHashMacroImplToGetAroundDeprecationWarning_Private(ta, key);
  PetscInt hashstep = (PetscInt)PetscHashStepMacroImplToGetAroundDeprecationWarning_Private(ta, key);

  PetscFunctionBegin;
  PetscCheck(key > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "key (value %" PetscInt_FMT ") <= 0", key);
  PetscCheck(key <= ta->maxkey, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "key %" PetscInt_FMT " is greater than largest key allowed %" PetscInt_FMT, key, ta->maxkey);

  for (i = 0; i < ta->tablesize; i++) {
    if (ta->keytable[hash] == key) {
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (!ta->keytable[hash]) {
      if (ta->count < 5 * (ta->tablesize / 6) - 1) {
        ta->count++; /* add */
        ta->keytable[hash] = key;
        ta->table[hash]    = ta->count;
      } else PetscCall(PetscTableAddCountExpand(ta, key));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    hash = (hash + hashstep) % ta->tablesize;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_COR, "Full table");
  /* PetscFunctionReturn(PETSC_SUCCESS); */
}

/*
    PetscTableFind - finds data in table from a given key, if the key is valid but not in the table returns 0
*/
PETSC_TABLE_DEPRECATION_WARNING("Use PetscHMapIGetWithDefault()") static inline PetscErrorCode PetscTableFind(PetscTable ta, PetscInt key, PetscInt *data)
{
  PetscInt ii       = 0;
  PetscInt hash     = (PetscInt)PetscHashMacroImplToGetAroundDeprecationWarning_Private(ta, key);
  PetscInt hashstep = (PetscInt)PetscHashStepMacroImplToGetAroundDeprecationWarning_Private(ta, key);

  PetscFunctionBegin;
  *data = 0;
  PetscCheck(key > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "key (value %" PetscInt_FMT ") <= 0", key);
  PetscCheck(key <= ta->maxkey, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "key %" PetscInt_FMT " is greater than largest key allowed %" PetscInt_FMT, key, ta->maxkey);

  while (ii++ < ta->tablesize) {
    if (!ta->keytable[hash]) break;
    else if (ta->keytable[hash] == key) {
      *data = ta->table[hash];
      break;
    }
    hash = (hash + hashstep) % ta->tablesize;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef PetscHashMacroImplToGetAroundDeprecationWarning_Private
#undef PetscHashStepMacroImplToGetAroundDeprecationWarning_Private
#undef PETSC_TABLE_DEPRECATION_WARNING

#endif /* PETSCCTABLE_H */
