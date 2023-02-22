#ifndef PETSC_HASHSET_H
#define PETSC_HASHSET_H

#include <petsc/private/hashtable.h>

/* SUBMANSEC = PetscH */

/*MC
  PETSC_HASH_SET - Instantiate a new PETSc hash set type

  Synopsis:
  #include <petsc/private/hashset.h>
  PETSC_HASH_SET(HSetT, KeyType, HashFunc, EqualFunc)

  Input Parameters:
+ HSetT - The hash set type name suffix, i.e. the name of the object created is PetscHSet<HSetT>
. KeyType - The type of entries, may be a basic type such as int or a struct
. HashFunc - Routine or function-like macro that computes hash values from entries
- EqualFunc - Routine or function-like macro that computes whether two values are equal

  Level: developer

  Developer Note:
    Each time this macro is used to create a new hash set type, the make rule for allmanpages in $PETSC_DIR/makefile should
    be updated to cause the automatic generation of appropriate manual pages for that type. The manual pages
    are generated from the templated version of the documentation in include/petsc/private/hashset.txt.

  References:
    This code uses the standalone and portable C language khash software https://github.com/attractivechaos/klib

.seealso: `PetscHSetI`, `PetscHSetICreate()`, `PetscHSetIJ`, `PetscHSetIJCreate()`, `PETSC_HASH_MAP()`
M*/

#define PETSC_HASH_SET(HashT, KeyType, HashFunc, EqualFunc) \
\
  KHASH_INIT(HashT, KeyType, char, 0, HashFunc, EqualFunc) \
\
  typedef khash_t(HashT) *Petsc##HashT; \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Create(Petsc##HashT *ht) \
  { \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    *ht = kh_init(HashT); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Destroy(Petsc##HashT *ht) \
  { \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    if (!*ht) PetscFunctionReturn(PETSC_SUCCESS); \
    kh_destroy(HashT, *ht); \
    *ht = NULL; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Reset(Petsc##HashT ht) \
  { \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    kh_reset(HashT, ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Duplicate(Petsc##HashT ht, Petsc##HashT *hd) \
  { \
    int     ret; \
    KeyType key; \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(hd, 2)); \
    *hd = kh_init(HashT); \
    ret = kh_resize(HashT, *hd, kh_size(ht)); \
    PetscHashAssert(ret == 0); \
    kh_foreach_key(ht, key, { \
      kh_put(HashT, *hd, key, &ret); \
      PetscHashAssert(ret >= 0); \
    }) PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Update(Petsc##HashT ht, Petsc##HashT hta) \
  { \
    int     ret; \
    KeyType key; \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(hta, 2)); \
    kh_foreach_key(hta, key, { \
      kh_put(HashT, ht, key, &ret); \
      PetscHashAssert(ret >= 0); \
    }) PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Clear(Petsc##HashT ht) \
  { \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    kh_clear(HashT, ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Resize(Petsc##HashT ht, PetscInt nb) \
  { \
    int ret; \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    ret = kh_resize(HashT, ht, (khint_t)nb); \
    PetscHashAssert(ret == 0); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetSize(Petsc##HashT ht, PetscInt *n) \
  { \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidIntPointer(n, 2)); \
    *n = (PetscInt)kh_size(ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetCapacity(Petsc##HashT ht, PetscInt *n) \
  { \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidIntPointer(n, 2)); \
    *n = (PetscInt)kh_n_buckets(ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Has(Petsc##HashT ht, KeyType key, PetscBool *has) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(has, 3)); \
    iter = kh_get(HashT, ht, key); \
    *has = (iter != kh_end(ht)) ? PETSC_TRUE : PETSC_FALSE; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Add(Petsc##HashT ht, KeyType key) \
  { \
    int                   ret; \
    PETSC_UNUSED khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    iter = kh_put(HashT, ht, key, &ret); \
    (void)iter; \
    PetscHashAssert(ret >= 0); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Del(Petsc##HashT ht, KeyType key) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    iter = kh_get(HashT, ht, key); \
    kh_del(HashT, ht, iter); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##QueryAdd(Petsc##HashT ht, KeyType key, PetscBool *missing) \
  { \
    int                   ret; \
    PETSC_UNUSED khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(missing, 3)); \
    iter = kh_put(HashT, ht, key, &ret); \
    (void)iter; \
    PetscHashAssert(ret >= 0); \
    *missing = ret ? PETSC_TRUE : PETSC_FALSE; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##QueryDel(Petsc##HashT ht, KeyType key, PetscBool *present) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(present, 3)); \
    iter = kh_get(HashT, ht, key); \
    if (iter != kh_end(ht)) { \
      kh_del(HashT, ht, iter); \
      *present = PETSC_TRUE; \
    } else { \
      *present = PETSC_FALSE; \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetElems(Petsc##HashT ht, PetscInt *off, KeyType array[]) \
  { \
    KeyType  key; \
    PetscInt pos; \
    PetscFunctionBegin; \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidPointer(ht, 1)); \
    PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidIntPointer(off, 2)); \
    pos = *off; \
    kh_foreach_key(ht, key, array[pos++] = key); \
    *off = pos; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

#endif /* PETSC_HASHSET_H */
