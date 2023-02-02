#ifndef PETSC_HASHMAP_H
#define PETSC_HASHMAP_H

#include <petsc/private/hashtable.h>

/* SUBMANSEC = PetscH */

/*MC
  PETSC_HASH_MAP - Instantiate a PETSc hash table map type

  Synopsis:
  #include <petsc/private/hashmap.h>
  PETSC_HASH_MAP(HMapT, KeyType, ValType, HashFunc, EqualFunc, DefaultValue)

  Input Parameters:
+ HMapT - The hash table map type name suffix
. KeyType - The type of keys
. ValType - The type of values
. HashFunc - Routine or function-like macro computing hash values from keys
. EqualFunc - Routine or function-like macro computing whether two values are equal
- DefaultValue - Default value to use for queries in case of missing keys

  Level: developer

  Developer Note:
    Each time this macro is used to create a new hash map type, the make rule for allmanpages in $PETSC_DIR/makefile should
    be updated to cause the automatic generation of appropriate manual pages for that type. The manual pages
    are generated from the templated version of the documentation in include/petsc/private/hashmap.txt.

  References:
    This code uses the standalone and portable C language khash software https://github.com/attractivechaos/klib

.seealso: `PETSC_HASH_MAP_DECL()`, `PetscHMapI`, `PetscHMapICreate()`, `PetscHMapIJ`,
`PetscHMapIJCreate()`, `PETSC_HASH_SET()`
M*/

#define PETSC_HASH_MAP_DECL(HashT, KeyType, ValType) \
  typedef kh_##HashT##_t                   *Petsc##HashT; \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Create(Petsc##HashT *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##CreateWithSize(PetscInt, Petsc##HashT *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Destroy(Petsc##HashT *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Reset(Petsc##HashT); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Duplicate(Petsc##HashT, Petsc##HashT *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Clear(Petsc##HashT); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Resize(Petsc##HashT, PetscInt); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetSize(Petsc##HashT, PetscInt *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetCapacity(Petsc##HashT, PetscInt *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Has(Petsc##HashT, KeyType, PetscBool *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Get(Petsc##HashT, KeyType, ValType *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetWithDefault(Petsc##HashT, KeyType, ValType, ValType *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Set(Petsc##HashT, KeyType, ValType); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Del(Petsc##HashT, KeyType); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##QuerySet(Petsc##HashT, KeyType, ValType, PetscBool *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##QueryDel(Petsc##HashT, KeyType, PetscBool *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Find(Petsc##HashT, KeyType, PetscHashIter *, PetscBool *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Put(Petsc##HashT, KeyType, PetscHashIter *, PetscBool *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##IterGet(Petsc##HashT, PetscHashIter, ValType *); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##IterSet(Petsc##HashT, PetscHashIter, ValType); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##IterDel(Petsc##HashT, PetscHashIter); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetKeys(Petsc##HashT, PetscInt *, KeyType[]); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetVals(Petsc##HashT, PetscInt *, ValType[]); \
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetPairs(Petsc##HashT, PetscInt *, KeyType[], ValType[])

#define PETSC_HASH_MAP(HashT, KeyType, ValType, HashFunc, EqualFunc, DefaultValue) \
\
  KHASH_INIT(HashT, KeyType, ValType, 1, HashFunc, EqualFunc) \
  PETSC_HASH_MAP_DECL(HashT, KeyType, ValType); \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Create(Petsc##HashT *ht) \
  { \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    *ht = kh_init(HashT); \
    PetscHashAssert(*ht != NULL); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##CreateWithSize(PetscInt n, Petsc##HashT *ht) \
  { \
    PetscFunctionBegin; \
    PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Hash table size %" PetscInt_FMT " must be >= 0", n); \
    PetscValidPointer(ht, 2); \
    PetscCall(Petsc##HashT##Create(ht)); \
    if (n) PetscCall(Petsc##HashT##Resize(*ht, n)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Destroy(Petsc##HashT *ht) \
  { \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    if (!*ht) PetscFunctionReturn(PETSC_SUCCESS); \
    kh_destroy(HashT, *ht); \
    *ht = NULL; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Reset(Petsc##HashT ht) \
  { \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    kh_reset(HashT, ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Duplicate(Petsc##HashT ht, Petsc##HashT *hd) \
  { \
    int     ret; \
    KeyType key; \
    ValType val; \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(hd, 2); \
    *hd = kh_init(HashT); \
    PetscHashAssert(*hd != NULL); \
    ret = kh_resize(HashT, *hd, kh_size(ht)); \
    PetscHashAssert(ret == 0); \
    kh_foreach(ht, key, val, { \
      khiter_t i; \
      i = kh_put(HashT, *hd, key, &ret); \
      PetscHashAssert(ret >= 0); \
      kh_val(*hd, i) = val; \
    }) PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Clear(Petsc##HashT ht) \
  { \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    kh_clear(HashT, ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Resize(Petsc##HashT ht, PetscInt nb) \
  { \
    int ret; \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    ret = kh_resize(HashT, ht, (khint_t)nb); \
    PetscHashAssert(ret >= 0); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetSize(Petsc##HashT ht, PetscInt *n) \
  { \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    PetscValidIntPointer(n, 2); \
    *n = (PetscInt)kh_size(ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetCapacity(Petsc##HashT ht, PetscInt *n) \
  { \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    PetscValidIntPointer(n, 2); \
    *n = (PetscInt)kh_n_buckets(ht); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Has(Petsc##HashT ht, KeyType key, PetscBool *has) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(has, 3); \
    iter = kh_get(HashT, ht, key); \
    *has = (iter != kh_end(ht)) ? PETSC_TRUE : PETSC_FALSE; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Get(Petsc##HashT ht, KeyType key, ValType *val) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidIntPointer(val, 3); \
    iter = kh_get(HashT, ht, key); \
    *val = (iter != kh_end(ht)) ? kh_val(ht, iter) : (DefaultValue); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetWithDefault(Petsc##HashT ht, KeyType key, ValType default_val, ValType *val) \
  { \
    PetscHashIter it    = 0; \
    PetscBool     found = PETSC_FALSE; \
\
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(val, 4); \
    PetscCall(Petsc##HashT##Find(ht, key, &it, &found)); \
    if (found) { \
      PetscHashIterGetVal(ht, it, *val); \
    } else { \
      *val = default_val; \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Set(Petsc##HashT ht, KeyType key, ValType val) \
  { \
    int      ret; \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    iter = kh_put(HashT, ht, key, &ret); \
    PetscHashAssert(ret >= 0); \
    kh_val(ht, iter) = val; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Del(Petsc##HashT ht, KeyType key) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    iter = kh_get(HashT, ht, key); \
    kh_del(HashT, ht, iter); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##QuerySet(Petsc##HashT ht, KeyType key, ValType val, PetscBool *missing) \
  { \
    int      ret; \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(missing, 3); \
    iter = kh_put(HashT, ht, key, &ret); \
    PetscHashAssert(ret >= 0); \
    kh_val(ht, iter) = val; \
    *missing         = ret ? PETSC_TRUE : PETSC_FALSE; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##QueryDel(Petsc##HashT ht, KeyType key, PetscBool *present) \
  { \
    khiter_t iter; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(present, 3); \
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
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Find(Petsc##HashT ht, KeyType key, PetscHashIter *iter, PetscBool *found) \
\
  { \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(iter, 2); \
    PetscValidPointer(found, 3); \
    *iter  = kh_get(HashT, ht, key); \
    *found = (*iter != kh_end(ht)) ? PETSC_TRUE : PETSC_FALSE; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##Put(Petsc##HashT ht, KeyType key, PetscHashIter *iter, PetscBool *missing) \
  { \
    int ret; \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(iter, 2); \
    PetscValidPointer(missing, 3); \
    *iter = kh_put(HashT, ht, key, &ret); \
    PetscHashAssert(ret >= 0); \
    *missing = ret ? PETSC_TRUE : PETSC_FALSE; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##IterGet(Petsc##HashT ht, PetscHashIter iter, ValType *val) \
  { \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscValidPointer(val, 3); \
    *val = PetscLikely(iter < kh_end(ht) && kh_exist(ht, iter)) ? kh_val(ht, iter) : (DefaultValue); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##IterSet(Petsc##HashT ht, PetscHashIter iter, ValType val) \
  { \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    if (PetscLikely(iter < kh_end(ht) && kh_exist(ht, iter))) kh_val(ht, iter) = val; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##IterDel(Petsc##HashT ht, PetscHashIter iter) \
  { \
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    if (PetscLikely(iter < kh_end(ht))) kh_del(HashT, ht, iter); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetKeys(Petsc##HashT ht, PetscInt *off, KeyType array[]) \
  { \
    KeyType  key; \
    PetscInt pos; \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    PetscValidIntPointer(off, 2); \
    pos = *off; \
    kh_foreach_key(ht, key, array[pos++] = key); \
    *off = pos; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetVals(Petsc##HashT ht, PetscInt *off, ValType array[]) \
  { \
    ValType  val; \
    PetscInt pos; \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    PetscValidIntPointer(off, 2); \
    pos = *off; \
    kh_foreach_value(ht, val, array[pos++] = val); \
    *off = pos; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##GetPairs(Petsc##HashT ht, PetscInt *off, KeyType karray[], ValType varray[]) \
  { \
    ValType  val; \
    KeyType  key; \
    PetscInt pos; \
    PetscFunctionBegin; \
    PetscValidPointer(ht, 1); \
    PetscValidIntPointer(off, 2); \
    pos     = *off; \
    kh_foreach(ht, key, val, { \
      karray[pos]   = key; \
      varray[pos++] = val; \
    }) *off = pos; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

#define PETSC_HASH_MAP_EXTENDED_DECL(HashT, KeyType, ValType) static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##SetWithMode(Petsc##HashT, KeyType, ValType, InsertMode)

#define PETSC_HASH_MAP_EXTENDED(HashT, KeyType, ValType, HashFunc, EqualFunc, DefaultValue) \
  PETSC_HASH_MAP(HashT, KeyType, ValType, HashFunc, EqualFunc, DefaultValue) \
\
  PETSC_HASH_MAP_EXTENDED_DECL(HashT, KeyType, ValType); \
\
  static inline PETSC_UNUSED PetscErrorCode Petsc##HashT##SetWithMode(Petsc##HashT ht, KeyType key, ValType val, InsertMode mode) \
  { \
    PetscHashIter it      = 0; \
    PetscBool     missing = PETSC_FALSE; \
\
    PetscFunctionBeginHot; \
    PetscValidPointer(ht, 1); \
    PetscCall(Petsc##HashT##Put(ht, key, &it, &missing)); \
    if (!missing) { \
      ValType cur_val; \
\
      PetscHashIterGetVal(ht, it, cur_val); \
      switch (mode) { \
      case INSERT_VALUES: \
        break; \
      case ADD_VALUES: \
        val += cur_val; \
        break; \
      case MAX_VALUES: \
        val = PetscMax(cur_val, val); \
        break; \
      case MIN_VALUES: \
        val = PetscMin(cur_val, val); \
        break; \
      case NOT_SET_VALUES:    /* fallthrough */ \
      case INSERT_ALL_VALUES: /* fallthrough */ \
      case ADD_ALL_VALUES:    /* fallthrough */ \
      case INSERT_BC_VALUES:  /* fallthrough */ \
      case ADD_BC_VALUES:     /* fallthrough */ \
      default: \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported InsertMode %d", (int)mode); \
      } \
    } \
    PetscCall(Petsc##HashT##IterSet(ht, it, val)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

#endif /* PETSC_HASHMAP_H */
