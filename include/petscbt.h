#pragma once

#include <petscsystypes.h>
#include <petscviewertypes.h>
#include <petscstring.h>

/* SUBMANSEC = Sys */

/* convert an index i to an index suitable for indexing a PetscBT, such that
 * bt[PetscBTIndex(i)] returns the i'th value of the bt */
static inline size_t PetscBTIndex_Internal(PetscInt index)
{
  return (size_t)index / PETSC_BITS_PER_BYTE;
}

static inline char PetscBTMask_Internal(PetscInt index)
{
  return (char)(1 << index % PETSC_BITS_PER_BYTE);
}

static inline size_t PetscBTLength(PetscInt m)
{
  return (size_t)m / PETSC_BITS_PER_BYTE + 1;
}

static inline PetscErrorCode PetscBTMemzero(PetscInt m, PetscBT array)
{
  return PetscArrayzero(array, PetscBTLength(m));
}

static inline PetscErrorCode PetscBTDestroy(PetscBT *array)
{
  return (*array) ? PetscFree(*array) : PETSC_SUCCESS;
}

static inline PetscErrorCode PetscBTCreate(PetscInt m, PetscBT *array)
{
  return PetscCalloc1(PetscBTLength(m), array);
}

static inline char PetscBTLookup(PetscBT array, PetscInt index)
{
  return array[PetscBTIndex_Internal(index)] & PetscBTMask_Internal(index);
}

static inline PetscErrorCode PetscBTSet(PetscBT array, PetscInt index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] |= PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscBTNegate(PetscBT array, PetscInt index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] ^= PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscBTClear(PetscBT array, PetscInt index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] &= (char)~PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline char PetscBTLookupSet(PetscBT array, PetscInt index)
{
  const char ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTSet(array, index));
  return ret;
}

static inline char PetscBTLookupClear(PetscBT array, PetscInt index)
{
  const char ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTClear(array, index));
  return ret;
}

PETSC_EXTERN PetscErrorCode PetscBTView(PetscInt, const PetscBT, PetscViewer);
