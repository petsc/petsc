#pragma once

#include <petscsystypes.h>
#include <petscviewertypes.h>
#include <petscstring.h>

/* SUBMANSEC = Sys */

/* convert an index i to an index suitable for indexing a PetscBT, such that
 * bt[PetscBTIndex(i)] returns the i'th value of the bt */
static inline size_t PetscBTIndex_Internal(PetscCount index)
{
  return (size_t)index / PETSC_BITS_PER_BYTE;
}

static inline char PetscBTMask_Internal(PetscCount index)
{
  return (char)(1 << index % PETSC_BITS_PER_BYTE);
}

static inline size_t PetscBTLength(PetscCount m)
{
  return (size_t)m / PETSC_BITS_PER_BYTE + 1;
}

static inline PetscErrorCode PetscBTMemzero(PetscCount m, PetscBT array)
{
  return PetscArrayzero(array, PetscBTLength(m));
}

static inline PetscErrorCode PetscBTDestroy(PetscBT *array)
{
  return (*array) ? PetscFree(*array) : PETSC_SUCCESS;
}

static inline PetscErrorCode PetscBTCreate(PetscCount m, PetscBT *array)
{
  return PetscCalloc1(PetscBTLength(m), array);
}

static inline PetscErrorCode PetscBTCopy(PetscBT dest, PetscCount m, PetscBT source)
{
  return PetscArraycpy(dest, source, PetscBTLength(m));
}

static inline char PetscBTLookup(PetscBT array, PetscCount index)
{
  return array[PetscBTIndex_Internal(index)] & PetscBTMask_Internal(index);
}

static inline PetscErrorCode PetscBTSet(PetscBT array, PetscCount index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] |= PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscBTNegate(PetscBT array, PetscCount index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] ^= PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscBTClear(PetscBT array, PetscCount index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] &= (char)~PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline char PetscBTLookupSet(PetscBT array, PetscCount index)
{
  const char ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTSet(array, index));
  return ret;
}

static inline char PetscBTLookupClear(PetscBT array, PetscCount index)
{
  const char ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTClear(array, index));
  return ret;
}

static inline PetscCount PetscBTCountSet(PetscBT array, PetscCount m)
{
  PetscCount cnt = 0;
  for (size_t j = 0; j < PetscBTLength(m); j++) {
    unsigned char       byte = array[j];
    const unsigned char c1   = 0x55;
    const unsigned char c2   = 0x33;
    const unsigned char c4   = 0x0F;

    byte -= (byte >> 1) & c1;
    byte = ((byte >> 2) & c2) + (byte & c2);
    cnt += (byte + (byte >> 4)) & c4;
  }
  return cnt;
}

PETSC_EXTERN PetscErrorCode PetscBTView(PetscCount, const PetscBT, PetscViewer);
