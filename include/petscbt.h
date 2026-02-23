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

static inline PetscByte PetscBTMask_Internal(PetscCount index)
{
  return (PetscByte)(1 << index % PETSC_BITS_PER_BYTE);
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

static inline PetscByte PetscBTLookup(PetscBT array, PetscCount index)
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
  array[PetscBTIndex_Internal(index)] &= (PetscByte)~PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscByte PetscBTLookupSet(PetscBT array, PetscCount index)
{
  const PetscByte ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTSet(array, index));
  return ret;
}

static inline PetscByte PetscBTLookupClear(PetscBT array, PetscCount index)
{
  const PetscByte ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTClear(array, index));
  return ret;
}

static inline PetscCount PetscBTCountSet(PetscBT array, PetscCount m)
{
  PetscCount cnt = 0;
  for (size_t j = 0; j < PetscBTLength(m); j++) {
    PetscByte       byte = array[j];
    const PetscByte c1   = 0x55;
    const PetscByte c2   = 0x33;
    const PetscByte c4   = 0x0F;

    byte -= (byte >> 1) & c1;
    byte = ((byte >> 2) & c2) + (byte & c2);
    cnt += (byte + (byte >> 4)) & c4;
  }
  return cnt;
}

PETSC_EXTERN PetscErrorCode PetscBTView(PetscCount, const PetscBT, PetscViewer);
