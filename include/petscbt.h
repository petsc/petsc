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

/*@C
  PetscBTLength - Returns the number of bytes needed to store a `PetscBT`

  Not Collective; No Fortran Support

  Input Parameter:
. m  - the number of bits in `array`

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`, `PetscBTLookup()`, `PetscBTLookupSet()`
@*/
static inline size_t PetscBTLength(PetscCount m)
{
  return (size_t)m / PETSC_BITS_PER_BYTE + 1;
}

/*@C
  PetscBTMemzero - Zero the contents of a `PetscBT` (bit array), setting every bit to `0`

  Not Collective; No Fortran Support

  Input Parameters:
+ m     - the number of bits the array can hold
- array - the `PetscBT` whose bits should be cleared

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTDestroy()`, `PetscBTCopy()`
@*/
static inline PetscErrorCode PetscBTMemzero(PetscCount m, PetscBT array)
{
  return PetscArrayzero(array, PetscBTLength(m));
}

/*@C
  PetscBTDestroy - Destroy a `PetscBT` (bit array) created with `PetscBTCreate()`, freeing its storage and setting the pointer to `NULL`

  Not Collective; No Fortran Support

  Input Parameter:
. array - pointer to the `PetscBT` to destroy; `*array` is set to `NULL` on return

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`
@*/
static inline PetscErrorCode PetscBTDestroy(PetscBT *array)
{
  return (*array) ? PetscFree(*array) : PETSC_SUCCESS;
}

/*@C
  PetscBTCreate - Create a `PetscBT` (bit array) capable of storing `m` bits, with all bits initialized to `0`

  Not Collective; No Fortran Support

  Input Parameter:
. m - the number of bits the array can hold

  Output Parameter:
. array - the newly created `PetscBT`

  Level: developer

.seealso: `PetscBT`, `PetscBTDestroy()`, `PetscBTMemzero()`, `PetscBTSet()`, `PetscBTLookup()`
@*/
static inline PetscErrorCode PetscBTCreate(PetscCount m, PetscBT *array)
{
  return PetscCalloc1(PetscBTLength(m), array);
}

/*@C
  PetscBTCopy - Copy the contents of one `PetscBT` (bit array) into another

  Not Collective; No Fortran Support

  Input Parameters:
+ dest   - the destination `PetscBT`, which must already be allocated to hold at least `m` bits
. m      - the number of bits to copy
- source - the source `PetscBT` providing the bits

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`
@*/
static inline PetscErrorCode PetscBTCopy(PetscBT dest, PetscCount m, PetscBT source)
{
  return PetscArraycpy(dest, source, PetscBTLength(m));
}

/*@C
  PetscBTLookup - Check if a particular bit in a `PetscBT` is set

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- index - the bit index to check

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`, `PetscBTLookupSet()`
@*/
static inline PetscByte PetscBTLookup(PetscBT array, PetscCount index)
{
  return array[PetscBTIndex_Internal(index)] & PetscBTMask_Internal(index);
}

/*@C
  PetscBTSet - Set the bit at a given index in a `PetscBT` (bit array) to `1`

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- index - the bit index to set

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTLookup()`, `PetscBTClear()`, `PetscBTNegate()`, `PetscBTLookupSet()`
@*/
static inline PetscErrorCode PetscBTSet(PetscBT array, PetscCount index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] |= PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBTNegate - Flip (xor) the bit at a given index in a `PetscBT` (bit array)

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- index - the bit index to flip

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTSet()`, `PetscBTClear()`, `PetscBTLookup()`
@*/
static inline PetscErrorCode PetscBTNegate(PetscBT array, PetscCount index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] ^= PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBTClear - Clear the bit at a given index in a `PetscBT` (bit array), setting it to `0`

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- index - the bit index to clear

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTSet()`, `PetscBTNegate()`, `PetscBTLookup()`
@*/
static inline PetscErrorCode PetscBTClear(PetscBT array, PetscCount index)
{
  PetscFunctionBegin;
  array[PetscBTIndex_Internal(index)] &= (PetscByte)~PetscBTMask_Internal(index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBTLookupSet - Check if a particular bit in a `PetscBT` is set and then set it

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- index - the bit index to check and set

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`, `PetscBTLookup()`, `PetscBTLookupClear()`
@*/
static inline PetscByte PetscBTLookupSet(PetscBT array, PetscCount index)
{
  const PetscByte ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTSet(array, index));
  return ret;
}

/*@C
  PetscBTLookupClear - Check if a particular bit in a `PetscBT` is set and then clear it

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- index - the bit index to check and clear

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`, `PetscBTLookup()`, `PetscBTLookupSet()`
@*/
static inline PetscByte PetscBTLookupClear(PetscBT array, PetscCount index)
{
  const PetscByte ret = PetscBTLookup(array, index);
  PetscCallContinue(PetscBTClear(array, index));
  return ret;
}

/*@C
  PetscBTCountSet - Count the number of bits that are set in a `PetscBT`

  Not Collective; No Fortran Support

  Input Parameters:
+ array - the `PetscBT`
- m     - the number of bits in `array`

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTMemzero()`, `PetscBTLookup()`, `PetscBTLookupSet()`
@*/
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
