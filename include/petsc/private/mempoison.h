#ifndef PETSC_MEMORY_POISON_H
#define PETSC_MEMORY_POISON_H

#include <petsc/private/petscimpl.h>

/* SUBMANSEC = Sys */

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
  #include <sanitizer/asan_interface.h> // ASAN_POISON/UNPOISON_MEMORY_REGION

  #define PETSC_HAVE_ASAN 1
#endif

#ifndef ASAN_POISON_MEMORY_REGION // use poison as canary
  #define ASAN_POISON_MEMORY_REGION(addr, size)   ((void)(addr), (void)(size))
  #define ASAN_UNPOISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#endif

#if !PetscDefined(HAVE_WINDOWS_COMPILERS) && !defined(__MINGW32__)
  #include <petsc/private/valgrind/memcheck.h> // VALGRIND_MAKE_MEM_*

  // defined in memcheck.h
  #if defined(PLAT_amd64_linux) || defined(PLAT_x86_linux) || defined(PLAT_amd64_darwin)
    #define PETSC_HAVE_VALGRIND_MEMPOISON 1
  #endif
#endif

#ifndef VALGRIND_MAKE_MEM_NOACCESS // use noaccess as canary
  #define VALGRIND_MAKE_MEM_NOACCESS(addr, size)  ((void)(addr), (void)(size))
  #define VALGRIND_MAKE_MEM_UNDEFINED(addr, size) ((void)(addr), (void)(size))
  #define VALGRIND_MAKE_MEM_DEFINED(addr, size)   ((void)(addr), (void)(size))
#endif

/*@C
  PetscPoisonMemoryRegion - Poison a region in memory, making it undereferencable

  Not Available From Fortran

  Input Parameters:
+ ptr  - The pointer to the start of the region
- size - The size (in bytes) of the region to poison

  Notes:
  `ptr` must not be `NULL`. It is OK to poison the same memory region repeatedly (it is a
  no-op).

  Any attempt to dereference the region after this routine returns results in an error being
  raised. The memory region may be un-poisoned using `PetscUnpoisonMemoryRegion()`, making it
  safe to dereference again.

  Example Usage:
.vb
  PetscInt *array;

  PetscMalloc1(15, &array);
  // OK, memory is normal
  array[0] = 10;
  array[1] = 15;

  PetscPoisonMemoryRegion(array, 15 * sizeof(*array));
  // ERROR this region is poisoned!
  array[0] = 10;
  // ERROR reading is not allowed either!
  PetscInt v = array[15];

  // OK can re-poison the region
  PetscPoisonMemoryRegion(array, 15 * sizeof(*array));
  // OK can re-poison any subregion too
  PetscPoisonMemoryRegion(array + 5, 1 * sizeof(*array));

  PetscUnpoisonMemoryRegion(array, 1 * sizeof(*array));
  // OK the first entry has been unpoisoned
  array[0] = 10;
  // ERROR the rest of the region is still poisoned!
  array[1] = 12345;

  PetscUnpoisonMemoryRegion(array + 10, sizeof(*array));
  // OK this region is unpoisoned (even though surrounding memory is still poisoned!)
  array[10] = 0;

  PetscInt stack_array[10];

  // OK can poison stack memory as well
  PetscPoisonMemoryRegion(stack_array, 10 * sizeof(*stack_array));
  // ERROR stack array is poisoned!
  stack_array[0] = 10;
.ve

  Level: developer

.seealso: `PetscUnpoisonMemoryRegion()`, `PetscIsRegionPoisoned()`
@*/
static inline PetscErrorCode PetscPoisonMemoryRegion(const void *ptr, size_t size)
{
  PetscFunctionBegin;
  // cannot check ptr as it may be poisoned
  // PetscValidPointer(ptr, 1);
  if (PetscDefined(HAVE_ASAN)) {
    ASAN_POISON_MEMORY_REGION(ptr, size);
  } else if (PetscDefined(HAVE_VALGRIND_MEMPOISON)) {
    (void)VALGRIND_MAKE_MEM_NOACCESS(ptr, size);
    (void)VALGRIND_MAKE_MEM_UNDEFINED(ptr, size);
  } else {
    (void)ptr;
    (void)size;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscUnpoisonMemoryRegion - Unpoison a previously poisoned memory region

  Input Parameters:
+ ptr  - The pointer to the start of the region
- size - The size (in bytes) of the region to unpoison

  Notes:
  Removes poisoning from a previously poisoned region. `ptr` may not be `NULL`. It is OK to
  unpoison an unpoisoned region.

  See `PetscPoisonMemoryRegion()` for example usage and further discussion.

  Level: developer

.seealso: `PetscPoisonMemoryRegion()`, `PetscIsRegionPoisoned()`
@*/
static inline PetscErrorCode PetscUnpoisonMemoryRegion(const void *ptr, size_t size)
{
  PetscFunctionBegin;
  // cannot check pointer as it is poisoned, duh!
  // PetscValidPointer(ptr, 1);
  if (PetscDefined(HAVE_ASAN)) {
    ASAN_UNPOISON_MEMORY_REGION(ptr, size);
  } else if (PetscDefined(HAVE_VALGRIND_MEMPOISON)) {
    (void)VALGRIND_MAKE_MEM_DEFINED(ptr, size);
  } else {
    (void)ptr;
    (void)size;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscIsRegionPoisoned - Query whether a particular memory region is poisoned

  Input Parameters:
+ ptr  - The pointer to the start of the region
- size - The size (in bytes) of the region to query

  Output Parameter:
. poisoned - Whether the region is known to be poisoned

  Notes:
  Sets `poisoned` to `PETSC_BOOL3_TRUE` if at least 1 byte in the range [`ptr`, `ptr + size`) is
  poisoned. Therefore a region must be entirely unpoisoned for `poisoned` to be `PETSC_BOOL3_FALSE`.

  If `ptr` is `NULL` or `size` is `0` then `poisoned` is set to `PETSC_BOOL3_FALSE`.

  If it is not possible to query the poisoned status of a region, then `poisoned` is set to
  `PETSC_BOOL3_UNKNOWN`.

  Level: developer

.seealso: `PetscPoisonMemoryRegion()`, `PetscUnpoisonMemoryRegion()`
@*/
static inline PetscErrorCode PetscIsRegionPoisoned(const void *ptr, size_t size, PetscBool3 *poisoned)
{
  PetscFunctionBegin;
  // cannot check pointer as may be poisoned
  // PetscValidPointer(ptr, 1);
  PetscValidBoolPointer(poisoned, 3);
  *poisoned = PETSC_BOOL3_FALSE;
  // if ptr is NULL, or if size = 0 then the "region" is not poisoned
  if (ptr && size) {
#if PetscDefined(HAVE_ASAN)
    if (__asan_region_is_poisoned((void *)ptr, size)) *poisoned = PETSC_BOOL3_TRUE;
#else
    // valgrind does not appear to have a way of querying the status without raising an error
    if (PetscDefined(HAVE_VALGRIND_MEMPOISON)) *poisoned = PETSC_BOOL3_UNKNOWN;
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif // PETSC_MEMORY_POISON_H
