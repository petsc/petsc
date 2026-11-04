#pragma once

/* SUBMANSEC = Sys */

#include <petscsystypes.h>
#include <petscmacros.h>

#include <stdio.h> // FILE used by PetscMallocDump()/PetscMallocView()

/*MC
   PetscMalloc - Allocates memory for use with PETSc. One should use `PetscNew()`, `PetscMalloc1()` or `PetscCalloc1()` usually instead of `PetscMalloc()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc(size_t m, void **result)

   Not Collective

   Input Parameter:
.  m - number of bytes to allocate

   Output Parameter:
.  result - memory allocated

   Level: beginner

   Notes:
   Memory is always allocated at least double aligned

   It is safe to allocate with an m of 0 and pass the resulting pointer to `PetscFree()`.
   However, the pointer should never be dereferenced or the program will crash.

   Developer Note:
   All the `PetscMallocN()` routines actually call `PetscMalloc()` behind the scenes.

   Except for data structures that store information about the PETSc options database all memory allocated by PETSc is
   obtained with `PetscMalloc()` or `PetscCalloc()`

.seealso: `PetscFree()`, `PetscNew()`, `PetscCalloc()`
M*/
#define PetscMalloc(a, b) ((*PetscTrMalloc)((a), PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void **)(b)))

/*MC
   PetscRealloc - Reallocates memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscRealloc(size_t m, void **result)

   Not Collective

   Input Parameters:
+  m      - number of bytes to allocate
-  result - previous memory

   Output Parameter:
.  result - new memory allocated

   Level: developer

   Notes:
   `results` must have already been obtained with `PetscMalloc()`

   Memory is always allocated at least double aligned

.seealso: `PetscMalloc()`, `PetscFree()`, `PetscNew()`
M*/
#define PetscRealloc(a, b) ((*PetscTrRealloc)((a), __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void **)(b)))

/*MC
   PetscAddrAlign - Rounds up an address to `PETSC_MEMALIGN` alignment

   Synopsis:
    #include <petscsys.h>
   void *PetscAddrAlign(void *addr)

   Not Collective

   Input Parameter:
.  addr - address to align (any pointer type)

   Level: developer

.seealso: `PetscMallocAlign()`
M*/
#define PetscAddrAlign(a) ((void *)((((PETSC_UINTPTR_T)(a)) + (PETSC_MEMALIGN - 1)) & ~(PETSC_MEMALIGN - 1)))

/*MC
   PetscCalloc - Allocates a cleared (zeroed) memory region aligned to `PETSC_MEMALIGN`, similar to `PetscMalloc()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc(size_t m, void **result)

   Not Collective

   Input Parameter:
.  m - number of bytes to allocate

   Output Parameter:
.  result - memory allocated

   Level: beginner

   Notes:
   Memory is always allocated at least double aligned. This macro is useful in allocating memory pointed by void pointers

   It is safe to allocate with an m of 0 and pass the resulting pointer to `PetscFree()`.

   However, the pointer should never be dereferenced or the program will crash.

   Developer Note:
   All `PetscCallocN()` routines call `PetscCalloc()` behind the scenes.

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`
M*/
#define PetscCalloc(m, result) PetscMallocA(1, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)m), (void *)(result))

/*MC
   PetscMalloc1 - Allocates an array of memory aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc1(size_t m1, type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate  (may be zero)

   Output Parameter:
.  r1 - memory allocated

   Level: beginner

   Note:
   This uses `sizeof()` of the memory type requested to determine the total memory to be allocated; therefore, you should not
   multiply the number of elements requested by the `sizeof()` the type. For example, use
.vb
  PetscInt *id;
  PetscMalloc1(10,&id);
.ve
       not
.vb
  PetscInt *id;
  PetscMalloc1(10*sizeof(PetscInt),&id);
.ve

  Does not zero the memory allocated, use `PetscCalloc1()` to obtain memory that has been zeroed.

  The `PetscMalloc[N]()` and `PetscCalloc[N]()` take an argument of type `size_t`! However, most codes use `value`, computed via `int` or `PetscInt` variables. This can overflow in
  32bit `int` computation - while computation in 64bit `size_t` would not overflow!
  It's best if any arithmetic that is done for size computations is done with `size_t` type - avoiding arithmetic overflow!

  `PetscMalloc[N]()` and `PetscCalloc[N]()` attempt to work-around this by casting the first variable to `size_t`.
  This works for most expressions, but not all, such as
.vb
  PetscInt *id, a, b;
  PetscMalloc1(use_a_squared ? a * a * b : a * b, &id); // use_a_squared is cast to size_t, but a and b are still PetscInt
  PetscMalloc1(a + b * b, &id); // a is cast to size_t, but b * b is performed at PetscInt precision first due to order-of-operations
.ve

  These expressions should either be avoided, or appropriately cast variables to `size_t`:
.vb
  PetscInt *id, a, b;
  PetscMalloc1(use_a_squared ? (size_t)a * a * b : (size_t)a * b, &id); // Cast a to size_t before multiplication
  PetscMalloc1(b * b + a, &id); // b is automatically cast to size_t and order-of-operations ensures size_t precision is maintained
.ve

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscCalloc1()`, `PetscMalloc2()`
M*/
#define PetscMalloc1(m1, r1) PetscMallocA(1, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1))

/*MC
   PetscCalloc1 - Allocates a cleared (zeroed) array of memory aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc1(size_t m1, type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate in 1st chunk  (may be zero)

   Output Parameter:
.  r1 - memory allocated

   Level: beginner

   Note:
   See `PetscMalloc1()` for more details on usage.

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc2()`
M*/
#define PetscCalloc1(m1, r1) PetscMallocA(1, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1))

/*MC
   PetscMalloc2 - Allocates 2 arrays of memory both aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc2(size_t m1, type **r1, size_t m2, type **r2)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
-  m2 - number of elements to allocate in 2nd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc2()`
M*/
#define PetscMalloc2(m1, r1, m2, r2) PetscMallocA(2, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2))

/*MC
   PetscCalloc2 - Allocates 2 cleared (zeroed) arrays of memory both aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc2(size_t m1, type **r1, size_t m2, type **r2)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
-  m2 - number of elements to allocate in 2nd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscCalloc1()`, `PetscMalloc2()`
M*/
#define PetscCalloc2(m1, r1, m2, r2) PetscMallocA(2, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2))

/*MC
   PetscMalloc3 - Allocates 3 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc3(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  m3 - number of elements to allocate in 3rd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc3()`, `PetscFree3()`
M*/
#define PetscMalloc3(m1, r1, m2, r2, m3, r3) \
  PetscMallocA(3, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3))

/*MC
   PetscCalloc3 - Allocates 3 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc3(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  m3 - number of elements to allocate in 3rd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscCalloc2()`, `PetscMalloc3()`, `PetscFree3()`
M*/
#define PetscCalloc3(m1, r1, m2, r2, m3, r3) \
  PetscMallocA(3, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3))

/*MC
   PetscMalloc4 - Allocates 4 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc4(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
-  m4 - number of elements to allocate in 4th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
-  r4 - memory allocated in fourth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc4()`, `PetscFree4()`
M*/
#define PetscMalloc4(m1, r1, m2, r2, m3, r3, m4, r4) \
  PetscMallocA(4, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4))

/*MC
   PetscCalloc4 - Allocates 4 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc4(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
-  m4 - number of elements to allocate in 4th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
-  r4 - memory allocated in fourth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc4()`, `PetscFree4()`
M*/
#define PetscCalloc4(m1, r1, m2, r2, m3, r3, m4, r4) \
  PetscMallocA(4, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4))

/*MC
   PetscMalloc5 - Allocates 5 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc5(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4, size_t m5, type **r5)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  m5 - number of elements to allocate in 5th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc5()`, `PetscFree5()`
M*/
#define PetscMalloc5(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5) \
  PetscMallocA(5, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5))

/*MC
   PetscCalloc5 - Allocates 5 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc5(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4, size_t m5, type **r5)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  m5 - number of elements to allocate in 5th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc5()`, `PetscFree5()`
M*/
#define PetscCalloc5(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5) \
  PetscMallocA(5, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5))

/*MC
   PetscMalloc6 - Allocates 6 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc6(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4, size_t m5, type **r5, size_t m6, type **r6)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  m6 - number of elements to allocate in 6th chunk  (may be zero)

   Output Parameteasr:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc6()`, `PetscFree3()`, `PetscFree4()`, `PetscFree5()`, `PetscFree6()`
M*/
#define PetscMalloc6(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6) \
  PetscMallocA(6, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6))

/*MC
   PetscCalloc6 - Allocates 6 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc6(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4, size_t m5, type **r5, size_t m6, type **r6)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  m6 - number of elements to allocate in 6th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscMalloc6()`, `PetscFree6()`
M*/
#define PetscCalloc6(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6) \
  PetscMallocA(6, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6))

/*MC
   PetscMalloc7 - Allocates 7 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc7(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4, size_t m5, type **r5, size_t m6, type **r6, size_t m7, type **r7)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  m7 - number of elements to allocate in 7th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc7()`, `PetscFree7()`
M*/
#define PetscMalloc7(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6, m7, r7) \
  PetscMallocA(7, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6), ((size_t)((size_t)m7) * sizeof(**(r7))), (r7))

/*MC
   PetscCalloc7 - Allocates 7 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc7(size_t m1, type **r1, size_t m2, type **r2, size_t m3, type **r3, size_t m4, type **r4, size_t m5, type **r5, size_t m6, type **r6, size_t m7, type **r7)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  m7 - number of elements to allocate in 7th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscMalloc7()`, `PetscFree7()`
M*/
#define PetscCalloc7(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6, m7, r7) \
  PetscMallocA(7, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (void *)(r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6), ((size_t)((size_t)m7) * sizeof(**(r7))), (r7))

/*MC
   PetscNew - Allocates memory of a particular type, zeros the memory! Aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscNew(type **result)

   Not Collective

   Output Parameter:
.  result - memory allocated, sized to match pointer `type`

   Level: beginner

   Developer Note:
   Calls `PetscCalloc()` with the appropriate memory size obtained from `type`

.seealso: `PetscFree()`, `PetscMalloc()`, `PetscCall()`, `PetscCalloc1()`, `PetscMalloc1()`
M*/
#define PetscNew(b) PetscCalloc1(1, (b))

/*MC
   PetscNewLog - Deprecated alias for `PetscNew()`; previously logged the allocation against a `PetscObject`

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode PetscNewLog(PetscObject obj, type **result)

   Not Collective

   Input Parameter:
.  obj - the `PetscObject` to which the allocation used to be charged (now ignored)

   Output Parameter:
.  result - the newly allocated memory, sized to match `type`

   Level: deprecated

   Note:
   This macro is retained only for source compatibility. Use `PetscNew()` instead; PETSc no longer associates allocations with a specific `PetscObject`.

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscCalloc1()`, `PetscFree()`
M*/
#define PetscNewLog(o, b) PETSC_DEPRECATED_MACRO(3, 18, 0, "PetscNew()", ) PetscNew(b)

/*MC
   PetscFree - Frees memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free (the pointer is ALWAYS set to `NULL` upon success)

   Level: beginner

   Notes:
   Do not free memory obtained with `PetscMalloc2()`, `PetscCalloc2()` etc, they must be freed with `PetscFree2()` etc.

   It is safe to call `PetscFree()` on a `NULL` pointer.

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc1()`
M*/
#define PetscFree(a) ((PetscErrorCode)((*PetscTrFree)((void *)(a), __LINE__, PETSC_FUNCTION_NAME, __FILE__) || ((a) = PETSC_NULLPTR, PETSC_SUCCESS)))

/*MC
   PetscFree2 - Frees 2 chunks of memory obtained with `PetscMalloc2()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree2(void *memory1, void *memory2)

   Not Collective

   Input Parameters:
+   memory1 - memory to free
-   memory2 - 2nd memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc2()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc2()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`
M*/
#define PetscFree2(m1, m2) PetscFreeA(2, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void *)&(m1), &(m2))

/*MC
   PetscFree3 - Frees 3 chunks of memory obtained with `PetscMalloc3()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree3(void *memory1, void *memory2, void *memory3)

   Not Collective

   Input Parameters:
+   memory1 - memory to free
.   memory2 - 2nd memory to free
-   memory3 - 3rd memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc3()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc3()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`
M*/
#define PetscFree3(m1, m2, m3) PetscFreeA(3, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void *)&(m1), &(m2), &(m3))

/*MC
   PetscFree4 - Frees 4 chunks of memory obtained with `PetscMalloc4()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree4(void *m1, void *m2, void *m3, void *m4)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
-   m4 - 4th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc4()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc4()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`
M*/
#define PetscFree4(m1, m2, m3, m4) PetscFreeA(4, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void *)&(m1), &(m2), &(m3), &(m4))

/*MC
   PetscFree5 - Frees 5 chunks of memory obtained with `PetscMalloc5()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree5(void *m1, void *m2, void *m3, void *m4, void *m5)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
-   m5 - 5th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc5()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc5()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`, `PetscMalloc5()`
M*/
#define PetscFree5(m1, m2, m3, m4, m5) PetscFreeA(5, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void *)&(m1), &(m2), &(m3), &(m4), &(m5))

/*MC
   PetscFree6 - Frees 6 chunks of memory obtained with `PetscMalloc6()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree6(void *m1, void *m2, void *m3, void *m4, void *m5, void *m6)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
.   m5 - 5th memory to free
-   m6 - 6th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc6()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc6()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`, `PetscMalloc5()`, `PetscMalloc6()`
M*/
#define PetscFree6(m1, m2, m3, m4, m5, m6) PetscFreeA(6, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void *)&(m1), &(m2), &(m3), &(m4), &(m5), &(m6))

/*MC
   PetscFree7 - Frees 7 chunks of memory obtained with `PetscMalloc7()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree7(void *m1, void *m2, void *m3, void *m4, void *m5, void *m6, void *m7)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
.   m5 - 5th memory to free
.   m6 - 6th memory to free
-   m7 - 7th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc7()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc7()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`, `PetscMalloc5()`, `PetscMalloc6()`,
          `PetscMalloc7()`
M*/
#define PetscFree7(m1, m2, m3, m4, m5, m6, m7) PetscFreeA(7, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void *)&(m1), &(m2), &(m3), &(m4), &(m5), &(m6), &(m7))

PETSC_EXTERN PetscErrorCode PetscMallocA(int, PetscBool, int, const char *, const char *, size_t, void *, ...);
PETSC_EXTERN PetscErrorCode PetscFreeA(int, int, const char *, const char *, void *, ...);
PETSC_EXTERN PetscErrorCode (*PetscTrMalloc)(size_t, PetscBool, int, const char[], const char[], void **);
PETSC_EXTERN PetscErrorCode (*PetscTrFree)(void *, int, const char[], const char[]);
PETSC_EXTERN PetscErrorCode (*PetscTrRealloc)(size_t, int, const char[], const char[], void **);
PETSC_EXTERN PetscErrorCode PetscMallocSetCoalesce(PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocSet(PetscErrorCode (*)(size_t, PetscBool, int, const char[], const char[], void **), PetscErrorCode (*)(void *, int, const char[], const char[]), PetscErrorCode (*)(size_t, int, const char[], const char[], void **));
PETSC_EXTERN PetscErrorCode PetscMallocClear(void);

/*
  Unlike PetscMallocSet and PetscMallocClear which overwrite the existing settings, these two functions save the previous choice of allocator, and should be used in pair.
*/
PETSC_EXTERN PetscErrorCode PetscMallocSetDRAM(void);
PETSC_EXTERN PetscErrorCode PetscMallocResetDRAM(void);
#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscMallocSetCUDAHost(void);
PETSC_EXTERN PetscErrorCode PetscMallocResetCUDAHost(void);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscMallocSetHIPHost(void);
PETSC_EXTERN PetscErrorCode PetscMallocResetHIPHost(void);
#endif

/*
   Routines for tracing memory corruption/bleeding with default PETSc memory allocation
*/
PETSC_EXTERN PetscErrorCode PetscMallocDump(FILE *);
PETSC_EXTERN PetscErrorCode PetscMallocView(FILE *);
PETSC_EXTERN PetscErrorCode PetscMallocGetCurrentUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocGetMaximumUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocPushMaximumUsage(int);
PETSC_EXTERN PetscErrorCode PetscMallocPopMaximumUsage(int, PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocSetDebug(PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocGetDebug(PetscBool *, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscMallocValidate(int, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscMallocViewSet(PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscMallocViewGet(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscMallocLogRequestedSizeSet(PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocLogRequestedSizeGet(PetscBool *);
