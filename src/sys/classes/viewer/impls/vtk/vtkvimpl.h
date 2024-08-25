#pragma once

#include <petsc/private/viewerimpl.h> /*I   "petscsys.h"   I*/

typedef struct _n_PetscViewerVTKObjectLink *PetscViewerVTKObjectLink;
struct _n_PetscViewerVTKObjectLink {
  PetscViewerVTKFieldType  ft;
  PetscObject              vec;
  PetscViewerVTKObjectLink next;
  PetscInt                 field;
};

typedef struct {
  char                    *filename;
  PetscFileMode            btype;
  PetscObject              dm;
  PetscViewerVTKObjectLink link;
  PetscErrorCode (*write)(PetscObject, PetscViewer);
} PetscViewer_VTK;

PETSC_EXTERN PetscErrorCode PetscViewerVTKFWrite(PetscViewer, FILE *, const void *, PetscCount, MPI_Datatype);

#if defined(PETSC_HAVE_STDINT_H) /* The VTK format requires a 32-bit integer */
typedef int32_t PetscVTKInt;
#else /* Hope int is 32-bits */
typedef int PetscVTKInt;
#endif
typedef unsigned char PetscVTKType;

#define PETSC_VTK_INT_MAX 2147483647
#define PETSC_VTK_INT_MIN -2147483647

/*@C
  PetscVTKIntCast - casts to a `PetscVTKInt` (which may be 32-bits in size), generates an
  error if the `PetscVTKInt` is not large enough to hold the number.

  Not Collective; No Fortran Support

  Input Parameter:
. a - the  value to cast

  Output Parameter:
. b - the resulting `PetscVTKInt` value

  Level: advanced

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscIntCast()`
@*/
static inline PetscErrorCode PetscVTKIntCast(PetscCount a, PetscVTKInt *b)
{
  PetscFunctionBegin;
  *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(a <= PETSC_VTK_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscCount_FMT " is too big for VTK integer. Maximum supported value is %d", a, PETSC_MPI_INT_MAX);
  *b = (PetscVTKInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the only problem we've encountered so far is spaces not being acceptable for paraview field names */
static inline PetscErrorCode PetscViewerVTKSanitizeName_Internal(char name[], size_t maxlen)
{
  size_t c;

  PetscFunctionBegin;
  for (c = 0; c < maxlen; c++) {
    char a = name[c];
    if (a == '\0') break;
    if (a == ' ') name[c] = '_';
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
