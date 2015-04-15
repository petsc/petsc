#if !defined(_VTKVIMPL_H)
#define _VTKVIMPL_H

#include <petsc/private/viewerimpl.h>    /*I   "petscsys.h"   I*/

typedef struct _n_PetscViewerVTKObjectLink *PetscViewerVTKObjectLink;
struct _n_PetscViewerVTKObjectLink {
  PetscViewerVTKFieldType  ft;
  PetscObject              vec;
  PetscViewerVTKObjectLink next;
};

typedef struct {
  char                     *filename;
  PetscFileMode            btype;
  PetscBool                written;
  PetscObject              dm;
  PetscViewerVTKObjectLink link;
  PetscErrorCode (*write)(PetscObject,PetscViewer);
} PetscViewer_VTK;

PETSC_EXTERN PetscErrorCode PetscViewerVTKFWrite(PetscViewer,FILE*,const void*,PetscInt,PetscDataType);

#if defined(PETSC_HAVE_STDINT_H) /* The VTK format requires a 32-bit integer */
typedef int32_t PetscVTKInt;
#else                            /* Hope int is 32 bits */
typedef int PetscVTKInt;
#endif

/* Hack: Our enums are guaranteed to be the same size as normal int. A better fix would be to distinguish PETSC_INT32
 * and PETSC_INT64 in the PetscDataType enum. */
#define PETSC_INT32 PETSC_ENUM

typedef unsigned char PetscVTKType;
#define PETSC_UINT8 PETSC_CHAR

#define PETSC_VTK_INT_MAX  2147483647
#define PETSC_VTK_INT_MIN -2147483647
#if defined(PETSC_USE_64BIT_INDICES)
#  define PetscVTKIntCheck(a)  if ((a) > PETSC_VTK_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for 32-bit VTK binary format")
#  define PetscVTKIntCast(a) (PetscVTKInt)(a);PetscVTKIntCheck(a)
#else
#  define PetscVTKIntCheck(a)
#  define PetscVTKIntCast(a) a
#endif

#endif
