#include <petscis.h> /*I  "petscis.h"  I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/viewerimpl.h>
#include <petsclayouthdf5.h>

PetscErrorCode ISView_Binary(IS is, PetscViewer viewer)
{
  PetscBool       skipHeader;
  PetscLayout     map;
  PetscInt        tr[2], n, s, N;
  const PetscInt *iarray;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &skipHeader));

  PetscCall(ISGetLayout(is, &map));
  PetscCall(PetscLayoutGetLocalSize(map, &n));
  PetscCall(PetscLayoutGetRange(map, &s, NULL));
  PetscCall(PetscLayoutGetSize(map, &N));

  /* write IS header */
  tr[0] = IS_FILE_CLASSID;
  tr[1] = N;
  if (!skipHeader) PetscCall(PetscViewerBinaryWrite(viewer, tr, 2, PETSC_INT));

  /* write IS indices */
  PetscCall(ISGetIndices(is, &iarray));
  PetscCall(PetscViewerBinaryWriteAll(viewer, iarray, n, s, N, PETSC_INT));
  PetscCall(ISRestoreIndices(is, &iarray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_HDF5)
/*
     This should handle properly the cases where PetscInt is 32 or 64 and hsize_t is 32 or 64. These means properly casting with
   checks back and forth between the two types of variables.
*/
static PetscErrorCode ISLoad_HDF5(IS is, PetscViewer viewer)
{
  PetscInt   *ind;
  const char *isname;

  PetscFunctionBegin;
  PetscCheck(((PetscObject)is)->name, PetscObjectComm((PetscObject)is), PETSC_ERR_SUP, "IS name must be given using PetscObjectSetName() before ISLoad() since HDF5 can store multiple objects in a single file");
  PetscCall(PetscObjectGetName((PetscObject)is, &isname));
  #if defined(PETSC_USE_64BIT_INDICES)
  PetscCall(PetscViewerHDF5Load(viewer, isname, is->map, H5T_NATIVE_LLONG, (void **)&ind));
  #else
  PetscCall(PetscViewerHDF5Load(viewer, isname, is->map, H5T_NATIVE_INT, (void **)&ind));
  #endif
  PetscCall(ISGeneralSetIndices(is, is->map->n, ind, PETSC_OWN_POINTER));
  PetscCall(PetscInfo(is, "Read IS object with name %s of size %" PetscInt_FMT ":%" PetscInt_FMT "\n", isname, is->map->n, is->map->N));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode ISLoad_Binary(IS is, PetscViewer viewer)
{
  PetscBool   isgeneral, skipHeader;
  PetscInt    tr[2], rows, N, n, s, *idx;
  PetscLayout map;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)is, ISGENERAL, &isgeneral));
  PetscCheck(isgeneral, PetscObjectComm((PetscObject)is), PETSC_ERR_ARG_INCOMP, "IS must be of type ISGENERAL to load into it");
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &skipHeader));

  PetscCall(ISGetLayout(is, &map));
  PetscCall(PetscLayoutGetSize(map, &N));

  /* read IS header */
  if (!skipHeader) {
    PetscCall(PetscViewerBinaryRead(viewer, tr, 2, NULL, PETSC_INT));
    PetscCheck(tr[0] == IS_FILE_CLASSID, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Not an IS next in file");
    PetscCheck(tr[1] >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "IS size (%" PetscInt_FMT ") in file is negative", tr[1]);
    PetscCheck(N < 0 || N == tr[1], PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "IS in file different size (%" PetscInt_FMT ") than input IS (%" PetscInt_FMT ")", tr[1], N);
    rows = tr[1];
  } else {
    PetscCheck(N >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "IS binary file header was skipped, thus the user must specify the global size of input IS");
    rows = N;
  }

  /* set IS size if not already set */
  if (N < 0) PetscCall(PetscLayoutSetSize(map, rows));
  PetscCall(PetscLayoutSetUp(map));

  /* get IS sizes and check global size */
  PetscCall(PetscLayoutGetSize(map, &N));
  PetscCall(PetscLayoutGetLocalSize(map, &n));
  PetscCall(PetscLayoutGetRange(map, &s, NULL));
  PetscCheck(N == rows, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "IS in file different size (%" PetscInt_FMT ") than input IS (%" PetscInt_FMT ")", rows, N);

  /* read IS indices */
  PetscCall(PetscMalloc1(n, &idx));
  PetscCall(PetscViewerBinaryReadAll(viewer, idx, n, s, N, PETSC_INT));
  PetscCall(ISGeneralSetIndices(is, n, idx, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ISLoad_Default(IS is, PetscViewer viewer)
{
  PetscBool isbinary, ishdf5;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  if (isbinary) {
    PetscCall(ISLoad_Binary(is, viewer));
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(ISLoad_HDF5(is, viewer));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
