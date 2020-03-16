#include <petscis.h>         /*I  "petscis.h"  I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/viewerimpl.h>
#include <petsclayouthdf5.h>

PetscErrorCode ISView_Binary(IS is,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      skipHeader;
  PetscLayout    map;
  PetscInt       tr[2],n,s,N;
  const PetscInt *iarray;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);

  ierr = ISGetLayout(is,&map);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map,&n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map,&s,NULL);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);

  /* write IS header */
  tr[0] = IS_FILE_CLASSID; tr[1] = N;
  if (!skipHeader) {ierr = PetscViewerBinaryWrite(viewer,tr,2,PETSC_INT);CHKERRQ(ierr);}

  /* write IS indices */
  ierr = ISGetIndices(is,&iarray);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWriteAll(viewer,iarray,n,s,N,PETSC_INT);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&iarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
/*
     This should handle properly the cases where PetscInt is 32 or 64 and hsize_t is 32 or 64. These means properly casting with
   checks back and forth between the two types of variables.
*/
PetscErrorCode ISLoad_HDF5(IS is, PetscViewer viewer)
{
  hid_t           inttype;    /* int type (H5T_NATIVE_INT or H5T_NATIVE_LLONG) */
  PetscInt       *ind;
  const char     *isname;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!((PetscObject)is)->name) SETERRQ(PetscObjectComm((PetscObject)is), PETSC_ERR_SUP, "Since HDF5 format gives ASCII name for each object in file; must use ISLoad() after setting name of Vec with PetscObjectSetName()");
#if defined(PETSC_USE_64BIT_INDICES)
  inttype = H5T_NATIVE_LLONG;
#else
  inttype = H5T_NATIVE_INT;
#endif
  ierr = PetscObjectGetName((PetscObject)is, &isname);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Load(viewer, isname, is->map, inttype, (void**)&ind);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(is, is->map->n, ind, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode ISLoad_Binary(IS is, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isgeneral,skipHeader;
  PetscInt       tr[2],rows,N,n,s,*idx;
  PetscLayout    map;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)is,ISGENERAL,&isgeneral);CHKERRQ(ierr);
  if (!isgeneral) SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"IS must be of type ISGENERAL to load into it");
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);

  ierr = ISGetLayout(is,&map);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);

  /* read IS header */
  if (!skipHeader) {
    ierr = PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT);CHKERRQ(ierr);
    if (tr[0] != IS_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not an IS next in file");
    if (tr[1] < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"IS size (%D) in file is negative",tr[1]);
    if (N >= 0 && N != tr[1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"IS in file different size (%D) than input IS (%D)",tr[1],N);
    rows = tr[1];
  } else {
    if (N < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"IS binary file header was skipped, thus the user must specify the global size of input IS");
    rows = N;
  }

  /* set IS size if not already set */
  if (N < 0) {ierr = PetscLayoutSetSize(map,rows);CHKERRQ(ierr);}
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);

  /* get IS sizes and check global size */
  ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map,&n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map,&s,NULL);CHKERRQ(ierr);
  if (N != rows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"IS in file different size (%D) than input IS (%D)",rows,N);

  /* read IS indices */
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadAll(viewer,idx,n,s,N,PETSC_INT);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(is,n,idx,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ISLoad_Default(IS is, PetscViewer viewer)
{
  PetscBool      isbinary,ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
  if (isbinary) {
    ierr = ISLoad_Binary(is, viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = ISLoad_HDF5(is, viewer);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}
