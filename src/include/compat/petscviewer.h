#ifndef _COMPAT_PETSC_VIEWER_H
#define _COMPAT_PETSC_VIEWER_H


#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PETSC_VIEWER_HDF5 PETSC_VIEWER_HDF4
#define PETSC_VIEWER_ASCII_MATRIXMARKET ((PetscViewerFormat)-1)
#define PETSC_VIEWER_ASCII_PYTHON       ((PetscViewerFormat)-1)
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PETSC_VIEWER_DEFAULT PETSC_VIEWER_ASCII_DEFAULT
#undef __FUNCT__
#define __FUNCT__ "PetscViewerGetFormat"
static PETSC_UNUSED
PetscErrorCode PetscViewerGetFormat_Compat(PetscViewer viewer, PetscViewerFormat *outformat)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(format,2);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_BINARY_DEFAULT)
    format = PETSC_VIEWER_DEFAULT;
  else if (format == PETSC_VIEWER_BINARY_NATIVE)
    format = PETSC_VIEWER_NATIVE;
  *outformat = format;
  PetscFunctionReturn(0);
}
#define PetscViewerGetFormat PetscViewerGetFormat_Compat
#undef __FUNCT__
#define __FUNCT__ "PetscViewerSetFormat"
static PETSC_UNUSED
PetscErrorCode PetscViewerSetFormat_Compat(PetscViewer viewer, PetscViewerFormat format)
{
  PetscTruth      isbinary;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  /*PetscValidType(viewer,1);*/
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary){
    if (format == PETSC_VIEWER_DEFAULT)
      format = PETSC_VIEWER_BINARY_DEFAULT;
    else if (format == PETSC_VIEWER_NATIVE)
      format = PETSC_VIEWER_BINARY_NATIVE;
  }
  ierr = PetscViewerSetFormat(viewer,format);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscViewerSetFormat PetscViewerSetFormat_Compat
#undef __FUNCT__
#define __FUNCT__ "PetscViewerPushFormat"
static PETSC_UNUSED
PetscErrorCode PetscViewerPushFormat_Compat(PetscViewer viewer, PetscViewerFormat format)
{
  PetscTruth      isbinary;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  /*PetscValidType(viewer,1);*/
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {
    if (format == PETSC_VIEWER_DEFAULT)
      format = PETSC_VIEWER_BINARY_DEFAULT;
    else if (format == PETSC_VIEWER_NATIVE)
      format = PETSC_VIEWER_BINARY_NATIVE;
  }
  ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscViewerPushFormat PetscViewerPushFormat_Compat
#endif

#endif /* _COMPAT_PETSC_VIEWER_H */
