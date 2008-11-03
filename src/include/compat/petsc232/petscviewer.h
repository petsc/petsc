#ifndef _PETSC_COMPAT_VIEWER_H
#define _PETSC_COMPAT_VIEWER_H

#define PETSC_VIEWER_DEFAULT PETSC_VIEWER_ASCII_DEFAULT

#undef __FUNCT__
#define __FUNCT__ "PetscViewerGetFormat_232"
static PETSC_UNUSED
PetscErrorCode PetscViewerGetFormat_232(PetscViewer viewer, PetscViewerFormat *outformat)
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
#define PetscViewerGetFormat PetscViewerGetFormat_232

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSetFormat_232"
static PETSC_UNUSED
PetscErrorCode PetscViewerSetFormat_232(PetscViewer viewer, PetscViewerFormat format)
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
#define PetscViewerSetFormat PetscViewerSetFormat_232

#undef __FUNCT__
#define __FUNCT__ "PetscViewerPushFormat_232"
static PETSC_UNUSED
PetscErrorCode PetscViewerPushFormat_232(PetscViewer viewer, PetscViewerFormat format)
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
#define PetscViewerPushFormat PetscViewerPushFormat_232


#endif /* _PETSC_COMPAT_VIEWER_H */
