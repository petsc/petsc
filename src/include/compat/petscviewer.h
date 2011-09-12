#ifndef _COMPAT_PETSC_VIEWER_H
#define _COMPAT_PETSC_VIEWER_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define PETSCVIEWERSOCKET       PETSC_VIEWER_SOCKET
#define PETSCVIEWERASCII        PETSC_VIEWER_ASCII
#define PETSCVIEWERBINARY       PETSC_VIEWER_BINARY
#define PETSCVIEWERSTRING       PETSC_VIEWER_STRING
#define PETSCVIEWERDRAW         PETSC_VIEWER_DRAW
#define PETSCVIEWERVU           PETSC_VIEWER_VU
#define PETSCVIEWERMATHEMATICA  PETSC_VIEWER_MATHEMATICA
#define PETSCVIEWERSILO         PETSC_VIEWER_SILO
#define PETSCVIEWERNETCDF       PETSC_VIEWER_NETCDF
#define PETSCVIEWERHDF5         PETSC_VIEWER_HDF5
#define PETSCVIEWERMATLAB       PETSC_VIEWER_MATLAB
#define PETSCVIEWERAMS          "ams"
#endif

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#define PetscViewerFileGetName(v,n) \
        PetscViewerFileGetName((v),(char**)(n))
#endif

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "PetscObjectPrintClassNamePrefixType"
static PetscErrorCode PetscObjectPrintClassNamePrefixType(PetscObject obj,PetscViewer viewer,const char string[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"%s:",string);CHKERRQ(ierr);
  if (obj->name) {
    ierr = PetscViewerASCIIPrintf(viewer,"%s",obj->name);CHKERRQ(ierr);
  }
  if (obj->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"(%s)",obj->prefix);CHKERRQ(ierr);
  }
  /*{
  MPI_Comm       comm;
  PetscMPIInt    size;
  ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer," %d MPI processes",size);CHKERRQ(ierr);
  }*/
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  if (obj->type_name) {
    ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",obj->type_name);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"  type not yet set\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#endif /* _COMPAT_PETSC_VIEWER_H */
