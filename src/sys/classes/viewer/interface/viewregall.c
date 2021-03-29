
#include <petsc/private/viewerimpl.h>  /*I "petscsys.h" I*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_Socket(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_ASCII(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_Binary(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_String(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_Draw(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_VU(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_Mathematica(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_HDF5(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_Matlab(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_SAWs(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_VTK(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_GLVis(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_ADIOS(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCreate_ExodusII(PetscViewer);

PetscBool PetscViewerRegisterAllCalled;

/*@C
  PetscViewerRegisterAll - Registers all of the graphics methods in the PetscViewer package.

  Not Collective

   Level: developer

.seealso:  PetscViewerRegisterDestroy()
@*/
PetscErrorCode  PetscViewerRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscViewerRegisterAllCalled) PetscFunctionReturn(0);
  PetscViewerRegisterAllCalled = PETSC_TRUE;

  ierr = PetscViewerRegister(PETSCVIEWERASCII,      PetscViewerCreate_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerRegister(PETSCVIEWERBINARY,     PetscViewerCreate_Binary);CHKERRQ(ierr);
  ierr = PetscViewerRegister(PETSCVIEWERSTRING,     PetscViewerCreate_String);CHKERRQ(ierr);
  ierr = PetscViewerRegister(PETSCVIEWERDRAW,       PetscViewerCreate_Draw);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = PetscViewerRegister(PETSCVIEWERSOCKET,     PetscViewerCreate_Socket);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscViewerRegister(PETSCVIEWERMATHEMATICA,PetscViewerCreate_Mathematica);CHKERRQ(ierr);
#endif
  ierr = PetscViewerRegister(PETSCVIEWERVU,         PetscViewerCreate_VU);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscViewerRegister(PETSCVIEWERHDF5,       PetscViewerCreate_HDF5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscViewerRegister(PETSCVIEWERMATLAB,     PetscViewerCreate_Matlab);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SAWS)
  ierr = PetscViewerRegister(PETSCVIEWERSAWS,        PetscViewerCreate_SAWs);CHKERRQ(ierr);
#endif
  ierr = PetscViewerRegister(PETSCVIEWERVTK,        PetscViewerCreate_VTK);CHKERRQ(ierr);
  ierr = PetscViewerRegister(PETSCVIEWERGLVIS,      PetscViewerCreate_GLVis);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ADIOS)
  ierr = PetscViewerRegister(PETSCVIEWERADIOS,      PetscViewerCreate_ADIOS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_EXODUSII)
  ierr = PetscViewerRegister(PETSCVIEWEREXODUSII,    PetscViewerCreate_ExodusII);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
