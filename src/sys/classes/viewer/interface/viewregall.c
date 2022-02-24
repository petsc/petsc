
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
@*/
PetscErrorCode  PetscViewerRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscViewerRegisterAllCalled) PetscFunctionReturn(0);
  PetscViewerRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscViewerRegister(PETSCVIEWERASCII,      PetscViewerCreate_ASCII));
  CHKERRQ(PetscViewerRegister(PETSCVIEWERBINARY,     PetscViewerCreate_Binary));
  CHKERRQ(PetscViewerRegister(PETSCVIEWERSTRING,     PetscViewerCreate_String));
  CHKERRQ(PetscViewerRegister(PETSCVIEWERDRAW,       PetscViewerCreate_Draw));
#if defined(PETSC_USE_SOCKET_VIEWER)
  CHKERRQ(PetscViewerRegister(PETSCVIEWERSOCKET,     PetscViewerCreate_Socket));
#endif
#if defined(PETSC_HAVE_MATHEMATICA)
  CHKERRQ(PetscViewerRegister(PETSCVIEWERMATHEMATICA,PetscViewerCreate_Mathematica));
#endif
  CHKERRQ(PetscViewerRegister(PETSCVIEWERVU,         PetscViewerCreate_VU));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscViewerRegister(PETSCVIEWERHDF5,       PetscViewerCreate_HDF5));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  CHKERRQ(PetscViewerRegister(PETSCVIEWERMATLAB,     PetscViewerCreate_Matlab));
#endif
#if defined(PETSC_HAVE_SAWS)
  CHKERRQ(PetscViewerRegister(PETSCVIEWERSAWS,        PetscViewerCreate_SAWs));
#endif
  CHKERRQ(PetscViewerRegister(PETSCVIEWERVTK,        PetscViewerCreate_VTK));
  CHKERRQ(PetscViewerRegister(PETSCVIEWERGLVIS,      PetscViewerCreate_GLVis));
#if defined(PETSC_HAVE_ADIOS)
  CHKERRQ(PetscViewerRegister(PETSCVIEWERADIOS,      PetscViewerCreate_ADIOS));
#endif
#if defined(PETSC_HAVE_EXODUSII)
  CHKERRQ(PetscViewerRegister(PETSCVIEWEREXODUSII,    PetscViewerCreate_ExodusII));
#endif
  PetscFunctionReturn(0);
}
