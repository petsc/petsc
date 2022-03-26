
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

  PetscCall(PetscViewerRegister(PETSCVIEWERASCII,      PetscViewerCreate_ASCII));
  PetscCall(PetscViewerRegister(PETSCVIEWERBINARY,     PetscViewerCreate_Binary));
  PetscCall(PetscViewerRegister(PETSCVIEWERSTRING,     PetscViewerCreate_String));
  PetscCall(PetscViewerRegister(PETSCVIEWERDRAW,       PetscViewerCreate_Draw));
#if defined(PETSC_USE_SOCKET_VIEWER)
  PetscCall(PetscViewerRegister(PETSCVIEWERSOCKET,     PetscViewerCreate_Socket));
#endif
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscCall(PetscViewerRegister(PETSCVIEWERMATHEMATICA,PetscViewerCreate_Mathematica));
#endif
  PetscCall(PetscViewerRegister(PETSCVIEWERVU,         PetscViewerCreate_VU));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscViewerRegister(PETSCVIEWERHDF5,       PetscViewerCreate_HDF5));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscCall(PetscViewerRegister(PETSCVIEWERMATLAB,     PetscViewerCreate_Matlab));
#endif
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscViewerRegister(PETSCVIEWERSAWS,        PetscViewerCreate_SAWs));
#endif
  PetscCall(PetscViewerRegister(PETSCVIEWERVTK,        PetscViewerCreate_VTK));
  PetscCall(PetscViewerRegister(PETSCVIEWERGLVIS,      PetscViewerCreate_GLVis));
#if defined(PETSC_HAVE_ADIOS)
  PetscCall(PetscViewerRegister(PETSCVIEWERADIOS,      PetscViewerCreate_ADIOS));
#endif
#if defined(PETSC_HAVE_EXODUSII)
  PetscCall(PetscViewerRegister(PETSCVIEWEREXODUSII,    PetscViewerCreate_ExodusII));
#endif
  PetscFunctionReturn(0);
}
