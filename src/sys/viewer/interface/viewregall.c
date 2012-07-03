
#include <petsc-private/viewerimpl.h>  /*I "petscsys.h" I*/  

EXTERN_C_BEGIN
extern PetscErrorCode PetscViewerCreate_Socket(PetscViewer);
extern PetscErrorCode PetscViewerCreate_ASCII(PetscViewer);
extern PetscErrorCode PetscViewerCreate_Binary(PetscViewer);
extern PetscErrorCode PetscViewerCreate_String(PetscViewer);
extern PetscErrorCode PetscViewerCreate_Draw(PetscViewer);
extern PetscErrorCode PetscViewerCreate_VU(PetscViewer);
extern PetscErrorCode PetscViewerCreate_Mathematica(PetscViewer);
extern PetscErrorCode PetscViewerCreate_Netcdf(PetscViewer);
extern PetscErrorCode PetscViewerCreate_HDF5(PetscViewer);
extern PetscErrorCode PetscViewerCreate_Matlab(PetscViewer);
extern PetscErrorCode PetscViewerCreate_AMS(PetscViewer);
extern PetscErrorCode PetscViewerCreate_VTK(PetscViewer);
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerRegisterAll" 
/*@C
  PetscViewerRegisterAll - Registers all of the graphics methods in the PetscViewer package.

  Not Collective

   Level: developer

.seealso:  PetscViewerRegisterDestroy()
@*/
PetscErrorCode  PetscViewerRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERASCII,      path,"PetscViewerCreate_ASCII",      PetscViewerCreate_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERBINARY,     path,"PetscViewerCreate_Binary",     PetscViewerCreate_Binary);CHKERRQ(ierr);
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERSTRING,     path,"PetscViewerCreate_String",     PetscViewerCreate_String);CHKERRQ(ierr);
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERDRAW,       path,"PetscViewerCreate_Draw",       PetscViewerCreate_Draw);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERSOCKET,     path,"PetscViewerCreate_Socket",     PetscViewerCreate_Socket);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERMATHEMATICA,path,"PetscViewerCreate_Mathematica",PetscViewerCreate_Mathematica);CHKERRQ(ierr); 
#endif
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERVU,         path,"PetscViewerCreate_VU",         PetscViewerCreate_VU);CHKERRQ(ierr); 
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERHDF5,       path,"PetscViewerCreate_HDF5",       PetscViewerCreate_HDF5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERMATLAB,     path,"PetscViewerCreate_Matlab",     PetscViewerCreate_Matlab);CHKERRQ(ierr); 
#endif
#if defined(PETSC_HAVE_AMS)
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERAMS,        path,"PetscViewerCreate_AMS",        PetscViewerCreate_AMS);CHKERRQ(ierr); 
#endif
  ierr = PetscViewerRegisterDynamic(PETSCVIEWERVTK,        path,"PetscViewerCreate_VTK",        PetscViewerCreate_VTK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

