!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#if !defined (__PETSCVIEWERDEF_H)
#define __PETSCVIEWERDEF_H

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define PetscViewer PetscFortranAddr
#endif

#define PetscViewers PetscFortranAddr
#define PetscFileMode PetscEnum
#define PetscViewerType character*(80)
#define PetscViewerFormat PetscEnum

#define PETSC_VIEWER_SOCKET 'socket'
#define PETSC_VIEWER_ASCII 'ascii'
#define PETSC_VIEWER_BINARY 'binary'
#define PETSC_VIEWER_STRING 'string'
#define PETSC_VIEWER_DRAW 'draw'
#define PETSC_VIEWER_AMS 'ams'
#define PETSC_VIEWER_HDF5 'hdf5'
#define PETSC_VIEWER_NETCDF 'netcdf'
#define PETSC_VIEWER_MATLAB 'matlab'

#endif
