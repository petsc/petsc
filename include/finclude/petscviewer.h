!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#if !defined (__VIEWER_H)
#define __VIEWER_H

#define PetscViewer PetscFortranAddr
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
#define PETSC_VIEWER_HDF4 'hdf4'
#define PETSC_VIEWER_NETCDF 'netcdf'
#define PETSC_VIEWER_MATLAB 'matlab'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  Flags for binary I/O
!
      PetscEnum FILE_MODE_READ,
      PetscEnum FILE_MODE_WRITE
      PetscEnum FILE_MODE_APPEND
      PetscEnum FILE_MODE_UPDATE,
      PetscEnum FILE_MODE_APPEND_UPDATE

      parameter (FILE_MODE_READ = 0)
      parameter (FILE_MODE_WRITE = 1)
      parameter (FILE_MODE_APPEND = 2)
      parameter (FILE_MODE_UPDATE = 3)
      parameter (FILE_MODE_APPEND_UPDATE = 4)

!
!  PetscViewer formats
!
      PetscEnum  PETSC_VIEWER_ASCII_DEFAULT
      PetscEnum  PETSC_VIEWER_ASCII_MATLAB
      PetscEnum  PETSC_VIEWER_ASCII_IMPL
      PetscEnum  PETSC_VIEWER_ASCII_INFO
      PetscEnum  PETSC_VIEWER_ASCII_INFO_DETAIL 
      PetscEnum  PETSC_VIEWER_ASCII_COMMON
      PetscEnum  PETSC_VIEWER_ASCII_SYMMODU
      PetscEnum  PETSC_VIEWER_ASCII_INDEX
      PetscEnum  PETSC_VIEWER_ASCII_DENSE

      parameter (PETSC_VIEWER_ASCII_DEFAULT = 0)
      parameter (PETSC_VIEWER_ASCII_MATLAB = 1)
      parameter (PETSC_VIEWER_ASCII_IMPL = 2)
      parameter (PETSC_VIEWER_ASCII_INFO = 3)
      parameter (PETSC_VIEWER_ASCII_INFO_DETAIL = 4)
      parameter (PETSC_VIEWER_ASCII_COMMON = 5)
      parameter (PETSC_VIEWER_ASCII_SYMMODU = 6)
      parameter (PETSC_VIEWER_ASCII_INDEX = 7)
      parameter (PETSC_VIEWER_ASCII_DENSE = 8)

      PetscEnum  PETSC_VIEWER_BINARY_DEFAULT
      PetscEnum  PETSC_VIEWER_BINARY_NATIVE

      parameter (PETSC_VIEWER_BINARY_DEFAULT = 9) 
      parameter (PETSC_VIEWER_BINARY_NATIVE = 10)

      PetscEnum PETSC_VIEWER_DRAW_BASIC
      PetscEnum PETSC_VIEWER_DRAW_LG
      PetscEnum PETSC_VIEWER_DRAW_CONTOUR
      PetscEnum PETSC_VIEWER_DRAW_PORTS

      parameter (PETSC_VIEWER_DRAW_BASIC = 11)
      parameter (PETSC_VIEWER_DRAW_LG = 12)
      parameter (PETSC_VIEWER_DRAW_CONTOUR = 13)
      parameter (PETSC_VIEWER_DRAW_PORTS = 14)

      PetscEnum PETSC_VIEWER_NATIVE
      parameter (PETSC_VIEWER_NATIVE = 15)
      PetscEnum PETSC_VIEWER_NOFORMAT
      parameter (PETSC_VIEWER_NOFORMAT = 16)
!
!  End of Fortran include file for the PetscViewer package in PETSc

#endif






