!
!  $Id: petscviewer.h,v 1.26 2000/09/25 18:05:08 balay Exp bsmith $;
!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#if !defined (__VIEWER_H)
#define __VIEWER_H

#define PetscViewer PetscFortranAddr
#define PetscViewerBinaryType integer
#define PetscViewerType character*(80)

#define PETSC_VIEWER_SOCKET 'socket'
#define PETSC_VIEWER_ASCII 'ascii'
#define PETSC_BINARY_VIEWER 'binary'
#define PETSC_VIEWER_STRING 'string'
#define PETSC_DRAW_VIEWER 'draw'
#define PETSC_VIEWER_AMS 'ams'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  Flags for binary I/O
!
      integer  PETSC_BINARY_RDONLY,PETSC_BINARY_WRONLY,PETSC_BINARY_CREATE

      parameter (PETSC_BINARY_RDONLY = 0,PETSC_BINARY_WRONLY = 1)
      parameter (PETSC_BINARY_CREATE = 2)
!
!  PetscViewer formats
!
      integer  PETSC_VIEWER_FORMAT_ASCII_DEFAULT
      integer  PETSC_VIEWER_FORMAT_ASCII_MATLAB
      integer  PETSC_VIEWER_FORMAT_ASCII_IMPL
      integer  PETSC_VIEWER_FORMAT_ASCII_INFO
      integer  PETSC_VIEWER_FORMAT_ASCII_INFO_LONG
      integer  PETSC_VIEWER_FORMAT_ASCII_COMMON
      integer  PETSC_VIEWER_FORMAT_ASCII_SYMMODU
      integer  PETSC_VIEWER_FORMAT_ASCII_INDEX
      integer  PETSC_VIEWER_FORMAT_ASCII_DENSE

      parameter (PETSC_VIEWER_FORMAT_ASCII_DEFAULT = 0)
      parameter (PETSC_VIEWER_FORMAT_ASCII_MATLAB = 1)
      parameter (PETSC_VIEWER_FORMAT_ASCII_IMPL = 2)
      parameter (PETSC_VIEWER_FORMAT_ASCII_INFO = 3)
      parameter (PETSC_VIEWER_FORMAT_ASCII_INFO_LONG = 4)
      parameter (PETSC_VIEWER_FORMAT_ASCII_COMMON = 5)
      parameter (PETSC_VIEWER_FORMAT_ASCII_SYMMODU = 6)
      parameter (PETSC_VIEWER_FORMAT_ASCII_INDEX = 7)
      parameter (PETSC_VIEWER_FORMAT_ASCII_DENSE = 8)

      integer  PETSC_VIEWER_FORMAT_BINARY_DEFAULT
      integer  PETSC_VIEWER_FORMAT_BINARY_NATIVE

      parameter (PETSC_VIEWER_FORMAT_BINARY_DEFAULT = 9) 
      parameter (PETSC_VIEWER_FORMAT_BINARY_NATIVE = 10)

      integer PETSC_VIEWER_FORMAT_DRAW_BASIC
      integer PETSC_VIEWER_FORMAT_DRAW_LG
      integer PETSC_VIEWER_FORMAT_DRAW_CONTOUR

      parameter (PETSC_VIEWER_FORMAT_DRAW_BASIC = 11)
      parameter (PETSC_VIEWER_FORMAT_DRAW_LG = 12)
      parameter (PETSC_VIEWER_FORMAT_DRAW_CONTOUR = 13)

      integer PETSC_VIEWER_FORMAT_NATIVE
      parameter (PETSC_VIEWER_FORMAT_NATIVE = 14)
!
!  End of Fortran include file for the PetscViewer package in PETSc

#endif






