!
!  $Id: petscviewer.h,v 1.31 2001/04/10 22:37:56 balay Exp $;
!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#if !defined (__VIEWER_H)
#define __VIEWER_H

#define PetscViewer PetscFortranAddr
#define PetscViewers PetscFortranAddr
#define PetscViewerBinaryType integer
#define PetscViewerType character*(80)
#define PetscViewerFormat integer

#define PETSC_VIEWER_SOCKET 'socket'
#define PETSC_VIEWER_ASCII 'ascii'
#define PETSC_VIEWER_BINARY 'binary'
#define PETSC_VIEWER_STRING 'string'
#define PETSC_VIEWER_DRAW 'draw'
#define PETSC_VIEWER_AMS 'ams'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  Flags for binary I/O
!
      integer  PETSC_BINARY_RDONLY,PETSC_BINARY_WRONLY
      integer  PETSC_BINARY_CREATE

      parameter (PETSC_BINARY_RDONLY = 0,PETSC_BINARY_WRONLY = 1)
      parameter (PETSC_BINARY_CREATE = 2)
!
!  PetscViewer formats
!
      integer  PETSC_VIEWER_ASCII_DEFAULT
      integer  PETSC_VIEWER_ASCII_MATLAB
      integer  PETSC_VIEWER_ASCII_IMPL
      integer  PETSC_VIEWER_ASCII_INFO
      integer  PETSC_VIEWER_ASCII_INFO_DETAIL 
      integer  PETSC_VIEWER_ASCII_COMMON
      integer  PETSC_VIEWER_ASCII_SYMMODU
      integer  PETSC_VIEWER_ASCII_INDEX
      integer  PETSC_VIEWER_ASCII_DENSE

      parameter (PETSC_VIEWER_ASCII_DEFAULT = 0)
      parameter (PETSC_VIEWER_ASCII_MATLAB = 1)
      parameter (PETSC_VIEWER_ASCII_IMPL = 2)
      parameter (PETSC_VIEWER_ASCII_INFO = 3)
      parameter (PETSC_VIEWER_ASCII_INFO_DETAIL = 4)
      parameter (PETSC_VIEWER_ASCII_COMMON = 5)
      parameter (PETSC_VIEWER_ASCII_SYMMODU = 6)
      parameter (PETSC_VIEWER_ASCII_INDEX = 7)
      parameter (PETSC_VIEWER_ASCII_DENSE = 8)

      integer  PETSC_VIEWER_BINARY_DEFAULT
      integer  PETSC_VIEWER_BINARY_NATIVE

      parameter (PETSC_VIEWER_BINARY_DEFAULT = 9) 
      parameter (PETSC_VIEWER_BINARY_NATIVE = 10)

      integer PETSC_VIEWER_DRAW_BASIC
      integer PETSC_VIEWER_DRAW_LG
      integer PETSC_VIEWER_DRAW_CONTOUR
      integer PETSC_VIEWER_DRAW_PORTS

      parameter (PETSC_VIEWER_DRAW_BASIC = 11)
      parameter (PETSC_VIEWER_DRAW_LG = 12)
      parameter (PETSC_VIEWER_DRAW_CONTOUR = 13)
      parameter (PETSC_VIEWER_DRAW_PORTS = 14)

      integer PETSC_VIEWER_NATIVE
      parameter (PETSC_VIEWER_NATIVE = 15)
      integer PETSC_VIEWER_NOFORMAT
      parameter (PETSC_VIEWER_NOFORMAT = 16)
!
!  End of Fortran include file for the PetscViewer package in PETSc

#endif






