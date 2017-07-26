!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#include "petsc/finclude/petscviewer.h"

      type tPetscViewer
        PetscFortranAddr:: v
      end type tPetscViewer

      PetscViewer, parameter :: PETSC_NULL_VIEWER                          &
     &            = tPetscViewer(-1)
!
!     The numbers used below should match those in
!     petsc/private/fortranimpl.h
!
      PetscViewer, parameter :: PETSC_VIEWER_STDOUT_SELF =                &
     &           tPetscViewer(9)
      PetscViewer, parameter :: PETSC_VIEWER_DRAW_WORLD   =                &
     &           tPetscViewer(4)
      PetscViewer, parameter :: PETSC_VIEWER_DRAW_SELF    =                &
     &           tPetscViewer(5)
      PetscViewer, parameter :: PETSC_VIEWER_SOCKET_WORLD =                &
     &           tPetscViewer(6)
      PetscViewer, parameter :: PETSC_VIEWER_SOCKET_SELF  =                &
     &           tPetscViewer(7)
      PetscViewer, parameter :: PETSC_VIEWER_STDOUT_WORLD =                &
     &           tPetscViewer(8)
      PetscViewer, parameter :: PETSC_VIEWER_STDERR_WORLD =                &
     &           tPetscViewer(10)
      PetscViewer, parameter :: PETSC_VIEWER_STDERR_SELF  =                &
     &           tPetscViewer(11)
      PetscViewer, parameter :: PETSC_VIEWER_BINARY_WORLD =                &
     &           tPetscViewer(12)
      PetscViewer, parameter :: PETSC_VIEWER_BINARY_SELF  =                &
     &           tPetscViewer(13)
      PetscViewer, parameter :: PETSC_VIEWER_MATLAB_WORLD =                &
     &           tPetscViewer(14)
      PetscViewer, parameter :: PETSC_VIEWER_MATLAB_SELF  =                &
     &           tPetscViewer(15)

      PetscViewer PETSC_VIEWER_STDOUT_
      external PETSC_VIEWER_STDOUT_
      external PetscViewerAndFormatDestroy
!
!  Flags for binary I/O
!
      PetscEnum FILE_MODE_READ
      PetscEnum FILE_MODE_WRITE
      PetscEnum FILE_MODE_APPEND
      PetscEnum FILE_MODE_UPDATE
      PetscEnum FILE_MODE_APPEND_UPDATE

      parameter (FILE_MODE_READ = 0)
      parameter (FILE_MODE_WRITE = 1)
      parameter (FILE_MODE_APPEND = 2)
      parameter (FILE_MODE_UPDATE = 3)
      parameter (FILE_MODE_APPEND_UPDATE = 4)

!
!  PetscViewer formats
!
      PetscEnum PETSC_VIEWER_DEFAULT
      PetscEnum PETSC_VIEWER_ASCII_MATLAB
      PetscEnum PETSC_VIEWER_ASCII_MATHEMATICA
      PetscEnum PETSC_VIEWER_ASCII_IMPL
      PetscEnum PETSC_VIEWER_ASCII_INFO
      PetscEnum PETSC_VIEWER_ASCII_INFO_DETAIL
      PetscEnum PETSC_VIEWER_ASCII_COMMON
      PetscEnum PETSC_VIEWER_ASCII_SYMMODU
      PetscEnum PETSC_VIEWER_ASCII_INDEX
      PetscEnum PETSC_VIEWER_ASCII_DENSE
      PetscEnum PETSC_VIEWER_ASCII_MATRIXMARKET
      PetscEnum PETSC_VIEWER_ASCII_VTK
      PetscEnum PETSC_VIEWER_ASCII_VTK_CELL
      PetscEnum PETSC_VIEWER_ASCII_VTK_COORDS
      PetscEnum PETSC_VIEWER_ASCII_PCICE
      PetscEnum PETSC_VIEWER_ASCII_PYTHON
      PetscEnum PETSC_VIEWER_ASCII_FACTOR_INFO
      PetscEnum PETSC_VIEWER_ASCII_LATEX
      PetscEnum PETSC_VIEWER_ASCII_XML
      PetscEnum PETSC_VIEWER_ASCII_GLVIS
      PetscEnum PETSC_VIEWER_DRAW_BASIC
      PetscEnum PETSC_VIEWER_DRAW_LG
      PetscEnum PETSC_VIEWER_DRAW_CONTOUR
      PetscEnum PETSC_VIEWER_DRAW_PORTS
      PetscEnum PETSC_VIEWER_VTK_VTS
      PetscEnum PETSC_VIEWER_VTK_VTR
      PetscEnum PETSC_VIEWER_VTK_VTU
      PetscEnum PETSC_VIEWER_BINARY_MATLAB
      PetscEnum PETSC_VIEWER_NATIVE
      PetscEnum PETSC_VIEWER_HDF5_VIZ
      PetscEnum PETSC_VIEWER_NOFORMAT

      parameter (PETSC_VIEWER_DEFAULT = 0)
      parameter (PETSC_VIEWER_ASCII_MATLAB = 1)
      parameter (PETSC_VIEWER_ASCII_MATHEMATICA = 2)
      parameter (PETSC_VIEWER_ASCII_IMPL = 3)
      parameter (PETSC_VIEWER_ASCII_INFO = 4)
      parameter (PETSC_VIEWER_ASCII_INFO_DETAIL = 5)
      parameter (PETSC_VIEWER_ASCII_COMMON = 6)
      parameter (PETSC_VIEWER_ASCII_SYMMODU = 7)
      parameter (PETSC_VIEWER_ASCII_INDEX = 8)
      parameter (PETSC_VIEWER_ASCII_DENSE = 9)
      parameter (PETSC_VIEWER_ASCII_MATRIXMARKET = 10)
      parameter (PETSC_VIEWER_ASCII_VTK = 11)
      parameter (PETSC_VIEWER_ASCII_VTK_CELL = 12)
      parameter (PETSC_VIEWER_ASCII_VTK_COORDS = 13)
      parameter (PETSC_VIEWER_ASCII_PCICE = 14)
      parameter (PETSC_VIEWER_ASCII_PYTHON = 15)
      parameter (PETSC_VIEWER_ASCII_FACTOR_INFO = 16)
      parameter (PETSC_VIEWER_ASCII_LATEX = 17)
      parameter (PETSC_VIEWER_ASCII_XML = 18)
      parameter (PETSC_VIEWER_ASCII_GLVIS = 19)
      parameter (PETSC_VIEWER_DRAW_BASIC = 20)
      parameter (PETSC_VIEWER_DRAW_LG = 21)
      parameter (PETSC_VIEWER_DRAW_CONTOUR = 22)
      parameter (PETSC_VIEWER_DRAW_PORTS = 23)
      parameter (PETSC_VIEWER_VTK_VTS = 24)
      parameter (PETSC_VIEWER_VTK_VTR = 25)
      parameter (PETSC_VIEWER_VTK_VTU = 26)
      parameter (PETSC_VIEWER_BINARY_MATLAB = 27)
      parameter (PETSC_VIEWER_NATIVE = 28)
      parameter (PETSC_VIEWER_HDF5_VIZ = 29)
      parameter (PETSC_VIEWER_NOFORMAT = 30)
!
!  End of Fortran include file for the PetscViewer package in PETSc







