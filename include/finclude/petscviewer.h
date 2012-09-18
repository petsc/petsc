!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#include "finclude/petscviewerdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
      PetscViewer PETSC_VIEWER_STDOUT_
      external PETSC_VIEWER_STDOUT_
#endif
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
      PetscEnum PETSC_VIEWER_DRAW_BASIC
      PetscEnum PETSC_VIEWER_DRAW_LG
      PetscEnum PETSC_VIEWER_DRAW_CONTOUR
      PetscEnum PETSC_VIEWER_DRAW_PORTS
      PetscEnum PETSC_VIEWER_VTK_VTS
      PetscEnum PETSC_VIEWER_NATIVE
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
      parameter (PETSC_VIEWER_DRAW_BASIC = 18)
      parameter (PETSC_VIEWER_DRAW_LG = 19)
      parameter (PETSC_VIEWER_DRAW_CONTOUR = 20)
      parameter (PETSC_VIEWER_DRAW_PORTS = 21)
      parameter (PETSC_VIEWER_VTK_VTS = 22)
      parameter (PETSC_VIEWER_NATIVE = 23)
      parameter (PETSC_VIEWER_NOFORMAT = 24)
!
!  End of Fortran include file for the PetscViewer package in PETSc







