!
!
!  Include file for Fortran use of the PetscDraw package in PETSc
!
#include "petsc/finclude/petscdraw.h"
!
!  Flags for draw
!
      PetscEnum, parameter :: PETSC_DRAW_BASIC_COLORS = 33
      PetscEnum, parameter :: PETSC_DRAW_ROTATE = -1
!
!  Colors for drawing
!
      PetscEnum, parameter :: PETSC_DRAW_WHITE = 0
      PetscEnum, parameter :: PETSC_DRAW_BLACK = 1
      PetscEnum, parameter :: PETSC_DRAW_RED = 2
      PetscEnum, parameter :: PETSC_DRAW_GREEN = 3
      PetscEnum, parameter :: PETSC_DRAW_CYAN = 4
      PetscEnum, parameter :: PETSC_DRAW_BLUE = 5
      PetscEnum, parameter :: PETSC_DRAW_MAGENTA = 6
      PetscEnum, parameter :: PETSC_DRAW_AQUAMARINE = 7
      PetscEnum, parameter :: PETSC_DRAW_FORESTGREEN = 8
      PetscEnum, parameter :: PETSC_DRAW_ORANGE = 9
      PetscEnum, parameter :: PETSC_DRAW_VIOLET = 10
      PetscEnum, parameter :: PETSC_DRAW_BROWN = 11
      PetscEnum, parameter :: PETSC_DRAW_PINK = 12
      PetscEnum, parameter :: PETSC_DRAW_CORAL = 13
      PetscEnum, parameter :: PETSC_DRAW_GRAY = 14
      PetscEnum, parameter :: PETSC_DRAW_YELLOW = 15
      PetscEnum, parameter :: PETSC_DRAW_GOLD = 16
      PetscEnum, parameter :: PETSC_DRAW_LIGHTPINK = 17
      PetscEnum, parameter :: PETSC_DRAW_MEDIUMTURQUOISE = 18
      PetscEnum, parameter :: PETSC_DRAW_KHAKI = 19
      PetscEnum, parameter :: PETSC_DRAW_DIMGRAY = 20
      PetscEnum, parameter :: PETSC_DRAW_YELLOWGREEN = 21
      PetscEnum, parameter :: PETSC_DRAW_SKYBLUE = 22
      PetscEnum, parameter :: PETSC_DRAW_DARKGREEN = 23
      PetscEnum, parameter :: PETSC_DRAW_NAVYBLUE = 24
      PetscEnum, parameter :: PETSC_DRAW_SANDYBROWN = 25
      PetscEnum, parameter :: PETSC_DRAW_CADETBLUE = 26
      PetscEnum, parameter :: PETSC_DRAW_POWDERBLUE = 27
      PetscEnum, parameter :: PETSC_DRAW_DEEPPINK = 28
      PetscEnum, parameter :: PETSC_DRAW_THISTLE = 29
      PetscEnum, parameter :: PETSC_DRAW_LIMEGREEN = 30
      PetscEnum, parameter :: PETSC_DRAW_LAVENDERBLUSH =31
      PetscEnum, parameter :: PETSC_DRAW_PLUM = 32

      PetscEnum, parameter :: PETSC_BUTTON_NONE = 0
      PetscEnum, parameter :: PETSC_BUTTON_LEFT = 1
      PetscEnum, parameter :: PETSC_BUTTON_CENTER = 2
      PetscEnum, parameter :: PETSC_BUTTON_RIGHT = 3
      PetscEnum, parameter :: PETSC_BUTTON_WHEEL_UP = 4
      PetscEnum, parameter :: PETSC_BUTTON_WHEEL_DOWN = 5
      PetscEnum, parameter :: PETSC_BUTTON_LEFT_SHIFT = 6
      PetscEnum, parameter :: PETSC_BUTTON_CENTER_SHIFT = 7
      PetscEnum, parameter :: PETSC_BUTTON_RIGHT_SHIFT = 8
!
!  End of Fortran include file for the PetscDraw package in PETSc

