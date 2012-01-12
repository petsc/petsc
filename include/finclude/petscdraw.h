!
!
!  Include file for Fortran use of the PetscDraw package in PETSc
!
#include "finclude/petscdrawdef.h"
!
!  Flags for draw
!
      PetscEnum PETSC_DRAW_BASIC_COLORS
      PetscEnum PETSC_DRAW_ROTATE
      parameter (PETSC_DRAW_BASIC_COLORS=33,PETSC_DRAW_ROTATE=-1)
!
!  Colors for drawing
!
      PetscEnum PETSC_DRAW_WHITE
      PetscEnum PETSC_DRAW_BLACK
      PetscEnum PETSC_DRAW_RED
      PetscEnum PETSC_DRAW_GREEN
      PetscEnum PETSC_DRAW_CYAN
      PetscEnum PETSC_DRAW_BLUE
      PetscEnum PETSC_DRAW_MAGENTA
      PetscEnum PETSC_DRAW_AQUAMARINE
      PetscEnum PETSC_DRAW_FORESTGREEN
      PetscEnum PETSC_DRAW_ORANGE
      PetscEnum PETSC_DRAW_BROWN
      PetscEnum PETSC_DRAW_PINK
      PetscEnum PETSC_DRAW_CORAL
      PetscEnum PETSC_DRAW_GRAY
      PetscEnum PETSC_DRAW_YELLOW
      PetscEnum PETSC_DRAW_GOLD
      PetscEnum PETSC_DRAW_LIGHTPINK
      PetscEnum PETSC_DRAW_MEDIUMTURQUOISE
      PetscEnum PETSC_DRAW_KHAKI
      PetscEnum PETSC_DRAW_DIMGRAY
      PetscEnum PETSC_DRAW_SKYBLUE
      PetscEnum PETSC_DRAW_DARKGREEN
      PetscEnum PETSC_DRAW_NAVYBLUE
      PetscEnum PETSC_DRAW_SANDYBROWN
      PetscEnum PETSC_DRAW_CADETBLUE
      PetscEnum PETSC_DRAW_POWDERBLUE
      PetscEnum PETSC_DRAW_DEEPPINK
      PetscEnum PETSC_DRAW_THISTLE
      PetscEnum PETSC_DRAW_LIMEGREEN
      PetscEnum PETSC_DRAW_LAVENDERBLUSH
      PetscEnum PETSC_DRAW_PLUM
      PetscEnum PETSC_DRAW_YELLOWGREEN
      PetscEnum PETSC_DRAW_VIOLET

      parameter (PETSC_DRAW_WHITE = 0,PETSC_DRAW_BLACK = 1)
      parameter (PETSC_DRAW_RED = 2,PETSC_DRAW_GREEN = 3)
      parameter (PETSC_DRAW_CYAN = 4,PETSC_DRAW_BLUE = 5)
      parameter (PETSC_DRAW_MAGENTA = 6,PETSC_DRAW_AQUAMARINE = 7)
      parameter (PETSC_DRAW_FORESTGREEN = 8,PETSC_DRAW_ORANGE = 9)
      parameter (PETSC_DRAW_VIOLET = 10,PETSC_DRAW_BROWN = 11)
      parameter (PETSC_DRAW_PINK = 12,PETSC_DRAW_CORAL = 13)
      parameter (PETSC_DRAW_GRAY = 14,PETSC_DRAW_YELLOW = 15)
      parameter (PETSC_DRAW_GOLD = 16,PETSC_DRAW_LIGHTPINK = 17)
      parameter (PETSC_DRAW_MEDIUMTURQUOISE = 18)
      parameter (PETSC_DRAW_KHAKI = 19)
      parameter (PETSC_DRAW_DIMGRAY = 20)
      parameter (PETSC_DRAW_YELLOWGREEN = 21)
      parameter (PETSC_DRAW_SKYBLUE = 22)
      parameter (PETSC_DRAW_DARKGREEN = 23)
      parameter (PETSC_DRAW_NAVYBLUE = 24)
      parameter (PETSC_DRAW_SANDYBROWN = 25)
      parameter (PETSC_DRAW_CADETBLUE = 26)
      parameter (PETSC_DRAW_POWDERBLUE = 27)
      parameter (PETSC_DRAW_DEEPPINK = 28)
      parameter (PETSC_DRAW_THISTLE = 29)
      parameter (PETSC_DRAW_LIMEGREEN = 30)
      parameter (PETSC_DRAW_LAVENDERBLUSH =31)
      parameter (PETSC_DRAW_PLUM = 32)

      PetscEnum BUTTON_NONE
      PetscEnum BUTTON_LEFT
      PetscEnum BUTTON_CENTER
      PetscEnum BUTTON_RIGHT
      PetscEnum BUTTON_LEFT_SHIFT
      PetscEnum BUTTON_CENTER_SHIFT
      PetscEnum BUTTON_RIGHT_SHIFT

      parameter (BUTTON_NONE = 0,BUTTON_LEFT = 1) 
      parameter (BUTTON_CENTER = 2,BUTTON_RIGHT = 3)
      parameter (BUTTON_LEFT_SHIFT = 4, BUTTON_CENTER_SHIFT = 5)
      parameter (BUTTON_RIGHT_SHIFT = 6)

!
!  End of Fortran include file for the PetscDraw package in PETSc

