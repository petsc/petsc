!
!
!  Include file for Fortran use of the PetscDraw package in PETSc
!

#if !defined (__PETSCDRAW_H)
#define __PETSCDRAW_H

#define PetscDraw PetscFortranAddr
#define PetscDrawLG PetscFortranAddr
#define PetscDrawAxis PetscFortranAddr
#define PetscDrawSP PetscFortranAddr
#define PetscDrawHG PetscFortranAddr
#define PetscDrawMesh PetscFortranAddr
#define PetscDrawButton integer
#define PetscDrawType character*(80)

!
!  types of draw context
!
#define PETSC_DRAW_X 'x'
#define PETSC_DRAW_NULL 'null'
#define PETSC_DRAW_PS 'ps'
#define PETSC_DRAW_WIN32 'win32'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  Flags for draw
!
      integer PETSC_DRAW_BASIC_COLORS,PETSC_DRAW_ROTATE
      parameter (PETSC_DRAW_BASIC_COLORS=33,PETSC_DRAW_ROTATE=-1)
!
!  Colors for drawing
!
      integer PETSC_DRAW_WHITE,PETSC_DRAW_BLACK,PETSC_DRAW_RED
      integer PETSC_DRAW_GREEN,PETSC_DRAW_CYAN,PETSC_DRAW_BLUE
      integer PETSC_DRAW_MAGENTA,PETSC_DRAW_AQUAMARINE
      integer PETSC_DRAW_FORESTGREEN,PETSC_DRAW_ORANGE,PETSC_DRAW_VIOLET
      integer PETSC_DRAW_BROWN,PETSC_DRAW_PINK,PETSC_DRAW_CORAL
      integer PETSC_DRAW_GRAY,PETSC_DRAW_YELLOW,PETSC_DRAW_GOLD
      integer PETSC_DRAW_LIGHTPINK,PETSC_DRAW_MEDIUMTURQUOISE
      integer PETSC_DRAW_KHAKI,PETSC_DRAW_DIMGRAY,PETSC_DRAW_YELLOWGREEN
      integer PETSC_DRAW_SKYBLUE,PETSC_DRAW_DARKGREEN
      integer PETSC_DRAW_NAVYBLUE,PETSC_DRAW_SANDYBROWN
      integer PETSC_DRAW_CADETBLUE,PETSC_DRAW_POWDERBLUE
      integer PETSC_DRAW_DEEPPINK,PETSC_DRAW_THISTLE
      integer PETSC_DRAW_LIMEGREEN,PETSC_DRAW_LAVENDERBLUSH
      integer PETSC_DRAW_PLUM


      parameter (PETSC_DRAW_WHITE = 0,PETSC_DRAW_BLACK = 1)
      parameter (PETSC_DRAW_RED = 2,PETSC_DRAW_GREEN = 3)
      parameter (PETSC_DRAW_CYAN = 4,PETSC_DRAW_BLUE = 5)
      parameter (PETSC_DRAW_MAGENTA = 6,PETSC_DRAW_AQUAMARINE = 7)
      parameter (PETSC_DRAW_FORESTGREEN = 8,PETSC_DRAW_ORANGE = 9)
      parameter (PETSC_DRAW_VIOLET = 10,PETSC_DRAW_BROWN = 11)
      parameter (PETSC_DRAW_PINK = 12,PETSC_DRAW_CORAL = 13)
      parameter (PETSC_DRAW_GRAY = 14,PETSC_DRAW_YELLOW = 15)
      parameter (PETSC_DRAW_GOLD = 16,PETSC_DRAW_LIGHTPINK = 17)
      parameter (PETSC_DRAW_MEDIUMTURQUOISE = 18,PETSC_DRAW_KHAKI = 19)
      parameter (PETSC_DRAW_DIMGRAY = 20,PETSC_DRAW_YELLOWGREEN = 21)
      parameter (PETSC_DRAW_SKYBLUE = 22,PETSC_DRAW_DARKGREEN = 23)
      parameter (PETSC_DRAW_NAVYBLUE = 24,PETSC_DRAW_SANDYBROWN = 25)
      parameter (PETSC_DRAW_CADETBLUE = 26,PETSC_DRAW_POWDERBLUE = 27)
      parameter (PETSC_DRAW_DEEPPINK = 28,PETSC_DRAW_THISTLE = 29)
      parameter (PETSC_DRAW_LIMEGREEN = 30,PETSC_DRAW_LAVENDERBLUSH =31)
      parameter (PETSC_DRAW_PLUM = 32)

      integer BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT

      parameter (BUTTON_NONE = 0,BUTTON_LEFT = 1) 
      parameter (BUTTON_CENTER = 2,BUTTON_RIGHT = 3)

!
!  End of Fortran include file for the PetscDraw package in PETSc

#endif
