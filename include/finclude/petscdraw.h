!
!  $Id: draw.h,v 1.19 1999/01/13 22:37:30 bsmith Exp bsmith $;
!
!  Include file for Fortran use of the Draw package in PETSc
!
#define Draw       PetscFortranAddr
#define DrawLG     PetscFortranAddr
#define DrawAxis   PetscFortranAddr
#define DrawSP     PetscFortranAddr
#define DrawHist   PetscFortranAddr
#define DrawMesh   PetscFortranAddr
#define DrawButton integer
#define DrawType   character*(80)

!
!  types of draw context
!
#define DRAW_X    'x'
#define DRAW_NULL 'null'
!
!  Colors for drawing
!
      integer DRAW_WHITE,DRAW_BLACK,DRAW_RED,DRAW_GREEN,DRAW_CYAN
      integer DRAW_BLUE,DRAW_MAGENTA,DRAW_AQUAMARINE,DRAW_FORESTGREEN
      integer DRAW_ORANGE,DRAW_VIOLET,DRAW_BROWN,DRAW_PINK,DRAW_CORAL
      integer DRAW_GRAY,DRAW_YELLOW,DRAW_GOLD,DRAW_LIGHTPINK
      integer DRAW_MEDIUMTURQUOISE,DRAW_KHAKI,DRAW_DIMGRAY
      integer DRAW_YELLOWGREEN,DRAW_SKYBLUE,DRAW_DARKGREEN
      integer DRAW_NAVYBLUE,DRAW_SANDYBROWN,DRAW_CADETBLUE
      integer DRAW_POWDERBLUE,DRAW_DEEPPINK,DRAW_THISTLE,DRAW_LIMEGREEN
      integer DRAW_LAVENDERBLUSH


      parameter (DRAW_WHITE = 0, DRAW_BLACK = 1, DRAW_RED = 2)
      parameter (DRAW_GREEN = 3, DRAW_CYAN = 4, DRAW_BLUE = 5)
      parameter (DRAW_MAGENTA = 6, DRAW_AQUAMARINE = 7)
      parameter (DRAW_FORESTGREEN = 8, DRAW_ORANGE = 9)
      parameter (DRAW_VIOLET = 10, DRAW_BROWN = 11, DRAW_PINK = 12)
      parameter (DRAW_CORAL = 13, DRAW_GRAY = 14, DRAW_YELLOW = 15)
      parameter (DRAW_GOLD = 16, DRAW_LIGHTPINK = 17)
      parameter (DRAW_MEDIUMTURQUOISE = 18, DRAW_KHAKI = 19)
      parameter (DRAW_DIMGRAY = 20, DRAW_YELLOWGREEN = 21)
      parameter (DRAW_SKYBLUE = 22, DRAW_DARKGREEN = 23)
      parameter (DRAW_NAVYBLUE = 24, DRAW_SANDYBROWN = 25)
      parameter (DRAW_CADETBLUE = 26, DRAW_POWDERBLUE = 27)
      parameter (DRAW_DEEPPINK = 28, DRAW_THISTLE = 29)
      parameter (DRAW_LIMEGREEN = 30, DRAW_LAVENDERBLUSH = 31)


      integer BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT

      parameter (BUTTON_NONE = 0, BUTTON_LEFT = 1) 
      parameter (BUTTON_CENTER = 2, BUTTON_RIGHT = 3)

!
!  End of Fortran include file for the Draw package in PETSc





