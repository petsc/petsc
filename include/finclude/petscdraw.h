!
!  $Id: draw.h,v 1.15 1998/03/25 00:37:25 balay Exp balay $;
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
!
!  types of draw context
!
      integer DRAW_XWINDOW, DRAW_NULLWINDOW, DRAW_VRML
      parameter (DRAW_XWINDOW=0, DRAW_NULLWINDOW=1, DRAW_VRML=2)
!
!  Colors for drawing
!
      integer DRAW_WHITE,DRAW_BLACK,DRAW_RED,DRAW_GREEN,DRAW_CYAN,
     *        DRAW_BLUE,DRAW_MAGENTA,DRAW_AQUAMARINE,DRAW_FORESTGREEN,
     *        DRAW_ORANGE,DRAW_VIOLET,DRAW_BROWN,DRAW_PINK,DRAW_CORAL,
     *        DRAW_GRAY,DRAW_YELLOW,DRAW_GOLD,DRAW_LIGHTPINK,
     *        DRAW_MEDIUMTURQUOISE,DRAW_KHAKI,DRAW_DIMGRAY,
     *        DRAW_YELLOWGREEN,DRAW_SKYBLUE,DRAW_DARKGREEN,
     *        DRAW_NAVYBLUE,DRAW_SANDYBROWN,DRAW_CADETBLUE,
     *        DRAW_POWDERBLUE,DRAW_DEEPPINK,DRAW_THISTLE,DRAW_LIMEGREEN,
     *        DRAW_LAVENDERBLUSH


      parameter (DRAW_WHITE = 0, DRAW_BLACK = 1, DRAW_RED = 2,
     *           DRAW_GREEN = 3, DRAW_CYAN = 4, DRAW_BLUE = 5,
     *           DRAW_MAGENTA = 6, DRAW_AQUAMARINE = 7,
     *           DRAW_FORESTGREEN = 8, DRAW_ORANGE = 9,
     *           DRAW_VIOLET = 10, DRAW_BROWN = 11, DRAW_PINK = 12,
     *           DRAW_CORAL = 13, DRAW_GRAY = 14, DRAW_YELLOW = 15,
     *           DRAW_GOLD = 16, DRAW_LIGHTPINK = 17,
     *           DRAW_MEDIUMTURQUOISE = 18, DRAW_KHAKI = 19,
     *           DRAW_DIMGRAY = 20, DRAW_YELLOWGREEN = 21,
     *           DRAW_SKYBLUE = 22, DRAW_DARKGREEN = 23,
     *           DRAW_NAVYBLUE = 24, DRAW_SANDYBROWN = 25,
     *           DRAW_CADETBLUE = 26, DRAW_POWDERBLUE = 27,
     *           DRAW_DEEPPINK = 28, DRAW_THISTLE = 29,
     *           DRAW_LIMEGREEN = 30, DRAW_LAVENDERBLUSH = 31)


      integer BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT

      parameter (BUTTON_NONE = 0, BUTTON_LEFT = 1, 
     *           BUTTON_CENTER = 2, BUTTON_RIGHT = 3)

!
!  End of Fortran include file for the Draw package in PETSc





