C
C  $Id: draw.h,v 1.10 1996/04/16 03:50:07 balay Exp bsmith $;
C
C  Include file for Fortran use of the Draw package in PETSc
C
#define Draw       integer
#define DrawLG     integer
#define DrawAxis   integer
#define DrawButton integer

C
C  types of draw context
C
      integer DRAW_XWINDOW, DRAW_NULLWINDOW
      parameter (DRAW_XWINDOW=0, DRAW_NULLWINDOW=1)
C
C  Colors for drawing
C
      integer DRAW_WHITE,DRAW_BLACK,DRAW_RED,DRAW_GREEN,DRAW_CYAN,
     *        DRAW_BLUE,DRAW_MAGENTA,DRAW_AQUAMARINE,DRAW_FORESTGREEN,
     *        DRAW_ORANGE,DRAW_VIOLET,DRAW_BROWN,DRAW_PINK,DRAW_CORAL,
     *        DRAW_GRAY,DRAW_YELLOW

      parameter (DRAW_WHITE = 0,DRAW_BLACK = 1,DRAW_RED = 2,
     *           DRAW_GREEN = 3,DRAW_CYAN = 4,DRAW_BLUE = 5,
     *           DRAW_MAGENTA = 6,DRAW_AQUAMARINE = 7,
     *           DRAW_FORESTGREEN = 8,DRAW_ORANGE = 9,
     *           DRAW_VIOLET = 10,DRAW_BROWN = 11,DRAW_PINK = 12,
     *           DRAW_CORAL =13,DRAW_GRAY = 14,DRAW_YELLOW = 15)


      integer BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT

      parameter (BUTTON_NONE = 0, BUTTON_LEFT = 1, 
     *           BUTTON_CENTER = 2, BUTTON_RIGHT = 3)

C
C  End of Fortran include file for the Draw package in PETSc





