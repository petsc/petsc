C
C  $Id: draw.h,v 1.7 1996/02/12 20:31:49 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the Draw package in PETSc
C
#define Draw       integer
#define DrawLG     integer
#define DrawAxis   integer
#define DrawButton integer
C
C  Colors for drawing
C
      integer DRAW_WHITE,DRAW_BLACK,DRAW_RED,DRAW_YELLOW,DRAW_GREEN,
     *        DRAW_CYAN,DRAW_BLUE

      parameter (DRAW_WHITE = 0,DRAW_BLACK = 1,DRAW_RED = 2,
     *           DRAW_YELLOW = 3,DRAW_GREEN = 4,
     *           DRAW_CYAN = 5,DRAW_BLUE= 6)

      integer BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT

      parameter (BUTTON_NONE = 0, BUTTON_LEFT = 1, 
     *           BUTTON_CENTER = 2, BUTTON_RIGHT = 3)

C
C  End of Fortran include file for the Draw package in PETSc

