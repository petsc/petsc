C
C  Include file for Fortran use of the Draw package in PETSc
C
#define Draw     integer
#define DrawLG   integer
#define DrawAxis integer
C
C  Colors for drawing
C
      integer draw_white,draw_black,draw_red,draw_yellow,draw_green,
     *        draw_cyan,draw_blue

      parameter (draw_white = 0,draw_black = 1,draw_red = 2,
     *           draw_yellow = 3,draw_green = 4,
     *           draw_cyan = 5,draw_blue = 6)
C
C  End of Fortran include file for the Draw package in PETSc

