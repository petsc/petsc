
static char help[] = "Demonstrates drawing primitives in a window\n";

#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw draw;

  int i,j,w,h;
  int k  = PETSC_DRAW_BLACK;
  int r  = PETSC_DRAW_RED;
  int g  = PETSC_DRAW_GREEN;
  int b  = PETSC_DRAW_BLUE;
  int y  = PETSC_DRAW_YELLOW;
  int c0 = PETSC_DRAW_BASIC_COLORS;
  int c2 = 255;
  int c1 = (c0+c2)/2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,0,"Draw Example",PETSC_DECIDE,PETSC_DECIDE,101,101,&draw));
  /*PetscCall(PetscDrawSetPause(draw,2.0));*/
  PetscCall(PetscDrawSetFromOptions(draw));

  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawGetWindowSize(draw,&w,&h));
  PetscCall(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  PetscCall(PetscDrawClear(draw));
  /* one-pixel lines in the window corners */
  PetscCall(PetscDrawLine(draw,0,0,0,0,r));
  PetscCall(PetscDrawLine(draw,w,0,w,0,r));
  PetscCall(PetscDrawLine(draw,0,h,0,h,r));
  PetscCall(PetscDrawLine(draw,w,h,w,h,r));
  /* border lines with two pixels from  borders */
  PetscCall(PetscDrawLine(draw,0+2,0,w-2,0,k));
  PetscCall(PetscDrawLine(draw,0+2,h,w-2,h,k));
  PetscCall(PetscDrawLine(draw,0,0+2,0,h-2,k));
  PetscCall(PetscDrawLine(draw,w,0+2,w,h-2,k));
  /* oblique lines */
  PetscCall(PetscDrawLine(draw,0+2,h/2,w-2,h-2,b));
  PetscCall(PetscDrawLine(draw,0+1,h-1,w-1,0+1,b));
  /* vertical up and down arrow, two pixels from borders  */
  PetscCall(PetscDrawArrow(draw,1*w/4,0+2,1*w/4,h-2,g));
  PetscCall(PetscDrawArrow(draw,3*w/4,h-2,3*w/4,0+2,g));
  /* horizontal right and left arrow, two pixels from borders  */
  PetscCall(PetscDrawArrow(draw,0+2,3*h/4,w-2,3*h/4,g));
  PetscCall(PetscDrawArrow(draw,w-2,1*h/4,0+2,1*h/4,g));
  /* flush, save, and pause */
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawSave(draw));
  PetscCall(PetscDrawPause(draw));

  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawGetWindowSize(draw,&w,&h));
  PetscCall(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  PetscCall(PetscDrawClear(draw));
  /* one-pixel rectangles in the window corners */
  PetscCall(PetscDrawRectangle(draw,0,0,0,0,k,k,k,k));
  PetscCall(PetscDrawRectangle(draw,w,0,w,0,k,k,k,k));
  PetscCall(PetscDrawRectangle(draw,0,h,0,h,k,k,k,k));
  PetscCall(PetscDrawRectangle(draw,w,h,w,h,k,k,k,k));
  /* border rectangles with two pixels from  borders */
  PetscCall(PetscDrawRectangle(draw,0+2,0,w-2,0,k,k,k,k));
  PetscCall(PetscDrawRectangle(draw,0+2,h,w-2,h,k,k,k,k));
  PetscCall(PetscDrawRectangle(draw,0,0+2,0,h-2,k,k,k,k));
  PetscCall(PetscDrawRectangle(draw,w,0+2,w,h-2,k,k,k,k));
  /* more rectangles */
  PetscCall(PetscDrawRectangle(draw,0+2,0+2,w/2-1,h/2-1,b,b,b,b));
  PetscCall(PetscDrawRectangle(draw,0+2,h/2+1,w/2-1,h-2,r,r,r,r));
  PetscCall(PetscDrawRectangle(draw,w/2+1,h/2+1,w-2,h-2,g,g,g,g));
  PetscCall(PetscDrawRectangle(draw,w/2+1,0+2,w-2,h/2-1,y,y,y,y));
  /* flush, save, and pause */
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawSave(draw));
  PetscCall(PetscDrawPause(draw));

  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawGetWindowSize(draw,&w,&h));
  PetscCall(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  PetscCall(PetscDrawClear(draw));
  /* interpolated triangles, one pixel from borders */
  PetscCall(PetscDrawTriangle(draw,0+1,0+1,w-1,0+1,w-1,h-1,c0,c1,c2));
  PetscCall(PetscDrawTriangle(draw,0+1,0+1,0+1,h-1,w-1,h-1,c0,c1,c2));
  /* interpolated triangle, oblique, inside canvas */
  PetscCall(PetscDrawTriangle(draw,w/4,h/4,w/2,3*h/4,3*w/4,h/2,c2,c1,c0));
  /* flush, save, and pause */
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawSave(draw));
  PetscCall(PetscDrawPause(draw));

  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawGetWindowSize(draw,&w,&h));
  PetscCall(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  PetscCall(PetscDrawClear(draw));
  /* circles and ellipses */
  PetscCall(PetscDrawEllipse(draw,w/2,h/2,w-1,h-1,r));
  PetscCall(PetscDrawEllipse(draw,w,h/2,w/2,h,g));
  PetscCall(PetscDrawEllipse(draw,0,0,w,h/2,b));
  PetscCall(PetscDrawEllipse(draw,w/4,3*h/4,w/2,h/4,y));
  PetscCall(PetscDrawCoordinateToPixel(draw,w/2,h/2,&i,&j));
  PetscCall(PetscDrawPointPixel(draw,i,j,k));
  /* flush, save, and pause */
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawSave(draw));
  PetscCall(PetscDrawPause(draw));

  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
