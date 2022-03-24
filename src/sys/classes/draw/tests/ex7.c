
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

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,0,"Draw Example",PETSC_DECIDE,PETSC_DECIDE,101,101,&draw));
  /*CHKERRQ(PetscDrawSetPause(draw,2.0));*/
  CHKERRQ(PetscDrawSetFromOptions(draw));

  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawGetWindowSize(draw,&w,&h));
  CHKERRQ(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  CHKERRQ(PetscDrawClear(draw));
  /* one-pixel lines in the window corners */
  CHKERRQ(PetscDrawLine(draw,0,0,0,0,r));
  CHKERRQ(PetscDrawLine(draw,w,0,w,0,r));
  CHKERRQ(PetscDrawLine(draw,0,h,0,h,r));
  CHKERRQ(PetscDrawLine(draw,w,h,w,h,r));
  /* border lines with two pixels from  borders */
  CHKERRQ(PetscDrawLine(draw,0+2,0,w-2,0,k));
  CHKERRQ(PetscDrawLine(draw,0+2,h,w-2,h,k));
  CHKERRQ(PetscDrawLine(draw,0,0+2,0,h-2,k));
  CHKERRQ(PetscDrawLine(draw,w,0+2,w,h-2,k));
  /* oblique lines */
  CHKERRQ(PetscDrawLine(draw,0+2,h/2,w-2,h-2,b));
  CHKERRQ(PetscDrawLine(draw,0+1,h-1,w-1,0+1,b));
  /* vertical up and down arrow, two pixels from borders  */
  CHKERRQ(PetscDrawArrow(draw,1*w/4,0+2,1*w/4,h-2,g));
  CHKERRQ(PetscDrawArrow(draw,3*w/4,h-2,3*w/4,0+2,g));
  /* horizontal right and left arrow, two pixels from borders  */
  CHKERRQ(PetscDrawArrow(draw,0+2,3*h/4,w-2,3*h/4,g));
  CHKERRQ(PetscDrawArrow(draw,w-2,1*h/4,0+2,1*h/4,g));
  /* flush, save, and pause */
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawSave(draw));
  CHKERRQ(PetscDrawPause(draw));

  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawGetWindowSize(draw,&w,&h));
  CHKERRQ(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  CHKERRQ(PetscDrawClear(draw));
  /* one-pixel rectangles in the window corners */
  CHKERRQ(PetscDrawRectangle(draw,0,0,0,0,k,k,k,k));
  CHKERRQ(PetscDrawRectangle(draw,w,0,w,0,k,k,k,k));
  CHKERRQ(PetscDrawRectangle(draw,0,h,0,h,k,k,k,k));
  CHKERRQ(PetscDrawRectangle(draw,w,h,w,h,k,k,k,k));
  /* border rectangles with two pixels from  borders */
  CHKERRQ(PetscDrawRectangle(draw,0+2,0,w-2,0,k,k,k,k));
  CHKERRQ(PetscDrawRectangle(draw,0+2,h,w-2,h,k,k,k,k));
  CHKERRQ(PetscDrawRectangle(draw,0,0+2,0,h-2,k,k,k,k));
  CHKERRQ(PetscDrawRectangle(draw,w,0+2,w,h-2,k,k,k,k));
  /* more rectangles */
  CHKERRQ(PetscDrawRectangle(draw,0+2,0+2,w/2-1,h/2-1,b,b,b,b));
  CHKERRQ(PetscDrawRectangle(draw,0+2,h/2+1,w/2-1,h-2,r,r,r,r));
  CHKERRQ(PetscDrawRectangle(draw,w/2+1,h/2+1,w-2,h-2,g,g,g,g));
  CHKERRQ(PetscDrawRectangle(draw,w/2+1,0+2,w-2,h/2-1,y,y,y,y));
  /* flush, save, and pause */
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawSave(draw));
  CHKERRQ(PetscDrawPause(draw));

  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawGetWindowSize(draw,&w,&h));
  CHKERRQ(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  CHKERRQ(PetscDrawClear(draw));
  /* interpolated triangles, one pixel from borders */
  CHKERRQ(PetscDrawTriangle(draw,0+1,0+1,w-1,0+1,w-1,h-1,c0,c1,c2));
  CHKERRQ(PetscDrawTriangle(draw,0+1,0+1,0+1,h-1,w-1,h-1,c0,c1,c2));
  /* interpolated triangle, oblique, inside canvas */
  CHKERRQ(PetscDrawTriangle(draw,w/4,h/4,w/2,3*h/4,3*w/4,h/2,c2,c1,c0));
  /* flush, save, and pause */
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawSave(draw));
  CHKERRQ(PetscDrawPause(draw));

  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawGetWindowSize(draw,&w,&h));
  CHKERRQ(PetscDrawSetCoordinates(draw,0,0,--w,--h));
  CHKERRQ(PetscDrawClear(draw));
  /* circles and ellipses */
  CHKERRQ(PetscDrawEllipse(draw,w/2,h/2,w-1,h-1,r));
  CHKERRQ(PetscDrawEllipse(draw,w,h/2,w/2,h,g));
  CHKERRQ(PetscDrawEllipse(draw,0,0,w,h/2,b));
  CHKERRQ(PetscDrawEllipse(draw,w/4,3*h/4,w/2,h/4,y));
  CHKERRQ(PetscDrawCoordinateToPixel(draw,w/2,h/2,&i,&j));
  CHKERRQ(PetscDrawPointPixel(draw,i,j,k));
  /* flush, save, and pause */
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawSave(draw));
  CHKERRQ(PetscDrawPause(draw));

  CHKERRQ(PetscDrawDestroy(&draw));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
