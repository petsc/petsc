
static char help[] = "Demonstrates drawing primitives in a window\n";

#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  int i,j,w,h;
  int k  = PETSC_DRAW_BLACK;
  int r  = PETSC_DRAW_RED;
  int g  = PETSC_DRAW_GREEN;
  int b  = PETSC_DRAW_BLUE;
  int y  = PETSC_DRAW_YELLOW;
  int c0 = PETSC_DRAW_BASIC_COLORS;
  int c2 = 255;
  int c1 = (c0+c2)/2;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,0,"Draw Example",PETSC_DECIDE,PETSC_DECIDE,101,101,&draw);CHKERRQ(ierr);
  /*ierr = PetscDrawSetPause(draw,2.0);CHKERRQ(ierr);*/
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawGetWindowSize(draw,&w,&h);CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,0,0,--w,--h);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  /* one-pixel lines in the window corners */
  ierr = PetscDrawLine(draw,0,0,0,0,r);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,w,0,w,0,r);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,0,h,0,h,r);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,w,h,w,h,r);CHKERRQ(ierr);
  /* border lines with two pixels from  borders */
  ierr = PetscDrawLine(draw,0+2,0,w-2,0,k);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,0+2,h,w-2,h,k);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,0,0+2,0,h-2,k);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,w,0+2,w,h-2,k);CHKERRQ(ierr);
  /* oblique lines */
  ierr = PetscDrawLine(draw,0+2,h/2,w-2,h-2,b);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,0+1,h-1,w-1,0+1,b);CHKERRQ(ierr);
  /* vertical up and down arrow, two pixels from borders  */
  ierr = PetscDrawArrow(draw,1*w/4,0+2,1*w/4,h-2,g);CHKERRQ(ierr);
  ierr = PetscDrawArrow(draw,3*w/4,h-2,3*w/4,0+2,g);CHKERRQ(ierr);
  /* horizontal right and left arrow, two pixels from borders  */
  ierr = PetscDrawArrow(draw,0+2,3*h/4,w-2,3*h/4,g);CHKERRQ(ierr);
  ierr = PetscDrawArrow(draw,w-2,1*h/4,0+2,1*h/4,g);CHKERRQ(ierr);
  /* flush, save, and pause */
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawGetWindowSize(draw,&w,&h);CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,0,0,--w,--h);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  /* one-pixel rectangles in the window corners */
  ierr = PetscDrawRectangle(draw,0,0,0,0,k,k,k,k);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,w,0,w,0,k,k,k,k);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,0,h,0,h,k,k,k,k);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,w,h,w,h,k,k,k,k);CHKERRQ(ierr);
  /* border rectangles with two pixels from  borders */
  ierr = PetscDrawRectangle(draw,0+2,0,w-2,0,k,k,k,k);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,0+2,h,w-2,h,k,k,k,k);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,0,0+2,0,h-2,k,k,k,k);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,w,0+2,w,h-2,k,k,k,k);CHKERRQ(ierr);
  /* more rectangles */
  ierr = PetscDrawRectangle(draw,0+2,0+2,w/2-1,h/2-1,b,b,b,b);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,0+2,h/2+1,w/2-1,h-2,r,r,r,r);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,w/2+1,h/2+1,w-2,h-2,g,g,g,g);CHKERRQ(ierr);
  ierr = PetscDrawRectangle(draw,w/2+1,0+2,w-2,h/2-1,y,y,y,y);CHKERRQ(ierr);
  /* flush, save, and pause */
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawGetWindowSize(draw,&w,&h);CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,0,0,--w,--h);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  /* interpolated triangles, one pixel from borders */
  ierr = PetscDrawTriangle(draw,0+1,0+1,w-1,0+1,w-1,h-1,c0,c1,c2);CHKERRQ(ierr);
  ierr = PetscDrawTriangle(draw,0+1,0+1,0+1,h-1,w-1,h-1,c0,c1,c2);CHKERRQ(ierr);
  /* interpolated triangle, oblique, inside canvas */
  ierr = PetscDrawTriangle(draw,w/4,h/4,w/2,3*h/4,3*w/4,h/2,c2,c1,c0);CHKERRQ(ierr);
  /* flush, save, and pause */
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawGetWindowSize(draw,&w,&h);CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,0,0,--w,--h);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  /* circles and ellipses */
  ierr = PetscDrawEllipse(draw,w/2,h/2,w-1,h-1,r);CHKERRQ(ierr);
  ierr = PetscDrawEllipse(draw,w,h/2,w/2,h,g);CHKERRQ(ierr);
  ierr = PetscDrawEllipse(draw,0,0,w,h/2,b);CHKERRQ(ierr);
  ierr = PetscDrawEllipse(draw,w/4,3*h/4,w/2,h/4,y);CHKERRQ(ierr);
  ierr = PetscDrawCoordinateToPixel(draw,w/2,h/2,&i,&j);CHKERRQ(ierr);
  ierr = PetscDrawPointPixel(draw,i,j,k);CHKERRQ(ierr);
  /* flush, save, and pause */
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
