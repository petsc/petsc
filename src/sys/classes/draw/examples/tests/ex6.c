static char help[] = "Demonstrates named colormaps\n";

#include <petscsys.h>
#include <petscdraw.h>


typedef PetscReal (*Function)(PetscReal,PetscReal);

typedef struct {
  Function function;
} FunctionCtx;

#define Exp PetscExpReal
#define Pow PetscPowReal
static PetscReal Peaks(PetscReal x,PetscReal y)
{
  return 3 * Pow(1-x,2) * Exp(-Pow(x,2) - Pow(y+1,2))
       - 10 * (x/5 - Pow(x,3) - Pow(y,5)) * Exp(-Pow(x,2) - Pow(y,2))
       - 1./3 * Exp(-Pow(x+1,2) - Pow(y,2));
}

static PetscErrorCode DrawFunction(PetscDraw draw,void *ctx)
{
  int            i,j,w,h;
  Function       function = ((FunctionCtx*)ctx)->function;
  PetscReal      min = PETSC_MAX_REAL, max = PETSC_MIN_REAL;
  MPI_Comm       comm = PetscObjectComm((PetscObject)draw);
  PetscMPIInt    size,rank;
  PetscDraw      popup;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawGetWindowSize(draw,&w,&h);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  for (j=rank; j<h; j+=size) {
    for (i=0; i<w; i++) {
      PetscReal x,y,f; int color;
      ierr = PetscDrawPixelToCoordinate(draw,i,j,&x,&y);CHKERRQ(ierr);
      f = function(x,y); color = PetscDrawRealToColor(f,-8,+8);
      ierr = PetscDrawPointPixel(draw,i,j,color);CHKERRQ(ierr);
      min = PetscMin(f,min); max = PetscMax(f,max);
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
  ierr = PetscDrawScalePopup(popup,-8,+8);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  char           title[64],cmap[32] = "";
  PetscDraw      draw;
  FunctionCtx    ctx;
  PetscErrorCode ierr;

  ctx.function = Peaks;
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-draw_cmap",cmap,sizeof(cmap),NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(title,sizeof(title),"Colormap: %s",cmap);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,NULL,title,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&draw);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)draw,"Peaks");CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,-3,-3,+3,+3);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,DrawFunction,&ctx);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);

  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

      build:
         requires: x

      test:
         args: -draw_cmap hue
         output_file: output/ex1_1.out

      test:
         suffix: 2
         args: -draw_cmap gray
         output_file: output/ex1_1.out

      test:
         suffix: 3
         args: -draw_cmap bone
         output_file: output/ex1_1.out

      test:
         suffix: 4
         args: -draw_cmap jet
         output_file: output/ex1_1.out

      test:
         suffix: 5
         args: -draw_cmap coolwarm
         output_file: output/ex1_1.out

      test:
         suffix: 6
         args: -draw_cmap parula
         output_file: output/ex1_1.out

      test:
         suffix: 7
         args: -draw_cmap viridis
         output_file: output/ex1_1.out

      test:
         suffix: 8
         args: -draw_cmap plasma
         output_file: output/ex1_1.out

TEST*/
