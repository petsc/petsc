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
  PetscCall(PetscDrawGetWindowSize(draw,&w,&h));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  ierr = PetscDrawCollectiveBegin(draw);PetscCall(ierr);
  for (j=rank; j<h; j+=size) {
    for (i=0; i<w; i++) {
      PetscReal x,y,f; int color;
      PetscCall(PetscDrawPixelToCoordinate(draw,i,j,&x,&y));
      f = function(x,y); color = PetscDrawRealToColor(f,-8,+8);
      PetscCall(PetscDrawPointPixel(draw,i,j,color));
      min = PetscMin(f,min); max = PetscMax(f,max);
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);PetscCall(ierr);

  PetscCall(PetscDrawGetPopup(draw,&popup));
  PetscCall(PetscDrawScalePopup(popup,-8,+8));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  char        title[64],cmap[32] = "";
  PetscDraw   draw;
  FunctionCtx ctx;

  ctx.function = Peaks;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-draw_cmap",cmap,sizeof(cmap),NULL));
  PetscCall(PetscSNPrintf(title,sizeof(title),"Colormap: %s",cmap));

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,NULL,title,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&draw));
  PetscCall(PetscObjectSetName((PetscObject)draw,"Peaks"));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawSetCoordinates(draw,-3,-3,+3,+3));
  PetscCall(PetscDrawZoom(draw,DrawFunction,&ctx));
  PetscCall(PetscDrawSave(draw));

  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
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
