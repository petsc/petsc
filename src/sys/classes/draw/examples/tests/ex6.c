static char help[] = "Demonstrates named colormaps\n";

#include <petscsys.h>
#include <petscdraw.h>

#define Exp PetscExpReal
#define Pow PetscPowReal

extern PetscErrorCode PetscDrawSetColormap(PetscDraw,const char[]);

static PetscReal Peaks(PetscReal x,PetscReal y)
{
  return 3 * Pow(1-x,2) * Exp(-Pow(x,2) - Pow(y+1,2))
       - 10 * (x/5 - Pow(x,3) - Pow(y,5)) * Exp(-Pow(x,2) - Pow(y,2))
       - 1./3 * Exp(-Pow(x+1,2) - Pow(y,2));
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  char           title[64],cmap[32] = "";
  PetscDraw      draw,popup;
  PetscMPIInt    size,rank;
  int            i,j, w = 600, h = 600;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-draw_cmap",cmap,sizeof(cmap),NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(title,sizeof(title),"Colormap: %s",cmap);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,NULL,title,PETSC_DECIDE,PETSC_DECIDE,w,h,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  for (j=rank; j<h; j+=size) {
    for (i=0; i<w; i++) {
      PetscReal u = (PetscReal)i/(w-1), x = 6*u-3;
      PetscReal v = (PetscReal)j/(h-1), y = 6*(1-v)-3;
      PetscReal f = Peaks(x,y);
      int color = PetscDrawRealToColor(f,-8,+8);
      ierr = PetscDrawPointPixel(draw,i,j,color);CHKERRQ(ierr);
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
  if (popup) {ierr = PetscDrawScalePopup(popup,-8,+8);CHKERRQ(ierr);}
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
