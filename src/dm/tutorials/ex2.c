static char help[] = "Demonstrates Conway's Game of Life using a 2d DMDA.\n\n";

/*
 At each step in time, the following transitions occur:

    Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    Any live cell with two or three live neighbours lives on to the next generation.
    Any live cell with more than three live neighbours dies, as if by overpopulation.
    Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

 https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
*/

#include <petscdm.h>
#include <petscdmda.h>

static const int GLIDER[3][3] = {
  {0, 1, 0},
  {0, 1, 1},
  {1, 0, 1}
};

int main(int argc,char **argv)
{
  PetscErrorCode   ierr;
  DM               da;
  PetscViewer      viewer;
  Vec              Xlocal, Xglobal;
  PetscInt         glider_loc[2] = {10, 20}, blinker_loc[2] = {20, 10}, two, steps = 100, viz_interval = 1;
  PetscInt         check_step_alive = -1, check_step_dead = -1;
  PetscBool        has_glider, has_blinker;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Conway's Game of Life","");CHKERRQ(ierr);
  {
    CHKERRQ(PetscOptionsIntArray("-glider","Coordinate at which to center a glider",NULL,glider_loc,(two=2,&two),&has_glider));
    CHKERRQ(PetscOptionsIntArray("-blinker","Coordinate at which to center a blinker",NULL,blinker_loc,(two=2,&two),&has_blinker));
    CHKERRQ(PetscOptionsInt("-steps","Number of steps to take",NULL,steps,&steps,NULL));
    CHKERRQ(PetscOptionsInt("-viz_interval","Vizualization interval",NULL,viz_interval,&viz_interval,NULL));
    CHKERRQ(PetscOptionsInt("-check_step_alive","Step on which to check that the simulation is alive",NULL,check_step_alive,&check_step_alive,NULL));
    CHKERRQ(PetscOptionsInt("-check_step_dead","Step on which to check that the simulation is dead",NULL,check_step_dead,&check_step_dead,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"Life",PETSC_DECIDE,PETSC_DECIDE,1000,1000,&viewer));

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,30,30,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateLocalVector(da,&Xlocal));
  CHKERRQ(DMCreateGlobalVector(da,&Xglobal));

  {  /* Initialize */
    DMDALocalInfo info;
    PetscScalar   **x;
    PetscInt      i,j;

    CHKERRQ(DMDAGetLocalInfo(da,&info));
    CHKERRQ(DMDAVecGetArray(da, Xlocal, &x));
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        if (has_glider && i == glider_loc[0] && j == glider_loc[1]) {
          PetscInt ii,jj;
          for (ii=-1; ii<=1; ii++)
            for (jj=-1; jj<=1; jj++)
              x[j+jj][i+ii] = GLIDER[1-jj][ii+1];
        }
        if (has_blinker && i == blinker_loc[0] && j == blinker_loc[1]) {
          x[j-1][i] = 1;
          x[j][i]   = 1;
          x[j+1][i] = 1;
        }
      }
    }
    CHKERRQ(DMDAVecRestoreArray(da, Xlocal, &x));
    CHKERRQ(DMLocalToGlobal(da, Xlocal, ADD_VALUES, Xglobal));
  }

  /* View the initial condition */
  CHKERRQ(VecView(Xglobal, viewer));

  {                             /* Play */
    PetscInt step;

    for (step=0; step<steps; step++) {
      const PetscScalar **x;
      PetscScalar       **y;
      DMDALocalInfo     info;
      PetscInt          i,j;

      CHKERRQ(DMGlobalToLocal(da, Xglobal, INSERT_VALUES, Xlocal));
      CHKERRQ(DMDAGetLocalInfo(da,&info));
      CHKERRQ(DMDAVecGetArrayRead(da, Xlocal, &x));
      CHKERRQ(DMDAVecGetArrayWrite(da, Xglobal, &y));
      for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
          PetscInt live_neighbors = 0;
          live_neighbors += PetscRealPart(x[j-1][i-1]) > 0;
          live_neighbors += PetscRealPart(x[j-1][i]) > 0;
          live_neighbors += PetscRealPart(x[j-1][i+1]) > 0;
          live_neighbors += PetscRealPart(x[j][i-1]) > 0;
          live_neighbors += PetscRealPart(x[j][i+1]) > 0;
          live_neighbors += PetscRealPart(x[j+1][i-1]) > 0;
          live_neighbors += PetscRealPart(x[j+1][i]) > 0;
          live_neighbors += PetscRealPart(x[j+1][i+1]) > 0;
          if (PetscRealPart(x[j][i]) > 0) {    /* Live cell */
            switch (live_neighbors) {
            case 2:
            case 3:
              y[j][i] = 1;      /* Survive */
              break;
            default:
              y[j][i] = 0;      /* Death */
            }
          } else {                                /* Dead cell */
            if (live_neighbors == 3) y[j][i] = 1; /* Birth */
            else y[j][i] = 0;
          }
        }
      }
      CHKERRQ(DMDAVecRestoreArrayRead(da, Xlocal, &x));
      CHKERRQ(DMDAVecRestoreArrayWrite(da, Xglobal, &y));
      if (step == check_step_alive || step == check_step_dead) {
        PetscScalar sum;
        CHKERRQ(VecSum(Xglobal, &sum));
        if (PetscAbsScalar(sum) > 0.1) {
          if (step == check_step_dead) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Simulation alive at step %D\n",step));
          }
        } else if (step == check_step_alive) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Simulation dead at step %D\n",step));
        }
      }
      if (step % viz_interval == 0) {
        CHKERRQ(VecView(Xglobal, viewer));
      }
    }
  }

  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&Xglobal));
  CHKERRQ(VecDestroy(&Xlocal));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: x
      nsize: 2
      args: -glider 5,6 -blinker 12,12 -steps 35 -check_step_alive 31 -check_step_dead 32 -da_grid_x 20 -da_grid_y 20 -nox

TEST*/
