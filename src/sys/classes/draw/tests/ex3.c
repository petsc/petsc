
static char help[] = "Plots a simple line graph.\n";

#if defined(PETSC_APPLE_FRAMEWORK)
  #import <PETSc/petscsys.h>
  #import <PETSc/petscdraw.h>
#else

  #include <petscsys.h>
  #include <petscdraw.h>
#endif

int main(int argc, char **argv)
{
  PetscDraw           draw;
  PetscDrawLG         lg;
  PetscDrawAxis       axis;
  PetscInt            n = 15, i, x = 0, y = 0, width = 400, height = 300, nports = 1;
  PetscBool           useports, flg;
  const char         *xlabel, *ylabel, *toplabel, *legend;
  PetscReal           xd, yd;
  PetscDrawViewPorts *ports = NULL;

  toplabel = "Top Label";
  xlabel   = "X-axis Label";
  ylabel   = "Y-axis Label";
  legend   = "Legend";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-x", &x, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-y", &y, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-width", &width, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-height", &height, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nports", &nports, &useports));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nolegend", &flg));
  if (flg) legend = NULL;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-notoplabel", &flg));
  if (flg) toplabel = NULL;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-noxlabel", &flg));
  if (flg) xlabel = NULL;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-noylabel", &flg));
  if (flg) ylabel = NULL;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nolabels", &flg));
  if (flg) {
    toplabel = NULL;
    xlabel   = NULL;
    ylabel   = NULL;
  }

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, 0, "Title", x, y, width, height, &draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  if (useports) {
    PetscCall(PetscDrawViewPortsCreate(draw, nports, &ports));
    PetscCall(PetscDrawViewPortsSet(ports, 0));
  }
  PetscCall(PetscDrawLGCreate(draw, 1, &lg));
  PetscCall(PetscDrawLGSetUseMarkers(lg, PETSC_TRUE));
  PetscCall(PetscDrawLGGetAxis(lg, &axis));
  PetscCall(PetscDrawAxisSetColors(axis, PETSC_DRAW_BLACK, PETSC_DRAW_RED, PETSC_DRAW_BLUE));
  PetscCall(PetscDrawAxisSetLabels(axis, toplabel, xlabel, ylabel));
  PetscCall(PetscDrawLGSetLegend(lg, &legend));
  PetscCall(PetscDrawLGSetFromOptions(lg));

  for (i = 0; i <= n; i++) {
    xd = (PetscReal)(i - 5);
    yd = xd * xd;
    PetscCall(PetscDrawLGAddPoint(lg, &xd, &yd));
  }
  PetscCall(PetscDrawLGDraw(lg));
  PetscCall(PetscDrawLGSave(lg));

  PetscCall(PetscDrawViewPortsDestroy(ports));
  PetscCall(PetscDrawLGDestroy(&lg));
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
