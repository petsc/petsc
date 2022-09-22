
static char help[] = "Makes a simple histogram.\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc, char **argv)
{
  PetscDraw     draw;
  PetscDrawHG   hist;
  PetscDrawAxis axis;
  int           n = 20, i, x = 0, y = 0, width = 400, height = 300, bins = 8;
  PetscInt      w = 400, h = 300, nn = 20, b = 8, c = PETSC_DRAW_GREEN;
  int           color = PETSC_DRAW_GREEN;
  const char   *xlabel, *ylabel, *toplabel;
  PetscReal     xd;
  PetscBool     flg;

  xlabel   = "X-axis Label";
  toplabel = "Top Label";
  ylabel   = "Y-axis Label";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-width", &w, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-height", &h, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &nn, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bins", &b, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-color", &c, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nolabels", &flg));
  width  = (int)w;
  height = (int)h;
  n      = (int)nn;
  bins   = (int)b;
  color  = (int)c;
  if (flg) {
    xlabel   = NULL;
    ylabel   = NULL;
    toplabel = NULL;
  }

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, 0, "Title", x, y, width, height, &draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawHGCreate(draw, bins, &hist));
  PetscCall(PetscDrawHGSetColor(hist, color));
  PetscCall(PetscDrawHGGetAxis(hist, &axis));
  PetscCall(PetscDrawAxisSetColors(axis, PETSC_DRAW_BLACK, PETSC_DRAW_RED, PETSC_DRAW_BLUE));
  PetscCall(PetscDrawAxisSetLabels(axis, toplabel, xlabel, ylabel));
  /* PetscCall(PetscDrawHGSetFromOptions(hist)); */

  for (i = 0; i < n; i++) {
    xd = (PetscReal)(i - 5);
    PetscCall(PetscDrawHGAddValue(hist, xd * xd));
  }
  PetscCall(PetscDrawHGDraw(hist));
  PetscCall(PetscDrawHGSave(hist));

  PetscCall(PetscDrawHGDestroy(&hist));
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
