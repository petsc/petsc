static char help[] = "Test DMStag ghosted boundaries in 3d\n\n";
/* This solves a very contrived problem - the "pressure" terms are set to a constant function
   and the "velocity" terms are just the sum of neighboring values of these, hence twice the
   constant */
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>

#define PRESSURE_CONST 2.0

PetscErrorCode ApplyOperator(Mat, Vec, Vec);

int main(int argc, char **argv)
{
  DM              dmSol;
  Vec             sol, solRef, solRefLocal, rhs, rhsLocal;
  Mat             A;
  KSP             ksp;
  PC              pc;
  PetscInt        startx, starty, startz, nx, ny, nz, ex, ey, ez, nExtrax, nExtray, nExtraz;
  PetscInt        iux, iuy, iuz, ip;
  PetscInt        dof0, dof1, dof2, dof3;
  PetscScalar ****arrSol, ****arrRHS;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  /* Note: these defaults are chosen to suit the problem. We allow adjusting
     them to check that things still work when you add unused extra dof */
  dof0 = 0;
  dof1 = 0;
  dof2 = 1;
  dof3 = 1;
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, 3, 3, 3, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2, dof3, DMSTAG_STENCIL_BOX, 1, NULL, NULL, NULL, &dmSol));
  PetscCall(DMSetFromOptions(dmSol));
  PetscCall(DMSetUp(dmSol));

  /* Compute reference solution on the grid, using direct array access */
  PetscCall(DMCreateGlobalVector(dmSol, &rhs));
  PetscCall(DMCreateGlobalVector(dmSol, &solRef));
  PetscCall(DMGetLocalVector(dmSol, &solRefLocal));
  PetscCall(DMGetLocalVector(dmSol, &rhsLocal));
  PetscCall(DMStagVecGetArray(dmSol, solRefLocal, &arrSol));

  PetscCall(DMStagGetCorners(dmSol, &startx, &starty, &startz, &nx, &ny, &nz, &nExtrax, &nExtray, &nExtraz));
  PetscCall(DMStagVecGetArray(dmSol, rhsLocal, &arrRHS));

  /* Get the correct entries for each of our variables in local element-wise storage */
  PetscCall(DMStagGetLocationSlot(dmSol, DMSTAG_LEFT, 0, &iux));
  PetscCall(DMStagGetLocationSlot(dmSol, DMSTAG_DOWN, 0, &iuy));
  PetscCall(DMStagGetLocationSlot(dmSol, DMSTAG_BACK, 0, &iuz));
  PetscCall(DMStagGetLocationSlot(dmSol, DMSTAG_ELEMENT, 0, &ip));
  for (ez = startz; ez < startz + nz + nExtraz; ++ez) {
    for (ey = starty; ey < starty + ny + nExtray; ++ey) {
      for (ex = startx; ex < startx + nx + nExtrax; ++ex) {
        arrSol[ez][ey][ex][iux] = 2 * PRESSURE_CONST;
        arrRHS[ez][ey][ex][iux] = 0.0;
        arrSol[ez][ey][ex][iuy] = 2 * PRESSURE_CONST;
        arrRHS[ez][ey][ex][iuy] = 0.0;
        arrSol[ez][ey][ex][iuz] = 2 * PRESSURE_CONST;
        arrRHS[ez][ey][ex][iuz] = 0.0;
        if (ex < startx + nx && ey < starty + ny && ez < startz + nz) {
          arrSol[ez][ey][ex][ip] = PRESSURE_CONST;
          arrRHS[ez][ey][ex][ip] = PRESSURE_CONST;
        }
      }
    }
  }
  PetscCall(DMStagVecRestoreArray(dmSol, rhsLocal, &arrRHS));
  PetscCall(DMLocalToGlobalBegin(dmSol, rhsLocal, INSERT_VALUES, rhs));
  PetscCall(DMLocalToGlobalEnd(dmSol, rhsLocal, INSERT_VALUES, rhs));
  PetscCall(DMStagVecRestoreArray(dmSol, solRefLocal, &arrSol));
  PetscCall(DMLocalToGlobalBegin(dmSol, solRefLocal, INSERT_VALUES, solRef));
  PetscCall(DMLocalToGlobalEnd(dmSol, solRefLocal, INSERT_VALUES, solRef));
  PetscCall(DMRestoreLocalVector(dmSol, &solRefLocal));
  PetscCall(DMRestoreLocalVector(dmSol, &rhsLocal));

  /* Matrix-free Operator */
  PetscCall(DMSetMatType(dmSol, MATSHELL));
  PetscCall(DMCreateMatrix(dmSol, &A));
  PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))ApplyOperator));

  /* Solve */
  PetscCall(DMCreateGlobalVector(dmSol, &sol));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, sol));

  /* Check Solution */
  {
    Vec       diff;
    PetscReal normsolRef, errAbs, errRel;

    PetscCall(VecDuplicate(sol, &diff));
    PetscCall(VecCopy(sol, diff));
    PetscCall(VecAXPY(diff, -1.0, solRef));
    PetscCall(VecNorm(diff, NORM_2, &errAbs));
    PetscCall(VecNorm(solRef, NORM_2, &normsolRef));
    errRel = errAbs / normsolRef;
    if (errAbs > 1e14 || errRel > 1e14) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmSol), "Error (abs): %g\nError (rel): %g\n", (double)errAbs, (double)errRel));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmSol), "Non-zero error. Probable failure.\n"));
    }
    PetscCall(VecDestroy(&diff));
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&solRef));
  PetscCall(VecDestroy(&rhs));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dmSol));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ApplyOperator(Mat A, Vec in, Vec out)
{
  DM              dm;
  Vec             inLocal, outLocal;
  PetscScalar ****arrIn;
  PetscScalar ****arrOut;
  PetscInt        startx, starty, startz, nx, ny, nz, nExtrax, nExtray, nExtraz, ex, ey, ez, idxP, idxUx, idxUy, idxUz, startGhostx, startGhosty, startGhostz, nGhostx, nGhosty, nGhostz;
  PetscBool       isFirstx, isFirsty, isFirstz, isLastx, isLasty, isLastz;

  PetscFunctionBeginUser;
  PetscCall(MatGetDM(A, &dm));
  PetscCall(DMGetLocalVector(dm, &inLocal));
  PetscCall(DMGetLocalVector(dm, &outLocal));
  PetscCall(DMGlobalToLocalBegin(dm, in, INSERT_VALUES, inLocal));
  PetscCall(DMGlobalToLocalEnd(dm, in, INSERT_VALUES, inLocal));
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, &nExtrax, &nExtray, &nExtraz));
  PetscCall(DMStagGetGhostCorners(dm, &startGhostx, &startGhosty, &startGhostz, &nGhostx, &nGhosty, &nGhostz));
  PetscCall(DMStagVecGetArrayRead(dm, inLocal, &arrIn));
  PetscCall(DMStagVecGetArray(dm, outLocal, &arrOut));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, 0, &idxUx));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, 0, &idxUy));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_BACK, 0, &idxUz));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &idxP));
  PetscCall(DMStagGetIsFirstRank(dm, &isFirstx, &isFirsty, &isFirstz));
  PetscCall(DMStagGetIsLastRank(dm, &isLastx, &isLasty, &isLastz));

  /* Set "pressures" on ghost boundaries by copying neighboring values*/
  if (isFirstx) {
    for (ez = startz; ez < startz + nz + nExtraz; ++ez) {
      for (ey = starty; ey < starty + ny + nExtray; ++ey) arrIn[ez][ey][-1][idxP] = arrIn[ez][ey][0][idxP];
    }
  }
  if (isLastx) {
    for (ez = startz; ez < startz + nz + nExtraz; ++ez) {
      for (ey = starty; ey < starty + ny + nExtray; ++ey) arrIn[ez][ey][startx + nx][idxP] = arrIn[ez][ey][startx + nx - 1][idxP];
    }
  }
  if (isFirsty) {
    for (ez = startz; ez < startz + nz + nExtraz; ++ez) {
      for (ex = startx; ex < startx + nx + nExtrax; ++ex) arrIn[ez][-1][ex][idxP] = arrIn[ez][0][ex][idxP];
    }
  }
  if (isLasty) {
    for (ez = startz; ez < startz + nz + nExtraz; ++ez) {
      for (ex = startx; ex < startx + nx + nExtrax; ++ex) arrIn[ez][starty + ny][ex][idxP] = arrIn[ez][starty + ny - 1][ex][idxP];
    }
  }

  if (isFirstz) {
    for (ey = starty; ey < starty + ny + nExtray; ++ey) {
      for (ex = startx; ex < startx + nx + nExtrax; ++ex) arrIn[-1][ey][ex][idxP] = arrIn[0][ey][ex][idxP];
    }
  }
  if (isLastz) {
    for (ey = starty; ey < starty + ny + nExtray; ++ey) {
      for (ex = startx; ex < startx + nx + nExtrax; ++ex) arrIn[startz + nz][ey][ex][idxP] = arrIn[startz + nz - 1][ey][ex][idxP];
    }
  }

  /* Apply operator on physical points */
  for (ez = startz; ez < startz + nz + nExtraz; ++ez) {
    for (ey = starty; ey < starty + ny + nExtray; ++ey) {
      for (ex = startx; ex < startx + nx + nExtrax; ++ex) {
        if (ex < startx + nx && ey < starty + ny && ez < startz + nz) { /* Don't compute pressure outside domain */
          arrOut[ez][ey][ex][idxP] = arrIn[ez][ey][ex][idxP];
        }
        arrOut[ez][ey][ex][idxUx] = arrIn[ez][ey][ex][idxP] + arrIn[ez][ey][ex - 1][idxP] - arrIn[ez][ey][ex][idxUx];
        arrOut[ez][ey][ex][idxUy] = arrIn[ez][ey][ex][idxP] + arrIn[ez][ey - 1][ex][idxP] - arrIn[ez][ey][ex][idxUy];
        arrOut[ez][ey][ex][idxUz] = arrIn[ez][ey][ex][idxP] + arrIn[ez - 1][ey][ex][idxP] - arrIn[ez][ey][ex][idxUz];
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm, inLocal, &arrIn));
  PetscCall(DMStagVecRestoreArray(dm, outLocal, &arrOut));
  PetscCall(DMLocalToGlobalBegin(dm, outLocal, INSERT_VALUES, out));
  PetscCall(DMLocalToGlobalEnd(dm, outLocal, INSERT_VALUES, out));
  PetscCall(DMRestoreLocalVector(dm, &inLocal));
  PetscCall(DMRestoreLocalVector(dm, &outLocal));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 8

   test:
      suffix: 3
      nsize: 1
      args: -stag_stencil_width 2

   test:
      suffix: 4
      nsize: 8
      args: -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 4 -stag_stencil_width 2

   test:
      suffix: 5
      nsize: 8
      args: -stag_dof_0 3 -stag_dof_1 2 -stag_dof_2 4 -stag_dof_3 2 -stag_stencil_width 3 -stag_grid_x 6 -stag_grid_y 6 -stag_grid_z 6

TEST*/
