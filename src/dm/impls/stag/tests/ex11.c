static char help[] = "Test DMStag ghosted boundaries in 2d\n\n";
/* This solves a very contrived problem - the "pressure" terms are set to a constant function
   and the "velocity" terms are just the sum of neighboring values of these, hence twice the
   constant */
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>

#define PRESSURE_CONST 2.0

PetscErrorCode ApplyOperator(Mat,Vec,Vec);

int main(int argc,char **argv)
{
  DM              dmSol;
  Vec             sol,solRef,solRefLocal,rhs,rhsLocal;
  Mat             A;
  KSP             ksp;
  PC              pc;
  PetscInt        startx,starty,nx,ny,ex,ey,nExtrax,nExtray;
  PetscInt        iux,iuy,ip;
  PetscInt        dof0,dof1,dof2;
  PetscScalar     ***arrSol,***arrRHS;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Note: these defaults are chosen to suit the problem. We allow adjusting
     them to check that things still work when you add unused extra dof */
  dof0 = 0;
  dof1 = 1;
  dof2 = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof2",&dof2,NULL));
  CHKERRQ(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,3,3,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dmSol));
  CHKERRQ(DMSetFromOptions(dmSol));
  CHKERRQ(DMSetUp(dmSol));

  /* Compute reference solution on the grid, using direct array access */
  CHKERRQ(DMCreateGlobalVector(dmSol,&rhs));
  CHKERRQ(DMCreateGlobalVector(dmSol,&solRef));
  CHKERRQ(DMGetLocalVector(dmSol,&solRefLocal));
  CHKERRQ(DMGetLocalVector(dmSol,&rhsLocal));
  CHKERRQ(DMStagVecGetArray(dmSol,solRefLocal,&arrSol));

  CHKERRQ(DMStagGetCorners(dmSol,&startx,&starty,NULL,&nx,&ny,NULL,&nExtrax,&nExtray,NULL));
  CHKERRQ(DMStagVecGetArray(dmSol,rhsLocal,&arrRHS));

  /* Get the correct entries for each of our variables in local element-wise storage */
  CHKERRQ(DMStagGetLocationSlot(dmSol,DMSTAG_LEFT,0,&iux));
  CHKERRQ(DMStagGetLocationSlot(dmSol,DMSTAG_DOWN,0,&iuy));
  CHKERRQ(DMStagGetLocationSlot(dmSol,DMSTAG_ELEMENT,0,&ip));
    for (ey=starty; ey<starty+ny+nExtray; ++ey) {
      for (ex=startx; ex<startx+nx+nExtrax; ++ex) {
        arrSol[ey][ex][iux] = 2*PRESSURE_CONST;
        arrRHS[ey][ex][iux] = 0.0;
        arrSol[ey][ex][iuy] = 2*PRESSURE_CONST;
        arrRHS[ey][ex][iuy] = 0.0;
        if (ex < startx+nx && ey < starty+ny) {
          arrSol[ey][ex][ip] = PRESSURE_CONST;
          arrRHS[ey][ex][ip] = PRESSURE_CONST;
        }
      }
    }
  CHKERRQ(DMStagVecRestoreArray(dmSol,rhsLocal,&arrRHS));
  CHKERRQ(DMLocalToGlobalBegin(dmSol,rhsLocal,INSERT_VALUES,rhs));
  CHKERRQ(DMLocalToGlobalEnd(dmSol,rhsLocal,INSERT_VALUES,rhs));
  CHKERRQ(DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol));
  CHKERRQ(DMLocalToGlobalBegin(dmSol,solRefLocal,INSERT_VALUES,solRef));
  CHKERRQ(DMLocalToGlobalEnd(dmSol,solRefLocal,INSERT_VALUES,solRef));
  CHKERRQ(DMRestoreLocalVector(dmSol,&solRefLocal));
  CHKERRQ(DMRestoreLocalVector(dmSol,&rhsLocal));

  /* Matrix-free Operator */
  CHKERRQ(DMSetMatType(dmSol,MATSHELL));
  CHKERRQ(DMCreateMatrix(dmSol,&A));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void(*) (void)) ApplyOperator));

  /* Solve */
  CHKERRQ(DMCreateGlobalVector(dmSol,&sol));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,rhs,sol));

  /* Check Solution */
  {
    Vec       diff;
    PetscReal normsolRef,errAbs,errRel;

    CHKERRQ(VecDuplicate(sol,&diff));
    CHKERRQ(VecCopy(sol,diff));
    CHKERRQ(VecAXPY(diff,-1.0,solRef));
    CHKERRQ(VecNorm(diff,NORM_2,&errAbs));
    CHKERRQ(VecNorm(solRef,NORM_2,&normsolRef));
    errRel = errAbs/normsolRef;
    if (errAbs > 1e14 || errRel > 1e14) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dmSol),"Error (abs): %g\nError (rel): %g\n",(double)errAbs,(double)errRel));
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dmSol),"Non-zero error. Probable failure.\n"));
    }
    CHKERRQ(VecDestroy(&diff));
  }

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&sol));
  CHKERRQ(VecDestroy(&solRef));
  CHKERRQ(VecDestroy(&rhs));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(DMDestroy(&dmSol));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode ApplyOperator(Mat A,Vec in,Vec out)
{
  DM                dm;
  Vec               inLocal,outLocal;
  PetscScalar       ***arrIn;
  PetscScalar       ***arrOut;
  PetscInt          startx,starty,nx,ny,nExtrax,nExtray,ex,ey,idxP,idxUx,idxUy,startGhostx,startGhosty,nGhostx,nGhosty;
  PetscBool         isFirstx,isFirsty,isFirstz,isLastx,isLasty,isLastz;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetDM(A,&dm));
  CHKERRQ(DMGetLocalVector(dm,&inLocal));
  CHKERRQ(DMGetLocalVector(dm,&outLocal));
  CHKERRQ(DMGlobalToLocalBegin(dm,in,INSERT_VALUES,inLocal));
  CHKERRQ(DMGlobalToLocalEnd(dm,in,INSERT_VALUES,inLocal));
  CHKERRQ(DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,&nExtrax,&nExtray,NULL));
  CHKERRQ(DMStagGetGhostCorners(dm,&startGhostx,&startGhosty,NULL,&nGhostx,&nGhosty,NULL));
  CHKERRQ(DMStagVecGetArrayRead(dm,inLocal,&arrIn));
  CHKERRQ(DMStagVecGetArray(dm,outLocal,&arrOut));
  CHKERRQ(DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxUx));
  CHKERRQ(DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&idxUy));
  CHKERRQ(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxP));
  CHKERRQ(DMStagGetIsFirstRank(dm,&isFirstx,&isFirsty,&isFirstz));
  CHKERRQ(DMStagGetIsLastRank(dm,&isLastx,&isLasty,&isLastz));

  /* Set "pressures" on ghost boundaries by copying neighboring values*/
  if (isFirstx) {
      for (ey=starty; ey<starty+ny+nExtray; ++ey) {
        arrIn[ey][-1][idxP] = arrIn[ey][0][idxP];
      }
  }
  if (isLastx) {
      for (ey=starty; ey<starty+ny+nExtray; ++ey) {
        arrIn[ey][startx + nx][idxP] = arrIn[ey][startx + nx - 1][idxP];
      }
  }
  if (isFirsty) {
      for (ex=startx; ex<startx+nx+nExtrax; ++ex) {
        arrIn[-1][ex][idxP] = arrIn[0][ex][idxP];
      }
  }
  if  (isLasty) {
      for (ex=startx; ex<startx+nx+nExtrax; ++ex) {
        arrIn[starty + ny][ex][idxP] = arrIn[starty + ny - 1][ex][idxP];
      }
  }

  /* Apply operator on physical points */
  for (ey=starty; ey<starty+ny+nExtray; ++ey) {
    for (ex=startx; ex<startx+nx+nExtrax; ++ex) {
      if (ex < startx+nx && ey < starty+ny) {/* Don't compute pressure outside domain */
        arrOut[ey][ex][idxP] = arrIn[ey][ex][idxP];
      }
      arrOut[ey][ex][idxUx] = arrIn[ey][ex][idxP] + arrIn[ey][ex-1][idxP] - arrIn[ey][ex][idxUx];
      arrOut[ey][ex][idxUy] = arrIn[ey][ex][idxP] + arrIn[ey-1][ex][idxP] - arrIn[ey][ex][idxUy];
    }
  }
  CHKERRQ(DMStagVecRestoreArrayRead(dm,inLocal,&arrIn));
  CHKERRQ(DMStagVecRestoreArray(dm,outLocal,&arrOut));
  CHKERRQ(DMLocalToGlobalBegin(dm,outLocal,INSERT_VALUES,out));
  CHKERRQ(DMLocalToGlobalEnd(dm,outLocal,INSERT_VALUES,out));
  CHKERRQ(DMRestoreLocalVector(dm,&inLocal));
  CHKERRQ(DMRestoreLocalVector(dm,&outLocal));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 4

   test:
      suffix: 3
      nsize: 1
      args: -stag_stencil_width 2

   test:
      suffix: 4
      nsize: 4
      args: -stag_grid_x 4 -stag_grid_y 5 -stag_stencil_width 2

   test:
      suffix: 5
      nsize: 4
      args: -stag_dof_0 3 -stag_dof_1 2 -stag_dof_2 4 -stag_stencil_width 3 -stag_grid_x 6 -stag_grid_y 6

TEST*/
