static char help[] = "Test DMStag ghosted boundaries in 1d\n\n";
/* This solves a very contrived problem - the "pressure" terms are set to a constant function
   and the "velocity" terms are just the sum of neighboring values of these, hence twice the
   constant */
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>

#define LEFT    DMSTAG_LEFT
#define RIGHT   DMSTAG_RIGHT
#define ELEMENT DMSTAG_ELEMENT

#define PRESSURE_CONST 2.0

PetscErrorCode ApplyOperator(Mat,Vec,Vec);

int main(int argc,char **argv)
{
  DM              dmSol;
  Vec             sol,solRef,solRefLocal,rhs,rhsLocal;
  Mat             A;
  KSP             ksp;
  PC              pc;
  PetscInt        start,n,e,nExtra;
  PetscInt        iu,ip;
  PetscScalar     **arrSol,**arrRHS;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dmSol));
  CHKERRQ(DMSetFromOptions(dmSol));
  CHKERRQ(DMSetUp(dmSol));

  /* Compute reference solution on the grid, using direct array access */
  CHKERRQ(DMCreateGlobalVector(dmSol,&rhs));
  CHKERRQ(DMCreateGlobalVector(dmSol,&solRef));
  CHKERRQ(DMGetLocalVector(dmSol,&solRefLocal));
  CHKERRQ(DMGetLocalVector(dmSol,&rhsLocal));
  CHKERRQ(DMStagVecGetArray(dmSol,solRefLocal,&arrSol));

  CHKERRQ(DMStagGetCorners(dmSol,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL));
  CHKERRQ(DMStagVecGetArray(dmSol,rhsLocal,&arrRHS));

  /* Get the correct entries for each of our variables in local element-wise storage */
  CHKERRQ(DMStagGetLocationSlot(dmSol,LEFT,0,&iu));
  CHKERRQ(DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip));
  for (e=start; e<start+n+nExtra; ++e) {
    {
      arrSol[e][iu] = 2*PRESSURE_CONST;
      arrRHS[e][iu] = 0.0;
    }
    if (e < start+n) {
      arrSol[e][ip] = PRESSURE_CONST;
      arrRHS[e][ip] = PRESSURE_CONST;
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
  PetscScalar       **arrIn;
  PetscScalar       **arrOut;
  PetscInt          start,n,nExtra,ex,idxP,idxU,startGhost,nGhost;
  DMBoundaryType    boundaryType;
  PetscBool         isFirst,isLast;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetDM(A,&dm));
  CHKERRQ(DMStagGetBoundaryTypes(dm,&boundaryType,NULL,NULL));
  PetscCheckFalse(boundaryType != DM_BOUNDARY_GHOSTED,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Ghosted boundaries required");
  CHKERRQ(DMGetLocalVector(dm,&inLocal));
  CHKERRQ(DMGetLocalVector(dm,&outLocal));
  CHKERRQ(DMGlobalToLocalBegin(dm,in,INSERT_VALUES,inLocal));
  CHKERRQ(DMGlobalToLocalEnd(dm,in,INSERT_VALUES,inLocal));
  CHKERRQ(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL));
  CHKERRQ(DMStagGetGhostCorners(dm,&startGhost,NULL,NULL,&nGhost,NULL,NULL));
  CHKERRQ(DMStagVecGetArrayRead(dm,inLocal,&arrIn));
  CHKERRQ(DMStagVecGetArray(dm,outLocal,&arrOut));
  CHKERRQ(DMStagGetLocationSlot(dm,LEFT,0,&idxU));
  CHKERRQ(DMStagGetLocationSlot(dm,ELEMENT,0,&idxP));
  CHKERRQ(DMStagGetIsFirstRank(dm,&isFirst,NULL,NULL));
  CHKERRQ(DMStagGetIsLastRank(dm,&isLast,NULL,NULL));

  /* Set "pressures" on ghost boundaries by copying neighboring values*/
  if (isFirst) {
    arrIn[-1][idxP] = arrIn[0][idxP];
  }
  if (isLast) {
    arrIn[start + n][idxP] = arrIn[start + n - 1][idxP];
  }

  /* Apply operator on physical points */
  for (ex=start; ex<start + n + nExtra; ++ex) {
    if (ex < start + n) { /* Don't compute pressure outside domain */
      arrOut[ex][idxP] = arrIn[ex][idxP];
    }
    arrOut[ex][idxU] = arrIn[ex][idxP] + arrIn[ex-1][idxP] - arrIn[ex][idxU];
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
      nsize: 2

   test:
      suffix: 3
      nsize: 3
      args: -stag_grid_x 19

   test:
      suffix: 4
      nsize: 5
      args: -stag_grid_x 21 -stag_stencil_width 2

TEST*/
