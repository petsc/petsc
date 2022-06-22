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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dmSol));
  PetscCall(DMSetFromOptions(dmSol));
  PetscCall(DMSetUp(dmSol));

  /* Compute reference solution on the grid, using direct array access */
  PetscCall(DMCreateGlobalVector(dmSol,&rhs));
  PetscCall(DMCreateGlobalVector(dmSol,&solRef));
  PetscCall(DMGetLocalVector(dmSol,&solRefLocal));
  PetscCall(DMGetLocalVector(dmSol,&rhsLocal));
  PetscCall(DMStagVecGetArray(dmSol,solRefLocal,&arrSol));

  PetscCall(DMStagGetCorners(dmSol,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL));
  PetscCall(DMStagVecGetArray(dmSol,rhsLocal,&arrRHS));

  /* Get the correct entries for each of our variables in local element-wise storage */
  PetscCall(DMStagGetLocationSlot(dmSol,LEFT,0,&iu));
  PetscCall(DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip));
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
  PetscCall(DMStagVecRestoreArray(dmSol,rhsLocal,&arrRHS));
  PetscCall(DMLocalToGlobalBegin(dmSol,rhsLocal,INSERT_VALUES,rhs));
  PetscCall(DMLocalToGlobalEnd(dmSol,rhsLocal,INSERT_VALUES,rhs));
  PetscCall(DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol));
  PetscCall(DMLocalToGlobalBegin(dmSol,solRefLocal,INSERT_VALUES,solRef));
  PetscCall(DMLocalToGlobalEnd(dmSol,solRefLocal,INSERT_VALUES,solRef));
  PetscCall(DMRestoreLocalVector(dmSol,&solRefLocal));
  PetscCall(DMRestoreLocalVector(dmSol,&rhsLocal));

  /* Matrix-free Operator */
  PetscCall(DMSetMatType(dmSol,MATSHELL));
  PetscCall(DMCreateMatrix(dmSol,&A));
  PetscCall(MatShellSetOperation(A,MATOP_MULT,(void(*) (void)) ApplyOperator));

  /* Solve */
  PetscCall(DMCreateGlobalVector(dmSol,&sol));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,rhs,sol));

  /* Check Solution */
  {
    Vec       diff;
    PetscReal normsolRef,errAbs,errRel;

    PetscCall(VecDuplicate(sol,&diff));
    PetscCall(VecCopy(sol,diff));
    PetscCall(VecAXPY(diff,-1.0,solRef));
    PetscCall(VecNorm(diff,NORM_2,&errAbs));
    PetscCall(VecNorm(solRef,NORM_2,&normsolRef));
    errRel = errAbs/normsolRef;
    if (errAbs > 1e14 || errRel > 1e14) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmSol),"Error (abs): %g\nError (rel): %g\n",(double)errAbs,(double)errRel));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmSol),"Non-zero error. Probable failure.\n"));
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
  PetscCall(MatGetDM(A,&dm));
  PetscCall(DMStagGetBoundaryTypes(dm,&boundaryType,NULL,NULL));
  PetscCheck(boundaryType == DM_BOUNDARY_GHOSTED,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Ghosted boundaries required");
  PetscCall(DMGetLocalVector(dm,&inLocal));
  PetscCall(DMGetLocalVector(dm,&outLocal));
  PetscCall(DMGlobalToLocalBegin(dm,in,INSERT_VALUES,inLocal));
  PetscCall(DMGlobalToLocalEnd(dm,in,INSERT_VALUES,inLocal));
  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL));
  PetscCall(DMStagGetGhostCorners(dm,&startGhost,NULL,NULL,&nGhost,NULL,NULL));
  PetscCall(DMStagVecGetArrayRead(dm,inLocal,&arrIn));
  PetscCall(DMStagVecGetArray(dm,outLocal,&arrOut));
  PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idxU));
  PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idxP));
  PetscCall(DMStagGetIsFirstRank(dm,&isFirst,NULL,NULL));
  PetscCall(DMStagGetIsLastRank(dm,&isLast,NULL,NULL));

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
  PetscCall(DMStagVecRestoreArrayRead(dm,inLocal,&arrIn));
  PetscCall(DMStagVecRestoreArray(dm,outLocal,&arrOut));
  PetscCall(DMLocalToGlobalBegin(dm,outLocal,INSERT_VALUES,out));
  PetscCall(DMLocalToGlobalEnd(dm,outLocal,INSERT_VALUES,out));
  PetscCall(DMRestoreLocalVector(dm,&inLocal));
  PetscCall(DMRestoreLocalVector(dm,&outLocal));
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
