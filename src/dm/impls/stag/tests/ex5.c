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
  PetscErrorCode  ierr;
  DM              dmSol;
  Vec             sol,solRef,solRefLocal,rhs,rhsLocal;
  Mat             A;
  KSP             ksp;
  PC              pc;
  PetscInt        start,n,e,nExtra;
  PetscInt        iu,ip;
  PetscScalar     **arrSol,**arrRHS;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dmSol);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmSol);CHKERRQ(ierr);
  ierr = DMSetUp(dmSol);CHKERRQ(ierr);

  /* Compute reference solution on the grid, using direct array access */
  ierr = DMCreateGlobalVector(dmSol,&rhs);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmSol,&solRef);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmSol,&rhsLocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmSol,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmSol,rhsLocal,&arrRHS);CHKERRQ(ierr);

  /* Get the correct entries for each of our variables in local element-wise storage */
  ierr = DMStagGetLocationSlot(dmSol,LEFT,0,&iu);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip);CHKERRQ(ierr);
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
  ierr = DMStagVecRestoreArray(dmSol,rhsLocal,&arrRHS);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmSol,rhsLocal,INSERT_VALUES,rhs);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmSol,rhsLocal,INSERT_VALUES,rhs);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmSol,solRefLocal,INSERT_VALUES,solRef);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmSol,solRefLocal,INSERT_VALUES,solRef);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmSol,&rhsLocal);CHKERRQ(ierr);

  /* Matrix-free Operator */
  ierr = DMSetMatType(dmSol,MATSHELL);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmSol,&A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT,(void(*) (void)) ApplyOperator);CHKERRQ(ierr);

  /* Solve */
  ierr = DMCreateGlobalVector(dmSol,&sol);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);

  /* Check Solution */
  {
    Vec       diff;
    PetscReal normsolRef,errAbs,errRel;

    ierr = VecDuplicate(sol,&diff);CHKERRQ(ierr);
    ierr = VecCopy(sol,diff);CHKERRQ(ierr);
    ierr = VecAXPY(diff,-1.0,solRef);CHKERRQ(ierr);
    ierr = VecNorm(diff,NORM_2,&errAbs);CHKERRQ(ierr);
    ierr = VecNorm(solRef,NORM_2,&normsolRef);CHKERRQ(ierr);
    errRel = errAbs/normsolRef;
    if (errAbs > 1e14 || errRel > 1e14) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)dmSol),"Error (abs): %g\nError (rel): %g\n",(double)errAbs,(double)errRel);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)dmSol),"Non-zero error. Probable failure.\n");CHKERRQ(ierr);
    }
    ierr = VecDestroy(&diff);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  ierr = VecDestroy(&solRef);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSol);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ApplyOperator(Mat A,Vec in,Vec out)
{
  PetscErrorCode    ierr;
  DM                dm;
  Vec               inLocal,outLocal;
  PetscScalar       **arrIn;
  PetscScalar       **arrOut;
  PetscInt          start,n,nExtra,ex,idxP,idxU,startGhost,nGhost;
  DMBoundaryType    boundaryType;
  PetscBool         isFirst,isLast;

  PetscFunctionBeginUser;
  ierr = MatGetDM(A,&dm);CHKERRQ(ierr);
  ierr = DMStagGetBoundaryTypes(dm,&boundaryType,NULL,NULL);CHKERRQ(ierr);
  if (boundaryType != DM_BOUNDARY_GHOSTED) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Ghosted boundaries required");
  ierr = DMGetLocalVector(dm,&inLocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&outLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,in,INSERT_VALUES,inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,in,INSERT_VALUES,inLocal);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGhostCorners(dm,&startGhost,NULL,NULL,&nGhost,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,inLocal,&arrIn);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,outLocal,&arrOut);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,LEFT,0,&idxU);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idxP);CHKERRQ(ierr);
  ierr = DMStagGetIsFirstRank(dm,&isFirst,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetIsLastRank(dm,&isLast,NULL,NULL);CHKERRQ(ierr);

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
  ierr = DMStagVecRestoreArrayRead(dm,inLocal,&arrIn);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,outLocal,&arrOut);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,outLocal,INSERT_VALUES,out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,outLocal,INSERT_VALUES,out);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&inLocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&outLocal);CHKERRQ(ierr);
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
