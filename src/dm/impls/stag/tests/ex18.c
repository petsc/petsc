static char help[] = "Test: Solve a toy 2D problem on a staggered grid using 2-level multigrid. \n"
                     "The solve is only applied to the u-u block.\n\n";
/*

  Solves only the velocity block of a isoviscous incompressible Stokes problem on a rectangular 2D
  domain.

  u_xx + u_yy - p_x = f^x
  v_xx + v_yy - p_y = f^y
  u_x + v_y         = g

  g is 0 in the physical case.

  Boundary conditions give prescribed flow perpendicular to the boundaries,
  and zero derivative perpendicular to them (free slip).

  Supply the -analyze flag to activate a custom KSP monitor. Note that
  this does an additional direct solve of the velocity block of the system to have an "exact"
  solution to the discrete system available (see KSPSetOptionsPrefix
  call below for the prefix).

  This is for testing purposes, and uses some routines to make
  sure that transfer operators are consistent with extracting submatrices.

  -extractTransferOperators (true by default) defines transfer operators for
  the velocity-velocity system by extracting submatrices from the operators for
  the full system.

*/
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h> /* Includes petscdmproduct.h */

/* Shorter, more convenient names for DMStagStencilLocation entries */
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

static PetscErrorCode CreateSystem(DM, Mat *, Vec *);
static PetscErrorCode CreateNumericalReferenceSolution(Mat, Vec, Vec *);
static PetscErrorCode DMStagAnalysisKSPMonitor(KSP, PetscInt, PetscReal, void *);
typedef struct {
  DM  dm;
  Vec solRef, solRefNum, solPrev;
} DMStagAnalysisKSPMonitorContext;

/* Manufactured solution. Chosen to be higher order than can be solved exactly,
and to have a zero derivative for flow parallel to the boundaries. That is,
d(ux)/dy = 0 at the top and bottom boundaries, and d(uy)/dx = 0 at the right
and left boundaries. */
static PetscScalar uxRef(PetscScalar x, PetscScalar y)
{
  return 0.0 * x + y * y - 2.0 * y * y * y + y * y * y * y;
} /* no x-dependence  */
static PetscScalar uyRef(PetscScalar x, PetscScalar y)
{
  return x * x - 2.0 * x * x * x + x * x * x * x + 0.0 * y;
} /* no y-dependence  */
static PetscScalar fx(PetscScalar x, PetscScalar y)
{
  return 0.0 * x + 2.0 - 12.0 * y + 12.0 * y * y + 1.0;
} /* no x-dependence  */
static PetscScalar fy(PetscScalar x, PetscScalar y)
{
  return 2.0 - 12.0 * x + 12.0 * x * x + 3.0 * y;
}
static PetscScalar g(PetscScalar x, PetscScalar y)
{
  return 0.0 * x * y;
} /* identically zero */

int main(int argc, char **argv)
{
  DM            dmSol, dmSolc, dmuu, dmuuc;
  KSP           ksp, kspc;
  PC            pc;
  Mat           IIu, Ru, Auu, Auuc;
  Vec           su, xu, fu;
  IS            isuf, ispf, isuc, ispc;
  PetscInt      cnt = 0;
  DMStagStencil stencil_set[1 + 1 + 1];
  PetscBool     extractTransferOperators, extractSystem;

  DMStagAnalysisKSPMonitorContext mctx;
  PetscBool                       analyze;

  /* Initialize PETSc and process command line arguments */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  analyze = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-analyze", &analyze, NULL));
  extractTransferOperators = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-extractTransferOperators", &extractTransferOperators, NULL));
  extractSystem = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-extractSystem", &extractSystem, NULL));

  /* Create 2D DMStag for the solution, and set up. */
  {
    const PetscInt dof0 = 0, dof1 = 1, dof2 = 1; /* 1 dof on each edge and element center */
    const PetscInt stencilWidth = 1;
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2, DMSTAG_STENCIL_BOX, stencilWidth, NULL, NULL, &dmSol));
    PetscCall(DMSetFromOptions(dmSol));
    PetscCall(DMSetUp(dmSol));
  }

  /* Create coarse DMStag (which we may or may not directly use) */
  PetscCall(DMCoarsen(dmSol, MPI_COMM_NULL, &dmSolc));
  PetscCall(DMSetUp(dmSolc));

  /* Create compatible DMStags with only velocity dof (which we may or may not
     directly use) */
  PetscCall(DMStagCreateCompatibleDMStag(dmSol, 0, 1, 0, 0, &dmuu)); /* vel-only */
  PetscCall(DMSetUp(dmuu));
  PetscCall(DMCoarsen(dmuu, MPI_COMM_NULL, &dmuuc));
  PetscCall(DMSetUp(dmuuc));

  /* Define uniform coordinates as a product of 1D arrays */
  PetscCall(DMStagSetUniformCoordinatesProduct(dmSol, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
  PetscCall(DMStagSetUniformCoordinatesProduct(dmSolc, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
  PetscCall(DMStagSetUniformCoordinatesProduct(dmuu, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
  PetscCall(DMStagSetUniformCoordinatesProduct(dmuuc, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

  /* Create ISs for the velocity and pressure blocks */
  /* i,j will be ignored */
  stencil_set[cnt].loc = DMSTAG_DOWN;
  stencil_set[cnt].c   = 0;
  cnt++; /* u */
  stencil_set[cnt].loc = DMSTAG_LEFT;
  stencil_set[cnt].c   = 0;
  cnt++; /* v */
  stencil_set[cnt].loc = DMSTAG_ELEMENT;
  stencil_set[cnt].c   = 0;
  cnt++; /* p */

  PetscCall(DMStagCreateISFromStencils(dmSol, 2, stencil_set, &isuf));
  PetscCall(DMStagCreateISFromStencils(dmSol, 1, &stencil_set[2], &ispf));
  PetscCall(DMStagCreateISFromStencils(dmSolc, 2, stencil_set, &isuc));
  PetscCall(DMStagCreateISFromStencils(dmSolc, 1, &stencil_set[2], &ispc));

  /* Assemble velocity-velocity system */
  if (extractSystem) {
    Mat A, Ac;
    Vec tmp, rhs;

    PetscCall(CreateSystem(dmSol, &A, &rhs));
    PetscCall(CreateSystem(dmSolc, &Ac, NULL));
    PetscCall(MatCreateSubMatrix(A, isuf, isuf, MAT_INITIAL_MATRIX, &Auu));
    PetscCall(MatCreateSubMatrix(Ac, isuc, isuc, MAT_INITIAL_MATRIX, &Auuc));
    PetscCall(MatCreateVecs(Auu, &xu, &fu));
    PetscCall(VecGetSubVector(rhs, isuf, &tmp));
    PetscCall(VecCopy(tmp, fu));
    PetscCall(VecRestoreSubVector(rhs, isuf, &tmp));
    PetscCall(MatDestroy(&Ac));
    PetscCall(VecDestroy(&rhs));
    PetscCall(MatDestroy(&A));
  } else {
    PetscCall(CreateSystem(dmuu, &Auu, &fu));
    PetscCall(CreateSystem(dmuuc, &Auuc, NULL));
    PetscCall(MatCreateVecs(Auu, &xu, NULL));
  }
  PetscCall(PetscObjectSetName((PetscObject)Auu, "Auu"));
  PetscCall(PetscObjectSetName((PetscObject)Auuc, "Auuc"));

  /* Create Transfer Operators and scaling for the velocity-velocity block */
  if (extractTransferOperators) {
    Mat II, R;
    Vec s, tmp;

    PetscCall(DMCreateInterpolation(dmSolc, dmSol, &II, NULL));
    PetscCall(DMCreateRestriction(dmSolc, dmSol, &R));
    PetscCall(MatCreateSubMatrix(II, isuf, isuc, MAT_INITIAL_MATRIX, &IIu));
    PetscCall(MatCreateSubMatrix(R, isuc, isuf, MAT_INITIAL_MATRIX, &Ru));
    PetscCall(DMCreateInterpolationScale(dmSolc, dmSol, II, &s));
    PetscCall(MatCreateVecs(IIu, &su, NULL));
    PetscCall(VecGetSubVector(s, isuc, &tmp));
    PetscCall(VecCopy(tmp, su));
    PetscCall(VecRestoreSubVector(s, isuc, &tmp));
    PetscCall(MatDestroy(&R));
    PetscCall(MatDestroy(&II));
    PetscCall(VecDestroy(&s));
  } else {
    PetscCall(DMCreateInterpolation(dmuuc, dmuu, &IIu, NULL));
    PetscCall(DMCreateInterpolationScale(dmuuc, dmuu, IIu, &su));
    PetscCall(DMCreateRestriction(dmuuc, dmuu, &Ru));
  }

  /* Create and configure solver */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, Auu, Auu));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  PetscCall(PCMGSetInterpolation(pc, 1, IIu));
  /* PetscCall(PCMGSetRestriction(pc,1,Ru)); */
  PetscCall(PCMGSetRScale(pc, 1, su));

  PetscCall(PCMGGetCoarseSolve(pc, &kspc));
  PetscCall(KSPSetOperators(kspc, Auuc, Auuc));
  PetscCall(KSPSetFromOptions(ksp));

  if (analyze) {
    mctx.dm      = dmuu;
    mctx.solRef  = NULL; /* Reference solution not computed for u-u only */
    mctx.solPrev = NULL; /* Populated automatically */
    PetscCall(CreateNumericalReferenceSolution(Auu, fu, &mctx.solRefNum));
    PetscCall(KSPMonitorSet(ksp, DMStagAnalysisKSPMonitor, &mctx, NULL));
  }

  /* Solve */
  PetscCall(KSPSolve(ksp, fu, xu));

  /* Clean up and finalize PETSc */
  if (analyze) {
    PetscCall(VecDestroy(&mctx.solPrev));
    PetscCall(VecDestroy(&mctx.solRef));
    PetscCall(VecDestroy(&mctx.solRefNum));
  }
  PetscCall(DMDestroy(&dmuu));
  PetscCall(DMDestroy(&dmuuc));
  PetscCall(VecDestroy(&su));
  PetscCall(ISDestroy(&ispc));
  PetscCall(ISDestroy(&ispf));
  PetscCall(ISDestroy(&isuc));
  PetscCall(ISDestroy(&isuf));
  PetscCall(MatDestroy(&Ru));
  PetscCall(MatDestroy(&IIu));
  PetscCall(MatDestroy(&Auuc));
  PetscCall(MatDestroy(&Auu));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&fu));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dmSolc));
  PetscCall(DMDestroy(&dmSol));
  PetscCall(PetscFinalize());
  return 0;
}

/*
Note: in this system all stencil coefficients which are not related to the Dirichlet boundary are scaled by dv = dx*dy.
This scaling is necessary for multigrid to converge.
*/
static PetscErrorCode CreateSystem(DM dm, Mat *pA, Vec *pRhs)
{
  PetscInt      N[2], dof[3];
  PetscBool     isLastRankx, isLastRanky, isFirstRankx, isFirstRanky;
  PetscInt      ex, ey, startx, starty, nx, ny;
  PetscInt      iprev, icenter, inext;
  Mat           A;
  Vec           rhs;
  PetscReal     hx, hy, dv, bogusScale;
  PetscScalar **cArrX, **cArrY;
  PetscBool     hasPressure;

  PetscFunctionBeginUser;

  /* Determine whether or not to create system including pressure dof (on elements) */
  PetscCall(DMStagGetDOF(dm, &dof[0], &dof[1], &dof[2], NULL));
  PetscCheck(dof[0] == 0 && dof[1] == 1 && (dof[2] == 1 || dof[2] == 0), PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "CreateSystem only implemented for velocity-only or velocity+pressure grids");
  hasPressure = (PetscBool)(dof[2] == 1);

  bogusScale = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-bogus", &bogusScale, NULL)); /* Use this to break MG (try small values) */

  PetscCall(DMCreateMatrix(dm, pA));
  A = *pA;
  if (pRhs) {
    PetscCall(DMCreateGlobalVector(dm, pRhs));
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  PetscCall(DMStagGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetGlobalSizes(dm, &N[0], &N[1], NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetIsFirstRank(dm, &isFirstRankx, &isFirstRanky, NULL));
  hx = 1.0 / N[0];
  hy = 1.0 / N[1];
  dv = hx * hy;
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &cArrX, &cArrY, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, RIGHT, &inext));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, ELEMENT, &icenter));

  /* Loop over all local elements. Note that it may be more efficient in real
     applications to loop over each boundary separately */
  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ++ex) {
      if (ex == N[0] - 1) {
        /* Right Boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;

        const PetscScalar valA = bogusScale * 1.0;
        row.i                  = ex;
        row.j                  = ey;
        row.loc                = RIGHT;
        row.c                  = 0;
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 1, &row, &valA, INSERT_VALUES));
        if (rhs) {
          valRhs = bogusScale * uxRef(cArrX[ex][inext], cArrY[ey][icenter]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      }
      if (ey == N[1] - 1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i                  = ex;
        row.j                  = ey;
        row.loc                = UP;
        row.c                  = 0;
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 1, &row, &valA, INSERT_VALUES));
        if (rhs) {
          valRhs = uyRef(cArrX[ex][icenter], cArrY[ey][inext]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i                  = ex;
        row.j                  = ey;
        row.loc                = DOWN;
        row.c                  = 0;
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 1, &row, &valA, INSERT_VALUES));
        if (rhs) {
          valRhs = uyRef(cArrX[ex][icenter], cArrY[ey][iprev]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y */
        DMStagStencil row, col[7];
        PetscScalar   valA[7], valRhs;
        PetscInt      nEntries;

        row.i   = ex;
        row.j   = ey;
        row.loc = DOWN;
        row.c   = 0;
        if (ex == 0) {
          nEntries   = 4;
          col[0].i   = ex;
          col[0].j   = ey;
          col[0].loc = DOWN;
          col[0].c   = 0;
          valA[0]    = -dv * 1.0 / (hx * hx) - dv * 2.0 / (hy * hy);
          col[1].i   = ex;
          col[1].j   = ey - 1;
          col[1].loc = DOWN;
          col[1].c   = 0;
          valA[1]    = dv * 1.0 / (hy * hy);
          col[2].i   = ex;
          col[2].j   = ey + 1;
          col[2].loc = DOWN;
          col[2].c   = 0;
          valA[2]    = dv * 1.0 / (hy * hy);
          /* Missing left element */
          col[3].i   = ex + 1;
          col[3].j   = ey;
          col[3].loc = DOWN;
          col[3].c   = 0;
          valA[3]    = dv * 1.0 / (hx * hx);
          if (hasPressure) {
            nEntries += 2;
            col[4].i   = ex;
            col[4].j   = ey - 1;
            col[4].loc = ELEMENT;
            col[4].c   = 0;
            valA[4]    = dv * 1.0 / hy;
            col[5].i   = ex;
            col[5].j   = ey;
            col[5].loc = ELEMENT;
            col[5].c   = 0;
            valA[5]    = -dv * 1.0 / hy;
          }
        } else if (ex == N[0] - 1) {
          /* Right boundary y velocity stencil */
          nEntries   = 4;
          col[0].i   = ex;
          col[0].j   = ey;
          col[0].loc = DOWN;
          col[0].c   = 0;
          valA[0]    = -dv * 1.0 / (hx * hx) - dv * 2.0 / (hy * hy);
          col[1].i   = ex;
          col[1].j   = ey - 1;
          col[1].loc = DOWN;
          col[1].c   = 0;
          valA[1]    = dv * 1.0 / (hy * hy);
          col[2].i   = ex;
          col[2].j   = ey + 1;
          col[2].loc = DOWN;
          col[2].c   = 0;
          valA[2]    = dv * 1.0 / (hy * hy);
          col[3].i   = ex - 1;
          col[3].j   = ey;
          col[3].loc = DOWN;
          col[3].c   = 0;
          valA[3]    = dv * 1.0 / (hx * hx);
          /* Missing right element */
          if (hasPressure) {
            nEntries += 2;
            col[4].i   = ex;
            col[4].j   = ey - 1;
            col[4].loc = ELEMENT;
            col[4].c   = 0;
            valA[4]    = dv * 1.0 / hy;
            col[5].i   = ex;
            col[5].j   = ey;
            col[5].loc = ELEMENT;
            col[5].c   = 0;
            valA[5]    = -dv * 1.0 / hy;
          }
        } else {
          nEntries   = 5;
          col[0].i   = ex;
          col[0].j   = ey;
          col[0].loc = DOWN;
          col[0].c   = 0;
          valA[0]    = -dv * 2.0 / (hx * hx) - dv * 2.0 / (hy * hy);
          col[1].i   = ex;
          col[1].j   = ey - 1;
          col[1].loc = DOWN;
          col[1].c   = 0;
          valA[1]    = dv * 1.0 / (hy * hy);
          col[2].i   = ex;
          col[2].j   = ey + 1;
          col[2].loc = DOWN;
          col[2].c   = 0;
          valA[2]    = dv * 1.0 / (hy * hy);
          col[3].i   = ex - 1;
          col[3].j   = ey;
          col[3].loc = DOWN;
          col[3].c   = 0;
          valA[3]    = dv * 1.0 / (hx * hx);
          col[4].i   = ex + 1;
          col[4].j   = ey;
          col[4].loc = DOWN;
          col[4].c   = 0;
          valA[4]    = dv * 1.0 / (hx * hx);
          if (hasPressure) {
            nEntries += 2;
            col[5].i   = ex;
            col[5].j   = ey - 1;
            col[5].loc = ELEMENT;
            col[5].c   = 0;
            valA[5]    = dv * 1.0 / hy;
            col[6].i   = ex;
            col[6].j   = ey;
            col[6].loc = ELEMENT;
            col[6].c   = 0;
            valA[6]    = -dv * 1.0 / hy;
          }
        }
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, nEntries, col, valA, INSERT_VALUES));
        if (rhs) {
          valRhs = dv * fy(cArrX[ex][icenter], cArrY[ey][iprev]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      }

      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i                  = ex;
        row.j                  = ey;
        row.loc                = LEFT;
        row.c                  = 0;
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 1, &row, &valA, INSERT_VALUES));
        if (rhs) {
          valRhs = uxRef(cArrX[ex][iprev], cArrY[ey][icenter]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
        DMStagStencil row, col[7];
        PetscScalar   valA[7], valRhs;
        PetscInt      nEntries;
        row.i   = ex;
        row.j   = ey;
        row.loc = LEFT;
        row.c   = 0;

        if (ey == 0) {
          nEntries   = 4;
          col[0].i   = ex;
          col[0].j   = ey;
          col[0].loc = LEFT;
          col[0].c   = 0;
          valA[0]    = -dv * 2.0 / (hx * hx) - dv * 1.0 / (hy * hy);
          /* missing term from element below */
          col[1].i   = ex;
          col[1].j   = ey + 1;
          col[1].loc = LEFT;
          col[1].c   = 0;
          valA[1]    = dv * 1.0 / (hy * hy);
          col[2].i   = ex - 1;
          col[2].j   = ey;
          col[2].loc = LEFT;
          col[2].c   = 0;
          valA[2]    = dv * 1.0 / (hx * hx);
          col[3].i   = ex + 1;
          col[3].j   = ey;
          col[3].loc = LEFT;
          col[3].c   = 0;
          valA[3]    = dv * 1.0 / (hx * hx);
          if (hasPressure) {
            nEntries += 2;
            col[4].i   = ex - 1;
            col[4].j   = ey;
            col[4].loc = ELEMENT;
            col[4].c   = 0;
            valA[4]    = dv * 1.0 / hx;
            col[5].i   = ex;
            col[5].j   = ey;
            col[5].loc = ELEMENT;
            col[5].c   = 0;
            valA[5]    = -dv * 1.0 / hx;
          }
        } else if (ey == N[1] - 1) {
          /* Top boundary x velocity stencil */
          nEntries   = 4;
          row.i      = ex;
          row.j      = ey;
          row.loc    = LEFT;
          row.c      = 0;
          col[0].i   = ex;
          col[0].j   = ey;
          col[0].loc = LEFT;
          col[0].c   = 0;
          valA[0]    = -dv * 2.0 / (hx * hx) - dv * 1.0 / (hy * hy);
          col[1].i   = ex;
          col[1].j   = ey - 1;
          col[1].loc = LEFT;
          col[1].c   = 0;
          valA[1]    = dv * 1.0 / (hy * hy);
          /* Missing element above term */
          col[2].i   = ex - 1;
          col[2].j   = ey;
          col[2].loc = LEFT;
          col[2].c   = 0;
          valA[2]    = dv * 1.0 / (hx * hx);
          col[3].i   = ex + 1;
          col[3].j   = ey;
          col[3].loc = LEFT;
          col[3].c   = 0;
          valA[3]    = dv * 1.0 / (hx * hx);
          if (hasPressure) {
            nEntries += 2;
            col[4].i   = ex - 1;
            col[4].j   = ey;
            col[4].loc = ELEMENT;
            col[4].c   = 0;
            valA[4]    = dv * 1.0 / hx;
            col[5].i   = ex;
            col[5].j   = ey;
            col[5].loc = ELEMENT;
            col[5].c   = 0;
            valA[5]    = -dv * 1.0 / hx;
          }
        } else {
          /* Note how this is identical to the stencil for U_y, with "DOWN" replaced by "LEFT" and the pressure derivative in the other direction */
          nEntries   = 5;
          col[0].i   = ex;
          col[0].j   = ey;
          col[0].loc = LEFT;
          col[0].c   = 0;
          valA[0]    = -dv * 2.0 / (hx * hx) + -dv * 2.0 / (hy * hy);
          col[1].i   = ex;
          col[1].j   = ey - 1;
          col[1].loc = LEFT;
          col[1].c   = 0;
          valA[1]    = dv * 1.0 / (hy * hy);
          col[2].i   = ex;
          col[2].j   = ey + 1;
          col[2].loc = LEFT;
          col[2].c   = 0;
          valA[2]    = dv * 1.0 / (hy * hy);
          col[3].i   = ex - 1;
          col[3].j   = ey;
          col[3].loc = LEFT;
          col[3].c   = 0;
          valA[3]    = dv * 1.0 / (hx * hx);
          col[4].i   = ex + 1;
          col[4].j   = ey;
          col[4].loc = LEFT;
          col[4].c   = 0;
          valA[4]    = dv * 1.0 / (hx * hx);
          if (hasPressure) {
            nEntries += 2;
            col[5].i   = ex - 1;
            col[5].j   = ey;
            col[5].loc = ELEMENT;
            col[5].c   = 0;
            valA[5]    = dv * 1.0 / hx;
            col[6].i   = ex;
            col[6].j   = ey;
            col[6].loc = ELEMENT;
            col[6].c   = 0;
            valA[6]    = -dv * 1.0 / hx;
          }
        }
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, nEntries, col, valA, INSERT_VALUES));
        if (rhs) {
          valRhs = dv * fx(cArrX[ex][iprev], cArrY[ey][icenter]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      }

      /* P equation : u_x + v_y = g
         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
      if (hasPressure) {
        DMStagStencil row, col[5];
        PetscScalar   valA[5], valRhs;

        row.i   = ex;
        row.j   = ey;
        row.loc = ELEMENT;
        row.c   = 0;
        /* Note: the scaling by dv here may not be optimal (but this test isn't concerned with these equations) */
        col[0].i   = ex;
        col[0].j   = ey;
        col[0].loc = LEFT;
        col[0].c   = 0;
        valA[0]    = -dv * 1.0 / hx;
        col[1].i   = ex;
        col[1].j   = ey;
        col[1].loc = RIGHT;
        col[1].c   = 0;
        valA[1]    = dv * 1.0 / hx;
        col[2].i   = ex;
        col[2].j   = ey;
        col[2].loc = DOWN;
        col[2].c   = 0;
        valA[2]    = -dv * 1.0 / hy;
        col[3].i   = ex;
        col[3].j   = ey;
        col[3].loc = UP;
        col[3].c   = 0;
        valA[3]    = dv * 1.0 / hy;
        col[4]     = row;
        valA[4]    = dv * 0.0;
        PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 5, col, valA, INSERT_VALUES));
        if (rhs) {
          valRhs = dv * g(cArrX[ex][icenter], cArrY[ey][icenter]);
          PetscCall(DMStagVecSetValuesStencil(dm, rhs, 1, &row, &valRhs, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &cArrX, &cArrY, NULL));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (rhs) {
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(VecAssemblyEnd(rhs));
  }

  PetscFunctionReturn(0);
}

/* A custom monitor function for analysis purposes. Computes and dumps
   residuals and errors for each KSP iteration */
PetscErrorCode DMStagAnalysisKSPMonitor(KSP ksp, PetscInt it, PetscReal rnorm, void *mctx)
{
  DM                               dm;
  Vec                              r, sol;
  DMStagAnalysisKSPMonitorContext *ctx = (DMStagAnalysisKSPMonitorContext *)mctx;

  PetscFunctionBeginUser;
  PetscCall(KSPBuildSolution(ksp, NULL, &sol)); /* don't destroy sol */
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &r));
  dm = ctx->dm; /* Would typically get this with VecGetDM(), KSPGetDM() */
  PetscCheck(dm, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Analaysis monitor requires a DM which is properly associated with the solution Vec");
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), " *** DMStag Analysis KSP Monitor (it. %" PetscInt_FMT ") ***\n", it));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), " Residual Norm: %g\n", (double)rnorm));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), " Dumping files...\n"));

  /* Note: these blocks are almost entirely duplicated */
  {
    const DMStagStencilLocation loc = DMSTAG_LEFT;
    const PetscInt              c   = 0;
    Vec                         vec;
    DM                          da;
    PetscViewer                 viewer;
    char                        filename[PETSC_MAX_PATH_LEN];

    PetscCall(DMStagVecSplitToDMDA(dm, r, loc, c, &da, &vec));
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "res_vx_%" PetscInt_FMT ".vtr", it));
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(vec, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&vec));
    PetscCall(DMDestroy(&da));
  }

  {
    const DMStagStencilLocation loc = DMSTAG_DOWN;
    const PetscInt              c   = 0;
    Vec                         vec;
    DM                          da;
    PetscViewer                 viewer;
    char                        filename[PETSC_MAX_PATH_LEN];

    PetscCall(DMStagVecSplitToDMDA(dm, r, loc, c, &da, &vec));
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "res_vy_%" PetscInt_FMT ".vtr", it));
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(vec, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&vec));
    PetscCall(DMDestroy(&da));
  }

  if (ctx->solRef) {
    Vec e;
    PetscCall(VecDuplicate(ctx->solRef, &e));
    PetscCall(VecCopy(ctx->solRef, e));
    PetscCall(VecAXPY(e, -1.0, sol));

    {
      const DMStagStencilLocation loc = DMSTAG_LEFT;
      const PetscInt              c   = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      PetscCall(DMStagVecSplitToDMDA(dm, e, loc, c, &da, &vec));
      PetscCall(PetscSNPrintf(filename, sizeof(filename), "err_vx_%" PetscInt_FMT ".vtr", it));
      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(vec, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDestroy(&vec));
      PetscCall(DMDestroy(&da));
    }

    {
      const DMStagStencilLocation loc = DMSTAG_DOWN;
      const PetscInt              c   = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      PetscCall(DMStagVecSplitToDMDA(dm, e, loc, c, &da, &vec));
      PetscCall(PetscSNPrintf(filename, sizeof(filename), "err_vy_%" PetscInt_FMT ".vtr", it));
      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(vec, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDestroy(&vec));
      PetscCall(DMDestroy(&da));
    }

    PetscCall(VecDestroy(&e));
  }

  /* Repeat error computations wrt an "exact" solution to the discrete equations */
  if (ctx->solRefNum) {
    Vec e;
    PetscCall(VecDuplicate(ctx->solRefNum, &e));
    PetscCall(VecCopy(ctx->solRefNum, e));
    PetscCall(VecAXPY(e, -1.0, sol));

    {
      const DMStagStencilLocation loc = DMSTAG_LEFT;
      const PetscInt              c   = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      PetscCall(DMStagVecSplitToDMDA(dm, e, loc, c, &da, &vec));
      PetscCall(PetscSNPrintf(filename, sizeof(filename), "err_num_vx_%" PetscInt_FMT ".vtr", it));
      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(vec, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDestroy(&vec));
      PetscCall(DMDestroy(&da));
    }

    {
      const DMStagStencilLocation loc = DMSTAG_DOWN;
      const PetscInt              c   = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      PetscCall(DMStagVecSplitToDMDA(dm, e, loc, c, &da, &vec));
      PetscCall(PetscSNPrintf(filename, sizeof(filename), "err_num_vy_%" PetscInt_FMT ".vtr", it));
      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(vec, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDestroy(&vec));
      PetscCall(DMDestroy(&da));
    }

    PetscCall(VecDestroy(&e));
  }

  /* Step */
  if (!ctx->solPrev) {
    PetscCall(VecDuplicate(sol, &ctx->solPrev));
  } else {
    Vec diff;
    PetscCall(VecDuplicate(sol, &diff));
    PetscCall(VecCopy(sol, diff));
    PetscCall(VecAXPY(diff, -1.0, ctx->solPrev));
    {
      const DMStagStencilLocation loc = DMSTAG_LEFT;
      const PetscInt              c   = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      PetscCall(DMStagVecSplitToDMDA(dm, diff, loc, c, &da, &vec));
      PetscCall(PetscSNPrintf(filename, sizeof(filename), "diff_vx_%" PetscInt_FMT ".vtr", it));
      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(vec, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDestroy(&vec));
      PetscCall(DMDestroy(&da));
    }

    {
      const DMStagStencilLocation loc = DMSTAG_DOWN;
      const PetscInt              c   = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      PetscCall(DMStagVecSplitToDMDA(dm, diff, loc, c, &da, &vec));
      PetscCall(PetscSNPrintf(filename, sizeof(filename), "diff_vy_%" PetscInt_FMT ".vtr", it));
      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)r), filename, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(vec, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDestroy(&vec));
      PetscCall(DMDestroy(&da));
    }
    PetscCall(VecDestroy(&diff));
  }
  PetscCall(VecCopy(sol, ctx->solPrev));

  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(0);
}

/* Use a direct solver to create an "exact" solution to the discrete system
   useful for testing solvers (in that it doesn't include discretization error) */
static PetscErrorCode CreateNumericalReferenceSolution(Mat A, Vec rhs, Vec *px)
{
  KSP ksp;
  PC  pc;
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)A), &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERUMFPACK));
  PetscCall(KSPSetOptionsPrefix(ksp, "numref_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(VecDuplicate(rhs, px));
  x = *px;
  PetscCall(KSPSolve(ksp, rhs, x));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: gmg_1
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason

   test:
      suffix: gmg_1_b
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractTransferOperators false

   test:
      suffix: gmg_1_c
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false

   test:
      suffix: gmg_1_d
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false -extractTransferOperators false

   test:
      suffix: gmg_1_bigger
      requires: !complex
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -stag_grid_x 32 -stag_grid_y 32

   test:
      suffix: gmg_8
      requires: !single !complex
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason

   test:
      suffix: gmg_8_b
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractTransferOperators false

   test:
      suffix: gmg_8_c
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false

   test:
      suffix: gmg_8_d
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false -extractTransferOperators false

   test:
      suffix: gmg_8_galerkin
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false -extractTransferOperators false -pc_mg_galerkin

   test:
      suffix: gmg_8_bigger
      requires: !complex
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -stag_grid_x 32 -stag_grid_y 32

TEST*/
