static char help[] = "Solve a toy 1D problem on a staggered grid.\n\
                      Accepts command line options -a, -b, and -c\n\
                      and approximately solves\n\
                      u''(x) = c, u(0) = 1, u(1) = b\n\n";
/*

   To demonstrate the basic functionality of DMStag, solves a second-order ODE,

       u''(x) = f(x),  0 < x < 1
       u(0) = a
       u(1) = b

   in mixed form, that is by introducing an auxiliary variable p

      p'(x) = f(x), p - u'(x) = 0, 0 < x < 1
      u(0) = a
      u(1) = b

   For f == c, the solution is

     u(x) = a + (b - a - (c/2)) x + (c/2) x^2
     p(x) = (b - a - (c/2)) + c x

   To find an approximate solution, discretize by storing values of p in
   elements and values of u on their boundaries, and using first-order finite
   differences.

   This should in fact produce a (nodal) solution with no discretization error,
   so differences from the reference solution will be restricted to those induced
   by floating point operations (in particular, the finite differences) and the
   accuracy of the linear solve.

   Parameters for the main grid can be controlled with command line options, e.g.

     -stag_grid_x 10

  In particular to notice in this example are the two methods of indexing. The
  first is analogous to the use of MatStencil with DMDA, and the second is
  analogous to the use of DMDAVecGetArrayDOF().

  The first, recommended for ease of use, is based on naming an element in the
  global grid, a location in its support, and a component. For example,
  one might consider element e, the left side (a vertex in 1d), and the first
  component (index 0). This is accomplished by populating a DMStagStencil struct,
  e.g.

      DMStagStencil stencil;
      stencil.i   = i
      stencil.loc = DMSTAG_LEFT;
      stencil.c   = 0

  Note that below, for convenenience, we #define an alias LEFT for DMSTAG_LEFT.

  The second, which ensures maximum efficiency, makes use of the underlying
  block structure of local DMStag-derived vectors, and requires the user to
  obtain the correct local offset for the degrees of freedom they would like to
  use. This is made simple with the helper function DMStagGetLocationSlot().

  Note that the linear system being solved is indefinite, so is not entirely
  trivial to invert. The default solver here (GMRES/Jacobi) is a poor choice,
  made to avoid depending on an external package. To solve a larger system,
  the usual method for a 1-d problem such as this is to choose a sophisticated
  direct solver, e.g. configure --download-suitesparse and run

    $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n 3 ./stag_ex2 -stag_grid_x 100 -pc_type lu -pc_factor_mat_solver_package umfpack

  You can also impose a periodic boundary condition, in which case -b and -c are
  ignored; b = a and c = 0.0 are used, giving a constant u == a , p == 0.

      -stag_boundary_type_x periodic

*/
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>

/* Shorter, more convenient names for DMStagStencilLocation entries */
#define LEFT    DMSTAG_LEFT
#define RIGHT   DMSTAG_RIGHT
#define ELEMENT DMSTAG_ELEMENT

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  DM                dmSol,dmForcing;
  DM                dmCoordSol;
  Vec               sol,solRef,solRefLocal,f,fLocal,rhs,coordSolLocal;
  Mat               A;
  PetscScalar       a,b,c,h;
  KSP               ksp;
  PC                pc;
  PetscInt          start,n,e,nExtra;
  PetscInt          iu,ip,ixu,ixp;
  PetscBool         isLastRank,isFirstRank;
  PetscScalar       **arrSol,**arrCoordSol;
  DMBoundaryType    boundary;

  const PetscReal domainSize = 1.0;

  /* Initialize PETSc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Create 1D DMStag for the solution, and set up. Note that you can supply many
     command line options (see the man page for DMStagCreate1d)
  */
  ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dmSol);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmSol);CHKERRQ(ierr);
  ierr = DMSetUp(dmSol);CHKERRQ(ierr);

  /* Create uniform coordinates. Note that in higher-dimensional examples,
      coordinates are created differently.*/
  ierr = DMStagSetUniformCoordinatesExplicit(dmSol,0.0,domainSize,0.0,0.0,0.0,0.0);CHKERRQ(ierr);

  /* Determine boundary type */
  ierr = DMStagGetBoundaryTypes(dmSol,&boundary,NULL,NULL);CHKERRQ(ierr);

  /* Process command line options (depends on DMStag setup) */
  a = 1.0; b = 2.0; c = 1.0;
  ierr = PetscOptionsGetScalar(NULL,NULL,"-a",&a,NULL);CHKERRQ(ierr);
  if (boundary == DM_BOUNDARY_PERIODIC) {
    b = a;
    c = 0.0;
  } else {
    ierr = PetscOptionsGetScalar(NULL,NULL,"-b",&b,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL,NULL,"-c",&c,NULL);CHKERRQ(ierr);
  }

  /* Compute reference solution on the grid, using direct array access */
  ierr = DMCreateGlobalVector(dmSol,&solRef);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmSol,&dmCoordSol);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmSol,&coordSolLocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmCoordSol,coordSolLocal,&arrCoordSol);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmSol,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL);CHKERRQ(ierr);

  /* Get the correct entries for each of our variables in local element-wise storage */
  ierr = DMStagGetLocationSlot(dmSol,LEFT,   0,&iu);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoordSol,LEFT,   0,&ixu);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoordSol,ELEMENT,0,&ixp);CHKERRQ(ierr);
  for (e=start; e<start + n + nExtra; ++e) {
    {
      const PetscScalar coordu = arrCoordSol[e][ixu];
      arrSol[e][iu] = a  + (b - a - (c/2.0)) * coordu + (c/2.0)*coordu*coordu;
    }
    if (e < start+n) {
      const PetscScalar coordp = arrCoordSol[e][ixp];
      arrSol[e][ip] = b - a - (c/2.0) + c * coordp;
    }
  }
  ierr = DMStagVecRestoreArrayRead(dmCoordSol,coordSolLocal,&arrCoordSol);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmSol,solRefLocal,INSERT_VALUES,solRef);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);

  /* Create another 1D DMStag for the forcing term, and populate a field on it.
     Here this is not really necessary, but in other contexts we may have auxiliary
     fields which we use to construct the linear system.

     This second DM represents the same physical domain, but has a different
     "default section" (though the current implementation does NOT actually use
     PetscSection). Since it is created as a derivative of the original DMStag,
     we can be confident that it is compatible. One could check with DMGetCompatiblity() */
  ierr = DMStagCreateCompatibleDMStag(dmSol,1,0,0,0,&dmForcing);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmForcing,&f);CHKERRQ(ierr);
  ierr = VecSet(f,c);CHKERRQ(ierr); /* Dummy for logic which depends on auxiliary data */

  /* Assemble System */
  ierr = DMCreateMatrix(dmSol,&A);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmSol,&rhs);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmForcing,&fLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmForcing,f,INSERT_VALUES,fLocal);CHKERRQ(ierr);

  /* Note: if iterating over all the elements, you will usually need to do something
     special at one of the boundaries. You can either make use of the existence
     of a "extra" partial dummy element on the right/top/front, or you can use a different stencil.
     The construction of the reference solution above uses the first method,
     so here we will use the second */

  ierr = DMStagGetIsLastRank(dmSol,&isLastRank,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetIsFirstRank(dmSol,&isFirstRank,NULL,NULL);CHKERRQ(ierr);
  for (e = start; e<start+n; ++e) {
    DMStagStencil pos[3];
    PetscScalar   val[3];
    PetscInt      idxLoc;

    idxLoc = 0;
    pos[idxLoc].i   = e;       /* This element in the 1d ordering */
    pos[idxLoc].loc = ELEMENT; /* Element-centered dofs (p) */
    pos[idxLoc].c   = 0;       /* Component 0 : first (and only) p dof */
    val[idxLoc]     = 0.0;     /* p - u'(x) = 0 */
    ++idxLoc;

    if (isFirstRank && e == start) {
      /* Special case on left boundary */
      pos[idxLoc].i   = e;     /* This element in the 1d ordering */
      pos[idxLoc].loc = LEFT;  /* Left vertex */
      pos[idxLoc].c   = 0;
      val[idxLoc]     = a;     /* u(0) = a */
      ++idxLoc;
    } else {
      PetscScalar fVal;
      /* Usual case - deal with velocity on left side of cell
         Here, we obtain a value of f (even though it's constant here,
         this demonstrates the more-realistic case of a pre-computed coefficient) */
      pos[idxLoc].i   = e;     /* This element in the 1d ordering */
      pos[idxLoc].loc = LEFT;  /* vertex-centered dof (u) */
      pos[idxLoc].c   = 0;

      ierr = DMStagVecGetValuesStencil(dmForcing,fLocal,1,&pos[idxLoc],&fVal);CHKERRQ(ierr);

      val[idxLoc]     = fVal;     /* p'(x) = f, in interior */
      ++idxLoc;
    }
    if (boundary != DM_BOUNDARY_PERIODIC && isLastRank && e == start+n-1) {
      /* Special case on right boundary (in addition to usual case) */
      pos[idxLoc].i   = e;     /* This element in the 1d ordering */
      pos[idxLoc].loc = RIGHT;
      pos[idxLoc].c   = 0;
      val[idxLoc]     = b;     /* u(1) = b */
      ++idxLoc;
    }
    ierr = DMStagVecSetValuesStencil(dmSol,rhs,idxLoc,pos,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dmForcing,&fLocal);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);

  /* Note: normally it would be more efficient to assemble the RHS and the matrix
     in the same loop over elements, but we separate them for clarity here */
  ierr = DMGetCoordinatesLocal(dmSol,&coordSolLocal);CHKERRQ(ierr);
  for (e = start; e<start+n; ++e) {

    /* Velocity is either a BC or an interior point */
    if (isFirstRank && e == start) {
      DMStagStencil row;
      PetscScalar   val;

      row.i   = e;
      row.loc = LEFT;
      row.c   = 0;
      val     = 1.0;
      ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      DMStagStencil row,col[3];
      PetscScalar   val[3],xp[2];

      row.i      = e;
      row.loc    = LEFT; /* In general, opt for LEFT/DOWN/BACK  and iterate over elements */
      row.c      = 0;

      col[0].i   = e;
      col[0].loc = ELEMENT;
      col[0].c   = 0;

      col[1].i   = e-1;
      col[1].loc = ELEMENT;
      col[1].c   = 0;

      ierr = DMStagVecGetValuesStencil(dmCoordSol,coordSolLocal,2,col,xp);CHKERRQ(ierr);
      h = xp[0]- xp[1];
      if (boundary == DM_BOUNDARY_PERIODIC && PetscRealPart(h) < 0.0) h += domainSize;

      val[0]     = 1.0/h;
      val[1]     = -1.0/h;

      /* For convenience, we add an explicit 0 on the diagonal. This is a waste,
         but it allows for easier use of a direct solver, if desired */
      col[2].i   = e;
      col[2].loc = LEFT;
      col[2].c   = 0;
      val[2]     = 0.0;

      ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }

    /* Additional velocity point (BC) on the right */
    if (isLastRank && e == start+n-1) {
      DMStagStencil row;
      PetscScalar   val;

      row.i = e;
      row.loc = RIGHT;
      row.c = 0;
      val = 1.0;
      ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    }

    /* Equation on pressure (element) variables */
    {
      DMStagStencil row,col[3];
      PetscScalar   val[3],xu[2];

      row.i      = e;
      row.loc    = ELEMENT;
      row.c      = 0;

      col[0].i = e;
      col[0].loc = RIGHT;
      col[0].c   = 0;

      col[1].i = e;
      col[1].loc = LEFT;
      col[1].c   = 0;

      ierr = DMStagVecGetValuesStencil(dmCoordSol,coordSolLocal,2,col,xu);CHKERRQ(ierr);
      h = xu[0]- xu[1];
      if (boundary == DM_BOUNDARY_PERIODIC && PetscRealPart(h) < 0.0) h += domainSize;

      val[0]     = -1.0/h;
      val[1]     = 1.0/h;

      col[2].i   = e;
      col[2].loc = ELEMENT;
      col[2].c   = 0;
      val[2]     = 1.0;

      ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Solve */
  ierr = DMCreateGlobalVector(dmSol,&sol);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr); /* A simple, but non-scalable, solver choice */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);

  /* View the components of the solution, demonstrating DMStagMigrateVec() */
  {
    DM  dmVertsOnly,dmElementsOnly;
    Vec u,p;

    ierr = DMStagCreateCompatibleDMStag(dmSol,1,0,0,0,&dmVertsOnly);CHKERRQ(ierr);
    ierr = DMStagCreateCompatibleDMStag(dmSol,0,1,0,0,&dmElementsOnly);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmVertsOnly,&u);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmElementsOnly,&p);CHKERRQ(ierr);

    ierr = DMStagMigrateVec(dmSol,sol,dmVertsOnly,u);CHKERRQ(ierr);
    ierr = DMStagMigrateVec(dmSol,sol,dmElementsOnly,p);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)u,"Sol_u");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)p,"Sol_p");CHKERRQ(ierr);
    ierr = VecView(p,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = DMRestoreGlobalVector(dmVertsOnly,&u);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmElementsOnly,&p);CHKERRQ(ierr);
    ierr = DMDestroy(&dmVertsOnly);CHKERRQ(ierr);
    ierr = DMDestroy(&dmElementsOnly);CHKERRQ(ierr);
  }

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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error (abs): %g\nError (rel): %g\n",(double)errAbs,(double)errRel);CHKERRQ(ierr);
    ierr = VecDestroy(&diff);CHKERRQ(ierr);
  }

  /* Clean up and finalize PETSc */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  ierr = VecDestroy(&solRef);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSol);CHKERRQ(ierr);
  ierr = DMDestroy(&dmForcing);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 7
      args: -dm_view -stag_grid_x 11 -stag_stencil_type star -a 1.33 -b 7.22 -c 347.2 -ksp_monitor_short

   test:
      suffix: periodic
      nsize: 3
      args: -dm_view -stag_grid_x 13 -stag_boundary_type_x periodic -a 1.1234

   test:
      suffix: periodic_seq
      nsize: 1
      args: -dm_view -stag_grid_x 13 -stag_boundary_type_x periodic -a 1.1234

   test:
      suffix: ghosted_vacuous
      nsize: 3
      args: -dm_view -stag_grid_x 13 -stag_boundary_type_x ghosted -a 1.1234

TEST*/
