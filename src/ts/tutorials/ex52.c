static char help[] = "Simple Advection-diffusion equation solved using FVM in DMPLEX\n";

/*
   Solves the simple advection equation given by

   q_t + u (q_x) + v (q_y) - D (q_xx + q_yy) = 0 using FVM and First Order Upwind discretization.

   with a user defined initial condition.

   with dirichlet/neumann conditions on the four boundaries of the domain.

   User can define the mesh parameters either in the command line or inside
   the ProcessOptions() routine.

   Contributed by: Mukkund Sunjii, Domenico Lahaye
*/

#include <petscdmplex.h>
#include <petscts.h>
#include <petscblaslapack.h>

#if defined(PETSC_HAVE_CGNS)
#undef I
#include <cgnslib.h>
#endif
/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS, PetscReal, Vec, Vec, void *), FormInitialSolution(DM, Vec);
extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void *);
extern PetscErrorCode MySNESMonitor(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);

/* Defining the usr defined context */
typedef struct {
    PetscScalar diffusion;
    PetscReal   u, v;
    PetscScalar delta_x, delta_y;
} AppCtx;

/* Options for the scenario */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
    PetscFunctionBeginUser;
    options->u = 2.5;
    options->v = 0.0;
    options->diffusion = 0.0;
    PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
    PetscCall(PetscOptionsReal("-u", "The x component of the convective coefficient", "advection_DMPLEX.c", options->u, &options->u, NULL));
    PetscCall(PetscOptionsReal("-v", "The y component of the convective coefficient", "advection_DMPLEX.c", options->v, &options->v, NULL));
    PetscCall(PetscOptionsScalar("-diffus", "The diffusive coefficient", "advection_DMPLEX.c", options->diffusion, &options->diffusion, NULL));
    PetscOptionsEnd();
    PetscFunctionReturn(0);
}

/*
  User can provide the file containing the mesh.
  Or can generate the mesh using DMPlexCreateBoxMesh with the specified options.
*/
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
    PetscFunctionBeginUser;
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
    {
      DMLabel label;
      PetscCall(DMGetLabel(*dm, "boundary", &label));
      PetscCall(DMPlexLabelComplete(*dm, label));
    }
    PetscFunctionReturn(0);
}

    /* This routine is responsible for defining the local solution vector x
    with a given initial solution.
    The initial solution can be modified accordingly inside the loops.
    No need for a local vector because there is exchange of information
    across the processors. Unlike for FormFunction which depends on the neighbours */
PetscErrorCode FormInitialSolution(DM da, Vec U)
{
    PetscScalar    *u;
    PetscInt       cell, cStart, cEnd;
    PetscReal      cellvol, centroid[3], normal[3];

    PetscFunctionBeginUser;
    /* Get pointers to vector data */
    PetscCall(VecGetArray(U, &u));
    /* Get local grid boundaries */
    PetscCall(DMPlexGetHeightStratum(da, 0, &cStart, &cEnd));
    /* Assigning the values at the cell centers based on x and y directions */
    PetscCall(DMGetCoordinatesLocalSetUp(da));
    for (cell = cStart; cell < cEnd; cell++) {
        PetscCall(DMPlexComputeCellGeometryFVM(da, cell, &cellvol, centroid, normal));
        if (centroid[0] > 0.9 && centroid[0] < 0.95) {
            if (centroid[1] > 0.9 && centroid[1] < 0.95) u[cell] = 2.0;
        }
        else u[cell] = 0;
    }
    PetscCall(VecRestoreArray(U, &u));
    PetscFunctionReturn(0);
}

PetscErrorCode MyTSMonitor(TS ts, PetscInt step, PetscReal ptime, Vec v, void *ctx)
{
    PetscReal      norm;
    MPI_Comm       comm;

    PetscFunctionBeginUser;
    if (step < 0) PetscFunctionReturn(0); /* step of -1 indicates an interpolated solution */
    PetscCall(VecNorm(v, NORM_2, &norm));
    PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
    PetscCall(PetscPrintf(comm, "timestep %" PetscInt_FMT " time %g norm %g\n", step, (double) ptime, (double) norm));
    PetscFunctionReturn(0);
}

/*
   MySNESMonitor - illustrate how to set user-defined monitoring routine for SNES.
   Input Parameters:
     snes - the SNES context
     its - iteration number
     fnorm - 2-norm function value (may be estimated)
     ctx - optional user-defined context for private data for the
         monitor routine, as set by SNESMonitorSet()
*/
PetscErrorCode MySNESMonitor(SNES snes, PetscInt its, PetscReal fnorm, PetscViewerAndFormat *vf)
{
    PetscFunctionBeginUser;
    PetscCall(SNESMonitorDefaultShort(snes, its, fnorm, vf));
    PetscFunctionReturn(0);
}

/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(TS ts, PetscReal ftime, Vec X, Vec F, void *ctx)
{
    AppCtx *user = (AppCtx *) ctx;
    DM da;
    PetscScalar *x, *f;
    Vec localX;
    PetscInt fStart, fEnd, nF;
    PetscInt cell, cStart, cEnd, nC;
    DM dmFace;      /* DMPLEX for face geometry */
    PetscFV fvm;                /* specify type of FVM discretization */
    Vec cellGeom, faceGeom; /* vector of structs related to cell/face geometry*/
    const PetscScalar *fgeom;             /* values stored in the vector facegeom */
    PetscFVFaceGeom *fgA;               /* struct with face geometry information */
    const PetscInt *cellcone, *cellsupport;
    PetscScalar flux_east, flux_west, flux_north, flux_south, flux_centre;
    PetscScalar centroid_x[2], centroid_y[2], boundary = 0.0;
    PetscScalar boundary_left = 0.0;
    PetscReal u_plus, u_minus, v_plus, v_minus, zero = 0.0;
    PetscScalar delta_x, delta_y;

    /* Get the local vector from the DM object. */
    PetscFunctionBeginUser;
    PetscCall(TSGetDM(ts, &da));
    PetscCall(DMGetLocalVector(da, &localX));

    /* Scatter ghost points to local vector,using the 2-step process
       DMGlobalToLocalBegin(),DMGlobalToLocalEnd(). */
    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
    /* Get pointers to vector data. */
    PetscCall(VecGetArray(localX, &x));
    PetscCall(VecGetArray(F, &f));

    /* Obtaining local cell and face ownership */
    PetscCall(DMPlexGetHeightStratum(da, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetHeightStratum(da, 1, &fStart, &fEnd));

    /* Creating the PetscFV object to obtain face and cell geometry.
    Later to be used to compute face centroid to find cell widths. */

    PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
    PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND));
    /*....Retrieve precomputed cell geometry....*/
    PetscCall(DMPlexGetDataFVM(da, fvm, &cellGeom, &faceGeom, NULL));
    PetscCall(VecGetDM(faceGeom, &dmFace));
    PetscCall(VecGetArrayRead(faceGeom, &fgeom));

    /* Spanning through all the cells and an inner loop through the faces. Find the
    face neighbors and pick the upwinded cell value for flux. */

    u_plus = PetscMax(user->u, zero);
    u_minus = PetscMin(user->u, zero);
    v_plus = PetscMax(user->v, zero);
    v_minus = PetscMin(user->v, zero);

    for (cell = cStart; cell < cEnd; cell++) {
        /* Obtaining the faces of the cell */
        PetscCall(DMPlexGetConeSize(da, cell, &nF));
        PetscCall(DMPlexGetCone(da, cell, &cellcone));

        /* south */
        PetscCall(DMPlexPointLocalRead(dmFace, cellcone[0], fgeom, &fgA));
        centroid_y[0] = fgA->centroid[1];
        /* North */
        PetscCall(DMPlexPointLocalRead(dmFace, cellcone[2], fgeom, &fgA));
        centroid_y[1] = fgA->centroid[1];
        /* West */
        PetscCall(DMPlexPointLocalRead(dmFace, cellcone[3], fgeom, &fgA));
        centroid_x[0] = fgA->centroid[0];
        /* East */
        PetscCall(DMPlexPointLocalRead(dmFace, cellcone[1], fgeom, &fgA));
        centroid_x[1] = fgA->centroid[0];

        /* Computing the cell widths in the x and y direction */
        delta_x = centroid_x[1] - centroid_x[0];
        delta_y = centroid_y[1] - centroid_y[0];

        /* Getting the neighbors of each face
           Going through the faces by the order (cellcone) */

        /* cellcone[0] - south */
        PetscCall(DMPlexGetSupportSize(da, cellcone[0], &nC));
        PetscCall(DMPlexGetSupport(da, cellcone[0], &cellsupport));
        if (nC == 2) flux_south = (x[cellsupport[0]] * (-v_plus - user->diffusion * delta_x)) / delta_y;
        else flux_south = (boundary * (-v_plus - user->diffusion * delta_x)) / delta_y;

        /* cellcone[1] - east */
        PetscCall(DMPlexGetSupportSize(da, cellcone[1], &nC));
        PetscCall(DMPlexGetSupport(da, cellcone[1], &cellsupport));
        if (nC == 2) flux_east = (x[cellsupport[1]] * (u_minus - user->diffusion * delta_y)) / delta_x;
        else flux_east = (boundary * (u_minus - user->diffusion * delta_y)) / delta_x;

        /* cellcone[2] - north */
        PetscCall(DMPlexGetSupportSize(da, cellcone[2], &nC));
        PetscCall(DMPlexGetSupport(da, cellcone[2], &cellsupport));
        if (nC == 2) flux_north = (x[cellsupport[1]] * (v_minus - user->diffusion * delta_x)) / delta_y;
        else flux_north = (boundary * (v_minus - user->diffusion * delta_x)) / delta_y;

        /* cellcone[3] - west */
        PetscCall(DMPlexGetSupportSize(da, cellcone[3], &nC));
        PetscCall(DMPlexGetSupport(da, cellcone[3], &cellsupport));
        if (nC == 2) flux_west = (x[cellsupport[0]] * (-u_plus - user->diffusion * delta_y)) / delta_x;
        else flux_west = (boundary_left * (-u_plus - user->diffusion * delta_y)) / delta_x;

        /* Contribution by the cell to the fluxes */
        flux_centre = x[cell] * ((u_plus - u_minus + 2 * user->diffusion * delta_y) / delta_x +
                                 (v_plus - v_minus + 2 * user->diffusion * delta_x) / delta_y);

        /* Calculating the net flux for each cell
           and computing the RHS time derivative f[.] */
        f[cell] = -(flux_centre + flux_east + flux_west + flux_north + flux_south);
    }
    PetscCall(PetscFVDestroy(&fvm));
    PetscCall(VecRestoreArray(localX, &x));
    PetscCall(VecRestoreArray(F, &f));
    PetscCall(DMRestoreLocalVector(da, &localX));
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    TS                   ts;                         /* time integrator */
    SNES                 snes;
    Vec                  x, r;                        /* solution, residual vectors */
    DM                   da;
    PetscMPIInt          rank;
    PetscViewerAndFormat *vf;
    AppCtx               user;                             /* mesh context */
    PetscInt             dim, numFields = 1, numBC, i;
    PetscInt             numComp[1];
    PetscInt             numDof[12];
    PetscInt             bcField[1];
    PetscSection         section;
    IS                   bcPointIS[1];

    /* Initialize program */
    PetscCall(PetscInitialize(&argc, &argv, (char *) 0, help));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    /* Create distributed array (DMPLEX) to manage parallel grid and vectors */
    PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
    PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &da));
    PetscCall(DMGetDimension(da, &dim));

    /* Specifying the fields and dof for the formula through PETSc Section
    Create a scalar field u with 1 component on cells, faces and edges.
    Alternatively, the field information could be added through a PETSCFV object
    using DMAddField(...).*/
    numComp[0] = 1;

    for (i = 0; i < numFields * (dim + 1); ++i) numDof[i] = 0;

    numDof[0 * (dim + 1)] = 1;
    numDof[0 * (dim + 1) + dim - 1] = 1;
    numDof[0 * (dim + 1) + dim] = 1;

    /* Setup boundary conditions */
    numBC = 1;
    /* Prescribe a Dirichlet condition on u on the boundary
       Label "marker" is made by the mesh creation routine  */
    bcField[0] = 0;
    PetscCall(DMGetStratumIS(da, "marker", 1, &bcPointIS[0]));

    /* Create a PetscSection with this data layout */
    PetscCall(DMSetNumFields(da, numFields));
    PetscCall(DMPlexCreateSection(da, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &section));

    /* Name the Field variables */
    PetscCall(PetscSectionSetFieldName(section, 0, "u"));

    /* Tell the DM to use this section (with the specified fields and dof) */
    PetscCall(DMSetLocalSection(da, section));

    /* Extract global vectors from DMDA; then duplicate for remaining
       vectors that are the same types */

    /* Create a Vec with this layout and view it */
    PetscCall(DMGetGlobalVector(da, &x));
    PetscCall(VecDuplicate(x, &r));

    /* Create timestepping solver context */
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
    PetscCall(TSSetRHSFunction(ts, NULL, FormFunction, &user));

    PetscCall(TSSetMaxTime(ts, 1.0));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
    PetscCall(TSMonitorSet(ts, MyTSMonitor, PETSC_VIEWER_STDOUT_WORLD, NULL));
    PetscCall(TSSetDM(ts, da));

    /* Customize nonlinear solver */
    PetscCall(TSSetType(ts, TSEULER));
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf));
    PetscCall(SNESMonitorSet(snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *)) MySNESMonitor, vf,(PetscErrorCode (*)(void **)) PetscViewerAndFormatDestroy));

     /* Set initial conditions */
    PetscCall(FormInitialSolution(da, x));
    PetscCall(TSSetTimeStep(ts, .0001));
    PetscCall(TSSetSolution(ts, x));
    /* Set runtime options */
    PetscCall(TSSetFromOptions(ts));
    /* Solve nonlinear system */
    PetscCall(TSSolve(ts, x));

    /* Clean up routine */
    PetscCall(DMRestoreGlobalVector(da, &x));
    PetscCall(ISDestroy(&bcPointIS[0]));
    PetscCall(PetscSectionDestroy(&section));
    PetscCall(VecDestroy(&r));
    PetscCall(TSDestroy(&ts));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

    test:
      suffix: 0
      args: -dm_plex_simplex 0 -dm_plex_box_faces 20,20 -dm_plex_boundary_label boundary -ts_max_steps 5 -ts_type rk
      requires: !single !complex triangle ctetgen

TEST*/
