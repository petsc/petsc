static char help[] = "Poisson Problem in 2d with tensor product finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel block-structured mesh (DMFOREST) to discretize it.\n\n\n";

//#include <petscdmforest.h>
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

#ifdef PETSC_HAVE_P4EST
  #include <p4est_bits.h>
  #include <p4est_ghost.h>
  #include <p4est_lnodes.h>
#else
  #error Please install p4est using --download-p4est
#endif

PetscInt spatialDim = 0;

typedef enum {NEUMANN, DIRICHLET} BCType;

typedef struct {
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  /* Problem definition */
  BCType        bcType;
  void       (**exactFuncs)(const PetscReal x[], PetscScalar *u, void *ctx);
} AppCtx;

void zero_scalar(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
}
void zero_vector(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) u[d] = 0.0;
}

void quadratic_u_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1];
}

void f0_u(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 4.0;
}

void f0_bd_u(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  PetscInt  d;
  for (d = 0, f0[0] = 0.0; d < spatialDim; ++d) f0[0] += -n[d]*2.0*x[d];
}

void f0_bd_zero(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

void f1_bd_zero(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  PetscInt comp;
  for (comp = 0; comp < spatialDim; ++comp) f1[comp] = 0.0;
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_u(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = u_x[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = 1.0;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  PetscInt       bc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim    = 2;
  options->bcType = DIRICHLET;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMFOREST");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex66.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex66.c",bcTypes,2,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim      = user->dim;
  const PetscInt cells[3] = {3, 3, 3};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupProblem"
PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  switch (user->dim) {
  case 2:
    user->exactFuncs[0] = quadratic_u_2d;
    if (user->bcType == NEUMANN) {ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u, f1_bd_zero);CHKERRQ(ierr);}
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupDiscretization"
PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm   = dm;
  const PetscInt dim   = user->dim;
  const PetscInt id    = 1;
  PetscFE        feBd  = NULL;
  PetscFE        fe;
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    ierr = PetscFECreateDefault(dm, dim-1, 1, PETSC_TRUE, "bd_", -1, &feBd);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) feBd, "potential");CHKERRQ(ierr);
  }
  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscDSSetBdDiscretization(prob, 0, (PetscObject) feBd);CHKERRQ(ierr);
    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(cdm, user->bcType == DIRICHLET, "wall", user->bcType == NEUMANN ? "boundary" : "marker", 0, 0, NULL, user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feBd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/** Callback function to decide on refinement.
 *
 * This function is called for every processor-local quadrant in order; its
 * return value is understood as a boolean refinement flag.  We refine around a
 * h = 1/8 block with left front lower corner (5/8, 2/8, 6/8).
 */
static int refine_fn(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  /* Compute the integer coordinate extent of a quadrant of length 2^(-3). */
  const p4est_qcoord_t eighth = P4EST_QUADRANT_LEN(3);
  /* Compute the length of the current quadrant in integer coordinates. */
  const p4est_qcoord_t length = P4EST_QUADRANT_LEN(quadrant->level);

  /* Refine if the quadrant intersects the block in question. */
  return ((quadrant->x + length > 5 * eighth && quadrant->x < 6 * eighth) &&
          (quadrant->y + length > 2 * eighth && quadrant->y < 3 * eighth) &&
#ifdef P4_TO_P8
          (quadrant->z + length > 6 * eighth && quadrant->z < 7 * eighth) &&
#endif
          1);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes; /* nonlinear solver */
  DM             dm;   /* problem definition */
  Vec            u;    /* solution, residual vectors */
  AppCtx         user; /* user-defined work context */
  PetscSection   section;
  PetscInt       v, lsize;
  PetscReal      ferrors[1];
  void         (*initialGuess[2])(const PetscReal x[], PetscScalar *u, void* ctx) = {zero_vector, zero_scalar};
  PetscErrorCode ierr;

  p4est_connectivity_t *conn;
  p4est_t              *p4est;
  p4est_ghost_t        *ghost;
  p4est_lnodes_t       *lnodes;
  int                   startlevel, endlevel, level;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);

  sc_init(PETSC_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  p4est_init(NULL, SC_LP_PRODUCTION);  /* SC_LP_ERROR for silence. */

  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);

  /* Create a forest that consists of multiple quadtrees/octrees.
   * This file is compiled for both 2D and 3D: the macro P4_TO_P8 can be
   * checked to execute dimension-dependent code. */
  conn = p4est_connectivity_new_unitsquare();

  /* Create a forest that is not refined; it consists of the root octant.
   * The p4est_new_ext function can take a startlevel for a load-balanced
   * initial uniform refinement.  Here we refine adaptively instead. */
  startlevel = 0;
  p4est = p4est_new(PETSC_COMM_WORLD, conn, 0, NULL, NULL);

  /* Refine the forest iteratively, load balancing at each iteration.
   * This is important when starting with an unrefined forest */
  endlevel = 5;
  for (level = startlevel; level < endlevel; ++level) {
    p4est_refine(p4est, 0, refine_fn, NULL);
    /* Refinement has lead to up to 8x more elements; redistribute them. */
    p4est_partition(p4est, 0, NULL);
  }
  if (startlevel < endlevel) {
    /* For finite elements this corner balance is not strictly required.
     * We call balance only once since it's more expensive than both
     * coarsen/refine and partition. */
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    /* We repartition with coarsening in mind to allow for
     * partition-independent a-posteriori adaptation (not covered in step4). */
    p4est_partition(p4est, 1, NULL);
  }

  /* Write the forest to disk for visualization, one file per processor. */
  p4est_vtk_write_file(p4est, NULL, "ex66");

  /* Create the ghost layer to learn about parallel neighbors. */
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  /* Create a node numbering for continuous linear finite elements. */
  lnodes = p4est_lnodes_new(p4est, ghost, 1);

  /* Create a section from the lnodes */
  ierr = PetscSectionCreate(PETSC_COMM_WORLD, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(section, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 0, "potential");CHKERRQ(ierr);
  /* We can count the various thing from lnodes, but it would be nice to
     get the info directly from p4est, ask Toby.

     Cells:    [0, num_local_elements)
     Vertices: [num_local_elements, num_local_elements+num_local_nodes)
     Faces:    [)
   */
  ierr = PetscSectionSetChart(section, 0, lnodes->num_local_elements + lnodes->num_local_nodes);CHKERRQ(ierr);
  for (v = lnodes->num_local_elements; v < lnodes->num_local_elements + lnodes->num_local_nodes; ++v) {
    ierr = PetscSectionSetDof(section, v, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &lsize);CHKERRQ(ierr);
  if (lsize != lnodes->num_local_nodes) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local section size %d != %d from lnodes", lsize, lnodes->num_local_nodes);

  /* Create SF */

  /* Destroy the ghost structure -- no longer needed after node creation. */
  p4est_ghost_destroy(ghost);
  ghost = NULL;

  /* We are done with the FE node numbering. */
  p4est_lnodes_destroy (lnodes);

  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = DMComputeL2FieldDiff(dm, user.exactFuncs, NULL, u, ferrors);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %.3g\n", ferrors[0]);CHKERRQ(ierr);
#if 0
  if (user.showError) {
    Vec r;

    ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, user.exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
    ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution Error\n");CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  }
#endif
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);

  /* Destroy the p4est and the connectivity structure. */
  p4est_destroy (p4est);
  p4est_connectivity_destroy (conn);

  /* Verify that allocations internal to p4est and sc do not leak memory.
   * This should be called if sc_init () has been called earlier. */
  sc_finalize();

  ierr = PetscFinalize();
  return 0;
}
