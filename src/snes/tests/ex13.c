static char help[] = "Benchmark Poisson Problem in 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  /* Domain and mesh definition */
  PetscBool benchmark;
  PetscInt  cells[3];
  PetscInt  processGrid[3];
  PetscInt  nodeGrid[3];
} AppCtx;


static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscInt       asz, dim=2; /* should be default of DMPLex (yuck) */

  PetscFunctionBeginUser;
  options->benchmark= PETSC_FALSE;
  for (asz=0;asz<3;asz++) options->processGrid[asz] = options->cells[asz] = options->nodeGrid[asz] = 1;
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_plex_box_dim","dim in ex13","ex13.c",dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-benchmark", "Solve the benchmark problem", "ex13.c", options->benchmark, &options->benchmark, NULL);CHKERRQ(ierr);
  asz  = dim;
  ierr = PetscOptionsIntArray("-dm_plex_box_faces","Mesh size (cells) for benchmarking ex13","ex13.c",options->cells,&asz,NULL);CHKERRQ(ierr);
  asz  = dim;
  ierr = PetscOptionsIntArray("-process_grid_size","Number of processors (np) in each dimension (cells[i]%np[i]==0 && prod(np[i]==#procs)","ex13.c",options->processGrid,&asz,NULL);CHKERRQ(ierr);
  asz  = dim;
  ierr = PetscOptionsIntArray("-node_grid_size","Number of nodes (nnodes) in each dimension (np[i]%nnodes[i]==0)","ex13.c",options->nodeGrid,&asz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBeginUser;
  /* Create box mesh */
  ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* TODO: This should be pulled into the library */
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  if (user->benchmark) {
    PetscPartitioner part;
    PetscInt         cEnd, ii, np,cells_proc[3],procs_node[3];
    PetscMPIInt      rank, size;
    PetscInt         *sizes = NULL, *points = NULL;
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(*dm, 0, NULL, &cEnd);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
    for (ii=0,np=1;ii<dim;ii++) np *= user->processGrid[ii]; /* check number of processors */
    if (np!=size)SETERRQ2(comm,PETSC_ERR_SUP,"invalid process grid sum = %D, -n %D",np, size);
    for (ii=0,np=1;ii<dim;ii++) np *= user->cells[ii];
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
    if (!rank) {
      if (np!=cEnd)SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP," cell grid %D != num cells = %D",np,cEnd);
      for (ii=0;ii<dim;ii++) {
        if (user->processGrid[ii]%user->nodeGrid[ii]) SETERRQ3(comm,PETSC_ERR_SUP,"dir %D, invalid node grid size %D, process grid %D",ii,user->nodeGrid[ii],user->processGrid[ii]);
        procs_node[ii] = user->processGrid[ii]/user->nodeGrid[ii];
        if (user->cells[ii]%user->processGrid[ii]) SETERRQ3(comm,PETSC_ERR_SUP,"dir %D, invalid process grid size %D, cells %D",ii,user->processGrid[ii],user->cells[ii]);
        cells_proc[ii] = user->cells[ii]/user->processGrid[ii];
        ierr = PetscPrintf(comm, "%D) cells_proc=%D procs_node=%D processGrid=%D cells=%D nodeGrid=%D\n",ii,cells_proc[ii],procs_node[ii],user->processGrid[ii],user->cells[ii],user->nodeGrid[ii]);CHKERRQ(ierr);
      }
      for (/* */;ii<3;ii++) {
        procs_node[ii] = cells_proc[ii] = 1;
        ierr = PetscPrintf(comm, "%D) cells_proc=%D procs_node=%D processGrid=%D cells=%D nodeGrid=%D\n",ii,cells_proc[ii],procs_node[ii],user->processGrid[ii],user->cells[ii],user->nodeGrid[ii]);CHKERRQ(ierr);
      }
      PetscInt  pi,pj,pk,ni,nj,nk,ci,cj,ck,pid=0;
      ierr = PetscMalloc2(size, &sizes, cEnd, &points);CHKERRQ(ierr);
      for (ii=0,np=1;ii<dim;ii++) np *= cells_proc[ii];
      for (ii=0;ii<size;ii++) sizes[ii] = np;
      for (ii=0;ii<cEnd;ii++) points[ii] = -1;
      for (nk=0;nk<user->nodeGrid[2];nk++) { /* node loop */
        PetscInt idx_2 = nk*cells_proc[2]*procs_node[2]*user->cells[0]*user->cells[1];
        for (nj=0;nj<user->nodeGrid[1];nj++) { /* node loop */
          PetscInt idx_1 = idx_2 + nj*cells_proc[1]*procs_node[1]*user->cells[0];
          for (ni=0;ni<user->nodeGrid[0];ni++) { /* node loop */
            PetscInt idx_0 = idx_1 + ni*cells_proc[0]*procs_node[0];
            for (pk=0;pk<procs_node[2];pk++) { /* process loop */
              PetscInt idx_22 = idx_0 + pk*cells_proc[2]*user->cells[0]*user->cells[1];
              for (pj=0;pj<procs_node[1];pj++) { /* process loop */
                PetscInt idx_11 = idx_22 + pj*cells_proc[1]*user->cells[0];
                for (pi=0;pi<procs_node[0];pi++) { /* process loop */
                  PetscInt idx_00 = idx_11 + pi*cells_proc[0];
                  for (ck=0;ck<cells_proc[2];ck++) { /* cell loop */
                    PetscInt idx_222 = idx_00 + ck*user->cells[0]*user->cells[1];
                    for (cj=0;cj<cells_proc[1];cj++) { /* cell loop */
                      PetscInt idx_111 = idx_222 + cj*user->cells[0];
                      for (ci=0;ci<cells_proc[0];ci++) { /* cell loop */
                        PetscInt idx_000 = idx_111 + ci;
                        points[idx_000] = pid;
                      }
                    }
                  }
                  pid++;
                }
              }
            }
          }
        }
      }
      if (pid!=size) SETERRQ2(comm,PETSC_ERR_SUP,"pid %D != size %D",pid,size);
      /* view */
      ierr = PetscPrintf(comm, "points:\n");CHKERRQ(ierr);
      pid=0;
      for (ck=0;ck<user->cells[2];ck++) {
        for (cj=0;cj<user->cells[1];cj++) {
          for (ci=0;ci<user->cells[0];ci++) {
            ierr = PetscPrintf(comm, "%6D",points[pid++]);CHKERRQ(ierr);
          }
          ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
        }
        ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
      }
    }
    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, size, sizes, points);CHKERRQ(ierr);
    ierr = PetscFree2(sizes, points);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, trig_u, user);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) trig_u, NULL, 1, &id, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  /* Benchmark system */
  if (user.benchmark) {
#if defined(PETSC_USE_LOG)
    PetscLogStage stage;
#endif
    KSP           ksp;
    Vec           b;
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = VecZeroEntries(u);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes, &b, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, u, b);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("KSP Solve only", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, u);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: bench
    nsize: 16
    args: -dm_plex_box_dim 2 -dm_plex_box_faces 8,8 -ksp_type cg -pc_type gamg -dm_plex_box_simplex 0 -dm_refine 1 \
          -potential_petscspace_degree 2 -dm_distribute -petscpartitioner_type simple -dm_view \
          -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -process_grid_size 4,4 -node_grid_size 2,2 -benchmark true

TEST*/
