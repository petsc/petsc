#pragma once
#include <petscksp.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/pcmgimpl.h> /*I "petscksp.h" I*/
#include <petscmatcoarsen.h>        /*I "petscmatcoarsen.h" I*/
#include <petsc/private/matimpl.h>

struct _PCGAMGOps {
  PetscErrorCode (*creategraph)(PC, Mat, Mat *);
  PetscErrorCode (*coarsen)(PC, Mat *, PetscCoarsenData **);
  PetscErrorCode (*prolongator)(PC, Mat, PetscCoarsenData *, Mat *);
  PetscErrorCode (*optprolongator)(PC, Mat, Mat *);
  PetscErrorCode (*createlevel)(PC, Mat, PetscInt, Mat *, Mat *, PetscMPIInt *, IS *, PetscBool);
  PetscErrorCode (*createdefaultdata)(PC, Mat); /* for data methods that have a default (SA) */
  PetscErrorCode (*setfromoptions)(PC, PetscOptionItems);
  PetscErrorCode (*destroy)(PC);
  PetscErrorCode (*view)(PC, PetscViewer);
};
/* Private context for the GAMG preconditioner */
typedef struct gamg_TAG {
  PCGAMGType       type;
  PetscInt         Nlevels;
  PetscBool        repart;
  PetscBool        reuse_prol;
  PetscBool        use_aggs_in_asm;
  PetscBool        use_parallel_coarse_grid_solver;
  PCGAMGLayoutType layout_type;
  PetscBool        cpu_pin_coarse_grids;
  PetscInt         min_eq_proc;
  PetscInt         asm_hem_aggs;
  MatCoarsen       asm_crs; /* used to generate ASM aggregates */
  PetscInt         coarse_eq_limit;
  PetscReal        threshold_scale;
  PetscReal        threshold[PETSC_MG_MAXLEVELS]; /* common quantity to many AMG methods so keep it up here */
  PetscInt         level_reduction_factors[PETSC_MG_MAXLEVELS];
  PetscInt         current_level; /* stash construction state */
  /* these 4 are all related to the method data and should be in the subctx */
  PetscInt   data_sz; /* nloc*data_rows*data_cols */
  PetscInt   data_cell_rows;
  PetscInt   data_cell_cols;
  PetscInt   orig_data_cell_rows;
  PetscInt   orig_data_cell_cols;
  PetscReal *data;      /* [data_sz] blocked vector of vertex data on fine grid (coordinates/nullspace) */
  PetscReal *orig_data; /* cache data */

  struct _PCGAMGOps *ops;
  char              *gamg_type_name;

  void *subctx;

  PetscBool use_sa_esteig;
  PetscReal emin, emax;
  PetscBool recompute_esteig;
  PetscInt  injection_index_size;
  PetscInt  injection_index[MAT_COARSEN_STRENGTH_INDEX_SIZE];
} PC_GAMG;

/* hooks create derivied classes */
PetscErrorCode PCCreateGAMG_GEO(PC);
PetscErrorCode PCCreateGAMG_AGG(PC);
PetscErrorCode PCCreateGAMG_Classical(PC);

PetscErrorCode PCDestroy_GAMG(PC);

/* helper methods */
PetscErrorCode PCGAMGGetDataWithGhosts(Mat, PetscInt, PetscReal[], PetscInt *, PetscReal **);

enum tag {
  GAMG_SETUP = 0,
  GAMG_MESH,
  GAMG_MATRIX,
  GAMG_GRAPH,
  GAMG_COARSEN,
  GAMG_SQUARE,
  GAMG_MIS,
  GAMG_PROL,
  GAMG_PROLA,
  GAMG_PROLB,
  GAMG_OPT,
  GAMG_OPTSM,
  GAMG_LEVEL,
  GAMG_PTAP,
  GAMG_REDUCE,
  GAMG_REPART,
  SET13,
  SET14,
  SET15,
  GAMG_NUM_SET
};
PETSC_EXTERN PetscLogEvent petsc_gamg_setup_events[GAMG_NUM_SET];
PETSC_EXTERN PetscLogEvent petsc_gamg_setup_matmat_events[PETSC_MG_MAXLEVELS][3];
