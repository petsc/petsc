
#ifndef PETSCPC_TELESCOPE_H
#define PETSCPC_TELESCOPE_H

/* Telescope */
typedef enum {
  TELESCOPE_DEFAULT = 0,
  TELESCOPE_DMDA,
  TELESCOPE_DMPLEX,
  TELESCOPE_COARSEDM
} PCTelescopeType;

typedef struct _PC_Telescope *PC_Telescope;
struct _PC_Telescope {
  PetscSubcomm     psubcomm;
  PetscSubcommType subcommtype;
  MPI_Comm         subcomm;
  PetscInt         redfactor; /* factor to reduce comm size by */
  KSP              ksp;
  IS               isin;
  VecScatter       scatter;
  Vec              xred, yred, xtmp;
  Mat              Bred;
  PetscBool        ignore_dm, ignore_kspcomputeoperators, use_coarse_dm;
  PCTelescopeType  sr_type;
  void            *dm_ctx;
  PetscErrorCode (*pctelescope_setup_type)(PC, PC_Telescope);
  PetscErrorCode (*pctelescope_matcreate_type)(PC, PC_Telescope, MatReuse, Mat *);
  PetscErrorCode (*pctelescope_matnullspacecreate_type)(PC, PC_Telescope, Mat);
  PetscErrorCode (*pctelescope_reset_type)(PC);
};

/* DMDA */
typedef struct {
  DM        dmrepart;
  Mat       permutation;
  Vec       xp;
  PetscInt  Mp_re, Np_re, Pp_re;
  PetscInt *range_i_re, *range_j_re, *range_k_re;
  PetscInt *start_i_re, *start_j_re, *start_k_re;
} PC_Telescope_DMDACtx;

static inline PetscBool PetscSubcomm_isActiveRank(PetscSubcomm scomm)
{
  if (scomm->color == 0) return (PETSC_TRUE);
  else return (PETSC_FALSE);
}

static inline PetscBool PCTelescope_isActiveRank(PC_Telescope sred)
{
  if (sred->psubcomm) return (PetscSubcomm_isActiveRank(sred->psubcomm));
  else {
    if (sred->subcomm != MPI_COMM_NULL) return (PETSC_TRUE);
    else return (PETSC_FALSE);
  }
}

PetscErrorCode PCTelescopeSetUp_dmda(PC, PC_Telescope);
PetscErrorCode PCTelescopeMatCreate_dmda(PC, PC_Telescope, MatReuse, Mat *);
PetscErrorCode PCTelescopeMatNullSpaceCreate_dmda(PC, PC_Telescope, Mat);
PetscErrorCode PCApply_Telescope_dmda(PC, Vec, Vec);
PetscErrorCode PCApplyRichardson_Telescope_dmda(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);
PetscErrorCode PCReset_Telescope_dmda(PC);
PetscErrorCode PCTelescopeSetUp_CoarseDM(PC, PC_Telescope);
PetscErrorCode PCApply_Telescope_CoarseDM(PC, Vec, Vec);
PetscErrorCode PCTelescopeMatNullSpaceCreate_CoarseDM(PC, PC_Telescope, Mat);
PetscErrorCode PCReset_Telescope_CoarseDM(PC);
PetscErrorCode PCApplyRichardson_Telescope_CoarseDM(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);
PetscErrorCode DMView_DA_Short(DM, PetscViewer);

#endif
