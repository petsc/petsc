
#ifndef __PETSCPC_TELESCOPE_H
#define __PETSCPC_TELESCOPE_H

/* Telescope */
typedef enum { TELESCOPE_DEFAULT = 0, TELESCOPE_DMDA, TELESCOPE_DMPLEX } PCTelescopeType;

typedef struct _PC_Telescope *PC_Telescope;
struct _PC_Telescope {
  PetscSubcomm      psubcomm;
  PetscSubcommType  subcommtype;
  PetscInt          redfactor; /* factor to reduce comm size by */
  KSP               ksp;
  IS                isin;
  VecScatter        scatter;
  Vec               xred,yred,xtmp;
  Mat               Bred;
  PetscBool         ignore_dm,ignore_kspcomputeoperators;
  PCTelescopeType   sr_type;
  void              *dm_ctx;
  PetscErrorCode    (*pctelescope_setup_type)(PC,PC_Telescope);
  PetscErrorCode    (*pctelescope_matcreate_type)(PC,PC_Telescope,MatReuse,Mat*);
  PetscErrorCode    (*pctelescope_matnullspacecreate_type)(PC,PC_Telescope,Mat);
  PetscErrorCode    (*pctelescope_reset_type)(PC);
};

 PetscBool isActiveRank(PetscSubcomm);
 DM private_PCTelescopeGetSubDM(PC_Telescope);

/* DMDA */
typedef struct {
  DM              dmrepart;
  Mat             permutation;
  Vec             xp;
  PetscInt        Mp_re,Np_re,Pp_re;
  PetscInt        *range_i_re,*range_j_re,*range_k_re;
  PetscInt        *start_i_re,*start_j_re,*start_k_re;
} PC_Telescope_DMDACtx;

 PetscErrorCode _DMDADetermineRankFromGlobalIJK(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,
                                               PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,
                                               PetscMPIInt*,PetscMPIInt*,PetscMPIInt*,PetscMPIInt*);

 PetscErrorCode _DMDADetermineGlobalS0(PetscInt,PetscMPIInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

 PetscErrorCode PCTelescopeSetUp_dmda(PC,PC_Telescope);
 PetscErrorCode PCTelescopeMatCreate_dmda(PC,PC_Telescope,MatReuse,Mat*);
 PetscErrorCode PCTelescopeMatNullSpaceCreate_dmda(PC,PC_Telescope,Mat);
 PetscErrorCode PCApply_Telescope_dmda(PC,Vec,Vec);
PetscErrorCode PCApplyRichardson_Telescope_dmda(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason);
PetscErrorCode PCReset_Telescope_dmda(PC);
PetscErrorCode DMView_DMDAShort(DM,PetscViewer);

#endif
