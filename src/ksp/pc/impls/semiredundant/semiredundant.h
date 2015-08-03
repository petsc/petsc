
#ifndef __PETSCPC_SEMIREDUNDANT_H
#define __PETSCPC_SEMIREDUNDANT_H

/* SemiRedundant */
typedef enum { SR_DEFAULT = 0, SR_DMDA, SR_DMPLEX } SemiRedundantType;
typedef struct _p_PC_SemiRedundant PC_SemiRedundant;

struct _p_PC_SemiRedundant {
  PetscSubcomm      psubcomm;
  PetscInt          redfactor; /* factor to reduce comm size by */
  KSP               ksp;
  IS                isin;
  VecScatter        scatter;
  Vec               xred,yred,xtmp;
  Mat               Bred;
  PetscBool         ignore_dm;
  SemiRedundantType sr_type;
  void              *dm_ctx;
  PetscErrorCode    (*pcsemired_setup_type)(PC,PC_SemiRedundant*);
  PetscErrorCode    (*pcsemired_matcreate_type)(PC,PC_SemiRedundant*,MatReuse,Mat*);
  PetscErrorCode    (*pcsemired_matnullspacecreate_type)(PC,PC_SemiRedundant*,Mat);
  PetscErrorCode    (*pcsemired_reset_type)(PC);
};

PetscBool isActiveRank(PetscSubcomm);
DM private_PCSemiRedundantGetSubDM(PC_SemiRedundant*);

/* DMDA */
typedef struct {
  DM              dmrepart;
  Mat             permutation;
  Vec             xp;
  PetscInt        Mp_re,Np_re,Pp_re;
  PetscInt        *range_i_re,*range_j_re,*range_k_re;
  PetscInt        *start_i_re,*start_j_re,*start_k_re;
} PC_SemiRedundant_DMDACtx;

PetscErrorCode _DMDADetermineRankFromGlobalIJK(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,
                                               PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,
                                               PetscMPIInt*,PetscMPIInt*,PetscMPIInt*,PetscMPIInt*);

PetscErrorCode _DMDADetermineGlobalS0(PetscInt,PetscMPIInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

PetscErrorCode PCSemiRedundantSetUp_dmda(PC,PC_SemiRedundant*);
PetscErrorCode PCSemiRedundantMatCreate_dmda(PC,PC_SemiRedundant*,MatReuse,Mat*);
PetscErrorCode PCSemiRedundantMatNullSpaceCreate_dmda(PC,PC_SemiRedundant*,Mat);
PetscErrorCode PCApply_SemiRedundant_dmda(PC,Vec,Vec);
PetscErrorCode PCReset_SemiRedundant_dmda(PC);
PetscErrorCode DMView_DMDAShort(DM,PetscViewer);

#endif
