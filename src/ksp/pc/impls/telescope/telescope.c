#include <petsc/private/petscimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/
#include <petscdm.h>  /*I "petscdm.h" I*/
#include "../src/ksp/pc/impls/telescope/telescope.h"

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@inproceedings{MaySananRuppKnepleySmith2016,\n"
                               "  title     = {Extreme-Scale Multigrid Components within PETSc},\n"
                               "  author    = {Dave A. May and Patrick Sanan and Karl Rupp and Matthew G. Knepley and Barry F. Smith},\n"
                               "  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},\n"
                               "  series    = {PASC '16},\n"
                               "  isbn      = {978-1-4503-4126-4},\n"
                               "  location  = {Lausanne, Switzerland},\n"
                               "  pages     = {5:1--5:12},\n"
                               "  articleno = {5},\n"
                               "  numpages  = {12},\n"
                               "  url       = {https://doi.acm.org/10.1145/2929908.2929913},\n"
                               "  doi       = {10.1145/2929908.2929913},\n"
                               "  acmid     = {2929913},\n"
                               "  publisher = {ACM},\n"
                               "  address   = {New York, NY, USA},\n"
                               "  keywords  = {GPU, HPC, agglomeration, coarse-level solver, multigrid, parallel computing, preconditioning},\n"
                               "  year      = {2016}\n"
                               "}\n";

/*
 default setup mode

 [1a] scatter to (FORWARD)
 x(comm) -> xtmp(comm)
 [1b] local copy (to) ranks with color = 0
 xred(subcomm) <- xtmp

 [2] solve on sub KSP to obtain yred(subcomm)

 [3a] local copy (from) ranks with color = 0
 yred(subcomm) --> xtmp
 [2b] scatter from (REVERSE)
 xtmp(comm) -> y(comm)
*/

/*
  Collective[comm_f]
  Notes
   * Using comm_f = MPI_COMM_NULL will result in an error
   * Using comm_c = MPI_COMM_NULL is valid. If all instances of comm_c are NULL the subcomm is not valid.
   * If any non NULL comm_c communicator cannot map any of its ranks to comm_f, the subcomm is not valid.
*/
PetscErrorCode PCTelescopeTestValidSubcomm(MPI_Comm comm_f, MPI_Comm comm_c, PetscBool *isvalid)
{
  PetscInt     valid = 1;
  MPI_Group    group_f, group_c;
  PetscMPIInt  count, k, size_f = 0, size_c = 0, size_c_sum = 0;
  PetscMPIInt *ranks_f, *ranks_c;

  PetscFunctionBegin;
  PetscCheck(comm_f != MPI_COMM_NULL, PETSC_COMM_SELF, PETSC_ERR_SUP, "comm_f cannot be MPI_COMM_NULL");

  PetscCallMPI(MPI_Comm_group(comm_f, &group_f));
  if (comm_c != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_group(comm_c, &group_c));

  PetscCallMPI(MPI_Comm_size(comm_f, &size_f));
  if (comm_c != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_size(comm_c, &size_c));

  /* check not all comm_c's are NULL */
  size_c_sum = size_c;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &size_c_sum, 1, MPI_INT, MPI_SUM, comm_f));
  if (size_c_sum == 0) valid = 0;

  /* check we can map at least 1 rank in comm_c to comm_f */
  PetscCall(PetscMalloc1(size_f, &ranks_f));
  PetscCall(PetscMalloc1(size_c, &ranks_c));
  for (k = 0; k < size_f; k++) ranks_f[k] = MPI_UNDEFINED;
  for (k = 0; k < size_c; k++) ranks_c[k] = k;

  /*
   MPI_Group_translate_ranks() returns a non-zero exit code if any rank cannot be translated.
   I do not want the code to terminate immediately if this occurs, rather I want to throw
   the error later (during PCSetUp_Telescope()) via SETERRQ() with a message indicating
   that comm_c is not a valid sub-communicator.
   Hence I purposefully do not call PetscCall() after MPI_Group_translate_ranks().
  */
  count = 0;
  if (comm_c != MPI_COMM_NULL) {
    (void)MPI_Group_translate_ranks(group_c, size_c, ranks_c, group_f, ranks_f);
    for (k = 0; k < size_f; k++) {
      if (ranks_f[k] == MPI_UNDEFINED) count++;
    }
  }
  if (count == size_f) valid = 0;

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPIU_INT, MPI_MIN, comm_f));
  if (valid == 1) *isvalid = PETSC_TRUE;
  else *isvalid = PETSC_FALSE;

  PetscCall(PetscFree(ranks_f));
  PetscCall(PetscFree(ranks_c));
  PetscCallMPI(MPI_Group_free(&group_f));
  if (comm_c != MPI_COMM_NULL) PetscCallMPI(MPI_Group_free(&group_c));
  PetscFunctionReturn(0);
}

DM private_PCTelescopeGetSubDM(PC_Telescope sred)
{
  DM subdm = NULL;

  if (!PCTelescope_isActiveRank(sred)) {
    subdm = NULL;
  } else {
    switch (sred->sr_type) {
    case TELESCOPE_DEFAULT:
      subdm = NULL;
      break;
    case TELESCOPE_DMDA:
      subdm = ((PC_Telescope_DMDACtx *)sred->dm_ctx)->dmrepart;
      break;
    case TELESCOPE_DMPLEX:
      subdm = NULL;
      break;
    case TELESCOPE_COARSEDM:
      if (sred->ksp) KSPGetDM(sred->ksp, &subdm);
      break;
    }
  }
  return (subdm);
}

PetscErrorCode PCTelescopeSetUp_default(PC pc, PC_Telescope sred)
{
  PetscInt   m, M, bs, st, ed;
  Vec        x, xred, yred, xtmp;
  Mat        B;
  MPI_Comm   comm, subcomm;
  VecScatter scatter;
  IS         isin;
  VecType    vectype;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "PCTelescope: setup (default)\n"));
  comm    = PetscSubcommParent(sred->psubcomm);
  subcomm = PetscSubcommChild(sred->psubcomm);

  PetscCall(PCGetOperators(pc, NULL, &B));
  PetscCall(MatGetSize(B, &M, NULL));
  PetscCall(MatGetBlockSize(B, &bs));
  PetscCall(MatCreateVecs(B, &x, NULL));
  PetscCall(MatGetVecType(B, &vectype));

  xred = NULL;
  m    = 0;
  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(VecCreate(subcomm, &xred));
    PetscCall(VecSetSizes(xred, PETSC_DECIDE, M));
    PetscCall(VecSetBlockSize(xred, bs));
    PetscCall(VecSetType(xred, vectype)); /* Use the preconditioner matrix's vectype by default */
    PetscCall(VecSetFromOptions(xred));
    PetscCall(VecGetLocalSize(xred, &m));
  }

  yred = NULL;
  if (PCTelescope_isActiveRank(sred)) PetscCall(VecDuplicate(xred, &yred));

  PetscCall(VecCreate(comm, &xtmp));
  PetscCall(VecSetSizes(xtmp, m, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(xtmp, bs));
  PetscCall(VecSetType(xtmp, vectype));

  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(VecGetOwnershipRange(xred, &st, &ed));
    PetscCall(ISCreateStride(comm, (ed - st), st, 1, &isin));
  } else {
    PetscCall(VecGetOwnershipRange(x, &st, &ed));
    PetscCall(ISCreateStride(comm, 0, st, 1, &isin));
  }
  PetscCall(ISSetBlockSize(isin, bs));

  PetscCall(VecScatterCreate(x, isin, xtmp, NULL, &scatter));

  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xred    = xred;
  sred->yred    = yred;
  sred->xtmp    = xtmp;
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatCreate_default(PC pc, PC_Telescope sred, MatReuse reuse, Mat *A)
{
  MPI_Comm comm, subcomm;
  Mat      Bred, B;
  PetscInt nr, nc, bs;
  IS       isrow, iscol;
  Mat      Blocal, *_Blocal;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "PCTelescope: updating the redundant preconditioned operator (default)\n"));
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  subcomm = PetscSubcommChild(sred->psubcomm);
  PetscCall(PCGetOperators(pc, NULL, &B));
  PetscCall(MatGetSize(B, &nr, &nc));
  isrow = sred->isin;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nc, 0, 1, &iscol));
  PetscCall(ISSetIdentity(iscol));
  PetscCall(MatGetBlockSizes(B, NULL, &bs));
  PetscCall(ISSetBlockSize(iscol, bs));
  PetscCall(MatSetOption(B, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  PetscCall(MatCreateSubMatrices(B, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &_Blocal));
  Blocal = *_Blocal;
  PetscCall(PetscFree(_Blocal));
  Bred = NULL;
  if (PCTelescope_isActiveRank(sred)) {
    PetscInt mm;

    if (reuse != MAT_INITIAL_MATRIX) Bred = *A;

    PetscCall(MatGetSize(Blocal, &mm, NULL));
    PetscCall(MatCreateMPIMatConcatenateSeqMat(subcomm, Blocal, mm, reuse, &Bred));
  }
  *A = Bred;
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&Blocal));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSubNullSpaceCreate_Telescope(PC pc, PC_Telescope sred, MatNullSpace nullspace, MatNullSpace *sub_nullspace)
{
  PetscBool  has_const;
  const Vec *vecs;
  Vec       *sub_vecs = NULL;
  PetscInt   i, k, n = 0;
  MPI_Comm   subcomm;

  PetscFunctionBegin;
  subcomm = PetscSubcommChild(sred->psubcomm);
  PetscCall(MatNullSpaceGetVecs(nullspace, &has_const, &n, &vecs));

  if (PCTelescope_isActiveRank(sred)) {
    if (n) PetscCall(VecDuplicateVecs(sred->xred, n, &sub_vecs));
  }

  /* copy entries */
  for (k = 0; k < n; k++) {
    const PetscScalar *x_array;
    PetscScalar       *LA_sub_vec;
    PetscInt           st, ed;

    /* pull in vector x->xtmp */
    PetscCall(VecScatterBegin(sred->scatter, vecs[k], sred->xtmp, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sred->scatter, vecs[k], sred->xtmp, INSERT_VALUES, SCATTER_FORWARD));
    if (sub_vecs) {
      /* copy vector entries into xred */
      PetscCall(VecGetArrayRead(sred->xtmp, &x_array));
      if (sub_vecs[k]) {
        PetscCall(VecGetOwnershipRange(sub_vecs[k], &st, &ed));
        PetscCall(VecGetArray(sub_vecs[k], &LA_sub_vec));
        for (i = 0; i < ed - st; i++) LA_sub_vec[i] = x_array[i];
        PetscCall(VecRestoreArray(sub_vecs[k], &LA_sub_vec));
      }
      PetscCall(VecRestoreArrayRead(sred->xtmp, &x_array));
    }
  }

  if (PCTelescope_isActiveRank(sred)) {
    /* create new (near) nullspace for redundant object */
    PetscCall(MatNullSpaceCreate(subcomm, has_const, n, sub_vecs, sub_nullspace));
    PetscCall(VecDestroyVecs(n, &sub_vecs));
    PetscCheck(!nullspace->remove, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Propagation of custom remove callbacks not supported when propagating (near) nullspaces with PCTelescope");
    PetscCheck(!nullspace->rmctx, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Propagation of custom remove callback context not supported when propagating (near) nullspaces with PCTelescope");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeMatNullSpaceCreate_default(PC pc, PC_Telescope sred, Mat sub_mat)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(PCGetOperators(pc, NULL, &B));
  /* Propagate the nullspace if it exists */
  {
    MatNullSpace nullspace, sub_nullspace;
    PetscCall(MatGetNullSpace(B, &nullspace));
    if (nullspace) {
      PetscCall(PetscInfo(pc, "PCTelescope: generating nullspace (default)\n"));
      PetscCall(PCTelescopeSubNullSpaceCreate_Telescope(pc, sred, nullspace, &sub_nullspace));
      if (PCTelescope_isActiveRank(sred)) {
        PetscCall(MatSetNullSpace(sub_mat, sub_nullspace));
        PetscCall(MatNullSpaceDestroy(&sub_nullspace));
      }
    }
  }
  /* Propagate the near nullspace if it exists */
  {
    MatNullSpace nearnullspace, sub_nearnullspace;
    PetscCall(MatGetNearNullSpace(B, &nearnullspace));
    if (nearnullspace) {
      PetscCall(PetscInfo(pc, "PCTelescope: generating near nullspace (default)\n"));
      PetscCall(PCTelescopeSubNullSpaceCreate_Telescope(pc, sred, nearnullspace, &sub_nearnullspace));
      if (PCTelescope_isActiveRank(sred)) {
        PetscCall(MatSetNearNullSpace(sub_mat, sub_nearnullspace));
        PetscCall(MatNullSpaceDestroy(&sub_nearnullspace));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Telescope(PC pc, PetscViewer viewer)
{
  PC_Telescope sred = (PC_Telescope)pc->data;
  PetscBool    iascii, isstring;
  PetscViewer  subviewer;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (iascii) {
    {
      MPI_Comm    comm, subcomm;
      PetscMPIInt comm_size, subcomm_size;
      DM          dm = NULL, subdm = NULL;

      PetscCall(PCGetDM(pc, &dm));
      subdm = private_PCTelescopeGetSubDM(sred);

      if (sred->psubcomm) {
        comm    = PetscSubcommParent(sred->psubcomm);
        subcomm = PetscSubcommChild(sred->psubcomm);
        PetscCallMPI(MPI_Comm_size(comm, &comm_size));
        PetscCallMPI(MPI_Comm_size(subcomm, &subcomm_size));

        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "petsc subcomm: parent comm size reduction factor = %" PetscInt_FMT "\n", sred->redfactor));
        PetscCall(PetscViewerASCIIPrintf(viewer, "petsc subcomm: parent_size = %d , subcomm_size = %d\n", (int)comm_size, (int)subcomm_size));
        switch (sred->subcommtype) {
        case PETSC_SUBCOMM_INTERLACED:
          PetscCall(PetscViewerASCIIPrintf(viewer, "petsc subcomm: type = %s\n", PetscSubcommTypes[sred->subcommtype]));
          break;
        case PETSC_SUBCOMM_CONTIGUOUS:
          PetscCall(PetscViewerASCIIPrintf(viewer, "petsc subcomm type = %s\n", PetscSubcommTypes[sred->subcommtype]));
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "General subcomm type not supported by PCTelescope");
        }
        PetscCall(PetscViewerASCIIPopTab(viewer));
      } else {
        PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
        subcomm = sred->subcomm;
        if (!PCTelescope_isActiveRank(sred)) subcomm = PETSC_COMM_SELF;

        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "subcomm: using user provided sub-communicator\n"));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }

      PetscCall(PetscViewerGetSubViewer(viewer, subcomm, &subviewer));
      if (PCTelescope_isActiveRank(sred)) {
        PetscCall(PetscViewerASCIIPushTab(subviewer));

        if (dm && sred->ignore_dm) PetscCall(PetscViewerASCIIPrintf(subviewer, "ignoring DM\n"));
        if (sred->ignore_kspcomputeoperators) PetscCall(PetscViewerASCIIPrintf(subviewer, "ignoring KSPComputeOperators\n"));
        switch (sred->sr_type) {
        case TELESCOPE_DEFAULT:
          PetscCall(PetscViewerASCIIPrintf(subviewer, "setup type: default\n"));
          break;
        case TELESCOPE_DMDA:
          PetscCall(PetscViewerASCIIPrintf(subviewer, "setup type: DMDA auto-repartitioning\n"));
          PetscCall(DMView_DA_Short(subdm, subviewer));
          break;
        case TELESCOPE_DMPLEX:
          PetscCall(PetscViewerASCIIPrintf(subviewer, "setup type: DMPLEX auto-repartitioning\n"));
          break;
        case TELESCOPE_COARSEDM:
          PetscCall(PetscViewerASCIIPrintf(subviewer, "setup type: coarse DM\n"));
          break;
        }

        if (dm) {
          PetscObject obj = (PetscObject)dm;
          PetscCall(PetscViewerASCIIPrintf(subviewer, "Parent DM object:"));
          PetscViewerASCIIUseTabs(subviewer, PETSC_FALSE);
          if (obj->type_name) { PetscViewerASCIIPrintf(subviewer, " type = %s;", obj->type_name); }
          if (obj->name) { PetscViewerASCIIPrintf(subviewer, " name = %s;", obj->name); }
          if (obj->prefix) PetscViewerASCIIPrintf(subviewer, " prefix = %s", obj->prefix);
          PetscCall(PetscViewerASCIIPrintf(subviewer, "\n"));
          PetscViewerASCIIUseTabs(subviewer, PETSC_TRUE);
        } else {
          PetscCall(PetscViewerASCIIPrintf(subviewer, "Parent DM object: NULL\n"));
        }
        if (subdm) {
          PetscObject obj = (PetscObject)subdm;
          PetscCall(PetscViewerASCIIPrintf(subviewer, "Sub DM object:"));
          PetscViewerASCIIUseTabs(subviewer, PETSC_FALSE);
          if (obj->type_name) { PetscViewerASCIIPrintf(subviewer, " type = %s;", obj->type_name); }
          if (obj->name) { PetscViewerASCIIPrintf(subviewer, " name = %s;", obj->name); }
          if (obj->prefix) PetscViewerASCIIPrintf(subviewer, " prefix = %s", obj->prefix);
          PetscCall(PetscViewerASCIIPrintf(subviewer, "\n"));
          PetscViewerASCIIUseTabs(subviewer, PETSC_TRUE);
        } else {
          PetscCall(PetscViewerASCIIPrintf(subviewer, "Sub DM object: NULL\n"));
        }

        PetscCall(KSPView(sred->ksp, subviewer));
        PetscCall(PetscViewerASCIIPopTab(subviewer));
      }
      PetscCall(PetscViewerRestoreSubViewer(viewer, subcomm, &subviewer));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Telescope(PC pc)
{
  PC_Telescope    sred = (PC_Telescope)pc->data;
  MPI_Comm        comm, subcomm = 0;
  PCTelescopeType sr_type;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));

  /* Determine type of setup/update */
  if (!pc->setupcalled) {
    PetscBool has_dm, same;
    DM        dm;

    sr_type = TELESCOPE_DEFAULT;
    has_dm  = PETSC_FALSE;
    PetscCall(PCGetDM(pc, &dm));
    if (dm) has_dm = PETSC_TRUE;
    if (has_dm) {
      /* check for dmda */
      PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMDA, &same));
      if (same) {
        PetscCall(PetscInfo(pc, "PCTelescope: found DMDA\n"));
        sr_type = TELESCOPE_DMDA;
      }
      /* check for dmplex */
      PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &same));
      if (same) {
        PetscCall(PetscInfo(pc, "PCTelescope: found DMPLEX\n"));
        sr_type = TELESCOPE_DMPLEX;
      }

      if (sred->use_coarse_dm) {
        PetscCall(PetscInfo(pc, "PCTelescope: using coarse DM\n"));
        sr_type = TELESCOPE_COARSEDM;
      }

      if (sred->ignore_dm) {
        PetscCall(PetscInfo(pc, "PCTelescope: ignoring DM\n"));
        sr_type = TELESCOPE_DEFAULT;
      }
    }
    sred->sr_type = sr_type;
  } else {
    sr_type = sred->sr_type;
  }

  /* set function pointers for repartition setup, matrix creation/update, matrix (near) nullspace, and reset functionality */
  switch (sr_type) {
  case TELESCOPE_DEFAULT:
    sred->pctelescope_setup_type              = PCTelescopeSetUp_default;
    sred->pctelescope_matcreate_type          = PCTelescopeMatCreate_default;
    sred->pctelescope_matnullspacecreate_type = PCTelescopeMatNullSpaceCreate_default;
    sred->pctelescope_reset_type              = NULL;
    break;
  case TELESCOPE_DMDA:
    pc->ops->apply                            = PCApply_Telescope_dmda;
    pc->ops->applyrichardson                  = PCApplyRichardson_Telescope_dmda;
    sred->pctelescope_setup_type              = PCTelescopeSetUp_dmda;
    sred->pctelescope_matcreate_type          = PCTelescopeMatCreate_dmda;
    sred->pctelescope_matnullspacecreate_type = PCTelescopeMatNullSpaceCreate_dmda;
    sred->pctelescope_reset_type              = PCReset_Telescope_dmda;
    break;
  case TELESCOPE_DMPLEX:
    SETERRQ(comm, PETSC_ERR_SUP, "Support for DMPLEX is currently not available");
  case TELESCOPE_COARSEDM:
    pc->ops->apply                            = PCApply_Telescope_CoarseDM;
    pc->ops->applyrichardson                  = PCApplyRichardson_Telescope_CoarseDM;
    sred->pctelescope_setup_type              = PCTelescopeSetUp_CoarseDM;
    sred->pctelescope_matcreate_type          = NULL;
    sred->pctelescope_matnullspacecreate_type = NULL; /* PCTelescopeMatNullSpaceCreate_CoarseDM; */
    sred->pctelescope_reset_type              = PCReset_Telescope_CoarseDM;
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Support only provided for: repartitioning an operator; repartitioning a DMDA; or using a coarse DM");
  }

  /* subcomm definition */
  if (!pc->setupcalled) {
    if ((sr_type == TELESCOPE_DEFAULT) || (sr_type == TELESCOPE_DMDA)) {
      if (!sred->psubcomm) {
        PetscCall(PetscSubcommCreate(comm, &sred->psubcomm));
        PetscCall(PetscSubcommSetNumber(sred->psubcomm, sred->redfactor));
        PetscCall(PetscSubcommSetType(sred->psubcomm, sred->subcommtype));
        sred->subcomm = PetscSubcommChild(sred->psubcomm);
      }
    } else { /* query PC for DM, check communicators */
      DM          dm, dm_coarse_partition          = NULL;
      MPI_Comm    comm_fine, comm_coarse_partition = MPI_COMM_NULL;
      PetscMPIInt csize_fine = 0, csize_coarse_partition = 0, cs[2], csg[2], cnt = 0;
      PetscBool   isvalidsubcomm;

      PetscCall(PCGetDM(pc, &dm));
      comm_fine = PetscObjectComm((PetscObject)dm);
      PetscCall(DMGetCoarseDM(dm, &dm_coarse_partition));
      if (dm_coarse_partition) cnt = 1;
      PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &cnt, 1, MPI_INT, MPI_SUM, comm_fine));
      PetscCheck(cnt != 0, comm_fine, PETSC_ERR_SUP, "Zero instances of a coarse DM were found");

      PetscCallMPI(MPI_Comm_size(comm_fine, &csize_fine));
      if (dm_coarse_partition) {
        comm_coarse_partition = PetscObjectComm((PetscObject)dm_coarse_partition);
        PetscCallMPI(MPI_Comm_size(comm_coarse_partition, &csize_coarse_partition));
      }

      cs[0] = csize_fine;
      cs[1] = csize_coarse_partition;
      PetscCallMPI(MPI_Allreduce(cs, csg, 2, MPI_INT, MPI_MAX, comm_fine));
      PetscCheck(csg[0] != csg[1], comm_fine, PETSC_ERR_SUP, "Coarse DM uses the same size communicator as the parent DM attached to the PC");

      PetscCall(PCTelescopeTestValidSubcomm(comm_fine, comm_coarse_partition, &isvalidsubcomm));
      PetscCheck(isvalidsubcomm, comm_fine, PETSC_ERR_SUP, "Coarse DM communicator is not a sub-communicator of parentDM->comm");
      sred->subcomm = comm_coarse_partition;
    }
  }
  subcomm = sred->subcomm;

  /* internal KSP */
  if (!pc->setupcalled) {
    const char *prefix;

    if (PCTelescope_isActiveRank(sred)) {
      PetscCall(KSPCreate(subcomm, &sred->ksp));
      PetscCall(KSPSetErrorIfNotConverged(sred->ksp, pc->erroriffailure));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)sred->ksp, (PetscObject)pc, 1));
      PetscCall(KSPSetType(sred->ksp, KSPPREONLY));
      PetscCall(PCGetOptionsPrefix(pc, &prefix));
      PetscCall(KSPSetOptionsPrefix(sred->ksp, prefix));
      PetscCall(KSPAppendOptionsPrefix(sred->ksp, "telescope_"));
    }
  }

  /* setup */
  if (!pc->setupcalled && sred->pctelescope_setup_type) PetscCall(sred->pctelescope_setup_type(pc, sred));
  /* update */
  if (!pc->setupcalled) {
    if (sred->pctelescope_matcreate_type) PetscCall(sred->pctelescope_matcreate_type(pc, sred, MAT_INITIAL_MATRIX, &sred->Bred));
    if (sred->pctelescope_matnullspacecreate_type) PetscCall(sred->pctelescope_matnullspacecreate_type(pc, sred, sred->Bred));
  } else {
    if (sred->pctelescope_matcreate_type) PetscCall(sred->pctelescope_matcreate_type(pc, sred, MAT_REUSE_MATRIX, &sred->Bred));
  }

  /* common - no construction */
  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(KSPSetOperators(sred->ksp, sred->Bred, sred->Bred));
    if (pc->setfromoptionscalled && !pc->setupcalled) PetscCall(KSPSetFromOptions(sred->ksp));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Telescope(PC pc, Vec x, Vec y)
{
  PC_Telescope       sred = (PC_Telescope)pc->data;
  Vec                xtmp, xred, yred;
  PetscInt           i, st, ed;
  VecScatter         scatter;
  PetscScalar       *array;
  const PetscScalar *x_array;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));

  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  xred    = sred->xred;
  yred    = sred->yred;

  /* pull in vector x->xtmp */
  PetscCall(VecScatterBegin(scatter, x, xtmp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, x, xtmp, INSERT_VALUES, SCATTER_FORWARD));

  /* copy vector entries into xred */
  PetscCall(VecGetArrayRead(xtmp, &x_array));
  if (xred) {
    PetscScalar *LA_xred;
    PetscCall(VecGetOwnershipRange(xred, &st, &ed));
    PetscCall(VecGetArray(xred, &LA_xred));
    for (i = 0; i < ed - st; i++) LA_xred[i] = x_array[i];
    PetscCall(VecRestoreArray(xred, &LA_xred));
  }
  PetscCall(VecRestoreArrayRead(xtmp, &x_array));
  /* solve */
  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(KSPSolve(sred->ksp, xred, yred));
    PetscCall(KSPCheckSolve(sred->ksp, pc, yred));
  }
  /* return vector */
  PetscCall(VecGetArray(xtmp, &array));
  if (yred) {
    const PetscScalar *LA_yred;
    PetscCall(VecGetOwnershipRange(yred, &st, &ed));
    PetscCall(VecGetArrayRead(yred, &LA_yred));
    for (i = 0; i < ed - st; i++) array[i] = LA_yred[i];
    PetscCall(VecRestoreArrayRead(yred, &LA_yred));
  }
  PetscCall(VecRestoreArray(xtmp, &array));
  PetscCall(VecScatterBegin(scatter, xtmp, y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter, xtmp, y, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyRichardson_Telescope(PC pc, Vec x, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool zeroguess, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_Telescope       sred = (PC_Telescope)pc->data;
  Vec                xtmp, yred;
  PetscInt           i, st, ed;
  VecScatter         scatter;
  const PetscScalar *x_array;
  PetscBool          default_init_guess_value;

  PetscFunctionBegin;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  yred    = sred->yred;

  PetscCheck(its <= 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCApplyRichardson_Telescope only supports max_it = 1");
  *reason = (PCRichardsonConvergedReason)0;

  if (!zeroguess) {
    PetscCall(PetscInfo(pc, "PCTelescope: Scattering y for non-zero initial guess\n"));
    /* pull in vector y->xtmp */
    PetscCall(VecScatterBegin(scatter, y, xtmp, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, y, xtmp, INSERT_VALUES, SCATTER_FORWARD));

    /* copy vector entries into xred */
    PetscCall(VecGetArrayRead(xtmp, &x_array));
    if (yred) {
      PetscScalar *LA_yred;
      PetscCall(VecGetOwnershipRange(yred, &st, &ed));
      PetscCall(VecGetArray(yred, &LA_yred));
      for (i = 0; i < ed - st; i++) LA_yred[i] = x_array[i];
      PetscCall(VecRestoreArray(yred, &LA_yred));
    }
    PetscCall(VecRestoreArrayRead(xtmp, &x_array));
  }

  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(KSPGetInitialGuessNonzero(sred->ksp, &default_init_guess_value));
    if (!zeroguess) PetscCall(KSPSetInitialGuessNonzero(sred->ksp, PETSC_TRUE));
  }

  PetscCall(PCApply_Telescope(pc, x, y));

  if (PCTelescope_isActiveRank(sred)) PetscCall(KSPSetInitialGuessNonzero(sred->ksp, default_init_guess_value));

  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Telescope(PC pc)
{
  PC_Telescope sred = (PC_Telescope)pc->data;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&sred->isin));
  PetscCall(VecScatterDestroy(&sred->scatter));
  PetscCall(VecDestroy(&sred->xred));
  PetscCall(VecDestroy(&sred->yred));
  PetscCall(VecDestroy(&sred->xtmp));
  PetscCall(MatDestroy(&sred->Bred));
  PetscCall(KSPReset(sred->ksp));
  if (sred->pctelescope_reset_type) PetscCall(sred->pctelescope_reset_type(pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Telescope(PC pc)
{
  PC_Telescope sred = (PC_Telescope)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_Telescope(pc));
  PetscCall(KSPDestroy(&sred->ksp));
  PetscCall(PetscSubcommDestroy(&sred->psubcomm));
  PetscCall(PetscFree(sred->dm_ctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetSubcommType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetSubcommType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetReductionFactor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetReductionFactor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetIgnoreDM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetIgnoreDM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetIgnoreKSPComputeOperators_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetIgnoreKSPComputeOperators_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetDM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetUseCoarseDM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetUseCoarseDM_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Telescope(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_Telescope     sred = (PC_Telescope)pc->data;
  MPI_Comm         comm;
  PetscMPIInt      size;
  PetscBool        flg;
  PetscSubcommType subcommtype;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscOptionsHeadBegin(PetscOptionsObject, "Telescope options");
  PetscCall(PetscOptionsEnum("-pc_telescope_subcomm_type", "Subcomm type (interlaced or contiguous)", "PCTelescopeSetSubcommType", PetscSubcommTypes, (PetscEnum)sred->subcommtype, (PetscEnum *)&subcommtype, &flg));
  if (flg) PetscCall(PCTelescopeSetSubcommType(pc, subcommtype));
  PetscCall(PetscOptionsInt("-pc_telescope_reduction_factor", "Factor to reduce comm size by", "PCTelescopeSetReductionFactor", sred->redfactor, &sred->redfactor, NULL));
  PetscCheck(sred->redfactor <= size, comm, PETSC_ERR_ARG_WRONG, "-pc_telescope_reduction_factor <= comm size");
  PetscCall(PetscOptionsBool("-pc_telescope_ignore_dm", "Ignore any DM attached to the PC", "PCTelescopeSetIgnoreDM", sred->ignore_dm, &sred->ignore_dm, NULL));
  PetscCall(PetscOptionsBool("-pc_telescope_ignore_kspcomputeoperators", "Ignore method used to compute A", "PCTelescopeSetIgnoreKSPComputeOperators", sred->ignore_kspcomputeoperators, &sred->ignore_kspcomputeoperators, NULL));
  PetscCall(PetscOptionsBool("-pc_telescope_use_coarse_dm", "Define sub-communicator from the coarse DM", "PCTelescopeSetUseCoarseDM", sred->use_coarse_dm, &sred->use_coarse_dm, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/* PC simplementation specific API's */

static PetscErrorCode PCTelescopeGetKSP_Telescope(PC pc, KSP *ksp)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (ksp) *ksp = red->ksp;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetSubcommType_Telescope(PC pc, PetscSubcommType *subcommtype)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (subcommtype) *subcommtype = red->subcommtype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetSubcommType_Telescope(PC pc, PetscSubcommType subcommtype)
{
  PC_Telescope red = (PC_Telescope)pc->data;

  PetscFunctionBegin;
  PetscCheck(!pc->setupcalled, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "You cannot change the subcommunicator type for PCTelescope after it has been set up.");
  red->subcommtype = subcommtype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetReductionFactor_Telescope(PC pc, PetscInt *fact)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (fact) *fact = red->redfactor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetReductionFactor_Telescope(PC pc, PetscInt fact)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscMPIInt  size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCheck(fact > 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Reduction factor of telescoping PC %" PetscInt_FMT " must be positive", fact);
  PetscCheck(fact <= size, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Reduction factor of telescoping PC %" PetscInt_FMT " must be <= comm.size", fact);
  red->redfactor = fact;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetIgnoreDM_Telescope(PC pc, PetscBool *v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (v) *v = red->ignore_dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetIgnoreDM_Telescope(PC pc, PetscBool v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  red->ignore_dm = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetUseCoarseDM_Telescope(PC pc, PetscBool *v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (v) *v = red->use_coarse_dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetUseCoarseDM_Telescope(PC pc, PetscBool v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  red->use_coarse_dm = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetIgnoreKSPComputeOperators_Telescope(PC pc, PetscBool *v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (v) *v = red->ignore_kspcomputeoperators;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetIgnoreKSPComputeOperators_Telescope(PC pc, PetscBool v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  red->ignore_kspcomputeoperators = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetDM_Telescope(PC pc, DM *dm)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  *dm = private_PCTelescopeGetSubDM(red);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetKSP - Gets the `KSP` created by the telescoping `PC`.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  subksp - the `KSP` defined the smaller set of processes

 Level: advanced

.seealso: `PCTELESCOPE`
@*/
PetscErrorCode PCTelescopeGetKSP(PC pc, KSP *subksp)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetKSP_C", (PC, KSP *), (pc, subksp));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetReductionFactor - Gets the factor by which the original number of MPI ranks  has been reduced by.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  fact - the reduction factor

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetReductionFactor()`
@*/
PetscErrorCode PCTelescopeGetReductionFactor(PC pc, PetscInt *fact)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetReductionFactor_C", (PC, PetscInt *), (pc, fact));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetReductionFactor - Sets the factor by which the original number of MPI ranks will been reduced by.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  fact - the reduction factor

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeGetReductionFactor()`
@*/
PetscErrorCode PCTelescopeSetReductionFactor(PC pc, PetscInt fact)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCTelescopeSetReductionFactor_C", (PC, PetscInt), (pc, fact));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetIgnoreDM - Get the flag indicating if any `DM` attached to the `PC` will be used.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - the flag

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetIgnoreDM()`
@*/
PetscErrorCode PCTelescopeGetIgnoreDM(PC pc, PetscBool *v)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetIgnoreDM_C", (PC, PetscBool *), (pc, v));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetIgnoreDM - Set a flag to ignore any DM attached to the PC.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - Use PETSC_TRUE to ignore any DM

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeGetIgnoreDM()`
@*/
PetscErrorCode PCTelescopeSetIgnoreDM(PC pc, PetscBool v)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCTelescopeSetIgnoreDM_C", (PC, PetscBool), (pc, v));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetUseCoarseDM - Get the flag indicating if the coarse `DM` attached to `DM` associated with the `PC` will be used.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - the flag

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetIgnoreDM()`, `PCTelescopeSetUseCoarseDM()`
@*/
PetscErrorCode PCTelescopeGetUseCoarseDM(PC pc, PetscBool *v)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetUseCoarseDM_C", (PC, PetscBool *), (pc, v));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetUseCoarseDM - Set a flag to query the `DM` attached to the `PC` if it also has a coarse `DM`

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - Use `PETSC_FALSE` to ignore any coarse `DM`

 Notes:
 When you have specified to use a coarse `DM`, the communicator used to create the sub-KSP within `PCTELESCOPE`
 will be that of the coarse `DM`. Hence the flags -pc_telescope_reduction_factor and
 -pc_telescope_subcomm_type will no longer have any meaning.
 It is required that the communicator associated with the parent (fine) and the coarse `DM` are of different sizes.
 An error will occur of the size of the communicator associated with the coarse `DM`
 is the same as that of the parent `DM`.
 Furthermore, it is required that the communicator on the coarse DM is a sub-communicator of the parent.
 This will be checked at the time the preconditioner is setup and an error will occur if
 the coarse DM does not define a sub-communicator of that used by the parent DM.

 The particular Telescope setup invoked when using a coarse DM is agnostic with respect to the type of
 the `DM` used (e.g. it supports `DMSHELL`, `DMPLEX`, etc).

 Support is currently only provided for the case when you are using `KSPSetComputeOperators()`

 The user is required to compose a function with the parent DM to facilitate the transfer of fields (`Vec`) between the different decompositions defined by the fine and coarse `DM`s.
 In the user code, this is achieved via
.vb
   {
     DM dm_fine;
     PetscObjectCompose((PetscObject)dm_fine,"PCTelescopeFieldScatter",your_field_scatter_method);
   }
.ve
 The signature of the user provided field scatter method is
.vb
   PetscErrorCode your_field_scatter_method(DM dm_fine,Vec x_fine,ScatterMode mode,DM dm_coarse,Vec x_coarse);
.ve
 The user must provide support for both mode = `SCATTER_FORWARD` and mode = `SCATTER_REVERSE`.
 `SCATTER_FORWARD` implies the direction of transfer is from the parent (fine) `DM` to the coarse `DM`.

 Optionally, the user may also compose a function with the parent DM to facilitate the transfer
 of state variables between the fine and coarse `DM`s.
 In the context of a finite element discretization, an example state variable might be
 values associated with quadrature points within each element.
 A user provided state scatter method is composed via
.vb
   {
     DM dm_fine;
     PetscObjectCompose((PetscObject)dm_fine,"PCTelescopeStateScatter",your_state_scatter_method);
   }
.ve
 The signature of the user provided state scatter method is
.vb
   PetscErrorCode your_state_scatter_method(DM dm_fine,ScatterMode mode,DM dm_coarse);
.ve
 `SCATTER_FORWARD` implies the direction of transfer is from the fine `DM` to the coarse `DM`.
 The user is only required to support mode = `SCATTER_FORWARD`.
 No assumption is made about the data type of the state variables.
 These must be managed by the user and must be accessible from the `DM`.

 Care must be taken in defining the user context passed to `KSPSetComputeOperators()` which is to be
 associated with the sub-`KSP` residing within `PCTELESCOPE`.
 In general, `PCTELESCOPE` assumes that the context on the fine and coarse `DM` used with
 `KSPSetComputeOperators()` should be "similar" in type or origin.
 Specifically the following rules are used to infer what context on the sub-`KSP` should be.

 First the contexts from the `KSP` and the fine and coarse `DM`s are retrieved.
 Note that the special case of a `DMSHELL` context is queried.

.vb
   DMKSPGetComputeOperators(dm_fine,&dmfine_kspfunc,&dmfine_kspctx);
   DMGetApplicationContext(dm_fine,&dmfine_appctx);
   DMShellGetContext(dm_fine,&dmfine_shellctx);

   DMGetApplicationContext(dm_coarse,&dmcoarse_appctx);
   DMShellGetContext(dm_coarse,&dmcoarse_shellctx);
.ve

 The following rules are then enforced:

 1. If dmfine_kspctx = NULL, then we provide a NULL pointer as the context for the sub-KSP:
 `KSPSetComputeOperators`(sub_ksp,dmfine_kspfunc,NULL);

 2. If dmfine_kspctx != NULL and dmfine_kspctx == dmfine_appctx,
 check that dmcoarse_appctx is also non-NULL. If this is true, then:
 `KSPSetComputeOperators`(sub_ksp,dmfine_kspfunc,dmcoarse_appctx);

 3. If dmfine_kspctx != NULL and dmfine_kspctx == dmfine_shellctx,
 check that dmcoarse_shellctx is also non-NULL. If this is true, then:
 `KSPSetComputeOperators`(sub_ksp,dmfine_kspfunc,dmcoarse_shellctx);

 If neither of the above three tests passed, then `PCTELESCOPE` cannot safely determine what
 context should be provided to `KSPSetComputeOperators()` for use with the sub-`KSP`.
 In this case, an additional mechanism is provided via a composed function which will return
 the actual context to be used. To use this feature you must compose the "getter" function
 with the coarse `DM`, e.g.
.vb
   {
     DM dm_coarse;
     PetscObjectCompose((PetscObject)dm_coarse,"PCTelescopeGetCoarseDMKSPContext",your_coarse_context_getter);
   }
.ve
 The signature of the user provided method is
.vb
   PetscErrorCode your_coarse_context_getter(DM dm_coarse,void **your_kspcontext);
.ve

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetIgnoreDM()`, `PCTelescopeSetUseCoarseDM()`
@*/
PetscErrorCode PCTelescopeSetUseCoarseDM(PC pc, PetscBool v)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCTelescopeSetUseCoarseDM_C", (PC, PetscBool), (pc, v));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetIgnoreKSPComputeOperators - Get the flag indicating if `KSPComputeOperators()` will be used.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - the flag

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetIgnoreDM()`, `PCTelescopeSetUseCoarseDM()`, `PCTelescopeSetIgnoreKSPComputeOperators()`
@*/
PetscErrorCode PCTelescopeGetIgnoreKSPComputeOperators(PC pc, PetscBool *v)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetIgnoreKSPComputeOperators_C", (PC, PetscBool *), (pc, v));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetIgnoreKSPComputeOperators - Set a flag to ignore `KSPComputeOperators()`.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - Use `PETSC_TRUE` to ignore the method (if defined) set via `KSPSetComputeOperators()` on pc

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetIgnoreDM()`, `PCTelescopeSetUseCoarseDM()`, `PCTelescopeGetIgnoreKSPComputeOperators()`
@*/
PetscErrorCode PCTelescopeSetIgnoreKSPComputeOperators(PC pc, PetscBool v)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCTelescopeSetIgnoreKSPComputeOperators_C", (PC, PetscBool), (pc, v));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetDM - Get the re-partitioned `DM` attached to the sub-`KSP`.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  subdm - The re-partitioned DM

 Level: advanced

.seealso: `PCTELESCOPE`, `PCTelescopeSetIgnoreDM()`, `PCTelescopeSetUseCoarseDM()`, `PCTelescopeGetIgnoreKSPComputeOperators()`
@*/
PetscErrorCode PCTelescopeGetDM(PC pc, DM *subdm)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetDM_C", (PC, DM *), (pc, subdm));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetSubcommType - set subcommunicator type (interlaced or contiguous)

 Logically Collective

 Input Parameters:
+  pc - the preconditioner context
-  subcommtype - the subcommunicator type (see `PetscSubcommType`)

 Level: advanced

.seealso: `PetscSubcommType`, `PetscSubcomm`, `PCTELESCOPE`
@*/
PetscErrorCode PCTelescopeSetSubcommType(PC pc, PetscSubcommType subcommtype)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCTelescopeSetSubcommType_C", (PC, PetscSubcommType), (pc, subcommtype));
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetSubcommType - Get the subcommunicator type (interlaced or contiguous)

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  subcommtype - the subcommunicator type (see `PetscSubcommType`)

 Level: advanced

.seealso: `PetscSubcomm`, `PetscSubcommType`, `PCTELESCOPE`
@*/
PetscErrorCode PCTelescopeGetSubcommType(PC pc, PetscSubcommType *subcommtype)
{
  PetscFunctionBegin;
  PetscUseMethod(pc, "PCTelescopeGetSubcommType_C", (PC, PetscSubcommType *), (pc, subcommtype));
  PetscFunctionReturn(0);
}

/*MC
   PCTELESCOPE - Runs a `KSP` solver on a sub-communicator. MPI ranks not in the sub-communicator are idle during the solve.

   Options Database Keys:
+  -pc_telescope_reduction_factor <r> - factor to reduce the communicator size by. e.g. with 64 MPI ranks and r=4, the new sub-communicator will have 64/4 = 16 ranks.
.  -pc_telescope_ignore_dm  - flag to indicate whether an attached DM should be ignored.
.  -pc_telescope_subcomm_type <interlaced,contiguous> - defines the selection of MPI ranks on the sub-communicator. see PetscSubcomm for more information.
.  -pc_telescope_ignore_kspcomputeoperators - flag to indicate whether `KSPSetComputeOperators()` should be used on the sub-KSP.
-  -pc_telescope_use_coarse_dm - flag to indicate whether the coarse `DM` should be used to define the sub-communicator.

   Level: advanced

   Notes:
   Assuming that the parent preconditioner `PC` is defined on a communicator c, this implementation
   creates a child sub-communicator (c') containing fewer MPI ranks than the original parent preconditioner `PC`.
   The preconditioner is deemed telescopic as it only calls `KSPSolve()` on a single
   sub-communicator, in contrast with `PCREDUNDANT` which calls `KSPSolve()` on N sub-communicators.
   This means there will be MPI ranks which will be idle during the application of this preconditioner.
   Additionally, in comparison with `PCREDUNDANT`, `PCTELESCOPE` can utilize an attached `DM`.

   The default type of the sub `KSP` (the `KSP` defined on c') is `KSPPREONLY`.

   There are three setup mechanisms for `PCTELESCOPE`. Features support by each type are described below.
   In the following, we will refer to the operators B and B', these are the Bmat provided to the `KSP` on the
   communicators c and c' respectively.

   [1] Default setup
   The sub-communicator c' is created via `PetscSubcommCreate()`.
   Explicitly defined nullspace and near nullspace vectors will be propagated from B to B'.
   Currently there is no support define nullspaces via a user supplied method (e.g. as passed to `MatNullSpaceSetFunction()`).
   No support is provided for `KSPSetComputeOperators()`.
   Currently there is no support for the flag -pc_use_amat.

   [2] `DM` aware setup
   If a `DM` is attached to the `PC`, it is re-partitioned on the sub-communicator c'.
   c' is created via `PetscSubcommCreate()`.
   Both the Bmat operator and the right hand side vector are permuted into the new DOF ordering defined by the re-partitioned `DM`.
   Currently only support for re-partitioning a `DMDA` is provided.
   Any explicitly defined nullspace or near nullspace vectors attached to the original Bmat operator (B) are extracted, re-partitioned and set on the re-partitioned Bmat operator (B').
   Currently there is no support define nullspaces via a user supplied method (e.g. as passed to `MatNullSpaceSetFunction()`).
   Support is provided for `KSPSetComputeOperators()`. The user provided function and context is propagated to the sub `KSP`.
   This is fragile since the user must ensure that their user context is valid for use on c'.
   Currently there is no support for the flag -pc_use_amat.

   [3] Coarse `DM` setup
   If a `DM` (dmfine) is attached to the `PC`, dmfine is queried for a "coarse" `DM` (call this dmcoarse) via `DMGetCoarseDM()`.
   `PCTELESCOPE` will interpret the coarse `DM` as being defined on a sub-communicator of c.
   The communicator associated with dmcoarse will define the c' to be used within `PCTELESCOPE`.
   `PCTELESCOPE` will check that c' is in fact a sub-communicator of c. If it is not, an error will be reported.
   The intention of this setup type is that` PCTELESCOPE` will use an existing (e.g. user defined) communicator hierarchy, say as would be
   available with using multi-grid on unstructured meshes.
   This setup will not use the command line options -pc_telescope_reduction_factor or -pc_telescope_subcomm_type.
   Any explicitly defined nullspace or near nullspace vectors attached to the original Bmat operator (B) are extracted, scattered into the correct ordering consistent with dmcoarse and set on B'.
   Currently there is no support define nullspaces via a user supplied method (e.g. as passed to `MatNullSpaceSetFunction()`).
   There is no general method to permute field orderings, hence only `KSPSetComputeOperators()` is supported.
   The user must use `PetscObjectComposeFunction()` with dmfine to define the method to scatter fields from dmfine to dmcoarse.
   Propagation of the user context for `KSPSetComputeOperators()` on the sub `KSP` is attempted by querying the `DM` contexts associated with dmfine and dmcoarse. Alternatively, the user may use `PetscObjectComposeFunction()` with dmcoarse to define a method which will return the appropriate user context for `KSPSetComputeOperators()`.
   Currently there is no support for the flag -pc_use_amat.
   This setup can be invoked by the option -pc_telescope_use_coarse_dm or by calling `PCTelescopeSetUseCoarseDM`(pc,`PETSC_TRUE`);
   Further information about the user-provided methods required by this setup type are described here `PCTelescopeSetUseCoarseDM()`.

   Developer Notes:
   During `PCSetup()`, the B operator is scattered onto c'.
   Within `PCApply()`, the RHS vector (x) is scattered into a redundant vector, xred (defined on c').
   Then, `KSPSolve()` is executed on the c' communicator.

   The communicator used within the telescoping preconditioner is defined by a `PetscSubcomm` using the INTERLACED
   creation routine by default (this can be changed with -pc_telescope_subcomm_type). We run the sub `KSP` on only the ranks within the communicator which have a color equal to zero.

   The telescoping preconditioner is aware of nullspaces and near nullspaces which are attached to the B operator.
   In the case where B has a (near) nullspace attached, the (near) nullspace vectors are extracted from B and mapped into
   a new (near) nullspace, defined on the sub-communicator, which is attached to B' (the B operator which was scattered to c')

   The telescoping preconditioner can re-partition an attached DM if it is a `DMDA` (2D or 3D -
   support for 1D `DMDA`s is not provided). If a `DMDA` is found, a topologically equivalent `DMDA` is created on c'
   and this new `DM` is attached the sub `KSP`. The design of telescope is such that it should be possible to extend support
   for re-partitioning other to DM's (e.g. `DMPLEX`). The user can supply a flag to ignore attached DMs.
   Alternatively, user-provided re-partitioned DMs can be used via -pc_telescope_use_coarse_dm.

   With the default setup mode, B' is defined by fusing rows (in order) associated with MPI ranks common to c and c'.

   When a `DMDA` is attached to the parent preconditioner, B' is defined by: (i) performing a symmetric permutation of B
   into the ordering defined by the `DMDA` on c', (ii) extracting the local chunks via `MatCreateSubMatrices()`, (iii) fusing the
   locally (sequential) matrices defined on the ranks common to c and c' into B' using `MatCreateMPIMatConcatenateSeqMat()`

   Limitations/improvements include the following.
   `VecPlaceArray()` could be used within `PCApply()` to improve efficiency and reduce memory usage.
   A unified mechanism to query for user contexts as required by `KSPSetComputeOperators()` and `MatNullSpaceSetFunction()`.

   The symmetric permutation used when a `DMDA` is encountered is performed via explicitly assembling a permutation matrix P,
   and performing P^T.A.P. Possibly it might be more efficient to use `MatPermute()`. We opted to use P^T.A.P as it appears
   `VecPermute()` does not support the use case required here. By computing P, one can permute both the operator and RHS in a
   consistent manner.

   Mapping of vectors (default setup mode) is performed in the following way.
   Suppose the parent communicator size was 4, and we set a reduction factor of 2; this would give a comm size on c' of 2.
   Using the interlaced creation routine, the ranks in c with color = 0 will be rank 0 and 2.
   We perform the scatter to the sub-communicator in the following way.
   [1] Given a vector x defined on communicator c

.vb
   rank(c)  local values of x
   ------- ----------------------------------------
        0   [  0.0,  1.0,  2.0,  3.0,  4.0,  5.0 ]
        1   [  6.0,  7.0,  8.0,  9.0, 10.0, 11.0 ]
        2   [ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0 ]
        3   [ 18.0, 19.0, 20.0, 21.0, 22.0, 23.0 ]
.ve

   scatter into xtmp defined also on comm c, so that we have the following values

.vb
   rank(c)  local values of xtmp
   ------- ----------------------------------------------------------------------------
        0   [  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0 ]
        1   [ ]
        2   [ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0 ]
        3   [ ]
.ve

   The entries on rank 1 and 3 (ranks which do not have a color = 0 in c') have no values

   [2] Copy the values from ranks 0, 2 (indices with respect to comm c) into the vector xred which is defined on communicator c'.
   Ranks 0 and 2 are the only ranks in the subcomm which have a color = 0.

.vb
   rank(c')  local values of xred
   -------- ----------------------------------------------------------------------------
         0   [  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0 ]
         1   [ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0 ]
.ve

  Contributed by Dave May

  Reference:
  Dave A. May, Patrick Sanan, Karl Rupp, Matthew G. Knepley, and Barry F. Smith, "Extreme-Scale Multigrid Components within PETSc". 2016. In Proceedings of the Platform for Advanced Scientific Computing Conference (PASC '16). DOI: 10.1145/2929908.2929913

.seealso: `PCTelescopeGetKSP()`, `PCTelescopeGetDM()`, `PCTelescopeGetReductionFactor()`, `PCTelescopeSetReductionFactor()`, `PCTelescopeGetIgnoreDM()`, `PCTelescopeSetIgnoreDM()`, `PCREDUNDANT`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_Telescope(PC pc)
{
  struct _PC_Telescope *sred;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sred));
  sred->psubcomm                   = NULL;
  sred->subcommtype                = PETSC_SUBCOMM_INTERLACED;
  sred->subcomm                    = MPI_COMM_NULL;
  sred->redfactor                  = 1;
  sred->ignore_dm                  = PETSC_FALSE;
  sred->ignore_kspcomputeoperators = PETSC_FALSE;
  sred->use_coarse_dm              = PETSC_FALSE;
  pc->data                         = (void *)sred;

  pc->ops->apply           = PCApply_Telescope;
  pc->ops->applytranspose  = NULL;
  pc->ops->applyrichardson = PCApplyRichardson_Telescope;
  pc->ops->setup           = PCSetUp_Telescope;
  pc->ops->destroy         = PCDestroy_Telescope;
  pc->ops->reset           = PCReset_Telescope;
  pc->ops->setfromoptions  = PCSetFromOptions_Telescope;
  pc->ops->view            = PCView_Telescope;

  sred->pctelescope_setup_type              = PCTelescopeSetUp_default;
  sred->pctelescope_matcreate_type          = PCTelescopeMatCreate_default;
  sred->pctelescope_matnullspacecreate_type = PCTelescopeMatNullSpaceCreate_default;
  sred->pctelescope_reset_type              = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetKSP_C", PCTelescopeGetKSP_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetSubcommType_C", PCTelescopeGetSubcommType_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetSubcommType_C", PCTelescopeSetSubcommType_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetReductionFactor_C", PCTelescopeGetReductionFactor_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetReductionFactor_C", PCTelescopeSetReductionFactor_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetIgnoreDM_C", PCTelescopeGetIgnoreDM_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetIgnoreDM_C", PCTelescopeSetIgnoreDM_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetIgnoreKSPComputeOperators_C", PCTelescopeGetIgnoreKSPComputeOperators_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetIgnoreKSPComputeOperators_C", PCTelescopeSetIgnoreKSPComputeOperators_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetDM_C", PCTelescopeGetDM_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeGetUseCoarseDM_C", PCTelescopeGetUseCoarseDM_Telescope));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCTelescopeSetUseCoarseDM_C", PCTelescopeSetUseCoarseDM_Telescope));
  PetscFunctionReturn(0);
}
