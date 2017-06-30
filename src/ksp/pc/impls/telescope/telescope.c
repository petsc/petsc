
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/
#include <petscdm.h> /*I "petscdm.h" I*/
#include "../src/ksp/pc/impls/telescope/telescope.h"

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
"@inproceedings{MaySananRuppKnepleySmith2016,\n"
"  title     = {Extreme-Scale Multigrid Components within PETSc},\n"
"  author    = {Dave A. May and Patrick Sanan and Karl Rupp and Matthew G. Knepley and Barry F. Smith},\n"
"  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},\n"
"  series    = {PASC '16},\n"
"  isbn      = {978-1-4503-4126-4},\n"
"  location  = {Lausanne, Switzerland},\n"
"  pages     = {5:1--5:12},\n"
"  articleno = {5},\n"
"  numpages  = {12},\n"
"  url       = {http://doi.acm.org/10.1145/2929908.2929913},\n"
"  doi       = {10.1145/2929908.2929913},\n"
"  acmid     = {2929913},\n"
"  publisher = {ACM},\n"
"  address   = {New York, NY, USA},\n"
"  keywords  = {GPU, HPC, agglomeration, coarse-level solver, multigrid, parallel computing, preconditioning},\n"
"  year      = {2016}\n"
"}\n";

/*
 PCTelescopeSetUp_default()
 PCTelescopeMatCreate_default()

 default

 // scatter in
 x(comm) -> xtmp(comm)

 xred(subcomm) <- xtmp
 yred(subcomm)

 yred(subcomm) --> xtmp

 // scatter out
 xtmp(comm) -> y(comm)
*/

PetscBool isActiveRank(PetscSubcomm scomm)
{
  if (scomm->color == 0) { return PETSC_TRUE; }
  else { return PETSC_FALSE; }
}

DM private_PCTelescopeGetSubDM(PC_Telescope sred)
{
  DM subdm = NULL;

  if (!isActiveRank(sred->psubcomm)) { subdm = NULL; }
  else {
    switch (sred->sr_type) {
    case TELESCOPE_DEFAULT: subdm = NULL;
      break;
    case TELESCOPE_DMDA:    subdm = ((PC_Telescope_DMDACtx*)sred->dm_ctx)->dmrepart;
      break;
    case TELESCOPE_DMPLEX:  subdm = NULL;
      break;
    }
  }
  return(subdm);
}

PetscErrorCode PCTelescopeSetUp_default(PC pc,PC_Telescope sred)
{
  PetscErrorCode ierr;
  PetscInt       m,M,bs,st,ed;
  Vec            x,xred,yred,xtmp;
  Mat            B;
  MPI_Comm       comm,subcomm;
  VecScatter     scatter;
  IS             isin;

  PetscFunctionBegin;
  ierr = PetscInfo(pc,"PCTelescope: setup (default)\n");CHKERRQ(ierr);
  comm = PetscSubcommParent(sred->psubcomm);
  subcomm = PetscSubcommChild(sred->psubcomm);

  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatGetSize(B,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(B,&bs);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,NULL);CHKERRQ(ierr);

  xred = NULL;
  m    = 0;
  if (isActiveRank(sred->psubcomm)) {
    ierr = VecCreate(subcomm,&xred);CHKERRQ(ierr);
    ierr = VecSetSizes(xred,PETSC_DECIDE,M);CHKERRQ(ierr);
    ierr = VecSetBlockSize(xred,bs);CHKERRQ(ierr);
    ierr = VecSetFromOptions(xred);CHKERRQ(ierr);
    ierr = VecGetLocalSize(xred,&m);CHKERRQ(ierr);
  }

  yred = NULL;
  if (isActiveRank(sred->psubcomm)) {
    ierr = VecDuplicate(xred,&yred);CHKERRQ(ierr);
  }

  ierr = VecCreate(comm,&xtmp);CHKERRQ(ierr);
  ierr = VecSetSizes(xtmp,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(xtmp,bs);CHKERRQ(ierr);
  ierr = VecSetType(xtmp,((PetscObject)x)->type_name);CHKERRQ(ierr);

  if (isActiveRank(sred->psubcomm)) {
    ierr = VecGetOwnershipRange(xred,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,(ed-st),st,1,&isin);CHKERRQ(ierr);
  } else {
    ierr = VecGetOwnershipRange(x,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,0,st,1,&isin);CHKERRQ(ierr);
  }
  ierr = ISSetBlockSize(isin,bs);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,isin,xtmp,NULL,&scatter);CHKERRQ(ierr);

  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xred    = xred;
  sred->yred    = yred;
  sred->xtmp    = xtmp;
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatCreate_default(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A)
{
  PetscErrorCode ierr;
  MPI_Comm       comm,subcomm;
  Mat            Bred,B;
  PetscInt       nr,nc;
  IS             isrow,iscol;
  Mat            Blocal,*_Blocal;

  PetscFunctionBegin;
  ierr = PetscInfo(pc,"PCTelescope: updating the redundant preconditioned operator (default)\n");CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  subcomm = PetscSubcommChild(sred->psubcomm);
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatGetSize(B,&nr,&nc);CHKERRQ(ierr);
  isrow = sred->isin;
  ierr = ISCreateStride(comm,nc,0,1,&iscol);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(B,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&_Blocal);CHKERRQ(ierr);
  Blocal = *_Blocal;
  ierr = PetscFree(_Blocal);CHKERRQ(ierr);
  Bred = NULL;
  if (isActiveRank(sred->psubcomm)) {
    PetscInt mm;

    if (reuse != MAT_INITIAL_MATRIX) { Bred = *A; }

    ierr = MatGetSize(Blocal,&mm,NULL);CHKERRQ(ierr);
    ierr = MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,mm,reuse,&Bred);CHKERRQ(ierr);
  }
  *A = Bred;
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&Blocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSubNullSpaceCreate_Telescope(PC pc,PC_Telescope sred,MatNullSpace nullspace,MatNullSpace *sub_nullspace)
{
  PetscErrorCode   ierr;
  PetscBool        has_const;
  const Vec        *vecs;
  Vec              *sub_vecs = NULL;
  PetscInt         i,k,n = 0;
  MPI_Comm         subcomm;

  PetscFunctionBegin;
  subcomm = PetscSubcommChild(sred->psubcomm);
  ierr = MatNullSpaceGetVecs(nullspace,&has_const,&n,&vecs);CHKERRQ(ierr);

  if (isActiveRank(sred->psubcomm)) {
    if (n) {
      ierr = VecDuplicateVecs(sred->xred,n,&sub_vecs);CHKERRQ(ierr);
    }
  }

  /* copy entries */
  for (k=0; k<n; k++) {
    const PetscScalar *x_array;
    PetscScalar       *LA_sub_vec;
    PetscInt          st,ed;

    /* pull in vector x->xtmp */
    ierr = VecScatterBegin(sred->scatter,vecs[k],sred->xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sred->scatter,vecs[k],sred->xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (sub_vecs) {
      /* copy vector entries into xred */
      ierr = VecGetArrayRead(sred->xtmp,&x_array);CHKERRQ(ierr);
      if (sub_vecs[k]) {
        ierr = VecGetOwnershipRange(sub_vecs[k],&st,&ed);CHKERRQ(ierr);
        ierr = VecGetArray(sub_vecs[k],&LA_sub_vec);CHKERRQ(ierr);
        for (i=0; i<ed-st; i++) {
          LA_sub_vec[i] = x_array[i];
        }
        ierr = VecRestoreArray(sub_vecs[k],&LA_sub_vec);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(sred->xtmp,&x_array);CHKERRQ(ierr);
    }
  }

  if (isActiveRank(sred->psubcomm)) {
    /* create new (near) nullspace for redundant object */
    ierr = MatNullSpaceCreate(subcomm,has_const,n,sub_vecs,sub_nullspace);CHKERRQ(ierr);
    ierr = VecDestroyVecs(n,&sub_vecs);CHKERRQ(ierr);
    if (nullspace->remove) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Propagation of custom remove callbacks not supported when propagating (near) nullspaces with PCTelescope");
    if (nullspace->rmctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Propagation of custom remove callback context not supported when propagating (near) nullspaces with PCTelescope");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeMatNullSpaceCreate_default(PC pc,PC_Telescope sred,Mat sub_mat)
{
  PetscErrorCode   ierr;
  Mat              B;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);

  /* Propagate the nullspace if it exists */
  {
    MatNullSpace nullspace,sub_nullspace;
    ierr = MatGetNullSpace(B,&nullspace);CHKERRQ(ierr);
    if (nullspace) {
      ierr = PetscInfo(pc,"PCTelescope: generating nullspace (default)\n");CHKERRQ(ierr);
      ierr = PCTelescopeSubNullSpaceCreate_Telescope(pc,sred,nullspace,&sub_nullspace);CHKERRQ(ierr);
      if (isActiveRank(sred->psubcomm)) {
        ierr = MatSetNullSpace(sub_mat,sub_nullspace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&sub_nullspace);CHKERRQ(ierr);
      }
    }
  }

  /* Propagate the near nullspace if it exists */
  {
    MatNullSpace nearnullspace,sub_nearnullspace;
    ierr = MatGetNearNullSpace(B,&nearnullspace);CHKERRQ(ierr);
    if (nearnullspace) {
      ierr = PetscInfo(pc,"PCTelescope: generating near nullspace (default)\n");CHKERRQ(ierr);
      ierr = PCTelescopeSubNullSpaceCreate_Telescope(pc,sred,nearnullspace,&sub_nearnullspace);CHKERRQ(ierr);
      if (isActiveRank(sred->psubcomm)) {
        ierr = MatSetNearNullSpace(sub_mat,sub_nearnullspace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&sub_nearnullspace);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Telescope(PC pc,PetscViewer viewer)
{
  PC_Telescope     sred = (PC_Telescope)pc->data;
  PetscErrorCode   ierr;
  PetscBool        iascii,isstring;
  PetscViewer      subviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (!sred->psubcomm) {
      ierr = PetscViewerASCIIPrintf(viewer,"  preconditioner not yet setup\n");CHKERRQ(ierr);
    } else {
      MPI_Comm    comm,subcomm;
      PetscMPIInt comm_size,subcomm_size;
      DM          dm,subdm;

      ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
      subdm = private_PCTelescopeGetSubDM(sred);
      comm = PetscSubcommParent(sred->psubcomm);
      subcomm = PetscSubcommChild(sred->psubcomm);
      ierr = MPI_Comm_size(comm,&comm_size);CHKERRQ(ierr);
      ierr = MPI_Comm_size(subcomm,&subcomm_size);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  parent comm size reduction factor = %D\n",sred->redfactor);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  comm_size = %d , subcomm_size = %d\n",(int)comm_size,(int)subcomm_size);CHKERRQ(ierr);
      switch (sred->subcommtype) {
        case PETSC_SUBCOMM_INTERLACED :
          ierr = PetscViewerASCIIPrintf(viewer,"  subcomm type: interlaced\n",sred->subcommtype);CHKERRQ(ierr);
          break;
        case PETSC_SUBCOMM_CONTIGUOUS :
          ierr = PetscViewerASCIIPrintf(viewer,"  subcomm type: contiguous\n",sred->subcommtype);CHKERRQ(ierr);
          break;
        default :
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"General subcomm type not supported by PCTelescope");
      }
      ierr = PetscViewerGetSubViewer(viewer,subcomm,&subviewer);CHKERRQ(ierr);
      if (isActiveRank(sred->psubcomm)) {
        ierr = PetscViewerASCIIPushTab(subviewer);CHKERRQ(ierr);

        if (dm && sred->ignore_dm) {
          ierr = PetscViewerASCIIPrintf(subviewer,"  ignoring DM\n");CHKERRQ(ierr);
        }
        if (sred->ignore_kspcomputeoperators) {
          ierr = PetscViewerASCIIPrintf(subviewer,"  ignoring KSPComputeOperators\n");CHKERRQ(ierr);
        }
        switch (sred->sr_type) {
        case TELESCOPE_DEFAULT:
          ierr = PetscViewerASCIIPrintf(subviewer,"  using default setup\n");CHKERRQ(ierr);
          break;
        case TELESCOPE_DMDA:
          ierr = PetscViewerASCIIPrintf(subviewer,"  DMDA detected\n");CHKERRQ(ierr);
          ierr = DMView_DMDAShort(subdm,subviewer);CHKERRQ(ierr);
          break;
        case TELESCOPE_DMPLEX:
          ierr = PetscViewerASCIIPrintf(subviewer,"  DMPLEX detected\n");CHKERRQ(ierr);
          break;
        }

        ierr = KSPView(sred->ksp,subviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(subviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer,subcomm,&subviewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Telescope(PC pc)
{
  PC_Telescope      sred = (PC_Telescope)pc->data;
  PetscErrorCode    ierr;
  MPI_Comm          comm,subcomm=0;
  PCTelescopeType   sr_type;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);

  /* subcomm definition */
  if (!pc->setupcalled) {
    if (!sred->psubcomm) {
      ierr = PetscSubcommCreate(comm,&sred->psubcomm);CHKERRQ(ierr);
      ierr = PetscSubcommSetNumber(sred->psubcomm,sred->redfactor);CHKERRQ(ierr);
      ierr = PetscSubcommSetType(sred->psubcomm,sred->subcommtype);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)pc,sizeof(PetscSubcomm));CHKERRQ(ierr);
    }
  }
  subcomm = PetscSubcommChild(sred->psubcomm);

  /* internal KSP */
  if (!pc->setupcalled) {
    const char *prefix;

    if (isActiveRank(sred->psubcomm)) {
      ierr = KSPCreate(subcomm,&sred->ksp);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(sred->ksp,pc->erroriffailure);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)sred->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)sred->ksp);CHKERRQ(ierr);
      ierr = KSPSetType(sred->ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(sred->ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(sred->ksp,"telescope_");CHKERRQ(ierr);
    }
  }
  /* Determine type of setup/update */
  if (!pc->setupcalled) {
    PetscBool has_dm,same;
    DM        dm;

    sr_type = TELESCOPE_DEFAULT;
    has_dm = PETSC_FALSE;
    ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
    if (dm) { has_dm = PETSC_TRUE; }
    if (has_dm) {
      /* check for dmda */
      ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&same);CHKERRQ(ierr);
      if (same) {
        ierr = PetscInfo(pc,"PCTelescope: found DMDA\n");CHKERRQ(ierr);
        sr_type = TELESCOPE_DMDA;
      }
      /* check for dmplex */
      ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&same);CHKERRQ(ierr);
      if (same) {
        PetscInfo(pc,"PCTelescope: found DMPLEX\n");
        sr_type = TELESCOPE_DMPLEX;
      }
    }

    if (sred->ignore_dm) {
      PetscInfo(pc,"PCTelescope: ignore DM\n");
      sr_type = TELESCOPE_DEFAULT;
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
  case TELESCOPE_DMPLEX: SETERRQ(comm,PETSC_ERR_SUP,"Support for DMPLEX is currently not available");
    break;
  default: SETERRQ(comm,PETSC_ERR_SUP,"Only support for repartitioning DMDA is provided");
    break;
  }

  /* setup */
  if (sred->pctelescope_setup_type) {
    ierr = sred->pctelescope_setup_type(pc,sred);CHKERRQ(ierr);
  }
  /* update */
  if (!pc->setupcalled) {
    if (sred->pctelescope_matcreate_type) {
      ierr = sred->pctelescope_matcreate_type(pc,sred,MAT_INITIAL_MATRIX,&sred->Bred);CHKERRQ(ierr);
    }
    if (sred->pctelescope_matnullspacecreate_type) {
      ierr = sred->pctelescope_matnullspacecreate_type(pc,sred,sred->Bred);CHKERRQ(ierr);
    }
  } else {
    if (sred->pctelescope_matcreate_type) {
      ierr = sred->pctelescope_matcreate_type(pc,sred,MAT_REUSE_MATRIX,&sred->Bred);CHKERRQ(ierr);
    }
  }

  /* common - no construction */
  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPSetOperators(sred->ksp,sred->Bred,sred->Bred);CHKERRQ(ierr);
    if (pc->setfromoptionscalled && !pc->setupcalled){
      ierr = KSPSetFromOptions(sred->ksp);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Telescope(PC pc,Vec x,Vec y)
{
  PC_Telescope      sred = (PC_Telescope)pc->data;
  PetscErrorCode    ierr;
  Vec               xtmp,xred,yred;
  PetscInt          i,st,ed;
  VecScatter        scatter;
  PetscScalar       *array;
  const PetscScalar *x_array;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  xred    = sred->xred;
  yred    = sred->yred;

  /* pull in vector x->xtmp */
  ierr = VecScatterBegin(scatter,x,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,x,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* copy vector entries into xred */
  ierr = VecGetArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  if (xred) {
    PetscScalar *LA_xred;
    ierr = VecGetOwnershipRange(xred,&st,&ed);CHKERRQ(ierr);
    ierr = VecGetArray(xred,&LA_xred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      LA_xred[i] = x_array[i];
    }
    ierr = VecRestoreArray(xred,&LA_xred);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  /* solve */
  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPSolve(sred->ksp,xred,yred);CHKERRQ(ierr);
  }
  /* return vector */
  ierr = VecGetArray(xtmp,&array);CHKERRQ(ierr);
  if (yred) {
    const PetscScalar *LA_yred;
    ierr = VecGetOwnershipRange(yred,&st,&ed);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yred,&LA_yred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      array[i] = LA_yred[i];
    }
    ierr = VecRestoreArrayRead(yred,&LA_yred);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xtmp,&array);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,xtmp,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,xtmp,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyRichardson_Telescope(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_Telescope      sred = (PC_Telescope)pc->data;
  PetscErrorCode    ierr;
  Vec               xtmp,yred;
  PetscInt          i,st,ed;
  VecScatter        scatter;
  const PetscScalar *x_array;
  PetscBool         default_init_guess_value;

  PetscFunctionBegin;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  yred    = sred->yred;

  if (its > 1) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCApplyRichardson_Telescope only supports max_it = 1");
  *reason = (PCRichardsonConvergedReason)0;

  if (!zeroguess) {
    ierr = PetscInfo(pc,"PCTelescope: Scattering y for non-zero initial guess\n");CHKERRQ(ierr);
    /* pull in vector y->xtmp */
    ierr = VecScatterBegin(scatter,y,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter,y,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* copy vector entries into xred */
    ierr = VecGetArrayRead(xtmp,&x_array);CHKERRQ(ierr);
    if (yred) {
      PetscScalar *LA_yred;
      ierr = VecGetOwnershipRange(yred,&st,&ed);CHKERRQ(ierr);
      ierr = VecGetArray(yred,&LA_yred);CHKERRQ(ierr);
      for (i=0; i<ed-st; i++) {
        LA_yred[i] = x_array[i];
      }
      ierr = VecRestoreArray(yred,&LA_yred);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  }

  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPGetInitialGuessNonzero(sred->ksp,&default_init_guess_value);CHKERRQ(ierr);
    if (!zeroguess) ierr = KSPSetInitialGuessNonzero(sred->ksp,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = PCApply_Telescope(pc,x,y);CHKERRQ(ierr);

  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPSetInitialGuessNonzero(sred->ksp,default_init_guess_value);CHKERRQ(ierr);
  }

  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Telescope(PC pc)
{
  PC_Telescope   sred = (PC_Telescope)pc->data;
  PetscErrorCode ierr;

  ierr = ISDestroy(&sred->isin);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sred->scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&sred->xred);CHKERRQ(ierr);
  ierr = VecDestroy(&sred->yred);CHKERRQ(ierr);
  ierr = VecDestroy(&sred->xtmp);CHKERRQ(ierr);
  ierr = MatDestroy(&sred->Bred);CHKERRQ(ierr);
  ierr = KSPReset(sred->ksp);CHKERRQ(ierr);
  if (sred->pctelescope_reset_type) {
    ierr = sred->pctelescope_reset_type(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Telescope(PC pc)
{
  PC_Telescope     sred = (PC_Telescope)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCReset_Telescope(pc);CHKERRQ(ierr);
  ierr = KSPDestroy(&sred->ksp);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&sred->psubcomm);CHKERRQ(ierr);
  ierr = PetscFree(sred->dm_ctx);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Telescope(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Telescope     sred = (PC_Telescope)pc->data;
  PetscErrorCode   ierr;
  MPI_Comm         comm;
  PetscMPIInt      size;
  PetscBool        flg;
  PetscSubcommType subcommtype;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Telescope options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_telescope_subcomm_type","Subcomm type (interlaced or contiguous)","PCTelescopeSetSubcommType",PetscSubcommTypes,(PetscEnum)sred->subcommtype,(PetscEnum*)&subcommtype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCTelescopeSetSubcommType(pc,subcommtype);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-pc_telescope_reduction_factor","Factor to reduce comm size by","PCTelescopeSetReductionFactor",sred->redfactor,&sred->redfactor,0);CHKERRQ(ierr);
  if (sred->redfactor > size) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"-pc_telescope_reduction_factor <= comm size");
  ierr = PetscOptionsBool("-pc_telescope_ignore_dm","Ignore any DM attached to the PC","PCTelescopeSetIgnoreDM",sred->ignore_dm,&sred->ignore_dm,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_telescope_ignore_kspcomputeoperators","Ignore method used to compute A","PCTelescopeSetIgnoreKSPComputeOperators",sred->ignore_kspcomputeoperators,&sred->ignore_kspcomputeoperators,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* PC simplementation specific API's */

static PetscErrorCode PCTelescopeGetKSP_Telescope(PC pc,KSP *ksp)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (ksp) *ksp = red->ksp;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetSubcommType_Telescope(PC pc,PetscSubcommType *subcommtype)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (subcommtype) *subcommtype = red->subcommtype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetSubcommType_Telescope(PC pc,PetscSubcommType subcommtype)
{
  PC_Telescope     red = (PC_Telescope)pc->data;

  PetscFunctionBegin;
  if (pc->setupcalled) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"You cannot change the subcommunicator type for PCTelescope after it has been set up.");
  red->subcommtype = subcommtype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetReductionFactor_Telescope(PC pc,PetscInt *fact)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (fact) *fact = red->redfactor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetReductionFactor_Telescope(PC pc,PetscInt fact)
{
  PC_Telescope     red = (PC_Telescope)pc->data;
  PetscMPIInt      size;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  if (fact <= 0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Reduction factor of telescoping PC %D must be positive",fact);
  if (fact > size) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Reduction factor of telescoping PC %D must be <= comm.size",fact);
  red->redfactor = fact;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetIgnoreDM_Telescope(PC pc,PetscBool *v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (v) *v = red->ignore_dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetIgnoreDM_Telescope(PC pc,PetscBool v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  red->ignore_dm = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetIgnoreKSPComputeOperators_Telescope(PC pc,PetscBool *v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  if (v) *v = red->ignore_kspcomputeoperators;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetIgnoreKSPComputeOperators_Telescope(PC pc,PetscBool v)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  red->ignore_kspcomputeoperators = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeGetDM_Telescope(PC pc,DM *dm)
{
  PC_Telescope red = (PC_Telescope)pc->data;
  PetscFunctionBegin;
  *dm = private_PCTelescopeGetSubDM(red);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetKSP - Gets the KSP created by the telescoping PC.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  subksp - the KSP defined the smaller set of processes

 Level: advanced

.keywords: PC, telescopting solve
@*/
PetscErrorCode PCTelescopeGetKSP(PC pc,KSP *subksp)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(pc,"PCTelescopeGetKSP_C",(PC,KSP*),(pc,subksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetReductionFactor - Gets the factor by which the original number of processes has been reduced by.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  fact - the reduction factor

 Level: advanced

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeGetReductionFactor(PC pc,PetscInt *fact)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(pc,"PCTelescopeGetReductionFactor_C",(PC,PetscInt*),(pc,fact));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetReductionFactor - Sets the factor by which the original number of processes has been reduced by.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  fact - the reduction factor

 Level: advanced

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeSetReductionFactor(PC pc,PetscInt fact)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCTelescopeSetReductionFactor_C",(PC,PetscInt),(pc,fact));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetIgnoreDM - Get the flag indicating if any DM attached to the PC will be used.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - the flag

 Level: advanced

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeGetIgnoreDM(PC pc,PetscBool *v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(pc,"PCTelescopeGetIgnoreDM_C",(PC,PetscBool*),(pc,v));CHKERRQ(ierr);
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

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeSetIgnoreDM(PC pc,PetscBool v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCTelescopeSetIgnoreDM_C",(PC,PetscBool),(pc,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetIgnoreKSPComputeOperators - Get the flag indicating if KSPComputeOperators will be used.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - the flag

 Level: advanced

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeGetIgnoreKSPComputeOperators(PC pc,PetscBool *v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(pc,"PCTelescopeGetIgnoreKSPComputeOperators_C",(PC,PetscBool*),(pc,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetIgnoreKSPComputeOperators - Set a flag to ignore KSPComputeOperators.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  v - Use PETSC_TRUE to ignore the method (if defined) set via KSPSetComputeOperators on pc

 Level: advanced

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeSetIgnoreKSPComputeOperators(PC pc,PetscBool v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCTelescopeSetIgnoreKSPComputeOperators_C",(PC,PetscBool),(pc,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetDM - Get the re-partitioned DM attached to the sub KSP.

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  subdm - The re-partitioned DM

 Level: advanced

.keywords: PC, telescoping solve
@*/
PetscErrorCode PCTelescopeGetDM(PC pc,DM *subdm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(pc,"PCTelescopeGetDM_C",(PC,DM*),(pc,subdm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeSetSubcommType - set subcommunicator type (interlaced or contiguous)

 Logically Collective

 Input Parameter:
+  pc - the preconditioner context
-  subcommtype - the subcommunicator type (see PetscSubcommType)

 Level: advanced

.keywords: PC, telescoping solve

.seealso: PetscSubcommType, PetscSubcomm, PCTELESCOPE
@*/
PetscErrorCode PCTelescopeSetSubcommType(PC pc, PetscSubcommType subcommtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCTelescopeSetSubcommType_C",(PC,PetscSubcommType),(pc,subcommtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PCTelescopeGetSubcommType - Get the subcommunicator type (interlaced or contiguous)

 Not Collective

 Input Parameter:
.  pc - the preconditioner context

 Output Parameter:
.  subcommtype - the subcommunicator type (see PetscSubcommType)

 Level: advanced

.keywords: PC, telescoping solve

.seealso: PetscSubcomm, PetscSubcommType, PCTELESCOPE
@*/
PetscErrorCode PCTelescopeGetSubcommType(PC pc, PetscSubcommType *subcommtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(pc,"PCTelescopeGetSubcommType_C",(PC,PetscSubcommType*),(pc,subcommtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCTELESCOPE - Runs a KSP solver on a sub-group of processors. MPI processes not in the sub-communicator are idle during the solve.

   Options Database:
+  -pc_telescope_reduction_factor <r> - factor to use communicator size by. e.g. with 64 MPI processes and r=4, the new sub-communicator will have 64/4 = 16 ranks.
-  -pc_telescope_ignore_dm  - flag to indicate whether an attached DM should be ignored
-  -pc_telescope_subcomm_type <interlaced,contiguous> - how to define the reduced communicator. see PetscSubcomm for more.

   Level: advanced

   Notes:
   The preconditioner is deemed telescopic as it only calls KSPSolve() on a single
   sub-communicator, in contrast with PCREDUNDANT which calls KSPSolve() on N sub-communicators.
   This means there will be MPI processes which will be idle during the application of this preconditioner.

   The default KSP is PREONLY. If a DM is attached to the PC, it is re-partitioned on the sub-communicator.
   Both the Bmat operator and the right hand side vector are permuted into the new DOF ordering defined by the re-partitioned DM.
   Currently only support for re-partitioning a DMDA is provided.
   Any nullspace attached to the original Bmat operator is extracted, re-partitioned and set on the repartitioned Bmat operator.
   KSPSetComputeOperators() is not propagated to the sub KSP.
   Currently there is no support for the flag -pc_use_amat

   Assuming that the parent preconditioner (PC) is defined on a communicator c, this implementation
   creates a child sub-communicator (c') containing fewer MPI processes than the original parent preconditioner (PC).

  Developer Notes:
   During PCSetup, the B operator is scattered onto c'.
   Within PCApply, the RHS vector (x) is scattered into a redundant vector, xred (defined on c').
   Then, KSPSolve() is executed on the c' communicator.

   The communicator used within the telescoping preconditioner is defined by a PetscSubcomm using the INTERLACED 
   creation routine by default (this can be changed with -pc_telescope_subcomm_type). We run the sub KSP on only the ranks within the communicator which have a color equal to zero.

   The telescoping preconditioner is aware of nullspaces and near nullspaces which are attached to the B operator.
   In the case where B has a (near) nullspace attached, the (near) nullspace vectors are extracted from B and mapped into
   a new (near) nullspace, defined on the sub-communicator, which is attached to B' (the B operator which was scattered to c')

   The telescoping preconditioner is aware of an attached DM. In the event that the DM is of type DMDA (2D or 3D - 
   1D support for 1D DMDAs is not provided), a new DMDA is created on c' (e.g. it is re-partitioned), and this new DM 
   is attached the sub KSPSolve(). The design of telescope is such that it should be possible to extend support 
   for re-partitioning other to DM's (e.g. DMPLEX). The user can supply a flag to ignore attached DMs.

   By default, B' is defined by simply fusing rows from different MPI processes

   When a DMDA is attached to the parent preconditioner, B' is defined by: (i) performing a symmetric permutation of B
   into the ordering defined by the DMDA on c', (ii) extracting the local chunks via MatCreateSubMatrices(), (iii) fusing the
   locally (sequential) matrices defined on the ranks common to c and c' into B' using MatCreateMPIMatConcatenateSeqMat()

   Limitations/improvements include the following.
   VecPlaceArray() could be used within PCApply() to improve efficiency and reduce memory usage.

   The symmetric permutation used when a DMDA is encountered is performed via explicitly assmbleming a permutation matrix P,
   and performing P^T.A.P. Possibly it might be more efficient to use MatPermute(). We opted to use P^T.A.P as it appears
   VecPermute() does not supported for the use case required here. By computing P, one can permute both the operator and RHS in a 
   consistent manner.

   Mapping of vectors is performed in the following way.
   Suppose the parent comm size was 4, and we set a reduction factor of 2; this would give a comm size on c' of 2.
   Using the interlaced creation routine, the ranks in c with color = 0 will be rank 0 and 2.
   We perform the scatter to the sub-comm in the following way.
   [1] Given a vector x defined on comm c

   rank(c) : _________ 0 ______  ________ 1 _______  ________ 2 _____________ ___________ 3 __________
         x : [0, 1, 2, 3, 4, 5] [6, 7, 8, 9, 10, 11] [12, 13, 14, 15, 16, 17] [18, 19, 20, 21, 22, 23]

   scatter to xtmp defined also on comm c so that we have the following values

   rank(c) : ___________________ 0 ________________  _1_  ______________________ 2 _______________________  __3_
      xtmp : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] [  ] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] [  ]

   The entries on rank 1 and 3 (ranks which do not have a color = 0 in c') have no values


   [2] Copy the value from rank 0, 2 (indices with respect to comm c) into the vector xred which is defined on communicator c'.
   Ranks 0 and 2 are the only ranks in the subcomm which have a color = 0.

    rank(c') : ___________________ 0 _______________  ______________________ 1 _____________________
      xred : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


  Contributed by Dave May

  Reference:
  Dave A. May, Patrick Sanan, Karl Rupp, Matthew G. Knepley, and Barry F. Smith, "Extreme-Scale Multigrid Components within PETSc". 2016. In Proceedings of the Platform for Advanced Scientific Computing Conference (PASC '16). DOI: 10.1145/2929908.2929913

.seealso:  PCTelescopeGetKSP(), PCTelescopeGetDM(), PCTelescopeGetReductionFactor(), PCTelescopeSetReductionFactor(), PCTelescopeGetIgnoreDM(), PCTelescopeSetIgnoreDM(), PCREDUNDANT
M*/
PETSC_EXTERN PetscErrorCode PCCreate_Telescope(PC pc)
{
  PetscErrorCode       ierr;
  struct _PC_Telescope *sred;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&sred);CHKERRQ(ierr);
  sred->subcommtype    = PETSC_SUBCOMM_INTERLACED;
  sred->redfactor      = 1;
  sred->ignore_dm      = PETSC_FALSE;
  sred->ignore_kspcomputeoperators = PETSC_FALSE;
  pc->data             = (void*)sred;

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

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeGetKSP_C",PCTelescopeGetKSP_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeGetSubcommType_C",PCTelescopeGetSubcommType_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeSetSubcommType_C",PCTelescopeSetSubcommType_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeGetReductionFactor_C",PCTelescopeGetReductionFactor_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeSetReductionFactor_C",PCTelescopeSetReductionFactor_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeGetIgnoreDM_C",PCTelescopeGetIgnoreDM_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeSetIgnoreDM_C",PCTelescopeSetIgnoreDM_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeGetIgnoreKSPComputeOperators_C",PCTelescopeGetIgnoreKSPComputeOperators_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeSetIgnoreKSPComputeOperators_C",PCTelescopeSetIgnoreKSPComputeOperators_Telescope);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCTelescopeGetDM_C",PCTelescopeGetDM_Telescope);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
