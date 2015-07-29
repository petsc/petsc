
/*
 Defines a semi-redundant preconditioner.
 
 Assuming that the parent preconditioner (PC) is defined on a communicator c, this implementation
 creates a child sub-communicator (c') containing less MPI processes than the original parent preconditioner (PC).
 
 During PCSetup, the B operator is scattered onto c'.
 Within PCApply, the RHS vector (x) is scattered into a redundant vector, xred (defined on c').
 Then KSPSolve() is executed on the c' communicator.

 The preconditioner is deemed "semi" redundant as it only calls KSPSolve() on a single
 sub-communicator in contrast with PCREDUNDANT which calls KSPSolve() on N sub-communicators.
 This means there will be MPI processes within c, which will be idle during the application of this preconditioner.
 
 Comments:
 - The semi-redundant preconditioner is aware of nullspaces which are attached to the only B operator.
 In case where B has a n nullspace attached, these nullspaces vectors are extract from B and mapped into
 a new nullspace (defined on the sub-communicator) which is attached to B' (the B operator which was scattered to c')

 - The semi-redundant preconditioner is aware of an attached DM. In the event that the DM is of type DMDA (2D or 3D - 
 1D support for 1D DMDAs is not provided), a new DMDA is created on c' (e.g. it is re-partitioned), and this new DM 
 is attached the sub KSPSolve(). The design of semi-redundant is such that it should be possible to extend support 
 for re-partitioning other DM's (e.g. DMPLEX). The user can supply a flag to ignore attached DMs.
 
 - By default, B' is defined by simply fusing rows from different MPI processes

 - When a DMDA is attached to the parent preconditioner, B' is defined by: (i) performing a symmetric permuting of B
 into the ordering defined by the DMDA on c', (ii) extracting the local chunks via MatGetSubMatrices(), (iii) fusing the
 locally (sequential) matrices defined on the ranks common to c and c' into B' using MatCreateMPIMatConcatenateSeqMat()
 
 Limitations/improvements
 - VecPlaceArray could be used within PCApply() to improve efficiency and reduce memory usage.
 
 - The symmetric permutation used when a DMDA is encountered is performed via explicitly assmbleming a permutation matrix P,
 and performing P^T.A.P. Possibly it might be more efficient to use MatPermute(). I opted to use P^T.A.P as it appears
 VecPermute() does not supported for the use case required here. By computing P, I can permute both the operator and RHS in a 
 consistent manner.
 
 - Currently the coordinates of the DMDA on c are not propagated to the sub DM defined on c'
 
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
#include <petscksp.h> /*I "petscksp.h" I*/
#include <petscdm.h> /*I "petscdm.h" I*/

#include "semiredundant.h"

/*
 PCSemiRedundantSetUp_default()
 PCSemiRedundantMatCreate_default()
 
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

#undef __FUNCT__
#define __FUNCT__ "private_PCSemiRedundantGetSubDM"
DM private_PCSemiRedundantGetSubDM(PC_SemiRedundant *sred)
{
  DM subdm;
  
  if (!isActiveRank(sred->psubcomm)) { subdm = NULL; }
  else {
    switch (sred->sr_type) {
      case SR_DEFAULT: subdm = NULL;
        break;
      case SR_DMDA:    subdm = ((PC_SemiRedundant_DMDACtx*)sred->dm_ctx)->dmrepart;
        break;
      case SR_DMPLEX:  subdm = NULL;
        break;
    }
  }
  return(subdm);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantSetUp_default"
PetscErrorCode PCSemiRedundantSetUp_default(PC pc,PC_SemiRedundant *sred)
{
  PetscErrorCode ierr;
  PetscInt       m,M,bs,st,ed;
  Vec            x,xred,yred,xtmp;
  Mat            B;
  MPI_Comm       comm,subcomm;
  VecScatter     scatter;
  IS             isin;
  
  PetscInfo(pc,"PCSemiRedundant: setup (default)\n");
  comm = PetscSubcommParent(sred->psubcomm);
  subcomm = PetscSubcommChild(sred->psubcomm);
  
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatGetSize(B,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(B,&bs);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,NULL);CHKERRQ(ierr);
  
  xred = NULL;
  m = bs;
  if (isActiveRank(sred->psubcomm)) {
    ierr = VecCreate(subcomm,&xred);CHKERRQ(ierr);
    ierr = VecSetSizes(xred,PETSC_DECIDE,M);CHKERRQ(ierr);
    ierr = VecSetBlockSize(xred,bs);CHKERRQ(ierr);
    ierr = VecSetFromOptions(xred);CHKERRQ(ierr);
    ierr = VecGetLocalSize(xred,&m);
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
    ierr = ISCreateStride(comm,bs,st,1,&isin);CHKERRQ(ierr);
  }
  ierr = ISSetBlockSize(isin,bs);CHKERRQ(ierr);
  
  ierr = VecScatterCreate(x,isin,xtmp,NULL,&scatter);CHKERRQ(ierr);
  
  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xred    = xred;
  sred->yred    = yred;
  sred->xtmp    = xtmp;
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantMatCreate_default"
PetscErrorCode PCSemiRedundantMatCreate_default(PC pc,PC_SemiRedundant *sred,MatReuse reuse,Mat *A)
{
  PetscErrorCode ierr;
  MPI_Comm       comm,subcomm;
  Mat            Bred,B;
  PetscInt       nr,nc;
  IS             isrow,iscol;
  Mat            Blocal,*_Blocal;
  
  PetscInfo(pc,"PCSemiRedundant: updating the redundant preconditioned operator (default)\n");
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  subcomm = PetscSubcommChild(sred->psubcomm);
  
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatGetSize(B,&nr,&nc);CHKERRQ(ierr);
  
  if (reuse == MAT_INITIAL_MATRIX) {
    
    Bred = NULL;
    /*
     if (isActiveRank(sred->psubcomm)) {
     ierr = VecGetLocalSize(sred->xred,&fused_length);
     ierr = MatCreate(subcomm,&Bred);CHKERRQ(ierr);
     ierr = MatSetSizes(Bred,fused_length,PETSC_DECIDE,PETSC_DETERMINE,nc);CHKERRQ(ierr);
     ierr = MatSetBlockSize(Bred,bsize);CHKERRQ(ierr);
     ierr = MatSetFromOptions(Bred);CHKERRQ(ierr);
     }
     */
    *A = Bred;
  } else {
    Bred = *A;
  }
  
  isrow = sred->isin;
  ierr = ISCreateStride(comm,nc,0,1,&iscol);CHKERRQ(ierr);
  
  ierr = MatGetSubMatrices(B,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&_Blocal);CHKERRQ(ierr);
  Blocal = *_Blocal;
  
  
  Bred = NULL;
  if (isActiveRank(sred->psubcomm)) {
    PetscInt mm;
    
    ierr = MatGetSize(Blocal,&mm,NULL);CHKERRQ(ierr);
    //ierr = MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,PETSC_DECIDE,reuse,&Bred);CHKERRQ(ierr);
    ierr = MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,mm,reuse,&Bred);CHKERRQ(ierr);
  }
  *A = Bred;
  
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&Blocal);CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantMatNullSpaceCreate_default"
PetscErrorCode PCSemiRedundantMatNullSpaceCreate_default(PC pc,PC_SemiRedundant *sred,Mat sub_mat)
{
  PetscErrorCode   ierr;
  MatNullSpace     nullspace,sub_nullspace;
  Mat              A,B;
  PetscBool        has_const;
  PetscInt         i,k,n;
  const Vec        *vecs;
  Vec              *sub_vecs;
  MPI_Comm         subcomm;
  
  ierr = PCGetOperators(pc,&A,&B);CHKERRQ(ierr);
  ierr = MatGetNullSpace(B,&nullspace);CHKERRQ(ierr);
  if (!nullspace) return(0);
  
  PetscInfo(pc,"PCSemiRedundant: generating nullspace (default)\n");
  subcomm = PetscSubcommChild(sred->psubcomm);
  ierr = MatNullSpaceGetVecs(nullspace,&has_const,&n,&vecs);CHKERRQ(ierr);
  
  if (isActiveRank(sred->psubcomm)) {
    sub_vecs = NULL;
    
    /* create new vectors */
    if (n != 0) {
      PetscMalloc(sizeof(Vec)*n,&sub_vecs);
      for (k=0; k<n; k++) {
        ierr = VecDuplicate(sred->xred,&sub_vecs[k]);CHKERRQ(ierr);
      }
    }
  }
  
  /* copy entries */
  for (k=0; k<n; k++) {
    const PetscScalar *x_array;
    PetscScalar *LA_sub_vec;
    PetscInt st,ed,bs;
    
    /* pull in vector x->xtmp */
    ierr = VecScatterBegin(sred->scatter,vecs[k],sred->xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sred->scatter,vecs[k],sred->xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    
    /* copy vector entires into xred */
    ierr = VecGetBlockSize(sred->xtmp,&bs);CHKERRQ(ierr);
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
  
  if (isActiveRank(sred->psubcomm)) {
    /* create new nullspace for redundant object */
    ierr = MatNullSpaceCreate(subcomm,has_const,n,sub_vecs,&sub_nullspace);CHKERRQ(ierr);
    
    /* attach redundant nullspace to Bred */
    ierr = MatSetNullSpace(sub_mat,sub_nullspace);CHKERRQ(ierr);

    for (k=0; k<n; k++) {
      ierr = VecDestroy(&sub_vecs[k]);CHKERRQ(ierr);
    }
    if (sub_vecs) PetscFree(sub_vecs);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_SemiRedundant"
static PetscErrorCode PCView_SemiRedundant(PC pc,PetscViewer viewer)
{
  PC_SemiRedundant *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode   ierr;
  PetscBool        iascii,isstring;
  PetscViewer      subviewer;
  
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (!sred->psubcomm) {
      ierr = PetscViewerASCIIPrintf(viewer,"  SemiRedundant: preconditioner not yet setup\n");CHKERRQ(ierr);
    } else {
      MPI_Comm    comm,subcomm;
      PetscMPIInt comm_size,subcomm_size;
      DM          dm,subdm;
      
      ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
      subdm = private_PCSemiRedundantGetSubDM(sred);
      comm = PetscSubcommParent(sred->psubcomm);
      subcomm = PetscSubcommChild(sred->psubcomm);
      ierr = MPI_Comm_size(comm,&comm_size);CHKERRQ(ierr);
      ierr = MPI_Comm_size(subcomm,&subcomm_size);CHKERRQ(ierr);
      
      ierr = PetscViewerASCIIPrintf(viewer,"  SemiRedundant: parent comm size reduction factor = %D\n",sred->redfactor);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  SemiRedundant: comm_size = %d , subcomm_size = %d\n",(int)comm_size,(int)subcomm_size);CHKERRQ(ierr);
      ierr = PetscViewerGetSubcomm(viewer,subcomm,&subviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      if (isActiveRank(sred->psubcomm)) {
        
        if (dm && sred->ignore_dm) {
          ierr = PetscViewerASCIIPrintf(subviewer,"  SemiRedundant: ignoring DM\n");CHKERRQ(ierr);
        }
        switch (sred->sr_type) {
          case SR_DEFAULT:
            ierr = PetscViewerASCIIPrintf(subviewer,"  SemiRedundant: using default setup\n");CHKERRQ(ierr);
            break;
          case SR_DMDA:
            ierr = PetscViewerASCIIPrintf(subviewer,"  SemiRedundant: DMDA detected\n");CHKERRQ(ierr);
            ierr = DMView_DMDAShort(subdm,subviewer);CHKERRQ(ierr);
            break;
          case SR_DMPLEX:
            ierr = PetscViewerASCIIPrintf(subviewer,"  SemiRedundant: DMPLEX detected\n");CHKERRQ(ierr);
            break;
        }
        
        ierr = KSPView(sred->ksp,subviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerRestoreSubcomm(viewer,subcomm,&subviewer);CHKERRQ(ierr);
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SemiRedundant"
static PetscErrorCode PCSetUp_SemiRedundant(PC pc)
{
  PC_SemiRedundant  *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode    ierr;
  MPI_Comm          comm,subcomm;
  SemiRedundantType sr_type;
  
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  
  /* subcomm definition */
  if (!pc->setupcalled) {
    if (!sred->psubcomm) {
      ierr = PetscSubcommCreate(comm,&sred->psubcomm);CHKERRQ(ierr);
      ierr = PetscSubcommSetNumber(sred->psubcomm,sred->redfactor);CHKERRQ(ierr);
      ierr = PetscSubcommSetType(sred->psubcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
      /* disable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
      /* ierr = PetscSubcommSetFromOptions(sred->psubcomm);CHKERRQ(ierr); */
      ierr = PetscLogObjectMemory((PetscObject)pc,sizeof(PetscSubcomm));CHKERRQ(ierr);
      
      /* create a new PC that processors in each subcomm have copy of */
      subcomm = PetscSubcommChild(sred->psubcomm);
    }
  } else {
    subcomm = PetscSubcommChild(sred->psubcomm);
  }
  
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
      ierr = KSPAppendOptionsPrefix(sred->ksp,"semiredundant_");CHKERRQ(ierr);
    }
  }
  
  /* Determine type of setup/update */
  if (!pc->setupcalled) {
    PetscBool has_dm,same;
    DM        dm;
    
    sr_type = SR_DEFAULT;
    
    has_dm = PETSC_FALSE;
    ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
    if (dm) { has_dm = PETSC_TRUE; }
    
    if (has_dm) {
      /* check for dmda */
      ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&same);CHKERRQ(ierr);
      if (same) {
        PetscInfo(pc,"PCSemiRedundant: found DMDA\n");
        sr_type = SR_DMDA;
      }
      
      /* check for dmplex */
      ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&same);CHKERRQ(ierr);
      if (same) {
        PetscInfo(pc,"PCSemiRedundant: found DMPLEX\n");
        sr_type = SR_DMPLEX;
      }
    }
    
    if (sred->ignore_dm) {
      PetscInfo(pc,"PCSemiRedundant: ignore DM\n");
      sr_type = SR_DEFAULT;
    }
    sred->sr_type = sr_type;
    
  } else {
    sr_type = sred->sr_type;
  }
  
  /* setup */
  switch (sr_type) {
    case SR_DEFAULT:
      ierr = PCSemiRedundantSetUp_default(pc,sred);CHKERRQ(ierr);
      break;
    case SR_DMDA:
      pc->ops->apply          = PCApply_SemiRedundant_dmda;
      pc->ops->applytranspose = NULL;
      ierr = PCSemiRedundantSetUp_dmda(pc,sred);CHKERRQ(ierr);
      break;
    case SR_DMPLEX:
      break;
  }
  
  /* update */
  switch (sr_type) {
    case SR_DEFAULT:
      if (!pc->setupcalled) {
        ierr = PCSemiRedundantMatCreate_default(pc,sred,MAT_INITIAL_MATRIX,&sred->Bred);CHKERRQ(ierr);
        ierr = PCSemiRedundantMatNullSpaceCreate_default(pc,sred,sred->Bred);CHKERRQ(ierr);
      } else {
        ierr = PCSemiRedundantMatCreate_default(pc,sred,MAT_REUSE_MATRIX,&sred->Bred);CHKERRQ(ierr);
      }
      break;
      
    case SR_DMDA:
      if (!pc->setupcalled) {
        ierr = PCSemiRedundantMatCreate_dmda(pc,sred,MAT_INITIAL_MATRIX,&sred->Bred);CHKERRQ(ierr);
        ierr = PCSemiRedundantMatNullSpaceCreate_dmda(pc,sred,sred->Bred);CHKERRQ(ierr);
      } else {
        ierr = PCSemiRedundantMatCreate_dmda(pc,sred,MAT_REUSE_MATRIX,&sred->Bred);CHKERRQ(ierr);
      }
      break;
      
    case SR_DMPLEX:
      break;
  }
  
  /* common - no construction */
  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPSetOperators(sred->ksp,sred->Bred,sred->Bred);CHKERRQ(ierr);
    if (pc->setfromoptionscalled && !pc->setupcalled){
      ierr = KSPSetFromOptions(sred->ksp);CHKERRQ(ierr);
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_SemiRedundant"
static PetscErrorCode PCApply_SemiRedundant(PC pc,Vec x,Vec y)
{
  PC_SemiRedundant  *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode    ierr;
  Vec               xtmp,xred,yred;
  PetscInt          i,st,ed,bs;
  VecScatter        scatter;
  PetscScalar       *array;
  const PetscScalar *x_array;
  
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  xred    = sred->xred;
  yred    = sred->yred;
  
  /* pull in vector x->xtmp */
  ierr = VecScatterBegin(scatter,x,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,x,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  /* copy vector entires into xred */
  ierr = VecGetBlockSize(xtmp,&bs);CHKERRQ(ierr);
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
  ierr = VecGetBlockSize(xtmp,&bs);CHKERRQ(ierr);
  ierr = VecGetArray(xtmp,&array);CHKERRQ(ierr);
  if (yred) {
    const PetscScalar *LA_yred;
    
    ierr = VecGetOwnershipRange(yred,&st,&ed);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(yred,&LA_yred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      array[i] = LA_yred[i];
    }
    ierr = VecRestoreArrayRead(yred,&LA_yred);CHKERRQ(ierr);
  } else {
    for (i=0; i<bs; i++) {
      array[i] = 0.0;
    }
  }
  ierr = VecRestoreArray(xtmp,&array);CHKERRQ(ierr);
  
  ierr = VecScatterBegin(scatter,xtmp,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,xtmp,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_SemiRedundant"
static PetscErrorCode PCApplyTranspose_SemiRedundant(PC pc,Vec x,Vec y)
{
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_SemiRedundant"
static PetscErrorCode PCReset_SemiRedundant(PC pc)
{
  PC_SemiRedundant *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode   ierr;
  
  ierr = ISDestroy(&sred->isin);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sred->scatter);CHKERRQ(ierr);
  if (sred->xred) { ierr = VecDestroy(&sred->xred);CHKERRQ(ierr); }
  if (sred->yred) { ierr = VecDestroy(&sred->yred);CHKERRQ(ierr); }
  if (sred->xtmp) { ierr = VecDestroy(&sred->xtmp);CHKERRQ(ierr); }
  if (sred->Bred) { ierr = MatDestroy(&sred->Bred);CHKERRQ(ierr); }
  if (sred->ksp) { ierr = KSPReset(sred->ksp);CHKERRQ(ierr); }
  switch (sred->sr_type) {
    case SR_DEFAULT:
      break;
    case SR_DMDA:
      ierr = PCReset_SemiRedundant_dmda(pc);CHKERRQ(ierr);
      break;
    case SR_DMPLEX:
      break;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_SemiRedundant"
static PetscErrorCode PCDestroy_SemiRedundant(PC pc)
{
  PC_SemiRedundant *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode   ierr;
  
  ierr = PCReset_SemiRedundant(pc);CHKERRQ(ierr);
  if (sred->ksp) { ierr = KSPDestroy(&sred->ksp);CHKERRQ(ierr); }
  ierr = PetscSubcommDestroy(&sred->psubcomm);CHKERRQ(ierr);
  if (sred->dm_ctx) PetscFree(sred->dm_ctx);
  PetscFree(pc->data);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_SemiRedundant"
static PetscErrorCode PCSetFromOptions_SemiRedundant(PetscOptions *PetscOptionsObject,PC pc)
{
  PC_SemiRedundant *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode   ierr;
  MPI_Comm         comm;
  PetscMPIInt      size;
  
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"SemiRedundant options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_semiredundant_reduction_factor","Factor to reduce comm size by","PCSemiRedundantSetReductionFactor",sred->redfactor,&sred->redfactor,0);CHKERRQ(ierr);
  if (sred->redfactor > size) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"-pc_semiredundant_reduction_factor <= comm size");
  ierr = PetscOptionsBool("-pc_semiredundant_ignore_dm","Ignore any DM attached to the PC","PCSemiRedundantSetIgnoreDM",sred->ignore_dm,&sred->ignore_dm,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  return(0);
}

/* PC simplementation specific API's */

static PetscErrorCode PCSemiRedundantGetKSP_SemiRedundant(PC pc,KSP *ksp)
{
  PC_SemiRedundant *red = (PC_SemiRedundant*)pc->data;
  if (ksp) *ksp = red->ksp;
  return(0);
}

static PetscErrorCode PCSemiRedundantGetReductionFactor_SemiRedundant(PC pc,PetscInt *fact)
{
  PC_SemiRedundant *red = (PC_SemiRedundant*)pc->data;
  if (fact) *fact = red->redfactor;
  return(0);
}
static PetscErrorCode PCSemiRedundantSetReductionFactor_SemiRedundant(PC pc,PetscInt fact)
{
  PC_SemiRedundant *red = (PC_SemiRedundant*)pc->data;
  PetscMPIInt      size;
  PetscErrorCode   ierr;
  
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  if (fact <= 0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Reduction factor of semi-redundant PC %D must be positive",fact);
  if (fact > size) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Reduction factor of semi-redundant PC %D must be <= comm.size",fact);
  red->redfactor = fact;
  return(0);
}

static PetscErrorCode PCSemiRedundantGetIgnoreDM_SemiRedundant(PC pc,PetscBool *v)
{
  PC_SemiRedundant *red = (PC_SemiRedundant*)pc->data;
  if (v) *v = red->ignore_dm;
  return(0);
}
static PetscErrorCode PCSemiRedundantSetIgnoreDM_SemiRedundant(PC pc,PetscBool v)
{
  PC_SemiRedundant *red = (PC_SemiRedundant*)pc->data;
  red->ignore_dm = v;
  return(0);
}

static PetscErrorCode PCSemiRedundantGetDM_SemiRedundant(PC pc,DM *dm)
{
  PC_SemiRedundant *red = (PC_SemiRedundant*)pc->data;
  *dm = private_PCSemiRedundantGetSubDM(red);
  return(0);
}

/*@
 PCSemiRedundantGetKSP - Gets the KSP created by the semi-redundant PC.
 
 Not Collective
 
 Input Parameter:
 .  pc - the preconditioner context
 
 Output Parameter:
 .  subksp - the KSP defined the smaller set of processes
 
 Level: advanced
 
 .keywords: PC, semi-redundant solve
 @*/
PetscErrorCode PCSemiRedundantGetKSP(PC pc,KSP *subksp)
{
  PetscTryMethod(pc,"PCSemiRedundantGetKSP_C",(PC,KSP*),(pc,subksp));
  return(0);
}

/*@
 PCSemiRedundantGetReductionFactor - Gets the factor by which the original number of processes has been reduced by.
 
 Not Collective
 
 Input Parameter:
 .  pc - the preconditioner context
 
 Output Parameter:
 .  fact - the reduction factor
 
 Level: advanced
 
 .keywords: PC, semi-redundant solve
 @*/
PetscErrorCode PCSemiRedundantGetReductionFactor(PC pc,PetscInt *fact)
{
  PetscTryMethod(pc,"PCSemiRedundantGetReductionFactor_C",(PC,PetscInt*),(pc,fact));
  return(0);
}

/*@
 PCSemiRedundantSetReductionFactor - Sets the factor by which the original number of processes has been reduced by.
 
 Not Collective
 
 Input Parameter:
 .  pc - the preconditioner context
 
 Output Parameter:
 .  fact - the reduction factor
 
 Level: advanced
 
 .keywords: PC, semi-redundant solve
 @*/
PetscErrorCode PCSemiRedundantSetReductionFactor(PC pc,PetscInt fact)
{
  PetscTryMethod(pc,"PCSemiRedundantSetReductionFactor_C",(PC,PetscInt),(pc,fact));
  return(0);
}

/*@
 PCSemiRedundantGetIgnoreDM - Get the flag indicating if any DM attached to the PC will be used.
 
 Not Collective
 
 Input Parameter:
 .  pc - the preconditioner context
 
 Output Parameter:
 .  v - the flag
 
 Level: advanced
 
 .keywords: PC, semi-redundant solve
 @*/
PetscErrorCode PCSemiRedundantGetIgnoreDM(PC pc,PetscBool *v)
{
  PetscTryMethod(pc,"PCSemiRedundantGetIgnoreDM_C",(PC,PetscBool*),(pc,v));
  return(0);
}

/*@
 PCSemiRedundantSetIgnoreDM - Set a flag to ignore any DM attached to the PC.
 
 Not Collective
 
 Input Parameter:
 .  pc - the preconditioner context
 
 Output Parameter:
 .  v - Use PETSC_TRUE to ignore any DM
 
 Level: advanced
 
 .keywords: PC, semi-redundant solve
 @*/
PetscErrorCode PCSemiRedundantSetIgnoreDM(PC pc,PetscInt v)
{
  PetscTryMethod(pc,"PCSemiRedundantSetIgnoreDM_C",(PC,PetscBool),(pc,v));
  return(0);
}

/*@
 PCSemiRedundantGetDM - Get the re-partitioned DM attached to the sub KSP.
 
 Not Collective
 
 Input Parameter:
 .  pc - the preconditioner context
 
 Output Parameter:
 .  subdm - The re-partitioned DM
 
 Level: advanced
 
 .keywords: PC, semi-redundant solve
 @*/
PetscErrorCode PCSemiRedundantGetDM(PC pc,DM *subdm)
{
  PetscTryMethod(pc,"PCSemiRedundantGetDM_C",(PC,DM*),(pc,subdm));
  return(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
 PCSEMIREDUNDANT - Runs a KSP solver on a sub-group of processors. MPI processes not in the sub-communicator are idle during the solve.
 
 Options Database:
 +   -pc_semiredundant_reduction_factor <n> - factor to use communicator size by, for example if you are using 64 MPI processes and
 use an n of 4, the new sub-communicator will be 4 defined with 64/4 processes
 -   -pc_semiredundant_ignore_dm <false> - flag to indicate whether an attached DM should be ignored
 
 Level: advanced
 
 Notes: The default KSP is PREONLY. If a DM is attached to the PC, it is re-partitioned on the sub-communicator. Both the B mat operator
 and the right hand side vector are permuted into the new DOF ordering defined by the re-partitioned DM.
 Currently only support for re-partitioning a DMDA is provided.
 Any nullspace attached to the original PC are extracted, re-partitioned and set on the operator in the sub KSP.
 KSPSetComputeOperators() is not propagated to the sub KSP.
 Optimization: (i) Memory re-use could be used in the scatters between the vectors defined on different sized communicators;
 (ii) Memory re-use and faster set-up would follow if the result of P^T.A.P was not re-allocated each time (DMDA attached).
 
 Contributed by Dave May
 
 .seealso:  PCSemiRedundantGetKSP(),
 PCSemiRedundantGetReductionFactor(), PCSemiRedundantSetReductionFactor(),
 PCSemiRedundantGetIgnoreDM(), PCSemiRedundantSetIgnoreDM(), PCREDUNDANT
 M*/
#undef __FUNCT__
#define __FUNCT__ "PCCreate_SemiRedundant"
PETSC_EXTERN PetscErrorCode PCCreate_SemiRedundant(PC pc)
{
  PetscErrorCode   ierr;
  PC_SemiRedundant *sred;
  
  ierr = PetscNewLog(pc,&sred);CHKERRQ(ierr);
  
  sred->redfactor      = 1;
  sred->ignore_dm      = PETSC_FALSE;
  pc->data             = (void*)sred;
  
  pc->ops->apply          = PCApply_SemiRedundant;
  pc->ops->applytranspose = PCApplyTranspose_SemiRedundant;
  pc->ops->setup          = PCSetUp_SemiRedundant;
  pc->ops->destroy        = PCDestroy_SemiRedundant;
  pc->ops->reset          = PCReset_SemiRedundant;
  pc->ops->setfromoptions = PCSetFromOptions_SemiRedundant;
  pc->ops->view           = PCView_SemiRedundant;
  
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSemiRedundantGetKSP_C",PCSemiRedundantGetKSP_SemiRedundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSemiRedundantGetReductionFactor_C",PCSemiRedundantGetReductionFactor_SemiRedundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSemiRedundantSetReductionFactor_C",PCSemiRedundantSetReductionFactor_SemiRedundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSemiRedundantGetIgnoreDM_C",PCSemiRedundantGetIgnoreDM_SemiRedundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSemiRedundantSetIgnoreDM_C",PCSemiRedundantSetIgnoreDM_SemiRedundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSemiRedundantGetDM_C",PCSemiRedundantGetDM_SemiRedundant);CHKERRQ(ierr);
  return(0);
}
