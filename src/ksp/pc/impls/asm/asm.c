#define PETSCKSP_DLL

/*
  This file defines an additive Schwarz preconditioner for any Mat implementation.

  Note that each processor may have any number of subdomains. But in order to 
  deal easily with the VecScatter(), we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number of subdomains on this processor
       n_local = maximum over all processors of n_local_true
*/
#include "src/ksp/pc/pcimpl.h"     /*I "petscpc.h" I*/

typedef struct {
  PetscInt   n,n_local,n_local_true;
  PetscTruth is_flg;              /* flg set to 1 if the IS created in pcsetup */
  PetscInt   overlap;             /* overlap requested by user */
  KSP        *ksp;               /* linear solvers for each block */
  VecScatter *scat;               /* mapping to subregion */
  Vec        *x,*y;
  IS         *is;                 /* index set that defines each subdomain */
  Mat        *mat,*pmat;          /* mat is not currently used */
  PCASMType  type;                /* use reduced interpolation, restriction or both */
  PetscTruth type_set;            /* if user set this value (so won't change it for symmetric problems) */
  PetscTruth same_local_solves;   /* flag indicating whether all local solvers are same */
  PetscTruth inplace;             /* indicates that the sub-matrices are deleted after 
                                     PCSetUpOnBlocks() is done. Similar to inplace 
                                     factorization in the case of LU and ILU */
} PC_ASM;

#undef __FUNCT__  
#define __FUNCT__ "PCView_ASM"
static PetscErrorCode PCView_ASM(PC pc,PetscViewer viewer)
{
  PC_ASM         *jac = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i;
  PetscTruth     iascii,isstring;
  PetscViewer    sviewer;


  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (jac->n > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: total subdomain blocks = %D, amount of overlap = %D\n",jac->n,jac->overlap);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: total subdomain blocks not yet set, amount of overlap = %D\n",jac->overlap);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: restriction/interpolation type - %s\n",PCASMTypes[jac->type]);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
    if (jac->same_local_solves) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
      if (!rank && jac->ksp) {
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] number of local blocks = %D\n",rank,jac->n_local_true);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      for (i=0; i<jac->n_local_true; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] local block number %D\n",rank,i);CHKERRQ(ierr);
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        ierr = KSPView(jac->ksp[i],sviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (i != jac->n_local_true-1) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blks=%D, overlap=%D, type=%D",jac->n,jac->overlap,jac->type);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
      if (jac->ksp) {ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCASM",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_ASM"
static PetscErrorCode PCSetUp_ASM(PC pc)
{
  PC_ASM         *osm  = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,m,n_local = osm->n_local,n_local_true = osm->n_local_true;
  PetscInt       start,start_val,end_val,sz,bs;
  PetscMPIInt    size;
  MatReuse       scall = MAT_REUSE_MATRIX;
  IS             isl;
  KSP            ksp;
  PC             subpc;
  const char     *prefix,*pprefix;
  Vec            vec;

  PetscFunctionBegin;
  ierr = MatGetVecs(pc->pmat,&vec,0);CHKERRQ(ierr);
  if (!pc->setupcalled) {
    if (osm->n == PETSC_DECIDE && osm->n_local_true == PETSC_DECIDE) { 
      /* no subdomains given, use one per processor */
      osm->n_local_true = osm->n_local = 1;
      ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
      osm->n = size;
    } else if (osm->n == PETSC_DECIDE) { /* determine global number of subdomains */
      PetscInt inwork[2],outwork[2];
      inwork[0] = inwork[1] = osm->n_local_true;
      ierr = MPI_Allreduce(inwork,outwork,1,MPIU_2INT,PetscMaxSum_Op,pc->comm);CHKERRQ(ierr);
      osm->n_local = outwork[0];
      osm->n       = outwork[1];
    }
    n_local      = osm->n_local;
    n_local_true = osm->n_local_true;  
    if (!osm->is){ /* build the index sets */
      ierr  = PetscMalloc((n_local_true+1)*sizeof(IS **),&osm->is);CHKERRQ(ierr);
      ierr  = MatGetOwnershipRange(pc->pmat,&start_val,&end_val);CHKERRQ(ierr);
      ierr  = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);
      sz    = end_val - start_val;
      start = start_val;
      if (end_val/bs*bs != end_val || start_val/bs*bs != start_val) {
        SETERRQ(PETSC_ERR_ARG_WRONG,"Bad distribution for matrix block size");
      }
      for (i=0; i<n_local_true; i++){
        size       =  ((sz/bs)/n_local_true + (((sz/bs) % n_local_true) > i))*bs;
        ierr       =  ISCreateStride(PETSC_COMM_SELF,size,start,1,&isl);CHKERRQ(ierr);
        start      += size;
        osm->is[i] =  isl;
      }
      osm->is_flg = PETSC_TRUE;
    }

    ierr   = PetscMalloc((n_local_true+1)*sizeof(KSP **),&osm->ksp);CHKERRQ(ierr);
    ierr   = PetscMalloc(n_local*sizeof(VecScatter **),&osm->scat);CHKERRQ(ierr);
    ierr   = PetscMalloc(2*n_local*sizeof(Vec **),&osm->x);CHKERRQ(ierr);
    osm->y = osm->x + n_local;

    /*  Extend the "overlapping" regions by a number of steps  */
    ierr = MatIncreaseOverlap(pc->pmat,n_local_true,osm->is,osm->overlap);CHKERRQ(ierr);
    for (i=0; i<n_local_true; i++) {
      ierr = ISSort(osm->is[i]);CHKERRQ(ierr);
    }

    /* create the local work vectors and scatter contexts */
    for (i=0; i<n_local_true; i++) {
      ierr = ISGetLocalSize(osm->is[i],&m);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,m,&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec,osm->is[i],osm->x[i],isl,&osm->scat[i]);CHKERRQ(ierr);
      ierr = ISDestroy(isl);CHKERRQ(ierr);
    }
    for (i=n_local_true; i<n_local; i++) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec,isl,osm->x[i],isl,&osm->scat[i]);CHKERRQ(ierr);
      ierr = ISDestroy(isl);CHKERRQ(ierr);   
    }

   /* 
       Create the local solvers.
    */
    for (i=0; i<n_local_true; i++) {
      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
      osm->ksp[i] = ksp;
    }
    scall = MAT_INITIAL_MATRIX;
  } else {
    /* 
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
      scall = MAT_INITIAL_MATRIX;
    }
  }

  /* extract out the submatrices */
  ierr = MatGetSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat);CHKERRQ(ierr);

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,osm->n_local,osm->is,osm->is,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /* loop over subdomains putting them into local ksp */
  ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
  for (i=0; i<n_local_true; i++) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,osm->pmat[i]);CHKERRQ(ierr);
    ierr = KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i],pc->flag);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(osm->ksp[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroy(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUpOnBlocks_ASM"
static PetscErrorCode PCSetUpOnBlocks_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local_true; i++) {
    ierr = KSPSetUp(osm->ksp[i]);CHKERRQ(ierr);
  }
  /* 
     If inplace flag is set, then destroy the matrix after the setup
     on blocks is done.
  */   
  if (osm->inplace && osm->n_local_true > 0) {
    ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ASM"
static PetscErrorCode PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n_local = osm->n_local,n_local_true = osm->n_local_true;
  PetscScalar    zero = 0.0;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
       Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0). 
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    for (i=0; i<n_local_true; i++) {
      ierr = VecSet(osm->x[i],zero);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  for (i=0; i<n_local; i++) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
  }
  ierr = VecSet(y,zero);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<n_local_true; i++) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = KSPSolve(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr); 
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  /* handle the rest of the scatters that do not have local solves */
  for (i=n_local_true; i<n_local; i++) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  for (i=0; i<n_local; i++) {
    ierr = VecScatterEnd(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_ASM"
static PetscErrorCode PCApplyTranspose_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n_local = osm->n_local,n_local_true = osm->n_local_true;
  PetscScalar    zero = 0.0;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
       Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0).

       Note: these are reversed from the PCApply_ASM() because we are applying the 
     transpose of the three terms 
  */
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    for (i=0; i<n_local_true; i++) {
      ierr = VecSet(osm->x[i],zero);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_RESTRICT)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  for (i=0; i<n_local; i++) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
  }
  ierr = VecSet(y,zero);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<n_local_true; i++) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr); 
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  /* handle the rest of the scatters that do not have local solves */
  for (i=n_local_true; i<n_local; i++) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  for (i=0; i<n_local; i++) {
    ierr = VecScatterEnd(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ASM"
static PetscErrorCode PCDestroy_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local; i++) {
    ierr = VecScatterDestroy(osm->scat[i]);CHKERRQ(ierr);
    ierr = VecDestroy(osm->x[i]);CHKERRQ(ierr);
    ierr = VecDestroy(osm->y[i]);CHKERRQ(ierr);
  }
  if (osm->n_local_true > 0 && !osm->inplace && osm->pmat) {
    ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
  }
  if (osm->ksp) {
    for (i=0; i<osm->n_local_true; i++) {
      ierr = KSPDestroy(osm->ksp[i]);CHKERRQ(ierr);
    }
  }
  if (osm->is_flg) {
    for (i=0; i<osm->n_local_true; i++) {ierr = ISDestroy(osm->is[i]);CHKERRQ(ierr);}
    ierr = PetscFree(osm->is);CHKERRQ(ierr);
  }
  if (osm->ksp) {ierr = PetscFree(osm->ksp);CHKERRQ(ierr);}
  if (osm->scat) {ierr = PetscFree(osm->scat);CHKERRQ(ierr);}
  if (osm->x) {ierr = PetscFree(osm->x);CHKERRQ(ierr);}
  ierr = PetscFree(osm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ASM"
static PetscErrorCode PCSetFromOptions_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks,ovl;
  PetscTruth     flg,set,sym;

  PetscFunctionBegin;
  /* set the type to symmetric if matrix is symmetric */
  if (pc->pmat && !osm->type_set) {
    ierr = MatIsSymmetricKnown(pc->pmat,&set,&sym);CHKERRQ(ierr);
    if (set && sym) {
      osm->type = PC_ASM_BASIC;
    }
  }
  ierr = PetscOptionsHead("Additive Schwarz options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_asm_blocks","Number of subdomains","PCASMSetTotalSubdomains",osm->n,&blocks,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCASMSetTotalSubdomains(pc,blocks,PETSC_NULL);CHKERRQ(ierr); }
    ierr = PetscOptionsInt("-pc_asm_overlap","Number of grid points overlap","PCASMSetOverlap",osm->overlap,&ovl,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCASMSetOverlap(pc,ovl);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-pc_asm_in_place","Perform matrix factorization inplace","PCASMSetUseInPlace",&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCASMSetUseInPlace(pc);CHKERRQ(ierr); }
    ierr = PetscOptionsEnum("-pc_asm_type","Type of restriction/extension","PCASMSetType",PCASMTypes,(PetscEnum)osm->type,(PetscEnum*)&osm->type,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetLocalSubdomains_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetLocalSubdomains_ASM(PC pc,PetscInt n,IS is[])
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 0 or more blocks");

  if (pc->setupcalled && n != osm->n_local_true) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"PCASMSetLocalSubdomains() should be called before calling PCSetup().");
  }
  if (!pc->setupcalled){
    osm->n_local_true = n;
    osm->is           = is;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetTotalSubdomains_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetTotalSubdomains_ASM(PC pc,PetscInt N,IS *is)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n;

  PetscFunctionBegin;

  if (is) SETERRQ(PETSC_ERR_SUP,"Use PCASMSetLocalSubdomains() to set specific index sets\n\
they cannot be set globally yet.");

  /*
     Split the subdomains equally amoung all processors 
  */
  ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  n = N/size + ((N % size) > rank);
  if (pc->setupcalled && n != osm->n_local_true) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"PCASMSetTotalSubdomains() should be called before PCSetup().");
  if (!pc->setupcalled){
    osm->n_local_true = n;
    osm->is           = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetOverlap_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetOverlap_ASM(PC pc,PetscInt ovl)
{
  PC_ASM *osm;

  PetscFunctionBegin;
  if (ovl < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");

  osm               = (PC_ASM*)pc->data;
  osm->overlap      = ovl;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetType_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetType_ASM(PC pc,PCASMType type)
{
  PC_ASM *osm;

  PetscFunctionBegin;
  osm           = (PC_ASM*)pc->data;
  osm->type     = type;
  osm->type_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMGetSubKSP_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetSubKSP_ASM(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_ASM         *jac = (PC_ASM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (jac->n_local_true < 0) {
    SETERRQ(PETSC_ERR_ORDER,"Need to call PCSetUP() on PC (or KSPSetUp() on the outer KSP object) before calling here");
  }

  if (n_local)     *n_local     = jac->n_local_true;
  if (first_local) {
    ierr = MPI_Scan(&jac->n_local_true,first_local,1,MPIU_INT,MPI_SUM,pc->comm);CHKERRQ(ierr);
    *first_local -= jac->n_local_true;
  }
  *ksp                         = jac->ksp;
  jac->same_local_solves        = PETSC_FALSE; /* Assume that local solves are now different;
                                      not necessarily true though!  This flag is 
                                      used only for PCView_ASM() */
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetUseInPlace_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetUseInPlace_ASM(PC pc)
{
  PC_ASM *dir;

  PetscFunctionBegin;
  dir          = (PC_ASM*)pc->data;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*----------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetUseInPlace"
/*@
   PCASMSetUseInPlace - Tells the system to destroy the matrix after setup is done.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_asm_in_place - Activates in-place factorization

   Note:
   PCASMSetUseInplace() can only be used with the KSP method KSPPREONLY, and
   when the original matrix is not required during the Solve process.
   This destroys the matrix, early thus, saving on memory usage.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, ASM

.seealso: PCILUSetUseInPlace(), PCLUSetUseInPlace ()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetUseInPlace(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetUseInPlace_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
/*----------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetLocalSubdomains"
/*@C
    PCASMSetLocalSubdomains - Sets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner. 

    Collective on PC 

    Input Parameters:
+   pc - the preconditioner context
.   n - the number of subdomains for this processor (default value = 1)
-   is - the index sets that define the subdomains for this processor
         (or PETSC_NULL for PETSc to determine subdomains)

    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    By default the ASM preconditioner uses 1 block per processor.  

    These index sets cannot be destroyed until after completion of the
    linear solves for which the ASM preconditioner is being used.

    Use PCASMSetTotalSubdomains() to set the subdomains for all processors.

    Level: advanced

.keywords: PC, ASM, set, local, subdomains, additive Schwarz

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetLocalSubdomains(PC pc,PetscInt n,IS is[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,IS[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetLocalSubdomains_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,is);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetTotalSubdomains"
/*@C
    PCASMSetTotalSubdomains - Sets the subdomains for all processor for the 
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine, with the same index sets.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
.   n - the number of subdomains for all processors
-   is - the index sets that define the subdomains for all processor
         (or PETSC_NULL for PETSc to determine subdomains)

    Options Database Key:
    To set the total number of subdomain blocks rather than specify the
    index sets, use the option
.    -pc_asm_blocks <blks> - Sets total blocks

    Notes:
    Currently you cannot use this to set the actual subdomains with the argument is.

    By default the ASM preconditioner uses 1 block per processor.  

    These index sets cannot be destroyed until after completion of the
    linear solves for which the ASM preconditioner is being used.

    Use PCASMSetLocalSubdomains() to set local subdomains.

    Level: advanced

.keywords: PC, ASM, set, total, global, subdomains, additive Schwarz

.seealso: PCASMSetLocalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetTotalSubdomains(PC pc,PetscInt N,IS *is)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,IS *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetTotalSubdomains_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,N,is);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetOverlap"
/*@
    PCASMSetOverlap - Sets the overlap between a pair of subdomains for the
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine. 

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   ovl - the amount of overlap between subdomains (ovl >= 0, default value = 1)

    Options Database Key:
.   -pc_asm_overlap <ovl> - Sets overlap

    Notes:
    By default the ASM preconditioner uses 1 block per processor.  To use
    multiple blocks per perocessor, see PCASMSetTotalSubdomains() and
    PCASMSetLocalSubdomains() (and the option -pc_asm_blocks <blks>).

    The overlap defaults to 1, so if one desires that no additional
    overlap be computed beyond what may have been set with a call to
    PCASMSetTotalSubdomains() or PCASMSetLocalSubdomains(), then ovl
    must be set to be 0.  In particular, if one does not explicitly set
    the subdomains an application code, then all overlap would be computed
    internally by PETSc, and using an overlap of 0 would result in an ASM 
    variant that is equivalent to the block Jacobi preconditioner.  

    Note that one can define initial index sets with any overlap via
    PCASMSetTotalSubdomains() or PCASMSetLocalSubdomains(); the routine
    PCASMSetOverlap() merely allows PETSc to extend that overlap further
    if desired.

    Level: intermediate

.keywords: PC, ASM, set, overlap

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetOverlap(PC pc,PetscInt ovl)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetOverlap_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ovl);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetType"
/*@
    PCASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   type - variant of ASM, one of
.vb
      PC_ASM_BASIC       - full interpolation and restriction
      PC_ASM_RESTRICT    - full restriction, local processor interpolation
      PC_ASM_INTERPOLATE - full interpolation, local processor restriction
      PC_ASM_NONE        - local processor restriction and interpolation
.ve

    Options Database Key:
.   -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

    Level: intermediate

.keywords: PC, ASM, set, type

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetType(PC pc,PCASMType type)
{
  PetscErrorCode ierr,(*f)(PC,PCASMType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMGetSubKSP"
/*@C
   PCASMGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.
   
   Collective on PC iff first_local is requested

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor or PETSC_NULL
.  first_local - the global number of the first block on this processor or PETSC_NULL,
                 all processors must request or all must pass PETSC_NULL
-  ksp - the array of KSP contexts

   Note:  
   After PCASMGetSubKSP() the array of KSPes is not to be freed

   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call KSPSetUp() before calling PCASMGetSubKSP().

   Level: advanced

.keywords: PC, ASM, additive Schwarz, get, sub, KSP, context

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMSetOverlap(),
          PCASMCreateSubdomains2D(),
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*,PetscInt*,KSP **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMGetSubKSP_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n_local,first_local,ksp);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get subksp for this type of PC");
  }

 PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCASM - Use the (restricted) additive Schwarz method, each block is (approximately) solved with 
           its own KSP object.

   Options Database Keys:
+  -pc_asm_truelocal - Activates PCASMSetUseTrueLocal()
.  -pc_asm_in_place - Activates in-place factorization
.  -pc_asm_blocks <blks> - Sets total blocks
.  -pc_asm_overlap <ovl> - Sets overlap
-  -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

     IMPORTANT: If you run with, for example, 3 blocks on 1 processor or 3 blocks on 3 processors you 
      will get a different convergence rate due to the default option of -pc_asm_type restrict. Use
      -pc_asm_type basic to use the standard ASM. 

   Notes: Each processor can have one or more blocks, but a block cannot be shared by more
     than one processor. Defaults to one block per processor.

     To set options on the solvers for each block append -sub_ to all the KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_ilu_levels 1 -sub_ksp_type preonly
        
     To set the options on the solvers seperate for each block call PCASMGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         with KSPGetPC())


   Level: beginner

   Concepts: additive Schwarz method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCBJACOBI, PCASMSetUseTrueLocal(), PCASMGetSubKSP(), PCASMSetLocalSubdomains(),
           PCASMSetTotalSubdomains(), PCSetModifySubmatrices(), PCASMSetOverlap(), PCASMSetType(),
           PCASMSetUseInPlace()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ASM"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_ASM(PC pc)
{
  PetscErrorCode ierr;
  PC_ASM         *osm;

  PetscFunctionBegin;
  ierr = PetscNew(PC_ASM,&osm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_ASM));CHKERRQ(ierr);
  osm->n                 = PETSC_DECIDE;
  osm->n_local           = 0;
  osm->n_local_true      = PETSC_DECIDE;
  osm->overlap           = 1;
  osm->is_flg            = PETSC_FALSE;
  osm->ksp              = 0;
  osm->scat              = 0;
  osm->is                = 0;
  osm->mat               = 0;
  osm->pmat              = 0;
  osm->type              = PC_ASM_RESTRICT;
  osm->same_local_solves = PETSC_TRUE;
  osm->inplace           = PETSC_FALSE;
  pc->data               = (void*)osm;

  pc->ops->apply             = PCApply_ASM;
  pc->ops->applytranspose    = PCApplyTranspose_ASM;
  pc->ops->setup             = PCSetUp_ASM;
  pc->ops->destroy           = PCDestroy_ASM;
  pc->ops->setfromoptions    = PCSetFromOptions_ASM;
  pc->ops->setuponblocks     = PCSetUpOnBlocks_ASM;
  pc->ops->view              = PCView_ASM;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetLocalSubdomains_C","PCASMSetLocalSubdomains_ASM",
                    PCASMSetLocalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetTotalSubdomains_C","PCASMSetTotalSubdomains_ASM",
                    PCASMSetTotalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetOverlap_C","PCASMSetOverlap_ASM",
                    PCASMSetOverlap_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetType_C","PCASMSetType_ASM",
                    PCASMSetType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMGetSubKSP_C","PCASMGetSubKSP_ASM",
                    PCASMGetSubKSP_ASM);CHKERRQ(ierr);
ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetUseInPlace_C","PCASMSetUseInPlace_ASM",
                    PCASMSetUseInPlace_ASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PCASMCreateSubdomains2D"
/*@
   PCASMCreateSubdomains2D - Creates the index sets for the overlapping Schwarz 
   preconditioner for a two-dimensional problem on a regular grid.

   Not Collective

   Input Parameters:
+  m, n - the number of mesh points in the x and y directions
.  M, N - the number of subdomains in the x and y directions
.  dof - degrees of freedom per node
-  overlap - overlap in mesh lines

   Output Parameters:
+  Nsub - the number of subdomains created
-  is - the array of index sets defining the subdomains

   Note:
   Presently PCAMSCreateSubdomains2d() is valid only for sequential
   preconditioners.  More general related routines are
   PCASMSetTotalSubdomains() and PCASMSetLocalSubdomains().

   Level: advanced

.keywords: PC, ASM, additive Schwarz, create, subdomains, 2D, regular grid

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubKSP(),
          PCASMSetOverlap()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMCreateSubdomains2D(PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt dof,PetscInt overlap,PetscInt *Nsub,IS **is)
{
  PetscInt       i,j,height,width,ystart,xstart,yleft,yright,xleft,xright,loc_outter;
  PetscErrorCode ierr;
  PetscInt       nidx,*idx,loc,ii,jj,count;

  PetscFunctionBegin;
  if (dof != 1) SETERRQ(PETSC_ERR_SUP," ");

  *Nsub = N*M;
  ierr = PetscMalloc((*Nsub)*sizeof(IS **),is);CHKERRQ(ierr);
  ystart = 0;
  loc_outter = 0;
  for (i=0; i<N; i++) {
    height = n/N + ((n % N) > i); /* height of subdomain */
    if (height < 2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many N subdomains for mesh dimension n");
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    xstart = 0;
    for (j=0; j<M; j++) {
      width = m/M + ((m % M) > j); /* width of subdomain */
      if (width < 2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many M subdomains for mesh dimension m");
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;
      nidx   = (xright - xleft)*(yright - yleft);
      ierr = PetscMalloc(nidx*sizeof(PetscInt),&idx);CHKERRQ(ierr);
      loc    = 0;
      for (ii=yleft; ii<yright; ii++) {
        count = m*ii + xleft;
        for (jj=xleft; jj<xright; jj++) {
          idx[loc++] = count++;
        }
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx,(*is)+loc_outter++);CHKERRQ(ierr);
      ierr = PetscFree(idx);CHKERRQ(ierr);
      xstart += width;
    }
    ystart += height;
  }
  for (i=0; i<*Nsub; i++) { ierr = ISSort((*is)[i]);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMGetLocalSubdomains"
/*@C
    PCASMGetLocalSubdomains - Gets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner. 

    Collective on PC 

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of subdomains for this processor (default value = 1)
-   is - the index sets that define the subdomains for this processor
         

    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.keywords: PC, ASM, set, local, subdomains, additive Schwarz

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubmatrices()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetLocalSubdomains(PC pc,PetscInt *n,IS *is[])
{
  PC_ASM *osm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(n,2);
  if (!pc->setupcalled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUP() or PCSetUp().");
  }

  osm = (PC_ASM*)pc->data;
  if (n)  *n = osm->n_local_true;
  if (is) *is = osm->is;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMGetLocalSubmatrices"
/*@C
    PCASMGetLocalSubmatrices - Gets the local submatrices (for this processor
    only) for the additive Schwarz preconditioner. 

    Collective on PC 

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of matrices for this processor (default value = 1)
-   mat - the matrices
         

    Level: advanced

.keywords: PC, ASM, set, local, subdomains, additive Schwarz, block Jacobi

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetLocalSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_ASM *osm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(n,2);
  if (!pc->setupcalled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUP() or PCSetUp().");
  }

  osm = (PC_ASM*)pc->data;
  if (n)   *n   = osm->n_local_true;
  if (mat) *mat = osm->pmat;
  PetscFunctionReturn(0);
}

