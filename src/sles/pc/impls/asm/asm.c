/*$Id: asm.c,v 1.108 1999/11/24 21:54:41 bsmith Exp bsmith $*/
/*
  This file defines an additive Schwarz preconditioner for any Mat implementation.

  Note that each processor may have any number of subdomains. But in order to 
  deal easily with the VecScatter(), we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number of subdomains on this processor
       n_local = maximum over all processors of n_local_true
*/
#include "src/sles/pc/pcimpl.h"     /*I "pc.h" I*/
#include "sles.h"

typedef struct {
  int        n,n_local,n_local_true;
  int        is_flg;              /* flg set to 1 if the IS created in pcsetup */
  int        overlap;             /* overlap requested by user */
  SLES       *sles;               /* linear solvers for each block */
  VecScatter *scat;               /* mapping to subregion */
  Vec        *x,*y;
  IS         *is;                 /* index set that defines each subdomain */
  Mat        *mat,*pmat;          /* mat is not currently used */
  PCASMType  type;                /* use reduced interpolation, restriction or both */
  int        same_local_solves;   /* flag indicating whether all local solvers are same */
  int        inplace;             /* indicates that the sub-matrices are deleted after 
                                     PCSetUpOnBlocks() is done. Similar to inplace 
                                     factorization in the case of LU and ILU */
} PC_ASM;

#undef __FUNC__  
#define __FUNC__ "PCView_ASM"
static int PCView_ASM(PC pc,Viewer viewer)
{
  PC_ASM     *jac = (PC_ASM*)pc->data;
  int        rank,ierr,i;
  char       *cstring = 0;
  PetscTruth isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"  Additive Schwarz: total subdomain blocks = %d, amount of overlap = %d\n",jac->n,jac->overlap);CHKERRQ(ierr);
    if (jac->type == PC_ASM_NONE)             cstring = "limited restriction and interpolation (PC_ASM_NONE)";
    else if (jac->type == PC_ASM_RESTRICT)    cstring = "full restriction (PC_ASM_RESTRICT)";
    else if (jac->type == PC_ASM_INTERPOLATE) cstring = "full interpolation (PC_ASM_INTERPOLATE)";
    else if (jac->type == PC_ASM_BASIC)       cstring = "full restriction and interpolation (PC_ASM_BASIC)";
    else                                      cstring = "Unknown ASM type";
    ierr = ViewerASCIIPrintf(viewer,"  Additive Schwarz: type - %s\n",cstring);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
    if (jac->same_local_solves) {
      Viewer sviewer;

      ierr = ViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");CHKERRQ(ierr);
      if (!rank && jac->sles) {
        ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = ViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        ierr = SLESView(jac->sles[0],sviewer);CHKERRQ(ierr);
        ierr = ViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
        ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    } else {
      ierr = ViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = ViewerASCIISynchronizedPrintf(viewer,"Proc %d: number of local blocks = %d\n",rank,jac->n_local);CHKERRQ(ierr);
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) {
        Viewer sviewer;

        ierr = ViewerASCIISynchronizedPrintf(viewer,"Proc %d: local block number %d\n",rank,i);CHKERRQ(ierr);
        ierr = ViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        ierr = SLESView(jac->sles[i],sviewer);CHKERRQ(ierr);
        ierr = ViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (i != jac->n_local-1) {
          ierr = ViewerASCIISynchronizedPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
        }
      }
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = ViewerFlush(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer," blks=%d, overlap=%d, type=%d",jac->n,jac->overlap,jac->type);CHKERRQ(ierr);
    if (jac->sles) {ierr = SLESView(jac->sles[0],viewer);CHKERRQ(ierr);}
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for PCASM",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_ASM"
static int PCSetUp_ASM(PC pc)
{
  PC_ASM   *osm  = (PC_ASM*)pc->data;
  int      i,ierr,m,n_local = osm->n_local,n_local_true = osm->n_local_true;
  int      start,start_val,end_val,size,sz,bs;
  MatReuse scall = MAT_REUSE_MATRIX;
  IS       isl;
  SLES     sles;
  KSP      subksp;
  PC       subpc;
  char     *prefix,*pprefix;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    if (osm->n == PETSC_DECIDE && osm->n_local_true == PETSC_DECIDE) { 
      /* no subdomains given, use one per processor */
      osm->n_local_true = osm->n_local = 1;
      ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
      osm->n = size;
    } else if (osm->n == PETSC_DECIDE) { /* determine global number of subdomains */
      int inwork[2],outwork[2];
      inwork[0] = inwork[1] = osm->n_local_true;
      ierr = MPI_Allreduce(inwork,outwork,2,MPI_INT,PetscMaxSum_Op,pc->comm);CHKERRQ(ierr);
      osm->n_local = outwork[0];
      osm->n       = outwork[1];
    }
    n_local      = osm->n_local;
    n_local_true = osm->n_local_true;  
    if (!osm->is){ /* build the index sets */
      osm->is    = (IS*)PetscMalloc((n_local_true+1)*sizeof(IS **));CHKPTRQ(osm->is);
      ierr  = MatGetOwnershipRange(pc->pmat,&start_val,&end_val);CHKERRQ(ierr);
      ierr  = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);
      sz    = end_val - start_val;
      start = start_val;
      if (end_val/bs*bs != end_val || start_val/bs*bs != start_val) {
        SETERRQ(PETSC_ERR_ARG_WRONG,0,"Bad distribution for matrix block size");
      }
      for (i=0; i<n_local_true; i++){
        size       =  ((sz/bs)/n_local_true + (((sz/bs) % n_local_true) > i))*bs;
        ierr       =  ISCreateStride(PETSC_COMM_SELF,size,start,1,&isl);CHKERRQ(ierr);
        start      += size;
        osm->is[i] =  isl;
      }
      osm->is_flg = PETSC_TRUE;
    }

    osm->sles = (SLES*)PetscMalloc((n_local_true+1)*sizeof(SLES **));CHKPTRQ(osm->sles);
    osm->scat = (VecScatter*)PetscMalloc(n_local*sizeof(VecScatter **));CHKPTRQ(osm->scat);
    osm->x    = (Vec*)PetscMalloc(2*n_local*sizeof(Vec **));CHKPTRQ(osm->x);
    osm->y    = osm->x + n_local;

    /*  Extend the "overlapping" regions by a number of steps  */
    ierr = MatIncreaseOverlap(pc->pmat,n_local_true,osm->is,osm->overlap);CHKERRQ(ierr);
    for (i=0; i< n_local_true; i++) {
      ierr = ISSort(osm->is[i]);CHKERRQ(ierr);
    }

    /* create the local work vectors and scatter contexts */
    for (i=0; i<n_local_true; i++) {
      ierr = ISGetSize(osm->is[i],&m);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,m,&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,osm->is[i],osm->x[i],isl,&osm->scat[i]);CHKERRQ(ierr);
      ierr = ISDestroy(isl);CHKERRQ(ierr);
    }
    for (i=n_local_true; i<n_local; i++) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,isl,osm->x[i],isl,&osm->scat[i]);CHKERRQ(ierr);
      ierr = ISDestroy(isl);CHKERRQ(ierr);   
    }

   /* 
       Create the local solvers.
    */
    for (i=0; i<n_local_true; i++) {
      ierr = SLESCreate(PETSC_COMM_SELF,&sles);CHKERRQ(ierr);
      PLogObjectParent(pc,sles);
      ierr = SLESGetKSP(sles,&subksp);CHKERRQ(ierr);
      ierr = KSPSetType(subksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&subpc);CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCILU);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = SLESSetOptionsPrefix(sles,prefix);CHKERRQ(ierr);
      ierr = SLESAppendOptionsPrefix(sles,"sub_");CHKERRQ(ierr);
      ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
      osm->sles[i] = sles;
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

  /* loop over subdomains putting them into local sles */
  ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
  for (i=0; i<n_local_true; i++) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix);CHKERRQ(ierr);
    PLogObjectParent(pc,osm->pmat[i]);
    ierr = SLESSetOperators(osm->sles[i],osm->pmat[i],osm->pmat[i],pc->flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUpOnBlocks_ASM"
static int PCSetUpOnBlocks_ASM(PC pc)
{
  PC_ASM *osm = (PC_ASM*)pc->data;
  int    i,ierr;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local_true; i++) {
    ierr = SLESSetUp(osm->sles[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
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

#undef __FUNC__  
#define __FUNC__ "PCApply_ASM"
static int PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM      *osm = (PC_ASM*)pc->data;
  int         i,n_local = osm->n_local,n_local_true = osm->n_local_true,ierr,its;
  Scalar      zero = 0.0;
  ScatterMode forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
       Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0). 
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    for (i=0; i<n_local; i++) {
      ierr = VecSet(&zero,osm->x[i]);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  for (i=0; i<n_local; i++) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
  }
  ierr = VecSet(&zero,y);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<n_local_true; i++) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = SLESSolve(osm->sles[i],osm->x[i],osm->y[i],&its);CHKERRQ(ierr); 
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

#undef __FUNC__  
#define __FUNC__ "PCApplyTranspose_ASM"
static int PCApplyTranspose_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM      *osm = (PC_ASM*)pc->data;
  int         i,n_local = osm->n_local,n_local_true = osm->n_local_true,ierr,its;
  Scalar      zero = 0.0;
  ScatterMode forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

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
    for (i=0; i<n_local; i++) {
      ierr = VecSet(&zero,osm->x[i]);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_RESTRICT)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  for (i=0; i<n_local; i++) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
  }
  ierr = VecSet(&zero,y);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<n_local_true; i++) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = SLESSolveTranspose(osm->sles[i],osm->x[i],osm->y[i],&its);CHKERRQ(ierr); 
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

#undef __FUNC__  
#define __FUNC__ "PCDestroy_ASM"
static int PCDestroy_ASM(PC pc)
{
  PC_ASM *osm = (PC_ASM*)pc->data;
  int    i,ierr;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local; i++) {
    ierr = VecScatterDestroy(osm->scat[i]);CHKERRQ(ierr);
    ierr = VecDestroy(osm->x[i]);CHKERRQ(ierr);
    ierr = VecDestroy(osm->y[i]);CHKERRQ(ierr);
  }
  if (osm->n_local_true > 0 && !osm->inplace) {
    ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
  }
  for (i=0; i<osm->n_local_true; i++) {
    ierr = SLESDestroy(osm->sles[i]);CHKERRQ(ierr);
  }
  if (osm->is_flg) {
    for (i=0; i<osm->n_local_true; i++) {ierr = ISDestroy(osm->is[i]);CHKERRQ(ierr);}
    ierr = PetscFree(osm->is);CHKERRQ(ierr);
  }
  if (osm->sles) {ierr = PetscFree(osm->sles);CHKERRQ(ierr);}
  if (osm->scat) {ierr = PetscFree(osm->scat);CHKERRQ(ierr);}
  if (osm->x) {ierr = PetscFree(osm->x);CHKERRQ(ierr);}
  ierr = PetscFree(osm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_ASM"
static int PCPrintHelp_ASM(PC pc,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(pc->comm," Options for PCASM preconditioner:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_asm_blocks <blks>: total subdomain blocks\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_asm_overlap <ovl>: amount of overlap between subdomains, defaults to 1\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_asm_inplace: delete the sub-matrices after PCSetUpOnBlocks() is done\n",p);CHKERRQ(ierr); 
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_asm_type <basic,restrict,interpolate,none>: type of restriction/interpolation\n",p);CHKERRQ(ierr); 
  ierr = (*PetscHelpPrintf)(pc->comm," %ssub : prefix to control options for individual blocks.\
  Add before the \n      usual KSP and PC option names (e.g., %ssub_ksp_type\
  <type>)\n",p,p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_ASM"
static int PCSetFromOptions_ASM(PC pc)
{
  int        blocks,ovl,ierr;
  PetscTruth flg;
  char       buff[16];

  PetscFunctionBegin;
  ierr = OptionsGetInt(pc->prefix,"-pc_asm_blocks",&blocks,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetTotalSubdomains(pc,blocks,PETSC_NULL);CHKERRQ(ierr); }
  ierr = OptionsGetInt(pc->prefix,"-pc_asm_overlap",&ovl,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetOverlap(pc,ovl);CHKERRQ(ierr); }
  ierr = OptionsHasName(pc->prefix,"-pc_asm_in_place",&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetUseInPlace(pc);CHKERRQ(ierr); }
  ierr = OptionsGetString(pc->prefix,"-pc_asm_type",buff,15,&flg);CHKERRQ(ierr);
  if (flg) {
    PCASMType  type;
    PetscTruth isbasic,isrestrict,isinterpolate,isnone;

    ierr = PetscStrcmp(buff,"basic",&isbasic);CHKERRQ(ierr);
    ierr = PetscStrcmp(buff,"restrict",&isrestrict);CHKERRQ(ierr);
    ierr = PetscStrcmp(buff,"interpolate",&isinterpolate);CHKERRQ(ierr);
    ierr = PetscStrcmp(buff,"none",&isnone);CHKERRQ(ierr);

    if (isbasic)            type = PC_ASM_BASIC;
    else if (isrestrict)    type = PC_ASM_RESTRICT;
    else if (isinterpolate) type = PC_ASM_INTERPOLATE;
    else if (isnone)        type = PC_ASM_NONE;
    else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown type");
    ierr = PCASMSetType(pc,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCASMSetLocalSubdomains_ASM"
int PCASMSetLocalSubdomains_ASM(PC pc,int n,IS *is)
{
  PC_ASM *osm;

  PetscFunctionBegin;

  if (pc->setupcalled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"PCASMSetLocalSubdomains() should be called before calling PCSetup().");
  }

  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Each process must have 0 or more blocks");
  osm               = (PC_ASM*)pc->data;
  osm->n_local_true = n;
  osm->is           = is;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCASMSetTotalSubdomains_ASM"
int PCASMSetTotalSubdomains_ASM(PC pc,int N,IS *is)
{
  PC_ASM *osm;
  int    rank,size,ierr;

  PetscFunctionBegin;
  if (pc->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,
"PCASMSetTotalSubdomains() should be called before calling PCSetup().");

  if (is) SETERRQ(PETSC_ERR_SUP,0,
"Use PCASMSetLocalSubdomains to set specific index sets\n\
they cannot be set globally yet.");

  osm               = (PC_ASM*)pc->data;
  /*
     Split the subdomains equally amoung all processors 
  */
  ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  osm->n_local_true = N/size + ((N % size) > rank);
  if (osm->n_local_true < 0) {
    SETERRQ(PETSC_ERR_SUP,0,"Each process must have 0 or more blocks");
  }
  osm->is           = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCASMSetOverlap_ASM"
int PCASMSetOverlap_ASM(PC pc,int ovl)
{
  PC_ASM *osm;

  PetscFunctionBegin;
  if (ovl < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative overlap value requested");

  osm               = (PC_ASM*)pc->data;
  osm->overlap      = ovl;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCASMSetType_ASM"
int PCASMSetType_ASM(PC pc,PCASMType type)
{
  PC_ASM *osm;

  PetscFunctionBegin;
  osm        = (PC_ASM*)pc->data;
  osm->type  = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCASMGetSubSLES_ASM"
int PCASMGetSubSLES_ASM(PC pc,int *n_local,int *first_local,SLES **sles)
{
  PC_ASM   *jac = (PC_ASM*)pc->data;;

  PetscFunctionBegin;
  if (jac->n_local_true < 0) {
    SETERRQ(1,1,"Need to call PCSetUP() on PC before calling here");
  }

  if (n_local)     *n_local     = jac->n_local_true;
  if (first_local) *first_local = -1; /* need to determine global number of local blocks*/
  *sles                         = jac->sles;
  jac->same_local_solves        = 0; /* Assume that local solves are now different;
                                      not necessarily true though!  This flag is 
                                      used only for PCView_ASM() */
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCASMSetUseInPlace_ASM"
int PCASMSetUseInPlace_ASM(PC pc)
{
  PC_ASM *dir;

  PetscFunctionBegin;
  dir          = (PC_ASM*)pc->data;
  dir->inplace = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*----------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PCASMSetUseInPlace"
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
int PCASMSetUseInPlace(PC pc)
{
  int ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetUseInPlace_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
/*----------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCASMSetLocalSubdomains"
/*@
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

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetLocalSubdomains(PC pc,int n,IS *is)
{
  int ierr,(*f)(PC,int,IS *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetLocalSubdomains_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,is);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetTotalSubdomains"
/*@
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

.seealso: PCASMSetLocalSubdomains(), PCASMSetOverlap(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetTotalSubdomains(PC pc,int N,IS *is)
{
  int ierr,(*f)(PC,int,IS *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetTotalSubdomains_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,N,is);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetOverlap"
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

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetOverlap(PC pc,int ovl)
{
  int ierr,(*f)(PC,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetOverlap_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ovl);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetType"
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
$   -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

    Level: intermediate

.keywords: PC, ASM, set, type

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetType(PC pc,PCASMType type)
{
  int ierr,(*f)(PC,PCASMType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMSetType_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCASMGetSubSLES"
/*@C
   PCASMGetSubSLES - Gets the local SLES contexts for all blocks on
   this processor.
   
   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor or PETSC_NULL
.  first_local - the global number of the first block on this processor or PETSC_NULL
-  sles - the array of SLES contexts

   Note:  
   After PCASMGetSubSLES() the array of SLESes is not to be freed

   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call SLESSetUp() before calling PCASMGetSubSLES().

   Level: advanced

.keywords: PC, ASM, additive Schwarz, get, sub, SLES, context

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMSetOverlap(),
          PCASMCreateSubdomains2D(),
@*/
int PCASMGetSubSLES(PC pc,int *n_local,int *first_local,SLES **sles)
{
  int ierr,(*f)(PC,int*,int*,SLES **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCASMGetSubSLES_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n_local,first_local,sles);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Cannot get subsles for this type of PC");
  }

 PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_ASM"
int PCCreate_ASM(PC pc)
{
  int    ierr;
  PC_ASM *osm = PetscNew(PC_ASM);CHKPTRQ(osm);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_ASM));
  ierr = PetscMemzero(osm,sizeof(PC_ASM));CHKERRQ(ierr);
  osm->n                 = PETSC_DECIDE;
  osm->n_local           = 0;
  osm->n_local_true      = PETSC_DECIDE;
  osm->overlap           = 1;
  osm->is_flg            = PETSC_FALSE;
  osm->sles              = 0;
  osm->scat              = 0;
  osm->is                = 0;
  osm->mat               = 0;
  osm->pmat              = 0;
  osm->type              = PC_ASM_RESTRICT;
  osm->same_local_solves = 1;
  osm->inplace           = 0;
  pc->data               = (void*)osm;

  pc->ops->apply             = PCApply_ASM;
  pc->ops->applytranspose    = PCApplyTranspose_ASM;
  pc->ops->setup             = PCSetUp_ASM;
  pc->ops->destroy           = PCDestroy_ASM;
  pc->ops->printhelp         = PCPrintHelp_ASM;
  pc->ops->setfromoptions    = PCSetFromOptions_ASM;
  pc->ops->setuponblocks     = PCSetUpOnBlocks_ASM;
  pc->ops->view              = PCView_ASM;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetLocalSubdomains_C","PCASMSetLocalSubdomains_ASM",
                    (void*)PCASMSetLocalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetTotalSubdomains_C","PCASMSetTotalSubdomains_ASM",
                    (void*)PCASMSetTotalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetOverlap_C","PCASMSetOverlap_ASM",
                    (void*)PCASMSetOverlap_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetType_C","PCASMSetType_ASM",
                    (void*)PCASMSetType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMGetSubSLES_C","PCASMGetSubSLES_ASM",
                    (void*)PCASMGetSubSLES_ASM);CHKERRQ(ierr);
ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetUseInPlace_C","PCASMSetUseInPlace_ASM",
                    (void*)PCASMSetUseInPlace_ASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNC__  
#define __FUNC__ "PCASMCreateSubdomains2D"
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

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubSLES(),
          PCASMSetOverlap()
@*/
int PCASMCreateSubdomains2D(int m,int n,int M,int N,int dof,int overlap,int *Nsub,IS **is)
{
  int i,j,height,width,ystart,xstart,yleft,yright,xleft,xright,loc_outter;
  int nidx,*idx,loc,ii,jj,ierr,count;

  PetscFunctionBegin;
  if (dof != 1) SETERRQ(PETSC_ERR_SUP,0,"");

  *Nsub = N*M;
  *is = (IS*)PetscMalloc((*Nsub)*sizeof(IS **));CHKPTRQ(is);
  ystart = 0;
  loc_outter = 0;
  for (i=0; i<N; i++) {
    height = n/N + ((n % N) > i); /* height of subdomain */
    if (height < 2) SETERRA(1,0,"Too many N subdomains for mesh dimension n");
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    xstart = 0;
    for (j=0; j<M; j++) {
      width = m/M + ((m % M) > j); /* width of subdomain */
      if (width < 2) SETERRA(1,0,"Too many M subdomains for mesh dimension m");
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;
      nidx   = (xright - xleft)*(yright - yleft);
      idx    = (int*)PetscMalloc(nidx*sizeof(int));CHKPTRQ(idx);
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

