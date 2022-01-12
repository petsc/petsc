#include <petsc/private/snesimpl.h>             /*I   "petscsnes.h"   I*/
#include <petscdm.h>

typedef struct {
  PetscInt   n;                   /* local subdomains */
  SNES       *subsnes;            /* nonlinear solvers for each subdomain */
  Vec        *x;                  /* solution vectors */
  Vec        *xl;                 /* solution local vectors */
  Vec        *y;                  /* step vectors */
  Vec        *b;                  /* rhs vectors */
  Vec        weight;              /* weighting for adding updates on overlaps, in global space */
  VecScatter *oscatter;           /* scatter from global space to the subdomain global space */
  VecScatter *oscatter_copy;      /* copy of the above */
  VecScatter *iscatter;           /* scatter from global space to the nonoverlapping subdomain space */
  VecScatter *gscatter;           /* scatter from global space to the subdomain local space */
  PCASMType  type;                /* ASM type */
  PetscBool  usesdm;              /* use the DM for setting up the subproblems */
  PetscBool  finaljacobian;       /* compute the jacobian of the converged solution */
  PetscReal  damping;             /* damping parameter for updates from the blocks */
  PetscBool  weight_set;          /* use a weight in the overlap updates */

  /* logging events */
  PetscLogEvent eventrestrictinterp;
  PetscLogEvent eventsubsolve;

  PetscInt      fjtype;            /* type of computed jacobian */
  Vec           xinit;             /* initial solution in case the final jacobian type is computed as first */
} SNES_NASM;

const char *const SNESNASMTypes[] = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCASMType","PC_ASM_",NULL};
const char *const SNESNASMFJTypes[] = {"FINALOUTER","FINALINNER","INITIAL"};

static PetscErrorCode SNESReset_NASM(SNES snes)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nasm->n; i++) {
    if (nasm->xl) { ierr = VecDestroy(&nasm->xl[i]);CHKERRQ(ierr); }
    if (nasm->x) { ierr = VecDestroy(&nasm->x[i]);CHKERRQ(ierr); }
    if (nasm->y) { ierr = VecDestroy(&nasm->y[i]);CHKERRQ(ierr); }
    if (nasm->b) { ierr = VecDestroy(&nasm->b[i]);CHKERRQ(ierr); }

    if (nasm->subsnes) { ierr = SNESDestroy(&nasm->subsnes[i]);CHKERRQ(ierr); }
    if (nasm->oscatter) { ierr = VecScatterDestroy(&nasm->oscatter[i]);CHKERRQ(ierr); }
    if (nasm->oscatter_copy) { ierr = VecScatterDestroy(&nasm->oscatter_copy[i]);CHKERRQ(ierr); }
    if (nasm->iscatter) { ierr = VecScatterDestroy(&nasm->iscatter[i]);CHKERRQ(ierr); }
    if (nasm->gscatter) { ierr = VecScatterDestroy(&nasm->gscatter[i]);CHKERRQ(ierr); }
  }

  ierr = PetscFree(nasm->x);CHKERRQ(ierr);
  ierr = PetscFree(nasm->xl);CHKERRQ(ierr);
  ierr = PetscFree(nasm->y);CHKERRQ(ierr);
  ierr = PetscFree(nasm->b);CHKERRQ(ierr);

  if (nasm->xinit) {ierr = VecDestroy(&nasm->xinit);CHKERRQ(ierr);}

  ierr = PetscFree(nasm->subsnes);CHKERRQ(ierr);
  ierr = PetscFree(nasm->oscatter);CHKERRQ(ierr);
  ierr = PetscFree(nasm->oscatter_copy);CHKERRQ(ierr);
  ierr = PetscFree(nasm->iscatter);CHKERRQ(ierr);
  ierr = PetscFree(nasm->gscatter);CHKERRQ(ierr);

  if (nasm->weight_set) {
    ierr = VecDestroy(&nasm->weight);CHKERRQ(ierr);
  }

  nasm->eventrestrictinterp = 0;
  nasm->eventsubsolve = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_NASM(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NASM(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalSubDomainDirichletHook_Private(DM dm,Vec g,InsertMode mode,Vec l,void *ctx)
{
  PetscErrorCode ierr;
  Vec            bcs = (Vec)ctx;

  PetscFunctionBegin;
  ierr = VecCopy(bcs,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_NASM(SNES snes)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
  PetscErrorCode ierr;
  DM             dm,subdm;
  DM             *subdms;
  PetscInt       i;
  const char     *optionsprefix;
  Vec            F;
  PetscMPIInt    size;
  KSP            ksp;
  PC             pc;

  PetscFunctionBegin;
  if (!nasm->subsnes) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    if (dm) {
      nasm->usesdm = PETSC_TRUE;
      ierr         = DMCreateDomainDecomposition(dm,&nasm->n,NULL,NULL,NULL,&subdms);CHKERRQ(ierr);
      if (!subdms) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM has no default decomposition defined.  Set subsolves manually with SNESNASMSetSubdomains().");
      ierr = DMCreateDomainDecompositionScatters(dm,nasm->n,subdms,&nasm->iscatter,&nasm->oscatter,&nasm->gscatter);CHKERRQ(ierr);
      ierr = PetscMalloc1(nasm->n, &nasm->oscatter_copy);CHKERRQ(ierr);
      for (i=0; i<nasm->n; i++) {
        ierr = VecScatterCopy(nasm->oscatter[i], &nasm->oscatter_copy[i]);CHKERRQ(ierr);
      }

      ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
      ierr = PetscMalloc1(nasm->n,&nasm->subsnes);CHKERRQ(ierr);
      for (i=0; i<nasm->n; i++) {
        ierr = SNESCreate(PETSC_COMM_SELF,&nasm->subsnes[i]);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)nasm->subsnes[i], (PetscObject)snes, 1);CHKERRQ(ierr);
        ierr = SNESAppendOptionsPrefix(nasm->subsnes[i],optionsprefix);CHKERRQ(ierr);
        ierr = SNESAppendOptionsPrefix(nasm->subsnes[i],"sub_");CHKERRQ(ierr);
        ierr = SNESSetDM(nasm->subsnes[i],subdms[i]);CHKERRQ(ierr);
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)nasm->subsnes[i]),&size);CHKERRMPI(ierr);
        if (size == 1) {
          ierr = SNESGetKSP(nasm->subsnes[i],&ksp);CHKERRQ(ierr);
          ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
          ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
          ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
        }
        ierr = SNESSetFromOptions(nasm->subsnes[i]);CHKERRQ(ierr);
        ierr = DMDestroy(&subdms[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(subdms);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Cannot construct local problems automatically without a DM!");
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Must set subproblems manually if there is no DM!");
  /* allocate the global vectors */
  if (!nasm->x) {
    ierr = PetscCalloc1(nasm->n,&nasm->x);CHKERRQ(ierr);
  }
  if (!nasm->xl) {
    ierr = PetscCalloc1(nasm->n,&nasm->xl);CHKERRQ(ierr);
  }
  if (!nasm->y) {
    ierr = PetscCalloc1(nasm->n,&nasm->y);CHKERRQ(ierr);
  }
  if (!nasm->b) {
    ierr = PetscCalloc1(nasm->n,&nasm->b);CHKERRQ(ierr);
  }

  for (i=0; i<nasm->n; i++) {
    ierr = SNESGetFunction(nasm->subsnes[i],&F,NULL,NULL);CHKERRQ(ierr);
    if (!nasm->x[i]) {ierr = VecDuplicate(F,&nasm->x[i]);CHKERRQ(ierr);}
    if (!nasm->y[i]) {ierr = VecDuplicate(F,&nasm->y[i]);CHKERRQ(ierr);}
    if (!nasm->b[i]) {ierr = VecDuplicate(F,&nasm->b[i]);CHKERRQ(ierr);}
    if (!nasm->xl[i]) {
      ierr = SNESGetDM(nasm->subsnes[i],&subdm);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(subdm,&nasm->xl[i]);CHKERRQ(ierr);
      ierr = DMGlobalToLocalHookAdd(subdm,DMGlobalToLocalSubDomainDirichletHook_Private,NULL,nasm->xl[i]);CHKERRQ(ierr);
    }
  }
  if (nasm->finaljacobian) {
    ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
    if (nasm->fjtype == 2) {
      ierr = VecDuplicate(snes->vec_sol,&nasm->xinit);CHKERRQ(ierr);
    }
    for (i=0; i<nasm->n;i++) {
      ierr = SNESSetUpMatrices(nasm->subsnes[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_NASM(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  PetscErrorCode    ierr;
  PCASMType         asmtype;
  PetscBool         flg,monflg;
  SNES_NASM         *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Nonlinear Additive Schwarz options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_nasm_type","Type of restriction/extension","",SNESNASMTypes,(PetscEnum)nasm->type,(PetscEnum*)&asmtype,&flg);CHKERRQ(ierr);
  if (flg) {ierr = SNESNASMSetType(snes,asmtype);CHKERRQ(ierr);}
  flg    = PETSC_FALSE;
  monflg = PETSC_TRUE;
  ierr   = PetscOptionsReal("-snes_nasm_damping","The new solution is obtained as old solution plus dmp times (sum of the solutions on the subdomains)","SNESNASMSetDamping",nasm->damping,&nasm->damping,&flg);CHKERRQ(ierr);
  if (flg) {ierr = SNESNASMSetDamping(snes,nasm->damping);CHKERRQ(ierr);}
  ierr   = PetscOptionsDeprecated("-snes_nasm_sub_view",NULL,"3.15","Use -snes_view ::ascii_info_detail");CHKERRQ(ierr);
  ierr   = PetscOptionsBool("-snes_nasm_finaljacobian","Compute the global jacobian of the final iterate (for ASPIN)","",nasm->finaljacobian,&nasm->finaljacobian,NULL);CHKERRQ(ierr);
  ierr   = PetscOptionsEList("-snes_nasm_finaljacobian_type","The type of the final jacobian computed.","",SNESNASMFJTypes,3,SNESNASMFJTypes[0],&nasm->fjtype,NULL);CHKERRQ(ierr);
  ierr   = PetscOptionsBool("-snes_nasm_log","Log times for subSNES solves and restriction","",monflg,&monflg,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscLogEventRegister("SNESNASMSubSolve",((PetscObject)snes)->classid,&nasm->eventsubsolve);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("SNESNASMRestrict",((PetscObject)snes)->classid,&nasm->eventrestrictinterp);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_NASM(SNES snes, PetscViewer viewer)
{
  SNES_NASM         *nasm = (SNES_NASM*)snes->data;
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  PetscInt          i,N,bsz;
  PetscBool         iascii,isstring;
  PetscViewer       sviewer;
  MPI_Comm          comm;
  PetscViewerFormat format;
  const char        *prefix;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPIU_Allreduce(&nasm->n,&N,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  total subdomain blocks = %D\n",N);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (nasm->subsnes) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Local solver information for first block on rank 0:\n");CHKERRQ(ierr);
        ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Use -%ssnes_view ::ascii_info_detail to display information for all blocks\n",prefix?prefix:"");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
        if (rank == 0) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = SNESView(nasm->subsnes[0],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    } else {
      /* print the solver on each block */
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] number of local blocks = %D\n",(int)rank,nasm->n);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solver information for each block is in the following SNES objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      for (i=0; i<nasm->n; i++) {
        ierr = VecGetLocalSize(nasm->x[i],&bsz);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(sviewer,"[%d] local block number %D, size = %D\n",(int)rank,i,bsz);CHKERRQ(ierr);
        ierr = SNESView(nasm->subsnes[i],sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blocks=%D,type=%s",N,SNESNASMTypes[nasm->type]);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (nasm->subsnes && rank == 0) {ierr = SNESView(nasm->subsnes[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   SNESNASMSetType - Set the type of subdomain update used

   Logically Collective on SNES

   Input Parameters:
+  SNES - the SNES context
-  type - the type of update, PC_ASM_BASIC or PC_ASM_RESTRICT

   Level: intermediate

.seealso: SNESNASM, SNESNASMGetType(), PCASMSetType()
@*/
PetscErrorCode SNESNASMSetType(SNES snes,PCASMType type)
{
  PetscErrorCode ierr;
  PetscErrorCode (*f)(SNES,PCASMType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetType_C",&f);CHKERRQ(ierr);
  if (f) {ierr = (f)(snes,type);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMSetType_NASM(SNES snes,PCASMType type)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  if (type != PC_ASM_BASIC && type != PC_ASM_RESTRICT) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"SNESNASM only supports basic and restrict types");
  nasm->type = type;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMGetType - Get the type of subdomain update used

   Logically Collective on SNES

   Input Parameters:
.  SNES - the SNES context

   Output Parameters:
.  type - the type of update

   Level: intermediate

.seealso: SNESNASM, SNESNASMSetType(), PCASMGetType()
@*/
PetscErrorCode SNESNASMGetType(SNES snes,PCASMType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(snes,"SNESNASMGetType_C",(SNES,PCASMType*),(snes,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMGetType_NASM(SNES snes,PCASMType *type)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  *type = nasm->type;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMSetSubdomains - Manually Set the context required to restrict and solve subdomain problems.

   Not Collective

   Input Parameters:
+  SNES - the SNES context
.  n - the number of local subdomains
.  subsnes - solvers defined on the local subdomains
.  iscatter - scatters into the nonoverlapping portions of the local subdomains
.  oscatter - scatters into the overlapping portions of the local subdomains
-  gscatter - scatters into the (ghosted) local vector of the local subdomain

   Level: intermediate

.seealso: SNESNASM, SNESNASMGetSubdomains()
@*/
PetscErrorCode SNESNASMSetSubdomains(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[])
{
  PetscErrorCode ierr;
  PetscErrorCode (*f)(SNES,PetscInt,SNES*,VecScatter*,VecScatter*,VecScatter*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetSubdomains_C",&f);CHKERRQ(ierr);
  if (f) {ierr = (f)(snes,n,subsnes,iscatter,oscatter,gscatter);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMSetSubdomains_NASM(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  if (snes->setupcalled) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"SNESNASMSetSubdomains() should be called before calling SNESSetUp().");

  /* tear down the previously set things */
  ierr = SNESReset(snes);CHKERRQ(ierr);

  nasm->n = n;
  if (oscatter) {
    for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)oscatter[i]);CHKERRQ(ierr);}
  }
  if (iscatter) {
    for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)iscatter[i]);CHKERRQ(ierr);}
  }
  if (gscatter) {
    for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)gscatter[i]);CHKERRQ(ierr);}
  }
  if (oscatter) {
    ierr = PetscMalloc1(n,&nasm->oscatter);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&nasm->oscatter_copy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->oscatter[i] = oscatter[i];
      ierr = VecScatterCopy(oscatter[i], &nasm->oscatter_copy[i]);CHKERRQ(ierr);
    }
  }
  if (iscatter) {
    ierr = PetscMalloc1(n,&nasm->iscatter);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->iscatter[i] = iscatter[i];
    }
  }
  if (gscatter) {
    ierr = PetscMalloc1(n,&nasm->gscatter);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->gscatter[i] = gscatter[i];
    }
  }

  if (subsnes) {
    ierr = PetscMalloc1(n,&nasm->subsnes);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->subsnes[i] = subsnes[i];
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SNESNASMGetSubdomains - Get the local subdomain context.

   Not Collective

   Input Parameter:
.  SNES - the SNES context

   Output Parameters:
+  n - the number of local subdomains
.  subsnes - solvers defined on the local subdomains
.  iscatter - scatters into the nonoverlapping portions of the local subdomains
.  oscatter - scatters into the overlapping portions of the local subdomains
-  gscatter - scatters into the (ghosted) local vector of the local subdomain

   Level: intermediate

.seealso: SNESNASM, SNESNASMSetSubdomains()
@*/
PetscErrorCode SNESNASMGetSubdomains(SNES snes,PetscInt *n,SNES *subsnes[],VecScatter *iscatter[],VecScatter *oscatter[],VecScatter *gscatter[])
{
  PetscErrorCode ierr;
  PetscErrorCode (*f)(SNES,PetscInt*,SNES**,VecScatter**,VecScatter**,VecScatter**);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMGetSubdomains_C",&f);CHKERRQ(ierr);
  if (f) {ierr = (f)(snes,n,subsnes,iscatter,oscatter,gscatter);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMGetSubdomains_NASM(SNES snes,PetscInt *n,SNES *subsnes[],VecScatter *iscatter[],VecScatter *oscatter[],VecScatter *gscatter[])
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  if (n) *n = nasm->n;
  if (oscatter) *oscatter = nasm->oscatter;
  if (iscatter) *iscatter = nasm->iscatter;
  if (gscatter) *gscatter = nasm->gscatter;
  if (subsnes)  *subsnes  = nasm->subsnes;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMGetSubdomainVecs - Get the processor-local subdomain vectors

   Not Collective

   Input Parameter:
.  SNES - the SNES context

   Output Parameters:
+  n - the number of local subdomains
.  x - The subdomain solution vector
.  y - The subdomain step vector
.  b - The subdomain RHS vector
-  xl - The subdomain local vectors (ghosted)

   Level: developer

.seealso: SNESNASM, SNESNASMGetSubdomains()
@*/
PetscErrorCode SNESNASMGetSubdomainVecs(SNES snes,PetscInt *n,Vec **x,Vec **y,Vec **b, Vec **xl)
{
  PetscErrorCode ierr;
  PetscErrorCode (*f)(SNES,PetscInt*,Vec**,Vec**,Vec**,Vec**);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMGetSubdomainVecs_C",&f);CHKERRQ(ierr);
  if (f) {ierr = (f)(snes,n,x,y,b,xl);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMGetSubdomainVecs_NASM(SNES snes,PetscInt *n,Vec **x,Vec **y,Vec **b,Vec **xl)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  if (n)  *n  = nasm->n;
  if (x)  *x  = nasm->x;
  if (y)  *y  = nasm->y;
  if (b)  *b  = nasm->b;
  if (xl) *xl = nasm->xl;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMSetComputeFinalJacobian - Schedules the computation of the global and subdomain Jacobians upon convergence

   Collective on SNES

   Input Parameters:
+  SNES - the SNES context
-  flg - indication of whether to compute the Jacobians or not

   Level: developer

   Notes:
   This is used almost exclusively in the implementation of ASPIN, where the converged subdomain and global Jacobian
   is needed at each linear iteration.

.seealso: SNESNASM, SNESNASMGetSubdomains()
@*/
PetscErrorCode SNESNASMSetComputeFinalJacobian(SNES snes,PetscBool flg)
{
  PetscErrorCode (*f)(SNES,PetscBool);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetComputeFinalJacobian_C",&f);CHKERRQ(ierr);
  if (f) {ierr = (f)(snes,flg);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMSetComputeFinalJacobian_NASM(SNES snes,PetscBool flg)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  nasm->finaljacobian = flg;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMSetDamping - Sets the update damping for NASM

   Logically collective on SNES

   Input Parameters:
+  SNES - the SNES context
-  dmp - damping

   Level: intermediate

   Notes:
    The new solution is obtained as old solution plus dmp times (sum of the solutions on the subdomains)

.seealso: SNESNASM, SNESNASMGetDamping()
@*/
PetscErrorCode SNESNASMSetDamping(SNES snes,PetscReal dmp)
{
  PetscErrorCode (*f)(SNES,PetscReal);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetDamping_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {ierr = (f)(snes,dmp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMSetDamping_NASM(SNES snes,PetscReal dmp)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  nasm->damping = dmp;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMGetDamping - Gets the update damping for NASM

   Not Collective

   Input Parameters:
+  SNES - the SNES context
-  dmp - damping

   Level: intermediate

.seealso: SNESNASM, SNESNASMSetDamping()
@*/
PetscErrorCode SNESNASMGetDamping(SNES snes,PetscReal *dmp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(snes,"SNESNASMGetDamping_C",(SNES,PetscReal*),(snes,dmp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMGetDamping_NASM(SNES snes,PetscReal *dmp)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  *dmp = nasm->damping;
  PetscFunctionReturn(0);
}

/*
  Input Parameters:
+ snes - The solver
. B - The RHS vector
- X - The initial guess

  Output Parameters:
. Y - The solution update

  TODO: All scatters should be packed into one
*/
PetscErrorCode SNESNASMSolveLocal_Private(SNES snes,Vec B,Vec Y,Vec X)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
  SNES           subsnes;
  PetscInt       i;
  PetscReal      dmp;
  PetscErrorCode ierr;
  Vec            Xl,Bl,Yl,Xlloc;
  VecScatter     iscat,oscat,gscat,oscat_copy;
  DM             dm,subdm;
  PCASMType      type;

  PetscFunctionBegin;
  ierr = SNESNASMGetType(snes,&type);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = VecSet(Y,0);CHKERRQ(ierr);
  if (nasm->eventrestrictinterp) {ierr = PetscLogEventBegin(nasm->eventrestrictinterp,snes,0,0,0);CHKERRQ(ierr);}
  for (i=0; i<nasm->n; i++) {
    /* scatter the solution to the global solution and the local solution */
    Xl      = nasm->x[i];
    Xlloc   = nasm->xl[i];
    oscat   = nasm->oscatter[i];
    oscat_copy = nasm->oscatter_copy[i];
    gscat   = nasm->gscatter[i];
    ierr = VecScatterBegin(oscat,X,Xl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (B) {
      /* scatter the RHS to the local RHS */
      Bl   = nasm->b[i];
      ierr = VecScatterBegin(oscat_copy,B,Bl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  }
  if (nasm->eventrestrictinterp) {ierr = PetscLogEventEnd(nasm->eventrestrictinterp,snes,0,0,0);CHKERRQ(ierr);}

  if (nasm->eventsubsolve) {ierr = PetscLogEventBegin(nasm->eventsubsolve,snes,0,0,0);CHKERRQ(ierr);}
  for (i=0; i<nasm->n; i++) {
    Xl    = nasm->x[i];
    Xlloc = nasm->xl[i];
    Yl    = nasm->y[i];
    subsnes = nasm->subsnes[i];
    ierr    = SNESGetDM(subsnes,&subdm);CHKERRQ(ierr);
    iscat   = nasm->iscatter[i];
    oscat   = nasm->oscatter[i];
    oscat_copy = nasm->oscatter_copy[i];
    gscat   = nasm->gscatter[i];
    ierr = VecScatterEnd(oscat,X,Xl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (B) {
      Bl   = nasm->b[i];
      ierr = VecScatterEnd(oscat_copy,B,Bl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    } else Bl = NULL;

    ierr = DMSubDomainRestrict(dm,oscat,gscat,subdm);CHKERRQ(ierr);
    ierr = VecCopy(Xl,Yl);CHKERRQ(ierr);
    ierr = SNESSolve(subsnes,Bl,Xl);CHKERRQ(ierr);
    ierr = VecAYPX(Yl,-1.0,Xl);CHKERRQ(ierr);
    ierr = VecScale(Yl, nasm->damping);CHKERRQ(ierr);
    if (type == PC_ASM_BASIC) {
      ierr = VecScatterBegin(oscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    } else if (type == PC_ASM_RESTRICT) {
      ierr = VecScatterBegin(iscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(iscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Only basic and restrict types are supported for SNESNASM");
  }
  if (nasm->eventsubsolve) {ierr = PetscLogEventEnd(nasm->eventsubsolve,snes,0,0,0);CHKERRQ(ierr);}
  if (nasm->eventrestrictinterp) {ierr = PetscLogEventBegin(nasm->eventrestrictinterp,snes,0,0,0);CHKERRQ(ierr);}
  if (nasm->weight_set) {
    ierr = VecPointwiseMult(Y,Y,nasm->weight);CHKERRQ(ierr);
  }
  if (nasm->eventrestrictinterp) {ierr = PetscLogEventEnd(nasm->eventrestrictinterp,snes,0,0,0);CHKERRQ(ierr);}
  ierr = SNESNASMGetDamping(snes,&dmp);CHKERRQ(ierr);
  ierr = VecAXPY(X,dmp,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMComputeFinalJacobian_Private(SNES snes, Vec Xfinal)
{
  Vec            X = Xfinal;
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
  SNES           subsnes;
  PetscInt       i,lag = 1;
  PetscErrorCode ierr;
  Vec            Xlloc,Xl,Fl,F;
  VecScatter     oscat,gscat;
  DM             dm,subdm;

  PetscFunctionBegin;
  if (nasm->fjtype == 2) X = nasm->xinit;
  F = snes->vec_func;
  if (snes->normschedule == SNES_NORM_NONE) {ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);}
  ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  if (nasm->eventrestrictinterp) {ierr = PetscLogEventBegin(nasm->eventrestrictinterp,snes,0,0,0);CHKERRQ(ierr);}
  if (nasm->fjtype != 1) {
    for (i=0; i<nasm->n; i++) {
      Xlloc = nasm->xl[i];
      gscat = nasm->gscatter[i];
      ierr = VecScatterBegin(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  }
  if (nasm->eventrestrictinterp) {ierr = PetscLogEventEnd(nasm->eventrestrictinterp,snes,0,0,0);CHKERRQ(ierr);}
  for (i=0; i<nasm->n; i++) {
    Fl      = nasm->subsnes[i]->vec_func;
    Xl      = nasm->x[i];
    Xlloc   = nasm->xl[i];
    subsnes = nasm->subsnes[i];
    oscat   = nasm->oscatter[i];
    gscat   = nasm->gscatter[i];
    if (nasm->fjtype != 1) {ierr = VecScatterEnd(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);}
    ierr = SNESGetDM(subsnes,&subdm);CHKERRQ(ierr);
    ierr = DMSubDomainRestrict(dm,oscat,gscat,subdm);CHKERRQ(ierr);
    if (nasm->fjtype != 1) {
      ierr = DMLocalToGlobalBegin(subdm,Xlloc,INSERT_VALUES,Xl);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(subdm,Xlloc,INSERT_VALUES,Xl);CHKERRQ(ierr);
    }
    if (subsnes->lagjacobian == -1)    subsnes->lagjacobian = -2;
    else if (subsnes->lagjacobian > 1) lag = subsnes->lagjacobian;
    ierr = SNESComputeFunction(subsnes,Xl,Fl);CHKERRQ(ierr);
    ierr = SNESComputeJacobian(subsnes,Xl,subsnes->jacobian,subsnes->jacobian_pre);CHKERRQ(ierr);
    if (lag > 1) subsnes->lagjacobian = lag;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_NASM(SNES snes)
{
  Vec              F;
  Vec              X;
  Vec              B;
  Vec              Y;
  PetscInt         i;
  PetscReal        fnorm = 0.0;
  PetscErrorCode   ierr;
  SNESNormSchedule normschedule;
  SNES_NASM        *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;

  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  ierr = PetscCitationsRegister(SNESCitation,&SNEScite);CHKERRQ(ierr);
  X = snes->vec_sol;
  Y = snes->vec_sol_update;
  F = snes->vec_func;
  B = snes->vec_rhs;

  ierr         = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter   = 0;
  snes->norm   = 0.;
  ierr         = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;
  ierr         = SNESGetNormSchedule(snes, &normschedule);CHKERRQ(ierr);
  if (normschedule == SNES_NORM_ALWAYS || normschedule == SNES_NORM_INITIAL_ONLY || normschedule == SNES_NORM_INITIAL_FINAL_ONLY) {
    /* compute the initial function and preconditioned update delX */
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    } else snes->vec_func_init_set = PETSC_FALSE;

    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    SNESCheckFunctionNorm(snes,fnorm);
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);

    /* test convergence */
    ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }
  /* copy the initial solution over for later */
  if (nasm->fjtype == 2) {ierr = VecCopy(X,nasm->xinit);CHKERRQ(ierr);}

  for (i=0; i < snes->max_its; i++) {
    ierr = SNESNASMSolveLocal_Private(snes,B,Y,X);CHKERRQ(ierr);
    if (normschedule == SNES_NORM_ALWAYS || ((i == snes->max_its - 1) && (normschedule == SNES_NORM_INITIAL_FINAL_ONLY || normschedule == SNES_NORM_FINAL_ONLY))) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      SNESCheckFunctionNorm(snes,fnorm);
    }
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (normschedule == SNES_NORM_ALWAYS) {ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);}
    if (snes->reason) break;
    /* Call general purpose update function */
    if (snes->ops->update) {ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);}
  }
  if (nasm->finaljacobian) {
    ierr = SNESNASMComputeFinalJacobian_Private(snes,X);CHKERRQ(ierr);
    SNESCheckJacobianDomainerror(snes);
  }
  if (normschedule == SNES_NORM_ALWAYS) {
    if (i == snes->max_its) {
      ierr = PetscInfo(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
      if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
    }
  } else if (!snes->reason) snes->reason = SNES_CONVERGED_ITS; /* NASM is meant to be used as a preconditioner */
  PetscFunctionReturn(0);
}

/*MC
  SNESNASM - Nonlinear Additive Schwarz

   Options Database:
+  -snes_nasm_log - enable logging events for the communication and solve stages
.  -snes_nasm_type <basic,restrict> - type of subdomain update used
.  -snes_asm_damping <dmp> - the new solution is obtained as old solution plus dmp times (sum of the solutions on the subdomains)
.  -snes_nasm_finaljacobian - compute the local and global jacobians of the final iterate
.  -snes_nasm_finaljacobian_type <finalinner,finalouter,initial> - pick state the jacobian is calculated at
.  -sub_snes_ - options prefix of the subdomain nonlinear solves
.  -sub_ksp_ - options prefix of the subdomain Krylov solver
-  -sub_pc_ - options prefix of the subdomain preconditioner

   Level: advanced

   Developer Note: This is a non-Newton based nonlinear solver that does not directly require a Jacobian; hence the flag snes->usesksp is set to
       false and SNESView() and -snes_view do not display a KSP object. However, if the flag nasm->finaljacobian is set (for example, if
       NASM is used as a nonlinear preconditioner for  KSPASPIN) then SNESSetUpMatrices() is called to generate the Jacobian (needed by KSPASPIN)
       and this utilizes the KSP for storing the matrices, but the KSP is never used for solving a linear system. Note that when SNESNASM is
       used by SNESASPIN they share the same Jacobian matrices because SNESSetUp() (called on the outer SNES KSPASPIN) causes the inner SNES
       object (in this case SNESNASM) to inherit the outer Jacobian matrices.

   References:
.  1. - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types), SNESNASMSetType(), SNESNASMGetType(), SNESNASMSetSubdomains(), SNESNASMGetSubdomains(), SNESNASMGetSubdomainVecs(), SNESNASMSetComputeFinalJacobian(), SNESNASMSetDamping(), SNESNASMGetDamping()
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_NASM(SNES snes)
{
  SNES_NASM      *nasm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr       = PetscNewLog(snes,&nasm);CHKERRQ(ierr);
  snes->data = (void*)nasm;

  nasm->n        = PETSC_DECIDE;
  nasm->subsnes  = NULL;
  nasm->x        = NULL;
  nasm->xl       = NULL;
  nasm->y        = NULL;
  nasm->b        = NULL;
  nasm->oscatter = NULL;
  nasm->oscatter_copy = NULL;
  nasm->iscatter = NULL;
  nasm->gscatter = NULL;
  nasm->damping  = 1.;

  nasm->type              = PC_ASM_BASIC;
  nasm->finaljacobian     = PETSC_FALSE;
  nasm->weight_set        = PETSC_FALSE;

  snes->ops->destroy        = SNESDestroy_NASM;
  snes->ops->setup          = SNESSetUp_NASM;
  snes->ops->setfromoptions = SNESSetFromOptions_NASM;
  snes->ops->view           = SNESView_NASM;
  snes->ops->solve          = SNESSolve_NASM;
  snes->ops->reset          = SNESReset_NASM;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  nasm->fjtype              = 0;
  nasm->xinit               = NULL;
  nasm->eventrestrictinterp = 0;
  nasm->eventsubsolve       = 0;

  if (!snes->tolerancesset) {
    snes->max_its   = 10000;
    snes->max_funcs = 10000;
  }

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetType_C",SNESNASMSetType_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetType_C",SNESNASMGetType_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetSubdomains_C",SNESNASMSetSubdomains_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetSubdomains_C",SNESNASMGetSubdomains_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetDamping_C",SNESNASMSetDamping_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetDamping_C",SNESNASMGetDamping_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetSubdomainVecs_C",SNESNASMGetSubdomainVecs_NASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetComputeFinalJacobian_C",SNESNASMSetComputeFinalJacobian_NASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   SNESNASMGetSNES - Gets a subsolver

   Not collective

   Input Parameters:
+  snes - the SNES context
-  i - the number of the subsnes to get

   Output Parameters:
.  subsnes - the subsolver context

   Level: intermediate

.seealso: SNESNASM, SNESNASMGetNumber()
@*/
PetscErrorCode SNESNASMGetSNES(SNES snes,PetscInt i,SNES *subsnes)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  if (i < 0 || i >= nasm->n) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"No such subsolver");
  *subsnes = nasm->subsnes[i];
  PetscFunctionReturn(0);
}

/*@
   SNESNASMGetNumber - Gets number of subsolvers

   Not collective

   Input Parameters:
.  snes - the SNES context

   Output Parameters:
.  n - the number of subsolvers

   Level: intermediate

.seealso: SNESNASM, SNESNASMGetSNES()
@*/
PetscErrorCode SNESNASMGetNumber(SNES snes,PetscInt *n)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  *n = nasm->n;
  PetscFunctionReturn(0);
}

/*@
   SNESNASMSetWeight - Sets weight to use when adding overlapping updates

   Collective

   Input Parameters:
+  snes - the SNES context
-  weight - the weights to use (typically 1/N for each dof, where N is the number of patches it appears in)

   Level: intermediate

.seealso: SNESNASM
@*/
PetscErrorCode SNESNASMSetWeight(SNES snes,Vec weight)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecDestroy(&nasm->weight);CHKERRQ(ierr);
  nasm->weight_set = PETSC_TRUE;
  nasm->weight     = weight;
  ierr = PetscObjectReference((PetscObject)nasm->weight);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
