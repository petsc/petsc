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
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nasm->n; i++) {
    if (nasm->xl) PetscCall(VecDestroy(&nasm->xl[i]));
    if (nasm->x) PetscCall(VecDestroy(&nasm->x[i]));
    if (nasm->y) PetscCall(VecDestroy(&nasm->y[i]));
    if (nasm->b) PetscCall(VecDestroy(&nasm->b[i]));

    if (nasm->subsnes) PetscCall(SNESDestroy(&nasm->subsnes[i]));
    if (nasm->oscatter) PetscCall(VecScatterDestroy(&nasm->oscatter[i]));
    if (nasm->oscatter_copy) PetscCall(VecScatterDestroy(&nasm->oscatter_copy[i]));
    if (nasm->iscatter) PetscCall(VecScatterDestroy(&nasm->iscatter[i]));
    if (nasm->gscatter) PetscCall(VecScatterDestroy(&nasm->gscatter[i]));
  }

  PetscCall(PetscFree(nasm->x));
  PetscCall(PetscFree(nasm->xl));
  PetscCall(PetscFree(nasm->y));
  PetscCall(PetscFree(nasm->b));

  if (nasm->xinit) PetscCall(VecDestroy(&nasm->xinit));

  PetscCall(PetscFree(nasm->subsnes));
  PetscCall(PetscFree(nasm->oscatter));
  PetscCall(PetscFree(nasm->oscatter_copy));
  PetscCall(PetscFree(nasm->iscatter));
  PetscCall(PetscFree(nasm->gscatter));

  if (nasm->weight_set) {
    PetscCall(VecDestroy(&nasm->weight));
  }

  nasm->eventrestrictinterp = 0;
  nasm->eventsubsolve = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_NASM(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESReset_NASM(snes));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalSubDomainDirichletHook_Private(DM dm,Vec g,InsertMode mode,Vec l,void *ctx)
{
  Vec            bcs = (Vec)ctx;

  PetscFunctionBegin;
  PetscCall(VecCopy(bcs,l));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_NASM(SNES snes)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
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
    PetscCall(SNESGetDM(snes,&dm));
    if (dm) {
      nasm->usesdm = PETSC_TRUE;
      PetscCall(DMCreateDomainDecomposition(dm,&nasm->n,NULL,NULL,NULL,&subdms));
      PetscCheck(subdms,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM has no default decomposition defined.  Set subsolves manually with SNESNASMSetSubdomains().");
      PetscCall(DMCreateDomainDecompositionScatters(dm,nasm->n,subdms,&nasm->iscatter,&nasm->oscatter,&nasm->gscatter));
      PetscCall(PetscMalloc1(nasm->n, &nasm->oscatter_copy));
      for (i=0; i<nasm->n; i++) {
        PetscCall(VecScatterCopy(nasm->oscatter[i], &nasm->oscatter_copy[i]));
      }

      PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
      PetscCall(PetscMalloc1(nasm->n,&nasm->subsnes));
      for (i=0; i<nasm->n; i++) {
        PetscCall(SNESCreate(PETSC_COMM_SELF,&nasm->subsnes[i]));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)nasm->subsnes[i], (PetscObject)snes, 1));
        PetscCall(SNESAppendOptionsPrefix(nasm->subsnes[i],optionsprefix));
        PetscCall(SNESAppendOptionsPrefix(nasm->subsnes[i],"sub_"));
        PetscCall(SNESSetDM(nasm->subsnes[i],subdms[i]));
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)nasm->subsnes[i]),&size));
        if (size == 1) {
          PetscCall(SNESGetKSP(nasm->subsnes[i],&ksp));
          PetscCall(KSPGetPC(ksp,&pc));
          PetscCall(KSPSetType(ksp,KSPPREONLY));
          PetscCall(PCSetType(pc,PCLU));
        }
        PetscCall(SNESSetFromOptions(nasm->subsnes[i]));
        PetscCall(DMDestroy(&subdms[i]));
      }
      PetscCall(PetscFree(subdms));
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Cannot construct local problems automatically without a DM!");
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Must set subproblems manually if there is no DM!");
  /* allocate the global vectors */
  if (!nasm->x) {
    PetscCall(PetscCalloc1(nasm->n,&nasm->x));
  }
  if (!nasm->xl) {
    PetscCall(PetscCalloc1(nasm->n,&nasm->xl));
  }
  if (!nasm->y) {
    PetscCall(PetscCalloc1(nasm->n,&nasm->y));
  }
  if (!nasm->b) {
    PetscCall(PetscCalloc1(nasm->n,&nasm->b));
  }

  for (i=0; i<nasm->n; i++) {
    PetscCall(SNESGetFunction(nasm->subsnes[i],&F,NULL,NULL));
    if (!nasm->x[i]) PetscCall(VecDuplicate(F,&nasm->x[i]));
    if (!nasm->y[i]) PetscCall(VecDuplicate(F,&nasm->y[i]));
    if (!nasm->b[i]) PetscCall(VecDuplicate(F,&nasm->b[i]));
    if (!nasm->xl[i]) {
      PetscCall(SNESGetDM(nasm->subsnes[i],&subdm));
      PetscCall(DMCreateLocalVector(subdm,&nasm->xl[i]));
      PetscCall(DMGlobalToLocalHookAdd(subdm,DMGlobalToLocalSubDomainDirichletHook_Private,NULL,nasm->xl[i]));
    }
  }
  if (nasm->finaljacobian) {
    PetscCall(SNESSetUpMatrices(snes));
    if (nasm->fjtype == 2) {
      PetscCall(VecDuplicate(snes->vec_sol,&nasm->xinit));
    }
    for (i=0; i<nasm->n;i++) {
      PetscCall(SNESSetUpMatrices(nasm->subsnes[i]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_NASM(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  PCASMType         asmtype;
  PetscBool         flg,monflg;
  SNES_NASM         *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Nonlinear Additive Schwarz options");
  PetscCall(PetscOptionsEnum("-snes_nasm_type","Type of restriction/extension","",SNESNASMTypes,(PetscEnum)nasm->type,(PetscEnum*)&asmtype,&flg));
  if (flg) PetscCall(SNESNASMSetType(snes,asmtype));
  flg    = PETSC_FALSE;
  monflg = PETSC_TRUE;
  PetscCall(PetscOptionsReal("-snes_nasm_damping","The new solution is obtained as old solution plus dmp times (sum of the solutions on the subdomains)","SNESNASMSetDamping",nasm->damping,&nasm->damping,&flg));
  if (flg) PetscCall(SNESNASMSetDamping(snes,nasm->damping));
  PetscCall(PetscOptionsDeprecated("-snes_nasm_sub_view",NULL,"3.15","Use -snes_view ::ascii_info_detail"));
  PetscCall(PetscOptionsBool("-snes_nasm_finaljacobian","Compute the global jacobian of the final iterate (for ASPIN)","",nasm->finaljacobian,&nasm->finaljacobian,NULL));
  PetscCall(PetscOptionsEList("-snes_nasm_finaljacobian_type","The type of the final jacobian computed.","",SNESNASMFJTypes,3,SNESNASMFJTypes[0],&nasm->fjtype,NULL));
  PetscCall(PetscOptionsBool("-snes_nasm_log","Log times for subSNES solves and restriction","",monflg,&monflg,&flg));
  if (flg) {
    PetscCall(PetscLogEventRegister("SNESNASMSubSolve",((PetscObject)snes)->classid,&nasm->eventsubsolve));
    PetscCall(PetscLogEventRegister("SNESNASMRestrict",((PetscObject)snes)->classid,&nasm->eventrestrictinterp));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_NASM(SNES snes, PetscViewer viewer)
{
  SNES_NASM         *nasm = (SNES_NASM*)snes->data;
  PetscMPIInt       rank,size;
  PetscInt          i,N,bsz;
  PetscBool         iascii,isstring;
  PetscViewer       sviewer;
  MPI_Comm          comm;
  PetscViewerFormat format;
  const char        *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)snes,&comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(MPIU_Allreduce(&nasm->n,&N,1,MPIU_INT,MPI_SUM,comm));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  total subdomain blocks = %" PetscInt_FMT "\n",N));
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (nasm->subsnes) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Local solver information for first block on rank 0:\n"));
        PetscCall(SNESGetOptionsPrefix(snes,&prefix));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Use -%ssnes_view ::ascii_info_detail to display information for all blocks\n",prefix?prefix:""));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        if (rank == 0) {
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(SNESView(nasm->subsnes[0],sviewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
    } else {
      /* print the solver on each block */
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] number of local blocks = %" PetscInt_FMT "\n",(int)rank,nasm->n));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Local solver information for each block is in the following SNES objects:\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n"));
      PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      for (i=0; i<nasm->n; i++) {
        PetscCall(VecGetLocalSize(nasm->x[i],&bsz));
        PetscCall(PetscViewerASCIIPrintf(sviewer,"[%d] local block number %" PetscInt_FMT ", size = %" PetscInt_FMT "\n",(int)rank,i,bsz));
        PetscCall(SNESView(nasm->subsnes[i],sviewer));
        PetscCall(PetscViewerASCIIPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n"));
      }
      PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer," blocks=%" PetscInt_FMT ",type=%s",N,SNESNASMTypes[nasm->type]));
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (nasm->subsnes && rank == 0) PetscCall(SNESView(nasm->subsnes[0],sviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
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
  PetscErrorCode (*f)(SNES,PCASMType);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetType_C",&f));
  if (f) PetscCall((f)(snes,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMSetType_NASM(SNES snes,PCASMType type)
{
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  PetscCheck(type == PC_ASM_BASIC || type == PC_ASM_RESTRICT,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"SNESNASM only supports basic and restrict types");
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
  PetscFunctionBegin;
  PetscUseMethod(snes,"SNESNASMGetType_C",(SNES,PCASMType*),(snes,type));
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
  PetscErrorCode (*f)(SNES,PetscInt,SNES*,VecScatter*,VecScatter*,VecScatter*);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetSubdomains_C",&f));
  if (f) PetscCall((f)(snes,n,subsnes,iscatter,oscatter,gscatter));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMSetSubdomains_NASM(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[])
{
  PetscInt       i;
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;
  PetscCheck(!snes->setupcalled,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"SNESNASMSetSubdomains() should be called before calling SNESSetUp().");

  /* tear down the previously set things */
  PetscCall(SNESReset(snes));

  nasm->n = n;
  if (oscatter) {
    for (i=0; i<n; i++) PetscCall(PetscObjectReference((PetscObject)oscatter[i]));
  }
  if (iscatter) {
    for (i=0; i<n; i++) PetscCall(PetscObjectReference((PetscObject)iscatter[i]));
  }
  if (gscatter) {
    for (i=0; i<n; i++) PetscCall(PetscObjectReference((PetscObject)gscatter[i]));
  }
  if (oscatter) {
    PetscCall(PetscMalloc1(n,&nasm->oscatter));
    PetscCall(PetscMalloc1(n,&nasm->oscatter_copy));
    for (i=0; i<n; i++) {
      nasm->oscatter[i] = oscatter[i];
      PetscCall(VecScatterCopy(oscatter[i], &nasm->oscatter_copy[i]));
    }
  }
  if (iscatter) {
    PetscCall(PetscMalloc1(n,&nasm->iscatter));
    for (i=0; i<n; i++) {
      nasm->iscatter[i] = iscatter[i];
    }
  }
  if (gscatter) {
    PetscCall(PetscMalloc1(n,&nasm->gscatter));
    for (i=0; i<n; i++) {
      nasm->gscatter[i] = gscatter[i];
    }
  }

  if (subsnes) {
    PetscCall(PetscMalloc1(n,&nasm->subsnes));
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
  PetscErrorCode (*f)(SNES,PetscInt*,SNES**,VecScatter**,VecScatter**,VecScatter**);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNASMGetSubdomains_C",&f));
  if (f) PetscCall((f)(snes,n,subsnes,iscatter,oscatter,gscatter));
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
  PetscErrorCode (*f)(SNES,PetscInt*,Vec**,Vec**,Vec**,Vec**);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNASMGetSubdomainVecs_C",&f));
  if (f) PetscCall((f)(snes,n,x,y,b,xl));
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

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetComputeFinalJacobian_C",&f));
  if (f) PetscCall((f)(snes,flg));
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

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetDamping_C",(void (**)(void))&f));
  if (f) PetscCall((f)(snes,dmp));
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
  PetscFunctionBegin;
  PetscUseMethod(snes,"SNESNASMGetDamping_C",(SNES,PetscReal*),(snes,dmp));
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
  Vec            Xl,Bl,Yl,Xlloc;
  VecScatter     iscat,oscat,gscat,oscat_copy;
  DM             dm,subdm;
  PCASMType      type;

  PetscFunctionBegin;
  PetscCall(SNESNASMGetType(snes,&type));
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(VecSet(Y,0));
  if (nasm->eventrestrictinterp) PetscCall(PetscLogEventBegin(nasm->eventrestrictinterp,snes,0,0,0));
  for (i=0; i<nasm->n; i++) {
    /* scatter the solution to the global solution and the local solution */
    Xl      = nasm->x[i];
    Xlloc   = nasm->xl[i];
    oscat   = nasm->oscatter[i];
    oscat_copy = nasm->oscatter_copy[i];
    gscat   = nasm->gscatter[i];
    PetscCall(VecScatterBegin(oscat,X,Xl,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD));
    if (B) {
      /* scatter the RHS to the local RHS */
      Bl   = nasm->b[i];
      PetscCall(VecScatterBegin(oscat_copy,B,Bl,INSERT_VALUES,SCATTER_FORWARD));
    }
  }
  if (nasm->eventrestrictinterp) PetscCall(PetscLogEventEnd(nasm->eventrestrictinterp,snes,0,0,0));

  if (nasm->eventsubsolve) PetscCall(PetscLogEventBegin(nasm->eventsubsolve,snes,0,0,0));
  for (i=0; i<nasm->n; i++) {
    Xl    = nasm->x[i];
    Xlloc = nasm->xl[i];
    Yl    = nasm->y[i];
    subsnes = nasm->subsnes[i];
    PetscCall(SNESGetDM(subsnes,&subdm));
    iscat   = nasm->iscatter[i];
    oscat   = nasm->oscatter[i];
    oscat_copy = nasm->oscatter_copy[i];
    gscat   = nasm->gscatter[i];
    PetscCall(VecScatterEnd(oscat,X,Xl,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD));
    if (B) {
      Bl   = nasm->b[i];
      PetscCall(VecScatterEnd(oscat_copy,B,Bl,INSERT_VALUES,SCATTER_FORWARD));
    } else Bl = NULL;

    PetscCall(DMSubDomainRestrict(dm,oscat,gscat,subdm));
    PetscCall(VecCopy(Xl,Yl));
    PetscCall(SNESSolve(subsnes,Bl,Xl));
    PetscCall(VecAYPX(Yl,-1.0,Xl));
    PetscCall(VecScale(Yl, nasm->damping));
    if (type == PC_ASM_BASIC) {
      PetscCall(VecScatterBegin(oscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(oscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE));
    } else if (type == PC_ASM_RESTRICT) {
      PetscCall(VecScatterBegin(iscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(iscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE));
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Only basic and restrict types are supported for SNESNASM");
  }
  if (nasm->eventsubsolve) PetscCall(PetscLogEventEnd(nasm->eventsubsolve,snes,0,0,0));
  if (nasm->eventrestrictinterp) PetscCall(PetscLogEventBegin(nasm->eventrestrictinterp,snes,0,0,0));
  if (nasm->weight_set) {
    PetscCall(VecPointwiseMult(Y,Y,nasm->weight));
  }
  if (nasm->eventrestrictinterp) PetscCall(PetscLogEventEnd(nasm->eventrestrictinterp,snes,0,0,0));
  PetscCall(SNESNASMGetDamping(snes,&dmp));
  PetscCall(VecAXPY(X,dmp,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNASMComputeFinalJacobian_Private(SNES snes, Vec Xfinal)
{
  Vec            X = Xfinal;
  SNES_NASM      *nasm = (SNES_NASM*)snes->data;
  SNES           subsnes;
  PetscInt       i,lag = 1;
  Vec            Xlloc,Xl,Fl,F;
  VecScatter     oscat,gscat;
  DM             dm,subdm;

  PetscFunctionBegin;
  if (nasm->fjtype == 2) X = nasm->xinit;
  F = snes->vec_func;
  if (snes->normschedule == SNES_NORM_NONE) PetscCall(SNESComputeFunction(snes,X,F));
  PetscCall(SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre));
  PetscCall(SNESGetDM(snes,&dm));
  if (nasm->eventrestrictinterp) PetscCall(PetscLogEventBegin(nasm->eventrestrictinterp,snes,0,0,0));
  if (nasm->fjtype != 1) {
    for (i=0; i<nasm->n; i++) {
      Xlloc = nasm->xl[i];
      gscat = nasm->gscatter[i];
      PetscCall(VecScatterBegin(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD));
    }
  }
  if (nasm->eventrestrictinterp) PetscCall(PetscLogEventEnd(nasm->eventrestrictinterp,snes,0,0,0));
  for (i=0; i<nasm->n; i++) {
    Fl      = nasm->subsnes[i]->vec_func;
    Xl      = nasm->x[i];
    Xlloc   = nasm->xl[i];
    subsnes = nasm->subsnes[i];
    oscat   = nasm->oscatter[i];
    gscat   = nasm->gscatter[i];
    if (nasm->fjtype != 1) PetscCall(VecScatterEnd(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(SNESGetDM(subsnes,&subdm));
    PetscCall(DMSubDomainRestrict(dm,oscat,gscat,subdm));
    if (nasm->fjtype != 1) {
      PetscCall(DMLocalToGlobalBegin(subdm,Xlloc,INSERT_VALUES,Xl));
      PetscCall(DMLocalToGlobalEnd(subdm,Xlloc,INSERT_VALUES,Xl));
    }
    if (subsnes->lagjacobian == -1)    subsnes->lagjacobian = -2;
    else if (subsnes->lagjacobian > 1) lag = subsnes->lagjacobian;
    PetscCall(SNESComputeFunction(subsnes,Xl,Fl));
    PetscCall(SNESComputeJacobian(subsnes,Xl,subsnes->jacobian,subsnes->jacobian_pre));
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
  SNESNormSchedule normschedule;
  SNES_NASM        *nasm = (SNES_NASM*)snes->data;

  PetscFunctionBegin;

  PetscCheck(!snes->xl & !snes->xu && !snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscCitationsRegister(SNESCitation,&SNEScite));
  X = snes->vec_sol;
  Y = snes->vec_sol_update;
  F = snes->vec_func;
  B = snes->vec_rhs;

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter   = 0;
  snes->norm   = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  snes->reason = SNES_CONVERGED_ITERATING;
  PetscCall(SNESGetNormSchedule(snes, &normschedule));
  if (normschedule == SNES_NORM_ALWAYS || normschedule == SNES_NORM_INITIAL_ONLY || normschedule == SNES_NORM_INITIAL_FINAL_ONLY) {
    /* compute the initial function and preconditioned update delX */
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes,X,F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
    SNESCheckFunctionNorm(snes,fnorm);
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = 0;
    snes->norm = fnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,0,snes->norm));

    /* test convergence */
    PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,0,snes->norm));
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    PetscCall((*snes->ops->update)(snes, snes->iter));
  }
  /* copy the initial solution over for later */
  if (nasm->fjtype == 2) PetscCall(VecCopy(X,nasm->xinit));

  for (i=0; i < snes->max_its; i++) {
    PetscCall(SNESNASMSolveLocal_Private(snes,B,Y,X));
    if (normschedule == SNES_NORM_ALWAYS || ((i == snes->max_its - 1) && (normschedule == SNES_NORM_INITIAL_FINAL_ONLY || normschedule == SNES_NORM_FINAL_ONLY))) {
      PetscCall(SNESComputeFunction(snes,X,F));
      PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
      SNESCheckFunctionNorm(snes,fnorm);
    }
    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i+1;
    snes->norm = fnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence */
    if (normschedule == SNES_NORM_ALWAYS) PetscCall((*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;
    /* Call general purpose update function */
    if (snes->ops->update) PetscCall((*snes->ops->update)(snes, snes->iter));
  }
  if (nasm->finaljacobian) {
    PetscCall(SNESNASMComputeFinalJacobian_Private(snes,X));
    SNESCheckJacobianDomainerror(snes);
  }
  if (normschedule == SNES_NORM_ALWAYS) {
    if (i == snes->max_its) {
      PetscCall(PetscInfo(snes,"Maximum number of iterations has been reached: %" PetscInt_FMT "\n",snes->max_its));
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
.  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType, SNESNASMSetType(), SNESNASMGetType(), SNESNASMSetSubdomains(), SNESNASMGetSubdomains(), SNESNASMGetSubdomainVecs(), SNESNASMSetComputeFinalJacobian(), SNESNASMSetDamping(), SNESNASMGetDamping()
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_NASM(SNES snes)
{
  SNES_NASM      *nasm;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(snes,&nasm));
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

  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetType_C",SNESNASMSetType_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetType_C",SNESNASMGetType_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetSubdomains_C",SNESNASMSetSubdomains_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetSubdomains_C",SNESNASMGetSubdomains_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetDamping_C",SNESNASMSetDamping_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetDamping_C",SNESNASMGetDamping_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMGetSubdomainVecs_C",SNESNASMGetSubdomainVecs_NASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNASMSetComputeFinalJacobian_C",SNESNASMSetComputeFinalJacobian_NASM));
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
  PetscCheck(i >= 0 && i < nasm->n,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"No such subsolver");
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

  PetscFunctionBegin;

  PetscCall(VecDestroy(&nasm->weight));
  nasm->weight_set = PETSC_TRUE;
  nasm->weight     = weight;
  PetscCall(PetscObjectReference((PetscObject)nasm->weight));

  PetscFunctionReturn(0);
}
