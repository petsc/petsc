#include <petsc/private/dmimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/petschpddm.h> /*I "petscpc.h" I*/
#include <petsc/private/pcimpl.h> /* this must be included after petschpddm.h so that _PCIMPL_H is not defined            */
                                  /* otherwise, it is assumed that one is compiling libhpddm_petsc => circular dependency */
#if defined(PETSC_HAVE_FORTRAN)
#include <petsc/private/fortranimpl.h>
#endif

static PetscErrorCode (*loadedSym)(HPDDM::Schwarz<PetscScalar>* const, IS, Mat, Mat, Mat, std::vector<Vec>, PC_HPDDM_Level** const) = NULL;

static PetscBool PCHPDDMPackageInitialized = PETSC_FALSE;

PetscLogEvent PC_HPDDM_Strc;
PetscLogEvent PC_HPDDM_PtAP;
PetscLogEvent PC_HPDDM_PtBP;
PetscLogEvent PC_HPDDM_Next;
PetscLogEvent PC_HPDDM_SetUp[PETSC_PCHPDDM_MAXLEVELS];
PetscLogEvent PC_HPDDM_Solve[PETSC_PCHPDDM_MAXLEVELS];

const char *const PCHPDDMCoarseCorrectionTypes[] = { "DEFLATED", "ADDITIVE", "BALANCED", "PCHPDDMCoarseCorrectionType", "PC_HPDDM_COARSE_CORRECTION_", NULL };

static PetscErrorCode PCReset_HPDDM(PC pc)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;
  PetscInt i;

  PetscFunctionBegin;
  if (data->levels) {
    for (i = 0; i < PETSC_PCHPDDM_MAXLEVELS && data->levels[i]; ++i) {
      PetscCall(KSPDestroy(&data->levels[i]->ksp));
      PetscCall(PCDestroy(&data->levels[i]->pc));
      PetscCall(PetscFree(data->levels[i]));
    }
    PetscCall(PetscFree(data->levels));
  }

  PetscCall(ISDestroy(&data->is));
  PetscCall(MatDestroy(&data->aux));
  PetscCall(MatDestroy(&data->B));
  PetscCall(VecDestroy(&data->normal));
  data->correction = PC_HPDDM_COARSE_CORRECTION_DEFLATED;
  data->Neumann    = PETSC_FALSE;
  data->deflation  = PETSC_FALSE;
  data->setup      = NULL;
  data->setup_ctx  = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_HPDDM(PC pc)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_HPDDM(pc));
  PetscCall(PetscFree(data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)pc, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetAuxiliaryMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMHasNeumannMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetRHSMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetCoarseCorrectionType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetCoarseCorrectionType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetSTShareSubKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetDeflationMat_C", NULL));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PCHPDDMSetAuxiliaryMat_Private(PC pc, IS is, Mat A, PetscBool deflation)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  if (is) {
    PetscCall(PetscObjectReference((PetscObject)is));
    if (data->is) { /* new overlap definition resets the PC */
      PetscCall(PCReset_HPDDM(pc));
      pc->setfromoptionscalled = 0;
    }
    PetscCall(ISDestroy(&data->is));
    data->is = is;
  }
  if (A) {
    PetscCall(PetscObjectReference((PetscObject)A));
    PetscCall(MatDestroy(&data->aux));
    data->aux = A;
  }
  data->deflation = deflation;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetAuxiliaryMat_HPDDM(PC pc, IS is, Mat A, PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void* setup_ctx)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCHPDDMSetAuxiliaryMat_Private(pc, is, A, PETSC_FALSE));
  if (setup) {
    data->setup = setup;
    data->setup_ctx = setup_ctx;
  }
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetAuxiliaryMat - Sets the auxiliary matrix used by PCHPDDM for the concurrent GenEO problems at the finest level. As an example, in a finite element context with nonoverlapping subdomains plus (overlapping) ghost elements, this could be the unassembled (Neumann) local overlapping operator. As opposed to the assembled (Dirichlet) local overlapping operator obtained by summing neighborhood contributions at the interface of ghost elements.

   Input Parameters:
+     pc - preconditioner context
.     is - index set of the local auxiliary, e.g., Neumann, matrix
.     A - auxiliary sequential matrix
.     setup - function for generating the auxiliary matrix
-     setup_ctx - context for setup

   Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCHPDDMSetRHSMat()`, `MATIS`
@*/
PetscErrorCode PCHPDDMSetAuxiliaryMat(PC pc, IS is, Mat A, PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void* setup_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (is) PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  if (A) PetscValidHeaderSpecific(A, MAT_CLASSID, 3);
#if defined(PETSC_HAVE_FORTRAN)
  if (reinterpret_cast<void*>(setup) == reinterpret_cast<void*>(PETSC_NULL_FUNCTION_Fortran)) setup = NULL;
  if (setup_ctx == PETSC_NULL_INTEGER_Fortran) setup_ctx = NULL;
#endif
  PetscTryMethod(pc, "PCHPDDMSetAuxiliaryMat_C", (PC, IS, Mat, PetscErrorCode (*)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void*), (pc, is, A, setup, setup_ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMHasNeumannMat_HPDDM(PC pc, PetscBool has)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  data->Neumann = has;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMHasNeumannMat - Informs PCHPDDM that the Mat passed to PCHPDDMSetAuxiliaryMat() is the local Neumann matrix. This may be used to bypass a call to MatCreateSubMatrices() and to MatConvert() for MATMPISBAIJ matrices. If a DMCreateNeumannOverlap() implementation is available in the DM attached to the Pmat, or the Amat, or the PC, the flag is internally set to PETSC_TRUE. Its default value is otherwise PETSC_FALSE.

   Input Parameters:
+     pc - preconditioner context
-     has - Boolean value

   Level: intermediate

.seealso: `PCHPDDM`, `PCHPDDMSetAuxiliaryMat()`
@*/
PetscErrorCode PCHPDDMHasNeumannMat(PC pc, PetscBool has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCHPDDMHasNeumannMat_C", (PC, PetscBool), (pc, has));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetRHSMat_HPDDM(PC pc, Mat B)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)B));
  PetscCall(MatDestroy(&data->B));
  data->B = B;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetRHSMat - Sets the right-hand side matrix used by PCHPDDM for the concurrent GenEO problems at the finest level. Must be used in conjunction with PCHPDDMSetAuxiliaryMat(N), so that Nv = lambda Bv is solved using EPSSetOperators(N, B). It is assumed that N and B are provided using the same numbering. This provides a means to try more advanced methods such as GenEO-II or H-GenEO.

   Input Parameters:
+     pc - preconditioner context
-     B - right-hand side sequential matrix

   Level: advanced

.seealso: `PCHPDDMSetAuxiliaryMat()`, `PCHPDDM`
@*/
PetscErrorCode PCHPDDMSetRHSMat(PC pc, Mat B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (B) {
    PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
    PetscTryMethod(pc, "PCHPDDMSetRHSMat_C", (PC, Mat), (pc, B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_HPDDM(PetscOptionItems *PetscOptionsObject, PC pc)
{
  PC_HPDDM                    *data = (PC_HPDDM*)pc->data;
  PC_HPDDM_Level              **levels = data->levels;
  char                        prefix[256];
  int                         i = 1;
  PetscMPIInt                 size, previous;
  PetscInt                    n;
  PCHPDDMCoarseCorrectionType type;
  PetscBool                   flg = PETSC_TRUE;

  PetscFunctionBegin;
  if (!data->levels) {
    PetscCall(PetscCalloc1(PETSC_PCHPDDM_MAXLEVELS, &levels));
    data->levels = levels;
  }
  PetscOptionsHeadBegin(PetscOptionsObject, "PCHPDDM options");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  previous = size;
  while (i < PETSC_PCHPDDM_MAXLEVELS) {
    PetscInt p = 1;

    if (!data->levels[i - 1]) PetscCall(PetscNewLog(pc, data->levels + i - 1));
    data->levels[i - 1]->parent = data;
    /* if the previous level has a single process, it is not possible to coarsen further */
    if (previous == 1 || !flg) break;
    data->levels[i - 1]->nu = 0;
    data->levels[i - 1]->threshold = -1.0;
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_nev", i));
    PetscCall(PetscOptionsInt(prefix, "Local number of deflation vectors computed by SLEPc", "EPSSetDimensions", data->levels[i - 1]->nu, &data->levels[i - 1]->nu, NULL));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold", i));
    PetscCall(PetscOptionsReal(prefix, "Local threshold for selecting deflation vectors returned by SLEPc", "PCHPDDM", data->levels[i - 1]->threshold, &data->levels[i - 1]->threshold, NULL));
    if (i == 1) {
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_1_st_share_sub_ksp"));
      PetscCall(PetscOptionsBool(prefix, "Shared KSP between SLEPc ST and the fine-level subdomain solver", "PCHPDDMGetSTShareSubKSP", PETSC_FALSE, &data->share, NULL));
    }
    /* if there is no prescribed coarsening, just break out of the loop */
    if (data->levels[i - 1]->threshold <= 0.0 && data->levels[i - 1]->nu <= 0 && !(data->deflation && i == 1)) break;
    else {
      ++i;
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_nev", i));
      PetscCall(PetscOptionsHasName(PetscOptionsObject->options, PetscOptionsObject->prefix, prefix, &flg));
      if (!flg) {
        PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold", i));
        PetscCall(PetscOptionsHasName(PetscOptionsObject->options, PetscOptionsObject->prefix, prefix, &flg));
      }
      if (flg) {
        /* if there are coarsening options for the next level, then register it  */
        /* otherwise, don't to avoid having both options levels_N_p and coarse_p */
        PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_p", i));
        PetscCall(PetscOptionsRangeInt(prefix, "Number of processes used to assemble the coarse operator at this level", "PCHPDDM", p, &p, &flg, 1, PetscMax(1, previous / 2)));
        previous = p;
      }
    }
  }
  data->N = i;
  n = 1;
  if (i > 1) {
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_coarse_p"));
    PetscCall(PetscOptionsRangeInt(prefix, "Number of processes used to assemble the coarsest operator", "PCHPDDM", n, &n, NULL, 1, PetscMax(1, previous / 2)));
#if defined(PETSC_HAVE_MUMPS)
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "pc_hpddm_coarse_"));
    PetscCall(PetscOptionsHasName(NULL, prefix, "-mat_mumps_use_omp_threads", &flg));
    if (flg) {
      char type[64]; /* same size as in src/ksp/pc/impls/factor/factimpl.c */
      if (n == 1) PetscCall(PetscStrcpy(type, MATSOLVERPETSC)); /* default solver for a sequential Mat */
      PetscCall(PetscOptionsGetString(NULL, prefix, "-pc_factor_mat_solver_type", type, sizeof(type), &flg));
      if (flg) PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "-%smat_mumps_use_omp_threads and -%spc_factor_mat_solver_type != %s", prefix, prefix, MATSOLVERMUMPS);
      size = n;
      n = -1;
      PetscCall(PetscOptionsGetInt(NULL, prefix, "-mat_mumps_use_omp_threads", &n, NULL));
      PetscCheck(n >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Need to specify a positive integer for -%smat_mumps_use_omp_threads", prefix);
      PetscCheck(n * size <= previous, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%d MPI process%s x %d OpenMP thread%s greater than %d available MPI process%s for the coarsest operator", (int)size, size > 1 ? "es" : "", (int)n, n > 1 ? "s" : "", (int)previous, previous > 1 ? "es" : "");
    }
#endif
    PetscCall(PetscOptionsEnum("-pc_hpddm_coarse_correction", "Type of coarse correction applied each iteration", "PCHPDDMSetCoarseCorrectionType", PCHPDDMCoarseCorrectionTypes, (PetscEnum)data->correction, (PetscEnum*)&type, &flg));
    if (flg) PetscCall(PCHPDDMSetCoarseCorrectionType(pc, type));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_has_neumann"));
    PetscCall(PetscOptionsBool(prefix, "Is the auxiliary Mat the local Neumann matrix?", "PCHPDDMHasNeumannMat", data->Neumann, &data->Neumann, NULL));
    data->log_separate = PETSC_FALSE;
    if (PetscDefined(USE_LOG)) {
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_log_separate"));
      PetscCall(PetscOptionsBool(prefix, "Log events level by level instead of inside PCSetUp()/KSPSolve()", NULL, data->log_separate, &data->log_separate, NULL));
    }
  }
  PetscOptionsHeadEnd();
  while (i < PETSC_PCHPDDM_MAXLEVELS && data->levels[i]) PetscCall(PetscFree(data->levels[i++]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_HPDDM(PC pc, Vec x, Vec y)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  PetscCheck(data->levels[0]->ksp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No KSP attached to PCHPDDM");
  if (data->log_separate) PetscCall(PetscLogEventBegin(PC_HPDDM_Solve[0], data->levels[0]->ksp, 0, 0, 0)); /* coarser-level events are directly triggered in HPDDM */
  PetscCall(KSPSolve(data->levels[0]->ksp, x, y));
  if (data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_Solve[0], data->levels[0]->ksp, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_HPDDM(PC pc, Mat X, Mat Y)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  PetscCheck(data->levels[0]->ksp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No KSP attached to PCHPDDM");
  PetscCall(KSPMatSolve(data->levels[0]->ksp, X, Y));
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMGetComplexities - Computes the grid and operator complexities.

   Input Parameter:
.     pc - preconditioner context

   Output Parameters:
+     gc - grid complexity = sum_i(m_i) / m_1
-     oc - operator complexity = sum_i(nnz_i) / nnz_1

   Notes:
     PCGAMG does not follow the usual convention and names the grid complexity what is usually referred to as the operator complexity. PCHPDDM follows what is found in the literature, and in particular, what you get with PCHYPRE and -pc_hypre_boomeramg_print_statistics.

   Level: advanced

.seealso: `PCMGGetGridComplexity()`, `PCHPDDM`
@*/
static PetscErrorCode PCHPDDMGetComplexities(PC pc, PetscReal *gc, PetscReal *oc)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  MatInfo        info;
  PetscInt       n, m;
  PetscLogDouble accumulate[2] { }, nnz1 = 1.0, m1 = 1.0;

  PetscFunctionBegin;
  for (n = 0, *gc = 0, *oc = 0; n < data->N; ++n) {
    if (data->levels[n]->ksp) {
      Mat P, A;
      PetscCall(KSPGetOperators(data->levels[n]->ksp, NULL, &P));
      PetscCall(MatGetSize(P, &m, NULL));
      accumulate[0] += m;
      if (n == 0) {
        PetscBool flg;
        PetscCall(PetscObjectTypeCompare((PetscObject)P, MATNORMAL, &flg));
        if (flg) {
          PetscCall(MatConvert(P, MATAIJ, MAT_INITIAL_MATRIX, &A));
          P = A;
        } else PetscCall(PetscObjectReference((PetscObject)P));
      }
      if (P->ops->getinfo) {
        PetscCall(MatGetInfo(P, MAT_GLOBAL_SUM, &info));
        accumulate[1] += info.nz_used;
      }
      if (n == 0) {
        m1 = m;
        if (P->ops->getinfo) nnz1 = info.nz_used;
        PetscCall(MatDestroy(&P));
      }
    }
  }
  *gc = static_cast<PetscReal>(accumulate[0]/m1);
  *oc = static_cast<PetscReal>(accumulate[1]/nnz1);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_HPDDM(PC pc, PetscViewer viewer)
{
  PC_HPDDM     *data = (PC_HPDDM*)pc->data;
  PetscViewer  subviewer;
  PetscSubcomm subcomm;
  PetscReal    oc, gc;
  PetscInt     i, tabs;
  PetscMPIInt  size, color, rank;
  PetscBool    ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &ascii));
  if (ascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "level%s: %" PetscInt_FMT "\n", data->N > 1 ? "s" : "", data->N));
    PetscCall(PCHPDDMGetComplexities(pc, &gc, &oc));
    if (data->N > 1) {
      if (!data->deflation) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Neumann matrix attached? %s\n", PetscBools[data->Neumann]));
        PetscCall(PetscViewerASCIIPrintf(viewer, "shared subdomain KSP between SLEPc and PETSc? %s\n", PetscBools[data->share]));
      } else PetscCall(PetscViewerASCIIPrintf(viewer, "user-supplied deflation matrix\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "coarse correction: %s\n", PCHPDDMCoarseCorrectionTypes[data->correction]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "on process #0, value%s (+ threshold%s if available) for selecting deflation vectors:", data->N > 2 ? "s" : "", data->N > 2 ? "s" : ""));
      PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
      PetscCall(PetscViewerASCIISetTab(viewer, 0));
      for (i = 1; i < data->N; ++i) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, data->levels[i - 1]->nu));
        if (data->levels[i - 1]->threshold > -0.1) PetscCall(PetscViewerASCIIPrintf(viewer, " (%g)", (double)data->levels[i - 1]->threshold));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIISetTab(viewer, tabs));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "grid and operator complexities: %g %g\n", (double)gc, (double)oc));
    if (data->levels[0]->ksp) {
      PetscCall(KSPView(data->levels[0]->ksp, viewer));
      if (data->levels[0]->pc) PetscCall(PCView(data->levels[0]->pc, viewer));
      for (i = 1; i < data->N; ++i) {
        if (data->levels[i]->ksp) color = 1;
        else color = 0;
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
        PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
        PetscCall(PetscSubcommCreate(PetscObjectComm((PetscObject)pc), &subcomm));
        PetscCall(PetscSubcommSetNumber(subcomm, PetscMin(size, 2)));
        PetscCall(PetscSubcommSetTypeGeneral(subcomm, color, rank));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerGetSubViewer(viewer, PetscSubcommChild(subcomm), &subviewer));
        if (color == 1) {
          PetscCall(KSPView(data->levels[i]->ksp, subviewer));
          if (data->levels[i]->pc) PetscCall(PCView(data->levels[i]->pc, subviewer));
          PetscCall(PetscViewerFlush(subviewer));
        }
        PetscCall(PetscViewerRestoreSubViewer(viewer, PetscSubcommChild(subcomm), &subviewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscCall(PetscSubcommDestroy(&subcomm));
        PetscCall(PetscViewerFlush(viewer));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolve_HPDDM(PC pc, KSP ksp, Vec, Vec)
{
  PC_HPDDM  *data = (PC_HPDDM*)pc->data;
  PetscBool flg;
  Mat       A;

  PetscFunctionBegin;
  if (ksp) {
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPLSQR, &flg));
    if (flg && !data->normal) {
      PetscCall(KSPGetOperators(ksp, &A, NULL));
      PetscCall(MatCreateVecs(A, NULL, &data->normal)); /* temporary Vec used in PCHPDDMShellApply() for coarse grid corrections */
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMShellSetUp(PC pc)
{
  PC_HPDDM_Level *ctx;
  Mat            A, P;
  Vec            x;
  const char     *pcpre;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(KSPGetOptionsPrefix(ctx->ksp, &pcpre));
  PetscCall(KSPGetOperators(ctx->ksp, &A, &P));
  /* smoother */
  PetscCall(PCSetOptionsPrefix(ctx->pc, pcpre));
  PetscCall(PCSetOperators(ctx->pc, A, P));
  if (!ctx->v[0]) {
    PetscCall(VecDuplicateVecs(ctx->D, 1, &ctx->v[0]));
    if (!std::is_same<PetscScalar, PetscReal>::value) PetscCall(VecDestroy(&ctx->D));
    PetscCall(MatCreateVecs(A, &x, NULL));
    PetscCall(VecDuplicateVecs(x, 2, &ctx->v[1]));
    PetscCall(VecDestroy(&x));
  }
  std::fill_n(ctx->V, 3, nullptr);
  PetscFunctionReturn(0);
}

template<class Type, typename std::enable_if<std::is_same<Type, Vec>::value>::type* = nullptr>
static inline PetscErrorCode PCHPDDMDeflate_Private(PC pc, Type x, Type y)
{
  PC_HPDDM_Level *ctx;
  PetscScalar    *out;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  /* going from PETSc to HPDDM numbering */
  PetscCall(VecScatterBegin(ctx->scatter, x, ctx->v[0][0], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatter, x, ctx->v[0][0], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayWrite(ctx->v[0][0], &out));
  ctx->P->deflation<false>(NULL, out, 1); /* y = Q x */
  PetscCall(VecRestoreArrayWrite(ctx->v[0][0], &out));
  /* going from HPDDM to PETSc numbering */
  PetscCall(VecScatterBegin(ctx->scatter, ctx->v[0][0], y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatter, ctx->v[0][0], y, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

template<class Type, typename std::enable_if<std::is_same<Type, Mat>::value>::type* = nullptr>
static inline PetscErrorCode PCHPDDMDeflate_Private(PC pc, Type X, Type Y)
{
  PC_HPDDM_Level *ctx;
  Vec            vX, vY, vC;
  PetscScalar    *out;
  PetscInt       i, N;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MatGetSize(X, NULL, &N));
  /* going from PETSc to HPDDM numbering */
  for (i = 0; i < N; ++i) {
    PetscCall(MatDenseGetColumnVecRead(X, i, &vX));
    PetscCall(MatDenseGetColumnVecWrite(ctx->V[0], i, &vC));
    PetscCall(VecScatterBegin(ctx->scatter, vX, vC, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx->scatter, vX, vC, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(MatDenseRestoreColumnVecWrite(ctx->V[0], i, &vC));
    PetscCall(MatDenseRestoreColumnVecRead(X, i, &vX));
  }
  PetscCall(MatDenseGetArrayWrite(ctx->V[0], &out));
  ctx->P->deflation<false>(NULL, out, N); /* Y = Q X */
  PetscCall(MatDenseRestoreArrayWrite(ctx->V[0], &out));
  /* going from HPDDM to PETSc numbering */
  for (i = 0; i < N; ++i) {
    PetscCall(MatDenseGetColumnVecRead(ctx->V[0], i, &vC));
    PetscCall(MatDenseGetColumnVecWrite(Y, i, &vY));
    PetscCall(VecScatterBegin(ctx->scatter, vC, vY, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ctx->scatter, vC, vY, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &vY));
    PetscCall(MatDenseRestoreColumnVecRead(ctx->V[0], i, &vC));
  }
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMShellApply - Applies a (2) deflated, (1) additive, or (3) balanced coarse correction. In what follows, E = Z Pmat Z^T and Q = Z^T E^-1 Z.

.vb
   (1) y =                Pmat^-1              x + Q x,
   (2) y =                Pmat^-1 (I - Amat Q) x + Q x (default),
   (3) y = (I - Q Amat^T) Pmat^-1 (I - Amat Q) x + Q x.
.ve

   Input Parameters:
+     pc - preconditioner context
-     x - input vector

   Output Parameter:
.     y - output vector

   Application Interface Routine: PCApply()

   Notes:
     The options of Pmat^1 = pc(Pmat) are prefixed by -pc_hpddm_levels_1_pc_. Z is a tall-and-skiny matrix assembled by HPDDM. The number of processes on which (Z Pmat Z^T) is aggregated is set via -pc_hpddm_coarse_p.
     The options of (Z Pmat Z^T)^-1 = ksp(Z Pmat Z^T) are prefixed by -pc_hpddm_coarse_ (KSPPREONLY and PCCHOLESKY by default), unless a multilevel correction is turned on, in which case, this function is called recursively at each level except the coarsest one.
     (1) and (2) visit the "next" level (in terms of coarsening) once per application, while (3) visits it twice, so it is asymptotically twice costlier. (2) is not symmetric even if both Amat and Pmat are symmetric.

   Level: advanced

.seealso: `PCHPDDM`, `PCHPDDMCoarseCorrectionType`
@*/
static PetscErrorCode PCHPDDMShellApply(PC pc, Vec x, Vec y)
{
  PC_HPDDM_Level *ctx;
  Mat            A;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object");
  PetscCall(KSPGetOperators(ctx->ksp, &A, NULL));
  PetscCall(PCHPDDMDeflate_Private(pc, x, y));                    /* y = Q x                          */
  if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
    if (!ctx->parent->normal || ctx != ctx->parent->levels[0]) PetscCall(MatMult(A, y, ctx->v[1][0])); /* y = A Q x */
    else { /* KSPLSQR and finest level */
      PetscCall(MatMult(A, y, ctx->parent->normal));              /* y = A Q x                        */
      PetscCall(MatMultTranspose(A, ctx->parent->normal, ctx->v[1][0])); /* y = A^T A Q x             */
    }
    PetscCall(VecWAXPY(ctx->v[1][1], -1.0, ctx->v[1][0], x));     /* y = (I - A Q) x                  */
    PetscCall(PCApply(ctx->pc, ctx->v[1][1], ctx->v[1][0]));      /* y = M^-1 (I - A Q) x             */
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      if (!ctx->parent->normal || ctx != ctx->parent->levels[0]) PetscCall(MatMultTranspose(A, ctx->v[1][0], ctx->v[1][1])); /* z = A^T y */
      else {
        PetscCall(MatMult(A, ctx->v[1][0], ctx->parent->normal));
        PetscCall(MatMultTranspose(A, ctx->parent->normal, ctx->v[1][1])); /* z = A^T A y             */
      }
      PetscCall(PCHPDDMDeflate_Private(pc, ctx->v[1][1], ctx->v[1][1]));
      PetscCall(VecAXPBYPCZ(y, -1.0, 1.0, 1.0, ctx->v[1][1], ctx->v[1][0])); /* y = (I - Q A^T) y + Q x */
    } else PetscCall(VecAXPY(y, 1.0, ctx->v[1][0]));              /* y = Q M^-1 (I - A Q) x + Q x     */
  } else {
    PetscCheck(ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction);
    PetscCall(PCApply(ctx->pc, x, ctx->v[1][0]));
    PetscCall(VecAXPY(y, 1.0, ctx->v[1][0]));                     /* y = M^-1 x + Q x                 */
  }
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMShellMatApply - Variant of PCHPDDMShellApply() for blocks of vectors.

   Input Parameters:
+     pc - preconditioner context
-     X - block of input vectors

   Output Parameter:
.     Y - block of output vectors

   Application Interface Routine: PCApply()

   Level: advanced

.seealso: `PCHPDDM`, `PCHPDDMShellMatApply()`, `PCHPDDMCoarseCorrectionType`
@*/
static PetscErrorCode PCHPDDMShellMatApply(PC pc, Mat X, Mat Y)
{
  PC_HPDDM_Level *ctx;
  Mat            A, *ptr;
  PetscContainer container = NULL;
  PetscScalar    *array;
  PetscInt       m, M, N, prev = 0;
  PetscBool      reset = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object");
  PetscCall(MatGetSize(X, NULL, &N));
  PetscCall(KSPGetOperators(ctx->ksp, &A, NULL));
  PetscCall(PetscObjectQuery((PetscObject)A, "_HPDDM_MatProduct", (PetscObject*)&container));
  if (container) { /* MatProduct container already attached */
    PetscCall(PetscContainerGetPointer(container, (void**)&ptr));
    if (ptr[1] != ctx->V[2]) /* Mat has changed or may have been set first in KSPHPDDM */
      for (m = 0; m < 2; ++m) {
        PetscCall(MatDestroy(ctx->V + m + 1));
        ctx->V[m + 1] = ptr[m];
        PetscCall(PetscObjectReference((PetscObject)ctx->V[m + 1]));
      }
  }
  if (ctx->V[1]) PetscCall(MatGetSize(ctx->V[1], NULL, &prev));
  if (N != prev || !ctx->V[0]) {
    PetscCall(MatDestroy(ctx->V));
    PetscCall(VecGetLocalSize(ctx->v[0][0], &m));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)pc), m, PETSC_DECIDE, PETSC_DECIDE, N, NULL, ctx->V));
    if (N != prev) {
      PetscCall(MatDestroy(ctx->V + 1));
      PetscCall(MatDestroy(ctx->V + 2));
      PetscCall(MatGetLocalSize(X, &m, NULL));
      PetscCall(MatGetSize(X, &M, NULL));
      if (ctx->parent->correction != PC_HPDDM_COARSE_CORRECTION_BALANCED) PetscCall(MatDenseGetArrayWrite(ctx->V[0], &array));
      else array = NULL;
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)pc), m, PETSC_DECIDE, M, N, array, ctx->V + 1));
      if (ctx->parent->correction != PC_HPDDM_COARSE_CORRECTION_BALANCED) PetscCall(MatDenseRestoreArrayWrite(ctx->V[0], &array));
      else {
        PetscCall(MatAssemblyBegin(ctx->V[1], MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(ctx->V[1], MAT_FINAL_ASSEMBLY));
      }
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)pc), m, PETSC_DECIDE, M, N, NULL, ctx->V + 2));
      PetscCall(MatAssemblyBegin(ctx->V[2], MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(ctx->V[2], MAT_FINAL_ASSEMBLY));
      PetscCall(MatProductCreateWithMat(A, Y, NULL, ctx->V[1]));
      PetscCall(MatProductSetType(ctx->V[1], MATPRODUCT_AB));
      PetscCall(MatProductSetFromOptions(ctx->V[1]));
      PetscCall(MatProductSymbolic(ctx->V[1]));
      if (!container) { /* no MatProduct container attached, create one to be queried in KSPHPDDM or at the next call to PCMatApply() */
        PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)A), &container));
        PetscCall(PetscObjectCompose((PetscObject)A, "_HPDDM_MatProduct", (PetscObject)container));
      }
      PetscCall(PetscContainerSetPointer(container, ctx->V + 1)); /* need to compose B and D from MatProductCreateWithMath(A, B, NULL, D), which are stored in the contiguous array ctx->V */
    }
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      PetscCall(MatProductCreateWithMat(A, ctx->V[1], NULL, ctx->V[2]));
      PetscCall(MatProductSetType(ctx->V[2], MATPRODUCT_AtB));
      PetscCall(MatProductSetFromOptions(ctx->V[2]));
      PetscCall(MatProductSymbolic(ctx->V[2]));
    }
    ctx->P->start(N);
  }
  if (N == prev || container) { /* when MatProduct container is attached, always need to MatProductReplaceMats() since KSPHPDDM may have replaced the Mat as well */
    PetscCall(MatProductReplaceMats(NULL, Y, NULL, ctx->V[1]));
    if (container && ctx->parent->correction != PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      PetscCall(MatDenseGetArrayWrite(ctx->V[0], &array));
      PetscCall(MatDensePlaceArray(ctx->V[1], array));
      PetscCall(MatDenseRestoreArrayWrite(ctx->V[0], &array));
      reset = PETSC_TRUE;
    }
  }
  PetscCall(PCHPDDMDeflate_Private(pc, X, Y));
  if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
    PetscCall(MatProductNumeric(ctx->V[1]));
    PetscCall(MatCopy(ctx->V[1], ctx->V[2], SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(ctx->V[2], -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(PCMatApply(ctx->pc, ctx->V[2], ctx->V[1]));
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      PetscCall(MatProductNumeric(ctx->V[2]));
      PetscCall(PCHPDDMDeflate_Private(pc, ctx->V[2], ctx->V[2]));
      PetscCall(MatAXPY(ctx->V[1], -1.0, ctx->V[2], SAME_NONZERO_PATTERN));
    }
    PetscCall(MatAXPY(Y, -1.0, ctx->V[1], SAME_NONZERO_PATTERN));
  } else {
    PetscCheck(ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction);
    PetscCall(PCMatApply(ctx->pc, X, ctx->V[1]));
    PetscCall(MatAXPY(Y, 1.0, ctx->V[1], SAME_NONZERO_PATTERN));
  }
  if (reset) PetscCall(MatDenseResetArray(ctx->V[1]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMShellDestroy(PC pc)
{
  PC_HPDDM_Level *ctx;
  PetscContainer container;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(HPDDM::Schwarz<PetscScalar>::destroy(ctx, PETSC_TRUE));
  PetscCall(VecDestroyVecs(1, &ctx->v[0]));
  PetscCall(VecDestroyVecs(2, &ctx->v[1]));
  PetscCall(PetscObjectQuery((PetscObject)(ctx->pc)->mat, "_HPDDM_MatProduct", (PetscObject*)&container));
  PetscCall(PetscContainerDestroy(&container));
  PetscCall(PetscObjectCompose((PetscObject)(ctx->pc)->mat, "_HPDDM_MatProduct", NULL));
  PetscCall(MatDestroy(ctx->V));
  PetscCall(MatDestroy(ctx->V + 1));
  PetscCall(MatDestroy(ctx->V + 2));
  PetscCall(VecDestroy(&ctx->D));
  PetscCall(VecScatterDestroy(&ctx->scatter));
  PetscCall(PCDestroy(&ctx->pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSolve_Private(const PC_HPDDM_Level *ctx, PetscScalar *rhs, const unsigned short& mu)
{
  Mat      B, X;
  PetscInt n, N, j = 0;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ctx->ksp, &B, NULL));
  PetscCall(MatGetLocalSize(B, &n, NULL));
  PetscCall(MatGetSize(B, &N, NULL));
  if (ctx->parent->log_separate) {
    j = std::distance(ctx->parent->levels, std::find(ctx->parent->levels, ctx->parent->levels + ctx->parent->N, ctx));
    PetscCall(PetscLogEventBegin(PC_HPDDM_Solve[j], ctx->ksp, 0, 0, 0));
  }
  if (mu == 1) {
    if (!ctx->ksp->vec_rhs) {
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ctx->ksp), 1, n, N, NULL, &ctx->ksp->vec_rhs));
      PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ctx->ksp), n, N, &ctx->ksp->vec_sol));
    }
    PetscCall(VecPlaceArray(ctx->ksp->vec_rhs, rhs));
    PetscCall(KSPSolve(ctx->ksp, NULL, NULL));
    PetscCall(VecCopy(ctx->ksp->vec_sol, ctx->ksp->vec_rhs));
    PetscCall(VecResetArray(ctx->ksp->vec_rhs));
  } else {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ctx->ksp), n, PETSC_DECIDE, N, mu, rhs, &B));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ctx->ksp), n, PETSC_DECIDE, N, mu, NULL, &X));
    PetscCall(KSPMatSolve(ctx->ksp, B, X));
    PetscCall(MatCopy(X, B, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&B));
  }
  if (ctx->parent->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_Solve[j], ctx->ksp, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetUpNeumannOverlap_Private(PC pc)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  if (data->setup) {
    Mat       P;
    Vec       x, xt = NULL;
    PetscReal t = 0.0, s = 0.0;

    PetscCall(PCGetOperators(pc, NULL, &P));
    PetscCall(PetscObjectQuery((PetscObject)P, "__SNES_latest_X", (PetscObject*)&x));
    PetscStackPush("PCHPDDM Neumann callback");
    PetscCall((*data->setup)(data->aux, t, x, xt, s, data->is, data->setup_ctx));
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMCreateSubMatrices_Private(Mat mat, PetscInt n, const IS*, const IS*, MatReuse scall, Mat *submat[])
{
  Mat A;

  PetscFunctionBegin;
  PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "MatCreateSubMatrices() called to extract %" PetscInt_FMT " submatrices, which is different than 1", n);
  /* previously composed Mat */
  PetscCall(PetscObjectQuery((PetscObject)mat, "_PCHPDDM_SubMatrices", (PetscObject*)&A));
  PetscCheck(A, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SubMatrices not found in Mat");
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscCalloc1(1, submat));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, *submat));
  } else PetscCall(MatCopy(A, (*submat)[0], SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMCommunicationAvoidingPCASM_Private(PC pc, Mat C, PetscBool sorted)
{
  void (*op)(void);

  PetscFunctionBegin;
  /* previously-composed Mat */
  PetscCall(PetscObjectCompose((PetscObject)pc->pmat, "_PCHPDDM_SubMatrices", (PetscObject)C));
  PetscCall(MatGetOperation(pc->pmat, MATOP_CREATE_SUBMATRICES, &op));
  /* trick suggested by Barry https://lists.mcs.anl.gov/pipermail/petsc-dev/2020-January/025491.html */
  PetscCall(MatSetOperation(pc->pmat, MATOP_CREATE_SUBMATRICES, (void(*)(void))PCHPDDMCreateSubMatrices_Private));
  if (sorted) PetscCall(PCASMSetSortIndices(pc, PETSC_FALSE)); /* everything is already sorted */
  PetscCall(PCSetFromOptions(pc)); /* otherwise -pc_hpddm_levels_1_pc_asm_sub_mat_type is not used */
  PetscCall(PCSetUp(pc));
  /* reset MatCreateSubMatrices() */
  PetscCall(MatSetOperation(pc->pmat, MATOP_CREATE_SUBMATRICES, op));
  PetscCall(PetscObjectCompose((PetscObject)pc->pmat, "_PCHPDDM_SubMatrices", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMPermute_Private(IS is, IS in_is, IS *out_is, Mat in_C, Mat *out_C, IS *p)
{
  IS                           perm;
  const PetscInt               *ptr;
  PetscInt                     *concatenate, size, n;
  std::map<PetscInt, PetscInt> order;
  PetscBool                    sorted;

  PetscFunctionBegin;
  PetscCall(ISSorted(is, &sorted));
  if (!sorted) {
    PetscCall(ISGetLocalSize(is, &size));
    PetscCall(ISGetIndices(is, &ptr));
    /* MatCreateSubMatrices(), called by PCASM, follows the global numbering of Pmat */
    for (n = 0; n < size; ++n) order.insert(std::make_pair(ptr[n], n));
    PetscCall(ISRestoreIndices(is, &ptr));
    if (out_C) {
      PetscCall(PetscMalloc1(size, &concatenate));
      for (const std::pair<const PetscInt, PetscInt>& i : order) *concatenate++ = i.second;
      concatenate -= size;
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)in_C), size, concatenate, PETSC_OWN_POINTER, &perm));
      PetscCall(ISSetPermutation(perm));
      /* permute user-provided Mat so that it matches with MatCreateSubMatrices() numbering */
      PetscCall(MatPermute(in_C, perm, perm, out_C));
      if (p) *p = perm;
      else PetscCall(ISDestroy(&perm)); /* no need to save the permutation */
    }
    if (out_is) {
      PetscCall(PetscMalloc1(size, &concatenate));
      for (const std::pair<const PetscInt, PetscInt>& i : order) *concatenate++ = i.first;
      concatenate -= size;
      /* permute user-provided IS so that it matches with MatCreateSubMatrices() numbering */
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)in_is), size, concatenate, PETSC_OWN_POINTER, out_is));
    }
  } else { /* input IS is sorted, nothing to permute, simply duplicate inputs when needed */
    if (out_C) PetscCall(MatDuplicate(in_C, MAT_COPY_VALUES, out_C));
    if (out_is) PetscCall(ISDuplicate(in_is, out_is));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMDestroySubMatrices_Private(PetscBool flg, PetscBool algebraic, Mat *sub)
{
  IS is;

  PetscFunctionBegin;
  if (!flg) {
    if (algebraic) {
      PetscCall(PetscObjectQuery((PetscObject)sub[0], "_PCHPDDM_Embed", (PetscObject*)&is));
      PetscCall(ISDestroy(&is));
      PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Embed", NULL));
      PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Compact", NULL));
    }
    PetscCall(MatDestroySubMatrices(algebraic ? 2 : 1, &sub));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMAlgebraicAuxiliaryMat_Private(Mat P, IS *is, Mat *sub[], PetscBool block)
{
  IS        icol[3], irow[2];
  Mat       *M, Q;
  PetscReal *ptr;
  PetscInt  *idx, p = 0, n;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, P->cmap->N, 0, 1, icol + 2));
  PetscCall(ISSetBlockSize(icol[2], P->cmap->bs));
  PetscCall(ISSetIdentity(icol[2]));
  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATMPISBAIJ, &flg));
  if (flg) {
    /* MatCreateSubMatrices() does not handle MATMPISBAIJ properly when iscol != isrow, so convert first to MATMPIBAIJ */
    PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &Q));
    std::swap(P, Q);
  }
  PetscCall(MatCreateSubMatrices(P, 1, is, icol + 2, MAT_INITIAL_MATRIX, &M));
  if (flg) {
    std::swap(P, Q);
    PetscCall(MatDestroy(&Q));
  }
  PetscCall(ISDestroy(icol + 2));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, M[0]->rmap->N, 0, 1, irow));
  PetscCall(ISSetBlockSize(irow[0], P->cmap->bs));
  PetscCall(ISSetIdentity(irow[0]));
  if (!block) {
    PetscCall(PetscMalloc2(P->cmap->N, &ptr, P->cmap->N, &idx));
    PetscCall(MatGetColumnNorms(M[0], NORM_INFINITY, ptr));
    /* check for nonzero columns so that M[0] may be expressed in compact form */
    for (n = 0; n < P->cmap->N; n += P->cmap->bs)
      if (std::find_if(ptr + n, ptr + n + P->cmap->bs, [](PetscReal v) { return v > PETSC_MACHINE_EPSILON; }) != ptr + n + P->cmap->bs) {
        std::iota(idx + p, idx + p + P->cmap->bs, n);
        p += P->cmap->bs;
      }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, p, idx, PETSC_USE_POINTER, icol + 1));
    PetscCall(ISSetBlockSize(icol[1], P->cmap->bs));
    PetscCall(ISSetInfo(icol[1], IS_SORTED, IS_GLOBAL, PETSC_TRUE, PETSC_TRUE));
    PetscCall(ISEmbed(*is, icol[1], PETSC_FALSE, icol + 2));
    irow[1] = irow[0];
    /* first Mat will be used in PCASM (if it is used as a PC on this level) and as the left-hand side of GenEO */
    icol[0] = is[0];
    PetscCall(MatCreateSubMatrices(M[0], 2, irow, icol, MAT_INITIAL_MATRIX, sub));
    PetscCall(ISDestroy(icol + 1));
    PetscCall(PetscFree2(ptr, idx));
    /* IS used to go back and forth between the augmented and the original local linear system, see eq. (3.4) of [2022b] */
    PetscCall(PetscObjectCompose((PetscObject)(*sub)[0], "_PCHPDDM_Embed", (PetscObject)icol[2]));
    /* Mat used in eq. (3.1) of [2022b] */
    PetscCall(PetscObjectCompose((PetscObject)(*sub)[0], "_PCHPDDM_Compact", (PetscObject)(*sub)[1]));
  } else {
    Mat aux;
    PetscCall(MatSetOption(M[0], MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
    /* diagonal block of the overlapping rows */
    PetscCall(MatCreateSubMatrices(M[0], 1, irow, is, MAT_INITIAL_MATRIX, sub));
    PetscCall(MatDuplicate((*sub)[0], MAT_COPY_VALUES, &aux));
    PetscCall(MatSetOption(aux, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    if (P->cmap->bs == 1) { /* scalar case */
      Vec sum[2];
      PetscCall(MatCreateVecs(aux, sum, sum + 1));
      PetscCall(MatGetRowSum(M[0], sum[0]));
      PetscCall(MatGetRowSum(aux, sum[1]));
      /* off-diagonal block row sum (full rows - diagonal block rows) */
      PetscCall(VecAXPY(sum[0], -1.0, sum[1]));
      /* subdomain matrix plus off-diagonal block row sum */
      PetscCall(MatDiagonalSet(aux, sum[0], ADD_VALUES));
      PetscCall(VecDestroy(sum));
      PetscCall(VecDestroy(sum + 1));
    } else { /* vectorial case */
      /* TODO: missing MatGetValuesBlocked(), so the code below is     */
      /* an extension of the scalar case for when bs > 1, but it could */
      /* be more efficient by avoiding all these MatMatMult()          */
      Mat         sum[2], ones;
      PetscScalar *ptr;
      PetscCall(PetscCalloc1(M[0]->cmap->n * P->cmap->bs, &ptr));
      PetscCall(MatCreateDense(PETSC_COMM_SELF, M[0]->cmap->n, P->cmap->bs, M[0]->cmap->n, P->cmap->bs, ptr, &ones));
      for (n = 0; n < M[0]->cmap->n; n += P->cmap->bs) {
        for (p = 0; p < P->cmap->bs; ++p) ptr[n + p * (M[0]->cmap->n + 1)] = 1.0;
      }
      PetscCall(MatMatMult(M[0], ones, MAT_INITIAL_MATRIX, PETSC_DEFAULT, sum));
      PetscCall(MatDestroy(&ones));
      PetscCall(MatCreateDense(PETSC_COMM_SELF, aux->cmap->n, P->cmap->bs, aux->cmap->n, P->cmap->bs, ptr, &ones));
      PetscCall(MatDenseSetLDA(ones, M[0]->cmap->n));
      PetscCall(MatMatMult(aux, ones, MAT_INITIAL_MATRIX, PETSC_DEFAULT, sum + 1));
      PetscCall(MatDestroy(&ones));
      PetscCall(PetscFree(ptr));
      /* off-diagonal block row sum (full rows - diagonal block rows) */
      PetscCall(MatAXPY(sum[0], -1.0, sum[1], SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(sum + 1));
      /* re-order values to be consistent with MatSetValuesBlocked()           */
      /* equivalent to MatTranspose() which does not truly handle              */
      /* MAT_INPLACE_MATRIX in the rectangular case, as it calls PetscMalloc() */
      PetscCall(MatDenseGetArrayWrite(sum[0], &ptr));
      HPDDM::Wrapper<PetscScalar>::imatcopy<'T'>(P->cmap->bs, sum[0]->rmap->n, ptr, sum[0]->rmap->n, P->cmap->bs);
      /* subdomain matrix plus off-diagonal block row sum */
      for (n = 0; n < aux->cmap->n / P->cmap->bs; ++n) PetscCall(MatSetValuesBlocked(aux, 1, &n, 1, &n, ptr + n * P->cmap->bs * P->cmap->bs, ADD_VALUES));
      PetscCall(MatAssemblyBegin(aux, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(aux, MAT_FINAL_ASSEMBLY));
      PetscCall(MatDenseRestoreArrayWrite(sum[0], &ptr));
      PetscCall(MatDestroy(sum));
    }
    PetscCall(MatSetOption(aux, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    /* left-hand side of GenEO, with the same sparsity pattern as PCASM subdomain solvers  */
    PetscCall(PetscObjectCompose((PetscObject)(*sub)[0], "_PCHPDDM_Neumann_Mat", (PetscObject)aux));
  }
  PetscCall(ISDestroy(irow));
  PetscCall(MatDestroySubMatrices(1, &M));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_HPDDM(PC pc)
{
  PC_HPDDM                 *data = (PC_HPDDM*)pc->data;
  PC                       inner;
  KSP                      *ksp;
  Mat                      *sub, A, P, N, C = NULL, uaux = NULL, weighted, subA[2];
  Vec                      xin, v;
  std::vector<Vec>         initial;
  IS                       is[1], loc, uis = data->is;
  ISLocalToGlobalMapping   l2g;
  char                     prefix[256];
  const char               *pcpre;
  const PetscScalar *const *ev;
  PetscInt                 n, requested = data->N, reused = 0;
  MatStructure             structure = UNKNOWN_NONZERO_PATTERN;
  PetscBool                subdomains = PETSC_FALSE, flg = PETSC_FALSE, ismatis, swap = PETSC_FALSE, algebraic = PETSC_FALSE, block = PETSC_FALSE;
  DM                       dm;

  PetscFunctionBegin;
  PetscCheck(data->levels && data->levels[0], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not a single level allocated");
  PetscCall(PCGetOptionsPrefix(pc, &pcpre));
  PetscCall(PCGetOperators(pc, &A, &P));
  if (!data->levels[0]->ksp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->ksp));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_%s_", pcpre ? pcpre : "", data->N > 1 ? "levels_1" : "coarse"));
    PetscCall(KSPSetOptionsPrefix(data->levels[0]->ksp, prefix));
    PetscCall(KSPSetType(data->levels[0]->ksp, KSPPREONLY));
  } else if (data->levels[0]->ksp->pc && data->levels[0]->ksp->pc->setupcalled == 1 && data->levels[0]->ksp->pc->reusepreconditioner) {
    /* if the fine-level PCSHELL exists, its setup has succeeded, and one wants to reuse it, */
    /* then just propagate the appropriate flag to the coarser levels                        */
    for (n = 0; n < PETSC_PCHPDDM_MAXLEVELS && data->levels[n]; ++n) {
      /* the following KSP and PC may be NULL for some processes, hence the check            */
      if (data->levels[n]->ksp) PetscCall(KSPSetReusePreconditioner(data->levels[n]->ksp, PETSC_TRUE));
      if (data->levels[n]->pc) PetscCall(PCSetReusePreconditioner(data->levels[n]->pc, PETSC_TRUE));
    }
    /* early bail out because there is nothing to do */
    PetscFunctionReturn(0);
  } else {
    /* reset coarser levels */
    for (n = 1; n < PETSC_PCHPDDM_MAXLEVELS && data->levels[n]; ++n) {
      if (data->levels[n]->ksp && data->levels[n]->ksp->pc && data->levels[n]->ksp->pc->setupcalled == 1 && data->levels[n]->ksp->pc->reusepreconditioner && n < data->N) {
        reused = data->N - n;
        break;
      }
      PetscCall(KSPDestroy(&data->levels[n]->ksp));
      PetscCall(PCDestroy(&data->levels[n]->pc));
    }
    /* check if some coarser levels are being reused */
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &reused, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)pc)));
    const int *addr = data->levels[0]->P ? data->levels[0]->P->getAddrLocal() : &HPDDM::i__0;

    if (addr != &HPDDM::i__0 && reused != data->N - 1) {
      /* reuse previously computed eigenvectors */
      ev = data->levels[0]->P->getVectors();
      if (ev) {
        initial.reserve(*addr);
        PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, data->levels[0]->P->getDof(), ev[0], &xin));
        for (n = 0; n < *addr; ++n) {
          PetscCall(VecDuplicate(xin, &v));
          PetscCall(VecPlaceArray(xin, ev[n]));
          PetscCall(VecCopy(xin, v));
          initial.emplace_back(v);
          PetscCall(VecResetArray(xin));
        }
        PetscCall(VecDestroy(&xin));
      }
    }
  }
  data->N -= reused;
  PetscCall(KSPSetOperators(data->levels[0]->ksp, A, P));

  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATIS, &ismatis));
  if (!data->is && !ismatis) {
    PetscErrorCode (*create)(DM, IS*, Mat*, PetscErrorCode (**)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void**) = NULL;
    PetscErrorCode (*usetup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*) = NULL;
    void           *uctx = NULL;

    /* first see if we can get the data from the DM */
    PetscCall(MatGetDM(P, &dm));
    if (!dm) PetscCall(MatGetDM(A, &dm));
    if (!dm) PetscCall(PCGetDM(pc, &dm));
    if (dm) { /* this is the hook for DMPLEX and DMDA for which the auxiliary Mat is the local Neumann matrix */
      PetscCall(PetscObjectQueryFunction((PetscObject)dm, "DMCreateNeumannOverlap_C", &create));
      if (create) {
        PetscCall((*create)(dm, &uis, &uaux, &usetup, &uctx));
        data->Neumann = PETSC_TRUE;
      }
    }
    if (!create) {
      if (!uis) {
        PetscCall(PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Neumann_IS", (PetscObject*)&uis));
        PetscCall(PetscObjectReference((PetscObject)uis));
      }
      if (!uaux) {
        PetscCall(PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Neumann_Mat", (PetscObject*)&uaux));
        PetscCall(PetscObjectReference((PetscObject)uaux));
      }
      /* look inside the Pmat instead of the PC, needed for MatSchurComplementComputeExplicitOperator() */
      if (!uis) {
        PetscCall(PetscObjectQuery((PetscObject)P, "_PCHPDDM_Neumann_IS", (PetscObject*)&uis));
        PetscCall(PetscObjectReference((PetscObject)uis));
      }
      if (!uaux) {
        PetscCall(PetscObjectQuery((PetscObject)P, "_PCHPDDM_Neumann_Mat", (PetscObject*)&uaux));
        PetscCall(PetscObjectReference((PetscObject)uaux));
      }
    }
    PetscCall(PCHPDDMSetAuxiliaryMat(pc, uis, uaux, usetup, uctx));
    PetscCall(MatDestroy(&uaux));
    PetscCall(ISDestroy(&uis));
  }

  if (!ismatis) {
    PetscCall(PCHPDDMSetUpNeumannOverlap_Private(pc));
    if (!data->is && data->N > 1) {
      char type[256] = { }; /* same size as in src/ksp/pc/interface/pcset.c */
      PetscCall(PetscOptionsGetString(NULL, pcpre, "-pc_hpddm_levels_1_st_pc_type", type, sizeof(type), NULL));
      PetscCall(PetscStrcmp(type, PCMAT, &algebraic));
      PetscCall(PetscOptionsGetBool(NULL, pcpre, "-pc_hpddm_block_splitting", &block, NULL));
      PetscCheck(!algebraic || !block, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "-pc_hpddm_levels_1_st_pc_type mat and -pc_hpddm_block_splitting");
      if (block) algebraic = PETSC_TRUE;
      if (algebraic) {
        PetscCall(ISCreateStride(PETSC_COMM_SELF, P->rmap->n, P->rmap->rstart, 1, &data->is));
        PetscCall(MatIncreaseOverlap(P, 1, &data->is, 1));
        PetscCall(ISSort(data->is));
      } else PetscCall(PetscInfo(pc, "Cannot assemble a fully-algebraic coarse operator with an assembled Pmat and -%spc_hpddm_levels_1_st_pc_type != mat and -%spc_hpddm_block_splitting != true\n", pcpre ? pcpre : "", pcpre ? pcpre : ""));
    }
  }

  if (data->is || (ismatis && data->N > 1)) {
    if (ismatis) {
      std::initializer_list<std::string> list = { MATSEQBAIJ, MATSEQSBAIJ };
      PetscCall(MatISGetLocalMat(P, &N));
      std::initializer_list<std::string>::const_iterator it = std::find(list.begin(), list.end(), ((PetscObject)N)->type_name);
      PetscCall(MatISRestoreLocalMat(P, &N));
      switch (std::distance(list.begin(), it)) {
      case 0:
        PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &C));
        break;
      case 1:
        /* MatCreateSubMatrices() does not work with MATSBAIJ and unsorted ISes, so convert to MPIBAIJ */
        PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &C));
        PetscCall(MatSetOption(C, MAT_SYMMETRIC, PETSC_TRUE));
        break;
      default:
        PetscCall(MatConvert(P, MATMPIAIJ, MAT_INITIAL_MATRIX, &C));
      }
      PetscCall(MatISGetLocalToGlobalMapping(P, &l2g, NULL));
      PetscCall(PetscObjectReference((PetscObject)P));
      PetscCall(KSPSetOperators(data->levels[0]->ksp, A, C));
      std::swap(C, P);
      PetscCall(ISLocalToGlobalMappingGetSize(l2g, &n));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &loc));
      PetscCall(ISLocalToGlobalMappingApplyIS(l2g, loc, &is[0]));
      PetscCall(ISDestroy(&loc));
      /* the auxiliary Mat is _not_ the local Neumann matrix                                */
      /* it is the local Neumann matrix augmented (with zeros) through MatIncreaseOverlap() */
      data->Neumann = PETSC_FALSE;
      structure = SAME_NONZERO_PATTERN;
      if (data->share) {
        data->share = PETSC_FALSE;
        PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc with a Pmat of type MATIS\n"));
      }
    } else {
      is[0] = data->is;
      if (algebraic) subdomains = PETSC_TRUE;
      PetscCall(PetscOptionsGetBool(NULL, pcpre, "-pc_hpddm_define_subdomains", &subdomains, NULL));
      if (data->share) {
        if (!subdomains) {
          PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since -%spc_hpddm_define_subdomains is not true\n", pcpre ? pcpre : ""));
          data->share = PETSC_FALSE;
        }
        if (data->deflation) {
          PetscCall(PetscInfo(pc, "Nothing to share since PCHPDDMSetDeflationMat() has been called\n"));
          data->share = PETSC_FALSE;
        }
      }
      if (data->Neumann) {
        PetscCheck(!block, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "-pc_hpddm_block_splitting and -pc_hpddm_has_neumann");
        PetscCheck(!algebraic, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "-pc_hpddm_levels_1_st_pc_type mat and -pc_hpddm_has_neumann");
      }
      if (data->Neumann || block) structure = SAME_NONZERO_PATTERN;
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)data->is), P->rmap->n, P->rmap->rstart, 1, &loc));
    }
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_", pcpre ? pcpre : ""));
    PetscCall(PetscOptionsGetEnum(NULL, prefix, "-st_matstructure", MatStructures, (PetscEnum*)&structure, &flg)); /* if not user-provided, force its value when possible */
    if (!flg && structure == SAME_NONZERO_PATTERN) { /* cannot call STSetMatStructure() yet, insert the appropriate option in the database, parsed by STSetFromOptions() */
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-%spc_hpddm_levels_1_st_matstructure", pcpre ? pcpre : ""));
      PetscCall(PetscOptionsSetValue(NULL, prefix, MatStructures[structure]));
    }
    if (data->N > 1 && (data->aux || ismatis || algebraic)) {
      PetscCheck(loadedSym, PETSC_COMM_SELF, PETSC_ERR_PLIB, "HPDDM library not loaded, cannot use more than one level");
      PetscCall(MatSetOption(P, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
      if (ismatis) {
        /* needed by HPDDM (currently) so that the partition of unity is 0 on subdomain interfaces */
        PetscCall(MatIncreaseOverlap(P, 1, is, 1));
        PetscCall(ISDestroy(&data->is));
        data->is = is[0];
      } else {
        if (PetscDefined(USE_DEBUG)) {
          PetscBool equal;
          IS        intersect;

          PetscCall(ISIntersect(data->is, loc, &intersect));
          PetscCall(ISEqualUnsorted(loc, intersect, &equal));
          PetscCall(ISDestroy(&intersect));
          PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "IS of the auxiliary Mat does not include all local rows of A");
        }
        PetscCall(PetscObjectComposeFunction((PetscObject)pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_Private_C", PCHPDDMAlgebraicAuxiliaryMat_Private));
        if (!data->Neumann && !algebraic) {
          PetscCall(PetscObjectTypeCompare((PetscObject)P, MATMPISBAIJ, &flg));
          if (flg) {
            /* maybe better to ISSort(is[0]), MatCreateSubMatrices(), and then MatPermute() */
            /* but there is no MatPermute_SeqSBAIJ(), so as before, just use MATMPIBAIJ     */
            PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &uaux));
            flg = PETSC_FALSE;
          }
        }
      }
      if (algebraic) {
        PetscUseMethod(pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_Private_C", (Mat, IS*, Mat*[], PetscBool), (P, is, &sub, block));
        if (block) {
          PetscCall(PetscObjectQuery((PetscObject)sub[0], "_PCHPDDM_Neumann_Mat", (PetscObject*)&data->aux));
          PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Neumann_Mat", NULL));
        }
      } else if (!uaux) {
        if (data->Neumann) sub = &data->aux;
        else PetscCall(MatCreateSubMatrices(P, 1, is, is, MAT_INITIAL_MATRIX, &sub));
      } else {
        PetscCall(MatCreateSubMatrices(uaux, 1, is, is, MAT_INITIAL_MATRIX, &sub));
        PetscCall(MatDestroy(&uaux));
        PetscCall(MatConvert(sub[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, sub));
      }
      /* Vec holding the partition of unity */
      if (!data->levels[0]->D) {
        PetscCall(ISGetLocalSize(data->is, &n));
        PetscCall(VecCreateMPI(PETSC_COMM_SELF, n, PETSC_DETERMINE, &data->levels[0]->D));
      }
      if (data->share && structure == SAME_NONZERO_PATTERN) { /* share the KSP only when the MatStructure is SAME_NONZERO_PATTERN */
        Mat      D;
        IS       perm = NULL;
        PetscInt size = -1;
        PetscCall(PCHPDDMPermute_Private(*is, data->is, &uis, data->Neumann || block ? sub[0] : data->aux, &C, &perm));
        if (!data->Neumann && !block) {
          PetscCall(MatPermute(sub[0], perm, perm, &D)); /* permute since PCASM will call ISSort() */
          PetscCall(MatHeaderReplace(sub[0], &D));
        }
        if (data->B) { /* see PCHPDDMSetRHSMat() */
          PetscCall(MatPermute(data->B, perm, perm, &D));
          PetscCall(MatHeaderReplace(data->B, &D));
        }
        PetscCall(ISDestroy(&perm));
        if (!data->levels[0]->pc) {
          PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_", pcpre ? pcpre : ""));
          PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->pc));
          PetscCall(PCSetOptionsPrefix(data->levels[0]->pc, prefix));
          PetscCall(PCSetOperators(data->levels[0]->pc, A, P));
        }
        PetscCall(PCSetType(data->levels[0]->pc, PCASM));
        if (!data->levels[0]->pc->setupcalled) PetscCall(PCASMSetLocalSubdomains(data->levels[0]->pc, 1, is, &loc));
        PetscCall(PCSetFromOptions(data->levels[0]->pc));
        if (block) PetscCall(PCHPDDMCommunicationAvoidingPCASM_Private(data->levels[0]->pc, C, algebraic));
        else PetscCall(PCSetUp(data->levels[0]->pc));
        PetscTryMethod(data->levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt*, PetscInt*, KSP**), (data->levels[0]->pc, &size, NULL, &ksp));
        if (size != 1) {
          PetscCall(PCDestroy(&data->levels[0]->pc));
          PetscCall(MatDestroy(&C));
          PetscCall(ISDestroy(&uis));
          data->share = PETSC_FALSE;
          PetscCheck(size == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of subdomain solver %" PetscInt_FMT " != 1", size);
          PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since PCASMGetSubKSP() not found in fine-level PC\n"));
        } else {
          const char *matpre;
          PetscBool  cmp[2];
          PetscCall(KSPGetOperators(ksp[0], subA, subA + 1));
          PetscCall(MatDuplicate(subA[1], MAT_SHARE_NONZERO_PATTERN, &D));
          PetscCall(MatGetOptionsPrefix(subA[1], &matpre));
          PetscCall(MatSetOptionsPrefix(D, matpre));
          PetscCall(PetscObjectTypeCompare((PetscObject)D, MATNORMAL, cmp));
          PetscCall(PetscObjectTypeCompare((PetscObject)C, MATNORMAL, cmp + 1));
          PetscCheck(cmp[0] == cmp[1], PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "-pc_hpddm_levels_1_pc_asm_sub_mat_type %s and auxiliary Mat of type %s",((PetscObject)D)->type_name,((PetscObject)C)->type_name);
          if (!cmp[0]) {
            if (!block) PetscCall(MatAXPY(D, 1.0, C, SUBSET_NONZERO_PATTERN));
            else PetscCall(MatAXPY(D, 1.0, data->aux, SAME_NONZERO_PATTERN));
          } else {
            Mat mat[2];
            PetscCall(MatNormalGetMat(D, mat));
            PetscCall(MatNormalGetMat(C, mat + 1));
            PetscCall(MatAXPY(mat[0], 1.0, mat[1], SUBSET_NONZERO_PATTERN));
          }
          PetscCall(MatPropagateSymmetryOptions(C, D));
          PetscCall(MatDestroy(&C));
          C = D;
          /* swap pointers so that variables stay consistent throughout PCSetUp() */
          std::swap(C, data->aux);
          std::swap(uis, data->is);
          swap = PETSC_TRUE;
        }
      } else if (data->share) {
        data->share = PETSC_FALSE;
        PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since -%spc_hpddm_levels_1_st_matstructure %s (!= %s)\n", pcpre ? pcpre : "", MatStructures[structure], MatStructures[SAME_NONZERO_PATTERN]));
      }
      if (!data->levels[0]->scatter) {
        PetscCall(MatCreateVecs(P, &xin, NULL));
        if (ismatis) PetscCall(MatDestroy(&P));
        PetscCall(VecScatterCreate(xin, data->is, data->levels[0]->D, NULL, &data->levels[0]->scatter));
        PetscCall(VecDestroy(&xin));
      }
      if (data->levels[0]->P) {
        /* if the pattern is the same and PCSetUp() has previously succeeded, reuse HPDDM buffers and connectivity */
        PetscCall(HPDDM::Schwarz<PetscScalar>::destroy(data->levels[0], pc->setupcalled < 1 || pc->flag == DIFFERENT_NONZERO_PATTERN ? PETSC_TRUE : PETSC_FALSE));
      }
      if (!data->levels[0]->P) data->levels[0]->P = new HPDDM::Schwarz<PetscScalar>();
      if (data->log_separate) PetscCall(PetscLogEventBegin(PC_HPDDM_SetUp[0], data->levels[0]->ksp, 0, 0, 0));
      else PetscCall(PetscLogEventBegin(PC_HPDDM_Strc, data->levels[0]->ksp, 0, 0, 0));
      /* HPDDM internal data structure */
      PetscCall(data->levels[0]->P->structure(loc, data->is, sub[0], ismatis ? C : data->aux, data->levels));
      if (!data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_Strc, data->levels[0]->ksp, 0, 0, 0));
      /* matrix pencil of the generalized eigenvalue problem on the overlap (GenEO) */
      if (data->deflation) weighted = data->aux;
      else if (!data->B) {
        PetscCall(MatDuplicate(sub[0], MAT_COPY_VALUES, &weighted));
        PetscCall(PetscObjectTypeCompare((PetscObject)weighted, MATNORMAL, &flg));
        if (!flg) PetscCall(MatDiagonalScale(weighted, data->levels[0]->D, data->levels[0]->D));
        else { /* MATNORMAL applies MatDiagonalScale() in a matrix-free fashion, not what is needed since this won't be passed to SLEPc during the eigensolve */
          PetscCall(MatNormalGetMat(weighted, &data->B));
          PetscCall(MatDiagonalScale(data->B, NULL, data->levels[0]->D));
          data->B = NULL;
          flg = PETSC_FALSE;
        }
        /* neither MatDuplicate() nor MatDiagonaleScale() handles the symmetry options, so propagate the options explicitly */
        /* only useful for -mat_type baij -pc_hpddm_levels_1_st_pc_type cholesky (no problem with MATAIJ or MATSBAIJ)       */
        PetscCall(MatPropagateSymmetryOptions(sub[0], weighted));
      } else weighted = data->B;
      /* SLEPc is used inside the loaded symbol */
      PetscCall((*loadedSym)(data->levels[0]->P, data->is, ismatis ? C : (algebraic && !block ? sub[0] : data->aux), weighted, data->B, initial, data->levels));
      if (data->share) {
        Mat st[2];
        PetscCall(KSPGetOperators(ksp[0], st, st + 1));
        PetscCall(MatCopy(subA[0], st[0], SAME_NONZERO_PATTERN));
        if (subA[1] != subA[0] || st[1] != st[0]) PetscCall(MatCopy(subA[1], st[1], SAME_NONZERO_PATTERN));
      }
      if (data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_SetUp[0], data->levels[0]->ksp, 0, 0, 0));
      if (ismatis) PetscCall(MatISGetLocalMat(C, &N));
      else N = data->aux;
      P = sub[0];
      /* going through the grid hierarchy */
      for (n = 1; n < data->N; ++n) {
        if (data->log_separate) PetscCall(PetscLogEventBegin(PC_HPDDM_SetUp[n], data->levels[n]->ksp, 0, 0, 0));
        /* method composed in the loaded symbol since there, SLEPc is used as well */
        PetscTryMethod(data->levels[0]->ksp, "PCHPDDMSetUp_Private_C", (Mat*, Mat*, PetscInt, PetscInt* const, PC_HPDDM_Level** const), (&P, &N, n, &data->N, data->levels));
        if (data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_SetUp[n], data->levels[n]->ksp, 0, 0, 0));
      }
      /* reset to NULL to avoid any faulty use */
      PetscCall(PetscObjectComposeFunction((PetscObject)data->levels[0]->ksp, "PCHPDDMSetUp_Private_C", NULL));
      if (!ismatis) PetscCall(PetscObjectComposeFunction((PetscObject)pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_C", NULL));
      else PetscCall(PetscObjectDereference((PetscObject)C)); /* matching PetscObjectReference() above */
      for (n = 0; n < data->N - 1; ++n)
        if (data->levels[n]->P) {
          /* HPDDM internal work buffers */
          data->levels[n]->P->setBuffer();
          data->levels[n]->P->super::start();
        }
      if (ismatis || !subdomains) PetscCall(PCHPDDMDestroySubMatrices_Private(data->Neumann, PetscBool(algebraic && !block), sub));
      if (ismatis) data->is = NULL;
      for (n = 0; n < data->N - 1 + (reused > 0); ++n) {
        if (data->levels[n]->P) {
          PC spc;

          /* force the PC to be PCSHELL to do the coarse grid corrections */
          PetscCall(KSPSetSkipPCSetFromOptions(data->levels[n]->ksp, PETSC_TRUE));
          PetscCall(KSPGetPC(data->levels[n]->ksp, &spc));
          PetscCall(PCSetType(spc, PCSHELL));
          PetscCall(PCShellSetContext(spc, data->levels[n]));
          PetscCall(PCShellSetSetUp(spc, PCHPDDMShellSetUp));
          PetscCall(PCShellSetApply(spc, PCHPDDMShellApply));
          PetscCall(PCShellSetMatApply(spc, PCHPDDMShellMatApply));
          PetscCall(PCShellSetDestroy(spc, PCHPDDMShellDestroy));
          if (!data->levels[n]->pc) PetscCall(PCCreate(PetscObjectComm((PetscObject)data->levels[n]->ksp), &data->levels[n]->pc));
          if (n < reused) {
            PetscCall(PCSetReusePreconditioner(spc, PETSC_TRUE));
            PetscCall(PCSetReusePreconditioner(data->levels[n]->pc, PETSC_TRUE));
          }
          PetscCall(PCSetUp(spc));
        }
      }
    } else flg = reused ? PETSC_FALSE : PETSC_TRUE;
    if (!ismatis && subdomains) {
      if (flg) PetscCall(KSPGetPC(data->levels[0]->ksp, &inner));
      else inner = data->levels[0]->pc;
      if (inner) {
        PetscCall(PCSetType(inner, PCASM)); /* inner is the fine-level PC for which one must ensure                       */
                                                      /* PCASMSetLocalSubdomains() has been called when -pc_hpddm_define_subdomains */
        if (!inner->setupcalled) { /* evaluates to PETSC_FALSE when -pc_hpddm_block_splitting */
          PetscCall(PCASMSetLocalSubdomains(inner, 1, is, &loc));
          if (!data->Neumann && data->N > 1) { /* subdomain matrices are already created for the eigenproblem, reuse them for the fine-level PC */
            PetscCall(PCHPDDMPermute_Private(*is, NULL, NULL, sub[0], &C, NULL));
            PetscCall(PCHPDDMCommunicationAvoidingPCASM_Private(inner, C, algebraic));
            PetscCall(MatDestroy(&C));
          }
        }
      }
      if (data->N > 1) PetscCall(PCHPDDMDestroySubMatrices_Private(data->Neumann, PetscBool(algebraic && !block), sub));
    }
    PetscCall(ISDestroy(&loc));
  } else data->N = 1 + reused; /* enforce this value to 1 + reused if there is no way to build another level */
  if (requested != data->N + reused) {
    PetscCall(PetscInfo(pc, "%" PetscInt_FMT " levels requested, only %" PetscInt_FMT " built + %" PetscInt_FMT " reused. Options for level(s) > %" PetscInt_FMT ", including -%spc_hpddm_coarse_ will not be taken into account\n", requested, data->N, reused, data->N, pcpre ? pcpre : ""));
    PetscCall(PetscInfo(pc, "It is best to tune parameters, e.g., a higher value for -%spc_hpddm_levels_%" PetscInt_FMT "_eps_threshold so that at least one local deflation vector will be selected\n", pcpre ? pcpre : "", data->N));
    /* cannot use PCHPDDMShellDestroy() because PCSHELL not set for unassembled levels */
    for (n = data->N - 1; n < requested - 1; ++n) {
      if (data->levels[n]->P) {
        PetscCall(HPDDM::Schwarz<PetscScalar>::destroy(data->levels[n], PETSC_TRUE));
        PetscCall(VecDestroyVecs(1, &data->levels[n]->v[0]));
        PetscCall(VecDestroyVecs(2, &data->levels[n]->v[1]));
        PetscCall(MatDestroy(data->levels[n]->V));
        PetscCall(MatDestroy(data->levels[n]->V + 1));
        PetscCall(MatDestroy(data->levels[n]->V + 2));
        PetscCall(VecDestroy(&data->levels[n]->D));
        PetscCall(VecScatterDestroy(&data->levels[n]->scatter));
      }
    }
    if (reused) {
      for (n = reused; n < PETSC_PCHPDDM_MAXLEVELS && data->levels[n]; ++n) {
        PetscCall(KSPDestroy(&data->levels[n]->ksp));
        PetscCall(PCDestroy(&data->levels[n]->pc));
      }
    }
    PetscCheck(!PetscDefined(USE_DEBUG), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT " levels requested, only %" PetscInt_FMT " built + %" PetscInt_FMT " reused. Options for level(s) > %" PetscInt_FMT ", including -%spc_hpddm_coarse_ will not be taken into account. It is best to tune parameters, e.g., a higher value for -%spc_hpddm_levels_%" PetscInt_FMT "_eps_threshold so that at least one local deflation vector will be selected. If you don't want this to error out, compile --with-debugging=0", requested, data->N, reused, data->N, pcpre ? pcpre : "", pcpre ? pcpre : "", data->N);
  }
  /* these solvers are created after PCSetFromOptions() is called */
  if (pc->setfromoptionscalled) {
    for (n = 0; n < data->N; ++n) {
      if (data->levels[n]->ksp) PetscCall(KSPSetFromOptions(data->levels[n]->ksp));
      if (data->levels[n]->pc) PetscCall(PCSetFromOptions(data->levels[n]->pc));
    }
    pc->setfromoptionscalled = 0;
  }
  data->N += reused;
  if (data->share && swap) {
    /* swap back pointers so that variables follow the user-provided numbering */
    std::swap(C, data->aux);
    std::swap(uis, data->is);
    PetscCall(MatDestroy(&C));
    PetscCall(ISDestroy(&uis));
  }
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetCoarseCorrectionType - Sets the coarse correction type.

   Input Parameters:
+     pc - preconditioner context
-     type - PC_HPDDM_COARSE_CORRECTION_DEFLATED, PC_HPDDM_COARSE_CORRECTION_ADDITIVE, or PC_HPDDM_COARSE_CORRECTION_BALANCED

   Options Database Key:
.   -pc_hpddm_coarse_correction <deflated, additive, balanced> - type of coarse correction to apply

   Level: intermediate

.seealso: `PCHPDDMGetCoarseCorrectionType()`, `PCHPDDM`, `PCHPDDMCoarseCorrectionType`
@*/
PetscErrorCode PCHPDDMSetCoarseCorrectionType(PC pc, PCHPDDMCoarseCorrectionType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(pc, type, 2);
  PetscTryMethod(pc, "PCHPDDMSetCoarseCorrectionType_C", (PC, PCHPDDMCoarseCorrectionType), (pc, type));
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMGetCoarseCorrectionType - Gets the coarse correction type.

   Input Parameter:
.     pc - preconditioner context

   Output Parameter:
.     type - PC_HPDDM_COARSE_CORRECTION_DEFLATED, PC_HPDDM_COARSE_CORRECTION_ADDITIVE, or PC_HPDDM_COARSE_CORRECTION_BALANCED

   Level: intermediate

.seealso: `PCHPDDMSetCoarseCorrectionType()`, `PCHPDDM`, `PCHPDDMCoarseCorrectionType`
@*/
PetscErrorCode PCHPDDMGetCoarseCorrectionType(PC pc, PCHPDDMCoarseCorrectionType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (type) {
    PetscValidPointer(type, 2);
    PetscUseMethod(pc, "PCHPDDMGetCoarseCorrectionType_C", (PC, PCHPDDMCoarseCorrectionType*), (pc, type));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetCoarseCorrectionType_HPDDM(PC pc, PCHPDDMCoarseCorrectionType type)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  PetscCheck(type >= 0 && type <= 2, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PCHPDDMCoarseCorrectionType %d", type);
  data->correction = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMGetCoarseCorrectionType_HPDDM(PC pc, PCHPDDMCoarseCorrectionType *type)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  *type = data->correction;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMGetSTShareSubKSP - Gets whether the KSP in SLEPc ST and the fine-level subdomain solver is shared.

   Input Parameter:
.     pc - preconditioner context

   Output Parameter:
.     share - whether the KSP is shared or not

   Notes:
     This is not the same as PCGetReusePreconditioner(). The return value is unlikely to be true, but when it is, a symbolic factorization can be skipped
     when using a subdomain PCType such as PCLU or PCCHOLESKY.

   Level: advanced

@*/
PetscErrorCode PCHPDDMGetSTShareSubKSP(PC pc, PetscBool *share)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (share) {
    PetscValidBoolPointer(share, 2);
    PetscUseMethod(pc, "PCHPDDMGetSTShareSubKSP_C", (PC, PetscBool*), (pc, share));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMGetSTShareSubKSP_HPDDM(PC pc, PetscBool *share)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  *share = data->share;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetDeflationMat - Sets the deflation space used to assemble a coarser operator.

   Input Parameters:
+     pc - preconditioner context
.     is - index set of the local deflation matrix
-     U - deflation sequential matrix stored as a MATSEQDENSE

   Level: advanced

.seealso: `PCDeflationSetSpace()`, `PCMGSetRestriction()`
@*/
PetscErrorCode PCHPDDMSetDeflationMat(PC pc, IS is, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 3);
  PetscUseMethod(pc, "PCHPDDMSetDeflationMat_C", (PC, IS, Mat), (pc, is, U));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetDeflationMat_HPDDM(PC pc, IS is, Mat U)
{
  PetscFunctionBegin;
  PetscCall(PCHPDDMSetAuxiliaryMat_Private(pc, is, U, PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode HPDDMLoadDL_Private(PetscBool *found)
{
  char lib[PETSC_MAX_PATH_LEN], dlib[PETSC_MAX_PATH_LEN], dir[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidBoolPointer(found, 1);
  PetscCall(PetscStrcpy(dir, "${PETSC_LIB_DIR}"));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hpddm_dir", dir, sizeof(dir), NULL));
  PetscCall(PetscSNPrintf(lib, sizeof(lib), "%s/libhpddm_petsc", dir));
  PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found));
#if defined(SLEPC_LIB_DIR) /* this variable is passed during SLEPc ./configure since    */
  if (!*found) {           /* slepcconf.h is not yet built (and thus can't be included) */
    PetscCall(PetscStrcpy(dir, HPDDM_STR(SLEPC_LIB_DIR)));
    PetscCall(PetscSNPrintf(lib, sizeof(lib), "%s/libhpddm_petsc", dir));
    PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found));
  }
#endif
  PetscCheck(*found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s not found", lib);
  PetscCall(PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, dlib));
  PetscFunctionReturn(0);
}

/*MC
     PCHPDDM - Interface with the HPDDM library.

   This PC may be used to build multilevel spectral domain decomposition methods based on the GenEO framework [2011, 2019]. It may be viewed as an alternative to spectral AMGe or PCBDDC with adaptive selection of constraints. A chronological bibliography of relevant publications linked with PC available in HPDDM through PCHPDDM may be found below. The interface is explained in details in [2021].

   The matrix to be preconditioned (Pmat) may be unassembled (MATIS), assembled (MATMPIAIJ, MATMPIBAIJ, or MATMPISBAIJ), hierarchical (MATHTOOL), or MATNORMAL. For multilevel preconditioning, when using an assembled or hierarchical Pmat, one must provide an auxiliary local Mat (unassembled local operator for GenEO) using PCHPDDMSetAuxiliaryMat(). Calling this routine is not needed when using a MATIS Pmat, assembly done internally using MatConvert().

   Options Database Keys:
+   -pc_hpddm_define_subdomains <true, default=false> - on the finest level, calls PCASMSetLocalSubdomains() with the IS supplied in PCHPDDMSetAuxiliaryMat() (not relevant with an unassembled Pmat)
.   -pc_hpddm_has_neumann <true, default=false> - on the finest level, informs the PC that the local Neumann matrix is supplied in PCHPDDMSetAuxiliaryMat()
-   -pc_hpddm_coarse_correction <type, default=deflated> - determines the PCHPDDMCoarseCorrectionType when calling PCApply

   Options for subdomain solvers, subdomain eigensolvers (for computing deflation vectors), and the coarse solver can be set with
.vb
      -pc_hpddm_levels_%d_pc_
      -pc_hpddm_levels_%d_ksp_
      -pc_hpddm_levels_%d_eps_
      -pc_hpddm_levels_%d_p
      -pc_hpddm_levels_%d_mat_type_
      -pc_hpddm_coarse_
      -pc_hpddm_coarse_p
      -pc_hpddm_coarse_mat_type_
.ve
   e.g., -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 10 -pc_hpddm_levels_2_p 4 -pc_hpddm_levels_2_sub_pc_type lu -pc_hpddm_levels_2_eps_nev 10 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_mat_type baij will use 10 deflation vectors per subdomain on the fine "level 1", aggregate the fine subdomains into 4 "level 2" subdomains, then use 10 deflation vectors per subdomain on "level 2", and assemble the coarse matrix (of dimension 4 x 10 = 40) on two processes as a MATMPIBAIJ (default is MATMPISBAIJ).

   In order to activate a "level N+1" coarse correction, it is mandatory to call -pc_hpddm_levels_N_eps_nev <nu> or -pc_hpddm_levels_N_eps_threshold <val>. The default -pc_hpddm_coarse_p value is 1, meaning that the coarse operator is aggregated on a single process.

   This preconditioner requires that you build PETSc with SLEPc (--download-slepc=1). By default, the underlying concurrent eigenproblems are solved using SLEPc shift-and-invert spectral transformation. This is usually what gives the best performance for GenEO, cf. [2011, 2013]. As stated above, SLEPc options are available through -pc_hpddm_levels_%d_, e.g., -pc_hpddm_levels_1_eps_type arpack -pc_hpddm_levels_1_eps_threshold 0.1 -pc_hpddm_levels_1_st_type sinvert.

   References:
+   2011 - A robust two-level domain decomposition preconditioner for systems of PDEs. Spillane, Dolean, Hauret, Nataf, Pechstein, and Scheichl. Comptes Rendus Mathematique.
.   2013 - Scalable domain decomposition preconditioners for heterogeneous elliptic problems. Jolivet, Hecht, Nataf, and Prud'homme. SC13.
.   2015 - An introduction to domain decomposition methods: algorithms, theory, and parallel implementation. Dolean, Jolivet, and Nataf. SIAM.
.   2019 - A multilevel Schwarz preconditioner based on a hierarchy of robust coarse spaces. Al Daas, Grigori, Jolivet, and Tournier. SIAM Journal on Scientific Computing.
.   2021 - KSPHPDDM and PCHPDDM: extending PETSc with advanced Krylov methods and robust multilevel overlapping Schwarz preconditioners. Jolivet, Roman, and Zampini. Computer & Mathematics with Applications.
.   2022a - A robust algebraic domain decomposition preconditioner for sparse normal equations. Al Daas, Jolivet, and Scott. SIAM Journal on Scientific Computing.
-   2022b - A robust algebraic multilevel domain decomposition preconditioner for sparse symmetric positive definite matrices. Al Daas and Jolivet.

   Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCHPDDMSetAuxiliaryMat()`, `MATIS`, `PCBDDC`, `PCDEFLATION`, `PCTELESCOPE`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_HPDDM(PC pc)
{
  PC_HPDDM  *data;
  PetscBool found;

  PetscFunctionBegin;
  if (!loadedSym) {
    PetscCall(HPDDMLoadDL_Private(&found));
    if (found) PetscCall(PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, NULL, "PCHPDDM_Internal", (void**)&loadedSym));
  }
  PetscCheck(loadedSym, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCHPDDM_Internal symbol not found in loaded libhpddm_petsc");
  PetscCall(PetscNewLog(pc, &data));
  pc->data                = data;
  pc->ops->reset          = PCReset_HPDDM;
  pc->ops->destroy        = PCDestroy_HPDDM;
  pc->ops->setfromoptions = PCSetFromOptions_HPDDM;
  pc->ops->setup          = PCSetUp_HPDDM;
  pc->ops->apply          = PCApply_HPDDM;
  pc->ops->matapply       = PCMatApply_HPDDM;
  pc->ops->view           = PCView_HPDDM;
  pc->ops->presolve       = PCPreSolve_HPDDM;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetAuxiliaryMat_C", PCHPDDMSetAuxiliaryMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMHasNeumannMat_C", PCHPDDMHasNeumannMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetRHSMat_C", PCHPDDMSetRHSMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetCoarseCorrectionType_C", PCHPDDMSetCoarseCorrectionType_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetCoarseCorrectionType_C", PCHPDDMGetCoarseCorrectionType_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetSTShareSubKSP_C", PCHPDDMGetSTShareSubKSP_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetDeflationMat_C", PCHPDDMSetDeflationMat_HPDDM));
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMInitializePackage - This function initializes everything in the PCHPDDM package. It is called from PCInitializePackage().

   Level: intermediate

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PCHPDDMInitializePackage(void)
{
  char     ename[32];
  PetscInt i;

  PetscFunctionBegin;
  if (PCHPDDMPackageInitialized) PetscFunctionReturn(0);
  PCHPDDMPackageInitialized = PETSC_TRUE;
  PetscCall(PetscRegisterFinalize(PCHPDDMFinalizePackage));
  /* general events registered once during package initialization */
  /* some of these events are not triggered in libpetsc,          */
  /* but rather directly in libhpddm_petsc,                       */
  /* which is in charge of performing the following operations    */

  /* domain decomposition structure from Pmat sparsity pattern    */
  PetscCall(PetscLogEventRegister("PCHPDDMStrc", PC_CLASSID, &PC_HPDDM_Strc));
  /* Galerkin product, redistribution, and setup (not triggered in libpetsc)                */
  PetscCall(PetscLogEventRegister("PCHPDDMPtAP", PC_CLASSID, &PC_HPDDM_PtAP));
  /* Galerkin product with summation, redistribution, and setup (not triggered in libpetsc) */
  PetscCall(PetscLogEventRegister("PCHPDDMPtBP", PC_CLASSID, &PC_HPDDM_PtBP));
  /* next level construction using PtAP and PtBP (not triggered in libpetsc)                */
  PetscCall(PetscLogEventRegister("PCHPDDMNext", PC_CLASSID, &PC_HPDDM_Next));
  static_assert(PETSC_PCHPDDM_MAXLEVELS <= 9, "PETSC_PCHPDDM_MAXLEVELS value is too high");
  for (i = 1; i < PETSC_PCHPDDM_MAXLEVELS; ++i) {
    PetscCall(PetscSNPrintf(ename, sizeof(ename), "PCHPDDMSetUp L%1" PetscInt_FMT, i));
    /* events during a PCSetUp() at level #i _except_ the assembly */
    /* of the Galerkin operator of the coarser level #(i + 1)      */
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &PC_HPDDM_SetUp[i - 1]));
    PetscCall(PetscSNPrintf(ename, sizeof(ename), "PCHPDDMSolve L%1" PetscInt_FMT, i));
    /* events during a PCApply() at level #i _except_              */
    /* the KSPSolve() of the coarser level #(i + 1)                */
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &PC_HPDDM_Solve[i - 1]));
  }
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMFinalizePackage - This function frees everything from the PCHPDDM package. It is called from PetscFinalize().

   Level: intermediate

.seealso: `PetscFinalize()`
@*/
PetscErrorCode PCHPDDMFinalizePackage(void)
{
  PetscFunctionBegin;
  PCHPDDMPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}
