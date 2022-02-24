#define PETSCKSP_DLL

/*
   Provides an interface to pARMS.
   Requires pARMS 3.2 or later.
*/

#include <petsc/private/pcimpl.h>          /*I "petscpc.h" I*/

#if defined(PETSC_USE_COMPLEX)
#define DBL_CMPLX
#else
#define DBL
#endif
#define USE_MPI
#define REAL double
#define HAS_BLAS
#define FORTRAN_UNDERSCORE
#include "parms_sys.h"
#undef FLOAT
#define FLOAT PetscScalar
#include <parms.h>

/*
   Private context (data structure) for the  preconditioner.
*/
typedef struct {
  parms_Map         map;
  parms_Mat         A;
  parms_PC          pc;
  PCPARMSGlobalType global;
  PCPARMSLocalType  local;
  PetscInt          levels, blocksize, maxdim, maxits, lfil[7];
  PetscBool         nonsymperm, meth[8];
  PetscReal         solvetol, indtol, droptol[7];
  PetscScalar       *lvec0, *lvec1;
} PC_PARMS;

static PetscErrorCode PCSetUp_PARMS(PC pc)
{
  Mat                pmat;
  PC_PARMS          *parms = (PC_PARMS*)pc->data;
  const PetscInt    *mapptr0;
  PetscInt           n, lsize, low, high, i, pos, ncols, length;
  int               *maptmp, *mapptr, *ia, *ja, *ja1, *im;
  PetscScalar       *aa, *aa1;
  const PetscInt    *cols;
  PetscInt           meth[8];
  const PetscScalar *values;
  MatInfo            matinfo;
  PetscMPIInt        rank, npro;

  PetscFunctionBegin;
  /* Get preconditioner matrix from PETSc and setup pARMS structs */
  CHKERRQ(PCGetOperators(pc,NULL,&pmat));
  MPI_Comm_size(PetscObjectComm((PetscObject)pmat),&npro);
  MPI_Comm_rank(PetscObjectComm((PetscObject)pmat),&rank);

  CHKERRQ(MatGetSize(pmat,&n,NULL));
  CHKERRQ(PetscMalloc1(npro+1,&mapptr));
  CHKERRQ(PetscMalloc1(n,&maptmp));
  CHKERRQ(MatGetOwnershipRanges(pmat,&mapptr0));
  low   = mapptr0[rank];
  high  = mapptr0[rank+1];
  lsize = high - low;

  for (i=0; i<npro+1; i++) mapptr[i] = mapptr0[i]+1;
  for (i = 0; i<n; i++) maptmp[i] = i+1;

  /* if created, destroy the previous map */
  if (parms->map) {
    parms_MapFree(&parms->map);
    parms->map = NULL;
  }

  /* create pARMS map object */
  parms_MapCreateFromPtr(&parms->map,(int)n,maptmp,mapptr,PetscObjectComm((PetscObject)pmat),1,NONINTERLACED);

  /* if created, destroy the previous pARMS matrix */
  if (parms->A) {
    parms_MatFree(&parms->A);
    parms->A = NULL;
  }

  /* create pARMS mat object */
  parms_MatCreate(&parms->A,parms->map);

  /* setup and copy csr data structure for pARMS */
  CHKERRQ(PetscMalloc1(lsize+1,&ia));
  ia[0]  = 1;
  CHKERRQ(MatGetInfo(pmat,MAT_LOCAL,&matinfo));
  length = matinfo.nz_used;
  CHKERRQ(PetscMalloc1(length,&ja));
  CHKERRQ(PetscMalloc1(length,&aa));

  for (i = low; i<high; i++) {
    pos         = ia[i-low]-1;
    CHKERRQ(MatGetRow(pmat,i,&ncols,&cols,&values));
    ia[i-low+1] = ia[i-low] + ncols;

    if (ia[i-low+1] >= length) {
      length += ncols;
      CHKERRQ(PetscMalloc1(length,&ja1));
      CHKERRQ(PetscArraycpy(ja1,ja,ia[i-low]-1));
      CHKERRQ(PetscFree(ja));
      ja      = ja1;
      CHKERRQ(PetscMalloc1(length,&aa1));
      CHKERRQ(PetscArraycpy(aa1,aa,ia[i-low]-1));
      CHKERRQ(PetscFree(aa));
      aa      = aa1;
    }
    CHKERRQ(PetscArraycpy(&ja[pos],cols,ncols));
    CHKERRQ(PetscArraycpy(&aa[pos],values,ncols));
    CHKERRQ(MatRestoreRow(pmat,i,&ncols,&cols,&values));
  }

  /* csr info is for local matrix so initialize im[] locally */
  CHKERRQ(PetscMalloc1(lsize,&im));
  CHKERRQ(PetscArraycpy(im,&maptmp[mapptr[rank]-1],lsize));

  /* 1-based indexing */
  for (i=0; i<ia[lsize]-1; i++) ja[i] = ja[i]+1;

  /* Now copy csr matrix to parms_mat object */
  parms_MatSetValues(parms->A,(int)lsize,im,ia,ja,aa,INSERT);

  /* free memory */
  CHKERRQ(PetscFree(maptmp));
  CHKERRQ(PetscFree(mapptr));
  CHKERRQ(PetscFree(aa));
  CHKERRQ(PetscFree(ja));
  CHKERRQ(PetscFree(ia));
  CHKERRQ(PetscFree(im));

  /* setup parms matrix */
  parms_MatSetup(parms->A);

  /* if created, destroy the previous pARMS pc */
  if (parms->pc) {
    parms_PCFree(&parms->pc);
    parms->pc = NULL;
  }

  /* Now create pARMS preconditioner object based on A */
  parms_PCCreate(&parms->pc,parms->A);

  /* Transfer options from PC to pARMS */
  switch (parms->global) {
  case 0: parms_PCSetType(parms->pc, PCRAS); break;
  case 1: parms_PCSetType(parms->pc, PCSCHUR); break;
  case 2: parms_PCSetType(parms->pc, PCBJ); break;
  }
  switch (parms->local) {
  case 0: parms_PCSetILUType(parms->pc, PCILU0); break;
  case 1: parms_PCSetILUType(parms->pc, PCILUK); break;
  case 2: parms_PCSetILUType(parms->pc, PCILUT); break;
  case 3: parms_PCSetILUType(parms->pc, PCARMS); break;
  }
  parms_PCSetInnerEps(parms->pc, parms->solvetol);
  parms_PCSetNlevels(parms->pc, parms->levels);
  parms_PCSetPermType(parms->pc, parms->nonsymperm ? 1 : 0);
  parms_PCSetBsize(parms->pc, parms->blocksize);
  parms_PCSetTolInd(parms->pc, parms->indtol);
  parms_PCSetInnerKSize(parms->pc, parms->maxdim);
  parms_PCSetInnerMaxits(parms->pc, parms->maxits);
  for (i=0; i<8; i++) meth[i] = parms->meth[i] ? 1 : 0;
  parms_PCSetPermScalOptions(parms->pc, &meth[0], 1);
  parms_PCSetPermScalOptions(parms->pc, &meth[4], 0);
  parms_PCSetFill(parms->pc, parms->lfil);
  parms_PCSetTol(parms->pc, parms->droptol);

  parms_PCSetup(parms->pc);

  /* Allocate two auxiliary vector of length lsize */
  if (parms->lvec0) CHKERRQ(PetscFree(parms->lvec0));
  CHKERRQ(PetscMalloc1(lsize, &parms->lvec0));
  if (parms->lvec1) CHKERRQ(PetscFree(parms->lvec1));
  CHKERRQ(PetscMalloc1(lsize, &parms->lvec1));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_PARMS(PC pc,PetscViewer viewer)
{
  PetscBool  iascii;
  PC_PARMS  *parms = (PC_PARMS*)pc->data;
  char      *str;
  double     fill_fact;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    parms_PCGetName(parms->pc,&str);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  global preconditioner: %s\n",str));
    parms_PCILUGetName(parms->pc,&str);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  local preconditioner: %s\n",str));
    parms_PCGetRatio(parms->pc,&fill_fact);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  non-zero elements/original non-zero entries: %-4.2f\n",fill_fact));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Tolerance for local solve: %g\n",parms->solvetol));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Number of levels: %d\n",parms->levels));
    if (parms->nonsymperm) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using nonsymmetric permutation\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Block size: %d\n",parms->blocksize));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Tolerance for independent sets: %g\n",parms->indtol));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Inner Krylov dimension: %d\n",parms->maxdim));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Maximum number of inner iterations: %d\n",parms->maxits));
    if (parms->meth[0]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using nonsymmetric permutation for interlevel blocks\n"));
    }
    if (parms->meth[1]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using column permutation for interlevel blocks\n"));
    }
    if (parms->meth[2]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using row scaling for interlevel blocks\n"));
    }
    if (parms->meth[3]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using column scaling for interlevel blocks\n"));
    }
    if (parms->meth[4]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using nonsymmetric permutation for last level blocks\n"));
    }
    if (parms->meth[5]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using column permutation for last level blocks\n"));
    }
    if (parms->meth[6]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using row scaling for last level blocks\n"));
    }
    if (parms->meth[7]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using column scaling for last level blocks\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  amount of fill-in for ilut, iluk and arms: %d\n",parms->lfil[0]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  amount of fill-in for schur: %d\n",parms->lfil[4]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  amount of fill-in for ILUT L and U: %d\n",parms->lfil[5]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  drop tolerance for L, U, L^{-1}F and EU^{-1}: %g\n",parms->droptol[0]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  drop tolerance for schur complement at each level: %g\n",parms->droptol[4]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  drop tolerance for ILUT in last level schur complement: %g\n",parms->droptol[5]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PARMS(PC pc)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if (parms->map) parms_MapFree(&parms->map);
  if (parms->A) parms_MatFree(&parms->A);
  if (parms->pc) parms_PCFree(&parms->pc);
  if (parms->lvec0) {
    CHKERRQ(PetscFree(parms->lvec0));
  }
  if (parms->lvec1) {
    CHKERRQ(PetscFree(parms->lvec1));
  }
  CHKERRQ(PetscFree(pc->data));

  CHKERRQ(PetscObjectChangeTypeName((PetscObject)pc,0));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetGlobal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetLocal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetSolveTolerances_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetSolveRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetNonsymPerm_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetFill_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_PARMS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_PARMS          *parms = (PC_PARMS*)pc->data;
  PetscBool          flag;
  PCPARMSGlobalType  global;
  PCPARMSLocalType   local;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"PARMS Options"));
  CHKERRQ(PetscOptionsEnum("-pc_parms_global","Global preconditioner","PCPARMSSetGlobal",PCPARMSGlobalTypes,(PetscEnum)parms->global,(PetscEnum*)&global,&flag));
  if (flag) CHKERRQ(PCPARMSSetGlobal(pc,global));
  CHKERRQ(PetscOptionsEnum("-pc_parms_local","Local preconditioner","PCPARMSSetLocal",PCPARMSLocalTypes,(PetscEnum)parms->local,(PetscEnum*)&local,&flag));
  if (flag) CHKERRQ(PCPARMSSetLocal(pc,local));
  CHKERRQ(PetscOptionsReal("-pc_parms_solve_tol","Tolerance for local solve","PCPARMSSetSolveTolerances",parms->solvetol,&parms->solvetol,NULL));
  CHKERRQ(PetscOptionsInt("-pc_parms_levels","Number of levels","None",parms->levels,&parms->levels,NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_nonsymmetric_perm","Use nonsymmetric permutation","PCPARMSSetNonsymPerm",parms->nonsymperm,&parms->nonsymperm,NULL));
  CHKERRQ(PetscOptionsInt("-pc_parms_blocksize","Block size","None",parms->blocksize,&parms->blocksize,NULL));
  CHKERRQ(PetscOptionsReal("-pc_parms_ind_tol","Tolerance for independent sets","None",parms->indtol,&parms->indtol,NULL));
  CHKERRQ(PetscOptionsInt("-pc_parms_max_dim","Inner Krylov dimension","PCPARMSSetSolveRestart",parms->maxdim,&parms->maxdim,NULL));
  CHKERRQ(PetscOptionsInt("-pc_parms_max_it","Maximum number of inner iterations","PCPARMSSetSolveTolerances",parms->maxits,&parms->maxits,NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_inter_nonsymmetric_perm","nonsymmetric permutation for interlevel blocks","None",parms->meth[0],&parms->meth[0],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_inter_column_perm","column permutation for interlevel blocks","None",parms->meth[1],&parms->meth[1],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_inter_row_scaling","row scaling for interlevel blocks","None",parms->meth[2],&parms->meth[2],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_inter_column_scaling","column scaling for interlevel blocks","None",parms->meth[3],&parms->meth[3],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_last_nonsymmetric_perm","nonsymmetric permutation for last level blocks","None",parms->meth[4],&parms->meth[4],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_last_column_perm","column permutation for last level blocks","None",parms->meth[5],&parms->meth[5],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_last_row_scaling","row scaling for last level blocks","None",parms->meth[6],&parms->meth[6],NULL));
  CHKERRQ(PetscOptionsBool("-pc_parms_last_column_scaling","column scaling for last level blocks","None",parms->meth[7],&parms->meth[7],NULL));
  CHKERRQ(PetscOptionsInt("-pc_parms_lfil_ilu_arms","amount of fill-in for ilut, iluk and arms","PCPARMSSetFill",parms->lfil[0],&parms->lfil[0],&flag));
  if (flag) parms->lfil[1] = parms->lfil[2] = parms->lfil[3] = parms->lfil[0];
  CHKERRQ(PetscOptionsInt("-pc_parms_lfil_schur","amount of fill-in for schur","PCPARMSSetFill",parms->lfil[4],&parms->lfil[4],NULL));
  CHKERRQ(PetscOptionsInt("-pc_parms_lfil_ilut_L_U","amount of fill-in for ILUT L and U","PCPARMSSetFill",parms->lfil[5],&parms->lfil[5],&flag));
  if (flag) parms->lfil[6] = parms->lfil[5];
  CHKERRQ(PetscOptionsReal("-pc_parms_droptol_factors","drop tolerance for L, U, L^{-1}F and EU^{-1}","None",parms->droptol[0],&parms->droptol[0],NULL));
  CHKERRQ(PetscOptionsReal("-pc_parms_droptol_schur_compl","drop tolerance for schur complement at each level","None",parms->droptol[4],&parms->droptol[4],&flag));
  if (flag) parms->droptol[1] = parms->droptol[2] = parms->droptol[3] = parms->droptol[0];
  CHKERRQ(PetscOptionsReal("-pc_parms_droptol_last_schur","drop tolerance for ILUT in last level schur complement","None",parms->droptol[5],&parms->droptol[5],&flag));
  if (flag) parms->droptol[6] = parms->droptol[5];
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PARMS(PC pc,Vec b,Vec x)
{
  PC_PARMS          *parms = (PC_PARMS*)pc->data;
  const PetscScalar *b1;
  PetscScalar       *x1;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(b,&b1));
  CHKERRQ(VecGetArray(x,&x1));
  parms_VecPermAux((PetscScalar*)b1,parms->lvec0,parms->map);
  parms_PCApply(parms->pc,parms->lvec0,parms->lvec1);
  parms_VecInvPermAux(parms->lvec1,x1,parms->map);
  CHKERRQ(VecRestoreArrayRead(b,&b1));
  CHKERRQ(VecRestoreArray(x,&x1));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPARMSSetGlobal_PARMS(PC pc,PCPARMSGlobalType type)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if (type != parms->global) {
    parms->global   = type;
    pc->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   PCPARMSSetGlobal - Sets the global preconditioner to be used in PARMS.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - the global preconditioner type, one of
.vb
     PC_PARMS_GLOBAL_RAS   - Restricted additive Schwarz
     PC_PARMS_GLOBAL_SCHUR - Schur complement
     PC_PARMS_GLOBAL_BJ    - Block Jacobi
.ve

   Options Database Keys:
   -pc_parms_global [ras,schur,bj] - Sets global preconditioner

   Level: intermediate

   Notes:
   See the pARMS function parms_PCSetType for more information.

.seealso: PCPARMS, PCPARMSSetLocal()
@*/
PetscErrorCode PCPARMSSetGlobal(PC pc,PCPARMSGlobalType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  CHKERRQ(PetscTryMethod(pc,"PCPARMSSetGlobal_C",(PC,PCPARMSGlobalType),(pc,type)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPARMSSetLocal_PARMS(PC pc,PCPARMSLocalType type)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if (type != parms->local) {
    parms->local    = type;
    pc->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   PCPARMSSetLocal - Sets the local preconditioner to be used in PARMS.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - the local preconditioner type, one of
.vb
     PC_PARMS_LOCAL_ILU0   - ILU0 preconditioner
     PC_PARMS_LOCAL_ILUK   - ILU(k) preconditioner
     PC_PARMS_LOCAL_ILUT   - ILUT preconditioner
     PC_PARMS_LOCAL_ARMS   - ARMS preconditioner
.ve

   Options Database Keys:
   -pc_parms_local [ilu0,iluk,ilut,arms] - Sets local preconditioner

   Level: intermediate

   Notes:
   For the ARMS preconditioner, one can use either the symmetric ARMS or the non-symmetric
   variant (ARMS-ddPQ) by setting the permutation type with PCPARMSSetNonsymPerm().

   See the pARMS function parms_PCILUSetType for more information.

.seealso: PCPARMS, PCPARMSSetGlobal(), PCPARMSSetNonsymPerm()

@*/
PetscErrorCode PCPARMSSetLocal(PC pc,PCPARMSLocalType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  CHKERRQ(PetscTryMethod(pc,"PCPARMSSetLocal_C",(PC,PCPARMSLocalType),(pc,type)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPARMSSetSolveTolerances_PARMS(PC pc,PetscReal tol,PetscInt maxits)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if (tol != parms->solvetol) {
    parms->solvetol = tol;
    pc->setupcalled = 0;
  }
  if (maxits != parms->maxits) {
    parms->maxits   = maxits;
    pc->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   PCPARMSSetSolveTolerances - Sets the convergence tolerance and the maximum iterations for the
   inner GMRES solver, when the Schur global preconditioner is used.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  tol - the convergence tolerance
-  maxits - the maximum number of iterations to use

   Options Database Keys:
+  -pc_parms_solve_tol - set the tolerance for local solve
-  -pc_parms_max_it - set the maximum number of inner iterations

   Level: intermediate

   Notes:
   See the pARMS functions parms_PCSetInnerEps and parms_PCSetInnerMaxits for more information.

.seealso: PCPARMS, PCPARMSSetSolveRestart()
@*/
PetscErrorCode PCPARMSSetSolveTolerances(PC pc,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCPARMSSetSolveTolerances_C",(PC,PetscReal,PetscInt),(pc,tol,maxits)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPARMSSetSolveRestart_PARMS(PC pc,PetscInt restart)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if (restart != parms->maxdim) {
    parms->maxdim   = restart;
    pc->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   PCPARMSSetSolveRestart - Sets the number of iterations at which the
   inner GMRES solver restarts.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  restart - maximum dimension of the Krylov subspace

   Options Database Keys:
.  -pc_parms_max_dim - sets the inner Krylov dimension

   Level: intermediate

   Notes:
   See the pARMS function parms_PCSetInnerKSize for more information.

.seealso: PCPARMS, PCPARMSSetSolveTolerances()
@*/
PetscErrorCode PCPARMSSetSolveRestart(PC pc,PetscInt restart)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCPARMSSetSolveRestart_C",(PC,PetscInt),(pc,restart)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPARMSSetNonsymPerm_PARMS(PC pc,PetscBool nonsym)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if ((nonsym && !parms->nonsymperm) || (!nonsym && parms->nonsymperm)) {
    parms->nonsymperm = nonsym;
    pc->setupcalled   = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   PCPARMSSetNonsymPerm - Sets the type of permutation for the ARMS preconditioner: the standard
   symmetric ARMS or the non-symmetric ARMS (ARMS-ddPQ).

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  nonsym - PETSC_TRUE indicates the non-symmetric ARMS is used;
            PETSC_FALSE indicates the symmetric ARMS is used

   Options Database Keys:
.  -pc_parms_nonsymmetric_perm - sets the use of nonsymmetric permutation

   Level: intermediate

   Notes:
   See the pARMS function parms_PCSetPermType for more information.

.seealso: PCPARMS
@*/
PetscErrorCode PCPARMSSetNonsymPerm(PC pc,PetscBool nonsym)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCPARMSSetNonsymPerm_C",(PC,PetscBool),(pc,nonsym)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPARMSSetFill_PARMS(PC pc,PetscInt lfil0,PetscInt lfil1,PetscInt lfil2)
{
  PC_PARMS *parms = (PC_PARMS*)pc->data;

  PetscFunctionBegin;
  if (lfil0 != parms->lfil[0] || lfil0 != parms->lfil[1] || lfil0 != parms->lfil[2] || lfil0 != parms->lfil[3]) {
    parms->lfil[1]  = parms->lfil[2] = parms->lfil[3] = parms->lfil[0] = lfil0;
    pc->setupcalled = 0;
  }
  if (lfil1 != parms->lfil[4]) {
    parms->lfil[4]  = lfil1;
    pc->setupcalled = 0;
  }
  if (lfil2 != parms->lfil[5] || lfil2 != parms->lfil[6]) {
    parms->lfil[5]  = parms->lfil[6] = lfil2;
    pc->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   PCPARMSSetFill - Sets the fill-in parameters for ILUT, ILUK and ARMS preconditioners.
   Consider the original matrix A = [B F; E C] and the approximate version
   M = [LB 0; E/UB I]*[UB LB\F; 0 S].

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  fil0 - the level of fill-in kept in LB, UB, E/UB and LB\F
.  fil1 - the level of fill-in kept in S
-  fil2 - the level of fill-in kept in the L and U parts of the LU factorization of S

   Options Database Keys:
+  -pc_parms_lfil_ilu_arms - set the amount of fill-in for ilut, iluk and arms
.  -pc_parms_lfil_schur - set the amount of fill-in for schur
-  -pc_parms_lfil_ilut_L_U - set the amount of fill-in for ILUT L and U

   Level: intermediate

   Notes:
   See the pARMS function parms_PCSetFill for more information.

.seealso: PCPARMS
@*/
PetscErrorCode PCPARMSSetFill(PC pc,PetscInt lfil0,PetscInt lfil1,PetscInt lfil2)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCPARMSSetFill_C",(PC,PetscInt,PetscInt,PetscInt),(pc,lfil0,lfil1,lfil2)));
  PetscFunctionReturn(0);
}

/*MC
   PCPARMS - Allows the use of the parallel Algebraic Recursive Multilevel Solvers
      available in the package pARMS

   Options Database Keys:
+  -pc_parms_global - one of ras, schur, bj
.  -pc_parms_local - one of ilu0, iluk, ilut, arms
.  -pc_parms_solve_tol - set the tolerance for local solve
.  -pc_parms_levels - set the number of levels
.  -pc_parms_nonsymmetric_perm - set the use of nonsymmetric permutation
.  -pc_parms_blocksize - set the block size
.  -pc_parms_ind_tol - set the tolerance for independent sets
.  -pc_parms_max_dim - set the inner krylov dimension
.  -pc_parms_max_it - set the maximum number of inner iterations
.  -pc_parms_inter_nonsymmetric_perm - set the use of nonsymmetric permutation for interlevel blocks
.  -pc_parms_inter_column_perm - set the use of column permutation for interlevel blocks
.  -pc_parms_inter_row_scaling - set the use of row scaling for interlevel blocks
.  -pc_parms_inter_column_scaling - set the use of column scaling for interlevel blocks
.  -pc_parms_last_nonsymmetric_perm - set the use of nonsymmetric permutation for last level blocks
.  -pc_parms_last_column_perm - set the use of column permutation for last level blocks
.  -pc_parms_last_row_scaling - set the use of row scaling for last level blocks
.  -pc_parms_last_column_scaling - set the use of column scaling for last level blocks
.  -pc_parms_lfil_ilu_arms - set the amount of fill-in for ilut, iluk and arms
.  -pc_parms_lfil_schur - set the amount of fill-in for schur
.  -pc_parms_lfil_ilut_L_U - set the amount of fill-in for ILUT L and U
.  -pc_parms_droptol_factors - set the drop tolerance for L, U, L^{-1}F and EU^{-1}
.  -pc_parms_droptol_schur_compl - set the drop tolerance for schur complement at each level
-  -pc_parms_droptol_last_schur - set the drop tolerance for ILUT in last level schur complement

   IMPORTANT:
   Unless configured appropriately, this preconditioner performs an inexact solve
   as part of the preconditioner application. Therefore, it must be used in combination
   with flexible variants of iterative solvers, such as KSPFGMRES or KSPGCR.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

PETSC_EXTERN PetscErrorCode PCCreate_PARMS(PC pc)
{
  PC_PARMS *parms;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&parms));

  parms->map        = 0;
  parms->A          = 0;
  parms->pc         = 0;
  parms->global     = PC_PARMS_GLOBAL_RAS;
  parms->local      = PC_PARMS_LOCAL_ARMS;
  parms->levels     = 10;
  parms->nonsymperm = PETSC_TRUE;
  parms->blocksize  = 250;
  parms->maxdim     = 0;
  parms->maxits     = 0;
  parms->meth[0]    = PETSC_FALSE;
  parms->meth[1]    = PETSC_FALSE;
  parms->meth[2]    = PETSC_FALSE;
  parms->meth[3]    = PETSC_FALSE;
  parms->meth[4]    = PETSC_FALSE;
  parms->meth[5]    = PETSC_FALSE;
  parms->meth[6]    = PETSC_FALSE;
  parms->meth[7]    = PETSC_FALSE;
  parms->solvetol   = 0.01;
  parms->indtol     = 0.4;
  parms->lfil[0]    = parms->lfil[1] = parms->lfil[2] = parms->lfil[3] = 20;
  parms->lfil[4]    = parms->lfil[5] = parms->lfil[6] = 20;
  parms->droptol[0] = parms->droptol[1] = parms->droptol[2] = parms->droptol[3] = 0.00001;
  parms->droptol[4] = 0.001;
  parms->droptol[5] = parms->droptol[6] = 0.001;

  pc->data                = parms;
  pc->ops->destroy        = PCDestroy_PARMS;
  pc->ops->setfromoptions = PCSetFromOptions_PARMS;
  pc->ops->setup          = PCSetUp_PARMS;
  pc->ops->apply          = PCApply_PARMS;
  pc->ops->view           = PCView_PARMS;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetGlobal_C",PCPARMSSetGlobal_PARMS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetLocal_C",PCPARMSSetLocal_PARMS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetSolveTolerances_C",PCPARMSSetSolveTolerances_PARMS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetSolveRestart_C",PCPARMSSetSolveRestart_PARMS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetNonsymPerm_C",PCPARMSSetNonsymPerm_PARMS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPARMSSetFill_C",PCPARMSSetFill_PARMS));
  PetscFunctionReturn(0);
}
