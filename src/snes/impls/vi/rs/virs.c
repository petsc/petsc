
#include <../src/snes/impls/vi/rs/virsimpl.h> /*I "petscsnes.h" I*/
#include <petsc/private/dmimpl.h>
#include <petsc/private/vecimpl.h>

/*
   SNESVIGetInactiveSet - Gets the global indices for the inactive set variables (these correspond to the degrees of freedom the linear
     system is solved on)

   Input parameter:
.  snes - the SNES context

   Output parameter:
.  inact - inactive set index set

 */
PetscErrorCode SNESVIGetInactiveSet(SNES snes,IS *inact)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*)snes->data;

  PetscFunctionBegin;
  *inact = vi->IS_inact;
  PetscFunctionReturn(0);
}

/*
    Provides a wrapper to a DM to allow it to be used to generated the interpolation/restriction from the DM for the smaller matrices and vectors
  defined by the reduced space method.

    Simple calls the regular DM interpolation and restricts it to operation on the variables not associated with active constraints.

*/
typedef struct {
  PetscInt n;                                              /* size of vectors in the reduced DM space */
  IS       inactive;

  PetscErrorCode (*createinterpolation)(DM,DM,Mat*,Vec*);  /* DM's original routines */
  PetscErrorCode (*coarsen)(DM, MPI_Comm, DM*);
  PetscErrorCode (*createglobalvector)(DM,Vec*);
  PetscErrorCode (*createinjection)(DM,DM,Mat*);
  PetscErrorCode (*hascreateinjection)(DM,PetscBool*);

  DM dm;                                                  /* when destroying this object we need to reset the above function into the base DM */
} DM_SNESVI;

/*
     DMCreateGlobalVector_SNESVI - Creates global vector of the size of the reduced space

*/
PetscErrorCode  DMCreateGlobalVector_SNESVI(DM dm,Vec *vec)
{
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Composed SNES is missing");
  CHKERRQ(PetscContainerGetPointer(isnes,(void**)&dmsnesvi));
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)dm),dmsnesvi->n,PETSC_DETERMINE,vec));
  CHKERRQ(VecSetDM(*vec, dm));
  PetscFunctionReturn(0);
}

/*
     DMCreateInterpolation_SNESVI - Modifieds the interpolation obtained from the DM by removing all rows and columns associated with active constraints.

*/
PetscErrorCode  DMCreateInterpolation_SNESVI(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi1,*dmsnesvi2;
  Mat            interp;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  CHKERRQ(PetscContainerGetPointer(isnes,(void**)&dmsnesvi1));
  CHKERRQ(PetscObjectQuery((PetscObject)dm2,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm2),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  CHKERRQ(PetscContainerGetPointer(isnes,(void**)&dmsnesvi2));

  CHKERRQ((*dmsnesvi1->createinterpolation)(dm1,dm2,&interp,NULL));
  CHKERRQ(MatCreateSubMatrix(interp,dmsnesvi2->inactive,dmsnesvi1->inactive,MAT_INITIAL_MATRIX,mat));
  CHKERRQ(MatDestroy(&interp));
  *vec = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetVI(DM,IS);
static PetscErrorCode DMDestroyVI(DM);

/*
     DMCoarsen_SNESVI - Computes the regular coarsened DM then computes additional information about its inactive set

*/
PetscErrorCode  DMCoarsen_SNESVI(DM dm1,MPI_Comm comm,DM *dm2)
{
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi1;
  Vec            finemarked,coarsemarked;
  IS             inactive;
  Mat            inject;
  const PetscInt *index;
  PetscInt       n,k,cnt = 0,rstart,*coarseindex;
  PetscScalar    *marked;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  CHKERRQ(PetscContainerGetPointer(isnes,(void**)&dmsnesvi1));

  /* get the original coarsen */
  CHKERRQ((*dmsnesvi1->coarsen)(dm1,comm,dm2));

  /* not sure why this extra reference is needed, but without the dm2 disappears too early */
  /* Updating the KSPCreateVecs() to avoid using DMGetGlobalVector() when matrix is available removes the need for this reference? */
  /*  CHKERRQ(PetscObjectReference((PetscObject)*dm2));*/

  /* need to set back global vectors in order to use the original injection */
  CHKERRQ(DMClearGlobalVectors(dm1));

  dm1->ops->createglobalvector = dmsnesvi1->createglobalvector;

  CHKERRQ(DMCreateGlobalVector(dm1,&finemarked));
  CHKERRQ(DMCreateGlobalVector(*dm2,&coarsemarked));

  /*
     fill finemarked with locations of inactive points
  */
  CHKERRQ(ISGetIndices(dmsnesvi1->inactive,&index));
  CHKERRQ(ISGetLocalSize(dmsnesvi1->inactive,&n));
  CHKERRQ(VecSet(finemarked,0.0));
  for (k=0; k<n; k++) {
    CHKERRQ(VecSetValue(finemarked,index[k],1.0,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(finemarked));
  CHKERRQ(VecAssemblyEnd(finemarked));

  CHKERRQ(DMCreateInjection(*dm2,dm1,&inject));
  CHKERRQ(MatRestrict(inject,finemarked,coarsemarked));
  CHKERRQ(MatDestroy(&inject));

  /*
     create index set list of coarse inactive points from coarsemarked
  */
  CHKERRQ(VecGetLocalSize(coarsemarked,&n));
  CHKERRQ(VecGetOwnershipRange(coarsemarked,&rstart,NULL));
  CHKERRQ(VecGetArray(coarsemarked,&marked));
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) cnt++;
  }
  CHKERRQ(PetscMalloc1(cnt,&coarseindex));
  cnt  = 0;
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) coarseindex[cnt++] = k + rstart;
  }
  CHKERRQ(VecRestoreArray(coarsemarked,&marked));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)coarsemarked),cnt,coarseindex,PETSC_OWN_POINTER,&inactive));

  CHKERRQ(DMClearGlobalVectors(dm1));

  dm1->ops->createglobalvector = DMCreateGlobalVector_SNESVI;

  CHKERRQ(DMSetVI(*dm2,inactive));

  CHKERRQ(VecDestroy(&finemarked));
  CHKERRQ(VecDestroy(&coarsemarked));
  CHKERRQ(ISDestroy(&inactive));
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_SNESVI(DM_SNESVI *dmsnesvi)
{
  PetscFunctionBegin;
  /* reset the base methods in the DM object that were changed when the DM_SNESVI was reset */
  dmsnesvi->dm->ops->createinterpolation = dmsnesvi->createinterpolation;
  dmsnesvi->dm->ops->coarsen             = dmsnesvi->coarsen;
  dmsnesvi->dm->ops->createglobalvector  = dmsnesvi->createglobalvector;
  dmsnesvi->dm->ops->createinjection     = dmsnesvi->createinjection;
  dmsnesvi->dm->ops->hascreateinjection  = dmsnesvi->hascreateinjection;
  /* need to clear out this vectors because some of them may not have a reference to the DM
    but they are counted as having references to the DM in DMDestroy() */
  CHKERRQ(DMClearGlobalVectors(dmsnesvi->dm));

  CHKERRQ(ISDestroy(&dmsnesvi->inactive));
  CHKERRQ(PetscFree(dmsnesvi));
  PetscFunctionReturn(0);
}

/*
     DMSetVI - Marks a DM as associated with a VI problem. This causes the interpolation/restriction operators to
               be restricted to only those variables NOT associated with active constraints.

*/
static PetscErrorCode DMSetVI(DM dm,IS inactive)
{
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);

  CHKERRQ(PetscObjectReference((PetscObject)inactive));

  CHKERRQ(PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes));
  if (!isnes) {
    CHKERRQ(PetscContainerCreate(PetscObjectComm((PetscObject)dm),&isnes));
    CHKERRQ(PetscContainerSetUserDestroy(isnes,(PetscErrorCode (*)(void*))DMDestroy_SNESVI));
    CHKERRQ(PetscNew(&dmsnesvi));
    CHKERRQ(PetscContainerSetPointer(isnes,(void*)dmsnesvi));
    CHKERRQ(PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)isnes));
    CHKERRQ(PetscContainerDestroy(&isnes));

    dmsnesvi->createinterpolation = dm->ops->createinterpolation;
    dm->ops->createinterpolation  = DMCreateInterpolation_SNESVI;
    dmsnesvi->coarsen             = dm->ops->coarsen;
    dm->ops->coarsen              = DMCoarsen_SNESVI;
    dmsnesvi->createglobalvector  = dm->ops->createglobalvector;
    dm->ops->createglobalvector   = DMCreateGlobalVector_SNESVI;
    dmsnesvi->createinjection     = dm->ops->createinjection;
    dm->ops->createinjection      = NULL;
    dmsnesvi->hascreateinjection  = dm->ops->hascreateinjection;
    dm->ops->hascreateinjection   = NULL;
  } else {
    CHKERRQ(PetscContainerGetPointer(isnes,(void**)&dmsnesvi));
    CHKERRQ(ISDestroy(&dmsnesvi->inactive));
  }
  CHKERRQ(DMClearGlobalVectors(dm));
  CHKERRQ(ISGetLocalSize(inactive,&dmsnesvi->n));

  dmsnesvi->inactive = inactive;
  dmsnesvi->dm       = dm;
  PetscFunctionReturn(0);
}

/*
     DMDestroyVI - Frees the DM_SNESVI object contained in the DM
         - also resets the function pointers in the DM for createinterpolation() etc to use the original DM
*/
static PetscErrorCode DMDestroyVI(DM dm)
{
  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)NULL));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/

PetscErrorCode SNESCreateIndexSets_VINEWTONRSLS(SNES snes,Vec X,Vec F,IS *ISact,IS *ISinact)
{
  PetscFunctionBegin;
  CHKERRQ(SNESVIGetActiveSetIS(snes,X,F,ISact));
  CHKERRQ(ISComplement(*ISact,X->map->rstart,X->map->rend,ISinact));
  PetscFunctionReturn(0);
}

/* Create active and inactive set vectors. The local size of this vector is set and petsc computes the global size */
PetscErrorCode SNESCreateSubVectors_VINEWTONRSLS(SNES snes,PetscInt n,Vec *newv)
{
  Vec            v;

  PetscFunctionBegin;
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)snes),&v));
  CHKERRQ(VecSetSizes(v,n,PETSC_DECIDE));
  CHKERRQ(VecSetType(v,VECSTANDARD));
  *newv = v;
  PetscFunctionReturn(0);
}

/* Resets the snes PC and KSP when the active set sizes change */
PetscErrorCode SNESVIResetPCandKSP(SNES snes,Mat Amat,Mat Pmat)
{
  KSP            snesksp;

  PetscFunctionBegin;
  CHKERRQ(SNESGetKSP(snes,&snesksp));
  CHKERRQ(KSPReset(snesksp));
  CHKERRQ(KSPResetFromOptions(snesksp));

  /*
  KSP                    kspnew;
  PC                     pcnew;
  MatSolverType          stype;

  CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)snes),&kspnew));
  kspnew->pc_side = snesksp->pc_side;
  kspnew->rtol    = snesksp->rtol;
  kspnew->abstol    = snesksp->abstol;
  kspnew->max_it  = snesksp->max_it;
  CHKERRQ(KSPSetType(kspnew,((PetscObject)snesksp)->type_name));
  CHKERRQ(KSPGetPC(kspnew,&pcnew));
  CHKERRQ(PCSetType(kspnew->pc,((PetscObject)snesksp->pc)->type_name));
  CHKERRQ(PCSetOperators(kspnew->pc,Amat,Pmat));
  CHKERRQ(PCFactorGetMatSolverType(snesksp->pc,&stype));
  CHKERRQ(PCFactorSetMatSolverType(kspnew->pc,stype));
  CHKERRQ(KSPDestroy(&snesksp));
  snes->ksp = kspnew;
  CHKERRQ(PetscLogObjectParent((PetscObject)snes,(PetscObject)kspnew));
   CHKERRQ(KSPSetFromOptions(kspnew));*/
  PetscFunctionReturn(0);
}

/* Variational Inequality solver using reduce space method. No semismooth algorithm is
   implemented in this algorithm. It basically identifies the active constraints and does
   a linear solve on the other variables (those not associated with the active constraints). */
PetscErrorCode SNESSolve_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS    *vi = (SNES_VINEWTONRSLS*)snes->data;
  PetscInt             maxits,i,lits;
  SNESLineSearchReason lssucceed;
  PetscReal            fnorm,gnorm,xnorm=0,ynorm;
  Vec                  Y,X,F;
  KSPConvergedReason   kspreason;
  KSP                  ksp;
  PC                   pc;

  PetscFunctionBegin;
  /* Multigrid must use Galerkin for coarse grids with active set/reduced space methods; cannot rediscretize on coarser grids*/
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH));

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */

  CHKERRQ(SNESLineSearchSetVIFunctions(snes->linesearch, SNESVIProjectOntoBounds, SNESVIComputeInactiveSetFnorm));
  CHKERRQ(SNESLineSearchSetVecs(snes->linesearch, X, NULL, NULL, NULL, NULL));
  CHKERRQ(SNESLineSearchSetUp(snes->linesearch));

  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.0;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));

  CHKERRQ(SNESVIProjectOntoBounds(snes,X));
  CHKERRQ(SNESComputeFunction(snes,X,F));
  CHKERRQ(SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm));
  CHKERRQ(VecNorm(X,NORM_2,&xnorm));        /* xnorm <- ||x||  */
  SNESCheckFunctionNorm(snes,fnorm);
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
  CHKERRQ(SNESLogConvergenceHistory(snes,fnorm,0));
  CHKERRQ(SNESMonitor(snes,0,fnorm));

  /* test convergence */
  CHKERRQ((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {

    IS         IS_act; /* _act -> active set _inact -> inactive set */
    IS         IS_redact; /* redundant active set */
    VecScatter scat_act,scat_inact;
    PetscInt   nis_act,nis_inact;
    Vec        Y_act,Y_inact,F_inact;
    Mat        jac_inact_inact,prejac_inact_inact;
    PetscBool  isequal;

    /* Call general purpose update function */
    if (snes->ops->update) {
      CHKERRQ((*snes->ops->update)(snes, snes->iter));
    }
    CHKERRQ(SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre));
    SNESCheckJacobianDomainerror(snes);

    /* Create active and inactive index sets */

    /*original
    CHKERRQ(SNESVICreateIndexSets_RS(snes,X,F,&IS_act,&vi->IS_inact));
     */
    CHKERRQ(SNESVIGetActiveSetIS(snes,X,F,&IS_act));

    if (vi->checkredundancy) {
      CHKERRQ((*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP));
      if (IS_redact) {
        CHKERRQ(ISSort(IS_redact));
        CHKERRQ(ISComplement(IS_redact,X->map->rstart,X->map->rend,&vi->IS_inact));
        CHKERRQ(ISDestroy(&IS_redact));
      } else {
        CHKERRQ(ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact));
      }
    } else {
      CHKERRQ(ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact));
    }

    /* Create inactive set submatrix */
    CHKERRQ(MatCreateSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact));

    if (0) {                    /* Dead code (temporary developer hack) */
      IS keptrows;
      CHKERRQ(MatFindNonzeroRows(jac_inact_inact,&keptrows));
      if (keptrows) {
        PetscInt       cnt,*nrows,k;
        const PetscInt *krows,*inact;
        PetscInt       rstart;

        CHKERRQ(MatGetOwnershipRange(jac_inact_inact,&rstart,NULL));
        CHKERRQ(MatDestroy(&jac_inact_inact));
        CHKERRQ(ISDestroy(&IS_act));

        CHKERRQ(ISGetLocalSize(keptrows,&cnt));
        CHKERRQ(ISGetIndices(keptrows,&krows));
        CHKERRQ(ISGetIndices(vi->IS_inact,&inact));
        CHKERRQ(PetscMalloc1(cnt,&nrows));
        for (k=0; k<cnt; k++) nrows[k] = inact[krows[k]-rstart];
        CHKERRQ(ISRestoreIndices(keptrows,&krows));
        CHKERRQ(ISRestoreIndices(vi->IS_inact,&inact));
        CHKERRQ(ISDestroy(&keptrows));
        CHKERRQ(ISDestroy(&vi->IS_inact));

        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)snes),cnt,nrows,PETSC_OWN_POINTER,&vi->IS_inact));
        CHKERRQ(ISComplement(vi->IS_inact,F->map->rstart,F->map->rend,&IS_act));
        CHKERRQ(MatCreateSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact));
      }
    }
    CHKERRQ(DMSetVI(snes->dm,vi->IS_inact));
    /* remove later */

    /*
    CHKERRQ(VecView(vi->xu,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xu))->comm)));
    CHKERRQ(VecView(vi->xl,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xl))->comm)));
    CHKERRQ(VecView(X,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)X))));
    CHKERRQ(VecView(F,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)F))));
    CHKERRQ(ISView(vi->IS_inact,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)vi->IS_inact))));
     */

    /* Get sizes of active and inactive sets */
    CHKERRQ(ISGetLocalSize(IS_act,&nis_act));
    CHKERRQ(ISGetLocalSize(vi->IS_inact,&nis_inact));

    /* Create active and inactive set vectors */
    CHKERRQ(SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&F_inact));
    CHKERRQ(SNESCreateSubVectors_VINEWTONRSLS(snes,nis_act,&Y_act));
    CHKERRQ(SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&Y_inact));

    /* Create scatter contexts */
    CHKERRQ(VecScatterCreate(Y,IS_act,Y_act,NULL,&scat_act));
    CHKERRQ(VecScatterCreate(Y,vi->IS_inact,Y_inact,NULL,&scat_inact));

    /* Do a vec scatter to active and inactive set vectors */
    CHKERRQ(VecScatterBegin(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD));

    CHKERRQ(VecScatterBegin(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD));

    CHKERRQ(VecScatterBegin(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD));

    /* Active set direction = 0 */
    CHKERRQ(VecSet(Y_act,0));
    if (snes->jacobian != snes->jacobian_pre) {
      CHKERRQ(MatCreateSubMatrix(snes->jacobian_pre,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact));
    } else prejac_inact_inact = jac_inact_inact;

    CHKERRQ(ISEqual(vi->IS_inact_prev,vi->IS_inact,&isequal));
    if (!isequal) {
      CHKERRQ(SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact));
      CHKERRQ(PCFieldSplitRestrictIS(pc,vi->IS_inact));
    }

    /*      CHKERRQ(ISView(vi->IS_inact,0)); */
    /*      CHKERRQ(ISView(IS_act,0));*/
    /*      ierr = MatView(snes->jacobian_pre,0); */

    CHKERRQ(KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact));
    CHKERRQ(KSPSetUp(snes->ksp));
    {
      PC        pc;
      PetscBool flg;
      CHKERRQ(KSPGetPC(snes->ksp,&pc));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&flg));
      if (flg) {
        KSP *subksps;
        CHKERRQ(PCFieldSplitGetSubKSP(pc,NULL,&subksps));
        CHKERRQ(KSPGetPC(subksps[0],&pc));
        CHKERRQ(PetscFree(subksps));
        CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg));
        if (flg) {
          PetscInt       n,N = 101*101,j,cnts[3] = {0,0,0};
          const PetscInt *ii;

          CHKERRQ(ISGetSize(vi->IS_inact,&n));
          CHKERRQ(ISGetIndices(vi->IS_inact,&ii));
          for (j=0; j<n; j++) {
            if (ii[j] < N) cnts[0]++;
            else if (ii[j] < 2*N) cnts[1]++;
            else if (ii[j] < 3*N) cnts[2]++;
          }
          CHKERRQ(ISRestoreIndices(vi->IS_inact,&ii));

          CHKERRQ(PCBJacobiSetTotalBlocks(pc,3,cnts));
        }
      }
    }

    CHKERRQ(KSPSolve(snes->ksp,F_inact,Y_inact));
    CHKERRQ(VecScatterBegin(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE));

    CHKERRQ(VecDestroy(&F_inact));
    CHKERRQ(VecDestroy(&Y_act));
    CHKERRQ(VecDestroy(&Y_inact));
    CHKERRQ(VecScatterDestroy(&scat_act));
    CHKERRQ(VecScatterDestroy(&scat_inact));
    CHKERRQ(ISDestroy(&IS_act));
    if (!isequal) {
      CHKERRQ(ISDestroy(&vi->IS_inact_prev));
      CHKERRQ(ISDuplicate(vi->IS_inact,&vi->IS_inact_prev));
    }
    CHKERRQ(ISDestroy(&vi->IS_inact));
    CHKERRQ(MatDestroy(&jac_inact_inact));
    if (snes->jacobian != snes->jacobian_pre) {
      CHKERRQ(MatDestroy(&prejac_inact_inact));
    }

    CHKERRQ(KSPGetConvergedReason(snes->ksp,&kspreason));
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        CHKERRQ(PetscInfo(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures));
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }

    CHKERRQ(KSPGetIterationNumber(snes->ksp,&lits));
    snes->linear_its += lits;
    CHKERRQ(PetscInfo(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits));
    /*
    if (snes->ops->precheck) {
      PetscBool changed_y = PETSC_FALSE;
      CHKERRQ((*snes->ops->precheck)(snes,X,Y,snes->precheck,&changed_y));
    }

    if (PetscLogPrintInfo) {
      CHKERRQ(SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W));
    }
    */
    /* Compute a (scaled) negative update in the line search routine:
         Y <- X - lambda*Y
       and evaluate G = function(Y) (depends on the line search).
    */
    CHKERRQ(VecCopy(Y,snes->vec_sol_update));
    ynorm = 1; gnorm = fnorm;
    CHKERRQ(SNESLineSearchApply(snes->linesearch, X, F, &gnorm, Y));
    CHKERRQ(SNESLineSearchGetReason(snes->linesearch, &lssucceed));
    CHKERRQ(SNESLineSearchGetNorms(snes->linesearch, &xnorm, &gnorm, &ynorm));
    CHKERRQ(PetscInfo(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed));
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      CHKERRQ(DMDestroyVI(snes->dm));
      PetscFunctionReturn(0);
    }
    if (lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        CHKERRQ(SNESVICheckLocalMin_Private(snes,snes->jacobian,F,X,gnorm,&ismin));
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
   }
   CHKERRQ(DMDestroyVI(snes->dm));
    /* Update function and solution vectors */
    fnorm = gnorm;
    /* Monitor convergence */
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
    CHKERRQ(SNESLogConvergenceHistory(snes,snes->norm,lits));
    CHKERRQ(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESConvergedSkip) CHKERRQ(VecNorm(X,NORM_2,&xnorm));
    CHKERRQ((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;
  }
  /* make sure that the VI information attached to the DM is removed if the for loop above was broken early due to some exceptional conditional */
  CHKERRQ(DMDestroyVI(snes->dm));
  if (i == maxits) {
    CHKERRQ(PetscInfo(snes,"Maximum number of iterations has been reached: %D\n",maxits));
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVISetRedundancyCheck(SNES snes,PetscErrorCode (*func)(SNES,IS,IS*,void*),void *ctx)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  vi->checkredundancy = func;
  vi->ctxP            = ctx;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>
#include <mex.h>
typedef struct {char *funcname; mxArray *ctx;} SNESMatlabContext;

PetscErrorCode SNESVIRedundancyCheck_Matlab(SNES snes,IS is_act,IS *is_redact,void *ctx)
{
  SNESMatlabContext *sctx = (SNESMatlabContext*)ctx;
  int               nlhs  = 1, nrhs = 5;
  mxArray           *plhs[1], *prhs[5];
  long long int     l1      = 0, l2 = 0, ls = 0;
  PetscInt          *indices=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(is_act,IS_CLASSID,2);
  PetscValidPointer(is_redact,3);
  PetscCheckSameComm(snes,1,is_act,2);

  /* Create IS for reduced active set of size 0, its size and indices will
   bet set by the Matlab function */
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)snes),0,indices,PETSC_OWN_POINTER,is_redact));
  /* call Matlab function in ctx */
  CHKERRQ(PetscArraycpy(&ls,&snes,1));
  CHKERRQ(PetscArraycpy(&l1,&is_act,1));
  CHKERRQ(PetscArraycpy(&l2,is_redact,1));
  prhs[0] = mxCreateDoubleScalar((double)ls);
  prhs[1] = mxCreateDoubleScalar((double)l1);
  prhs[2] = mxCreateDoubleScalar((double)l2);
  prhs[3] = mxCreateString(sctx->funcname);
  prhs[4] = sctx->ctx;
  CHKERRQ(mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESVIRedundancyCheckInternal"));
  CHKERRQ(mxGetScalar(plhs[0]));
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVISetRedundancyCheckMatlab(SNES snes,const char *func,mxArray *ctx)
{
  SNESMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  CHKERRQ(PetscNew(&sctx));
  CHKERRQ(PetscStrallocpy(func,&sctx->funcname));
  sctx->ctx = mxDuplicateArray(ctx);
  CHKERRQ(SNESVISetRedundancyCheck(snes,SNESVIRedundancyCheck_Matlab,sctx));
  PetscFunctionReturn(0);
}

#endif

/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VINEWTONRSLS - Sets up the internal data structures for the later use
   of the SNESVI nonlinear solver.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
PetscErrorCode SNESSetUp_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;
  PetscInt          *indices;
  PetscInt          i,n,rstart,rend;
  SNESLineSearch    linesearch;

  PetscFunctionBegin;
  CHKERRQ(SNESSetUp_VI(snes));

  /* Set up previous active index set for the first snes solve
   vi->IS_inact_prev = 0,1,2,....N */

  CHKERRQ(VecGetOwnershipRange(snes->vec_sol,&rstart,&rend));
  CHKERRQ(VecGetLocalSize(snes->vec_sol,&n));
  CHKERRQ(PetscMalloc1(n,&indices));
  for (i=0; i < n; i++) indices[i] = rstart + i;
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)snes),n,indices,PETSC_OWN_POINTER,&vi->IS_inact_prev));

  /* set the line search functions */
  if (!snes->linesearch) {
    CHKERRQ(SNESGetLineSearch(snes, &linesearch));
    CHKERRQ(SNESLineSearchSetType(linesearch, SNESLINESEARCHBT));
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode SNESReset_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;

  PetscFunctionBegin;
  CHKERRQ(SNESReset_VI(snes));
  CHKERRQ(ISDestroy(&vi->IS_inact_prev));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESVINEWTONRSLS - Reduced space active set solvers for variational inequalities based on Newton's method

   Options Database:
+   -snes_type <vinewtonssls,vinewtonrsls> - a semi-smooth solver, a reduced space active set method
-   -snes_vi_monitor - prints the number of active constraints at each iteration.

   Level: beginner

   References:
.  * - T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNESVISetVariableBounds(), SNESVISetComputeVariableBounds(), SNESCreate(), SNES, SNESSetType(), SNESVINEWTONSSLS, SNESNEWTONTR, SNESLineSearchSetType(),SNESLineSearchSetPostCheck(), SNESLineSearchSetPreCheck()

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS *vi;
  SNESLineSearch    linesearch;

  PetscFunctionBegin;
  snes->ops->reset          = SNESReset_VINEWTONRSLS;
  snes->ops->setup          = SNESSetUp_VINEWTONRSLS;
  snes->ops->solve          = SNESSolve_VINEWTONRSLS;
  snes->ops->destroy        = SNESDestroy_VI;
  snes->ops->setfromoptions = SNESSetFromOptions_VI;
  snes->ops->view           = NULL;
  snes->ops->converged      = SNESConvergedDefault_VI;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  CHKERRQ(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    CHKERRQ(SNESLineSearchSetType(linesearch, SNESLINESEARCHBT));
  }
  CHKERRQ(SNESLineSearchBTSetAlpha(linesearch, 0.0));

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  CHKERRQ(PetscNewLog(snes,&vi));
  snes->data          = (void*)vi;
  vi->checkredundancy = NULL;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESVISetVariableBounds_C",SNESVISetVariableBounds_VI));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",SNESVISetComputeVariableBounds_VI));
  PetscFunctionReturn(0);
}
