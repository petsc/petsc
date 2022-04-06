
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
  PetscCall(PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Composed SNES is missing");
  PetscCall(PetscContainerGetPointer(isnes,(void**)&dmsnesvi));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)dm),dmsnesvi->n,PETSC_DETERMINE,vec));
  PetscCall(VecSetDM(*vec, dm));
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
  PetscCall(PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  PetscCall(PetscContainerGetPointer(isnes,(void**)&dmsnesvi1));
  PetscCall(PetscObjectQuery((PetscObject)dm2,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm2),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  PetscCall(PetscContainerGetPointer(isnes,(void**)&dmsnesvi2));

  PetscCall((*dmsnesvi1->createinterpolation)(dm1,dm2,&interp,NULL));
  PetscCall(MatCreateSubMatrix(interp,dmsnesvi2->inactive,dmsnesvi1->inactive,MAT_INITIAL_MATRIX,mat));
  PetscCall(MatDestroy(&interp));
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
  PetscCall(PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes));
  PetscCheck(isnes,PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  PetscCall(PetscContainerGetPointer(isnes,(void**)&dmsnesvi1));

  /* get the original coarsen */
  PetscCall((*dmsnesvi1->coarsen)(dm1,comm,dm2));

  /* not sure why this extra reference is needed, but without the dm2 disappears too early */
  /* Updating the KSPCreateVecs() to avoid using DMGetGlobalVector() when matrix is available removes the need for this reference? */
  /*  PetscCall(PetscObjectReference((PetscObject)*dm2));*/

  /* need to set back global vectors in order to use the original injection */
  PetscCall(DMClearGlobalVectors(dm1));

  dm1->ops->createglobalvector = dmsnesvi1->createglobalvector;

  PetscCall(DMCreateGlobalVector(dm1,&finemarked));
  PetscCall(DMCreateGlobalVector(*dm2,&coarsemarked));

  /*
     fill finemarked with locations of inactive points
  */
  PetscCall(ISGetIndices(dmsnesvi1->inactive,&index));
  PetscCall(ISGetLocalSize(dmsnesvi1->inactive,&n));
  PetscCall(VecSet(finemarked,0.0));
  for (k=0; k<n; k++) {
    PetscCall(VecSetValue(finemarked,index[k],1.0,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(finemarked));
  PetscCall(VecAssemblyEnd(finemarked));

  PetscCall(DMCreateInjection(*dm2,dm1,&inject));
  PetscCall(MatRestrict(inject,finemarked,coarsemarked));
  PetscCall(MatDestroy(&inject));

  /*
     create index set list of coarse inactive points from coarsemarked
  */
  PetscCall(VecGetLocalSize(coarsemarked,&n));
  PetscCall(VecGetOwnershipRange(coarsemarked,&rstart,NULL));
  PetscCall(VecGetArray(coarsemarked,&marked));
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) cnt++;
  }
  PetscCall(PetscMalloc1(cnt,&coarseindex));
  cnt  = 0;
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) coarseindex[cnt++] = k + rstart;
  }
  PetscCall(VecRestoreArray(coarsemarked,&marked));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)coarsemarked),cnt,coarseindex,PETSC_OWN_POINTER,&inactive));

  PetscCall(DMClearGlobalVectors(dm1));

  dm1->ops->createglobalvector = DMCreateGlobalVector_SNESVI;

  PetscCall(DMSetVI(*dm2,inactive));

  PetscCall(VecDestroy(&finemarked));
  PetscCall(VecDestroy(&coarsemarked));
  PetscCall(ISDestroy(&inactive));
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
  PetscCall(DMClearGlobalVectors(dmsnesvi->dm));

  PetscCall(ISDestroy(&dmsnesvi->inactive));
  PetscCall(PetscFree(dmsnesvi));
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

  PetscCall(PetscObjectReference((PetscObject)inactive));

  PetscCall(PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes));
  if (!isnes) {
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)dm),&isnes));
    PetscCall(PetscContainerSetUserDestroy(isnes,(PetscErrorCode (*)(void*))DMDestroy_SNESVI));
    PetscCall(PetscNew(&dmsnesvi));
    PetscCall(PetscContainerSetPointer(isnes,(void*)dmsnesvi));
    PetscCall(PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)isnes));
    PetscCall(PetscContainerDestroy(&isnes));

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
    PetscCall(PetscContainerGetPointer(isnes,(void**)&dmsnesvi));
    PetscCall(ISDestroy(&dmsnesvi->inactive));
  }
  PetscCall(DMClearGlobalVectors(dm));
  PetscCall(ISGetLocalSize(inactive,&dmsnesvi->n));

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
  PetscCall(PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)NULL));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/

PetscErrorCode SNESCreateIndexSets_VINEWTONRSLS(SNES snes,Vec X,Vec F,IS *ISact,IS *ISinact)
{
  PetscFunctionBegin;
  PetscCall(SNESVIGetActiveSetIS(snes,X,F,ISact));
  PetscCall(ISComplement(*ISact,X->map->rstart,X->map->rend,ISinact));
  PetscFunctionReturn(0);
}

/* Create active and inactive set vectors. The local size of this vector is set and petsc computes the global size */
PetscErrorCode SNESCreateSubVectors_VINEWTONRSLS(SNES snes,PetscInt n,Vec *newv)
{
  Vec            v;

  PetscFunctionBegin;
  PetscCall(VecCreate(PetscObjectComm((PetscObject)snes),&v));
  PetscCall(VecSetSizes(v,n,PETSC_DECIDE));
  PetscCall(VecSetType(v,VECSTANDARD));
  *newv = v;
  PetscFunctionReturn(0);
}

/* Resets the snes PC and KSP when the active set sizes change */
PetscErrorCode SNESVIResetPCandKSP(SNES snes,Mat Amat,Mat Pmat)
{
  KSP            snesksp;

  PetscFunctionBegin;
  PetscCall(SNESGetKSP(snes,&snesksp));
  PetscCall(KSPReset(snesksp));
  PetscCall(KSPResetFromOptions(snesksp));

  /*
  KSP                    kspnew;
  PC                     pcnew;
  MatSolverType          stype;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)snes),&kspnew));
  kspnew->pc_side = snesksp->pc_side;
  kspnew->rtol    = snesksp->rtol;
  kspnew->abstol    = snesksp->abstol;
  kspnew->max_it  = snesksp->max_it;
  PetscCall(KSPSetType(kspnew,((PetscObject)snesksp)->type_name));
  PetscCall(KSPGetPC(kspnew,&pcnew));
  PetscCall(PCSetType(kspnew->pc,((PetscObject)snesksp->pc)->type_name));
  PetscCall(PCSetOperators(kspnew->pc,Amat,Pmat));
  PetscCall(PCFactorGetMatSolverType(snesksp->pc,&stype));
  PetscCall(PCFactorSetMatSolverType(kspnew->pc,stype));
  PetscCall(KSPDestroy(&snesksp));
  snes->ksp = kspnew;
  PetscCall(PetscLogObjectParent((PetscObject)snes,(PetscObject)kspnew));
   PetscCall(KSPSetFromOptions(kspnew));*/
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
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH));

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */

  PetscCall(SNESLineSearchSetVIFunctions(snes->linesearch, SNESVIProjectOntoBounds, SNESVIComputeInactiveSetFnorm));
  PetscCall(SNESLineSearchSetVecs(snes->linesearch, X, NULL, NULL, NULL, NULL));
  PetscCall(SNESLineSearchSetUp(snes->linesearch));

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  PetscCall(SNESVIProjectOntoBounds(snes,X));
  PetscCall(SNESComputeFunction(snes,X,F));
  PetscCall(SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm));
  PetscCall(VecNorm(X,NORM_2,&xnorm));        /* xnorm <- ||x||  */
  SNESCheckFunctionNorm(snes,fnorm);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes,fnorm,0));
  PetscCall(SNESMonitor(snes,0,fnorm));

  /* test convergence */
  PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
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
      PetscCall((*snes->ops->update)(snes, snes->iter));
    }
    PetscCall(SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre));
    SNESCheckJacobianDomainerror(snes);

    /* Create active and inactive index sets */

    /*original
    PetscCall(SNESVICreateIndexSets_RS(snes,X,F,&IS_act,&vi->IS_inact));
     */
    PetscCall(SNESVIGetActiveSetIS(snes,X,F,&IS_act));

    if (vi->checkredundancy) {
      PetscCall((*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP));
      if (IS_redact) {
        PetscCall(ISSort(IS_redact));
        PetscCall(ISComplement(IS_redact,X->map->rstart,X->map->rend,&vi->IS_inact));
        PetscCall(ISDestroy(&IS_redact));
      } else {
        PetscCall(ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact));
      }
    } else {
      PetscCall(ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact));
    }

    /* Create inactive set submatrix */
    PetscCall(MatCreateSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact));

    if (0) {                    /* Dead code (temporary developer hack) */
      IS keptrows;
      PetscCall(MatFindNonzeroRows(jac_inact_inact,&keptrows));
      if (keptrows) {
        PetscInt       cnt,*nrows,k;
        const PetscInt *krows,*inact;
        PetscInt       rstart;

        PetscCall(MatGetOwnershipRange(jac_inact_inact,&rstart,NULL));
        PetscCall(MatDestroy(&jac_inact_inact));
        PetscCall(ISDestroy(&IS_act));

        PetscCall(ISGetLocalSize(keptrows,&cnt));
        PetscCall(ISGetIndices(keptrows,&krows));
        PetscCall(ISGetIndices(vi->IS_inact,&inact));
        PetscCall(PetscMalloc1(cnt,&nrows));
        for (k=0; k<cnt; k++) nrows[k] = inact[krows[k]-rstart];
        PetscCall(ISRestoreIndices(keptrows,&krows));
        PetscCall(ISRestoreIndices(vi->IS_inact,&inact));
        PetscCall(ISDestroy(&keptrows));
        PetscCall(ISDestroy(&vi->IS_inact));

        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)snes),cnt,nrows,PETSC_OWN_POINTER,&vi->IS_inact));
        PetscCall(ISComplement(vi->IS_inact,F->map->rstart,F->map->rend,&IS_act));
        PetscCall(MatCreateSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact));
      }
    }
    PetscCall(DMSetVI(snes->dm,vi->IS_inact));
    /* remove later */

    /*
    PetscCall(VecView(vi->xu,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xu))->comm)));
    PetscCall(VecView(vi->xl,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xl))->comm)));
    PetscCall(VecView(X,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)X))));
    PetscCall(VecView(F,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)F))));
    PetscCall(ISView(vi->IS_inact,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)vi->IS_inact))));
     */

    /* Get sizes of active and inactive sets */
    PetscCall(ISGetLocalSize(IS_act,&nis_act));
    PetscCall(ISGetLocalSize(vi->IS_inact,&nis_inact));

    /* Create active and inactive set vectors */
    PetscCall(SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&F_inact));
    PetscCall(SNESCreateSubVectors_VINEWTONRSLS(snes,nis_act,&Y_act));
    PetscCall(SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&Y_inact));

    /* Create scatter contexts */
    PetscCall(VecScatterCreate(Y,IS_act,Y_act,NULL,&scat_act));
    PetscCall(VecScatterCreate(Y,vi->IS_inact,Y_inact,NULL,&scat_inact));

    /* Do a vec scatter to active and inactive set vectors */
    PetscCall(VecScatterBegin(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD));

    PetscCall(VecScatterBegin(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD));

    PetscCall(VecScatterBegin(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD));

    /* Active set direction = 0 */
    PetscCall(VecSet(Y_act,0));
    if (snes->jacobian != snes->jacobian_pre) {
      PetscCall(MatCreateSubMatrix(snes->jacobian_pre,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact));
    } else prejac_inact_inact = jac_inact_inact;

    PetscCall(ISEqual(vi->IS_inact_prev,vi->IS_inact,&isequal));
    if (!isequal) {
      PetscCall(SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact));
      PetscCall(PCFieldSplitRestrictIS(pc,vi->IS_inact));
    }

    /*      PetscCall(ISView(vi->IS_inact,0)); */
    /*      PetscCall(ISView(IS_act,0));*/
    /*      ierr = MatView(snes->jacobian_pre,0); */

    PetscCall(KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact));
    PetscCall(KSPSetUp(snes->ksp));
    {
      PC        pc;
      PetscBool flg;
      PetscCall(KSPGetPC(snes->ksp,&pc));
      PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&flg));
      if (flg) {
        KSP *subksps;
        PetscCall(PCFieldSplitGetSubKSP(pc,NULL,&subksps));
        PetscCall(KSPGetPC(subksps[0],&pc));
        PetscCall(PetscFree(subksps));
        PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg));
        if (flg) {
          PetscInt       n,N = 101*101,j,cnts[3] = {0,0,0};
          const PetscInt *ii;

          PetscCall(ISGetSize(vi->IS_inact,&n));
          PetscCall(ISGetIndices(vi->IS_inact,&ii));
          for (j=0; j<n; j++) {
            if (ii[j] < N) cnts[0]++;
            else if (ii[j] < 2*N) cnts[1]++;
            else if (ii[j] < 3*N) cnts[2]++;
          }
          PetscCall(ISRestoreIndices(vi->IS_inact,&ii));

          PetscCall(PCBJacobiSetTotalBlocks(pc,3,cnts));
        }
      }
    }

    PetscCall(KSPSolve(snes->ksp,F_inact,Y_inact));
    PetscCall(VecScatterBegin(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE));

    PetscCall(VecDestroy(&F_inact));
    PetscCall(VecDestroy(&Y_act));
    PetscCall(VecDestroy(&Y_inact));
    PetscCall(VecScatterDestroy(&scat_act));
    PetscCall(VecScatterDestroy(&scat_inact));
    PetscCall(ISDestroy(&IS_act));
    if (!isequal) {
      PetscCall(ISDestroy(&vi->IS_inact_prev));
      PetscCall(ISDuplicate(vi->IS_inact,&vi->IS_inact_prev));
    }
    PetscCall(ISDestroy(&vi->IS_inact));
    PetscCall(MatDestroy(&jac_inact_inact));
    if (snes->jacobian != snes->jacobian_pre) {
      PetscCall(MatDestroy(&prejac_inact_inact));
    }

    PetscCall(KSPGetConvergedReason(snes->ksp,&kspreason));
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        PetscCall(PetscInfo(snes,"iter=%" PetscInt_FMT ", number linear solve failures %" PetscInt_FMT " greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures));
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }

    PetscCall(KSPGetIterationNumber(snes->ksp,&lits));
    snes->linear_its += lits;
    PetscCall(PetscInfo(snes,"iter=%" PetscInt_FMT ", linear solve iterations=%" PetscInt_FMT "\n",snes->iter,lits));
    /*
    if (snes->ops->precheck) {
      PetscBool changed_y = PETSC_FALSE;
      PetscCall((*snes->ops->precheck)(snes,X,Y,snes->precheck,&changed_y));
    }

    if (PetscLogPrintInfo) {
      PetscCall(SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W));
    }
    */
    /* Compute a (scaled) negative update in the line search routine:
         Y <- X - lambda*Y
       and evaluate G = function(Y) (depends on the line search).
    */
    PetscCall(VecCopy(Y,snes->vec_sol_update));
    ynorm = 1; gnorm = fnorm;
    PetscCall(SNESLineSearchApply(snes->linesearch, X, F, &gnorm, Y));
    PetscCall(SNESLineSearchGetReason(snes->linesearch, &lssucceed));
    PetscCall(SNESLineSearchGetNorms(snes->linesearch, &xnorm, &gnorm, &ynorm));
    PetscCall(PetscInfo(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed));
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscCall(DMDestroyVI(snes->dm));
      PetscFunctionReturn(0);
    }
    if (lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscCall(SNESVICheckLocalMin_Private(snes,snes->jacobian,F,X,gnorm,&ismin));
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
   }
   PetscCall(DMDestroyVI(snes->dm));
    /* Update function and solution vectors */
    fnorm = gnorm;
    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,lits));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESConvergedSkip) PetscCall(VecNorm(X,NORM_2,&xnorm));
    PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;
  }
  /* make sure that the VI information attached to the DM is removed if the for loop above was broken early due to some exceptional conditional */
  PetscCall(DMDestroyVI(snes->dm));
  if (i == maxits) {
    PetscCall(PetscInfo(snes,"Maximum number of iterations has been reached: %" PetscInt_FMT "\n",maxits));
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
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)snes),0,indices,PETSC_OWN_POINTER,is_redact));
  /* call Matlab function in ctx */
  PetscCall(PetscArraycpy(&ls,&snes,1));
  PetscCall(PetscArraycpy(&l1,&is_act,1));
  PetscCall(PetscArraycpy(&l2,is_redact,1));
  prhs[0] = mxCreateDoubleScalar((double)ls);
  prhs[1] = mxCreateDoubleScalar((double)l1);
  prhs[2] = mxCreateDoubleScalar((double)l2);
  prhs[3] = mxCreateString(sctx->funcname);
  prhs[4] = sctx->ctx;
  PetscCall(mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESVIRedundancyCheckInternal"));
  PetscCall(mxGetScalar(plhs[0]));
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
  PetscCall(PetscNew(&sctx));
  PetscCall(PetscStrallocpy(func,&sctx->funcname));
  sctx->ctx = mxDuplicateArray(ctx);
  PetscCall(SNESVISetRedundancyCheck(snes,SNESVIRedundancyCheck_Matlab,sctx));
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
  PetscCall(SNESSetUp_VI(snes));

  /* Set up previous active index set for the first snes solve
   vi->IS_inact_prev = 0,1,2,....N */

  PetscCall(VecGetOwnershipRange(snes->vec_sol,&rstart,&rend));
  PetscCall(VecGetLocalSize(snes->vec_sol,&n));
  PetscCall(PetscMalloc1(n,&indices));
  for (i=0; i < n; i++) indices[i] = rstart + i;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)snes),n,indices,PETSC_OWN_POINTER,&vi->IS_inact_prev));

  /* set the line search functions */
  if (!snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes, &linesearch));
    PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBT));
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode SNESReset_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;

  PetscFunctionBegin;
  PetscCall(SNESReset_VI(snes));
  PetscCall(ISDestroy(&vi->IS_inact_prev));
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

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBT));
  }
  PetscCall(SNESLineSearchBTSetAlpha(linesearch, 0.0));

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNewLog(snes,&vi));
  snes->data          = (void*)vi;
  vi->checkredundancy = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESVISetVariableBounds_C",SNESVISetVariableBounds_VI));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",SNESVISetComputeVariableBounds_VI));
  PetscFunctionReturn(0);
}
