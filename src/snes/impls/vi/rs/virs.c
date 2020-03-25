
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

<*/
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
  PetscErrorCode ierr;
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Composed SNES is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)dm),dmsnesvi->n,PETSC_DETERMINE,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     DMCreateInterpolation_SNESVI - Modifieds the interpolation obtained from the DM by removing all rows and columns associated with active constraints.

*/
PetscErrorCode  DMCreateInterpolation_SNESVI(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  PetscErrorCode ierr;
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi1,*dmsnesvi2;
  Mat            interp;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi1);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm2,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm2),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi2);CHKERRQ(ierr);

  ierr = (*dmsnesvi1->createinterpolation)(dm1,dm2,&interp,NULL);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(interp,dmsnesvi2->inactive,dmsnesvi1->inactive,MAT_INITIAL_MATRIX,mat);CHKERRQ(ierr);
  ierr = MatDestroy(&interp);CHKERRQ(ierr);
  *vec = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetVI(DM,IS);
static PetscErrorCode DMDestroyVI(DM);

/*
     DMCoarsen_SNESVI - Computes the regular coarsened DM then computes additional information about its inactive set

*/
PetscErrorCode  DMCoarsen_SNESVI(DM dm1,MPI_Comm comm,DM *dm2)
{
  PetscErrorCode ierr;
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi1;
  Vec            finemarked,coarsemarked;
  IS             inactive;
  Mat            inject;
  const PetscInt *index;
  PetscInt       n,k,cnt = 0,rstart,*coarseindex;
  PetscScalar    *marked;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi1);CHKERRQ(ierr);

  /* get the original coarsen */
  ierr = (*dmsnesvi1->coarsen)(dm1,comm,dm2);CHKERRQ(ierr);

  /* not sure why this extra reference is needed, but without the dm2 disappears too early */
  /* Updating the KSPCreateVecs() to avoid using DMGetGlobalVector() when matrix is available removes the need for this reference? */
  /*  ierr = PetscObjectReference((PetscObject)*dm2);CHKERRQ(ierr);*/

  /* need to set back global vectors in order to use the original injection */
  ierr = DMClearGlobalVectors(dm1);CHKERRQ(ierr);

  dm1->ops->createglobalvector = dmsnesvi1->createglobalvector;

  ierr = DMCreateGlobalVector(dm1,&finemarked);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(*dm2,&coarsemarked);CHKERRQ(ierr);

  /*
     fill finemarked with locations of inactive points
  */
  ierr = ISGetIndices(dmsnesvi1->inactive,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(dmsnesvi1->inactive,&n);CHKERRQ(ierr);
  ierr = VecSet(finemarked,0.0);CHKERRQ(ierr);
  for (k=0; k<n; k++) {
    ierr = VecSetValue(finemarked,index[k],1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(finemarked);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(finemarked);CHKERRQ(ierr);

  ierr = DMCreateInjection(*dm2,dm1,&inject);CHKERRQ(ierr);
  ierr = MatRestrict(inject,finemarked,coarsemarked);CHKERRQ(ierr);
  ierr = MatDestroy(&inject);CHKERRQ(ierr);

  /*
     create index set list of coarse inactive points from coarsemarked
  */
  ierr = VecGetLocalSize(coarsemarked,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(coarsemarked,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(coarsemarked,&marked);CHKERRQ(ierr);
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) cnt++;
  }
  ierr = PetscMalloc1(cnt,&coarseindex);CHKERRQ(ierr);
  cnt  = 0;
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) coarseindex[cnt++] = k + rstart;
  }
  ierr = VecRestoreArray(coarsemarked,&marked);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)coarsemarked),cnt,coarseindex,PETSC_OWN_POINTER,&inactive);CHKERRQ(ierr);

  ierr = DMClearGlobalVectors(dm1);CHKERRQ(ierr);

  dm1->ops->createglobalvector = DMCreateGlobalVector_SNESVI;

  ierr = DMSetVI(*dm2,inactive);CHKERRQ(ierr);

  ierr = VecDestroy(&finemarked);CHKERRQ(ierr);
  ierr = VecDestroy(&coarsemarked);CHKERRQ(ierr);
  ierr = ISDestroy(&inactive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_SNESVI(DM_SNESVI *dmsnesvi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* reset the base methods in the DM object that were changed when the DM_SNESVI was reset */
  dmsnesvi->dm->ops->createinterpolation = dmsnesvi->createinterpolation;
  dmsnesvi->dm->ops->coarsen             = dmsnesvi->coarsen;
  dmsnesvi->dm->ops->createglobalvector  = dmsnesvi->createglobalvector;
  dmsnesvi->dm->ops->createinjection     = dmsnesvi->createinjection;
  dmsnesvi->dm->ops->hascreateinjection  = dmsnesvi->hascreateinjection;
  /* need to clear out this vectors because some of them may not have a reference to the DM
    but they are counted as having references to the DM in DMDestroy() */
  ierr = DMClearGlobalVectors(dmsnesvi->dm);CHKERRQ(ierr);

  ierr = ISDestroy(&dmsnesvi->inactive);CHKERRQ(ierr);
  ierr = PetscFree(dmsnesvi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     DMSetVI - Marks a DM as associated with a VI problem. This causes the interpolation/restriction operators to
               be restricted to only those variables NOT associated with active constraints.

*/
static PetscErrorCode DMSetVI(DM dm,IS inactive)
{
  PetscErrorCode ierr;
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);

  ierr = PetscObjectReference((PetscObject)inactive);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) {
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)dm),&isnes);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(isnes,(PetscErrorCode (*)(void*))DMDestroy_SNESVI);CHKERRQ(ierr);
    ierr = PetscNew(&dmsnesvi);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(isnes,(void*)dmsnesvi);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)isnes);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&isnes);CHKERRQ(ierr);

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
    ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi);CHKERRQ(ierr);
    ierr = ISDestroy(&dmsnesvi->inactive);CHKERRQ(ierr);
  }
  ierr = DMClearGlobalVectors(dm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(inactive,&dmsnesvi->n);CHKERRQ(ierr);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);
  ierr = PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/


PetscErrorCode SNESCreateIndexSets_VINEWTONRSLS(SNES snes,Vec X,Vec F,IS *ISact,IS *ISinact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESVIGetActiveSetIS(snes,X,F,ISact);CHKERRQ(ierr);
  ierr = ISComplement(*ISact,X->map->rstart,X->map->rend,ISinact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create active and inactive set vectors. The local size of this vector is set and petsc computes the global size */
PetscErrorCode SNESCreateSubVectors_VINEWTONRSLS(SNES snes,PetscInt n,Vec *newv)
{
  PetscErrorCode ierr;
  Vec            v;

  PetscFunctionBegin;
  ierr  = VecCreate(PetscObjectComm((PetscObject)snes),&v);CHKERRQ(ierr);
  ierr  = VecSetSizes(v,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr  = VecSetType(v,VECSTANDARD);CHKERRQ(ierr);
  *newv = v;
  PetscFunctionReturn(0);
}

/* Resets the snes PC and KSP when the active set sizes change */
PetscErrorCode SNESVIResetPCandKSP(SNES snes,Mat Amat,Mat Pmat)
{
  PetscErrorCode ierr;
  KSP            snesksp;

  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&snesksp);CHKERRQ(ierr);
  ierr = KSPReset(snesksp);CHKERRQ(ierr);
  ierr = KSPResetFromOptions(snesksp);CHKERRQ(ierr);

  /*
  KSP                    kspnew;
  PC                     pcnew;
  MatSolverType          stype;


  ierr = KSPCreate(PetscObjectComm((PetscObject)snes),&kspnew);CHKERRQ(ierr);
  kspnew->pc_side = snesksp->pc_side;
  kspnew->rtol    = snesksp->rtol;
  kspnew->abstol    = snesksp->abstol;
  kspnew->max_it  = snesksp->max_it;
  ierr = KSPSetType(kspnew,((PetscObject)snesksp)->type_name);CHKERRQ(ierr);
  ierr = KSPGetPC(kspnew,&pcnew);CHKERRQ(ierr);
  ierr = PCSetType(kspnew->pc,((PetscObject)snesksp->pc)->type_name);CHKERRQ(ierr);
  ierr = PCSetOperators(kspnew->pc,Amat,Pmat);CHKERRQ(ierr);
  ierr = PCFactorGetMatSolverType(snesksp->pc,&stype);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverType(kspnew->pc,stype);CHKERRQ(ierr);
  ierr = KSPDestroy(&snesksp);CHKERRQ(ierr);
  snes->ksp = kspnew;
  ierr = PetscLogObjectParent((PetscObject)snes,(PetscObject)kspnew);CHKERRQ(ierr);
   ierr = KSPSetFromOptions(kspnew);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

/* Variational Inequality solver using reduce space method. No semismooth algorithm is
   implemented in this algorithm. It basically identifies the active constraints and does
   a linear solve on the other variables (those not associated with the active constraints). */
PetscErrorCode SNESSolve_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS    *vi = (SNES_VINEWTONRSLS*)snes->data;
  PetscErrorCode       ierr;
  PetscInt             maxits,i,lits;
  SNESLineSearchReason lssucceed;
  PetscReal            fnorm,gnorm,xnorm=0,ynorm;
  Vec                  Y,X,F;
  KSPConvergedReason   kspreason;
  KSP                  ksp;
  PC                   pc;

  PetscFunctionBegin;
  /* Multigrid must use Galerkin for coarse grids with active set/reduced space methods; cannot rediscretize on coarser grids*/
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */

  ierr = SNESLineSearchSetVIFunctions(snes->linesearch, SNESVIProjectOntoBounds, SNESVIComputeInactiveSetFnorm);CHKERRQ(ierr);
  ierr = SNESLineSearchSetVecs(snes->linesearch, X, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchSetUp(snes->linesearch);CHKERRQ(ierr);

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  ierr = SNESVIProjectOntoBounds(snes,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  ierr = SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm);CHKERRQ(ierr);
  ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);        /* xnorm <- ||x||  */
  SNESCheckFunctionNorm(snes,fnorm);
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
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
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    SNESCheckJacobianDomainerror(snes);

    /* Create active and inactive index sets */

    /*original
    ierr = SNESVICreateIndexSets_RS(snes,X,F,&IS_act,&vi->IS_inact);CHKERRQ(ierr);
     */
    ierr = SNESVIGetActiveSetIS(snes,X,F,&IS_act);CHKERRQ(ierr);

    if (vi->checkredundancy) {
      (*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP);CHKERRQ(ierr);
      if (IS_redact) {
        ierr = ISSort(IS_redact);CHKERRQ(ierr);
        ierr = ISComplement(IS_redact,X->map->rstart,X->map->rend,&vi->IS_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_redact);CHKERRQ(ierr);
      } else {
        ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact);CHKERRQ(ierr);
      }
    } else {
      ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact);CHKERRQ(ierr);
    }


    /* Create inactive set submatrix */
    ierr = MatCreateSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);

    if (0) {                    /* Dead code (temporary developer hack) */
      IS keptrows;
      ierr = MatFindNonzeroRows(jac_inact_inact,&keptrows);CHKERRQ(ierr);
      if (keptrows) {
        PetscInt       cnt,*nrows,k;
        const PetscInt *krows,*inact;
        PetscInt       rstart;

        ierr = MatGetOwnershipRange(jac_inact_inact,&rstart,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_act);CHKERRQ(ierr);

        ierr = ISGetLocalSize(keptrows,&cnt);CHKERRQ(ierr);
        ierr = ISGetIndices(keptrows,&krows);CHKERRQ(ierr);
        ierr = ISGetIndices(vi->IS_inact,&inact);CHKERRQ(ierr);
        ierr = PetscMalloc1(cnt,&nrows);CHKERRQ(ierr);
        for (k=0; k<cnt; k++) nrows[k] = inact[krows[k]-rstart];
        ierr = ISRestoreIndices(keptrows,&krows);CHKERRQ(ierr);
        ierr = ISRestoreIndices(vi->IS_inact,&inact);CHKERRQ(ierr);
        ierr = ISDestroy(&keptrows);CHKERRQ(ierr);
        ierr = ISDestroy(&vi->IS_inact);CHKERRQ(ierr);

        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)snes),cnt,nrows,PETSC_OWN_POINTER,&vi->IS_inact);CHKERRQ(ierr);
        ierr = ISComplement(vi->IS_inact,F->map->rstart,F->map->rend,&IS_act);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);
      }
    }
    ierr = DMSetVI(snes->dm,vi->IS_inact);CHKERRQ(ierr);
    /* remove later */

    /*
    ierr = VecView(vi->xu,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xu))->comm));CHKERRQ(ierr);
    ierr = VecView(vi->xl,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xl))->comm));CHKERRQ(ierr);
    ierr = VecView(X,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)X)));CHKERRQ(ierr);
    ierr = VecView(F,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)F)));CHKERRQ(ierr);
    ierr = ISView(vi->IS_inact,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)vi->IS_inact)));CHKERRQ(ierr);
     */

    /* Get sizes of active and inactive sets */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    ierr = ISGetLocalSize(vi->IS_inact,&nis_inact);CHKERRQ(ierr);

    /* Create active and inactive set vectors */
    ierr = SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&F_inact);CHKERRQ(ierr);
    ierr = SNESCreateSubVectors_VINEWTONRSLS(snes,nis_act,&Y_act);CHKERRQ(ierr);
    ierr = SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&Y_inact);CHKERRQ(ierr);

    /* Create scatter contexts */
    ierr = VecScatterCreate(Y,IS_act,Y_act,NULL,&scat_act);CHKERRQ(ierr);
    ierr = VecScatterCreate(Y,vi->IS_inact,Y_inact,NULL,&scat_inact);CHKERRQ(ierr);

    /* Do a vec scatter to active and inactive set vectors */
    ierr = VecScatterBegin(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* Active set direction = 0 */
    ierr = VecSet(Y_act,0);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatCreateSubMatrix(snes->jacobian_pre,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact);CHKERRQ(ierr);
    } else prejac_inact_inact = jac_inact_inact;

    ierr = ISEqual(vi->IS_inact_prev,vi->IS_inact,&isequal);CHKERRQ(ierr);
    if (!isequal) {
      ierr = SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
      ierr = PCFieldSplitRestrictIS(pc,vi->IS_inact);CHKERRQ(ierr);
    }

    /*      ierr = ISView(vi->IS_inact,0);CHKERRQ(ierr); */
    /*      ierr = ISView(IS_act,0);CHKERRQ(ierr);*/
    /*      ierr = MatView(snes->jacobian_pre,0); */



    ierr = KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
    ierr = KSPSetUp(snes->ksp);CHKERRQ(ierr);
    {
      PC        pc;
      PetscBool flg;
      ierr = KSPGetPC(snes->ksp,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&flg);CHKERRQ(ierr);
      if (flg) {
        KSP *subksps;
        ierr = PCFieldSplitGetSubKSP(pc,NULL,&subksps);CHKERRQ(ierr);
        ierr = KSPGetPC(subksps[0],&pc);CHKERRQ(ierr);
        ierr = PetscFree(subksps);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg);CHKERRQ(ierr);
        if (flg) {
          PetscInt       n,N = 101*101,j,cnts[3] = {0,0,0};
          const PetscInt *ii;

          ierr = ISGetSize(vi->IS_inact,&n);CHKERRQ(ierr);
          ierr = ISGetIndices(vi->IS_inact,&ii);CHKERRQ(ierr);
          for (j=0; j<n; j++) {
            if (ii[j] < N) cnts[0]++;
            else if (ii[j] < 2*N) cnts[1]++;
            else if (ii[j] < 3*N) cnts[2]++;
          }
          ierr = ISRestoreIndices(vi->IS_inact,&ii);CHKERRQ(ierr);

          ierr = PCBJacobiSetTotalBlocks(pc,3,cnts);CHKERRQ(ierr);
        }
      }
    }

    ierr = KSPSolve(snes->ksp,F_inact,Y_inact);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecDestroy(&F_inact);CHKERRQ(ierr);
    ierr = VecDestroy(&Y_act);CHKERRQ(ierr);
    ierr = VecDestroy(&Y_inact);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_act);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_inact);CHKERRQ(ierr);
    ierr = ISDestroy(&IS_act);CHKERRQ(ierr);
    if (!isequal) {
      ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
      ierr = ISDuplicate(vi->IS_inact,&vi->IS_inact_prev);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&vi->IS_inact);CHKERRQ(ierr);
    ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatDestroy(&prejac_inact_inact);CHKERRQ(ierr);
    }

    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr         = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }

    ierr              = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr              = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (snes->ops->precheck) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheck)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo) {
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine:
         Y <- X - lambda*Y
       and evaluate G = function(Y) (depends on the line search).
    */
    ierr  = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr  = SNESLineSearchApply(snes->linesearch, X, F, &gnorm, Y);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetReason(snes->linesearch, &lssucceed);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
    ierr  = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      ierr         = DMDestroyVI(snes->dm);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr         = SNESVICheckLocalMin_Private(snes,snes->jacobian,F,X,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
   }
   ierr = DMDestroyVI(snes->dm);CHKERRQ(ierr);
    /* Update function and solution vectors */
    fnorm = gnorm;
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESConvergedSkip) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  /* make sure that the VI information attached to the DM is removed if the for loop above was broken early due to some exceptional conditional */
  ierr = DMDestroyVI(snes->dm);CHKERRQ(ierr);
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
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
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)snes),0,indices,PETSC_OWN_POINTER,is_redact);CHKERRQ(ierr);
  /* call Matlab function in ctx */
  ierr    = PetscArraycpy(&ls,&snes,1);CHKERRQ(ierr);
  ierr    = PetscArraycpy(&l1,&is_act,1);CHKERRQ(ierr);
  ierr    = PetscArraycpy(&l2,is_redact,1);CHKERRQ(ierr);
  prhs[0] = mxCreateDoubleScalar((double)ls);
  prhs[1] = mxCreateDoubleScalar((double)l1);
  prhs[2] = mxCreateDoubleScalar((double)l2);
  prhs[3] = mxCreateString(sctx->funcname);
  prhs[4] = sctx->ctx;
  ierr    = mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESVIRedundancyCheckInternal");CHKERRQ(ierr);
  ierr    = mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVISetRedundancyCheckMatlab(SNES snes,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr      = PetscNew(&sctx);CHKERRQ(ierr);
  ierr      = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  sctx->ctx = mxDuplicateArray(ctx);
  ierr      = SNESVISetRedundancyCheck(snes,SNESVIRedundancyCheck_Matlab,sctx);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;
  PetscInt          *indices;
  PetscInt          i,n,rstart,rend;
  SNESLineSearch    linesearch;

  PetscFunctionBegin;
  ierr = SNESSetUp_VI(snes);CHKERRQ(ierr);

  /* Set up previous active index set for the first snes solve
   vi->IS_inact_prev = 0,1,2,....N */

  ierr = VecGetOwnershipRange(snes->vec_sol,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(snes->vec_sol,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);
  for (i=0; i < n; i++) indices[i] = rstart + i;
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)snes),n,indices,PETSC_OWN_POINTER,&vi->IS_inact_prev);CHKERRQ(ierr);

  /* set the line search functions */
  if (!snes->linesearch) {
    ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode SNESReset_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SNESReset_VI(snes);CHKERRQ(ierr);
  ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
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
.  1. - T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNESVISetVariableBounds(), SNESVISetComputeVariableBounds(), SNESCreate(), SNES, SNESSetType(), SNESVINEWTONSSLS, SNESNEWTONTR, SNESLineSearchSetType(),SNESLineSearchSetPostCheck(), SNESLineSearchSetPreCheck()

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_VINEWTONRSLS(SNES snes)
{
  PetscErrorCode    ierr;
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

  ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);CHKERRQ(ierr);
  ierr = SNESLineSearchBTSetAlpha(linesearch, 0.0);CHKERRQ(ierr);

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  ierr                = PetscNewLog(snes,&vi);CHKERRQ(ierr);
  snes->data          = (void*)vi;
  vi->checkredundancy = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESVISetVariableBounds_C",SNESVISetVariableBounds_VI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",SNESVISetComputeVariableBounds_VI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

