
#include <../src/snes/impls/vi/rsaug/virsaugimpl.h> /*I "petscsnes.h" I*/
#include <../include/petsc-private/kspimpl.h>
#include <../include/petsc-private/matimpl.h>
#include <../include/petsc-private/dmimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGSetComputeVariableBounds"
/*@C
   SNESVIRSAUGSetComputeVariableBounds - Sets a function  that is called to compute the variable bounds

   Input parameter
+  snes - the SNES context
-  compute - computes the bounds

   Level: advanced

@*/
PetscErrorCode SNESVIRSAUGSetComputeVariableBounds(SNES snes, PetscErrorCode (*compute)(SNES,Vec,Vec))
{
  PetscErrorCode   ierr;
  SNES_VIRSAUG       *vi;

  PetscFunctionBegin;
  ierr = SNESSetType(snes,SNESVIRS);CHKERRQ(ierr);
  vi = (SNES_VIRSAUG*)snes->data;
  vi->computevariablebounds = compute;
  PetscFunctionReturn(0);
}
  

#if 0
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeInactiveSetIS"
/*
   SNESVIComputeInactiveSetIS - Gets the global indices for the bogus inactive set variables

   Input parameter
.  snes - the SNES context
.  X    - the snes solution vector

   Output parameter
.  ISact - active set index set

 */
PetscErrorCode SNESVIComputeInactiveSetIS(Vec upper,Vec lower,Vec X,Vec F,IS* inact)
{
  PetscErrorCode   ierr;
  const PetscScalar *x,*xl,*xu,*f;
  PetscInt          *idx_act,i,nlocal,nloc_isact=0,ilow,ihigh,i1=0;
  
  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&ilow,&ihigh);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(lower,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(upper,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F,&f);CHKERRQ(ierr);
  /* Compute inactive set size */
  for (i=0; i < nlocal;i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) nloc_isact++;
  }

  ierr = PetscMalloc(nloc_isact*sizeof(PetscInt),&idx_act);CHKERRQ(ierr);

  /* Set inactive set indices */
  for (i=0; i < nlocal; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) idx_act[i1++] = ilow+i;
  }

   /* Create inactive set IS */
  ierr = ISCreateGeneral(((PetscObject)upper)->comm,nloc_isact,idx_act,PETSC_OWN_POINTER,inact);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(lower,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(upper,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/*
    Provides a wrapper to a DM to allow it to be used to generated the interpolation/restriction from the DM for the smaller matrices and vectors 
  defined by the reduced space method. 

    Simple calls the regular DM interpolation and restricts it to operation on the variables not associated with active constraints.

<*/
typedef struct {
  PetscInt       n;                                        /* size of vectors in the reduced DM space */
  IS             inactive;
  PetscErrorCode (*createinterpolation)(DM,DM,Mat*,Vec*);    /* DM's original routines */
  PetscErrorCode (*coarsen)(DM, MPI_Comm, DM*); 
  PetscErrorCode (*createglobalvector)(DM,Vec*);
  DM             dm;                                      /* when destroying this object we need to reset the above function into the base DM */
} DM_SNESVI;

#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector_SNESVIRSAUG"
/*
     DMCreateGlobalVector_SNESVIRSAUG - Creates global vector of the size of the reduced space

*/
PetscErrorCode  DMCreateGlobalVector_SNESVIRSAUG(DM dm,Vec *vec)
{
  PetscErrorCode          ierr;
  PetscContainer          isnes;
  DM_SNESVI               *dmsnesvi;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm,"VI",(PetscObject *)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_PLIB,"Composed SNES is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi);CHKERRQ(ierr);
  ierr = VecCreateMPI(((PetscObject)dm)->comm,dmsnesvi->n,PETSC_DETERMINE,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_SNESVIRSAUG"
/*
     DMCreateInterpolation_SNESVIRSAUG - Modifieds the interpolation obtained from the DM by removing all rows and columns associated with active constraints.

*/
PetscErrorCode  DMCreateInterpolation_SNESVIRSAUG(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  PetscErrorCode          ierr;
  PetscContainer          isnes;
  DM_SNESVI               *dmsnesvi1,*dmsnesvi2;
  Mat                     interp;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject *)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(((PetscObject)dm1)->comm,PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi1);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm2,"VI",(PetscObject *)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(((PetscObject)dm2)->comm,PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi2);CHKERRQ(ierr);
  
  ierr = (*dmsnesvi1->createinterpolation)(dm1,dm2,&interp,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(interp,dmsnesvi2->inactive,dmsnesvi1->inactive,MAT_INITIAL_MATRIX,mat);CHKERRQ(ierr);
  ierr = MatDestroy(&interp);CHKERRQ(ierr);
  *vec = 0;
  PetscFunctionReturn(0);
}

extern PetscErrorCode  DMSetVI(DM,IS);

#undef __FUNCT__  
#define __FUNCT__ "DMCoarsen_SNESVIRSAUG"
/*
     DMCoarsen_SNESVIRSAUG - Computes the regular coarsened DM then computes additional information about its inactive set

*/
PetscErrorCode  DMCoarsen_SNESVIRSAUG(DM dm1,MPI_Comm comm,DM *dm2)
{
  PetscErrorCode          ierr;
  PetscContainer          isnes;
  DM_SNESVI               *dmsnesvi1;
  Vec                     finemarked,coarsemarked;
  IS                      inactive;
  VecScatter              inject;
  const PetscInt          *index;
  PetscInt                n,k,cnt = 0,rstart,*coarseindex;
  PetscScalar             *marked;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject *)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(((PetscObject)dm1)->comm,PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi1);CHKERRQ(ierr);
  
  /* get the original coarsen */
  ierr = (*dmsnesvi1->coarsen)(dm1,comm,dm2);CHKERRQ(ierr);

  /* not sure why this extra reference is needed, but without the dm2 disappears too early */
  ierr = PetscObjectReference((PetscObject)*dm2);CHKERRQ(ierr);

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
  for (k=0;k<n;k++){
      ierr = VecSetValue(finemarked,index[k],1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(finemarked);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(finemarked);CHKERRQ(ierr);

  ierr = DMCreateInjection(*dm2,dm1,&inject);CHKERRQ(ierr);
  ierr = VecScatterBegin(inject,finemarked,coarsemarked,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(inject,finemarked,coarsemarked,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&inject);CHKERRQ(ierr);

  /*
     create index set list of coarse inactive points from coarsemarked
  */
  ierr = VecGetLocalSize(coarsemarked,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(coarsemarked,&rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(coarsemarked,&marked);CHKERRQ(ierr);
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) cnt++;
  }
  ierr = PetscMalloc(cnt*sizeof(PetscInt),&coarseindex);CHKERRQ(ierr);
  cnt  = 0;
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) coarseindex[cnt++] = k + rstart;
  }
  ierr = VecRestoreArray(coarsemarked,&marked);CHKERRQ(ierr);
  ierr = ISCreateGeneral(((PetscObject)coarsemarked)->comm,cnt,coarseindex,PETSC_OWN_POINTER,&inactive);CHKERRQ(ierr);

  ierr = DMClearGlobalVectors(dm1);CHKERRQ(ierr);
  dm1->ops->createglobalvector = DMCreateGlobalVector_SNESVIRSAUG;
  ierr = DMSetVI(*dm2,inactive);CHKERRQ(ierr);

  ierr = VecDestroy(&finemarked);CHKERRQ(ierr);
  ierr = VecDestroy(&coarsemarked);CHKERRQ(ierr);
  ierr = ISDestroy(&inactive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy_SNESVIRSAUG"
PetscErrorCode DMDestroy_SNESVIRSAUG(DM_SNESVI *dmsnesvi)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  /* reset the base methods in the DM object that were changed when the DM_SNESVI was reset */
  dmsnesvi->dm->ops->createinterpolation   = dmsnesvi->createinterpolation;
  dmsnesvi->dm->ops->coarsen            = dmsnesvi->coarsen;
  dmsnesvi->dm->ops->createglobalvector = dmsnesvi->createglobalvector;
  /* need to clear out this vectors because some of them may not have a reference to the DM
    but they are counted as having references to the DM in DMDestroy() */
  ierr = DMClearGlobalVectors(dmsnesvi->dm);CHKERRQ(ierr);

  ierr = ISDestroy(&dmsnesvi->inactive);CHKERRQ(ierr);
  ierr = PetscFree(dmsnesvi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetVIRSAUG"
/*
     DMSetVI - Marks a DM as associated with a VI problem. This causes the interpolation/restriction operators to 
               be restricted to only those variables NOT associated with active constraints.

*/
PetscErrorCode  DMSetVIRSAUG(DM dm,IS inactive)
{
  PetscErrorCode          ierr;
  PetscContainer          isnes;
  DM_SNESVI               *dmsnesvi;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);

  ierr = PetscObjectReference((PetscObject)inactive);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject)dm,"VI",(PetscObject *)&isnes);CHKERRQ(ierr);
  if (!isnes) {
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&isnes);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(isnes,(PetscErrorCode (*)(void*))DMDestroy_SNESVIRSAUG);CHKERRQ(ierr);
    ierr = PetscNew(DM_SNESVI,&dmsnesvi);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(isnes,(void*)dmsnesvi);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)isnes);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&isnes);CHKERRQ(ierr);
    dmsnesvi->createinterpolation   = dm->ops->createinterpolation;
    dm->ops->createinterpolation    = DMCreateInterpolation_SNESVIRSAUG;
    dmsnesvi->coarsen            = dm->ops->coarsen;
    dm->ops->coarsen             = DMCoarsen_SNESVIRSAUG;
    dmsnesvi->createglobalvector = dm->ops->createglobalvector;
    dm->ops->createglobalvector  = DMCreateGlobalVector_SNESVIRSAUG;
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

#undef __FUNCT__  
#define __FUNCT__ "DMDestroyVIRSAUG"
/*
     DMDestroyVIRSAUG - Frees the DM_SNESVI object contained in the DM 
         - also resets the function pointers in the DM for createinterpolation() etc to use the original DM 
*/
PetscErrorCode  DMDestroyVIRSAUG(DM dm)
{
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);
  ierr = PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorVIRSAUG"
PetscErrorCode  SNESMonitorVIRSAUG(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode    ierr;
  SNES_VIRSAUG          *vi = (SNES_VIRSAUG*)snes->data;
  PetscViewer        viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);
  const PetscScalar  *x,*xl,*xu,*f;
  PetscInt           i,n,act[2] = {0,0},fact[2],N;
  /* Number of components that actually hit the bounds (c.f. active variables) */
  PetscInt           act_bound[2] = {0,0},fact_bound[2];
  PetscReal          rnorm,fnorm;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(snes->vec_sol,&n);CHKERRQ(ierr);
  ierr = VecGetSize(snes->vec_sol,&N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_sol,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_func,&f);CHKERRQ(ierr);
  
  rnorm = 0.0;
  for (i=0; i<n; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
    else if (PetscRealPart(x[i]) <= PetscRealPart(xl[i]) + 1.e-8 && PetscRealPart(f[i]) >= 0.0) act[0]++;
    else if (PetscRealPart(x[i]) >= PetscRealPart(xu[i]) - 1.e-8 && PetscRealPart(f[i]) <= 0.0) act[1]++;
    else SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"Can never get here");
  }

  for (i=0; i<n; i++) {
    if (PetscRealPart(x[i]) <= PetscRealPart(xl[i]) + 1.e-8) act_bound[0]++; 
    else if (PetscRealPart(x[i]) >= PetscRealPart(xu[i]) - 1.e-8) act_bound[1]++;
  }
  ierr = VecRestoreArrayRead(snes->vec_func,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->vec_sol,&x);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&rnorm,&fnorm,1,MPIU_REAL,MPIU_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(act,fact,2,MPIU_INT,MPI_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(act_bound,fact_bound,2,MPIU_INT,MPI_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  fnorm = PetscSqrtReal(fnorm);
  
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES VI Function norm %14.12e Active lower constraints %D/%D upper constraints %D/%D Percent of total %g Percent of bounded %g\n",its,(double)fnorm,fact[0],fact_bound[0],fact[1],fact_bound[1],((double)(fact[0]+fact[1]))/((double)N),((double)(fact[0]+fact[1]))/((double)vi->ntruebounds));CHKERRQ(ierr);
  
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the norm of the function,
    || F(u) ||_2 but not a zero, F(u) = 0. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More 
    for this trick. One assumes that the probability that W is in the null space of J is very, very small.
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESVIRSAUGCheckLocalMin_Private"
PetscErrorCode SNESVIRSAUGCheckLocalMin_Private(SNES snes,Mat A,Vec F,Vec W,PetscReal fnorm,PetscBool *ismin)
{
  PetscReal      a1;
  PetscErrorCode ierr;
  PetscBool     hastranspose;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    /* Compute || J^T F|| */
    ierr = MatMultTranspose(A,F,W);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&a1);CHKERRQ(ierr);
    ierr = PetscInfo1(snes,"|| J^T F|| %g near zero implies found a local minimum\n",(double)(a1/fnorm));CHKERRQ(ierr);
    if (a1/fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    ierr = VecSetRandom(W,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&wnorm);CHKERRQ(ierr);
    ierr = VecDuplicate(W,&work);CHKERRQ(ierr);
    ierr = MatMult(A,W,work);CHKERRQ(ierr);
    ierr = VecDot(F,work,&result);CHKERRQ(ierr);
    ierr = VecDestroy(&work);CHKERRQ(ierr);
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    ierr = PetscInfo1(snes,"(F^T J random)/(|| F ||*||random|| %g near zero implies found a local minimum\n",(double)a1);CHKERRQ(ierr);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#if 0
/*
     Checks if J^T(F - J*X) = 0 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESVICheckResidual_Private"
PetscErrorCode SNESVICheckResidual_Private(SNES snes,Mat A,Vec F,Vec X,Vec W1,Vec W2)
{
  PetscReal      a1,a2;
  PetscErrorCode ierr;
  PetscBool     hastranspose;

  PetscFunctionBegin;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    ierr = MatMult(A,X,W1);CHKERRQ(ierr);
    ierr = VecAXPY(W1,-1.0,F);CHKERRQ(ierr);

    /* Compute || J^T W|| */
    ierr = MatMultTranspose(A,W1,W2);CHKERRQ(ierr);
    ierr = VecNorm(W1,NORM_2,&a1);CHKERRQ(ierr);
    ierr = VecNorm(W2,NORM_2,&a2);CHKERRQ(ierr);
    if (a1 != 0.0) {
      ierr = PetscInfo1(snes,"||J^T(F-Ax)||/||F-AX|| %g near zero implies inconsistent rhs\n",(double)(a2/a1));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
#endif

/*
  SNESDefaultConverged_VIRSAUG - Checks the convergence of the semismooth newton algorithm.

  Notes:
  The convergence criterion currently implemented is
  merit < abstol
  merit < rtol*merit_initial
*/
#undef __FUNCT__
#define __FUNCT__ "SNESDefaultConverged_VIRSAUG"
PetscErrorCode SNESDefaultConverged_VIRSAUG(SNES snes,PetscInt it,PetscReal xnorm,PetscReal gradnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);
  
  *reason = SNES_CONVERGED_ITERATING;

  if (!it) {
    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;
  }
  if (fnorm != fnorm) {
    ierr = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm < snes->abstol) {
    ierr = PetscInfo2(snes,"Converged due to function norm %g < %g\n",(double)fnorm,(double)snes->abstol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (fnorm < snes->ttol) {
      ierr = PetscInfo2(snes,"Converged due to function norm %g < %g (relative tolerance)\n",(double)fnorm,(double)snes->ttol);CHKERRQ(ierr);
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }
  } 
  PetscFunctionReturn(0);
}

/*
  SNESVIRSAUGComputeMeritFunction - Evaluates the merit function for the mixed complementarity problem.

  Input Parameter:
. phi - the semismooth function

  Output Parameter:
. merit - the merit function
. phinorm - ||phi||

  Notes:
  The merit function for the mixed complementarity problem is defined as
     merit = 0.5*phi^T*phi
*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGComputeMeritFunction"
static PetscErrorCode SNESVIRSAUGComputeMeritFunction(Vec phi, PetscReal* merit,PetscReal* phinorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNormBegin(phi,NORM_2,phinorm);CHKERRQ(ierr);
  ierr = VecNormEnd(phi,NORM_2,phinorm);CHKERRQ(ierr);

  *merit = 0.5*(*phinorm)*(*phinorm);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscScalar Phi(PetscScalar a,PetscScalar b)
{
  return a + b - PetscSqrtScalar(a*a + b*b);
}

PETSC_STATIC_INLINE PetscScalar DPhi(PetscScalar a,PetscScalar b)
{
  if ((PetscAbsScalar(a) >= 1.e-6) || (PetscAbsScalar(b) >= 1.e-6)) return  1.0 - a/ PetscSqrtScalar(a*a + b*b);
  else return .5;
}

/* 
   SNESVIComputeFunction - Reformulates a system of nonlinear equations in mixed complementarity form to a system of nonlinear equations in semismooth form. 

   Input Parameters:  
.  snes - the SNES context
.  x - current iterate
.  functx - user defined function context

   Output Parameters:
.  phi - Semismooth function

*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeFunction"
static PetscErrorCode SNESVIComputeFunction(SNES snes,Vec X,Vec phi,void* functx)
{
  PetscErrorCode  ierr;
  SNES_VIRSAUG       *vi = (SNES_VIRSAUG*)snes->data;
  Vec             Xl = vi->xl,Xu = vi->xu,F = snes->vec_func;
  PetscScalar     *phi_arr,*x_arr,*f_arr,*l,*u;
  PetscInt        i,nlocal;

  PetscFunctionBegin;
  ierr = (*vi->computeuserfunction)(snes,X,F,functx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x_arr);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f_arr);CHKERRQ(ierr);
  ierr = VecGetArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(phi,&phi_arr);CHKERRQ(ierr);

  for (i=0;i < nlocal;i++) {
    if ((PetscRealPart(l[i]) <= SNES_VI_NINF) && (PetscRealPart(u[i]) >= SNES_VI_INF)) { /* no constraints on variable */
      phi_arr[i] = f_arr[i];
    } else if (PetscRealPart(l[i]) <= SNES_VI_NINF) {                      /* upper bound on variable only */
      phi_arr[i] = -Phi(u[i] - x_arr[i],-f_arr[i]);
    } else if (PetscRealPart(u[i]) >= SNES_VI_INF) {                       /* lower bound on variable only */
      phi_arr[i] = Phi(x_arr[i] - l[i],f_arr[i]);
    } else if (l[i] == u[i]) {
      phi_arr[i] = l[i] - x_arr[i];
    } else {                                                /* both bounds on variable */
      phi_arr[i] = Phi(x_arr[i] - l[i],-Phi(u[i] - x_arr[i],-f_arr[i]));
    }
  }
  
  ierr = VecRestoreArray(X,&x_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(phi,&phi_arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
   SNESVIRSAUGComputeBsubdifferentialVectors - Computes the diagonal shift (Da) and row scaling (Db) vectors needed for the
                                          the semismooth jacobian.
*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGComputeBsubdifferentialVectors"
PetscErrorCode SNESVIRSAUGComputeBsubdifferentialVectors(SNES snes,Vec X,Vec F,Mat jac,Vec Da,Vec Db)
{
  PetscErrorCode ierr;
  SNES_VIRSAUG      *vi = (SNES_VIRSAUG*)snes->data;
  PetscScalar    *l,*u,*x,*f,*da,*db,da1,da2,db1,db2;
  PetscInt       i,nlocal;

  PetscFunctionBegin;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArray(vi->xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(vi->xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Da,&da);CHKERRQ(ierr);
  ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  
  for (i=0;i< nlocal;i++) {
    if ((PetscRealPart(l[i]) <= SNES_VI_NINF) && (PetscRealPart(u[i]) >= SNES_VI_INF)) {/* no constraints on variable */
      da[i] = 0; 
      db[i] = 1;
    } else if (PetscRealPart(l[i]) <= SNES_VI_NINF) {                     /* upper bound on variable only */
      da[i] = DPhi(u[i] - x[i], -f[i]);
      db[i] = DPhi(-f[i],u[i] - x[i]);
    } else if (PetscRealPart(u[i]) >= SNES_VI_INF) {                      /* lower bound on variable only */
      da[i] = DPhi(x[i] - l[i], f[i]);
      db[i] = DPhi(f[i],x[i] - l[i]);
    } else if (l[i] == u[i]) {                              /* fixed variable */
      da[i] = 1;
      db[i] = 0;
    } else {                                                /* upper and lower bounds on variable */
      da1 = DPhi(x[i] - l[i], -Phi(u[i] - x[i], -f[i]));
      db1 = DPhi(-Phi(u[i] - x[i], -f[i]),x[i] - l[i]);
      da2 = DPhi(u[i] - x[i], -f[i]);
      db2 = DPhi(-f[i],u[i] - x[i]);
      da[i] = da1 + db1*da2;
      db[i] = db1*db2;
    }
  }

  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArray(vi->xl,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(vi->xu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(Da,&da);CHKERRQ(ierr);
  ierr = VecRestoreArray(Db,&db);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESVIRSAUGComputeJacobian - Computes the jacobian of the semismooth function.The Jacobian for the semismooth function is an element of the B-subdifferential of the Fischer-Burmeister function for complementarity problems.

   Input Parameters:
.  Da       - Diagonal shift vector for the semismooth jacobian.
.  Db       - Row scaling vector for the semismooth jacobian. 

   Output Parameters:
.  jac      - semismooth jacobian
.  jac_pre  - optional preconditioning matrix

   Notes:
   The semismooth jacobian matrix is given by
   jac = Da + Db*jacfun
   where Db is the row scaling matrix stored as a vector,
         Da is the diagonal perturbation matrix stored as a vector
   and   jacfun is the jacobian of the original nonlinear function.	 
*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGComputeJacobian"
PetscErrorCode SNESVIRSAUGComputeJacobian(Mat jac, Mat jac_pre,Vec Da, Vec Db)
{
  PetscErrorCode ierr;
  
  /* Do row scaling  and add diagonal perturbation */
  ierr = MatDiagonalScale(jac,Db,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(jac,Da,ADD_VALUES);CHKERRQ(ierr);
  if (jac != jac_pre) { /* If jac and jac_pre are different */
    ierr = MatDiagonalScale(jac_pre,Db,PETSC_NULL);
    ierr = MatDiagonalSet(jac_pre,Da,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   SNESVIRSAUGComputeMeritFunctionGradient - Computes the gradient of the merit function psi.

   Input Parameters:
   phi - semismooth function.
   H   - semismooth jacobian
   
   Output Parameters:
   dpsi - merit function gradient

   Notes:
  The merit function gradient is computed as follows
        dpsi = H^T*phi
*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGComputeMeritFunctionGradient"
PetscErrorCode SNESVIRSAUGComputeMeritFunctionGradient(Mat H, Vec phi, Vec dpsi)
{
  PetscErrorCode ierr;
    
  PetscFunctionBegin;
  ierr = MatMultTranspose(H,phi,dpsi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESVIRSAUGProjectOntoBounds - Projects X onto the feasible region so that Xl[i] <= X[i] <= Xu[i] for i = 1...n.

   Input Parameters:
.  SNES - nonlinear solver context

   Output Parameters:
.  X - Bound projected X

*/

#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGProjectOntoBounds"
PetscErrorCode SNESVIRSAUGProjectOntoBounds(SNES snes,Vec X)
{
  PetscErrorCode    ierr;
  SNES_VIRSAUG         *vi = (SNES_VIRSAUG*)snes->data;
  const PetscScalar *xl,*xu;
  PetscScalar       *x;
  PetscInt          i,n;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xu,&xu);CHKERRQ(ierr);

  for (i = 0;i<n;i++) {
    if (PetscRealPart(x[i]) < PetscRealPart(xl[i])) x[i] = xl[i];
    else if (PetscRealPart(x[i]) > PetscRealPart(xu[i])) x[i] = xu[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*  -------------------------------------------------------------------- 

     This file implements a semismooth truncated Newton method with a line search,
     for solving a system of nonlinear equations in complementarity form, using the KSP, Vec, 
     and Mat interfaces for linear solvers, vectors, and matrices, 
     respectively.

     The following basic routines are required for each nonlinear solver:
          SNESCreate_XXX()          - Creates a nonlinear solver context
          SNESSetFromOptions_XXX()  - Sets runtime options
          SNESSolve_XXX()           - Solves the nonlinear system
          SNESDestroy_XXX()         - Destroys the nonlinear solver context
     The suffix "_XXX" denotes a particular implementation, in this case
     we use _VI (e.g., SNESCreate_VI, SNESSolve_VI) for solving
     systems of nonlinear equations with a line search (LS) method.
     These routines are actually called via the common user interface
     routines SNESCreate(), SNESSetFromOptions(), SNESSolve(), and 
     SNESDestroy(), so the application code interface remains identical 
     for all nonlinear solvers.

     Another key routine is:
          SNESSetUp_XXX()           - Prepares for the use of a nonlinear solver
     by setting data structures and options.   The interface routine SNESSetUp()
     is not usually called directly by the user, but instead is called by
     SNESSolve() if necessary.

     Additional basic routines are:
          SNESView_XXX()            - Prints details of runtime options that
                                      have actually been used.
     These are called by application codes via the interface routines
     SNESView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  

    -------------------------------------------------------------------- */
/*
   SNESSolveVI_SS - Solves the complementarity problem with a semismooth Newton
   method using a line search.

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  outits - number of iterations until termination

   Application Interface Routine: SNESSolve()

   Notes:
   This implements essentially a semismooth Newton method with a
   line search. The default line search does not do any line seach
   but rather takes a full newton step.
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolveVI_SS"
PetscErrorCode SNESSolveVI_SS(SNES snes)
{ 
  SNES_VIRSAUG          *vi = (SNES_VIRSAUG*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           maxits,i,lits;
  PetscBool          lssucceed;
  MatStructure       flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal          gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;
  DM                 dm;
  SNESDM             sdm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  vi->computeuserfunction = sdm->computefunction;
  sdm->computefunction    = SNESVIComputeFunction;

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  ierr = SNESVIProjectOntoBounds(snes,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,vi->phi);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    sdm->computefunction = vi->computeuserfunction;
    PetscFunctionReturn(0);
  }
   /* Compute Merit function */
  ierr = SNESVIRSAUGComputeMeritFunction(vi->phi,&vi->merit,&vi->phinorm);CHKERRQ(ierr);

  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(vi->merit)) SETERRQ(((PetscObject)X)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = vi->phinorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,vi->phinorm,0);
  ierr = SNESMonitor(snes,0,vi->phinorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = vi->phinorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) {
    sdm->computefunction = vi->computeuserfunction;
    PetscFunctionReturn(0);
  }

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
 
    /* Solve J Y = Phi, where J is the semismooth jacobian */
    /* Get the nonlinear function jacobian */
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    /* Get the diagonal shift and row scaling vectors */
    ierr = SNESVIRSAUGComputeBsubdifferentialVectors(snes,X,F,snes->jacobian,vi->Da,vi->Db);CHKERRQ(ierr);
    /* Compute the semismooth jacobian */
    ierr = SNESVIRSAUGComputeJacobian(snes->jacobian,snes->jacobian_pre,vi->Da,vi->Db);CHKERRQ(ierr);
    /* Compute the merit function gradient */
    ierr = SNESVIRSAUGComputeMeritFunctionGradient(snes->jacobian,vi->phi,vi->dpsi);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);
    ierr = SNES_KSPSolve(snes,snes->ksp,vi->phi,Y);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);

    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (snes->ops->precheckstep) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheckstep)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = vi->phinorm;
    ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,vi->phi,Y,vi->phinorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)vi->phinorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      sdm->computefunction = vi->computeuserfunction;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESVIRSAUGCheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    vi->phinorm = gnorm;
    vi->merit = 0.5*vi->phinorm*vi->phinorm;
    ierr = VecCopy(G,vi->phi);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = vi->phinorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  sdm->computefunction = vi->computeuserfunction;
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "SNESVIGetActiveSetIS"
/*
   SNESVIGetActiveSetIndices - Gets the global indices for the active set variables

   Input parameter
.  snes - the SNES context
.  X    - the snes solution vector
.  F    - the nonlinear function vector

   Output parameter
.  ISact - active set index set
 */
PetscErrorCode SNESVIGetActiveSetIS(SNES snes,Vec X,Vec F,IS* ISact)
{
  PetscErrorCode   ierr;
  SNES_VIRSAUG        *vi = (SNES_VIRSAUG*)snes->data;
  Vec               Xl=vi->xl,Xu=vi->xu;
  const PetscScalar *x,*f,*xl,*xu;
  PetscInt          *idx_act,i,nlocal,nloc_isact=0,ilow,ihigh,i1=0;
  
  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&ilow,&ihigh);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F,&f);CHKERRQ(ierr);
  /* Compute active set size */
  for (i=0; i < nlocal;i++) {
    if (!vi->ignorefunctionsign) {
      if (!((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) nloc_isact++;
    } else {
      if (!(PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8  && PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8)) nloc_isact++;
    }
  }

  ierr = PetscMalloc(nloc_isact*sizeof(PetscInt),&idx_act);CHKERRQ(ierr);

  /* Set active set indices */
  for (i=0; i < nlocal; i++) {
    if (!vi->ignorefunctionsign) {
      if (!((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) idx_act[i1++] = ilow+i;
    } else {
      if (!(PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8  && PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8)) idx_act[i1++] = ilow+i;
    }
  }

   /* Create active set IS */
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,nloc_isact,idx_act,PETSC_OWN_POINTER,ISact);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "SNESVICreateIndexSets_RSAUG"
PetscErrorCode SNESVICreateIndexSets_RSAUG(SNES snes,Vec X,Vec F,IS* ISact,IS* ISinact)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = SNESVIGetActiveSetIS(snes,X,F,ISact);CHKERRQ(ierr);
  ierr = ISComplement(*ISact,X->map->rstart,X->map->rend,ISinact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create active and inactive set vectors. The local size of this vector is set and petsc computes the global size */
#undef __FUNCT__
#define __FUNCT__ "SNESVICreateSubVectors"
PetscErrorCode SNESVICreateSubVectors(SNES snes,PetscInt n,Vec* newv)
{
  PetscErrorCode ierr;
  Vec            v;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)snes)->comm,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  *newv = v;

  PetscFunctionReturn(0);
}

extern PetscErrorCode SNESVIResetPCandKSP(SNES snes,Mat Amat,Mat Pmat);
#if 0
/* Resets the snes PC and KSP when the active set sizes change */
#undef __FUNCT__
#define __FUNCT__ "SNESVIResetPCandKSP"
PetscErrorCode SNESVIResetPCandKSP(SNES snes,Mat Amat,Mat Pmat)
{
  PetscErrorCode         ierr;
  KSP                    snesksp;

  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&snesksp);CHKERRQ(ierr);
  ierr = KSPReset(snesksp);CHKERRQ(ierr);

  /*
  KSP                    kspnew;
  PC                     pcnew;
  const MatSolverPackage stype;


  ierr = KSPCreate(((PetscObject)snes)->comm,&kspnew);CHKERRQ(ierr);
  kspnew->pc_side = snesksp->pc_side;
  kspnew->rtol    = snesksp->rtol;
  kspnew->abstol    = snesksp->abstol;
  kspnew->max_it  = snesksp->max_it;
  ierr = KSPSetType(kspnew,((PetscObject)snesksp)->type_name);CHKERRQ(ierr);
  ierr = KSPGetPC(kspnew,&pcnew);CHKERRQ(ierr);
  ierr = PCSetType(kspnew->pc,((PetscObject)snesksp->pc)->type_name);CHKERRQ(ierr);
  ierr = PCSetOperators(kspnew->pc,Amat,Pmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCFactorGetMatSolverPackage(snesksp->pc,&stype);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(kspnew->pc,stype);CHKERRQ(ierr);
  ierr = KSPDestroy(&snesksp);CHKERRQ(ierr);
  snes->ksp = kspnew;
  ierr = PetscLogObjectParent(snes,kspnew);CHKERRQ(ierr);
   ierr = KSPSetFromOptions(kspnew);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeInactiveSetFnorm"
PetscErrorCode SNESVIComputeInactiveSetFnorm(SNES snes,Vec F,Vec X,PetscReal *fnorm)
{
  PetscErrorCode    ierr;
  SNES_VIRSAUG         *vi = (SNES_VIRSAUG*)snes->data;
  const PetscScalar *x,*xl,*xu,*f;
  PetscInt          i,n;
  PetscReal         rnorm;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F,&f);CHKERRQ(ierr);
  rnorm = 0.0;
  if (!vi->ignorefunctionsign) {
    for (i=0; i<n; i++) {
      if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
    }
  } else {
    for (i=0; i<n; i++) {
      if ((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8) && (PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8)) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
    }
  }
  ierr = VecRestoreArrayRead(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&rnorm,fnorm,1,MPIU_REAL,MPIU_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  *fnorm = PetscSqrtReal(*fnorm);
  PetscFunctionReturn(0);
}
#endif

/* Variational Inequality solver using reduce space method. No semismooth algorithm is
   implemented in this algorithm. It basically identifies the active constraints and does
   a linear solve on the other variables (those not associated with the active constraints). */
#undef __FUNCT__  
#define __FUNCT__ "SNESSolveVI_RS"
PetscErrorCode SNESSolveVI_RS(SNES snes)
{ 
  SNES_VIRSAUG        *vi = (SNES_VIRSAUG*)snes->data;
  PetscErrorCode    ierr;
  PetscInt          maxits,i,lits;
  PetscBool         lssucceed;
  MatStructure      flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal         fnorm,gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  ierr = SNESVIProjectOntoBounds(snes,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(((PetscObject)X)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);


  for (i=0; i<maxits; i++) {

    IS         IS_act,IS_inact; /* _act -> active set _inact -> inactive set */
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
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);


        /* Create active and inactive index sets */
    
    /*original
    ierr = SNESVICreateIndexSets_RSAUG(snes,X,F,&IS_act,&IS_inact);CHKERRQ(ierr);
     */
    ierr = SNESVIGetActiveSetIS(snes,X,F,&IS_act);CHKERRQ(ierr);
    
    if (vi->checkredundancy) {
      (*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP);CHKERRQ(ierr);
      if (IS_redact){
        ierr = ISSort(IS_redact);CHKERRQ(ierr);
        ierr = ISComplement(IS_redact,X->map->rstart,X->map->rend,&IS_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_redact);CHKERRQ(ierr);
      }
      else {
        ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&IS_inact);CHKERRQ(ierr);
      }
    } else {
      ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&IS_inact);CHKERRQ(ierr);
    }
    

    /* Create inactive set submatrix */
    ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);

    if (0) {                    /* Dead code (temporary developer hack) */
      IS keptrows;
      ierr = MatFindNonzeroRows(jac_inact_inact,&keptrows);CHKERRQ(ierr);
      if (0) {
        PetscInt       cnt,*nrows,k;
        const PetscInt *krows,*inact;
        PetscInt       rstart=jac_inact_inact->rmap->rstart;

        ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_act);CHKERRQ(ierr);

        ierr = ISGetLocalSize(keptrows,&cnt);CHKERRQ(ierr);
        ierr = ISGetIndices(keptrows,&krows);CHKERRQ(ierr);
        ierr = ISGetIndices(IS_inact,&inact);CHKERRQ(ierr);
        ierr = PetscMalloc(cnt*sizeof(PetscInt),&nrows);CHKERRQ(ierr);
        for (k=0; k<cnt; k++) {
          nrows[k] = inact[krows[k]-rstart];
        }
        ierr = ISRestoreIndices(keptrows,&krows);CHKERRQ(ierr);
        ierr = ISRestoreIndices(IS_inact,&inact);CHKERRQ(ierr);
        ierr = ISDestroy(&keptrows);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_inact);CHKERRQ(ierr);

        ierr = ISCreateGeneral(((PetscObject)sness)->comm,cnt,nrows,PETSC_OWN_POINTER,&IS_inact);CHKERRQ(ierr);
        ierr = ISComplement(IS_inact,F->map->rstart,F->map->rend,&IS_act);CHKERRQ(ierr);
        ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);
      }
    }
    ierr = DMSetVIRSAUG(snes->dm,IS_inact);CHKERRQ(ierr);
    /* remove later */

    /*
  ierr = VecView(vi->xu,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xu))->comm));CHKERRQ(ierr);
  ierr = VecView(vi->xl,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xl))->comm));CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_BINARY_(((PetscObject)X)->comm));CHKERRQ(ierr);
  ierr = VecView(F,PETSC_VIEWER_BINARY_(((PetscObject)F)->comm));CHKERRQ(ierr);
  ierr = ISView(IS_inact,PETSC_VIEWER_BINARY_(((PetscObject)IS_inact)->comm));CHKERRQ(ierr);
     */

    /* Get sizes of active and inactive sets */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    ierr = ISGetLocalSize(IS_inact,&nis_inact);CHKERRQ(ierr);

    /* Create active and inactive set vectors */
    ierr = SNESVICreateSubVectors(snes,nis_inact,&F_inact);CHKERRQ(ierr);
    ierr = SNESVICreateSubVectors(snes,nis_act,&Y_act);CHKERRQ(ierr);
    ierr = SNESVICreateSubVectors(snes,nis_inact,&Y_inact);CHKERRQ(ierr);

    /* Create scatter contexts */
    ierr = VecScatterCreate(Y,IS_act,Y_act,PETSC_NULL,&scat_act);CHKERRQ(ierr);
    ierr = VecScatterCreate(Y,IS_inact,Y_inact,PETSC_NULL,&scat_inact);CHKERRQ(ierr);

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
      ierr = MatGetSubMatrix(snes->jacobian_pre,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact);CHKERRQ(ierr);
    } else prejac_inact_inact = jac_inact_inact;

    ierr = ISEqual(vi->IS_inact_prev,IS_inact,&isequal);CHKERRQ(ierr);
    if (!isequal) {
      ierr = SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
      flg  = DIFFERENT_NONZERO_PATTERN;
    }
    
    /*      ierr = ISView(IS_inact,0);CHKERRQ(ierr); */
    /*      ierr = ISView(IS_act,0);CHKERRQ(ierr);*/
    /*      ierr = MatView(snes->jacobian_pre,0); */

   
   
    ierr = KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact,flg);CHKERRQ(ierr);
    ierr = KSPSetUp(snes->ksp);CHKERRQ(ierr);
    {
      PC        pc;
      PetscBool flg;
      ierr = KSPGetPC(snes->ksp,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&flg);CHKERRQ(ierr);
      if (flg) {
        KSP      *subksps;
        ierr = PCFieldSplitGetSubKSP(pc,PETSC_NULL,&subksps);CHKERRQ(ierr);
        ierr = KSPGetPC(subksps[0],&pc);CHKERRQ(ierr);
        ierr = PetscFree(subksps);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg);CHKERRQ(ierr);
        if (flg) {
          PetscInt       n,N = 101*101,j,cnts[3] = {0,0,0};
          const PetscInt *ii;

          ierr = ISGetSize(IS_inact,&n);CHKERRQ(ierr);
          ierr = ISGetIndices(IS_inact,&ii);CHKERRQ(ierr);
          for (j=0; j<n; j++) {
            if (ii[j] < N) cnts[0]++;
            else if (ii[j] < 2*N) cnts[1]++;
            else if (ii[j] < 3*N) cnts[2]++;
          }
          ierr = ISRestoreIndices(IS_inact,&ii);CHKERRQ(ierr);

          ierr = PCBJacobiSetTotalBlocks(pc,3,cnts);CHKERRQ(ierr);
        }
      }
    }

    ierr = SNES_KSPSolve(snes,snes->ksp,F_inact,Y_inact);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
     }

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
      ierr = ISDuplicate(IS_inact,&vi->IS_inact_prev);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&IS_inact);CHKERRQ(ierr);
    ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatDestroy(&prejac_inact_inact);CHKERRQ(ierr);
    }
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (snes->ops->precheckstep) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheckstep)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,F,Y,fnorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      ierr = DMDestroyVIRSAUG(snes->dm);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESVIRSAUGCheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    fnorm = gnorm;
    ierr = VecCopy(G,F);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  ierr = DMDestroyVIRSAUG(snes->dm);CHKERRQ(ierr);
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGSetRedundancyCheck"
PetscErrorCode SNESVIRSAUGSetRedundancyCheck(SNES snes,PetscErrorCode (*func)(SNES,IS,IS*,void*),void *ctx)
{
  SNES_VIRSAUG *vi = (SNES_VIRSAUG*)snes->data;

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

#undef __FUNCT__
#define __FUNCT__ "SNESVIRedundancyCheck_Matlab"
PetscErrorCode SNESVIRedundancyCheck_Matlab(SNES snes,IS is_act,IS* is_redact,void* ctx)
{
  PetscErrorCode      ierr;
  SNESMatlabContext   *sctx = (SNESMatlabContext*)ctx;
  int                 nlhs = 1, nrhs = 5;
  mxArray             *plhs[1], *prhs[5];
  long long int       l1 = 0, l2 = 0, ls = 0;
  PetscInt            *indices=PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(is_act,IS_CLASSID,2);
  PetscValidPointer(is_redact,3);
  PetscCheckSameComm(snes,1,is_act,2);

  /* Create IS for reduced active set of size 0, its size and indices will
   bet set by the Matlab function */
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,0,indices,PETSC_OWN_POINTER,is_redact);CHKERRQ(ierr);
  /* call Matlab function in ctx */
  ierr = PetscMemcpy(&ls,&snes,sizeof(snes));CHKERRQ(ierr);
  ierr = PetscMemcpy(&l1,&is_act,sizeof(is_act));CHKERRQ(ierr);
  ierr = PetscMemcpy(&l2,is_redact,sizeof(is_act));CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "SNESVISetRedundancyCheckMatlab"
PetscErrorCode SNESVISetRedundancyCheckMatlab(SNES snes,const char* func,mxArray* ctx)
{
  PetscErrorCode      ierr;
  SNESMatlabContext   *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(SNESMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = SNESVISetRedundancyCheck(snes,SNESVIRedundancyCheck_Matlab,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#endif


/* Variational Inequality solver using augmented space method. It does the opposite of the
   reduced space method i.e. it identifies the active set variables and instead of discarding
   them it augments the original system by introducing additional equality 
   constraint equations for active set variables. The user can optionally provide an IS for 
   a subset of 'redundant' active set variables which will treated as free variables.
   Specific implementation for Allen-Cahn problem 
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolveVI_RSAUG"
PetscErrorCode SNESSolveVI_RSAUG(SNES snes)
{ 
  SNES_VIRSAUG        *vi = (SNES_VIRSAUG*)snes->data;
  PetscErrorCode    ierr;
  PetscInt          maxits,i,lits;
  PetscBool         lssucceed;
  MatStructure      flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal         fnorm,gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  ierr = SNESVIProjectOntoBounds(snes,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(((PetscObject)X)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {
    IS             IS_act,IS_inact; /* _act -> active set _inact -> inactive set */
    IS             IS_redact; /* redundant active set */
    Mat            J_aug,Jpre_aug;
    Vec            F_aug,Y_aug;
    PetscInt       nis_redact,nis_act;
    const PetscInt *idx_redact,*idx_act;
    PetscInt       k;
    PetscInt       *idx_actkept=PETSC_NULL,nkept=0; /* list of kept active set */
    PetscScalar    *f,*f2;
    PetscBool      isequal;
    PetscInt       i1=0,j1=0;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);

    /* Create active and inactive index sets */
    ierr = SNESVICreateIndexSets_RSAUG(snes,X,F,&IS_act,&IS_inact);CHKERRQ(ierr);

    /* Get local active set size */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    if (nis_act) {
      ierr = ISGetIndices(IS_act,&idx_act);CHKERRQ(ierr);
      IS_redact  = PETSC_NULL;
      if (vi->checkredundancy) {
	(*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP);
      }

      if (!IS_redact) {
	/* User called checkredundancy function but didn't create IS_redact because
           there were no redundant active set variables */
	/* Copy over all active set indices to the list */
	ierr = PetscMalloc(nis_act*sizeof(PetscInt),&idx_actkept);CHKERRQ(ierr);
	for (k=0;k < nis_act;k++) idx_actkept[k] = idx_act[k];
	nkept = nis_act;
      } else {
	ierr = ISGetLocalSize(IS_redact,&nis_redact);CHKERRQ(ierr);
	ierr = PetscMalloc((nis_act-nis_redact)*sizeof(PetscInt),&idx_actkept);CHKERRQ(ierr);

	/* Create reduced active set list */ 
	ierr = ISGetIndices(IS_act,&idx_act);CHKERRQ(ierr);
	ierr = ISGetIndices(IS_redact,&idx_redact);CHKERRQ(ierr);
	j1 = 0;nkept = 0;
	for (k=0;k<nis_act;k++) {
	  if (j1 < nis_redact && idx_act[k] == idx_redact[j1]) j1++;
	  else idx_actkept[nkept++] = idx_act[k];
	}
	ierr = ISRestoreIndices(IS_act,&idx_act);CHKERRQ(ierr);
	ierr = ISRestoreIndices(IS_redact,&idx_redact);CHKERRQ(ierr);

        ierr = ISDestroy(&IS_redact);CHKERRQ(ierr);
      }

      /* Create augmented F and Y */
      ierr = VecCreate(((PetscObject)snes)->comm,&F_aug);CHKERRQ(ierr);
      ierr = VecSetSizes(F_aug,F->map->n+nkept,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(F_aug);CHKERRQ(ierr);
      ierr = VecDuplicate(F_aug,&Y_aug);CHKERRQ(ierr);
      
      /* Copy over F to F_aug in the first n locations */
      ierr = VecGetArray(F,&f);CHKERRQ(ierr);
      ierr = VecGetArray(F_aug,&f2);CHKERRQ(ierr);
      ierr = PetscMemcpy(f2,f,F->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
      ierr = VecRestoreArray(F_aug,&f2);CHKERRQ(ierr);
      
      /* Create the augmented jacobian matrix */
      ierr = MatCreate(((PetscObject)snes)->comm,&J_aug);CHKERRQ(ierr);
      ierr = MatSetSizes(J_aug,X->map->n+nkept,X->map->n+nkept,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = MatSetFromOptions(J_aug);CHKERRQ(ierr);
      

      { /* local vars */
      /* Preallocate augmented matrix and set values in it...Doing only sequential case first*/
      PetscInt          ncols;
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscScalar        value[2];
      PetscInt           row,col[2];
      PetscInt           *d_nnz;
      value[0] = 1.0; value[1] = 0.0;
      ierr = PetscMalloc((X->map->n+nkept)*sizeof(PetscInt),&d_nnz);CHKERRQ(ierr);
      ierr = PetscMemzero(d_nnz,(X->map->n+nkept)*sizeof(PetscInt));CHKERRQ(ierr);
      for (row=0;row<snes->jacobian->rmap->n;row++) {
        ierr = MatGetRow(snes->jacobian,row,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        d_nnz[row] += ncols;
        ierr = MatRestoreRow(snes->jacobian,row,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      }

      for (k=0;k<nkept;k++) {
        d_nnz[idx_actkept[k]] += 1;
        d_nnz[snes->jacobian->rmap->n+k] += 2;
      }
      ierr = MatSeqAIJSetPreallocation(J_aug,PETSC_NULL,d_nnz);CHKERRQ(ierr);
      
      ierr = PetscFree(d_nnz);CHKERRQ(ierr);

      /* Set the original jacobian matrix in J_aug */
      for (row=0;row<snes->jacobian->rmap->n;row++) {
	ierr = MatGetRow(snes->jacobian,row,&ncols,&cols,&vals);CHKERRQ(ierr);
	ierr = MatSetValues(J_aug,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatRestoreRow(snes->jacobian,row,&ncols,&cols,&vals);CHKERRQ(ierr);
      }
      /* Add the augmented part */
      for (k=0;k<nkept;k++) {
	row = snes->jacobian->rmap->n+k;
	col[0] = idx_actkept[k]; col[1] = snes->jacobian->rmap->n+k;
	ierr = MatSetValues(J_aug,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatSetValues(J_aug,1,&col[0],1,&row,&value[0],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(J_aug,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(J_aug,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      /* Only considering prejac = jac for now */
      Jpre_aug = J_aug;
      } /* local vars*/
      ierr = ISRestoreIndices(IS_act,&idx_act);CHKERRQ(ierr);
      ierr = ISDestroy(&IS_act);CHKERRQ(ierr);
    } else {
      F_aug = F; J_aug = snes->jacobian; Y_aug = Y; Jpre_aug = snes->jacobian_pre;
    }

    ierr = ISEqual(vi->IS_inact_prev,IS_inact,&isequal);CHKERRQ(ierr);
    if (!isequal) {
      ierr = SNESVIResetPCandKSP(snes,J_aug,Jpre_aug);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(snes->ksp,J_aug,Jpre_aug,flg);CHKERRQ(ierr);
    ierr = KSPSetUp(snes->ksp);CHKERRQ(ierr);
    /*  {
      PC        pc;
      PetscBool flg;
      ierr = KSPGetPC(snes->ksp,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&flg);CHKERRQ(ierr);
      if (flg) {
        KSP      *subksps;
        ierr = PCFieldSplitGetSubKSP(pc,PETSC_NULL,&subksps);CHKERRQ(ierr);
        ierr = KSPGetPC(subksps[0],&pc);CHKERRQ(ierr);
        ierr = PetscFree(subksps);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg);CHKERRQ(ierr);
        if (flg) {
          PetscInt       n,N = 101*101,j,cnts[3] = {0,0,0};
          const PetscInt *ii;

          ierr = ISGetSize(IS_inact,&n);CHKERRQ(ierr);
          ierr = ISGetIndices(IS_inact,&ii);CHKERRQ(ierr);
          for (j=0; j<n; j++) {
            if (ii[j] < N) cnts[0]++;
            else if (ii[j] < 2*N) cnts[1]++;
            else if (ii[j] < 3*N) cnts[2]++;
          }
          ierr = ISRestoreIndices(IS_inact,&ii);CHKERRQ(ierr);

          ierr = PCBJacobiSetTotalBlocks(pc,3,cnts);CHKERRQ(ierr);
        }
      }
    }
    */
    ierr = SNES_KSPSolve(snes,snes->ksp,F_aug,Y_aug);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }

    if (nis_act) {
      PetscScalar *y1,*y2;
      ierr = VecGetArray(Y,&y1);CHKERRQ(ierr);
      ierr = VecGetArray(Y_aug,&y2);CHKERRQ(ierr);
      /* Copy over inactive Y_aug to Y */
      j1 = 0;
      for (i1=Y->map->rstart;i1 < Y->map->rend;i1++) {
	if (j1 < nkept && idx_actkept[j1] == i1) j1++;
	else y1[i1-Y->map->rstart] = y2[i1-Y->map->rstart];
      }
      ierr = VecRestoreArray(Y,&y1);CHKERRQ(ierr);
      ierr = VecRestoreArray(Y_aug,&y2);CHKERRQ(ierr);

      ierr = VecDestroy(&F_aug);CHKERRQ(ierr);
      ierr = VecDestroy(&Y_aug);CHKERRQ(ierr);
      ierr = MatDestroy(&J_aug);CHKERRQ(ierr);
      ierr = PetscFree(idx_actkept);CHKERRQ(ierr);
    }
	
    if (!isequal) {
      ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
      ierr = ISDuplicate(IS_inact,&vi->IS_inact_prev);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&IS_inact);CHKERRQ(ierr);

    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (snes->ops->precheckstep) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheckstep)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,F,Y,fnorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESVIRSAUGCheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    fnorm = gnorm;
    ierr = VecCopy(G,F);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VI - Sets up the internal data structures for the later use
   of the SNESVI nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_VIRSAUG"
PetscErrorCode SNESSetUp_VIRSAUG(SNES snes)
{
  PetscErrorCode ierr;
  SNES_VIRSAUG      *vi = (SNES_VIRSAUG*) snes->data;
  PetscInt       i_start[3],i_end[3];

  PetscFunctionBegin;

  ierr = SNESDefaultGetWork(snes,3);CHKERRQ(ierr);

  if (vi->computevariablebounds) {
    if (!vi->xl) {ierr = VecDuplicate(snes->vec_sol,&vi->xl);CHKERRQ(ierr);}
    if (!vi->xu) {ierr = VecDuplicate(snes->vec_sol,&vi->xu);CHKERRQ(ierr);}
    ierr = (*vi->computevariablebounds)(snes,vi->xl,vi->xu);CHKERRQ(ierr);
  } else if (!vi->xl && !vi->xu) {
    /* If the lower and upper bound on variables are not set, set it to -Inf and Inf */
    ierr = VecDuplicate(snes->vec_sol, &vi->xl);CHKERRQ(ierr);
    ierr = VecSet(vi->xl,SNES_VI_NINF);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->xu);CHKERRQ(ierr);
    ierr = VecSet(vi->xu,SNES_VI_INF);CHKERRQ(ierr);
  } else {
    /* Check if lower bound, upper bound and solution vector distribution across the processors is identical */
    ierr = VecGetOwnershipRange(snes->vec_sol,i_start,i_end);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vi->xl,i_start+1,i_end+1);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vi->xu,i_start+2,i_end+2);CHKERRQ(ierr);
    if ((i_start[0] != i_start[1]) || (i_start[0] != i_start[2]) || (i_end[0] != i_end[1]) || (i_end[0] != i_end[2]))
      SETERRQ(((PetscObject)(vi->xl))->comm,PETSC_ERR_ARG_SIZ,"Distribution of lower bound, upper bound and the solution vector should be identical across all the processors.");
  }
  if (snes->ops->solve != SNESSolveVI_SS) {
    /* Set up previous active index set for the first snes solve 
       vi->IS_inact_prev = 0,1,2,....N */
    PetscInt *indices;
    PetscInt  i,n,rstart,rend;

    ierr = VecGetOwnershipRange(snes->vec_sol,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetLocalSize(snes->vec_sol,&n);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscInt),&indices);CHKERRQ(ierr);
    for (i=0;i < n; i++) indices[i] = rstart + i;
    ierr = ISCreateGeneral(((PetscObject)snes)->comm,n,indices,PETSC_OWN_POINTER,&vi->IS_inact_prev);
  }

  if (snes->ops->solve == SNESSolveVI_SS) {
    ierr = VecDuplicate(snes->vec_sol, &vi->dpsi);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->phi);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->Da);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->Db);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->z);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESReset_VIRSAUG"
PetscErrorCode SNESReset_VIRSAUG(SNES snes)
{
  SNES_VIRSAUG      *vi = (SNES_VIRSAUG*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&vi->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->xu);CHKERRQ(ierr);
  if (snes->ops->solve != SNESSolveVI_SS) {
    ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   SNESDestroy_VI - Destroys the private SNES_VI context that was created
   with SNESCreate_VI().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_VIRSAUG"
PetscErrorCode SNESDestroy_VIRSAUG(SNES snes)
{
  SNES_VIRSAUG      *vi = (SNES_VIRSAUG*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (snes->ops->solve == SNESSolveVI_SS) {
    /* clear vectors */
    ierr = VecDestroy(&vi->dpsi);CHKERRQ(ierr);
    ierr = VecDestroy(&vi->phi);CHKERRQ(ierr);
    ierr = VecDestroy(&vi->Da);CHKERRQ(ierr);
    ierr = VecDestroy(&vi->Db);CHKERRQ(ierr);
    ierr = VecDestroy(&vi->z);CHKERRQ(ierr);
    ierr = VecDestroy(&vi->t);CHKERRQ(ierr);
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetMonitor_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchNo_VIRSAUG"

/*
  This routine does not actually do a line search but it takes a full newton
  step while ensuring that the new iterates remain within the constraints.
  
*/
PetscErrorCode SNESLineSearchNo_VIRSAUG(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode ierr;
  PetscBool      changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE; 
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);         /* ynorm = || y || */
  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  if (snes->ops->postcheckstep) {
   ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
  }
  if (changed_y) {
    ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y   */
    ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  }
  ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (!snes->domainerror) {
    if (snes->ops->solve != SNESSolveVI_SS) {
       ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);  /* gnorm = || g || */
    }
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  }
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Using full step: fnorm %g gnorm %g\n",(double)fnorm,(double)*gnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchNoNorms_VIRSAUG"

/*
  This routine is a copy of SNESLineSearchNoNorms in snes/impls/ls/ls.c
*/
PetscErrorCode SNESLineSearchNoNorms_VIRSAUG(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode ierr;
  PetscBool     changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE; 
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y      */
  ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  if (snes->ops->postcheckstep) {
   ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
  }
  if (changed_y) {
    ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y   */
    ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  }
  
  /* don't evaluate function the last time through */
  if (snes->iter < snes->max_its-1) {
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchCubic_VIRSAUG"
/*
  This routine implements a cubic line search while doing a projection on the variable bounds
*/
PetscErrorCode SNESLineSearchCubic_VIRSAUG(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscReal      initslope,lambdaprev,gnormprev,a,b,d,t1,t2,rellength;
  PetscReal      minlambda,lambda,lambdatemp;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscErrorCode ierr;
  PetscInt       count;
  PetscBool      changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Initial direction and size is 0\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    *gnorm = fnorm;
    ierr   = VecCopy(x,w);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend1;
  }
  if (*ynorm > snes->maxstep) {	/* Step too big, so scale back */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Scaling step by %g old ynorm %g\n",(double)(snes->maxstep/(*ynorm)),(double)(*ynorm));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    ierr = VecScale(y,snes->maxstep/(*ynorm));CHKERRQ(ierr);
    *ynorm = snes->maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = snes->steptol/rellength;
  ierr = MatMult(snes->jacobian,y,w);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(f,w,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(f,w,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
  ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend1;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->ops->solve != SNESSolveVI_SS) {
    ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  }
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscInfo4(snes,"Initial fnorm %g gnorm %g alpha %g initslope %g\n",(double)fnorm,(double)*gnorm,(double)snes->ls_alpha,(double)initslope);CHKERRQ(ierr);
  if ((*gnorm)*(*gnorm) <= (1.0 - snes->ls_alpha)*fnorm*fnorm ) { /* Sufficient reduction */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Using full step: fnorm %g gnorm %g\n",(double)fnorm,(double)(*gnorm));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    goto theend1;
  }

  /* Fit points with quadratic */
  lambda     = 1.0;
  lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
  lambdaprev = lambda;
  gnormprev  = *gnorm;
  if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
  if (lambdatemp <= .1*lambda) lambda = .1*lambda; 
  else                         lambda = lambdatemp;

  ierr  = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
  ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while attempting quadratic backtracking! %D \n",snes->nfuncs);CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend1;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->ops->solve != SNESSolveVI_SS) {
    ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  }
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: gnorm after quadratic fit %g\n",(double)(*gnorm));CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  if ((*gnorm)*(*gnorm) < (1.0 - snes->ls_alpha)*fnorm*fnorm ) { /* sufficient reduction */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Quadratically determined step, lambda=%18.16e\n",(double)lambda);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    goto theend1;
  }

  /* Fit points with cubic */
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { 
      if (snes->ls_monitor) {
        ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
 	ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: unable to find good step length! After %D tries \n",count);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)minlambda,(double)lambda,(double)initslope);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
      *flag = PETSC_FALSE; 
      break;
    }
    t1 = .5*((*gnorm)*(*gnorm) - fnorm*fnorm) - lambda*initslope;
    t2 = .5*(gnormprev*gnormprev  - fnorm*fnorm) - lambdaprev*initslope;
    a  = (t1/(lambda*lambda) - t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
    b  = (-lambdaprev*t1/(lambda*lambda) + lambda*t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
    d  = b*b - 3*a*initslope;
    if (d < 0.0) d = 0.0;
    if (a == 0.0) {
      lambdatemp = -initslope/(2.0*b);
    } else {
      lambdatemp = (-b + PetscSqrtReal(d))/(3.0*a);
    }
    lambdaprev = lambda;
    gnormprev  = *gnorm;
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda;
    else                         lambda     = lambdatemp;
    ierr = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
    ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
    if (snes->nfuncs >= snes->max_funcs) {
      ierr = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
      ierr = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)lambda,(double)initslope);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    if (snes->ops->solve != SNESSolveVI_SS) {
      ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    }
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if ((*gnorm)*(*gnorm) < (1.0 - snes->ls_alpha)*fnorm*fnorm) { /* is reduction enough? */
      if (snes->ls_monitor) {
	ierr = PetscPrintf(comm,"    Line search: Cubically determined step, current gnorm %g lambda=%18.16e\n",(double)(*gnorm),(double)lambda);CHKERRQ(ierr);
      }
      break;
    } else {
      if (snes->ls_monitor) {
        ierr = PetscPrintf(comm,"    Line search: Cubic step no good, shrinking lambda, current gnorm %g lambda=%18.16e\n",(double)(*gnorm),(double)lambda);CHKERRQ(ierr);
      }
    }
    count++;
  }
  theend1:
  /* Optional user-defined check for line search step validity */
  if (snes->ops->postcheckstep && *flag) {
    ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
    if (changed_y) {
      ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
      ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
    }
    if (changed_y || changed_w) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
      if (snes->ops->solve != SNESSolveVI_SS) {
        ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
      }
      if (snes->domainerror) {
        ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchQuadratic_VIRSAUG"
/*
  This routine does a quadratic line search while keeping the iterates within the variable bounds
*/
PetscErrorCode SNESLineSearchQuadratic_VIRSAUG(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm,and fnorm = || f ||_2.
   */
  PetscReal      initslope,minlambda,lambda,lambdatemp,rellength;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscErrorCode ierr;
  PetscInt       count;
  PetscBool     changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  ierr    = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"Line search: Direction and size is 0\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    *gnorm = fnorm;
    ierr   = VecCopy(x,w);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend2;
  }
  if (*ynorm > snes->maxstep) {	/* Step too big, so scale back */
    ierr   = VecScale(y,snes->maxstep/(*ynorm));CHKERRQ(ierr);
    *ynorm = snes->maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = snes->steptol/rellength;
  ierr = MatMult(snes->jacobian,y,w);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr      = VecDot(f,w,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(f,w,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;
  ierr = PetscInfo1(snes,"Initslope %g \n",(double)initslope);CHKERRQ(ierr);

  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
  ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend2;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->ops->solve != SNESSolveVI_SS) {
    ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  }
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  if ((*gnorm)*(*gnorm) <= (1.0 - snes->ls_alpha)*fnorm*fnorm) { /* Sufficient reduction */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Using full step: fnorm %g gnorm %g\n",(double)fnorm,(double)(*gnorm));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    goto theend2;
  }

  /* Fit points with quadratic */
  lambda = 1.0;
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      if (snes->ls_monitor) {
        ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"Line search: Unable to find good step length! %D \n",count);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"Line search: fnorm=%g, gnorm=%g, ynorm=%g, lambda=%g, initial slope=%g\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)lambda,(double)initslope);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
      ierr = VecCopy(x,w);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      break;
    }
    lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda; 
    else                         lambda     = lambdatemp;
    
    ierr = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
    ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
    if (snes->nfuncs >= snes->max_funcs) {
      ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
      ierr  = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)lambda,(double)initslope);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (snes->ops->solve != SNESSolveVI_SS) {
      ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    }
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if ((*gnorm)*(*gnorm) < (1.0 - snes->ls_alpha)*fnorm*fnorm) { /* sufficient reduction */
      if (snes->ls_monitor) {
        ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line Search: Quadratically determined step, lambda=%g\n",(double)lambda);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
      break;
    }
    count++;
  }
  theend2:
  /* Optional user-defined check for line search step validity */
  if (snes->ops->postcheckstep) {
    ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
    if (changed_y) {
      ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
      ierr = SNESVIProjectOntoBounds(snes,w);CHKERRQ(ierr);
    }
    if (changed_y || changed_w) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,w,g);
      if (snes->domainerror) {
        ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      if (snes->ops->solve != SNESSolveVI_SS) {
        ierr = SNESVIComputeInactiveSetFnorm(snes,g,w,gnorm);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
      }

      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(((PetscObject)g)->comm,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESView_VI - Prints info from the SNESVI data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESView_VIRSAUG"
PetscErrorCode SNESView_VIRSAUG(SNES snes,PetscViewer viewer)
{
  const char     *cstr,*tstr;
  PetscErrorCode ierr;
  PetscBool     iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    cstr = SNESLineSearchTypeName(snes->ls_type);
    if (snes->ops->solve == SNESSolveVI_SS)         tstr = "Semismooth";
    else if (snes->ops->solve == SNESSolveVI_RS)    tstr = "Reduced Space";
    else if (snes->ops->solve == SNESSolveVI_RSAUG) tstr = "Reduced space with augmented variables";
    else                                         tstr = "unknown";
    ierr = PetscViewerASCIIPrintf(viewer,"  VI algorithm: %s\n",tstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  line search variant: %s\n",cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  alpha=%g, maxstep=%g, minlambda=%g\n",(double)snes->ls_alpha,(double)snes->maxstep,(double)snes->steptol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGSetVariableBounds"
/*@C
   SNESVIRSAUGSetVariableBounds - Sets the lower and upper bounds for the solution vector. xl <= x <= xu.

   Input Parameters:
.  snes - the SNES context.
.  xl   - lower bound.
.  xu   - upper bound.

   Notes:
   If this routine is not called then the lower and upper bounds are set to 
   SNES_VI_INF and SNES_VI_NINF respectively during SNESSetUp().

   Level: advanced

@*/
PetscErrorCode SNESVIRSAUGSetVariableBounds(SNES snes, Vec xl, Vec xu)
{
  SNES_VIRSAUG         *vi;
  PetscErrorCode    ierr;
  const PetscScalar *xxl,*xxu;
  PetscInt          i,n, cnt = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(xl,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xu,VEC_CLASSID,3);
  if (!snes->vec_func) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() first");
  if (xl->map->N != snes->vec_func->map->N) SETERRQ2(((PetscObject)snes)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths lower bound = %D solution vector = %D",xl->map->N,snes->vec_func->map->N);
  if (xu->map->N != snes->vec_func->map->N) SETERRQ2(((PetscObject)snes)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths: upper bound = %D solution vector = %D",xu->map->N,snes->vec_func->map->N);
  ierr = SNESSetType(snes,SNESVIRS);CHKERRQ(ierr);
  vi = (SNES_VIRSAUG*)snes->data;
  ierr = PetscObjectReference((PetscObject)xl);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)xu);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->xu);CHKERRQ(ierr);
  vi->xl = xl;
  vi->xu = xu;
  ierr = VecGetLocalSize(xl,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xl,&xxl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xu,&xxu);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    cnt += ((xxl[i] != SNES_VI_NINF) || (xxu[i] != SNES_VI_INF));
  }
  ierr = MPI_Allreduce(&cnt,&vi->ntruebounds,1,MPIU_INT,MPI_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xl,&xxl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xu,&xxu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetFromOptions_VIRSAUG - Sets various parameters for the SNESVI method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_VIRSAUG"
PetscErrorCode SNESSetFromOptions_VIRSAUG(SNES snes)
{
  SNES_VIRSAUG          *vi = (SNES_VIRSAUG *)snes->data;
  const char         *vies[] = {"ss","rs","rsaug"};
  PetscErrorCode     ierr;
  PetscInt           indx;
  PetscBool          flg,flg2;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES semismooth method options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_vi_monitor","Monitor all non-active variables","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESMonitorSet(snes,SNESMonitorVIRSAUG,0,0);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-snes_vi_const_tol","constraint tolerance","None",vi->const_tol,&vi->const_tol,0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-snes_vi_type","Semismooth algorithm used","",vies,3,"rs",&indx,&flg2);CHKERRQ(ierr);
  if (flg2) {
    switch (indx) {
    case 0:
      snes->ops->solve = SNESSolveVI_SS;
      break;
    case 1:
      snes->ops->solve = SNESSolveVI_RS;
      break;
    case 2:
      snes->ops->solve = SNESSolveVI_RSAUG;
    }
  }
  ierr = PetscOptionsBool("-snes_vi_ignore_function_sign","Ignore the sign of function for active constraints","None",vi->ignorefunctionsign,&vi->ignorefunctionsign,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType_VIRSAUG"
PetscErrorCode  SNESLineSearchSetType_VIRSAUG(SNES snes, SNESLineSearchType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (type) {
  case SNES_LS_BASIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNo_VIRSAUG,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_BASIC_NONORMS:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms_VIRSAUG,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_QUADRATIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic_VIRSAUG,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_CUBIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchCubic_VIRSAUG,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_SUP,"Unknown line search type");
    break;
  }
  snes->ls_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*MC
      SNESVIRSAUG - Reduced space active set solver that augments the Jacobian with additional variables to enforce constraints instead of by eliminating, for variational inequalities based on Newton's method

   Options Database:
+   -snes_vi_type <ss,rs,rsaug> a semi-smooth solver, a reduced space active set method, and a reduced space active set method that does not eliminate the active constraints from the Jacobian instead augments the Jacobian with additional variables that enforce the constraints
-   -snes_vi_monitor - prints the number of active constraints at each iteration.


   Level: beginner

.seealso:  SNESVISetVariableBounds(), SNESVISetComputeVariableBounds(), SNESCreate(), SNES, SNESSetType(), SNESVIRS, SNESVISS, SNESTR, SNESLineSearchSet(),
           SNESLineSearchSetPostCheck(), SNESLineSearchNo(), SNESLineSearchCubic(), SNESLineSearchQuadratic(), 
           SNESLineSearchSet(), SNESLineSearchNoNorms(), SNESLineSearchSetPreCheck(), SNESLineSearchSetParams(), SNESLineSearchGetParams()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_VI"
PetscErrorCode SNESCreate_VIRSAUG(SNES snes)
{
  PetscErrorCode ierr;
  SNES_VIRSAUG      *vi;

  PetscFunctionBegin;
  snes->ops->reset           = SNESReset_VIRSAUG;
  snes->ops->setup           = SNESSetUp_VIRSAUG;
  snes->ops->solve           = SNESSolveVI_RS;
  snes->ops->destroy         = SNESDestroy_VIRSAUG;
  snes->ops->setfromoptions  = SNESSetFromOptions_VIRSAUG;
  snes->ops->view            = SNESView_VIRSAUG;
  snes->ops->converged       = SNESDefaultConverged_VIRSAUG;

  snes->usesksp             = PETSC_TRUE;
  snes->usespc              = PETSC_FALSE;

  ierr                  = PetscNewLog(snes,SNES_VIRSAUG,&vi);CHKERRQ(ierr);
  snes->data                 = (void*)vi;
  snes->ls_alpha             = 1.e-4;
  snes->maxstep              = 1.e8;
  snes->steptol              = 1.e-12;
  vi->const_tol              =  2.2204460492503131e-16;
  vi->checkredundancy        = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESVISetVariableBounds_C","SNESVISetVariableBounds_VI",SNESVISetVariableBounds_VI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESVISetComputeVariableBounds_C","SNESVISetComputeVariableBounds_VI",SNESVISetComputeVariableBounds_VI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetType_C","SNESLineSearchSetType_VI",SNESLineSearchSetType_VIRSAUG);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(snes, SNES_LS_CUBIC);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SNESVIRSAUGGetInactiveSet"
/*
   SNESVIRSAUGGetInactiveSet - Gets the global indices for the inactive set variables (these correspond to the degrees of freedom the linear
     system is solved on)

   Input parameter
.  snes - the SNES context

   Output parameter
.  ISact - active set index set

 */
PetscErrorCode SNESVIRSAUGGetInactiveSet(SNES snes,IS* inact)
{
  SNES_VIRSAUG *vi = (SNES_VIRSAUG*)snes->data;
  PetscFunctionBegin;
  *inact = vi->IS_inact_prev;
  PetscFunctionReturn(0);
}
