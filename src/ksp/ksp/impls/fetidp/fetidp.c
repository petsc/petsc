#include <petsc/private/kspimpl.h> /*I <petscksp.h> I*/
#include <../src/ksp/pc/impls/bddc/bddc.h>

/*
    This file implements the FETI-DP method in PETSc as part of KSP.
*/
typedef struct {
  KSP parentksp;
} KSP_FETIDPMon;

typedef struct {
  KSP           innerksp;        /* the KSP for the Lagrange multipliers */
  PC            innerbddc;       /* the inner BDDC object */
  PetscBool     fully_redundant; /* true for using a fully redundant set of multipliers */
  PetscBool     userbddc;        /* true if the user provided the PCBDDC object */
  KSP_FETIDPMon *monctx;         /* monitor context, used to pass user defined monitors
                                    in the physical space */
} KSP_FETIDP;

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerKSP_FETIDP"
static PetscErrorCode KSPFETIDPGetInnerKSP_FETIDP(KSP ksp,KSP* innerksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  *innerksp = fetidp->innerksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerKSP"
/*@
 KSPFETIDPGetInnerKSP - Get the KSP object for the Lagrange multipliers

   Input Parameters:
+  ksp - the FETI-DP KSP
-  innerksp - the KSP for the multipliers

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerBDDC
@*/
PetscErrorCode KSPFETIDPGetInnerKSP(KSP ksp,KSP* innerksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(innerksp,2);
  ierr = PetscUseMethod(ksp,"KSPFETIDPGetInnerKSP_C",(KSP,KSP*),(ksp,innerksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerBDDC_FETIDP"
static PetscErrorCode KSPFETIDPGetInnerBDDC_FETIDP(KSP ksp,PC* pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  *pc = fetidp->innerbddc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerBDDC"
/*@
 KSPFETIDPGetInnerBDDC - Get the PCBDDC object used to setup the FETI-DP matrix for the Lagrange multipliers

   Input Parameters:
+  ksp - the FETI-DP KSP
-  pc - the PCBDDC object

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerKSP
@*/
PetscErrorCode KSPFETIDPGetInnerBDDC(KSP ksp,PC* pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pc,2);
  ierr = PetscUseMethod(ksp,"KSPFETIDPGetInnerBDDC_C",(KSP,PC*),(ksp,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPSetInnerBDDC_FETIDP"
static PetscErrorCode KSPFETIDPSetInnerBDDC_FETIDP(KSP ksp,PC pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
  fetidp->innerbddc = pc;
  fetidp->userbddc = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPSetInnerBDDC"
/*@
 KSPFETIDPSetInnerBDDC - Set the PCBDDC object used to setup the FETI-DP matrix for the Lagrange multipliers

   Collective on KSP

   Input Parameters:
+  ksp - the FETI-DP KSP
-  pc - the PCBDDC object

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPGetInnerBDDC, KSPFETIDPGetInnerKSP
@*/
PetscErrorCode KSPFETIDPSetInnerBDDC(KSP ksp,PC pc)
{
  PetscBool      isbddc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc);CHKERRQ(ierr);
  if (!isbddc) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"KSPFETIDPSetInnerBDDC need a PCBDDC preconditioner");
  ierr = PetscTryMethod(ksp,"KSPFETIDPSetInnerBDDC_C",(KSP,PC),(ksp,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPBuildSolution_FETIDP"
static PetscErrorCode KSPBuildSolution_FETIDP(KSP ksp,Vec v,Vec *V)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  Mat            F;
  Vec            Xl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = KSPBuildSolution(fetidp->innerksp,NULL,&Xl);CHKERRQ(ierr);
  if (v) {
    ierr = PCBDDCMatFETIDPGetSolution(F,Xl,v);CHKERRQ(ierr);
    *V = v;
  } else {
    ierr = PCBDDCMatFETIDPGetSolution(F,Xl,*V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitor_FETIDP"
static PetscErrorCode KSPMonitor_FETIDP(KSP ksp,PetscInt it,PetscReal rnorm,void* ctx)
{
  KSP_FETIDPMon  *monctx = (KSP_FETIDPMon*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitor(monctx->parentksp,it,rnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPComputeEigenvalues_FETIDP"
static PetscErrorCode KSPComputeEigenvalues_FETIDP(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPComputeEigenvalues(fetidp->innerksp,nmax,r,c,neig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPComputeExtremeSingularValues_FETIDP"
static PetscErrorCode KSPComputeExtremeSingularValues_FETIDP(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPComputeExtremeSingularValues(fetidp->innerksp,emax,emin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPSetUpOperators"
static PetscErrorCode KSPFETIDPSetUpOperators(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  Mat            A,Ap;
  PetscInt       fid = -1;
  PetscBool      ismatis,saddle,pisz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  saddle = PETSC_FALSE;
  pisz   = PETSC_FALSE;
  ierr   = PetscOptionsBegin(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"FETI-DP options","PC");CHKERRQ(ierr);
  ierr   = PetscOptionsBool("-ksp_fetidp_saddlepoint","Activates support for saddle-point problems",NULL,saddle,&saddle,NULL);CHKERRQ(ierr);
  ierr   = PetscOptionsInt("-ksp_fetidp_pressure_field","Field id for pressures for saddle-point problems",NULL,fid,&fid,NULL);CHKERRQ(ierr);
  ierr   = PetscOptionsBool("-ksp_fetidp_pressure_iszero","Zero pressure block",NULL,pisz,&pisz,NULL);CHKERRQ(ierr);
  ierr   = PetscOptionsEnd();CHKERRQ(ierr);
  saddle = (fid >= 0 ? PETSC_TRUE : saddle);

  ierr = KSPGetOperators(ksp,&A,&Ap);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)Ap,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Pmat should be of type MATIS");

  /* see if MATIS has same fields attached */
  if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
    PetscContainer c;

    ierr = PetscObjectQuery((PetscObject)Ap,"_convert_nest_lfields",(PetscObject*)&c);CHKERRQ(ierr);
    if (c) {
      MatISLocalFields lf;
      ierr = PetscContainerGetPointer(c,(void**)&lf);CHKERRQ(ierr);
      ierr = PCBDDCSetDofsSplittingLocal(fetidp->innerbddc,lf->nr,lf->rf);CHKERRQ(ierr);
    }
  }

  if (!saddle) {
    ierr = PCSetOperators(fetidp->innerbddc,A,Ap);CHKERRQ(ierr);
  } else {
    Mat nA,lA;
    IS  pP,lP;

    ierr = MatISGetLocalMat(Ap,&lA);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",(PetscObject)lA);CHKERRQ(ierr);

    /* saved index sets */
    pP  = NULL;
    lP  = NULL;
    ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP" ,(PetscObject*)&pP);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP" ,(PetscObject*)&lP);CHKERRQ(ierr);
    if (!pP && !lP) { /* first time, need to compute boundary pressure dofs */
      PC_IS                  *pcis = (PC_IS*)fetidp->innerbddc->data;
      Mat_IS                 *matis = (Mat_IS*)(Ap->data);
      ISLocalToGlobalMapping l2g;
      IS                     II,pII,Pall;
      const PetscInt         *idxs;
      PetscInt               nl,ni,*widxs;
      PetscInt               i,j,n_neigh,*neigh,*n_shared,**shared,*count;
      PetscInt               rst,ren,n;
      PetscBool              ploc;

      ierr = MatGetLocalSize(Ap,&nl,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(Ap,&rst,&ren);CHKERRQ(ierr);
      ierr = MatGetLocalSize(lA,&n,NULL);CHKERRQ(ierr);

      if (!pcis->is_I_local) { /* need to compute interior dofs */
        ierr = PetscCalloc1(n,&count);CHKERRQ(ierr);
        ierr = MatGetLocalToGlobalMapping(Ap,&l2g,NULL);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetInfo(l2g,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
        for (i=1;i<n_neigh;i++)
          for (j=0;j<n_shared[i];j++)
            count[shared[i][j]] += 1;
        for (i=0,j=0;i<n;i++) if (!count[i]) count[j++] = i;
        ierr = ISLocalToGlobalMappingRestoreInfo(l2g,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,j,count,PETSC_OWN_POINTER,&II);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)pcis->is_I_local);CHKERRQ(ierr);
        II   = pcis->is_I_local;
      }

      /* interior dofs in layout */
      ierr = MatISSetUpSF(Ap);CHKERRQ(ierr);
      ierr = PetscMemzero(matis->sf_leafdata,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(matis->sf_rootdata,nl*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = ISGetLocalSize(II,&ni);CHKERRQ(ierr);
      ierr = ISGetIndices(II,&idxs);CHKERRQ(ierr);
      for (i=0;i<ni;i++) matis->sf_leafdata[idxs[i]] = 1;
      ierr = ISRestoreIndices(II,&idxs);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscMalloc1(PetscMax(nl,n),&widxs);CHKERRQ(ierr);
      for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&pII);CHKERRQ(ierr);

      /* pressure space at the interface */
      ploc = PETSC_FALSE;
      if (fid >= 0) {
        if (pcbddc->n_ISForDofsLocal) {
          PetscInt np;

          if (fid >= pcbddc->n_ISForDofsLocal) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Invalid field id for pressure %D, max %D",fid,pcbddc->n_ISForDofsLocal);
          /* need a sequential IS */
          ierr = ISGetLocalSize(pcbddc->ISForDofsLocal[fid],&np);CHKERRQ(ierr);
          ierr = ISGetIndices(pcbddc->ISForDofsLocal[fid],&idxs);CHKERRQ(ierr);
          ierr = ISCreateGeneral(PETSC_COMM_SELF,np,idxs,PETSC_COPY_VALUES,&Pall);CHKERRQ(ierr);
          ierr = ISRestoreIndices(pcbddc->ISForDofsLocal[fid],&idxs);CHKERRQ(ierr);
          ploc = PETSC_TRUE;
        } else if (pcbddc->n_ISForDofs) {
          if (fid >= pcbddc->n_ISForDofs) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Invalid field id for pressure %D, max %D",fid,pcbddc->n_ISForDofs);
          ierr = PetscObjectReference((PetscObject)pcbddc->ISForDofs[fid]);CHKERRQ(ierr);
          Pall = pcbddc->ISForDofs[fid];
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Missing fields! Use PCBDDCSetDofsSplitting/Local");
      } else { /* fallback to zero pressure block */
        ierr = MatFindZeroDiagonals(Ap,&Pall);CHKERRQ(ierr);
      }
      if (ploc) {
        ierr = ISDifference(Pall,II,&lP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP",(PetscObject)lP);CHKERRQ(ierr);
      } else {
        ierr = ISDifference(Pall,pII,&pP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",(PetscObject)pP);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&Pall);CHKERRQ(ierr);
      ierr = ISDestroy(&pII);CHKERRQ(ierr);
      ierr = ISDestroy(&II);CHKERRQ(ierr);

      /* local interface pressures in subdomain-wise and global ordering */
      ierr = PetscMemzero(matis->sf_leafdata,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(matis->sf_rootdata,nl*sizeof(PetscInt));CHKERRQ(ierr);
      if (pP) {
        ierr = ISGetLocalSize(pP,&ni);CHKERRQ(ierr);
        ierr = ISGetIndices(pP,&idxs);CHKERRQ(ierr);
        for (i=0;i<ni;i++) matis->sf_rootdata[idxs[i]-rst] = 1;
        ierr = ISRestoreIndices(pP,&idxs);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
        for (i=0,ni=0;i<n;i++) if (matis->sf_leafdata[i]) widxs[ni++] = i;
        ierr = ISLocalToGlobalMappingApply(l2g,ni,widxs,widxs+ni);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,ni,widxs,PETSC_COPY_VALUES,&lP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP",(PetscObject)lP);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs+ni,PETSC_COPY_VALUES,&Pall);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_gP",(PetscObject)Pall);CHKERRQ(ierr);
        ierr = ISDestroy(&Pall);CHKERRQ(ierr);
      } else {
        ierr = ISGetLocalSize(lP,&ni);CHKERRQ(ierr);
        ierr = ISGetIndices(lP,&idxs);CHKERRQ(ierr);
        for (i=0;i<ni;i++)
          if (idxs[i] >=0 && idxs[i] < n)
            matis->sf_leafdata[idxs[i]] = 1;
        ierr = ISRestoreIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPIU_REPLACE);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingApply(l2g,ni,idxs,widxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&Pall);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_gP",(PetscObject)Pall);CHKERRQ(ierr);
        ierr = ISDestroy(&Pall);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPIU_REPLACE);CHKERRQ(ierr);
        for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&pP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",(PetscObject)pP);CHKERRQ(ierr);
      }
      ierr = PetscFree(widxs);CHKERRQ(ierr);

      /* exclude interface pressures from the inner BDDC */
      if (pcbddc->DirichletBoundariesLocal) {
        IS       list[2],plP,isout;
        PetscInt np;

        /* need a parallel IS */
        ierr = ISGetLocalSize(lP,&np);CHKERRQ(ierr);
        ierr = ISGetIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),np,idxs,PETSC_USE_POINTER,&plP);CHKERRQ(ierr);
        list[0] = plP;
        list[1] = pcbddc->DirichletBoundariesLocal;
        ierr = ISConcatenate(PetscObjectComm((PetscObject)ksp),2,list,&isout);CHKERRQ(ierr);
        ierr = ISDestroy(&plP);CHKERRQ(ierr);
        ierr = ISRestoreIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc,isout);CHKERRQ(ierr);
        ierr = ISDestroy(&isout);CHKERRQ(ierr);
      } else if (pcbddc->DirichletBoundaries) {
        IS list[2],isout;

        list[0] = pP;
        list[1] = pcbddc->DirichletBoundaries;
        ierr = ISConcatenate(PetscObjectComm((PetscObject)ksp),2,list,&isout);CHKERRQ(ierr);
        ierr = PCBDDCSetDirichletBoundaries(fetidp->innerbddc,isout);CHKERRQ(ierr);
        ierr = ISDestroy(&isout);CHKERRQ(ierr);
      } else {
        IS       plP;
        PetscInt np;

        /* need a parallel IS */
        ierr = ISGetLocalSize(lP,&np);CHKERRQ(ierr);
        ierr = ISGetIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),np,idxs,PETSC_COPY_VALUES,&plP);CHKERRQ(ierr);
        ierr = PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc,plP);CHKERRQ(ierr);
        ierr = ISDestroy(&plP);CHKERRQ(ierr);
        ierr = ISRestoreIndices(lP,&idxs);CHKERRQ(ierr);
      }
    } else {
      if (!pP) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Missing global interface pressure field");
      if (!lP) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Missing local interface pressure field");
      ierr = PetscObjectReference((PetscObject)pP);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)lP);CHKERRQ(ierr);
    }

    /* Set operator for inner BDDC */
    ierr = MatDuplicate(Ap,MAT_COPY_VALUES,&nA);CHKERRQ(ierr);
    ierr = MatSetOption(nA,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(nA,pP,1.,NULL,NULL);CHKERRQ(ierr);
    ierr = PCSetOperators(fetidp->innerbddc,nA,nA);CHKERRQ(ierr);
    ierr = MatDestroy(&nA);CHKERRQ(ierr);

    /* non-zero rhs on interior dofs when applying the preconditioner */
    pcbddc->switch_static = PETSC_TRUE;

    /* Extract pressure block */
    if (!pisz) {
      Mat C;

      ierr = MatGetSubMatrix(Ap,pP,pP,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
      ierr = MatScale(C,-1.);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_C",(PetscObject)C);CHKERRQ(ierr);
      /* default operators for preconditioning boundary pressures */
      ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PAmat",(PetscObject)C);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject)C);CHKERRQ(ierr);
      ierr = MatDestroy(&C);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&lP);CHKERRQ(ierr);
    ierr = ISDestroy(&pP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_FETIDP"
static PetscErrorCode KSPSetUp_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPFETIDPSetUpOperators(ksp);CHKERRQ(ierr);
  /* set up BDDC */
  ierr = PCSetUp(fetidp->innerbddc);CHKERRQ(ierr);

  /* FETI-DP as it is implemented needs an exact coarse solver */
  if (pcbddc->coarse_ksp) {
    ierr = KSPSetTolerances(pcbddc->coarse_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000);CHKERRQ(ierr);
    ierr = KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_DEFAULT);CHKERRQ(ierr);
  }
  /* FETI-DP as it is implemented needs exact local Neumann solvers */
  ierr = KSPSetTolerances(pcbddc->ksp_R,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = KSPSetNormType(pcbddc->ksp_R,KSP_NORM_DEFAULT);CHKERRQ(ierr);

  /* if the primal space is changed, setup F */
  if (pcbddc->new_primal_space || !pcbddc->coarse_size) {
    Mat F; /* the FETI-DP matrix */
    PC  D; /* the FETI-DP preconditioner */
    ierr = KSPReset(fetidp->innerksp);CHKERRQ(ierr);
    ierr = PCBDDCCreateFETIDPOperators(fetidp->innerbddc,fetidp->fully_redundant,&F,&D);CHKERRQ(ierr);
    ierr = KSPSetOperators(fetidp->innerksp,F,F);CHKERRQ(ierr);
    ierr = KSPSetTolerances(fetidp->innerksp,ksp->rtol,ksp->abstol,ksp->divtol,ksp->max_it);CHKERRQ(ierr);
    ierr = KSPSetPC(fetidp->innerksp,D);CHKERRQ(ierr);
    ierr = MatCreateVecs(F,&(fetidp->innerksp)->vec_rhs,&(fetidp->innerksp)->vec_sol);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = PCDestroy(&D);CHKERRQ(ierr);
  }

  /* propagate settings to the inner solve */
  ierr = KSPGetComputeSingularValues(ksp,&flg);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues(fetidp->innerksp,flg);CHKERRQ(ierr);
  if (ksp->res_hist) {
    ierr = KSPSetResidualHistory(fetidp->innerksp,ksp->res_hist,ksp->res_hist_max,ksp->res_hist_reset);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(fetidp->innerksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_FETIDP"
static PetscErrorCode KSPSolve_FETIDP(KSP ksp)
{
  PetscErrorCode ierr;
  Mat            F;
  Vec            X,B,Xl,Bl;
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;

  PetscFunctionBegin;
  ierr = KSPGetRhs(ksp,&B);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&X);CHKERRQ(ierr);
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = KSPGetRhs(fetidp->innerksp,&Bl);CHKERRQ(ierr);
  ierr = KSPGetSolution(fetidp->innerksp,&Xl);CHKERRQ(ierr);
  ierr = PCBDDCMatFETIDPGetRHS(F,B,Bl);CHKERRQ(ierr);
  if (ksp->transpose_solve) {
    ierr = KSPSolveTranspose(fetidp->innerksp,Bl,Xl);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(fetidp->innerksp,Bl,Xl);CHKERRQ(ierr);
  }
  ierr = PCBDDCMatFETIDPGetSolution(F,Xl,X);CHKERRQ(ierr);
  /* update ksp with stats from inner ksp */
  ierr = KSPGetConvergedReason(fetidp->innerksp,&ksp->reason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(fetidp->innerksp,&ksp->its);CHKERRQ(ierr);
  ksp->totalits += ksp->its;
  ierr = KSPGetResidualHistory(fetidp->innerksp,NULL,&ksp->res_hist_len);CHKERRQ(ierr);
  /* restore defaults for inner BDDC (Pre/PostSolve flags) */
  pcbddc->temp_solution_used        = PETSC_FALSE;
  pcbddc->rhs_change                = PETSC_FALSE;
  pcbddc->exact_dirichlet_trick_app = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_FETIDP"
static PetscErrorCode KSPDestroy_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = KSPDestroy(&fetidp->innerksp);CHKERRQ(ierr);
  ierr = PetscFree(fetidp->monctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_FETIDP"
static PetscErrorCode KSPView_FETIDP(KSP ksp,PetscViewer viewer)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  FETI_DP: fully redundant: %d\n",fetidp->fully_redundant);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  FETI_DP: inner solver details\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,2);CHKERRQ(ierr);
  }
  ierr = KSPView(fetidp->innerksp,viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISubtractTab(viewer,2);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  FETI_DP: BDDC solver details\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,2);CHKERRQ(ierr);
  }
  ierr = PCView(fetidp->innerbddc,viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISubtractTab(viewer,2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_FETIDP"
static PetscErrorCode KSPSetFromOptions_FETIDP(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* set options prefixes for the inner objects, since the parent prefix will be valid at this point */
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerksp,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerksp,"fetidp_");CHKERRQ(ierr);
  if (!fetidp->userbddc) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerbddc,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerbddc,"fetidp_bddc_");CHKERRQ(ierr);
  }
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP FETIDP options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_fullyredundant","Use fully redundant multipliers","none",fetidp->fully_redundant,&fetidp->fully_redundant,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PCSetFromOptions(fetidp->innerbddc);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(fetidp->innerksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPFETIDP - The FETI-DP method

   This class implements the FETI-DP method [1].
   The preconditioning matrix for the KSP must be of type MATIS.
   The FETI-DP linear system (automatically generated constructing an internal PCBDDC object) is solved using an inner KSP object.

   Options Database Keys:
.   -ksp_fetidp_fullyredundant <false> : use a fully redundant set of Lagrange multipliers

   Level: Advanced

   Notes: Options for the inner KSP and for the customization of the PCBDDC object can be specified at command line by using the prefixes -fetidp_ and -fetidp_bddc_. E.g.,
.vb
      -fetidp_ksp_type gmres -fetidp_bddc_pc_bddc_symmetric false
.ve
   will use GMRES for the solution of the linear system on the Lagrange multipliers, generated using a non-symmetric PCBDDC.

   References:
.  [1] - C. Farhat, M. Lesoinne, P. LeTallec, K. Pierson, and D. Rixen, FETI-DP: a dual-primal unified FETI method. I. A faster alternative to the two-level FETI method, Internat. J. Numer. Methods Engrg., 50 (2001), pp. 1523--1544

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerBDDC, KSPFETIDPGetInnerKSP
M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_FETIDP"
PETSC_EXTERN PetscErrorCode KSPCreate_FETIDP(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_FETIDP     *fetidp;
  KSP_FETIDPMon  *monctx;
  PC_BDDC        *pcbddc;
  PC             pc;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&fetidp);CHKERRQ(ierr);
  ksp->data = (void*)fetidp;
  ksp->ops->setup                        = KSPSetUp_FETIDP;
  ksp->ops->solve                        = KSPSolve_FETIDP;
  ksp->ops->destroy                      = KSPDestroy_FETIDP;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_FETIDP;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_FETIDP;
  ksp->ops->view                         = KSPView_FETIDP;
  ksp->ops->setfromoptions               = KSPSetFromOptions_FETIDP;
  ksp->ops->buildsolution                = KSPBuildSolution_FETIDP;
  ksp->ops->buildresidual                = KSPBuildResidualDefault;
  /* create the inner KSP for the Lagrange multipliers */
  ierr = KSPCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerksp);CHKERRQ(ierr);
  ierr = KSPGetPC(fetidp->innerksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerksp);CHKERRQ(ierr);
  /* monitor */
  ierr = PetscNew(&monctx);CHKERRQ(ierr);
  monctx->parentksp = ksp;
  fetidp->monctx = monctx;
  ierr = KSPMonitorSet(fetidp->innerksp,KSPMonitor_FETIDP,fetidp->monctx,NULL);CHKERRQ(ierr);
  /* create the inner BDDC */
  ierr = PCCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = PCSetType(fetidp->innerbddc,PCBDDC);CHKERRQ(ierr);
  /* make sure we always obtain a consistent FETI-DP matrix
     for symmetric problems, the user can always customize it through the command line */
  pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerbddc);CHKERRQ(ierr);
  /* composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",KSPFETIDPSetInnerBDDC_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",KSPFETIDPGetInnerBDDC_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",KSPFETIDPGetInnerKSP_FETIDP);CHKERRQ(ierr);
  /* need to call KSPSetUp_FETIDP even with KSP_SETUP_NEWMATRIX */
  ksp->setupnewmatrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}
