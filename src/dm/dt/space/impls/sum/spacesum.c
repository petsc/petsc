#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
/*@
  PetscSpaceSumGetNumSubspaces - Get the number of spaces in the sum

  Input Parameter:
  . sp  - the function space object

  Output Parameter:
  . numSumSpaces - the number of spaces

Level: intermediate

.seealso: PetscSpaceSumSetNumSubspaces(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace sp,PetscInt *numSumSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  PetscValidIntPointer(numSumSpaces,2);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetNumSubspaces_C",(PetscSpace,PetscInt*),(sp,numSumSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumSetNumSubspaces - Set the number of spaces in the sum

  Input Parameters:
  + sp  - the function space object
  - numSumSpaces - the number of spaces

Level: intermediate

.seealso: PetscSpaceSumGetNumSubspaces(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace sp,PetscInt numSumSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr = PetscTryMethod(sp,"PetscSpaceSumSetNumSubspaces_C",(PetscSpace,PetscInt),(sp,numSumSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PetscSpaceSumGetConcatenate - Get the concatenate flag for this space.
 A concatenated sum space will have number of components equal to the sum of the number of components of all subspaces.A non-concatenated,
 or direct sum space will have the same number of components as its subspaces .

 Input Parameters:
 . sp - the function space object

 Output Parameters:
 . concatenate - flag indicating whether subspaces are concatenated.

Level: intermediate

.seealso: PetscSpaceSumSetConcatenate()
@*/
PetscErrorCode PetscSpaceSumGetConcatenate(PetscSpace sp,PetscBool *concatenate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetConcatenate_C",(PetscSpace,PetscBool*),(sp,concatenate));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumSetConcatenate - Sets the concatenate flag for this space.
 A concatenated sum space will have number of components equal to the sum of the number of components of all subspaces.A non-concatenated,
 or direct sum space will have the same number of components as its subspaces .

 Input Parameters:
  + sp - the function space object
  - concatenate - are subspaces concatenated components (true) or direct summands (false)

Level: intermediate
.seealso: PetscSpaceSumGetConcatenate()
@*/
PetscErrorCode PetscSpaceSumSetConcatenate(PetscSpace sp,PetscBool concatenate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr = PetscTryMethod(sp,"PetscSpaceSumSetConcatenate_C",(PetscSpace,PetscBool),(sp,concatenate));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumGetSubspace - Get a space in the sum

  Input Parameters:
  + sp - the function space object
  - s  - The space number

  Output Parameter:
  . subsp - the PetscSpace

Level: intermediate

.seealso: PetscSpaceSumSetSubspace(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumGetSubspace(PetscSpace sp,PetscInt s,PetscSpace *subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  PetscValidPointer(subsp,3);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetSubspace_C",(PetscSpace,PetscInt,PetscSpace*),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumSetSubspace - Set a space in the sum

  Input Parameters:
  + sp    - the function space object
  . s     - The space number
  - subsp - the number of spaces

Level: intermediate

.seealso: PetscSpaceSumGetSubspace(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumSetSubspace(PetscSpace sp,PetscInt s,PetscSpace subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  if (subsp) PetscValidHeaderSpecific(subsp,PETSCSPACE_CLASSID,3);
  ierr = PetscTryMethod(sp,"PetscSpaceSumSetSubspace_C",(PetscSpace,PetscInt,PetscSpace),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetNumSubspaces_Sum(PetscSpace space,PetscInt *numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;

  PetscFunctionBegin;
  *numSumSpaces = sum->numSumSpaces;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetNumSubspaces_Sum(PetscSpace space,PetscInt numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns   = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of subspaces after setup called");
  if (numSumSpaces == Ns) PetscFunctionReturn(0);
  if (Ns >= 0) {
    PetscInt s;
    for (s=0; s<Ns; ++s) {
      ierr = PetscSpaceDestroy(&sum->sumspaces[s]);CHKERRQ(ierr);
    }
    ierr = PetscFree(sum->sumspaces);CHKERRQ(ierr);
  }

  Ns   = sum->numSumSpaces = numSumSpaces;
  ierr = PetscCalloc1(Ns,&sum->sumspaces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetConcatenate_Sum(PetscSpace sp,PetscBool *concatenate)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;

  PetscFunctionBegin;
  *concatenate = sum->concatenate;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetConcatenate_Sum(PetscSpace sp,PetscBool concatenate)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change space concatenation after setup called.");

  sum->concatenate = concatenate;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace *subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns   = sum->numSumSpaces;

  PetscFunctionBegin;
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first");
  if (s<0 || s>=Ns) SETERRQ1 (PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D",subspace);

  *subspace = sum->sumspaces[s];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns   = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called");
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first");
  if (s < 0 || s >= Ns) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D",subspace);

  ierr              = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
  ierr              = PetscSpaceDestroy(&sum->sumspaces[s]);CHKERRQ(ierr);
  sum->sumspaces[s] = subspace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetFromOptions_Sum(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       Ns,Nc,Nv,deg,i;
  PetscBool      concatenate = PETSC_TRUE;
  const char     *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp,&Nv);CHKERRQ(ierr);
  if (!Nv) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSumGetNumSubspaces(sp,&Ns);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp,&deg,NULL);CHKERRQ(ierr);
  Ns   = (Ns == PETSC_DEFAULT) ? 1 : Ns;

  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace sum options");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscspace_sum_spaces","The number of subspaces","PetscSpaceSumSetNumSubspaces",Ns,&Ns,NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_sum_concatenate","Subspaces are concatenated components of the final space","PetscSpaceSumSetFromOptions",
                          concatenate,&concatenate,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (Ns < 0 || (Nv > 0 && Ns == 0)) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a sum space of %D spaces",Ns);
  if (Ns != sum->numSumSpaces) {
    ierr = PetscSpaceSumSetNumSubspaces(sp,Ns);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetOptionsPrefix((PetscObject)sp,&prefix);CHKERRQ(ierr);
  for (i=0; i<Ns; ++i) {
    PetscInt   sNv;
    PetscSpace subspace;

    ierr = PetscSpaceSumGetSubspace(sp,i,&subspace);CHKERRQ(ierr);
    if (!subspace) {
      char subspacePrefix[256];

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)sp),&subspace);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)subspace,prefix);CHKERRQ(ierr);
      ierr = PetscSNPrintf(subspacePrefix,256,"sumcomp_%D_",i);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)subspace,subspacePrefix);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
    }
    ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumVariables(subspace,&sNv);CHKERRQ(ierr);
    if (!sNv) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Subspace %D has not been set properly, number of variables is 0.",i);
    ierr = PetscSpaceSumSetSubspace(sp,i,subspace);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscBool      concatenate = PETSC_TRUE;
  PetscBool      uniform;
  PetscInt       Nv,Ns,Nc,i,sum_Nc = 0,deg = PETSC_MAX_INT,maxDeg = PETSC_MIN_INT;
  PetscInt       minNc,maxNc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) PetscFunctionReturn(0);

  ierr = PetscSpaceGetNumVariables(sp,&Nv);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSumGetNumSubspaces(sp,&Ns);CHKERRQ(ierr);
  if (Ns == PETSC_DEFAULT) {
    Ns   = 1;
    ierr = PetscSpaceSumSetNumSubspaces(sp,Ns);CHKERRQ(ierr);
  }
  if (Ns < 0) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have %D subspaces", Ns);
  uniform = PETSC_TRUE;
  if (Ns) {
    PetscSpace s0;

    ierr = PetscSpaceSumGetSubspace(sp,0,&s0);CHKERRQ(ierr);
    for (PetscInt i = 1; i < Ns; i++) {
      PetscSpace si;

      ierr = PetscSpaceSumGetSubspace(sp,i,&si);CHKERRQ(ierr);
      if (si != s0) {
        uniform = PETSC_FALSE;
        break;
      }
    }
  }

  minNc = Nc;
  maxNc = Nc;
  for (i=0; i<Ns; ++i) {
    PetscInt   sNv,sNc,iDeg,iMaxDeg;
    PetscSpace si;

    ierr = PetscSpaceSumGetSubspace(sp,i,&si);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(si);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumVariables(si,&sNv);CHKERRQ(ierr);
    if (sNv != Nv) SETERRQ3(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Subspace %D has %D variables, space has %D.",i,sNv,Nv);
    ierr = PetscSpaceGetNumComponents(si,&sNc);CHKERRQ(ierr);
    if (i == 0 && sNc == Nc) concatenate = PETSC_FALSE;
    minNc = PetscMin(minNc, sNc);
    maxNc = PetscMax(maxNc, sNc);
    sum_Nc += sNc;
    ierr    = PetscSpaceSumGetSubspace(sp,i,&si);CHKERRQ(ierr);
    ierr    = PetscSpaceGetDegree(si,&iDeg,&iMaxDeg);CHKERRQ(ierr);
    deg     = PetscMin(deg,iDeg);
    maxDeg  = PetscMax(maxDeg,iMaxDeg);
  }

  if (concatenate) {
    if (sum_Nc != Nc) {
      SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Total number of subspace components (%D) does not match number of target space components (%D).",sum_Nc,Nc);
    }
  } else {
    if (minNc != Nc || maxNc != Nc) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspaces must have same number of components as the target space.");
  }

  sp->degree       = deg;
  sp->maxDegree    = maxDeg;
  sum->concatenate = concatenate;
  sum->uniform     = uniform;
  sum->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumView_Ascii(PetscSpace sp,PetscViewer v)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscBool      concatenate = sum->concatenate;
  PetscInt       i,Ns         = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (concatenate) {
    ierr = PetscViewerASCIIPrintf(v,"Sum space of %D concatenated subspaces%s\n",Ns, sum->uniform ? " (all identical)": "");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(v,"Sum space of %D subspaces%s\n",Ns, sum->uniform ? " (all identical)" : "");CHKERRQ(ierr);
  }
  for (i=0; i < (sum->uniform ? (Ns > 0 ? 1 : 0) : Ns); ++i) {
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscSpaceView(sum->sumspaces[i],v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp,PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscSpaceSumView_Ascii(sp,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       i,Ns   = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<Ns; ++i) {
    ierr = PetscSpaceDestroy(&sum->sumspaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sum->sumspaces);CHKERRQ(ierr);
  if (sum->heightsubspaces) {
    PetscInt d;

    /* sp->Nv is the spatial dimension, so it is equal to the number
     * of subspaces on higher co-dimension points */
    for (d = 0; d < sp->Nv; ++d) {
      ierr = PetscSpaceDestroy(&sum->heightsubspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(sum->heightsubspaces);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(sum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp,PetscInt *dim)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       i,d = 0,Ns = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    ierr = PetscSpaceGetDimension(sp, dim);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  for (i=0; i<Ns; ++i) {
    PetscInt id;

    ierr = PetscSpaceGetDimension(sum->sumspaces[i],&id);CHKERRQ(ierr);
    d   += id;
  }

  *dim = d;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Sum(PetscSpace sp,PetscInt npoints,const PetscReal points[],PetscReal B[],PetscReal D[],PetscReal H[])
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscBool      concatenate = sum->concatenate;
  DM             dm = sp->dm;
  PetscInt       Nc = sp->Nc,Nv = sp->Nv,Ns = sum->numSumSpaces;
  PetscInt       i,s,offset,ncoffset,pdimfull,numelB,numelD,numelH;
  PetscReal      *sB = NULL,*sD = NULL,*sH = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(sp, npoints, points, B, D, H);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr   = PetscSpaceGetDimension(sp,&pdimfull);CHKERRQ(ierr);
  numelB = npoints*pdimfull*Nc;
  numelD = numelB*Nv;
  numelH = numelD*Nv;
  if (B || D || H) {
    ierr = DMGetWorkArray(dm,numelB,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMGetWorkArray(dm,numelD,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (H) {
    ierr = DMGetWorkArray(dm,numelH,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (B)
    for (i=0; i<numelB; ++i) B[i] = 0.;
  if (D)
    for (i=0; i<numelD; ++i) D[i] = 0.;
  if (H)
    for (i=0; i<numelH; ++i) H[i] = 0.;

  for (s=0,offset=0,ncoffset=0; s<Ns; ++s) {
    PetscInt sNv,spdim,sNc,p;

    ierr = PetscSpaceGetNumVariables(sum->sumspaces[s],&sNv);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(sum->sumspaces[s],&sNc);CHKERRQ(ierr);
    ierr = PetscSpaceGetDimension(sum->sumspaces[s],&spdim);CHKERRQ(ierr);
    if (offset + spdim > pdimfull) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspace dimensions exceed target space dimension.");
    if (s == 0 || !sum->uniform) {
      ierr = PetscSpaceEvaluate(sum->sumspaces[s],npoints,points,sB,sD,sH);CHKERRQ(ierr);
    }
    if (B || D || H) {
      for (p=0; p<npoints; ++p) {
        PetscInt j;

        for (j=0; j<spdim; ++j) {
          PetscInt c;

          for (c=0; c<sNc; ++c) {
            PetscInt compoffset,BInd,sBInd;

            compoffset = concatenate ? c+ncoffset : c;
            BInd       = (p*pdimfull + j + offset)*Nc + compoffset;
            sBInd      = (p*spdim + j)*sNc + c;
            if (B) B[BInd] = sB[sBInd];
            if (D || H) {
              PetscInt v;

              for (v=0; v<Nv; ++v) {
                PetscInt DInd,sDInd;

                DInd  = BInd*Nv + v;
                sDInd = sBInd*Nv + v;
                if (D) D[DInd] = sD[sDInd];
                if (H) {
                  PetscInt v2;

                  for (v2=0; v2<Nv; ++v2) {
                    PetscInt HInd,sHInd;

                    HInd    = DInd*Nv + v2;
                    sHInd   = sDInd*Nv + v2;
                    H[HInd] = sH[sHInd];
                  }
                }
              }
            }
          }
        }
      }
    }
    offset   += spdim;
    ncoffset += sNc;
  }

  if (H) {
    ierr = DMRestoreWorkArray(dm,numelH,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMRestoreWorkArray(dm,numelD,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (B || D || H) {
    ierr = DMRestoreWorkArray(dm,numelB,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Sum(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Sum  *sum = (PetscSpace_Sum *) sp->data;
  PetscInt         Nc, dim, order;
  PetscBool        tensor;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(sp, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &order, NULL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(sp, &tensor);CHKERRQ(ierr);
  if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!sum->heightsubspaces) {ierr = PetscCalloc1(dim, &sum->heightsubspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!sum->heightsubspaces[height-1]) {
      PetscSpace  sub;
      const char *name;

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) sp,  &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACESUM);CHKERRQ(ierr);
      ierr = PetscSpaceSumSetNumSubspaces(sub, sum->numSumSpaces);CHKERRQ(ierr);
      ierr = PetscSpaceSumSetConcatenate(sub, sum->concatenate);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      for (PetscInt i = 0; i < sum->numSumSpaces; i++) {
        PetscSpace subh;

        ierr = PetscSpaceGetHeightSubspace(sum->sumspaces[i], height, &subh);CHKERRQ(ierr);
        ierr = PetscSpaceSumSetSubspace(sub, i, subh);CHKERRQ(ierr);
      }
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      sum->heightsubspaces[height-1] = sub;
    }
    *subsp = sum->heightsubspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Sum(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Sum;
  sp->ops->setup             = PetscSpaceSetUp_Sum;
  sp->ops->view              = PetscSpaceView_Sum;
  sp->ops->destroy           = PetscSpaceDestroy_Sum;
  sp->ops->getdimension      = PetscSpaceGetDimension_Sum;
  sp->ops->evaluate          = PetscSpaceEvaluate_Sum;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Sum;

  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",PetscSpaceSumGetNumSubspaces_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",PetscSpaceSumSetNumSubspaces_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",PetscSpaceSumGetSubspace_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",PetscSpaceSumSetSubspace_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetConcatenate_C",PetscSpaceSumGetConcatenate_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetConcatenate_C",PetscSpaceSumSetConcatenate_Sum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACESUM = "sum" - A PetscSpace object that encapsulates a sum of subspaces.
  That sum can either be direct or concatenate a concatenation.For example if A and B are spaces each with 2 components,
  the direct sum of A and B will also have 2 components while the concatenated sum will have 4 components.In both cases A and B must be defined over the
  same number of variables.

Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr     = PetscNewLog(sp,&sum);CHKERRQ(ierr);
  sum->numSumSpaces = PETSC_DEFAULT;
  sp->data = sum;
  ierr     = PetscSpaceInitialize_Sum(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces,const PetscSpace subspaces[],PetscBool concatenate,PetscSpace *sumSpace)
{
  PetscInt       i,Nv,Nc = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sumSpace) {
    ierr = PetscSpaceDestroy(sumSpace);CHKERRQ(ierr);
  }
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]),sumSpace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*sumSpace,PETSCSPACESUM);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetNumSubspaces(*sumSpace,numSubspaces);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetConcatenate(*sumSpace,concatenate);CHKERRQ(ierr);
  for (i=0; i<numSubspaces; ++i) {
    PetscInt sNc;

    ierr = PetscSpaceSumSetSubspace(*sumSpace,i,subspaces[i]);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(subspaces[i],&sNc);CHKERRQ(ierr);
    if (concatenate) Nc += sNc;
    else Nc = sNc;
  }
  ierr = PetscSpaceGetNumVariables(subspaces[0],&Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(*sumSpace,Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*sumSpace,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(*sumSpace);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
