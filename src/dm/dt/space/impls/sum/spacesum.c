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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  PetscValidIntPointer(numSumSpaces,2);
  CHKERRQ(PetscTryMethod(sp,"PetscSpaceSumGetNumSubspaces_C",(PetscSpace,PetscInt*),(sp,numSumSpaces)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  CHKERRQ(PetscTryMethod(sp,"PetscSpaceSumSetNumSubspaces_C",(PetscSpace,PetscInt),(sp,numSumSpaces)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  CHKERRQ(PetscTryMethod(sp,"PetscSpaceSumGetConcatenate_C",(PetscSpace,PetscBool*),(sp,concatenate)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  CHKERRQ(PetscTryMethod(sp,"PetscSpaceSumSetConcatenate_C",(PetscSpace,PetscBool),(sp,concatenate)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  PetscValidPointer(subsp,3);
  CHKERRQ(PetscTryMethod(sp,"PetscSpaceSumGetSubspace_C",(PetscSpace,PetscInt,PetscSpace*),(sp,s,subsp)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  if (subsp) PetscValidHeaderSpecific(subsp,PETSCSPACE_CLASSID,3);
  CHKERRQ(PetscTryMethod(sp,"PetscSpaceSumSetSubspace_C",(PetscSpace,PetscInt,PetscSpace),(sp,s,subsp)));
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

  PetscFunctionBegin;
  PetscCheckFalse(sum->setupCalled,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of subspaces after setup called");
  if (numSumSpaces == Ns) PetscFunctionReturn(0);
  if (Ns >= 0) {
    PetscInt s;
    for (s=0; s<Ns; ++s) {
      CHKERRQ(PetscSpaceDestroy(&sum->sumspaces[s]));
    }
    CHKERRQ(PetscFree(sum->sumspaces));
  }

  Ns   = sum->numSumSpaces = numSumSpaces;
  CHKERRQ(PetscCalloc1(Ns,&sum->sumspaces));
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
  PetscCheckFalse(sum->setupCalled,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change space concatenation after setup called.");

  sum->concatenate = concatenate;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace *subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns   = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheckFalse(Ns < 0,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first");
  PetscCheckFalse(s<0 || s>=Ns,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D",subspace);

  *subspace = sum->sumspaces[s];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns   = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheckFalse(sum->setupCalled,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called");
  PetscCheckFalse(Ns < 0,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first");
  PetscCheckFalse(s < 0 || s >= Ns,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D",subspace);

  CHKERRQ(PetscObjectReference((PetscObject)subspace));
  CHKERRQ(PetscSpaceDestroy(&sum->sumspaces[s]));
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
  CHKERRQ(PetscSpaceGetNumVariables(sp,&Nv));
  if (!Nv) PetscFunctionReturn(0);
  CHKERRQ(PetscSpaceGetNumComponents(sp,&Nc));
  CHKERRQ(PetscSpaceSumGetNumSubspaces(sp,&Ns));
  CHKERRQ(PetscSpaceGetDegree(sp,&deg,NULL));
  Ns   = (Ns == PETSC_DEFAULT) ? 1 : Ns;

  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"PetscSpace sum options"));
  CHKERRQ(PetscOptionsBoundedInt("-petscspace_sum_spaces","The number of subspaces","PetscSpaceSumSetNumSubspaces",Ns,&Ns,NULL,0));
  ierr = PetscOptionsBool("-petscspace_sum_concatenate","Subspaces are concatenated components of the final space","PetscSpaceSumSetFromOptions",
                          concatenate,&concatenate,NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsTail());

  PetscCheckFalse(Ns < 0 || (Nv > 0 && Ns == 0),PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a sum space of %D spaces",Ns);
  if (Ns != sum->numSumSpaces) {
    CHKERRQ(PetscSpaceSumSetNumSubspaces(sp,Ns));
  }
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)sp,&prefix));
  for (i=0; i<Ns; ++i) {
    PetscInt   sNv;
    PetscSpace subspace;

    CHKERRQ(PetscSpaceSumGetSubspace(sp,i,&subspace));
    if (!subspace) {
      char subspacePrefix[256];

      CHKERRQ(PetscSpaceCreate(PetscObjectComm((PetscObject)sp),&subspace));
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)subspace,prefix));
      CHKERRQ(PetscSNPrintf(subspacePrefix,256,"sumcomp_%D_",i));
      CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)subspace,subspacePrefix));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)subspace));
    }
    CHKERRQ(PetscSpaceSetFromOptions(subspace));
    CHKERRQ(PetscSpaceGetNumVariables(subspace,&sNv));
    PetscCheckFalse(!sNv,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Subspace %D has not been set properly, number of variables is 0.",i);
    CHKERRQ(PetscSpaceSumSetSubspace(sp,i,subspace));
    CHKERRQ(PetscSpaceDestroy(&subspace));
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

  PetscFunctionBegin;
  if (sum->setupCalled) PetscFunctionReturn(0);

  CHKERRQ(PetscSpaceGetNumVariables(sp,&Nv));
  CHKERRQ(PetscSpaceGetNumComponents(sp,&Nc));
  CHKERRQ(PetscSpaceSumGetNumSubspaces(sp,&Ns));
  if (Ns == PETSC_DEFAULT) {
    Ns   = 1;
    CHKERRQ(PetscSpaceSumSetNumSubspaces(sp,Ns));
  }
  PetscCheckFalse(Ns < 0,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have %D subspaces", Ns);
  uniform = PETSC_TRUE;
  if (Ns) {
    PetscSpace s0;

    CHKERRQ(PetscSpaceSumGetSubspace(sp,0,&s0));
    for (PetscInt i = 1; i < Ns; i++) {
      PetscSpace si;

      CHKERRQ(PetscSpaceSumGetSubspace(sp,i,&si));
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

    CHKERRQ(PetscSpaceSumGetSubspace(sp,i,&si));
    CHKERRQ(PetscSpaceSetUp(si));
    CHKERRQ(PetscSpaceGetNumVariables(si,&sNv));
    PetscCheckFalse(sNv != Nv,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Subspace %D has %D variables, space has %D.",i,sNv,Nv);
    CHKERRQ(PetscSpaceGetNumComponents(si,&sNc));
    if (i == 0 && sNc == Nc) concatenate = PETSC_FALSE;
    minNc = PetscMin(minNc, sNc);
    maxNc = PetscMax(maxNc, sNc);
    sum_Nc += sNc;
    CHKERRQ(PetscSpaceSumGetSubspace(sp,i,&si));
    CHKERRQ(PetscSpaceGetDegree(si,&iDeg,&iMaxDeg));
    deg     = PetscMin(deg,iDeg);
    maxDeg  = PetscMax(maxDeg,iMaxDeg);
  }

  if (concatenate) {
    if (sum_Nc != Nc) {
      SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Total number of subspace components (%D) does not match number of target space components (%D).",sum_Nc,Nc);
    }
  } else {
    PetscCheckFalse(minNc != Nc || maxNc != Nc,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspaces must have same number of components as the target space.");
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

  PetscFunctionBegin;
  if (concatenate) {
    CHKERRQ(PetscViewerASCIIPrintf(v,"Sum space of %D concatenated subspaces%s\n",Ns, sum->uniform ? " (all identical)": ""));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(v,"Sum space of %D subspaces%s\n",Ns, sum->uniform ? " (all identical)" : ""));
  }
  for (i=0; i < (sum->uniform ? (Ns > 0 ? 1 : 0) : Ns); ++i) {
    CHKERRQ(PetscViewerASCIIPushTab(v));
    CHKERRQ(PetscSpaceView(sum->sumspaces[i],v));
    CHKERRQ(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscSpaceSumView_Ascii(sp,viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       i,Ns   = sum->numSumSpaces;

  PetscFunctionBegin;
  for (i=0; i<Ns; ++i) {
    CHKERRQ(PetscSpaceDestroy(&sum->sumspaces[i]));
  }
  CHKERRQ(PetscFree(sum->sumspaces));
  if (sum->heightsubspaces) {
    PetscInt d;

    /* sp->Nv is the spatial dimension, so it is equal to the number
     * of subspaces on higher co-dimension points */
    for (d = 0; d < sp->Nv; ++d) {
      CHKERRQ(PetscSpaceDestroy(&sum->heightsubspaces[d]));
    }
  }
  CHKERRQ(PetscFree(sum->heightsubspaces));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",NULL));
  CHKERRQ(PetscFree(sum));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp,PetscInt *dim)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       i,d = 0,Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    CHKERRQ(PetscSpaceSetUp(sp));
    CHKERRQ(PetscSpaceGetDimension(sp, dim));
    PetscFunctionReturn(0);
  }

  for (i=0; i<Ns; ++i) {
    PetscInt id;

    CHKERRQ(PetscSpaceGetDimension(sum->sumspaces[i],&id));
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

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    CHKERRQ(PetscSpaceSetUp(sp));
    CHKERRQ(PetscSpaceEvaluate(sp, npoints, points, B, D, H));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscSpaceGetDimension(sp,&pdimfull));
  numelB = npoints*pdimfull*Nc;
  numelD = numelB*Nv;
  numelH = numelD*Nv;
  if (B || D || H) {
    CHKERRQ(DMGetWorkArray(dm,numelB,MPIU_REAL,&sB));
  }
  if (D || H) {
    CHKERRQ(DMGetWorkArray(dm,numelD,MPIU_REAL,&sD));
  }
  if (H) {
    CHKERRQ(DMGetWorkArray(dm,numelH,MPIU_REAL,&sH));
  }
  if (B)
    for (i=0; i<numelB; ++i) B[i] = 0.;
  if (D)
    for (i=0; i<numelD; ++i) D[i] = 0.;
  if (H)
    for (i=0; i<numelH; ++i) H[i] = 0.;

  for (s=0,offset=0,ncoffset=0; s<Ns; ++s) {
    PetscInt sNv,spdim,sNc,p;

    CHKERRQ(PetscSpaceGetNumVariables(sum->sumspaces[s],&sNv));
    CHKERRQ(PetscSpaceGetNumComponents(sum->sumspaces[s],&sNc));
    CHKERRQ(PetscSpaceGetDimension(sum->sumspaces[s],&spdim));
    PetscCheckFalse(offset + spdim > pdimfull,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspace dimensions exceed target space dimension.");
    if (s == 0 || !sum->uniform) {
      CHKERRQ(PetscSpaceEvaluate(sum->sumspaces[s],npoints,points,sB,sD,sH));
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
    CHKERRQ(DMRestoreWorkArray(dm,numelH,MPIU_REAL,&sH));
  }
  if (D || H) {
    CHKERRQ(DMRestoreWorkArray(dm,numelD,MPIU_REAL,&sD));
  }
  if (B || D || H) {
    CHKERRQ(DMRestoreWorkArray(dm,numelB,MPIU_REAL,&sB));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Sum(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Sum  *sum = (PetscSpace_Sum *) sp->data;
  PetscInt         Nc, dim, order;
  PetscBool        tensor;

  PetscFunctionBegin;
  CHKERRQ(PetscSpaceGetNumComponents(sp, &Nc));
  CHKERRQ(PetscSpaceGetNumVariables(sp, &dim));
  CHKERRQ(PetscSpaceGetDegree(sp, &order, NULL));
  CHKERRQ(PetscSpacePolynomialGetTensor(sp, &tensor));
  PetscCheckFalse(height > dim || height < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!sum->heightsubspaces) CHKERRQ(PetscCalloc1(dim, &sum->heightsubspaces));
  if (height <= dim) {
    if (!sum->heightsubspaces[height-1]) {
      PetscSpace  sub;
      const char *name;

      CHKERRQ(PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub));
      CHKERRQ(PetscObjectGetName((PetscObject) sp,  &name));
      CHKERRQ(PetscObjectSetName((PetscObject) sub,  name));
      CHKERRQ(PetscSpaceSetType(sub, PETSCSPACESUM));
      CHKERRQ(PetscSpaceSumSetNumSubspaces(sub, sum->numSumSpaces));
      CHKERRQ(PetscSpaceSumSetConcatenate(sub, sum->concatenate));
      CHKERRQ(PetscSpaceSetNumComponents(sub, Nc));
      CHKERRQ(PetscSpaceSetNumVariables(sub, dim-height));
      for (PetscInt i = 0; i < sum->numSumSpaces; i++) {
        PetscSpace subh;

        CHKERRQ(PetscSpaceGetHeightSubspace(sum->sumspaces[i], height, &subh));
        CHKERRQ(PetscSpaceSumSetSubspace(sub, i, subh));
      }
      CHKERRQ(PetscSpaceSetUp(sub));
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
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Sum;
  sp->ops->setup             = PetscSpaceSetUp_Sum;
  sp->ops->view              = PetscSpaceView_Sum;
  sp->ops->destroy           = PetscSpaceDestroy_Sum;
  sp->ops->getdimension      = PetscSpaceGetDimension_Sum;
  sp->ops->evaluate          = PetscSpaceEvaluate_Sum;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Sum;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",PetscSpaceSumGetNumSubspaces_Sum));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",PetscSpaceSumSetNumSubspaces_Sum));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",PetscSpaceSumGetSubspace_Sum));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",PetscSpaceSumSetSubspace_Sum));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetConcatenate_C",PetscSpaceSumGetConcatenate_Sum));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetConcatenate_C",PetscSpaceSumSetConcatenate_Sum));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  CHKERRQ(PetscNewLog(sp,&sum));
  sum->numSumSpaces = PETSC_DEFAULT;
  sp->data = sum;
  CHKERRQ(PetscSpaceInitialize_Sum(sp));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces,const PetscSpace subspaces[],PetscBool concatenate,PetscSpace *sumSpace)
{
  PetscInt       i,Nv,Nc = 0;

  PetscFunctionBegin;
  if (sumSpace) {
    CHKERRQ(PetscSpaceDestroy(sumSpace));
  }
  CHKERRQ(PetscSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]),sumSpace));
  CHKERRQ(PetscSpaceSetType(*sumSpace,PETSCSPACESUM));
  CHKERRQ(PetscSpaceSumSetNumSubspaces(*sumSpace,numSubspaces));
  CHKERRQ(PetscSpaceSumSetConcatenate(*sumSpace,concatenate));
  for (i=0; i<numSubspaces; ++i) {
    PetscInt sNc;

    CHKERRQ(PetscSpaceSumSetSubspace(*sumSpace,i,subspaces[i]));
    CHKERRQ(PetscSpaceGetNumComponents(subspaces[i],&sNc));
    if (concatenate) Nc += sNc;
    else Nc = sNc;
  }
  CHKERRQ(PetscSpaceGetNumVariables(subspaces[0],&Nv));
  CHKERRQ(PetscSpaceSetNumComponents(*sumSpace,Nc));
  CHKERRQ(PetscSpaceSetNumVariables(*sumSpace,Nv));
  CHKERRQ(PetscSpaceSetUp(*sumSpace));

  PetscFunctionReturn(0);
}
