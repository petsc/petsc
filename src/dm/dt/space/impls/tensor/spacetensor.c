#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceTensorCreateSubspace(PetscSpace space, PetscInt Nvs, PetscInt Ncs, PetscSpace *subspace)
{
  PetscInt       degree;
  const char    *prefix;
  const char    *name;
  char           subname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetDegree(space, &degree, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)space, &prefix);CHKERRQ(ierr);
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)space), subspace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*subspace, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*subspace, Nvs);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(*subspace, Ncs);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(*subspace, degree, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)*subspace, prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)*subspace, "tensorcomp_");CHKERRQ(ierr);
  if (((PetscObject) space)->name) {
    ierr = PetscObjectGetName((PetscObject)space, &name);CHKERRQ(ierr);
    ierr = PetscSNPrintf(subname, PETSC_MAX_PATH_LEN-1, "%s tensor component", name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)*subspace, subname);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectSetName((PetscObject)*subspace, "tensor component");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetFromOptions_Tensor(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscInt           Ns, Nc, i, Nv, deg;
  PetscBool          uniform = PETSC_TRUE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
  if (!Nv) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceTensorGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &deg, NULL);CHKERRQ(ierr);
  if (Ns > 1) {
    PetscSpace s0;

    ierr = PetscSpaceTensorGetSubspace(sp, 0, &s0);CHKERRQ(ierr);
    for (i = 1; i < Ns; i++) {
      PetscSpace si;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
      if (si != s0) {uniform = PETSC_FALSE; break;}
    }
  }
  Ns = (Ns == PETSC_DEFAULT) ? PetscMax(Nv,1) : Ns;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace tensor options");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscspace_tensor_spaces", "The number of subspaces", "PetscSpaceTensorSetNumSubspaces", Ns, &Ns, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_tensor_uniform", "Subspaces are identical", "PetscSpaceTensorSetFromOptions", uniform, &uniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscCheckFalse(Ns < 0 || (Nv > 0 && Ns == 0),PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a tensor space made up of %D spaces",Ns);
  PetscCheckFalse(Nv > 0 && Ns > Nv,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a tensor space with %D subspaces over %D variables", Ns, Nv);
  if (Ns != tens->numTensSpaces) {ierr = PetscSpaceTensorSetNumSubspaces(sp, Ns);CHKERRQ(ierr);}
  if (uniform) {
    PetscInt   Nvs = Nv / Ns;
    PetscInt   Ncs;
    PetscSpace subspace;

    PetscCheckFalse(Nv % Ns,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Cannot use %D uniform subspaces for %D variable space", Ns, Nv);
    Ncs = (PetscInt) PetscPowReal((PetscReal) Nc, 1./Ns);
    PetscCheckFalse(Nc % PetscPowInt(Ncs, Ns),PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Cannot use %D uniform subspaces for %D component space", Ns, Nc);
    ierr = PetscSpaceTensorGetSubspace(sp, 0, &subspace);CHKERRQ(ierr);
    if (!subspace) {ierr = PetscSpaceTensorCreateSubspace(sp, Nvs, Ncs, &subspace);CHKERRQ(ierr);}
    else           {ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);}
    ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
    for (i = 0; i < Ns; i++) {ierr = PetscSpaceTensorSetSubspace(sp, i, subspace);CHKERRQ(ierr);}
    ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
  } else {
    for (i = 0; i < Ns; i++) {
      PetscSpace subspace;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &subspace);CHKERRQ(ierr);
      if (!subspace) {
        char tprefix[128];

        ierr = PetscSpaceTensorCreateSubspace(sp, 1, 1, &subspace);CHKERRQ(ierr);
        ierr = PetscSNPrintf(tprefix, 128, "%d_",(int)i);CHKERRQ(ierr);
        ierr = PetscObjectAppendOptionsPrefix((PetscObject)subspace, tprefix);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
      }
      ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
      ierr = PetscSpaceTensorSetSubspace(sp, i, subspace);CHKERRQ(ierr);
      ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscBool          uniform = PETSC_TRUE;
  PetscInt           Ns = tens->numTensSpaces, i, n;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  for (i = 1; i < Ns; i++) {
    if (tens->tensspaces[i] != tens->tensspaces[0]) {uniform = PETSC_FALSE; break;}
  }
  if (uniform) {ierr = PetscViewerASCIIPrintf(v, "Tensor space of %D subspaces (all identical)\n", Ns);CHKERRQ(ierr);}
  else         {ierr = PetscViewerASCIIPrintf(v, "Tensor space of %D subspaces\n", Ns);CHKERRQ(ierr);}
  n = uniform ? 1 : Ns;
  for (i = 0; i < n; i++) {
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscSpaceView(tens->tensspaces[i], v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Tensor(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpaceTensorView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Tensor(PetscSpace sp)
{
  PetscSpace_Tensor *tens    = (PetscSpace_Tensor *) sp->data;
  PetscInt           Nc, Nv, Ns;
  PetscBool          uniform = PETSC_TRUE;
  PetscInt           deg, maxDeg;
  PetscInt           Ncprod;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (tens->setupCalled) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceTensorGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  if (Ns == PETSC_DEFAULT) {
    Ns = Nv;
    ierr = PetscSpaceTensorSetNumSubspaces(sp, Ns);CHKERRQ(ierr);
  }
  if (!Ns) {
    SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have zero subspaces");
  } else {
    PetscSpace s0;

    PetscCheckFalse(Nv > 0 && Ns > Nv,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a tensor space with %D subspaces over %D variables", Ns, Nv);
    ierr = PetscSpaceTensorGetSubspace(sp, 0, &s0);CHKERRQ(ierr);
    for (PetscInt i = 1; i < Ns; i++) {
      PetscSpace si;

      ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
      if (si != s0) {uniform = PETSC_FALSE; break;}
    }
    if (uniform) {
      PetscInt Nvs = Nv / Ns;
      PetscInt Ncs;

      PetscCheckFalse(Nv % Ns,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Cannot use %D uniform subspaces for %D variable space", Ns, Nv);
      Ncs = (PetscInt) (PetscPowReal((PetscReal) Nc, 1./Ns));
      PetscCheckFalse(Nc % PetscPowInt(Ncs, Ns),PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Cannot use %D uniform subspaces for %D component space", Ns, Nc);
      if (!s0) {ierr = PetscSpaceTensorCreateSubspace(sp, Nvs, Ncs, &s0);CHKERRQ(ierr);}
      else     {ierr = PetscObjectReference((PetscObject) s0);CHKERRQ(ierr);}
      ierr = PetscSpaceSetUp(s0);CHKERRQ(ierr);
      for (PetscInt i = 0; i < Ns; i++) {ierr = PetscSpaceTensorSetSubspace(sp, i, s0);CHKERRQ(ierr);}
      ierr = PetscSpaceDestroy(&s0);CHKERRQ(ierr);
      Ncprod = PetscPowInt(Ncs, Ns);
    } else {
      PetscInt Nvsum = 0;

      Ncprod = 1;
      for (PetscInt i = 0 ; i < Ns; i++) {
        PetscInt   Nvs, Ncs;
        PetscSpace si;

        ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
        if (!si) {ierr = PetscSpaceTensorCreateSubspace(sp, 1, 1, &si);CHKERRQ(ierr);}
        else     {ierr = PetscObjectReference((PetscObject) si);CHKERRQ(ierr);}
        ierr = PetscSpaceSetUp(si);CHKERRQ(ierr);
        ierr = PetscSpaceTensorSetSubspace(sp, i, si);CHKERRQ(ierr);
        ierr = PetscSpaceGetNumVariables(si, &Nvs);CHKERRQ(ierr);
        ierr = PetscSpaceGetNumComponents(si, &Ncs);CHKERRQ(ierr);
        ierr = PetscSpaceDestroy(&si);CHKERRQ(ierr);

        Nvsum += Nvs;
        Ncprod *= Ncs;
      }

      PetscCheckFalse(Nvsum != Nv,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Sum of subspace variables %D does not equal the number of variables %D", Nvsum, Nv);
      PetscCheckFalse(Nc % Ncprod,PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONG,"Product of subspace components %D does not divide the number of components %D", Ncprod, Nc);
    }
    if (Ncprod != Nc) {
      PetscInt    Ncopies = Nc / Ncprod;
      PetscInt    Nv = sp->Nv;
      const char *prefix;
      const char *name;
      char        subname[PETSC_MAX_PATH_LEN];
      PetscSpace  subsp;

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &subsp);CHKERRQ(ierr);
      ierr = PetscObjectGetOptionsPrefix((PetscObject)sp, &prefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)subsp, prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)subsp, "sumcomp_");CHKERRQ(ierr);
      if (((PetscObject)sp)->name) {
        ierr = PetscObjectGetName((PetscObject)sp, &name);CHKERRQ(ierr);
        ierr = PetscSNPrintf(subname, PETSC_MAX_PATH_LEN-1, "%s sum component", name);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)subsp, subname);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectSetName((PetscObject)subsp, "sum component");CHKERRQ(ierr);
      }
      ierr = PetscSpaceSetType(subsp, PETSCSPACETENSOR);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(subsp, Nv);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(subsp, Ncprod);CHKERRQ(ierr);
      ierr = PetscSpaceTensorSetNumSubspaces(subsp, Ns);CHKERRQ(ierr);
      for (PetscInt i = 0; i < Ns; i++) {
        PetscSpace ssp;

        ierr = PetscSpaceTensorGetSubspace(sp, i, &ssp);CHKERRQ(ierr);
        ierr = PetscSpaceTensorSetSubspace(subsp, i, ssp);CHKERRQ(ierr);
      }
      ierr = PetscSpaceSetUp(subsp);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sp, PETSCSPACESUM);CHKERRQ(ierr);
      ierr = PetscSpaceSumSetNumSubspaces(sp, Ncopies);CHKERRQ(ierr);
      for (PetscInt i = 0; i < Ncopies; i++) {
        ierr = PetscSpaceSumSetSubspace(sp, i, subsp);CHKERRQ(ierr);
      }
      ierr = PetscSpaceDestroy(&subsp);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  deg = PETSC_MAX_INT;
  maxDeg = 0;
  for (PetscInt i = 0; i < Ns; i++) {
    PetscSpace si;
    PetscInt   iDeg, iMaxDeg;

    ierr = PetscSpaceTensorGetSubspace(sp, i, &si);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(si, &iDeg, &iMaxDeg);CHKERRQ(ierr);
    deg    = PetscMin(deg, iDeg);
    maxDeg += iMaxDeg;
  }
  sp->degree    = deg;
  sp->maxDegree = maxDeg;
  tens->uniform = uniform;
  tens->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Tensor(PetscSpace sp)
{
  PetscSpace_Tensor *tens    = (PetscSpace_Tensor *) sp->data;
  PetscInt           Ns, i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  Ns = tens->numTensSpaces;
  if (tens->heightsubspaces) {
    PetscInt d;

    /* sp->Nv is the spatial dimension, so it is equal to the number
     * of subspaces on higher co-dimension points */
    for (d = 0; d < sp->Nv; ++d) {
      ierr = PetscSpaceDestroy(&tens->heightsubspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(tens->heightsubspaces);CHKERRQ(ierr);
  for (i = 0; i < Ns; i++) {ierr = PetscSpaceDestroy(&tens->tensspaces[i]);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetSubspace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetSubspace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetNumSubspaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetNumSubspaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(tens->tensspaces);CHKERRQ(ierr);
  ierr = PetscFree(tens);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Tensor(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscInt           i, Ns, d;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  Ns = tens->numTensSpaces;
  d  = 1;
  for (i = 0; i < Ns; i++) {
    PetscInt id;

    ierr = PetscSpaceGetDimension(tens->tensspaces[i], &id);CHKERRQ(ierr);
    d *= id;
  }
  *dim = d;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Tensor(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Tensor *tens  = (PetscSpace_Tensor *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         Nc      = sp->Nc;
  PetscInt         Nv      = sp->Nv;
  PetscInt         Ns;
  PetscReal       *lpoints, *sB = NULL, *sD = NULL, *sH = NULL;
  PetscInt         pdim;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!tens->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(sp, npoints, points, B, D, H);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  Ns = tens->numTensSpaces;
  ierr = PetscSpaceGetDimension(sp,&pdim);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  if (B || D || H) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc,       MPIU_REAL, &sB);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*Nv,    MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (H)           {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (B) {
    for (PetscInt i = 0; i < npoints*pdim*Nc; i++) B[i] = 1.;
  }
  if (D) {
    for (PetscInt i = 0; i < npoints*pdim*Nc*Nv; i++) D[i] = 1.;
  }
  if (H) {
    for (PetscInt i = 0; i < npoints*pdim*Nc*Nv*Nv; i++) H[i] = 1.;
  }
  for (PetscInt s = 0, d = 0, vstep = 1, cstep = 1; s < Ns; s++) {
    PetscInt sNv, sNc, spdim;
    PetscInt vskip, cskip;

    ierr = PetscSpaceGetNumVariables(tens->tensspaces[s], &sNv);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(tens->tensspaces[s], &sNc);CHKERRQ(ierr);
    ierr = PetscSpaceGetDimension(tens->tensspaces[s], &spdim);CHKERRQ(ierr);
    PetscCheckFalse((pdim % vstep) || (pdim % spdim),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad tensor loop: Nv %d, Ns %D, pdim %D, s %D, vstep %D, spdim %D", Nv, Ns, pdim, s, vstep, spdim);
    PetscCheckFalse((Nc % cstep) || (Nc % sNc),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad tensor loop: Nv %d, Ns %D, Nc %D, s %D, cstep %D, sNc %D", Nv, Ns, Nc, s, cstep, spdim);
    vskip = pdim / (vstep * spdim);
    cskip = Nc / (cstep * sNc);
    for (PetscInt p = 0; p < npoints; ++p) {
      for (PetscInt i = 0; i < sNv; i++) {
        lpoints[p * sNv + i] = points[p*Nv + d + i];
      }
    }
    ierr = PetscSpaceEvaluate(tens->tensspaces[s], npoints, lpoints, sB, sD, sH);CHKERRQ(ierr);
    if (B) {
      for (PetscInt k = 0; k < vskip; k++) {
        for (PetscInt si = 0; si < spdim; si++) {
          for (PetscInt j = 0; j < vstep; j++) {
            PetscInt i = (k * spdim + si) * vstep + j;

            for (PetscInt l = 0; l < cskip; l++) {
              for (PetscInt sc = 0; sc < sNc; sc++) {
                for (PetscInt m = 0; m < cstep; m++) {
                  PetscInt c = (l * sNc + sc) * cstep + m;

                  for (PetscInt p = 0; p < npoints; p++) {
                    B[(pdim * p + i) * Nc + c] *= sB[(spdim * p + si) * sNc + sc];
                  }
                }
              }
            }
          }
        }
      }
    }
    if (D) {
      for (PetscInt k = 0; k < vskip; k++) {
        for (PetscInt si = 0; si < spdim; si++) {
          for (PetscInt j = 0; j < vstep; j++) {
            PetscInt i = (k * spdim + si) * vstep + j;

            for (PetscInt l = 0; l < cskip; l++) {
              for (PetscInt sc = 0; sc < sNc; sc++) {
                for (PetscInt m = 0; m < cstep; m++) {
                  PetscInt c = (l * sNc + sc) * cstep + m;

                  for (PetscInt der = 0; der < Nv; der++) {
                    if (der >= d && der < d + sNv) {
                      for (PetscInt p = 0; p < npoints; p++) {
                        D[((pdim * p + i) * Nc + c)*Nv + der] *= sD[((spdim * p + si) * sNc + sc) * sNv + der - d];
                      }
                    } else {
                      for (PetscInt p = 0; p < npoints; p++) {
                        D[((pdim * p + i) * Nc + c)*Nv + der] *= sB[(spdim * p + si) * sNc + sc];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if (H) {
      for (PetscInt k = 0; k < vskip; k++) {
        for (PetscInt si = 0; si < spdim; si++) {
          for (PetscInt j = 0; j < vstep; j++) {
            PetscInt i = (k * spdim + si) * vstep + j;

            for (PetscInt l = 0; l < cskip; l++) {
              for (PetscInt sc = 0; sc < sNc; sc++) {
                for (PetscInt m = 0; m < cstep; m++) {
                  PetscInt c = (l * sNc + sc) * cstep + m;

                  for (PetscInt der = 0; der < Nv; der++) {
                    for (PetscInt der2 = 0; der2 < Nv; der2++) {
                      if (der >= d && der < d + sNv && der2 >= d && der2 < d + sNv) {
                        for (PetscInt p = 0; p < npoints; p++) {
                          H[(((pdim * p + i) * Nc + c)*Nv + der) * Nv + der2] *= sH[(((spdim * p + si) * sNc + sc) * sNv + der - d) * sNv + der2 - d];
                        }
                      } else if (der >= d && der < d + sNv) {
                        for (PetscInt p = 0; p < npoints; p++) {
                          H[(((pdim * p + i) * Nc + c)*Nv + der) * Nv + der2] *= sD[((spdim * p + si) * sNc + sc) * sNv + der - d];
                        }
                      } else if (der2 >= d && der2 < d + sNv) {
                        for (PetscInt p = 0; p < npoints; p++) {
                          H[(((pdim * p + i) * Nc + c)*Nv + der) * Nv + der2] *= sD[((spdim * p + si) * sNc + sc) * sNv + der2 - d];
                        }
                      } else {
                        for (PetscInt p = 0; p < npoints; p++) {
                          H[(((pdim * p + i) * Nc + c)*Nv + der) * Nv + der2] *= sB[(spdim * p + si) * sNc + sc];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    d += sNv;
    vstep *= spdim;
    cstep *= sNc;
  }
  if (H)           {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*Nv,    MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (B || D || H) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc,       MPIU_REAL, &sB);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceTensorSetNumSubspaces - Set the number of spaces in the tensor product

  Input Parameters:
+ sp  - the function space object
- numTensSpaces - the number of spaces

  Level: intermediate

.seealso: PetscSpaceTensorGetNumSubspaces(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceTensorSetNumSubspaces(PetscSpace sp, PetscInt numTensSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorSetNumSubspaces_C",(PetscSpace,PetscInt),(sp,numTensSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceTensorGetNumSubspaces - Get the number of spaces in the tensor product

  Input Parameter:
. sp  - the function space object

  Output Parameter:
. numTensSpaces - the number of spaces

  Level: intermediate

.seealso: PetscSpaceTensorSetNumSubspaces(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceTensorGetNumSubspaces(PetscSpace sp, PetscInt *numTensSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(numTensSpaces, 2);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorGetNumSubspaces_C",(PetscSpace,PetscInt*),(sp,numTensSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceTensorSetSubspace - Set a space in the tensor product

  Input Parameters:
+ sp    - the function space object
. s     - The space number
- subsp - the number of spaces

  Level: intermediate

.seealso: PetscSpaceTensorGetSubspace(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceTensorSetSubspace(PetscSpace sp, PetscInt s, PetscSpace subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (subsp) PetscValidHeaderSpecific(subsp, PETSCSPACE_CLASSID, 3);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorSetSubspace_C",(PetscSpace,PetscInt,PetscSpace),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceTensorGetSubspace - Get a space in the tensor product

  Input Parameters:
+ sp - the function space object
- s  - The space number

  Output Parameter:
. subsp - the PetscSpace

  Level: intermediate

.seealso: PetscSpaceTensorSetSubspace(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceTensorGetSubspace(PetscSpace sp, PetscInt s, PetscSpace *subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  ierr = PetscTryMethod(sp,"PetscSpaceTensorGetSubspace_C",(PetscSpace,PetscInt,PetscSpace*),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorSetNumSubspaces_Tensor(PetscSpace space, PetscInt numTensSpaces)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckFalse(tens->setupCalled,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of subspaces after setup called");
  Ns = tens->numTensSpaces;
  if (numTensSpaces == Ns) PetscFunctionReturn(0);
  if (Ns >= 0) {
    PetscInt s;

    for (s = 0; s < Ns; s++) {ierr = PetscSpaceDestroy(&tens->tensspaces[s]);CHKERRQ(ierr);}
    ierr = PetscFree(tens->tensspaces);CHKERRQ(ierr);
  }
  Ns = tens->numTensSpaces = numTensSpaces;
  ierr = PetscCalloc1(Ns, &tens->tensspaces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorGetNumSubspaces_Tensor(PetscSpace space, PetscInt *numTensSpaces)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;

  PetscFunctionBegin;
  *numTensSpaces = tens->numTensSpaces;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorSetSubspace_Tensor(PetscSpace space, PetscInt s, PetscSpace subspace)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckFalse(tens->setupCalled,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called");
  Ns = tens->numTensSpaces;
  PetscCheckFalse(Ns < 0,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceTensorSetNumSubspaces() first");
  PetscCheckFalse(s < 0 || s >= Ns,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D",subspace);
  ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&tens->tensspaces[s]);CHKERRQ(ierr);
  tens->tensspaces[s] = subspace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Tensor(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) sp->data;
  PetscInt         Nc, dim, order, i;
  PetscSpace       bsp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp, &dim);CHKERRQ(ierr);
  PetscCheckFalse(!tens->uniform || tens->numTensSpaces != dim,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Can only get a generic height subspace of a uniform tensor space of 1d spaces.");
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &order, NULL);CHKERRQ(ierr);
  PetscCheckFalse(height > dim || height < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!tens->heightsubspaces) {ierr = PetscCalloc1(dim, &tens->heightsubspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!tens->heightsubspaces[height-1]) {
      PetscSpace  sub;
      const char *name;

      ierr = PetscSpaceTensorGetSubspace(sp, 0, &bsp);CHKERRQ(ierr);
      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) sp,  &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACETENSOR);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(sub, order, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      ierr = PetscSpaceTensorSetNumSubspaces(sub, dim-height);CHKERRQ(ierr);
      for (i = 0; i < dim - height; i++) {
        ierr = PetscSpaceTensorSetSubspace(sub, i, bsp);CHKERRQ(ierr);
      }
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      tens->heightsubspaces[height-1] = sub;
    }
    *subsp = tens->heightsubspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceTensorGetSubspace_Tensor(PetscSpace space, PetscInt s, PetscSpace *subspace)
{
  PetscSpace_Tensor *tens = (PetscSpace_Tensor *) space->data;
  PetscInt           Ns;

  PetscFunctionBegin;
  Ns = tens->numTensSpaces;
  PetscCheckFalse(Ns < 0,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceTensorSetNumSubspaces() first");
  PetscCheckFalse(s < 0 || s >= Ns,PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D",subspace);
  *subspace = tens->tensspaces[s];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Tensor(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Tensor;
  sp->ops->setup             = PetscSpaceSetUp_Tensor;
  sp->ops->view              = PetscSpaceView_Tensor;
  sp->ops->destroy           = PetscSpaceDestroy_Tensor;
  sp->ops->getdimension      = PetscSpaceGetDimension_Tensor;
  sp->ops->evaluate          = PetscSpaceEvaluate_Tensor;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Tensor;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetNumSubspaces_C", PetscSpaceTensorGetNumSubspaces_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetNumSubspaces_C", PetscSpaceTensorSetNumSubspaces_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorGetSubspace_C", PetscSpaceTensorGetSubspace_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceTensorSetSubspace_C", PetscSpaceTensorSetSubspace_Tensor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACETENSOR = "tensor" - A PetscSpace object that encapsulates a tensor product space.
                     A tensor product is created of the components of the subspaces as well.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Tensor(PetscSpace sp)
{
  PetscSpace_Tensor *tens;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&tens);CHKERRQ(ierr);
  sp->data = tens;

  tens->numTensSpaces = PETSC_DEFAULT;

  ierr = PetscSpaceInitialize_Tensor(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
