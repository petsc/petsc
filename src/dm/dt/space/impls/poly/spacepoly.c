#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

const char *const PetscSpacePolynomialTypes[] = {"P", "PMINUS_HDIV", "PMINUS_HCURL", "PetscSpacePolynomialType", "PETSCSPACE_POLYNOMIALTYPE_",0};

static PetscErrorCode PetscSpaceSetFromOptions_Polynomial(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace polynomial options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_sym", "Use only symmetric polynomials", "PetscSpacePolynomialSetSymmetric", poly->symmetric, &poly->symmetric, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_tensor", "Use the tensor product polynomials", "PetscSpacePolynomialSetTensor", poly->tensor, &poly->tensor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-petscspace_poly_type", "Type of polynomial space", "PetscSpacePolynomialSetType", PetscSpacePolynomialTypes, (PetscEnum)poly->ptype, (PetscEnum*)&poly->ptype, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(v, "%s%s%s space of degree %D\n", poly->ptype ? PetscSpacePolynomialTypes[poly->ptype] : "", poly->ptype ? " " : "", poly->tensor ? "Tensor polynomial" : "Polynomial", sp->degree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Polynomial(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePolynomialView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  PetscInt         ndegree = sp->degree+1;
  PetscInt         deg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (poly->setupCalled) PetscFunctionReturn(0);
  ierr = PetscMalloc1(ndegree, &poly->degrees);CHKERRQ(ierr);
  for (deg = 0; deg < ndegree; ++deg) poly->degrees[deg] = deg;
  if (poly->tensor) {
    sp->maxDegree = sp->degree + PetscMax(sp->Nv - 1,0);
  } else {
    sp->maxDegree = sp->degree;
  }
  poly->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(poly->degrees);CHKERRQ(ierr);
  if (poly->subspaces) {
    PetscInt d;

    for (d = 0; d < sp->Nv; ++d) {
      ierr = PetscSpaceDestroy(&poly->subspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(poly->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(poly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* We treat the space as a tensor product of scalar polynomial spaces, so the dimension is multiplied by Nc */
static PetscErrorCode PetscSpaceGetDimension_Polynomial(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         deg  = sp->degree;
  PetscInt         n    = sp->Nv, N, i;
  PetscReal        D    = 1.0;

  PetscFunctionBegin;
  if ((poly->ptype == PETSCSPACE_POLYNOMIALTYPE_PMINUS_HDIV) || (poly->ptype == PETSCSPACE_POLYNOMIALTYPE_PMINUS_HCURL)) --deg;
  if (poly->tensor) {
    N = 1;
    for (i = 0; i < n; ++i) N *= (deg+1);
  } else {
    for (i = 1; i <= n; ++i) {
      D *= ((PetscReal) (deg+i))/i;
    }
    N = (PetscInt) (D + 0.5);
  }
  if ((poly->ptype == PETSCSPACE_POLYNOMIALTYPE_PMINUS_HDIV) || (poly->ptype == PETSCSPACE_POLYNOMIALTYPE_PMINUS_HCURL)) {
    N *= sp->Nc + 1;
  } else {
    N *= sp->Nc;
  }
  *dim = N;
  PetscFunctionReturn(0);
}

/*
  LatticePoint_Internal - Returns all tuples of size 'len' with nonnegative integers that sum up to 'sum'.

  Input Parameters:
+ len - The length of the tuple
. sum - The sum of all entries in the tuple
- ind - The current multi-index of the tuple, initialized to the 0 tuple

  Output Parameter:
+ ind - The multi-index of the tuple, -1 indicates the iteration has terminated
. tup - A tuple of len integers addig to sum

  Level: developer

.seealso:
*/
static PetscErrorCode LatticePoint_Internal(PetscInt len, PetscInt sum, PetscInt ind[], PetscInt tup[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (len == 1) {
    ind[0] = -1;
    tup[0] = sum;
  } else if (sum == 0) {
    for (i = 0; i < len; ++i) {ind[0] = -1; tup[i] = 0;}
  } else {
    tup[0] = sum - ind[0];
    ierr = LatticePoint_Internal(len-1, ind[0], &ind[1], &tup[1]);CHKERRQ(ierr);
    if (ind[1] < 0) {
      if (ind[0] == sum) {ind[0] = -1;}
      else               {ind[1] = 0; ++ind[0];}
    }
  }
  PetscFunctionReturn(0);
}

/*
  TensorPoint_Internal - Returns all tuples of size 'len' with nonnegative integers that are less than 'max'.

  Input Parameters:
+ len - The length of the tuple
. max - The max for all entries in the tuple
- ind - The current multi-index of the tuple, initialized to the 0 tuple

  Output Parameter:
+ ind - The multi-index of the tuple, -1 indicates the iteration has terminated
. tup - A tuple of len integers less than max

  Level: developer

.seealso:
*/
static PetscErrorCode TensorPoint_Internal(PetscInt len, PetscInt max, PetscInt ind[], PetscInt tup[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (len == 1) {
    tup[0] = ind[0]++;
    ind[0] = ind[0] >= max ? -1 : ind[0];
  } else if (max == 0) {
    for (i = 0; i < len; ++i) {ind[0] = -1; tup[i] = 0;}
  } else {
    tup[0] = ind[0];
    ierr = TensorPoint_Internal(len-1, max, &ind[1], &tup[1]);CHKERRQ(ierr);
    if (ind[1] < 0) {
      ind[1] = 0;
      if (ind[0] == max-1) {ind[0] = -1;}
      else                 {++ind[0];}
    }
  }
  PetscFunctionReturn(0);
}

/*
  p in [0, npoints), i in [0, pdim), c in [0, Nc)

  B[p][i][c] = B[p][i_scalar][c][c]
*/
static PetscErrorCode PetscSpaceEvaluate_Polynomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  const PetscInt eps[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         Nc      = sp->Nc;
  PetscInt         ndegree = sp->degree+1;
  PetscInt        *degrees = poly->degrees;
  PetscInt         dim     = sp->Nv;
  PetscReal       *lpoints, *tmp, *LB, *LD, *LH;
  PetscInt        *ind, *tup;
  PetscInt         c, pdim, pdimRed, d, e, der, der2, i, p, deg, o;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  pdim /= Nc;
  ierr = DMGetWorkArray(dm, npoints, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, npoints*ndegree*3, MPIU_REAL, &tmp);CHKERRQ(ierr);
  if (B || D || H) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, MPIU_REAL, &LB);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, MPIU_REAL, &LD);CHKERRQ(ierr);}
  if (H)           {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, MPIU_REAL, &LH);CHKERRQ(ierr);}
  for (d = 0; d < dim; ++d) {
    for (p = 0; p < npoints; ++p) {
      lpoints[p] = points[p*dim+d];
    }
    ierr = PetscDTLegendreEval(npoints, lpoints, ndegree, degrees, tmp, &tmp[1*npoints*ndegree], &tmp[2*npoints*ndegree]);CHKERRQ(ierr);
    /* LB, LD, LH (ndegree * dim x npoints) */
    for (deg = 0; deg < ndegree; ++deg) {
      for (p = 0; p < npoints; ++p) {
        if (B || D || H) LB[(deg*dim + d)*npoints + p] = tmp[(0*npoints + p)*ndegree+deg];
        if (D || H)      LD[(deg*dim + d)*npoints + p] = tmp[(1*npoints + p)*ndegree+deg];
        if (H)           LH[(deg*dim + d)*npoints + p] = tmp[(2*npoints + p)*ndegree+deg];
      }
    }
  }
  /* Multiply by A (pdim x ndegree * dim) */
  ierr = PetscMalloc2(dim,&ind,dim,&tup);CHKERRQ(ierr);
  if (B) {
    PetscInt topDegree = sp->degree;

    /* B (npoints x pdim x Nc) */
    ierr = PetscArrayzero(B, npoints*pdim*Nc*Nc);CHKERRQ(ierr);
    if ((poly->ptype == PETSCSPACE_POLYNOMIALTYPE_PMINUS_HDIV) || (poly->ptype == PETSCSPACE_POLYNOMIALTYPE_PMINUS_HCURL)) topDegree--;
    /* Make complete space portion */
    if (poly->tensor) {
      if (poly->ptype != PETSCSPACE_POLYNOMIALTYPE_P) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Tensor spaces not supported for P^- spaces (%s)", PetscSpacePolynomialTypes[poly->ptype]);
      i = 0;
      ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->degree+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          B[(p*pdim + i)*Nc*Nc] = 1.0;
          for (d = 0; d < dim; ++d) {
            B[(p*pdim + i)*Nc*Nc] *= LB[(tup[d]*dim + d)*npoints + p];
          }
        }
        ++i;
      }
    } else {
      i = 0;
      for (o = 0; o <= topDegree; ++o) {
        ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
        while (ind[0] >= 0) {
          ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
          for (p = 0; p < npoints; ++p) {
            B[(p*pdim + i)*Nc*Nc] = 1.0;
            for (d = 0; d < dim; ++d) {
              B[(p*pdim + i)*Nc*Nc] *= LB[(tup[d]*dim + d)*npoints + p];
            }
          }
          ++i;
        }
      }
    }
    pdimRed = i;
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdimRed; ++i) {
        for (c = 1; c < Nc; ++c) {
          B[(p*pdim*Nc + i*Nc + c)*Nc + c] = B[(p*pdim + i)*Nc*Nc];
        }
      }
    }
    /* Make homogeneous part */
    if (topDegree < sp->degree) {
      if (poly->tensor) {
      } else {
        i = pdimRed;
        ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
        while (ind[0] >= 0) {
          ierr = LatticePoint_Internal(dim, topDegree, ind, tup);CHKERRQ(ierr);
          for (p = 0; p < npoints; ++p) {
            for (c = 0; c < Nc; ++c) {
              B[(p*pdim*Nc + i*Nc + c)*Nc + c] = 1.0;
              for (d = 0; d < dim; ++d) {
                B[(p*pdim*Nc + i*Nc + c)*Nc + c] *= LB[(tup[d]*dim + d)*npoints + p];
              }
              switch (poly->ptype) {
              case PETSCSPACE_POLYNOMIALTYPE_PMINUS_HDIV:
                B[(p*pdim*Nc + i*Nc + c)*Nc + c] *= LB[(c*dim + d)*npoints + p];break;
              case PETSCSPACE_POLYNOMIALTYPE_PMINUS_HCURL:
              {
                PetscReal sum = 0.0;
                for (d = 0; d < dim; ++d) for (e = 0; e < dim; ++e) sum += eps[c][d][e]*LB[(d*dim + d)*npoints + p];
                B[(p*pdim*Nc + i*Nc + c)*Nc + c] *= sum;
                break;
              }
              default: SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Invalid polynomial type %s", PetscSpacePolynomialTypes[poly->ptype]);
              }
            }
          }
          ++i;
        }
      }
    }
  }
  if (D) {
    if (poly->ptype != PETSCSPACE_POLYNOMIALTYPE_P) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Derivatives not supported for P^- spaces (%s)", PetscSpacePolynomialTypes[poly->ptype]);
    /* D (npoints x pdim x Nc x dim) */
    ierr = PetscArrayzero(D, npoints*pdim*Nc*Nc*dim);CHKERRQ(ierr);
    if (poly->tensor) {
      i = 0;
      ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->degree+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          for (der = 0; der < dim; ++der) {
            D[(p*pdim + i)*Nc*Nc*dim + der] = 1.0;
            for (d = 0; d < dim; ++d) {
              if (d == der) {
                D[(p*pdim + i)*Nc*Nc*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
              } else {
                D[(p*pdim + i)*Nc*Nc*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
              }
            }
          }
        }
        ++i;
      }
    } else {
      i = 0;
      for (o = 0; o <= sp->degree; ++o) {
        ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
        while (ind[0] >= 0) {
          ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
          for (p = 0; p < npoints; ++p) {
            for (der = 0; der < dim; ++der) {
              D[(p*pdim + i)*Nc*Nc*dim + der] = 1.0;
              for (d = 0; d < dim; ++d) {
                if (d == der) {
                  D[(p*pdim + i)*Nc*Nc*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
                } else {
                  D[(p*pdim + i)*Nc*Nc*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
                }
              }
            }
          }
          ++i;
        }
      }
    }
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          for (d = 0; d < dim; ++d) {
            D[((p*pdim*Nc + i*Nc + c)*Nc + c)*dim + d] = D[(p*pdim + i)*Nc*Nc*dim + d];
          }
        }
      }
    }
  }
  if (H) {
    if (poly->ptype != PETSCSPACE_POLYNOMIALTYPE_P) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Hessians not supported for P^- spaces (%s)", PetscSpacePolynomialTypes[poly->ptype]);
    /* H (npoints x pdim x Nc x Nc x dim x dim) */
    ierr = PetscArrayzero(H, npoints*pdim*Nc*Nc*dim*dim);CHKERRQ(ierr);
    if (poly->tensor) {
      i = 0;
      ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->degree+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          for (der = 0; der < dim; ++der) {
            H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der] = 1.0;
            for (d = 0; d < dim; ++d) {
              if (d == der) {
                H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der] *= LH[(tup[d]*dim + d)*npoints + p];
              } else {
                H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
              }
            }
            for (der2 = der + 1; der2 < dim; ++der2) {
              H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2] = 1.0;
              for (d = 0; d < dim; ++d) {
                if (d == der || d == der2) {
                  H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2] *= LD[(tup[d]*dim + d)*npoints + p];
                } else {
                  H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2] *= LB[(tup[d]*dim + d)*npoints + p];
                }
              }
              H[((p*pdim + i)*Nc*Nc*dim + der2) * dim + der] = H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2];
            }
          }
        }
        ++i;
      }
    } else {
      i = 0;
      for (o = 0; o <= sp->degree; ++o) {
        ierr = PetscArrayzero(ind, dim);CHKERRQ(ierr);
        while (ind[0] >= 0) {
          ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
          for (p = 0; p < npoints; ++p) {
            for (der = 0; der < dim; ++der) {
              H[((p*pdim + i)*Nc*Nc*dim + der)*dim + der] = 1.0;
              for (d = 0; d < dim; ++d) {
                if (d == der) {
                  H[((p*pdim + i)*Nc*Nc*dim + der)*dim + der] *= LH[(tup[d]*dim + d)*npoints + p];
                } else {
                  H[((p*pdim + i)*Nc*Nc*dim + der)*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
                }
              }
              for (der2 = der + 1; der2 < dim; ++der2) {
                H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2] = 1.0;
                for (d = 0; d < dim; ++d) {
                  if (d == der || d == der2) {
                    H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2] *= LD[(tup[d]*dim + d)*npoints + p];
                  } else {
                    H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2] *= LB[(tup[d]*dim + d)*npoints + p];
                  }
                }
                H[((p*pdim + i)*Nc*Nc*dim + der2) * dim + der] = H[((p*pdim + i)*Nc*Nc*dim + der) * dim + der2];
              }
            }
          }
          ++i;
        }
      }
    }
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          for (d = 0; d < dim; ++d) {
            for (e = 0; e < dim; ++e) {
              H[(((p*pdim*Nc + i*Nc + c)*Nc + c)*dim + d)*dim + e] = H[((p*pdim + i)*Nc*Nc*dim + d)*dim + e];
            }
          }
        }
      }
    }
  }
  ierr = PetscFree2(ind,tup);CHKERRQ(ierr);
  if (H)           {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, MPIU_REAL, &LH);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, MPIU_REAL, &LD);CHKERRQ(ierr);}
  if (B || D || H) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, MPIU_REAL, &LB);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*ndegree*3, MPIU_REAL, &tmp);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, npoints, MPIU_REAL, &lpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialSetTensor - Set whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variabl is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
+ sp     - the function space object
- tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Options Database:
. -petscspace_poly_tensor <bool> - Whether to use tensor product polynomials in higher dimension

  Level: intermediate

.seealso: PetscSpacePolynomialGetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace sp, PetscBool tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpacePolynomialSetTensor_C",(PetscSpace,PetscBool),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialGetTensor - Get whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variabl is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
. sp     - the function space object

  Output Parameters:
. tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Level: intermediate

.seealso: PetscSpacePolynomialSetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace sp, PetscBool *tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  ierr = PetscTryMethod(sp,"PetscSpacePolynomialGetTensor_C",(PetscSpace,PetscBool*),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialSetTensor_Polynomial(PetscSpace sp, PetscBool tensor)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  poly->tensor = tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialGetTensor_Polynomial(PetscSpace sp, PetscBool *tensor)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  *tensor = poly->tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Polynomial(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         Nc, dim, order;
  PetscBool        tensor;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(sp, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &order, NULL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(sp, &tensor);CHKERRQ(ierr);
  if (height > dim || height < 0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);}
  if (!poly->subspaces) {ierr = PetscCalloc1(dim, &poly->subspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!poly->subspaces[height-1]) {
      PetscSpace  sub;
      const char *name;

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) sp,  &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(sub, order, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      ierr = PetscSpacePolynomialSetTensor(sub, tensor);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      poly->subspaces[height-1] = sub;
    }
    *subsp = poly->subspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Polynomial(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Polynomial;
  sp->ops->setup             = PetscSpaceSetUp_Polynomial;
  sp->ops->view              = PetscSpaceView_Polynomial;
  sp->ops->destroy           = PetscSpaceDestroy_Polynomial;
  sp->ops->getdimension      = PetscSpaceGetDimension_Polynomial;
  sp->ops->evaluate          = PetscSpaceEvaluate_Polynomial;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Polynomial;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", PetscSpacePolynomialGetTensor_Polynomial);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", PetscSpacePolynomialSetTensor_Polynomial);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOLYNOMIAL = "poly" - A PetscSpace object that encapsulates a polynomial space, e.g. P1 is the space of
  linear polynomials. The space is replicated for each component.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&poly);CHKERRQ(ierr);
  sp->data = poly;

  poly->symmetric    = PETSC_FALSE;
  poly->tensor       = PETSC_FALSE;
  poly->degrees      = NULL;
  poly->subspaces    = NULL;

  ierr = PetscSpaceInitialize_Polynomial(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialSetSymmetric - Set whether a function space is a space of symmetric polynomials

  Input Parameters:
+ sp  - the function space object
- sym - flag for symmetric polynomials

  Options Database:
. -petscspace_poly_sym <bool> - Whether to use symmetric polynomials

  Level: intermediate

.seealso: PetscSpacePolynomialGetSymmetric(), PetscSpacePolynomialGetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace sp, PetscBool sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->symmetric = sym;
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialGetSymmetric - Get whether a function space is a space of symmetric polynomials

  Input Parameter:
. sp  - the function space object

  Output Parameter:
. sym - flag for symmetric polynomials

  Level: intermediate

.seealso: PetscSpacePolynomialSetSymmetric(), PetscSpacePolynomialGetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialGetSymmetric(PetscSpace sp, PetscBool *sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(sym, 2);
  *sym = poly->symmetric;
  PetscFunctionReturn(0);
}
