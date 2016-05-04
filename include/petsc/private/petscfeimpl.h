#if !defined(_PETSCFEIMPL_H)
#define _PETSCFEIMPL_H

#include <petscfe.h>
#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmpleximpl.h>

PETSC_EXTERN PetscBool PetscSpaceRegisterAllCalled;
PETSC_EXTERN PetscBool PetscDualSpaceRegisterAllCalled;
PETSC_EXTERN PetscBool PetscFERegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscFERegisterAll(void);

typedef struct _PetscSpaceOps *PetscSpaceOps;
struct _PetscSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscSpace);
  PetscErrorCode (*setup)(PetscSpace);
  PetscErrorCode (*view)(PetscSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscSpace);

  PetscErrorCode (*getdimension)(PetscSpace,PetscInt*);
  PetscErrorCode (*evaluate)(PetscSpace,PetscInt,const PetscReal*,PetscReal*,PetscReal*,PetscReal*);
};

struct _p_PetscSpace {
  PETSCHEADER(struct _PetscSpaceOps);
  void    *data;  /* Implementation object */
  PetscInt order; /* The approximation order of the space */
  DM       dm;    /* Shell to use for temp allocation */
};

typedef struct {
  PetscInt   numVariables; /* The number of variables in the space, e.g. x and y */
  PetscBool  symmetric;    /* Use only symmetric polynomials */
  PetscBool  tensor;       /* Flag for tensor product */
  PetscInt  *degrees;      /* Degrees of single variable which we need to compute */
} PetscSpace_Poly;

typedef struct {
  PetscInt        numVariables; /* The spatial dimension */
  PetscQuadrature quad;         /* The points defining the space */
} PetscSpace_DG;

typedef struct _PetscDualSpaceOps *PetscDualSpaceOps;
struct _PetscDualSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscDualSpace);
  PetscErrorCode (*setup)(PetscDualSpace);
  PetscErrorCode (*view)(PetscDualSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscDualSpace);

  PetscErrorCode (*duplicate)(PetscDualSpace,PetscDualSpace*);
  PetscErrorCode (*getdimension)(PetscDualSpace,PetscInt*);
  PetscErrorCode (*getnumdof)(PetscDualSpace,const PetscInt**);
  PetscErrorCode (*getheightsubspace)(PetscDualSpace,PetscInt,PetscDualSpace *);
};

struct _p_PetscDualSpace {
  PETSCHEADER(struct _PetscDualSpaceOps);
  void            *data;       /* Implementation object */
  DM               dm;         /* The integration region K */
  PetscInt         order;      /* The approximation order of the space */
  PetscQuadrature *functional; /* The basis of functionals for this space */
  PetscBool        setupcalled;
};

typedef struct {
  PetscInt       *numDof;
  PetscBool       simplex;
  PetscBool       continuous;
  PetscInt        height;
  PetscDualSpace *subspaces;
} PetscDualSpace_Lag;

typedef struct {
  PetscInt  dim;
  PetscInt *numDof;
} PetscDualSpace_Simple;

typedef struct _PetscFEOps *PetscFEOps;
struct _PetscFEOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscFE);
  PetscErrorCode (*setup)(PetscFE);
  PetscErrorCode (*view)(PetscFE,PetscViewer);
  PetscErrorCode (*destroy)(PetscFE);
  PetscErrorCode (*getdimension)(PetscFE,PetscInt*);
  PetscErrorCode (*gettabulation)(PetscFE,PetscInt,const PetscReal*,PetscReal*,PetscReal*,PetscReal*);
  /* Element integration */
  PetscErrorCode (*integrate)(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscReal[]);
  PetscErrorCode (*integrateresidual)(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integratebdresidual)(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integratejacobianaction)(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integratejacobian)(PetscFE, PetscDS, PetscFEJacobianType, PetscInt, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integratebdjacobian)(PetscFE, PetscDS, PetscInt, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
};

struct _p_PetscFE {
  PETSCHEADER(struct _PetscFEOps);
  void           *data;          /* Implementation object */
  PetscSpace      basisSpace;    /* The basis space P */
  PetscDualSpace  dualSpace;     /* The dual space P' */
  PetscInt        numComponents; /* The number of field components */
  PetscQuadrature quadrature;    /* Suitable quadrature on K */
  PetscInt       *numDof;        /* The number of dof on mesh points of each depth */
  PetscReal      *invV;          /* Change of basis matrix, from prime to nodal basis set */
  PetscReal      *B, *D, *H;     /* Tabulation of basis and derivatives at quadrature points */
  PetscReal      *F;             /* Tabulation of basis at face centroids */
  PetscInt        blockSize, numBlocks;  /* Blocks are processed concurrently */
  PetscInt        batchSize, numBatches; /* A batch is made up of blocks, Batches are processed in serial */
};

typedef struct {
  PetscInt cellType;
} PetscFE_Basic;

typedef struct {
  PetscInt dummy;
} PetscFE_Nonaffine;

#ifdef PETSC_HAVE_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
  cl_platform_id   pf_id;
  cl_device_id     dev_id;
  cl_context       ctx_id;
  cl_command_queue queue_id;
  PetscDataType    realType;
  PetscLogEvent    residualEvent;
  PetscInt         op; /* ANDY: Stand-in for real equation code generation */
} PetscFE_OpenCL;
#endif

typedef struct {
  CellRefiner   cellRefiner;    /* The cell refiner defining the cell division */
  PetscInt      numSubelements; /* The number of subelements */
  PetscReal    *v0;             /* The affine transformation for each subelement */
  PetscReal    *jac, *invjac;
  PetscInt     *embedding;      /* Map from subelements dofs to element dofs */
} PetscFE_Composite;

/* Utility functions */
#undef __FUNCT__
#define __FUNCT__ "CoordinatesRefToReal"
PETSC_STATIC_INLINE void CoordinatesRefToReal(PetscInt dimReal, PetscInt dimRef, const PetscReal v0[], const PetscReal J[], const PetscReal xi[], PetscReal x[])
{
  PetscInt d, e;

  for (d = 0; d < dimReal; ++d) {
    x[d] = v0[d];
    for (e = 0; e < dimRef; ++e) {
      x[d] += J[d*dimReal+e]*(xi[e] + 1.0);
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "CoordinatesRealToRef"
PETSC_STATIC_INLINE void CoordinatesRealToRef(PetscInt dimReal, PetscInt dimRef, const PetscReal v0[], const PetscReal invJ[], const PetscReal x[], PetscReal xi[])
{
  PetscInt d, e;

  for (d = 0; d < dimRef; ++d) {
    xi[d] = -1.;
    for (e = 0; e < dimReal; ++e) {
      xi[d] += invJ[d*dimRef+e]*(x[e] - v0[e]);
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "EvaluateFieldJets"
PETSC_STATIC_INLINE PetscErrorCode EvaluateFieldJets(PetscDS prob, PetscBool bd, PetscInt q, const PetscReal invJ[], const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscScalar u[], PetscScalar u_x[], PetscScalar u_t[])
{
  PetscScalar   *refSpaceDer;
  PetscReal    **basisField, **basisFieldDer;
  PetscInt       dOffset = 0, fOffset = 0;
  PetscInt       Nf, Nc, dimRef, dimReal, d, f;
  PetscErrorCode ierr;

  if (!prob) return 0;
  ierr = PetscDSGetSpatialDimension(prob, &dimReal);CHKERRQ(ierr);
  dimRef = dimReal;
  if (bd) dimRef -= 1;
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, NULL, &refSpaceDer);CHKERRQ(ierr);
  if (bd) {ierr = PetscDSGetBdTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);}
  else    {ierr = PetscDSGetTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);}
  for (d = 0; d < Nc; ++d)          {u[d]   = 0.0;}
  for (d = 0; d < dimReal*Nc; ++d)      {u_x[d] = 0.0;}
  if (u_t) for (d = 0; d < Nc; ++d) {u_t[d] = 0.0;}
  for (f = 0; f < Nf; ++f) {
    const PetscReal *basis    = basisField[f];
    const PetscReal *basisDer = basisFieldDer[f];
    PetscObject      obj;
    PetscClassId     id;
    PetscInt         Nb, Ncf, b, c, e;

    if (bd) {ierr = PetscDSGetBdDiscretization(prob, f, &obj);CHKERRQ(ierr);}
    else    {ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);}
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Ncf);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      /* TODO Should also support reconstruction here */
      Nb   = 1;
      ierr = PetscFVGetNumComponents(fv, &Ncf);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
    for (d = 0; d < dimRef*Ncf; ++d) refSpaceDer[d] = 0.0;
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Ncf; ++c) {
        const PetscInt cidx = b*Ncf+c;

        u[fOffset+c]   += coefficients[dOffset+cidx]*basis[q*Nb*Ncf+cidx];
        for (d = 0; d < dimRef; ++d) refSpaceDer[c*dimRef+d] += coefficients[dOffset+cidx]*basisDer[(q*Nb*Ncf+cidx)*dimRef+d];
      }
    }
    for (c = 0; c < Ncf; ++c) for (d = 0; d < dimReal; ++d) for (e = 0; e < dimRef; ++e) u_x[(fOffset+c)*dimReal+d] += invJ[e*dimReal+d]*refSpaceDer[c*dimRef+e];
    if (u_t) {
      for (b = 0; b < Nb; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b*Ncf+c;

          u_t[fOffset+c] += coefficients_t[dOffset+cidx]*basis[q*Nb*Ncf+cidx];
        }
      }
    }
#if 0
    for (c = 0; c < Ncf; ++c) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, c, PetscRealPart(u[fOffset+c]));CHKERRQ(ierr);
      if (u_t) {ierr = PetscPrintf(PETSC_COMM_SELF, "    u_t[%d,%d]: %g\n", f, c, PetscRealPart(u_t[fOffset+c]));CHKERRQ(ierr);}
      for (d = 0; d < dimRef; ++d) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, c, 'x'+d, PetscRealPart(u_x[(fOffset+c)*dimReal+d]));CHKERRQ(ierr);
      }
    }
#endif
    fOffset += Ncf;
    dOffset += Nb*Ncf;
  }
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "EvaluateFaceFields"
PETSC_STATIC_INLINE PetscErrorCode EvaluateFaceFields(PetscDS prob, PetscInt field, PetscInt faceLoc, const PetscScalar coefficients[], PetscScalar u[])
{
  PetscFE        fe;
  PetscReal     *faceBasis;
  PetscInt       Nb, Nc, b, c;
  PetscErrorCode ierr;

  if (!prob) return 0;
  ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscFEGetFaceTabulation(fe, &faceBasis);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) {u[c] = 0.0;}
  for (b = 0; b < Nb; ++b) {
    for (c = 0; c < Nc; ++c) {
      const PetscInt cidx = b*Nc+c;

      u[c] += coefficients[cidx]*faceBasis[faceLoc*Nb*Nc+cidx];
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TransformF"
PETSC_STATIC_INLINE void TransformF(PetscInt dimReal, PetscInt dimRef, PetscInt Nc, PetscInt q, const PetscReal invJ[], PetscReal detJ, const PetscReal quadWeights[], PetscScalar refSpaceDer[], PetscScalar f0[], PetscScalar f1[])
{
  PetscInt c, d, e;

  for (c = 0; c < Nc; ++c) f0[q*Nc+c] *= detJ*quadWeights[q];
  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dimRef; ++d) {
      f1[(q*Nc + c)*dimRef+d] = 0.0;
      for (e = 0; e < dimReal; ++e) f1[(q*Nc + c)*dimRef+d] += invJ[d*dimReal+e]*refSpaceDer[c*dimReal+e];
      f1[(q*Nc + c)*dimRef+d] *= detJ*quadWeights[q];
    }
  }
#if 0
  if (debug > 1) {
    for (c = 0; c < Nc; ++c) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, PetscRealPart(f0[q*Nc+c]));CHKERRQ(ierr);
      for (d = 0; d < dimRef; ++d) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, PetscRealPart(f1[(q*Nc + c)*dimRef+d]));CHKERRQ(ierr);
      }
    }
  }
#endif
}

#undef __FUNCT__
#define __FUNCT__ "UpdateElementVec"
PETSC_STATIC_INLINE void UpdateElementVec(PetscInt dim, PetscInt Nq, PetscInt Nb, PetscInt Nc, PetscReal basis[], PetscReal basisDer[], PetscScalar f0[], PetscScalar f1[], PetscScalar elemVec[])
{
  PetscInt b, c;

  for (b = 0; b < Nb; ++b) {
    for (c = 0; c < Nc; ++c) {
      const PetscInt cidx = b*Nc+c;
      PetscInt       q;

      elemVec[cidx] = 0.0;
      for (q = 0; q < Nq; ++q) {
        PetscInt d;

        elemVec[cidx] += basis[q*Nb*Nc+cidx]*f0[q*Nc+c];
        for (d = 0; d < dim; ++d) elemVec[cidx] += basisDer[(q*Nb*Nc+cidx)*dim+d]*f1[(q*Nc+c)*dim+d];
      }
    }
  }
#if 0
  if (debug > 1) {
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, c, PetscRealPart(elemVec[b*Nc+c]));CHKERRQ(ierr);
      }
    }
  }
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEInterpolate_Static"
PETSC_STATIC_INLINE PetscErrorCode PetscFEInterpolate_Static(PetscFE fe, const PetscScalar x[], PetscInt q, PetscScalar interpolant[])
{
  PetscReal     *basis;
  PetscInt       Nb, Nc, fc, f;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscFEGetDefaultTabulation(fe, &basis, NULL, NULL);CHKERRQ(ierr);
  for (fc = 0; fc < Nc; ++fc) {
    interpolant[fc] = 0.0;
    for (f = 0; f < Nb; ++f) {
      const PetscInt fidx = f*Nc+fc;
      interpolant[fc] += x[fidx]*basis[q*Nb*Nc+fidx];
    }
  }
  PetscFunctionReturn(0);
}

#endif
