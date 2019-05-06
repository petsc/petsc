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

PETSC_EXTERN PetscBool FEcite;
PETSC_EXTERN const char FECitation[];

typedef struct _PetscSpaceOps *PetscSpaceOps;
struct _PetscSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscSpace);
  PetscErrorCode (*setup)(PetscSpace);
  PetscErrorCode (*view)(PetscSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscSpace);

  PetscErrorCode (*getdimension)(PetscSpace,PetscInt*);
  PetscErrorCode (*evaluate)(PetscSpace,PetscInt,const PetscReal*,PetscReal*,PetscReal*,PetscReal*);
  PetscErrorCode (*getheightsubspace)(PetscSpace,PetscInt,PetscSpace *);
};

struct _p_PetscSpace {
  PETSCHEADER(struct _PetscSpaceOps);
  void                   *data;          /* Implementation object */
  PetscInt                degree;        /* The approximation order of the space */
  PetscInt                maxDegree;     /* The containing approximation order of the space */
  PetscInt                Nc;            /* The number of components */
  PetscInt                Nv;            /* The number of variables in the space, e.g. x and y */
  PetscInt                dim;           /* The dimension of the space */
  DM                      dm;            /* Shell to use for temp allocation */
};

typedef struct {
  PetscBool                symmetric;   /* Use only symmetric polynomials */
  PetscBool                tensor;      /* Flag for tensor product */
  PetscInt                *degrees;     /* Degrees of single variable which we need to compute */
  PetscSpacePolynomialType ptype;       /* Allows us to make the Hdiv and Hcurl spaces */
  PetscBool                setupCalled;
  PetscSpace              *subspaces;   /* Subspaces for each dimension */
} PetscSpace_Poly;

typedef struct {
  PetscSpace *tensspaces;
  PetscInt    numTensSpaces;
  PetscInt    dim;
  PetscBool   uniform;
  PetscBool   setupCalled;
  PetscSpace *heightsubspaces;    /* Height subspaces */
} PetscSpace_Tensor;

typedef struct {
  PetscQuadrature quad;         /* The points defining the space */
} PetscSpace_Point;

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
  PetscErrorCode (*getpointsubspace)(PetscDualSpace,PetscInt,PetscDualSpace *);
  PetscErrorCode (*getsymmetries)(PetscDualSpace,const PetscInt****,const PetscScalar****);
  PetscErrorCode (*apply)(PetscDualSpace, PetscInt, PetscReal, PetscFEGeom *, PetscInt, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *, PetscScalar *);
  PetscErrorCode (*applyall)(PetscDualSpace, const PetscScalar *, PetscScalar *);
  PetscErrorCode (*createallpoints)(PetscDualSpace, PetscQuadrature *);
};

struct _p_PetscDualSpace {
  PETSCHEADER(struct _PetscDualSpaceOps);
  void            *data;       /* Implementation object */
  DM               dm;         /* The integration region K */
  PetscInt         order;      /* The approximation order of the space */
  PetscInt         Nc;         /* The number of components */
  PetscQuadrature *functional; /* The basis of functionals for this space */
  PetscQuadrature  allPoints;  /* Collects all quadrature points representing functionals in the basis */
  PetscInt         k;          /* k-simplex corresponding to the dofs in this basis (we always use the 3D complex right now) */
  PetscBool        setupcalled;
  PetscBool        setfromoptionscalled;
};

typedef struct {
  PetscInt       *numDof;      /* [d]: Number of dofs for d-dimensional point */
  PetscBool       simplexCell; /* Flag for simplices, as opposed to tensor cells */
  PetscBool       tensorSpace; /* Flag for tensor product space of polynomials, as opposed to a space of maximum degree */
  PetscBool       continuous;  /* Flag for a continuous basis, as opposed to discontinuous across element boundaries */
  PetscInt        height;      /* The number of subspaces stored */
  PetscDualSpace *subspaces;   /* [h]: The subspace for dimension dim-(h+1) */
  PetscInt     ***symmetries;
  PetscInt        numSelfSym;
  PetscInt        selfSymOff;
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
  PetscErrorCode (*integrate)(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integratebd)(PetscDS, PetscInt, PetscBdPointFunc, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integrateresidual)(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
  PetscErrorCode (*integratebdresidual)(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
  PetscErrorCode (*integratejacobianaction)(PetscFE, PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
  PetscErrorCode (*integratejacobian)(PetscDS, PetscFEJacobianType, PetscInt, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
  PetscErrorCode (*integratebdjacobian)(PetscDS, PetscInt, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
};

struct _p_PetscFE {
  PETSCHEADER(struct _PetscFEOps);
  void           *data;                  /* Implementation object */
  PetscSpace      basisSpace;            /* The basis space P */
  PetscDualSpace  dualSpace;             /* The dual space P' */
  PetscInt        numComponents;         /* The number of field components */
  PetscQuadrature quadrature;            /* Suitable quadrature on K */
  PetscQuadrature faceQuadrature;        /* Suitable face quadrature on \partial K */
  PetscFE        *subspaces;             /* Subspaces for each dimension */
  PetscReal      *invV;                  /* Change of basis matrix, from prime to nodal basis set */
  PetscReal      *B,  *D,  *H;           /* Tabulation of basis and derivatives at quadrature points */
  PetscReal      *Bf, *Df, *Hf;          /* Tabulation of basis and derivatives at quadrature points on each face */
  PetscReal      *F;                     /* Tabulation of basis at face centroids */
  PetscInt        blockSize, numBlocks;  /* Blocks are processed concurrently */
  PetscInt        batchSize, numBatches; /* A batch is made up of blocks, Batches are processed in serial */
  PetscBool       setupcalled;
};

typedef struct {
  PetscInt cellType;
} PetscFE_Basic;

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
PETSC_STATIC_INLINE void CoordinatesRefToReal(PetscInt dimReal, PetscInt dimRef, const PetscReal xi0[], const PetscReal v0[], const PetscReal J[], const PetscReal xi[], PetscReal x[])
{
  PetscInt d, e;

  for (d = 0; d < dimReal; ++d) {
    x[d] = v0[d];
    for (e = 0; e < dimRef; ++e) {
      x[d] += J[d*dimReal+e]*(xi[e] - xi0[e]);
    }
  }
}

PETSC_STATIC_INLINE void CoordinatesRealToRef(PetscInt dimReal, PetscInt dimRef, const PetscReal xi0[], const PetscReal v0[], const PetscReal invJ[], const PetscReal x[], PetscReal xi[])
{
  PetscInt d, e;

  for (d = 0; d < dimRef; ++d) {
    xi[d] = xi0[d];
    for (e = 0; e < dimReal; ++e) {
      xi[d] += invJ[d*dimReal+e]*(x[e] - v0[e]);
    }
  }
}

PETSC_STATIC_INLINE PetscErrorCode EvaluateFieldJets(PetscDS ds, PetscInt dim, PetscInt Nf, const PetscInt Nb[], const PetscInt Nc[], PetscInt q, PetscReal *basisField[], PetscReal *basisFieldDer[], PetscFEGeom *fegeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscScalar u[], PetscScalar u_x[], PetscScalar u_t[])
{
  PetscInt       dOffset = 0, fOffset = 0, f;
  PetscErrorCode ierr;

  for (f = 0; f < Nf; ++f) {
    PetscFE          fe;
    const PetscInt   Nbf = Nb[f], Ncf = Nc[f];
    const PetscReal *Bq = &basisField[f][q*Nbf*Ncf];
    const PetscReal *Dq = &basisFieldDer[f][q*Nbf*Ncf*dim];
    PetscInt         b, c, d;

    ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fe);CHKERRQ(ierr);
    for (c = 0; c < Ncf; ++c)     u[fOffset+c] = 0.0;
    for (d = 0; d < dim*Ncf; ++d) u_x[fOffset*dim+d] = 0.0;
    for (b = 0; b < Nbf; ++b) {
      for (c = 0; c < Ncf; ++c) {
        const PetscInt cidx = b*Ncf+c;

        u[fOffset+c] += Bq[cidx]*coefficients[dOffset+b];
        for (d = 0; d < dim; ++d) u_x[(fOffset+c)*dim+d] += Dq[cidx*dim+d]*coefficients[dOffset+b];
      }
    }
    ierr = PetscFEPushforward(fe, fegeom, 1, &u[fOffset]);CHKERRQ(ierr);
    ierr = PetscFEPushforwardGradient(fe, fegeom, 1, &u_x[fOffset*dim]);CHKERRQ(ierr);
    if (u_t) {
      for (c = 0; c < Ncf; ++c) u_t[fOffset+c] = 0.0;
      for (b = 0; b < Nbf; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b*Ncf+c;

          u_t[fOffset+c] += Bq[cidx]*coefficients_t[dOffset+b];
        }
      }
      ierr = PetscFEPushforward(fe, fegeom, 1, &u_t[fOffset]);CHKERRQ(ierr);
    }
    fOffset += Ncf;
    dOffset += Nbf;
  }
  return 0;
}

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
  ierr = PetscFEGetFaceCentroidTabulation(fe, &faceBasis);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) {u[c] = 0.0;}
  for (b = 0; b < Nb; ++b) {
    for (c = 0; c < Nc; ++c) {
      const PetscInt cidx = b*Nc+c;

      u[c] += coefficients[cidx]*faceBasis[faceLoc*Nb*Nc+cidx];
    }
  }
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode UpdateElementVec(PetscFE fe, PetscInt dim, PetscInt Nq, PetscInt Nb, PetscInt Nc, PetscReal basis[], PetscReal basisDer[], PetscReal tmpBasis[], PetscReal tmpBasisDer[], PetscFEGeom *fegeom, PetscScalar f0[], PetscScalar f1[], PetscScalar elemVec[])
{
  PetscInt       q, b, c, d;
  PetscErrorCode ierr;

  for (b = 0; b < Nb; ++b) elemVec[b] = 0.0;
  for (q = 0; q < Nq; ++q) {
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b*Nc+c;

        tmpBasis[bcidx] = basis[q*Nb*Nc+bcidx];
        for (d = 0; d < dim; ++d) tmpBasisDer[bcidx*dim+d] = basisDer[q*Nb*Nc*dim+bcidx*dim+d];
      }
    }
    ierr = PetscFEPushforward(fe, fegeom, Nb, tmpBasis);CHKERRQ(ierr);
    ierr = PetscFEPushforwardGradient(fe, fegeom, Nb, tmpBasisDer);CHKERRQ(ierr);
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b*Nc+c;
        const PetscInt qcidx = q*Nc+c;

        elemVec[b] += tmpBasis[bcidx]*f0[qcidx];
        for (d = 0; d < dim; ++d) elemVec[b] += tmpBasisDer[bcidx*dim+d]*f1[qcidx*dim+d];
      }
    }
  }
  return(0);
}
PETSC_STATIC_INLINE PetscErrorCode UpdateElementMat(PetscFE feI, PetscFE feJ, PetscInt dim, PetscInt NbI, PetscInt NcI, const PetscReal basisI[], const PetscReal basisDerI[], PetscReal tmpBasisI[], PetscReal tmpBasisDerI[], PetscInt NbJ, PetscInt NcJ, const PetscReal basisJ[], const PetscReal basisDerJ[], PetscReal tmpBasisJ[], PetscReal tmpBasisDerJ[], PetscFEGeom *fegeom, const PetscScalar g0[], const PetscScalar g1[], const PetscScalar g2[], const PetscScalar g3[], PetscInt eOffset, PetscInt totDim, PetscInt offsetI, PetscInt offsetJ, PetscScalar elemMat[])
{
  PetscInt       f, fc, g, gc, df, dg;
  PetscErrorCode ierr;

  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f*NcI+fc; /* Test function basis index */

      tmpBasisI[fidx] = basisI[fidx];
      for (df = 0; df < dim; ++df) tmpBasisDerI[fidx*dim+df] = basisDerI[fidx*dim+df];
    }
  }
  ierr = PetscFEPushforward(feI, fegeom, NbI, tmpBasisI);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(feI, fegeom, NbI, tmpBasisDerI);CHKERRQ(ierr);
  for (g = 0; g < NbJ; ++g) {
    for (gc = 0; gc < NcJ; ++gc) {
      const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */

      tmpBasisJ[gidx] = basisJ[gidx];
      for (dg = 0; dg < dim; ++dg) tmpBasisDerJ[gidx*dim+dg] = basisDerJ[gidx*dim+dg];
    }
  }
  ierr = PetscFEPushforward(feJ, fegeom, NbJ, tmpBasisJ);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(feJ, fegeom, NbJ, tmpBasisDerJ);CHKERRQ(ierr);
  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f*NcI+fc; /* Test function basis index */
      const PetscInt i    = offsetI+f; /* Element matrix row */
      for (g = 0; g < NbJ; ++g) {
        for (gc = 0; gc < NcJ; ++gc) {
          const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
          const PetscInt j    = offsetJ+g; /* Element matrix column */
          const PetscInt fOff = eOffset+i*totDim+j;

          elemMat[fOff] += tmpBasisI[fidx]*g0[fc*NcJ+gc]*tmpBasisJ[gidx];
          for (df = 0; df < dim; ++df) {
            elemMat[fOff] += tmpBasisI[fidx]*g1[(fc*NcJ+gc)*dim+df]*tmpBasisDerJ[gidx*dim+df];
            elemMat[fOff] += tmpBasisDerI[fidx*dim+df]*g2[(fc*NcJ+gc)*dim+df]*tmpBasisJ[gidx];
            for (dg = 0; dg < dim; ++dg) {
              elemMat[fOff] += tmpBasisDerI[fidx*dim+df]*g3[((fc*NcJ+gc)*dim+df)*dim+dg]*tmpBasisDerJ[gidx*dim+dg];
            }
          }
        }
      }
    }
  }
  return(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscFEInterpolate_Static(PetscFE fe, const PetscScalar x[], PetscFEGeom *fegeom, PetscInt q, PetscScalar interpolant[])
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
      interpolant[fc] += x[f]*basis[(q*Nb + f)*Nc + fc];
    }
  }
  ierr = PetscFEPushforward(fe, fegeom, 1, interpolant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscFEInterpolateGradient_Static(PetscFE fe, const PetscScalar x[], PetscFEGeom *fegeom, PetscInt q, PetscScalar interpolant[])
{
  PetscReal     *basisDer;
  PetscInt       Nb, Nc, fc, f, d;
  const PetscInt dim = fegeom->dimEmbed;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscFEGetDefaultTabulation(fe, NULL, &basisDer, NULL);CHKERRQ(ierr);
  for (fc = 0; fc < Nc; ++fc) {
    for (d = 0; d < dim; ++d) interpolant[fc*dim+d] = 0.0;
    for (f = 0; f < Nb; ++f) {
      for (d = 0; d < dim; ++d) {
        interpolant[fc*dim+d] += x[f]*basisDer[((q*Nb + f)*Nc + fc)*dim + d];
      }
    }
  }
  ierr = PetscFEPushforwardGradient(fe, fegeom, 1, interpolant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscFEInterpolateFieldAndGradient_Static(PetscFE fe, const PetscScalar x[], PetscFEGeom *fegeom, PetscInt q, PetscScalar interpolant[], PetscScalar interpolantGrad[])
{
  PetscReal     *basis, *basisDer;
  PetscInt       Nb, Nc, fc, f, d;
  const PetscInt dim = fegeom->dimEmbed;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscFEGetDefaultTabulation(fe, &basis, &basisDer, NULL);CHKERRQ(ierr);
  for (fc = 0; fc < Nc; ++fc) {
    interpolant[fc] = 0.0;
    for (d = 0; d < dim; ++d) interpolantGrad[fc*dim+d] = 0.0;
    for (f = 0; f < Nb; ++f) {
      interpolant[fc] += x[f]*basis[(q*Nb + f)*Nc + fc];
      for (d = 0; d < dim; ++d) interpolantGrad[fc*dim+d] += x[f]*basisDer[((q*Nb + f)*Nc + fc)*dim + d];
    }
  }
  ierr = PetscFEPushforward(fe, fegeom, 1, interpolant);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(fe, fegeom, 1, interpolantGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscDualSpaceLatticePointLexicographic_Internal(PetscInt, PetscInt, PetscInt[]);
PETSC_INTERN PetscErrorCode PetscDualSpaceTensorPointLexicographic_Internal(PetscInt, PetscInt, PetscInt[]);

PETSC_EXTERN PetscErrorCode PetscFEGetDimension_Basic(PetscFE, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateResidual_Basic(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar [], const PetscScalar [], PetscDS, const PetscScalar [], PetscReal, PetscScalar []);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar [], const PetscScalar [], PetscDS, const PetscScalar [], PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscDS, PetscFEJacobianType, PetscInt, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar [], const PetscScalar [], PetscDS, const PetscScalar [], PetscReal, PetscReal, PetscScalar []);
#endif
