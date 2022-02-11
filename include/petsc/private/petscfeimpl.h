#if !defined(PETSCFEIMPL_H)
#define PETSCFEIMPL_H

#include <petscfe.h>
#ifdef PETSC_HAVE_LIBCEED
#include <petscfeceed.h>
#endif
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

PETSC_EXTERN PetscLogEvent PETSCDUALSPACE_SetUp;
PETSC_EXTERN PetscLogEvent PETSCFE_SetUp;

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
  PetscBool   tensor;      /* Flag for tensor product */
  PetscBool   setupCalled;
  PetscSpace *subspaces;   /* Subspaces for each dimension */
} PetscSpace_Poly;

typedef struct {
  PetscInt    formDegree;
  PetscBool   setupCalled;
  PetscSpace *subspaces;
} PetscSpace_Ptrimmed;

typedef struct {
  PetscSpace *tensspaces;
  PetscInt    numTensSpaces;
  PetscInt    dim;
  PetscBool   uniform;
  PetscBool   setupCalled;
  PetscSpace *heightsubspaces;    /* Height subspaces */
} PetscSpace_Tensor;

typedef struct {
  PetscSpace *sumspaces;
  PetscInt    numSumSpaces;
  PetscBool   uniform;
  PetscBool   concatenate;
  PetscBool   setupCalled;
  PetscSpace *heightsubspaces;    /* Height subspaces */
} PetscSpace_Sum;

typedef struct {
  PetscQuadrature quad;         /* The points defining the space */
} PetscSpace_Point;

typedef struct _PetscDualSpaceOps *PetscDualSpaceOps;
struct _PetscDualSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscDualSpace);
  PetscErrorCode (*setup)(PetscDualSpace);
  PetscErrorCode (*view)(PetscDualSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscDualSpace);

  PetscErrorCode (*duplicate)(PetscDualSpace,PetscDualSpace);
  PetscErrorCode (*createheightsubspace)(PetscDualSpace,PetscInt,PetscDualSpace *);
  PetscErrorCode (*createpointsubspace)(PetscDualSpace,PetscInt,PetscDualSpace *);
  PetscErrorCode (*getsymmetries)(PetscDualSpace,const PetscInt****,const PetscScalar****);
  PetscErrorCode (*apply)(PetscDualSpace, PetscInt, PetscReal, PetscFEGeom *, PetscInt, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *, PetscScalar *);
  PetscErrorCode (*applyall)(PetscDualSpace, const PetscScalar *, PetscScalar *);
  PetscErrorCode (*applyint)(PetscDualSpace, const PetscScalar *, PetscScalar *);
  PetscErrorCode (*createalldata)(PetscDualSpace, PetscQuadrature *, Mat *);
  PetscErrorCode (*createintdata)(PetscDualSpace, PetscQuadrature *, Mat *);
};

struct _p_PetscDualSpace {
  PETSCHEADER(struct _PetscDualSpaceOps);
  void            *data;       /* Implementation object */
  DM               dm;         /* The integration region K */
  PetscInt         order;      /* The approximation order of the space */
  PetscInt         Nc;         /* The number of components */
  PetscQuadrature *functional; /* The basis of functionals for this space */
  Mat              allMat;
  PetscQuadrature  allNodes;   /* Collects all quadrature points representing functionals in the basis */
  Vec              allNodeValues;
  Vec              allDofValues;
  Mat              intMat;
  PetscQuadrature  intNodes;   /* Collects all quadrature points representing functionals in the basis in the interior of the cell */
  Vec              intNodeValues;
  Vec              intDofValues;
  PetscInt         spdim;      /* The dual-space dimension */
  PetscInt         spintdim;   /* The dual-space interior dimension */
  PetscInt         k;          /* k-simplex corresponding to the dofs in this basis (we always use the 3D complex right now) */
  PetscBool        uniform;
  PetscBool        setupcalled;
  PetscBool        setfromoptionscalled;
  PetscSection     pointSection;
  PetscDualSpace  *pointSpaces;
  PetscDualSpace  *heightSpaces;
  PetscInt        *numDof;
};

typedef struct _n_Petsc1DNodeFamily *Petsc1DNodeFamily;
typedef struct _n_PetscLagNodeIndices *PetscLagNodeIndices;

PETSC_EXTERN PetscErrorCode PetscLagNodeIndicesGetData_Internal(PetscLagNodeIndices, PetscInt *, PetscInt *, PetscInt *, const PetscInt *[], const PetscReal *[]);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreateInteriorSymmetryMatrix_Lagrange(PetscDualSpace sp, PetscInt ornt, Mat *symMat);

typedef struct {
  /* these describe the types of dual spaces implemented */
  PetscBool         tensorCell;  /* Flag for tensor product cell */
  PetscBool         tensorSpace; /* Flag for tensor product space of polynomials, as opposed to a space of maximum degree */
  PetscBool         trimmed;     /* Flag for dual space of trimmed polynomial spaces */
  PetscBool         continuous;  /* Flag for a continuous basis, as opposed to discontinuous across element boundaries */

  PetscBool         interiorOnly; /* To make setup faster for tensor elements, only construct interior dofs in recursive calls */

  /* these keep track of symmetries */
  PetscInt       ***symperms;
  PetscScalar    ***symflips;
  PetscInt          numSelfSym;
  PetscInt          selfSymOff;
  PetscBool         symComputed;

  /* these describe different schemes of placing nodes in a simplex, from
   * which are derived all dofs in Lagrange dual spaces */
  PetscDTNodeType   nodeType;
  PetscBool         endNodes;
  PetscReal         nodeExponent;
  PetscInt          numNodeSkip; /* The number of end nodes from the 1D Node family to skip */
  Petsc1DNodeFamily nodeFamily;

  PetscInt          numCopies;

  PetscBool         useMoments;  /* Use moments for functionals */
  PetscInt          momentOrder; /* Order for moment quadrature */

  /* these are ways of indexing nodes in a way that makes
   * the computation of symmetries programmatic */
  PetscLagNodeIndices vertIndices;
  PetscLagNodeIndices intNodeIndices;
  PetscLagNodeIndices allNodeIndices;
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
  PetscErrorCode (*createtabulation)(PetscFE,PetscInt,const PetscReal*,PetscInt,PetscTabulation);
  /* Element integration */
  PetscErrorCode (*integrate)(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integratebd)(PetscDS, PetscInt, PetscBdPointFunc, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*integrateresidual)(PetscDS, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
  PetscErrorCode (*integratebdresidual)(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
  PetscErrorCode (*integratehybridresidual)(PetscDS, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
  PetscErrorCode (*integratejacobianaction)(PetscFE, PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
  PetscErrorCode (*integratejacobian)(PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
  PetscErrorCode (*integratebdjacobian)(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
  PetscErrorCode (*integratehybridjacobian)(PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
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
  PetscTabulation T;                     /* Tabulation of basis and derivatives at quadrature points */
  PetscTabulation Tf;                    /* Tabulation of basis and derivatives at quadrature points on each face */
  PetscTabulation Tc;                    /* Tabulation of basis at face centroids */
  PetscInt        blockSize, numBlocks;  /* Blocks are processed concurrently */
  PetscInt        batchSize, numBatches; /* A batch is made up of blocks, Batches are processed in serial */
  PetscBool       setupcalled;
#ifdef PETSC_HAVE_LIBCEED
  Ceed            ceed;                  /* The LibCEED context, usually set by the DM */
  CeedBasis       ceedBasis;             /* Basis for libCEED matching this element */
#endif
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
  PetscInt   numSubelements; /* The number of subelements */
  PetscReal *v0;             /* The affine transformation for each subelement */
  PetscReal *jac, *invjac;
  PetscInt  *embedding;      /* Map from subelements dofs to element dofs */
} PetscFE_Composite;

/* Utility functions */
static inline void CoordinatesRefToReal(PetscInt dimReal, PetscInt dimRef, const PetscReal xi0[], const PetscReal v0[], const PetscReal J[], const PetscReal xi[], PetscReal x[])
{
  PetscInt d, e;

  for (d = 0; d < dimReal; ++d) {
    x[d] = v0[d];
    for (e = 0; e < dimRef; ++e) {
      x[d] += J[d*dimReal+e]*(xi[e] - xi0[e]);
    }
  }
}

static inline void CoordinatesRealToRef(PetscInt dimReal, PetscInt dimRef, const PetscReal xi0[], const PetscReal v0[], const PetscReal invJ[], const PetscReal x[], PetscReal xi[])
{
  PetscInt d, e;

  for (d = 0; d < dimRef; ++d) {
    xi[d] = xi0[d];
    for (e = 0; e < dimReal; ++e) {
      xi[d] += invJ[d*dimReal+e]*(x[e] - v0[e]);
    }
  }
}

static inline PetscErrorCode PetscFEInterpolate_Static(PetscFE fe, const PetscScalar x[], PetscFEGeom *fegeom, PetscInt q, PetscScalar interpolant[])
{
  PetscTabulation T;
  PetscInt        fc, f;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = PetscFEGetCellTabulation(fe, 0, &T);CHKERRQ(ierr);
  {
    const PetscReal *basis = T->T[0];
    const PetscInt   Nb    = T->Nb;
    const PetscInt   Nc    = T->Nc;
    for (fc = 0; fc < Nc; ++fc) {
      interpolant[fc] = 0.0;
      for (f = 0; f < Nb; ++f) {
        interpolant[fc] += x[f]*basis[(q*Nb + f)*Nc + fc];
      }
    }
  }
  ierr = PetscFEPushforward(fe, fegeom, 1, interpolant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscFEInterpolateGradient_Static(PetscFE fe, PetscInt k, const PetscScalar x[], PetscFEGeom *fegeom, PetscInt q, PetscScalar interpolant[])
{
  PetscTabulation T;
  PetscInt        fc, f, d;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = PetscFEGetCellTabulation(fe, k, &T);CHKERRQ(ierr);
  {
    const PetscReal *basisDer = T->T[1];
    const PetscReal *basisHes = k > 1 ? T->T[2] : NULL;
    const PetscInt   Nb       = T->Nb;
    const PetscInt   Nc       = T->Nc;
    const PetscInt   cdim     = T->cdim;

    PetscCheck(cdim == fegeom->dimEmbed,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Geometry dim %" PetscInt_FMT " must match tabulation dim %" PetscInt_FMT, fegeom->dimEmbed, cdim);
    for (fc = 0; fc < Nc; ++fc) {
      for (d = 0; d < cdim; ++d) interpolant[fc*cdim+d] = 0.0;
      for (f = 0; f < Nb; ++f) {
        for (d = 0; d < cdim; ++d) {
          interpolant[fc*cdim+d] += x[f]*basisDer[((q*Nb + f)*Nc + fc)*cdim + d];
        }
      }
    }
    if (k > 1) {
      const PetscInt off = Nc*cdim;

      for (fc = 0; fc < Nc; ++fc) {
        for (d = 0; d < cdim*cdim; ++d) interpolant[off+fc*cdim*cdim+d] = 0.0;
        for (f = 0; f < Nb; ++f) {
          for (d = 0; d < cdim*cdim; ++d) interpolant[off+fc*cdim+d] += x[f]*basisHes[((q*Nb + f)*Nc + fc)*cdim*cdim + d];
        }
      }
    }
  }
  ierr = PetscFEPushforwardGradient(fe, fegeom, 1, interpolant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscFEFreeInterpolateGradient_Static(PetscFE fe, const PetscReal basisDer[], const PetscScalar x[], PetscInt dim, const PetscReal invJ[], const PetscReal n[], PetscInt q, PetscScalar interpolant[])
{
 PetscReal      realSpaceDer[3];
 PetscScalar    compGradient[3];
 PetscInt       Nb, Nc, fc, f=0, d, g;
 PetscErrorCode ierr;

 PetscFunctionBeginHot;
 ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
 ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);

 for (fc = 0; fc < Nc; ++fc) {
   interpolant[fc] = 0.0;
   for (d = 0; d < dim; ++d) compGradient[d] = 0.0;
   for (d = 0; d < dim; ++d) {
      for (d = 0; d < dim; ++d) {
        realSpaceDer[d] = 0.0;
        for (g = 0; g < dim; ++g) {
          realSpaceDer[d] += invJ[g*dim+d]*basisDer[((q*Nb + f)*Nc + fc)*dim + g];
        }
        compGradient[d] += x[f]*realSpaceDer[d];
      }
   }
   if (n) {
     for (d = 0; d < dim; ++d) interpolant[fc] += compGradient[d]*n[d];
   } else {
     for (d = 0; d < dim; ++d) interpolant[d] = compGradient[d];
   }
 }
 PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscFEInterpolateFieldAndGradient_Static(PetscFE fe, PetscInt k, const PetscScalar x[], PetscFEGeom *fegeom, PetscInt q, PetscScalar interpolant[], PetscScalar interpolantGrad[])
{
  PetscTabulation T;
  PetscInt        fc, f, d;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = PetscFEGetCellTabulation(fe, k, &T);CHKERRQ(ierr);
  {
    const PetscReal *basis    = T->T[0];
    const PetscReal *basisDer = T->T[1];
    const PetscReal *basisHes = k > 1 ? T->T[2] : NULL;
    const PetscInt   Nb       = T->Nb;
    const PetscInt   Nc       = T->Nc;
    const PetscInt   cdim     = T->cdim;

    PetscCheck(cdim == fegeom->dimEmbed,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Geometry dim %" PetscInt_FMT " must match tabulation dim %" PetscInt_FMT, fegeom->dimEmbed, cdim);
    for (fc = 0; fc < Nc; ++fc) {
      interpolant[fc] = 0.0;
      for (d = 0; d < cdim; ++d) interpolantGrad[fc*cdim+d] = 0.0;
      for (f = 0; f < Nb; ++f) {
        interpolant[fc] += x[f]*basis[(q*Nb + f)*Nc + fc];
        for (d = 0; d < cdim; ++d) interpolantGrad[fc*cdim+d] += x[f]*basisDer[((q*Nb + f)*Nc + fc)*cdim + d];
      }
    }
    if (k > 1) {
      const PetscInt off = Nc*cdim;

      for (fc = 0; fc < Nc; ++fc) {
        for (d = 0; d < cdim*cdim; ++d) interpolantGrad[off+fc*cdim*cdim+d] = 0.0;
        for (f = 0; f < Nb; ++f) {
          for (d = 0; d < cdim*cdim; ++d) interpolantGrad[off+fc*cdim+d] += x[f]*basisHes[((q*Nb + f)*Nc + fc)*cdim*cdim + d];
        }
      }
      ierr = PetscFEPushforwardHessian(fe, fegeom, 1, &interpolantGrad[off]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFEPushforward(fe, fegeom, 1, interpolant);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(fe, fegeom, 1, interpolantGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscDualSpaceLatticePointLexicographic_Internal(PetscInt, PetscInt, PetscInt[]);
PETSC_INTERN PetscErrorCode PetscDualSpaceTensorPointLexicographic_Internal(PetscInt, PetscInt, PetscInt[]);

PETSC_INTERN PetscErrorCode PetscDualSpaceSectionCreate_Internal(PetscDualSpace, PetscSection*);
PETSC_INTERN PetscErrorCode PetscDualSpaceSectionSetUp_Internal(PetscDualSpace, PetscSection);
PETSC_INTERN PetscErrorCode PetscDualSpacePushForwardSubspaces_Internal(PetscDualSpace, PetscInt, PetscInt);

PETSC_INTERN PetscErrorCode PetscFEEvaluateFieldJets_Internal(PetscDS, PetscInt, PetscInt, PetscInt, PetscTabulation[], PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscScalar[], PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEEvaluateFaceFields_Internal(PetscDS, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEUpdateElementVec_Internal(PetscFE, PetscTabulation, PetscInt, PetscScalar[], PetscScalar[], PetscInt, PetscFEGeom *, PetscScalar[], PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEUpdateElementMat_Internal(PetscFE, PetscFE, PetscInt, PetscInt, PetscTabulation, PetscScalar[], PetscScalar[], PetscTabulation, PetscScalar[], PetscScalar[], PetscFEGeom *, const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar[]);

PETSC_INTERN PetscErrorCode PetscFEEvaluateFieldJets_Hybrid_Internal(PetscDS, PetscInt, PetscInt, PetscInt, PetscTabulation[], PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscScalar[], PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEUpdateElementVec_Hybrid_Internal(PetscFE, PetscTabulation, PetscInt, PetscInt, PetscScalar[], PetscScalar[], PetscFEGeom *, PetscScalar[], PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEUpdateElementMat_Hybrid_Internal(PetscFE, PetscBool, PetscFE, PetscBool, PetscInt, PetscInt, PetscInt, PetscTabulation, PetscScalar[], PetscScalar[], PetscTabulation, PetscScalar[], PetscScalar[], PetscFEGeom *, const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar[]);

PETSC_EXTERN PetscErrorCode PetscFEGetDimension_Basic(PetscFE, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateResidual_Basic(PetscDS, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar [], const PetscScalar [], PetscDS, const PetscScalar [], PetscReal, PetscScalar []);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar [], const PetscScalar [], PetscDS, const PetscScalar [], PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar [], const PetscScalar [], PetscDS, const PetscScalar [], PetscReal, PetscReal, PetscScalar []);
#endif
