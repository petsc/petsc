/* Discretization tools */

#include <petscdt.h>            /*I "petscdt.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>
#include <petscviewer.h>
#include <petscdmplex.h>
#include <petscdmshell.h>

#if defined(PETSC_HAVE_MPFR)
#include <mpfr.h>
#endif

const char *const PetscDTNodeTypes[] = {"gaussjacobi", "equispaced", "tanhsinh", "PETSCDTNODES_", NULL};

static PetscBool GolubWelschCite       = PETSC_FALSE;
const char       GolubWelschCitation[] = "@article{GolubWelsch1969,\n"
                                         "  author  = {Golub and Welsch},\n"
                                         "  title   = {Calculation of Quadrature Rules},\n"
                                         "  journal = {Math. Comp.},\n"
                                         "  volume  = {23},\n"
                                         "  number  = {106},\n"
                                         "  pages   = {221--230},\n"
                                         "  year    = {1969}\n}\n";

/* Numerical tests in src/dm/dt/tests/ex1.c show that when computing the nodes and weights of Gauss-Jacobi
   quadrature rules:

   - in double precision, Newton's method and Golub & Welsch both work for moderate degrees (< 100),
   - in single precision, Newton's method starts producing incorrect roots around n = 15, but
     the weights from Golub & Welsch become a problem before then: they produces errors
     in computing the Jacobi-polynomial Gram matrix around n = 6.

   So we default to Newton's method (required fewer dependencies) */
PetscBool PetscDTGaussQuadratureNewton_Internal = PETSC_TRUE;

PetscClassId PETSCQUADRATURE_CLASSID = 0;

/*@
  PetscQuadratureCreate - Create a PetscQuadrature object

  Collective

  Input Parameter:
. comm - The communicator for the PetscQuadrature object

  Output Parameter:
. q  - The PetscQuadrature object

  Level: beginner

.seealso: PetscQuadratureDestroy(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscQuadratureCreate(MPI_Comm comm, PetscQuadrature *q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(q, 2);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*q,PETSCQUADRATURE_CLASSID,"PetscQuadrature","Quadrature","DT",comm,PetscQuadratureDestroy,PetscQuadratureView);CHKERRQ(ierr);
  (*q)->dim       = -1;
  (*q)->Nc        =  1;
  (*q)->order     = -1;
  (*q)->numPoints = 0;
  (*q)->points    = NULL;
  (*q)->weights   = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscQuadratureDuplicate - Create a deep copy of the PetscQuadrature object

  Collective on q

  Input Parameter:
. q  - The PetscQuadrature object

  Output Parameter:
. r  - The new PetscQuadrature object

  Level: beginner

.seealso: PetscQuadratureCreate(), PetscQuadratureDestroy(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature q, PetscQuadrature *r)
{
  PetscInt         order, dim, Nc, Nq;
  const PetscReal *points, *weights;
  PetscReal       *p, *w;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(q, 1);
  ierr = PetscQuadratureCreate(PetscObjectComm((PetscObject) q), r);CHKERRQ(ierr);
  ierr = PetscQuadratureGetOrder(q, &order);CHKERRQ(ierr);
  ierr = PetscQuadratureSetOrder(*r, order);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, &dim, &Nc, &Nq, &points, &weights);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nq*dim, &p);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nq*Nc, &w);CHKERRQ(ierr);
  ierr = PetscArraycpy(p, points, Nq*dim);CHKERRQ(ierr);
  ierr = PetscArraycpy(w, weights, Nc * Nq);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*r, dim, Nc, Nq, p, w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscQuadratureDestroy - Destroys a PetscQuadrature object

  Collective on q

  Input Parameter:
. q  - The PetscQuadrature object

  Level: beginner

.seealso: PetscQuadratureCreate(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*q) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*q),PETSCQUADRATURE_CLASSID,1);
  if (--((PetscObject)(*q))->refct > 0) {
    *q = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscFree((*q)->points);CHKERRQ(ierr);
  ierr = PetscFree((*q)->weights);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscQuadratureGetOrder - Return the order of the method

  Not collective

  Input Parameter:
. q - The PetscQuadrature object

  Output Parameter:
. order - The order of the quadrature, i.e. the highest degree polynomial that is exactly integrated

  Level: intermediate

.seealso: PetscQuadratureSetOrder(), PetscQuadratureGetData(), PetscQuadratureSetData()
@*/
PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature q, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  PetscValidPointer(order, 2);
  *order = q->order;
  PetscFunctionReturn(0);
}

/*@
  PetscQuadratureSetOrder - Return the order of the method

  Not collective

  Input Parameters:
+ q - The PetscQuadrature object
- order - The order of the quadrature, i.e. the highest degree polynomial that is exactly integrated

  Level: intermediate

.seealso: PetscQuadratureGetOrder(), PetscQuadratureGetData(), PetscQuadratureSetData()
@*/
PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature q, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  q->order = order;
  PetscFunctionReturn(0);
}

/*@
  PetscQuadratureGetNumComponents - Return the number of components for functions to be integrated

  Not collective

  Input Parameter:
. q - The PetscQuadrature object

  Output Parameter:
. Nc - The number of components

  Note: We are performing an integral int f(x) . w(x) dx, where both f and w (the weight) have Nc components.

  Level: intermediate

.seealso: PetscQuadratureSetNumComponents(), PetscQuadratureGetData(), PetscQuadratureSetData()
@*/
PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature q, PetscInt *Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  PetscValidPointer(Nc, 2);
  *Nc = q->Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscQuadratureSetNumComponents - Return the number of components for functions to be integrated

  Not collective

  Input Parameters:
+ q  - The PetscQuadrature object
- Nc - The number of components

  Note: We are performing an integral int f(x) . w(x) dx, where both f and w (the weight) have Nc components.

  Level: intermediate

.seealso: PetscQuadratureGetNumComponents(), PetscQuadratureGetData(), PetscQuadratureSetData()
@*/
PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature q, PetscInt Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  q->Nc = Nc;
  PetscFunctionReturn(0);
}

/*@C
  PetscQuadratureGetData - Returns the data defining the quadrature

  Not collective

  Input Parameter:
. q  - The PetscQuadrature object

  Output Parameters:
+ dim - The spatial dimension
. Nc - The number of components
. npoints - The number of quadrature points
. points - The coordinates of each quadrature point
- weights - The weight of each quadrature point

  Level: intermediate

  Fortran Notes:
    From Fortran you must call PetscQuadratureRestoreData() when you are done with the data

.seealso: PetscQuadratureCreate(), PetscQuadratureSetData()
@*/
PetscErrorCode PetscQuadratureGetData(PetscQuadrature q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, const PetscReal *points[], const PetscReal *weights[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  if (dim) {
    PetscValidPointer(dim, 2);
    *dim = q->dim;
  }
  if (Nc) {
    PetscValidPointer(Nc, 3);
    *Nc = q->Nc;
  }
  if (npoints) {
    PetscValidPointer(npoints, 4);
    *npoints = q->numPoints;
  }
  if (points) {
    PetscValidPointer(points, 5);
    *points = q->points;
  }
  if (weights) {
    PetscValidPointer(weights, 6);
    *weights = q->weights;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTJacobianInverse_Internal(PetscInt m, PetscInt n, const PetscReal J[], PetscReal Jinv[])
{
  PetscScalar    *Js, *Jinvs;
  PetscInt       i, j, k;
  PetscBLASInt   bm, bn, info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0);
  ierr = PetscBLASIntCast(m, &bm);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n, &bn);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc2(m*n, &Js, m*n, &Jinvs);CHKERRQ(ierr);
  for (i = 0; i < m*n; i++) Js[i] = J[i];
#else
  Js = (PetscReal *) J;
  Jinvs = Jinv;
#endif
  if (m == n) {
    PetscBLASInt *pivots;
    PetscScalar *W;

    ierr = PetscMalloc2(m, &pivots, m, &W);CHKERRQ(ierr);

    ierr = PetscArraycpy(Jinvs, Js, m * m);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&bm, &bm, Jinvs, &bm, pivots, &info));
    PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetrf %D",(PetscInt)info);
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&bm, Jinvs, &bm, pivots, W, &bm, &info));
    PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetri %D",(PetscInt)info);
    ierr = PetscFree2(pivots, W);CHKERRQ(ierr);
  } else if (m < n) {
    PetscScalar *JJT;
    PetscBLASInt *pivots;
    PetscScalar *W;

    ierr = PetscMalloc1(m*m, &JJT);CHKERRQ(ierr);
    ierr = PetscMalloc2(m, &pivots, m, &W);CHKERRQ(ierr);
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
        PetscScalar val = 0.;

        for (k = 0; k < n; k++) val += Js[i * n + k] * Js[j * n + k];
        JJT[i * m + j] = val;
      }
    }

    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&bm, &bm, JJT, &bm, pivots, &info));
    PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetrf %D",(PetscInt)info);
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&bm, JJT, &bm, pivots, W, &bm, &info));
    PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetri %D",(PetscInt)info);
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        PetscScalar val = 0.;

        for (k = 0; k < m; k++) val += Js[k * n + i] * JJT[k * m + j];
        Jinvs[i * m + j] = val;
      }
    }
    ierr = PetscFree2(pivots, W);CHKERRQ(ierr);
    ierr = PetscFree(JJT);CHKERRQ(ierr);
  } else {
    PetscScalar *JTJ;
    PetscBLASInt *pivots;
    PetscScalar *W;

    ierr = PetscMalloc1(n*n, &JTJ);CHKERRQ(ierr);
    ierr = PetscMalloc2(n, &pivots, n, &W);CHKERRQ(ierr);
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        PetscScalar val = 0.;

        for (k = 0; k < m; k++) val += Js[k * n + i] * Js[k * n + j];
        JTJ[i * n + j] = val;
      }
    }

    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&bn, &bn, JTJ, &bn, pivots, &info));
    PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetrf %D",(PetscInt)info);
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&bn, JTJ, &bn, pivots, W, &bn, &info));
    PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetri %D",(PetscInt)info);
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        PetscScalar val = 0.;

        for (k = 0; k < n; k++) val += JTJ[i * n + k] * Js[j * n + k];
        Jinvs[i * m + j] = val;
      }
    }
    ierr = PetscFree2(pivots, W);CHKERRQ(ierr);
    ierr = PetscFree(JTJ);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  for (i = 0; i < m*n; i++) Jinv[i] = PetscRealPart(Jinvs[i]);
  ierr = PetscFree2(Js, Jinvs);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@
   PetscQuadraturePushForward - Push forward a quadrature functional under an affine transformation.

   Collecive on PetscQuadrature

   Input Parameters:
+  q - the quadrature functional
.  imageDim - the dimension of the image of the transformation
.  origin - a point in the original space
.  originImage - the image of the origin under the transformation
.  J - the Jacobian of the image: an [imageDim x dim] matrix in row major order
-  formDegree - transform the quadrature weights as k-forms of this form degree (if the number of components is a multiple of (dim choose formDegree), it is assumed that they represent multiple k-forms) [see PetscDTAltVPullback() for interpretation of formDegree]

   Output Parameters:
.  Jinvstarq - a quadrature rule where each point is the image of a point in the original quadrature rule, and where the k-form weights have been pulled-back by the pseudoinverse of J to the k-form weights in the image space.

   Note: the new quadrature rule will have a different number of components if spaces have different dimensions.  For example, pushing a 2-form forward from a two dimensional space to a three dimensional space changes the number of components from 1 to 3.

   Level: intermediate

.seealso: PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscQuadraturePushForward(PetscQuadrature q, PetscInt imageDim, const PetscReal origin[], const PetscReal originImage[], const PetscReal J[], PetscInt formDegree, PetscQuadrature *Jinvstarq)
{
  PetscInt         dim, Nc, imageNc, formSize, Ncopies, imageFormSize, Npoints, pt, i, j, c;
  const PetscReal *points;
  const PetscReal *weights;
  PetscReal       *imagePoints, *imageWeights;
  PetscReal       *Jinv;
  PetscReal       *Jinvstar;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  PetscCheckFalse(imageDim < PetscAbsInt(formDegree),PetscObjectComm((PetscObject)q), PETSC_ERR_ARG_INCOMP, "Cannot represent a %D-form in %D dimensions", PetscAbsInt(formDegree), imageDim);
  ierr = PetscQuadratureGetData(q, &dim, &Nc, &Npoints, &points, &weights);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &formSize);CHKERRQ(ierr);
  PetscCheckFalse(Nc % formSize,PetscObjectComm((PetscObject)q), PETSC_ERR_ARG_INCOMP, "Number of components %D is not a multiple of formSize %D", Nc, formSize);
  Ncopies = Nc / formSize;
  ierr = PetscDTBinomialInt(imageDim, PetscAbsInt(formDegree), &imageFormSize);CHKERRQ(ierr);
  imageNc = Ncopies * imageFormSize;
  ierr = PetscMalloc1(Npoints * imageDim, &imagePoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(Npoints * imageNc, &imageWeights);CHKERRQ(ierr);
  ierr = PetscMalloc2(imageDim * dim, &Jinv, formSize * imageFormSize, &Jinvstar);CHKERRQ(ierr);
  ierr = PetscDTJacobianInverse_Internal(imageDim, dim, J, Jinv);CHKERRQ(ierr);
  ierr = PetscDTAltVPullbackMatrix(imageDim, dim, Jinv, formDegree, Jinvstar);CHKERRQ(ierr);
  for (pt = 0; pt < Npoints; pt++) {
    const PetscReal *point = &points[pt * dim];
    PetscReal       *imagePoint = &imagePoints[pt * imageDim];

    for (i = 0; i < imageDim; i++) {
      PetscReal val = originImage[i];

      for (j = 0; j < dim; j++) val += J[i * dim + j] * (point[j] - origin[j]);
      imagePoint[i] = val;
    }
    for (c = 0; c < Ncopies; c++) {
      const PetscReal *form = &weights[pt * Nc + c * formSize];
      PetscReal       *imageForm = &imageWeights[pt * imageNc + c * imageFormSize];

      for (i = 0; i < imageFormSize; i++) {
        PetscReal val = 0.;

        for (j = 0; j < formSize; j++) val += Jinvstar[i * formSize + j] * form[j];
        imageForm[i] = val;
      }
    }
  }
  ierr = PetscQuadratureCreate(PetscObjectComm((PetscObject)q), Jinvstarq);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*Jinvstarq, imageDim, imageNc, Npoints, imagePoints, imageWeights);CHKERRQ(ierr);
  ierr = PetscFree2(Jinv, Jinvstar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscQuadratureSetData - Sets the data defining the quadrature

  Not collective

  Input Parameters:
+ q  - The PetscQuadrature object
. dim - The spatial dimension
. Nc - The number of components
. npoints - The number of quadrature points
. points - The coordinates of each quadrature point
- weights - The weight of each quadrature point

  Note: This routine owns the references to points and weights, so they must be allocated using PetscMalloc() and the user should not free them.

  Level: intermediate

.seealso: PetscQuadratureCreate(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscQuadratureSetData(PetscQuadrature q, PetscInt dim, PetscInt Nc, PetscInt npoints, const PetscReal points[], const PetscReal weights[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  if (dim >= 0)     q->dim       = dim;
  if (Nc >= 0)      q->Nc        = Nc;
  if (npoints >= 0) q->numPoints = npoints;
  if (points) {
    PetscValidPointer(points, 5);
    q->points = points;
  }
  if (weights) {
    PetscValidPointer(weights, 6);
    q->weights = weights;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscQuadratureView_Ascii(PetscQuadrature quad, PetscViewer v)
{
  PetscInt          q, d, c;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (quad->Nc > 1) {ierr = PetscViewerASCIIPrintf(v, "Quadrature of order %D on %D points (dim %D) with %D components\n", quad->order, quad->numPoints, quad->dim, quad->Nc);CHKERRQ(ierr);}
  else              {ierr = PetscViewerASCIIPrintf(v, "Quadrature of order %D on %D points (dim %D)\n", quad->order, quad->numPoints, quad->dim);CHKERRQ(ierr);}
  ierr = PetscViewerGetFormat(v, &format);CHKERRQ(ierr);
  if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  for (q = 0; q < quad->numPoints; ++q) {
    ierr = PetscViewerASCIIPrintf(v, "p%D (", q);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(v, PETSC_FALSE);CHKERRQ(ierr);
    for (d = 0; d < quad->dim; ++d) {
      if (d) {ierr = PetscViewerASCIIPrintf(v, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(v, "%+g", (double)quad->points[q*quad->dim+d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(v, ") ");CHKERRQ(ierr);
    if (quad->Nc > 1) {ierr = PetscViewerASCIIPrintf(v, "w%D (", q);CHKERRQ(ierr);}
    for (c = 0; c < quad->Nc; ++c) {
      if (c) {ierr = PetscViewerASCIIPrintf(v, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(v, "%+g", (double)quad->weights[q*quad->Nc+c]);CHKERRQ(ierr);
    }
    if (quad->Nc > 1) {ierr = PetscViewerASCIIPrintf(v, ")");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(v, PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscQuadratureView - Views a PetscQuadrature object

  Collective on quad

  Input Parameters:
+ quad  - The PetscQuadrature object
- viewer - The PetscViewer object

  Level: beginner

.seealso: PetscQuadratureCreate(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscQuadratureView(PetscQuadrature quad, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(quad, 1);
  if (viewer) PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) quad), &viewer);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (iascii) {ierr = PetscQuadratureView_Ascii(quad, viewer);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscQuadratureExpandComposite - Return a quadrature over the composite element, which has the original quadrature in each subelement

  Not collective

  Input Parameters:
+ q - The original PetscQuadrature
. numSubelements - The number of subelements the original element is divided into
. v0 - An array of the initial points for each subelement
- jac - An array of the Jacobian mappings from the reference to each subelement

  Output Parameters:
. dim - The dimension

  Note: Together v0 and jac define an affine mapping from the original reference element to each subelement

 Not available from Fortran

  Level: intermediate

.seealso: PetscFECreate(), PetscSpaceGetDimension(), PetscDualSpaceGetDimension()
@*/
PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature q, PetscInt numSubelements, const PetscReal v0[], const PetscReal jac[], PetscQuadrature *qref)
{
  const PetscReal *points,    *weights;
  PetscReal       *pointsRef, *weightsRef;
  PetscInt         dim, Nc, order, npoints, npointsRef, c, p, cp, d, e;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 1);
  PetscValidPointer(v0, 3);
  PetscValidPointer(jac, 4);
  PetscValidPointer(qref, 5);
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, qref);CHKERRQ(ierr);
  ierr = PetscQuadratureGetOrder(q, &order);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, &weights);CHKERRQ(ierr);
  npointsRef = npoints*numSubelements;
  ierr = PetscMalloc1(npointsRef*dim,&pointsRef);CHKERRQ(ierr);
  ierr = PetscMalloc1(npointsRef*Nc, &weightsRef);CHKERRQ(ierr);
  for (c = 0; c < numSubelements; ++c) {
    for (p = 0; p < npoints; ++p) {
      for (d = 0; d < dim; ++d) {
        pointsRef[(c*npoints + p)*dim+d] = v0[c*dim+d];
        for (e = 0; e < dim; ++e) {
          pointsRef[(c*npoints + p)*dim+d] += jac[(c*dim + d)*dim+e]*(points[p*dim+e] + 1.0);
        }
      }
      /* Could also use detJ here */
      for (cp = 0; cp < Nc; ++cp) weightsRef[(c*npoints+p)*Nc+cp] = weights[p*Nc+cp]/numSubelements;
    }
  }
  ierr = PetscQuadratureSetOrder(*qref, order);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*qref, dim, Nc, npointsRef, pointsRef, weightsRef);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Compute the coefficients for the Jacobi polynomial recurrence,
 *
 * J^{a,b}_n(x) = (cnm1 + cnm1x * x) * J^{a,b}_{n-1}(x) - cnm2 * J^{a,b}_{n-2}(x).
 */
#define PetscDTJacobiRecurrence_Internal(n,a,b,cnm1,cnm1x,cnm2) \
do {                                                            \
  PetscReal _a = (a);                                           \
  PetscReal _b = (b);                                           \
  PetscReal _n = (n);                                           \
  if (n == 1) {                                                 \
    (cnm1) = (_a-_b) * 0.5;                                     \
    (cnm1x) = (_a+_b+2.)*0.5;                                   \
    (cnm2) = 0.;                                                \
  } else {                                                      \
    PetscReal _2n = _n+_n;                                      \
    PetscReal _d = (_2n*(_n+_a+_b)*(_2n+_a+_b-2));              \
    PetscReal _n1 = (_2n+_a+_b-1.)*(_a*_a-_b*_b);               \
    PetscReal _n1x = (_2n+_a+_b-1.)*(_2n+_a+_b)*(_2n+_a+_b-2);  \
    PetscReal _n2 = 2.*((_n+_a-1.)*(_n+_b-1.)*(_2n+_a+_b));     \
    (cnm1) = _n1 / _d;                                          \
    (cnm1x) = _n1x / _d;                                        \
    (cnm2) = _n2 / _d;                                          \
  }                                                             \
} while (0)

/*@
  PetscDTJacobiNorm - Compute the weighted L2 norm of a Jacobi polynomial.

  $\| P^{\alpha,\beta}_n \|_{\alpha,\beta}^2 = \int_{-1}^1 (1 + x)^{\alpha} (1 - x)^{\beta} P^{\alpha,\beta}_n (x)^2 dx.$

  Input Parameters:
- alpha - the left exponent > -1
. beta - the right exponent > -1
+ n - the polynomial degree

  Output Parameter:
. norm - the weighted L2 norm

  Level: beginner

.seealso: PetscDTJacobiEval()
@*/
PetscErrorCode PetscDTJacobiNorm(PetscReal alpha, PetscReal beta, PetscInt n, PetscReal *norm)
{
  PetscReal twoab1;
  PetscReal gr;

  PetscFunctionBegin;
  PetscCheckFalse(alpha <= -1.,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exponent alpha %g <= -1. invalid", (double) alpha);
  PetscCheckFalse(beta <= -1.,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exponent beta %g <= -1. invalid", (double) beta);
  PetscCheckFalse(n < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n %D < 0 invalid", n);
  twoab1 = PetscPowReal(2., alpha + beta + 1.);
#if defined(PETSC_HAVE_LGAMMA)
  if (!n) {
    gr = PetscExpReal(PetscLGamma(alpha+1.) + PetscLGamma(beta+1.) - PetscLGamma(alpha+beta+2.));
  } else {
    gr = PetscExpReal(PetscLGamma(n+alpha+1.) + PetscLGamma(n+beta+1.) - (PetscLGamma(n+1.) + PetscLGamma(n+alpha+beta+1.))) / (n+n+alpha+beta+1.);
  }
#else
  {
    PetscInt alphai = (PetscInt) alpha;
    PetscInt betai = (PetscInt) beta;
    PetscInt i;

    gr = n ? (1. / (n+n+alpha+beta+1.)) : 1.;
    if ((PetscReal) alphai == alpha) {
      if (!n) {
        for (i = 0; i < alphai; i++) gr *= (i+1.) / (beta+i+1.);
        gr /= (alpha+beta+1.);
      } else {
        for (i = 0; i < alphai; i++) gr *= (n+i+1.) / (n+beta+i+1.);
      }
    } else if ((PetscReal) betai == beta) {
      if (!n) {
        for (i = 0; i < betai; i++) gr *= (i+1.) / (alpha+i+2.);
        gr /= (alpha+beta+1.);
      } else {
        for (i = 0; i < betai; i++) gr *= (n+i+1.) / (n+alpha+i+1.);
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"lgamma() - math routine is unavailable.");
  }
#endif
  *norm = PetscSqrtReal(twoab1 * gr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTJacobiEval_Internal(PetscInt npoints, PetscReal a, PetscReal b, PetscInt k, const PetscReal *points, PetscInt ndegree, const PetscInt *degrees, PetscReal *p)
{
  PetscReal ak, bk;
  PetscReal abk1;
  PetscInt i,l,maxdegree;

  PetscFunctionBegin;
  maxdegree = degrees[ndegree-1] - k;
  ak = a + k;
  bk = b + k;
  abk1 = a + b + k + 1.;
  if (maxdegree < 0) {
    for (i = 0; i < npoints; i++) for (l = 0; l < ndegree; l++) p[i*ndegree+l] = 0.;
    PetscFunctionReturn(0);
  }
  for (i=0; i<npoints; i++) {
    PetscReal pm1,pm2,x;
    PetscReal cnm1, cnm1x, cnm2;
    PetscInt  j,m;

    x    = points[i];
    pm2  = 1.;
    PetscDTJacobiRecurrence_Internal(1,ak,bk,cnm1,cnm1x,cnm2);
    pm1 = (cnm1 + cnm1x*x);
    l    = 0;
    while (l < ndegree && degrees[l] - k < 0) {
      p[l++] = 0.;
    }
    while (l < ndegree && degrees[l] - k == 0) {
      p[l] = pm2;
      for (m = 0; m < k; m++) p[l] *= (abk1 + m) * 0.5;
      l++;
    }
    while (l < ndegree && degrees[l] - k == 1) {
      p[l] = pm1;
      for (m = 0; m < k; m++) p[l] *= (abk1 + 1 + m) * 0.5;
      l++;
    }
    for (j=2; j<=maxdegree; j++) {
      PetscReal pp;

      PetscDTJacobiRecurrence_Internal(j,ak,bk,cnm1,cnm1x,cnm2);
      pp   = (cnm1 + cnm1x*x)*pm1 - cnm2*pm2;
      pm2  = pm1;
      pm1  = pp;
      while (l < ndegree && degrees[l] - k == j) {
        p[l] = pp;
        for (m = 0; m < k; m++) p[l] *= (abk1 + j + m) * 0.5;
        l++;
      }
    }
    p += ndegree;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDTJacobiEvalJet - Evaluate the jet (function and derivatives) of the Jacobi polynomials basis up to a given degree.  The Jacobi polynomials with indices $\alpha$ and $\beta$ are orthogonal with respect to the weighted inner product $\langle f, g \rangle = \int_{-1}^1 (1+x)^{\alpha} (1-x)^{\beta) f(x) g(x) dx$.

  Input Parameters:
+ alpha - the left exponent of the weight
. beta - the right exponetn of the weight
. npoints - the number of points to evaluate the polynomials at
. points - [npoints] array of point coordinates
. degree - the maximm degree polynomial space to evaluate, (degree + 1) will be evaluated total.
- k - the maximum derivative to evaluate in the jet, (k + 1) will be evaluated total.

  Output Argments:
- p - an array containing the evaluations of the Jacobi polynomials's jets on the points.  the size is (degree + 1) x
  (k + 1) x npoints, which also describes the order of the dimensions of this three-dimensional array: the first
  (slowest varying) dimension is polynomial degree; the second dimension is derivative order; the third (fastest
  varying) dimension is the index of the evaluation point.

  Level: advanced

.seealso: PetscDTJacobiEval(), PetscDTPKDEvalJet()
@*/
PetscErrorCode PetscDTJacobiEvalJet(PetscReal alpha, PetscReal beta, PetscInt npoints, const PetscReal points[], PetscInt degree, PetscInt k, PetscReal p[])
{
  PetscInt        i, j, l;
  PetscInt       *degrees;
  PetscReal      *psingle;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (degree == 0) {
    PetscInt zero = 0;

    for (i = 0; i <= k; i++) {
      ierr = PetscDTJacobiEval_Internal(npoints, alpha, beta, i, points, 1, &zero, &p[i*npoints]);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(degree + 1, &degrees);CHKERRQ(ierr);
  ierr = PetscMalloc1((degree + 1) * npoints, &psingle);CHKERRQ(ierr);
  for (i = 0; i <= degree; i++) degrees[i] = i;
  for (i = 0; i <= k; i++) {
    ierr = PetscDTJacobiEval_Internal(npoints, alpha, beta, i, points, degree + 1, degrees, psingle);CHKERRQ(ierr);
    for (j = 0; j <= degree; j++) {
      for (l = 0; l < npoints; l++) {
        p[(j * (k + 1) + i) * npoints + l] = psingle[l * (degree + 1) + j];
      }
    }
  }
  ierr = PetscFree(psingle);CHKERRQ(ierr);
  ierr = PetscFree(degrees);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDTJacobiEval - evaluate Jacobi polynomials for the weight function $(1.+x)^{\alpha} (1.-x)^{\beta}$
                       at points

   Not Collective

   Input Parameters:
+  npoints - number of spatial points to evaluate at
.  alpha - the left exponent > -1
.  beta - the right exponent > -1
.  points - array of locations to evaluate at
.  ndegree - number of basis degrees to evaluate
-  degrees - sorted array of degrees to evaluate

   Output Parameters:
+  B - row-oriented basis evaluation matrix B[point*ndegree + degree] (dimension npoints*ndegrees, allocated by caller) (or NULL)
.  D - row-oriented derivative evaluation matrix (or NULL)
-  D2 - row-oriented second derivative evaluation matrix (or NULL)

   Level: intermediate

.seealso: PetscDTGaussQuadrature()
@*/
PetscErrorCode PetscDTJacobiEval(PetscInt npoints,PetscReal alpha, PetscReal beta, const PetscReal *points,PetscInt ndegree,const PetscInt *degrees,PetscReal *B,PetscReal *D,PetscReal *D2)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(alpha <= -1.,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"alpha must be > -1.");
  PetscCheckFalse(beta <= -1.,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"beta must be > -1.");
  if (!npoints || !ndegree) PetscFunctionReturn(0);
  if (B)  {ierr = PetscDTJacobiEval_Internal(npoints, alpha, beta, 0, points, ndegree, degrees, B);CHKERRQ(ierr);}
  if (D)  {ierr = PetscDTJacobiEval_Internal(npoints, alpha, beta, 1, points, ndegree, degrees, D);CHKERRQ(ierr);}
  if (D2) {ierr = PetscDTJacobiEval_Internal(npoints, alpha, beta, 2, points, ndegree, degrees, D2);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
   PetscDTLegendreEval - evaluate Legendre polynomials at points

   Not Collective

   Input Parameters:
+  npoints - number of spatial points to evaluate at
.  points - array of locations to evaluate at
.  ndegree - number of basis degrees to evaluate
-  degrees - sorted array of degrees to evaluate

   Output Parameters:
+  B - row-oriented basis evaluation matrix B[point*ndegree + degree] (dimension npoints*ndegrees, allocated by caller) (or NULL)
.  D - row-oriented derivative evaluation matrix (or NULL)
-  D2 - row-oriented second derivative evaluation matrix (or NULL)

   Level: intermediate

.seealso: PetscDTGaussQuadrature()
@*/
PetscErrorCode PetscDTLegendreEval(PetscInt npoints,const PetscReal *points,PetscInt ndegree,const PetscInt *degrees,PetscReal *B,PetscReal *D,PetscReal *D2)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTJacobiEval(npoints, 0., 0., points, ndegree, degrees, B, D, D2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTIndexToGradedOrder - convert an index into a tuple of monomial degrees in a graded order (that is, if the degree sum of tuple x is less than the degree sum of tuple y, then the index of x is smaller than the index of y)

  Input Parameters:
+ len - the desired length of the degree tuple
- index - the index to convert: should be >= 0

  Output Parameter:
. degtup - will be filled with a tuple of degrees

  Level: beginner

  Note: for two tuples x and y with the same degree sum, partial degree sums over the final elements of the tuples
  acts as a tiebreaker.  For example, (2, 1, 1) and (1, 2, 1) have the same degree sum, but the degree sum over the
  last two elements is smaller for the former, so (2, 1, 1) < (1, 2, 1).

.seealso: PetscDTGradedOrderToIndex()
@*/
PetscErrorCode PetscDTIndexToGradedOrder(PetscInt len, PetscInt index, PetscInt degtup[])
{
  PetscInt i, total;
  PetscInt sum;

  PetscFunctionBeginHot;
  PetscCheckFalse(len < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "length must be non-negative");
  PetscCheckFalse(index < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index must be non-negative");
  total = 1;
  sum = 0;
  while (index >= total) {
    index -= total;
    total = (total * (len + sum)) / (sum + 1);
    sum++;
  }
  for (i = 0; i < len; i++) {
    PetscInt c;

    degtup[i] = sum;
    for (c = 0, total = 1; c < sum; c++) {
      /* going into the loop, total is the number of way to have a tuple of sum exactly c with length len - 1 - i */
      if (index < total) break;
      index -= total;
      total = (total * (len - 1 - i + c)) / (c + 1);
      degtup[i]--;
    }
    sum -= degtup[i];
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDTGradedOrderToIndex - convert a tuple into an index in a graded order, the inverse of PetscDTIndexToGradedOrder().

  Input Parameters:
+ len - the length of the degree tuple
- degtup - tuple with this length

  Output Parameter:
. index - index in graded order: >= 0

  Level: Beginner

  Note: for two tuples x and y with the same degree sum, partial degree sums over the final elements of the tuples
  acts as a tiebreaker.  For example, (2, 1, 1) and (1, 2, 1) have the same degree sum, but the degree sum over the
  last two elements is smaller for the former, so (2, 1, 1) < (1, 2, 1).

.seealso: PetscDTIndexToGradedOrder()
@*/
PetscErrorCode PetscDTGradedOrderToIndex(PetscInt len, const PetscInt degtup[], PetscInt *index)
{
  PetscInt i, idx, sum, total;

  PetscFunctionBeginHot;
  PetscCheckFalse(len < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "length must be non-negative");
  for (i = 0, sum = 0; i < len; i++) sum += degtup[i];
  idx = 0;
  total = 1;
  for (i = 0; i < sum; i++) {
    idx += total;
    total = (total * (len + i)) / (i + 1);
  }
  for (i = 0; i < len - 1; i++) {
    PetscInt c;

    total = 1;
    sum -= degtup[i];
    for (c = 0; c < sum; c++) {
      idx += total;
      total = (total * (len - 1 - i + c)) / (c + 1);
    }
  }
  *index = idx;
  PetscFunctionReturn(0);
}

static PetscBool PKDCite = PETSC_FALSE;
const char       PKDCitation[] = "@article{Kirby2010,\n"
                                 "  title={Singularity-free evaluation of collapsed-coordinate orthogonal polynomials},\n"
                                 "  author={Kirby, Robert C},\n"
                                 "  journal={ACM Transactions on Mathematical Software (TOMS)},\n"
                                 "  volume={37},\n"
                                 "  number={1},\n"
                                 "  pages={1--16},\n"
                                 "  year={2010},\n"
                                 "  publisher={ACM New York, NY, USA}\n}\n";

/*@
  PetscDTPKDEvalJet - Evaluate the jet (function and derivatives) of the Proriol-Koornwinder-Dubiner (PKD) basis for
  the space of polynomials up to a given degree.  The PKD basis is L2-orthonormal on the biunit simplex (which is used
  as the reference element for finite elements in PETSc), which makes it a stable basis to use for evaluating
  polynomials in that domain.

  Input Parameters:
+ dim - the number of variables in the multivariate polynomials
. npoints - the number of points to evaluate the polynomials at
. points - [npoints x dim] array of point coordinates
. degree - the degree (sum of degrees on the variables in a monomial) of the polynomial space to evaluate.  There are ((dim + degree) choose dim) polynomials in this space.
- k - the maximum order partial derivative to evaluate in the jet.  There are (dim + k choose dim) partial derivatives
  in the jet.  Choosing k = 0 means to evaluate just the function and no derivatives

  Output Argments:
- p - an array containing the evaluations of the PKD polynomials' jets on the points.  The size is ((dim + degree)
  choose dim) x ((dim + k) choose dim) x npoints, which also describes the order of the dimensions of this
  three-dimensional array: the first (slowest varying) dimension is basis function index; the second dimension is jet
  index; the third (fastest varying) dimension is the index of the evaluation point.

  Level: advanced

  Note: The ordering of the basis functions, and the ordering of the derivatives in the jet, both follow the graded
  ordering of PetscDTIndexToGradedOrder() and PetscDTGradedOrderToIndex().  For example, in 3D, the polynomial with
  leading monomial x^2,y^0,z^1, which has degree tuple (2,0,1), which by PetscDTGradedOrderToIndex() has index 12 (it is the 13th basis function in the space);
  the partial derivative $\partial_x \partial_z$ has order tuple (1,0,1), appears at index 6 in the jet (it is the 7th partial derivative in the jet).

  The implementation uses Kirby's singularity-free evaluation algorithm, https://doi.org/10.1145/1644001.1644006.

.seealso: PetscDTGradedOrderToIndex(), PetscDTIndexToGradedOrder(), PetscDTJacobiEvalJet()
@*/
PetscErrorCode PetscDTPKDEvalJet(PetscInt dim, PetscInt npoints, const PetscReal points[], PetscInt degree, PetscInt k, PetscReal p[])
{
  PetscInt        degidx, kidx, d, pt;
  PetscInt        Nk, Ndeg;
  PetscInt       *ktup, *degtup;
  PetscReal      *scales, initscale, scaleexp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(PKDCitation, &PKDCite);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + k, k, &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(degree + dim, degree, &Ndeg);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim, &degtup, dim, &ktup);CHKERRQ(ierr);
  ierr = PetscMalloc1(Ndeg, &scales);CHKERRQ(ierr);
  initscale = 1.;
  if (dim > 1) {
    ierr = PetscDTBinomial(dim,2,&scaleexp);CHKERRQ(ierr);
    initscale = PetscPowReal(2.,scaleexp*0.5);
  }
  for (degidx = 0; degidx < Ndeg; degidx++) {
    PetscInt e, i;
    PetscInt m1idx = -1, m2idx = -1;
    PetscInt n;
    PetscInt degsum;
    PetscReal alpha;
    PetscReal cnm1, cnm1x, cnm2;
    PetscReal norm;

    ierr = PetscDTIndexToGradedOrder(dim, degidx, degtup);CHKERRQ(ierr);
    for (d = dim - 1; d >= 0; d--) if (degtup[d]) break;
    if (d < 0) { /* constant is 1 everywhere, all derivatives are zero */
      scales[degidx] = initscale;
      for (e = 0; e < dim; e++) {
        ierr = PetscDTJacobiNorm(e,0.,0,&norm);CHKERRQ(ierr);
        scales[degidx] /= norm;
      }
      for (i = 0; i < npoints; i++) p[degidx * Nk * npoints + i] = 1.;
      for (i = 0; i < (Nk - 1) * npoints; i++) p[(degidx * Nk + 1) * npoints + i] = 0.;
      continue;
    }
    n = degtup[d];
    degtup[d]--;
    ierr = PetscDTGradedOrderToIndex(dim, degtup, &m1idx);CHKERRQ(ierr);
    if (degtup[d] > 0) {
      degtup[d]--;
      ierr = PetscDTGradedOrderToIndex(dim, degtup, &m2idx);CHKERRQ(ierr);
      degtup[d]++;
    }
    degtup[d]++;
    for (e = 0, degsum = 0; e < d; e++) degsum += degtup[e];
    alpha = 2 * degsum + d;
    PetscDTJacobiRecurrence_Internal(n,alpha,0.,cnm1,cnm1x,cnm2);

    scales[degidx] = initscale;
    for (e = 0, degsum = 0; e < dim; e++) {
      PetscInt  f;
      PetscReal ealpha;
      PetscReal enorm;

      ealpha = 2 * degsum + e;
      for (f = 0; f < degsum; f++) scales[degidx] *= 2.;
      ierr = PetscDTJacobiNorm(ealpha,0.,degtup[e],&enorm);CHKERRQ(ierr);
      scales[degidx] /= enorm;
      degsum += degtup[e];
    }

    for (pt = 0; pt < npoints; pt++) {
      /* compute the multipliers */
      PetscReal thetanm1, thetanm1x, thetanm2;

      thetanm1x = dim - (d+1) + 2.*points[pt * dim + d];
      for (e = d+1; e < dim; e++) thetanm1x += points[pt * dim + e];
      thetanm1x *= 0.5;
      thetanm1 = (2. - (dim-(d+1)));
      for (e = d+1; e < dim; e++) thetanm1 -= points[pt * dim + e];
      thetanm1 *= 0.5;
      thetanm2 = thetanm1 * thetanm1;

      for (kidx = 0; kidx < Nk; kidx++) {
        PetscInt f;

        ierr = PetscDTIndexToGradedOrder(dim, kidx, ktup);CHKERRQ(ierr);
        /* first sum in the same derivative terms */
        p[(degidx * Nk + kidx) * npoints + pt] = (cnm1 * thetanm1 + cnm1x * thetanm1x) * p[(m1idx * Nk + kidx) * npoints + pt];
        if (m2idx >= 0) {
          p[(degidx * Nk + kidx) * npoints + pt] -= cnm2 * thetanm2 * p[(m2idx * Nk + kidx) * npoints + pt];
        }

        for (f = d; f < dim; f++) {
          PetscInt km1idx, mplty = ktup[f];

          if (!mplty) continue;
          ktup[f]--;
          ierr = PetscDTGradedOrderToIndex(dim, ktup, &km1idx);CHKERRQ(ierr);

          /* the derivative of  cnm1x * thetanm1x  wrt x variable f is 0.5 * cnm1x if f > d otherwise it is cnm1x */
          /* the derivative of  cnm1  * thetanm1   wrt x variable f is 0 if f == d, otherwise it is -0.5 * cnm1 */
          /* the derivative of -cnm2  * thetanm2   wrt x variable f is 0 if f == d, otherwise it is cnm2 * thetanm1 */
          if (f > d) {
            PetscInt f2;

            p[(degidx * Nk + kidx) * npoints + pt] += mplty * 0.5 * (cnm1x - cnm1) * p[(m1idx * Nk + km1idx) * npoints + pt];
            if (m2idx >= 0) {
              p[(degidx * Nk + kidx) * npoints + pt] += mplty * cnm2 * thetanm1 * p[(m2idx * Nk + km1idx) * npoints + pt];
              /* second derivatives of -cnm2  * thetanm2   wrt x variable f,f2 is like - 0.5 * cnm2 */
              for (f2 = f; f2 < dim; f2++) {
                PetscInt km2idx, mplty2 = ktup[f2];
                PetscInt factor;

                if (!mplty2) continue;
                ktup[f2]--;
                ierr = PetscDTGradedOrderToIndex(dim, ktup, &km2idx);CHKERRQ(ierr);

                factor = mplty * mplty2;
                if (f == f2) factor /= 2;
                p[(degidx * Nk + kidx) * npoints + pt] -= 0.5 * factor * cnm2 * p[(m2idx * Nk + km2idx) * npoints + pt];
                ktup[f2]++;
              }
            }
          } else {
            p[(degidx * Nk + kidx) * npoints + pt] += mplty * cnm1x * p[(m1idx * Nk + km1idx) * npoints + pt];
          }
          ktup[f]++;
        }
      }
    }
  }
  for (degidx = 0; degidx < Ndeg; degidx++) {
    PetscReal scale = scales[degidx];
    PetscInt i;

    for (i = 0; i < Nk * npoints; i++) p[degidx*Nk*npoints + i] *= scale;
  }
  ierr = PetscFree(scales);CHKERRQ(ierr);
  ierr = PetscFree2(degtup, ktup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTPTrimmedSize - The size of the trimmed polynomial space of k-forms with a given degree and form degree,
  which can be evaluated in PetscDTPTrimmedEvalJet().

  Input Parameters:
+ dim - the number of variables in the multivariate polynomials
. degree - the degree (sum of degrees on the variables in a monomial) of the trimmed polynomial space.
- formDegree - the degree of the form

  Output Argments:
- size - The number ((dim + degree) choose (dim + formDegree)) x ((degree + formDegree - 1) choose (formDegree))

  Level: advanced

.seealso: PetscDTPTrimmedEvalJet()
@*/
PetscErrorCode PetscDTPTrimmedSize(PetscInt dim, PetscInt degree, PetscInt formDegree, PetscInt *size)
{
  PetscInt       Nrk, Nbpt; // number of trimmed polynomials
  PetscErrorCode ierr;

  PetscFunctionBegin;
  formDegree = PetscAbsInt(formDegree);
  ierr = PetscDTBinomialInt(degree + dim, degree + formDegree, &Nbpt);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(degree + formDegree - 1, formDegree, &Nrk);CHKERRQ(ierr);
  Nbpt *= Nrk;
  *size = Nbpt;
  PetscFunctionReturn(0);
}

/* there was a reference implementation based on section 4.4 of Arnold, Falk & Winther (acta numerica, 2006), but it
 * was inferior to this implementation */
static PetscErrorCode PetscDTPTrimmedEvalJet_Internal(PetscInt dim, PetscInt npoints, const PetscReal points[], PetscInt degree, PetscInt formDegree, PetscInt jetDegree, PetscReal p[])
{
  PetscInt       formDegreeOrig = formDegree;
  PetscBool      formNegative = (formDegreeOrig < 0) ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  formDegree = PetscAbsInt(formDegreeOrig);
  if (formDegree == 0) {
    ierr = PetscDTPKDEvalJet(dim, npoints, points, degree, jetDegree, p);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (formDegree == dim) {
    ierr = PetscDTPKDEvalJet(dim, npoints, points, degree - 1, jetDegree, p);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscInt Nbpt;
  ierr = PetscDTPTrimmedSize(dim, degree, formDegree, &Nbpt);CHKERRQ(ierr);
  PetscInt Nf;
  ierr = PetscDTBinomialInt(dim, formDegree, &Nf);CHKERRQ(ierr);
  PetscInt Nk;
  ierr = PetscDTBinomialInt(dim + jetDegree, dim, &Nk);CHKERRQ(ierr);
  ierr = PetscArrayzero(p, Nbpt * Nf * Nk * npoints);CHKERRQ(ierr);

  PetscInt Nbpm1; // number of scalar polynomials up to degree - 1;
  ierr = PetscDTBinomialInt(dim + degree - 1, dim, &Nbpm1);CHKERRQ(ierr);
  PetscReal *p_scalar;
  ierr = PetscMalloc1(Nbpm1 * Nk * npoints, &p_scalar);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, npoints, points, degree - 1, jetDegree, p_scalar);CHKERRQ(ierr);
  PetscInt total = 0;
  // First add the full polynomials up to degree - 1 into the basis: take the scalar
  // and copy one for each form component
  for (PetscInt i = 0; i < Nbpm1; i++) {
    const PetscReal *src = &p_scalar[i * Nk * npoints];
    for (PetscInt f = 0; f < Nf; f++) {
      PetscReal *dest = &p[(total++ * Nf + f) * Nk * npoints];
      ierr = PetscArraycpy(dest, src, Nk * npoints);CHKERRQ(ierr);
    }
  }
  PetscInt *form_atoms;
  ierr = PetscMalloc1(formDegree + 1, &form_atoms);CHKERRQ(ierr);
  // construct the interior product pattern
  PetscInt (*pattern)[3];
  PetscInt Nf1; // number of formDegree + 1 forms
  ierr = PetscDTBinomialInt(dim, formDegree + 1, &Nf1);CHKERRQ(ierr);
  PetscInt nnz = Nf1 * (formDegree+1);
  ierr = PetscMalloc1(Nf1 * (formDegree+1), &pattern);CHKERRQ(ierr);
  ierr = PetscDTAltVInteriorPattern(dim, formDegree+1, pattern);CHKERRQ(ierr);
  PetscReal centroid = (1. - dim) / (dim + 1.);
  PetscInt *deriv;
  ierr = PetscMalloc1(dim, &deriv);CHKERRQ(ierr);
  for (PetscInt d = dim; d >= formDegree + 1; d--) {
    PetscInt Nfd1; // number of formDegree + 1 forms in dimension d that include dx_0
                   // (equal to the number of formDegree forms in dimension d-1)
    ierr = PetscDTBinomialInt(d - 1, formDegree, &Nfd1);CHKERRQ(ierr);
    // The number of homogeneous (degree-1) scalar polynomials in d variables
    PetscInt Nh;
    ierr = PetscDTBinomialInt(d - 1 + degree - 1, d - 1, &Nh);CHKERRQ(ierr);
    const PetscReal *h_scalar = &p_scalar[(Nbpm1 - Nh) * Nk * npoints];
    for (PetscInt b = 0; b < Nh; b++) {
      const PetscReal *h_s = &h_scalar[b * Nk * npoints];
      for (PetscInt f = 0; f < Nfd1; f++) {
        // construct all formDegree+1 forms that start with dx_(dim - d) /\ ...
        form_atoms[0] = dim - d;
        ierr = PetscDTEnumSubset(d-1, formDegree, f, &form_atoms[1]);CHKERRQ(ierr);
        for (PetscInt i = 0; i < formDegree; i++) {
          form_atoms[1+i] += form_atoms[0] + 1;
        }
        PetscInt f_ind; // index of the resulting form
        ierr = PetscDTSubsetIndex(dim, formDegree + 1, form_atoms, &f_ind);CHKERRQ(ierr);
        PetscReal *p_f = &p[total++ * Nf * Nk * npoints];
        for (PetscInt nz = 0; nz < nnz; nz++) {
          PetscInt i = pattern[nz][0]; // formDegree component
          PetscInt j = pattern[nz][1]; // (formDegree + 1) component
          PetscInt v = pattern[nz][2]; // coordinate component
          PetscReal scale = v < 0 ? -1. : 1.;

          i = formNegative ? (Nf - 1 - i) : i;
          scale = (formNegative && (i & 1)) ? -scale : scale;
          v = v < 0 ? -(v + 1) : v;
          if (j != f_ind) {
            continue;
          }
          PetscReal *p_i = &p_f[i * Nk * npoints];
          for (PetscInt jet = 0; jet < Nk; jet++) {
            const PetscReal *h_jet = &h_s[jet * npoints];
            PetscReal *p_jet = &p_i[jet * npoints];

            for (PetscInt pt = 0; pt < npoints; pt++) {
              p_jet[pt] += scale * h_jet[pt] * (points[pt * dim + v] - centroid);
            }
            ierr = PetscDTIndexToGradedOrder(dim, jet, deriv);CHKERRQ(ierr);
            deriv[v]++;
            PetscReal mult = deriv[v];
            PetscInt l;
            ierr = PetscDTGradedOrderToIndex(dim, deriv, &l);CHKERRQ(ierr);
            if (l >= Nk) {
              continue;
            }
            p_jet = &p_i[l * npoints];
            for (PetscInt pt = 0; pt < npoints; pt++) {
              p_jet[pt] += scale * mult * h_jet[pt];
            }
            deriv[v]--;
          }
        }
      }
    }
  }
  PetscCheckFalse(total != Nbpt,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrectly counted P trimmed polynomials");
  ierr = PetscFree(deriv);CHKERRQ(ierr);
  ierr = PetscFree(pattern);CHKERRQ(ierr);
  ierr = PetscFree(form_atoms);CHKERRQ(ierr);
  ierr = PetscFree(p_scalar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTPTrimmedEvalJet - Evaluate the jet (function and derivatives) of a basis of the trimmed polynomial k-forms up to
  a given degree.

  Input Parameters:
+ dim - the number of variables in the multivariate polynomials
. npoints - the number of points to evaluate the polynomials at
. points - [npoints x dim] array of point coordinates
. degree - the degree (sum of degrees on the variables in a monomial) of the trimmed polynomial space to evaluate.
           There are ((dim + degree) choose (dim + formDegree)) x ((degree + formDegree - 1) choose (formDegree)) polynomials in this space.
           (You can use PetscDTPTrimmedSize() to compute this size.)
. formDegree - the degree of the form
- jetDegree - the maximum order partial derivative to evaluate in the jet.  There are ((dim + jetDegree) choose dim) partial derivatives
              in the jet.  Choosing jetDegree = 0 means to evaluate just the function and no derivatives

  Output Argments:
- p - an array containing the evaluations of the PKD polynomials' jets on the points.  The size is
      PetscDTPTrimmedSize() x ((dim + formDegree) choose dim) x ((dim + k) choose dim) x npoints,
      which also describes the order of the dimensions of this
      four-dimensional array:
        the first (slowest varying) dimension is basis function index;
        the second dimension is component of the form;
        the third dimension is jet index;
        the fourth (fastest varying) dimension is the index of the evaluation point.

  Level: advanced

  Note: The ordering of the basis functions is not graded, so the basis functions are not nested by degree like PetscDTPKDEvalJet().
        The basis functions are not an L2-orthonormal basis on any particular domain.

  The implementation is based on the description of the trimmed polynomials up to degree r as
  the direct sum of polynomials up to degree (r-1) and the Koszul differential applied to
  homogeneous polynomials of degree (r-1).

.seealso: PetscDTPKDEvalJet(), PetscDTPTrimmedSize()
@*/
PetscErrorCode PetscDTPTrimmedEvalJet(PetscInt dim, PetscInt npoints, const PetscReal points[], PetscInt degree, PetscInt formDegree, PetscInt jetDegree, PetscReal p[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTPTrimmedEvalJet_Internal(dim, npoints, points, degree, formDegree, jetDegree, p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* solve the symmetric tridiagonal eigenvalue system, writing the eigenvalues into eigs and the eigenvectors into V
 * with lds n; diag and subdiag are overwritten */
static PetscErrorCode PetscDTSymmetricTridiagonalEigensolve(PetscInt n, PetscReal diag[], PetscReal subdiag[],
                                                            PetscReal eigs[], PetscScalar V[])
{
  char jobz = 'V'; /* eigenvalues and eigenvectors */
  char range = 'A'; /* all eigenvalues will be found */
  PetscReal VL = 0.; /* ignored because range is 'A' */
  PetscReal VU = 0.; /* ignored because range is 'A' */
  PetscBLASInt IL = 0; /* ignored because range is 'A' */
  PetscBLASInt IU = 0; /* ignored because range is 'A' */
  PetscReal abstol = 0.; /* unused */
  PetscBLASInt bn, bm, ldz; /* bm will equal bn on exit */
  PetscBLASInt *isuppz;
  PetscBLASInt lwork, liwork;
  PetscReal workquery;
  PetscBLASInt  iworkquery;
  PetscBLASInt *iwork;
  PetscBLASInt info;
  PetscReal *work = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSCDTGAUSSIANQUADRATURE_EIG)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "A LAPACK symmetric tridiagonal eigensolver could not be found");
#endif
  ierr = PetscBLASIntCast(n, &bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n, &ldz);CHKERRQ(ierr);
#if !defined(PETSC_MISSING_LAPACK_STEGR)
  ierr = PetscMalloc1(2 * n, &isuppz);CHKERRQ(ierr);
  lwork = -1;
  liwork = -1;
  PetscStackCallBLAS("LAPACKstegr",LAPACKstegr_(&jobz,&range,&bn,diag,subdiag,&VL,&VU,&IL,&IU,&abstol,&bm,eigs,V,&ldz,isuppz,&workquery,&lwork,&iworkquery,&liwork,&info));
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_PLIB,"xSTEGR error");
  lwork = (PetscBLASInt) workquery;
  liwork = (PetscBLASInt) iworkquery;
  ierr = PetscMalloc2(lwork, &work, liwork, &iwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKstegr",LAPACKstegr_(&jobz,&range,&bn,diag,subdiag,&VL,&VU,&IL,&IU,&abstol,&bm,eigs,V,&ldz,isuppz,work,&lwork,iwork,&liwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_PLIB,"xSTEGR error");
  ierr = PetscFree2(work, iwork);CHKERRQ(ierr);
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
#elif !defined(PETSC_MISSING_LAPACK_STEQR)
  jobz = 'I'; /* Compute eigenvalues and eigenvectors of the
                 tridiagonal matrix.  Z is initialized to the identity
                 matrix. */
  ierr = PetscMalloc1(PetscMax(1,2*n-2),&work);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKsteqr",LAPACKsteqr_("I",&bn,diag,subdiag,V,&ldz,work,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_PLIB,"xSTEQR error");
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscArraycpy(eigs,diag,n);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* Formula for the weights at the endpoints (-1 and 1) of Gauss-Lobatto-Jacobi
 * quadrature rules on the interval [-1, 1] */
static PetscErrorCode PetscDTGaussLobattoJacobiEndweights_Internal(PetscInt n, PetscReal alpha, PetscReal beta, PetscReal *leftw, PetscReal *rightw)
{
  PetscReal twoab1;
  PetscInt  m = n - 2;
  PetscReal a = alpha + 1.;
  PetscReal b = beta + 1.;
  PetscReal gra, grb;

  PetscFunctionBegin;
  twoab1 = PetscPowReal(2., a + b - 1.);
#if defined(PETSC_HAVE_LGAMMA)
  grb = PetscExpReal(2. * PetscLGamma(b+1.) + PetscLGamma(m+1.) + PetscLGamma(m+a+1.) -
                     (PetscLGamma(m+b+1) + PetscLGamma(m+a+b+1.)));
  gra = PetscExpReal(2. * PetscLGamma(a+1.) + PetscLGamma(m+1.) + PetscLGamma(m+b+1.) -
                     (PetscLGamma(m+a+1) + PetscLGamma(m+a+b+1.)));
#else
  {
    PetscInt alphai = (PetscInt) alpha;
    PetscInt betai = (PetscInt) beta;
    PetscErrorCode ierr;

    if ((PetscReal) alphai == alpha && (PetscReal) betai == beta) {
      PetscReal binom1, binom2;

      ierr = PetscDTBinomial(m+b, b, &binom1);CHKERRQ(ierr);
      ierr = PetscDTBinomial(m+a+b, b, &binom2);CHKERRQ(ierr);
      grb = 1./ (binom1 * binom2);
      ierr = PetscDTBinomial(m+a, a, &binom1);CHKERRQ(ierr);
      ierr = PetscDTBinomial(m+a+b, a, &binom2);CHKERRQ(ierr);
      gra = 1./ (binom1 * binom2);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"lgamma() - math routine is unavailable.");
  }
#endif
  *leftw = twoab1 * grb / b;
  *rightw = twoab1 * gra / a;
  PetscFunctionReturn(0);
}

/* Evaluates the nth jacobi polynomial with weight parameters a,b at a point x.
   Recurrence relations implemented from the pseudocode given in Karniadakis and Sherwin, Appendix B */
static inline PetscErrorCode PetscDTComputeJacobi(PetscReal a, PetscReal b, PetscInt n, PetscReal x, PetscReal *P)
{
  PetscReal pn1, pn2;
  PetscReal cnm1, cnm1x, cnm2;
  PetscInt  k;

  PetscFunctionBegin;
  if (!n) {*P = 1.0; PetscFunctionReturn(0);}
  PetscDTJacobiRecurrence_Internal(1,a,b,cnm1,cnm1x,cnm2);
  pn2 = 1.;
  pn1 = cnm1 + cnm1x*x;
  if (n == 1) {*P = pn1; PetscFunctionReturn(0);}
  *P  = 0.0;
  for (k = 2; k < n+1; ++k) {
    PetscDTJacobiRecurrence_Internal(k,a,b,cnm1,cnm1x,cnm2);

    *P  = (cnm1 + cnm1x*x)*pn1 - cnm2*pn2;
    pn2 = pn1;
    pn1 = *P;
  }
  PetscFunctionReturn(0);
}

/* Evaluates the first derivative of P_{n}^{a,b} at a point x. */
static inline PetscErrorCode PetscDTComputeJacobiDerivative(PetscReal a, PetscReal b, PetscInt n, PetscReal x, PetscInt k, PetscReal *P)
{
  PetscReal      nP;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *P = 0.0;
  if (k > n) PetscFunctionReturn(0);
  ierr = PetscDTComputeJacobi(a+k, b+k, n-k, x, &nP);CHKERRQ(ierr);
  for (i = 0; i < k; i++) nP *= (a + b + n + 1. + i) * 0.5;
  *P = nP;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTGaussJacobiQuadrature_Newton_Internal(PetscInt npoints, PetscReal a, PetscReal b, PetscReal x[], PetscReal w[])
{
  PetscInt       maxIter = 100;
  PetscReal      eps     = PetscExpReal(0.75 * PetscLogReal(PETSC_MACHINE_EPSILON));
  PetscReal      a1, a6, gf;
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  a1 = PetscPowReal(2.0, a+b+1);
#if defined(PETSC_HAVE_LGAMMA)
  {
    PetscReal a2, a3, a4, a5;
    a2 = PetscLGamma(a + npoints + 1);
    a3 = PetscLGamma(b + npoints + 1);
    a4 = PetscLGamma(a + b + npoints + 1);
    a5 = PetscLGamma(npoints + 1);
    gf = PetscExpReal(a2 + a3 - (a4 + a5));
  }
#else
  {
    PetscInt ia, ib;

    ia = (PetscInt) a;
    ib = (PetscInt) b;
    gf = 1.;
    if (ia == a && ia >= 0) { /* compute ratio of rising factorals wrt a */
      for (k = 0; k < ia; k++) gf *= (npoints + 1. + k) / (npoints + b + 1. + k);
    } else if (b == b && ib >= 0) { /* compute ratio of rising factorials wrt b */
      for (k = 0; k < ib; k++) gf *= (npoints + 1. + k) / (npoints + a + 1. + k);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"lgamma() - math routine is unavailable.");
  }
#endif

  a6   = a1 * gf;
  /* Computes the m roots of P_{m}^{a,b} on [-1,1] by Newton's method with Chebyshev points as initial guesses.
   Algorithm implemented from the pseudocode given by Karniadakis and Sherwin and Python in FIAT */
  for (k = 0; k < npoints; ++k) {
    PetscReal r = PetscCosReal(PETSC_PI * (1. - (4.*k + 3. + 2.*b) / (4.*npoints + 2.*(a + b + 1.)))), dP;
    PetscInt  j;

    if (k > 0) r = 0.5 * (r + x[k-1]);
    for (j = 0; j < maxIter; ++j) {
      PetscReal s = 0.0, delta, f, fp;
      PetscInt  i;

      for (i = 0; i < k; ++i) s = s + 1.0 / (r - x[i]);
      ierr = PetscDTComputeJacobi(a, b, npoints, r, &f);CHKERRQ(ierr);
      ierr = PetscDTComputeJacobiDerivative(a, b, npoints, r, 1, &fp);CHKERRQ(ierr);
      delta = f / (fp - f * s);
      r     = r - delta;
      if (PetscAbsReal(delta) < eps) break;
    }
    x[k] = r;
    ierr = PetscDTComputeJacobiDerivative(a, b, npoints, x[k], 1, &dP);CHKERRQ(ierr);
    w[k] = a6 / (1.0 - PetscSqr(x[k])) / PetscSqr(dP);
  }
  PetscFunctionReturn(0);
}

/* Compute the diagonals of the Jacobi matrix used in Golub & Welsch algorithms for Gauss-Jacobi
 * quadrature weight calculations on [-1,1] for exponents (1. + x)^a (1.-x)^b */
static PetscErrorCode PetscDTJacobiMatrix_Internal(PetscInt nPoints, PetscReal a, PetscReal b, PetscReal *d, PetscReal *s)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i = 0; i < nPoints; i++) {
    PetscReal A, B, C;

    PetscDTJacobiRecurrence_Internal(i+1,a,b,A,B,C);
    d[i] = -A / B;
    if (i) s[i-1] *= C / B;
    if (i < nPoints - 1) s[i] = 1. / B;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTGaussJacobiQuadrature_GolubWelsch_Internal(PetscInt npoints, PetscReal a, PetscReal b, PetscReal *x, PetscReal *w)
{
  PetscReal mu0;
  PetscReal ga, gb, gab;
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(GolubWelschCitation, &GolubWelschCite);CHKERRQ(ierr);

#if defined(PETSC_HAVE_TGAMMA)
  ga  = PetscTGamma(a + 1);
  gb  = PetscTGamma(b + 1);
  gab = PetscTGamma(a + b + 2);
#else
  {
    PetscInt ia, ib;

    ia = (PetscInt) a;
    ib = (PetscInt) b;
    if (ia == a && ib == b && ia + 1 > 0 && ib + 1 > 0 && ia + ib + 2 > 0) { /* All gamma(x) terms are (x-1)! terms */
      ierr = PetscDTFactorial(ia, &ga);CHKERRQ(ierr);
      ierr = PetscDTFactorial(ib, &gb);CHKERRQ(ierr);
      ierr = PetscDTFactorial(ia + ib + 1, &gb);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"tgamma() - math routine is unavailable.");
  }
#endif
  mu0 = PetscPowReal(2.,a + b + 1.) * ga * gb / gab;

#if defined(PETSCDTGAUSSIANQUADRATURE_EIG)
  {
    PetscReal *diag, *subdiag;
    PetscScalar *V;

    ierr = PetscMalloc2(npoints, &diag, npoints, &subdiag);CHKERRQ(ierr);
    ierr = PetscMalloc1(npoints*npoints, &V);CHKERRQ(ierr);
    ierr = PetscDTJacobiMatrix_Internal(npoints, a, b, diag, subdiag);CHKERRQ(ierr);
    for (i = 0; i < npoints - 1; i++) subdiag[i] = PetscSqrtReal(subdiag[i]);
    ierr = PetscDTSymmetricTridiagonalEigensolve(npoints, diag, subdiag, x, V);CHKERRQ(ierr);
    for (i = 0; i < npoints; i++) w[i] = PetscSqr(PetscRealPart(V[i * npoints])) * mu0;
    ierr = PetscFree(V);CHKERRQ(ierr);
    ierr = PetscFree2(diag, subdiag);CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "A LAPACK symmetric tridiagonal eigensolver could not be found");
#endif
  { /* As of March 2, 2020, The Sun Performance Library breaks the LAPACK contract for xstegr and xsteqr: the
       eigenvalues are not guaranteed to be in ascending order.  So we heave a passive aggressive sigh and check that
       the eigenvalues are sorted */
    PetscBool sorted;

    ierr = PetscSortedReal(npoints, x, &sorted);CHKERRQ(ierr);
    if (!sorted) {
      PetscInt *order, i;
      PetscReal *tmp;

      ierr = PetscMalloc2(npoints, &order, npoints, &tmp);CHKERRQ(ierr);
      for (i = 0; i < npoints; i++) order[i] = i;
      ierr = PetscSortRealWithPermutation(npoints, x, order);CHKERRQ(ierr);
      ierr = PetscArraycpy(tmp, x, npoints);CHKERRQ(ierr);
      for (i = 0; i < npoints; i++) x[i] = tmp[order[i]];
      ierr = PetscArraycpy(tmp, w, npoints);CHKERRQ(ierr);
      for (i = 0; i < npoints; i++) w[i] = tmp[order[i]];
      ierr = PetscFree2(order, tmp);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTGaussJacobiQuadrature_Internal(PetscInt npoints,PetscReal alpha, PetscReal beta, PetscReal x[], PetscReal w[], PetscBool newton)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(npoints < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of points must be positive");
  /* If asking for a 1D Lobatto point, just return the non-Lobatto 1D point */
  PetscCheckFalse(alpha <= -1.,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"alpha must be > -1.");
  PetscCheckFalse(beta <= -1.,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"beta must be > -1.");

  if (newton) {
    ierr = PetscDTGaussJacobiQuadrature_Newton_Internal(npoints, alpha, beta, x, w);CHKERRQ(ierr);
  } else {
    ierr = PetscDTGaussJacobiQuadrature_GolubWelsch_Internal(npoints, alpha, beta, x, w);CHKERRQ(ierr);
  }
  if (alpha == beta) { /* symmetrize */
    PetscInt i;
    for (i = 0; i < (npoints + 1) / 2; i++) {
      PetscInt  j  = npoints - 1 - i;
      PetscReal xi = x[i];
      PetscReal xj = x[j];
      PetscReal wi = w[i];
      PetscReal wj = w[j];

      x[i] = (xi - xj) / 2.;
      x[j] = (xj - xi) / 2.;
      w[i] = w[j] = (wi + wj) / 2.;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDTGaussJacobiQuadrature - quadrature for the interval [a, b] with the weight function
  $(x-a)^\alpha (x-b)^\beta$.

  Not collective

  Input Parameters:
+ npoints - the number of points in the quadrature rule
. a - the left endpoint of the interval
. b - the right endpoint of the interval
. alpha - the left exponent
- beta - the right exponent

  Output Parameters:
+ x - array of length npoints, the locations of the quadrature points
- w - array of length npoints, the weights of the quadrature points

  Level: intermediate

  Note: this quadrature rule is exact for polynomials up to degree 2*npoints - 1.
@*/
PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt npoints,PetscReal a, PetscReal b, PetscReal alpha, PetscReal beta, PetscReal x[], PetscReal w[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTGaussJacobiQuadrature_Internal(npoints, alpha, beta, x, w, PetscDTGaussQuadratureNewton_Internal);CHKERRQ(ierr);
  if (a != -1. || b != 1.) { /* shift */
    for (i = 0; i < npoints; i++) {
      x[i] = (x[i] + 1.) * ((b - a) / 2.) + a;
      w[i] *= (b - a) / 2.;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTGaussLobattoJacobiQuadrature_Internal(PetscInt npoints,PetscReal alpha, PetscReal beta, PetscReal x[], PetscReal w[], PetscBool newton)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(npoints < 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of points must be positive");
  /* If asking for a 1D Lobatto point, just return the non-Lobatto 1D point */
  PetscCheckFalse(alpha <= -1.,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"alpha must be > -1.");
  PetscCheckFalse(beta <= -1.,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"beta must be > -1.");

  x[0] = -1.;
  x[npoints-1] = 1.;
  if (npoints > 2) {
    ierr = PetscDTGaussJacobiQuadrature_Internal(npoints-2, alpha+1., beta+1., &x[1], &w[1], newton);CHKERRQ(ierr);
  }
  for (i = 1; i < npoints - 1; i++) {
    w[i] /= (1. - x[i]*x[i]);
  }
  ierr = PetscDTGaussLobattoJacobiEndweights_Internal(npoints, alpha, beta, &w[0], &w[npoints-1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTGaussLobattoJacobiQuadrature - quadrature for the interval [a, b] with the weight function
  $(x-a)^\alpha (x-b)^\beta$, with endpoints a and b included as quadrature points.

  Not collective

  Input Parameters:
+ npoints - the number of points in the quadrature rule
. a - the left endpoint of the interval
. b - the right endpoint of the interval
. alpha - the left exponent
- beta - the right exponent

  Output Parameters:
+ x - array of length npoints, the locations of the quadrature points
- w - array of length npoints, the weights of the quadrature points

  Level: intermediate

  Note: this quadrature rule is exact for polynomials up to degree 2*npoints - 3.
@*/
PetscErrorCode PetscDTGaussLobattoJacobiQuadrature(PetscInt npoints,PetscReal a, PetscReal b, PetscReal alpha, PetscReal beta, PetscReal x[], PetscReal w[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTGaussLobattoJacobiQuadrature_Internal(npoints, alpha, beta, x, w, PetscDTGaussQuadratureNewton_Internal);CHKERRQ(ierr);
  if (a != -1. || b != 1.) { /* shift */
    for (i = 0; i < npoints; i++) {
      x[i] = (x[i] + 1.) * ((b - a) / 2.) + a;
      w[i] *= (b - a) / 2.;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTGaussQuadrature - create Gauss-Legendre quadrature

   Not Collective

   Input Parameters:
+  npoints - number of points
.  a - left end of interval (often-1)
-  b - right end of interval (often +1)

   Output Parameters:
+  x - quadrature points
-  w - quadrature weights

   Level: intermediate

   References:
.  * - Golub and Welsch, Calculation of Quadrature Rules, Math. Comp. 23(106), 1969.

.seealso: PetscDTLegendreEval()
@*/
PetscErrorCode PetscDTGaussQuadrature(PetscInt npoints,PetscReal a,PetscReal b,PetscReal *x,PetscReal *w)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTGaussJacobiQuadrature_Internal(npoints, 0., 0., x, w, PetscDTGaussQuadratureNewton_Internal);CHKERRQ(ierr);
  if (a != -1. || b != 1.) { /* shift */
    for (i = 0; i < npoints; i++) {
      x[i] = (x[i] + 1.) * ((b - a) / 2.) + a;
      w[i] *= (b - a) / 2.;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDTGaussLobattoLegendreQuadrature - creates a set of the locations and weights of the Gauss-Lobatto-Legendre
                      nodes of a given size on the domain [-1,1]

   Not Collective

   Input Parameters:
+  n - number of grid nodes
-  type - PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA or PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON

   Output Parameters:
+  x - quadrature points
-  w - quadrature weights

   Notes:
    For n > 30  the Newton approach computes duplicate (incorrect) values for some nodes because the initial guess is apparently not
          close enough to the desired solution

   These are useful for implementing spectral methods based on Gauss-Lobatto-Legendre (GLL) nodes

   See  https://epubs.siam.org/doi/abs/10.1137/110855442  https://epubs.siam.org/doi/abs/10.1137/120889873 for better ways to compute GLL nodes

   Level: intermediate

.seealso: PetscDTGaussQuadrature()

@*/
PetscErrorCode PetscDTGaussLobattoLegendreQuadrature(PetscInt npoints,PetscGaussLobattoLegendreCreateType type,PetscReal *x,PetscReal *w)
{
  PetscBool      newton;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(npoints < 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must provide at least 2 grid points per element");
  newton = (PetscBool) (type == PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON);
  ierr = PetscDTGaussLobattoJacobiQuadrature_Internal(npoints, 0., 0., x, w, newton);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTGaussTensorQuadrature - creates a tensor-product Gauss quadrature

  Not Collective

  Input Parameters:
+ dim     - The spatial dimension
. Nc      - The number of components
. npoints - number of points in one dimension
. a       - left end of interval (often-1)
- b       - right end of interval (often +1)

  Output Parameter:
. q - A PetscQuadrature object

  Level: intermediate

.seealso: PetscDTGaussQuadrature(), PetscDTLegendreEval()
@*/
PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt dim, PetscInt Nc, PetscInt npoints, PetscReal a, PetscReal b, PetscQuadrature *q)
{
  PetscInt       totpoints = dim > 1 ? dim > 2 ? npoints*PetscSqr(npoints) : PetscSqr(npoints) : npoints, i, j, k, c;
  PetscReal     *x, *w, *xw, *ww;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(totpoints*dim,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(totpoints*Nc,&w);CHKERRQ(ierr);
  /* Set up the Golub-Welsch system */
  switch (dim) {
  case 0:
    ierr = PetscFree(x);CHKERRQ(ierr);
    ierr = PetscFree(w);CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &x);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nc, &w);CHKERRQ(ierr);
    x[0] = 0.0;
    for (c = 0; c < Nc; ++c) w[c] = 1.0;
    break;
  case 1:
    ierr = PetscMalloc1(npoints,&ww);CHKERRQ(ierr);
    ierr = PetscDTGaussQuadrature(npoints, a, b, x, ww);CHKERRQ(ierr);
    for (i = 0; i < npoints; ++i) for (c = 0; c < Nc; ++c) w[i*Nc+c] = ww[i];
    ierr = PetscFree(ww);CHKERRQ(ierr);
    break;
  case 2:
    ierr = PetscMalloc2(npoints,&xw,npoints,&ww);CHKERRQ(ierr);
    ierr = PetscDTGaussQuadrature(npoints, a, b, xw, ww);CHKERRQ(ierr);
    for (i = 0; i < npoints; ++i) {
      for (j = 0; j < npoints; ++j) {
        x[(i*npoints+j)*dim+0] = xw[i];
        x[(i*npoints+j)*dim+1] = xw[j];
        for (c = 0; c < Nc; ++c) w[(i*npoints+j)*Nc+c] = ww[i] * ww[j];
      }
    }
    ierr = PetscFree2(xw,ww);CHKERRQ(ierr);
    break;
  case 3:
    ierr = PetscMalloc2(npoints,&xw,npoints,&ww);CHKERRQ(ierr);
    ierr = PetscDTGaussQuadrature(npoints, a, b, xw, ww);CHKERRQ(ierr);
    for (i = 0; i < npoints; ++i) {
      for (j = 0; j < npoints; ++j) {
        for (k = 0; k < npoints; ++k) {
          x[((i*npoints+j)*npoints+k)*dim+0] = xw[i];
          x[((i*npoints+j)*npoints+k)*dim+1] = xw[j];
          x[((i*npoints+j)*npoints+k)*dim+2] = xw[k];
          for (c = 0; c < Nc; ++c) w[((i*npoints+j)*npoints+k)*Nc+c] = ww[i] * ww[j] * ww[k];
        }
      }
    }
    ierr = PetscFree2(xw,ww);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct quadrature rule for dimension %d", dim);
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, q);CHKERRQ(ierr);
  ierr = PetscQuadratureSetOrder(*q, 2*npoints-1);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*q, dim, Nc, totpoints, x, w);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*q,"GaussTensor");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTStroudConicalQuadrature - create Stroud conical quadrature for a simplex

  Not Collective

  Input Parameters:
+ dim     - The simplex dimension
. Nc      - The number of components
. npoints - The number of points in one dimension
. a       - left end of interval (often-1)
- b       - right end of interval (often +1)

  Output Parameter:
. q - A PetscQuadrature object

  Level: intermediate

  References:
. * - Karniadakis and Sherwin.  FIAT

  Note: For dim == 1, this is Gauss-Legendre quadrature

.seealso: PetscDTGaussTensorQuadrature(), PetscDTGaussQuadrature()
@*/
PetscErrorCode PetscDTStroudConicalQuadrature(PetscInt dim, PetscInt Nc, PetscInt npoints, PetscReal a, PetscReal b, PetscQuadrature *q)
{
  PetscInt       totprev, totrem;
  PetscInt       totpoints;
  PetscReal     *p1, *w1;
  PetscReal     *x, *w;
  PetscInt       i, j, k, l, m, pt, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse((a != -1.0) || (b != 1.0),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must use default internal right now");
  totpoints = 1;
  for (i = 0, totpoints = 1; i < dim; i++) totpoints *= npoints;
  ierr = PetscMalloc1(totpoints*dim, &x);CHKERRQ(ierr);
  ierr = PetscMalloc1(totpoints*Nc, &w);CHKERRQ(ierr);
  ierr = PetscMalloc2(npoints, &p1, npoints, &w1);CHKERRQ(ierr);
  for (i = 0; i < totpoints*Nc; i++) w[i] = 1.;
  for (i = 0, totprev = 1, totrem = totpoints / npoints; i < dim; i++) {
    PetscReal mul;

    mul = PetscPowReal(2.,-i);
    ierr = PetscDTGaussJacobiQuadrature(npoints, -1., 1., i, 0.0, p1, w1);CHKERRQ(ierr);
    for (pt = 0, l = 0; l < totprev; l++) {
      for (j = 0; j < npoints; j++) {
        for (m = 0; m < totrem; m++, pt++) {
          for (k = 0; k < i; k++) x[pt*dim+k] = (x[pt*dim+k]+1.)*(1.-p1[j])*0.5 - 1.;
          x[pt * dim + i] = p1[j];
          for (c = 0; c < Nc; c++) w[pt*Nc + c] *= mul * w1[j];
        }
      }
    }
    totprev *= npoints;
    totrem /= npoints;
  }
  ierr = PetscFree2(p1, w1);CHKERRQ(ierr);
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, q);CHKERRQ(ierr);
  ierr = PetscQuadratureSetOrder(*q, 2*npoints-1);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*q, dim, Nc, totpoints, x, w);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*q,"StroudConical");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDTTanhSinhTensorQuadrature - create tanh-sinh quadrature for a tensor product cell

  Not Collective

  Input Parameters:
+ dim   - The cell dimension
. level - The number of points in one dimension, 2^l
. a     - left end of interval (often-1)
- b     - right end of interval (often +1)

  Output Parameter:
. q - A PetscQuadrature object

  Level: intermediate

.seealso: PetscDTGaussTensorQuadrature()
@*/
PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt dim, PetscInt level, PetscReal a, PetscReal b, PetscQuadrature *q)
{
  const PetscInt  p     = 16;                        /* Digits of precision in the evaluation */
  const PetscReal alpha = (b-a)/2.;                  /* Half-width of the integration interval */
  const PetscReal beta  = (b+a)/2.;                  /* Center of the integration interval */
  const PetscReal h     = PetscPowReal(2.0, -level); /* Step size, length between x_k */
  PetscReal       xk;                                /* Quadrature point x_k on reference domain [-1, 1] */
  PetscReal       wk    = 0.5*PETSC_PI;              /* Quadrature weight at x_k */
  PetscReal      *x, *w;
  PetscInt        K, k, npoints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCheckFalse(dim > 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Dimension %d not yet implemented", dim);
  PetscCheckFalse(!level,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must give a number of significant digits");
  /* Find K such that the weights are < 32 digits of precision */
  for (K = 1; PetscAbsReal(PetscLog10Real(wk)) < 2*p; ++K) {
    wk = 0.5*h*PETSC_PI*PetscCoshReal(K*h)/PetscSqr(PetscCoshReal(0.5*PETSC_PI*PetscSinhReal(K*h)));
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, q);CHKERRQ(ierr);
  ierr = PetscQuadratureSetOrder(*q, 2*K+1);CHKERRQ(ierr);
  npoints = 2*K-1;
  ierr = PetscMalloc1(npoints*dim, &x);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints, &w);CHKERRQ(ierr);
  /* Center term */
  x[0] = beta;
  w[0] = 0.5*alpha*PETSC_PI;
  for (k = 1; k < K; ++k) {
    wk = 0.5*alpha*h*PETSC_PI*PetscCoshReal(k*h)/PetscSqr(PetscCoshReal(0.5*PETSC_PI*PetscSinhReal(k*h)));
    xk = PetscTanhReal(0.5*PETSC_PI*PetscSinhReal(k*h));
    x[2*k-1] = -alpha*xk+beta;
    w[2*k-1] = wk;
    x[2*k+0] =  alpha*xk+beta;
    w[2*k+0] = wk;
  }
  ierr = PetscQuadratureSetData(*q, dim, 1, npoints, x, w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDTTanhSinhIntegrate(void (*func)(const PetscReal[], void *, PetscReal *), PetscReal a, PetscReal b, PetscInt digits, void *ctx, PetscReal *sol)
{
  const PetscInt  p     = 16;        /* Digits of precision in the evaluation */
  const PetscReal alpha = (b-a)/2.;  /* Half-width of the integration interval */
  const PetscReal beta  = (b+a)/2.;  /* Center of the integration interval */
  PetscReal       h     = 1.0;       /* Step size, length between x_k */
  PetscInt        l     = 0;         /* Level of refinement, h = 2^{-l} */
  PetscReal       osum  = 0.0;       /* Integral on last level */
  PetscReal       psum  = 0.0;       /* Integral on the level before the last level */
  PetscReal       sum;               /* Integral on current level */
  PetscReal       yk;                /* Quadrature point 1 - x_k on reference domain [-1, 1] */
  PetscReal       lx, rx;            /* Quadrature points to the left and right of 0 on the real domain [a, b] */
  PetscReal       wk;                /* Quadrature weight at x_k */
  PetscReal       lval, rval;        /* Terms in the quadature sum to the left and right of 0 */
  PetscInt        d;                 /* Digits of precision in the integral */

  PetscFunctionBegin;
  PetscCheckFalse(digits <= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must give a positive number of significant digits");
  /* Center term */
  func(&beta, ctx, &lval);
  sum = 0.5*alpha*PETSC_PI*lval;
  /* */
  do {
    PetscReal lterm, rterm, maxTerm = 0.0, d1, d2, d3, d4;
    PetscInt  k = 1;

    ++l;
    /* PetscPrintf(PETSC_COMM_SELF, "LEVEL %D sum: %15.15f\n", l, sum); */
    /* At each level of refinement, h --> h/2 and sum --> sum/2 */
    psum = osum;
    osum = sum;
    h   *= 0.5;
    sum *= 0.5;
    do {
      wk = 0.5*h*PETSC_PI*PetscCoshReal(k*h)/PetscSqr(PetscCoshReal(0.5*PETSC_PI*PetscSinhReal(k*h)));
      yk = 1.0/(PetscExpReal(0.5*PETSC_PI*PetscSinhReal(k*h)) * PetscCoshReal(0.5*PETSC_PI*PetscSinhReal(k*h)));
      lx = -alpha*(1.0 - yk)+beta;
      rx =  alpha*(1.0 - yk)+beta;
      func(&lx, ctx, &lval);
      func(&rx, ctx, &rval);
      lterm   = alpha*wk*lval;
      maxTerm = PetscMax(PetscAbsReal(lterm), maxTerm);
      sum    += lterm;
      rterm   = alpha*wk*rval;
      maxTerm = PetscMax(PetscAbsReal(rterm), maxTerm);
      sum    += rterm;
      ++k;
      /* Only need to evaluate every other point on refined levels */
      if (l != 1) ++k;
    } while (PetscAbsReal(PetscLog10Real(wk)) < p); /* Only need to evaluate sum until weights are < 32 digits of precision */

    d1 = PetscLog10Real(PetscAbsReal(sum - osum));
    d2 = PetscLog10Real(PetscAbsReal(sum - psum));
    d3 = PetscLog10Real(maxTerm) - p;
    if (PetscMax(PetscAbsReal(lterm), PetscAbsReal(rterm)) == 0.0) d4 = 0.0;
    else d4 = PetscLog10Real(PetscMax(PetscAbsReal(lterm), PetscAbsReal(rterm)));
    d  = PetscAbsInt(PetscMin(0, PetscMax(PetscMax(PetscMax(PetscSqr(d1)/d2, 2*d1), d3), d4)));
  } while (d < digits && l < 12);
  *sol = sum;

  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPFR)
PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*func)(const PetscReal[], void *, PetscReal *), PetscReal a, PetscReal b, PetscInt digits, void *ctx, PetscReal *sol)
{
  const PetscInt  safetyFactor = 2;  /* Calculate abcissa until 2*p digits */
  PetscInt        l            = 0;  /* Level of refinement, h = 2^{-l} */
  mpfr_t          alpha;             /* Half-width of the integration interval */
  mpfr_t          beta;              /* Center of the integration interval */
  mpfr_t          h;                 /* Step size, length between x_k */
  mpfr_t          osum;              /* Integral on last level */
  mpfr_t          psum;              /* Integral on the level before the last level */
  mpfr_t          sum;               /* Integral on current level */
  mpfr_t          yk;                /* Quadrature point 1 - x_k on reference domain [-1, 1] */
  mpfr_t          lx, rx;            /* Quadrature points to the left and right of 0 on the real domain [a, b] */
  mpfr_t          wk;                /* Quadrature weight at x_k */
  PetscReal       lval, rval, rtmp;  /* Terms in the quadature sum to the left and right of 0 */
  PetscInt        d;                 /* Digits of precision in the integral */
  mpfr_t          pi2, kh, msinh, mcosh, maxTerm, curTerm, tmp;

  PetscFunctionBegin;
  PetscCheckFalse(digits <= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must give a positive number of significant digits");
  /* Create high precision storage */
  mpfr_inits2(PetscCeilReal(safetyFactor*digits*PetscLogReal(10.)/PetscLogReal(2.)), alpha, beta, h, sum, osum, psum, yk, wk, lx, rx, tmp, maxTerm, curTerm, pi2, kh, msinh, mcosh, NULL);
  /* Initialization */
  mpfr_set_d(alpha, 0.5*(b-a), MPFR_RNDN);
  mpfr_set_d(beta,  0.5*(b+a), MPFR_RNDN);
  mpfr_set_d(osum,  0.0,       MPFR_RNDN);
  mpfr_set_d(psum,  0.0,       MPFR_RNDN);
  mpfr_set_d(h,     1.0,       MPFR_RNDN);
  mpfr_const_pi(pi2, MPFR_RNDN);
  mpfr_mul_d(pi2, pi2, 0.5, MPFR_RNDN);
  /* Center term */
  rtmp = 0.5*(b+a);
  func(&rtmp, ctx, &lval);
  mpfr_set(sum, pi2, MPFR_RNDN);
  mpfr_mul(sum, sum, alpha, MPFR_RNDN);
  mpfr_mul_d(sum, sum, lval, MPFR_RNDN);
  /* */
  do {
    PetscReal d1, d2, d3, d4;
    PetscInt  k = 1;

    ++l;
    mpfr_set_d(maxTerm, 0.0, MPFR_RNDN);
    /* PetscPrintf(PETSC_COMM_SELF, "LEVEL %D sum: %15.15f\n", l, sum); */
    /* At each level of refinement, h --> h/2 and sum --> sum/2 */
    mpfr_set(psum, osum, MPFR_RNDN);
    mpfr_set(osum,  sum, MPFR_RNDN);
    mpfr_mul_d(h,   h,   0.5, MPFR_RNDN);
    mpfr_mul_d(sum, sum, 0.5, MPFR_RNDN);
    do {
      mpfr_set_si(kh, k, MPFR_RNDN);
      mpfr_mul(kh, kh, h, MPFR_RNDN);
      /* Weight */
      mpfr_set(wk, h, MPFR_RNDN);
      mpfr_sinh_cosh(msinh, mcosh, kh, MPFR_RNDN);
      mpfr_mul(msinh, msinh, pi2, MPFR_RNDN);
      mpfr_mul(mcosh, mcosh, pi2, MPFR_RNDN);
      mpfr_cosh(tmp, msinh, MPFR_RNDN);
      mpfr_sqr(tmp, tmp, MPFR_RNDN);
      mpfr_mul(wk, wk, mcosh, MPFR_RNDN);
      mpfr_div(wk, wk, tmp, MPFR_RNDN);
      /* Abscissa */
      mpfr_set_d(yk, 1.0, MPFR_RNDZ);
      mpfr_cosh(tmp, msinh, MPFR_RNDN);
      mpfr_div(yk, yk, tmp, MPFR_RNDZ);
      mpfr_exp(tmp, msinh, MPFR_RNDN);
      mpfr_div(yk, yk, tmp, MPFR_RNDZ);
      /* Quadrature points */
      mpfr_sub_d(lx, yk, 1.0, MPFR_RNDZ);
      mpfr_mul(lx, lx, alpha, MPFR_RNDU);
      mpfr_add(lx, lx, beta, MPFR_RNDU);
      mpfr_d_sub(rx, 1.0, yk, MPFR_RNDZ);
      mpfr_mul(rx, rx, alpha, MPFR_RNDD);
      mpfr_add(rx, rx, beta, MPFR_RNDD);
      /* Evaluation */
      rtmp = mpfr_get_d(lx, MPFR_RNDU);
      func(&rtmp, ctx, &lval);
      rtmp = mpfr_get_d(rx, MPFR_RNDD);
      func(&rtmp, ctx, &rval);
      /* Update */
      mpfr_mul(tmp, wk, alpha, MPFR_RNDN);
      mpfr_mul_d(tmp, tmp, lval, MPFR_RNDN);
      mpfr_add(sum, sum, tmp, MPFR_RNDN);
      mpfr_abs(tmp, tmp, MPFR_RNDN);
      mpfr_max(maxTerm, maxTerm, tmp, MPFR_RNDN);
      mpfr_set(curTerm, tmp, MPFR_RNDN);
      mpfr_mul(tmp, wk, alpha, MPFR_RNDN);
      mpfr_mul_d(tmp, tmp, rval, MPFR_RNDN);
      mpfr_add(sum, sum, tmp, MPFR_RNDN);
      mpfr_abs(tmp, tmp, MPFR_RNDN);
      mpfr_max(maxTerm, maxTerm, tmp, MPFR_RNDN);
      mpfr_max(curTerm, curTerm, tmp, MPFR_RNDN);
      ++k;
      /* Only need to evaluate every other point on refined levels */
      if (l != 1) ++k;
      mpfr_log10(tmp, wk, MPFR_RNDN);
      mpfr_abs(tmp, tmp, MPFR_RNDN);
    } while (mpfr_get_d(tmp, MPFR_RNDN) < safetyFactor*digits); /* Only need to evaluate sum until weights are < 32 digits of precision */
    mpfr_sub(tmp, sum, osum, MPFR_RNDN);
    mpfr_abs(tmp, tmp, MPFR_RNDN);
    mpfr_log10(tmp, tmp, MPFR_RNDN);
    d1 = mpfr_get_d(tmp, MPFR_RNDN);
    mpfr_sub(tmp, sum, psum, MPFR_RNDN);
    mpfr_abs(tmp, tmp, MPFR_RNDN);
    mpfr_log10(tmp, tmp, MPFR_RNDN);
    d2 = mpfr_get_d(tmp, MPFR_RNDN);
    mpfr_log10(tmp, maxTerm, MPFR_RNDN);
    d3 = mpfr_get_d(tmp, MPFR_RNDN) - digits;
    mpfr_log10(tmp, curTerm, MPFR_RNDN);
    d4 = mpfr_get_d(tmp, MPFR_RNDN);
    d  = PetscAbsInt(PetscMin(0, PetscMax(PetscMax(PetscMax(PetscSqr(d1)/d2, 2*d1), d3), d4)));
  } while (d < digits && l < 8);
  *sol = mpfr_get_d(sum, MPFR_RNDN);
  /* Cleanup */
  mpfr_clears(alpha, beta, h, sum, osum, psum, yk, wk, lx, rx, tmp, maxTerm, curTerm, pi2, kh, msinh, mcosh, NULL);
  PetscFunctionReturn(0);
}
#else

PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*func)(const PetscReal[], void *, PetscReal *), PetscReal a, PetscReal b, PetscInt digits, void *ctx, PetscReal *sol)
{
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "This method will not work without MPFR. Reconfigure using --download-mpfr --download-gmp");
}
#endif

/*@
  PetscDTTensorQuadratureCreate - create the tensor product quadrature from two lower-dimensional quadratures

  Not Collective

  Input Parameters:
+ q1 - The first quadrature
- q2 - The second quadrature

  Output Parameter:
. q - A PetscQuadrature object

  Level: intermediate

.seealso: PetscDTGaussTensorQuadrature()
@*/
PetscErrorCode PetscDTTensorQuadratureCreate(PetscQuadrature q1, PetscQuadrature q2, PetscQuadrature *q)
{
  const PetscReal *x1, *w1, *x2, *w2;
  PetscReal       *x, *w;
  PetscInt         dim1, Nc1, Np1, order1, qa, d1;
  PetscInt         dim2, Nc2, Np2, order2, qb, d2;
  PetscInt         dim,  Nc,  Np,  order, qc, d;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(q1, PETSCQUADRATURE_CLASSID, 1);
  PetscValidHeaderSpecific(q2, PETSCQUADRATURE_CLASSID, 2);
  PetscValidPointer(q, 3);
  ierr = PetscQuadratureGetOrder(q1, &order1);CHKERRQ(ierr);
  ierr = PetscQuadratureGetOrder(q2, &order2);CHKERRQ(ierr);
  PetscCheck(order1 == order2, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Order1 %" PetscInt_FMT " != %" PetscInt_FMT " Order2", order1, order2);
  ierr = PetscQuadratureGetData(q1, &dim1, &Nc1, &Np1, &x1, &w1);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q2, &dim2, &Nc2, &Np2, &x2, &w2);CHKERRQ(ierr);
  PetscCheck(Nc1 == Nc2, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "NumComp1 %" PetscInt_FMT " != %" PetscInt_FMT " NumComp2", Nc1, Nc2);

  dim   = dim1 + dim2;
  Nc    = Nc1;
  Np    = Np1 * Np2;
  order = order1;
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, q);CHKERRQ(ierr);
  ierr = PetscQuadratureSetOrder(*q, order);CHKERRQ(ierr);
  ierr = PetscMalloc1(Np*dim, &x);CHKERRQ(ierr);
  ierr = PetscMalloc1(Np, &w);CHKERRQ(ierr);
  for (qa = 0, qc = 0; qa < Np1; ++qa) {
    for (qb = 0; qb < Np2; ++qb, ++qc) {
      for (d1 = 0, d = 0; d1 < dim1; ++d1, ++d) {
        x[qc*dim+d] = x1[qa*dim1+d1];
      }
      for (d2 = 0; d2 < dim2; ++d2, ++d) {
        x[qc*dim+d] = x2[qb*dim2+d2];
      }
      w[qc] = w1[qa] * w2[qb];
    }
  }
  ierr = PetscQuadratureSetData(*q, dim, Nc, Np, x, w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Overwrites A. Can only handle full-rank problems with m>=n
 * A in column-major format
 * Ainv in row-major format
 * tau has length m
 * worksize must be >= max(1,n)
 */
static PetscErrorCode PetscDTPseudoInverseQR(PetscInt m,PetscInt mstride,PetscInt n,PetscReal *A_in,PetscReal *Ainv_out,PetscScalar *tau,PetscInt worksize,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscBLASInt   M,N,K,lda,ldb,ldwork,info;
  PetscScalar    *A,*Ainv,*R,*Q,Alpha;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  {
    PetscInt i,j;
    ierr = PetscMalloc2(m*n,&A,m*n,&Ainv);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      for (i=0; i<m; i++) A[i+m*j] = A_in[i+mstride*j];
    }
    mstride = m;
  }
#else
  A = A_in;
  Ainv = Ainv_out;
#endif

  ierr = PetscBLASIntCast(m,&M);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(mstride,&lda);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(worksize,&ldwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&M,&N,A,&lda,tau,work,&ldwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");
  R = A; /* Upper triangular part of A now contains R, the rest contains the elementary reflectors */

  /* Extract an explicit representation of Q */
  Q = Ainv;
  ierr = PetscArraycpy(Q,A,mstride*n);CHKERRQ(ierr);
  K = N;                        /* full rank */
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&M,&N,&K,Q,&lda,tau,work,&ldwork,&info));
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");

  /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
  Alpha = 1.0;
  ldb = lda;
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("Right","Upper","ConjugateTranspose","NotUnitTriangular",&M,&N,&Alpha,R,&lda,Q,&ldb));
  /* Ainv is Q, overwritten with inverse */

#if defined(PETSC_USE_COMPLEX)
  {
    PetscInt i;
    for (i=0; i<m*n; i++) Ainv_out[i] = PetscRealPart(Ainv[i]);
    ierr = PetscFree2(A,Ainv);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* Computes integral of L_p' over intervals {(x0,x1),(x1,x2),...} */
static PetscErrorCode PetscDTLegendreIntegrate(PetscInt ninterval,const PetscReal *x,PetscInt ndegree,const PetscInt *degrees,PetscBool Transpose,PetscReal *B)
{
  PetscErrorCode ierr;
  PetscReal      *Bv;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc1((ninterval+1)*ndegree,&Bv);CHKERRQ(ierr);
  /* Point evaluation of L_p on all the source vertices */
  ierr = PetscDTLegendreEval(ninterval+1,x,ndegree,degrees,Bv,NULL,NULL);CHKERRQ(ierr);
  /* Integral over each interval: \int_a^b L_p' = L_p(b)-L_p(a) */
  for (i=0; i<ninterval; i++) {
    for (j=0; j<ndegree; j++) {
      if (Transpose) B[i+ninterval*j] = Bv[(i+1)*ndegree+j] - Bv[i*ndegree+j];
      else           B[i*ndegree+j]   = Bv[(i+1)*ndegree+j] - Bv[i*ndegree+j];
    }
  }
  ierr = PetscFree(Bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDTReconstructPoly - create matrix representing polynomial reconstruction using cell intervals and evaluation at target intervals

   Not Collective

   Input Parameters:
+  degree - degree of reconstruction polynomial
.  nsource - number of source intervals
.  sourcex - sorted coordinates of source cell boundaries (length nsource+1)
.  ntarget - number of target intervals
-  targetx - sorted coordinates of target cell boundaries (length ntarget+1)

   Output Parameter:
.  R - reconstruction matrix, utarget = sum_s R[t*nsource+s] * usource[s]

   Level: advanced

.seealso: PetscDTLegendreEval()
@*/
PetscErrorCode PetscDTReconstructPoly(PetscInt degree,PetscInt nsource,const PetscReal *sourcex,PetscInt ntarget,const PetscReal *targetx,PetscReal *R)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,*bdegrees,worksize;
  PetscReal      xmin,xmax,center,hscale,*sourcey,*targety,*Bsource,*Bsinv,*Btarget;
  PetscScalar    *tau,*work;

  PetscFunctionBegin;
  PetscValidRealPointer(sourcex,3);
  PetscValidRealPointer(targetx,5);
  PetscValidRealPointer(R,6);
  PetscCheckFalse(degree >= nsource,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Reconstruction degree %D must be less than number of source intervals %D",degree,nsource);
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<nsource; i++) {
      PetscCheckFalse(sourcex[i] >= sourcex[i+1],PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Source interval %D has negative orientation (%g,%g)",i,(double)sourcex[i],(double)sourcex[i+1]);
    }
    for (i=0; i<ntarget; i++) {
      PetscCheckFalse(targetx[i] >= targetx[i+1],PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Target interval %D has negative orientation (%g,%g)",i,(double)targetx[i],(double)targetx[i+1]);
    }
  }
  xmin = PetscMin(sourcex[0],targetx[0]);
  xmax = PetscMax(sourcex[nsource],targetx[ntarget]);
  center = (xmin + xmax)/2;
  hscale = (xmax - xmin)/2;
  worksize = nsource;
  ierr = PetscMalloc4(degree+1,&bdegrees,nsource+1,&sourcey,nsource*(degree+1),&Bsource,worksize,&work);CHKERRQ(ierr);
  ierr = PetscMalloc4(nsource,&tau,nsource*(degree+1),&Bsinv,ntarget+1,&targety,ntarget*(degree+1),&Btarget);CHKERRQ(ierr);
  for (i=0; i<=nsource; i++) sourcey[i] = (sourcex[i]-center)/hscale;
  for (i=0; i<=degree; i++) bdegrees[i] = i+1;
  ierr = PetscDTLegendreIntegrate(nsource,sourcey,degree+1,bdegrees,PETSC_TRUE,Bsource);CHKERRQ(ierr);
  ierr = PetscDTPseudoInverseQR(nsource,nsource,degree+1,Bsource,Bsinv,tau,nsource,work);CHKERRQ(ierr);
  for (i=0; i<=ntarget; i++) targety[i] = (targetx[i]-center)/hscale;
  ierr = PetscDTLegendreIntegrate(ntarget,targety,degree+1,bdegrees,PETSC_FALSE,Btarget);CHKERRQ(ierr);
  for (i=0; i<ntarget; i++) {
    PetscReal rowsum = 0;
    for (j=0; j<nsource; j++) {
      PetscReal sum = 0;
      for (k=0; k<degree+1; k++) {
        sum += Btarget[i*(degree+1)+k] * Bsinv[k*nsource+j];
      }
      R[i*nsource+j] = sum;
      rowsum += sum;
    }
    for (j=0; j<nsource; j++) R[i*nsource+j] /= rowsum; /* normalize each row */
  }
  ierr = PetscFree4(bdegrees,sourcey,Bsource,work);CHKERRQ(ierr);
  ierr = PetscFree4(tau,Bsinv,targety,Btarget);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreIntegrate - Compute the L2 integral of a function on the GLL points

   Not Collective

   Input Parameters:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
.  weights - the GLL weights
-  f - the function values at the nodes

   Output Parameter:
.  in - the value of the integral

   Level: beginner

.seealso: PetscDTGaussLobattoLegendreQuadrature()

@*/
PetscErrorCode PetscGaussLobattoLegendreIntegrate(PetscInt n,PetscReal *nodes,PetscReal *weights,const PetscReal *f,PetscReal *in)
{
  PetscInt          i;

  PetscFunctionBegin;
  *in = 0.;
  for (i=0; i<n; i++) {
    *in += f[i]*f[i]*weights[i];
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreElementLaplacianCreate - computes the Laplacian for a single 1d GLL element

   Not Collective

   Input Parameters:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
-  weights - the GLL weights

   Output Parameter:
.  A - the stiffness element

   Level: beginner

   Notes:
    Destroy this with PetscGaussLobattoLegendreElementLaplacianDestroy()

   You can access entries in this array with AA[i][j] but in memory it is stored in contiguous memory, row oriented (the array is symmetric)

.seealso: PetscDTGaussLobattoLegendreQuadrature(), PetscGaussLobattoLegendreElementLaplacianDestroy()

@*/
PetscErrorCode PetscGaussLobattoLegendreElementLaplacianCreate(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA)
{
  PetscReal        **A;
  PetscErrorCode  ierr;
  const PetscReal  *gllnodes = nodes;
  const PetscInt   p = n-1;
  PetscReal        z0,z1,z2 = -1,x,Lpj,Lpr;
  PetscInt         i,j,nn,r;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*n,&A[0]);CHKERRQ(ierr);
  for (i=1; i<n; i++) A[i] = A[i-1]+n;

  for (j=1; j<p; j++) {
    x  = gllnodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj=z2;
    for (r=1; r<p; r++) {
      if (r == j) {
        A[j][j]=2./(3.*(1.-gllnodes[j]*gllnodes[j])*Lpj*Lpj);
      } else {
        x  = gllnodes[r];
        z0 = 1.;
        z1 = x;
        for (nn=1; nn<p; nn++) {
          z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
          z0 = z1;
          z1 = z2;
        }
        Lpr     = z2;
        A[r][j] = 4./(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*Lpr*(gllnodes[j]-gllnodes[r])*(gllnodes[j]-gllnodes[r]));
      }
    }
  }
  for (j=1; j<p+1; j++) {
    x  = gllnodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj     = z2;
    A[j][0] = 4.*PetscPowRealInt(-1.,p)/(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*(1.+gllnodes[j])*(1.+gllnodes[j]));
    A[0][j] = A[j][0];
  }
  for (j=0; j<p; j++) {
    x  = gllnodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj=z2;

    A[p][j] = 4./(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*(1.-gllnodes[j])*(1.-gllnodes[j]));
    A[j][p] = A[p][j];
  }
  A[0][0]=0.5+(((PetscReal)p)*(((PetscReal)p)+1.)-2.)/6.;
  A[p][p]=A[0][0];
  *AA = A;
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreElementLaplacianDestroy - frees the Laplacian for a single 1d GLL element

   Not Collective

   Input Parameters:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
.  weights - the GLL weightss
-  A - the stiffness element

   Level: beginner

.seealso: PetscDTGaussLobattoLegendreQuadrature(), PetscGaussLobattoLegendreElementLaplacianCreate()

@*/
PetscErrorCode PetscGaussLobattoLegendreElementLaplacianDestroy(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreElementGradientCreate - computes the gradient for a single 1d GLL element

   Not Collective

   Input Parameter:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
.  weights - the GLL weights

   Output Parameters:
.  AA - the stiffness element
-  AAT - the transpose of AA (pass in NULL if you do not need this array)

   Level: beginner

   Notes:
    Destroy this with PetscGaussLobattoLegendreElementGradientDestroy()

   You can access entries in these arrays with AA[i][j] but in memory it is stored in contiguous memory, row oriented

.seealso: PetscDTGaussLobattoLegendreQuadrature(), PetscGaussLobattoLegendreElementLaplacianDestroy()

@*/
PetscErrorCode PetscGaussLobattoLegendreElementGradientCreate(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA,PetscReal ***AAT)
{
  PetscReal        **A, **AT = NULL;
  PetscErrorCode  ierr;
  const PetscReal  *gllnodes = nodes;
  const PetscInt   p = n-1;
  PetscReal        Li, Lj,d0;
  PetscInt         i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*n,&A[0]);CHKERRQ(ierr);
  for (i=1; i<n; i++) A[i] = A[i-1]+n;

  if (AAT) {
    ierr = PetscMalloc1(n,&AT);CHKERRQ(ierr);
    ierr = PetscMalloc1(n*n,&AT[0]);CHKERRQ(ierr);
    for (i=1; i<n; i++) AT[i] = AT[i-1]+n;
  }

  if (n==1) {A[0][0] = 0.;}
  d0 = (PetscReal)p*((PetscReal)p+1.)/4.;
  for  (i=0; i<n; i++) {
    for  (j=0; j<n; j++) {
      A[i][j] = 0.;
      ierr = PetscDTComputeJacobi(0., 0., p, gllnodes[i], &Li);CHKERRQ(ierr);
      ierr = PetscDTComputeJacobi(0., 0., p, gllnodes[j], &Lj);CHKERRQ(ierr);
      if (i!=j)             A[i][j] = Li/(Lj*(gllnodes[i]-gllnodes[j]));
      if ((j==i) && (i==0)) A[i][j] = -d0;
      if (j==i && i==p)     A[i][j] = d0;
      if (AT) AT[j][i] = A[i][j];
    }
  }
  if (AAT) *AAT = AT;
  *AA  = A;
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreElementGradientDestroy - frees the gradient for a single 1d GLL element obtained with PetscGaussLobattoLegendreElementGradientCreate()

   Not Collective

   Input Parameters:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
.  weights - the GLL weights
.  AA - the stiffness element
-  AAT - the transpose of the element

   Level: beginner

.seealso: PetscDTGaussLobattoLegendreQuadrature(), PetscGaussLobattoLegendreElementLaplacianCreate(), PetscGaussLobattoLegendreElementAdvectionCreate()

@*/
PetscErrorCode PetscGaussLobattoLegendreElementGradientDestroy(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA,PetscReal ***AAT)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  if (*AAT) {
    ierr = PetscFree((*AAT)[0]);CHKERRQ(ierr);
    ierr = PetscFree(*AAT);CHKERRQ(ierr);
    *AAT  = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreElementAdvectionCreate - computes the advection operator for a single 1d GLL element

   Not Collective

   Input Parameters:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
-  weights - the GLL weightss

   Output Parameter:
.  AA - the stiffness element

   Level: beginner

   Notes:
    Destroy this with PetscGaussLobattoLegendreElementAdvectionDestroy()

   This is the same as the Gradient operator multiplied by the diagonal mass matrix

   You can access entries in this array with AA[i][j] but in memory it is stored in contiguous memory, row oriented

.seealso: PetscDTGaussLobattoLegendreQuadrature(), PetscGaussLobattoLegendreElementLaplacianCreate(), PetscGaussLobattoLegendreElementAdvectionDestroy()

@*/
PetscErrorCode PetscGaussLobattoLegendreElementAdvectionCreate(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA)
{
  PetscReal       **D;
  PetscErrorCode  ierr;
  const PetscReal  *gllweights = weights;
  const PetscInt   glln = n;
  PetscInt         i,j;

  PetscFunctionBegin;
  ierr = PetscGaussLobattoLegendreElementGradientCreate(n,nodes,weights,&D,NULL);CHKERRQ(ierr);
  for (i=0; i<glln; i++) {
    for (j=0; j<glln; j++) {
      D[i][j] = gllweights[i]*D[i][j];
    }
  }
  *AA = D;
  PetscFunctionReturn(0);
}

/*@C
   PetscGaussLobattoLegendreElementAdvectionDestroy - frees the advection stiffness for a single 1d GLL element

   Not Collective

   Input Parameters:
+  n - the number of GLL nodes
.  nodes - the GLL nodes
.  weights - the GLL weights
-  A - advection

   Level: beginner

.seealso: PetscDTGaussLobattoLegendreQuadrature(), PetscGaussLobattoLegendreElementAdvectionCreate()

@*/
PetscErrorCode PetscGaussLobattoLegendreElementAdvectionDestroy(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGaussLobattoLegendreElementMassCreate(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA)
{
  PetscReal        **A;
  PetscErrorCode  ierr;
  const PetscReal  *gllweights = weights;
  const PetscInt   glln = n;
  PetscInt         i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc1(glln,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(glln*glln,&A[0]);CHKERRQ(ierr);
  for (i=1; i<glln; i++) A[i] = A[i-1]+glln;
  if (glln==1) {A[0][0] = 0.;}
  for  (i=0; i<glln; i++) {
    for  (j=0; j<glln; j++) {
      A[i][j] = 0.;
      if (j==i)     A[i][j] = gllweights[i];
    }
  }
  *AA  = A;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGaussLobattoLegendreElementMassDestroy(PetscInt n,PetscReal *nodes,PetscReal *weights,PetscReal ***AA)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscDTIndexToBary - convert an index into a barycentric coordinate.

  Input Parameters:
+ len - the desired length of the barycentric tuple (usually 1 more than the dimension it represents, so a barycentric coordinate in a triangle has length 3)
. sum - the value that the sum of the barycentric coordinates (which will be non-negative integers) should sum to
- index - the index to convert: should be >= 0 and < Binomial(len - 1 + sum, sum)

  Output Parameter:
. coord - will be filled with the barycentric coordinate

  Level: beginner

  Note: the indices map to barycentric coordinates in lexicographic order, where the first index is the
  least significant and the last index is the most significant.

.seealso: PetscDTBaryToIndex()
@*/
PetscErrorCode PetscDTIndexToBary(PetscInt len, PetscInt sum, PetscInt index, PetscInt coord[])
{
  PetscInt c, d, s, total, subtotal, nexttotal;

  PetscFunctionBeginHot;
  PetscCheckFalse(len < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "length must be non-negative");
  PetscCheckFalse(index < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index must be non-negative");
  if (!len) {
    if (!sum && !index) PetscFunctionReturn(0);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid index or sum for length 0 barycentric coordinate");
  }
  for (c = 1, total = 1; c <= len; c++) {
    /* total is the number of ways to have a tuple of length c with sum */
    if (index < total) break;
    total = (total * (sum + c)) / c;
  }
  PetscCheckFalse(c > len,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index out of range");
  for (d = c; d < len; d++) coord[d] = 0;
  for (s = 0, subtotal = 1, nexttotal = 1; c > 0;) {
    /* subtotal is the number of ways to have a tuple of length c with sum s */
    /* nexttotal is the number of ways to have a tuple of length c-1 with sum s */
    if ((index + subtotal) >= total) {
      coord[--c] = sum - s;
      index -= (total - subtotal);
      sum = s;
      total = nexttotal;
      subtotal = 1;
      nexttotal = 1;
      s = 0;
    } else {
      subtotal = (subtotal * (c + s)) / (s + 1);
      nexttotal = (nexttotal * (c - 1 + s)) / (s + 1);
      s++;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDTBaryToIndex - convert a barycentric coordinate to an index

  Input Parameters:
+ len - the desired length of the barycentric tuple (usually 1 more than the dimension it represents, so a barycentric coordinate in a triangle has length 3)
. sum - the value that the sum of the barycentric coordinates (which will be non-negative integers) should sum to
- coord - a barycentric coordinate with the given length and sum

  Output Parameter:
. index - the unique index for the coordinate, >= 0 and < Binomial(len - 1 + sum, sum)

  Level: beginner

  Note: the indices map to barycentric coordinates in lexicographic order, where the first index is the
  least significant and the last index is the most significant.

.seealso: PetscDTIndexToBary
@*/
PetscErrorCode PetscDTBaryToIndex(PetscInt len, PetscInt sum, const PetscInt coord[], PetscInt *index)
{
  PetscInt c;
  PetscInt i;
  PetscInt total;

  PetscFunctionBeginHot;
  PetscCheckFalse(len < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "length must be non-negative");
  if (!len) {
    if (!sum) {
      *index = 0;
      PetscFunctionReturn(0);
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid index or sum for length 0 barycentric coordinate");
  }
  for (c = 1, total = 1; c < len; c++) total = (total * (sum + c)) / c;
  i = total - 1;
  c = len - 1;
  sum -= coord[c];
  while (sum > 0) {
    PetscInt subtotal;
    PetscInt s;

    for (s = 1, subtotal = 1; s < sum; s++) subtotal = (subtotal * (c + s)) / s;
    i   -= subtotal;
    sum -= coord[--c];
  }
  *index = i;
  PetscFunctionReturn(0);
}
