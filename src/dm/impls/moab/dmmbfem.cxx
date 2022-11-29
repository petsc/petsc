
#include <petscconf.h>
#include <petscdt.h>                /*I "petscdt.h" I*/
#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/

/* Utility functions */
static inline PetscReal DMatrix_Determinant_2x2_Internal(const PetscReal inmat[2 * 2])
{
  return inmat[0] * inmat[3] - inmat[1] * inmat[2];
}

static inline PetscErrorCode DMatrix_Invert_2x2_Internal(const PetscReal *inmat, PetscReal *outmat, PetscReal *determinant)
{
  PetscReal det = DMatrix_Determinant_2x2_Internal(inmat);
  if (outmat) {
    outmat[0] = inmat[3] / det;
    outmat[1] = -inmat[1] / det;
    outmat[2] = -inmat[2] / det;
    outmat[3] = inmat[0] / det;
  }
  if (determinant) *determinant = det;
  return 0;
}

static inline PetscReal DMatrix_Determinant_3x3_Internal(const PetscReal inmat[3 * 3])
{
  return inmat[0] * (inmat[8] * inmat[4] - inmat[7] * inmat[5]) - inmat[3] * (inmat[8] * inmat[1] - inmat[7] * inmat[2]) + inmat[6] * (inmat[5] * inmat[1] - inmat[4] * inmat[2]);
}

static inline PetscErrorCode DMatrix_Invert_3x3_Internal(const PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  PetscReal det = DMatrix_Determinant_3x3_Internal(inmat);
  if (outmat) {
    outmat[0] = (inmat[8] * inmat[4] - inmat[7] * inmat[5]) / det;
    outmat[1] = -(inmat[8] * inmat[1] - inmat[7] * inmat[2]) / det;
    outmat[2] = (inmat[5] * inmat[1] - inmat[4] * inmat[2]) / det;
    outmat[3] = -(inmat[8] * inmat[3] - inmat[6] * inmat[5]) / det;
    outmat[4] = (inmat[8] * inmat[0] - inmat[6] * inmat[2]) / det;
    outmat[5] = -(inmat[5] * inmat[0] - inmat[3] * inmat[2]) / det;
    outmat[6] = (inmat[7] * inmat[3] - inmat[6] * inmat[4]) / det;
    outmat[7] = -(inmat[7] * inmat[0] - inmat[6] * inmat[1]) / det;
    outmat[8] = (inmat[4] * inmat[0] - inmat[3] * inmat[1]) / det;
  }
  if (determinant) *determinant = det;
  return 0;
}

inline PetscReal DMatrix_Determinant_4x4_Internal(PetscReal inmat[4 * 4])
{
  return inmat[0 + 0 * 4] * (inmat[1 + 1 * 4] * (inmat[2 + 2 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 2 * 4]) - inmat[1 + 2 * 4] * (inmat[2 + 1 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 1 * 4]) + inmat[1 + 3 * 4] * (inmat[2 + 1 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 1 * 4])) - inmat[0 + 1 * 4] * (inmat[1 + 0 * 4] * (inmat[2 + 2 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 2 * 4]) - inmat[1 + 2 * 4] * (inmat[2 + 0 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 0 * 4]) + inmat[1 + 3 * 4] * (inmat[2 + 0 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 0 * 4])) + inmat[0 + 2 * 4] * (inmat[1 + 0 * 4] * (inmat[2 + 1 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 1 * 4]) - inmat[1 + 1 * 4] * (inmat[2 + 0 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 0 * 4]) + inmat[1 + 3 * 4] * (inmat[2 + 0 * 4] * inmat[3 + 1 * 4] - inmat[2 + 1 * 4] * inmat[3 + 0 * 4])) - inmat[0 + 3 * 4] * (inmat[1 + 0 * 4] * (inmat[2 + 1 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 1 * 4]) - inmat[1 + 1 * 4] * (inmat[2 + 0 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 0 * 4]) + inmat[1 + 2 * 4] * (inmat[2 + 0 * 4] * inmat[3 + 1 * 4] - inmat[2 + 1 * 4] * inmat[3 + 0 * 4]));
}

inline PetscErrorCode DMatrix_Invert_4x4_Internal(PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  PetscReal det = DMatrix_Determinant_4x4_Internal(inmat);
  if (outmat) {
    outmat[0]  = (inmat[5] * inmat[10] * inmat[15] + inmat[6] * inmat[11] * inmat[13] + inmat[7] * inmat[9] * inmat[14] - inmat[5] * inmat[11] * inmat[14] - inmat[6] * inmat[9] * inmat[15] - inmat[7] * inmat[10] * inmat[13]) / det;
    outmat[1]  = (inmat[1] * inmat[11] * inmat[14] + inmat[2] * inmat[9] * inmat[15] + inmat[3] * inmat[10] * inmat[13] - inmat[1] * inmat[10] * inmat[15] - inmat[2] * inmat[11] * inmat[13] - inmat[3] * inmat[9] * inmat[14]) / det;
    outmat[2]  = (inmat[1] * inmat[6] * inmat[15] + inmat[2] * inmat[7] * inmat[13] + inmat[3] * inmat[5] * inmat[14] - inmat[1] * inmat[7] * inmat[14] - inmat[2] * inmat[5] * inmat[15] - inmat[3] * inmat[6] * inmat[13]) / det;
    outmat[3]  = (inmat[1] * inmat[7] * inmat[10] + inmat[2] * inmat[5] * inmat[11] + inmat[3] * inmat[6] * inmat[9] - inmat[1] * inmat[6] * inmat[11] - inmat[2] * inmat[7] * inmat[9] - inmat[3] * inmat[5] * inmat[10]) / det;
    outmat[4]  = (inmat[4] * inmat[11] * inmat[14] + inmat[6] * inmat[8] * inmat[15] + inmat[7] * inmat[10] * inmat[12] - inmat[4] * inmat[10] * inmat[15] - inmat[6] * inmat[11] * inmat[12] - inmat[7] * inmat[8] * inmat[14]) / det;
    outmat[5]  = (inmat[0] * inmat[10] * inmat[15] + inmat[2] * inmat[11] * inmat[12] + inmat[3] * inmat[8] * inmat[14] - inmat[0] * inmat[11] * inmat[14] - inmat[2] * inmat[8] * inmat[15] - inmat[3] * inmat[10] * inmat[12]) / det;
    outmat[6]  = (inmat[0] * inmat[7] * inmat[14] + inmat[2] * inmat[4] * inmat[15] + inmat[3] * inmat[6] * inmat[12] - inmat[0] * inmat[6] * inmat[15] - inmat[2] * inmat[7] * inmat[12] - inmat[3] * inmat[4] * inmat[14]) / det;
    outmat[7]  = (inmat[0] * inmat[6] * inmat[11] + inmat[2] * inmat[7] * inmat[8] + inmat[3] * inmat[4] * inmat[10] - inmat[0] * inmat[7] * inmat[10] - inmat[2] * inmat[4] * inmat[11] - inmat[3] * inmat[6] * inmat[8]) / det;
    outmat[8]  = (inmat[4] * inmat[9] * inmat[15] + inmat[5] * inmat[11] * inmat[12] + inmat[7] * inmat[8] * inmat[13] - inmat[4] * inmat[11] * inmat[13] - inmat[5] * inmat[8] * inmat[15] - inmat[7] * inmat[9] * inmat[12]) / det;
    outmat[9]  = (inmat[0] * inmat[11] * inmat[13] + inmat[1] * inmat[8] * inmat[15] + inmat[3] * inmat[9] * inmat[12] - inmat[0] * inmat[9] * inmat[15] - inmat[1] * inmat[11] * inmat[12] - inmat[3] * inmat[8] * inmat[13]) / det;
    outmat[10] = (inmat[0] * inmat[5] * inmat[15] + inmat[1] * inmat[7] * inmat[12] + inmat[3] * inmat[4] * inmat[13] - inmat[0] * inmat[7] * inmat[13] - inmat[1] * inmat[4] * inmat[15] - inmat[3] * inmat[5] * inmat[12]) / det;
    outmat[11] = (inmat[0] * inmat[7] * inmat[9] + inmat[1] * inmat[4] * inmat[11] + inmat[3] * inmat[5] * inmat[8] - inmat[0] * inmat[5] * inmat[11] - inmat[1] * inmat[7] * inmat[8] - inmat[3] * inmat[4] * inmat[9]) / det;
    outmat[12] = (inmat[4] * inmat[10] * inmat[13] + inmat[5] * inmat[8] * inmat[14] + inmat[6] * inmat[9] * inmat[12] - inmat[4] * inmat[9] * inmat[14] - inmat[5] * inmat[10] * inmat[12] - inmat[6] * inmat[8] * inmat[13]) / det;
    outmat[13] = (inmat[0] * inmat[9] * inmat[14] + inmat[1] * inmat[10] * inmat[12] + inmat[2] * inmat[8] * inmat[13] - inmat[0] * inmat[10] * inmat[13] - inmat[1] * inmat[8] * inmat[14] - inmat[2] * inmat[9] * inmat[12]) / det;
    outmat[14] = (inmat[0] * inmat[6] * inmat[13] + inmat[1] * inmat[4] * inmat[14] + inmat[2] * inmat[5] * inmat[12] - inmat[0] * inmat[5] * inmat[14] - inmat[1] * inmat[6] * inmat[12] - inmat[2] * inmat[4] * inmat[13]) / det;
    outmat[15] = (inmat[0] * inmat[5] * inmat[10] + inmat[1] * inmat[6] * inmat[8] + inmat[2] * inmat[4] * inmat[9] - inmat[0] * inmat[6] * inmat[9] - inmat[1] * inmat[4] * inmat[10] - inmat[2] * inmat[5] * inmat[8]) / det;
  }
  if (determinant) *determinant = det;
  return 0;
}

/*@C
  Compute_Lagrange_Basis_1D_Internal - Evaluate bases and derivatives at quadrature points for a EDGE2 or EDGE3 element.

  The routine is given the coordinates of the vertices of a linear or quadratic edge element.

  The routine evaluates the basis functions associated with each quadrature point provided,
  and their derivatives with respect to X.

  Notes:

  Example Physical Element
.vb
    1-------2        1----3----2
      EDGE2             EDGE3
.ve

  Input Parameters:
+  PetscInt  nverts -          the number of element vertices
.  PetscReal coords[3*nverts] - the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts -            the number of evaluation points (quadrature points)
-  PetscReal quad[3*npts] -    the evaluation points (quadrature points) in the reference space

  Output Parameters:
+  PetscReal phypts[3*npts] -  the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts] -       the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal phi[npts] -       the bases evaluated at the specified quadrature points
.  PetscReal dphidx[npts] -    the derivative of the bases wrt X-direction evaluated at the specified quadrature points
.  jacobian -                  jacobian
.  ijacobian -                 ijacobian
-  volume -                    volume

  Level: advanced

@*/
PetscErrorCode Compute_Lagrange_Basis_1D_Internal(const PetscInt nverts, const PetscReal *coords, const PetscInt npts, const PetscReal *quad, PetscReal *phypts, PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  int i, j;

  PetscFunctionBegin;
  PetscValidPointer(jacobian, 9);
  PetscValidPointer(ijacobian, 10);
  PetscValidPointer(volume, 11);
  if (phypts) PetscCall(PetscArrayzero(phypts, npts * 3));
  if (dphidx) { /* Reset arrays. */
    PetscCall(PetscArrayzero(dphidx, npts * nverts));
  }
  if (nverts == 2) { /* Linear Edge */

    for (j = 0; j < npts; j++) {
      const PetscInt  offset = j * nverts;
      const PetscReal r      = quad[j];

      phi[0 + offset] = (1.0 - r);
      phi[1 + offset] = (r);

      const PetscReal dNi_dxi[2] = {-1.0, 1.0};

      jacobian[0] = ijacobian[0] = volume[0] = 0.0;
      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertices = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        if (phypts) phypts[3 * j + 0] += phi[i + offset] * vertices[0];
      }

      /* invert the jacobian */
      *volume      = jacobian[0];
      ijacobian[0] = 1.0 / jacobian[0];
      jxw[j] *= *volume;

      /*  Divide by element jacobian. */
      for (i = 0; i < nverts; i++) {
        if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[0];
      }
    }
  } else if (nverts == 3) { /* Quadratic Edge */

    for (j = 0; j < npts; j++) {
      const PetscInt  offset = j * nverts;
      const PetscReal r      = quad[j];

      phi[0 + offset] = 1.0 + r * (2.0 * r - 3.0);
      phi[1 + offset] = 4.0 * r * (1.0 - r);
      phi[2 + offset] = r * (2.0 * r - 1.0);

      const PetscReal dNi_dxi[3] = {4 * r - 3.0, 4 * (1.0 - 2.0 * r), 4.0 * r - 1.0};

      jacobian[0] = ijacobian[0] = volume[0] = 0.0;
      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertices = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        if (phypts) phypts[3 * j + 0] += phi[i + offset] * vertices[0];
      }

      /* invert the jacobian */
      *volume      = jacobian[0];
      ijacobian[0] = 1.0 / jacobian[0];
      if (jxw) jxw[j] *= *volume;

      /*  Divide by element jacobian. */
      for (i = 0; i < nverts; i++) {
        if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[0];
      }
    }
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of entity vertices are invalid. Currently only support EDGE2 and EDGE3 basis evaluations in 1-D : %" PetscInt_FMT, nverts);
  PetscFunctionReturn(0);
}

/*@C
  Compute_Lagrange_Basis_2D_Internal - Evaluate bases and derivatives at quadrature points for a QUAD4 or TRI3 element.

  The routine is given the coordinates of the vertices of a quadrangle or triangle.

  The routine evaluates the basis functions associated with each quadrature point provided,
  and their derivatives with respect to X and Y.

  Notes:

  Example Physical Element (QUAD4)
.vb
    4------3        s
    |      |        |
    |      |        |
    |      |        |
    1------2        0-------r
.ve

  Input Parameters:
+  PetscInt  nverts -          the number of element vertices
.  PetscReal coords[3*nverts] - the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts -            the number of evaluation points (quadrature points)
-  PetscReal quad[3*npts] -    the evaluation points (quadrature points) in the reference space

  Output Parameters:
+  PetscReal phypts[3*npts] -  the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts] -       the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal phi[npts] -       the bases evaluated at the specified quadrature points
.  PetscReal dphidx[npts] -    the derivative of the bases wrt X-direction evaluated at the specified quadrature points
.  PetscReal dphidy[npts] -    the derivative of the bases wrt Y-direction evaluated at the specified quadrature points
.  jacobian -                  jacobian
.  ijacobian -                 ijacobian
-  volume -                    volume

  Level: advanced

@*/
PetscErrorCode Compute_Lagrange_Basis_2D_Internal(const PetscInt nverts, const PetscReal *coords, const PetscInt npts, const PetscReal *quad, PetscReal *phypts, PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy, PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  PetscInt i, j, k;

  PetscFunctionBegin;
  PetscValidPointer(jacobian, 10);
  PetscValidPointer(ijacobian, 11);
  PetscValidPointer(volume, 12);
  PetscCall(PetscArrayzero(phi, npts));
  if (phypts) PetscCall(PetscArrayzero(phypts, npts * 3));
  if (dphidx) { /* Reset arrays. */
    PetscCall(PetscArrayzero(dphidx, npts * nverts));
    PetscCall(PetscArrayzero(dphidy, npts * nverts));
  }
  if (nverts == 4) { /* Linear Quadrangle */

    for (j = 0; j < npts; j++) {
      const PetscInt  offset = j * nverts;
      const PetscReal r      = quad[0 + j * 2];
      const PetscReal s      = quad[1 + j * 2];

      phi[0 + offset] = (1.0 - r) * (1.0 - s);
      phi[1 + offset] = r * (1.0 - s);
      phi[2 + offset] = r * s;
      phi[3 + offset] = (1.0 - r) * s;

      const PetscReal dNi_dxi[4]  = {-1.0 + s, 1.0 - s, s, -s};
      const PetscReal dNi_deta[4] = {-1.0 + r, -r, r, 1.0 - r};

      PetscCall(PetscArrayzero(jacobian, 4));
      PetscCall(PetscArrayzero(ijacobian, 4));
      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertices = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        jacobian[2] += dNi_dxi[i] * vertices[1];
        jacobian[1] += dNi_deta[i] * vertices[0];
        jacobian[3] += dNi_deta[i] * vertices[1];
        if (phypts) {
          phypts[3 * j + 0] += phi[i + offset] * vertices[0];
          phypts[3 * j + 1] += phi[i + offset] * vertices[1];
          phypts[3 * j + 2] += phi[i + offset] * vertices[2];
        }
      }

      /* invert the jacobian */
      PetscCall(DMatrix_Invert_2x2_Internal(jacobian, ijacobian, volume));
      PetscCheck(*volume >= 1e-12, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Quadrangular element has zero volume: %g. Degenerate element or invalid connectivity", *volume);

      if (jxw) jxw[j] *= *volume;

      /*  Let us compute the bases derivatives by scaling with inverse jacobian. */
      if (dphidx) {
        for (i = 0; i < nverts; i++) {
          for (k = 0; k < 2; ++k) {
            if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[k * 2 + 0];
            if (dphidy) dphidy[i + offset] += dNi_deta[i] * ijacobian[k * 2 + 1];
          } /* for k=[0..2) */
        }   /* for i=[0..nverts) */
      }     /* if (dphidx) */
    }
  } else if (nverts == 3) { /* Linear triangle */
    const PetscReal x2 = coords[2 * 3 + 0], y2 = coords[2 * 3 + 1];

    PetscCall(PetscArrayzero(jacobian, 4));
    PetscCall(PetscArrayzero(ijacobian, 4));

    /* Jacobian is constant */
    jacobian[0] = (coords[0 * 3 + 0] - x2);
    jacobian[1] = (coords[1 * 3 + 0] - x2);
    jacobian[2] = (coords[0 * 3 + 1] - y2);
    jacobian[3] = (coords[1 * 3 + 1] - y2);

    /* invert the jacobian */
    PetscCall(DMatrix_Invert_2x2_Internal(jacobian, ijacobian, volume));
    PetscCheck(*volume >= 1e-12, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Triangular element has zero volume: %g. Degenerate element or invalid connectivity", (double)*volume);

    const PetscReal Dx[3] = {ijacobian[0], ijacobian[2], -ijacobian[0] - ijacobian[2]};
    const PetscReal Dy[3] = {ijacobian[1], ijacobian[3], -ijacobian[1] - ijacobian[3]};

    for (j = 0; j < npts; j++) {
      const PetscInt  offset = j * nverts;
      const PetscReal r      = quad[0 + j * 2];
      const PetscReal s      = quad[1 + j * 2];

      if (jxw) jxw[j] *= 0.5;

      /* const PetscReal Ni[3]  = { r, s, 1.0 - r - s }; */
      const PetscReal phipts_x = coords[2 * 3 + 0] + jacobian[0] * r + jacobian[1] * s;
      const PetscReal phipts_y = coords[2 * 3 + 1] + jacobian[2] * r + jacobian[3] * s;
      if (phypts) {
        phypts[offset + 0] = phipts_x;
        phypts[offset + 1] = phipts_y;
      }

      /* \phi_0 = (b.y - c.y) x + (b.x - c.x) y + c.x b.y - b.x c.y */
      phi[0 + offset] = (ijacobian[0] * (phipts_x - x2) + ijacobian[1] * (phipts_y - y2));
      /* \phi_1 = (c.y - a.y) x + (a.x - c.x) y + c.x a.y - a.x c.y */
      phi[1 + offset] = (ijacobian[2] * (phipts_x - x2) + ijacobian[3] * (phipts_y - y2));
      phi[2 + offset] = 1.0 - phi[0 + offset] - phi[1 + offset];

      if (dphidx) {
        dphidx[0 + offset] = Dx[0];
        dphidx[1 + offset] = Dx[1];
        dphidx[2 + offset] = Dx[2];
      }

      if (dphidy) {
        dphidy[0 + offset] = Dy[0];
        dphidy[1 + offset] = Dy[1];
        dphidy[2 + offset] = Dy[2];
      }
    }
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of entity vertices are invalid. Currently only support QUAD4 and TRI3 basis evaluations in 2-D : %" PetscInt_FMT, nverts);
  PetscFunctionReturn(0);
}

/*@C
  Compute_Lagrange_Basis_3D_Internal - Evaluate bases and derivatives at quadrature points for a HEX8 or TET4 element.

  The routine is given the coordinates of the vertices of a hexahedra or tetrahedra.

  The routine evaluates the basis functions associated with each quadrature point provided,
  and their derivatives with respect to X, Y and Z.

  Notes:

  Example Physical Element (HEX8)
.vb
      8------7
     /|     /|        t  s
    5------6 |        | /
    | |    | |        |/
    | 4----|-3        0-------r
    |/     |/
    1------2
.ve

  Input Parameters:
+  PetscInt  nverts -          the number of element vertices
.  PetscReal coords[3*nverts] - the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts -            the number of evaluation points (quadrature points)
-  PetscReal quad[3*npts] -    the evaluation points (quadrature points) in the reference space

  Output Parameters:
+  PetscReal phypts[3*npts] -  the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts] -       the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal phi[npts] -       the bases evaluated at the specified quadrature points
.  PetscReal dphidx[npts] -    the derivative of the bases wrt X-direction evaluated at the specified quadrature points
.  PetscReal dphidy[npts] -    the derivative of the bases wrt Y-direction evaluated at the specified quadrature points
.  PetscReal dphidz[npts] -    the derivative of the bases wrt Z-direction evaluated at the specified quadrature points
.  jacobian -                  jacobian
.  ijacobian -                 ijacobian
-  volume -                    volume

  Level: advanced

@*/
PetscErrorCode Compute_Lagrange_Basis_3D_Internal(const PetscInt nverts, const PetscReal *coords, const PetscInt npts, const PetscReal *quad, PetscReal *phypts, PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy, PetscReal *dphidz, PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  PetscInt i, j, k;

  PetscFunctionBegin;
  PetscValidPointer(jacobian, 11);
  PetscValidPointer(ijacobian, 12);
  PetscValidPointer(volume, 13);

  PetscCall(PetscArrayzero(phi, npts));
  if (phypts) PetscCall(PetscArrayzero(phypts, npts * 3));
  if (dphidx) {
    PetscCall(PetscArrayzero(dphidx, npts * nverts));
    PetscCall(PetscArrayzero(dphidy, npts * nverts));
    PetscCall(PetscArrayzero(dphidz, npts * nverts));
  }

  if (nverts == 8) { /* Linear Hexahedra */

    for (j = 0; j < npts; j++) {
      const PetscInt   offset = j * nverts;
      const PetscReal &r      = quad[j * 3 + 0];
      const PetscReal &s      = quad[j * 3 + 1];
      const PetscReal &t      = quad[j * 3 + 2];

      phi[offset + 0] = (1.0 - r) * (1.0 - s) * (1.0 - t); /* 0,0,0 */
      phi[offset + 1] = (r) * (1.0 - s) * (1.0 - t);       /* 1,0,0 */
      phi[offset + 2] = (r) * (s) * (1.0 - t);             /* 1,1,0 */
      phi[offset + 3] = (1.0 - r) * (s) * (1.0 - t);       /* 0,1,0 */
      phi[offset + 4] = (1.0 - r) * (1.0 - s) * (t);       /* 0,0,1 */
      phi[offset + 5] = (r) * (1.0 - s) * (t);             /* 1,0,1 */
      phi[offset + 6] = (r) * (s) * (t);                   /* 1,1,1 */
      phi[offset + 7] = (1.0 - r) * (s) * (t);             /* 0,1,1 */

      const PetscReal dNi_dxi[8] = {-(1.0 - s) * (1.0 - t), (1.0 - s) * (1.0 - t), (s) * (1.0 - t), -(s) * (1.0 - t), -(1.0 - s) * (t), (1.0 - s) * (t), (s) * (t), -(s) * (t)};

      const PetscReal dNi_deta[8] = {-(1.0 - r) * (1.0 - t), -(r) * (1.0 - t), (r) * (1.0 - t), (1.0 - r) * (1.0 - t), -(1.0 - r) * (t), -(r) * (t), (r) * (t), (1.0 - r) * (t)};

      const PetscReal dNi_dzeta[8] = {-(1.0 - r) * (1.0 - s), -(r) * (1.0 - s), -(r) * (s), -(1.0 - r) * (s), (1.0 - r) * (1.0 - s), (r) * (1.0 - s), (r) * (s), (1.0 - r) * (s)};

      PetscCall(PetscArrayzero(jacobian, 9));
      PetscCall(PetscArrayzero(ijacobian, 9));
      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertex = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertex[0];
        jacobian[3] += dNi_dxi[i] * vertex[1];
        jacobian[6] += dNi_dxi[i] * vertex[2];
        jacobian[1] += dNi_deta[i] * vertex[0];
        jacobian[4] += dNi_deta[i] * vertex[1];
        jacobian[7] += dNi_deta[i] * vertex[2];
        jacobian[2] += dNi_dzeta[i] * vertex[0];
        jacobian[5] += dNi_dzeta[i] * vertex[1];
        jacobian[8] += dNi_dzeta[i] * vertex[2];
        if (phypts) {
          phypts[3 * j + 0] += phi[i + offset] * vertex[0];
          phypts[3 * j + 1] += phi[i + offset] * vertex[1];
          phypts[3 * j + 2] += phi[i + offset] * vertex[2];
        }
      }

      /* invert the jacobian */
      PetscCall(DMatrix_Invert_3x3_Internal(jacobian, ijacobian, volume));
      PetscCheck(*volume >= 1e-8, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Hexahedral element has zero volume: %g. Degenerate element or invalid connectivity", *volume);

      if (jxw) jxw[j] *= (*volume);

      /*  Divide by element jacobian. */
      for (i = 0; i < nverts; ++i) {
        for (k = 0; k < 3; ++k) {
          if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[0 * 3 + k];
          if (dphidy) dphidy[i + offset] += dNi_deta[i] * ijacobian[1 * 3 + k];
          if (dphidz) dphidz[i + offset] += dNi_dzeta[i] * ijacobian[2 * 3 + k];
        }
      }
    }
  } else if (nverts == 4) { /* Linear Tetrahedra */
    PetscReal       Dx[4] = {0, 0, 0, 0}, Dy[4] = {0, 0, 0, 0}, Dz[4] = {0, 0, 0, 0};
    const PetscReal x0 = coords[/*0 * 3 +*/ 0], y0 = coords[/*0 * 3 +*/ 1], z0 = coords[/*0 * 3 +*/ 2];

    PetscCall(PetscArrayzero(jacobian, 9));
    PetscCall(PetscArrayzero(ijacobian, 9));

    /* compute the jacobian */
    jacobian[0] = coords[1 * 3 + 0] - x0;
    jacobian[1] = coords[2 * 3 + 0] - x0;
    jacobian[2] = coords[3 * 3 + 0] - x0;
    jacobian[3] = coords[1 * 3 + 1] - y0;
    jacobian[4] = coords[2 * 3 + 1] - y0;
    jacobian[5] = coords[3 * 3 + 1] - y0;
    jacobian[6] = coords[1 * 3 + 2] - z0;
    jacobian[7] = coords[2 * 3 + 2] - z0;
    jacobian[8] = coords[3 * 3 + 2] - z0;

    /* invert the jacobian */
    PetscCall(DMatrix_Invert_3x3_Internal(jacobian, ijacobian, volume));
    PetscCheck(*volume >= 1e-8, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Tetrahedral element has zero volume: %g. Degenerate element or invalid connectivity", (double)*volume);

    if (dphidx) {
      Dx[0] = (coords[1 + 2 * 3] * (coords[2 + 1 * 3] - coords[2 + 3 * 3]) - coords[1 + 1 * 3] * (coords[2 + 2 * 3] - coords[2 + 3 * 3]) - coords[1 + 3 * 3] * (coords[2 + 1 * 3] - coords[2 + 2 * 3])) / *volume;
      Dx[1] = -(coords[1 + 2 * 3] * (coords[2 + 0 * 3] - coords[2 + 3 * 3]) - coords[1 + 0 * 3] * (coords[2 + 2 * 3] - coords[2 + 3 * 3]) - coords[1 + 3 * 3] * (coords[2 + 0 * 3] - coords[2 + 2 * 3])) / *volume;
      Dx[2] = (coords[1 + 1 * 3] * (coords[2 + 0 * 3] - coords[2 + 3 * 3]) - coords[1 + 0 * 3] * (coords[2 + 1 * 3] - coords[2 + 3 * 3]) - coords[1 + 3 * 3] * (coords[2 + 0 * 3] - coords[2 + 1 * 3])) / *volume;
      Dx[3] = -(Dx[0] + Dx[1] + Dx[2]);
      Dy[0] = (coords[0 + 1 * 3] * (coords[2 + 2 * 3] - coords[2 + 3 * 3]) - coords[0 + 2 * 3] * (coords[2 + 1 * 3] - coords[2 + 3 * 3]) + coords[0 + 3 * 3] * (coords[2 + 1 * 3] - coords[2 + 2 * 3])) / *volume;
      Dy[1] = -(coords[0 + 0 * 3] * (coords[2 + 2 * 3] - coords[2 + 3 * 3]) - coords[0 + 2 * 3] * (coords[2 + 0 * 3] - coords[2 + 3 * 3]) + coords[0 + 3 * 3] * (coords[2 + 0 * 3] - coords[2 + 2 * 3])) / *volume;
      Dy[2] = (coords[0 + 0 * 3] * (coords[2 + 1 * 3] - coords[2 + 3 * 3]) - coords[0 + 1 * 3] * (coords[2 + 0 * 3] - coords[2 + 3 * 3]) + coords[0 + 3 * 3] * (coords[2 + 0 * 3] - coords[2 + 1 * 3])) / *volume;
      Dy[3] = -(Dy[0] + Dy[1] + Dy[2]);
      Dz[0] = (coords[0 + 1 * 3] * (coords[1 + 3 * 3] - coords[1 + 2 * 3]) - coords[0 + 2 * 3] * (coords[1 + 3 * 3] - coords[1 + 1 * 3]) + coords[0 + 3 * 3] * (coords[1 + 2 * 3] - coords[1 + 1 * 3])) / *volume;
      Dz[1] = -(coords[0 + 0 * 3] * (coords[1 + 3 * 3] - coords[1 + 2 * 3]) + coords[0 + 2 * 3] * (coords[1 + 0 * 3] - coords[1 + 3 * 3]) - coords[0 + 3 * 3] * (coords[1 + 0 * 3] - coords[1 + 2 * 3])) / *volume;
      Dz[2] = (coords[0 + 0 * 3] * (coords[1 + 3 * 3] - coords[1 + 1 * 3]) + coords[0 + 1 * 3] * (coords[1 + 0 * 3] - coords[1 + 3 * 3]) - coords[0 + 3 * 3] * (coords[1 + 0 * 3] - coords[1 + 1 * 3])) / *volume;
      Dz[3] = -(Dz[0] + Dz[1] + Dz[2]);
    }

    for (j = 0; j < npts; j++) {
      const PetscInt   offset = j * nverts;
      const PetscReal &r      = quad[j * 3 + 0];
      const PetscReal &s      = quad[j * 3 + 1];
      const PetscReal &t      = quad[j * 3 + 2];

      if (jxw) jxw[j] *= *volume;

      phi[offset + 0] = 1.0 - r - s - t;
      phi[offset + 1] = r;
      phi[offset + 2] = s;
      phi[offset + 3] = t;

      if (phypts) {
        for (i = 0; i < nverts; ++i) {
          const PetscScalar *vertices = coords + i * 3;
          phypts[3 * j + 0] += phi[i + offset] * vertices[0];
          phypts[3 * j + 1] += phi[i + offset] * vertices[1];
          phypts[3 * j + 2] += phi[i + offset] * vertices[2];
        }
      }

      /* Now set the derivatives */
      if (dphidx) {
        dphidx[0 + offset] = Dx[0];
        dphidx[1 + offset] = Dx[1];
        dphidx[2 + offset] = Dx[2];
        dphidx[3 + offset] = Dx[3];

        dphidy[0 + offset] = Dy[0];
        dphidy[1 + offset] = Dy[1];
        dphidy[2 + offset] = Dy[2];
        dphidy[3 + offset] = Dy[3];

        dphidz[0 + offset] = Dz[0];
        dphidz[1 + offset] = Dz[1];
        dphidz[2 + offset] = Dz[2];
        dphidz[3 + offset] = Dz[3];
      }

    } /* Tetrahedra -- ends */
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of entity vertices are invalid. Currently only support HEX8 and TET4 basis evaluations in 3-D : %" PetscInt_FMT, nverts);
  PetscFunctionReturn(0);
}

/*@C
  DMMoabFEMComputeBasis - Evaluate bases and derivatives at quadrature points for a linear EDGE/QUAD/TRI/HEX/TET element.

  The routine takes the coordinates of the vertices of an element and computes basis functions associated with
  each quadrature point provided, and their derivatives with respect to X, Y and Z as appropriate.

  Input Parameters:
+  PetscInt  nverts -           the number of element vertices
.  PetscReal coords[3*nverts] - the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts -             the number of evaluation points (quadrature points)
-  PetscReal quad[3*npts] -     the evaluation points (quadrature points) in the reference space

  Output Parameters:
+  PetscReal phypts[3*npts] -   the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts] -        the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal fe_basis[npts] -   the bases values evaluated at the specified quadrature points
-  PetscReal fe_basis_derivatives[dim][npts] - the derivative of the bases wrt (X,Y,Z)-directions (depending on the dimension) evaluated at the specified quadrature points

  Level: advanced

@*/
PetscErrorCode DMMoabFEMComputeBasis(const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscQuadrature quadrature, PetscReal *phypts, PetscReal *jacobian_quadrature_weight_product, PetscReal *fe_basis, PetscReal **fe_basis_derivatives)
{
  PetscInt         npoints, idim;
  bool             compute_der;
  const PetscReal *quadpts, *quadwts;
  PetscReal        jacobian[9], ijacobian[9], volume;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidHeaderSpecific(quadrature, PETSCQUADRATURE_CLASSID, 4);
  PetscValidPointer(fe_basis, 7);
  compute_der = (fe_basis_derivatives != NULL);

  /* Get the quadrature points and weights for the given quadrature rule */
  PetscCall(PetscQuadratureGetData(quadrature, &idim, NULL, &npoints, &quadpts, &quadwts));
  PetscCheck(idim == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Dimension mismatch: provided (%" PetscInt_FMT ") vs quadrature (%" PetscInt_FMT ")", idim, dim);
  if (jacobian_quadrature_weight_product) PetscCall(PetscArraycpy(jacobian_quadrature_weight_product, quadwts, npoints));

  switch (dim) {
  case 1:
    PetscCall(Compute_Lagrange_Basis_1D_Internal(nverts, coordinates, npoints, quadpts, phypts, jacobian_quadrature_weight_product, fe_basis, (compute_der ? fe_basis_derivatives[0] : NULL), jacobian, ijacobian, &volume));
    break;
  case 2:
    PetscCall(Compute_Lagrange_Basis_2D_Internal(nverts, coordinates, npoints, quadpts, phypts, jacobian_quadrature_weight_product, fe_basis, (compute_der ? fe_basis_derivatives[0] : NULL), (compute_der ? fe_basis_derivatives[1] : NULL), jacobian, ijacobian, &volume));
    break;
  case 3:
    PetscCall(Compute_Lagrange_Basis_3D_Internal(nverts, coordinates, npoints, quadpts, phypts, jacobian_quadrature_weight_product, fe_basis, (compute_der ? fe_basis_derivatives[0] : NULL), (compute_der ? fe_basis_derivatives[1] : NULL), (compute_der ? fe_basis_derivatives[2] : NULL), jacobian, ijacobian, &volume));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension; should be in [1,3] : %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMMoabFEMCreateQuadratureDefault - Create default quadrature rules for integration over an element with a given
  dimension and polynomial order (deciphered from number of element vertices).

  Input Parameters:

+  PetscInt  dim   -   the element dimension (1=EDGE, 2=QUAD/TRI, 3=HEX/TET)
-  PetscInt nverts -   the number of vertices in the physical element

  Output Parameter:
.  PetscQuadrature quadrature -  the quadrature object with default settings to integrate polynomials defined over the element

  Level: advanced

@*/
PetscErrorCode DMMoabFEMCreateQuadratureDefault(const PetscInt dim, const PetscInt nverts, PetscQuadrature *quadrature)
{
  PetscReal *w, *x;
  PetscInt   nc = 1;

  PetscFunctionBegin;
  /* Create an appropriate quadrature rule to sample basis */
  switch (dim) {
  case 1:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    PetscCall(PetscDTStroudConicalQuadrature(1, nc, nverts, 0, 1.0, quadrature));
    break;
  case 2:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    if (nverts == 3) {
      const PetscInt order   = 2;
      const PetscInt npoints = (order == 2 ? 3 : 6);
      PetscCall(PetscMalloc2(npoints * 2, &x, npoints, &w));
      if (npoints == 3) {
        x[0] = x[1] = x[2] = x[5] = 1.0 / 6.0;
        x[3] = x[4] = 2.0 / 3.0;
        w[0] = w[1] = w[2] = 1.0 / 3.0;
      } else if (npoints == 6) {
        x[0] = x[1] = x[2] = 0.44594849091597;
        x[3] = x[4] = 0.10810301816807;
        x[5]        = 0.44594849091597;
        x[6] = x[7] = x[8] = 0.09157621350977;
        x[9] = x[10] = 0.81684757298046;
        x[11]        = 0.09157621350977;
        w[0] = w[1] = w[2] = 0.22338158967801;
        w[3] = w[4] = w[5] = 0.10995174365532;
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Triangle quadrature rules for points 3 and 6 supported; npoints : %" PetscInt_FMT, npoints);
      PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, quadrature));
      PetscCall(PetscQuadratureSetOrder(*quadrature, order));
      PetscCall(PetscQuadratureSetData(*quadrature, dim, nc, npoints, x, w));
      /* PetscCall(PetscDTStroudConicalQuadrature(dim, nc, nverts, 0.0, 1.0, quadrature)); */
    } else PetscCall(PetscDTGaussTensorQuadrature(dim, nc, nverts, 0.0, 1.0, quadrature));
    break;
  case 3:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    if (nverts == 4) {
      PetscInt       order;
      const PetscInt npoints = 4; // Choose between 4 and 10
      PetscCall(PetscMalloc2(npoints * 3, &x, npoints, &w));
      if (npoints == 4) { /*  KEAST1, K1,  N=4, O=4 */
        const PetscReal x_4[12] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
                                   0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105};
        PetscCall(PetscArraycpy(x, x_4, 12));

        w[0] = w[1] = w[2] = w[3] = 1.0 / 24.0;
        order                     = 4;
      } else if (npoints == 10) { /*  KEAST3, K3  N=10, O=10 */
        const PetscReal x_10[30] = {0.5684305841968444, 0.1438564719343852, 0.1438564719343852, 0.1438564719343852, 0.1438564719343852, 0.1438564719343852, 0.1438564719343852, 0.1438564719343852, 0.5684305841968444, 0.1438564719343852,
                                    0.5684305841968444, 0.1438564719343852, 0.0000000000000000, 0.5000000000000000, 0.5000000000000000, 0.5000000000000000, 0.0000000000000000, 0.5000000000000000, 0.5000000000000000, 0.5000000000000000,
                                    0.0000000000000000, 0.5000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.5000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.5000000000000000};
        PetscCall(PetscArraycpy(x, x_10, 30));

        w[0] = w[1] = w[2] = w[3] = 0.2177650698804054;
        w[4] = w[5] = w[6] = w[7] = w[8] = w[9] = 0.0214899534130631;
        order                                   = 10;
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Tetrahedral quadrature rules for points 4 and 10 supported; npoints : %" PetscInt_FMT, npoints);
      PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, quadrature));
      PetscCall(PetscQuadratureSetOrder(*quadrature, order));
      PetscCall(PetscQuadratureSetData(*quadrature, dim, nc, npoints, x, w));
      /* PetscCall(PetscDTStroudConicalQuadrature(dim, nc, nverts, 0.0, 1.0, quadrature)); */
    } else PetscCall(PetscDTGaussTensorQuadrature(dim, nc, nverts, 0.0, 1.0, quadrature));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension; should be in [1,3] : %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(0);
}

/* Compute Jacobians */
PetscErrorCode ComputeJacobian_Internal(const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscReal *quad, PetscReal *phypts, PetscReal *jacobian, PetscReal *ijacobian, PetscReal *dvolume)
{
  PetscInt  i;
  PetscReal volume = 1.0;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidPointer(quad, 4);
  PetscValidPointer(jacobian, 5);
  PetscCall(PetscArrayzero(jacobian, dim * dim));
  if (ijacobian) PetscCall(PetscArrayzero(ijacobian, dim * dim));
  if (phypts) PetscCall(PetscArrayzero(phypts, /*npts=1 * */ 3));

  if (dim == 1) {
    const PetscReal &r = quad[0];
    if (nverts == 2) { /* Linear Edge */
      const PetscReal dNi_dxi[2] = {-1.0, 1.0};

      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertices = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
      }
    } else if (nverts == 3) { /* Quadratic Edge */
      const PetscReal dNi_dxi[3] = {4 * r - 3.0, 4 * (1.0 - 2.0 * r), 4.0 * r - 1.0};

      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertices = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
      }
    } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of 1-D entity vertices are invalid. Currently only support EDGE2 and EDGE3 basis evaluations in 1-D : %" PetscInt_FMT, nverts);

    if (ijacobian) {
      /* invert the jacobian */
      ijacobian[0] = 1.0 / jacobian[0];
    }
  } else if (dim == 2) {
    if (nverts == 4) { /* Linear Quadrangle */
      const PetscReal &r = quad[0];
      const PetscReal &s = quad[1];

      const PetscReal dNi_dxi[4]  = {-1.0 + s, 1.0 - s, s, -s};
      const PetscReal dNi_deta[4] = {-1.0 + r, -r, r, 1.0 - r};

      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertices = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        jacobian[2] += dNi_dxi[i] * vertices[1];
        jacobian[1] += dNi_deta[i] * vertices[0];
        jacobian[3] += dNi_deta[i] * vertices[1];
      }
    } else if (nverts == 3) { /* Linear triangle */
      const PetscReal x2 = coordinates[2 * 3 + 0], y2 = coordinates[2 * 3 + 1];

      /* Jacobian is constant */
      jacobian[0] = (coordinates[0 * 3 + 0] - x2);
      jacobian[1] = (coordinates[1 * 3 + 0] - x2);
      jacobian[2] = (coordinates[0 * 3 + 1] - y2);
      jacobian[3] = (coordinates[1 * 3 + 1] - y2);
    } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of 2-D entity vertices are invalid. Currently only support QUAD4 and TRI3 basis evaluations in 2-D : %" PetscInt_FMT, nverts);

    /* invert the jacobian */
    if (ijacobian) PetscCall(DMatrix_Invert_2x2_Internal(jacobian, ijacobian, &volume));
  } else {
    if (nverts == 8) { /* Linear Hexahedra */
      const PetscReal &r          = quad[0];
      const PetscReal &s          = quad[1];
      const PetscReal &t          = quad[2];
      const PetscReal  dNi_dxi[8] = {-(1.0 - s) * (1.0 - t), (1.0 - s) * (1.0 - t), (s) * (1.0 - t), -(s) * (1.0 - t), -(1.0 - s) * (t), (1.0 - s) * (t), (s) * (t), -(s) * (t)};

      const PetscReal dNi_deta[8] = {-(1.0 - r) * (1.0 - t), -(r) * (1.0 - t), (r) * (1.0 - t), (1.0 - r) * (1.0 - t), -(1.0 - r) * (t), -(r) * (t), (r) * (t), (1.0 - r) * (t)};

      const PetscReal dNi_dzeta[8] = {-(1.0 - r) * (1.0 - s), -(r) * (1.0 - s), -(r) * (s), -(1.0 - r) * (s), (1.0 - r) * (1.0 - s), (r) * (1.0 - s), (r) * (s), (1.0 - r) * (s)};

      for (i = 0; i < nverts; ++i) {
        const PetscReal *vertex = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i] * vertex[0];
        jacobian[3] += dNi_dxi[i] * vertex[1];
        jacobian[6] += dNi_dxi[i] * vertex[2];
        jacobian[1] += dNi_deta[i] * vertex[0];
        jacobian[4] += dNi_deta[i] * vertex[1];
        jacobian[7] += dNi_deta[i] * vertex[2];
        jacobian[2] += dNi_dzeta[i] * vertex[0];
        jacobian[5] += dNi_dzeta[i] * vertex[1];
        jacobian[8] += dNi_dzeta[i] * vertex[2];
      }
    } else if (nverts == 4) { /* Linear Tetrahedra */
      const PetscReal x0 = coordinates[/*0 * 3 +*/ 0], y0 = coordinates[/*0 * 3 +*/ 1], z0 = coordinates[/*0 * 3 +*/ 2];

      /* compute the jacobian */
      jacobian[0] = coordinates[1 * 3 + 0] - x0;
      jacobian[1] = coordinates[2 * 3 + 0] - x0;
      jacobian[2] = coordinates[3 * 3 + 0] - x0;
      jacobian[3] = coordinates[1 * 3 + 1] - y0;
      jacobian[4] = coordinates[2 * 3 + 1] - y0;
      jacobian[5] = coordinates[3 * 3 + 1] - y0;
      jacobian[6] = coordinates[1 * 3 + 2] - z0;
      jacobian[7] = coordinates[2 * 3 + 2] - z0;
      jacobian[8] = coordinates[3 * 3 + 2] - z0;
    } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of 3-D entity vertices are invalid. Currently only support HEX8 and TET4 basis evaluations in 3-D : %" PetscInt_FMT, nverts);

    if (ijacobian) {
      /* invert the jacobian */
      PetscCall(DMatrix_Invert_3x3_Internal(jacobian, ijacobian, &volume));
    }
  }
  PetscCheck(volume >= 1e-12, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Element has zero volume: %g. Degenerate element or invalid connectivity", volume);
  if (dvolume) *dvolume = volume;
  PetscFunctionReturn(0);
}

PetscErrorCode FEMComputeBasis_JandF(const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscReal *quadrature, PetscReal *phypts, PetscReal *phibasis, PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  PetscFunctionBegin;
  switch (dim) {
  case 1:
    PetscCall(Compute_Lagrange_Basis_1D_Internal(nverts, coordinates, 1, quadrature, phypts, NULL, phibasis, NULL, jacobian, ijacobian, volume));
    break;
  case 2:
    PetscCall(Compute_Lagrange_Basis_2D_Internal(nverts, coordinates, 1, quadrature, phypts, NULL, phibasis, NULL, NULL, jacobian, ijacobian, volume));
    break;
  case 3:
    PetscCall(Compute_Lagrange_Basis_3D_Internal(nverts, coordinates, 1, quadrature, phypts, NULL, phibasis, NULL, NULL, NULL, jacobian, ijacobian, volume));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension; should be in [1,3] : %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMMoabPToRMapping - Compute the mapping from the physical coordinate system for a given element to the
  canonical reference element. In addition to finding the inverse mapping evaluation through Newton iteration,
  the basis function at the parametric point is also evaluated optionally.

  Input Parameters:
+  PetscInt  dim -         the element dimension (1=EDGE, 2=QUAD/TRI, 3=HEX/TET)
.  PetscInt nverts -       the number of vertices in the physical element
.  PetscReal coordinates - the coordinates of vertices in the physical element
-  PetscReal[3] xphy -     the coordinates of physical point for which natural coordinates (in reference frame) are sought

  Output Parameters:
+  PetscReal[3] natparam - the natural coordinates (in reference frame) corresponding to xphy
-  PetscReal[nverts] phi - the basis functions evaluated at the natural coordinates (natparam)

  Level: advanced

@*/
PetscErrorCode DMMoabPToRMapping(const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscReal *xphy, PetscReal *natparam, PetscReal *phi)
{
  /* Perform inverse evaluation for the mapping with use of Newton Raphson iteration */
  const PetscReal tol            = 1.0e-10;
  const PetscInt  max_iterations = 10;
  const PetscReal error_tol_sqr  = tol * tol;
  PetscReal       phibasis[8], jacobian[9], ijacobian[9], volume;
  PetscReal       phypts[3] = {0.0, 0.0, 0.0};
  PetscReal       delta[3]  = {0.0, 0.0, 0.0};
  PetscInt        iters     = 0;
  PetscReal       error     = 1.0;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidPointer(xphy, 4);
  PetscValidPointer(natparam, 5);

  PetscCall(PetscArrayzero(jacobian, dim * dim));
  PetscCall(PetscArrayzero(ijacobian, dim * dim));
  PetscCall(PetscArrayzero(phibasis, nverts));

  /* zero initial guess */
  natparam[0] = natparam[1] = natparam[2] = 0.0;

  /* Compute delta = evaluate( xi) - x */
  PetscCall(FEMComputeBasis_JandF(dim, nverts, coordinates, natparam, phypts, phibasis, jacobian, ijacobian, &volume));

  error = 0.0;
  switch (dim) {
  case 3:
    delta[2] = phypts[2] - xphy[2];
    error += (delta[2] * delta[2]);
  case 2:
    delta[1] = phypts[1] - xphy[1];
    error += (delta[1] * delta[1]);
  case 1:
    delta[0] = phypts[0] - xphy[0];
    error += (delta[0] * delta[0]);
    break;
  }

  while (error > error_tol_sqr) {
    PetscCheck(++iters <= max_iterations, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Maximum Newton iterations (10) reached. Current point in reference space : (%g, %g, %g)", natparam[0], natparam[1], natparam[2]);

    /* Compute natparam -= J.inverse() * delta */
    switch (dim) {
    case 1:
      natparam[0] -= ijacobian[0] * delta[0];
      break;
    case 2:
      natparam[0] -= ijacobian[0] * delta[0] + ijacobian[1] * delta[1];
      natparam[1] -= ijacobian[2] * delta[0] + ijacobian[3] * delta[1];
      break;
    case 3:
      natparam[0] -= ijacobian[0] * delta[0] + ijacobian[1] * delta[1] + ijacobian[2] * delta[2];
      natparam[1] -= ijacobian[3] * delta[0] + ijacobian[4] * delta[1] + ijacobian[5] * delta[2];
      natparam[2] -= ijacobian[6] * delta[0] + ijacobian[7] * delta[1] + ijacobian[8] * delta[2];
      break;
    }

    /* Compute delta = evaluate( xi) - x */
    PetscCall(FEMComputeBasis_JandF(dim, nverts, coordinates, natparam, phypts, phibasis, jacobian, ijacobian, &volume));

    error = 0.0;
    switch (dim) {
    case 3:
      delta[2] = phypts[2] - xphy[2];
      error += (delta[2] * delta[2]);
    case 2:
      delta[1] = phypts[1] - xphy[1];
      error += (delta[1] * delta[1]);
    case 1:
      delta[0] = phypts[0] - xphy[0];
      error += (delta[0] * delta[0]);
      break;
    }
  }
  if (phi) PetscCall(PetscArraycpy(phi, phibasis, nverts));
  PetscFunctionReturn(0);
}
