
#include <petscconf.h>
#include <petscdt.h>            /*I "petscdt.h" I*/
#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/

/* Utility functions */
static inline PetscReal DMatrix_Determinant_2x2_Internal ( const PetscReal inmat[2 * 2] )
{
  return  inmat[0] * inmat[3] - inmat[1] * inmat[2];
}

#undef __FUNCT__
#define __FUNCT__ "DMatrix_Invert_2x2_Internal"
static inline PetscErrorCode DMatrix_Invert_2x2_Internal (const PetscReal *inmat, PetscReal *outmat, PetscReal *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_POINTER, "Invalid input matrix specified for 2x2 inversion.");
  PetscReal det = DMatrix_Determinant_2x2_Internal(inmat);
  if (outmat) {
    outmat[0] = inmat[3] / det;
    outmat[1] = -inmat[1] / det;
    outmat[2] = -inmat[2] / det;
    outmat[3] = inmat[0] / det;
  }
  if (determinant) *determinant = det;
  PetscFunctionReturn(0);
}

static inline PetscReal DMatrix_Determinant_3x3_Internal ( const PetscReal inmat[3 * 3] )
{
  return   inmat[0] * (inmat[8] * inmat[4] - inmat[7] * inmat[5])
           - inmat[3] * (inmat[8] * inmat[1] - inmat[7] * inmat[2])
           + inmat[6] * (inmat[5] * inmat[1] - inmat[4] * inmat[2]);
}

#undef __FUNCT__
#define __FUNCT__ "DMatrix_Invert_3x3_Internal"
static inline PetscErrorCode DMatrix_Invert_3x3_Internal (const PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_POINTER, "Invalid input matrix specified for 3x3 inversion.");
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
  PetscFunctionReturn(0);
}

inline PetscReal DMatrix_Determinant_4x4_Internal ( PetscReal inmat[4 * 4] )
{
  return
    inmat[0 + 0 * 4] * (
      inmat[1 + 1 * 4] * ( inmat[2 + 2 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 2 * 4] )
      - inmat[1 + 2 * 4] * ( inmat[2 + 1 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 1 * 4] )
      + inmat[1 + 3 * 4] * ( inmat[2 + 1 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 1 * 4] ) )
    - inmat[0 + 1 * 4] * (
      inmat[1 + 0 * 4] * ( inmat[2 + 2 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 2 * 4] )
      - inmat[1 + 2 * 4] * ( inmat[2 + 0 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 0 * 4] )
      + inmat[1 + 3 * 4] * ( inmat[2 + 0 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 0 * 4] ) )
    + inmat[0 + 2 * 4] * (
      inmat[1 + 0 * 4] * ( inmat[2 + 1 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 1 * 4] )
      - inmat[1 + 1 * 4] * ( inmat[2 + 0 * 4] * inmat[3 + 3 * 4] - inmat[2 + 3 * 4] * inmat[3 + 0 * 4] )
      + inmat[1 + 3 * 4] * ( inmat[2 + 0 * 4] * inmat[3 + 1 * 4] - inmat[2 + 1 * 4] * inmat[3 + 0 * 4] ) )
    - inmat[0 + 3 * 4] * (
      inmat[1 + 0 * 4] * ( inmat[2 + 1 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 1 * 4] )
      - inmat[1 + 1 * 4] * ( inmat[2 + 0 * 4] * inmat[3 + 2 * 4] - inmat[2 + 2 * 4] * inmat[3 + 0 * 4] )
      + inmat[1 + 2 * 4] * ( inmat[2 + 0 * 4] * inmat[3 + 1 * 4] - inmat[2 + 1 * 4] * inmat[3 + 0 * 4] ) );
}

#undef __FUNCT__
#define __FUNCT__ "DMatrix_Invert_4x4_Internal"
inline PetscErrorCode DMatrix_Invert_4x4_Internal (PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_POINTER, "Invalid input matrix specified for 4x4 inversion.");
  PetscReal det = DMatrix_Determinant_4x4_Internal(inmat);
  if (outmat) {
    outmat[0] =  (inmat[5] * inmat[10] * inmat[15] + inmat[6] * inmat[11] * inmat[13] + inmat[7] * inmat[9] * inmat[14] - inmat[5] * inmat[11] * inmat[14] - inmat[6] * inmat[9] * inmat[15] - inmat[7] * inmat[10] * inmat[13]) / det;
    outmat[1] =  (inmat[1] * inmat[11] * inmat[14] + inmat[2] * inmat[9] * inmat[15] + inmat[3] * inmat[10] * inmat[13] - inmat[1] * inmat[10] * inmat[15] - inmat[2] * inmat[11] * inmat[13] - inmat[3] * inmat[9] * inmat[14]) / det;
    outmat[2] =  (inmat[1] * inmat[6] * inmat[15] + inmat[2] * inmat[7] * inmat[13] + inmat[3] * inmat[5] * inmat[14] - inmat[1] * inmat[7] * inmat[14] - inmat[2] * inmat[5] * inmat[15] - inmat[3] * inmat[6] * inmat[13]) / det;
    outmat[3] =  (inmat[1] * inmat[7] * inmat[10] + inmat[2] * inmat[5] * inmat[11] + inmat[3] * inmat[6] * inmat[9] - inmat[1] * inmat[6] * inmat[11] - inmat[2] * inmat[7] * inmat[9] - inmat[3] * inmat[5] * inmat[10]) / det;
    outmat[4] =  (inmat[4] * inmat[11] * inmat[14] + inmat[6] * inmat[8] * inmat[15] + inmat[7] * inmat[10] * inmat[12] - inmat[4] * inmat[10] * inmat[15] - inmat[6] * inmat[11] * inmat[12] - inmat[7] * inmat[8] * inmat[14]) / det;
    outmat[5] =  (inmat[0] * inmat[10] * inmat[15] + inmat[2] * inmat[11] * inmat[12] + inmat[3] * inmat[8] * inmat[14] - inmat[0] * inmat[11] * inmat[14] - inmat[2] * inmat[8] * inmat[15] - inmat[3] * inmat[10] * inmat[12]) / det;
    outmat[6] =  (inmat[0] * inmat[7] * inmat[14] + inmat[2] * inmat[4] * inmat[15] + inmat[3] * inmat[6] * inmat[12] - inmat[0] * inmat[6] * inmat[15] - inmat[2] * inmat[7] * inmat[12] - inmat[3] * inmat[4] * inmat[14]) / det;
    outmat[7] =  (inmat[0] * inmat[6] * inmat[11] + inmat[2] * inmat[7] * inmat[8] + inmat[3] * inmat[4] * inmat[10] - inmat[0] * inmat[7] * inmat[10] - inmat[2] * inmat[4] * inmat[11] - inmat[3] * inmat[6] * inmat[8]) / det;
    outmat[8] =  (inmat[4] * inmat[9] * inmat[15] + inmat[5] * inmat[11] * inmat[12] + inmat[7] * inmat[8] * inmat[13] - inmat[4] * inmat[11] * inmat[13] - inmat[5] * inmat[8] * inmat[15] - inmat[7] * inmat[9] * inmat[12]) / det;
    outmat[9] =  (inmat[0] * inmat[11] * inmat[13] + inmat[1] * inmat[8] * inmat[15] + inmat[3] * inmat[9] * inmat[12] - inmat[0] * inmat[9] * inmat[15] - inmat[1] * inmat[11] * inmat[12] - inmat[3] * inmat[8] * inmat[13]) / det;
    outmat[10] = (inmat[0] * inmat[5] * inmat[15] + inmat[1] * inmat[7] * inmat[12] + inmat[3] * inmat[4] * inmat[13] - inmat[0] * inmat[7] * inmat[13] - inmat[1] * inmat[4] * inmat[15] - inmat[3] * inmat[5] * inmat[12]) / det;
    outmat[11] = (inmat[0] * inmat[7] * inmat[9] + inmat[1] * inmat[4] * inmat[11] + inmat[3] * inmat[5] * inmat[8] - inmat[0] * inmat[5] * inmat[11] - inmat[1] * inmat[7] * inmat[8] - inmat[3] * inmat[4] * inmat[9]) / det;
    outmat[12] = (inmat[4] * inmat[10] * inmat[13] + inmat[5] * inmat[8] * inmat[14] + inmat[6] * inmat[9] * inmat[12] - inmat[4] * inmat[9] * inmat[14] - inmat[5] * inmat[10] * inmat[12] - inmat[6] * inmat[8] * inmat[13]) / det;
    outmat[13] = (inmat[0] * inmat[9] * inmat[14] + inmat[1] * inmat[10] * inmat[12] + inmat[2] * inmat[8] * inmat[13] - inmat[0] * inmat[10] * inmat[13] - inmat[1] * inmat[8] * inmat[14] - inmat[2] * inmat[9] * inmat[12]) / det;
    outmat[14] = (inmat[0] * inmat[6] * inmat[13] + inmat[1] * inmat[4] * inmat[14] + inmat[2] * inmat[5] * inmat[12] - inmat[0] * inmat[5] * inmat[14] - inmat[1] * inmat[6] * inmat[12] - inmat[2] * inmat[4] * inmat[13]) / det;
    outmat[15] = (inmat[0] * inmat[5] * inmat[10] + inmat[1] * inmat[6] * inmat[8] + inmat[2] * inmat[4] * inmat[9] - inmat[0] * inmat[6] * inmat[9] - inmat[1] * inmat[4] * inmat[10] - inmat[2] * inmat[5] * inmat[8]) / det;
  }
  if (determinant) *determinant = det;
  PetscFunctionReturn(0);
}


/*@
  Compute_Lagrange_Basis_1D_Internal: Evaluate bases and derivatives at quadrature points for a EDGE2 or EDGE3 element.

  The routine is given the coordinates of the vertices of a linear or quadratic edge element.

  The routine evaluates the basis functions associated with each quadrature point provided,
  and their derivatives with respect to X.

  Notes:

  Example Physical Element

    1-------2        1----3----2
      EDGE2             EDGE3

  Input Parameter:

.  PetscInt  nverts,           the number of element vertices
.  PetscReal coords[3*nverts], the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts,             the number of evaluation points (quadrature points)
.  PetscReal quad[3*npts],     the evaluation points (quadrature points) in the reference space

  Output Parameter:
.  PetscReal phypts[3*npts],   the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts],        the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal phi[npts],        the bases evaluated at the specified quadrature points
.  PetscReal dphidx[npts],     the derivative of the bases wrt X-direction evaluated at the specified quadrature points

.keywords: DMMoab, FEM, 1-D
@*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Lagrange_Basis_1D_Internal"
PetscErrorCode Compute_Lagrange_Basis_1D_Internal ( const PetscInt nverts, const PetscReal *coords, const PetscInt npts, const PetscReal *quad, PetscReal *phypts,
    PetscReal *jxw, PetscReal *phi, PetscReal *dphidx,
    PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  int i, j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(jacobian, 9);
  PetscValidPointer(ijacobian, 10);
  PetscValidPointer(volume, 11);
  if (phypts) {
    ierr = PetscMemzero(phypts, npts * 3 * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (dphidx) { /* Reset arrays. */
    ierr = PetscMemzero(dphidx, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (nverts == 2) { /* Linear Edge */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const PetscReal r = quad[j];

      phi[0 + offset] = ( 1.0 - r );
      phi[1 + offset] = (       r );

      const PetscReal dNi_dxi[2]  = { -1.0, 1.0 };

      jacobian[0] = ijacobian[0] = volume[0] = 0.0;
      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertices = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        if (phypts) {
          phypts[3 * j + 0] += phi[i + offset] * vertices[0];
        }
      }

      /* invert the jacobian */
      *volume = jacobian[0];
      ijacobian[0] = 1.0 / jacobian[0];
      jxw[j] *= *volume;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[0];
      }

    }
  }
  else if (nverts == 3) { /* Quadratic Edge */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const PetscReal r = quad[j];

      phi[0 + offset] = 1.0 + r * ( 2.0 * r - 3.0 );
      phi[1 + offset] = 4.0 * r * ( 1.0 - r );
      phi[2 + offset] = r * ( 2.0 * r - 1.0 );

      const PetscReal dNi_dxi[3]  = { 4 * r - 3.0, 4 * ( 1.0 - 2.0 * r ), 4.0 * r - 1.0};

      jacobian[0] = ijacobian[0] = volume[0] = 0.0;
      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertices = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        if (phypts) {
          phypts[3 * j + 0] += phi[i + offset] * vertices[0];
        }
      }

      /* invert the jacobian */
      *volume = jacobian[0];
      ijacobian[0] = 1.0 / jacobian[0];
      if (jxw) jxw[j] *= *volume;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[0];
      }
    }
  }
  else {
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of entity vertices are invalid. Currently only support EDGE2 and EDGE3 basis evaluations in 1-D : %D", nverts);
  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < npts; j++ ) {
    PetscScalar phisum = 0, dphixsum = 0;
    for ( i = 0; i < nverts; i++ ) {
      phisum += phi[i + j * nverts];
      if (dphidx) dphixsum += dphidx[i + j * nverts];
    }
    PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D = %g, %g, %g\n", j, phisum, dphixsum);
  }
#endif
  PetscFunctionReturn(0);
}


/*@
  Compute_Lagrange_Basis_2D_Internal: Evaluate bases and derivatives at quadrature points for a QUAD4 or TRI3 element.

  The routine is given the coordinates of the vertices of a quadrangle or triangle.

  The routine evaluates the basis functions associated with each quadrature point provided,
  and their derivatives with respect to X and Y.

  Notes:

  Example Physical Element (QUAD4)

    4------3        s
    |      |        |
    |      |        |
    |      |        |
    1------2        0-------r

  Input Parameter:

.  PetscInt  nverts,           the number of element vertices
.  PetscReal coords[3*nverts], the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts,             the number of evaluation points (quadrature points)
.  PetscReal quad[3*npts],     the evaluation points (quadrature points) in the reference space

  Output Parameter:
.  PetscReal phypts[3*npts],   the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts],        the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal phi[npts],        the bases evaluated at the specified quadrature points
.  PetscReal dphidx[npts],     the derivative of the bases wrt X-direction evaluated at the specified quadrature points
.  PetscReal dphidy[npts],     the derivative of the bases wrt Y-direction evaluated at the specified quadrature points

.keywords: DMMoab, FEM, 2-D
@*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Lagrange_Basis_2D_Internal"
PetscErrorCode Compute_Lagrange_Basis_2D_Internal ( const PetscInt nverts, const PetscReal *coords, const PetscInt npts, const PetscReal *quad, PetscReal *phypts,
    PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy,
    PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  int i, j;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(jacobian, 10);
  PetscValidPointer(ijacobian, 11);
  PetscValidPointer(volume, 12);
  ierr = PetscMemzero(phi, npts * sizeof(PetscReal));CHKERRQ(ierr);
  if (phypts) {
    ierr = PetscMemzero(phypts, npts * 3 * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (dphidx) { /* Reset arrays. */
    ierr = PetscMemzero(dphidx, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidy, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (nverts == 4) { /* Linear Quadrangle */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const PetscReal r = quad[0 + j * 2];
      const PetscReal s = quad[1 + j * 2];

      phi[0 + offset] = ( 1.0 - r ) * ( 1.0 - s );
      phi[1 + offset] =         r   * ( 1.0 - s );
      phi[2 + offset] =         r   *         s;
      phi[3 + offset] = ( 1.0 - r ) *         s;

      const PetscReal dNi_dxi[4]  = { -1.0 + s, 1.0 - s, s, -s };
      const PetscReal dNi_deta[4] = { -1.0 + r, -r, r, 1.0 - r };

      ierr = PetscMemzero(jacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(ijacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);
      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertices = coords + i * 3;
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
      ierr = DMatrix_Invert_2x2_Internal(jacobian, ijacobian, volume);CHKERRQ(ierr);
      if ( *volume < 1e-12 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Quadrangular element has zero volume: %g. Degenerate element or invalid connectivity\n", *volume);

      if (jxw) jxw[j] *= *volume;

      /*  Let us compute the bases derivatives by scaling with inverse jacobian. */
      if (dphidx) {
        for ( i = 0; i < nverts; i++ ) {
          for (int k = 0; k < 2; ++k) {
            if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[k * 2 + 0];
            if (dphidy) dphidy[i + offset] += dNi_deta[i] * ijacobian[k * 2 + 1];
          } /* for k=[0..2) */
        } /* for i=[0..nverts) */
      } /* if (dphidx) */
    }
  }
  else if (nverts == 3) { /* Linear triangle */

    ierr = PetscMemzero(jacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(ijacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);

    const PetscReal x2 = coords[2 * 3 + 0], y2 = coords[2 * 3 + 1];

    /* Jacobian is constant */
    jacobian[0] = (coords[0 * 3 + 0] - x2); jacobian[1] = (coords[1 * 3 + 0] - x2);
    jacobian[2] = (coords[0 * 3 + 1] - y2); jacobian[3] = (coords[1 * 3 + 1] - y2);

    /* invert the jacobian */
    ierr = DMatrix_Invert_2x2_Internal(jacobian, ijacobian, volume);CHKERRQ(ierr);
    if ( *volume < 1e-12 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Triangular element has zero volume: %g. Degenerate element or invalid connectivity\n", *volume);

    const PetscReal Dx[3] = { ijacobian[0], ijacobian[2], - ijacobian[0] - ijacobian[2] };
    const PetscReal Dy[3] = { ijacobian[1], ijacobian[3], - ijacobian[1] - ijacobian[3] };

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const PetscReal r = quad[0 + j * 2];
      const PetscReal s = quad[1 + j * 2];

      if (jxw) jxw[j] *= 0.5;

      /* const PetscReal Ni[3]  = { r, s, 1.0 - r - s }; */
      const PetscReal phipts_x = coords[2 * 3 + 0] + jacobian[0] * r + jacobian[1] * s;
      const PetscReal phipts_y = coords[2 * 3 + 1] + jacobian[2] * r + jacobian[3] * s;
      if (phypts) {
        phypts[offset + 0] = phipts_x;
        phypts[offset + 1] = phipts_y;
      }

      /* \phi_0 = (b.y − c.y) x + (b.x − c.x) y + c.x b.y − b.x c.y */
      phi[0 + offset] = (  ijacobian[0] * (phipts_x - x2) + ijacobian[1] * (phipts_y - y2) );
      /* \phi_1 = (c.y − a.y) x + (a.x − c.x) y + c.x a.y − a.x c.y */
      phi[1 + offset] = (  ijacobian[2] * (phipts_x - x2) + ijacobian[3] * (phipts_y - y2) );
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
  }
  else {
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of entity vertices are invalid. Currently only support QUAD4 and TRI3 basis evaluations in 2-D : %D", nverts);
  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < npts; j++ ) {
    PetscScalar phisum = 0, dphixsum = 0, dphiysum = 0;
    for ( i = 0; i < nverts; i++ ) {
      phisum += phi[i + j * nverts];
      if (dphidx) dphixsum += dphidx[i + j * nverts];
      if (dphidy) dphiysum += dphidy[i + j * nverts];
    }
    PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D = %g, %g, %g\n", j, phisum, dphixsum, dphiysum);
  }
#endif
  PetscFunctionReturn(0);
}


/*@
  Compute_Lagrange_Basis_3D_Internal: Evaluate bases and derivatives at quadrature points for a HEX8 or TET4 element.

  The routine is given the coordinates of the vertices of a hexahedra or tetrahedra.

  The routine evaluates the basis functions associated with each quadrature point provided,
  and their derivatives with respect to X, Y and Z.

  Notes:

  Example Physical Element (HEX8)

      8------7
     /|     /|        t  s
    5------6 |        | /
    | |    | |        |/
    | 4----|-3        0-------r
    |/     |/
    1------2

  Input Parameter:

.  PetscInt  nverts,           the number of element vertices
.  PetscReal coords[3*nverts], the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts,             the number of evaluation points (quadrature points)
.  PetscReal quad[3*npts],     the evaluation points (quadrature points) in the reference space

  Output Parameter:
.  PetscReal phypts[3*npts],   the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts],        the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal phi[npts],        the bases evaluated at the specified quadrature points
.  PetscReal dphidx[npts],     the derivative of the bases wrt X-direction evaluated at the specified quadrature points
.  PetscReal dphidy[npts],     the derivative of the bases wrt Y-direction evaluated at the specified quadrature points
.  PetscReal dphidz[npts],     the derivative of the bases wrt Z-direction evaluated at the specified quadrature points

.keywords: DMMoab, FEM, 3-D
@*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Lagrange_Basis_3D_Internal"
PetscErrorCode Compute_Lagrange_Basis_3D_Internal ( const PetscInt nverts, const PetscReal *coords, const PetscInt npts, const PetscReal *quad, PetscReal *phypts,
    PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy, PetscReal *dphidz,
    PetscReal *jacobian, PetscReal *ijacobian, PetscReal *volume)
{
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(jacobian, 11);
  PetscValidPointer(ijacobian, 12);
  PetscValidPointer(volume, 13);
  /* Reset arrays. */
  ierr = PetscMemzero(phi, npts * sizeof(PetscReal));CHKERRQ(ierr);
  if (phypts) {
    ierr = PetscMemzero(phypts, npts * 3 * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (dphidx) {
    ierr = PetscMemzero(dphidx, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidy, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidz, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }

  if (nverts == 8) { /* Linear Hexahedra */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const PetscReal& r = quad[j * 3 + 0];
      const PetscReal& s = quad[j * 3 + 1];
      const PetscReal& t = quad[j * 3 + 2];

      phi[offset + 0] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 - t ) / 8;
      phi[offset + 1] = ( 1.0 + r ) * ( 1.0 - s ) * ( 1.0 - t ) / 8;
      phi[offset + 2] = ( 1.0 + r ) * ( 1.0 + s ) * ( 1.0 - t ) / 8;
      phi[offset + 3] = ( 1.0 - r ) * ( 1.0 + s ) * ( 1.0 - t ) / 8;
      phi[offset + 4] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 + t ) / 8;
      phi[offset + 5] = ( 1.0 + r ) * ( 1.0 - s ) * ( 1.0 + t ) / 8;
      phi[offset + 6] = ( 1.0 + r ) * ( 1.0 + s ) * ( 1.0 + t ) / 8;
      phi[offset + 7] = ( 1.0 - r ) * ( 1.0 + s ) * ( 1.0 + t ) / 8;

      const PetscReal dNi_dxi[8]  = { - ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 + s ) * ( 1.0 - t ),
                                     - ( 1.0 + s ) * ( 1.0 - t ),
                                     - ( 1.0 - s ) * ( 1.0 + t ),
                                     ( 1.0 - s ) * ( 1.0 + t ),
                                     ( 1.0 + s ) * ( 1.0 + t ),
                                     - ( 1.0 + s ) * ( 1.0 + t )
                                   };

      const PetscReal dNi_deta[8]  = { - ( 1.0 - r ) * ( 1.0 - t ),
                                      - ( 1.0 + r ) * ( 1.0 - t ),
                                      ( 1.0 + r ) * ( 1.0 - t ),
                                      ( 1.0 - r ) * ( 1.0 - t ),
                                      - ( 1.0 - r ) * ( 1.0 + t ),
                                      - ( 1.0 + r ) * ( 1.0 + t ),
                                      ( 1.0 + r ) * ( 1.0 + t ),
                                      ( 1.0 - r ) * ( 1.0 + t )
                                    };

      const PetscReal dNi_dzeta[8]  = { - ( 1.0 - r ) * ( 1.0 - s ),
                                       - ( 1.0 + r ) * ( 1.0 - s ),
                                       - ( 1.0 + r ) * ( 1.0 + s ),
                                       - ( 1.0 - r ) * ( 1.0 + s ),
                                       ( 1.0 - r ) * ( 1.0 - s ),
                                       ( 1.0 + r ) * ( 1.0 - s ),
                                       ( 1.0 + r ) * ( 1.0 + s ),
                                       ( 1.0 - r ) * ( 1.0 + s )
                                     };

      ierr = PetscMemzero(jacobian, 9 * sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(ijacobian, 9 * sizeof(PetscReal));CHKERRQ(ierr);
      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertex = coords + i * 3;
        jacobian[0] += dNi_dxi[i]   * vertex[0];
        jacobian[3] += dNi_dxi[i]   * vertex[1];
        jacobian[6] += dNi_dxi[i]   * vertex[2];
        jacobian[1] += dNi_deta[i]  * vertex[0];
        jacobian[4] += dNi_deta[i]  * vertex[1];
        jacobian[7] += dNi_deta[i]  * vertex[2];
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
      ierr = DMatrix_Invert_3x3_Internal(jacobian, ijacobian, volume);CHKERRQ(ierr);
      if ( *volume < 1e-8 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Hexahedral element has zero volume: %g. Degenerate element or invalid connectivity\n", volume);

      const PetscReal factor = 1.0 / 8; /* Our basis is defined on [-1 to 1]^3 */
      if (jxw) jxw[j] *= factor * (*volume);

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; ++i ) {
        for (int k = 0; k < 3; ++k) {
          if (dphidx) dphidx[i + offset] += dNi_dxi[i]   * ijacobian[0 * 3 + k];
          if (dphidy) dphidy[i + offset] += dNi_deta[i]  * ijacobian[1 * 3 + k];
          if (dphidz) dphidz[i + offset] += dNi_dzeta[i] * ijacobian[2 * 3 + k];
        }
      }
    }
  }
  else if (nverts == 4) { /* Linear Tetrahedra */

    ierr = PetscMemzero(jacobian, 9 * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(ijacobian, 9 * sizeof(PetscReal));CHKERRQ(ierr);

    const PetscReal x0 = coords[/*0 * 3 +*/ 0], y0 = coords[/*0 * 3 +*/ 1], z0 = coords[/*0 * 3 +*/ 2];

    /* compute the jacobian */
    jacobian[0] = coords[1 * 3 + 0] - x0;  jacobian[1] = coords[2 * 3 + 0] - x0; jacobian[2] = coords[3 * 3 + 0] - x0;
    jacobian[3] = coords[1 * 3 + 1] - y0;  jacobian[4] = coords[2 * 3 + 1] - y0; jacobian[5] = coords[3 * 3 + 1] - y0;
    jacobian[6] = coords[1 * 3 + 2] - z0;  jacobian[7] = coords[2 * 3 + 2] - z0; jacobian[8] = coords[3 * 3 + 2] - z0;

    /* invert the jacobian */
    ierr = DMatrix_Invert_3x3_Internal(jacobian, ijacobian, volume);CHKERRQ(ierr);
    if ( *volume < 1e-8 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Tetrahedral element has zero volume: %g. Degenerate element or invalid connectivity\n", volume);

    // const PetscReal Dx[4] = { ijacobian[0], ijacobian[3], ijacobian[6], - ijacobian[0] - ijacobian[3] - ijacobian[6] };
    // const PetscReal Dy[4] = { ijacobian[1], ijacobian[4], ijacobian[7], - ijacobian[1] - ijacobian[4] - ijacobian[7] };
    // const PetscReal Dz[4] = { ijacobian[2], ijacobian[5], ijacobian[8], - ijacobian[2] - ijacobian[5] - ijacobian[8] };
    PetscReal Dx[4], Dy[4], Dz[4];
    if (dphidx) {
      Dx[0] =   ( coords[1 + 2 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                 - coords[1 + 1 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                 - coords[1 + 3 * 3] * ( coords[2 + 1 * 3] - coords[2 + 2 * 3] )
                ) / *volume;
      Dx[1] = - ( coords[1 + 2 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                 - coords[1 + 0 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                 - coords[1 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 2 * 3] )
                ) / *volume;
      Dx[2] =   ( coords[1 + 1 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                 - coords[1 + 0 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                 - coords[1 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 1 * 3] )
                ) / *volume;
      Dx[3] =  - ( Dx[0] + Dx[1] + Dx[2] );
      Dy[0] =   ( coords[0 + 1 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                 - coords[0 + 2 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                 + coords[0 + 3 * 3] * ( coords[2 + 1 * 3] - coords[2 + 2 * 3] )
                ) / *volume;
      Dy[1] = - ( coords[0 + 0 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                 - coords[0 + 2 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                 + coords[0 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 2 * 3] )
                ) / *volume;
      Dy[2] =   ( coords[0 + 0 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                 - coords[0 + 1 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                 + coords[0 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 1 * 3] )
                ) / *volume;
      Dy[3] =  - ( Dy[0] + Dy[1] + Dy[2] );
      Dz[0] =   ( coords[0 + 1 * 3] * (coords[1 + 3 * 3] - coords[1 + 2 * 3] )
                 - coords[0 + 2 * 3] * (coords[1 + 3 * 3] - coords[1 + 1 * 3] )
                 + coords[0 + 3 * 3] * (coords[1 + 2 * 3] - coords[1 + 1 * 3] )
                ) / *volume;
      Dz[1] = - ( coords[0 + 0 * 3] * (coords[1 + 3 * 3] - coords[1 + 2 * 3] )
                  + coords[0 + 2 * 3] * (coords[1 + 0 * 3] - coords[1 + 3 * 3] )
                  - coords[0 + 3 * 3] * (coords[1 + 0 * 3] - coords[1 + 2 * 3] )
                ) / *volume;
      Dz[2] =   ( coords[0 + 0 * 3] * (coords[1 + 3 * 3] - coords[1 + 1 * 3] )
                 + coords[0 + 1 * 3] * (coords[1 + 0 * 3] - coords[1 + 3 * 3] )
                 - coords[0 + 3 * 3] * (coords[1 + 0 * 3] - coords[1 + 1 * 3] )
                ) / *volume;
      Dz[3] =  - ( Dz[0] + Dz[1] + Dz[2] );
    }

    for ( j = 0; j < npts; j++ )
    {
      const int offset = j * nverts;
      const PetscReal& r = quad[j * 3 + 0];
      const PetscReal& s = quad[j * 3 + 1];
      const PetscReal& t = quad[j * 3 + 2];

      if (jxw) jxw[j] *= *volume;

      phi[offset + 0] = 1.0 - r - s - t;
      phi[offset + 1] = r;
      phi[offset + 2] = s;
      phi[offset + 3] = t;

      if (phypts) {
        for (i = 0; i < nverts; ++i) {
          const PetscScalar* vertices = coords + i * 3;
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
      }

      if (dphidy) {
        dphidy[0 + offset] = Dy[0];
        dphidy[1 + offset] = Dy[1];
        dphidy[2 + offset] = Dy[2];
        dphidy[3 + offset] = Dy[3];
      }

      if (dphidz) {
        dphidz[0 + offset] = Dz[0];
        dphidz[1 + offset] = Dz[1];
        dphidz[2 + offset] = Dz[2];
        dphidz[3 + offset] = Dz[3];
      }

    } /* Tetrahedra -- ends */
  }
  else
  {
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of entity vertices are invalid. Currently only support HEX8 and TET4 basis evaluations in 3-D : %D", nverts);
  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < npts; j++ ) {
    PetscScalar phisum = 0, dphixsum = 0, dphiysum = 0, dphizsum = 0;
    const int offset = j * nverts;
    for ( i = 0; i < nverts; i++ ) {
      phisum += phi[i + offset];
      if (dphidx) dphixsum += dphidx[i + offset];
      if (dphidy) dphiysum += dphidy[i + offset];
      if (dphidz) dphizsum += dphidz[i + offset];
      if (dphidx) PetscPrintf(PETSC_COMM_WORLD, "\t Values [%d]: [JxW] [phi, dphidx, dphidy, dphidz] = %g, %g, %g, %g, %g\n", j, jxw[j], phi[i + offset], dphidx[i + offset], dphidy[i + offset], dphidz[i + offset]);
    }
    if (dphidx) PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D (%g, %g, %g) = %g, %g, %g, %g\n", j, quad[3 * j + 0], quad[3 * j + 1], quad[3 * j + 2], phisum, dphixsum, dphiysum, dphizsum);
  }
#endif
  PetscFunctionReturn(0);
}



/*@
  DMMoabFEMComputeBasis: Evaluate bases and derivatives at quadrature points for a linear EDGE/QUAD/TRI/HEX/TET element.

  The routine takes the coordinates of the vertices of an element and computes basis functions associated with
  each quadrature point provided, and their derivatives with respect to X, Y and Z as appropriate.

  Input Parameter:

.  PetscInt  nverts,           the number of element vertices
.  PetscReal coords[3*nverts], the physical coordinates of the vertices (in canonical numbering)
.  PetscInt  npts,             the number of evaluation points (quadrature points)
.  PetscReal quad[3*npts],     the evaluation points (quadrature points) in the reference space

  Output Parameter:
.  PetscReal phypts[3*npts],   the evaluation points (quadrature points) transformed to the physical space
.  PetscReal jxw[npts],        the jacobian determinant * quadrature weight necessary for assembling discrete contributions
.  PetscReal fe_basis[npts],        the bases values evaluated at the specified quadrature points
.  PetscReal fe_basis_derivatives[dim][npts],  the derivative of the bases wrt (X,Y,Z)-directions (depending on the dimension) evaluated at the specified quadrature points

.keywords: DMMoab, FEM, 3-D
@*/
#undef __FUNCT__
#define __FUNCT__ "DMMoabFEMComputeBasis"
PetscErrorCode DMMoabFEMComputeBasis ( const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscQuadrature quadrature, 
                                       PetscReal *phypts, PetscReal *jacobian_quadrature_weight_product, 
                                       PetscReal *fe_basis, PetscReal **fe_basis_derivatives)
{
  PetscErrorCode  ierr;
  PetscInt        npoints;
  bool            compute_der;
  const PetscReal *quadpts, *quadwts;
  PetscReal       jacobian[9], ijacobian[9], volume;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidHeaderSpecific(quadrature, PETSC_OBJECT_CLASSID, 4);
  PetscValidPointer(fe_basis, 7);
  compute_der = (fe_basis_derivatives != NULL);

  /* Get the quadrature points and weights for the given quadrature rule */
  ierr = PetscQuadratureGetData(quadrature, NULL, &npoints, &quadpts, &quadwts);CHKERRQ(ierr);
  if (jacobian_quadrature_weight_product) {
    ierr = PetscMemcpy(jacobian_quadrature_weight_product, quadwts, npoints * sizeof(PetscReal));CHKERRQ(ierr);
  }

  switch (dim) {
  case 1:
    ierr = Compute_Lagrange_Basis_1D_Internal(nverts, coordinates, npoints, quadpts, phypts,
           jacobian_quadrature_weight_product, fe_basis,
           (compute_der ? fe_basis_derivatives[0] : NULL),
           jacobian, ijacobian, &volume);CHKERRQ(ierr);
    break;
  case 2:
    ierr = Compute_Lagrange_Basis_2D_Internal(nverts, coordinates, npoints, quadpts, phypts,
           jacobian_quadrature_weight_product, fe_basis,
           (compute_der ? fe_basis_derivatives[0] : NULL),
           (compute_der ? fe_basis_derivatives[1] : NULL),
           jacobian, ijacobian, &volume);CHKERRQ(ierr);
    break;
  case 3:
    ierr = Compute_Lagrange_Basis_3D_Internal(nverts, coordinates, npoints, quadpts, phypts,
           jacobian_quadrature_weight_product, fe_basis,
           (compute_der ? fe_basis_derivatives[0] : NULL),
           (compute_der ? fe_basis_derivatives[1] : NULL),
           (compute_der ? fe_basis_derivatives[2] : NULL),
           jacobian, ijacobian, &volume);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension; should be in [1,3] : %D", dim);
  }
  PetscFunctionReturn(0);
}



/*@
  DMMoabFEMCreateQuadratureDefault: Create default quadrature rules for integration over an element with a given
  dimension and polynomial order (deciphered from number of element vertices).

  Input Parameter:

.  PetscInt  dim,           the element dimension (1=EDGE, 2=QUAD/TRI, 3=HEX/TET)
.  PetscInt nverts,      the number of vertices in the physical element

  Output Parameter:
.  PetscQuadrature quadrature,  the quadrature object with default settings to integrate polynomials defined over the element

.keywords: DMMoab, Quadrature, PetscDT
@*/
#undef __FUNCT__
#define __FUNCT__ "DMMoabFEMCreateQuadratureDefault"
PetscErrorCode DMMoabFEMCreateQuadratureDefault ( const PetscInt dim, const PetscInt nverts, PetscQuadrature *quadrature )
{
  PetscReal *w, *x;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create an appropriate quadrature rule to sample basis */
  switch (dim)
  {
  case 1:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    ierr = PetscDTGaussJacobiQuadrature(1, nverts, 0, 1.0, quadrature);CHKERRQ(ierr);
    break;
  case 2:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    if (nverts == 3) {
      const int order = 2;
      const int npoints = (order == 2 ? 3 : 6);
      ierr = PetscMalloc2(npoints * 2, &x, npoints, &w);CHKERRQ(ierr);
      if (npoints == 3) {
        x[0] = x[1] = x[2] = x[5] = 1.0 / 6.0;
        x[3] = x[4] = 2.0 / 3.0;
        w[0] = w[1] = w[2] = 1.0 / 3.0;
      }
      else if (npoints == 6) {
        x[0] = x[1] = x[2] = 0.44594849091597;
        x[3] = x[4] = 0.10810301816807;
        x[5] = 0.44594849091597;
        x[6] = x[7] = x[8] = 0.09157621350977;
        x[9] = x[10] = 0.81684757298046;
        x[11] = 0.09157621350977;
        w[0] = w[1] = w[2] = 0.22338158967801;
        w[3] = w[4] = w[5] = 0.10995174365532;
      }
      else {
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Triangle quadrature rules for points 3 and 6 supported; npoints : %D", npoints);
      }
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, quadrature);CHKERRQ(ierr);
      ierr = PetscQuadratureSetOrder(*quadrature, order);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(*quadrature, dim, npoints, x, w);CHKERRQ(ierr);
      /* ierr = PetscDTGaussJacobiQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr); */
    }
    else {
      ierr = PetscDTGaussTensorQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr);
    }
    break;
  case 3:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    if (nverts == 4) {
      int order;
      const int npoints = 4; // Choose between 4 and 10
      ierr = PetscMalloc2(npoints * 3, &x, npoints, &w);CHKERRQ(ierr);
      if (npoints == 4) { /*  KEAST1, K1,  N=4, O=4 */
        const PetscReal x_4[12] = { 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
                                    0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
                                    0.1381966011250105, 0.1381966011250105, 0.5854101966249685,
                                    0.1381966011250105, 0.5854101966249685, 0.1381966011250105
                                  };
        ierr = PetscMemcpy(x, x_4, 12 * sizeof(PetscReal));CHKERRQ(ierr);

        w[0] = w[1] = w[2] = w[3] = 1.0 / 24.0;
        order = 4;
      }
      else if (npoints == 10) { /*  KEAST3, K3  N=10, O=10 */
        const PetscReal x_10[30] = { 0.5684305841968444, 0.1438564719343852, 0.1438564719343852,
                                     0.1438564719343852, 0.1438564719343852, 0.1438564719343852,
                                     0.1438564719343852, 0.1438564719343852, 0.5684305841968444,
                                     0.1438564719343852, 0.5684305841968444, 0.1438564719343852,
                                     0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
                                     0.5000000000000000, 0.0000000000000000, 0.5000000000000000,
                                     0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
                                     0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
                                     0.0000000000000000, 0.5000000000000000, 0.0000000000000000,
                                     0.0000000000000000, 0.0000000000000000, 0.5000000000000000
                                   };
        ierr = PetscMemcpy(x, x_10, 30 * sizeof(PetscReal));CHKERRQ(ierr);

        w[0] = w[1] = w[2] = w[3] = 0.2177650698804054;
        w[4] = w[5] = w[6] = w[7] = w[8] = w[9] = 0.0214899534130631;
        order = 10;
      }
      else {
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Tetrahedral quadrature rules for points 4 and 10 supported; npoints : %D", npoints);
      }
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, quadrature);CHKERRQ(ierr);
      ierr = PetscQuadratureSetOrder(*quadrature, order);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(*quadrature, dim, npoints, x, w);CHKERRQ(ierr);
      /* ierr = PetscDTGaussJacobiQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr); */
    }
    else {
      ierr = PetscDTGaussTensorQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension; should be in [1,3] : %D", dim);
  }
  PetscFunctionReturn(0);
}

/* Compute Jacobians */
#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian_Internal"
PetscErrorCode ComputeJacobian_Internal ( const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscReal *quad, PetscReal *phypts,
  PetscReal *jacobian, PetscReal *ijacobian, PetscReal* volume)
{
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidPointer(quad, 4);
  PetscValidPointer(jacobian, 5);
  ierr = PetscMemzero(jacobian, dim * dim * sizeof(PetscReal));CHKERRQ(ierr);
  if (ijacobian) {
    ierr = PetscMemzero(ijacobian, dim * dim * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (phypts) {
    ierr = PetscMemzero(phypts, /*npts=1 * */ 3 * sizeof(PetscReal));CHKERRQ(ierr);
  }

  if (dim == 1) {

    const PetscReal& r = quad[0];
    if (nverts == 2) { /* Linear Edge */
      const PetscReal dNi_dxi[2]  = { -1.0, 1.0 };

      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertices = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
      }
    }
    else if (nverts == 3) { /* Quadratic Edge */
      const PetscReal dNi_dxi[3]  = { 4 * r - 3.0, 4 * ( 1.0 - 2.0 * r ), 4.0 * r - 1.0};

      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertices = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
      }
    }
    else {
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of 1-D entity vertices are invalid. Currently only support EDGE2 and EDGE3 basis evaluations in 1-D : %D", nverts);
    }

    if (ijacobian) {
      /* invert the jacobian */
      ijacobian[0] = 1.0 / jacobian[0];
    }

  }
  else if (dim == 2) {

    if (nverts == 4) { /* Linear Quadrangle */
      const PetscReal& r = quad[0];
      const PetscReal& s = quad[1];

      const PetscReal dNi_dxi[4]  = { -1.0 + s, 1.0 - s, s, -s };
      const PetscReal dNi_deta[4] = { -1.0 + r, -r, r, 1.0 - r };

      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertices = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i]  * vertices[0];
        jacobian[2] += dNi_dxi[i]  * vertices[1];
        jacobian[1] += dNi_deta[i] * vertices[0];
        jacobian[3] += dNi_deta[i] * vertices[1];
      }
    }
    else if (nverts == 3) { /* Linear triangle */
      const PetscReal x2 = coordinates[2 * 3 + 0], y2 = coordinates[2 * 3 + 1];

      /* Jacobian is constant */
      jacobian[0] = (coordinates[0 * 3 + 0] - x2); jacobian[1] = (coordinates[1 * 3 + 0] - x2);
      jacobian[2] = (coordinates[0 * 3 + 1] - y2); jacobian[3] = (coordinates[1 * 3 + 1] - y2);
    }
    else {
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of 2-D entity vertices are invalid. Currently only support QUAD4 and TRI3 basis evaluations in 2-D : %D", nverts);
    }

    /* invert the jacobian */
    if (ijacobian) {
      ierr = DMatrix_Invert_2x2_Internal(jacobian, ijacobian, volume);CHKERRQ(ierr);
    }

  }
  else {

    if (nverts == 8) { /* Linear Hexahedra */
      const PetscReal& r = quad[0];
      const PetscReal& s = quad[1];
      const PetscReal& t = quad[2];
      const PetscReal dNi_dxi[8]  = { - ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 + s ) * ( 1.0 - t ),
                                     - ( 1.0 + s ) * ( 1.0 - t ),
                                     - ( 1.0 - s ) * ( 1.0 + t ),
                                     ( 1.0 - s ) * ( 1.0 + t ),
                                     ( 1.0 + s ) * ( 1.0 + t ),
                                     - ( 1.0 + s ) * ( 1.0 + t )
                                   };

      const PetscReal dNi_deta[8]  = { - ( 1.0 - r ) * ( 1.0 - t ),
                                      - ( 1.0 + r ) * ( 1.0 - t ),
                                      ( 1.0 + r ) * ( 1.0 - t ),
                                      ( 1.0 - r ) * ( 1.0 - t ),
                                      - ( 1.0 - r ) * ( 1.0 + t ),
                                      - ( 1.0 + r ) * ( 1.0 + t ),
                                      ( 1.0 + r ) * ( 1.0 + t ),
                                      ( 1.0 - r ) * ( 1.0 + t )
                                    };

      const PetscReal dNi_dzeta[8]  = { - ( 1.0 - r ) * ( 1.0 - s ),
                                       - ( 1.0 + r ) * ( 1.0 - s ),
                                       - ( 1.0 + r ) * ( 1.0 + s ),
                                       - ( 1.0 - r ) * ( 1.0 + s ),
                                       ( 1.0 - r ) * ( 1.0 - s ),
                                       ( 1.0 + r ) * ( 1.0 - s ),
                                       ( 1.0 + r ) * ( 1.0 + s ),
                                       ( 1.0 - r ) * ( 1.0 + s )
                                     };
      for (i = 0; i < nverts; ++i) {
        const PetscReal* vertex = coordinates + i * 3;
        jacobian[0] += dNi_dxi[i]   * vertex[0];
        jacobian[3] += dNi_dxi[i]   * vertex[1];
        jacobian[6] += dNi_dxi[i]   * vertex[2];
        jacobian[1] += dNi_deta[i]  * vertex[0];
        jacobian[4] += dNi_deta[i]  * vertex[1];
        jacobian[7] += dNi_deta[i]  * vertex[2];
        jacobian[2] += dNi_dzeta[i] * vertex[0];
        jacobian[5] += dNi_dzeta[i] * vertex[1];
        jacobian[8] += dNi_dzeta[i] * vertex[2];
      }

    }
    else if (nverts == 4) { /* Linear Tetrahedra */
      const PetscReal x0 = coordinates[/*0 * 3 +*/ 0], y0 = coordinates[/*0 * 3 +*/ 1], z0 = coordinates[/*0 * 3 +*/ 2];

      /* compute the jacobian */
      jacobian[0] = coordinates[1 * 3 + 0] - x0;  jacobian[1] = coordinates[2 * 3 + 0] - x0; jacobian[2] = coordinates[3 * 3 + 0] - x0;
      jacobian[3] = coordinates[1 * 3 + 1] - y0;  jacobian[4] = coordinates[2 * 3 + 1] - y0; jacobian[5] = coordinates[3 * 3 + 1] - y0;
      jacobian[6] = coordinates[1 * 3 + 2] - z0;  jacobian[7] = coordinates[2 * 3 + 2] - z0; jacobian[8] = coordinates[3 * 3 + 2] - z0;
    } /* Tetrahedra -- ends */
    else
    {
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The number of 3-D entity vertices are invalid. Currently only support HEX8 and TET4 basis evaluations in 3-D : %D", nverts);
    }

    if (ijacobian) {
      /* invert the jacobian */
      ierr = DMatrix_Invert_3x3_Internal(jacobian, ijacobian, volume);CHKERRQ(ierr);
    }

  }
  if ( volume && *volume < 1e-12 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Element has zero volume: %g. Degenerate element or invalid connectivity\n", volume);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FEMComputeBasis_JandF"
PetscErrorCode FEMComputeBasis_JandF ( const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscReal *quadrature, PetscReal *phypts,
                                       PetscReal *phibasis, PetscReal *jacobian, PetscReal *ijacobian, PetscReal* volume  )
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  switch (dim) {
  case 1:
    ierr = Compute_Lagrange_Basis_1D_Internal(nverts, coordinates, 1, quadrature, phypts,
           NULL, phibasis, NULL, jacobian, ijacobian, volume);CHKERRQ(ierr);
    break;
  case 2:
    ierr = Compute_Lagrange_Basis_2D_Internal(nverts, coordinates, 1, quadrature, phypts,
           NULL, phibasis, NULL, NULL, jacobian, ijacobian, volume);CHKERRQ(ierr);
    break;
  case 3:
    ierr = Compute_Lagrange_Basis_3D_Internal(nverts, coordinates, 1, quadrature, phypts,
           NULL, phibasis, NULL, NULL, NULL, jacobian, ijacobian, volume);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension; should be in [1,3] : %D", dim);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "DMMoabPToRMapping"
PetscErrorCode DMMoabPToRMapping( const PetscInt dim, const PetscInt nverts, const PetscReal *coordinates, const PetscReal* xphy, PetscReal* natparam, PetscReal* phi)
{
  // Perform inverse evaluation for the mapping with use of Newton Raphson iteration
  const PetscReal tol = 1.0e-10;
  const PetscInt max_iterations = 10;
  const PetscReal error_tol_sqr = tol*tol;
  PetscReal phibasis[8], jacobian[9], ijacobian[9], volume;
  PetscReal phypts[3] = {0.0, 0.0, 0.0};
  PetscReal delta[3] = {0.0, 0.0, 0.0};
  PetscErrorCode  ierr;
  PetscInt iters=0;
  PetscReal error=1.0;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidPointer(xphy, 4);
  PetscValidPointer(natparam, 5);

  ierr = PetscMemzero(natparam, 3 * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(jacobian, dim * dim * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(ijacobian, dim * dim * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(phibasis, nverts * sizeof(PetscReal));CHKERRQ(ierr);

  /* Compute delta = evaluate( xi ) - x */
  ierr = FEMComputeBasis_JandF ( dim, nverts, coordinates, natparam, phypts, phi, jacobian, ijacobian, &volume );CHKERRQ(ierr);

  delta[0] = phypts[0] - xphy[0];
  delta[1] = phypts[1] - xphy[1];
  delta[2] = phypts[2] - xphy[2];
  error = (delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
  
  while (error > error_tol_sqr) {

    if(++iters > max_iterations)
      SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Maximum Newton iterations (10) reached. Current point in reference space : (%g, %g, %g)", natparam[0], natparam[1], natparam[2]);

    /* Compute natparam -= J.inverse() * delta */
    switch(dim) {
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

    /* Compute delta = evaluate( xi ) - x */
    ierr = FEMComputeBasis_JandF ( dim, nverts, coordinates, natparam, phypts, phi, jacobian, ijacobian, &volume );CHKERRQ(ierr);

    delta[0] = phypts[0] - xphy[0];
    delta[1] = phypts[1] - xphy[1];
    delta[2] = phypts[2] - xphy[2];
    error = (delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
  }
  if (phi) {
    ierr = PetscMemcpy(phi, phibasis, nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

