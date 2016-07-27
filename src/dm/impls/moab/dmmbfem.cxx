
#include <petscconf.h>
#include <petscdt.h>            /*I "petscdt.h" I*/
#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/

/* Utility functions */
static inline PetscReal DMatrix_Determinant_2x2_Internal ( const PetscReal inmat[2 * 2] )
{
  return  inmat[0] * inmat[3] - inmat[1] * inmat[2];
}

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

static inline double DMatrix_Determinant_3x3_Internal ( const PetscReal inmat[3 * 3] )
{
  return   inmat[0] * (inmat[8] * inmat[4] - inmat[7] * inmat[5])
           - inmat[3] * (inmat[8] * inmat[1] - inmat[7] * inmat[2])
           + inmat[6] * (inmat[5] * inmat[1] - inmat[4] * inmat[2]);
}

static inline PetscErrorCode DMatrix_Invert_3x3_Internal (const PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_POINTER, "Invalid input matrix specified for 3x3 inversion.");
  double det = DMatrix_Determinant_3x3_Internal(inmat);
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
    PetscReal *jxw, PetscReal *phi, PetscReal *dphidx)
{
  int i, j;
  PetscReal jacobian, ijacobian;
  PetscErrorCode  ierr;
  PetscFunctionBegin;

  ierr = PetscMemzero(phypts, npts * 3 * sizeof(PetscReal));CHKERRQ(ierr);
  if (dphidx) { /* Reset arrays. */
    ierr = PetscMemzero(dphidx, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (nverts == 2) { /* Linear Edge */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const double r = quad[j];

      phi[0 + offset] = 0.5 * ( 1.0 - r );
      phi[1 + offset] = 0.5 * ( 1.0 + r );

      const double dNi_dxi[2]  = { -0.5, 0.5 };

      jacobian = ijacobian = 0.0;
      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertices = coords + i * 3;
        jacobian += dNi_dxi[i] * vertices[0];
        for (int k = 0; k < 3; ++k)
          phypts[3 * j + k] += phi[i + offset] * vertices[k];
      }

      /* invert the jacobian */
      ijacobian = 1.0 / jacobian;

      jxw[j] *= jacobian;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian;
      }

    }
  }
  else if (nverts == 3) { /* Quadratic Edge */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const double r = quad[j];

      phi[0 + offset] = 0.5 * r * (r - 1.0);
      phi[1 + offset] = 0.5 * r * (r + 1.0);
      phi[2 + offset] = ( 1.0 - r * r );

      const double dNi_dxi[3]  = { r - 0.5, r + 0.5, -2.0 * r};

      jacobian = ijacobian = 0.0;
      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertices = coords + i * 3;
        jacobian += dNi_dxi[i] * vertices[0];
        for (int k = 0; k < 3; ++k)
          phypts[3 * j + k] += phi[i + offset] * vertices[k];
      }

      /* invert the jacobian */
      ijacobian = 1.0 / jacobian;

      jxw[j] *= jacobian;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian;
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
    PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy)
{
  int i, j;
  PetscReal jacobian[4], ijacobian[4];
  double jacobiandet;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(phi, npts * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(phypts, npts * 3 * sizeof(PetscReal));CHKERRQ(ierr);
  if (dphidx) { /* Reset arrays. */
    ierr = PetscMemzero(dphidx, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidy, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (nverts == 4) { /* Linear Quadrangle */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const double r = quad[0 + j * 2];
      const double s = quad[1 + j * 2];

      phi[0 + offset] = ( 1.0 - r ) * ( 1.0 - s );
      phi[1 + offset] =         r   * ( 1.0 - s );
      phi[2 + offset] =         r   *         s;
      phi[3 + offset] = ( 1.0 - r ) *         s;

      const double dNi_dxi[4]  = { -1.0 + s, 1.0 - s, s, -s };
      const double dNi_deta[4] = { -1.0 + r, -r, r, 1.0 - r };

      ierr = PetscMemzero(jacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(ijacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);
      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertices = coords + i * 3;
        jacobian[0] += dNi_dxi[i] * vertices[0];
        jacobian[2] += dNi_dxi[i] * vertices[1];
        jacobian[1] += dNi_deta[i] * vertices[0];
        jacobian[3] += dNi_deta[i] * vertices[1];
        for (int k = 0; k < 3; ++k)
          phypts[3 * j + k] += phi[i + offset] * vertices[k];
      }

      /* invert the jacobian */
      ierr = DMatrix_Invert_2x2_Internal(jacobian, ijacobian, &jacobiandet);CHKERRQ(ierr);

      jxw[j] *= jacobiandet;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        for (int k = 0; k < 2; ++k) {
          if (dphidx) dphidx[i + offset] += dNi_dxi[i] * ijacobian[k * 2 + 0];
          if (dphidy) dphidy[i + offset] += dNi_deta[i] * ijacobian[k * 2 + 1];
        }
      }

    }
  }
  else if (nverts == 3) { /* Linear triangle */

    ierr = PetscMemzero(jacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(ijacobian, 4 * sizeof(PetscReal));CHKERRQ(ierr);

    /* Jacobian is constant */
    jacobian[0] = (coords[0 * 3 + 0] - coords[2 * 3 + 0]); jacobian[1] = (coords[1 * 3 + 0] - coords[2 * 3 + 0]);
    jacobian[2] = (coords[0 * 3 + 1] - coords[2 * 3 + 1]); jacobian[3] = (coords[1 * 3 + 1] - coords[2 * 3 + 1]);

    /* invert the jacobian */
    ierr = DMatrix_Invert_2x2_Internal(jacobian, ijacobian, &jacobiandet);CHKERRQ(ierr);
    // std::cout << "Triangle area = " << jacobiandet << "\n";
    if ( jacobiandet < 1e-8 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Triangular element has zero volume: %g. Degenerate element or invalid connectivity\n", jacobiandet);

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const double r = quad[0 + j * 2];
      const double s = quad[1 + j * 2];

      jxw[j] *= 0.5;

      // const double Ni[3]  = { r, s, 1.0 - r - s };
      // for (i = 0; i < nverts; ++i) {
      //   const PetscScalar* vertices = coords+i*3;
      //   for (int k = 0; k < 3; ++k)
      //     phypts[offset+k] += Ni[i] * vertices[k];
      // }
      phypts[offset + 0] = coords[2 * 3 + 0] + jacobian[0] * r + jacobian[1] * s;
      phypts[offset + 1] = coords[2 * 3 + 1] + jacobian[2] * r + jacobian[3] * s;

      phi[0 + offset] = (  jacobian[3] * (phypts[offset + 0] - coords[2 * 3 + 0]) - jacobian[1] * (phypts[offset + 1] - coords[2 * 3 + 1]) ) / jacobiandet; // (b.y − c.y) x + (b.x − c.x) y + c.x b.y − b.x c.y
      phi[1 + offset] = ( -jacobian[2] * (phypts[offset + 0] - coords[2 * 3 + 0]) + jacobian[0] * (phypts[offset + 1] - coords[2 * 3 + 1]) ) / jacobiandet; // (c.y − a.y) x + (a.x − c.x) y + c.x a.y − a.x c.y
      phi[2 + offset] = 1.0 - phi[0 + offset] - phi[1 + offset];

      if (dphidx) {
        dphidx[0 + offset] = jacobian[3] / jacobiandet;
        dphidx[1 + offset] = -jacobian[2] / jacobiandet;
        dphidx[2 + offset] = -dphidx[0 + offset] - dphidx[1 + offset];
      }

      if (dphidy) {
        dphidy[0 + offset] = -jacobian[1] / jacobiandet;
        dphidy[1 + offset] = jacobian[0] / jacobiandet;
        dphidy[2 + offset] = -dphidy[0 + offset] - dphidy[1 + offset];
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
    PetscReal *jxw, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy, PetscReal *dphidz)
{
  PetscReal volume;
  int i, j;
  PetscReal jacobian[9], ijacobian[9];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Reset arrays. */
  ierr = PetscMemzero(phi, npts * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(phypts, npts * 3 * sizeof(PetscReal));CHKERRQ(ierr);
  if (dphidx) {
    ierr = PetscMemzero(dphidx, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidy, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidz, npts * nverts * sizeof(PetscReal));CHKERRQ(ierr);
  }

  if (nverts == 8) { /* Linear Hexahedra */

    for (j = 0; j < npts; j++)
    {
      const int offset = j * nverts;
      const double& r = quad[j * 3 + 0];
      const double& s = quad[j * 3 + 1];
      const double& t = quad[j * 3 + 2];

      phi[offset + 0] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 - t ) / 8;
      phi[offset + 1] = ( 1.0 + r ) * ( 1.0 - s ) * ( 1.0 - t ) / 8;
      phi[offset + 2] = ( 1.0 + r ) * ( 1.0 + s ) * ( 1.0 - t ) / 8;
      phi[offset + 3] = ( 1.0 - r ) * ( 1.0 + s ) * ( 1.0 - t ) / 8;
      phi[offset + 4] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 + t ) / 8;
      phi[offset + 5] = ( 1.0 + r ) * ( 1.0 - s ) * ( 1.0 + t ) / 8;
      phi[offset + 6] = ( 1.0 + r ) * ( 1.0 + s ) * ( 1.0 + t ) / 8;
      phi[offset + 7] = ( 1.0 - r ) * ( 1.0 + s ) * ( 1.0 + t ) / 8;

      const double dNi_dxi[8]  = { - ( 1.0 - s ) * ( 1.0 - t ),
                                   ( 1.0 - s ) * ( 1.0 - t ),
                                   ( 1.0 + s ) * ( 1.0 - t ),
                                   - ( 1.0 + s ) * ( 1.0 - t ),
                                   - ( 1.0 - s ) * ( 1.0 + t ),
                                   ( 1.0 - s ) * ( 1.0 + t ),
                                   ( 1.0 + s ) * ( 1.0 + t ),
                                   - ( 1.0 + s ) * ( 1.0 + t )
                                 };

      const double dNi_deta[8]  = { - ( 1.0 - r ) * ( 1.0 - t ),
                                    - ( 1.0 + r ) * ( 1.0 - t ),
                                    ( 1.0 + r ) * ( 1.0 - t ),
                                    ( 1.0 - r ) * ( 1.0 - t ),
                                    - ( 1.0 - r ) * ( 1.0 + t ),
                                    - ( 1.0 + r ) * ( 1.0 + t ),
                                    ( 1.0 + r ) * ( 1.0 + t ),
                                    ( 1.0 - r ) * ( 1.0 + t )
                                  };

      const double dNi_dzeta[8]  = { - ( 1.0 - r ) * ( 1.0 - s ),
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
      double factor = 1.0 / 8;
      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertex = coords + i * 3;
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

      /* invert the jacobian */
      ierr = DMatrix_Invert_3x3_Internal(jacobian, ijacobian, &volume);CHKERRQ(ierr);

      jxw[j] *= factor * volume;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; ++i ) {
        const PetscScalar* vertex = coords + i * 3;
        for (int k = 0; k < 3; ++k) {
          phypts[3 * j + k] += phi[i + offset] * vertex[k];
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

    jacobian[0] = coords[1 * 3 + 0] - coords[0 * 3 + 0];  jacobian[1] = coords[2 * 3 + 0] - coords[0 * 3 + 0]; jacobian[2] = coords[3 * 3 + 0] - coords[0 * 3 + 0];
    jacobian[3] = coords[1 * 3 + 1] - coords[0 * 3 + 1];  jacobian[4] = coords[2 * 3 + 1] - coords[0 * 3 + 1]; jacobian[5] = coords[3 * 3 + 1] - coords[0 * 3 + 1];
    jacobian[6] = coords[1 * 3 + 2] - coords[0 * 3 + 2];  jacobian[7] = coords[2 * 3 + 2] - coords[0 * 3 + 2]; jacobian[8] = coords[3 * 3 + 2] - coords[0 * 3 + 2];

    /* invert the jacobian */
    ierr = DMatrix_Invert_3x3_Internal(jacobian, ijacobian, &volume);CHKERRQ(ierr);

    if ( volume < 1e-8 ) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Tetrahedral element has zero volume: %g. Degenerate element or invalid connectivity\n", volume);

    for ( j = 0; j < npts; j++ )
    {
      const int offset = j * nverts;
      const double factor = 1.0 / 6;
      const double& r = quad[j * 3 + 0];
      const double& s = quad[j * 3 + 1];
      const double& t = quad[j * 3 + 2];

      jxw[j] *= factor * volume;

      phi[offset + 0] = 1.0 - r - s - t;
      phi[offset + 1] = r;
      phi[offset + 2] = s;
      phi[offset + 3] = t;

      if (dphidx) {
        dphidx[0 + offset] = ( coords[1 + 2 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                               - coords[1 + 1 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                               - coords[1 + 3 * 3] * ( coords[2 + 1 * 3] - coords[2 + 2 * 3] )
                             ) / volume;
        dphidx[1 + offset] = -( coords[1 + 2 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                                - coords[1 + 0 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                                - coords[1 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 2 * 3] )
                              ) / volume;
        dphidx[2 + offset] = ( coords[1 + 1 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                               - coords[1 + 0 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                               - coords[1 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 1 * 3] )
                             ) / volume;
        dphidx[3 + offset] = -dphidx[0 + offset] - dphidx[1 + offset] - dphidx[2 + offset];
      }

      if (dphidy) {
        dphidy[0 + offset] = ( coords[0 + 1 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                               - coords[0 + 2 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                               + coords[0 + 3 * 3] * ( coords[2 + 1 * 3] - coords[2 + 2 * 3] )
                             ) / volume;
        dphidy[1 + offset] = -( coords[0 + 0 * 3] * ( coords[2 + 2 * 3] - coords[2 + 3 * 3] )
                                - coords[0 + 2 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                                + coords[0 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 2 * 3] )
                              ) / volume;
        dphidy[2 + offset] = ( coords[0 + 0 * 3] * ( coords[2 + 1 * 3] - coords[2 + 3 * 3] )
                               - coords[0 + 1 * 3] * ( coords[2 + 0 * 3] - coords[2 + 3 * 3] )
                               + coords[0 + 3 * 3] * ( coords[2 + 0 * 3] - coords[2 + 1 * 3] )
                             ) / volume;
        dphidy[3 + offset] = -dphidy[0 + offset] - dphidy[1 + offset] - dphidy[2 + offset];
      }


      if (dphidz) {
        dphidz[0 + offset] = ( coords[0 + 1 * 3] * (coords[1 + 3 * 3] - coords[1 + 2 * 3])
                               - coords[0 + 2 * 3] * (coords[1 + 3 * 3] - coords[1 + 1 * 3])
                               + coords[0 + 3 * 3] * (coords[1 + 2 * 3] - coords[1 + 1 * 3])
                             ) / volume;
        dphidz[1 + offset] = -( coords[0 + 0 * 3] * (coords[1 + 3 * 3] - coords[1 + 2 * 3])
                                + coords[0 + 2 * 3] * (coords[1 + 0 * 3] - coords[1 + 3 * 3])
                                - coords[0 + 3 * 3] * (coords[1 + 0 * 3] - coords[1 + 2 * 3])
                              ) / volume;
        dphidz[2 + offset] = ( coords[0 + 0 * 3] * (coords[1 + 3 * 3] - coords[1 + 1 * 3])
                               + coords[0 + 1 * 3] * (coords[1 + 0 * 3] - coords[1 + 3 * 3])
                               - coords[0 + 3 * 3] * (coords[1 + 0 * 3] - coords[1 + 1 * 3])
                             ) / volume;
        dphidz[3 + offset] = -dphidz[0 + offset] - dphidz[1 + offset] - dphidz[2 + offset];
      }

      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertices = coords + i * 3;
        for (int k = 0; k < 3; ++k)
          phypts[3 * j + k] += phi[i + offset] * vertices[k];
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
PetscErrorCode DMMoabFEMComputeBasis ( PetscInt dim, PetscInt nverts, PetscReal *coordinates, PetscQuadrature quadrature, PetscReal *phypts,
                                       PetscReal *jacobian_quadrature_weight_product, PetscReal *fe_basis, PetscReal **fe_basis_derivatives)
{
  PetscErrorCode  ierr;
  PetscInt        npoints;
  bool            compute_der;
  const PetscReal *quadpts, *quadwts;

  PetscFunctionBegin;
  PetscValidPointer(coordinates, 3);
  PetscValidHeaderSpecific(quadrature, PETSC_OBJECT_CLASSID, 4);
  PetscValidPointer(phypts, 5);
  PetscValidPointer(jacobian_quadrature_weight_product, 6);
  PetscValidPointer(fe_basis, 7);
  compute_der = (fe_basis_derivatives != NULL);

  /* Get the quadrature points and weights for the given quadrature rule */
  ierr = PetscQuadratureGetData(quadrature, NULL, &npoints, &quadpts, &quadwts);CHKERRQ(ierr);
  ierr = PetscMemcpy(jacobian_quadrature_weight_product, quadwts, npoints * sizeof(PetscReal));CHKERRQ(ierr);

  switch (dim) {
  case 1:
    ierr = Compute_Lagrange_Basis_1D_Internal(nverts, coordinates, npoints, quadpts, phypts,
           jacobian_quadrature_weight_product, fe_basis,
           (compute_der ? fe_basis_derivatives[0] : NULL));CHKERRQ(ierr);
    break;
  case 2:
    ierr = Compute_Lagrange_Basis_2D_Internal(nverts, coordinates, npoints, quadpts, phypts,
           jacobian_quadrature_weight_product, fe_basis,
           (compute_der ? fe_basis_derivatives[0] : NULL),
           (compute_der ? fe_basis_derivatives[1] : NULL));CHKERRQ(ierr);
    break;
  case 3:
    ierr = Compute_Lagrange_Basis_3D_Internal(nverts, coordinates, npoints, quadpts, phypts,
           jacobian_quadrature_weight_product, fe_basis,
           (compute_der ? fe_basis_derivatives[0] : NULL),
           (compute_der ? fe_basis_derivatives[1] : NULL),
           (compute_der ? fe_basis_derivatives[2] : NULL));CHKERRQ(ierr);
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
PetscErrorCode DMMoabFEMCreateQuadratureDefault ( PetscInt dim, PetscInt nverts, PetscQuadrature *quadrature )
{
  PetscReal *w, *x;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create an appropriate quadrature rule to sample basis */
  switch (dim)
  {
  case 1:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    ierr = PetscDTGaussJacobiQuadrature(1, nverts, -1.0, 1.0, quadrature);CHKERRQ(ierr);
    break;
  case 2:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    if (nverts == 3) {
      const int order = 2;
      const int npoints = (order == 2 ? 3 : 6);
      ierr = PetscMalloc2(npoints * 2, &x, npoints, &w);CHKERRQ(ierr);
      switch (npoints) {
      case 3:
        x[0] = x[1] = x[2] = x[5] = 1.0 / 6.0;
        x[3] = x[4] = 2.0 / 3.0;
        w[0] = w[1] = w[2] = 1.0 / 3.0;
        break;
      case 6:
        x[0] = x[1] = x[2] = 0.44594849091597;
        x[3] = x[4] = 0.10810301816807;
        x[5] = 0.44594849091597;
        x[6] = x[7] = x[8] = 0.09157621350977;
        x[9] = x[10] = 0.81684757298046;
        x[11] = 0.09157621350977;
        w[0] = w[1] = w[2] = 0.22338158967801;
        w[0] = w[1] = w[2] = 0.10995174365532;
        break;
      }
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, quadrature);CHKERRQ(ierr);
      ierr = PetscQuadratureSetOrder(*quadrature, order);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(*quadrature, 2, npoints, x, w);CHKERRQ(ierr);
      //ierr = PetscDTGaussJacobiQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr);
    }
    else {
      ierr = PetscDTGaussTensorQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr);
    }
    break;
  case 3:
    /* Create Gauss quadrature rules with <order = nverts> in the span [-1, 1] */
    if (nverts == 4) {
      ierr = PetscDTGaussJacobiQuadrature(dim, nverts, 0.0, 1.0, quadrature);CHKERRQ(ierr);
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

