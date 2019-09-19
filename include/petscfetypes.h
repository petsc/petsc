#if !defined(PETSCFETYPES_H)
#define PETSCFETYPES_H

/*S
  PetscSpace - PETSc object that manages a linear space, e.g. the space of d-dimensional polynomials of given degree

  Level: beginner

.seealso: PetscSpaceCreate(), PetscDualSpaceCreate(), PetscSpaceSetType(), PetscSpaceType
S*/
typedef struct _p_PetscSpace *PetscSpace;

/*MC
  PetscSpacePolynomialType - The type of polynomial space

  Notes:
$ PETSCSPACE_POLYNOMIALTYPE_P - This is the normal polynomial space of degree q, P_q or Q_q.
$ PETSCSPACE_POLYNOMIALTYPE_PMINUS_HDIV - This is the smallest polynomial space contained in P_q/Q_q such that the divergence is in P_{q-1}/Q_{q-1}. Making this space is straightforward:
$   P^-_q = P_{q-1} + P_{(q-1)} x
$ where P_{(q-1)} is the space of homogeneous polynomials of degree q-1.
$ PETSCSPACE_POLYNOMIALTYPE_PMINUS_HCURL - This is the smallest polynomial space contained in P_q/Q_q such that the curl is in P_{q-1}/Q_{q-1}. Making this space is straightforward:
$   P^-_q = P_{q-1} + P_{(q-1)} rot x
$ where P_{(q-1)} is the space of homogeneous polynomials of degree q-1, and rot x is (-y, x) in 2D, and (z - y, x - z, y - x) in 3D, being the generators of the rotation algebra.

  Level: beginner

.seealso: PetscSpace
M*/
typedef enum { PETSCSPACE_POLYNOMIALTYPE_P, PETSCSPACE_POLYNOMIALTYPE_PMINUS_HDIV, PETSCSPACE_POLYNOMIALTYPE_PMINUS_HCURL } PetscSpacePolynomialType;
PETSC_EXTERN const char * const PetscSpacePolynomialTypes[];

/*S
  PetscDualSpace - PETSc object that manages the dual space to a linear space, e.g. the space of evaluation functionals at the vertices of a triangle

  Level: beginner

.seealso: PetscDualSpaceCreate(), PetscSpaceCreate(), PetscDualSpaceSetType(), PetscDualSpaceType
S*/
typedef struct _p_PetscDualSpace *PetscDualSpace;

/*MC
  PetscDualSpaceReferenceCell - The type of reference cell

  Notes: This is used only for automatic creation of reference cells. A PetscDualSpace can accept an arbitary DM for a reference cell.

  Level: beginner

.seealso: PetscSpace
M*/
typedef enum { PETSCDUALSPACE_REFCELL_SIMPLEX, PETSCDUALSPACE_REFCELL_TENSOR } PetscDualSpaceReferenceCell;
PETSC_EXTERN const char * const PetscDualSpaceReferenceCells[];

/*MC
  PetscDualSpaceTransformType - The type of function transform

  Notes: These transforms, and their inverses, are used to move functions and functionals between the reference element and real space. Suppose that we have a mapping $\phi$ which maps the reference cell to real space, and its Jacobian $J$. If we want to transform function $F$ on the reference element, so that it acts on real space, we use the pushforward transform $\sigma^*$. The pullback $\sigma_*$ is the inverse transform.

$ Covariant Piola: $\sigma^*(F) = J^{-T} F \circ \phi^{-1)$
$ Contravariant Piola: $\sigma^*(F) = 1/|J| J F \circ \phi^{-1)$

  Note: For details, please see Rognes, Kirby, and Logg, Efficient Assembly of Hdiv and Hrot Conforming Finite Elements, SISC, 31(6), 4130-4151, arXiv 1205.3085, 2010

  Level: beginner

.seealso: PetscDualSpaceGetDeRahm()
M*/
typedef enum {IDENTITY_TRANSFORM, COVARIANT_PIOLA_TRANSFORM, CONTRAVARIANT_PIOLA_TRANSFORM} PetscDualSpaceTransformType;

/*S
  PetscFE - PETSc object that manages a finite element space, e.g. the P_1 Lagrange element

  Level: beginner

.seealso: PetscFECreate(), PetscSpaceCreate(), PetscDualSpaceCreate(), PetscFESetType(), PetscFEType
S*/
typedef struct _p_PetscFE *PetscFE;

/*MC
  PetscFEJacobianType - indicates which pointwise functions should be used to fill the Jacobian matrix

  Level: beginner

.seealso: PetscFEIntegrateJacobian()
M*/
typedef enum { PETSCFE_JACOBIAN, PETSCFE_JACOBIAN_PRE, PETSCFE_JACOBIAN_DYN } PetscFEJacobianType;

#endif
