#!/usr/bin/env python
import user
#import importer
import script

femIntegrationCode = r'''
#undef __FUNCT__
#define __FUNCT__ "FEMIntegrateResidualBatch"
/*C
  FEMIntegrateResidualBatch - Produce the element residual vector for a batch of elements by quadrature integration

  Not collective

  Input Parameters:
+ Ne                   - The number of elements in the batch
. numFields            - The number of physical fields
. field                - The field being integrated
. quad                 - PetscQuadrature objects for each field
. coefficients         - The array of FEM basis coefficients for the elements
. v0s                  - The coordinates of the initial vertex for each element (the constant part of the transform from the reference element)
. jacobians            - The Jacobian for each element (the linear part of the transform from the reference element)
. jacobianInverses     - The Jacobian inverse for each element (the linear part of the transform to the reference element)
. jacobianDeterminants - The Jacobian determinant for each element
. f0_func              - f_0 function from the first order FEM model
- f1_func              - f_1 function from the first order FEM model

  Output Parameter
. elemVec              - the element residual vectors from each element

   Calling sequence of f0_func and f1_func:
$    void f0(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])

  Note:
$ Loop over batch of elements (e):
$   Loop over quadrature points (q):
$     Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
$     Call f_0 and f_1
$   Loop over element vector entries (f,fc --> i):
$     elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)
*/
PetscErrorCode FEMIntegrateResidualBatch(PetscInt Ne, PetscInt numFields, PetscInt field, PetscQuadrature quad[], const PetscScalar coefficients[],
                                         const PetscReal v0s[], const PetscReal jacobians[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[],
                                         void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]),
                                         void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]), PetscScalar elemVec[])
{
  const PetscInt debug   = 0;
  const PetscInt dim     = SPATIAL_DIM_0;
  const PetscInt numComponents = NUM_BASIS_COMPONENTS_TOTAL;
  PetscInt       cOffset = 0;
  PetscInt       eOffset = 0, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(IntegrateResidualEvent,0,0,0,0);CHKERRQ(ierr); */
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ = jacobianDeterminants[e];
    const PetscReal *v0   = &v0s[e*dim];
    const PetscReal *J    = &jacobians[e*dim*dim];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    const PetscInt   Nq   = quad[field].numQuadPoints;
    PetscScalar      f0[NUM_QUADRATURE_POINTS_0*dim];
    PetscScalar      f1[NUM_QUADRATURE_POINTS_0*dim*dim];
    PetscInt         q, f;

    if (Nq > NUM_QUADRATURE_POINTS_0) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Number of quadrature points %d should be <= %d", Nq, NUM_QUADRATURE_POINTS_0);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar      u[NUM_BASIS_COMPONENTS_TOTAL];
      PetscScalar      gradU[dim*(NUM_BASIS_COMPONENTS_TOTAL)];
      PetscReal        x[SPATIAL_DIM_0];
      PetscInt         fOffset     = 0;
      PetscInt         dOffset     = cOffset;
      const PetscInt   Ncomp       = quad[field].numComponents;
      const PetscReal *quadPoints  = quad[field].quadPoints;
      const PetscReal *quadWeights = quad[field].quadWeights;
      PetscInt         d, d2, f, i;

      for (d = 0; d < numComponents; ++d)       {u[d]     = 0.0;}
      for (d = 0; d < dim*(numComponents); ++d) {gradU[d] = 0.0;}
      for (d = 0; d < dim; ++d) {
        x[d] = v0[d];
        for (d2 = 0; d2 < dim; ++d2) {
          x[d] += J[d*dim+d2]*(quadPoints[q*dim+d2] + 1.0);
        }
      }
      for (f = 0; f < numFields; ++f) {
        const PetscInt   Nb       = quad[f].numBasisFuncs;
        const PetscInt   Ncomp    = quad[f].numComponents;
        const PetscReal *basis    = quad[f].basis;
        const PetscReal *basisDer = quad[f].basisDer;
        PetscInt         b, comp;

        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscScalar    realSpaceDer[dim];
            PetscInt       d, g;

            u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradU[(fOffset+comp)*dim+d] += coefficients[dOffset+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
            }
          }
        }
        fOffset += Ncomp;
        dOffset += Nb*Ncomp;
      }

      f0_func(u, gradU, x, &f0[q*Ncomp]);
      for (i = 0; i < Ncomp; ++i) {
        f0[q*Ncomp+i] *= detJ*quadWeights[q];
      }
      f1_func(u, gradU, x, &f1[q*Ncomp*dim]);
      for (i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        PetscInt c,d;
        for (c = 0; c < Ncomp; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, f0[q*Ncomp+c]);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, f1[(q*Ncomp + c)*dim+d]);CHKERRQ(ierr);
          }
        }
      }
      if (q == Nq-1) {cOffset = dOffset;}
    }
    for (f = 0; f < numFields; ++f) {
      const PetscInt   Nq       = quad[f].numQuadPoints;
      const PetscInt   Nb       = quad[f].numBasisFuncs;
      const PetscInt   Ncomp    = quad[f].numComponents;
      const PetscReal *basis    = quad[f].basis;
      const PetscReal *basisDer = quad[f].basisDer;
      PetscInt         b, comp;

      if (f == field) {
      for (b = 0; b < Nb; ++b) {
        for (comp = 0; comp < Ncomp; ++comp) {
          const PetscInt cidx = b*Ncomp+comp;
          PetscInt       q;

          elemVec[eOffset+cidx] = 0.0;
          for (q = 0; q < Nq; ++q) {
            PetscScalar realSpaceDer[dim];
            PetscInt    d, g;

            elemVec[eOffset+cidx] += basis[q*Nb*Ncomp+cidx]*f0[q*Ncomp+comp];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              elemVec[eOffset+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
            }
          }
        }
      }
      if (debug > 1) {
        PetscInt b, comp;

        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, comp, elemVec[eOffset+b*Ncomp+comp]);CHKERRQ(ierr);
          }
        }
      }
      }
      eOffset += Nb*Ncomp;
    }
  }
  /* TODO ierr = PetscLogFlops();CHKERRQ(ierr); */
  /* ierr = PetscLogEventEnd(IntegrateResidualEvent,0,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FEMIntegrateJacobianActionBatch"
/*C
  FEMIntegrateJacobianActionBatch - Produce the action of the element Jacobian on an element vector for a batch of elements by quadrature integration

  Not collective

  Input Parameters:
+ Ne                   - The number of elements in the batch
. numFields            - The number of physical fields
. fieldI               - The field being integrated
. quad                 - PetscQuadrature objects for each field
. coefficients         - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. argCoefficients      - The array of FEM basis coefficients for the elements for the argument vector
. v0s                  - The coordinates of the initial vertex for each element (the constant part of the transform from the reference element)
. jacobians            - The Jacobian for each element (the linear part of the transform from the reference element)
. jacobianInverses     - The Jacobian inverse for each element (the linear part of the transform to the reference element)
. jacobianDeterminants - The Jacobian determinant for each element
. g0_func              - g_0 function from the first order FEM model
. g1_func              - g_1 function from the first order FEM model
. g2_func              - g_2 function from the first order FEM model
- g3_func              - g_3 function from the first order FEM model

  Output Parameter
. elemVec              - the element vectors for the Jacobian action from each element

   Calling sequence of g0_func, g1_func, g2_func and g3_func:
$    void g0(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])

  Note:
$ Loop over batch of elements (e):
$   Loop over element vector entries (f,fc --> i):
$     Sum over element matrix columns entries (g,gc --> j):
$       Loop over quadrature points (q):
$         Make u_q and gradU_q (loops over fields,Nb,Ncomp)
$           elemVec[i] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
$                      + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
*/
PetscErrorCode FEMIntegrateJacobianActionBatch(PetscInt Ne, PetscInt numFields, PetscInt fieldI, PetscQuadrature quad[], const PetscScalar coefficients[], const PetscScalar argCoefficients[],
                                               const PetscReal v0s[], const PetscReal jacobians[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[],
                                               void (**g0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g0[]),
                                               void (**g1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[]),
                                               void (**g2_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[]),
                                               void (**g3_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[]), PetscScalar elemVec[]) {
  const PetscReal *basisI    = quad[fieldI].basis;
  const PetscReal *basisDerI = quad[fieldI].basisDer;
  const PetscInt   debug   = 0;
  const PetscInt   dim     = SPATIAL_DIM_0;
  const PetscInt numComponents = NUM_BASIS_COMPONENTS_TOTAL;
  PetscInt         cellDof = 0; /* Total number of dof on a cell */
  PetscInt         cOffset = 0; /* Offset into coefficients[], argCoefficients[], elemVec[] for element e */
  PetscInt         offsetI = 0; /* Offset into an element vector for fieldI */
  PetscInt         fieldJ, offsetJ, field, e;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(IntegrateJacActionEvent,0,0,0,0);CHKERRQ(ierr); */
  for (field = 0; field < numFields; ++field) {
    if (field == fieldI) {offsetI = cellDof;}
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ    = jacobianDeterminants[e];
    const PetscReal *v0      = &v0s[e*dim];
    const PetscReal *J       = &jacobians[e*dim*dim];
    const PetscReal *invJ    = &jacobianInverses[e*dim*dim];
    const PetscInt   Nb_i    = quad[fieldI].numBasisFuncs;
    const PetscInt   Ncomp_i = quad[fieldI].numComponents;
    PetscInt         f, fc, g, gc;

    for (f = 0; f < Nb_i; ++f) {
      const PetscInt   Nq          = quad[fieldI].numQuadPoints;
      const PetscReal *quadPoints  = quad[fieldI].quadPoints;
      const PetscReal *quadWeights = quad[fieldI].quadWeights;
      PetscInt         q;

      for (fc = 0; fc < Ncomp_i; ++fc) {
        const PetscInt fidx = f*Ncomp_i+fc; /* Test function basis index */
        const PetscInt i    = offsetI+fidx; /* Element vector row */
        elemVec[cOffset+i] = 0.0;
      }
      for (q = 0; q < Nq; ++q) {
        PetscScalar u[NUM_BASIS_COMPONENTS_TOTAL];
        PetscScalar gradU[dim*(NUM_BASIS_COMPONENTS_TOTAL)];
        PetscReal   x[SPATIAL_DIM_0];
        PetscInt    fOffset            = 0;       /* Offset into u[] for field_q (like offsetI) */
        PetscInt    dOffset            = cOffset; /* Offset into coefficients[] for field_q */
        PetscInt    field_q, d, d2;
        PetscScalar g0[dim*dim];         /* Ncomp_i*Ncomp_j */
        PetscScalar g1[dim*dim*dim];     /* Ncomp_i*Ncomp_j*dim */
        PetscScalar g2[dim*dim*dim];     /* Ncomp_i*Ncomp_j*dim */
        PetscScalar g3[dim*dim*dim*dim]; /* Ncomp_i*Ncomp_j*dim*dim */
        PetscInt    c;

        if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
        for (d = 0; d < numComponents; ++d)       {u[d]     = 0.0;}
        for (d = 0; d < dim*(numComponents); ++d) {gradU[d] = 0.0;}
        for (d = 0; d < dim; ++d) {
          x[d] = v0[d];
          for (d2 = 0; d2 < dim; ++d2) {
            x[d] += J[d*dim+d2]*(quadPoints[q*dim+d2] + 1.0);
          }
        }
        for (field_q = 0; field_q < numFields; ++field_q) {
          const PetscInt   Nb          = quad[field_q].numBasisFuncs;
          const PetscInt   Ncomp       = quad[field_q].numComponents;
          const PetscReal *basis       = quad[field_q].basis;
          const PetscReal *basisDer    = quad[field_q].basisDer;
          PetscInt         b, comp;

          for (b = 0; b < Nb; ++b) {
            for (comp = 0; comp < Ncomp; ++comp) {
              const PetscInt cidx = b*Ncomp+comp;
              PetscScalar    realSpaceDer[dim];
              PetscInt       d1, d2;

              u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
              for (d1 = 0; d1 < dim; ++d1) {
                realSpaceDer[d1] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  realSpaceDer[d1] += invJ[d2*dim+d1]*basisDer[(q*Nb*Ncomp+cidx)*dim+d2];
                }
                gradU[(fOffset+comp)*dim+d1] += coefficients[dOffset+cidx]*realSpaceDer[d1];
              }
            }
          }
          if (debug > 1) {
            for (comp = 0; comp < Ncomp; ++comp) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
              for (d = 0; d < dim; ++d) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
              }
            }
          }
          fOffset += Ncomp;
          dOffset += Nb*Ncomp;
        }

        for (fieldJ = 0, offsetJ = 0; fieldJ < numFields; offsetJ += quad[fieldJ].numBasisFuncs*quad[fieldJ].numComponents,  ++fieldJ) {
          const PetscReal *basisJ    = quad[fieldJ].basis;
          const PetscReal *basisDerJ = quad[fieldJ].basisDer;
          const PetscInt   Nb_j      = quad[fieldJ].numBasisFuncs;
          const PetscInt   Ncomp_j   = quad[fieldJ].numComponents;

          for (g = 0; g < Nb_j; ++g) {
            if ((Ncomp_i > dim) || (Ncomp_j > dim)) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Number of components %d and %d should be <= %d", Ncomp_i, Ncomp_j, dim);
            ierr = PetscMemzero(g0, Ncomp_i*Ncomp_j         * sizeof(PetscScalar));CHKERRQ(ierr);
            ierr = PetscMemzero(g1, Ncomp_i*Ncomp_j*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
            ierr = PetscMemzero(g2, Ncomp_i*Ncomp_j*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
            ierr = PetscMemzero(g3, Ncomp_i*Ncomp_j*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
            if (g0_func[fieldI*numFields+fieldJ]) {
              g0_func[fieldI*numFields+fieldJ](u, gradU, x, g0);
              for (c = 0; c < Ncomp_i*Ncomp_j; ++c) {
                g0[c] *= detJ*quadWeights[q];
              }
            }
            if (g1_func[fieldI*numFields+fieldJ]) {
              g1_func[fieldI*numFields+fieldJ](u, gradU, x, g1);
              for (c = 0; c < Ncomp_i*Ncomp_j*dim; ++c) {
                g1[c] *= detJ*quadWeights[q];
              }
            }
            if (g2_func[fieldI*numFields+fieldJ]) {
              g2_func[fieldI*numFields+fieldJ](u, gradU, x, g2);
              for (c = 0; c < Ncomp_i*Ncomp_j*dim; ++c) {
                g2[c] *= detJ*quadWeights[q];
              }
            }
            if (g3_func[fieldI*numFields+fieldJ]) {
              g3_func[fieldI*numFields+fieldJ](u, gradU, x, g3);
              for (c = 0; c < Ncomp_i*Ncomp_j*dim*dim; ++c) {
                g3[c] *= detJ*quadWeights[q];
              }
            }

            for (fc = 0; fc < Ncomp_i; ++fc) {
              const PetscInt fidx = f*Ncomp_i+fc; /* Test function basis index */
              const PetscInt i    = offsetI+fidx; /* Element matrix row */
              for (gc = 0; gc < Ncomp_j; ++gc) {
                const PetscInt gidx  = g*Ncomp_j+gc; /* Trial function basis index */
                const PetscInt j     = offsetJ+gidx; /* Element matrix column */
                PetscScalar    entry = 0.0;          /* The (i,j) entry in the element matrix */
                PetscScalar    realSpaceDerI[dim];
                PetscScalar    realSpaceDerJ[dim];
                PetscInt       d, d2;

                for (d = 0; d < dim; ++d) {
                  realSpaceDerI[d] = 0.0;
                  realSpaceDerJ[d] = 0.0;
                  for (d2 = 0; d2 < dim; ++d2) {
                    realSpaceDerI[d] += invJ[d2*dim+d]*basisDerI[(q*Nb_i*Ncomp_i+fidx)*dim+d2];
                    realSpaceDerJ[d] += invJ[d2*dim+d]*basisDerJ[(q*Nb_j*Ncomp_j+gidx)*dim+d2];
                  }
                }
                entry += basisI[q*Nb_i*Ncomp_i+fidx]*g0[fc*Ncomp_j+gc]*basisJ[q*Nb_j*Ncomp_j+gidx];
                for (d = 0; d < dim; ++d) {
                  entry += basisI[q*Nb_i*Ncomp_i+fidx]*g1[(fc*Ncomp_j+gc)*dim+d]*realSpaceDerJ[d];
                  entry += realSpaceDerI[d]*g2[(fc*Ncomp_j+gc)*dim+d]*basisJ[q*Nb_j*Ncomp_j+gidx];
                  for (d2 = 0; d2 < dim; ++d2) {
                    entry += realSpaceDerI[d]*g3[((fc*Ncomp_j+gc)*dim+d)*dim+d2]*realSpaceDerJ[d2];
                  }
                }
                elemVec[cOffset+i] += entry*argCoefficients[cOffset+j];
              }
            }
          }
        }
      }
    }
    if (debug > 1) {
      PetscInt fc, f;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element %d action vector for field %d\n", e, fieldI);CHKERRQ(ierr);
      for (fc = 0; fc < Ncomp_i; ++fc) {
        for (f = 0; f < Nb_i; ++f) {
          const PetscInt i = offsetI + f*Ncomp_i+fc;
          ierr = PetscPrintf(PETSC_COMM_SELF, "    argCoef[%d,%d]: %g\n", f, fc, argCoefficients[cOffset+i]);CHKERRQ(ierr);
        }
      }
      for (fc = 0; fc < Ncomp_i; ++fc) {
        for (f = 0; f < Nb_i; ++f) {
          const PetscInt i = offsetI + f*Ncomp_i+fc;
          ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", f, fc, elemVec[cOffset+i]);CHKERRQ(ierr);
        }
      }
    }
    cOffset += cellDof;
  }
  /* ierr = PetscLogEventEnd(IntegrateJacActionEvent,0,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FEMIntegrateJacobianBatch"
/*C
  FEMIntegrateJacobianActionBatch - Produce the action of the element Jacobian on an element vector for a batch of elements by quadrature integration

  Not collective

  Input Parameters:
+ Ne                   - The number of elements in the batch
. numFields            - The number of physical fields
. fieldI               - The test field being integrated
. fieldJ               - The basis field being integrated
. quad                 - PetscQuadrature objects for each field
. coefficients         - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. v0s                  - The coordinates of the initial vertex for each element (the constant part of the transform from the reference element)
. jacobians            - The Jacobian for each element (the linear part of the transform from the reference element)
. jacobianInverses     - The Jacobian inverse for each element (the linear part of the transform to the reference element)
. jacobianDeterminants - The Jacobian determinant for each element
. g0_func              - g_0 function from the first order FEM model
. g1_func              - g_1 function from the first order FEM model
. g2_func              - g_2 function from the first order FEM model
- g3_func              - g_3 function from the first order FEM model

  Output Parameter
. elemMat              - the element matrices for the Jacobian from each element

   Calling sequence of g0_func, g1_func, g2_func and g3_func:
$    void g0(PetscScalar u[], const PetscScalar gradU[], PetscScalar x[], PetscScalar f0[])

  Note:
$ Loop over batch of elements (e):
$   Loop over element matrix entries (f,fc,g,gc --> i,j):
$     Loop over quadrature points (q):
$       Make u_q and gradU_q (loops over fields,Nb,Ncomp)
$         elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
$                      + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
*/
PetscErrorCode FEMIntegrateJacobianBatch(PetscInt Ne, PetscInt numFields, PetscInt fieldI, PetscInt fieldJ, PetscQuadrature quad[], const PetscScalar coefficients[],
                                         const PetscReal v0s[], const PetscReal jacobians[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[],
                                         void (*g0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g0[]),
                                         void (*g1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[]),
                                         void (*g2_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[]),
                                         void (*g3_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[]), PetscScalar elemMat[]) {
  const PetscReal *basisI    = quad[fieldI].basis;
  const PetscReal *basisDerI = quad[fieldI].basisDer;
  const PetscReal *basisJ    = quad[fieldJ].basis;
  const PetscReal *basisDerJ = quad[fieldJ].basisDer;
  const PetscInt   debug   = 0;
  const PetscInt   dim     = SPATIAL_DIM_0;
  PetscInt         cellDof = 0; /* Total number of dof on a cell */
  PetscInt         cOffset = 0; /* Offset into coefficients[] for element e */
  PetscInt         eOffset = 0; /* Offset into elemMat[] for element e */
  PetscInt         offsetI = 0; /* Offset into an element vector for fieldI */
  PetscInt         offsetJ = 0; /* Offset into an element vector for fieldJ */
  PetscInt         field, e;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for (field = 0; field < numFields; ++field) {
    if (field == fieldI) {offsetI = cellDof;}
    if (field == fieldJ) {offsetJ = cellDof;}
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  /* ierr = PetscLogEventBegin(IntegrateJacobianEvent,0,0,0,0);CHKERRQ(ierr); */
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ    = jacobianDeterminants[e];
    const PetscReal *v0      = &v0s[e*dim];
    const PetscReal *J       = &jacobians[e*dim*dim];
    const PetscReal *invJ    = &jacobianInverses[e*dim*dim];
    const PetscInt   Nb_i    = quad[fieldI].numBasisFuncs;
    const PetscInt   Ncomp_i = quad[fieldI].numComponents;
    const PetscInt   Nb_j    = quad[fieldJ].numBasisFuncs;
    const PetscInt   Ncomp_j = quad[fieldJ].numComponents;
    PetscInt         f, g;

    for (f = 0; f < Nb_i; ++f) {
      for (g = 0; g < Nb_j; ++g) {
        const PetscInt   Nq          = quad[fieldI].numQuadPoints;
        const PetscReal *quadPoints  = quad[fieldI].quadPoints;
        const PetscReal *quadWeights = quad[fieldI].quadWeights;
        PetscInt         q;

        for (q = 0; q < Nq; ++q) {
          PetscScalar u[dim+1];
          PetscScalar gradU[dim*(dim+1)];
          PetscReal   x[SPATIAL_DIM_0];
          PetscInt    fOffset            = 0;       /* Offset into u[] for field_q (like offsetI) */
          PetscInt    dOffset            = cOffset; /* Offset into coefficients[] for field_q */
          PetscInt    field_q, d, d2;
          PetscScalar g0[dim*dim];         /* Ncomp_i*Ncomp_j */
          PetscScalar g1[dim*dim*dim];     /* Ncomp_i*Ncomp_j*dim */
          PetscScalar g2[dim*dim*dim];     /* Ncomp_i*Ncomp_j*dim */
          PetscScalar g3[dim*dim*dim*dim]; /* Ncomp_i*Ncomp_j*dim*dim */
          PetscInt    fc, gc, c;

          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
          for (d = 0; d <= dim; ++d)        {u[d]     = 0.0;}
          for (d = 0; d < dim*(dim+1); ++d) {gradU[d] = 0.0;}
          for (d = 0; d < dim; ++d) {
            x[d] = v0[d];
            for (d2 = 0; d2 < dim; ++d2) {
              x[d] += J[d*dim+d2]*(quadPoints[q*dim+d2] + 1.0);
            }
          }
          for (field_q = 0; field_q < numFields; ++field_q) {
            const PetscInt   Nb          = quad[field_q].numBasisFuncs;
            const PetscInt   Ncomp       = quad[field_q].numComponents;
            const PetscReal *basis       = quad[field_q].basis;
            const PetscReal *basisDer    = quad[field_q].basisDer;
            PetscInt         b, comp;

            for (b = 0; b < Nb; ++b) {
              for (comp = 0; comp < Ncomp; ++comp) {
                const PetscInt cidx = b*Ncomp+comp;
                PetscScalar    realSpaceDer[dim];
                PetscInt       d1, d2;

                u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
                for (d1 = 0; d1 < dim; ++d1) {
                  realSpaceDer[d1] = 0.0;
                  for (d2 = 0; d2 < dim; ++d2) {
                    realSpaceDer[d1] += invJ[d2*dim+d1]*basisDer[(q*Nb*Ncomp+cidx)*dim+d2];
                  }
                  gradU[(fOffset+comp)*dim+d1] += coefficients[dOffset+cidx]*realSpaceDer[d1];
                }
              }
            }
            if (debug > 1) {
              for (comp = 0; comp < Ncomp; ++comp) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
                for (d = 0; d < dim; ++d) {
                  ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
                }
              }
            }
            fOffset += Ncomp;
            dOffset += Nb*Ncomp;
          }

          if ((Ncomp_i > dim) || (Ncomp_j > dim)) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Number of components %d and %d should be <= %d", Ncomp_i, Ncomp_j, dim);
          ierr = PetscMemzero(g0, Ncomp_i*Ncomp_j         * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g1, Ncomp_i*Ncomp_j*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g2, Ncomp_i*Ncomp_j*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g3, Ncomp_i*Ncomp_j*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
          if (g0_func) {
            g0_func(u, gradU, x, g0);
            for (c = 0; c < Ncomp_i*Ncomp_j; ++c) {
              g0[c] *= detJ*quadWeights[q];
            }
          }
          if (g1_func) {
            g1_func(u, gradU, x, g1);
            for (c = 0; c < Ncomp_i*Ncomp_j*dim; ++c) {
              g1[c] *= detJ*quadWeights[q];
            }
          }
          if (g2_func) {
            g2_func(u, gradU, x, g2);
            for (c = 0; c < Ncomp_i*Ncomp_j*dim; ++c) {
              g2[c] *= detJ*quadWeights[q];
            }
          }
          if (g3_func) {
            g3_func(u, gradU, x, g3);
            for (c = 0; c < Ncomp_i*Ncomp_j*dim*dim; ++c) {
              g3[c] *= detJ*quadWeights[q];
            }
          }

          for (fc = 0; fc < Ncomp_i; ++fc) {
            const PetscInt fidx = f*Ncomp_i+fc; /* Test function basis index */
            const PetscInt i    = offsetI+fidx; /* Element matrix row */
            for (gc = 0; gc < Ncomp_j; ++gc) {
              const PetscInt gidx = g*Ncomp_j+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+gidx; /* Element matrix column */
              PetscScalar    realSpaceDerI[dim];
              PetscScalar    realSpaceDerJ[dim];
              PetscInt       d, d2;

              for (d = 0; d < dim; ++d) {
                realSpaceDerI[d] = 0.0;
                realSpaceDerJ[d] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  realSpaceDerI[d] += invJ[d2*dim+d]*basisDerI[(q*Nb_i*Ncomp_i+fidx)*dim+d2];
                  realSpaceDerJ[d] += invJ[d2*dim+d]*basisDerJ[(q*Nb_j*Ncomp_j+gidx)*dim+d2];
                }
              }
              elemMat[eOffset+i*cellDof+j] += basisI[q*Nb_i*Ncomp_i+fidx]*g0[fc*Ncomp_j+gc]*basisJ[q*Nb_j*Ncomp_j+gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[eOffset+i*cellDof+j] += basisI[q*Nb_i*Ncomp_i+fidx]*g1[(fc*Ncomp_j+gc)*dim+d]*realSpaceDerJ[d];
                elemMat[eOffset+i*cellDof+j] += realSpaceDerI[d]*g2[(fc*Ncomp_j+gc)*dim+d]*basisJ[q*Nb_j*Ncomp_j+gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[eOffset+i*cellDof+j] += realSpaceDerI[d]*g3[((fc*Ncomp_j+gc)*dim+d)*dim+d2]*realSpaceDerJ[d2];
                }
              }
            }
          }
        }
      }
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < Ncomp_i; ++fc) {
        for (f = 0; f < Nb_i; ++f) {
          const PetscInt i = offsetI + f*Ncomp_i+fc;
          for (gc = 0; gc < Ncomp_j; ++gc) {
            for (g = 0; g < Nb_j; ++g) {
              const PetscInt j = offsetJ + g*Ncomp_j+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, elemMat[eOffset+i*cellDof+j]);CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset += cellDof;
    eOffset += cellDof*cellDof;
  }
  /* ierr = PetscLogEventEnd(IntegrateJacobianEvent,0,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}
'''

class QuadratureGenerator(script.Script):
  def __init__(self):
    import RDict
    script.Script.__init__(self, argDB = RDict.RDict())
    import os
    self.baseDir    = os.getcwd()
    self.quadDegree = -1
    self.gpuScalarType = 'float'
    return

  def setupPaths(self):
    import sys, os

    petscDir = os.getenv('PETSC_DIR')
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'fiat-dev'))
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'ffc-0.2.3'))
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'Generator'))
    return

  def setup(self):
    script.Script.setup(self)
    self.setupPaths()
    try:
      import Cxx, CxxHelper
    except ImportError:
      raise RuntimeError('Unable to find Generator package!\nReconfigure PETSc using --download-generator.')
    self.Cxx = CxxHelper.Cxx()
    if len(self.debugSections) == 0:
      self.debugSections = ['screen']
    return

  def createQuadrature(self, shape, degree):
    import FIAT.quadrature
    return FIAT.quadrature.make_quadrature(shape, degree)

  def createFaceQuadrature(self, shape, degree):
    from FIAT.reference_element import default_simplex

    if shape == default_simplex(2):
      q0 = self.createQuadrature(default_simplex(1), degree)
      q0.x = [[p, -1.0] for p in q0.x]
      q1 = self.createQuadrature(default_simplex(1), degree)
      q1.x = [[-1.0, p] for p in q1.x]
      q2 = self.createQuadrature(default_simplex(1), degree)
      q2.x = [[p, p] for p in q2.x]
      return (q0, q1, q2)
    else:
      raise RuntimeError("ERROR: not yet implemented")
    return

  def getMeshType(self):
    '''Was ALE::Mesh, is now PETSC_MESH_TYPE'''
    return 'PETSC_MESH_TYPE'

  def getArray(self, name, values, comment = None, typeName = 'double', static = True, packSize = 1):
    from Cxx import Array
    from Cxx import Initializer
    import numpy

    values = numpy.array(values)
    arrayInit = Initializer()
    arrayInit.children = map(self.Cxx.getDouble, numpy.ravel(values))
    arrayInit.list = True
    arrayDecl = Array()
    arrayDecl.children = [name]
    arrayDecl.type = self.Cxx.typeMap[typeName]
    arrayDecl.size = self.Cxx.getInteger(numpy.size(values)/packSize)
    arrayDecl.static = static
    arrayDecl.initializer = arrayInit
    return self.Cxx.getDecl(arrayDecl, comment)

  def getQuadratureStructs(self, degree, quadrature, num, tensor = 0):
    '''Return C arrays with the quadrature points and weights
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Define
    import numpy as np

    self.logPrint('Generating quadrature structures for degree '+str(degree), debugSection = 'codegen')
    ext = '_'+str(num)
    if tensor:
      numPointsDim = Define()
      numPointsDim.identifier = 'NUM_QUADRATURE_POINTS_DIM'+ext
      numPointsDim.replacementText = str(len(quadrature.get_points()))
      numPoints = Define()
      numPoints.identifier = 'NUM_QUADRATURE_POINTS'+ext
      numPoints.replacementText = str(len(quadrature.get_points())**tensor)
      points = quadrature.get_points()
      weights = quadrature.get_weights()
      for d in range(2, tensor+1):
#        points  = np.meshgrid(points, quadrature.get_points())
        weights = np.outer(weights, quadrature.get_weights())
      if tensor == 2:
        newPoints = np.zeros((len(points), len(points), 2))
        for i, p in enumerate(points):
          for j, q in enumerate(points):
            newPoints[i,j, 0] = p
            newPoints[i,j, 1] = q
        points = newPoints
      elif tensor == 3:
        newPoints = np.zeros((len(points), len(points), len(points), 3))
        for i, p in enumerate(points):
          for j, q in enumerate(points):
            for k, r in enumerate(points):
              newPoints[i,j,k, 0] = p
              newPoints[i,j,k, 1] = q
              newPoints[i,j,k, 2] = r
        points = newPoints
      else:
        raise RuntimeError('Not implemented')
      code = [numPointsDim, numPoints,
              self.getArray(self.Cxx.getVar('points_dim'+ext), quadrature.get_points(), 'Quadrature points along each dimension\n   - (x1,y1,x2,y2,...)', 'PetscReal'),
              self.getArray(self.Cxx.getVar('weights_dim'+ext), quadrature.get_weights(), 'Quadrature weights along each dimension\n   - (v1,v2,...)', 'PetscReal'),
              self.getArray(self.Cxx.getVar('points'+ext), points, 'Quadrature points\n   - (x1,y1,x2,y2,...)', 'PetscReal'),
              self.getArray(self.Cxx.getVar('weights'+ext), weights, 'Quadrature weights\n   - (v1,v2,...)', 'PetscReal')]
    else:
      numPoints = Define()
      numPoints.identifier = 'NUM_QUADRATURE_POINTS'+ext
      numPoints.replacementText = str(len(quadrature.get_points()))
      code = [numPoints,
              self.getArray(self.Cxx.getVar('points'+ext), quadrature.get_points(), 'Quadrature points\n   - (x1,y1,x2,y2,...)', 'PetscReal'),
              self.getArray(self.Cxx.getVar('weights'+ext), quadrature.get_weights(), 'Quadrature weights\n   - (v1,v2,...)', 'PetscReal')]
    return code

  def getQuadratureStructsInline(self, degree, quadrature, num, tensor = 0):
    '''Return C arrays with the quadrature points and weights
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Declarator
    import numpy as np

    self.logPrint('Generating quadrature structures for degree '+str(degree), debugSection = 'codegen')
    ext = '_'+str(num)
    if tensor:
      numPointsDim = Declarator()
      numPointsDim.identifier  = 'numQuadraturePointsDim'+ext
      numPointsDim.type        = self.Cxx.typeMap['const int']
      numPointsDim.initializer = self.Cxx.getInteger(len(quadrature.get_points()))
      numPoints = Declarator()
      numPoints.identifier  = 'numQuadraturePoints'+ext
      numPoints.type        = self.Cxx.typeMap['const int']
      numPoints.initializer = self.Cxx.getInteger(len(quadrature.get_points())**tensor)
      points = quadrature.get_points()
      weights = quadrature.get_weights()
      for d in range(2, tensor+1):
#        points  = np.outer(points, quadrature.get_points())
        weights = np.outer(weights, quadrature.get_weights())
      if tensor == 2:
        newPoints = np.zeros((len(points), len(points), 2))
        for i, p in enumerate(points):
          for j, q in enumerate(points):
            newPoints[i,j, 0] = p
            newPoints[i,j, 1] = q
        points = newPoints
      elif tensor == 3:
        newPoints = np.zeros((len(points), len(points), len(points), 3))
        for i, p in enumerate(points):
          for j, q in enumerate(points):
            for k, r in enumerate(points):
              newPoints[i,j,k, 0] = p
              newPoints[i,j,k, 1] = q
              newPoints[i,j,k, 2] = r
        points = newPoints
      else:
        raise RuntimeError('Not implemented')
      code = [self.Cxx.getDecl(numPointsDim), self.Cxx.getDecl(numPoints),
              self.getArray(self.Cxx.getVar('points_dim'+ext), quadrature.get_points(), 'Quadrature points along each dimension\n   - (x1,y1,x2,y2,...)', 'const PetscReal', static = False),
              self.getArray(self.Cxx.getVar('weights_dim'+ext), quadrature.get_weights(), 'Quadrature weights along each dimension\n   - (v1,v2,...)', 'const PetscReal', static = False),
              self.getArray(self.Cxx.getVar('points'+ext), points, 'Quadrature points\n   - (x1,y1,x2,y2,...)', 'const PetscReal', static = False),
              self.getArray(self.Cxx.getVar('weights'+ext), weights, 'Quadrature weights\n   - (v1,v2,...)', 'const PetscReal', static = False)]
    else:
      numPoints = Declarator()
      numPoints.identifier  = 'numQuadraturePoints'+ext
      numPoints.type        = self.Cxx.typeMap['const int']
      numPoints.initializer = self.Cxx.getInteger(len(quadrature.get_points()))
      code = [self.Cxx.getDecl(numPoints),
              self.getArray(self.Cxx.getVar('points'+ext), quadrature.get_points(), 'Quadrature points\n   - (x1,y1,x2,y2,...)', 'const PetscReal', static = False),
              self.getArray(self.Cxx.getVar('weights'+ext), quadrature.get_weights(), 'Quadrature weights\n   - (v1,v2,...)', 'const PetscReal', static = False)]
    return code

  def getBasisFuncOrder(self, element):
    '''Map from FIAT order to Sieve order
       - In 2D, FIAT uses the numbering, and in 3D
         v2                                     v2
         |\                                     |\
         | \                                    |\\
       e1|  \e0                                 | |\
         |   \                                e1| | \e0
         v0--v1                                 | \  \
           e2                                   |  |e5\
                                                |f1|   \
                                                |  | f0 \
                                                |  v3    \
                                                | /  \e4  \
                                                | |e3 ----\\
                                                |/    f2   \\
                                                v0-----------v1
                                                    e2
    '''
    basis = element.get_nodal_basis()
    dim   = element.get_reference_element().get_spatial_dimension()
    ids   = element.entity_dofs()
    if dim == 1:
      perm = []
      for e in ids[1]:
        perm.extend(ids[1][e])
      for v in ids[0]:
        perm.extend(ids[0][v])
    elif dim == 2:
      perm = []
      for f in ids[2]:
        perm.extend(ids[2][f])
      for e in ids[1]:
        perm.extend(ids[1][(e+2)%3])
      for v in ids[0]:
        perm.extend(ids[0][v])
    elif dim == 3:
      perm = []
      for c in ids[3]:
        perm.extend(ids[3][c])
      for f in [3, 2, 0, 1]:
        perm.extend(ids[2][f])
      for e in [2, 0, 1, 3]:
        perm.extend(ids[1][e])
      for e in [4, 5]:
        if len(ids[1][e]):
          perm.extend(ids[1][e][::-1])
      for v in ids[0]:
        perm.extend(ids[0][v])
    else:
      perm = None
    print [f.get_point_dict() for f in element.dual_basis()]
    print element.entity_dofs()
    print 'Perm:',perm
    return perm

  def getTensorBasisFuncOrder(self, element, dim):
    basis = element.get_nodal_basis()
    ids   = element.entity_dofs()
    if dim == 2:
      perm = []
      print ids
      if len(ids[1]) > 1: raise RuntimeError('More than 1 edge in a 1D discretization')
      numEdgeDof   = len(ids[1][0])
      numVertexDof = len(ids[0][0])
      if numEdgeDof == 1 and numVertexDof == 0:
        perm.append(0)
      elif numEdgeDof == 0 and numVertexDof == 1:
        perm.extend([0, 2, 3, 1])
      else:
        raise RuntimeError('I have not figured this out yet')
    else:
      perm = None
    print [f.get_point_dict() for f in element.dual_basis()]
    print element.entity_dofs()
    print 'Perm:',perm
    return perm

  def getReferenceTensor(self, element, quadrature):
    import numpy

    components = element.function_space().tensor_shape()[0]
    points = quadrature.get_points()
    weights = quadrature.get_weights()
    elemMats = []
    for i in range(components):
      basis = element.function_space().select_vector_component(i)
      dim = element.get_reference_element().get_spatial_dimension()
      elemMat = numpy.zeros((len(basis), len(basis), dim, dim), dtype = numpy.float32)
      basisTab = numpy.transpose(basis.tabulate(points))
      basisDerTab = numpy.transpose([basis.deriv_all(d).tabulate(points) for d in range(dim)])
      perm = self.getBasisFuncOrder(element)
      if not perm is None:
        basisTabOld    = numpy.array(basisTab)
        basisDerTabOld = numpy.array(basisDerTab)
        for q in range(len(points)):
          for i,pi in enumerate(perm):
            basisTab[q][i]    = basisTabOld[q][pi]
            basisDerTab[q][i] = basisDerTabOld[q][pi]
      # Integrate for Laplacian
      for i in range(len(basis)):
        for j in range(len(basis)):
          for d in range(dim):
            for e in range(dim):
              for q in range(len(points)):
                elemMat[i][j][d][e] += basisDerTab[q][i][d]*basisDerTab[q][j][e]*weights[q]
      elemMats.append(elemMat)
    return elemMats

  def getAggregateBasisStructs(self, elements):
    '''Return defines for overall sizes'''
    from Cxx import Define

    numComp = 0
    for element in elements:
      numComp += getattr(element, 'numComponents', 1)
    numFields = Define()
    numFields.identifier = 'NUM_FIELDS'
    numFields.replacementText = str(len(elements))
    numComponents = Define()
    numComponents.identifier = 'NUM_BASIS_COMPONENTS_TOTAL'
    numComponents.replacementText = str(numComp)
    code    = [numFields, numComponents]
    return code

  def getBasisStructs(self, name, element, quadrature, num, tensor = 0):
    '''Return C arrays with the basis functions and their derivatives evalauted at the quadrature points
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from FIAT.polynomial_set import mis
    from Cxx import Define
    import numpy

    self.logPrint('Generating basis structures for element '+str(element.__class__), debugSection = 'codegen')
    points  = quadrature.get_points()
    numComp = getattr(element, 'numComponents', 1)
    code    = []
    basis   = element.get_nodal_basis()
    dim     = element.get_reference_element().get_spatial_dimension()
    ext     = '_'+str(num)
    if tensor:
      spatialDim = Define()
      spatialDim.identifier = 'SPATIAL_DIM'+ext
      spatialDim.replacementText = str(tensor)
      numFunctionsDim = Define()
      numFunctionsDim.identifier = 'NUM_BASIS_FUNCTIONS_DIM'+ext
      numFunctionsDim.replacementText = str(basis.get_num_members())
      numFunctions = Define()
      numFunctions.identifier = 'NUM_BASIS_FUNCTIONS'+ext
      numFunctions.replacementText = str(basis.get_num_members()**tensor)
      numComponents = Define()
      numComponents.identifier = 'NUM_BASIS_COMPONENTS'+ext
      numComponents.replacementText = str(numComp)
      numDofDimName= 'numDofDim'+ext
      numDofDims   = [numComp*len(ids[0]) for d, ids in element.entity_dofs().items()]
      numDofName   = 'numDof'+ext
      numDofs      = numpy.zeros((tensor+1,), dtype=int)
      numDofs[0]   = numDofDims[0]
      numDofs[1]   = numDofDims[1]
      for d in range(2, tensor+1):
        numDofs[d] = numDofs[d-1]**2
      basisDimName    = name+'BasisDim'+ext
      basisDerDimName = name+'BasisDerivativesDim'+ext
      # BROKEN
      perm            = self.getTensorBasisFuncOrder(element, tensor)
      evals           = basis.tabulate(points, 1)
      basisDimTab     = numpy.array(evals[mis(dim, 0)[0]]).transpose()
      basisDerDimTab  = numpy.array([evals[alpha] for alpha in mis(dim, 1)]).transpose()
      basisName       = name+'Basis'+ext
      basisDerName    = name+'BasisDerivatives'+ext
      basisTab        = numpy.zeros((len(points)**tensor,basisDimTab.shape[1]**tensor))
      basisDerTab     = numpy.zeros((len(points)**tensor,basisDimTab.shape[1]**tensor,tensor))
      if tensor == 2:
        nB = basisDimTab.shape[1]
        for q in range(len(points)):
          for r in range(len(points)):
            for b1 in range(nB):
              for b2 in range(nB):
                basisTab[q*len(points)+r][b1*nB+b2] = basisDimTab[q][b1]*basisDimTab[r][b2]
                basisDerTab[q*len(points)+r][b1*nB+b2][0] = basisDerDimTab[q][b1][0]*basisDimTab[r][b2]
                basisDerTab[q*len(points)+r][b1*nB+b2][1] = basisDimTab[q][b1]*basisDerDimTab[r][b2][0]
      elif tensor == 3:
        for q in range(len(points)):
          for r in range(len(points)):
            for s in range(len(points)):
              for b1 in range(basisDimTab.shape[1]):
                for b2 in range(basisDimTab.shape[1]):
                  for b3 in range(basisDimTab.shape[1]):
                    basisTab[(q*len(points)+r)*len(points)+s][(b1*basisDimTab.shape[1]+b2)*basisDimTab.shape[1]+b3] = basisDimTab[q][b1]*basisDimTab[r][b2]*basisDimTab[s][b3]
                    basisDerTab[(q*len(points)+r)*len(points)+s][(b1*basisDimTab.shape[1]+b2)*basisDimTab.shape[1]+b3][0] = basisDerDimTab[q][b1][0]*basisDimTab[r][b2]*basisDimTab[s][b3]
                    basisDerTab[(q*len(points)+r)*len(points)+s][(b1*basisDimTab.shape[1]+b2)*basisDimTab.shape[1]+b3][1] = basisDimTab[q][b1]*basisDerDimTab[r][b2][0]*basisDimTab[s][b3]
                    basisDerTab[(q*len(points)+r)*len(points)+s][(b1*basisDimTab.shape[1]+b2)*basisDimTab.shape[1]+b3][2] = basisDimTab[q][b1]*basisDimTab[r][b2]*basisDerDimTab[s][b3][0]
      else:
        raise RuntimeError('Cannot handle tensor dimension '+str(tensor))
      if numComp > 1:
        newShape          = list(basisDimTab.shape)
        newShape[1]       = newShape[1]*numComp
        basisDimTabNew    = numpy.zeros(newShape)
        newShape          = list(basisDerDimTab.shape)
        newShape[1]       = newShape[1]*numComp
        basisDerDimTabNew = numpy.zeros(newShape)
        for q in range(basisDimTab.shape[0]):
          for i in range(basisDimTab.shape[1]):
            for c in range(numComp):
              basisDimTabNew[q][i*numComp+c]    = basisDimTab[q][i]
              basisDerDimTabNew[q][i*numComp+c] = basisDerDimTab[q][i]
        basisDimTab    = basisDimTabNew
        basisDerDimTab = basisDerDimTabNew
      code.extend([spatialDim, numFunctionsDim, numFunctions, numComponents,
                   self.getArray(self.Cxx.getVar(numDofDimName), numDofDims, 'Number of degrees of freedom for each dimension', 'int'),
                   self.getArray(self.Cxx.getVar(basisDimName), basisDimTab, 'Nodal basis function evaluations along each dimension\n    - basis component is fastest varying, then basis function, then quad point', 'PetscReal'),
                   self.getArray(self.Cxx.getVar(basisDerDimName), basisDerDimTab, 'Nodal basis function derivative evaluations along each dimension,\n    - basis component is fastest varying, then derivative direction, then basis function, then quad point')])
    else:
      spatialDim = Define()
      spatialDim.identifier = 'SPATIAL_DIM'+ext
      spatialDim.replacementText = str(dim)
      numFunctions = Define()
      numFunctions.identifier = 'NUM_BASIS_FUNCTIONS'+ext
      numFunctions.replacementText = str(basis.get_num_members())
      numComponents = Define()
      numComponents.identifier = 'NUM_BASIS_COMPONENTS'+ext
      numComponents.replacementText = str(numComp)
      numDofName   = 'numDof'+ext
      numDofs      = [numComp*len(ids[0]) for d, ids in element.entity_dofs().items()]
      basisName    = name+'Basis'+ext
      basisDerName = name+'BasisDerivatives'+ext
      perm         = self.getBasisFuncOrder(element)
      evals        = basis.tabulate(points, 1)
      basisTab     = numpy.array(evals[mis(dim, 0)[0]]).transpose()
      basisDerTab  = numpy.array([evals[alpha] for alpha in mis(dim, 1)]).transpose()
      code.extend([spatialDim, numFunctions, numComponents])
    if not perm is None:
      basisTabOld    = numpy.array(basisTab)
      basisDerTabOld = numpy.array(basisDerTab)
      for q in range(basisTab.shape[0]):
        for i,pi in enumerate(perm):
          basisTab[q][i]    = basisTabOld[q][pi]
          basisDerTab[q][i] = basisDerTabOld[q][pi]
    if numComp > 1:
      newShape       = list(basisTab.shape)
      newShape[1]    = newShape[1]*numComp
      basisTabNew    = numpy.zeros(newShape)
      newShape       = list(basisDerTab.shape)
      newShape[1]    = newShape[1]*numComp
      basisDerTabNew = numpy.zeros(newShape)
      for q in range(basisTab.shape[0]):
        for i in range(basisTab.shape[1]):
          for c in range(numComp):
            basisTabNew[q][i*numComp+c]    = basisTab[q][i]
            basisDerTabNew[q][i*numComp+c] = basisDerTab[q][i]
      basisTab    = basisTabNew
      basisDerTab = basisDerTabNew
    code.extend([self.getArray(self.Cxx.getVar(numDofName), numDofs, 'Number of degrees of freedom for each dimension', 'int'),
                 self.getArray(self.Cxx.getVar(basisName), basisTab, 'Nodal basis function evaluations\n    - basis component is fastest varying, then basis function, then quad point', 'PetscReal'),
                 self.getArray(self.Cxx.getVar(basisDerName), basisDerTab, 'Nodal basis function derivative evaluations,\n    - basis component is fastest varying, then derivative direction, then basis function, then quad point', 'PetscReal')])
    return code

  def getBasisStructsInline(self, name, element, quadrature, num, tensor = 0):
    '''Return C arrays with the basis functions and their derivatives evalauted at the quadrature points
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from FIAT.polynomial_set import mis
    from Cxx import Declarator
    import numpy

    self.logPrint('Generating basis structures for element '+str(element.__class__), debugSection = 'codegen')
    points  = quadrature.get_points()
    numComp = getattr(element, 'numComponents', 1)
    code    = []
    # Handles vector elements which just repeat scalar values
    for i in range(1):
      basis = element.get_nodal_basis()
      dim = element.get_reference_element().get_spatial_dimension()
      ext = '_'+str(num+i)
      numFunctions = Declarator()
      numFunctions.identifier  = 'numBasisFunctions'+ext
      numFunctions.type        = self.Cxx.typeMap['const int']
      numFunctions.initializer = self.Cxx.getInteger(basis.get_num_members())
      numComponents = Declarator()
      numComponents.identifier  = 'numBasisComponents'+ext
      numComponents.type        = self.Cxx.typeMap['const int']
      numComponents.initializer = self.Cxx.getInteger(numComp)
      basisName    = name+'Basis'+ext
      basisDerName = name+'BasisDerivatives'+ext
      perm         = self.getBasisFuncOrder(element)
      evals        = basis.tabulate(points, 1)
      basisTab     = numpy.array(evals[mis(dim, 0)[0]]).transpose()
      basisDerTab  = numpy.array([evals[alpha] for alpha in mis(dim, 1)]).transpose()
      if not perm is None:
        basisTabOld    = numpy.array(basisTab)
        basisDerTabOld = numpy.array(basisDerTab)
        for q in range(len(points)):
          for i,pi in enumerate(perm):
            basisTab[q][i]    = basisTabOld[q][pi]
            basisDerTab[q][i] = basisDerTabOld[q][pi]
      if numComp > 1:
        newShape       = list(basisTab.shape)
        newShape[1]    = newShape[1]*numComp
        basisTabNew    = numpy.zeros(newShape)
        newShape       = list(basisDerTab.shape)
        newShape[1]    = newShape[1]*numComp
        basisDerTabNew = numpy.zeros(newShape)
        for q in range(basisTab.shape[0]):
          for i in range(basisTab.shape[1]):
            for c in range(numComp):
              basisTabNew[q][i*numComp+c]    = basisTab[q][i]
              basisDerTabNew[q][i*numComp+c] = basisDerTab[q][i]
        basisTab    = basisTabNew
        basisDerTab = basisDerTabNew
      code.extend([self.Cxx.getDecl(numFunctions), self.Cxx.getDecl(numComponents),
                   self.getArray(self.Cxx.getVar(basisName), basisTab, 'Nodal basis function evaluations\n    - basis function is fastest varying, then point', 'const PetscReal', static = False),
                   self.getArray(self.Cxx.getVar(basisDerName), basisDerTab, 'Nodal basis function derivative evaluations,\n    - derivative direction fastest varying, then basis function, then point', 'const '+self.gpuScalarType+str(dim), static = False, packSize = dim)])
    return code

  def getPhysicsRoutines(self, operator):
    '''Should eventually generate the entire evaluation at quadrature points. Now it just defines a name'''
    from Cxx import Define
    f1 = Define()
    f1.identifier = 'f1_func'
    f1.replacementText = 'f1_'+operator
    f1coef = Define()
    f1coef.identifier = 'f1_coef_func'
    f1coef.replacementText = 'f1_'+operator+'_coef'
    return [f1, f1coef]

  def getComputationTypes(self, element, num):
    '''Right now, this is used for GPU'''
    from Cxx import Define, Declarator
    dim  = element.get_reference_element().get_spatial_dimension()
    ext  = '_'+str(num)
    real = self.gpuScalarType

    spatialDim = Define()
    spatialDim.identifier = 'SPATIAL_DIM'+ext
    spatialDim.replacementText = str(dim)

    realType = Declarator()
    realType.identifier = 'realType'
    realType.type       = self.Cxx.typeMap[real]
    realType.typedef    = True

    vecType = Declarator()
    vecType.identifier = 'vecType'
    vecType.type       = self.Cxx.typeMap[real+str(dim)]
    vecType.typedef    = True
    return [spatialDim, self.Cxx.getDecl(realType, 'Type for scalars'), self.Cxx.getDecl(vecType, 'Type for vectors')]

  def getComputationLayoutStructs(self, numBlocks):
    '''Right now, this is used for GPU data layout'''
    from Cxx import Declarator
    N_bl = Declarator()
    N_bl.identifier  = 'N_bl'
    N_bl.type        = self.Cxx.typeMap['const int']
    N_bl.initializer = self.Cxx.getInteger(numBlocks)
    return [self.Cxx.getDecl(N_bl, 'Number of concurrent blocks')]

  def getQuadratureBlock(self, num):
    from Cxx import CompoundStatement
    cmpd = CompoundStatement()
    names = [('numQuadPoints', 'NUM_QUADRATURE_POINTS_'+str(num)),
             ('quadPoints',    'points_'+str(num)),
             ('quadWeights',   'weights_'+str(num)),
             ('numBasisFuncs', 'NUM_BASIS_FUNCTIONS_'+str(num)),
             ('basis',         'Basis_'+str(num)),
             ('basisDer',      'BasisDerivatives_'+str(num))]
    cmpd.children = [self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getIndirection(self.Cxx.getVar(a)), self.Cxx.getVar(b))) for a, b in names]
    return cmpd

  def getQuadratureSetup(self):
    from Cxx import Equality
    from Cxx import If
    funcName = 'setupUnstructuredQuadrature_gen'
    stmts = []
    stmts.append(self.Cxx.getExpStmt(self.Cxx.getVar('PetscFunctionBegin')))
    dimVar = self.Cxx.getVar('dim')
    dim3 = If()
    dim3.branch = self.Cxx.getEquality(dimVar, self.Cxx.getInteger(3))
    dim3.children = [self.getQuadratureBlock(2), self.Cxx.getPetscError('PETSC_ERR_SUP', 'Dimension not supported: %d', 'dim')]
    dim2 = If()
    dim2.branch = self.Cxx.getEquality(dimVar, self.Cxx.getInteger(2))
    dim2.children = [self.getQuadratureBlock(1), dim3]
    dim1 = If()
    dim1.branch = self.Cxx.getEquality(dimVar, self.Cxx.getInteger(1))
    dim1.children = [self.getQuadratureBlock(0), dim2]
    stmts.append(dim1)
    stmts.append(self.Cxx.getReturn(isPetsc = 1))
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('PetscErrorCode'),
                                [self.Cxx.getParameter('dim', self.Cxx.getTypeMap()['const int']),
                                 self.Cxx.getParameter('numQuadPoints', self.Cxx.getTypeMap()['int pointer']),
                                 self.Cxx.getParameter('quadPoints', self.Cxx.getTypeMap()['double pointer pointer']),
                                 self.Cxx.getParameter('quadWeights', self.Cxx.getTypeMap()['double pointer pointer']),
                                 self.Cxx.getParameter('numBasisFuncs', self.Cxx.getTypeMap()['int pointer']),
                                 self.Cxx.getParameter('basis', self.Cxx.getTypeMap()['double pointer pointer']),
                                 self.Cxx.getParameter('basisDer', self.Cxx.getTypeMap()['double pointer pointer'])],
                                [], stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def mapToRealSpace(self, dim, decls, stmts, refVar = None, realVar = None, isBd = False):
    '''Maps coordinates in the reference element to real space'''
    if refVar  is None: refVar  = self.Cxx.getVar('refCoords')
    if realVar is None: realVar = self.Cxx.getVar('coords')
    if isBd:
      embedDim = dim + 1
    else:
      embedDim = dim
    if not decls is None:
      decls.append(self.Cxx.getArray(refVar,  self.Cxx.getType('PetscReal'), dim))
      decls.append(self.Cxx.getArray(realVar, self.Cxx.getType('PetscReal'), embedDim))
    basisLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('e', 'int'), 0, dim)
    basisLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(self.Cxx.getArrayRef(realVar, 'd'), self.Cxx.getMultiplication(self.Cxx.getArrayRef('J', self.Cxx.getAddition(self.Cxx.getMultiplication('d', embedDim), 'e')), self.Cxx.getGroup(self.Cxx.getAddition(self.Cxx.getArrayRef(refVar, 'e'), 1.0))))))
    testLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('d', 'int'), 0, embedDim)
    testLoop.children[0].children.extend([self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(realVar, 'd'), self.Cxx.getArrayRef('v0', 'd'))), basisLoop])
    stmts.append(testLoop)
    return

  def cellToFaceTransform(self, shape, cmpd, coordVar, quadVar, face):
    from FIAT.reference_element import default_simplex
    from math import sqrt
    from Cxx import Break
    if shape == default_simplex(2):
      if face == 0:
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordVar, 0), self.Cxx.getArrayRef(quadVar, 'q')), caseLabel = face)
        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordVar, 1), -1.0)), Break()])
      elif face == 1:
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordVar, 0), self.Cxx.getMultiplication(sqrt(2)/2.0, self.Cxx.getArrayRef(quadVar, 'q'))), caseLabel = face)
        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordVar, 1), self.Cxx.getMultiplication(sqrt(2)/2.0, self.Cxx.getArrayRef(quadVar, 'q')))), Break()])
      elif face == 2:
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordVar, 0), -1.0), caseLabel = face)
        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordVar, 1), self.Cxx.getArrayRef(quadVar, 'q'))), Break()])
    return

  def getIntegratorPoints(self, n, element):
    from Cxx import Define
    import numpy

    ids  = element.entity_dofs()
    pts  = [f.get_point_dict().keys()[0] for f in element.dual_basis()]
    perm = self.getBasisFuncOrder(element)
    ext  = '_'+str(n)
    dim  = element.get_reference_element().get_spatial_dimension()
    if dim == 1:
      num = len(ids[1][0]) + len(ids[0][0])*2
    elif dim == 2:
      num = len(ids[2][0]) + len(ids[1][0])*3 + len(ids[0][0])*3
    elif dim == 3:
      num = len(ids[3][0]) + len(ids[2][0])*4 + len(ids[1][0])*6 + len(ids[0][0])*4
    numPoints = Define()
    numPoints.identifier = 'NUM_DUAL_POINTS'+ext
    numPoints.replacementText = str(num)
    dualPoints = numpy.zeros((num, dim))
    for i in range(num):
      for d in range(dim):
        dualPoints[i][d] = pts[perm[i]][d]
    return [numPoints, self.getArray(self.Cxx.getVar('dualPoints'+ext), dualPoints, 'Dual points\n   - (x1,y1,x2,y2,...)')]

  def getIntegratorSetup_PointEvaluation(self, n, element, isBd = False):
    from Cxx import Break, CompoundStatement, Function, Pointer, Switch
    dim  = element.get_reference_element().get_spatial_dimension()
    ids  = element.entity_dofs()
    pts  = [f.get_point_dict().keys()[0] for f in element.dual_basis()]
    perm = self.getBasisFuncOrder(element)
    p    = 0
    if isBd:
      funcName = 'IntegrateBdDualBasis_gen_'+str(n)
    else:
      funcName = 'IntegrateDualBasis_gen_'+str(n)
    idxVar  = self.Cxx.getVar('dualIndex')
    refVar  = self.Cxx.getVar('refCoords')
    realVar = self.Cxx.getVar('coords')
    decls = []
    stmts = []
    switch  = Switch()
    switch.branch = idxVar
    cmpd = CompoundStatement()
    if dim == 1:
      for i in range(len(ids[1][0]) + len(ids[0][0])*2):
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]), caseLabel = p)
        cmpd.children.extend([cStmt, Break()])
        p += 1
    elif dim == 2:
      for i in range(len(ids[2][0]) + len(ids[1][0])*3 + len(ids[0][0])*3):
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]), caseLabel = p)
        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
        p += 1
    elif dim == 3:
      for i in range(len(ids[3][0]) + len(ids[2][0])*4 + len(ids[1][0])*6 + len(ids[0][0])*4):
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]), caseLabel = p)
        cmpd.children.extend([cStmt,
                              self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])),
                              self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 2), pts[perm[p]][2])), Break()])
        p += 1
    cStmt = self.Cxx.getExpStmt(self.Cxx.getFunctionCall('printf', [self.Cxx.getString('dualIndex: %d\\n'), 'dualIndex']))
    cStmt.caseLabel = self.Cxx.getValue('default')
    cmpd.children.extend([cStmt, self.Cxx.getThrow(self.Cxx.getFunctionCall('ALE::Exception', [self.Cxx.getString('Bad dual index')]))])
    switch.children = [cmpd]
    stmts.append(switch)
    self.mapToRealSpace(dim, decls, stmts, refVar, realVar, isBd)
    stmts.append(self.Cxx.getReturn(self.Cxx.getFunctionCall(self.Cxx.getGroup(self.Cxx.getIndirection('func')), [realVar])))
    bcFunc = self.Cxx.getFunctionPointer('func', self.Cxx.getType('double'), [self.Cxx.getParameter('coords', self.Cxx.getType('PetscReal', 1, isConst = 1))])
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('double'),
                                [self.Cxx.getParameter('v0', self.Cxx.getType('PetscReal', 1, isConst = 1)),
                                 self.Cxx.getParameter('J',  self.Cxx.getType('PetscReal', 1, isConst = 1)),
                                 self.Cxx.getParameter(idxVar, self.Cxx.getType('int', isConst = 1)),
                                 self.Cxx.getParameter(None, bcFunc)],
                                decls, stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def getIntegratorSetup_IntegralMoment(self, n, element):
    from Cxx import Break, CompoundStatement, Function, Pointer, Switch
    code   = []
    idxVar = self.Cxx.getVar('dualIndex')
    refVar = self.Cxx.getVar('refCoords')
    realVar = self.Cxx.getVar('coords')
    valVar = self.Cxx.getVar('value')
    bcFunc = self.Cxx.getFunctionPointer('func', self.Cxx.getType('double'), [self.Cxx.getParameter('coords', self.Cxx.getType('double', 1, isConst = 1))])
    shape  = element.get_reference_element()
    dim    = shape.get_spatial_dimension()
    ids    = element.Udual.entity_ids
    for i in range(element.function_space().tensor_shape()[0]):
      funcName = 'IntegrateDualBasis_gen_'+str(n+i)
      decls = []
      decls.append(self.Cxx.getDeclaration(valVar, self.Cxx.getType('double')))
      stmts = []
      quadLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('q', 'int'), 0, 'NUM_QUADRATURE_POINTS_'+str(n+i)+'_face')
      cmpd = quadLoop.children[0]
      switch  = Switch()
      switch.branch = idxVar
      scmpd = CompoundStatement()
      cmpd.declarations.append(self.Cxx.getDeclaration('faceIndex', self.Cxx.getType('const int'), self.Cxx.getModulo(idxVar, len(element.function_space())/3)))
      if dim == 2:
        for f in range(len(ids[2][0]), len(ids[2][0]) + len(ids[1][0])*3):
          self.cellToFaceTransform(shape, scmpd, refVar, self.Cxx.getVar('points_'+str(n+i)+'_face'), f)
        #for i in range(len(ids[0][0]*3) + len(ids[1][0])*3, len(ids[0][0])*3 + len(ids[1][0])*3 + len(ids[2][0])):
      cStmt = self.Cxx.getExpStmt(self.Cxx.getFunctionCall('printf', [self.Cxx.getString('dualIndex: %d\\n'), 'dualIndex']))
      cStmt.caseLabel = self.Cxx.getValue('default')
      scmpd.children.extend([cStmt, self.Cxx.getThrow(self.Cxx.getFunctionCall('ALE::Exception', [self.Cxx.getString('Bad dual index')]))])
      switch.children = [scmpd]
      cmpd.children.append(switch)
      self.mapToRealSpace(dim, cmpd.declarations, cmpd.children, refVar, realVar)
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(valVar, self.Cxx.getMultiplication(self.Cxx.getFunctionCall(self.Cxx.getGroup(self.Cxx.getIndirection('func')), [realVar]), self.Cxx.getMultiplication(self.Cxx.getArrayRef('basis_'+str(n+i)+'_face', self.Cxx.getAddition(self.Cxx.getMultiplication('q', 'NUM_BASIS_FUNCTIONS_'+str(n+i)+'_face'), 'faceIndex')), self.Cxx.getArrayRef('weights_'+str(n+i)+'_face', 'q'))))))
      stmts.append(quadLoop)
      stmts.append(self.Cxx.getReturn(valVar))
      func = self.Cxx.getFunction(funcName, self.Cxx.getType('double'),
                                  [self.Cxx.getParameter('v0', self.Cxx.getType('double', 1, isConst = 1)),
                                   self.Cxx.getParameter('J',  self.Cxx.getType('double', 1, isConst = 1)),
                                   self.Cxx.getParameter(idxVar, self.Cxx.getType('int', isConst = 1)),
                                   self.Cxx.getParameter(None, bcFunc)],
                                  decls, stmts)
      code.extend(self.Cxx.getFunctionHeader(funcName)+[func])
    return code

  def getIntegratorSetup(self, n, element, isBd = False):
    import FIAT.functional

    if isinstance(element.dual_basis()[0], FIAT.functional.PointEvaluation):
      return self.getIntegratorSetup_PointEvaluation(n, element, isBd)
    elif isinstance(element.dual_basis()[0], FIAT.functional.IntegralMoment):
      return self.getIntegratorSetup_IntegralMoment(n, element)
    raise RuntimeError('Could not generate dual basis evaluation code')

  def getSectionSetup(self, n, element):
    from Cxx import CompoundStatement
    if len(element.value_shape()) > 0:
      rank    = element.value_shape()[0]
    else:
      rank    = 1
    code      = []
    dmVar     = self.Cxx.getVar('dm')
    numBCVar  = self.Cxx.getVar('numBC')
    markerVar = self.Cxx.getVar('markers')
    bcVar     = self.Cxx.getVar('bcFuncs')
    exactVar  = self.Cxx.getVar('exactFunc')
    decls = []
    decls.append(self.Cxx.getDeclaration('m', self.Cxx.getType('ALE::Obj<'+self.getMeshType()+'>'), isForward=1))
    decls.append(self.Cxx.getDeclaration('ierr', self.Cxx.getType('PetscErrorCode')))
    for i in range(rank):
      funcName  = 'CreateProblem_gen_'+str(n+i)
      stmts = []
      stmts.append(self.Cxx.getExpStmt(self.Cxx.getVar('PetscFunctionBegin')))
      stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('DMMeshGetMesh', [dmVar, 'm'])))
      cmpd = CompoundStatement()
      cmpd.declarations = [self.Cxx.getDeclaration('d', self.Cxx.getType('ALE::Obj<ALE::Discretization>&', isConst=1),
                                                   self.Cxx.getFunctionCall('new ALE::Discretization',
                                                                            [self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'comm')),
                                                                             self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'debug'))]))]
      for d, ids in element.entity_dofs().items():
        cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setNumDof'), [d, len(ids[0])])))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setQuadratureSize'), ['NUM_QUADRATURE_POINTS_'+str(n+i)])))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setQuadraturePoints'), ['points_'+str(n+i)])))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setQuadratureWeights'), ['weights_'+str(n+i)])))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setBasisSize'), ['NUM_BASIS_FUNCTIONS_'+str(n+i)])))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setBasis'), ['Basis_'+str(n+i)])))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setBasisDerivatives'), ['BasisDerivatives_'+str(n+i)])))
      bcCmpd = CompoundStatement()
      bcCmpd.declarations = [self.Cxx.getDeclaration('b', self.Cxx.getType('ALE::Obj<ALE::BoundaryCondition>&', isConst=1),
                                                     self.Cxx.getFunctionCall('new ALE::BoundaryCondition', [self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'comm')),
                                                                                                             self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'debug'))])),
                             self.Cxx.getDeclaration('name', self.Cxx.getType('ostringstream'), isForward = 1)]
      bcCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setLabelName'), [self.Cxx.getString('marker')])))
      bcCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setMarker'), [self.Cxx.getArrayRef(markerVar, 'i')])))
      bcCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setFunction'), [self.Cxx.getArrayRef(bcVar, 'i')])))
      bcCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setDualIntegrator'), ['IntegrateDualBasis_gen_'+str(n+i)])))
      bcCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getLeftShift('name', 'i')))
      bcCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setBoundaryCondition'), [self.Cxx.getFunctionCall(self.Cxx.getStructRef('name', 'str', 0)), 'b'])))
      cmpd.children.append(self.Cxx.getSimpleLoop('i', 0, numBCVar, isPrefix = 1, body = [bcCmpd]))
      exactCmpd = CompoundStatement()
      exactCmpd.declarations = [self.Cxx.getDeclaration('e', self.Cxx.getType('ALE::Obj<ALE::BoundaryCondition>&', isConst=1),
                                                        self.Cxx.getFunctionCall('new ALE::BoundaryCondition', [self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'comm')),
                                                                                                                self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'debug'))]))]
      exactCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('e', 'setLabelName'), [self.Cxx.getString('marker')])))
      exactCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('e', 'setFunction'), [exactVar])))
      exactCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('e', 'setDualIntegrator'), ['IntegrateDualBasis_gen_'+str(n+i)])))
      exactCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setExactSolution'), ['e'])))
      cmpd.children.append(self.Cxx.getIf(exactVar, [exactCmpd]))
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'setDiscretization'), ['name', 'd'])))
      stmts.append(cmpd)
      stmts.append(self.Cxx.getReturn(isPetsc = 1))
      func = self.Cxx.getFunction(funcName, self.Cxx.getType('PetscErrorCode'),
                                  [self.Cxx.getParameter('dm', self.Cxx.getType('DM')),
                                   self.Cxx.getParameter('name', self.Cxx.getType('char pointer', isConst = 1)),
                                   self.Cxx.getParameter('numBC', self.Cxx.getType('int', isConst = 1)),
                                   self.Cxx.getParameter('markers', self.Cxx.getType('int pointer', isConst = 1)),
                                   self.Cxx.getParameter(None, self.Cxx.getFunctionPointer(bcVar, self.Cxx.getType('double'), [self.Cxx.getParameter('coords', self.Cxx.getType('PetscReal', 1, isConst = 1))], numPointers = 2)),
                                   self.Cxx.getParameter(None, self.Cxx.getFunctionPointer(exactVar, self.Cxx.getType('double'), [self.Cxx.getParameter('coords', self.Cxx.getType('PetscReal', 1, isConst = 1))]))],
                                  decls, stmts)
      code.extend(self.Cxx.getFunctionHeader(funcName)+[func])
    return code

  def getRealCoordinates(self, dimVar, v0Var, JVar, coordsVar):
    '''Calculates the real coordinates of each quadrature point from reference coordinates'''
    dim2Loop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('e', 'int'), 0, dimVar)
    dim2Loop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(self.Cxx.getArrayRef(coordsVar, 'd'), self.Cxx.getMultiplication(self.Cxx.getArrayRef(JVar, self.Cxx.getAddition(self.Cxx.getMultiplication('d', dimVar), 'e')), self.Cxx.getGroup(self.Cxx.getAddition(self.Cxx.getArrayRef('quadPoints', self.Cxx.getAddition(self.Cxx.getMultiplication('q', dimVar), 'e')), self.Cxx.getDouble(1.0)))))))
    dimLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('d', 'int'), 0, dimVar)
    dimLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordsVar, 'd'), self.Cxx.getArrayRef(v0Var, 'd'))))
    dimLoop.children[0].children.append(dim2Loop)
    return dimLoop

  def getLinearAccumulation(self, inputSection, outputSection, elemIter, numBasisFuncs, elemVec, elemMat):
    '''Accumulates linear terms in the residual'''
    from Cxx import CompoundStatement
    scope = CompoundStatement()
    scope.declarations.append(self.Cxx.getDeclaration('x', self.Cxx.getType('PetscScalar', 1)))
    scope.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealRestrict', [inputSection, self.Cxx.getIndirection(elemIter), self.Cxx.getAddress('x')])))
    basisFuncLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('g', 'int'), 0, numBasisFuncs)
    basisFuncLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(self.Cxx.getArrayRef(elemVec, 'f'), self.Cxx.getMultiplication(self.Cxx.getArrayRef(elemMat, self.Cxx.getAddition(self.Cxx.getMultiplication('f', 'numBasisFuncs'), 'g')), self.Cxx.getArrayRef('x', 'g')))))
    testFuncLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('f', 'int'), 0, numBasisFuncs)
    testFuncLoop.children[0].children.append(basisFuncLoop)
    scope.children.append(testFuncLoop)
    scope.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealUpdateAdd', [outputSection, self.Cxx.getIndirection(elemIter), elemVec])))
    return scope

  def getElementIntegrals(self):
    '''Output the C++ residual calculation method'''
    from Cxx import CompoundStatement
    from Cxx import DeclaratorGroup
    from Cxx import Declarator
    from Cxx import Function
    from Cxx import FunctionCall
    from Cxx import Pointer
    from Cxx import Type
    # Parameters
    ctxVar = self.Cxx.getVar('ctx')
    # C Declarations
    decls = []
    optionsType = self.Cxx.getType('Options', 1)
    optionsVar = self.Cxx.getVar('options')
    funcVar = self.Cxx.getVar('func')
    meshVar = self.Cxx.getVar('mesh')
    mVar = self.Cxx.getVar('m')
    decls.append(self.Cxx.getDeclaration(optionsVar, optionsType, self.Cxx.castToType(ctxVar, optionsType)))
    paramDecl = Declarator()
    paramDecl.type = self.Cxx.getType('double', isConst = 1, numPointers = 1)
    funcHead = DeclaratorGroup()
    #   This is almost certainly wrong
    funcHead.setChildren([self.Cxx.getIndirection(funcVar)])
    funcDecl = Function()
    funcDecl.setChildren([funcHead])
    funcDecl.type = self.Cxx.getType('PetscScalar')
    funcDecl.parameters = [paramDecl]
    funcDecl.initializer = self.Cxx.getStructRef(optionsVar, 'rhsFunc')
    decls.append(self.Cxx.getDecl(funcDecl))
    decls.append(self.Cxx.getDeclaration(mVar, self.Cxx.getType('ALE::Obj<'+self.getMeshType()+'>'), self.Cxx.getNullVar()))
    decls.extend([self.Cxx.getDeclaration(self.Cxx.getVar('numQuadPoints'), self.Cxx.getTypeMap()['int']),
                  self.Cxx.getDeclaration(self.Cxx.getVar('numBasisFuncs'), self.Cxx.getTypeMap()['int'])])
    decls.extend([self.Cxx.getDeclaration(self.Cxx.getVar('quadPoints'),  self.Cxx.getType('double pointer')),
                  self.Cxx.getDeclaration(self.Cxx.getVar('quadWeights'), self.Cxx.getType('double pointer')),
                  self.Cxx.getDeclaration(self.Cxx.getVar('basis'),       self.Cxx.getType('double pointer')),
                  self.Cxx.getDeclaration(self.Cxx.getVar('basisDer'),    self.Cxx.getType('double pointer'))])
    decls.append(self.Cxx.getDeclaration(self.Cxx.getVar('ierr'),  self.Cxx.getType('PetscErrorCode')))
    # C++ Declarations and Setup
    stmts = []
    stmts.append(self.Cxx.getExpStmt(self.Cxx.getVar('PetscFunctionBegin')))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('MeshGetMesh', [meshVar, mVar])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('setupUnstructuredQuadrature_gen', [self.Cxx.getStructRef(optionsVar, 'dim'), self.Cxx.getAddress('numQuadPoints'), self.Cxx.getAddress('quadPoints'), self.Cxx.getAddress('quadWeights'), self.Cxx.getAddress('numBasisFuncs'), self.Cxx.getAddress('basis'), self.Cxx.getAddress('basisDer')])))
    patchVar = self.Cxx.getVar('patch')
    coordinatesVar = self.Cxx.getVar('coordinates')
    topologyVar = self.Cxx.getVar('topology')
    cellsVar = self.Cxx.getVar('cells')
    cornersVar = self.Cxx.getVar('corners')
    dimVar = self.Cxx.getVar('dim')
    t_derVar = self.Cxx.getVar('t_der')
    b_derVar = self.Cxx.getVar('b_der')
    coordsVar = self.Cxx.getVar('coords')
    v0Var = self.Cxx.getVar('v0')
    JVar = self.Cxx.getVar('J')
    invJVar = self.Cxx.getVar('invJ')
    detJVar = self.Cxx.getVar('detJ')
    elemVecVar = self.Cxx.getVar('elemVec')
    elemMatVar = self.Cxx.getVar('elemMat')
    decls.extend([self.Cxx.getDeclaration(t_derVar,   self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(b_derVar,   self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(coordsVar,  self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(v0Var,      self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(JVar,       self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(invJVar,    self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(detJVar,    self.Cxx.getTypeMap()['double']),
                  self.Cxx.getDeclaration(elemVecVar, self.Cxx.getType('PetscScalar', 1)),
                  self.Cxx.getDeclaration(elemMatVar, self.Cxx.getType('PetscScalar', 1))])
    cxxCmpd = CompoundStatement()
    cxxCmpd.declarations = [self.Cxx.getDeclaration(patchVar, self.Cxx.getType(self.getMeshType()+'::real_section_type::patch_type', 0, 1), self.Cxx.getInteger(0)),
                            self.Cxx.getDeclaration(coordinatesVar, self.Cxx.getType('ALE::Obj<'+self.getMeshType()+'::real_section_type>&', isConst = 1), self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'getRealSection'), [self.Cxx.getString('coordinates')])),
                            self.Cxx.getDeclaration(topologyVar, self.Cxx.getType('ALE::Obj<'+self.getMeshType()+'::topology_type>&', isConst = 1), self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'getTopology'))),
                            self.Cxx.getDeclaration(cellsVar, self.Cxx.getType('ALE::Obj<'+self.getMeshType()+'::topology_type::label_sequence>&', 0, 1), self.Cxx.getFunctionCall(self.Cxx.getStructRef(topologyVar, 'heightStratum'), [patchVar, 0])),
                            self.Cxx.getDeclaration(cornersVar, self.Cxx.getTypeMap()['const int'], self.Cxx.getFunctionCall(self.Cxx.getStructRef(self.Cxx.getFunctionCall(self.Cxx.getStructRef(self.Cxx.getFunctionCall(self.Cxx.getStructRef(topologyVar, 'getPatch'), [patchVar]), 'nCone'), [self.Cxx.getIndirection(self.Cxx.getFunctionCall(self.Cxx.getStructRef(cellsVar, 'begin'))), self.Cxx.getFunctionCall(self.Cxx.getStructRef(topologyVar, 'depth'))]), 'size'))),
                            self.Cxx.getDeclaration(dimVar, self.Cxx.getTypeMap()['const int'], self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'getDimension')))]
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealZero', ['section'])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMalloc', [self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar')), self.Cxx.getAddress(elemVecVar)])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMalloc', [self.Cxx.getMultiplication(cornersVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar'))), self.Cxx.getAddress(elemMatVar)])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMalloc6', [dimVar,'double',self.Cxx.getAddress(t_derVar),dimVar,'double',self.Cxx.getAddress(b_derVar),dimVar,'double',self.Cxx.getAddress(coordsVar),dimVar,'double',self.Cxx.getAddress(v0Var),self.Cxx.getMultiplication(dimVar,dimVar),'double',self.Cxx.getAddress(JVar),self.Cxx.getMultiplication(dimVar,dimVar),'double',self.Cxx.getAddress(invJVar)])))
    # Loop over elements
    lowerBound = FunctionCall()
    lowerBound.setChildren([self.Cxx.getStructRef(cellsVar, 'begin')])
    upperBound = FunctionCall()
    upperBound.setChildren([self.Cxx.getStructRef(cellsVar, 'end')])
    lType = Type()
    lType.identifier = self.getMeshType()+'::topology_type::label_sequence::iterator'
    decl = Declarator()
    decl.identifier = 'c_iter'
    decl.type = lType
    loop = self.Cxx.getSimpleLoop(decl, lowerBound, upperBound, allowInequality = 1, isPrefix = 1)
    loop.comments = [self.Cxx.getComment('Loop over elements')]
    loopCmpd = loop.children[0]
    # Loop body
    c_iterVar = self.Cxx.getVar('c_iter')
    loopCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMemzero', [elemVecVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar'))])))
    loopCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMemzero', [elemMatVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar')))])))
    loopCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'computeElementGeometry'), [coordinatesVar, self.Cxx.getIndirection(c_iterVar), v0Var, JVar, invJVar, detJVar])))
    #   Quadrature loop
    quadLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('q', 'int'), 0, 'numQuadPoints')
    quadLoop.comments = [self.Cxx.getComment('Loop over quadrature points')]
    quadCmpd = quadLoop.children[0]
    loopCmpd.children.append(quadLoop)
    #   Get real coordinates
    quadCmpd.children.append(self.getRealCoordinates(dimVar, v0Var, JVar, coordsVar))
    #   Accumulate constant terms
    funcValVar = self.Cxx.getVar('funcVal')
    quadCmpd.declarations.append(self.Cxx.getDeclaration(funcValVar, self.Cxx.getType('PetscScalar')))
    quadCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getAssignment(funcValVar, self.Cxx.getFunctionCall(self.Cxx.getGroup(self.Cxx.getIndirection(funcVar)), [coordsVar]))))
    testFuncLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('f', 'int'), 0, 'numBasisFuncs')
    testFuncLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getSubtractionAssignment(self.Cxx.getArrayRef(elemVecVar, 'f'), self.Cxx.getMultiplication(self.Cxx.getMultiplication(self.Cxx.getMultiplication(self.Cxx.getArrayRef('basis', self.Cxx.getAddition(self.Cxx.getMultiplication('q', 'numBasisFuncs'), 'f')), funcValVar), self.Cxx.getArrayRef('quadWeights', 'q')), detJVar))))
    quadCmpd.children.append(testFuncLoop)
    #   Accumulate linear terms
    loopCmpd.children.append(self.getLinearAccumulation('X', 'section', c_iterVar, 'numBasisFuncs', elemVecVar, elemMatVar))
    cxxCmpd.children.append(loop)
    # Cleanup
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscFree', [elemVecVar])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscFree', [elemMatVar])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscFree6', [t_derVar,b_derVar,coordsVar,v0Var,JVar,invJVar])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealComplete', ['section'])))
    # Pack up scopes
    stmts.append(cxxCmpd)
    stmts.append(self.Cxx.getReturn(isPetsc = 1))
    funcName = 'Rhs_Unstructured_gen'
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('PetscErrorCode'),
                                [self.Cxx.getParameter('mesh', self.Cxx.getType('Mesh')),
                                 self.Cxx.getParameter('X', self.Cxx.getType('SectionReal')),
                                 self.Cxx.getParameter('section', self.Cxx.getType('SectionReal')),
                                 self.Cxx.getParameter('ctx', self.Cxx.getTypeMap()['void pointer'])],
                                decls, stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def getQuadratureFile(self, filename, decls):
    from GenericCompiler import CodePurpose
    from Cxx import Include
    from Cxx import Header
    import os

    # Needed to define NULL
    stdInclude = Include()
    stdInclude.identifier = '<stdlib.h>'
    header            = Header()
    if filename:
      header.filename = filename
    else:
      header.filename = 'Integration.c'
    header.children   = [stdInclude]+decls
    header.purpose    = CodePurpose.SKELETON
    return header

  def getElementSource(self, elements, numBlocks = 1, operator = None, sourceType = 'CPU', tensor = 0):
    from GenericCompiler import CompilerException

    self.logPrint('Generating element module', debugSection = 'codegen')
    try:
      defns = []
      n     = 0
      for element in elements:
        #name      = element.family+str(element.n)
        name       = ''
        shape      = element.get_reference_element()
        order      = element.order
        if sourceType != 'GPU':
          if self.quadDegree < 0:
            quadrature = self.createQuadrature(shape, order)
          else:
            quadrature = self.createQuadrature(shape, self.quadDegree)
        if sourceType == 'GPU_inline':
          defns.extend(self.getQuadratureStructsInline(2*len(quadrature.pts)-1, quadrature, n, tensor))
          defns.extend(self.getBasisStructsInline(name, element, quadrature, n, tensor))
          defns.extend(self.getPhysicsRoutines(operator))
          defns.extend(self.getComputationLayoutStructs(numBlocks))
        elif sourceType == 'CPU':
          defns.extend(self.getQuadratureStructs(2*len(quadrature.pts)-1, quadrature, n, tensor))
          defns.extend(self.getBasisStructs(name, element, quadrature, n, tensor))
          #defns.extend(self.getIntegratorPoints(n, element))
          #defns.extend(self.getIntegratorSetup(n, element))
          #defns.extend(self.getIntegratorSetup(n, element, True))
          #defns.extend(self.getSectionSetup(n, element))
        else:
          defns.extend(self.getComputationTypes(element, n))
        if len(element.value_shape()) > 0:
          n += element.value_shape()[0]
        else:
          n += 1
      defns.extend(self.getAggregateBasisStructs(elements))
      #defns.extend(self.getQuadratureSetup())
      #defns.extend(self.getElementIntegrals())
    except CompilerException, e:
      print e
      raise RuntimeError('Quadrature source generation failed')
    return defns

  def outputElementSource(self, defns, filename = '', extra = ''):
    from GenericCompiler import CodePurpose
    import CxxVisitor

    # May need to move setupPETScLogging() here because PETSc clients are currently interfering with numpy
    source = {'Cxx': [self.getQuadratureFile(filename, defns)]}
    outputs = {'Cxx': CxxVisitor.Output()}
    self.logPrint('Writing element source', debugSection = 'codegen')
    for language,output in outputs.items():
      output.setRoot(CodePurpose.STUB, self.baseDir)
      output.setRoot(CodePurpose.IOR, self.baseDir)
      output.setRoot(CodePurpose.SKELETON, self.baseDir)
      try:
        map(lambda tree: tree.accept(output), source[language])
        for f in output.getFiles():
          self.logPrint('Created '+str(language)+' file '+str(f), debugSection = 'codegen')
        if extra:
          f = file(output.getFiles()[0], 'a')
          f.write(extra)
          f.close()
      except RuntimeError, e:
        print e
    return

  def run(self, elements, numBlocks, operator, filename = ''):
    import os
    if elements is None:
      from FIAT.reference_element import default_simplex
      from FIAT.Lagrange import lagrange
      order = 1
      elements =[lagrange(default_simplex(2), order)]
      self.logPrint('Making a P'+str(order)+' Lagrange element on a triangle')
    self.outputElementSource(self.getElementSource(elements), filename, extra = femIntegrationCode)
    self.outputElementSource(self.getElementSource(elements, numBlocks, operator, sourceType = 'GPU'), os.path.splitext(filename)[0]+'_gpu'+os.path.splitext(filename)[1])
    self.outputElementSource(self.getElementSource(elements, numBlocks, operator, sourceType = 'GPU_inline'), os.path.splitext(filename)[0]+'_gpu_inline'+os.path.splitext(filename)[1])
    return

  def runTensorProduct(self, dim, elements, numBlocks, operator, filename = ''):
    # Nothing is finished here
    import os
    self.outputElementSource(self.getElementSource(elements, tensor = dim), filename)
    self.outputElementSource(self.getElementSource(elements, numBlocks, operator, sourceType = 'GPU', tensor = dim), os.path.splitext(filename)[0]+'_gpu'+os.path.splitext(filename)[1])
    self.outputElementSource(self.getElementSource(elements, numBlocks, operator, sourceType = 'GPU_inline', tensor = dim), os.path.splitext(filename)[0]+'_gpu_inline'+os.path.splitext(filename)[1])
    return

if __name__ == '__main__':
  QuadratureGenerator().run()
