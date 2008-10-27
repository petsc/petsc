#ifndef included_ALE_Problem_Bratu_hh
#define included_ALE_Problem_Bratu_hh

#include <DMBuilder.hh>

#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

// How do we do this correctly?
#include "../examples/tutorials/bratu_quadrature.h"

namespace ALE {
  namespace Problem {
    typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
    typedef enum {NEUMANN, DIRICHLET} BCType;
    typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
    typedef union {SectionReal section; Vec vec;} ExactSolType;
    typedef struct {
      PetscInt      debug;                       // The debugging level
      RunType       run;                         // The run type
      PetscInt      dim;                         // The topological mesh dimension
      PetscTruth    reentrantMesh;               // Generate a reentrant mesh?
      PetscTruth    circularMesh;                // Generate a circular mesh?
      PetscTruth    refineSingularity;           // Generate an a priori graded mesh for the poisson problem
      PetscTruth    structured;                  // Use a structured mesh
      PetscTruth    generateMesh;                // Generate the unstructure mesh
      PetscTruth    interpolate;                 // Generate intermediate mesh elements
      PetscReal     refinementLimit;             // The largest allowable cell volume
      char          baseFilename[2048];          // The base filename for mesh files
      char          partitioner[2048];           // The graph partitioner
      PetscScalar (*func)(const double []);      // The function to project
      BCType        bcType;                      // The type of boundary conditions
      PetscScalar (*exactFunc)(const double []); // The exact solution function
      ExactSolType  exactSol;                    // The discrete exact solution
      ExactSolType  error;                       // The discrete cell-wise error
      AssemblyType  operatorAssembly;            // The type of operator assembly 
      double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
      double        lambda;                      // The parameter controlling nonlinearity
      double        reentrant_angle;              // The angle for the reentrant corner.
    } BratuOptions;
    namespace BratuFunctions {
      static PetscScalar lambda;

      PetscScalar zero(const double x[]) {
        return 0.0;
      };

      PetscScalar constant(const double x[]) {
        return -4.0;
      };

      PetscScalar nonlinear_2d(const double x[]) {
        return -4.0 - lambda*PetscExpScalar(x[0]*x[0] + x[1]*x[1]);
      };

      PetscScalar singularity_2d(const double x[]) {
        return 0.;
      };

      PetscScalar singularity_exact_2d(const double x[]) {
        double r = sqrt(x[0]*x[0] + x[1]*x[1]);
        double theta;
        if (r == 0.) {
          return 0.;
        } else theta = asin(x[1]/r);
        if (x[0] < 0) {
          theta = 2*M_PI - theta;
        }
        return pow(r, 2./3.)*sin((2./3.)*theta);
      };

      PetscScalar singularity_exact_3d(const double x[]) {
        return sin(x[0] + x[1] + x[2]);  
      };

      PetscScalar singularity_3d(const double x[]) {
        return (3)*sin(x[0] + x[1] + x[2]);
      };

      PetscScalar linear_2d(const double x[]) {
        return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5);
      };

      PetscScalar quadratic_2d(const double x[]) {
        return x[0]*x[0] + x[1]*x[1];
      };

      PetscScalar cubic_2d(const double x[]) {
        return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + 0.5;
      };

      PetscScalar nonlinear_3d(const double x[]) {
        return -4.0 - lambda*PetscExpScalar((2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
      };

      PetscScalar linear_3d(const double x[]) {
        return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5) - 6.0*(x[2] - 0.5);
      };

      PetscScalar quadratic_3d(const double x[]) {
        return (2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      };

      PetscScalar cubic_3d(const double x[]) {
        return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + x[2]*x[2]*x[2] - 1.5*x[2]*x[2] + 0.75;
      };

      PetscScalar cos_x(const double x[]) {
        return cos(2.0*PETSC_PI*x[0]);
      };
      #undef __FUNCT__
      #define __FUNCT__ "Function_Structured_2d"
      PetscErrorCode Function_Structured_2d(DALocalInfo *info, PetscScalar *x[], PetscScalar *f[], void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        PetscScalar  (*func)(const double *) = options->func;
        DA             coordDA;
        Vec            coordinates;
        DACoor2d     **coords;
        PetscInt       i, j;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
        ierr = DAGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
        ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        for(j = info->ys; j < info->ys+info->ym; j++) {
          for(i = info->xs; i < info->xs+info->xm; i++) {
            f[j][i] = func((PetscReal *) &coords[j][i]);
          }
        }
        ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        PetscFunctionReturn(0); 
      };
      #undef __FUNCT__
      #define __FUNCT__ "Rhs_Structured_2d_FD"
      PetscErrorCode Rhs_Structured_2d_FD(DALocalInfo *info, PetscScalar *x[], PetscScalar *f[], void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        PetscScalar  (*func)(const double *)   = options->func;
        PetscScalar  (*bcFunc)(const double *) = options->exactFunc;
        const double   lambda                  = options->lambda;
        DA             coordDA;
        Vec            coordinates;
        DACoor2d     **coords;
        PetscReal      hxa, hxb, hx, hya, hyb, hy;
        PetscInt       ie = info->xs+info->xm;
        PetscInt       je = info->ys+info->ym;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
        ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
        ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        // Loop over stencils
        for(int j = info->ys; j < je; j++) {
          for(int i = info->xs; i < ie; i++) {
            if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
              f[j][i] = x[j][i] - bcFunc((PetscReal *) &coords[j][i]);
            } else {
              hya = coords[j+1][i].y - coords[j][i].y;
              hyb = coords[j][i].y   - coords[j-1][i].y;
              hxa = coords[j][i+1].x - coords[j][i].x;
              hxb = coords[j][i].x   - coords[j][i-1].x;
              hy  = 0.5*(hya+hyb);
              hx  = 0.5*(hxa+hxb);
              f[j][i] = -func((const double *) &coords[j][i])*hx*hy -
                ((x[j][i+1] - x[j][i])/hxa - (x[j][i] - x[j][i-1])/hxb)*hy -
                ((x[j+1][i] - x[j][i])/hya - (x[j][i] - x[j-1][i])/hyb)*hx -
                lambda*hx*hy*PetscExpScalar(x[j][i]);
            }
          }
        }
        ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        PetscFunctionReturn(0); 
      };
      #undef __FUNCT__
      #define __FUNCT__ "Jac_Structured_2d_FD"
      PetscErrorCode Jac_Structured_2d_FD(DALocalInfo *info, PetscScalar *x[], Mat J, void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        const double   lambda  = options->lambda;
        DA             coordDA;
        Vec            coordinates;
        DACoor2d     **coords;
        MatStencil     row, col[5];
        PetscScalar    v[5];
        PetscReal      hxa, hxb, hx, hya, hyb, hy;
        PetscInt       ie = info->xs+info->xm;
        PetscInt       je = info->ys+info->ym;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
        ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
        ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        // Loop over stencils
        for(int j = info->ys; j < je; j++) {
          for(int i = info->xs; i < ie; i++) {
            row.j = j; row.i = i;
            if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
              v[0] = 1.0;
              ierr = MatSetValuesStencil(J, 1, &row, 1, &row, v, INSERT_VALUES);CHKERRQ(ierr);
            } else {
              hya = coords[j+1][i].y - coords[j][i].y;
              hyb = coords[j][i].y   - coords[j-1][i].y;
              hxa = coords[j][i+1].x - coords[j][i].x;
              hxb = coords[j][i].x   - coords[j][i-1].x;
              hy  = 0.5*(hya+hyb);
              hx  = 0.5*(hxa+hxb);
              v[0] = -hx/hyb;                                          col[0].j = j - 1; col[0].i = i;
              v[1] = -hy/hxb;                                          col[1].j = j;     col[1].i = i-1;
              v[2] = (hy/hxa + hy/hxb + hx/hya + hx/hyb);              col[2].j = row.j; col[2].i = row.i;
              v[3] = -hy/hxa;                                          col[3].j = j;     col[3].i = i+1;
              v[4] = -hx/hya;                                          col[4].j = j + 1; col[4].i = i;
              v[2] -= lambda*hx*hy*PetscExpScalar(x[j][i]);
              ierr = MatSetValuesStencil(J, 1, &row, 5, col, v, INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }
        ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        PetscFunctionReturn(0); 
      };
      #undef __FUNCT__
      #define __FUNCT__ "Function_Structured_3d"
      PetscErrorCode Function_Structured_3d(DALocalInfo *info, PetscScalar **x[], PetscScalar **f[], void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        PetscScalar  (*func)(const double *) = options->func;
        DA             coordDA;
        Vec            coordinates;
        DACoor3d    ***coords;
        PetscInt       i, j, k;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
        ierr = DAGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
        ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        for(k = info->zs; k < info->zs+info->zm; k++) {
          for(j = info->ys; j < info->ys+info->ym; j++) {
            for(i = info->xs; i < info->xs+info->xm; i++) {
              f[k][j][i] = func((PetscReal *) &coords[k][j][i]);
            }
          }
        }
        ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        PetscFunctionReturn(0); 
      };
      #undef __FUNCT__
      #define __FUNCT__ "Rhs_Structured_3d_FD"
      PetscErrorCode Rhs_Structured_3d_FD(DALocalInfo *info, PetscScalar **x[], PetscScalar **f[], void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        PetscScalar  (*func)(const double *)   = options->func;
        PetscScalar  (*bcFunc)(const double *) = options->exactFunc;
        const double   lambda                  = options->lambda;
        DA             coordDA;
        Vec            coordinates;
        DACoor3d    ***coords;
        PetscReal      hxa, hxb, hx, hya, hyb, hy, hza, hzb, hz;
        PetscInt       ie = info->xs+info->xm;
        PetscInt       je = info->ys+info->ym;
        PetscInt       ke = info->zs+info->zm;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
        ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
        ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        // Loop over stencils
        for(int k = info->zs; k < ke; k++) {
          for(int j = info->ys; j < je; j++) {
            for(int i = info->xs; i < ie; i++) {
              if (i == 0 || j == 0 || k == 0 || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
                f[k][j][i] = x[k][j][i] - bcFunc((PetscReal *) &coords[k][j][i]);
              } else {
                hza = coords[k+1][j][i].z - coords[k][j][i].z;
                hzb = coords[k][j][i].z   - coords[k-1][j][i].z;
                hya = coords[k][j+1][i].y - coords[k][j][i].y;
                hyb = coords[k][j][i].y   - coords[k][j-1][i].y;
                hxa = coords[k][j][i+1].x - coords[k][j][i].x;
                hxb = coords[k][j][i].x   - coords[k][j][i-1].x;
                hz  = 0.5*(hza+hzb);
                hy  = 0.5*(hya+hyb);
                hx  = 0.5*(hxa+hxb);
                f[k][j][i] = -func((const double *) &coords[k][j][i])*hx*hy*hz -
                  ((x[k][j][i+1] - x[k][j][i])/hxa - (x[k][j][i] - x[k][j][i-1])/hxb)*hy*hz -
                  ((x[k][j+1][i] - x[k][j][i])/hya - (x[k][j][i] - x[k][j-1][i])/hyb)*hx*hz - 
                  ((x[k+1][j][i] - x[k][j][i])/hza - (x[k][j][i] - x[k-1][j][i])/hzb)*hx*hy -
                  lambda*hx*hy*hz*PetscExpScalar(x[k][j][i]);
              }
            }
          }
        }
        ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        PetscFunctionReturn(0); 
      };
      #undef __FUNCT__
      #define __FUNCT__ "Jac_Structured_3d_FD"
      PetscErrorCode Jac_Structured_3d_FD(DALocalInfo *info, PetscScalar **x[], Mat J, void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        const double   lambda  = options->lambda;
        DA             coordDA;
        Vec            coordinates;
        DACoor3d    ***coords;
        MatStencil     row, col[7];
        PetscScalar    v[7];
        PetscReal      hxa, hxb, hx, hya, hyb, hy, hza, hzb, hz;
        PetscInt       ie = info->xs+info->xm;
        PetscInt       je = info->ys+info->ym;
        PetscInt       ke = info->zs+info->zm;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
        ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
        ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        // Loop over stencils
        for(int k = info->zs; k < ke; k++) {
          for(int j = info->ys; j < je; j++) {
            for(int i = info->xs; i < ie; i++) {
              row.k = k; row.j = j; row.i = i;
              if (i == 0 || j == 0 || k == 0 || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
                v[0] = 1.0;
                ierr = MatSetValuesStencil(J, 1, &row, 1, &row, v, INSERT_VALUES);CHKERRQ(ierr);
              } else {
                hza = coords[k+1][j][i].z - coords[k][j][i].z;
                hzb = coords[k][j][i].z   - coords[k-1][j][i].z;
                hya = coords[k][j+1][i].y - coords[k][j][i].y;
                hyb = coords[k][j][i].y   - coords[k][j-1][i].y;
                hxa = coords[k][j][i+1].x - coords[k][j][i].x;
                hxb = coords[k][j][i].x   - coords[k][j][i-1].x;
                hz  = 0.5*(hza+hzb);
                hy  = 0.5*(hya+hyb);
                hx  = 0.5*(hxa+hxb);
                v[0] = -hx*hy/hzb;                                       col[0].k = k - 1; col[0].j = j;     col[0].i = i;
                v[1] = -hx*hz/hyb;                                       col[1].k = k;     col[1].j = j - 1; col[1].i = i;
                v[2] = -hy*hz/hxb;                                       col[2].k = k;     col[2].j = j;     col[2].i = i - 1;
                v[3] = (hy*hz/hxa + hy*hz/hxb + hx*hz/hya + hx*hz/hyb + hx*hy/hza + hx*hy/hzb); col[3].k = row.k; col[3].j = row.j; col[3].i = row.i;
                v[4] = -hx*hy/hza;                                       col[4].k = k + 1; col[4].j = j;     col[4].i = i;
                v[5] = -hx*hz/hya;                                       col[5].k = k;     col[5].j = j + 1; col[5].i = i;
                v[6] = -hy*hz/hxa;                                       col[6].k = k;     col[6].j = j;     col[6].i = i + 1;
                v[3] -= lambda*hx*hy*hz*PetscExpScalar(x[k][j][i]);
                ierr = MatSetValuesStencil(J, 1, &row, 7, col, v, INSERT_VALUES);CHKERRQ(ierr);
              }
            }
          }
        }
        ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        PetscFunctionReturn(0); 
      };
      #undef __FUNCT__
      #define __FUNCT__ "Rhs_Unstructured"
      PetscErrorCode Rhs_Unstructured(::Mesh mesh, SectionReal X, SectionReal section, void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        PetscScalar  (*func)(const double *) = options->func;
        const double   lambda                = options->lambda;
        Obj<PETSC_MESH_TYPE> m;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
        const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
        const int                                numQuadPoints = disc->getQuadratureSize();
        const double                            *quadPoints    = disc->getQuadraturePoints();
        const double                            *quadWeights   = disc->getQuadratureWeights();
        const int                                numBasisFuncs = disc->getBasisSize();
        const double                            *basis         = disc->getBasis();
        const double                            *basisDer      = disc->getBasisDerivatives();
        const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates   = m->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = m->heightStratum(0);
        const int                                dim           = m->getDimension();
        double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
        PetscScalar *elemVec, *elemMat;

        ierr = SectionRealZero(section);CHKERRQ(ierr);
        ierr = PetscMalloc2(numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
        ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          if (detJ <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
          PetscScalar *x;

          ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
          // Loop over quadrature points
          for(int q = 0; q < numQuadPoints; ++q) {
            for(int d = 0; d < dim; d++) {
              coords[d] = v0[d];
              for(int e = 0; e < dim; e++) {
                coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
              }
            }
            const PetscScalar funcVal  = (*func)(coords);
            PetscScalar       fieldVal = 0.0;

            for(int f = 0; f < numBasisFuncs; ++f) {
              fieldVal += x[f]*basis[q*numBasisFuncs+f];
            }
            // Loop over trial functions
            for(int f = 0; f < numBasisFuncs; ++f) {
              // Constant part
              elemVec[f] -= basis[q*numBasisFuncs+f]*funcVal*quadWeights[q]*detJ;
              // Linear part
              for(int d = 0; d < dim; ++d) {
                t_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
              }
              // Loop over basis functions
              for(int g = 0; g < numBasisFuncs; ++g) {
                // Linear part
                for(int d = 0; d < dim; ++d) {
                  b_der[d] = 0.0;
                  for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
                }
                PetscScalar product = 0.0;
                for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
                elemMat[f*numBasisFuncs+g] += product*quadWeights[q]*detJ;
              }
              // Nonlinear part
              if (lambda != 0.0) {
                elemVec[f] -= basis[q*numBasisFuncs+f]*lambda*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
              }
            }
          }    
          // Add linear contribution
          for(int f = 0; f < numBasisFuncs; ++f) {
            for(int g = 0; g < numBasisFuncs; ++g) {
              elemVec[f] += elemMat[f*numBasisFuncs+g]*x[g];
            }
          }
          ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
        }
        ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
        ierr = PetscFree6(t_der,b_der,coords,v0,J,invJ);CHKERRQ(ierr);
        // Exchange neighbors
        ierr = SectionRealComplete(section);CHKERRQ(ierr);
        // Subtract the constant
        if (m->hasRealSection("constant")) {
          const Obj<PETSC_MESH_TYPE::real_section_type>& constant = m->getRealSection("constant");
          Obj<PETSC_MESH_TYPE::real_section_type>        s;

          ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
          s->axpy(-1.0, constant);
        }
	Obj<PETSC_MESH_TYPE::real_section_type>        s;
	ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
	s->view("RHS");
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "Jac_Unstructured"
      PetscErrorCode Jac_Unstructured(::Mesh mesh, SectionReal section, Mat A, void *ctx) {
        BratuOptions  *options = (BratuOptions *) ctx;
        const double   lambda  = options->lambda;
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        Obj<PETSC_MESH_TYPE> m;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = MatZeroEntries(A);CHKERRQ(ierr);
        ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
        ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
        const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
        const int                                numQuadPoints = disc->getQuadratureSize();
        const double                            *quadWeights   = disc->getQuadratureWeights();
        const int                                numBasisFuncs = disc->getBasisSize();
        const double                            *basis         = disc->getBasis();
        const double                            *basisDer      = disc->getBasisDerivatives();
        const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates   = m->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = m->heightStratum(0);
        const Obj<PETSC_MESH_TYPE::order_type>&        order         = m->getFactory()->getGlobalOrder(m, "default", s);
        const int                                dim           = m->getDimension();
        double      *t_der, *b_der, *v0, *J, *invJ, detJ;
        PetscScalar *elemMat;

        ierr = PetscMalloc(numBasisFuncs*numBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
        ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          PetscScalar *u;

          ierr = SectionRealRestrict(section, *c_iter, &u);CHKERRQ(ierr);
          // Loop over quadrature points
          for(int q = 0; q < numQuadPoints; ++q) {
            PetscScalar fieldVal = 0.0;

            for(int f = 0; f < numBasisFuncs; ++f) {
              fieldVal += u[f]*basis[q*numBasisFuncs+f];
            }
            // Loop over trial functions
            for(int f = 0; f < numBasisFuncs; ++f) {
              for(int d = 0; d < dim; ++d) {
                t_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
              }
              // Loop over basis functions
              for(int g = 0; g < numBasisFuncs; ++g) {
                for(int d = 0; d < dim; ++d) {
                  b_der[d] = 0.0;
                  for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
                }
                PetscScalar product = 0.0;
                for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
                elemMat[f*numBasisFuncs+g] += product*quadWeights[q]*detJ;
                // Nonlinear part
                if (lambda != 0.0) {
                  elemMat[f*numBasisFuncs+g] -= basis[q*numBasisFuncs+f]*basis[q*numBasisFuncs+g]*lambda*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
                }
              }
            }
          }
          ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = PetscFree(elemMat);CHKERRQ(ierr);
        ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatView(A, PETSC_VIEWER_STDOUT_SELF);
        PetscFunctionReturn(0);
      };
    };
    class Bratu : ALE::ParallelObject {
    public:
    protected:
      BratuOptions         _options;
      DM                   _dm;
      Obj<PETSC_MESH_TYPE> _mesh;
      DMMG                *_dmmg;
    public:
      Bratu(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
        PetscErrorCode ierr = this->processOptions(comm, &this->_options);CHKERRXX(ierr);
        this->_dm   = PETSC_NULL;
        this->_dmmg = PETSC_NULL;
      };
      ~Bratu() {
        PetscErrorCode ierr;

        if (this->_dmmg) {ierr = DMMGDestroy(this->_dmmg);CHKERRXX(ierr);}
        ierr = this->destroyExactSolution(this->_options.exactSol);CHKERRXX(ierr);
        ierr = this->destroyExactSolution(this->_options.error);CHKERRXX(ierr);
        ierr = this->destroyMesh();CHKERRXX(ierr);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "BratuProcessOptions"
      PetscErrorCode processOptions(MPI_Comm comm, BratuOptions *options) {
        const char    *runTypes[3] = {"full", "test", "mesh"};
        const char    *bcTypes[2]  = {"neumann", "dirichlet"};
        const char    *asTypes[4]  = {"full", "stored", "calculated"};
        ostringstream  filename;
        PetscInt       run, bc, as;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        options->debug            = 0;
        options->run              = RUN_FULL;
        options->dim              = 2;
        options->structured       = PETSC_TRUE;
        options->generateMesh     = PETSC_TRUE;
        options->interpolate      = PETSC_TRUE;
        options->refinementLimit  = 0.0;
        options->bcType           = DIRICHLET;
        options->operatorAssembly = ASSEMBLY_FULL;
        options->lambda           = 0.0;
        options->reentrantMesh    = PETSC_FALSE;
        options->circularMesh     = PETSC_FALSE;
        options->refineSingularity= PETSC_FALSE;

        ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMG");CHKERRQ(ierr);
          ierr = PetscOptionsInt("-debug", "The debugging level", "bratu.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
          run = options->run;
          ierr = PetscOptionsEList("-run", "The run type", "bratu.cxx", runTypes, 3, runTypes[options->run], &run, PETSC_NULL);CHKERRQ(ierr);
          options->run = (RunType) run;
          ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "bratu.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-reentrant", "Make a reentrant-corner mesh", "bratu.cxx", options->reentrantMesh, &options->reentrantMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-circular_mesh", "Make a reentrant-corner mesh", "bratu.cxx", options->circularMesh, &options->circularMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-singularity", "Refine the mesh around a singularity with a priori poisson error estimation", "bratu.cxx", options->refineSingularity, &options->refineSingularity, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-structured", "Use a structured mesh", "bratu.cxx", options->structured, &options->structured, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "bratu.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "bratu.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "bratu.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
          filename << "data/bratu_" << options->dim <<"d";
          ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
          ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "bratu.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
          ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
          bc = options->bcType;
          ierr = PetscOptionsEList("-bc_type","Type of boundary condition","bratu.cxx",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
          options->bcType = (BCType) bc;
          as = options->operatorAssembly;
          ierr = PetscOptionsEList("-assembly_type","Type of operator assembly","bratu.cxx",asTypes,3,asTypes[options->operatorAssembly],&as,PETSC_NULL);CHKERRQ(ierr);
          options->operatorAssembly = (AssemblyType) as;
          ierr = PetscOptionsReal("-lambda", "The parameter controlling nonlinearity", "bratu.cxx", options->lambda, &options->lambda, PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();

        ALE::Problem::BratuFunctions::lambda = options->lambda;
        this->setDebug(options->debug);
        PetscFunctionReturn(0);
      };
    public: // Accessors
      BratuOptions *getOptions() {return &this->_options;};
      int  dim() const {return this->_options.dim;};
      bool structured() const {return this->_options.structured;};
      void structured(const bool s) {this->_options.structured = (PetscTruth) s;};
      bool interpolated() const {return this->_options.interpolate;};
      void interpolated(const bool i) {this->_options.interpolate = (PetscTruth) i;};
      BCType bcType() const {return this->_options.bcType;};
      void bcType(const BCType bc) {this->_options.bcType = bc;};
      AssemblyType opAssembly() const {return this->_options.operatorAssembly;};
      void opAssembly(const AssemblyType at) {this->_options.operatorAssembly = at;};
      DM getDM() const {return this->_dm;};
      DMMG *getDMMG() const {return this->_dmmg;};
      ALE::Problem::ExactSolType exactSolution() const {return this->_options.exactSol;};
    public: // Mesh
      #undef __FUNCT__
      #define __FUNCT__ "CreateMesh"
      PetscErrorCode createMesh() {
        PetscTruth     view;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          DA       da;
          PetscInt dof = 1;
          PetscInt pd  = PETSC_DECIDE;

          if (dim() == 2) {
            ierr = DACreate2d(comm(), DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DACreate3d(comm(), DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, -3, pd, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = DASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
          this->_dm = (DM) da;
          PetscFunctionReturn(0);
        }
        if (_options.circularMesh) {
          if (_options.reentrantMesh) {
            _options.reentrant_angle = .9;
            ierr = ALE::DMBuilder::createReentrantSphericalMesh(comm(), dim(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          } else {
            ierr = ALE::DMBuilder::createSphericalMesh(comm(), dim(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          }
        } else {
          if (_options.reentrantMesh) {
            _options.reentrant_angle = .75;
            ierr = ALE::DMBuilder::createReentrantBoxMesh(comm(), dim(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          } else {
            ierr = ALE::DMBuilder::createBoxMesh(comm(), dim(), structured(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          }
        }
	ierr = refineMesh();CHKERRQ(ierr);

        if (this->commSize() > 1) {
          ::Mesh parallelMesh;

          ierr = MeshDistribute((::Mesh) this->_dm, _options.partitioner, &parallelMesh);CHKERRQ(ierr);
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
          this->_dm = (DM) parallelMesh;
        }
        ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);
        if (bcType() == DIRICHLET) {
          this->_mesh->markBoundaryCells("marker");
        }
        ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
        if (view) {
          PetscViewer viewer;

          ierr = PetscViewerCreate(this->comm(), &viewer);CHKERRQ(ierr);
          ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
          ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
          ierr = PetscViewerFileSetName(viewer, "bratu.vtk");CHKERRQ(ierr);
          ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
          ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
        }
        ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
        if (view) {this->_mesh->view("Mesh");}
        ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_simple", &view);CHKERRQ(ierr);
        if (view) {ierr = MeshView((::Mesh) this->_dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "RefineMesh"
      PetscErrorCode refineMesh() {
        PetscErrorCode ierr;
	PetscFunctionBegin;
        if (_options.refinementLimit > 0.0) {
          ::Mesh refinedMesh;

          ierr = MeshRefine((::Mesh) this->_dm, _options.refinementLimit, (PetscTruth) interpolated(), &refinedMesh);CHKERRQ(ierr);
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
          this->_dm = (DM) refinedMesh;
          ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);

          if (_options.refineSingularity) {
            ::Mesh refinedMesh2;
            double singularity[3] = {0.0, 0.0, 0.0};

            if (dim() == 2) {
              ierr = ALE::DMBuilder::MeshRefineSingularity((::Mesh) this->_dm, singularity, _options.reentrant_angle, &refinedMesh2);CHKERRQ(ierr);
            } else if (dim() == 3) {
              ierr = ALE::DMBuilder::MeshRefineSingularity_Fichera((::Mesh) this->_dm, singularity, 0.75, &refinedMesh2);CHKERRQ(ierr);
            }
            ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
            this->_dm = (DM) refinedMesh2;
            ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);
#ifndef PETSC_OPT_SIEVE
            ierr = MeshIDBoundary((::Mesh) this->_dm);CHKERRQ(ierr);
#endif
          }  
        }
	PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "DestroyMesh"
      PetscErrorCode destroyMesh() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          ierr = DADestroy((DA) this->_dm);CHKERRQ(ierr);
        } else {
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateProblem"
      PetscErrorCode createProblem() {
        PetscFunctionBegin;
        if (dim() == 2) {
          if (bcType() == DIRICHLET) {
            if (this->_options.lambda > 0.0) {
              this->_options.func      = ALE::Problem::BratuFunctions::nonlinear_2d;
              this->_options.exactFunc = ALE::Problem::BratuFunctions::quadratic_2d;
            } else if (this->_options.reentrantMesh) { 
              this->_options.func      = ALE::Problem::BratuFunctions::singularity_2d;
              this->_options.exactFunc = ALE::Problem::BratuFunctions::singularity_exact_2d;
            } else {
              this->_options.func      = ALE::Problem::BratuFunctions::constant;
              this->_options.exactFunc = ALE::Problem::BratuFunctions::quadratic_2d;
            }
          } else {
            this->_options.func      = ALE::Problem::BratuFunctions::linear_2d;
            this->_options.exactFunc = ALE::Problem::BratuFunctions::cubic_2d;
          }
        } else if (dim() == 3) {
          if (bcType() == DIRICHLET) {
            if (this->_options.reentrantMesh) {
              this->_options.func      = ALE::Problem::BratuFunctions::singularity_3d;
              this->_options.exactFunc = ALE::Problem::BratuFunctions::singularity_exact_3d;
            } else {
              if (this->_options.lambda > 0.0) {
                this->_options.func    = ALE::Problem::BratuFunctions::nonlinear_3d;
              } else {
                this->_options.func    = ALE::Problem::BratuFunctions::constant;
              }
              this->_options.exactFunc = ALE::Problem::BratuFunctions::quadratic_3d;
            }
          } else {
            this->_options.func      = ALE::Problem::BratuFunctions::linear_3d;
            this->_options.exactFunc = ALE::Problem::BratuFunctions::cubic_3d;
          }
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
        }
        if (!structured()) {
          int            numBC      = (bcType() == DIRICHLET) ? 1 : 0;
          int            markers[1] = {1};
          double       (*funcs[1])(const double *coords) = {this->_options.exactFunc};
          PetscErrorCode ierr;

          if (dim() == 1) {
            ierr = CreateProblem_gen_0(this->_dm, "u", numBC, markers, funcs, this->_options.exactFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_0;
          } else if (dim() == 2) {
            ierr = CreateProblem_gen_1(this->_dm, "u", numBC, markers, funcs, this->_options.exactFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_1;
          } else if (dim() == 3) {
            ierr = CreateProblem_gen_2(this->_dm, "u", numBC, markers, funcs, this->_options.exactFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_2;
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("default");
          s->setDebug(debug());
          this->_mesh->setupField(s);
          if (debug()) {s->view("Default field");}
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateExactSolution"
      PetscErrorCode createExactSolution() {
        PetscTruth     flag;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          DA            da = (DA) this->_dm;
          PetscScalar (*func)(const double *) = this->_options.func;
          Vec           X, U;

          ierr = DAGetGlobalVector(da, &X);CHKERRQ(ierr);
          ierr = DACreateGlobalVector(da, &this->_options.exactSol.vec);CHKERRQ(ierr);
          this->_options.func = this->_options.exactFunc;
          U                   = exactSolution().vec;
          if (dim() == 2) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::BratuFunctions::Function_Structured_2d, X, U, (void *) &this->_options);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::BratuFunctions::Function_Structured_3d, X, U, (void *) &this->_options);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = DARestoreGlobalVector(da, &X);CHKERRQ(ierr);
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
          if (flag) {ierr = VecView(U, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
          if (flag) {ierr = VecView(U, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
          this->_options.func = func;
          ierr = DACreateGlobalVector(da, &this->_options.error.vec);CHKERRQ(ierr);
        } else {
          ::Mesh mesh = (::Mesh) this->_dm;

          ierr = MeshGetSectionReal(mesh, "exactSolution", &this->_options.exactSol.section);CHKERRQ(ierr);
          const Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("exactSolution");
          this->_mesh->setupField(s);
          const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = this->_mesh->heightStratum(0);
          const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = this->_mesh->getRealSection("coordinates");
          const int                                       localDof    = this->_mesh->sizeWithBC(s, *cells->begin());
          PETSC_MESH_TYPE::real_section_type::value_type *values      = new PETSC_MESH_TYPE::real_section_type::value_type[localDof];
          double                                         *v0          = new double[dim()];
          double                                         *J           = new double[dim()*dim()];
          double                                          detJ;
          ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(this->_mesh->getSieve()->getMaxConeSize(), this->_mesh->depth())+1, true);

          for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), *c_iter, pV);
            const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
            const int                          oSize   = pV.getSize();
            int                                v       = 0;

            this->_mesh->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
            for(int cl = 0; cl < oSize; ++cl) {
              const int pointDim = s->getFiberDimension(oPoints[cl]);

              if (pointDim) {
                for(int d = 0; d < pointDim; ++d, ++v) {
                  values[v] = (*this->_options.integrate)(v0, J, v, this->_options.exactFunc);
                }
              }
            }
            this->_mesh->updateAll(s, *c_iter, values);
            pV.clear();
          }
          delete [] values;
          delete [] v0;
          delete [] J;
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
          if (flag) {s->view("Exact Solution");}
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
          if (flag) {
            PetscViewer viewer;

            ierr = PetscViewerCreate(this->comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "exact_sol.vtk");CHKERRQ(ierr);
            ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
            ierr = SectionRealView(exactSolution().section, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
          }
          ierr = MeshGetSectionReal(mesh, "error", &this->_options.error.section);CHKERRQ(ierr);
          const Obj<PETSC_MESH_TYPE::real_section_type>& e = this->_mesh->getRealSection("error");
          e->setChart(PETSC_MESH_TYPE::real_section_type::chart_type(*this->_mesh->heightStratum(0)));
          e->setFiberDimension(this->_mesh->heightStratum(0), 1);
          this->_mesh->allocate(e);
        }
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "DestroyExactSolution"
      PetscErrorCode destroyExactSolution(ALE::Problem::ExactSolType sol) {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          ierr = VecDestroy(sol.vec);CHKERRQ(ierr);
        } else {
          ierr = SectionRealDestroy(sol.section);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateSolver"
      PetscErrorCode createSolver() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGCreate(this->comm(), 1, &this->_options, &this->_dmmg);CHKERRQ(ierr);
        ierr = DMMGSetDM(this->_dmmg, this->_dm);CHKERRQ(ierr);
        if (structured()) {
          // Needed if using finite elements
          // ierr = PetscOptionsSetValue("-dmmg_form_function_ghost", PETSC_NULL);CHKERRQ(ierr);
          if (dim() == 2) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::BratuFunctions::Rhs_Structured_2d_FD, ALE::Problem::BratuFunctions::Jac_Structured_2d_FD, 0, 0);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::BratuFunctions::Rhs_Structured_3d_FD, ALE::Problem::BratuFunctions::Jac_Structured_3d_FD, 0, 0);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
          for(int l = 0; l < DMMGGetLevels(this->_dmmg); l++) {
            ierr = DASetUniformCoordinates((DA) (this->_dmmg)[l]->dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
          }
        } else {
          if (opAssembly() == ALE::Problem::ASSEMBLY_FULL) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::BratuFunctions::Rhs_Unstructured, ALE::Problem::BratuFunctions::Jac_Unstructured, 0, 0);CHKERRQ(ierr);
#if 0
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_CALCULATED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::BratuFunctions::Rhs_Unstructured, ALE::Problem::BratuFunctions::Jac_Unstructured_Calculated, 0, 0);CHKERRQ(ierr);
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_STORED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::BratuFunctions::Rhs_Unstructured, ALE::Problem::BratuFunctions::Jac_Unstructured_Stored, 0, 0);CHKERRQ(ierr);
#endif
          } else {
            SETERRQ1(PETSC_ERR_ARG_WRONG, "Assembly type not supported: %d", opAssembly());
          }
          ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
        }
        if (bcType() == ALE::Problem::NEUMANN) {
          // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
          ierr = DMMGSetNullSpace(this->_dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "BratuSolve"
      PetscErrorCode solve() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGSolve(this->_dmmg);CHKERRQ(ierr);
        // Report on solve
        SNES                snes = DMMGGetSNES(this->_dmmg);
        PetscInt            its;
        PetscTruth          flag;
        SNESConvergedReason reason;

        ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
        ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
        ierr = PetscPrintf(comm(), "Number of nonlinear iterations = %D\n", its);CHKERRQ(ierr);
        ierr = PetscPrintf(comm(), "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
        if (flag) {ierr = VecView(DMMGGetx(this->_dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
        if (flag && dim() == 2) {ierr = VecView(DMMGGetx(this->_dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
        if (structured()) {
          ALE::Problem::ExactSolType sol;

          sol.vec = DMMGGetx(this->_dmmg);
          if (DMMGGetLevels(this->_dmmg) == 1) {ierr = this->checkError(sol);CHKERRQ(ierr);}
        } else {
          const Obj<PETSC_MESH_TYPE::real_section_type>& sol = this->_mesh->getRealSection("default");
          SectionReal solution;
          double      error;

          ierr = MeshGetSectionReal((::Mesh) this->_dm, "default", &solution);CHKERRQ(ierr);
          ierr = SectionRealToVec(solution, (::Mesh) this->_dm, SCATTER_REVERSE, DMMGGetx(this->_dmmg));CHKERRQ(ierr);
          ierr = this->calculateError(solution, &error);CHKERRQ(ierr);
          ierr = PetscPrintf(comm(), "Total error: %g\n", error);CHKERRQ(ierr);
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
          if (flag) {
            PetscViewer viewer;

            ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "sol.vtk");CHKERRQ(ierr);
            ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
            ierr = SectionRealView(solution, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

            ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "error.vtk");CHKERRQ(ierr);
            ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
            ierr = SectionRealView(this->_options.error.section, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
          }
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
          if (flag) {sol->view("Solution");}
          ierr = PetscOptionsHasName(PETSC_NULL, "-hierarchy_vtk", &flag);CHKERRQ(ierr);
          if (flag) {
            double      offset[3] = {2.0, 0.0, 0.25};
            PetscViewer viewer;

            ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "mesh_hierarchy.vtk");CHKERRQ(ierr);
            ierr = PetscOptionsReal("-hierarchy_vtk", PETSC_NULL, "bratu.cxx", *offset, offset, PETSC_NULL);CHKERRQ(ierr);
            ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
            ierr = VTKViewer::writeHierarchyVertices(this->_dmmg, viewer, offset);CHKERRQ(ierr);
            ierr = VTKViewer::writeHierarchyElements(this->_dmmg, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
          }
          ierr = SectionRealDestroy(solution);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CalculateError"
      PetscErrorCode calculateError(SectionReal X, double *error) {
        PetscScalar  (*func)(const double *) = this->_options.exactFunc;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        const Obj<ALE::Discretization>&                disc          = this->_mesh->getDiscretization("u");
        const int                                      numQuadPoints = disc->getQuadratureSize();
        const double                                  *quadPoints    = disc->getQuadraturePoints();
        const double                                  *quadWeights   = disc->getQuadratureWeights();
        const int                                      numBasisFuncs = disc->getBasisSize();
        const double                                  *basis         = disc->getBasis();
        const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates   = this->_mesh->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = this->_mesh->heightStratum(0);
        const int                                      dim           = this->_mesh->getDimension();
        double *coords, *v0, *J, *invJ, detJ;
        double  localError = 0.0;

        ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          PetscScalar *x;
          double       elemError = 0.0;

          this->_mesh->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          if (debug()) {
            std::cout << "Element " << *c_iter << " v0: (" << v0[0]<<","<<v0[1]<<")" << "J " << J[0]<<","<<J[1]<<","<<J[2]<<","<<J[3] << " detJ " << detJ << std::endl;
          }
          ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
          // Loop over quadrature points
          for(int q = 0; q < numQuadPoints; ++q) {
            for(int d = 0; d < dim; d++) {
              coords[d] = v0[d];
              for(int e = 0; e < dim; e++) {
                coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
              }
              if (debug()) {std::cout << "q: "<<q<<"  coords["<<d<<"] " << coords[d] << std::endl;}
            }
            const PetscScalar funcVal = (*func)(coords);
            if (debug()) {std::cout << "q: "<<q<<"  funcVal " << funcVal << std::endl;}

            double interpolant = 0.0;
            for(int f = 0; f < numBasisFuncs; ++f) {
              interpolant += x[f]*basis[q*numBasisFuncs+f];
            }
            if (debug()) {std::cout << "q: "<<q<<"  interpolant " << interpolant << std::endl;}
            elemError += (interpolant - funcVal)*(interpolant - funcVal)*quadWeights[q];
            if (debug()) {std::cout << "q: "<<q<<"  elemError " << elemError << std::endl;}
          }
          if (debug()) {
            std::cout << "Element " << *c_iter << " error: " << elemError << std::endl;
          }
          ierr = SectionRealUpdateAdd(this->_options.error.section, *c_iter, &elemError);CHKERRQ(ierr);
          localError += elemError;
        }
        ierr = MPI_Allreduce(&localError, error, 1, MPI_DOUBLE, MPI_SUM, comm());CHKERRQ(ierr);
        ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
        *error = sqrt(*error);
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "CheckError"
      PetscErrorCode checkError(ALE::Problem::ExactSolType sol) {
        const char    *name;
        PetscScalar    norm;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          DA  da = (DA) this->_dm;
          Vec error;

          ierr = DAGetGlobalVector(da, &error);CHKERRQ(ierr);
          ierr = VecCopy(sol.vec, error);CHKERRQ(ierr);
          ierr = VecAXPY(error, -1.0, exactSolution().vec);CHKERRQ(ierr);
          ierr = VecNorm(error, NORM_2, &norm);CHKERRQ(ierr);
          ierr = DARestoreGlobalVector(da, &error);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.vec, &name);CHKERRQ(ierr);
        } else {
          ierr = this->calculateError(sol.section, &norm);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
        }
        PetscPrintf(comm(), "Error for trial solution %s: %g\n", name, norm);
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "CheckResidual"
      PetscErrorCode checkResidual(ALE::Problem::ExactSolType sol) {
        const char    *name;
        PetscScalar    norm;
        PetscTruth     flag;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
        if (structured()) {
          DA  da = (DA) this->_dm;
          Vec residual;

          ierr = DAGetGlobalVector(da, &residual);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
          if (dim() == 2) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::BratuFunctions::Rhs_Structured_2d_FD, sol.vec, residual, &this->_options);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::BratuFunctions::Rhs_Structured_3d_FD, sol.vec, residual, &this->_options);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = VecNorm(residual, NORM_2, &norm);CHKERRQ(ierr);
          if (flag) {ierr = VecView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = DARestoreGlobalVector(da, &residual);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.vec, &name);CHKERRQ(ierr);
        } else {
          ::Mesh      mesh = (::Mesh) this->_dm;
          SectionReal residual;

          ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
          ierr = ALE::Problem::BratuFunctions::Rhs_Unstructured(mesh, sol.section, residual, &this->_options);CHKERRQ(ierr);
          if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
          ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
        }
        PetscPrintf(comm(), "Residual for trial solution %s: %g\n", name, norm);
        PetscFunctionReturn(0);
      };
    };
  }
}

#endif
