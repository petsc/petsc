#ifndef included_ALE_Problem_LaplaceBEM_hh
#define included_ALE_Problem_LaplaceBEM_hh

#include <DMBuilder.hh>

#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

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
      PetscScalar (*exactDirichletFunc)(const double []); // The exact solution function for Dirichlet data
      PetscScalar (*exactNeumannFunc)(const double []);   // The exact solution function for Neumann data
      ExactSolType  exactSol;                    // The discrete exact solution
      ExactSolType  error;                       // The discrete cell-wise error
      AssemblyType  operatorAssembly;            // The type of operator assembly 
      double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
      double        lambda;                      // The parameter controlling nonlinearity
      double        reentrant_angle;             // The angle for the reentrant corner.
      PetscScalar   phiCoefficient;              // Coefficient C for phi = {0 in interior, 0.5 on smooth boundary}
    } LaplaceBEMOptions;
    namespace LaplaceBEMFunctions {
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

      // \frac{\partial u}{\partial n}
      PetscScalar quadratic_nder_2d(const double x[]) {
        if (x[0] + x[1] < 1.0) {
          // Bottom/Left
          return 0.0;
        } else {
        // Right/Top
          return 2.0;
        }
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
      #define __FUNCT__ "PointEvaluation"
      PetscErrorCode PointEvaluation(::Mesh mesh, SectionReal X, double coordsx[], double detJx, PetscScalar elemVec[]) {
        Obj<PETSC_MESH_TYPE> m;
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        SectionReal    boundaryData, normals;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
        ierr = MeshGetSectionReal(mesh, "boundaryData", &boundaryData);CHKERRQ(ierr);
        ierr = MeshGetSectionReal(mesh, "normals", &normals);CHKERRQ(ierr);
        ierr = SectionRealGetSection(X, s);CHKERRQ(ierr);

        const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = m->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = m->heightStratum(0);
        const PETSC_MESH_TYPE::label_sequence::iterator cEnd        = cells->end();
        const int                                embedDim           = m->getDimension()+1;
        const int                                closureSize        = m->sizeWithBC(s, *cells->begin()); // Should do a max of some sort
        const int                                numBasisFuncs      = 1;
        const double                             quadWeights[1]     = {1.0};
        const int                                qx                 = 0;
        const int                                matSize            = numBasisFuncs*numBasisFuncs;
        double         coordsy[2], v0y[2], Jy[4], invJy[4], detJy;
        PetscScalar    fMat[1], gMat[1], normal[2], x[1], bdData[1];

        for(PETSC_MESH_TYPE::label_sequence::iterator d_iter = cells->begin(); d_iter != cEnd; ++d_iter) {
          m->computeBdElementGeometry(coordinates, *d_iter, v0y, Jy, invJy, detJy);
          if (detJy <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJy, *d_iter);
          detJy = sqrt(detJy);
          ierr = PetscMemzero(fMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(gMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);

          ierr = SectionRealRestrictClosure(X, mesh, *d_iter, closureSize, x);CHKERRQ(ierr);
          ierr = SectionRealRestrictClosure(boundaryData, mesh, *d_iter, closureSize, bdData);CHKERRQ(ierr);
          ierr = SectionRealRestrictClosure(normals, mesh, *d_iter, embedDim, normal);CHKERRQ(ierr);

          for(int d = 0; d < embedDim; d++) {
            coordsy[d] = v0y[d];
          }
          // I was warned about screwy integration
          PetscScalar L = 2.0*detJy;
          PetscScalar A = PetscSqr(L);
          PetscScalar B = 2.0*L*(-normal[1]*(coordsy[0] - coordsx[0]) + normal[0]*(coordsy[1] - coordsx[1]));
          PetscScalar E = PetscSqr(coordsy[0] - coordsx[0]) + PetscSqr(coordsy[1] - coordsx[1]);
          PetscScalar D = sqrt(fabs(4.0*A*E - PetscSqr(B)));
          PetscScalar BA = B/A;
          PetscScalar EA = E/A;

          if (D < 1.0e-10) {
            gMat[0] -= (L/(2.0*M_PI))*(log(L) + (1.0 + 0.5*BA)*log(fabs(1.0 + 0.5*BA)) - 0.5*BA*log(fabs(0.5*BA)) - 1.0)*quadWeights[qx]*detJx;
            fMat[0]  = 0.0;
          } else {
            gMat[0] -= (L/(4.0*M_PI))*(2.0*(log(L) - 1.0) - 0.5*BA*log(fabs(EA)) + (1.0 + 0.5*BA)*log(fabs(1.0 + BA + EA))
                                       + (D/A)*(atan((2.0*A + B)/D) - atan(B/D)))*quadWeights[qx]*detJx;
            fMat[0] -= (L*(normal[0]*(coordsy[0] - coordsx[0]) + normal[1]*(coordsy[1] - coordsx[1]))/D * (atan((2.0*A + B)/D) - atan(B/D)))/M_PI*quadWeights[qx]*detJx;
          }
#if 1
          PetscPrintf(PETSC_COMM_WORLD, "Cell: %d\n", *d_iter);
          PetscPrintf(PETSC_COMM_WORLD, "x: ");
          for(int f = 0; f < numBasisFuncs; ++f) {
            PetscPrintf(PETSC_COMM_WORLD, "(%d) %g ", f, x[f]);
          }
          PetscPrintf(PETSC_COMM_WORLD, "\n");
          PetscPrintf(PETSC_COMM_WORLD, "bdData: ");
          for(int f = 0; f < numBasisFuncs; ++f) {
            PetscPrintf(PETSC_COMM_WORLD, "(%d) %g ", f, bdData[f]);
          }
          PetscPrintf(PETSC_COMM_WORLD, "\n");
          PetscPrintf(PETSC_COMM_WORLD, "fMat: ");
          for(int f = 0; f < numBasisFuncs; ++f) {
            for(int g = 0; g < numBasisFuncs; ++g) {
              PetscPrintf(PETSC_COMM_WORLD, "(%d,%d) %g ", f, g, fMat[f*numBasisFuncs+g]);
            }
          }
          PetscPrintf(PETSC_COMM_WORLD, "\n");
          PetscPrintf(PETSC_COMM_WORLD, "gMat: ");
          for(int f = 0; f < numBasisFuncs; ++f) {
            for(int g = 0; g < numBasisFuncs; ++g) {
              PetscPrintf(PETSC_COMM_WORLD, "(%d,%d) %g ", f, g, gMat[f*numBasisFuncs+g]);
            }
          }
          PetscPrintf(PETSC_COMM_WORLD, "\n");
#endif
          // Add linear contribution
          for(int f = 0; f < numBasisFuncs; ++f) {
            for(int g = 0; g < numBasisFuncs; ++g) {
              elemVec[f] += fMat[f*numBasisFuncs+g]*bdData[g] - gMat[f*numBasisFuncs+g]*x[g];
            }
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "Rhs_Unstructured"
      PetscErrorCode Rhs_Unstructured(::Mesh mesh, SectionReal X, SectionReal section, void *ctx) {
        LaplaceBEMOptions  *options = (LaplaceBEMOptions *) ctx;
        PetscScalar         C = options->phiCoefficient;
        Obj<PETSC_MESH_TYPE> m;
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        SectionReal    boundaryData, normals;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
        ierr = MeshGetSectionReal(mesh, "boundaryData", &boundaryData);CHKERRQ(ierr);
        ierr = MeshGetSectionReal(mesh, "normals", &normals);CHKERRQ(ierr);
        ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
        const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
        const int                                numQuadPoints = disc->getQuadratureSize();
        const double                            *quadPoints    = disc->getQuadraturePoints();
        const double                            *quadWeights   = disc->getQuadratureWeights();
        const int                                numBasisFuncs = disc->getBasisSize();
        const double                            *basis         = disc->getBasis();
        const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = m->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = m->heightStratum(0);
        const PETSC_MESH_TYPE::label_sequence::iterator cEnd        = cells->end();
        const int                                dim           = m->getDimension();
        const int                                embedDim      = m->getDimension()+1;
        const int                                closureSize   = m->sizeWithBC(s, *cells->begin()); // Should do a max of some sort
        const int                                matSize       = numBasisFuncs*numBasisFuncs;
        double      *coordsx, *v0x, *Jx, *invJx, detJx, *coordsy, *v0y, *Jy, *invJy, detJy;
        PetscScalar *elemVec, *iMat, *fMat, *gMat;
        PetscScalar *x, *bdData, *normal;

        ierr = SectionRealZero(section);CHKERRQ(ierr);
        ierr = PetscMalloc7(numBasisFuncs,PetscScalar,&elemVec,matSize,PetscScalar,&iMat,matSize,PetscScalar,&fMat,matSize,PetscScalar,&gMat,closureSize,PetscScalar,&x,closureSize,PetscScalar,&bdData,embedDim,PetscScalar,&normal);CHKERRQ(ierr);
        ierr = PetscMalloc4(embedDim,double,&coordsx,embedDim,double,&v0x,embedDim*embedDim,double,&Jx,embedDim*embedDim,double,&invJx);CHKERRQ(ierr);
        ierr = PetscMalloc4(embedDim,double,&coordsy,embedDim,double,&v0y,embedDim*embedDim,double,&Jy,embedDim*embedDim,double,&invJy);CHKERRQ(ierr);
        // Loop over cells
        CHKMEMQ;
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
          m->computeBdElementGeometry(coordinates, *c_iter, v0x, Jx, invJx, detJx);
          if (detJx <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJx, *c_iter);
          detJx = sqrt(detJx);
          ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(iMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);

          // Loop over x quadrature points
          for(int qx = 0; qx < numQuadPoints; ++qx) {
            // Loop over trial functions
            for(int f = 0; f < numBasisFuncs; ++f) {
              // Loop over basis functions
              for(int g = 0; g < numBasisFuncs; ++g) {
                PetscScalar identity = basis[qx*numBasisFuncs+f]*basis[qx*numBasisFuncs+g];

                iMat[f*numBasisFuncs+g] += 0.5*identity*quadWeights[qx]*detJx;
              }
            }
          }

          ierr = SectionRealRestrictClosure(boundaryData, mesh, *c_iter, closureSize, bdData);CHKERRQ(ierr);

          for(int f = 0; f < numBasisFuncs; ++f) {
            for(int g = 0; g < numBasisFuncs; ++g) {
              elemVec[f] += iMat[f*numBasisFuncs+g]*bdData[g];
            }
            //PetscPrintf(PETSC_COMM_WORLD, "Initial 1/2 phi[%d] %g\n", f, elemVec[f]);
          }

          // Loop over x quadrature points
          for(int qx = 0; qx < numQuadPoints; ++qx) {
            for(int d = 0; d < embedDim; d++) {
              coordsx[d] = v0x[d];
              for(int e = 0; e < dim; e++) {
                coordsx[d] += Jx[d*embedDim+e]*(quadPoints[qx*dim+e] + 1.0);
              }
            }

            for(PETSC_MESH_TYPE::label_sequence::iterator d_iter = cells->begin(); d_iter != cEnd; ++d_iter) {
              m->computeBdElementGeometry(coordinates, *d_iter, v0y, Jy, invJy, detJy);
              if (detJy <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJy, *d_iter);
              detJy = sqrt(detJy);
              ierr = PetscMemzero(fMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);
              ierr = PetscMemzero(gMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);

              ierr = SectionRealRestrictClosure(X, mesh, *d_iter, closureSize, x);CHKERRQ(ierr);
              ierr = SectionRealRestrictClosure(boundaryData, mesh, *d_iter, closureSize, bdData);CHKERRQ(ierr);
              ierr = SectionRealRestrictClosure(normals, mesh, *d_iter, embedDim, normal);CHKERRQ(ierr);

#if 1
              for(int d = 0; d < embedDim; d++) {
                coordsy[d] = v0y[d];
              }
              // I was warned about screwy integration
              PetscScalar L = 2.0*detJy;
              PetscScalar A = PetscSqr(L);
              PetscScalar B = 2.0*L*(-normal[1]*(coordsy[0] - coordsx[0]) + normal[0]*(coordsy[1] - coordsx[1]));
              PetscScalar E = PetscSqr(coordsy[0] - coordsx[0]) + PetscSqr(coordsy[1] - coordsx[1]);
              PetscScalar D = sqrt(fabs(4.0*A*E - PetscSqr(B)));
              PetscScalar BA = B/A;
              PetscScalar EA = E/A;

              if (D < 1.0e-10) {
                gMat[0] -= (L/(2.0*M_PI))*(log(L) + (1.0 + 0.5*BA)*log(fabs(1.0 + 0.5*BA)) - 0.5*BA*log(fabs(0.5*BA)) - 1.0)*quadWeights[qx]*detJx;
                fMat[0]  = 0.0;
              } else {
                gMat[0] -= (L/(4.0*M_PI))*(2.0*(log(L) - 1.0) - 0.5*BA*log(fabs(EA)) + (1.0 + 0.5*BA)*log(fabs(1.0 + BA + EA))
                                         + (D/A)*(atan((2.0*A + B)/D) - atan(B/D)))*quadWeights[qx]*detJx;
                fMat[0] -= (L*(normal[0]*(coordsy[0] - coordsx[0]) + normal[1]*(coordsy[1] - coordsx[1]))/D * (atan((2.0*A + B)/D) - atan(B/D)))/M_PI*quadWeights[qx]*detJx;
              }
#else
              // Loop over y quadrature points
              for(int qy = 0; qy < numQuadPoints; ++qy) {
                for(int d = 0; d < embedDim; d++) {
                  coordsy[d] = v0y[d];
                  for(int e = 0; e < dim; e++) {
                    coordsy[d] += Jy[d*embedDim+e]*(quadPoints[qy*dim+e] + 1.0);
                  }
                }
                CHKMEMQ;
                //PetscPrintf(PETSC_COMM_WORLD, "coordsx %g %g coordsy %g %g\n", coordsx[0], coordsx[1], coordsy[0], coordsy[1]);

                PetscScalar r = 0.0;
                for(int d = 0; d < embedDim; d++) {
                  r += PetscSqr(coordsx[d] - coordsy[d]);
                }
                r = sqrt(r);

                PetscScalar nDotR = 0.0;
                for(int d = 0; d < embedDim; d++) {
                  nDotR += normal[d]*(coordsy[d] - coordsx[d])/r;
                }
                CHKMEMQ;
                PetscPrintf(PETSC_COMM_WORLD, "r %g dr/dn %g\n", r, nDotR);

                // Loop over trial functions
                for(int f = 0; f < numBasisFuncs; ++f) {
                  // Loop over basis functions
                  for(int g = 0; g < numBasisFuncs; ++g) {
                    PetscScalar identity = basis[qx*numBasisFuncs+f]*basis[qy*numBasisFuncs+g];

                    //PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d %g %g\n", qx, qy, f, g, identity, r);
                    fMat[f*numBasisFuncs+g] += identity*(nDotR/(2.0*M_PI*r))*quadWeights[qx]*detJx*quadWeights[qy]*detJy;
                    gMat[f*numBasisFuncs+g] += identity*(log(1.0/r)/(2.0*M_PI))*quadWeights[qx]*detJx*quadWeights[qy]*detJy;
                  }
                }
              }
#endif
#if 0
            PetscPrintf(PETSC_COMM_WORLD, "Cell pair: %d and %d\n", *c_iter, *d_iter);
            PetscPrintf(PETSC_COMM_WORLD, "x: ");
            for(int f = 0; f < numBasisFuncs; ++f) {
              PetscPrintf(PETSC_COMM_WORLD, "(%d) %g ", f, x[f]);
            }
            PetscPrintf(PETSC_COMM_WORLD, "\n");
            PetscPrintf(PETSC_COMM_WORLD, "bdData: ");
            for(int f = 0; f < numBasisFuncs; ++f) {
              PetscPrintf(PETSC_COMM_WORLD, "(%d) %g ", f, bdData[f]);
            }
            PetscPrintf(PETSC_COMM_WORLD, "\n");
            PetscPrintf(PETSC_COMM_WORLD, "iMat: ");
            for(int f = 0; f < numBasisFuncs; ++f) {
              for(int g = 0; g < numBasisFuncs; ++g) {
                PetscPrintf(PETSC_COMM_WORLD, "(%d,%d) %g ", f, g, iMat[f*numBasisFuncs+g]);
              }
            }
            PetscPrintf(PETSC_COMM_WORLD, "\n");
            PetscPrintf(PETSC_COMM_WORLD, "fMat: ");
            for(int f = 0; f < numBasisFuncs; ++f) {
              for(int g = 0; g < numBasisFuncs; ++g) {
                PetscPrintf(PETSC_COMM_WORLD, "(%d,%d) %g ", f, g, fMat[f*numBasisFuncs+g]);
              }
            }
            PetscPrintf(PETSC_COMM_WORLD, "\n");
            PetscPrintf(PETSC_COMM_WORLD, "gMat: ");
            for(int f = 0; f < numBasisFuncs; ++f) {
              for(int g = 0; g < numBasisFuncs; ++g) {
                PetscPrintf(PETSC_COMM_WORLD, "(%d,%d) %g ", f, g, gMat[f*numBasisFuncs+g]);
              }
            }
            PetscPrintf(PETSC_COMM_WORLD, "\n");
#endif
            // Add linear contribution
            for(int f = 0; f < numBasisFuncs; ++f) {
              for(int g = 0; g < numBasisFuncs; ++g) {
                elemVec[f] += fMat[f*numBasisFuncs+g]*bdData[g] - gMat[f*numBasisFuncs+g]*x[g];
              }
              //PetscPrintf(PETSC_COMM_WORLD, "elemVec[%d] %g\n", f, elemVec[f]);
            }
          }
          }
          for(int f = 0; f < numBasisFuncs; ++f) {
            PetscPrintf(PETSC_COMM_WORLD, "Full residual[%d] %g\n", f, elemVec[f]);
          }
          ierr = SectionRealUpdateClosure(section, mesh, *c_iter, elemVec, ADD_VALUES);CHKERRQ(ierr);
        }
        CHKMEMQ;
        ierr = PetscFree7(elemVec,iMat,fMat,gMat,x,bdData,normal);CHKERRQ(ierr);
        ierr = PetscFree4(coordsx,v0x,Jx,invJx);CHKERRQ(ierr);
        ierr = PetscFree4(coordsy,v0y,Jy,invJy);CHKERRQ(ierr);
        // Exchange neighbors
        ierr = SectionRealComplete(section);CHKERRQ(ierr);
        // Subtract the constant
        if (m->hasRealSection("constant")) {
          const Obj<PETSC_MESH_TYPE::real_section_type>& constant = m->getRealSection("constant");
          Obj<PETSC_MESH_TYPE::real_section_type>        s;

          ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
          s->axpy(-1.0, constant);
        }
        if (m->debug()) {s->view("RHS");}
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "Jac_Unstructured"
      PetscErrorCode Jac_Unstructured(::Mesh mesh, SectionReal section, Mat M, void *ctx) {
        LaplaceBEMOptions  *options = (LaplaceBEMOptions *) ctx;
        Obj<PETSC_MESH_TYPE> m;
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        SectionReal    normals;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
        ierr = MatZeroEntries(M);CHKERRQ(ierr);
        ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
        ierr = MeshGetSectionReal(mesh, "normals", &normals);CHKERRQ(ierr);
        ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
        const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
        const int                                numQuadPoints = disc->getQuadratureSize();
        const double                            *quadPoints    = disc->getQuadraturePoints();
        const double                            *quadWeights   = disc->getQuadratureWeights();
        const int                                numBasisFuncs = disc->getBasisSize();
        const double                            *basis         = disc->getBasis();
        const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = m->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = m->heightStratum(0);
        const PETSC_MESH_TYPE::label_sequence::iterator cEnd        = cells->end();
        const Obj<PETSC_MESH_TYPE::order_type>&         order       = m->getFactory()->getGlobalOrder(m, "default", s);
        const int                                dim           = m->getDimension();
        const int                                embedDim      = m->getDimension()+1;
        const int                                matSize       = numBasisFuncs*numBasisFuncs;
        double      *coordsx, *v0x, *Jx, *invJx, detJx, *coordsy, *v0y, *Jy, *invJy, detJy;
        PetscScalar *normal;
        PetscScalar *elemMat;

        ierr = PetscMalloc2(matSize,PetscScalar,&elemMat,embedDim,PetscScalar,&normal);CHKERRQ(ierr);
        ierr = PetscMalloc4(embedDim,double,&coordsx,embedDim,double,&v0x,embedDim*embedDim,double,&Jx,embedDim*embedDim,double,&invJx);CHKERRQ(ierr);
        ierr = PetscMalloc4(embedDim,double,&coordsy,embedDim,double,&v0y,embedDim*embedDim,double,&Jy,embedDim*embedDim,double,&invJy);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
          ierr = PetscMemzero(elemMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);
          m->computeBdElementGeometry(coordinates, *c_iter, v0x, Jx, invJx, detJx);
          if (detJx <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJx, *c_iter);
          detJx = sqrt(detJx);

#if 0
          // Loop over x quadrature points
          for(int qx = 0; qx < numQuadPoints; ++qx) {
            // Loop over trial functions
            for(int f = 0; f < numBasisFuncs; ++f) {
              // Loop over basis functions
              for(int g = 0; g < numBasisFuncs; ++g) {
                PetscScalar identity = basis[qx*numBasisFuncs+f]*basis[qx*numBasisFuncs+g];

                elemMat[f*numBasisFuncs+g] += 0.5*identity*quadWeights[qx]*detJx;
              }
            }
          }
#endif
          ierr = updateOperator(M, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);

          // Loop over x quadrature points
          for(int qx = 0; qx < numQuadPoints; ++qx) {
            for(int d = 0; d < embedDim; d++) {
              coordsx[d] = v0x[d];
              for(int e = 0; e < dim; e++) {
                coordsx[d] += Jx[d*embedDim+e]*(quadPoints[qx*dim+e] + 1.0);
              }
            }

            for(PETSC_MESH_TYPE::label_sequence::iterator d_iter = cells->begin(); d_iter != cEnd; ++d_iter) {
              ierr = PetscMemzero(elemMat, matSize * sizeof(PetscScalar));CHKERRQ(ierr);
              m->computeBdElementGeometry(coordinates, *d_iter, v0y, Jy, invJy, detJy);
              if (detJy <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJy, *d_iter);
              detJy = sqrt(detJy);
              
              ierr = SectionRealRestrictClosure(normals, mesh, *d_iter, embedDim, normal);CHKERRQ(ierr);

#if 1
              for(int d = 0; d < embedDim; d++) {
                coordsy[d] = v0y[d];
              }
              // I was warned about screwy integration
              PetscScalar L = 2.0*detJy;
              PetscScalar A = PetscSqr(L);
              PetscScalar B = 2.0*L*(-normal[1]*(coordsy[0] - coordsx[0]) + normal[0]*(coordsy[1] - coordsx[1]));
              PetscScalar E = PetscSqr(coordsy[0] - coordsx[0]) + PetscSqr(coordsy[1] - coordsx[1]);
              PetscScalar D = sqrt(fabs(4.0*A*E - PetscSqr(B)));
              PetscScalar BA = B/A;
              PetscScalar EA = E/A;

              if (D < 1.0e-10) {
                elemMat[0] += (L/(2.0*M_PI))*(log(L) + (1.0 + 0.5*BA)*log(fabs(1.0 + 0.5*BA)) - 0.5*BA*log(fabs(0.5*BA)) - 1.0)*quadWeights[qx]*detJx;
              } else {
                elemMat[0] += (L/(4.0*M_PI))*(2.0*(log(L) - 1.0) - 0.5*BA*log(fabs(EA)) + (1.0 + 0.5*BA)*log(fabs(1.0 + BA + EA))
                                         + (D/A)*(atan((2.0*A + B)/D) - atan(B/D)))*quadWeights[qx]*detJx;
              }
#else
              // Loop over y quadrature points
              for(int qy = 0; qy < numQuadPoints; ++qy) {
                for(int d = 0; d < embedDim; d++) {
                  coordsy[d] = v0y[d];
                  for(int e = 0; e < dim; e++) {
                    coordsy[d] += Jy[d*embedDim+e]*(quadPoints[qy*dim+e] + 1.0);
                  }
                }

                PetscScalar r = 0.0;
                for(int d = 0; d < embedDim; d++) {
                  r += PetscSqr(coordsx[d] - coordsy[d]);
                }
                r = sqrt(r);

                PetscScalar nDotR = 0.0;
                for(int d = 0; d < embedDim; d++) {
                  nDotR += normal[d]*(coordsy[d] - coordsx[d])/r;
                }

                // Loop over trial functions
                for(int f = 0; f < numBasisFuncs; ++f) {
                  // Loop over basis functions
                  for(int g = 0; g < numBasisFuncs; ++g) {
                    PetscScalar identity = basis[qx*numBasisFuncs+f]*basis[qy*numBasisFuncs+g];

                    //PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d %g %g\n", qx, qy, f, g, identity, r);
                    //fMat[f*numBasisFuncs+g] += identity*(nDotR/(2.0*M_PI*r))*quadWeights[qx]*detJx*quadWeights[qy]*detJy;
                    elemMat[f*numBasisFuncs+g] -= identity*(log(1.0/r)/(2.0*M_PI))*quadWeights[qx]*detJx*quadWeights[qy]*detJy;
                  }
                }
              }
#endif
              ierr = updateOperatorGeneral(M, m, s, order, *c_iter, m, s, order, *d_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }
        ierr = PetscFree2(elemMat,normal);CHKERRQ(ierr);
        ierr = PetscFree4(coordsx,v0x,Jx,invJx);CHKERRQ(ierr);
        ierr = PetscFree4(coordsy,v0y,Jy,invJy);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatView(M, PETSC_VIEWER_STDOUT_SELF);
        PetscFunctionReturn(0);
      };
    };
    class LaplaceBEM : public ALE::ParallelObject {
    public:
    protected:
      LaplaceBEMOptions         _options;
      DM                   _dm;
      Obj<PETSC_MESH_TYPE> _mesh;
      DMMG                *_dmmg;
    public:
      LaplaceBEM(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
        PetscErrorCode ierr = this->processOptions(comm, &this->_options);CHKERRXX(ierr);
        this->_dm   = PETSC_NULL;
        this->_dmmg = PETSC_NULL;
        this->_options.exactSol.vec = PETSC_NULL;
        this->_options.error.vec    = PETSC_NULL;
      };
      ~LaplaceBEM() {
        PetscErrorCode ierr;

        if (this->_dmmg)                 {ierr = DMMGDestroy(this->_dmmg);CHKERRXX(ierr);}
        if (this->_options.exactSol.vec) {ierr = this->destroyExactSolution(this->_options.exactSol);CHKERRXX(ierr);}
        if (this->_options.error.vec)    {ierr = this->destroyExactSolution(this->_options.error);CHKERRXX(ierr);}
        if (this->_dm)                   {ierr = this->destroyMesh();CHKERRXX(ierr);}
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "LaplaceBEMProcessOptions"
      PetscErrorCode processOptions(MPI_Comm comm, LaplaceBEMOptions *options) {
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
        options->phiCoefficient   = 0.5;

        ierr = PetscOptionsBegin(comm, "", "LaplaceBEM Problem Options", "DMMG");CHKERRQ(ierr);
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

        ALE::Problem::LaplaceBEMFunctions::lambda = options->lambda;
        this->setDebug(options->debug);
        PetscFunctionReturn(0);
      };
    public: // Accessors
      LaplaceBEMOptions *getOptions() {return &this->_options;};
      int  dim() const {return this->_options.dim;};
      bool structured() const {return this->_options.structured;};
      void structured(const bool s) {this->_options.structured = (PetscTruth) s;};
      bool interpolated() const {return this->_options.interpolate;};
      void interpolated(const bool i) {this->_options.interpolate = (PetscTruth) i;};
      BCType bcType() const {return this->_options.bcType;};
      void bcType(const BCType bc) {this->_options.bcType = bc;};
      AssemblyType opAssembly() const {return this->_options.operatorAssembly;};
      void opAssembly(const AssemblyType at) {this->_options.operatorAssembly = at;};
      PETSC_MESH_TYPE *getMesh() {return this->_mesh;};
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
            //ierr = ALE::DMBuilder::createBasketMesh(comm(), dim(), structured(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
            ierr = ALE::DMBuilder::createBasketMesh(comm(), dim(), structured(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
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
              this->_options.func               = ALE::Problem::LaplaceBEMFunctions::nonlinear_2d;
              this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::quadratic_2d;
            } else if (this->_options.reentrantMesh) { 
              this->_options.func               = ALE::Problem::LaplaceBEMFunctions::singularity_2d;
              this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::singularity_exact_2d;
            } else {
              this->_options.func               = ALE::Problem::LaplaceBEMFunctions::constant;
              this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::quadratic_2d;
              this->_options.exactNeumannFunc   = ALE::Problem::LaplaceBEMFunctions::quadratic_nder_2d;
            }
          } else {
            this->_options.func               = ALE::Problem::LaplaceBEMFunctions::linear_2d;
            this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::cubic_2d;
          }
        } else if (dim() == 3) {
          if (bcType() == DIRICHLET) {
            if (this->_options.reentrantMesh) {
              this->_options.func               = ALE::Problem::LaplaceBEMFunctions::singularity_3d;
              this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::singularity_exact_3d;
            } else {
              if (this->_options.lambda > 0.0) {
                this->_options.func             = ALE::Problem::LaplaceBEMFunctions::nonlinear_3d;
              } else {
                this->_options.func             = ALE::Problem::LaplaceBEMFunctions::constant;
              }
              this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::quadratic_3d;
            }
          } else {
            this->_options.func               = ALE::Problem::LaplaceBEMFunctions::linear_3d;
            this->_options.exactDirichletFunc = ALE::Problem::LaplaceBEMFunctions::cubic_3d;
          }
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
        }
        if (!structured()) {
          // Should pass bcType()
          int            numBC      = 0;
          int            markers[1] = {1};
          double       (*funcs[1])(const double *coords) = {this->_options.exactDirichletFunc};
          PetscErrorCode ierr;

          if (dim() == 2) {
            ierr = CreateProblem_gen_0(this->_dm, "u", numBC, markers, funcs, this->_options.exactDirichletFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateBdDualBasis_gen_0;
          } else if (dim() == 3) {
            ierr = CreateProblem_gen_1(this->_dm, "u", numBC, markers, funcs, this->_options.exactDirichletFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_1;
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("default");
          s->setDebug(debug());
          this->_mesh->setupField(s, 2, true);
          if (debug()) {s->view("Default field");}
          const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& t = this->_mesh->getRealSection("boundaryData");
          t->setDebug(debug());
          this->_mesh->setupField(t, 2);
          typedef ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
          const Obj<PETSC_MESH_TYPE::label_type>&         cellExclusion = this->_mesh->getLabel("cellExclusion");
          const Obj<PETSC_MESH_TYPE::label_sequence>&     boundaryCells = this->_mesh->heightStratum(0);
          const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates   = this->_mesh->getRealSection("coordinates");
          const Obj<PETSC_MESH_TYPE::names_type>&         discs         = this->_mesh->getDiscretizations();
          const PETSC_MESH_TYPE::point_type               firstCell     = *boundaryCells->begin();
          const int                                       numFields     = discs->size();
          PETSC_MESH_TYPE::real_section_type::value_type *values        = new PETSC_MESH_TYPE::real_section_type::value_type[this->_mesh->sizeWithBC(t, firstCell)];
          int                                             embedDim      = dim();
          int                                            *v             = new int[numFields];
          double                                         *v0            = new double[embedDim];
          double                                         *J             = new double[embedDim*embedDim];
          double                                          detJ;
          Visitor pV((int) pow((double) this->_mesh->getSieve()->getMaxConeSize(), this->_mesh->depth())+1, true);

          for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
            ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), *c_iter, pV);
            const Visitor::point_type *oPoints = pV.getPoints();
            const int                  oSize   = pV.getSize();

            this->_mesh->computeBdElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
            for(int f = 0; f < numFields; ++f) v[f] = 0;
            for(int cl = 0; cl < oSize; ++cl) {
              int f = 0;

              for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>&        disc    = this->_mesh->getDiscretization(*f_iter);
                const Obj<PETSC_MESH_TYPE::names_type> bcs     = disc->getBoundaryConditions();
                const int                              fDim    = t->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
                const int                             *indices = disc->getIndices(this->_mesh->getValue(cellExclusion, *c_iter));

                //for(PETSC_MESH_TYPE::names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter) {
                  //const Obj<ALE::BoundaryCondition>& bc = disc->getBoundaryCondition(*bc_iter);

                  for(int d = 0; d < fDim; ++d, ++v[f]) {
                    //values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
                    values[indices[v[f]]] = IntegrateBdDualBasis_gen_0(v0, J, v[f], this->_options.exactDirichletFunc);
                  }
                //}
              }
            }
            this->_mesh->updateAll(t, *c_iter, values);
            pV.clear();
          }
          delete [] values;
          delete [] v;
          delete [] v0;
          delete [] J;

          if (debug()) {t->view("Boundary Data");}
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
          SETERRQ(PETSC_ERR_SUP, "Structured meshes not supported");
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

            this->_mesh->computeBdElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
            for(int cl = 0; cl < oSize; ++cl) {
              const int pointDim = s->getFiberDimension(oPoints[cl]);

              if (pointDim) {
                for(int d = 0; d < pointDim; ++d, ++v) {
                  values[v] = (*this->_options.integrate)(v0, J, v, this->_options.exactNeumannFunc);
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
          s->view("Exact Solution");
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
      // The BEM system to bbe solved is
      //
      //   (1/2 I + F) \phi = G \frac{\partial\phi}{\partial n}
      //
      // where
      //
      //   F_{ij} = \int_{S_j} \frac{\partial G(x_i, y)}{\partial n(y)} dy = \int_{S_j} \frac{1}{2\pi r} \frac{\partial r}{\partial n} dy
      //   G_{ij} = \int_{S_j} G(x_i, y) dy = \int_{S_j} \frac{1}{2\pi} \ln\frac{1}{r} dy
      PetscErrorCode createSolver() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGCreate(this->comm(), 1, &this->_options, &this->_dmmg);CHKERRQ(ierr);
        ierr = DMMGSetDM(this->_dmmg, this->_dm);CHKERRQ(ierr);
        if (structured()) {
          SETERRQ(PETSC_ERR_SUP, "Structured meshes not supported");
        } else {
          if (opAssembly() == ALE::Problem::ASSEMBLY_FULL) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::LaplaceBEMFunctions::Rhs_Unstructured, ALE::Problem::LaplaceBEMFunctions::Jac_Unstructured, 0, 0);CHKERRQ(ierr);
#if 0
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_CALCULATED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::LaplaceBEMFunctions::Rhs_Unstructured, ALE::Problem::LaplaceBEMFunctions::Jac_Unstructured_Calculated, 0, 0);CHKERRQ(ierr);
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_STORED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::LaplaceBEMFunctions::Rhs_Unstructured, ALE::Problem::LaplaceBEMFunctions::Jac_Unstructured_Stored, 0, 0);CHKERRQ(ierr);
#endif
          } else {
            SETERRQ1(PETSC_ERR_ARG_WRONG, "Assembly type not supported: %d", opAssembly());
          }
          ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
        }
#if 0
        if (bcType() == ALE::Problem::NEUMANN) {
          // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
          ierr = DMMGSetNullSpace(this->_dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
        }
#endif
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "LaplaceBEMSolve"
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
        if (debug()) {
          ierr = PetscPrintf(comm(), "Number of nonlinear iterations = %D\n", its);CHKERRQ(ierr);
          ierr = PetscPrintf(comm(), "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
        }
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
          if (debug()) {ierr = PetscPrintf(comm(), "Total error: %g\n", error);CHKERRQ(ierr);}
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
        Obj<PETSC_MESH_TYPE::real_section_type> u;
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        PetscScalar  (*func)(const double *) = this->_options.exactDirichletFunc;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = SectionRealGetSection(X, u);CHKERRQ(ierr);
        ierr = SectionRealGetSection(this->_options.error.section, s);CHKERRQ(ierr);
        const Obj<ALE::Discretization>&                disc          = this->_mesh->getDiscretization("u");
        const int                                      numQuadPoints = disc->getQuadratureSize();
        const double                                  *quadPoints    = disc->getQuadraturePoints();
        const double                                  *quadWeights   = disc->getQuadratureWeights();
        const int                                      numBasisFuncs = disc->getBasisSize();
        const double                                  *basis         = disc->getBasis();
        const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates   = this->_mesh->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = this->_mesh->heightStratum(0);
        const int                                      dim           = this->dim();
        const int                                      closureSize   = this->_mesh->sizeWithBC(u, *cells->begin()); // Should do a max of some sort
        double      *coords, *v0, *J, *invJ, detJ;
        PetscScalar *x;
        double       localError = 0.0;

        ierr = PetscMalloc(closureSize * sizeof(PetscScalar), &x);CHKERRQ(ierr);
        ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          double elemError = 0.0;

          this->_mesh->computeBdElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          if (debug()) {
            std::cout << "Element " << *c_iter << " v0: (" << v0[0]<<","<<v0[1]<<")" << "J " << J[0]<<","<<J[1]<<","<<J[2]<<","<<J[3] << " detJ " << detJ << std::endl;
          }
          this->_mesh->restrictClosure(u, *c_iter, x, closureSize);
          if (debug()) {
            for(int f = 0; f < numBasisFuncs; ++f) {
              std::cout << "x["<<f<<"] " << x[f] << std::endl;
            }
          }
          // Loop over quadrature points
          for(int q = 0; q < numQuadPoints; ++q) {
            for(int d = 0; d < dim; d++) {
              coords[d] = v0[d];
              for(int e = 0; e < dim-1; e++) {
                coords[d] += J[d*dim+e]*(quadPoints[q*(dim-1)+e] + 1.0);
              }
              if (debug()) {std::cout << "q: "<<q<<"  refCoord["<<d<<"] " << quadPoints[q*(dim-1)+d] << "  coords["<<d<<"] " << coords[d] << std::endl;}
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
          this->_mesh->updateAdd(s, *c_iter, &elemError);
          localError += elemError;
        }
        ierr = MPI_Allreduce(&localError, error, 1, MPI_DOUBLE, MPI_SUM, comm());CHKERRQ(ierr);
        ierr = PetscFree(x);CHKERRQ(ierr);
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
          SETERRQ(PETSC_ERR_SUP, "Structured meshes not supported");
        } else {
          ::Mesh      mesh = (::Mesh) this->_dm;
          SectionReal residual;

          ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
          ierr = ALE::Problem::LaplaceBEMFunctions::Rhs_Unstructured(mesh, sol.section, residual, &this->_options);CHKERRQ(ierr);
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
