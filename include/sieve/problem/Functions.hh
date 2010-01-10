#ifndef included_ALE_Problem_Functions_hh
#define included_ALE_Problem_Functions_hh

#include <sieve/problem/Base.hh>

namespace ALE {
  namespace Problem {
    namespace Functions {
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

      PetscScalar linear_2d_bem(const double x[]) {
        return x[0] + x[1];
      };

      // \frac{\partial u}{\partial n}
      PetscScalar linear_nder_2d(const double x[]) {
        if (x[0] + x[1] < 1.0) {
          // Bottom/Left
          return -1.0;
        } else {
        // Right/Top
          return 1.0;
        }
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
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
        ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
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
        const int                                closureSize   = m->sizeWithBC(s, *cells->begin()); // Should do a max of some sort
        double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
        PetscScalar *elemVec, *elemMat;
        PetscScalar *x;

        ierr = SectionRealZero(section);CHKERRQ(ierr);
        ierr = PetscMalloc3(numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat,closureSize,PetscScalar,&x);CHKERRQ(ierr);
        ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          if (detJ <= 0.0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);

          ierr = SectionRealRestrictClosure(X, mesh, *c_iter, closureSize, x);CHKERRQ(ierr);
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
          ierr = SectionRealUpdateClosure(section, mesh, *c_iter, elemVec, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = PetscFree3(elemVec,elemMat,x);CHKERRQ(ierr);
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
        if (m->debug()) {s->view("RHS");}
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
        const int                                closureSize   = m->sizeWithBC(s, *cells->begin()); // Should do a max of some sort
        double      *t_der, *b_der, *v0, *J, *invJ, detJ;
        PetscScalar *u;
        PetscScalar *elemMat;

        ierr = PetscMalloc2(numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat,closureSize,PetscScalar,&u);CHKERRQ(ierr);
        ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
          m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);

          ierr = SectionRealRestrictClosure(section, mesh, *c_iter, closureSize, u);CHKERRQ(ierr);
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
        ierr = PetscFree2(elemMat,u);CHKERRQ(ierr);
        ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        //ierr = MatView(A, PETSC_VIEWER_STDOUT_SELF);
        PetscFunctionReturn(0);
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
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "RhsBd_Unstructured"
      PetscErrorCode RhsBd_Unstructured(::Mesh mesh, SectionReal X, SectionReal section, void *ctx) {
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

                iMat[f*numBasisFuncs+g] += C*identity*quadWeights[qx]*detJx;
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
      #define __FUNCT__ "JacBd_Unstructured"
      PetscErrorCode JacBd_Unstructured(::Mesh mesh, SectionReal section, Mat M, void *ctx) {
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
    }
  }
}

#endif
