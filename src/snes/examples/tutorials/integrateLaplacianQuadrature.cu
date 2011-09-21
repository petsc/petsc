#include <petscsys.h>
#include <assert.h>

__device__ float2 laplacian(float u, float2 gradU) {
  return gradU;
}

// Number of spatial dimensions:        2
// N_b    Number of basis functions:    3
// N_q    Number of quadrature points:  1
// N_{bs} Number of block cells         LCM(N_b, N_q)  = 3
// N_{bl} Number of blocks              1
// N_t    Number of threads:            N_{bl} * N_{bs} = 3
// N_{cbc} Number of concurrent basis      cells:   N_{bl} * N_q = 1
// N_{cqc} Number of concurrent quadrature cells:   N_{bl} * N_b = 3
// N_{sbc} Number of serial     basis      cells:   N_{bs} / N_q = 3
// N_{sqc} Number of serial     quadrature cells:   N_{bs} / N_b = 1
// N_{cb} Number of cell batches:
// N_c    Number of total cells:        N_{cb}*N_{t} = 3
__global__ void integrateLaplacianQuadrature(int N_cb, float *coefficients, float *jacobianInverses, float *jacobianDeterminants, float *elemVec) {
  #include "ex52_inline.h"
  const int        dim  = 2;
  const int        N_b  = numBasisFunctions_0;   // The number of basis functions
  const int        N_q  = numQuadraturePoints_0; // The number of quadrature points
  const int        N_bs = N_b*N_q;               // The block size, LCM(N_b, N_q), Notice that a block is not process simultaneously
  const int        N_bl = 1;                     // The number of concurrent blocks
  const int        N_t  = N_bs*N_bl;             // The number of threads, N_bs * N_bl
  const int        N_c     = N_cb * N_t;
  const int        N_sbc   = N_bs / N_q;
  const int        N_sqc   = N_bs / N_b;
  /* Calculated indices */
  const int        tidx    = threadIdx.x + blockDim.x*threadIdx.y;
  const int        bidx    = tidx % N_b; // Basis function mapped to this thread
  const int        qidx    = tidx % N_q; // Quadrature point mapped to this thread
  const int        blbidx  = tidx % (N_bl * N_q); // Cell mapped to this thread in the basis phase
  const int        blqidx  = tidx % (N_bl * N_b); // Cell mapped to this thread in the quadrature phase
  const int        gidx    = blockIdx.y*gridDim.x + blockIdx.x;
  const int        Goffset = gidx*N_c;
  const int        Coffset = gidx*N_c*N_b;
  const int        Eoffset = gidx*N_c*N_b;
  /* Quadrature data */
  float             w;                 // $w_q$, Quadrature weight at $x_q$
//  __shared__ float  phi_i[N_b*N_q];    // $\phi_i(x_q)$, Value of the basis function $i$ at $x_q$
  __shared__ float2 phiDer_i[N_b*N_q]; // $\frac{\partial\phi_i(x_q)}{\partial x_d}$, Value of the derivative of basis function $i$ in direction $x_d$ at $x_q$
  /* Geometric data */
  __shared__ float  detJ[N_t];         // $|J(x_q)|$, Jacobian determinant at $x_q$
  __shared__ float  invJ[N_t*dim*dim]; // $J^{-1}(x_q)$, Jacobian inverse at $x_q$
  /* FEM data */
  __shared__ float  u_i[N_t*N_b];      // Coefficients $u_i$ of the field $u|_{\mathcal{T}} = \sum_i u_i \phi_i$
  /* Intermediate calculations */
// __shared__ float  f_0[N_t*N_sqc];    // $f_0(u(x_q), \nabla u(x_q)) |J(x_q)| w_q$
  __shared__ float2 f_1[N_t*N_sqc];    // $f_1(u(x_q), \nabla u(x_q)) |J(x_q)| w_q$
  /* Output data */
  float             e_i;               // Coefficient $e_i$ of the residual

  /* These should be generated inline */
  /* Load quadrature weights */
  w = weights_0[qidx];
  /* Load basis tabulation \phi_i for this cell */
  if (tidx < N_b*N_q) {
 // phi_i[tidx]    = Basis_0[tidx];
    phiDer_i[tidx] = BasisDerivatives_0[tidx];
  }

  for(int batch = 0; batch < N_cb; ++batch) {
    /* Load geometry */
    detJ[tidx] = jacobianDeterminants[Goffset+batch*N_t+tidx];
    for(int n = 0; n < dim*dim; ++n) {
      const int offset = n*N_t;
      invJ[offset+tidx] = jacobianInverses[(Goffset+batch*N_t)*dim*dim+offset+tidx];
    }
    /* Load coefficients u_i for this cell */
    for(int n = 0; n < N_b; ++n) {
      const int offset = n*N_t;
      u_i[offset+tidx] = coefficients[Coffset+batch*N_t*N_b+offset+tidx];
    }

    /* Map coefficients to values at quadrature points */
    for(int c = 0; c < N_sqc; ++c) {
      float     u     = 0.0;        // $u(x_q)$, Value of the field at $x_q$
      float2    gradU = {0.0, 0.0}; // $\nabla u(x_q)$, Value of the field gradient at $x_q$
   // float2    x     = {0.0, 0.0}; // Quadrature point $x_q$
      const int fidx  = c*N_t + tidx;
      const int cell  = c*N_bl*N_b + blqidx;

      /* Get field and derivatives at this quadrature point */
      for(int i = 0; i < N_b; ++i) {
        float2 realSpaceDer;

     // u += u_i[cell*N_b+i]*phi_i[qidx*N_b+i];
        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[qidx*N_b+i].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[qidx*N_b+i].y;
        gradU.x += u_i[cell*N_b+i]*realSpaceDer.x;
        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[qidx*N_b+i].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[qidx*N_b+i].y;
        gradU.y += u_i[cell*N_b+i]*realSpaceDer.y;
      }
      /* Process values at quadrature points */
      //f_0[fidx] = identity(u, gradU)*detJ[cell]*w;
      f_1[fidx] = laplacian(u, gradU);
      f_1[fidx].x *= detJ[cell]*w;
      f_1[fidx].y *= detJ[cell]*w;
    }

    /* ==== TRANSPOSE THREADS ==== */

    /* Map values at quadrature points to coefficients */
    for(int c = 0; c < N_sbc; ++c) {
      const int cell = c*N_bl*N_q + blbidx;

      e_i = 0.0;
      for(int q = 0; q < N_q; ++q) {
        float2 realSpaceDer;

     // e_i += phi_i[q*N_b+bidx]*f_0[cell*N_q+q];
        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[q*N_b+bidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[q*N_b+bidx].y;
        e_i += realSpaceDer.x*f_1[cell*N_q+q].x;
        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[q*N_b+bidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[q*N_b+bidx].y;
        e_i += realSpaceDer.y*f_1[cell*N_q+q].y;
      }
#if 0
      // Check f_1
      //if (bidx < N_q) {
      //  e_i = f_1[cell*N_q+bidx].x;
      //} else {
      //  e_i = 0.0;
      //}
      // Check that u_i is being used correctly
      e_i = u_i[cell*N_b+bidx];
      //e_i = coefficients[Coffset+(batch*N_sbc+c)*N_t+tidx];
      //e_i = Coffset+(batch*N_sbc+c)*N_t+tidx;
#endif
      /* Write element vector for N_{cbc} cells at a time */
      elemVec[Eoffset+(batch*N_sbc+c)*N_t+tidx] = e_i;
    }
    /* ==== Could do one write per batch ==== */
  }
  return;
}

__global__ void integrateLaplacianJacobianQuadrature() {
  /* Map coefficients to values at quadrature points */
  /* Process values at quadrature points */
  /* Map values at quadrature points to coefficients */
  return;
}

// Calculate a conforming thread grid for N kernels
#undef __FUNCT__
#define __FUNCT__ "calculateGrid"
PetscErrorCode calculateGrid(const int N, const int blockSize, unsigned int& x, unsigned int& y, unsigned int& z)
{
  PetscFunctionBegin;
  z = 1;
  if (N % blockSize) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid block size %d for %d elements", blockSize, N);}
  const int Nblocks = N/blockSize;
  for(x = (int) (sqrt(Nblocks) + 0.5); x > 0; --x) {
    y = Nblocks/x;
    if (x*y == Nblocks) break;
  }
  if (x*y != Nblocks) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Could not find partition for %d with block size %d", N, blockSize);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateElementBatchGPU"
PetscErrorCode IntegrateElementBatchGPU(PetscInt Ne, PetscInt Ncb, PetscInt Nbc, const PetscScalar coefficients[],
                                        const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[], PetscInt debug) {
  #include "ex52_inline.h"
  const int dim  = 2;
  const int N_b  = numBasisFunctions_0;   // The number of basis functions
  const int N_q  = numQuadraturePoints_0; // The number of quadrature points
  const int N_bs = N_b*N_q;               // The block size, LCM(N_b, N_q), Notice that a block is not process simultaneously
  const int N_bl = 1;                     // The number of concurrent blocks
  const int N_t  = N_bs*N_bl;             // The number of threads, N_bs * N_bl

  float *d_coefficients;
  float *d_jacobianInverses;
  float *d_jacobianDeterminants;
  float *d_elemVec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  assert(Nbc == N_t);
  // Marshalling
  ierr = cudaMalloc((void**) &d_coefficients,         Ne*N_b * sizeof(float));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_jacobianInverses,     Ne*dim*dim * sizeof(float));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_jacobianDeterminants, Ne * sizeof(float));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_elemVec,              Ne*N_b * sizeof(float));CHKERRQ(ierr);
  ierr = cudaMemcpy(d_coefficients,         coefficients,         Ne*N_b * sizeof(float), cudaMemcpyHostToDevice);CHKERRQ(ierr);
  ierr = cudaMemcpy(d_jacobianInverses,     jacobianInverses,     Ne*dim*dim * sizeof(float), cudaMemcpyHostToDevice);CHKERRQ(ierr);
  ierr = cudaMemcpy(d_jacobianDeterminants, jacobianDeterminants, Ne * sizeof(float), cudaMemcpyHostToDevice);CHKERRQ(ierr);
  // Kernel launch
  //   This does not consider N_bl yet
  unsigned int x, y, z;
  ierr = calculateGrid(Ne, Ncb*Nbc, x, y, z);CHKERRQ(ierr);
  dim3 grid(x, y, z);
  dim3 block(Nbc, 1, 1);

  if (debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "GPU layout grid(%d,%d,%d) block(%d,%d,%d) with %d batches\n",
                       grid.x, grid.y, grid.z, block.x, block.y, block.z, Ncb);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " N_t: %d, N_cb: %d\n", N_t, Ncb);
  }
  integrateLaplacianQuadrature<<<grid, block>>>(Ncb, d_coefficients, d_jacobianInverses, d_jacobianDeterminants, d_elemVec);
  // Marshalling
  ierr = cudaMemcpy(elemVec, d_elemVec, Ne*N_b * sizeof(float), cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr = cudaFree(d_coefficients);CHKERRQ(ierr);
  ierr = cudaFree(d_jacobianInverses);CHKERRQ(ierr);
  ierr = cudaFree(d_jacobianDeterminants);CHKERRQ(ierr);
  ierr = cudaFree(d_elemVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
