#include <petscsys.h>
#include <assert.h>

#include "ex52_gpu.h"

#if (SPATIAL_DIM_0 == 2)

__device__ vecType f1_laplacian_coef(realType u[], vecType gradU[], realType kappa, int comp) {
  vecType l = {kappa*gradU[comp].x, kappa*gradU[comp].y};
  return l;
}

__device__ vecType f1_elasticity_coef(realType u[], vecType gradU[], realType kappa, int comp) {
  vecType f1;

  switch(comp) {
  case 0:
    f1.x = 0.5*(gradU[0].x + gradU[0].x);
    f1.y = 0.5*(gradU[0].y + gradU[1].x);
    break;
  case 1:
    f1.x = 0.5*(gradU[1].x + gradU[0].y);
    f1.y = 0.5*(gradU[1].y + gradU[1].y);
  }
  return f1;
}

#elif (SPATIAL_DIM_0 == 3)

__device__ vecType f1_laplacian_coef(realType u[], vecType gradU[], realType kappa, int comp) {
  vecType l = {kappa*gradU[comp].x, kappa*gradU[comp].y, kappa*gradU[comp].z};
  return l;
}

__device__ vecType f1_elasticity_coef(realType u[], vecType gradU[], int comp) {
  vecType f1;

  switch(comp) {
  case 0:
    f1.x = 0.5*(gradU[0].x + gradU[0].x);
    f1.y = 0.5*(gradU[0].y + gradU[1].x);
    f1.z = 0.5*(gradU[0].z + gradU[2].x);
    break;
  case 1:
    f1.x = 0.5*(gradU[1].x + gradU[0].y);
    f1.y = 0.5*(gradU[1].y + gradU[1].y);
    f1.z = 0.5*(gradU[1].z + gradU[2].y);
    break;
  case 2:
    f1.x = 0.5*(gradU[2].x + gradU[0].z);
    f1.y = 0.5*(gradU[2].y + gradU[1].z);
    f1.z = 0.5*(gradU[2].z + gradU[2].z);
  }
  return f1;
}

#else

#error "Invalid spatial dimension"

#endif

// dim     Number of spatial dimensions:          2
// N_b     Number of basis functions:             generated
// N_{bt}  Number of total basis functions:       N_b * N_{comp}
// N_q     Number of quadrature points:           generated
// N_{bs}  Number of block cells                  LCM(N_b, N_q)
// N_{bst} Number of block cell components        LCM(N_{bt}, N_q)
// N_{bl}  Number of concurrent blocks            generated
// N_t     Number of threads:                     N_{bl} * N_{bs}
// N_{cbc} Number of concurrent basis      cells: N_{bl} * N_q
// N_{cqc} Number of concurrent quadrature cells: N_{bl} * N_b
// N_{sbc} Number of serial     basis      cells: N_{bs} / N_q
// N_{sqc} Number of serial     quadrature cells: N_{bs} / N_b
// N_{cb}  Number of serial cell batches:         input
// N_c     Number of total cells:                 N_{cb}*N_{t}/N_{comp}

__global__ void integrateElementCoefQuadrature(int N_cb, realType *coefficients, realType *physCoefficients, realType *jacobianInverses, realType *jacobianDeterminants, realType *elemVec) {
  #include "ex52_gpu_inline.h"
  const int        dim     = SPATIAL_DIM_0;
  const int        N_b     = numBasisFunctions_0;   // The number of basis functions
  const int        N_comp  = numBasisComponents_0;  // The number of basis function components
  const int        N_bt    = N_b*N_comp;            // The total number of scalar basis functions
  const int        N_q     = numQuadraturePoints_0; // The number of quadrature points
  const int        N_bst   = N_bt*N_q;              // The block size, LCM(N_b*N_comp, N_q), Notice that a block is not processed simultaneously
  const int        N_t     = N_bst*N_bl;            // The number of threads, N_bst * N_bl
  const int        N_bc    = N_t/N_comp;            // The number of cells per batch (N_b*N_q*N_bl)
  const int        N_c     = N_cb * N_bc;
  const int        N_sbc   = N_bst / (N_q * N_comp);
  const int        N_sqc   = N_bst / N_bt;
  /* Calculated indices */
  const int        tidx    = threadIdx.x + blockDim.x*threadIdx.y;
  const int        blidx   = tidx / N_bst;           // Block number for this thread
  const int        bidx    = tidx % N_bt;            // Basis function mapped to this thread
  const int        cidx    = tidx % N_comp;          // Basis component mapped to this thread
  const int        qidx    = tidx % N_q;             // Quadrature point mapped to this thread
  const int        blbidx  = tidx % N_q + blidx*N_q; // Cell mapped to this thread in the basis phase
  const int        blqidx  = tidx % N_b + blidx*N_b; // Cell mapped to this thread in the quadrature phase
  const int        gidx    = blockIdx.y*gridDim.x + blockIdx.x;
  const int        Goffset = gidx*N_c;
  const int        Coffset = gidx*N_c*N_bt;
  const int        Poffset = gidx*N_c*N_q;
  const int        Eoffset = gidx*N_c*N_bt;
  /* Quadrature data */
  realType             w;                   // $w_q$, Quadrature weight at $x_q$
//__shared__ realType  phi_i[N_bt*N_q];     // $\phi_i(x_q)$, Value of the basis function $i$ at $x_q$
  __shared__ vecType   phiDer_i[N_bt*N_q];  // $\frac{\partial\phi_i(x_q)}{\partial x_d}$, Value of the derivative of basis function $i$ in direction $x_d$ at $x_q$
  /* Geometric data */
  __shared__ realType  detJ[N_t];           // $|J(x_q)|$, Jacobian determinant at $x_q$
  __shared__ realType  invJ[N_t*dim*dim];   // $J^{-1}(x_q)$, Jacobian inverse at $x_q$
  /* FEM data */
  __shared__ realType  u_i[N_t*N_bt];       // Coefficients $u_i$ of the field $u|_{\mathcal{T}} = \sum_i u_i \phi_i$
  /* Physical coefficient data */
  realType             kappa;               // Physical coefficient $\kappa$ in the equation
  /* Intermediate calculations */
//__shared__ realType  f_0[N_t*N_sqc];      // $f_0(u(x_q), \nabla u(x_q)) |J(x_q)| w_q$
  __shared__ vecType   f_1[N_t*N_sqc];      // $f_1(u(x_q), \nabla u(x_q)) |J(x_q)| w_q$
  /* Output data */
  realType             e_i;                 // Coefficient $e_i$ of the residual

  /* These should be generated inline */
  /* Load quadrature weights */
  w = weights_0[qidx];
  /* Load basis tabulation \phi_i for this cell */
  if (tidx < N_bt*N_q) {
 // phi_i[tidx]    = Basis_0[tidx];
    phiDer_i[tidx] = BasisDerivatives_0[tidx];
  }

  for (int batch = 0; batch < N_cb; ++batch) {
    /* Load geometry */
    detJ[tidx] = jacobianDeterminants[Goffset+batch*N_bc+tidx];
    for (int n = 0; n < dim*dim; ++n) {
      const int offset = n*N_t;
      invJ[offset+tidx] = jacobianInverses[(Goffset+batch*N_bc)*dim*dim+offset+tidx];
    }
    /* Load coefficients u_i for this cell */
    for (int n = 0; n < N_bt; ++n) {
      const int offset = n*N_t;
      u_i[offset+tidx] = coefficients[Coffset+batch*N_t*N_b+offset+tidx];
    }
    /* Load physical coefficient for this cell */
    kappa = physCoefficients[Poffset+batch*N_t*N_q+tidx];

    /* Map coefficients to values at quadrature points */
    for (int c = 0; c < N_sqc; ++c) {
      realType  u[N_comp];     // $u(x_q)$, Value of the field at $x_q$
      vecType   gradU[N_comp]; // $\nabla u(x_q)$, Value of the field gradient at $x_q$
   // vecType   x             = {0.0, 0.0};           // Quadrature point $x_q$
      const int cell          = c*N_bl*N_b + blqidx;
      const int fidx          = (cell*N_q + qidx)*N_comp + cidx;

      for (int comp = 0; comp < N_comp; ++comp) {
        //u[comp] = 0.0;
#if SPATIAL_DIM_0 == 2
        gradU[comp].x = 0.0; gradU[comp].y = 0.0;
#elif  SPATIAL_DIM_0 == 3
        gradU[comp].x = 0.0; gradU[comp].y = 0.0; gradU[comp].z = 0.0;
#endif
      }
      /* Get field and derivatives at this quadrature point */
      for (int i = 0; i < N_b; ++i) {
        for (int comp = 0; comp < N_comp; ++comp) {
          const int b     = i*N_comp+comp;
          const int pidx  = qidx*N_bt + b;
          const int uidx  = cell*N_bt + b;
          vecType    realSpaceDer;

       // u[comp] += u_i[uidx]*phi_i[qidx*N_bt+bbidx];
#if SPATIAL_DIM_0 == 2
          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;
          gradU[comp].x += u_i[uidx]*realSpaceDer.x;
          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;
          gradU[comp].y += u_i[uidx]*realSpaceDer.y;
#elif  SPATIAL_DIM_0 == 3
          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;
          gradU[comp].x += u_i[uidx]*realSpaceDer.x;
          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;
          gradU[comp].y += u_i[uidx]*realSpaceDer.y;
          realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;
          gradU[comp].z += u_i[uidx]*realSpaceDer.z;
#endif
        }
      }
      /* Process values at quadrature points */
      f_1[fidx] = f1_coef_func(u, gradU, kappa, cidx);
#if SPATIAL_DIM_0 == 2
      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w;
#elif  SPATIAL_DIM_0 == 3
      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w; f_1[fidx].z *= detJ[cell]*w;
#endif
    }

    /* ==== TRANSPOSE THREADS ==== */
    __syncthreads();

    /* Map values at quadrature points to coefficients */
    for (int c = 0; c < N_sbc; ++c) {
      const int cell = c*N_bl*N_q + blbidx;

      e_i = 0.0;
      for (int q = 0; q < N_q; ++q) {
        const int pidx = q*N_bt + bidx;
        const int fidx = (cell*N_q + q)*N_comp + cidx;
        vecType realSpaceDer;

     // e_i += phi_i[pidx]*f_0[fidx];
#if SPATIAL_DIM_0 == 2
        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;
        e_i += realSpaceDer.x*f_1[fidx].x;
        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;
        e_i += realSpaceDer.y*f_1[fidx].y;
#elif  SPATIAL_DIM_0 == 3
        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;
        e_i += realSpaceDer.x*f_1[fidx].x;
        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;
        e_i += realSpaceDer.y*f_1[fidx].y;
        realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;
        e_i += realSpaceDer.z*f_1[fidx].z;
#endif
      }
#if 0
      // Check f_1
      {
        const int q = 0;
        const int i = bidx/N_comp;
        // Prints f1[0].x, f1[1].x, f1[0].y, f1[1].y
        switch(i) {
        case 0:
          e_i = f_1[(cell*N_q+q)*N_comp+cidx].x;break;
        case 1:
          e_i = f_1[(cell*N_q+q)*N_comp+cidx].y;break;
        //case 2:
          //e_i = f_1[(cell*N_q+q)*N_comp+cidx].z;break;
        default:
          e_i = 0.0;
        }
      }
      // Check that u_i is being used correctly
      //e_i = u_i[cell*N_bt+bidx];
      e_i = detJ[cell];
      //e_i = coefficients[Coffset+(batch*N_sbc+c)*N_t+tidx];
      //e_i = Coffset+(batch*N_sbc+c)*N_t+tidx;
      //e_i = cell*N_bt+bidx;
#endif
      /* Write element vector for N_{cbc} cells at a time */
      elemVec[Eoffset+(batch*N_sbc+c)*N_t+tidx] = e_i;
    }
    /* ==== Could do one write per batch ==== */
  }
  return;
}

// Calculate a conforming thread grid for N kernels
#undef __FUNCT__
#define __FUNCT__ "calculateGridCoef"
PetscErrorCode calculateGridCoef(const int N, const int blockSize, unsigned int& x, unsigned int& y, unsigned int& z)
{
  PetscFunctionBegin;
  z = 1;
  if (N % blockSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid block size %d for %d elements", blockSize, N);
  const int Nblocks = N/blockSize;
  for (x = (int) (sqrt(Nblocks) + 0.5); x > 0; --x) {
    y = Nblocks/x;
    if (x*y == Nblocks) break;
  }
  if (x*y != Nblocks) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Could not find partition for %d with block size %d", N, blockSize);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "IntegrateElementCoefBatchGPU"
/*
  IntegrateElementCoefBatchGPU - Produces element vectors from input element solution and geometric information via quadrature

  Input Parameters:
+ Ne - The total number of cells, Nchunk * Ncb * Nbc
. Ncb - The number of serial cell batches
. Nbc - The number of cells per batch
. Nbl - The number of concurrent cells blocks per thread block
. coefficients - An array of the solution vector for each cell
. physCoefficients - An array of the physical coefficient values at quadrature points for each cell
. jacobianInverses - An array of the inverse Jacobian for each cell
. jacobianDeterminants - An array of the Jacobian determinant for each cell
. event - A PetscEvent, used to log flops
- debug - A flag for debugging information

  Output Parameter:
. elemVec - An array of the element vectors for each cell
*/
PetscErrorCode IntegrateElementCoefBatchGPU(PetscInt Ne, PetscInt Ncb, PetscInt Nbc, PetscInt Nbl, const PetscScalar coefficients[], const PetscScalar physCoefficients[],
                                            const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[],
                                            PetscLogEvent event, PetscInt debug) {
  #include "ex52_gpu_inline.h"
  const int dim    = SPATIAL_DIM_0;
  const int N_b    = numBasisFunctions_0;   // The number of basis functions
  const int N_comp = numBasisComponents_0;  // The number of basis function components
  const int N_bt   = N_b*N_comp;            // The total number of scalar basis functions
  const int N_q    = numQuadraturePoints_0; // The number of quadrature points
  const int N_bst  = N_bt*N_q;              // The block size, LCM(N_bt, N_q), Notice that a block is not process simultaneously
  const int N_t    = N_bst*N_bl;            // The number of threads, N_bst * N_bl

  realType *d_coefficients;
  realType *d_physCoefficients;
  realType *d_jacobianInverses;
  realType *d_jacobianDeterminants;
  realType *d_elemVec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Nbl != N_bl) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsisten block size %d should be %d", Nbl, N_bl);
  if (Nbc*N_comp != N_t) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of threads %d should be %d * %d", N_t, Nbc, N_comp);
  if (!Ne) {
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog = PETSC_NULL;
    PetscInt          stage;

    ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);CHKERRQ(ierr);
    /* Log performance info */
    eventLog->eventInfo[event].count++;
    eventLog->eventInfo[event].time  += 0.0;
    eventLog->eventInfo[event].flops += 0;
    PetscFunctionReturn(0);
  }
  // Marshalling
  ierr = cudaMalloc((void**) &d_coefficients,         Ne*N_bt * sizeof(realType));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_physCoefficients,     Ne*N_bt * sizeof(realType));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_jacobianInverses,     Ne*dim*dim * sizeof(realType));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_jacobianDeterminants, Ne * sizeof(realType));CHKERRQ(ierr);
  ierr = cudaMalloc((void**) &d_elemVec,              Ne*N_bt * sizeof(realType));CHKERRQ(ierr);
  if (sizeof(PetscReal) == sizeof(realType)) {
    ierr = cudaMemcpy(d_coefficients,         coefficients,         Ne*N_bt    * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = cudaMemcpy(d_physCoefficients,     physCoefficients,     Ne*N_q     * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = cudaMemcpy(d_jacobianInverses,     jacobianInverses,     Ne*dim*dim * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = cudaMemcpy(d_jacobianDeterminants, jacobianDeterminants, Ne         * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
  } else {
    realType *c, *pc, *jI, *jD;
    PetscInt  i;

    ierr = PetscMalloc4(Ne*N_bt,realType,&c,Ne*N_q,realType,&pc,Ne*dim*dim,realType,&jI,Ne,realType,&jD);CHKERRQ(ierr);
    for (i = 0; i < Ne*N_bt;    ++i) {c[i]  = coefficients[i];}
    for (i = 0; i < Ne*N_q;     ++i) {pc[i] = physCoefficients[i];}
    for (i = 0; i < Ne*dim*dim; ++i) {jI[i] = jacobianInverses[i];}
    for (i = 0; i < Ne;         ++i) {jD[i] = jacobianDeterminants[i];}
    ierr = cudaMemcpy(d_coefficients,         c,  Ne*N_bt    * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = cudaMemcpy(d_physCoefficients,     pc, Ne*N_q     * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = cudaMemcpy(d_jacobianInverses,     jI, Ne*dim*dim * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = cudaMemcpy(d_jacobianDeterminants, jD, Ne         * sizeof(realType), cudaMemcpyHostToDevice);CHKERRQ(ierr);
    ierr = PetscFree4(c,pc,jI,jD);CHKERRQ(ierr);
  }
  // Kernel launch
  unsigned int x, y, z;
  ierr = calculateGridCoef(Ne, Ncb*Nbc, x, y, z);CHKERRQ(ierr);
  dim3 grid(x, y, z);
  dim3 block(Nbc*N_comp, 1, 1);
  cudaEvent_t start, stop;
  float msElapsedTime;

  ierr = cudaEventCreate(&start);CHKERRQ(ierr);
  ierr = cudaEventCreate(&stop);CHKERRQ(ierr);
  //if (debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "GPU layout grid(%d,%d,%d) block(%d,%d,%d) with %d batches\n",
                       grid.x, grid.y, grid.z, block.x, block.y, block.z, Ncb);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " N_t: %d, N_cb: %d\n", N_t, Ncb);
  //}
  ierr = cudaEventRecord(start, 0);CHKERRQ(ierr);
  integrateElementCoefQuadrature<<<grid, block>>>(Ncb, d_coefficients, d_physCoefficients, d_jacobianInverses, d_jacobianDeterminants, d_elemVec);
  ierr = cudaEventRecord(stop, 0);CHKERRQ(ierr);
  ierr = cudaEventSynchronize(stop);CHKERRQ(ierr);
  ierr = cudaEventElapsedTime(&msElapsedTime, start, stop);CHKERRQ(ierr);
  ierr = cudaEventDestroy(start);CHKERRQ(ierr);
  ierr = cudaEventDestroy(stop);CHKERRQ(ierr);
  // Marshalling
  if (sizeof(PetscReal) == sizeof(realType)) {
    ierr = cudaMemcpy(elemVec, d_elemVec, Ne*N_bt * sizeof(realType), cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  } else {
    realType *eV;
    PetscInt  i;

    ierr = PetscMalloc(Ne*N_bt * sizeof(realType), &eV);CHKERRQ(ierr);
    ierr = cudaMemcpy(eV, d_elemVec, Ne*N_bt * sizeof(realType), cudaMemcpyDeviceToHost);CHKERRQ(ierr);
    for (i = 0; i < Ne*N_bt; ++i) {elemVec[i] = eV[i];}
    ierr = PetscFree(eV);CHKERRQ(ierr);
  }
  ierr = cudaFree(d_coefficients);CHKERRQ(ierr);
  ierr = cudaFree(d_jacobianInverses);CHKERRQ(ierr);
  ierr = cudaFree(d_jacobianDeterminants);CHKERRQ(ierr);
  ierr = cudaFree(d_elemVec);CHKERRQ(ierr);
  {
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog = PETSC_NULL;
    PetscInt          stage;

    ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);CHKERRQ(ierr);
    /* Log performance info */
    eventLog->eventInfo[event].count++;
    eventLog->eventInfo[event].time  += msElapsedTime*1.0e-3;
    eventLog->eventInfo[event].flops += (((2+(2+2*dim)*dim)*N_comp*N_b+(2+2)*dim*N_comp)*N_q + (2+2*dim)*dim*N_q*N_comp*N_b)*Ne;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
