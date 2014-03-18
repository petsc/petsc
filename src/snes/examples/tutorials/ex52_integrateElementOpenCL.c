#include <petscsys.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//#define SPATIAL_DIM_0 2

typedef enum {LAPLACIAN = 0, ELASTICITY} OpType;

/* Put the OpenCL program into a source string.
 * This allows to generate all the code at runtime, no need for external Python magic as for CUDA
 *
 * The code uses snprintf() to concatenate strings, as this is safer than strcat().
 */
#undef __FUNCT__
#define __FUNCT__ "generateOpenCLSource"
PetscErrorCode generateOpenCLSource(char **string_buffer, PetscInt buffer_length, PetscInt spatial_dim, PetscInt N_bl, PetscInt pde_op)
{
  char            *string_tail   = *string_buffer;
  char            *end_of_buffer = *string_buffer + buffer_length;
  PetscInt        num_quadrature_points = 1;
  PetscInt        num_basis_components = (pde_op == LAPLACIAN) ? 1 : spatial_dim;
  PetscInt        num_basis_functions = 3;
  PetscInt        num_threads = num_basis_functions * num_basis_components * num_quadrature_points * N_bl; /* N_t */

/* dim     Number of spatial dimensions:          2                   */
/* N_b     Number of basis functions:             generated           */
/* N_{bt}  Number of total basis functions:       N_b * N_{comp}      */
/* N_q     Number of quadrature points:           generated           */
/* N_{bs}  Number of block cells                  LCM(N_b, N_q)       */
/* N_{bst} Number of block cell components        LCM(N_{bt}, N_q)    */
/* N_{bl}  Number of concurrent blocks            generated           */
/* N_t     Number of threads:                     N_{bl} * N_{bs}     */
/* N_{cbc} Number of concurrent basis      cells: N_{bl} * N_q        */
/* N_{cqc} Number of concurrent quadrature cells: N_{bl} * N_b        */
/* N_{sbc} Number of serial     basis      cells: N_{bs} / N_q        */
/* N_{sqc} Number of serial     quadrature cells: N_{bs} / N_b        */
/* N_{cb}  Number of serial cell batches:         input               */
/* N_c     Number of total cells:                 N_{cb}*N_{t}/N_{comp} */

#define STRING_ERROR_CHECK(MSG) \
  if (string_tail == end_of_buffer) {\
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, MSG);\
  }

  PetscFunctionBegin;
  char float_str[] = "float";
  char double_str[] = "double";
  char *numeric_str = &(float_str[0]);

  /* Enable device extension for double precision */
  if (sizeof(PetscReal) == sizeof(double)) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#endif\n");
    numeric_str = &(double_str[0]);
  }

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"\n"
"__kernel void integrateElementQuadrature(int N_cb, __global %s *coefficients, __global %s *jacobianInverses, __global %s *jacobianDeterminants, __global %s *elemVec)\n"
"{\n", numeric_str, numeric_str, numeric_str, numeric_str);STRING_ERROR_CHECK("Message to short");

  if (spatial_dim == 2) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"  const int numQuadraturePoints_0 = %d;\n"
"\n"
"  /* Quadrature points\n"
"   - (x1,y1,x2,y2,...) */\n"
"  const %s points_0[2] = {\n"
"    -0.333333333333,\n"
"    -0.333333333333};\n"
"\n"
"  /* Quadrature weights\n"
"   - (v1,v2,...) */\n"
"  const %s weights_0[1] = {2.0};\n"
"\n"
"  const int numBasisFunctions_0 = %d;\n"
"  const int numBasisComponents_0 = %d;\n", num_quadrature_points, numeric_str, numeric_str, num_basis_functions, num_basis_components);STRING_ERROR_CHECK("Message to short");

    if (pde_op == LAPLACIAN) {
      string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"\n"
"  /* Nodal basis function evaluations\n"
"    - basis function is fastest varying, then point */\n"
"  const %s Basis_0[3] = {\n"
"    0.333333333333,\n"
"    0.333333333333,\n"
"    0.333333333333};\n"
"\n"
"  /* Nodal basis function derivative evaluations,\n"
"      - derivative direction fastest varying, then basis function, then point */\n"
"  const %s2 BasisDerivatives_0[3] = {\n"
"    (%s2)(-0.5, -0.5),\n"
"    (%s2)(0.5, 0.0),\n"
"    (%s2)(0.0, 0.5)};\n"
"\n", numeric_str, numeric_str, numeric_str, numeric_str, numeric_str);STRING_ERROR_CHECK("Message to short");
    } else if (pde_op == ELASTICITY) {
      string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"\n"
"  /* Nodal basis function evaluations\n"
"    - basis function is fastest varying, then point */\n"
"  const %s Basis_0[6] = {\n"
"    0.333333333333,\n"
"    0.333333333333,\n"
"    0.333333333333,\n"
"    0.333333333333,\n"
"    0.333333333333,\n"
"    0.333333333333};\n"
"\n"
"  /* Nodal basis function derivative evaluations,\n"
"      - derivative direction fastest varying, then basis function, then point */\n"
"  const %s2 BasisDerivatives_0[6] = {\n"
"    (%s2)(-0.5, -0.5),\n"
"    (%s2)(-0.5, -0.5),\n"
"    (%s2)(0.5, 0.0),\n"
"    (%s2)(0.5, 0.0),\n"
"    (%s2)(0.0, 0.5),\n"
"    (%s2)(0.0, 0.5)};\n"
"\n", numeric_str, numeric_str, numeric_str, numeric_str, numeric_str, numeric_str, numeric_str, numeric_str);STRING_ERROR_CHECK("Message to short");
    }
  } else if (spatial_dim == 3) {
  }

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"  /* Number of concurrent blocks */\n"
"  const int N_bl = %d;\n"
"\n"
/* Argument */
"  const int dim    = %d;\n"
/* Argument */
"  const int N_b    = numBasisFunctions_0;           // The number of basis functions\n"
"  const int N_comp = numBasisComponents_0;          // The number of basis function components\n"
"  const int N_bt   = N_b*N_comp;                    // The total number of scalar basis functions\n"
"  const int N_q    = numQuadraturePoints_0;         // The number of quadrature points\n"
"  const int N_bst  = N_bt*N_q;                      // The block size, LCM(N_b*N_comp, N_q), Notice that a block is not processed simultaneously\n"
"  const int N_t    = N_bst*N_bl;                    // The number of threads, N_bst * N_bl\n"
"  const int N_bc   = N_t/N_comp;                    // The number of cells per batch (N_b*N_q*N_bl)\n"
"  const int N_c    = N_cb * N_bc;\n"
"  const int N_sbc  = N_bst / (N_q * N_comp);\n"
"  const int N_sqc  = N_bst / N_bt;\n"
"\n"
"  /* Calculated indices */\n"
"  const int tidx    = get_local_id(0) + get_local_size(0)*get_local_id(1);\n"
"  const int blidx   = tidx / N_bst;                  // Block number for this thread\n"
"  const int bidx    = tidx %% N_bt;                   // Basis function mapped to this thread\n"
"  const int cidx    = tidx %% N_comp;                 // Basis component mapped to this thread\n"
"  const int qidx    = tidx %% N_q;                    // Quadrature point mapped to this thread\n"
"  const int blbidx  = tidx %% N_q + blidx*N_q;        // Cell mapped to this thread in the basis phase\n"
"  const int blqidx  = tidx %% N_b + blidx*N_b;        // Cell mapped to this thread in the quadrature phase\n"
"  const int gidx    = get_group_id(1)*get_num_groups(0) + get_group_id(0);\n"
"  const int Goffset = gidx*N_c;\n"
"  const int Coffset = gidx*N_c*N_bt;\n"
"  const int Eoffset = gidx*N_c*N_bt;\n", N_bl, spatial_dim);STRING_ERROR_CHECK("Message to short");

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"\n"
"  /* Quadrature data */\n"
"  %s                w;                   // $w_q$, Quadrature weight at $x_q$\n"
"  __local %s%d       phiDer_i[%d]; //[N_bt*N_q];  // $\\frac{\\partial\\phi_i(x_q)}{\\partial x_d}$, Value of the derivative of basis function $i$ in direction $x_d$ at $x_q$\n"
"  /* Geometric data */\n"
"  __local %s        detJ[%d]; //[N_t];           // $|J(x_q)|$, Jacobian determinant at $x_q$\n"
"  __local %s        invJ[%d];//[N_t*dim*dim];   // $J^{-1}(x_q)$, Jacobian inverse at $x_q$\n"
"  /* FEM data */\n"
"  __local %s        u_i[%d]; //[N_t*N_bt];       // Coefficients $u_i$ of the field $u|_{\\mathcal{T}} = \\sum_i u_i \\phi_i$\n"
"  /* Intermediate calculations */\n"
"  __local %s%d       f_1[%d]; //[N_t*N_sqc];      // $f_1(u(x_q), \\nabla u(x_q)) |J(x_q)| w_q$\n"
"  /* Output data */\n"
"  %s                e_i;                 // Coefficient $e_i$ of the residual\n"
"\n", numeric_str,
      numeric_str, spatial_dim,
      num_basis_functions * num_basis_components * num_quadrature_points,     /* size of PhiDer_i */
      numeric_str, num_threads, /* size of detJ */
      numeric_str, num_threads * spatial_dim * spatial_dim, /* size of invJ */
      numeric_str, num_threads * num_basis_functions * num_basis_components, /* size of u_i */
      numeric_str, spatial_dim, num_threads * num_quadrature_points /* size of f_1 */,
      numeric_str);STRING_ERROR_CHECK("Message to short");

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"  /* These should be generated inline */\n"
"  /* Load quadrature weights */\n"
"  w = weights_0[qidx];\n"
"  /* Load basis tabulation \\phi_i for this cell */\n"
"  if (tidx < N_bt*N_q) {\n"
" // phi_i[tidx]    = Basis_0[tidx];\n"
"    phiDer_i[tidx] = BasisDerivatives_0[tidx];\n"
"  }\n"
"\n"
"  for (int batch = 0; batch < N_cb; ++batch) {\n"
"    /* Load geometry */\n"
"    detJ[tidx] = jacobianDeterminants[Goffset+batch*N_bc+tidx];\n"
"    for (int n = 0; n < dim*dim; ++n) {\n"
"      const int offset = n*N_t;\n"
"      invJ[offset+tidx] = jacobianInverses[(Goffset+batch*N_bc)*dim*dim+offset+tidx];\n"
"    }\n"
"    /* Load coefficients u_i for this cell */\n"
"    for (int n = 0; n < N_bt; ++n) {\n"
"      const int offset = n*N_t;\n"
"      u_i[offset+tidx] = coefficients[Coffset+batch*N_t*N_b+offset+tidx];\n"
"    }\n"
"\n"
"    /* Map coefficients to values at quadrature points */\n"
"    for (int c = 0; c < N_sqc; ++c) {\n"
"      %s  u[%d]; //[N_comp];     // $u(x_q)$, Value of the field at $x_q$\n"
"      %s%d   gradU[%d]; //[N_comp]; // $\\nabla u(x_q)$, Value of the field gradient at $x_q$\n"
"      const int cell          = c*N_bl*N_b + blqidx;\n"
"      const int fidx          = (cell*N_q + qidx)*N_comp + cidx;\n"
"\n"
"      for (int comp = 0; comp < N_comp; ++comp) {\n"
"        gradU[comp].x = 0.0; gradU[comp].y = 0.0;", numeric_str, num_basis_components, numeric_str, spatial_dim, num_basis_components);STRING_ERROR_CHECK("Message to short");

  if (spatial_dim == 3) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail, " gradU[comp].z = 0.0;");
    if (string_tail == end_of_buffer) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "String too short!");
    }
  }

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"\n"
"      }\n"
"      /* Get field and derivatives at this quadrature point */\n"
"      for (int i = 0; i < N_b; ++i) {\n"
"        for (int comp = 0; comp < N_comp; ++comp) {\n"
"          const int b    = i*N_comp+comp;\n"
"          const int pidx = qidx*N_bt + b;\n"
"          const int uidx = cell*N_bt + b;\n"
"          %s%d   realSpaceDer;\n"
"\n", numeric_str, spatial_dim);STRING_ERROR_CHECK("Message to short");

  if (spatial_dim == 2) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;\n"
"          gradU[comp].x += u_i[uidx]*realSpaceDer.x;\n"
"          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;\n"
"          gradU[comp].y += u_i[uidx]*realSpaceDer.y;\n");STRING_ERROR_CHECK("Message to short");
  } else {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;\n"
"          gradU[comp].x += u_i[uidx]*realSpaceDer.x;\n"
"          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;\n"
"          gradU[comp].y += u_i[uidx]*realSpaceDer.y;\n"
"          realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;\n"
"          gradU[comp].z += u_i[uidx]*realSpaceDer.z;\n");STRING_ERROR_CHECK("Message to short");
  }

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"        }\n"
"      }\n"
"      /* Process values at quadrature points */\n");STRING_ERROR_CHECK("Message to short");

  /* Process values at quadrature points as induced by the PDE operator */
  if (pde_op == LAPLACIAN) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail, "      f_1[fidx] = gradU[cidx];\n");STRING_ERROR_CHECK("Message to short");
  } else if (spatial_dim == 2 && pde_op == ELASTICITY) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"      switch (cidx) {\n"
"      case 0:\n"
"        f_1[fidx].x = 0.5*(gradU[0].x + gradU[0].x);\n"
"        f_1[fidx].y = 0.5*(gradU[0].y + gradU[1].x);\n"
"        break;\n"
"      case 1:\n"
"        f_1[fidx].x = 0.5*(gradU[1].x + gradU[0].y);\n"
"        f_1[fidx].y = 0.5*(gradU[1].y + gradU[1].y);\n"
"      }\n");STRING_ERROR_CHECK("Message to short");
  } else if (spatial_dim == 3 && pde_op == ELASTICITY) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"      switch (cidx) {\n"
"      case 0:\n"
"        f_1[fidx].x = 0.5*(gradU[0].x + gradU[0].x);\n"
"        f_1[fidx].y = 0.5*(gradU[0].y + gradU[1].x);\n"
"        f_1[fidx].z = 0.5*(gradU[0].z + gradU[2].x);\n"
"        break;\n"
"      case 1:\n"
"        f_1[fidx].x = 0.5*(gradU[1].x + gradU[0].y);\n"
"        f_1[fidx].y = 0.5*(gradU[1].y + gradU[1].y);\n"
"        f_1[fidx].z = 0.5*(gradU[1].y + gradU[2].y);\n"
"        break;\n"
"      case 2:\n"
"        f_1[fidx].x = 0.5*(gradU[2].x + gradU[0].z);\n"
"        f_1[fidx].y = 0.5*(gradU[2].y + gradU[1].z);\n"
"        f_1[fidx].z = 0.5*(gradU[2].y + gradU[2].z);\n"
"      }\n");STRING_ERROR_CHECK("Message to short");
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Combination of spatial dimension and PDE operator invalid");
  }

  if (spatial_dim == 2) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail, "      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w; \n");STRING_ERROR_CHECK("Message to short");
  } else if (spatial_dim == 2) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail, "      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w; f_1[fidx].z *= detJ[cell]*w;\n");STRING_ERROR_CHECK("Message to short");
  }

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"\n"
"    }\n"
"\n"
"    /* ==== TRANSPOSE THREADS ==== */\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"    /* Map values at quadrature points to coefficients */\n"
"    for (int c = 0; c < N_sbc; ++c) {\n"
"      const int cell = c*N_bl*N_q + blbidx;\n"
"\n"
"      e_i = 0.0;\n"
"      for (int q = 0; q < N_q; ++q) {\n"
"        const int pidx = q*N_bt + bidx;\n"
"        const int fidx = (cell*N_q + q)*N_comp + cidx;\n"
"        %s%d   realSpaceDer;\n"
"\n"
"        // e_i += phi_i[pidx]*f_0[fidx];\n", numeric_str, spatial_dim);STRING_ERROR_CHECK("Message to short");

  if (spatial_dim == 2) {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;\n"
"        e_i           += realSpaceDer.x*f_1[fidx].x;\n"
"        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;\n"
"        e_i           += realSpaceDer.y*f_1[fidx].y;\n");STRING_ERROR_CHECK("Message to short");
  } else {
    string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;\n"
"        e_i           += realSpaceDer.x*f_1[fidx].x;\n"
"        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;\n"
"        e_i           += realSpaceDer.y*f_1[fidx].y;\n"
"        realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;\n"
"        e_i           += realSpaceDer.z*f_1[fidx].z;\n");STRING_ERROR_CHECK("Message to short");
  }

  string_tail += snprintf(string_tail, end_of_buffer - string_tail,
"      }\n"
"      /* Write element vector for N_{cbc} cells at a time */\n"
"      elemVec[Eoffset+(batch*N_sbc+c)*N_t+tidx] = e_i;\n"
"    }\n"
"    /* ==== Could do one write per batch ==== */\n"
"  }\n"
"  return;\n"
"}  \n");STRING_ERROR_CHECK("Message to short");

  PetscFunctionReturn(0);
}


/* Struct collecting information for a typical OpenCL environment (one platform, one device, one context, one queue) */
typedef struct OpenCLEnvironment_s
{
  cl_platform_id    pf_id;
  cl_device_id      dev_id;
  cl_context        ctx_id;
  cl_command_queue  queue_id;
} OpenCLEnvironment;

// Calculate a conforming thread grid for N kernels
#undef __FUNCT__
#define __FUNCT__ "initializeOpenCL"
PetscErrorCode initializeOpenCL(OpenCLEnvironment * ocl_env)
{
  cl_uint            num_platforms;
  cl_platform_id     platform_ids[42];
  cl_uint            num_devices;
  cl_device_id       device_ids[42];
  cl_int             ierr;

  PetscFunctionBegin;
  /* Init Platform */
  ierr = clGetPlatformIDs(42, platform_ids, &num_platforms);CHKERRQ(ierr);
  if (num_platforms == 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "No OpenCL platform found.");
  }
  ocl_env->pf_id = platform_ids[0];

  /* Init Device */
  ierr = clGetDeviceIDs(ocl_env->pf_id, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices);CHKERRQ(ierr);
  if (num_platforms == 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "No OpenCL device found.");
  }
  ocl_env->dev_id = device_ids[0];

  /* Create context with one command queue */
  ocl_env->ctx_id   = clCreateContext(0, 1, &(device_ids[0]), NULL, NULL, &ierr);CHKERRQ(ierr);
  ocl_env->queue_id = clCreateCommandQueue(ocl_env->ctx_id, ocl_env->dev_id, CL_QUEUE_PROFILING_ENABLE, &ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "destroyOpenCL"
PetscErrorCode destroyOpenCL(OpenCLEnvironment * ocl_env)
{
  cl_int             ierr;

  PetscFunctionBegin;
  ierr = clReleaseCommandQueue(ocl_env->queue_id);CHKERRQ(ierr);
  ocl_env->queue_id = 0;

  ierr = clReleaseContext(ocl_env->ctx_id);CHKERRQ(ierr);
  ocl_env->ctx_id = 0;
  PetscFunctionReturn(0);
}

// Calculate a conforming thread grid for N kernels
#undef __FUNCT__
#define __FUNCT__ "calculateGridOpenCL"
PetscErrorCode calculateGridOpenCL(const int N, const int blockSize, unsigned int * x, unsigned int * y, unsigned int * z)
{
  PetscFunctionBegin;
  *z = 1;
  if (N % blockSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid block size %d for %d elements", blockSize, N);
  const int Nblocks = N/blockSize;
  for (*x = (int) (sqrt(Nblocks) + 0.5); *x > 0; --*x) {
    *y = Nblocks / *x;
    if (*x * *y == Nblocks) break;
  }
  if (*x * *y != Nblocks) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Could not find partition for %d with block size %d", N, blockSize);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateElementBatchGPU"
/*
  IntegrateElementBatchOpenCL - Produces element vectors from input element solution and geometric information via quadrature

  Input Parameters:
+ Ne - The total number of cells, Nchunk * Ncb * Nbc
. Ncb - The number of serial cell batches
. Nbc - The number of cells per batch
. Nbl - The number of concurrent cells blocks per thread block
. coefficients - An array of the solution vector for each cell
. jacobianInverses - An array of the inverse Jacobian for each cell
. jacobianDeterminants - An array of the Jacobian determinant for each cell
. event - A PetscEvent, used to log flops
- debug - A flag for debugging information

  Output Parameter:
. elemVec - An array of the element vectors for each cell
*/
PETSC_EXTERN PetscErrorCode IntegrateElementBatchGPU(PetscInt spatial_dim, PetscInt Ne, PetscInt Ncb, PetscInt Nbc, PetscInt N_bl, const PetscScalar coefficients[],
                                                     const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[],
                                                     PetscLogEvent event, PetscInt debug, PetscInt pde_op)
{
  const cl_int numQuadraturePoints_0 = 1;

  const cl_int numBasisFunctions_0 = 3;
  const cl_int numBasisComponents_0 = (pde_op == LAPLACIAN) ? 1 : spatial_dim;

  const cl_int dim    = spatial_dim;
  const cl_int N_b    = numBasisFunctions_0;   /* The number of basis functions */
  const cl_int N_comp = numBasisComponents_0;  /* The number of basis function components */
  const cl_int N_bt   = N_b*N_comp;            /* The total number of scalar basis functions */
  const cl_int N_q    = numQuadraturePoints_0; /* The number of quadrature points */
  const cl_int N_bst  = N_bt*N_q;              /* The block size, LCM(N_bt, N_q), Notice that a block is not process simultaneously */
  const cl_int N_t    = N_bst*N_bl;            /* The number of threads, N_bst * N_bl */

  char            *program_buffer;
  char            build_buffer[8192];
  cl_build_status status;

  cl_event          ocl_ev;         /* The event for tracking kernel execution */
  cl_ulong          ns_start;       /* Nanoseconds counter on GPU at kernel start */
  cl_ulong          ns_end;         /* Nanoseconds counter on GPU at kernel stop */

  cl_mem            d_coefficients;
  cl_mem            d_jacobianInverses;
  cl_mem            d_jacobianDeterminants;
  cl_mem            d_elemVec;

  OpenCLEnvironment ocl_env;
  cl_program        ocl_prog;
  cl_kernel         ocl_kernel;
  size_t            ocl_source_length;
  size_t            local_work_size[3];
  size_t            global_work_size[3];
  size_t            i;
  unsigned int      x, y, z;
  PetscErrorCode    ierr;
  cl_int            ierr2;


  PetscFunctionBegin;
  ierr = initializeOpenCL(&ocl_env);CHKERRQ(ierr);
  ierr = PetscMalloc1(8192, &program_buffer);CHKERRQ(ierr);
  ierr = generateOpenCLSource(&program_buffer, 8192, dim, N_bl, pde_op);CHKERRQ(ierr);
  ocl_source_length = strlen(program_buffer);
  ocl_prog = clCreateProgramWithSource(ocl_env.ctx_id, 1, (const char**)&program_buffer, &ocl_source_length, &ierr2);CHKERRQ(ierr2);
  ierr = clBuildProgram(ocl_prog, 0, NULL, NULL, NULL, NULL);
  if (ierr != CL_SUCCESS) {
    clGetProgramBuildInfo(ocl_prog, ocl_env.dev_id, CL_PROGRAM_BUILD_LOG, sizeof(char)*8192, &build_buffer, NULL);
    printf("Build failed! Log:\n %s", build_buffer);
  }
  CHKERRQ(ierr);
  ierr = PetscFree(program_buffer);CHKERRQ(ierr);

  ocl_kernel = clCreateKernel(ocl_prog, "integrateElementQuadrature", &ierr);CHKERRQ(ierr);

  if (Nbc*N_comp != N_t) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of threads %d should be %d * %d", N_t, Nbc, N_comp);
  if (!Ne) {
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog = NULL;
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

  /* Create buffers on the device and send data over */
  d_coefficients         = clCreateBuffer(ocl_env.ctx_id, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Ne*N_bt    * sizeof(PetscReal), (void*)coefficients,         &ierr);CHKERRQ(ierr);
  d_jacobianInverses     = clCreateBuffer(ocl_env.ctx_id, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Ne*dim*dim * sizeof(PetscReal), (void*)jacobianInverses,     &ierr);CHKERRQ(ierr);
  d_jacobianDeterminants = clCreateBuffer(ocl_env.ctx_id, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Ne         * sizeof(PetscReal), (void*)jacobianDeterminants, &ierr);CHKERRQ(ierr);
  d_elemVec              = clCreateBuffer(ocl_env.ctx_id, CL_MEM_READ_WRITE,                        Ne*N_bt    * sizeof(PetscReal), NULL,                        &ierr);CHKERRQ(ierr);

  /* Work size preparations */
  ierr = calculateGridOpenCL(Ne, Ncb*Nbc, &x, &y, &z);CHKERRQ(ierr);
  local_work_size[0] = Nbc*N_comp;
  local_work_size[1] = 1;
  local_work_size[2] = 1;
  global_work_size[0] = x * local_work_size[0];
  global_work_size[1] = y * local_work_size[1];
  global_work_size[2] = z * local_work_size[2];

  /* if (debug) { */
  ierr = PetscPrintf(PETSC_COMM_SELF, "GPU layout grid(%d,%d,%d) block(%d,%d,%d) with %d batches\n",
                     x, y, z,
                     local_work_size[0], local_work_size[1], local_work_size[2], Ncb);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, " N_t: %d, N_cb: %d\n", N_t, Ncb);
  /* } */

  /* Kernel launch */
  /* integrateElementQuadrature<<<grid, block>>>(Ncb, d_coefficients, d_jacobianInverses, d_jacobianDeterminants, d_elemVec); */
  ierr = clSetKernelArg(ocl_kernel, 0, sizeof(cl_int), (void*)&Ncb);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 1, sizeof(cl_mem), (void*)&d_coefficients);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 2, sizeof(cl_mem), (void*)&d_jacobianInverses);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 3, sizeof(cl_mem), (void*)&d_jacobianDeterminants);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 4, sizeof(cl_mem), (void*)&d_elemVec);CHKERRQ(ierr);

  ierr = clEnqueueNDRangeKernel(ocl_env.queue_id, ocl_kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &ocl_ev);CHKERRQ(ierr);

  /* Read data back from device */
  ierr = clEnqueueReadBuffer(ocl_env.queue_id, d_elemVec, CL_TRUE, 0, Ne*N_bt * sizeof(PetscReal), elemVec, 0, NULL, NULL);CHKERRQ(ierr);

  {
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog = NULL;
    PetscInt          stage;

    ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);CHKERRQ(ierr);
    /* Log performance info */
    ierr = clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ns_start, NULL);CHKERRQ(ierr);
    ierr = clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &ns_end,   NULL);CHKERRQ(ierr);
    eventLog->eventInfo[event].count++;
    eventLog->eventInfo[event].time  += (ns_end - ns_start)*1.0e-9;
    eventLog->eventInfo[event].flops += (((2+(2+2*dim)*dim)*N_comp*N_b+(2+2)*dim*N_comp)*N_q + (2+2*dim)*dim*N_q*N_comp*N_b)*Ne;
  }

  /* We are done, clean up */
  ierr = clReleaseMemObject(d_coefficients);CHKERRQ(ierr);
  ierr = clReleaseMemObject(d_jacobianInverses);CHKERRQ(ierr);
  ierr = clReleaseMemObject(d_jacobianDeterminants);CHKERRQ(ierr);
  ierr = clReleaseMemObject(d_elemVec);CHKERRQ(ierr);
  ierr = clReleaseKernel(ocl_kernel);CHKERRQ(ierr);
  ierr = clReleaseProgram(ocl_prog);CHKERRQ(ierr);
  ierr = destroyOpenCL(&ocl_env);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
