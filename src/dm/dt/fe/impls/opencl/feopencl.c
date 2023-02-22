#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

#if defined(PETSC_HAVE_OPENCL)

static PetscErrorCode PetscFEDestroy_OpenCL(PetscFE fem)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *)fem->data;

  PetscFunctionBegin;
  PetscCall(clReleaseCommandQueue(ocl->queue_id));
  ocl->queue_id = 0;
  PetscCall(clReleaseContext(ocl->ctx_id));
  ocl->ctx_id = 0;
  PetscCall(PetscFree(ocl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define PetscCallSTR(err) \
    do { \
      PetscCall(err); \
      string_tail += count; \
      PetscCheck(string_tail != end_of_buffer, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Buffer overflow"); \
    } while (0)
enum {
  LAPLACIAN  = 0,
  ELASTICITY = 1
};

/* NOTE: This is now broken for vector problems. Must redo loops to respect vector basis elements */
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
static PetscErrorCode PetscFEOpenCLGenerateIntegrationCode(PetscFE fem, char **string_buffer, PetscInt buffer_length, PetscBool useAux, PetscInt N_bl)
{
  PetscFE_OpenCL  *ocl = (PetscFE_OpenCL *)fem->data;
  PetscQuadrature  q;
  char            *string_tail   = *string_buffer;
  char            *end_of_buffer = *string_buffer + buffer_length;
  char             float_str[] = "float", double_str[] = "double";
  char            *numeric_str    = &(float_str[0]);
  PetscInt         op             = ocl->op;
  PetscBool        useField       = PETSC_FALSE;
  PetscBool        useFieldDer    = PETSC_TRUE;
  PetscBool        useFieldAux    = useAux;
  PetscBool        useFieldDerAux = PETSC_FALSE;
  PetscBool        useF0          = PETSC_TRUE;
  PetscBool        useF1          = PETSC_TRUE;
  const PetscReal *points, *weights;
  PetscTabulation  T;
  PetscInt         dim, qNc, N_b, N_c, N_q, N_t, p, d, b, c;
  size_t           count;

  PetscFunctionBegin;
  PetscCall(PetscFEGetSpatialDimension(fem, &dim));
  PetscCall(PetscFEGetDimension(fem, &N_b));
  PetscCall(PetscFEGetNumComponents(fem, &N_c));
  PetscCall(PetscFEGetQuadrature(fem, &q));
  PetscCall(PetscQuadratureGetData(q, NULL, &qNc, &N_q, &points, &weights));
  PetscCheck(qNc == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %" PetscInt_FMT " components", qNc);
  N_t = N_b * N_c * N_q * N_bl;
  /* Enable device extension for double precision */
  if (ocl->realType == PETSC_DOUBLE) {
    PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                    "#if defined(cl_khr_fp64)\n"
                                    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
                                    "#elif defined(cl_amd_fp64)\n"
                                    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
                                    "#endif\n",
                                    &count));
    numeric_str = &(double_str[0]);
  }
  /* Kernel API */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "\n"
                                  "__kernel void integrateElementQuadrature(int N_cb, __global %s *coefficients, __global %s *coefficientsAux, __global %s *jacobianInverses, __global %s *jacobianDeterminants, __global %s *elemVec)\n"
                                  "{\n",
                                  &count, numeric_str, numeric_str, numeric_str, numeric_str, numeric_str));
  /* Quadrature */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  /* Quadrature points\n"
                                  "   - (x1,y1,x2,y2,...) */\n"
                                  "  const %s points[%d] = {\n",
                                  &count, numeric_str, N_q * dim));
  for (p = 0; p < N_q; ++p) {
    for (d = 0; d < dim; ++d) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, points[p * dim + d]));
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count));
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  /* Quadrature weights\n"
                                  "   - (v1,v2,...) */\n"
                                  "  const %s weights[%d] = {\n",
                                  &count, numeric_str, N_q));
  for (p = 0; p < N_q; ++p) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, weights[p]));
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count));
  /* Basis Functions */
  PetscCall(PetscFEGetCellTabulation(fem, 1, &T));
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  /* Nodal basis function evaluations\n"
                                  "    - basis component is fastest varying, the basis function, then point */\n"
                                  "  const %s Basis[%d] = {\n",
                                  &count, numeric_str, N_q * N_b * N_c));
  for (p = 0; p < N_q; ++p) {
    for (b = 0; b < N_b; ++b) {
      for (c = 0; c < N_c; ++c) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, T->T[0][(p * N_b + b) * N_c + c]));
    }
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count));
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "\n"
                                  "  /* Nodal basis function derivative evaluations,\n"
                                  "      - derivative direction is fastest varying, then basis component, then basis function, then point */\n"
                                  "  const %s%d BasisDerivatives[%d] = {\n",
                                  &count, numeric_str, dim, N_q * N_b * N_c));
  for (p = 0; p < N_q; ++p) {
    for (b = 0; b < N_b; ++b) {
      for (c = 0; c < N_c; ++c) {
        PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "(%s%d)(", &count, numeric_str, dim));
        for (d = 0; d < dim; ++d) {
          if (d > 0) {
            PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, ", %g", &count, T->T[1][((p * N_b + b) * dim + d) * N_c + c]));
          } else {
            PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g", &count, T->T[1][((p * N_b + b) * dim + d) * N_c + c]));
          }
        }
        PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "),\n", &count));
      }
    }
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count));
  /* Sizes */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  const int dim    = %d;                           // The spatial dimension\n"
                                  "  const int N_bl   = %d;                           // The number of concurrent blocks\n"
                                  "  const int N_b    = %d;                           // The number of basis functions\n"
                                  "  const int N_comp = %d;                           // The number of basis function components\n"
                                  "  const int N_bt   = N_b*N_comp;                    // The total number of scalar basis functions\n"
                                  "  const int N_q    = %d;                           // The number of quadrature points\n"
                                  "  const int N_bst  = N_bt*N_q;                      // The block size, LCM(N_b*N_comp, N_q), Notice that a block is not processed simultaneously\n"
                                  "  const int N_t    = N_bst*N_bl;                    // The number of threads, N_bst * N_bl\n"
                                  "  const int N_bc   = N_t/N_comp;                    // The number of cells per batch (N_b*N_q*N_bl)\n"
                                  "  const int N_sbc  = N_bst / (N_q * N_comp);\n"
                                  "  const int N_sqc  = N_bst / N_bt;\n"
                                  "  /*const int N_c    = N_cb * N_bc;*/\n"
                                  "\n"
                                  "  /* Calculated indices */\n"
                                  "  /*const int tidx    = get_local_id(0) + get_local_size(0)*get_local_id(1);*/\n"
                                  "  const int tidx    = get_local_id(0);\n"
                                  "  const int blidx   = tidx / N_bst;                  // Block number for this thread\n"
                                  "  const int bidx    = tidx %% N_bt;                   // Basis function mapped to this thread\n"
                                  "  const int cidx    = tidx %% N_comp;                 // Basis component mapped to this thread\n"
                                  "  const int qidx    = tidx %% N_q;                    // Quadrature point mapped to this thread\n"
                                  "  const int blbidx  = tidx %% N_q + blidx*N_q;        // Cell mapped to this thread in the basis phase\n"
                                  "  const int blqidx  = tidx %% N_b + blidx*N_b;        // Cell mapped to this thread in the quadrature phase\n"
                                  "  const int gidx    = get_group_id(1)*get_num_groups(0) + get_group_id(0);\n"
                                  "  const int Goffset = gidx*N_cb*N_bc;\n",
                                  &count, dim, N_bl, N_b, N_c, N_q));
  /* Local memory */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "\n"
                                  "  /* Quadrature data */\n"
                                  "  %s                w;                   // $w_q$, Quadrature weight at $x_q$\n"
                                  "  __local %s         phi_i[%d];    //[N_bt*N_q];  // $\\phi_i(x_q)$, Value of the basis function $i$ at $x_q$\n"
                                  "  __local %s%d       phiDer_i[%d]; //[N_bt*N_q];  // $\\frac{\\partial\\phi_i(x_q)}{\\partial x_d}$, Value of the derivative of basis function $i$ in direction $x_d$ at $x_q$\n"
                                  "  /* Geometric data */\n"
                                  "  __local %s        detJ[%d]; //[N_t];           // $|J(x_q)|$, Jacobian determinant at $x_q$\n"
                                  "  __local %s        invJ[%d];//[N_t*dim*dim];   // $J^{-1}(x_q)$, Jacobian inverse at $x_q$\n",
                                  &count, numeric_str, numeric_str, N_b * N_c * N_q, numeric_str, dim, N_b * N_c * N_q, numeric_str, N_t, numeric_str, N_t * dim * dim));
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  /* FEM data */\n"
                                  "  __local %s        u_i[%d]; //[N_t*N_bt];       // Coefficients $u_i$ of the field $u|_{\\mathcal{T}} = \\sum_i u_i \\phi_i$\n",
                                  &count, numeric_str, N_t * N_b * N_c));
  if (useAux) {
    PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "  __local %s        a_i[%d]; //[N_t];            // Coefficients $a_i$ of the auxiliary field $a|_{\\mathcal{T}} = \\sum_i a_i \\phi^R_i$\n", &count, numeric_str, N_t));
  }
  if (useF0) {
    PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                    "  /* Intermediate calculations */\n"
                                    "  __local %s         f_0[%d]; //[N_t*N_sqc];      // $f_0(u(x_q), \\nabla u(x_q)) |J(x_q)| w_q$\n",
                                    &count, numeric_str, N_t * N_q));
  }
  if (useF1) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "  __local %s%d       f_1[%d]; //[N_t*N_sqc];      // $f_1(u(x_q), \\nabla u(x_q)) |J(x_q)| w_q$\n", &count, numeric_str, dim, N_t * N_q));
  /* TODO: If using elasticity, put in mu/lambda coefficients */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  /* Output data */\n"
                                  "  %s                e_i;                 // Coefficient $e_i$ of the residual\n\n",
                                  &count, numeric_str));
  /* One-time loads */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "  /* These should be generated inline */\n"
                                  "  /* Load quadrature weights */\n"
                                  "  w = weights[qidx];\n"
                                  "  /* Load basis tabulation \\phi_i for this cell */\n"
                                  "  if (tidx < N_bt*N_q) {\n"
                                  "    phi_i[tidx]    = Basis[tidx];\n"
                                  "    phiDer_i[tidx] = BasisDerivatives[tidx];\n"
                                  "  }\n\n",
                                  &count));
  /* Batch loads */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
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
                                  "      u_i[offset+tidx] = coefficients[(Goffset*N_bt)+batch*N_t*N_b+offset+tidx];\n"
                                  "    }\n",
                                  &count));
  if (useAux) {
    PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                    "    /* Load coefficients a_i for this cell */\n"
                                    "    /* TODO: This should not be N_t here, it should be N_bc*N_comp_aux */\n"
                                    "    a_i[tidx] = coefficientsAux[Goffset+batch*N_t+tidx];\n",
                                    &count));
  }
  /* Quadrature phase */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                                  "\n"
                                  "    /* Map coefficients to values at quadrature points */\n"
                                  "    for (int c = 0; c < N_sqc; ++c) {\n"
                                  "      const int cell          = c*N_bl*N_b + blqidx;\n"
                                  "      const int fidx          = (cell*N_q + qidx)*N_comp + cidx;\n",
                                  &count));
  if (useField) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      %s  u[%d]; //[N_comp];     // $u(x_q)$, Value of the field at $x_q$\n", &count, numeric_str, N_c));
  if (useFieldDer) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      %s%d   gradU[%d]; //[N_comp]; // $\\nabla u(x_q)$, Value of the field gradient at $x_q$\n", &count, numeric_str, dim, N_c));
  if (useFieldAux) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      %s  a[%d]; //[1];     // $a(x_q)$, Value of the auxiliary fields at $x_q$\n", &count, numeric_str, 1));
  if (useFieldDerAux) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      %s%d   gradA[%d]; //[1]; // $\\nabla a(x_q)$, Value of the auxiliary field gradient at $x_q$\n", &count, numeric_str, dim, 1));
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "\n"
                                  "      for (int comp = 0; comp < N_comp; ++comp) {\n",
                                  &count));
  if (useField) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        u[comp] = 0.0;\n", &count));
  if (useFieldDer) {
    switch (dim) {
    case 1:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        gradU[comp].x = 0.0;\n", &count));
      break;
    case 2:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        gradU[comp].x = 0.0; gradU[comp].y = 0.0;\n", &count));
      break;
    case 3:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        gradU[comp].x = 0.0; gradU[comp].y = 0.0; gradU[comp].z = 0.0;\n", &count));
      break;
    }
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      }\n", &count));
  if (useFieldAux) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      a[0] = 0.0;\n", &count));
  if (useFieldDerAux) {
    switch (dim) {
    case 1:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      gradA[0].x = 0.0;\n", &count));
      break;
    case 2:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      gradA[0].x = 0.0; gradA[0].y = 0.0;\n", &count));
      break;
    case 3:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      gradA[0].x = 0.0; gradA[0].y = 0.0; gradA[0].z = 0.0;\n", &count));
      break;
    }
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "      /* Get field and derivatives at this quadrature point */\n"
                                  "      for (int i = 0; i < N_b; ++i) {\n"
                                  "        for (int comp = 0; comp < N_comp; ++comp) {\n"
                                  "          const int b    = i*N_comp+comp;\n"
                                  "          const int pidx = qidx*N_bt + b;\n"
                                  "          const int uidx = cell*N_bt + b;\n"
                                  "          %s%d   realSpaceDer;\n\n",
                                  &count, numeric_str, dim));
  if (useField) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "          u[comp] += u_i[uidx]*phi_i[pidx];\n", &count));
  if (useFieldDer) {
    switch (dim) {
    case 2:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                      "          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;\n"
                                      "          gradU[comp].x += u_i[uidx]*realSpaceDer.x;\n"
                                      "          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;\n"
                                      "          gradU[comp].y += u_i[uidx]*realSpaceDer.y;\n",
                                      &count));
      break;
    case 3:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                      "          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;\n"
                                      "          gradU[comp].x += u_i[uidx]*realSpaceDer.x;\n"
                                      "          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;\n"
                                      "          gradU[comp].y += u_i[uidx]*realSpaceDer.y;\n"
                                      "          realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;\n"
                                      "          gradU[comp].z += u_i[uidx]*realSpaceDer.z;\n",
                                      &count));
      break;
    }
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "        }\n"
                                  "      }\n",
                                  &count));
  if (useFieldAux) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "          a[0] += a_i[cell];\n", &count));
  /* Calculate residual at quadrature points: Should be generated by an weak form egine */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      /* Process values at quadrature points */\n", &count));
  switch (op) {
  case LAPLACIAN:
    if (useF0) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_0[fidx] = 4.0;\n", &count));
    if (useF1) {
      if (useAux) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx] = a[0]*gradU[cidx];\n", &count));
      else PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx] = gradU[cidx];\n", &count));
    }
    break;
  case ELASTICITY:
    if (useF0) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_0[fidx] = 4.0;\n", &count));
    if (useF1) {
      switch (dim) {
      case 2:
        PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                        "      switch (cidx) {\n"
                                        "      case 0:\n"
                                        "        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[0].x + gradU[0].x);\n"
                                        "        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[0].y + gradU[1].x);\n"
                                        "        break;\n"
                                        "      case 1:\n"
                                        "        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[1].x + gradU[0].y);\n"
                                        "        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[1].y + gradU[1].y);\n"
                                        "      }\n",
                                        &count));
        break;
      case 3:
        PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                        "      switch (cidx) {\n"
                                        "      case 0:\n"
                                        "        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[0].x + gradU[0].x);\n"
                                        "        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[0].y + gradU[1].x);\n"
                                        "        f_1[fidx].z = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[0].z + gradU[2].x);\n"
                                        "        break;\n"
                                        "      case 1:\n"
                                        "        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[1].x + gradU[0].y);\n"
                                        "        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[1].y + gradU[1].y);\n"
                                        "        f_1[fidx].z = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[1].y + gradU[2].y);\n"
                                        "        break;\n"
                                        "      case 2:\n"
                                        "        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[2].x + gradU[0].z);\n"
                                        "        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[2].y + gradU[1].z);\n"
                                        "        f_1[fidx].z = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[2].y + gradU[2].z);\n"
                                        "      }\n",
                                        &count));
        break;
      }
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PDE operator %d is not supported", op);
  }
  if (useF0) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_0[fidx] *= detJ[cell]*w;\n", &count));
  if (useF1) {
    switch (dim) {
    case 1:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx].x *= detJ[cell]*w;\n", &count));
      break;
    case 2:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w;\n", &count));
      break;
    case 3:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w; f_1[fidx].z *= detJ[cell]*w;\n", &count));
      break;
    }
  }
  /* Thread transpose */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "    }\n\n"
                                  "    /* ==== TRANSPOSE THREADS ==== */\n"
                                  "    barrier(CLK_LOCAL_MEM_FENCE);\n\n",
                                  &count));
  /* Basis phase */
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "    /* Map values at quadrature points to coefficients */\n"
                                  "    for (int c = 0; c < N_sbc; ++c) {\n"
                                  "      const int cell = c*N_bl*N_q + blbidx; /* Cell number in batch */\n"
                                  "\n"
                                  "      e_i = 0.0;\n"
                                  "      for (int q = 0; q < N_q; ++q) {\n"
                                  "        const int pidx = q*N_bt + bidx;\n"
                                  "        const int fidx = (cell*N_q + q)*N_comp + cidx;\n"
                                  "        %s%d   realSpaceDer;\n\n",
                                  &count, numeric_str, dim));

  if (useF0) PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        e_i += phi_i[pidx]*f_0[fidx];\n", &count));
  if (useF1) {
    switch (dim) {
    case 2:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                      "        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;\n"
                                      "        e_i           += realSpaceDer.x*f_1[fidx].x;\n"
                                      "        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;\n"
                                      "        e_i           += realSpaceDer.y*f_1[fidx].y;\n",
                                      &count));
      break;
    case 3:
      PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                      "        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;\n"
                                      "        e_i           += realSpaceDer.x*f_1[fidx].x;\n"
                                      "        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;\n"
                                      "        e_i           += realSpaceDer.y*f_1[fidx].y;\n"
                                      "        realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;\n"
                                      "        e_i           += realSpaceDer.z*f_1[fidx].z;\n",
                                      &count));
      break;
    }
  }
  PetscCallSTR(PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
                                  "      }\n"
                                  "      /* Write element vector for N_{cbc} cells at a time */\n"
                                  "      elemVec[(Goffset + batch*N_bc + c*N_bl*N_q)*N_bt + tidx] = e_i;\n"
                                  "    }\n"
                                  "    /* ==== Could do one write per batch ==== */\n"
                                  "  }\n"
                                  "  return;\n"
                                  "}\n",
                                  &count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEOpenCLGetIntegrationKernel(PetscFE fem, PetscBool useAux, cl_program *ocl_prog, cl_kernel *ocl_kernel)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *)fem->data;
  PetscInt        dim, N_bl;
  PetscBool       flg;
  char           *buffer;
  size_t          len;
  char            errMsg[8192];
  cl_int          err;

  PetscFunctionBegin;
  PetscCall(PetscFEGetSpatialDimension(fem, &dim));
  PetscCall(PetscMalloc1(8192, &buffer));
  PetscCall(PetscFEGetTileSizes(fem, NULL, &N_bl, NULL, NULL));
  PetscCall(PetscFEOpenCLGenerateIntegrationCode(fem, &buffer, 8192, useAux, N_bl));
  PetscCall(PetscOptionsHasName(((PetscObject)fem)->options, ((PetscObject)fem)->prefix, "-petscfe_opencl_kernel_print", &flg));
  if (flg) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)fem), "OpenCL FE Integration Kernel:\n%s\n", buffer));
  PetscCall(PetscStrlen(buffer, &len));
  *ocl_prog = clCreateProgramWithSource(ocl->ctx_id, 1, (const char **)&buffer, &len, &err);
  PetscCall(err);
  err = clBuildProgram(*ocl_prog, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    err = clGetProgramBuildInfo(*ocl_prog, ocl->dev_id, CL_PROGRAM_BUILD_LOG, 8192 * sizeof(char), &errMsg, NULL);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Build failed! Log:\n %s", errMsg);
  }
  PetscCall(PetscFree(buffer));
  *ocl_kernel = clCreateKernel(*ocl_prog, "integrateElementQuadrature", &err);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEOpenCLCalculateGrid(PetscFE fem, PetscInt N, PetscInt blockSize, size_t *x, size_t *y, size_t *z)
{
  const PetscInt Nblocks = N / blockSize;

  PetscFunctionBegin;
  PetscCheck(!(N % blockSize), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid block size %d for %d elements", blockSize, N);
  *z = 1;
  *y = 1;
  for (*x = (size_t)(PetscSqrtReal(Nblocks) + 0.5); *x > 0; --*x) {
    *y = Nblocks / *x;
    if (*x * *y == (size_t)Nblocks) break;
  }
  PetscCheck(*x * *y == (size_t)Nblocks, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Could not find partition for %" PetscInt_FMT " with block size %" PetscInt_FMT, N, blockSize);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEOpenCLLogResidual(PetscFE fem, PetscLogDouble time, PetscLogDouble flops)
{
  PetscFE_OpenCL   *ocl = (PetscFE_OpenCL *)fem->data;
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  /* Log performance info */
  eventLog->eventInfo[ocl->residualEvent].count++;
  eventLog->eventInfo[ocl->residualEvent].time += time;
  eventLog->eventInfo[ocl->residualEvent].flops += flops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEIntegrateResidual_OpenCL(PetscDS prob, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  /* Nbc = batchSize */
  PetscFE         fem;
  PetscFE_OpenCL *ocl;
  PetscPointFunc  f0_func;
  PetscPointFunc  f1_func;
  PetscQuadrature q;
  PetscInt        dim, qNc;
  PetscInt        N_b;    /* The number of basis functions */
  PetscInt        N_comp; /* The number of basis function components */
  PetscInt        N_bt;   /* The total number of scalar basis functions */
  PetscInt        N_q;    /* The number of quadrature points */
  PetscInt        N_bst;  /* The block size, LCM(N_bt, N_q), Notice that a block is not process simultaneously */
  PetscInt        N_t;    /* The number of threads, N_bst * N_bl */
  PetscInt        N_bl;   /* The number of blocks */
  PetscInt        N_bc;   /* The batch size, N_bl*N_q*N_b */
  PetscInt        N_cb;   /* The number of batches */
  const PetscInt  field = key.field;
  PetscInt        numFlops, f0Flops = 0, f1Flops = 0;
  PetscBool       useAux      = probAux ? PETSC_TRUE : PETSC_FALSE;
  PetscBool       useField    = PETSC_FALSE;
  PetscBool       useFieldDer = PETSC_TRUE;
  PetscBool       useF0       = PETSC_TRUE;
  PetscBool       useF1       = PETSC_TRUE;
  /* OpenCL variables */
  cl_program       ocl_prog;
  cl_kernel        ocl_kernel;
  cl_event         ocl_ev;   /* The event for tracking kernel execution */
  cl_ulong         ns_start; /* Nanoseconds counter on GPU at kernel start */
  cl_ulong         ns_end;   /* Nanoseconds counter on GPU at kernel stop */
  cl_mem           o_jacobianInverses, o_jacobianDeterminants;
  cl_mem           o_coefficients, o_coefficientsAux, o_elemVec;
  float           *f_coeff = NULL, *f_coeffAux = NULL, *f_invJ = NULL, *f_detJ = NULL;
  double          *d_coeff = NULL, *d_coeffAux = NULL, *d_invJ = NULL, *d_detJ = NULL;
  PetscReal       *r_invJ = NULL, *r_detJ = NULL;
  void            *oclCoeff, *oclCoeffAux, *oclInvJ, *oclDetJ;
  size_t           local_work_size[3], global_work_size[3];
  size_t           realSize, x, y, z;
  const PetscReal *points, *weights;
  int              err;

  PetscFunctionBegin;
  PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *)&fem));
  ocl = (PetscFE_OpenCL *)fem->data;
  if (!Ne) {
    PetscCall(PetscFEOpenCLLogResidual(fem, 0.0, 0.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscFEGetSpatialDimension(fem, &dim));
  PetscCall(PetscFEGetQuadrature(fem, &q));
  PetscCall(PetscQuadratureGetData(q, NULL, &qNc, &N_q, &points, &weights));
  PetscCheck(qNc == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %" PetscInt_FMT " components", qNc);
  PetscCall(PetscFEGetDimension(fem, &N_b));
  PetscCall(PetscFEGetNumComponents(fem, &N_comp));
  PetscCall(PetscDSGetResidual(prob, field, &f0_func, &f1_func));
  PetscCall(PetscFEGetTileSizes(fem, NULL, &N_bl, &N_bc, &N_cb));
  N_bt  = N_b * N_comp;
  N_bst = N_bt * N_q;
  N_t   = N_bst * N_bl;
  PetscCheck(N_bc * N_comp == N_t, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of threads %d should be %d * %d", N_t, N_bc, N_comp);
  /* Calculate layout */
  if (Ne % (N_cb * N_bc)) { /* Remainder cells */
    PetscCall(PetscFEIntegrateResidual_Basic(prob, key, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscFEOpenCLCalculateGrid(fem, Ne, N_cb * N_bc, &x, &y, &z));
  local_work_size[0]  = N_bc * N_comp;
  local_work_size[1]  = 1;
  local_work_size[2]  = 1;
  global_work_size[0] = x * local_work_size[0];
  global_work_size[1] = y * local_work_size[1];
  global_work_size[2] = z * local_work_size[2];
  PetscCall(PetscInfo(fem, "GPU layout grid(%zu,%zu,%zu) block(%zu,%zu,%zu) with %d batches\n", x, y, z, local_work_size[0], local_work_size[1], local_work_size[2], N_cb));
  PetscCall(PetscInfo(fem, " N_t: %d, N_cb: %d\n", N_t, N_cb));
  /* Generate code */
  if (probAux) {
    PetscSpace P;
    PetscInt   NfAux, order, f;

    PetscCall(PetscDSGetNumFields(probAux, &NfAux));
    for (f = 0; f < NfAux; ++f) {
      PetscFE feAux;

      PetscCall(PetscDSGetDiscretization(probAux, f, (PetscObject *)&feAux));
      PetscCall(PetscFEGetBasisSpace(feAux, &P));
      PetscCall(PetscSpaceGetDegree(P, &order, NULL));
      PetscCheck(order <= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Can only handle P0 coefficient fields");
    }
  }
  PetscCall(PetscFEOpenCLGetIntegrationKernel(fem, useAux, &ocl_prog, &ocl_kernel));
  /* Create buffers on the device and send data over */
  PetscCall(PetscDataTypeGetSize(ocl->realType, &realSize));
  PetscCheck(cgeom->numPoints <= 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support affine geometry for OpenCL integration right now");
  if (sizeof(PetscReal) != realSize) {
    switch (ocl->realType) {
    case PETSC_FLOAT: {
      PetscInt c, b, d;

      PetscCall(PetscMalloc4(Ne * N_bt, &f_coeff, Ne, &f_coeffAux, Ne * dim * dim, &f_invJ, Ne, &f_detJ));
      for (c = 0; c < Ne; ++c) {
        f_detJ[c] = (float)cgeom->detJ[c];
        for (d = 0; d < dim * dim; ++d) f_invJ[c * dim * dim + d] = (float)cgeom->invJ[c * dim * dim + d];
        for (b = 0; b < N_bt; ++b) f_coeff[c * N_bt + b] = (float)coefficients[c * N_bt + b];
      }
      if (coefficientsAux) { /* Assume P0 */
        for (c = 0; c < Ne; ++c) f_coeffAux[c] = (float)coefficientsAux[c];
      }
      oclCoeff = (void *)f_coeff;
      if (coefficientsAux) {
        oclCoeffAux = (void *)f_coeffAux;
      } else {
        oclCoeffAux = NULL;
      }
      oclInvJ = (void *)f_invJ;
      oclDetJ = (void *)f_detJ;
    } break;
    case PETSC_DOUBLE: {
      PetscInt c, b, d;

      PetscCall(PetscMalloc4(Ne * N_bt, &d_coeff, Ne, &d_coeffAux, Ne * dim * dim, &d_invJ, Ne, &d_detJ));
      for (c = 0; c < Ne; ++c) {
        d_detJ[c] = (double)cgeom->detJ[c];
        for (d = 0; d < dim * dim; ++d) d_invJ[c * dim * dim + d] = (double)cgeom->invJ[c * dim * dim + d];
        for (b = 0; b < N_bt; ++b) d_coeff[c * N_bt + b] = (double)coefficients[c * N_bt + b];
      }
      if (coefficientsAux) { /* Assume P0 */
        for (c = 0; c < Ne; ++c) d_coeffAux[c] = (double)coefficientsAux[c];
      }
      oclCoeff = (void *)d_coeff;
      if (coefficientsAux) {
        oclCoeffAux = (void *)d_coeffAux;
      } else {
        oclCoeffAux = NULL;
      }
      oclInvJ = (void *)d_invJ;
      oclDetJ = (void *)d_detJ;
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported PETSc type %d", ocl->realType);
    }
  } else {
    PetscInt c, d;

    PetscCall(PetscMalloc2(Ne * dim * dim, &r_invJ, Ne, &r_detJ));
    for (c = 0; c < Ne; ++c) {
      r_detJ[c] = cgeom->detJ[c];
      for (d = 0; d < dim * dim; ++d) r_invJ[c * dim * dim + d] = cgeom->invJ[c * dim * dim + d];
    }
    oclCoeff    = (void *)coefficients;
    oclCoeffAux = (void *)coefficientsAux;
    oclInvJ     = (void *)r_invJ;
    oclDetJ     = (void *)r_detJ;
  }
  o_coefficients = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne * N_bt * realSize, oclCoeff, &err);
  if (coefficientsAux) {
    o_coefficientsAux = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne * realSize, oclCoeffAux, &err);
  } else {
    o_coefficientsAux = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY, Ne * realSize, oclCoeffAux, &err);
  }
  o_jacobianInverses     = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne * dim * dim * realSize, oclInvJ, &err);
  o_jacobianDeterminants = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne * realSize, oclDetJ, &err);
  o_elemVec              = clCreateBuffer(ocl->ctx_id, CL_MEM_WRITE_ONLY, Ne * N_bt * realSize, NULL, &err);
  /* Kernel launch */
  PetscCall(clSetKernelArg(ocl_kernel, 0, sizeof(cl_int), (void *)&N_cb));
  PetscCall(clSetKernelArg(ocl_kernel, 1, sizeof(cl_mem), (void *)&o_coefficients));
  PetscCall(clSetKernelArg(ocl_kernel, 2, sizeof(cl_mem), (void *)&o_coefficientsAux));
  PetscCall(clSetKernelArg(ocl_kernel, 3, sizeof(cl_mem), (void *)&o_jacobianInverses));
  PetscCall(clSetKernelArg(ocl_kernel, 4, sizeof(cl_mem), (void *)&o_jacobianDeterminants));
  PetscCall(clSetKernelArg(ocl_kernel, 5, sizeof(cl_mem), (void *)&o_elemVec));
  PetscCall(clEnqueueNDRangeKernel(ocl->queue_id, ocl_kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &ocl_ev));
  /* Read data back from device */
  if (sizeof(PetscReal) != realSize) {
    switch (ocl->realType) {
    case PETSC_FLOAT: {
      float   *elem;
      PetscInt c, b;

      PetscCall(PetscFree4(f_coeff, f_coeffAux, f_invJ, f_detJ));
      PetscCall(PetscMalloc1(Ne * N_bt, &elem));
      PetscCall(clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne * N_bt * realSize, elem, 0, NULL, NULL));
      for (c = 0; c < Ne; ++c) {
        for (b = 0; b < N_bt; ++b) elemVec[c * N_bt + b] = (PetscScalar)elem[c * N_bt + b];
      }
      PetscCall(PetscFree(elem));
    } break;
    case PETSC_DOUBLE: {
      double  *elem;
      PetscInt c, b;

      PetscCall(PetscFree4(d_coeff, d_coeffAux, d_invJ, d_detJ));
      PetscCall(PetscMalloc1(Ne * N_bt, &elem));
      PetscCall(clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne * N_bt * realSize, elem, 0, NULL, NULL));
      for (c = 0; c < Ne; ++c) {
        for (b = 0; b < N_bt; ++b) elemVec[c * N_bt + b] = (PetscScalar)elem[c * N_bt + b];
      }
      PetscCall(PetscFree(elem));
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported PETSc type %d", ocl->realType);
    }
  } else {
    PetscCall(PetscFree2(r_invJ, r_detJ));
    PetscCall(clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne * N_bt * realSize, elemVec, 0, NULL, NULL));
  }
  /* Log performance */
  PetscCall(clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ns_start, NULL));
  PetscCall(clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ns_end, NULL));
  f0Flops = 0;
  switch (ocl->op) {
  case LAPLACIAN:
    f1Flops = useAux ? dim : 0;
    break;
  case ELASTICITY:
    f1Flops = 2 * dim * dim;
    break;
  }
  numFlops = Ne * (N_q * (N_b * N_comp * ((useField ? 2 : 0) + (useFieldDer ? 2 * dim * (dim + 1) : 0))
                          /*+
       N_ba*N_compa*((useFieldAux ? 2 : 0) + (useFieldDerAux ? 2*dim*(dim + 1) : 0))*/
                          + N_comp * ((useF0 ? f0Flops + 2 : 0) + (useF1 ? f1Flops + 2 * dim : 0))) +
                   N_b * ((useF0 ? 2 : 0) + (useF1 ? 2 * dim * (dim + 1) : 0)));
  PetscCall(PetscFEOpenCLLogResidual(fem, (ns_end - ns_start) * 1.0e-9, numFlops));
  /* Cleanup */
  PetscCall(clReleaseMemObject(o_coefficients));
  PetscCall(clReleaseMemObject(o_coefficientsAux));
  PetscCall(clReleaseMemObject(o_jacobianInverses));
  PetscCall(clReleaseMemObject(o_jacobianDeterminants));
  PetscCall(clReleaseMemObject(o_elemVec));
  PetscCall(clReleaseKernel(ocl_kernel));
  PetscCall(clReleaseProgram(ocl_prog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscFESetUp_Basic(PetscFE);
PETSC_INTERN PetscErrorCode PetscFECreateTabulation_Basic(PetscFE, PetscInt, const PetscReal[], PetscInt, PetscTabulation);

static PetscErrorCode PetscFEInitialize_OpenCL(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Basic;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_OpenCL;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->createtabulation        = PetscFECreateTabulation_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_OpenCL;
  fem->ops->integratebdresidual     = NULL /* PetscFEIntegrateBdResidual_OpenCL */;
  fem->ops->integratejacobianaction = NULL /* PetscFEIntegrateJacobianAction_OpenCL */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCFEOPENCL = "opencl" - A `PetscFEType` that integrates using a vectorized OpenCL implementation

  Level: intermediate

.seealso: `PetscFEType`, `PetscFECreate()`, `PetscFESetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_OpenCL(PetscFE fem)
{
  PetscFE_OpenCL *ocl;
  cl_uint         num_platforms;
  cl_platform_id  platform_ids[42];
  cl_uint         num_devices;
  cl_device_id    device_ids[42];
  cl_int          err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscCall(PetscNew(&ocl));
  fem->data = ocl;

  /* Init Platform */
  PetscCall(clGetPlatformIDs(42, platform_ids, &num_platforms));
  PetscCheck(num_platforms, PetscObjectComm((PetscObject)fem), PETSC_ERR_SUP, "No OpenCL platform found.");
  ocl->pf_id = platform_ids[0];
  /* Init Device */
  PetscCall(clGetDeviceIDs(ocl->pf_id, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices));
  PetscCheck(num_devices, PetscObjectComm((PetscObject)fem), PETSC_ERR_SUP, "No OpenCL device found.");
  ocl->dev_id = device_ids[0];
  /* Create context with one command queue */
  ocl->ctx_id = clCreateContext(0, 1, &(ocl->dev_id), NULL, NULL, &err);
  PetscCall(err);
  ocl->queue_id = clCreateCommandQueue(ocl->ctx_id, ocl->dev_id, CL_QUEUE_PROFILING_ENABLE, &err);
  PetscCall(err);
  /* Types */
  ocl->realType = PETSC_FLOAT;
  /* Register events */
  PetscCall(PetscLogEventRegister("OpenCL FEResidual", PETSCFE_CLASSID, &ocl->residualEvent));
  /* Equation handling */
  ocl->op = LAPLACIAN;

  PetscCall(PetscFEInitialize_OpenCL(fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEOpenCLSetRealType - Set the scalar type for running on the OpenCL accelerator

  Input Parameters:
+ fem      - The `PetscFE`
- realType - The scalar type

  Level: developer

.seealso: `PetscFE`, `PetscFEOpenCLGetRealType()`
@*/
PetscErrorCode PetscFEOpenCLSetRealType(PetscFE fem, PetscDataType realType)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *)fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ocl->realType = realType;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEOpenCLGetRealType - Get the scalar type for running on the OpenCL accelerator

  Input Parameter:
. fem - The `PetscFE`

  Output Parameter:
. realType - The scalar type

  Level: developer

.seealso: `PetscFE`, `PetscFEOpenCLSetRealType()`
@*/
PetscErrorCode PetscFEOpenCLGetRealType(PetscFE fem, PetscDataType *realType)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *)fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(realType, 2);
  *realType = ocl->realType;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_HAVE_OPENCL */
