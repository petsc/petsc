static char help[] = "This example illustrates the use of PCBDDC/FETI-DP and its customization.\n\n\
Discrete system: 1D, 2D or 3D laplacian, discretized with spectral elements.\n\
Spectral degree can be specified by passing values to -p option.\n\
Global problem either with dirichlet boundary conditions on one side or in the pure neumann case (depending on runtime parameters).\n\
Domain is [-nex,nex]x[-ney,ney]x[-nez,nez]: ne_ number of elements in _ direction.\n\
Example usage: \n\
1D: mpiexec -n 4 ex59 -nex 7\n\
2D: mpiexec -n 4 ex59 -npx 2 -npy 2 -nex 2 -ney 2\n\
3D: mpiexec -n 4 ex59 -npx 2 -npy 2 -npz 1 -nex 2 -ney 2 -nez 1\n\
Subdomain decomposition can be specified with -np_ parameters.\n\
Dirichlet boundaries on one side by default:\n\
it does not iterate on dirichlet nodes by default: if -usezerorows is passed in, it also iterates on Dirichlet nodes.\n\
Pure Neumann case can be requested by passing in -pureneumann.\n\
In the latter case, in order to avoid runtime errors during factorization, please specify also -coarse_redundant_pc_factor_zeropivot 0\n\n";

#include <petscksp.h>
#include <petscpc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscblaslapack.h>
#define DEBUG 0

/* structure holding domain data */
typedef struct {
  /* communicator */
  MPI_Comm gcomm;
  /* space dimension */
  PetscInt dim;
  /* spectral degree */
  PetscInt p;
  /* subdomains per dimension */
  PetscInt npx, npy, npz;
  /* subdomain index in cartesian dimensions */
  PetscInt ipx, ipy, ipz;
  /* elements per dimension */
  PetscInt nex, ney, nez;
  /* local elements per dimension */
  PetscInt nex_l, ney_l, nez_l;
  /* global number of dofs per dimension */
  PetscInt xm, ym, zm;
  /* local number of dofs per dimension */
  PetscInt xm_l, ym_l, zm_l;
  /* starting global indexes for subdomain in lexicographic ordering */
  PetscInt startx, starty, startz;
  /* pure Neumann problem */
  PetscBool pure_neumann;
  /* Dirichlet BC implementation */
  PetscBool DBC_zerorows;
  /* Scaling factor for subdomain */
  PetscScalar scalingfactor;
  PetscBool   testkspfetidp;
} DomainData;

/* structure holding GLL data */
typedef struct {
  /* GLL nodes */
  PetscReal *zGL;
  /* GLL weights */
  PetscScalar *rhoGL;
  /* aux_mat */
  PetscScalar **A;
  /* Element matrix */
  Mat elem_mat;
} GLLData;

static PetscErrorCode BuildCSRGraph(DomainData dd, PetscInt **xadj, PetscInt **adjncy)
{
  PetscInt *xadj_temp, *adjncy_temp;
  PetscInt  i, j, k, ii, jj, kk, iindex, count_adj;
  PetscInt  istart_csr, iend_csr, jstart_csr, jend_csr, kstart_csr, kend_csr;
  PetscBool internal_node;

  /* first count dimension of adjncy */
  PetscFunctionBeginUser;
  count_adj = 0;
  for (k = 0; k < dd.zm_l; k++) {
    internal_node = PETSC_TRUE;
    kstart_csr    = k > 0 ? k - 1 : k;
    kend_csr      = k < dd.zm_l - 1 ? k + 2 : k + 1;
    if (k == 0 || k == dd.zm_l - 1) {
      internal_node = PETSC_FALSE;
      kstart_csr    = k;
      kend_csr      = k + 1;
    }
    for (j = 0; j < dd.ym_l; j++) {
      jstart_csr = j > 0 ? j - 1 : j;
      jend_csr   = j < dd.ym_l - 1 ? j + 2 : j + 1;
      if (j == 0 || j == dd.ym_l - 1) {
        internal_node = PETSC_FALSE;
        jstart_csr    = j;
        jend_csr      = j + 1;
      }
      for (i = 0; i < dd.xm_l; i++) {
        istart_csr = i > 0 ? i - 1 : i;
        iend_csr   = i < dd.xm_l - 1 ? i + 2 : i + 1;
        if (i == 0 || i == dd.xm_l - 1) {
          internal_node = PETSC_FALSE;
          istart_csr    = i;
          iend_csr      = i + 1;
        }
        if (internal_node) {
          istart_csr = i;
          iend_csr   = i + 1;
          jstart_csr = j;
          jend_csr   = j + 1;
          kstart_csr = k;
          kend_csr   = k + 1;
        }
        for (kk = kstart_csr; kk < kend_csr; kk++) {
          for (jj = jstart_csr; jj < jend_csr; jj++) {
            for (ii = istart_csr; ii < iend_csr; ii++) count_adj = count_adj + 1;
          }
        }
      }
    }
  }
  PetscCall(PetscMalloc1(dd.xm_l * dd.ym_l * dd.zm_l + 1, &xadj_temp));
  PetscCall(PetscMalloc1(count_adj, &adjncy_temp));

  /* now fill CSR data structure */
  count_adj = 0;
  for (k = 0; k < dd.zm_l; k++) {
    internal_node = PETSC_TRUE;
    kstart_csr    = k > 0 ? k - 1 : k;
    kend_csr      = k < dd.zm_l - 1 ? k + 2 : k + 1;
    if (k == 0 || k == dd.zm_l - 1) {
      internal_node = PETSC_FALSE;
      kstart_csr    = k;
      kend_csr      = k + 1;
    }
    for (j = 0; j < dd.ym_l; j++) {
      jstart_csr = j > 0 ? j - 1 : j;
      jend_csr   = j < dd.ym_l - 1 ? j + 2 : j + 1;
      if (j == 0 || j == dd.ym_l - 1) {
        internal_node = PETSC_FALSE;
        jstart_csr    = j;
        jend_csr      = j + 1;
      }
      for (i = 0; i < dd.xm_l; i++) {
        istart_csr = i > 0 ? i - 1 : i;
        iend_csr   = i < dd.xm_l - 1 ? i + 2 : i + 1;
        if (i == 0 || i == dd.xm_l - 1) {
          internal_node = PETSC_FALSE;
          istart_csr    = i;
          iend_csr      = i + 1;
        }
        iindex            = k * dd.xm_l * dd.ym_l + j * dd.xm_l + i;
        xadj_temp[iindex] = count_adj;
        if (internal_node) {
          istart_csr = i;
          iend_csr   = i + 1;
          jstart_csr = j;
          jend_csr   = j + 1;
          kstart_csr = k;
          kend_csr   = k + 1;
        }
        for (kk = kstart_csr; kk < kend_csr; kk++) {
          for (jj = jstart_csr; jj < jend_csr; jj++) {
            for (ii = istart_csr; ii < iend_csr; ii++) {
              iindex = kk * dd.xm_l * dd.ym_l + jj * dd.xm_l + ii;

              adjncy_temp[count_adj] = iindex;
              count_adj              = count_adj + 1;
            }
          }
        }
      }
    }
  }
  xadj_temp[dd.xm_l * dd.ym_l * dd.zm_l] = count_adj;

  *xadj   = xadj_temp;
  *adjncy = adjncy_temp;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeSpecialBoundaryIndices(DomainData dd, IS *dirichlet, IS *neumann)
{
  IS         temp_dirichlet = 0, temp_neumann = 0;
  PetscInt   localsize, i, j, k, *indices;
  PetscBool *touched;

  PetscFunctionBeginUser;
  localsize = dd.xm_l * dd.ym_l * dd.zm_l;

  PetscCall(PetscMalloc1(localsize, &indices));
  PetscCall(PetscMalloc1(localsize, &touched));
  for (i = 0; i < localsize; i++) touched[i] = PETSC_FALSE;

  if (dirichlet) {
    i = 0;
    /* west boundary */
    if (dd.ipx == 0) {
      for (k = 0; k < dd.zm_l; k++) {
        for (j = 0; j < dd.ym_l; j++) {
          indices[i]          = k * dd.ym_l * dd.xm_l + j * dd.xm_l;
          touched[indices[i]] = PETSC_TRUE;
          i++;
        }
      }
    }
    PetscCall(ISCreateGeneral(dd.gcomm, i, indices, PETSC_COPY_VALUES, &temp_dirichlet));
  }
  if (neumann) {
    i = 0;
    /* west boundary */
    if (dd.ipx == 0) {
      for (k = 0; k < dd.zm_l; k++) {
        for (j = 0; j < dd.ym_l; j++) {
          indices[i] = k * dd.ym_l * dd.xm_l + j * dd.xm_l;
          if (!touched[indices[i]]) {
            touched[indices[i]] = PETSC_TRUE;
            i++;
          }
        }
      }
    }
    /* east boundary */
    if (dd.ipx == dd.npx - 1) {
      for (k = 0; k < dd.zm_l; k++) {
        for (j = 0; j < dd.ym_l; j++) {
          indices[i] = k * dd.ym_l * dd.xm_l + j * dd.xm_l + dd.xm_l - 1;
          if (!touched[indices[i]]) {
            touched[indices[i]] = PETSC_TRUE;
            i++;
          }
        }
      }
    }
    /* south boundary */
    if (dd.ipy == 0 && dd.dim > 1) {
      for (k = 0; k < dd.zm_l; k++) {
        for (j = 0; j < dd.xm_l; j++) {
          indices[i] = k * dd.ym_l * dd.xm_l + j;
          if (!touched[indices[i]]) {
            touched[indices[i]] = PETSC_TRUE;
            i++;
          }
        }
      }
    }
    /* north boundary */
    if (dd.ipy == dd.npy - 1 && dd.dim > 1) {
      for (k = 0; k < dd.zm_l; k++) {
        for (j = 0; j < dd.xm_l; j++) {
          indices[i] = k * dd.ym_l * dd.xm_l + (dd.ym_l - 1) * dd.xm_l + j;
          if (!touched[indices[i]]) {
            touched[indices[i]] = PETSC_TRUE;
            i++;
          }
        }
      }
    }
    /* bottom boundary */
    if (dd.ipz == 0 && dd.dim > 2) {
      for (k = 0; k < dd.ym_l; k++) {
        for (j = 0; j < dd.xm_l; j++) {
          indices[i] = k * dd.xm_l + j;
          if (!touched[indices[i]]) {
            touched[indices[i]] = PETSC_TRUE;
            i++;
          }
        }
      }
    }
    /* top boundary */
    if (dd.ipz == dd.npz - 1 && dd.dim > 2) {
      for (k = 0; k < dd.ym_l; k++) {
        for (j = 0; j < dd.xm_l; j++) {
          indices[i] = (dd.zm_l - 1) * dd.ym_l * dd.xm_l + k * dd.xm_l + j;
          if (!touched[indices[i]]) {
            touched[indices[i]] = PETSC_TRUE;
            i++;
          }
        }
      }
    }
    PetscCall(ISCreateGeneral(dd.gcomm, i, indices, PETSC_COPY_VALUES, &temp_neumann));
  }
  if (dirichlet) *dirichlet = temp_dirichlet;
  if (neumann) *neumann = temp_neumann;
  PetscCall(PetscFree(indices));
  PetscCall(PetscFree(touched));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMapping(DomainData dd, ISLocalToGlobalMapping *isg2lmap)
{
  DM                     da;
  AO                     ao;
  DMBoundaryType         bx = DM_BOUNDARY_NONE, by = DM_BOUNDARY_NONE, bz = DM_BOUNDARY_NONE;
  DMDAStencilType        stype = DMDA_STENCIL_BOX;
  ISLocalToGlobalMapping temp_isg2lmap;
  PetscInt               i, j, k, ig, jg, kg, lindex, gindex, localsize;
  PetscInt              *global_indices;

  PetscFunctionBeginUser;
  /* Not an efficient mapping: this function computes a very simple lexicographic mapping
     just to illustrate the creation of a MATIS object */
  localsize = dd.xm_l * dd.ym_l * dd.zm_l;
  PetscCall(PetscMalloc1(localsize, &global_indices));
  for (k = 0; k < dd.zm_l; k++) {
    kg = dd.startz + k;
    for (j = 0; j < dd.ym_l; j++) {
      jg = dd.starty + j;
      for (i = 0; i < dd.xm_l; i++) {
        ig                     = dd.startx + i;
        lindex                 = k * dd.xm_l * dd.ym_l + j * dd.xm_l + i;
        gindex                 = kg * dd.xm * dd.ym + jg * dd.xm + ig;
        global_indices[lindex] = gindex;
      }
    }
  }
  if (dd.dim == 3) {
    PetscCall(DMDACreate3d(dd.gcomm, bx, by, bz, stype, dd.xm, dd.ym, dd.zm, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da));
  } else if (dd.dim == 2) {
    PetscCall(DMDACreate2d(dd.gcomm, bx, by, stype, dd.xm, dd.ym, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  } else {
    PetscCall(DMDACreate1d(dd.gcomm, bx, dd.xm, 1, 1, NULL, &da));
  }
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetAOType(da, AOMEMORYSCALABLE));
  PetscCall(DMDAGetAO(da, &ao));
  PetscCall(AOApplicationToPetsc(ao, dd.xm_l * dd.ym_l * dd.zm_l, global_indices));
  PetscCall(ISLocalToGlobalMappingCreate(dd.gcomm, 1, localsize, global_indices, PETSC_OWN_POINTER, &temp_isg2lmap));
  PetscCall(DMDestroy(&da));
  *isg2lmap = temp_isg2lmap;
  PetscFunctionReturn(0);
}
static PetscErrorCode ComputeSubdomainMatrix(DomainData dd, GLLData glldata, Mat *local_mat)
{
  PetscInt     localsize, zloc, yloc, xloc, auxnex, auxney, auxnez;
  PetscInt     ie, je, ke, i, j, k, ig, jg, kg, ii, ming;
  PetscInt    *indexg, *cols, *colsg;
  PetscScalar *vals;
  Mat          temp_local_mat, elem_mat_DBC = 0, *usedmat;
  IS           submatIS;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(glldata.elem_mat, &i, &j));
  PetscCall(PetscMalloc1(i, &indexg));
  PetscCall(PetscMalloc1(i, &colsg));
  /* get submatrix of elem_mat without dirichlet nodes */
  if (!dd.pure_neumann && !dd.DBC_zerorows && !dd.ipx) {
    xloc = dd.p + 1;
    yloc = 1;
    zloc = 1;
    if (dd.dim > 1) yloc = dd.p + 1;
    if (dd.dim > 2) zloc = dd.p + 1;
    ii = 0;
    for (k = 0; k < zloc; k++) {
      for (j = 0; j < yloc; j++) {
        for (i = 1; i < xloc; i++) {
          indexg[ii] = k * xloc * yloc + j * xloc + i;
          ii++;
        }
      }
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ii, indexg, PETSC_COPY_VALUES, &submatIS));
    PetscCall(MatCreateSubMatrix(glldata.elem_mat, submatIS, submatIS, MAT_INITIAL_MATRIX, &elem_mat_DBC));
    PetscCall(ISDestroy(&submatIS));
  }

  /* Assemble subdomain matrix */
  localsize = dd.xm_l * dd.ym_l * dd.zm_l;
  PetscCall(MatCreate(PETSC_COMM_SELF, &temp_local_mat));
  PetscCall(MatSetSizes(temp_local_mat, localsize, localsize, localsize, localsize));
  PetscCall(MatSetOptionsPrefix(temp_local_mat, "subdomain_"));
  /* set local matrices type: here we use SEQSBAIJ primarily for testing purpose */
  /* in order to avoid conversions inside the BDDC code, use SeqAIJ if possible */
  if (dd.DBC_zerorows && !dd.ipx) { /* in this case, we need to zero out some of the rows, so use seqaij */
    PetscCall(MatSetType(temp_local_mat, MATSEQAIJ));
  } else {
    PetscCall(MatSetType(temp_local_mat, MATSEQSBAIJ));
  }
  PetscCall(MatSetFromOptions(temp_local_mat));

  i = PetscPowRealInt(3.0 * (dd.p + 1.0), dd.dim);

  PetscCall(MatSeqAIJSetPreallocation(temp_local_mat, i, NULL));      /* very overestimated */
  PetscCall(MatSeqSBAIJSetPreallocation(temp_local_mat, 1, i, NULL)); /* very overestimated */
  PetscCall(MatSeqBAIJSetPreallocation(temp_local_mat, 1, i, NULL));  /* very overestimated */
  PetscCall(MatSetOption(temp_local_mat, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));

  yloc = dd.p + 1;
  zloc = dd.p + 1;
  if (dd.dim < 3) zloc = 1;
  if (dd.dim < 2) yloc = 1;

  auxnez = dd.nez_l;
  auxney = dd.ney_l;
  auxnex = dd.nex_l;
  if (dd.dim < 3) auxnez = 1;
  if (dd.dim < 2) auxney = 1;

  for (ke = 0; ke < auxnez; ke++) {
    for (je = 0; je < auxney; je++) {
      for (ie = 0; ie < auxnex; ie++) {
        /* customize element accounting for BC */
        xloc    = dd.p + 1;
        ming    = 0;
        usedmat = &glldata.elem_mat;
        if (!dd.pure_neumann && !dd.DBC_zerorows && !dd.ipx) {
          if (ie == 0) {
            xloc    = dd.p;
            usedmat = &elem_mat_DBC;
          } else {
            ming    = -1;
            usedmat = &glldata.elem_mat;
          }
        }
        /* local to the element/global to the subdomain indexing */
        for (k = 0; k < zloc; k++) {
          kg = ke * dd.p + k;
          for (j = 0; j < yloc; j++) {
            jg = je * dd.p + j;
            for (i = 0; i < xloc; i++) {
              ig         = ie * dd.p + i + ming;
              ii         = k * xloc * yloc + j * xloc + i;
              indexg[ii] = kg * dd.xm_l * dd.ym_l + jg * dd.xm_l + ig;
            }
          }
        }
        /* Set values */
        for (i = 0; i < xloc * yloc * zloc; i++) {
          PetscCall(MatGetRow(*usedmat, i, &j, (const PetscInt **)&cols, (const PetscScalar **)&vals));
          for (k = 0; k < j; k++) colsg[k] = indexg[cols[k]];
          PetscCall(MatSetValues(temp_local_mat, 1, &indexg[i], j, colsg, vals, ADD_VALUES));
          PetscCall(MatRestoreRow(*usedmat, i, &j, (const PetscInt **)&cols, (const PetscScalar **)&vals));
        }
      }
    }
  }
  PetscCall(PetscFree(indexg));
  PetscCall(PetscFree(colsg));
  PetscCall(MatAssemblyBegin(temp_local_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(temp_local_mat, MAT_FINAL_ASSEMBLY));
#if DEBUG
  {
    Vec       lvec, rvec;
    PetscReal norm;
    PetscCall(MatCreateVecs(temp_local_mat, &lvec, &rvec));
    PetscCall(VecSet(lvec, 1.0));
    PetscCall(MatMult(temp_local_mat, lvec, rvec));
    PetscCall(VecNorm(rvec, NORM_INFINITY, &norm));
    PetscCall(VecDestroy(&lvec));
    PetscCall(VecDestroy(&rvec));
  }
#endif
  *local_mat = temp_local_mat;
  PetscCall(MatDestroy(&elem_mat_DBC));
  PetscFunctionReturn(0);
}

static PetscErrorCode GLLStuffs(DomainData dd, GLLData *glldata)
{
  PetscReal   *M, si;
  PetscScalar  x, z0, z1, z2, Lpj, Lpr, rhoGLj, rhoGLk;
  PetscBLASInt pm1, lierr;
  PetscInt     i, j, n, k, s, r, q, ii, jj, p = dd.p;
  PetscInt     xloc, yloc, zloc, xyloc, xyzloc;

  PetscFunctionBeginUser;
  /* Gauss-Lobatto-Legendre nodes zGL on [-1,1] */
  PetscCall(PetscCalloc1(p + 1, &glldata->zGL));

  glldata->zGL[0] = -1.0;
  glldata->zGL[p] = 1.0;
  if (p > 1) {
    if (p == 2) glldata->zGL[1] = 0.0;
    else {
      PetscCall(PetscMalloc1(p - 1, &M));
      for (i = 0; i < p - 1; i++) {
        si   = (PetscReal)(i + 1.0);
        M[i] = 0.5 * PetscSqrtReal(si * (si + 2.0) / ((si + 0.5) * (si + 1.5)));
      }
      pm1 = p - 1;
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscCallBLAS("LAPACKsteqr", LAPACKsteqr_("N", &pm1, &glldata->zGL[1], M, &x, &pm1, M, &lierr));
      PetscCheck(!lierr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in STERF Lapack routine %d", (int)lierr);
      PetscCall(PetscFPTrapPop());
      PetscCall(PetscFree(M));
    }
  }

  /* Weights for 1D quadrature */
  PetscCall(PetscMalloc1(p + 1, &glldata->rhoGL));

  glldata->rhoGL[0] = 2.0 / (PetscScalar)(p * (p + 1.0));
  glldata->rhoGL[p] = glldata->rhoGL[0];
  z2                = -1; /* Dummy value to avoid -Wmaybe-initialized */
  for (i = 1; i < p; i++) {
    x  = glldata->zGL[i];
    z0 = 1.0;
    z1 = x;
    for (n = 1; n < p; n++) {
      z2 = x * z1 * (2.0 * n + 1.0) / (n + 1.0) - z0 * (PetscScalar)(n / (n + 1.0));
      z0 = z1;
      z1 = z2;
    }
    glldata->rhoGL[i] = 2.0 / (p * (p + 1.0) * z2 * z2);
  }

  /* Auxiliary mat for laplacian */
  PetscCall(PetscMalloc1(p + 1, &glldata->A));
  PetscCall(PetscMalloc1((p + 1) * (p + 1), &glldata->A[0]));
  for (i = 1; i < p + 1; i++) glldata->A[i] = glldata->A[i - 1] + p + 1;

  for (j = 1; j < p; j++) {
    x  = glldata->zGL[j];
    z0 = 1.0;
    z1 = x;
    for (n = 1; n < p; n++) {
      z2 = x * z1 * (2.0 * n + 1.0) / (n + 1.0) - z0 * (PetscScalar)(n / (n + 1.0));
      z0 = z1;
      z1 = z2;
    }
    Lpj = z2;
    for (r = 1; r < p; r++) {
      if (r == j) {
        glldata->A[j][j] = 2.0 / (3.0 * (1.0 - glldata->zGL[j] * glldata->zGL[j]) * Lpj * Lpj);
      } else {
        x  = glldata->zGL[r];
        z0 = 1.0;
        z1 = x;
        for (n = 1; n < p; n++) {
          z2 = x * z1 * (2.0 * n + 1.0) / (n + 1.0) - z0 * (PetscScalar)(n / (n + 1.0));
          z0 = z1;
          z1 = z2;
        }
        Lpr              = z2;
        glldata->A[r][j] = 4.0 / (p * (p + 1.0) * Lpj * Lpr * (glldata->zGL[j] - glldata->zGL[r]) * (glldata->zGL[j] - glldata->zGL[r]));
      }
    }
  }
  for (j = 1; j < p + 1; j++) {
    x  = glldata->zGL[j];
    z0 = 1.0;
    z1 = x;
    for (n = 1; n < p; n++) {
      z2 = x * z1 * (2.0 * n + 1.0) / (n + 1.0) - z0 * (PetscScalar)(n / (n + 1.0));
      z0 = z1;
      z1 = z2;
    }
    Lpj              = z2;
    glldata->A[j][0] = 4.0 * PetscPowRealInt(-1.0, p) / (p * (p + 1.0) * Lpj * (1.0 + glldata->zGL[j]) * (1.0 + glldata->zGL[j]));
    glldata->A[0][j] = glldata->A[j][0];
  }
  for (j = 0; j < p; j++) {
    x  = glldata->zGL[j];
    z0 = 1.0;
    z1 = x;
    for (n = 1; n < p; n++) {
      z2 = x * z1 * (2.0 * n + 1.0) / (n + 1.0) - z0 * (PetscScalar)(n / (n + 1.0));
      z0 = z1;
      z1 = z2;
    }
    Lpj = z2;

    glldata->A[p][j] = 4.0 / (p * (p + 1.0) * Lpj * (1.0 - glldata->zGL[j]) * (1.0 - glldata->zGL[j]));
    glldata->A[j][p] = glldata->A[p][j];
  }
  glldata->A[0][0] = 0.5 + (p * (p + 1.0) - 2.0) / 6.0;
  glldata->A[p][p] = glldata->A[0][0];

  /* compute element matrix */
  xloc = p + 1;
  yloc = p + 1;
  zloc = p + 1;
  if (dd.dim < 2) yloc = 1;
  if (dd.dim < 3) zloc = 1;
  xyloc  = xloc * yloc;
  xyzloc = xloc * yloc * zloc;

  PetscCall(MatCreate(PETSC_COMM_SELF, &glldata->elem_mat));
  PetscCall(MatSetSizes(glldata->elem_mat, xyzloc, xyzloc, xyzloc, xyzloc));
  PetscCall(MatSetType(glldata->elem_mat, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(glldata->elem_mat, xyzloc, NULL)); /* overestimated */
  PetscCall(MatZeroEntries(glldata->elem_mat));
  PetscCall(MatSetOption(glldata->elem_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));

  for (k = 0; k < zloc; k++) {
    if (dd.dim > 2) rhoGLk = glldata->rhoGL[k];
    else rhoGLk = 1.0;

    for (j = 0; j < yloc; j++) {
      if (dd.dim > 1) rhoGLj = glldata->rhoGL[j];
      else rhoGLj = 1.0;

      for (i = 0; i < xloc; i++) {
        ii = k * xyloc + j * xloc + i;
        s  = k;
        r  = j;
        for (q = 0; q < xloc; q++) {
          jj = s * xyloc + r * xloc + q;
          PetscCall(MatSetValue(glldata->elem_mat, jj, ii, glldata->A[i][q] * rhoGLj * rhoGLk, ADD_VALUES));
        }
        if (dd.dim > 1) {
          s = k;
          q = i;
          for (r = 0; r < yloc; r++) {
            jj = s * xyloc + r * xloc + q;
            PetscCall(MatSetValue(glldata->elem_mat, jj, ii, glldata->A[j][r] * glldata->rhoGL[i] * rhoGLk, ADD_VALUES));
          }
        }
        if (dd.dim > 2) {
          r = j;
          q = i;
          for (s = 0; s < zloc; s++) {
            jj = s * xyloc + r * xloc + q;
            PetscCall(MatSetValue(glldata->elem_mat, jj, ii, glldata->A[k][s] * rhoGLj * glldata->rhoGL[i], ADD_VALUES));
          }
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(glldata->elem_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(glldata->elem_mat, MAT_FINAL_ASSEMBLY));
#if DEBUG
  {
    Vec       lvec, rvec;
    PetscReal norm;
    PetscCall(MatCreateVecs(glldata->elem_mat, &lvec, &rvec));
    PetscCall(VecSet(lvec, 1.0));
    PetscCall(MatMult(glldata->elem_mat, lvec, rvec));
    PetscCall(VecNorm(rvec, NORM_INFINITY, &norm));
    PetscCall(VecDestroy(&lvec));
    PetscCall(VecDestroy(&rvec));
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DomainDecomposition(DomainData *dd)
{
  PetscMPIInt rank;
  PetscInt    i, j, k;

  PetscFunctionBeginUser;
  /* Subdomain index in cartesian coordinates */
  MPI_Comm_rank(dd->gcomm, &rank);
  dd->ipx = rank % dd->npx;
  if (dd->dim > 1) dd->ipz = rank / (dd->npx * dd->npy);
  else dd->ipz = 0;

  dd->ipy = rank / dd->npx - dd->ipz * dd->npy;
  /* number of local elements */
  dd->nex_l = dd->nex / dd->npx;
  if (dd->ipx < dd->nex % dd->npx) dd->nex_l++;
  if (dd->dim > 1) {
    dd->ney_l = dd->ney / dd->npy;
    if (dd->ipy < dd->ney % dd->npy) dd->ney_l++;
  } else {
    dd->ney_l = 0;
  }
  if (dd->dim > 2) {
    dd->nez_l = dd->nez / dd->npz;
    if (dd->ipz < dd->nez % dd->npz) dd->nez_l++;
  } else {
    dd->nez_l = 0;
  }
  /* local and global number of dofs */
  dd->xm_l = dd->nex_l * dd->p + 1;
  dd->xm   = dd->nex * dd->p + 1;
  dd->ym_l = dd->ney_l * dd->p + 1;
  dd->ym   = dd->ney * dd->p + 1;
  dd->zm_l = dd->nez_l * dd->p + 1;
  dd->zm   = dd->nez * dd->p + 1;
  if (!dd->pure_neumann) {
    if (!dd->DBC_zerorows && !dd->ipx) dd->xm_l--;
    if (!dd->DBC_zerorows) dd->xm--;
  }

  /* starting global index for local dofs (simple lexicographic order) */
  dd->startx = 0;
  j          = dd->nex / dd->npx;
  for (i = 0; i < dd->ipx; i++) {
    k = j;
    if (i < dd->nex % dd->npx) k++;
    dd->startx = dd->startx + k * dd->p;
  }
  if (!dd->pure_neumann && !dd->DBC_zerorows && dd->ipx) dd->startx--;

  dd->starty = 0;
  if (dd->dim > 1) {
    j = dd->ney / dd->npy;
    for (i = 0; i < dd->ipy; i++) {
      k = j;
      if (i < dd->ney % dd->npy) k++;
      dd->starty = dd->starty + k * dd->p;
    }
  }
  dd->startz = 0;
  if (dd->dim > 2) {
    j = dd->nez / dd->npz;
    for (i = 0; i < dd->ipz; i++) {
      k = j;
      if (i < dd->nez % dd->npz) k++;
      dd->startz = dd->startz + k * dd->p;
    }
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode ComputeMatrix(DomainData dd, Mat *A)
{
  GLLData                gll;
  Mat                    local_mat = 0, temp_A = 0;
  ISLocalToGlobalMapping matis_map   = 0;
  IS                     dirichletIS = 0;

  PetscFunctionBeginUser;
  /* Compute some stuff of Gauss-Legendre-Lobatto quadrature rule */
  PetscCall(GLLStuffs(dd, &gll));
  /* Compute matrix of subdomain Neumann problem */
  PetscCall(ComputeSubdomainMatrix(dd, gll, &local_mat));
  /* Compute global mapping of local dofs */
  PetscCall(ComputeMapping(dd, &matis_map));
  /* Create MATIS object needed by BDDC */
  PetscCall(MatCreateIS(dd.gcomm, 1, PETSC_DECIDE, PETSC_DECIDE, dd.xm * dd.ym * dd.zm, dd.xm * dd.ym * dd.zm, matis_map, matis_map, &temp_A));
  /* Set local subdomain matrices into MATIS object */
  PetscCall(MatScale(local_mat, dd.scalingfactor));
  PetscCall(MatISSetLocalMat(temp_A, local_mat));
  /* Call assembly functions */
  PetscCall(MatAssemblyBegin(temp_A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(temp_A, MAT_FINAL_ASSEMBLY));

  if (dd.DBC_zerorows) {
    PetscInt dirsize;

    PetscCall(ComputeSpecialBoundaryIndices(dd, &dirichletIS, NULL));
    PetscCall(MatSetOption(local_mat, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
    PetscCall(MatZeroRowsColumnsLocalIS(temp_A, dirichletIS, 1.0, NULL, NULL));
    PetscCall(ISGetLocalSize(dirichletIS, &dirsize));
    PetscCall(ISDestroy(&dirichletIS));
  }

  /* giving hints to local and global matrices could be useful for the BDDC */
  PetscCall(MatSetOption(local_mat, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(local_mat, MAT_SPD_ETERNAL, PETSC_TRUE));
#if DEBUG
  {
    Vec       lvec, rvec;
    PetscReal norm;
    PetscCall(MatCreateVecs(temp_A, &lvec, &rvec));
    PetscCall(VecSet(lvec, 1.0));
    PetscCall(MatMult(temp_A, lvec, rvec));
    PetscCall(VecNorm(rvec, NORM_INFINITY, &norm));
    PetscCall(VecDestroy(&lvec));
    PetscCall(VecDestroy(&rvec));
  }
#endif
  /* free allocated workspace */
  PetscCall(PetscFree(gll.zGL));
  PetscCall(PetscFree(gll.rhoGL));
  PetscCall(PetscFree(gll.A[0]));
  PetscCall(PetscFree(gll.A));
  PetscCall(MatDestroy(&gll.elem_mat));
  PetscCall(MatDestroy(&local_mat));
  PetscCall(ISLocalToGlobalMappingDestroy(&matis_map));
  /* give back the pointer to te MATIS object */
  *A = temp_A;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeKSPFETIDP(DomainData dd, KSP ksp_bddc, KSP *ksp_fetidp)
{
  KSP temp_ksp;
  PC  pc;

  PetscFunctionBeginUser;
  PetscCall(KSPGetPC(ksp_bddc, &pc));
  if (!dd.testkspfetidp) {
    PC  D;
    Mat F;

    PetscCall(PCBDDCCreateFETIDPOperators(pc, PETSC_TRUE, NULL, &F, &D));
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)F), &temp_ksp));
    PetscCall(KSPSetOperators(temp_ksp, F, F));
    PetscCall(KSPSetType(temp_ksp, KSPCG));
    PetscCall(KSPSetPC(temp_ksp, D));
    PetscCall(KSPSetComputeSingularValues(temp_ksp, PETSC_TRUE));
    PetscCall(KSPSetOptionsPrefix(temp_ksp, "fluxes_"));
    PetscCall(KSPSetFromOptions(temp_ksp));
    PetscCall(KSPSetUp(temp_ksp));
    PetscCall(MatDestroy(&F));
    PetscCall(PCDestroy(&D));
  } else {
    Mat A, Ap;

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksp_bddc), &temp_ksp));
    PetscCall(KSPGetOperators(ksp_bddc, &A, &Ap));
    PetscCall(KSPSetOperators(temp_ksp, A, Ap));
    PetscCall(KSPSetOptionsPrefix(temp_ksp, "fluxes_"));
    PetscCall(KSPSetType(temp_ksp, KSPFETIDP));
    PetscCall(KSPFETIDPSetInnerBDDC(temp_ksp, pc));
    PetscCall(KSPSetComputeSingularValues(temp_ksp, PETSC_TRUE));
    PetscCall(KSPSetFromOptions(temp_ksp));
    PetscCall(KSPSetUp(temp_ksp));
  }
  *ksp_fetidp = temp_ksp;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeKSPBDDC(DomainData dd, Mat A, KSP *ksp)
{
  KSP          temp_ksp;
  PC           pc;
  IS           primals, dirichletIS = 0, neumannIS = 0, *bddc_dofs_splitting;
  PetscInt     vidx[8], localsize, *xadj = NULL, *adjncy = NULL;
  MatNullSpace near_null_space;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(dd.gcomm, &temp_ksp));
  PetscCall(KSPSetOperators(temp_ksp, A, A));
  PetscCall(KSPSetType(temp_ksp, KSPCG));
  PetscCall(KSPGetPC(temp_ksp, &pc));
  PetscCall(PCSetType(pc, PCBDDC));

  localsize = dd.xm_l * dd.ym_l * dd.zm_l;

  /* BDDC customization */

  /* jumping coefficients case */
  PetscCall(PCISSetSubdomainScalingFactor(pc, dd.scalingfactor));

  /* Dofs splitting
     Simple stride-1 IS
     It is not needed since, by default, PCBDDC assumes a stride-1 split */
  PetscCall(PetscMalloc1(1, &bddc_dofs_splitting));
#if 1
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, localsize, 0, 1, &bddc_dofs_splitting[0]));
  PetscCall(PCBDDCSetDofsSplittingLocal(pc, 1, bddc_dofs_splitting));
#else
  /* examples for global ordering */

  /* each process lists the nodes it owns */
  PetscInt sr, er;
  PetscCall(MatGetOwnershipRange(A, &sr, &er));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, er - sr, sr, 1, &bddc_dofs_splitting[0]));
  PetscCall(PCBDDCSetDofsSplitting(pc, 1, bddc_dofs_splitting));
  /* Split can be passed in a more general way since any process can list any node */
#endif
  PetscCall(ISDestroy(&bddc_dofs_splitting[0]));
  PetscCall(PetscFree(bddc_dofs_splitting));

  /* Primal constraints implemented by using a near null space attached to A -> now it passes in only the constants
    (which in practice is not needed since by default PCBDDC builds the primal space using constants for quadrature formulas */
#if 0
  Vec vecs[2];
  PetscRandom rctx;
  PetscCall(MatCreateVecs(A,&vecs[0],&vecs[1]));
  PetscCall(PetscRandomCreate(dd.gcomm,&rctx));
  PetscCall(VecSetRandom(vecs[0],rctx));
  PetscCall(VecSetRandom(vecs[1],rctx));
  PetscCall(MatNullSpaceCreate(dd.gcomm,PETSC_TRUE,2,vecs,&near_null_space));
  PetscCall(VecDestroy(&vecs[0]));
  PetscCall(VecDestroy(&vecs[1]));
  PetscCall(PetscRandomDestroy(&rctx));
#else
  PetscCall(MatNullSpaceCreate(dd.gcomm, PETSC_TRUE, 0, NULL, &near_null_space));
#endif
  PetscCall(MatSetNearNullSpace(A, near_null_space));
  PetscCall(MatNullSpaceDestroy(&near_null_space));

  /* CSR graph of subdomain dofs */
  PetscCall(BuildCSRGraph(dd, &xadj, &adjncy));
  PetscCall(PCBDDCSetLocalAdjacencyGraph(pc, localsize, xadj, adjncy, PETSC_OWN_POINTER));

  /* Prescribe user-defined primal vertices: in this case we use the 8 corners in 3D (for 2D and 1D, the indices are duplicated) */
  vidx[0] = 0 * dd.xm_l + 0;
  vidx[1] = 0 * dd.xm_l + dd.xm_l - 1;
  vidx[2] = (dd.ym_l - 1) * dd.xm_l + 0;
  vidx[3] = (dd.ym_l - 1) * dd.xm_l + dd.xm_l - 1;
  vidx[4] = (dd.zm_l - 1) * dd.xm_l * dd.ym_l + 0 * dd.xm_l + 0;
  vidx[5] = (dd.zm_l - 1) * dd.xm_l * dd.ym_l + 0 * dd.xm_l + dd.xm_l - 1;
  vidx[6] = (dd.zm_l - 1) * dd.xm_l * dd.ym_l + (dd.ym_l - 1) * dd.xm_l + 0;
  vidx[7] = (dd.zm_l - 1) * dd.xm_l * dd.ym_l + (dd.ym_l - 1) * dd.xm_l + dd.xm_l - 1;
  PetscCall(ISCreateGeneral(dd.gcomm, 8, vidx, PETSC_COPY_VALUES, &primals));
  PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc, primals));
  PetscCall(ISDestroy(&primals));

  /* Neumann/Dirichlet indices on the global boundary */
  if (dd.DBC_zerorows) {
    /* Only in case you eliminate some rows matrix with zerorows function, you need to set dirichlet indices into PCBDDC data */
    PetscCall(ComputeSpecialBoundaryIndices(dd, &dirichletIS, &neumannIS));
    PetscCall(PCBDDCSetNeumannBoundariesLocal(pc, neumannIS));
    PetscCall(PCBDDCSetDirichletBoundariesLocal(pc, dirichletIS));
  } else {
    if (dd.pure_neumann) {
      /* In such a case, all interface nodes lying on the global boundary are neumann nodes */
      PetscCall(ComputeSpecialBoundaryIndices(dd, NULL, &neumannIS));
      PetscCall(PCBDDCSetNeumannBoundariesLocal(pc, neumannIS));
    } else {
      /* It is wrong setting dirichlet indices without having zeroed the corresponding rows in the global matrix */
      /* But we can still compute them since nodes near the dirichlet boundaries does not need to be defined as neumann nodes */
      PetscCall(ComputeSpecialBoundaryIndices(dd, &dirichletIS, &neumannIS));
      PetscCall(PCBDDCSetNeumannBoundariesLocal(pc, neumannIS));
    }
  }

  /* Pass local null space information to local matrices (needed when using approximate local solvers) */
  if (dd.ipx || dd.pure_neumann) {
    MatNullSpace nsp;
    Mat          local_mat;

    PetscCall(MatISGetLocalMat(A, &local_mat));
    PetscCall(MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, NULL, &nsp));
    PetscCall(MatSetNullSpace(local_mat, nsp));
    PetscCall(MatNullSpaceDestroy(&nsp));
  }
  PetscCall(KSPSetComputeSingularValues(temp_ksp, PETSC_TRUE));
  PetscCall(KSPSetOptionsPrefix(temp_ksp, "physical_"));
  PetscCall(KSPSetFromOptions(temp_ksp));
  PetscCall(KSPSetUp(temp_ksp));
  *ksp = temp_ksp;
  PetscCall(ISDestroy(&dirichletIS));
  PetscCall(ISDestroy(&neumannIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeDomainData(DomainData *dd)
{
  PetscMPIInt sizes, rank;
  PetscInt    factor;

  PetscFunctionBeginUser;
  dd->gcomm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(dd->gcomm, &sizes));
  PetscCallMPI(MPI_Comm_rank(dd->gcomm, &rank));
  /* Get information from command line */
  /* Processors/subdomains per dimension */
  /* Default is 1d problem */
  dd->npx = sizes;
  dd->npy = 0;
  dd->npz = 0;
  dd->dim = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npx", &dd->npx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npy", &dd->npy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npz", &dd->npz, NULL));
  if (dd->npy) dd->dim++;
  if (dd->npz) dd->dim++;
  /* Number of elements per dimension */
  /* Default is one element per subdomain */
  dd->nex = dd->npx;
  dd->ney = dd->npy;
  dd->nez = dd->npz;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nex", &dd->nex, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ney", &dd->ney, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nez", &dd->nez, NULL));
  if (!dd->npy) {
    dd->ney = 0;
    dd->nez = 0;
  }
  if (!dd->npz) dd->nez = 0;
  /* Spectral degree */
  dd->p = 3;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-p", &dd->p, NULL));
  /* pure neumann problem? */
  dd->pure_neumann = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-pureneumann", &dd->pure_neumann, NULL));

  /* How to enforce dirichlet boundary conditions (in case pureneumann has not been requested explicitly) */
  dd->DBC_zerorows = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-usezerorows", &dd->DBC_zerorows, NULL));
  if (dd->pure_neumann) dd->DBC_zerorows = PETSC_FALSE;
  dd->scalingfactor = 1.0;

  factor = 0.0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-jump", &factor, NULL));
  /* checkerboard pattern */
  dd->scalingfactor = PetscPowScalar(10.0, (PetscScalar)factor * PetscPowScalar(-1.0, (PetscScalar)rank));
  /* test data passed in */
  if (dd->dim == 1) {
    PetscCheck(sizes == dd->npx, dd->gcomm, PETSC_ERR_USER, "Number of mpi procs in 1D must be equal to npx");
    PetscCheck(dd->nex >= dd->npx, dd->gcomm, PETSC_ERR_USER, "Number of elements per dim must be greater/equal than number of procs per dim");
  } else if (dd->dim == 2) {
    PetscCheck(sizes == dd->npx * dd->npy, dd->gcomm, PETSC_ERR_USER, "Number of mpi procs in 2D must be equal to npx*npy");
    PetscCheck(dd->nex >= dd->npx || dd->ney < dd->npy, dd->gcomm, PETSC_ERR_USER, "Number of elements per dim must be greater/equal than number of procs per dim");
  } else {
    PetscCheck(sizes == dd->npx * dd->npy * dd->npz, dd->gcomm, PETSC_ERR_USER, "Number of mpi procs in 3D must be equal to npx*npy*npz");
    PetscCheck(dd->nex >= dd->npx && dd->ney >= dd->npy && dd->nez >= dd->npz, dd->gcomm, PETSC_ERR_USER, "Number of elements per dim must be greater/equal than number of ranks per dim");
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  DomainData         dd;
  PetscReal          norm, maxeig, mineig;
  PetscScalar        scalar_value;
  PetscInt           ndofs, its;
  Mat                A = NULL, F = NULL;
  KSP                KSPwithBDDC = NULL, KSPwithFETIDP = NULL;
  KSPConvergedReason reason;
  Vec                exact_solution = NULL, bddc_solution = NULL, bddc_rhs = NULL;
  PetscBool          testfetidp = PETSC_TRUE;

  /* Init PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  /* Initialize DomainData */
  PetscCall(InitializeDomainData(&dd));
  /* Decompose domain */
  PetscCall(DomainDecomposition(&dd));
#if DEBUG
  printf("Subdomain data\n");
  printf("IPS   : %d %d %d\n", dd.ipx, dd.ipy, dd.ipz);
  printf("NEG   : %d %d %d\n", dd.nex, dd.ney, dd.nez);
  printf("NEL   : %d %d %d\n", dd.nex_l, dd.ney_l, dd.nez_l);
  printf("LDO   : %d %d %d\n", dd.xm_l, dd.ym_l, dd.zm_l);
  printf("SIZES : %d %d %d\n", dd.xm, dd.ym, dd.zm);
  printf("STARTS: %d %d %d\n", dd.startx, dd.starty, dd.startz);
#endif
  dd.testkspfetidp = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-testfetidp", &testfetidp, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-testkspfetidp", &dd.testkspfetidp, NULL));
  /* assemble global matrix */
  PetscCall(ComputeMatrix(dd, &A));
  /* get work vectors */
  PetscCall(MatCreateVecs(A, &bddc_solution, NULL));
  PetscCall(VecDuplicate(bddc_solution, &bddc_rhs));
  PetscCall(VecDuplicate(bddc_solution, &exact_solution));
  /* create and customize KSP/PC for BDDC */
  PetscCall(ComputeKSPBDDC(dd, A, &KSPwithBDDC));
  /* create KSP/PC for FETIDP */
  if (testfetidp) PetscCall(ComputeKSPFETIDP(dd, KSPwithBDDC, &KSPwithFETIDP));
    /* create random exact solution */
#if defined(PETSC_USE_COMPLEX)
  PetscCall(VecSet(exact_solution, 1.0 + PETSC_i));
#else
  PetscCall(VecSetRandom(exact_solution, NULL));
#endif
  PetscCall(VecShift(exact_solution, -0.5));
  PetscCall(VecScale(exact_solution, 100.0));
  PetscCall(VecGetSize(exact_solution, &ndofs));
  if (dd.pure_neumann) {
    PetscCall(VecSum(exact_solution, &scalar_value));
    scalar_value = -scalar_value / (PetscScalar)ndofs;
    PetscCall(VecShift(exact_solution, scalar_value));
  }
  /* assemble BDDC rhs */
  PetscCall(MatMult(A, exact_solution, bddc_rhs));
  /* test ksp with BDDC */
  PetscCall(KSPSolve(KSPwithBDDC, bddc_rhs, bddc_solution));
  PetscCall(KSPGetIterationNumber(KSPwithBDDC, &its));
  PetscCall(KSPGetConvergedReason(KSPwithBDDC, &reason));
  PetscCall(KSPComputeExtremeSingularValues(KSPwithBDDC, &maxeig, &mineig));
  if (dd.pure_neumann) {
    PetscCall(VecSum(bddc_solution, &scalar_value));
    scalar_value = -scalar_value / (PetscScalar)ndofs;
    PetscCall(VecShift(bddc_solution, scalar_value));
  }
  /* check exact_solution and BDDC solultion */
  PetscCall(VecAXPY(bddc_solution, -1.0, exact_solution));
  PetscCall(VecNorm(bddc_solution, NORM_INFINITY, &norm));
  PetscCall(PetscPrintf(dd.gcomm, "---------------------BDDC stats-------------------------------\n"));
  PetscCall(PetscPrintf(dd.gcomm, "Number of degrees of freedom               : %8" PetscInt_FMT "\n", ndofs));
  if (reason < 0) {
    PetscCall(PetscPrintf(dd.gcomm, "Number of iterations                       : %8" PetscInt_FMT "\n", its));
    PetscCall(PetscPrintf(dd.gcomm, "Converged reason                           : %s\n", KSPConvergedReasons[reason]));
  }
  if (0.95 <= mineig && mineig <= 1.05) mineig = 1.0;
  PetscCall(PetscPrintf(dd.gcomm, "Eigenvalues preconditioned operator        : %1.1e %1.1e\n", (double)PetscFloorReal(100. * mineig) / 100., (double)PetscCeilReal(100. * maxeig) / 100.));
  if (norm > 1.e-1 || reason < 0) PetscCall(PetscPrintf(dd.gcomm, "Error between exact and computed solution : %1.2e\n", (double)norm));
  PetscCall(PetscPrintf(dd.gcomm, "--------------------------------------------------------------\n"));
  if (testfetidp) {
    Vec fetidp_solution_all = NULL, fetidp_solution = NULL, fetidp_rhs = NULL;

    PetscCall(VecDuplicate(bddc_solution, &fetidp_solution_all));
    if (!dd.testkspfetidp) {
      /* assemble fetidp rhs on the space of Lagrange multipliers */
      PetscCall(KSPGetOperators(KSPwithFETIDP, &F, NULL));
      PetscCall(MatCreateVecs(F, &fetidp_solution, &fetidp_rhs));
      PetscCall(PCBDDCMatFETIDPGetRHS(F, bddc_rhs, fetidp_rhs));
      PetscCall(VecSet(fetidp_solution, 0.0));
      /* test ksp with FETIDP */
      PetscCall(KSPSolve(KSPwithFETIDP, fetidp_rhs, fetidp_solution));
      /* assemble fetidp solution on physical domain */
      PetscCall(PCBDDCMatFETIDPGetSolution(F, fetidp_solution, fetidp_solution_all));
    } else {
      KSP kspF;
      PetscCall(KSPSolve(KSPwithFETIDP, bddc_rhs, fetidp_solution_all));
      PetscCall(KSPFETIDPGetInnerKSP(KSPwithFETIDP, &kspF));
      PetscCall(KSPGetOperators(kspF, &F, NULL));
    }
    PetscCall(MatGetSize(F, &ndofs, NULL));
    PetscCall(KSPGetIterationNumber(KSPwithFETIDP, &its));
    PetscCall(KSPGetConvergedReason(KSPwithFETIDP, &reason));
    PetscCall(KSPComputeExtremeSingularValues(KSPwithFETIDP, &maxeig, &mineig));
    /* check FETIDP sol */
    if (dd.pure_neumann) {
      PetscCall(VecSum(fetidp_solution_all, &scalar_value));
      scalar_value = -scalar_value / (PetscScalar)ndofs;
      PetscCall(VecShift(fetidp_solution_all, scalar_value));
    }
    PetscCall(VecAXPY(fetidp_solution_all, -1.0, exact_solution));
    PetscCall(VecNorm(fetidp_solution_all, NORM_INFINITY, &norm));
    PetscCall(PetscPrintf(dd.gcomm, "------------------FETI-DP stats-------------------------------\n"));
    PetscCall(PetscPrintf(dd.gcomm, "Number of degrees of freedom               : %8" PetscInt_FMT "\n", ndofs));
    if (reason < 0) {
      PetscCall(PetscPrintf(dd.gcomm, "Number of iterations                       : %8" PetscInt_FMT "\n", its));
      PetscCall(PetscPrintf(dd.gcomm, "Converged reason                           : %s\n", KSPConvergedReasons[reason]));
    }
    if (0.95 <= mineig && mineig <= 1.05) mineig = 1.0;
    PetscCall(PetscPrintf(dd.gcomm, "Eigenvalues preconditioned operator        : %1.1e %1.1e\n", (double)PetscFloorReal(100. * mineig) / 100., (double)PetscCeilReal(100. * maxeig) / 100.));
    if (norm > 1.e-1 || reason < 0) PetscCall(PetscPrintf(dd.gcomm, "Error between exact and computed solution : %1.2e\n", (double)norm));
    PetscCall(PetscPrintf(dd.gcomm, "--------------------------------------------------------------\n"));
    PetscCall(VecDestroy(&fetidp_solution));
    PetscCall(VecDestroy(&fetidp_solution_all));
    PetscCall(VecDestroy(&fetidp_rhs));
  }
  PetscCall(KSPDestroy(&KSPwithFETIDP));
  PetscCall(VecDestroy(&exact_solution));
  PetscCall(VecDestroy(&bddc_solution));
  PetscCall(VecDestroy(&bddc_rhs));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&KSPwithBDDC));
  /* Quit PETSc */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 testset:
   nsize: 4
   args: -nex 7 -physical_pc_bddc_coarse_eqs_per_proc 3 -physical_pc_bddc_switch_static
   output_file: output/ex59_bddc_fetidp_1.out
   test:
     suffix: bddc_fetidp_1
   test:
     requires: viennacl
     suffix: bddc_fetidp_1_viennacl
     args: -subdomain_mat_type aijviennacl
   test:
     requires: cuda
     suffix: bddc_fetidp_1_cuda
     args: -subdomain_mat_type aijcusparse -physical_pc_bddc_dirichlet_pc_factor_mat_solver_type cusparse

 testset:
   nsize: 4
   args: -npx 2 -npy 2 -nex 6 -ney 6 -fluxes_ksp_max_it 10 -physical_ksp_max_it 10
   requires: !single
   test:
     suffix: bddc_fetidp_2
   test:
     suffix: bddc_fetidp_3
     args: -npz 1 -nez 1
   test:
     suffix: bddc_fetidp_4
     args: -npz 1 -nez 1 -physical_pc_bddc_use_change_of_basis -physical_sub_schurs_mat_solver_type petsc -physical_pc_bddc_use_deluxe_scaling -physical_pc_bddc_deluxe_singlemat -fluxes_fetidp_ksp_type cg

 testset:
   nsize: 8
   suffix: bddc_fetidp_approximate
   args: -npx 2 -npy 2 -npz 2 -p 2 -nex 8 -ney 7 -nez 9 -physical_ksp_max_it 20 -subdomain_mat_type aij -physical_pc_bddc_switch_static -physical_pc_bddc_dirichlet_approximate -physical_pc_bddc_neumann_approximate -physical_pc_bddc_dirichlet_pc_type gamg -physical_pc_bddc_dirichlet_pc_gamg_esteig_ksp_type cg -physical_pc_bddc_dirichlet_pc_gamg_esteig_ksp_max_it 10 -physical_pc_bddc_dirichlet_mg_levels_ksp_max_it 3 -physical_pc_bddc_neumann_pc_type sor -physical_pc_bddc_neumann_approximate_scale -testfetidp 0

 testset:
   nsize: 4
   args: -npx 2 -npy 2 -nex 6 -ney 6 -fluxes_ksp_max_it 10 -physical_ksp_max_it 10 -physical_ksp_view -physical_pc_bddc_levels 1
   filter: grep -v "variant HERMITIAN"
   requires: !single
   test:
     suffix: bddc_fetidp_ml_1
     args: -physical_pc_bddc_coarsening_ratio 1
   test:
     suffix: bddc_fetidp_ml_2
     args: -physical_pc_bddc_coarsening_ratio 2 -mat_partitioning_type average
   test:
     suffix: bddc_fetidp_ml_3
     args: -physical_pc_bddc_coarsening_ratio 4

 testset:
   nsize: 9
   args: -npx 3 -npy 3 -p 2 -nex 6 -ney 6 -physical_pc_bddc_deluxe_singlemat -physical_sub_schurs_mat_solver_type petsc -physical_pc_bddc_use_deluxe_scaling -physical_pc_bddc_graph_maxcount 1 -physical_pc_bddc_levels 3 -physical_pc_bddc_coarsening_ratio 2 -physical_pc_bddc_coarse_ksp_type gmres
   output_file: output/ex59_bddc_fetidp_ml_eqlimit.out
   test:
     suffix: bddc_fetidp_ml_eqlimit_1
     args: -physical_pc_bddc_coarse_eqs_limit 31 -mat_partitioning_type average -physical_pc_bddc_coarse_pc_bddc_graph_maxcount 1
   test:
     suffix: bddc_fetidp_ml_eqlimit_2
     args: -physical_pc_bddc_coarse_eqs_limit 46

TEST*/
