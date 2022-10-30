
/*
Module Name: xxt
Module Info:

author:  Henry M. Tufo III
e-mail:  hmt@asci.uchicago.edu
contact:
+--------------------------------+--------------------------------+
|MCS Division - Building 221     |Department of Computer Science  |
|Argonne National Laboratory     |Ryerson 152                     |
|9700 S. Cass Avenue             |The University of Chicago       |
|Argonne, IL  60439              |Chicago, IL  60637              |
|(630) 252-5354/5986 ph/fx       |(773) 702-6019/8487 ph/fx       |
+--------------------------------+--------------------------------+

Last Modification: 3.20.01
*/
#include <../src/ksp/pc/impls/tfs/tfs.h>

#define LEFT  -1
#define RIGHT 1
#define BOTH  0

typedef struct xxt_solver_info {
  PetscInt      n, m, n_global, m_global;
  PetscInt      nnz, max_nnz, msg_buf_sz;
  PetscInt     *nsep, *lnsep, *fo, nfo, *stages;
  PetscInt     *col_sz, *col_indices;
  PetscScalar **col_vals, *x, *solve_uu, *solve_w;
  PetscInt      nsolves;
  PetscScalar   tot_solve_time;
} xxt_info;

typedef struct matvec_info {
  PetscInt     n, m, n_global, m_global;
  PetscInt    *local2global;
  PCTFS_gs_ADT PCTFS_gs_handle;
  PetscErrorCode (*matvec)(struct matvec_info *, PetscScalar *, PetscScalar *);
  void *grid_data;
} mv_info;

struct xxt_CDT {
  PetscInt  id;
  PetscInt  ns;
  PetscInt  level;
  xxt_info *info;
  mv_info  *mvi;
};

static PetscInt n_xxt         = 0;
static PetscInt n_xxt_handles = 0;

/* prototypes */
static PetscErrorCode do_xxt_solve(xxt_ADT xxt_handle, PetscScalar *rhs);
static PetscErrorCode check_handle(xxt_ADT xxt_handle);
static PetscErrorCode det_separators(xxt_ADT xxt_handle);
static PetscErrorCode do_matvec(mv_info *A, PetscScalar *v, PetscScalar *u);
static PetscErrorCode xxt_generate(xxt_ADT xxt_handle);
static PetscErrorCode do_xxt_factor(xxt_ADT xxt_handle);
static mv_info       *set_mvi(PetscInt *local2global, PetscInt n, PetscInt m, PetscErrorCode (*matvec)(mv_info *, PetscScalar *, PetscScalar *), void *grid_data);

xxt_ADT XXT_new(void)
{
  xxt_ADT xxt_handle;

  /* rolling count on n_xxt ... pot. problem here */
  n_xxt_handles++;
  xxt_handle       = (xxt_ADT)malloc(sizeof(struct xxt_CDT));
  xxt_handle->id   = ++n_xxt;
  xxt_handle->info = NULL;
  xxt_handle->mvi  = NULL;

  return (xxt_handle);
}

PetscErrorCode XXT_factor(xxt_ADT   xxt_handle,                                           /* prev. allocated xxt  handle */
                          PetscInt *local2global,                                         /* global column mapping       */
                          PetscInt  n,                                                    /* local num rows              */
                          PetscInt  m,                                                    /* local num cols              */
                          PetscErrorCode (*matvec)(void *, PetscScalar *, PetscScalar *), /* b_loc=A_local.x_loc         */
                          void *grid_data)                                                /* grid data for matvec        */
{
  PCTFS_comm_init();
  check_handle(xxt_handle);

  /* only 2^k for now and all nodes participating */
  PetscCheck((1 << (xxt_handle->level = PCTFS_i_log2_num_nodes)) == PCTFS_num_nodes, PETSC_COMM_SELF, PETSC_ERR_PLIB, "only 2^k for now and MPI_COMM_WORLD!!! %d != %d", 1 << PCTFS_i_log2_num_nodes, PCTFS_num_nodes);

  /* space for X info */
  xxt_handle->info = (xxt_info *)malloc(sizeof(xxt_info));

  /* set up matvec handles */
  xxt_handle->mvi = set_mvi(local2global, n, m, (PetscErrorCode(*)(mv_info *, PetscScalar *, PetscScalar *))matvec, grid_data);

  /* matrix is assumed to be of full rank */
  /* LATER we can reset to indicate rank def. */
  xxt_handle->ns = 0;

  /* determine separators and generate firing order - NB xxt info set here */
  det_separators(xxt_handle);

  return (do_xxt_factor(xxt_handle));
}

PetscErrorCode XXT_solve(xxt_ADT xxt_handle, PetscScalar *x, PetscScalar *b)
{
  PCTFS_comm_init();
  check_handle(xxt_handle);

  /* need to copy b into x? */
  if (b) PCTFS_rvec_copy(x, b, xxt_handle->mvi->n);
  return do_xxt_solve(xxt_handle, x);
}

PetscInt XXT_free(xxt_ADT xxt_handle)
{
  PCTFS_comm_init();
  check_handle(xxt_handle);
  n_xxt_handles--;

  free(xxt_handle->info->nsep);
  free(xxt_handle->info->lnsep);
  free(xxt_handle->info->fo);
  free(xxt_handle->info->stages);
  free(xxt_handle->info->solve_uu);
  free(xxt_handle->info->solve_w);
  free(xxt_handle->info->x);
  free(xxt_handle->info->col_vals);
  free(xxt_handle->info->col_sz);
  free(xxt_handle->info->col_indices);
  free(xxt_handle->info);
  free(xxt_handle->mvi->local2global);
  PCTFS_gs_free(xxt_handle->mvi->PCTFS_gs_handle);
  free(xxt_handle->mvi);
  free(xxt_handle);

  /* if the check fails we nuke */
  /* if NULL pointer passed to free we nuke */
  /* if the calls to free fail that's not my problem */
  return (0);
}

/* This function is currently unused */
PetscErrorCode XXT_stats(xxt_ADT xxt_handle)
{
  PetscInt    op[]  = {NON_UNIFORM, GL_MIN, GL_MAX, GL_ADD, GL_MIN, GL_MAX, GL_ADD, GL_MIN, GL_MAX, GL_ADD};
  PetscInt    fop[] = {NON_UNIFORM, GL_MIN, GL_MAX, GL_ADD};
  PetscInt    vals[9], work[9];
  PetscScalar fvals[3], fwork[3];

  PetscFunctionBegin;
  PetscCall(PCTFS_comm_init());
  PetscCall(check_handle(xxt_handle));

  /* if factorization not done there are no stats */
  if (!xxt_handle->info || !xxt_handle->mvi) {
    if (!PCTFS_my_id) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "XXT_stats() :: no stats available!\n"));
    PetscFunctionReturn(0);
  }

  vals[0] = vals[1] = vals[2] = xxt_handle->info->nnz;
  vals[3] = vals[4] = vals[5] = xxt_handle->mvi->n;
  vals[6] = vals[7] = vals[8] = xxt_handle->info->msg_buf_sz;
  PCTFS_giop(vals, work, PETSC_STATIC_ARRAY_LENGTH(op) - 1, op);

  fvals[0] = fvals[1] = fvals[2] = xxt_handle->info->tot_solve_time / xxt_handle->info->nsolves++;
  PCTFS_grop(fvals, fwork, PETSC_STATIC_ARRAY_LENGTH(fop) - 1, fop);

  if (!PCTFS_my_id) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: min   xxt_nnz=%" PetscInt_FMT "\n", PCTFS_my_id, vals[0]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: max   xxt_nnz=%" PetscInt_FMT "\n", PCTFS_my_id, vals[1]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: avg   xxt_nnz=%g\n", PCTFS_my_id, (double)(1.0 * vals[2] / PCTFS_num_nodes)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: tot   xxt_nnz=%" PetscInt_FMT "\n", PCTFS_my_id, vals[2]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: xxt   C(2d)  =%g\n", PCTFS_my_id, (double)(vals[2] / (PetscPowReal(1.0 * vals[5], 1.5)))));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: xxt   C(3d)  =%g\n", PCTFS_my_id, (double)(vals[2] / (PetscPowReal(1.0 * vals[5], 1.6667)))));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: min   xxt_n  =%" PetscInt_FMT "\n", PCTFS_my_id, vals[3]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: max   xxt_n  =%" PetscInt_FMT "\n", PCTFS_my_id, vals[4]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: avg   xxt_n  =%g\n", PCTFS_my_id, (double)(1.0 * vals[5] / PCTFS_num_nodes)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: tot   xxt_n  =%" PetscInt_FMT "\n", PCTFS_my_id, vals[5]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: min   xxt_buf=%" PetscInt_FMT "\n", PCTFS_my_id, vals[6]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: max   xxt_buf=%" PetscInt_FMT "\n", PCTFS_my_id, vals[7]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: avg   xxt_buf=%g\n", PCTFS_my_id, (double)(1.0 * vals[8] / PCTFS_num_nodes)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: min   xxt_slv=%g\n", PCTFS_my_id, (double)PetscRealPart(fvals[0])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: max   xxt_slv=%g\n", PCTFS_my_id, (double)PetscRealPart(fvals[1])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d :: avg   xxt_slv=%g\n", PCTFS_my_id, (double)PetscRealPart(fvals[2] / PCTFS_num_nodes)));
  }
  PetscFunctionReturn(0);
}

/*

Description: get A_local, local portion of global coarse matrix which
is a row dist. nxm matrix w/ n<m.
   o my_ml holds address of ML struct associated w/A_local and coarse grid
   o local2global holds global number of column i (i=0,...,m-1)
   o local2global holds global number of row    i (i=0,...,n-1)
   o mylocmatvec performs A_local . vec_local (note that gs is performed using
   PCTFS_gs_init/gop).

mylocmatvec = my_ml->Amat[grid_tag].matvec->external;
mylocmatvec (void :: void *data, double *in, double *out)
*/
static PetscErrorCode do_xxt_factor(xxt_ADT xxt_handle)
{
  return xxt_generate(xxt_handle);
}

static PetscErrorCode xxt_generate(xxt_ADT xxt_handle)
{
  PetscInt      i, j, k, idex;
  PetscInt      dim, col;
  PetscScalar  *u, *uu, *v, *z, *w, alpha, alpha_w;
  PetscInt     *segs;
  PetscInt      op[] = {GL_ADD, 0};
  PetscInt      off, len;
  PetscScalar  *x_ptr;
  PetscInt     *iptr, flag;
  PetscInt      start = 0, end, work;
  PetscInt      op2[] = {GL_MIN, 0};
  PCTFS_gs_ADT  PCTFS_gs_handle;
  PetscInt     *nsep, *lnsep, *fo;
  PetscInt      a_n            = xxt_handle->mvi->n;
  PetscInt      a_m            = xxt_handle->mvi->m;
  PetscInt     *a_local2global = xxt_handle->mvi->local2global;
  PetscInt      level;
  PetscInt      xxt_nnz = 0, xxt_max_nnz = 0;
  PetscInt      n, m;
  PetscInt     *col_sz, *col_indices, *stages;
  PetscScalar **col_vals, *x;
  PetscInt      n_global;
  PetscBLASInt  i1  = 1, dlen;
  PetscScalar   dm1 = -1.0;

  n               = xxt_handle->mvi->n;
  nsep            = xxt_handle->info->nsep;
  lnsep           = xxt_handle->info->lnsep;
  fo              = xxt_handle->info->fo;
  end             = lnsep[0];
  level           = xxt_handle->level;
  PCTFS_gs_handle = xxt_handle->mvi->PCTFS_gs_handle;

  /* is there a null space? */
  /* LATER add in ability to detect null space by checking alpha */
  for (i = 0, j = 0; i <= level; i++) j += nsep[i];

  m = j - xxt_handle->ns;
  if (m != j) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "xxt_generate() :: null space exists %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", m, j, xxt_handle->ns));

  /* get and initialize storage for x local         */
  /* note that x local is nxm and stored by columns */
  col_sz      = (PetscInt *)malloc(m * sizeof(PetscInt));
  col_indices = (PetscInt *)malloc((2 * m + 1) * sizeof(PetscInt));
  col_vals    = (PetscScalar **)malloc(m * sizeof(PetscScalar *));
  for (i = j = 0; i < m; i++, j += 2) {
    col_indices[j] = col_indices[j + 1] = col_sz[i] = -1;
    col_vals[i]                                     = NULL;
  }
  col_indices[j] = -1;

  /* size of separators for each sub-hc working from bottom of tree to top */
  /* this looks like nsep[]=segments */
  stages = (PetscInt *)malloc((level + 1) * sizeof(PetscInt));
  segs   = (PetscInt *)malloc((level + 1) * sizeof(PetscInt));
  PCTFS_ivec_zero(stages, level + 1);
  PCTFS_ivec_copy(segs, nsep, level + 1);
  for (i = 0; i < level; i++) segs[i + 1] += segs[i];
  stages[0] = segs[0];

  /* temporary vectors  */
  u  = (PetscScalar *)malloc(n * sizeof(PetscScalar));
  z  = (PetscScalar *)malloc(n * sizeof(PetscScalar));
  v  = (PetscScalar *)malloc(a_m * sizeof(PetscScalar));
  uu = (PetscScalar *)malloc(m * sizeof(PetscScalar));
  w  = (PetscScalar *)malloc(m * sizeof(PetscScalar));

  /* extra nnz due to replication of vertices across separators */
  for (i = 1, j = 0; i <= level; i++) j += nsep[i];

  /* storage for sparse x values */
  n_global    = xxt_handle->info->n_global;
  xxt_max_nnz = (PetscInt)(2.5 * PetscPowReal(1.0 * n_global, 1.6667) + j * n / 2) / PCTFS_num_nodes;
  x           = (PetscScalar *)malloc(xxt_max_nnz * sizeof(PetscScalar));
  xxt_nnz     = 0;

  /* LATER - can embed next sep to fire in gs */
  /* time to make the donuts - generate X factor */
  for (dim = i = j = 0; i < m; i++) {
    /* time to move to the next level? */
    while (i == segs[dim]) {
      PetscCheck(dim != level, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dim about to exceed level");
      stages[dim++] = i;
      end += lnsep[dim];
    }
    stages[dim] = i;

    /* which column are we firing? */
    /* i.e. set v_l */
    /* use new seps and do global min across hc to determine which one to fire */
    (start < end) ? (col = fo[start]) : (col = INT_MAX);
    PCTFS_giop_hc(&col, &work, 1, op2, dim);

    /* shouldn't need this */
    if (col == INT_MAX) {
      PetscCall(PetscInfo(0, "hey ... col==INT_MAX??\n"));
      continue;
    }

    /* do I own it? I should */
    PCTFS_rvec_zero(v, a_m);
    if (col == fo[start]) {
      start++;
      idex = PCTFS_ivec_linear_search(col, a_local2global, a_n);
      if (idex != -1) {
        v[idex] = 1.0;
        j++;
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "NOT FOUND!");
    } else {
      idex = PCTFS_ivec_linear_search(col, a_local2global, a_m);
      if (idex != -1) v[idex] = 1.0;
    }

    /* perform u = A.v_l */
    PCTFS_rvec_zero(u, n);
    do_matvec(xxt_handle->mvi, v, u);

    /* uu =  X^T.u_l (local portion) */
    /* technically only need to zero out first i entries */
    /* later turn this into an XXT_solve call ? */
    PCTFS_rvec_zero(uu, m);
    x_ptr = x;
    iptr  = col_indices;
    for (k = 0; k < i; k++) {
      off = *iptr++;
      len = *iptr++;
      PetscCall(PetscBLASIntCast(len, &dlen));
      PetscCallBLAS("BLASdot", uu[k] = BLASdot_(&dlen, u + off, &i1, x_ptr, &i1));
      x_ptr += len;
    }

    /* uu = X^T.u_l (comm portion) */
    PetscCall(PCTFS_ssgl_radd(uu, w, dim, stages));

    /* z = X.uu */
    PCTFS_rvec_zero(z, n);
    x_ptr = x;
    iptr  = col_indices;
    for (k = 0; k < i; k++) {
      off = *iptr++;
      len = *iptr++;
      PetscCall(PetscBLASIntCast(len, &dlen));
      PetscCallBLAS("BLASaxpy", BLASaxpy_(&dlen, &uu[k], x_ptr, &i1, z + off, &i1));
      x_ptr += len;
    }

    /* compute v_l = v_l - z */
    PCTFS_rvec_zero(v + a_n, a_m - a_n);
    PetscCall(PetscBLASIntCast(n, &dlen));
    PetscCallBLAS("BLASaxpy", BLASaxpy_(&dlen, &dm1, z, &i1, v, &i1));

    /* compute u_l = A.v_l */
    if (a_n != a_m) PCTFS_gs_gop_hc(PCTFS_gs_handle, v, "+\0", dim);
    PCTFS_rvec_zero(u, n);
    do_matvec(xxt_handle->mvi, v, u);

    /* compute sqrt(alpha) = sqrt(v_l^T.u_l) - local portion */
    PetscCall(PetscBLASIntCast(n, &dlen));
    PetscCallBLAS("BLASdot", alpha = BLASdot_(&dlen, u, &i1, v, &i1));
    /* compute sqrt(alpha) = sqrt(v_l^T.u_l) - comm portion */
    PCTFS_grop_hc(&alpha, &alpha_w, 1, op, dim);

    alpha = (PetscScalar)PetscSqrtReal((PetscReal)alpha);

    /* check for small alpha                             */
    /* LATER use this to detect and determine null space */
    PetscCheck(PetscAbsScalar(alpha) >= 1.0e-14, PETSC_COMM_SELF, PETSC_ERR_PLIB, "bad alpha! %g", (double)PetscAbsScalar(alpha));

    /* compute v_l = v_l/sqrt(alpha) */
    PCTFS_rvec_scale(v, 1.0 / alpha, n);

    /* add newly generated column, v_l, to X */
    flag = 1;
    off = len = 0;
    for (k = 0; k < n; k++) {
      if (v[k] != 0.0) {
        len = k;
        if (flag) {
          off  = k;
          flag = 0;
        }
      }
    }

    len -= (off - 1);

    if (len > 0) {
      if ((xxt_nnz + len) > xxt_max_nnz) {
        PetscCall(PetscInfo(0, "increasing space for X by 2x!\n"));
        xxt_max_nnz *= 2;
        x_ptr = (PetscScalar *)malloc(xxt_max_nnz * sizeof(PetscScalar));
        PCTFS_rvec_copy(x_ptr, x, xxt_nnz);
        free(x);
        x = x_ptr;
        x_ptr += xxt_nnz;
      }
      xxt_nnz += len;
      PCTFS_rvec_copy(x_ptr, v + off, len);

      col_indices[2 * i] = off;
      col_sz[i] = col_indices[2 * i + 1] = len;
      col_vals[i]                        = x_ptr;
    } else {
      col_indices[2 * i] = 0;
      col_sz[i] = col_indices[2 * i + 1] = 0;
      col_vals[i]                        = x_ptr;
    }
  }

  /* close off stages for execution phase */
  while (dim != level) {
    stages[dim++] = i;
    PetscCall(PetscInfo(0, "disconnected!!! dim(%" PetscInt_FMT ")!=level(%" PetscInt_FMT ")\n", dim, level));
  }
  stages[dim] = i;

  xxt_handle->info->n              = xxt_handle->mvi->n;
  xxt_handle->info->m              = m;
  xxt_handle->info->nnz            = xxt_nnz;
  xxt_handle->info->max_nnz        = xxt_max_nnz;
  xxt_handle->info->msg_buf_sz     = stages[level] - stages[0];
  xxt_handle->info->solve_uu       = (PetscScalar *)malloc(m * sizeof(PetscScalar));
  xxt_handle->info->solve_w        = (PetscScalar *)malloc(m * sizeof(PetscScalar));
  xxt_handle->info->x              = x;
  xxt_handle->info->col_vals       = col_vals;
  xxt_handle->info->col_sz         = col_sz;
  xxt_handle->info->col_indices    = col_indices;
  xxt_handle->info->stages         = stages;
  xxt_handle->info->nsolves        = 0;
  xxt_handle->info->tot_solve_time = 0.0;

  free(segs);
  free(u);
  free(v);
  free(uu);
  free(z);
  free(w);

  return (0);
}

static PetscErrorCode do_xxt_solve(xxt_ADT xxt_handle, PetscScalar *uc)
{
  PetscInt     off, len, *iptr;
  PetscInt     level       = xxt_handle->level;
  PetscInt     n           = xxt_handle->info->n;
  PetscInt     m           = xxt_handle->info->m;
  PetscInt    *stages      = xxt_handle->info->stages;
  PetscInt    *col_indices = xxt_handle->info->col_indices;
  PetscScalar *x_ptr, *uu_ptr;
  PetscScalar *solve_uu = xxt_handle->info->solve_uu;
  PetscScalar *solve_w  = xxt_handle->info->solve_w;
  PetscScalar *x        = xxt_handle->info->x;
  PetscBLASInt i1       = 1, dlen;

  PetscFunctionBegin;
  uu_ptr = solve_uu;
  PCTFS_rvec_zero(uu_ptr, m);

  /* x  = X.Y^T.b */
  /* uu = Y^T.b */
  for (x_ptr = x, iptr = col_indices; *iptr != -1; x_ptr += len) {
    off = *iptr++;
    len = *iptr++;
    PetscCall(PetscBLASIntCast(len, &dlen));
    PetscCallBLAS("BLASdot", *uu_ptr++ = BLASdot_(&dlen, uc + off, &i1, x_ptr, &i1));
  }

  /* communication of beta */
  uu_ptr = solve_uu;
  if (level) PetscCall(PCTFS_ssgl_radd(uu_ptr, solve_w, level, stages));

  PCTFS_rvec_zero(uc, n);

  /* x = X.uu */
  for (x_ptr = x, iptr = col_indices; *iptr != -1; x_ptr += len) {
    off = *iptr++;
    len = *iptr++;
    PetscCall(PetscBLASIntCast(len, &dlen));
    PetscCallBLAS("BLASaxpy", BLASaxpy_(&dlen, uu_ptr++, x_ptr, &i1, uc + off, &i1));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode check_handle(xxt_ADT xxt_handle)
{
  PetscInt vals[2], work[2], op[] = {NON_UNIFORM, GL_MIN, GL_MAX};

  PetscFunctionBegin;
  PetscCheck(xxt_handle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "check_handle() :: bad handle :: NULL %p", xxt_handle);

  vals[0] = vals[1] = xxt_handle->id;
  PCTFS_giop(vals, work, PETSC_STATIC_ARRAY_LENGTH(op) - 1, op);
  PetscCheck(!(vals[0] != vals[1]) && !(xxt_handle->id <= 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "check_handle() :: bad handle :: id mismatch min/max %" PetscInt_FMT "/%" PetscInt_FMT " %" PetscInt_FMT, vals[0], vals[1], xxt_handle->id);
  PetscFunctionReturn(0);
}

static PetscErrorCode det_separators(xxt_ADT xxt_handle)
{
  PetscInt     i, ct, id;
  PetscInt     mask, edge, *iptr;
  PetscInt    *dir, *used;
  PetscInt     sum[4], w[4];
  PetscScalar  rsum[4], rw[4];
  PetscInt     op[] = {GL_ADD, 0};
  PetscScalar *lhs, *rhs;
  PetscInt    *nsep, *lnsep, *fo, nfo = 0;
  PCTFS_gs_ADT PCTFS_gs_handle = xxt_handle->mvi->PCTFS_gs_handle;
  PetscInt    *local2global    = xxt_handle->mvi->local2global;
  PetscInt     n               = xxt_handle->mvi->n;
  PetscInt     m               = xxt_handle->mvi->m;
  PetscInt     level           = xxt_handle->level;
  PetscInt     shared          = 0;

  PetscFunctionBegin;
  dir   = (PetscInt *)malloc(sizeof(PetscInt) * (level + 1));
  nsep  = (PetscInt *)malloc(sizeof(PetscInt) * (level + 1));
  lnsep = (PetscInt *)malloc(sizeof(PetscInt) * (level + 1));
  fo    = (PetscInt *)malloc(sizeof(PetscInt) * (n + 1));
  used  = (PetscInt *)malloc(sizeof(PetscInt) * n);

  PCTFS_ivec_zero(dir, level + 1);
  PCTFS_ivec_zero(nsep, level + 1);
  PCTFS_ivec_zero(lnsep, level + 1);
  PCTFS_ivec_set(fo, -1, n + 1);
  PCTFS_ivec_zero(used, n);

  lhs = (PetscScalar *)malloc(sizeof(PetscScalar) * m);
  rhs = (PetscScalar *)malloc(sizeof(PetscScalar) * m);

  /* determine the # of unique dof */
  PCTFS_rvec_zero(lhs, m);
  PCTFS_rvec_set(lhs, 1.0, n);
  PCTFS_gs_gop_hc(PCTFS_gs_handle, lhs, "+\0", level);
  PCTFS_rvec_zero(rsum, 2);
  for (i = 0; i < n; i++) {
    if (lhs[i] != 0.0) {
      rsum[0] += 1.0 / lhs[i];
      rsum[1] += lhs[i];
    }
  }
  PCTFS_grop_hc(rsum, rw, 2, op, level);
  rsum[0] += 0.1;
  rsum[1] += 0.1;

  if (PetscAbsScalar(rsum[0] - rsum[1]) > EPS) shared = 1;

  xxt_handle->info->n_global = xxt_handle->info->m_global = (PetscInt)rsum[0];
  xxt_handle->mvi->n_global = xxt_handle->mvi->m_global = (PetscInt)rsum[0];

  /* determine separator sets top down */
  if (shared) {
    for (iptr = fo + n, id = PCTFS_my_id, mask = PCTFS_num_nodes >> 1, edge = level; edge > 0; edge--, mask >>= 1) {
      /* set rsh of hc, fire, and collect lhs responses */
      (id < mask) ? PCTFS_rvec_zero(lhs, m) : PCTFS_rvec_set(lhs, 1.0, m);
      PCTFS_gs_gop_hc(PCTFS_gs_handle, lhs, "+\0", edge);

      /* set lsh of hc, fire, and collect rhs responses */
      (id < mask) ? PCTFS_rvec_set(rhs, 1.0, m) : PCTFS_rvec_zero(rhs, m);
      PCTFS_gs_gop_hc(PCTFS_gs_handle, rhs, "+\0", edge);

      for (i = 0; i < n; i++) {
        if (id < mask) {
          if (lhs[i] != 0.0) lhs[i] = 1.0;
        }
        if (id >= mask) {
          if (rhs[i] != 0.0) rhs[i] = 1.0;
        }
      }

      if (id < mask) PCTFS_gs_gop_hc(PCTFS_gs_handle, lhs, "+\0", edge - 1);
      else PCTFS_gs_gop_hc(PCTFS_gs_handle, rhs, "+\0", edge - 1);

      /* count number of dofs I own that have signal and not in sep set */
      PCTFS_rvec_zero(rsum, 4);
      for (PCTFS_ivec_zero(sum, 4), ct = i = 0; i < n; i++) {
        if (!used[i]) {
          /* number of unmarked dofs on node */
          ct++;
          /* number of dofs to be marked on lhs hc */
          if (id < mask) {
            if (lhs[i] != 0.0) {
              sum[0]++;
              rsum[0] += 1.0 / lhs[i];
            }
          }
          /* number of dofs to be marked on rhs hc */
          if (id >= mask) {
            if (rhs[i] != 0.0) {
              sum[1]++;
              rsum[1] += 1.0 / rhs[i];
            }
          }
        }
      }

      /* go for load balance - choose half with most unmarked dofs, bias LHS */
      (id < mask) ? (sum[2] = ct) : (sum[3] = ct);
      (id < mask) ? (rsum[2] = ct) : (rsum[3] = ct);
      PCTFS_giop_hc(sum, w, 4, op, edge);
      PCTFS_grop_hc(rsum, rw, 4, op, edge);
      rsum[0] += 0.1;
      rsum[1] += 0.1;
      rsum[2] += 0.1;
      rsum[3] += 0.1;

      if (id < mask) {
        /* mark dofs I own that have signal and not in sep set */
        for (ct = i = 0; i < n; i++) {
          if ((!used[i]) && (lhs[i] != 0.0)) {
            ct++;
            nfo++;

            PetscCheck(nfo <= n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nfo about to exceed n");

            *--iptr = local2global[i];
            used[i] = edge;
          }
        }
        if (ct > 1) PCTFS_ivec_sort(iptr, ct);

        lnsep[edge] = ct;
        nsep[edge]  = (PetscInt)rsum[0];
        dir[edge]   = LEFT;
      }

      if (id >= mask) {
        /* mark dofs I own that have signal and not in sep set */
        for (ct = i = 0; i < n; i++) {
          if ((!used[i]) && (rhs[i] != 0.0)) {
            ct++;
            nfo++;

            PetscCheck(nfo <= n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nfo about to exceed n");

            *--iptr = local2global[i];
            used[i] = edge;
          }
        }
        if (ct > 1) PCTFS_ivec_sort(iptr, ct);

        lnsep[edge] = ct;
        nsep[edge]  = (PetscInt)rsum[1];
        dir[edge]   = RIGHT;
      }

      /* LATER or we can recur on these to order seps at this level */
      /* do we need full set of separators for this?                */

      /* fold rhs hc into lower */
      if (id >= mask) id -= mask;
    }
  } else {
    for (iptr = fo + n, id = PCTFS_my_id, mask = PCTFS_num_nodes >> 1, edge = level; edge > 0; edge--, mask >>= 1) {
      /* set rsh of hc, fire, and collect lhs responses */
      (id < mask) ? PCTFS_rvec_zero(lhs, m) : PCTFS_rvec_set(lhs, 1.0, m);
      PCTFS_gs_gop_hc(PCTFS_gs_handle, lhs, "+\0", edge);

      /* set lsh of hc, fire, and collect rhs responses */
      (id < mask) ? PCTFS_rvec_set(rhs, 1.0, m) : PCTFS_rvec_zero(rhs, m);
      PCTFS_gs_gop_hc(PCTFS_gs_handle, rhs, "+\0", edge);

      /* count number of dofs I own that have signal and not in sep set */
      for (PCTFS_ivec_zero(sum, 4), ct = i = 0; i < n; i++) {
        if (!used[i]) {
          /* number of unmarked dofs on node */
          ct++;
          /* number of dofs to be marked on lhs hc */
          if ((id < mask) && (lhs[i] != 0.0)) sum[0]++;
          /* number of dofs to be marked on rhs hc */
          if ((id >= mask) && (rhs[i] != 0.0)) sum[1]++;
        }
      }

      /* go for load balance - choose half with most unmarked dofs, bias LHS */
      (id < mask) ? (sum[2] = ct) : (sum[3] = ct);
      PCTFS_giop_hc(sum, w, 4, op, edge);

      /* lhs hc wins */
      if (sum[2] >= sum[3]) {
        if (id < mask) {
          /* mark dofs I own that have signal and not in sep set */
          for (ct = i = 0; i < n; i++) {
            if ((!used[i]) && (lhs[i] != 0.0)) {
              ct++;
              nfo++;
              *--iptr = local2global[i];
              used[i] = edge;
            }
          }
          if (ct > 1) PCTFS_ivec_sort(iptr, ct);
          lnsep[edge] = ct;
        }
        nsep[edge] = sum[0];
        dir[edge]  = LEFT;
      } else { /* rhs hc wins */
        if (id >= mask) {
          /* mark dofs I own that have signal and not in sep set */
          for (ct = i = 0; i < n; i++) {
            if ((!used[i]) && (rhs[i] != 0.0)) {
              ct++;
              nfo++;
              *--iptr = local2global[i];
              used[i] = edge;
            }
          }
          if (ct > 1) PCTFS_ivec_sort(iptr, ct);
          lnsep[edge] = ct;
        }
        nsep[edge] = sum[1];
        dir[edge]  = RIGHT;
      }
      /* LATER or we can recur on these to order seps at this level */
      /* do we need full set of separators for this?                */

      /* fold rhs hc into lower */
      if (id >= mask) id -= mask;
    }
  }

  /* level 0 is on processor case - so mark the remainder */
  for (ct = i = 0; i < n; i++) {
    if (!used[i]) {
      ct++;
      nfo++;
      *--iptr = local2global[i];
      used[i] = edge;
    }
  }
  if (ct > 1) PCTFS_ivec_sort(iptr, ct);
  lnsep[edge] = ct;
  nsep[edge]  = ct;
  dir[edge]   = LEFT;

  xxt_handle->info->nsep  = nsep;
  xxt_handle->info->lnsep = lnsep;
  xxt_handle->info->fo    = fo;
  xxt_handle->info->nfo   = nfo;

  free(dir);
  free(lhs);
  free(rhs);
  free(used);
  PetscFunctionReturn(0);
}

static mv_info *set_mvi(PetscInt *local2global, PetscInt n, PetscInt m, PetscErrorCode (*matvec)(mv_info *, PetscScalar *, PetscScalar *), void *grid_data)
{
  mv_info *mvi;

  mvi               = (mv_info *)malloc(sizeof(mv_info));
  mvi->n            = n;
  mvi->m            = m;
  mvi->n_global     = -1;
  mvi->m_global     = -1;
  mvi->local2global = (PetscInt *)malloc((m + 1) * sizeof(PetscInt));
  PCTFS_ivec_copy(mvi->local2global, local2global, m);
  mvi->local2global[m] = INT_MAX;
  mvi->matvec          = matvec;
  mvi->grid_data       = grid_data;

  /* set xxt communication handle to perform restricted matvec */
  mvi->PCTFS_gs_handle = PCTFS_gs_init(local2global, m, PCTFS_num_nodes);

  return (mvi);
}

static PetscErrorCode do_matvec(mv_info *A, PetscScalar *v, PetscScalar *u)
{
  PetscFunctionBegin;
  A->matvec((mv_info *)A->grid_data, v, u);
  PetscFunctionReturn(0);
}
