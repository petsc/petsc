
/*************************************xyt.c************************************
Module Name: xyt
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
**************************************xyt.c***********************************/
#include <../src/ksp/pc/impls/tfs/tfs.h>

#define LEFT  -1
#define RIGHT  1
#define BOTH   0

typedef struct xyt_solver_info {
  PetscInt    n, m, n_global, m_global;
  PetscInt    nnz, max_nnz, msg_buf_sz;
  PetscInt    *nsep, *lnsep, *fo, nfo, *stages;
  PetscInt    *xcol_sz, *xcol_indices;
  PetscScalar **xcol_vals, *x, *solve_uu, *solve_w;
  PetscInt    *ycol_sz, *ycol_indices;
  PetscScalar **ycol_vals, *y;
  PetscInt    nsolves;
  PetscScalar tot_solve_time;
} xyt_info;

typedef struct matvec_info {
  PetscInt     n, m, n_global, m_global;
  PetscInt     *local2global;
  PCTFS_gs_ADT PCTFS_gs_handle;
  PetscErrorCode (*matvec)(struct matvec_info*,PetscScalar*,PetscScalar*);
  void *grid_data;
} mv_info;

struct xyt_CDT {
  PetscInt id;
  PetscInt ns;
  PetscInt level;
  xyt_info *info;
  mv_info  *mvi;
};

static PetscInt n_xyt        =0;
static PetscInt n_xyt_handles=0;

/* prototypes */
static PetscErrorCode do_xyt_solve(xyt_ADT xyt_handle, PetscScalar *rhs);
static PetscErrorCode check_handle(xyt_ADT xyt_handle);
static PetscErrorCode det_separators(xyt_ADT xyt_handle);
static PetscErrorCode do_matvec(mv_info *A, PetscScalar *v, PetscScalar *u);
static PetscErrorCode xyt_generate(xyt_ADT xyt_handle);
static PetscErrorCode do_xyt_factor(xyt_ADT xyt_handle);
static mv_info *set_mvi(PetscInt *local2global, PetscInt n, PetscInt m, PetscErrorCode (*matvec)(mv_info*,PetscScalar*,PetscScalar*), void *grid_data);

/**************************************xyt.c***********************************/
xyt_ADT XYT_new(void)
{
  xyt_ADT xyt_handle;

  /* rolling count on n_xyt ... pot. problem here */
  n_xyt_handles++;
  xyt_handle       = (xyt_ADT)malloc(sizeof(struct xyt_CDT));
  xyt_handle->id   = ++n_xyt;
  xyt_handle->info = NULL;
  xyt_handle->mvi  = NULL;

  return(xyt_handle);
}

/**************************************xyt.c***********************************/
PetscErrorCode XYT_factor(xyt_ADT xyt_handle,     /* prev. allocated xyt  handle */
                    PetscInt *local2global, /* global column mapping       */
                    PetscInt n,             /* local num rows              */
                    PetscInt m,             /* local num cols              */
                    PetscErrorCode (*matvec)(void*,PetscScalar*,PetscScalar*), /* b_loc=A_local.x_loc         */
                    void *grid_data)        /* grid data for matvec        */
{

  PCTFS_comm_init();
  check_handle(xyt_handle);

  /* only 2^k for now and all nodes participating */
  PetscCheckFalse((1<<(xyt_handle->level=PCTFS_i_log2_num_nodes))!=PCTFS_num_nodes,PETSC_COMM_SELF,PETSC_ERR_PLIB,"only 2^k for now and MPI_COMM_WORLD!!! %D != %D",1<<PCTFS_i_log2_num_nodes,PCTFS_num_nodes);

  /* space for X info */
  xyt_handle->info = (xyt_info*)malloc(sizeof(xyt_info));

  /* set up matvec handles */
  xyt_handle->mvi = set_mvi(local2global, n, m, (PetscErrorCode (*)(mv_info*,PetscScalar*,PetscScalar*))matvec, grid_data);

  /* matrix is assumed to be of full rank */
  /* LATER we can reset to indicate rank def. */
  xyt_handle->ns=0;

  /* determine separators and generate firing order - NB xyt info set here */
  det_separators(xyt_handle);

  return(do_xyt_factor(xyt_handle));
}

/**************************************xyt.c***********************************/
PetscErrorCode XYT_solve(xyt_ADT xyt_handle, PetscScalar *x, PetscScalar *b)
{
  PCTFS_comm_init();
  check_handle(xyt_handle);

  /* need to copy b into x? */
  if (b) PCTFS_rvec_copy(x,b,xyt_handle->mvi->n);
  return do_xyt_solve(xyt_handle,x);
}

/**************************************xyt.c***********************************/
PetscErrorCode XYT_free(xyt_ADT xyt_handle)
{
  PCTFS_comm_init();
  check_handle(xyt_handle);
  n_xyt_handles--;

  free(xyt_handle->info->nsep);
  free(xyt_handle->info->lnsep);
  free(xyt_handle->info->fo);
  free(xyt_handle->info->stages);
  free(xyt_handle->info->solve_uu);
  free(xyt_handle->info->solve_w);
  free(xyt_handle->info->x);
  free(xyt_handle->info->xcol_vals);
  free(xyt_handle->info->xcol_sz);
  free(xyt_handle->info->xcol_indices);
  free(xyt_handle->info->y);
  free(xyt_handle->info->ycol_vals);
  free(xyt_handle->info->ycol_sz);
  free(xyt_handle->info->ycol_indices);
  free(xyt_handle->info);
  free(xyt_handle->mvi->local2global);
  PCTFS_gs_free(xyt_handle->mvi->PCTFS_gs_handle);
  free(xyt_handle->mvi);
  free(xyt_handle);

  /* if the check fails we nuke */
  /* if NULL pointer passed to free we nuke */
  /* if the calls to free fail that's not my problem */
  return(0);
}

/**************************************xyt.c***********************************/
PetscErrorCode XYT_stats(xyt_ADT xyt_handle)
{
  PetscInt    op[]  = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD,GL_MIN,GL_MAX,GL_ADD,GL_MIN,GL_MAX,GL_ADD};
  PetscInt    fop[] = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD};
  PetscInt    vals[9],  work[9];
  PetscScalar fvals[3], fwork[3];

  PCTFS_comm_init();
  check_handle(xyt_handle);

  /* if factorization not done there are no stats */
  if (!xyt_handle->info||!xyt_handle->mvi) {
    if (!PCTFS_my_id) PetscPrintf(PETSC_COMM_WORLD,"XYT_stats() :: no stats available!\n");
    return 1;
  }

  vals[0]=vals[1]=vals[2]=xyt_handle->info->nnz;
  vals[3]=vals[4]=vals[5]=xyt_handle->mvi->n;
  vals[6]=vals[7]=vals[8]=xyt_handle->info->msg_buf_sz;
  PCTFS_giop(vals,work,PETSC_STATIC_ARRAY_LENGTH(op)-1,op);

  fvals[0]=fvals[1]=fvals[2]=xyt_handle->info->tot_solve_time/xyt_handle->info->nsolves++;
  PCTFS_grop(fvals,fwork,PETSC_STATIC_ARRAY_LENGTH(fop)-1,fop);

  if (!PCTFS_my_id) {
    PetscPrintf(PETSC_COMM_WORLD,"%D :: min   xyt_nnz=%D\n",PCTFS_my_id,vals[0]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: max   xyt_nnz=%D\n",PCTFS_my_id,vals[1]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: avg   xyt_nnz=%g\n",PCTFS_my_id,1.0*vals[2]/PCTFS_num_nodes);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: tot   xyt_nnz=%D\n",PCTFS_my_id,vals[2]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: xyt   C(2d)  =%g\n",PCTFS_my_id,vals[2]/(PetscPowReal(1.0*vals[5],1.5)));
    PetscPrintf(PETSC_COMM_WORLD,"%D :: xyt   C(3d)  =%g\n",PCTFS_my_id,vals[2]/(PetscPowReal(1.0*vals[5],1.6667)));
    PetscPrintf(PETSC_COMM_WORLD,"%D :: min   xyt_n  =%D\n",PCTFS_my_id,vals[3]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: max   xyt_n  =%D\n",PCTFS_my_id,vals[4]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: avg   xyt_n  =%g\n",PCTFS_my_id,1.0*vals[5]/PCTFS_num_nodes);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: tot   xyt_n  =%D\n",PCTFS_my_id,vals[5]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: min   xyt_buf=%D\n",PCTFS_my_id,vals[6]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: max   xyt_buf=%D\n",PCTFS_my_id,vals[7]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: avg   xyt_buf=%g\n",PCTFS_my_id,1.0*vals[8]/PCTFS_num_nodes);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: min   xyt_slv=%g\n",PCTFS_my_id,fvals[0]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: max   xyt_slv=%g\n",PCTFS_my_id,fvals[1]);
    PetscPrintf(PETSC_COMM_WORLD,"%D :: avg   xyt_slv=%g\n",PCTFS_my_id,fvals[2]/PCTFS_num_nodes);
  }

  return(0);
}

/*************************************xyt.c************************************

Description: get A_local, local portion of global coarse matrix which
is a row dist. nxm matrix w/ n<m.
   o my_ml holds address of ML struct associated w/A_local and coarse grid
   o local2global holds global number of column i (i=0,...,m-1)
   o local2global holds global number of row    i (i=0,...,n-1)
   o mylocmatvec performs A_local . vec_local (note that gs is performed using
   PCTFS_gs_init/gop).

mylocmatvec = my_ml->Amat[grid_tag].matvec->external;
mylocmatvec (void :: void *data, double *in, double *out)
**************************************xyt.c***********************************/
static PetscErrorCode do_xyt_factor(xyt_ADT xyt_handle)
{
  return xyt_generate(xyt_handle);
}

/**************************************xyt.c***********************************/
static PetscErrorCode xyt_generate(xyt_ADT xyt_handle)
{
  PetscInt       i,j,k,idx;
  PetscInt       dim, col;
  PetscScalar    *u, *uu, *v, *z, *w, alpha, alpha_w;
  PetscInt       *segs;
  PetscInt       op[] = {GL_ADD,0};
  PetscInt       off, len;
  PetscScalar    *x_ptr, *y_ptr;
  PetscInt       *iptr, flag;
  PetscInt       start =0, end, work;
  PetscInt       op2[] = {GL_MIN,0};
  PCTFS_gs_ADT   PCTFS_gs_handle;
  PetscInt       *nsep, *lnsep, *fo;
  PetscInt       a_n            =xyt_handle->mvi->n;
  PetscInt       a_m            =xyt_handle->mvi->m;
  PetscInt       *a_local2global=xyt_handle->mvi->local2global;
  PetscInt       level;
  PetscInt       n, m;
  PetscInt       *xcol_sz, *xcol_indices, *stages;
  PetscScalar    **xcol_vals, *x;
  PetscInt       *ycol_sz, *ycol_indices;
  PetscScalar    **ycol_vals, *y;
  PetscInt       n_global;
  PetscInt       xt_nnz       =0, xt_max_nnz=0;
  PetscInt       yt_nnz       =0, yt_max_nnz=0;
  PetscInt       xt_zero_nnz  =0;
  PetscInt       xt_zero_nnz_0=0;
  PetscInt       yt_zero_nnz  =0;
  PetscInt       yt_zero_nnz_0=0;
  PetscBLASInt   i1           = 1,dlen;
  PetscScalar    dm1          = -1.0;

  n              =xyt_handle->mvi->n;
  nsep           =xyt_handle->info->nsep;
  lnsep          =xyt_handle->info->lnsep;
  fo             =xyt_handle->info->fo;
  end            =lnsep[0];
  level          =xyt_handle->level;
  PCTFS_gs_handle=xyt_handle->mvi->PCTFS_gs_handle;

  /* is there a null space? */
  /* LATER add in ability to detect null space by checking alpha */
  for (i=0, j=0; i<=level; i++) j+=nsep[i];

  m = j-xyt_handle->ns;
  if (m!=j) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"xyt_generate() :: null space exists %D %D %D\n",m,j,xyt_handle->ns));
  }

  PetscCall(PetscInfo(0,"xyt_generate() :: X(%D,%D)\n",n,m));

  /* get and initialize storage for x local         */
  /* note that x local is nxm and stored by columns */
  xcol_sz      = (PetscInt*) malloc(m*sizeof(PetscInt));
  xcol_indices = (PetscInt*) malloc((2*m+1)*sizeof(PetscInt));
  xcol_vals    = (PetscScalar**) malloc(m*sizeof(PetscScalar*));
  for (i=j=0; i<m; i++, j+=2) {
    xcol_indices[j]=xcol_indices[j+1]=xcol_sz[i]=-1;
    xcol_vals[i]   = NULL;
  }
  xcol_indices[j]=-1;

  /* get and initialize storage for y local         */
  /* note that y local is nxm and stored by columns */
  ycol_sz      = (PetscInt*) malloc(m*sizeof(PetscInt));
  ycol_indices = (PetscInt*) malloc((2*m+1)*sizeof(PetscInt));
  ycol_vals    = (PetscScalar**) malloc(m*sizeof(PetscScalar*));
  for (i=j=0; i<m; i++, j+=2) {
    ycol_indices[j]=ycol_indices[j+1]=ycol_sz[i]=-1;
    ycol_vals[i]   = NULL;
  }
  ycol_indices[j]=-1;

  /* size of separators for each sub-hc working from bottom of tree to top */
  /* this looks like nsep[]=segments */
  stages = (PetscInt*) malloc((level+1)*sizeof(PetscInt));
  segs   = (PetscInt*) malloc((level+1)*sizeof(PetscInt));
  PCTFS_ivec_zero(stages,level+1);
  PCTFS_ivec_copy(segs,nsep,level+1);
  for (i=0; i<level; i++) segs[i+1] += segs[i];
  stages[0] = segs[0];

  /* temporary vectors  */
  u  = (PetscScalar*) malloc(n*sizeof(PetscScalar));
  z  = (PetscScalar*) malloc(n*sizeof(PetscScalar));
  v  = (PetscScalar*) malloc(a_m*sizeof(PetscScalar));
  uu = (PetscScalar*) malloc(m*sizeof(PetscScalar));
  w  = (PetscScalar*) malloc(m*sizeof(PetscScalar));

  /* extra nnz due to replication of vertices across separators */
  for (i=1, j=0; i<=level; i++) j+=nsep[i];

  /* storage for sparse x values */
  n_global   = xyt_handle->info->n_global;
  xt_max_nnz = yt_max_nnz = (PetscInt)(2.5*PetscPowReal(1.0*n_global,1.6667) + j*n/2)/PCTFS_num_nodes;
  x          = (PetscScalar*) malloc(xt_max_nnz*sizeof(PetscScalar));
  y          = (PetscScalar*) malloc(yt_max_nnz*sizeof(PetscScalar));

  /* LATER - can embed next sep to fire in gs */
  /* time to make the donuts - generate X factor */
  for (dim=i=j=0; i<m; i++) {
    /* time to move to the next level? */
    while (i==segs[dim]) {
      PetscCheck(dim!=level,PETSC_COMM_SELF,PETSC_ERR_PLIB,"dim about to exceed level");
      stages[dim++]=i;
      end         +=lnsep[dim];
    }
    stages[dim]=i;

    /* which column are we firing? */
    /* i.e. set v_l */
    /* use new seps and do global min across hc to determine which one to fire */
    (start<end) ? (col=fo[start]) : (col=INT_MAX);
    PCTFS_giop_hc(&col,&work,1,op2,dim);

    /* shouldn't need this */
    if (col==INT_MAX) {
      PetscCall(PetscInfo(0,"hey ... col==INT_MAX??\n"));
      continue;
    }

    /* do I own it? I should */
    PCTFS_rvec_zero(v,a_m);
    if (col==fo[start]) {
      start++;
      idx=PCTFS_ivec_linear_search(col, a_local2global, a_n);
      if (idx!=-1) {
        v[idx] = 1.0;
        j++;
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"NOT FOUND!");
    } else {
      idx=PCTFS_ivec_linear_search(col, a_local2global, a_m);
      if (idx!=-1) v[idx] = 1.0;
    }

    /* perform u = A.v_l */
    PCTFS_rvec_zero(u,n);
    do_matvec(xyt_handle->mvi,v,u);

    /* uu =  X^T.u_l (local portion) */
    /* technically only need to zero out first i entries */
    /* later turn this into an XYT_solve call ? */
    PCTFS_rvec_zero(uu,m);
    y_ptr=y;
    iptr = ycol_indices;
    for (k=0; k<i; k++) {
      off   = *iptr++;
      len   = *iptr++;
      PetscCall(PetscBLASIntCast(len,&dlen));
      PetscStackCallBLAS("BLASdot",uu[k] = BLASdot_(&dlen,u+off,&i1,y_ptr,&i1));
      y_ptr+=len;
    }

    /* uu = X^T.u_l (comm portion) */
    PetscCall(PCTFS_ssgl_radd  (uu, w, dim, stages));

    /* z = X.uu */
    PCTFS_rvec_zero(z,n);
    x_ptr=x;
    iptr = xcol_indices;
    for (k=0; k<i; k++) {
      off  = *iptr++;
      len  = *iptr++;
      PetscCall(PetscBLASIntCast(len,&dlen));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&dlen,&uu[k],x_ptr,&i1,z+off,&i1));
      x_ptr+=len;
    }

    /* compute v_l = v_l - z */
    PCTFS_rvec_zero(v+a_n,a_m-a_n);
    PetscCall(PetscBLASIntCast(n,&dlen));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&dlen,&dm1,z,&i1,v,&i1));

    /* compute u_l = A.v_l */
    if (a_n!=a_m) PCTFS_gs_gop_hc(PCTFS_gs_handle,v,"+\0",dim);
    PCTFS_rvec_zero(u,n);
    do_matvec(xyt_handle->mvi,v,u);

    /* compute sqrt(alpha) = sqrt(u_l^T.u_l) - local portion */
    PetscCall(PetscBLASIntCast(n,&dlen));
    PetscStackCallBLAS("BLASdot",alpha = BLASdot_(&dlen,u,&i1,u,&i1));
    /* compute sqrt(alpha) = sqrt(u_l^T.u_l) - comm portion */
    PCTFS_grop_hc(&alpha, &alpha_w, 1, op, dim);

    alpha = (PetscScalar) PetscSqrtReal((PetscReal)alpha);

    /* check for small alpha                             */
    /* LATER use this to detect and determine null space */
    PetscCheck(PetscAbsScalar(alpha)>=1.0e-14,PETSC_COMM_SELF,PETSC_ERR_PLIB,"bad alpha! %g",alpha);

    /* compute v_l = v_l/sqrt(alpha) */
    PCTFS_rvec_scale(v,1.0/alpha,n);
    PCTFS_rvec_scale(u,1.0/alpha,n);

    /* add newly generated column, v_l, to X */
    flag = 1;
    off  =len=0;
    for (k=0; k<n; k++) {
      if (v[k]!=0.0) {
        len=k;
        if (flag) {off=k; flag=0;}
      }
    }

    len -= (off-1);

    if (len>0) {
      if ((xt_nnz+len)>xt_max_nnz) {
        PetscCall(PetscInfo(0,"increasing space for X by 2x!\n"));
        xt_max_nnz *= 2;
        x_ptr       = (PetscScalar*) malloc(xt_max_nnz*sizeof(PetscScalar));
        PCTFS_rvec_copy(x_ptr,x,xt_nnz);
        free(x);
        x     = x_ptr;
        x_ptr+=xt_nnz;
      }
      xt_nnz += len;
      PCTFS_rvec_copy(x_ptr,v+off,len);

      /* keep track of number of zeros */
      if (dim) {
        for (k=0; k<len; k++) {
          if (x_ptr[k]==0.0) xt_zero_nnz++;
        }
      } else {
        for (k=0; k<len; k++) {
          if (x_ptr[k]==0.0) xt_zero_nnz_0++;
        }
      }
      xcol_indices[2*i] = off;
      xcol_sz[i]        = xcol_indices[2*i+1] = len;
      xcol_vals[i]      = x_ptr;
    } else {
      xcol_indices[2*i] = 0;
      xcol_sz[i]        = xcol_indices[2*i+1] = 0;
      xcol_vals[i]      = x_ptr;
    }

    /* add newly generated column, u_l, to Y */
    flag = 1;
    off  =len=0;
    for (k=0; k<n; k++) {
      if (u[k]!=0.0) {
        len=k;
        if (flag) { off=k; flag=0; }
      }
    }

    len -= (off-1);

    if (len>0) {
      if ((yt_nnz+len)>yt_max_nnz) {
        PetscCall(PetscInfo(0,"increasing space for Y by 2x!\n"));
        yt_max_nnz *= 2;
        y_ptr       = (PetscScalar*) malloc(yt_max_nnz*sizeof(PetscScalar));
        PCTFS_rvec_copy(y_ptr,y,yt_nnz);
        free(y);
        y     = y_ptr;
        y_ptr+=yt_nnz;
      }
      yt_nnz += len;
      PCTFS_rvec_copy(y_ptr,u+off,len);

      /* keep track of number of zeros */
      if (dim) {
        for (k=0; k<len; k++) {
          if (y_ptr[k]==0.0) yt_zero_nnz++;
        }
      } else {
        for (k=0; k<len; k++) {
          if (y_ptr[k]==0.0) yt_zero_nnz_0++;
        }
      }
      ycol_indices[2*i] = off;
      ycol_sz[i]        = ycol_indices[2*i+1] = len;
      ycol_vals[i]      = y_ptr;
    } else {
      ycol_indices[2*i] = 0;
      ycol_sz[i]        = ycol_indices[2*i+1] = 0;
      ycol_vals[i]      = y_ptr;
    }
  }

  /* close off stages for execution phase */
  while (dim!=level) {
    stages[dim++]=i;
    PetscCall(PetscInfo(0,"disconnected!!! dim(%D)!=level(%D)\n",dim,level));
  }
  stages[dim]=i;

  xyt_handle->info->n           =xyt_handle->mvi->n;
  xyt_handle->info->m           =m;
  xyt_handle->info->nnz         =xt_nnz + yt_nnz;
  xyt_handle->info->max_nnz     =xt_max_nnz + yt_max_nnz;
  xyt_handle->info->msg_buf_sz  =stages[level]-stages[0];
  xyt_handle->info->solve_uu    = (PetscScalar*) malloc(m*sizeof(PetscScalar));
  xyt_handle->info->solve_w     = (PetscScalar*) malloc(m*sizeof(PetscScalar));
  xyt_handle->info->x           =x;
  xyt_handle->info->xcol_vals   =xcol_vals;
  xyt_handle->info->xcol_sz     =xcol_sz;
  xyt_handle->info->xcol_indices=xcol_indices;
  xyt_handle->info->stages      =stages;
  xyt_handle->info->y           =y;
  xyt_handle->info->ycol_vals   =ycol_vals;
  xyt_handle->info->ycol_sz     =ycol_sz;
  xyt_handle->info->ycol_indices=ycol_indices;

  free(segs);
  free(u);
  free(v);
  free(uu);
  free(z);
  free(w);

  return(0);
}

/**************************************xyt.c***********************************/
static PetscErrorCode do_xyt_solve(xyt_ADT xyt_handle,  PetscScalar *uc)
{
  PetscInt       off, len, *iptr;
  PetscInt       level        =xyt_handle->level;
  PetscInt       n            =xyt_handle->info->n;
  PetscInt       m            =xyt_handle->info->m;
  PetscInt       *stages      =xyt_handle->info->stages;
  PetscInt       *xcol_indices=xyt_handle->info->xcol_indices;
  PetscInt       *ycol_indices=xyt_handle->info->ycol_indices;
  PetscScalar    *x_ptr, *y_ptr, *uu_ptr;
  PetscScalar    *solve_uu=xyt_handle->info->solve_uu;
  PetscScalar    *solve_w =xyt_handle->info->solve_w;
  PetscScalar    *x       =xyt_handle->info->x;
  PetscScalar    *y       =xyt_handle->info->y;
  PetscBLASInt   i1       = 1,dlen;

  PetscFunctionBegin;
  uu_ptr=solve_uu;
  PCTFS_rvec_zero(uu_ptr,m);

  /* x  = X.Y^T.b */
  /* uu = Y^T.b */
  for (y_ptr=y,iptr=ycol_indices; *iptr!=-1; y_ptr+=len) {
    off       =*iptr++;
    len       =*iptr++;
    PetscCall(PetscBLASIntCast(len,&dlen));
    PetscStackCallBLAS("BLASdot",*uu_ptr++ = BLASdot_(&dlen,uc+off,&i1,y_ptr,&i1));
  }

  /* comunication of beta */
  uu_ptr=solve_uu;
  if (level) PetscCall(PCTFS_ssgl_radd(uu_ptr, solve_w, level, stages));
  PCTFS_rvec_zero(uc,n);

  /* x = X.uu */
  for (x_ptr=x,iptr=xcol_indices; *iptr!=-1; x_ptr+=len) {
    off  =*iptr++;
    len  =*iptr++;
    PetscCall(PetscBLASIntCast(len,&dlen));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&dlen,uu_ptr++,x_ptr,&i1,uc+off,&i1));
  }
  PetscFunctionReturn(0);
}

/**************************************xyt.c***********************************/
static PetscErrorCode check_handle(xyt_ADT xyt_handle)
{
  PetscInt vals[2], work[2], op[] = {NON_UNIFORM,GL_MIN,GL_MAX};

  PetscFunctionBegin;
  PetscCheck(xyt_handle,PETSC_COMM_SELF,PETSC_ERR_PLIB,"check_handle() :: bad handle :: NULL %D",xyt_handle);

  vals[0]=vals[1]=xyt_handle->id;
  PCTFS_giop(vals,work,PETSC_STATIC_ARRAY_LENGTH(op)-1,op);
  PetscCheck(!(vals[0]!=vals[1])&&!(xyt_handle->id<=0),PETSC_COMM_SELF,PETSC_ERR_PLIB,"check_handle() :: bad handle :: id mismatch min/max %D/%D %D", vals[0],vals[1], xyt_handle->id);
  PetscFunctionReturn(0);
}

/**************************************xyt.c***********************************/
static PetscErrorCode det_separators(xyt_ADT xyt_handle)
{
  PetscInt       i, ct, id;
  PetscInt       mask, edge, *iptr;
  PetscInt       *dir, *used;
  PetscInt       sum[4], w[4];
  PetscScalar    rsum[4], rw[4];
  PetscInt       op[] = {GL_ADD,0};
  PetscScalar    *lhs, *rhs;
  PetscInt       *nsep, *lnsep, *fo, nfo=0;
  PCTFS_gs_ADT   PCTFS_gs_handle=xyt_handle->mvi->PCTFS_gs_handle;
  PetscInt       *local2global  =xyt_handle->mvi->local2global;
  PetscInt       n              =xyt_handle->mvi->n;
  PetscInt       m              =xyt_handle->mvi->m;
  PetscInt       level          =xyt_handle->level;
  PetscInt       shared         =0;

  PetscFunctionBegin;
  dir  = (PetscInt*)malloc(sizeof(PetscInt)*(level+1));
  nsep = (PetscInt*)malloc(sizeof(PetscInt)*(level+1));
  lnsep= (PetscInt*)malloc(sizeof(PetscInt)*(level+1));
  fo   = (PetscInt*)malloc(sizeof(PetscInt)*(n+1));
  used = (PetscInt*)malloc(sizeof(PetscInt)*n);

  PCTFS_ivec_zero(dir,level+1);
  PCTFS_ivec_zero(nsep,level+1);
  PCTFS_ivec_zero(lnsep,level+1);
  PCTFS_ivec_set (fo,-1,n+1);
  PCTFS_ivec_zero(used,n);

  lhs = (PetscScalar*)malloc(sizeof(PetscScalar)*m);
  rhs = (PetscScalar*)malloc(sizeof(PetscScalar)*m);

  /* determine the # of unique dof */
  PCTFS_rvec_zero(lhs,m);
  PCTFS_rvec_set(lhs,1.0,n);
  PCTFS_gs_gop_hc(PCTFS_gs_handle,lhs,"+\0",level);
  PetscCall(PetscInfo(0,"done first PCTFS_gs_gop_hc\n"));
  PCTFS_rvec_zero(rsum,2);
  for (i=0; i<n; i++) {
    if (lhs[i]!=0.0) { rsum[0]+=1.0/lhs[i]; rsum[1]+=lhs[i]; }
    if (lhs[i]!=1.0) shared=1;
  }

  PCTFS_grop_hc(rsum,rw,2,op,level);
  rsum[0]+=0.1;
  rsum[1]+=0.1;

  xyt_handle->info->n_global=xyt_handle->info->m_global=(PetscInt) rsum[0];
  xyt_handle->mvi->n_global =xyt_handle->mvi->m_global =(PetscInt) rsum[0];

  /* determine separator sets top down */
  if (shared) {
    /* solution is to do as in the symmetric shared case but then */
    /* pick the sub-hc with the most free dofs and do a mat-vec   */
    /* and pick up the responses on the other sub-hc from the     */
    /* initial separator set obtained from the symm. shared case  */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"shared dof separator determination not ready ... see hmt!!!");
    /* [dead code deleted since it is unlikely to be completed] */
  } else {
    for (iptr=fo+n,id=PCTFS_my_id,mask=PCTFS_num_nodes>>1,edge=level;edge>0;edge--,mask>>=1) {
      /* set rsh of hc, fire, and collect lhs responses */
      (id<mask) ? PCTFS_rvec_zero(lhs,m) : PCTFS_rvec_set(lhs,1.0,m);
      PCTFS_gs_gop_hc(PCTFS_gs_handle,lhs,"+\0",edge);

      /* set lsh of hc, fire, and collect rhs responses */
      (id<mask) ? PCTFS_rvec_set(rhs,1.0,m) : PCTFS_rvec_zero(rhs,m);
      PCTFS_gs_gop_hc(PCTFS_gs_handle,rhs,"+\0",edge);

      /* count number of dofs I own that have signal and not in sep set */
      for (PCTFS_ivec_zero(sum,4),ct=i=0;i<n;i++) {
        if (!used[i]) {
          /* number of unmarked dofs on node */
          ct++;
          /* number of dofs to be marked on lhs hc */
          if ((id< mask)&&(lhs[i]!=0.0)) sum[0]++;
          /* number of dofs to be marked on rhs hc */
          if ((id>=mask)&&(rhs[i]!=0.0)) sum[1]++;
        }
      }

      /* for the non-symmetric case we need separators of width 2 */
      /* so take both sides */
      (id<mask) ? (sum[2]=ct) : (sum[3]=ct);
      PCTFS_giop_hc(sum,w,4,op,edge);

      ct=0;
      if (id<mask) {
        /* mark dofs I own that have signal and not in sep set */
        for (i=0;i<n;i++) {
          if ((!used[i])&&(lhs[i]!=0.0)) {
            ct++; nfo++;
            *--iptr = local2global[i];
            used[i] =edge;
          }
        }
        /* LSH hc summation of ct should be sum[0] */
      } else {
        /* mark dofs I own that have signal and not in sep set */
        for (i=0; i<n; i++) {
          if ((!used[i])&&(rhs[i]!=0.0)) {
            ct++; nfo++;
            *--iptr = local2global[i];
            used[i] = edge;
          }
        }
        /* RSH hc summation of ct should be sum[1] */
      }

      if (ct>1) PCTFS_ivec_sort(iptr,ct);
      lnsep[edge]=ct;
      nsep[edge] =sum[0]+sum[1];
      dir [edge] =BOTH;

      /* LATER or we can recur on these to order seps at this level */
      /* do we need full set of separators for this?                */

      /* fold rhs hc into lower */
      if (id>=mask) id-=mask;
    }
  }

  /* level 0 is on processor case - so mark the remainder */
  for (ct=i=0;i<n;i++) {
    if (!used[i]) {
      ct++; nfo++;
      *--iptr = local2global[i];
      used[i] = edge;
    }
  }
  if (ct>1) PCTFS_ivec_sort(iptr,ct);
  lnsep[edge]=ct;
  nsep [edge]=ct;
  dir  [edge]=BOTH;

  xyt_handle->info->nsep  = nsep;
  xyt_handle->info->lnsep = lnsep;
  xyt_handle->info->fo    = fo;
  xyt_handle->info->nfo   = nfo;

  free(dir);
  free(lhs);
  free(rhs);
  free(used);
  PetscFunctionReturn(0);
}

/**************************************xyt.c***********************************/
static mv_info *set_mvi(PetscInt *local2global, PetscInt n, PetscInt m, PetscErrorCode (*matvec)(mv_info*,PetscScalar*,PetscScalar*), void *grid_data)
{
  mv_info *mvi;

  mvi              = (mv_info*)malloc(sizeof(mv_info));
  mvi->n           = n;
  mvi->m           = m;
  mvi->n_global    = -1;
  mvi->m_global    = -1;
  mvi->local2global= (PetscInt*)malloc((m+1)*sizeof(PetscInt));

  PCTFS_ivec_copy(mvi->local2global,local2global,m);
  mvi->local2global[m] = INT_MAX;
  mvi->matvec          = matvec;
  mvi->grid_data       = grid_data;

  /* set xyt communication handle to perform restricted matvec */
  mvi->PCTFS_gs_handle = PCTFS_gs_init(local2global, m, PCTFS_num_nodes);

  return(mvi);
}

/**************************************xyt.c***********************************/
static PetscErrorCode do_matvec(mv_info *A, PetscScalar *v, PetscScalar *u)
{
  PetscFunctionBegin;
  A->matvec((mv_info*)A->grid_data,v,u);
  PetscFunctionReturn(0);
}
