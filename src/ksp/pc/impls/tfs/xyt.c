#define PETSCKSP_DLL

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


/*************************************xyt.c************************************
NOTES ON USAGE: 

**************************************xyt.c***********************************/
#include "src/ksp/pc/impls/tfs/tfs.h"

#define LEFT  -1
#define RIGHT  1
#define BOTH   0
#define MAX_FORTRAN_HANDLES  10

typedef struct xyt_solver_info {
  int n, m, n_global, m_global;
  int nnz, max_nnz, msg_buf_sz;
  int *nsep, *lnsep, *fo, nfo, *stages;
  int *xcol_sz, *xcol_indices; 
  PetscScalar **xcol_vals, *x, *solve_uu, *solve_w;
  int *ycol_sz, *ycol_indices; 
  PetscScalar **ycol_vals, *y;
  int nsolves;
  PetscScalar tot_solve_time;
} xyt_info;

 
typedef struct matvec_info {
  int n, m, n_global, m_global;
  int *local2global;
  gs_ADT gs_handle;
  PetscErrorCode (*matvec)(struct matvec_info*,PetscScalar*,PetscScalar*);
  void *grid_data;
} mv_info;

struct xyt_CDT{
  int id;
  int ns;
  int level;
  xyt_info *info;
  mv_info  *mvi;
};

static int n_xyt=0;
static int n_xyt_handles=0;

/* prototypes */
static void do_xyt_solve(xyt_ADT xyt_handle, PetscScalar *rhs);
static void check_init(void);
static void check_handle(xyt_ADT xyt_handle);
static void det_separators(xyt_ADT xyt_handle);
static void do_matvec(mv_info *A, PetscScalar *v, PetscScalar *u);
static int xyt_generate(xyt_ADT xyt_handle);
static int do_xyt_factor(xyt_ADT xyt_handle);
static mv_info *set_mvi(int *local2global, int n, int m, void *matvec, void *grid_data);


/*************************************xyt.c************************************
Function: XYT_new()

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
xyt_ADT 
XYT_new(void)
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


/*************************************xyt.c************************************
Function: XYT_factor()

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
int
XYT_factor(xyt_ADT xyt_handle, /* prev. allocated xyt  handle */
	   int *local2global,  /* global column mapping       */
	   int n,              /* local num rows              */
	   int m,              /* local num cols              */
	   void *matvec,       /* b_loc=A_local.x_loc         */
	   void *grid_data     /* grid data for matvec        */
	   )
{

  check_init();
  check_handle(xyt_handle);

  /* only 2^k for now and all nodes participating */
  if ((1<<(xyt_handle->level=i_log2_num_nodes))!=num_nodes)
    {error_msg_fatal("only 2^k for now and MPI_COMM_WORLD!!! %d != %d\n",1<<i_log2_num_nodes,num_nodes);}

  /* space for X info */
  xyt_handle->info = (xyt_info*)malloc(sizeof(xyt_info));

  /* set up matvec handles */
  xyt_handle->mvi  = set_mvi(local2global, n, m, matvec, grid_data);

  /* matrix is assumed to be of full rank */
  /* LATER we can reset to indicate rank def. */
  xyt_handle->ns=0;

  /* determine separators and generate firing order - NB xyt info set here */
  det_separators(xyt_handle);

  return(do_xyt_factor(xyt_handle));
}


/*************************************xyt.c************************************
Function: XYT_solve

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
int
XYT_solve(xyt_ADT xyt_handle, double *x, double *b)
{
  check_init();
  check_handle(xyt_handle);

  /* need to copy b into x? */
  if (b)
    {rvec_copy(x,b,xyt_handle->mvi->n);}
  do_xyt_solve(xyt_handle,x);

  return(0);
}


/*************************************xyt.c************************************
Function: XYT_free()

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
int
XYT_free(xyt_ADT xyt_handle)
{
  check_init();
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
   gs_free(xyt_handle->mvi->gs_handle);
  free(xyt_handle->mvi);
  free(xyt_handle);

 
  /* if the check fails we nuke */
  /* if NULL pointer passed to free we nuke */
  /* if the calls to free fail that's not my problem */
  return(0);
}



/*************************************xyt.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
**************************************xyt.c***********************************/
int
XYT_stats(xyt_ADT xyt_handle)
{
  int  op[] = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD,GL_MIN,GL_MAX,GL_ADD,GL_MIN,GL_MAX,GL_ADD};
  int fop[] = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD};
  int   vals[9],  work[9];
  PetscScalar fvals[3], fwork[3];


  check_init();
  check_handle(xyt_handle);

  /* if factorization not done there are no stats */
  if (!xyt_handle->info||!xyt_handle->mvi)
    {
      if (!my_id) 
	{printf("XYT_stats() :: no stats available!\n");}
      return 1;
    }

  vals[0]=vals[1]=vals[2]=xyt_handle->info->nnz;
  vals[3]=vals[4]=vals[5]=xyt_handle->mvi->n;
  vals[6]=vals[7]=vals[8]=xyt_handle->info->msg_buf_sz;
  giop(vals,work,sizeof(op)/sizeof(op[0])-1,op);

  fvals[0]=fvals[1]=fvals[2]
    =xyt_handle->info->tot_solve_time/xyt_handle->info->nsolves++;
  grop(fvals,fwork,sizeof(fop)/sizeof(fop[0])-1,fop);

  if (!my_id) 
    {
      printf("%d :: min   xyt_nnz=%d\n",my_id,vals[0]);
      printf("%d :: max   xyt_nnz=%d\n",my_id,vals[1]);
      printf("%d :: avg   xyt_nnz=%g\n",my_id,1.0*vals[2]/num_nodes);
      printf("%d :: tot   xyt_nnz=%d\n",my_id,vals[2]);
      printf("%d :: xyt   C(2d)  =%g\n",my_id,vals[2]/(pow(1.0*vals[5],1.5)));
      printf("%d :: xyt   C(3d)  =%g\n",my_id,vals[2]/(pow(1.0*vals[5],1.6667)));
      printf("%d :: min   xyt_n  =%d\n",my_id,vals[3]);
      printf("%d :: max   xyt_n  =%d\n",my_id,vals[4]);
      printf("%d :: avg   xyt_n  =%g\n",my_id,1.0*vals[5]/num_nodes);
      printf("%d :: tot   xyt_n  =%d\n",my_id,vals[5]);
      printf("%d :: min   xyt_buf=%d\n",my_id,vals[6]);
      printf("%d :: max   xyt_buf=%d\n",my_id,vals[7]);
      printf("%d :: avg   xyt_buf=%g\n",my_id,1.0*vals[8]/num_nodes);
      printf("%d :: min   xyt_slv=%g\n",my_id,fvals[0]);
      printf("%d :: max   xyt_slv=%g\n",my_id,fvals[1]);
      printf("%d :: avg   xyt_slv=%g\n",my_id,fvals[2]/num_nodes);
    }

  return(0);
}


/*************************************xyt.c************************************
Function: do_xyt_factor

Input : 
Output: 
Return: 
Description: get A_local, local portion of global coarse matrix which 
is a row dist. nxm matrix w/ n<m.
   o my_ml holds address of ML struct associated w/A_local and coarse grid
   o local2global holds global number of column i (i=0,...,m-1)
   o local2global holds global number of row    i (i=0,...,n-1)
   o mylocmatvec performs A_local . vec_local (note that gs is performed using 
   gs_init/gop).

mylocmatvec = my_ml->Amat[grid_tag].matvec->external;
mylocmatvec (void :: void *data, double *in, double *out)
**************************************xyt.c***********************************/
static
int
do_xyt_factor(xyt_ADT xyt_handle)
{
  int flag;


  flag=xyt_generate(xyt_handle);
  return(flag);
}


/*************************************xyt.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
**************************************xyt.c***********************************/
static
int
xyt_generate(xyt_ADT xyt_handle)
{
  int i,j,k,idx;
  int dim, col;
  PetscScalar *u, *uu, *v, *z, *w, alpha, alpha_w;
  int *segs;
  int op[] = {GL_ADD,0};
  int off, len;
  PetscScalar *x_ptr, *y_ptr;
  int *iptr, flag;
  int start=0, end, work;
  int op2[] = {GL_MIN,0};
  gs_ADT gs_handle;
  int *nsep, *lnsep, *fo;
  int a_n=xyt_handle->mvi->n;
  int a_m=xyt_handle->mvi->m;
  int *a_local2global=xyt_handle->mvi->local2global;
  int level;
  int n, m;
  int *xcol_sz, *xcol_indices, *stages; 
  PetscScalar **xcol_vals, *x;
  int *ycol_sz, *ycol_indices;
  PetscScalar **ycol_vals, *y;
  int n_global;
  int xt_nnz=0, xt_max_nnz=0;
  int yt_nnz=0, yt_max_nnz=0;
  int xt_zero_nnz  =0;
  int xt_zero_nnz_0=0;
  int yt_zero_nnz  =0;
  int yt_zero_nnz_0=0;
  PetscBLASInt i1 = 1;
  PetscScalar dm1 = -1.0;

  n=xyt_handle->mvi->n; 
  nsep=xyt_handle->info->nsep; 
  lnsep=xyt_handle->info->lnsep;
  fo=xyt_handle->info->fo;
  end=lnsep[0];
  level=xyt_handle->level;
  gs_handle=xyt_handle->mvi->gs_handle;

  /* is there a null space? */
  /* LATER add in ability to detect null space by checking alpha */
  for (i=0, j=0; i<=level; i++)
    {j+=nsep[i];}

  m = j-xyt_handle->ns;
  if (m!=j)
    {printf("xyt_generate() :: null space exists %d %d %d\n",m,j,xyt_handle->ns);}

  error_msg_warning("xyt_generate() :: X(%d,%d)\n",n,m);    

  /* get and initialize storage for x local         */
  /* note that x local is nxm and stored by columns */
  xcol_sz = (int*) malloc(m*sizeof(PetscInt));
  xcol_indices = (int*) malloc((2*m+1)*sizeof(int));
  xcol_vals = (PetscScalar **) malloc(m*sizeof(PetscScalar *));
  for (i=j=0; i<m; i++, j+=2)
    {
      xcol_indices[j]=xcol_indices[j+1]=xcol_sz[i]=-1;
      xcol_vals[i] = NULL;
    }
  xcol_indices[j]=-1;

  /* get and initialize storage for y local         */
  /* note that y local is nxm and stored by columns */
  ycol_sz = (int*) malloc(m*sizeof(PetscInt));
  ycol_indices = (int*) malloc((2*m+1)*sizeof(int));
  ycol_vals = (PetscScalar **) malloc(m*sizeof(PetscScalar *));
  for (i=j=0; i<m; i++, j+=2)
    {
      ycol_indices[j]=ycol_indices[j+1]=ycol_sz[i]=-1;
      ycol_vals[i] = NULL;
    }
  ycol_indices[j]=-1;

  /* size of separators for each sub-hc working from bottom of tree to top */
  /* this looks like nsep[]=segments */
  stages = (int*) malloc((level+1)*sizeof(PetscInt));
  segs   = (int*) malloc((level+1)*sizeof(PetscInt));
  ivec_zero(stages,level+1);
  ivec_copy(segs,nsep,level+1);
  for (i=0; i<level; i++)
    {segs[i+1] += segs[i];}
  stages[0] = segs[0];

  /* temporary vectors  */
  u  = (PetscScalar *) malloc(n*sizeof(PetscScalar));
  z  = (PetscScalar *) malloc(n*sizeof(PetscScalar));
  v  = (PetscScalar *) malloc(a_m*sizeof(PetscScalar));
  uu = (PetscScalar *) malloc(m*sizeof(PetscScalar));
  w  = (PetscScalar *) malloc(m*sizeof(PetscScalar));

  /* extra nnz due to replication of vertices across separators */
  for (i=1, j=0; i<=level; i++)
    {j+=nsep[i];}

  /* storage for sparse x values */
  n_global = xyt_handle->info->n_global;
  xt_max_nnz = yt_max_nnz = (int)(2.5*pow(1.0*n_global,1.6667) + j*n/2)/num_nodes;
  x = (PetscScalar *) malloc(xt_max_nnz*sizeof(PetscScalar));
  y = (PetscScalar *) malloc(yt_max_nnz*sizeof(PetscScalar));

  /* LATER - can embed next sep to fire in gs */
  /* time to make the donuts - generate X factor */
  for (dim=i=j=0;i<m;i++)
    {
      /* time to move to the next level? */
      while (i==segs[dim])
	{
#ifdef SAFE	  
	  if (dim==level)
	    {error_msg_fatal("dim about to exceed level\n"); break;}
#endif

	  stages[dim++]=i;
	  end+=lnsep[dim];
	}
      stages[dim]=i;

      /* which column are we firing? */
      /* i.e. set v_l */
      /* use new seps and do global min across hc to determine which one to fire */
      (start<end) ? (col=fo[start]) : (col=INT_MAX);
      giop_hc(&col,&work,1,op2,dim); 

      /* shouldn't need this */
      if (col==INT_MAX)
	{
	  error_msg_warning("hey ... col==INT_MAX??\n");
	  continue;
	}

      /* do I own it? I should */
      rvec_zero(v ,a_m);
      if (col==fo[start])
	{
	  start++;
	  idx=ivec_linear_search(col, a_local2global, a_n);
	  if (idx!=-1)
	    {v[idx] = 1.0; j++;}
	  else
	    {error_msg_fatal("NOT FOUND!\n");}
	}
      else
	{
	  idx=ivec_linear_search(col, a_local2global, a_m);
	  if (idx!=-1)
	    {v[idx] = 1.0;}
	}

      /* perform u = A.v_l */
      rvec_zero(u,n);
      do_matvec(xyt_handle->mvi,v,u);

      /* uu =  X^T.u_l (local portion) */
      /* technically only need to zero out first i entries */
      /* later turn this into an XYT_solve call ? */
      rvec_zero(uu,m);
      y_ptr=y;
      iptr = ycol_indices;
      for (k=0; k<i; k++)
	{
	  off = *iptr++;
	  len = *iptr++;

	  uu[k] = BLASdot_(&len,u+off,&i1,y_ptr,&i1);
	  y_ptr+=len;
	}

      /* uu = X^T.u_l (comm portion) */
      ssgl_radd  (uu, w, dim, stages);

      /* z = X.uu */
      rvec_zero(z,n);
      x_ptr=x;
      iptr = xcol_indices;
      for (k=0; k<i; k++)
	{
	  off = *iptr++;
	  len = *iptr++;

	  BLASaxpy_(&len,&uu[k],x_ptr,&i1,z+off,&i1);
	  x_ptr+=len;
	}

      /* compute v_l = v_l - z */
      rvec_zero(v+a_n,a_m-a_n);
      BLASaxpy_(&n,&dm1,z,&i1,v,&i1);

      /* compute u_l = A.v_l */
      if (a_n!=a_m)
	{gs_gop_hc(gs_handle,v,"+\0",dim);}
      rvec_zero(u,n);
     do_matvec(xyt_handle->mvi,v,u);

      /* compute sqrt(alpha) = sqrt(u_l^T.u_l) - local portion */
      alpha = BLASdot_(&n,u,&i1,u,&i1);
      /* compute sqrt(alpha) = sqrt(u_l^T.u_l) - comm portion */
      grop_hc(&alpha, &alpha_w, 1, op, dim);

      alpha = (PetscScalar) sqrt((double)alpha);

      /* check for small alpha                             */
      /* LATER use this to detect and determine null space */
      if (fabs(alpha)<1.0e-14)
	{error_msg_fatal("bad alpha! %g\n",alpha);}

      /* compute v_l = v_l/sqrt(alpha) */
      rvec_scale(v,1.0/alpha,n);
      rvec_scale(u,1.0/alpha,n);

      /* add newly generated column, v_l, to X */
      flag = 1;
      off=len=0;
      for (k=0; k<n; k++)
	{
	  if (v[k]!=0.0)
	    {
	      len=k;
	      if (flag)
		{off=k; flag=0;}
	    }
	}

      len -= (off-1);

      if (len>0)
	{
	  if ((xt_nnz+len)>xt_max_nnz)
	    {
	      error_msg_warning("increasing space for X by 2x!\n");
	      xt_max_nnz *= 2;
	      x_ptr = (PetscScalar *) malloc(xt_max_nnz*sizeof(PetscScalar));
	      rvec_copy(x_ptr,x,xt_nnz);
	      free(x);
	      x = x_ptr;
	      x_ptr+=xt_nnz;
	    }
	  xt_nnz += len;      
	  rvec_copy(x_ptr,v+off,len);

          /* keep track of number of zeros */
	  if (dim)
	    {
	      for (k=0; k<len; k++)
		{
		  if (x_ptr[k]==0.0)
		    {xt_zero_nnz++;}
		}
	    }
	  else
	    {
	      for (k=0; k<len; k++)
		{
		  if (x_ptr[k]==0.0)
		    {xt_zero_nnz_0++;}
		}
	    }
	  xcol_indices[2*i] = off;
	  xcol_sz[i] = xcol_indices[2*i+1] = len;
	  xcol_vals[i] = x_ptr;
	}
      else
	{
	  xcol_indices[2*i] = 0;
	  xcol_sz[i] = xcol_indices[2*i+1] = 0;
	  xcol_vals[i] = x_ptr;
	}


      /* add newly generated column, u_l, to Y */
      flag = 1;
      off=len=0;
      for (k=0; k<n; k++)
	{
	  if (u[k]!=0.0)
	    {
	      len=k;
	      if (flag)
		{off=k; flag=0;}
	    }
	}

      len -= (off-1);

      if (len>0)
	{
	  if ((yt_nnz+len)>yt_max_nnz)
	    {
	      error_msg_warning("increasing space for Y by 2x!\n");
	      yt_max_nnz *= 2;
	      y_ptr = (PetscScalar *) malloc(yt_max_nnz*sizeof(PetscScalar));
	      rvec_copy(y_ptr,y,yt_nnz);
	      free(y);
	      y = y_ptr;
	      y_ptr+=yt_nnz;
	    }
	  yt_nnz += len;      
	  rvec_copy(y_ptr,u+off,len);

          /* keep track of number of zeros */
	  if (dim)
	    {
	      for (k=0; k<len; k++)
		{
		  if (y_ptr[k]==0.0)
		    {yt_zero_nnz++;}
		}
	    }
	  else
	    {
	      for (k=0; k<len; k++)
		{
		  if (y_ptr[k]==0.0)
		    {yt_zero_nnz_0++;}
		}
	    }
	  ycol_indices[2*i] = off;
	  ycol_sz[i] = ycol_indices[2*i+1] = len;
	  ycol_vals[i] = y_ptr;
	}
      else
	{
	  ycol_indices[2*i] = 0;
	  ycol_sz[i] = ycol_indices[2*i+1] = 0;
	  ycol_vals[i] = y_ptr;
	}
    }

  /* close off stages for execution phase */
  while (dim!=level)
    {
      stages[dim++]=i;
      error_msg_warning("disconnected!!! dim(%d)!=level(%d)\n",dim,level);
    }
  stages[dim]=i;

  xyt_handle->info->n=xyt_handle->mvi->n;
  xyt_handle->info->m=m;
  xyt_handle->info->nnz=xt_nnz + yt_nnz;
  xyt_handle->info->max_nnz=xt_max_nnz + yt_max_nnz;
  xyt_handle->info->msg_buf_sz=stages[level]-stages[0];
  xyt_handle->info->solve_uu = (PetscScalar *) malloc(m*sizeof(PetscScalar));
  xyt_handle->info->solve_w  = (PetscScalar *) malloc(m*sizeof(PetscScalar));
  xyt_handle->info->x=x;
  xyt_handle->info->xcol_vals=xcol_vals;
  xyt_handle->info->xcol_sz=xcol_sz;
  xyt_handle->info->xcol_indices=xcol_indices;  
  xyt_handle->info->stages=stages;
  xyt_handle->info->y=y;
  xyt_handle->info->ycol_vals=ycol_vals;
  xyt_handle->info->ycol_sz=ycol_sz;
  xyt_handle->info->ycol_indices=ycol_indices;  

  free(segs);
  free(u);
  free(v);
  free(uu);
  free(z);
  free(w);

  return(0);
}


/*************************************xyt.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
**************************************xyt.c***********************************/
static
void
do_xyt_solve(xyt_ADT xyt_handle,  PetscScalar *uc)
{
  int off, len, *iptr;
  int level       =xyt_handle->level;
  int n           =xyt_handle->info->n;
  int m           =xyt_handle->info->m;
  int *stages     =xyt_handle->info->stages;
  int *xcol_indices=xyt_handle->info->xcol_indices;
  int *ycol_indices=xyt_handle->info->ycol_indices;
   PetscScalar *x_ptr, *y_ptr, *uu_ptr;
  PetscScalar *solve_uu=xyt_handle->info->solve_uu;
  PetscScalar *solve_w =xyt_handle->info->solve_w;
  PetscScalar *x       =xyt_handle->info->x;
  PetscScalar *y       =xyt_handle->info->y;
  PetscBLASInt i1 = 1;


  uu_ptr=solve_uu;
  rvec_zero(uu_ptr,m);

  /* x  = X.Y^T.b */
  /* uu = Y^T.b */
  for (y_ptr=y,iptr=ycol_indices; *iptr!=-1; y_ptr+=len)
    {
      off=*iptr++; len=*iptr++;
      *uu_ptr++ = BLASdot_(&len,uc+off,&i1,y_ptr,&i1);
    }

  /* comunication of beta */
  uu_ptr=solve_uu;
  if (level) {ssgl_radd(uu_ptr, solve_w, level, stages);}

  rvec_zero(uc,n);

  /* x = X.uu */
  for (x_ptr=x,iptr=xcol_indices; *iptr!=-1; x_ptr+=len)
    {
      off=*iptr++; len=*iptr++;
      BLASaxpy_(&len,uu_ptr++,x_ptr,&i1,uc+off,&i1);
    }

}


/*************************************Xyt.c************************************
Function: check_init

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
static
void
check_init(void)
{
  comm_init();


}


/*************************************xyt.c************************************
Function: check_handle()

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
static
void 
check_handle(xyt_ADT xyt_handle)
{
#ifdef SAFE
  int vals[2], work[2], op[] = {NON_UNIFORM,GL_MIN,GL_MAX};
#endif

  if (xyt_handle==NULL)
    {error_msg_fatal("check_handle() :: bad handle :: NULL %d\n",xyt_handle);}

#ifdef SAFE
  vals[0]=vals[1]=xyt_handle->id;
  giop(vals,work,sizeof(op)/sizeof(op[0])-1,op);
  if ((vals[0]!=vals[1])||(xyt_handle->id<=0))
    {error_msg_fatal("check_handle() :: bad handle :: id mismatch min/max %d/%d %d\n",
		     vals[0],vals[1], xyt_handle->id);}
#endif

}


/*************************************xyt.c************************************
Function: det_separators

Input :
Output:
Return:
Description:
  det_separators(xyt_handle, local2global, n, m, mylocmatvec, grid_data);
**************************************xyt.c***********************************/
static 
void 
det_separators(xyt_ADT xyt_handle)
{
  int i, ct, id;
  int mask, edge, *iptr; 
  int *dir, *used;
  int sum[4], w[4];
  PetscScalar rsum[4], rw[4];
  int op[] = {GL_ADD,0};
  PetscScalar *lhs, *rhs;
  int *nsep, *lnsep, *fo, nfo=0;
  gs_ADT gs_handle=xyt_handle->mvi->gs_handle;
  int *local2global=xyt_handle->mvi->local2global;
  int  n=xyt_handle->mvi->n;
  int  m=xyt_handle->mvi->m;
  int level=xyt_handle->level;
  int shared=FALSE; 

  dir  = (int*)malloc(sizeof(PetscInt)*(level+1));
  nsep = (int*)malloc(sizeof(PetscInt)*(level+1));
  lnsep= (int*)malloc(sizeof(PetscInt)*(level+1));
  fo   = (int*)malloc(sizeof(PetscInt)*(n+1));
  used = (int*)malloc(sizeof(PetscInt)*n);

  ivec_zero(dir  ,level+1);
  ivec_zero(nsep ,level+1);
  ivec_zero(lnsep,level+1);
  ivec_set (fo   ,-1,n+1);
  ivec_zero(used,n);

  lhs  = (double*)malloc(sizeof(PetscScalar)*m);
  rhs  = (double*)malloc(sizeof(PetscScalar)*m);

  /* determine the # of unique dof */
  rvec_zero(lhs,m);
  rvec_set(lhs,1.0,n);
  gs_gop_hc(gs_handle,lhs,"+\0",level);
  error_msg_warning("done first gs_gop_hc\n");
  rvec_zero(rsum,2);
  for (ct=i=0;i<n;i++)
    {
      if (lhs[i]!=0.0)
	{rsum[0]+=1.0/lhs[i]; rsum[1]+=lhs[i];}

      if (lhs[i]!=1.0)
	{
          shared=TRUE;
        }
    }

  grop_hc(rsum,rw,2,op,level);
  rsum[0]+=0.1;
  rsum[1]+=0.1;

  /*
      if (!my_id)
      {
      printf("xyt n unique = %d (%g)\n",(int) rsum[0], rsum[0]);
      printf("xyt n shared = %d (%g)\n",(int) rsum[1], rsum[1]);
      }
  */

  xyt_handle->info->n_global=xyt_handle->info->m_global=(int) rsum[0];
  xyt_handle->mvi->n_global =xyt_handle->mvi->m_global =(int) rsum[0];

  /* determine separator sets top down */
  if (shared)
    {
      /* solution is to do as in the symmetric shared case but then */
      /* pick the sub-hc with the most free dofs and do a mat-vec   */
      /* and pick up the responses on the other sub-hc from the     */
      /* initial separator set obtained from the symm. shared case  */
      error_msg_fatal("shared dof separator determination not ready ... see hmt!!!\n"); 
      for (iptr=fo+n,id=my_id,mask=num_nodes>>1,edge=level;edge>0;edge--,mask>>=1)
	{
	  /* set rsh of hc, fire, and collect lhs responses */
	  (id<mask) ? rvec_zero(lhs,m) : rvec_set(lhs,1.0,m);
	  gs_gop_hc(gs_handle,lhs,"+\0",edge);
	  
	  /* set lsh of hc, fire, and collect rhs responses */
	  (id<mask) ? rvec_set(rhs,1.0,m) : rvec_zero(rhs,m);
	  gs_gop_hc(gs_handle,rhs,"+\0",edge);
	  
	  for (i=0;i<n;i++)
	    {
	      if (id< mask)
		{		
		  if (lhs[i]!=0.0)
		    {lhs[i]=1.0;}
		}
	      if (id>=mask)
		{		
		  if (rhs[i]!=0.0)
		    {rhs[i]=1.0;}
		}
	    }

	  if (id< mask)
	    {gs_gop_hc(gs_handle,lhs,"+\0",edge-1);}
	  else
	    {gs_gop_hc(gs_handle,rhs,"+\0",edge-1);}

	  /* count number of dofs I own that have signal and not in sep set */
	  rvec_zero(rsum,4);
	  for (ivec_zero(sum,4),ct=i=0;i<n;i++)
	    {
	      if (!used[i]) 
		{
		  /* number of unmarked dofs on node */
		  ct++;
		  /* number of dofs to be marked on lhs hc */
		  if (id< mask)
		    {		
		      if (lhs[i]!=0.0)
			{sum[0]++; rsum[0]+=1.0/lhs[i];}
		    }
		  /* number of dofs to be marked on rhs hc */
		  if (id>=mask)
		    {		
		      if (rhs[i]!=0.0)
			{sum[1]++; rsum[1]+=1.0/rhs[i];}
		    }
		}
	    }

	  /* go for load balance - choose half with most unmarked dofs, bias LHS */
	  (id<mask) ? (sum[2]=ct) : (sum[3]=ct);
	  (id<mask) ? (rsum[2]=ct) : (rsum[3]=ct);
	  giop_hc(sum,w,4,op,edge);
	  grop_hc(rsum,rw,4,op,edge);
	  rsum[0]+=0.1; rsum[1]+=0.1; rsum[2]+=0.1; rsum[3]+=0.1;

	  if (id<mask)
	    {
	      /* mark dofs I own that have signal and not in sep set */
	      for (ct=i=0;i<n;i++)
		{
		  if ((!used[i])&&(lhs[i]!=0.0))
		    {
		      ct++; nfo++;

		      if (nfo>n)
			{error_msg_fatal("nfo about to exceed n\n");}

		      *--iptr = local2global[i];
		      used[i]=edge;
		    }
		}
	      if (ct>1) {ivec_sort(iptr,ct);}

	      lnsep[edge]=ct;
	      nsep[edge]=(int) rsum[0];
	      dir [edge]=LEFT;
	    }

	  if (id>=mask)
	    {
	      /* mark dofs I own that have signal and not in sep set */
	      for (ct=i=0;i<n;i++)
		{
		  if ((!used[i])&&(rhs[i]!=0.0))
		    {
		      ct++; nfo++;

		      if (nfo>n)
			{error_msg_fatal("nfo about to exceed n\n");}

		      *--iptr = local2global[i];
		      used[i]=edge;
		    }
		}
	      if (ct>1) {ivec_sort(iptr,ct);}

	      lnsep[edge]=ct;
	      nsep[edge]= (int) rsum[1];
	      dir [edge]=RIGHT;
	    }

	  /* LATER or we can recur on these to order seps at this level */
	  /* do we need full set of separators for this?                */

	  /* fold rhs hc into lower */
	  if (id>=mask)
	    {id-=mask;}
	}
    }
  else
    {
      for (iptr=fo+n,id=my_id,mask=num_nodes>>1,edge=level;edge>0;edge--,mask>>=1)
	{
	  /* set rsh of hc, fire, and collect lhs responses */
	  (id<mask) ? rvec_zero(lhs,m) : rvec_set(lhs,1.0,m);
	  gs_gop_hc(gs_handle,lhs,"+\0",edge);

	  /* set lsh of hc, fire, and collect rhs responses */
	  (id<mask) ? rvec_set(rhs,1.0,m) : rvec_zero(rhs,m);
	  gs_gop_hc(gs_handle,rhs,"+\0",edge);

	  /* count number of dofs I own that have signal and not in sep set */
	  for (ivec_zero(sum,4),ct=i=0;i<n;i++)
	    {
	      if (!used[i]) 
		{
		  /* number of unmarked dofs on node */
		  ct++;
		  /* number of dofs to be marked on lhs hc */
		  if ((id< mask)&&(lhs[i]!=0.0)) {sum[0]++;}
		  /* number of dofs to be marked on rhs hc */
		  if ((id>=mask)&&(rhs[i]!=0.0)) {sum[1]++;}
		}
	    }

	  /* for the non-symmetric case we need separators of width 2 */
	  /* so take both sides */
	  (id<mask) ? (sum[2]=ct) : (sum[3]=ct);
	  giop_hc(sum,w,4,op,edge);

	  ct=0;
	  if (id<mask)
	    {
	      /* mark dofs I own that have signal and not in sep set */
	      for (i=0;i<n;i++)
		{
		  if ((!used[i])&&(lhs[i]!=0.0))
		    {
		      ct++; nfo++;
		      *--iptr = local2global[i];
		      used[i]=edge;
		    }
		}
	      /* LSH hc summation of ct should be sum[0] */
	    }
	  else
	    {
	      /* mark dofs I own that have signal and not in sep set */
	      for (i=0;i<n;i++)
		{
		  if ((!used[i])&&(rhs[i]!=0.0))
		    {
		      ct++; nfo++;
		      *--iptr = local2global[i];
		      used[i]=edge;
		    }
		}
	      /* RSH hc summation of ct should be sum[1] */
	    }

	  if (ct>1) {ivec_sort(iptr,ct);}
	  lnsep[edge]=ct;
	  nsep[edge]=sum[0]+sum[1];
	  dir [edge]=BOTH;

	  /* LATER or we can recur on these to order seps at this level */
	  /* do we need full set of separators for this?                */

	  /* fold rhs hc into lower */
	  if (id>=mask)
	    {id-=mask;}
	}
    }

  /* level 0 is on processor case - so mark the remainder */
  for (ct=i=0;i<n;i++)
    {
      if (!used[i]) 
	{
	  ct++; nfo++;
	  *--iptr = local2global[i];
	  used[i]=edge;
	}
    }
  if (ct>1) {ivec_sort(iptr,ct);}
  lnsep[edge]=ct;
  nsep [edge]=ct;
  dir  [edge]=BOTH;

  xyt_handle->info->nsep=nsep;
  xyt_handle->info->lnsep=lnsep;
  xyt_handle->info->fo=fo;
  xyt_handle->info->nfo=nfo;

  free(dir);
  free(lhs);
  free(rhs);
  free(used);

}


/*************************************xyt.c************************************
Function: set_mvi

Input :
Output:
Return:
Description:
**************************************xyt.c***********************************/
static
mv_info *set_mvi(int *local2global, int n, int m, void *matvec, void *grid_data)
{
  mv_info *mvi;


  mvi = (mv_info*)malloc(sizeof(mv_info));
  mvi->n=n;
  mvi->m=m;
  mvi->n_global=-1;
  mvi->m_global=-1;
  mvi->local2global=(int*)malloc((m+1)*sizeof(PetscInt));
  ivec_copy(mvi->local2global,local2global,m);
  mvi->local2global[m] = INT_MAX;
  mvi->matvec=(PetscErrorCode (*)(mv_info*,PetscScalar*,PetscScalar*))matvec;
  mvi->grid_data=grid_data;

  /* set xyt communication handle to perform restricted matvec */
  mvi->gs_handle = gs_init(local2global, m, num_nodes);

  return(mvi);
}


/*************************************xyt.c************************************
Function: set_mvi

Input :
Output:
Return:
Description:

      computes u = A.v 
      do_matvec(xyt_handle->mvi,v,u);
**************************************xyt.c***********************************/
static void do_matvec(mv_info *A, PetscScalar *v, PetscScalar *u)
{
  A->matvec((mv_info*)A->grid_data,v,u);
}



