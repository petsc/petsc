/*$Id: da2.c,v 1.142 2000/05/04 18:58:59 balay Exp bsmith $*/
 
#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetOwnershipRange"
int DAGetOwnershipRange(DA da,int **lx,int **ly,int **lz)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (lx) *lx = da->lx;
  if (ly) *ly = da->ly;
  if (lz) *lz = da->lz;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAView_2d"
int DAView_2d(DA da,Viewer viewer)
{
  int        rank,ierr;
  PetscTruth isascii,isdraw,isbinary;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(da->comm,&rank);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %d N %d m %d n %d w %d s %d\n",rank,da->M,
                             da->N,da->m,da->n,da->w,da->s);CHKERRQ(ierr);
    ierr = ViewerASCIISynchronizedPrintf(viewer,"X range: %d %d, Y range: %d %d\n",da->xs,da->xe,da->ys,da->ye);CHKERRQ(ierr);
    ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    Draw       draw;
    double     ymin = -1*da->s-1,ymax = da->N+da->s;
    double     xmin = -1*da->s-1,xmax = da->M+da->s;
    double     x,y;
    int        base,*idx;
    char       node[10];
    PetscTruth isnull;
 
    ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
    ierr = DrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draw all node lines */
    if (!rank) {
      ymin = 0.0; ymax = da->N - 1;
      for (xmin=0; xmin<da->M; xmin++) {
        ierr = DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_BLACK);CHKERRQ(ierr);
      }
      xmin = 0.0; xmax = da->M - 1;
      for (ymin=0; ymin<da->N; ymin++) {
        ierr = DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_BLACK);CHKERRQ(ierr);
      }
    }
    ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = DrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = da->ys; ymax = da->ye - 1; xmin = da->xs/da->w; 
    xmax =(da->xe-1)/da->w;
    ierr = DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawLine(draw,xmax,ymin,xmax,ymax,DRAW_RED);CHKERRQ(ierr);

    /* put in numbers */
    base = (da->base)/da->w;
    for (y=ymin; y<=ymax; y++) {
      for (x=xmin; x<=xmax; x++) {
        sprintf(node,"%d",base++);
        ierr = DrawString(draw,x,y,DRAW_BLACK,node);CHKERRQ(ierr);
      }
    }

    ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = DrawPause(draw);CHKERRQ(ierr);
    /* overlay ghost numbers, useful for error checking */
    /* put in numbers */

    base = 0; idx = da->idx;
    ymin = da->Ys; ymax = da->Ye; xmin = da->Xs; xmax = da->Xe;
    for (y=ymin; y<ymax; y++) {
      for (x=xmin; x<xmax; x++) {
        if ((base % da->w) == 0) {
          sprintf(node,"%d",idx[base]/da->w);
          ierr = DrawString(draw,x/da->w,y,DRAW_BLUE,node);CHKERRQ(ierr);
        }
        base++;
      }
    }        
    ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = DrawPause(draw);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = DAView_Binary(da,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for DA2d",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
/*
      This function tells the AMS the layout of the vectors, it is called
   in the VecPublish_xx routines.
*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AMSSetFieldBlock_DA"
int AMSSetFieldBlock_DA(AMS_Memory amem,char *name,Vec vec)
{
  int        ierr,dof,dim,ends[4],shift = 0,starts[] = {0,0,0,0};
  DA         da = 0;
  PetscTruth isseq,ismpi;

  PetscFunctionBegin;
  if (((PetscObject)vec)->amem < 0) PetscFunctionReturn(0); /* return if not published */

  ierr = PetscObjectQuery((PetscObject)vec,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) PetscFunctionReturn(0);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (dof > 1) {dim++; shift = 1; ends[0] = dof;}

  ierr = PetscTypeCompare((PetscObject)vec,VEC_SEQ,&isseq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)vec,VEC_MPI,&ismpi);CHKERRQ(ierr);
  if (isseq) {
    ierr = DAGetGhostCorners(da,0,0,0,ends+shift,ends+shift+1,ends+shift+2);CHKERRQ(ierr);
    ends[shift]   += starts[shift]-1;
    ends[shift+1] += starts[shift+1]-1;
    ends[shift+2] += starts[shift+2]-1;
    ierr = AMS_Memory_set_field_block(amem,name,dim,starts,ends);CHKERRQ(ierr);
    if (ierr) {
      char *message;
      AMS_Explain_error(ierr,&message);
      SETERRQ(ierr,1,message);
    }
  } else if (ismpi) {
    ierr = DAGetCorners(da,starts+shift,starts+shift+1,starts+shift+2,
                           ends+shift,ends+shift+1,ends+shift+2);CHKERRQ(ierr);
    ends[shift]   += starts[shift]-1;
    ends[shift+1] += starts[shift+1]-1;
    ends[shift+2] += starts[shift+2]-1;
    ierr = AMS_Memory_set_field_block(amem,name,dim,starts,ends);
    if (ierr) {
      char *message;
      AMS_Explain_error(ierr,&message);
      SETERRQ(ierr,1,message);
    }
  } else {
    SETERRQ1(1,1,"Wrong vector type %s for this call",((PetscObject)vec)->type_name);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAPublish_Petsc"
int DAPublish_Petsc(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  DA          v = (DA) obj;
  int         ierr;
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/*
   This allows the DA vectors to properly tell Matlab their dimensions
*/
#if defined(PETSC_HAVE_MATLAB)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="VecMatlabEnginePut_DA2d"></a>*/"VecMatlabEnginePut_DA2d"
int VecMatlabEnginePut_DA2d(PetscObject obj,void *engine)
{
  int     ierr,n,m;
  Vec     vec = (Vec)obj;
  Scalar  *array;
  mxArray *mat;
  DA      da;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)vec,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(1,1,"Vector not associated with a DA");
  ierr = DAGetGhostCorners(da,0,0,0,&m,&n,0);CHKERRQ(ierr);

  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(n,m,(mxComplexity)0);
#else
  mat  = mxCreateDoubleMatrix(n,m,(mxComplexity)1);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,n*m*sizeof(Scalar));CHKERRQ(ierr);
  mxSetName(mat,obj->name);
  engPutArray((Engine *)engine,mat);
  
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DACreate2d"
/*@C
   DACreate2d -  Creates an object that will manage the communication of  two-dimensional 
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  wrap - type of periodicity should the array have. 
         Use one of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, or DA_XYPERIODIC.
.  stencil_type - stencil type.  Use either DA_STENCIL_BOX or DA_STENCIL_STAR.
.  M,N - global dimension in each direction of the array
.  m,n - corresponding number of processors in each dimension 
         (or PETSC_DECIDE to have calculated)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx, ly - arrays containing the number of nodes in each cell along
           the x and y coordinates, or PETSC_NULL. If non-null, these
           must be of length as m and n, and the corresponding
           m and n cannot be PETSC_DECIDE. The sum of the lx[] entries
           must be M, and the sum of the ly[] entries must be N.

   Output Parameter:
.  inra - the resulting distributed array object

   Options Database Key:
.  -da_view - Calls DAView() at the conclusion of DACreate2d()

   Level: beginner

   Notes:
   The stencil type DA_STENCIL_STAR with width 1 corresponds to the 
   standard 5-pt stencil, while DA_STENCIL_BOX with width 1 denotes
   the standard 9-pt stencil.

   The array data itself is NOT stored in the DA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DACreateGlobalVector()
   and DACreateLocalVector() and calls to VecDuplicate() if more are needed.

.keywords: distributed array, create, two-dimensional

.seealso: DADestroy(), DAView(), DACreate1d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DALocalToLocalBegin(), DALocalToLocalEnd(),
          DAGetInfo(), DACreateGlobalVector(), DACreateLocalVector(), DACreateNaturalVector(), DALoad(), DAView()

@*/
int DACreate2d(MPI_Comm comm,DAPeriodicType wrap,DAStencilType stencil_type,
                int M,int N,int m,int n,int dof,int s,int *lx,int *ly,DA *inra)
{
  int           rank,size,xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,ierr,start,end;
  int           up,down,left,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn;
  int           xbase,*bases,*ldims,j,x_t,y_t,s_t,base,count;
  int           s_x,s_y; /* s proportionalized to w */
  int           *gA,*gB,*gAall,*gBall,ict,ldim,gdim,*flx = 0,*fly = 0;
  int           sn0 = 0,sn2 = 0,sn6 = 0,sn8 = 0;
  PetscTruth    flg1,flg2;
  DA            da;
  Vec           local,global;
  VecScatter    ltog,gtol;
  IS            to,from;

  PetscFunctionBegin;
  *inra = 0;

  if (dof < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Must have 1 or more degrees of freedom per node: %d",dof);
  if (s < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Stencil width cannot be negative: %d",s);

  PetscHeaderCreate(da,_p_DA,int,DA_COOKIE,0,"DA",comm,DADestroy,DAView);
  PLogObjectCreate(da);
  da->bops->publish = DAPublish_Petsc;
  PLogObjectMemory(da,sizeof(struct _p_DA));
  da->dim        = 2;
  da->gtog1      = 0;
  da->localused  = PETSC_FALSE;
  da->globalused = PETSC_FALSE;
  da->fieldname  = (char**)PetscMalloc(dof*sizeof(char*));CHKPTRQ(da->fieldname);
  ierr = PetscMemzero(da->fieldname,dof*sizeof(char*));CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  if (m != PETSC_DECIDE) {
    if (m < 1) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Non-positive number of processors in X direction: %d",m);}
    else if (m > size) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Too many processors in X direction: %d %d",m,size);}
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Non-positive number of processors in Y direction: %d",n);}
    else if (n > size) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Too many processors in Y direction: %d %d",n,size);}
  }

  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    /* try for squarish distribution */
    /* This should use MPI_Dims_create instead */
    m = (int)(0.5 + sqrt(((double)M)*((double)size)/((double)N)));
    if (!m) m = 1;
    while (m > 0) {
      n = size/m;
      if (m*n == size) break;
      m--;
    }
    if (M > N && m < n) {int _m = m; m = n; n = _m;}
    if (m*n != size) SETERRQ(PETSC_ERR_PLIB,0,"Internally Created Bad Partition");
  } else if (m*n != size) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Given Bad partition"); 

  if (M < m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Partition in x direction is too fine! %d %d",M,m);
  if (N < n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Partition in y direction is too fine! %d %d",N,n);

  /*
     We should create an MPI Cartesian topology here, with reorder
     set to true.  That would create a NEW communicator that we would
     need to use for operations on this distributed array 
  */
  ierr = OptionsHasName(PETSC_NULL,"-da_partition_blockcomm",&flg1);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-da_partition_nodes_at_end",&flg2);CHKERRQ(ierr);

  /* 
     Determine locally owned region 
     xs is the first local node number, x is the number of local nodes 
  */
  if (lx) { /* user sets distribution */
    x  = lx[rank % m];
    xs = 0;
    for (i=0; i<(rank % m); i++) {
      xs += lx[i];
    }
    left = xs;
    for (i=(rank % m); i<m; i++) {
      left += lx[i];
    }
    if (left != M) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Sum of lx across processors not equal to M: %d %d",left,M);
    }
  } else if (flg1) {  /* Block Comm type Distribution */
    xs = (rank%m)*M/m;
    x  = (rank%m + 1)*M/m - xs;
    if (x < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Column width is too thin for stencil! %d %d",x,s);
    SETERRQ(PETSC_ERR_SUP,1,"-da_partition_blockcomm not supported");
  } else if (flg2) { 
    x = (M + rank%m)/m;
    if (x < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Column width is too thin for stencil! %d %d",x,s);
    if (M/m == x) { xs = (rank % m)*x; }
    else          { xs = (rank % m)*(x-1) + (M+(rank % m))%(x*m); }
    SETERRQ(PETSC_ERR_SUP,1,"-da_partition_nodes_at_end not supported");
  } else { /* Normal PETSc distribution */
    x = M/m + ((M % m) > (rank % m));
    if (x < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Column width is too thin for stencil! %d %d",x,s);
    if ((M % m) > (rank % m)) { xs = (rank % m)*x; }
    else                      { xs = (M % m)*(x+1) + ((rank % m)-(M % m))*x; }
    flx = lx = (int*)PetscMalloc(m*sizeof(int));CHKPTRQ(lx);
    for (i=0; i<m; i++) {
      lx[i] = M/m + ((M % m) > i);
    }
  }

  /* 
     Determine locally owned region 
     ys is the first local node number, y is the number of local nodes 
  */
  if (ly) { /* user sets distribution */
    y  = ly[rank/m];
    ys = 0;
    for (i=0; i<(rank/m); i++) {
      ys += ly[i];
    }
    left = ys;
    for (i=(rank/m); i<n; i++) {
      left += ly[i];
    }
    if (left != N) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Sum of ly across processors not equal to N: %d %d",left,N);
    }
  } else if (flg1) {  /* Block Comm type Distribution */
    ys = (rank/m)*N/n;
    y  = (rank/m + 1)*N/n - ys;
    if (y < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Row width is too thin for stencil! %d %d",y,s);      
    SETERRQ(PETSC_ERR_SUP,1,"-da_partition_blockcomm not supported");
  } else if (flg2) { 
    y = (N + rank/m)/n;
    if (y < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Row width is too thin for stencil! %d %d",y,s);
    if (N/n == y) { ys = (rank/m)*y;  }
    else          { ys = (rank/m)*(y-1) + (N+(rank/m))%(y*n); }
    SETERRQ(PETSC_ERR_SUP,1,"-da_partition_nodes_at_end not supported");
  } else { /* Normal PETSc distribution */
    y = N/n + ((N % n) > (rank/m));
    if (y < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Row width is too thin for stencil! %d %d",y,s);
    if ((N % n) > (rank/m)) { ys = (rank/m)*y; }
    else                    { ys = (N % n)*(y+1) + ((rank/m)-(N % n))*y; }
    fly = ly = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(lx);
    for (i=0; i<n; i++) {
      ly[i] = N/n + ((N % n) > i);
    }
  }

  xe = xs + x;
  ye = ys + y;

  /* determine ghost region */
  /* Assume No Periodicity */
  if (xs-s > 0) Xs = xs - s; else Xs = 0; 
  if (ys-s > 0) Ys = ys - s; else Ys = 0; 
  if (xe+s <= M) Xe = xe + s; else Xe = M; 
  if (ye+s <= N) Ye = ye + s; else Ye = N;

  /* X Periodic */
  if (wrap == DA_XPERIODIC || wrap ==  DA_XYPERIODIC) {
    Xs = xs - s; 
    Xe = xe + s; 
  }

  /* Y Periodic */
  if (wrap == DA_YPERIODIC || wrap ==  DA_XYPERIODIC) {
    Ys = ys - s;
    Ye = ye + s;
  }

  /* Resize all X parameters to reflect w */
  x   *= dof;
  xs  *= dof;
  xe  *= dof;
  Xs  *= dof;
  Xe  *= dof;
  s_x = s*dof;
  s_y = s;

  /* determine starting point of each processor */
  nn = x*y;
  bases = (int*)PetscMalloc((2*size+1)*sizeof(int));CHKPTRQ(bases);
  ldims = (int*)(bases+size+1);
  ierr = MPI_Allgather(&nn,1,MPI_INT,ldims,1,MPI_INT,comm);CHKERRQ(ierr);
  bases[0] = 0;
  for (i=1; i<=size; i++) {
    bases[i] = ldims[i-1];
  }
  for (i=1; i<=size; i++) {
    bases[i] += bases[i-1];
  }

  /* allocate the base parallel and sequential vectors */
  ierr = VecCreateMPI(comm,x*y,PETSC_DECIDE,&global);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,dof);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,(Xe-Xs)*(Ye-Ys),&local);CHKERRQ(ierr);
  ierr = VecSetBlockSize(local,dof);CHKERRQ(ierr);


  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x*y,start,1,&to);CHKERRQ(ierr);

  left  = xs - Xs; down  = ys - Ys; up    = down + y;
  idx = (int*)PetscMalloc(x*(up - down)*sizeof(int));CHKPTRQ(idx);
  count = 0;
  for (i=down; i<up; i++) {
    for (j=0; j<x; j++) {
      idx[count++] = left + i*(Xe-Xs) + j;
    }
  }
  ierr = ISCreateGeneral(comm,count,idx,&from);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);

  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* global to local must include ghost points */
  if (stencil_type == DA_STENCIL_BOX) {
    ierr = ISCreateStride(comm,(Xe-Xs)*(Ye-Ys),0,1,&to);CHKERRQ(ierr); 
  } else {
    /* must drop into cross shape region */
    /*       ---------|
            |  top    |
         |---         ---|
         |   middle      |
         |               |
         ----         ----
            | bottom  |
            -----------
        Xs xs        xe  Xe */
    /* bottom */
    left  = xs - Xs; down = ys - Ys; up    = down + y;
    count = down*(xe-xs) + (up-down)*(Xe-Xs) + (Ye-Ys-up)*(xe-xs);
    idx   = (int*)PetscMalloc(count*sizeof(int));CHKPTRQ(idx);
    count = 0;
    for (i=0; i<down; i++) {
      for (j=0; j<xe-xs; j++) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    /* middle */
    for (i=down; i<up; i++) {
      for (j=0; j<Xe-Xs; j++) {
        idx[count++] = i*(Xe-Xs) + j;
      }
    }
    /* top */
    for (i=up; i<Ye-Ys; i++) {
      for (j=0; j<xe-xs; j++) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    ierr = ISCreateGeneral(comm,count,idx,&to);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr);
  }


  /* determine who lies on each side of use stored in    n6 n7 n8
                                                         n3    n5
                                                         n0 n1 n2
  */

  /* Assume the Non-Periodic Case */
  n1 = rank - m; 
  if (rank % m) {
    n0 = n1 - 1; 
  } else {
    n0 = -1;
  }
  if ((rank+1) % m) {
    n2 = n1 + 1;
    n5 = rank + 1;
    n8 = rank + m + 1; if (n8 >= m*n) n8 = -1;
  } else {
    n2 = -1; n5 = -1; n8 = -1;
  }
  if (rank % m) {
    n3 = rank - 1; 
    n6 = n3 + m; if (n6 >= m*n) n6 = -1;
  } else {
    n3 = -1; n6 = -1;
  }
  n7 = rank + m; if (n7 >= m*n) n7 = -1;


  /* Modify for Periodic Cases */
  if (wrap == DA_YPERIODIC) {  /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;
  } else if (wrap == DA_XPERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  } else if (wrap == DA_XYPERIODIC) {

    /* Handle all four corners */
    if ((n6 < 0) && (n7 < 0) && (n3 < 0)) n6 = m-1;
    if ((n8 < 0) && (n7 < 0) && (n5 < 0)) n8 = 0;
    if ((n2 < 0) && (n5 < 0) && (n1 < 0)) n2 = size-m;
    if ((n0 < 0) && (n3 < 0) && (n1 < 0)) n0 = size-1;   

    /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;

    /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  }

  if (stencil_type == DA_STENCIL_STAR) {
    /* save corner processor numbers */
    sn0 = n0; sn2 = n2; sn6 = n6; sn8 = n8; 
    n0 = n2 = n6 = n8 = -1;
  }

  idx = (int *)PetscMalloc((x+2*s_x)*(y+2*s_y)*sizeof(int));CHKPTRQ(idx);
  PLogObjectMemory(da,(x+2*s_x)*(y+2*s_y)*sizeof(int));
  nn = 0;

  xbase = bases[rank];
  for (i=1; i<=s_y; i++) {
    if (n0 >= 0) { /* left below */
      x_t = lx[n0 % m]*dof;
      y_t = ly[(n0/m)];
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = ly[(n1/m)];
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n2 >= 0) { /* right below */
      x_t = lx[n2 % m]*dof;
      y_t = ly[(n2/m)];
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=0; i<y; i++) {
    if (n3 >= 0) { /* directly left */
      x_t = lx[n3 % m]*dof;
      y_t = y;
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }

    for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = lx[n5 % m]*dof;
      y_t = y;
      s_t = bases[n5] + (i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=1; i<=s_y; i++) {
    if (n6 >= 0) { /* left above */
      x_t = lx[n6 % m]*dof;
      y_t = ly[(n6/m)];
      s_t = bases[n6] + (i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      y_t = ly[(n7/m)];
      s_t = bases[n7] + (i-1)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = lx[n8 % m]*dof;
      y_t = ly[(n8/m)];
      s_t = bases[n8] + (i-1)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  base = bases[rank];
  ierr = ISCreateGeneral(comm,nn,idx,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,gtol);
  ISDestroy(to); ISDestroy(from);

  if (stencil_type == DA_STENCIL_STAR) {
    /*
        Recompute the local to global mappings, this time keeping the 
      information about the cross corner processor numbers.
    */
    n0 = sn0; n2 = sn2; n6 = sn6; n8 = sn8;
    nn = 0;
    xbase = bases[rank];
    for (i=1; i<=s_y; i++) {
      if (n0 >= 0) { /* left below */
        x_t = lx[n0 % m]*dof;
        y_t = ly[(n0/m)];
        s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n1/m)];
        s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
      }
      if (n2 >= 0) { /* right below */
        x_t = lx[n2 % m]*dof;
        y_t = ly[(n2/m)];
        s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }

    for (i=0; i<y; i++) {
      if (n3 >= 0) { /* directly left */
        x_t = lx[n3 % m]*dof;
        y_t = y;
        s_t = bases[n3] + (i+1)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }

      for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

      if (n5 >= 0) { /* directly right */
        x_t = lx[n5 % m]*dof;
        y_t = y;
        s_t = bases[n5] + (i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n6 >= 0) { /* left above */
        x_t = lx[n6 % m]*dof;
        y_t = ly[(n6/m)];
        s_t = bases[n6] + (i)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        y_t = ly[(n7/m)];
        s_t = bases[n7] + (i-1)*x_t;
        for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
      }
      if (n8 >= 0) { /* right above */
        x_t = lx[n8 % m]*dof;
        y_t = ly[(n8/m)];
        s_t = bases[n8] + (i-1)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }
  }

  da->M  = M;  da->N  = N;  da->m  = m;  da->n  = n;  da->w = dof;  da->s = s;
  da->xs = xs; da->xe = xe; da->ys = ys; da->ye = ye; da->zs = 0; da->ze = 1;
  da->Xs = Xs; da->Xe = Xe; da->Ys = Ys; da->Ye = Ye; da->Zs = 0; da->Ze = 1;
  da->P  = 1;  da->p  = 1;

  PLogObjectParent(da,global);
  PLogObjectParent(da,local);

  da->global       = global; 
  da->local        = local; 
  da->gtol         = gtol;
  da->ltog         = ltog;
  da->idx          = idx;
  da->Nl           = nn;
  da->base         = base;
  da->wrap         = wrap;
  da->view         = DAView_2d;
  da->stencil_type = stencil_type;

  /* 
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  {
    ISLocalToGlobalMapping isltog;
    ierr        = ISLocalToGlobalMappingCreate(comm,nn,idx,&isltog);CHKERRQ(ierr);
    ierr        = VecSetLocalToGlobalMapping(da->global,isltog);CHKERRQ(ierr);
    da->ltogmap = isltog; 
    ierr = PetscObjectReference((PetscObject)isltog);CHKERRQ(ierr);
    PLogObjectParent(da,isltog);
    ierr = ISLocalToGlobalMappingDestroy(isltog);CHKERRQ(ierr);
  }

  *inra = da;

  /* recalculate the idx including missed ghost points */
  /* Assume the Non-Periodic Case */
  n1 = rank - m; 
  if (rank % m) {
    n0 = n1 - 1; 
  } else {
    n0 = -1;
  }
  if ((rank+1) % m) {
    n2 = n1 + 1;
    n5 = rank + 1;
    n8 = rank + m + 1; if (n8 >= m*n) n8 = -1;
  } else {
    n2 = -1; n5 = -1; n8 = -1;
  }
  if (rank % m) {
    n3 = rank - 1; 
    n6 = n3 + m; if (n6 >= m*n) n6 = -1;
  } else {
    n3 = -1; n6 = -1;
  }
  n7 = rank + m; if (n7 >= m*n) n7 = -1;


  /* Modify for Periodic Cases */
  if (wrap == DA_YPERIODIC) {  /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;
  } else if (wrap == DA_XPERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  } else if (wrap == DA_XYPERIODIC) {

    /* Handle all four corners */
    if ((n6 < 0) && (n7 < 0) && (n3 < 0)) n6 = m-1;
    if ((n8 < 0) && (n7 < 0) && (n5 < 0)) n8 = 0;
    if ((n2 < 0) && (n5 < 0) && (n1 < 0)) n2 = size-m;
    if ((n0 < 0) && (n3 < 0) && (n1 < 0)) n0 = size-1;   

    /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;

    /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  }

  nn = 0;

  xbase = bases[rank];
  for (i=1; i<=s_y; i++) {
    if (n0 >= 0) { /* left below */
      x_t = lx[n0 % m]*dof;
      y_t = ly[(n0/m)];
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = ly[(n1/m)];
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n2 >= 0) { /* right below */
      x_t = lx[n2 % m]*dof;
      y_t = ly[(n2/m)];
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=0; i<y; i++) {
    if (n3 >= 0) { /* directly left */
      x_t = lx[n3 % m]*dof;
      y_t = y;
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }

    for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = lx[n5 % m]*dof;
      y_t = y;
      s_t = bases[n5] + (i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=1; i<=s_y; i++) {
    if (n6 >= 0) { /* left above */
      x_t = lx[n6 % m]*dof;
      y_t = ly[(n6/m)];
      s_t = bases[n6] + (i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      y_t = ly[(n7/m)];
      s_t = bases[n7] + (i-1)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = lx[n8 % m]*dof;
      y_t = ly[(n8/m)];
      s_t = bases[n8] + (i-1)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }
  /* keep bases for use at end of routine */
  /* ierr = PetscFree(bases);CHKERRQ(ierr); */

  /* construct the local to local scatter context */
  /* 
      We simply remap the values in the from part of 
    global to local to read from an array with the ghost values 
    rather then from the plan array.
  */
  ierr = VecScatterCopy(gtol,&da->ltol);CHKERRQ(ierr);
  PLogObjectParent(da,da->ltol);
  left  = xs - Xs; down  = ys - Ys; up    = down + y;
  idx = (int*)PetscMalloc(x*(up - down)*sizeof(int));CHKPTRQ(idx);
  count = 0;
  for (i=down; i<up; i++) {
    for (j=0; j<x; j++) {
      idx[count++] = left + i*(Xe-Xs) + j;
    }
  }
  ierr = VecScatterRemap(da->ltol,idx,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscFree(idx);CHKERRQ(ierr);

  /* 
     Build the natural ordering to PETSc ordering mappings.
  */
  {
    IS  ispetsc,isnatural;
    int *lidx,lict = 0,Nlocal = (da->xe-da->xs)*(da->ye-da->ys);

    ierr = ISCreateStride(comm,Nlocal,da->base,1,&ispetsc);CHKERRQ(ierr);

    lidx = (int*)PetscMalloc(Nlocal*sizeof(int));CHKPTRQ(lidx);
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
        /*  global number in natural ordering */
        lidx[lict++] = i + j*M*dof;
      }
    }
    ierr = ISCreateGeneral(comm,Nlocal,lidx,&isnatural);CHKERRQ(ierr);
    ierr = PetscFree(lidx);CHKERRQ(ierr);

    ierr = AOCreateBasicIS(isnatural,ispetsc,&da->ao);CHKERRQ(ierr);
    PLogObjectParent(da,da->ao);
    ierr = ISDestroy(ispetsc);CHKERRQ(ierr);
    ierr = ISDestroy(isnatural);CHKERRQ(ierr);
  }
  if (!flx) {
    flx  = (int*)PetscMalloc(m*sizeof(int));CHKPTRQ(flx);
    ierr = PetscMemcpy(flx,lx,m*sizeof(int));CHKERRQ(ierr);
  }
  if (!fly) {
    fly  = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(fly);
    ierr = PetscMemcpy(fly,ly,n*sizeof(int));CHKERRQ(ierr);
  }
  da->lx = flx;
  da->ly = fly;

  /*
     Note the following will be removed soon. Since the functionality 
    is replaced by the above.
  */
  /* Construct the mapping from current global ordering to global
     ordering that would be used if only 1 processor were employed.
     This mapping is intended only for internal use by discrete
     function and matrix viewers.

     Note: At this point, x has already been adjusted for multiple
     degrees of freedom per node.
   */
  ldim = x*y;
  ierr = VecGetSize(global,&gdim);CHKERRQ(ierr);
  da->gtog1 = (int *)PetscMalloc(gdim*sizeof(int));CHKPTRQ(da->gtog1);
  PLogObjectMemory(da,gdim*sizeof(int));
  gA        = (int *)PetscMalloc((2*(gdim+ldim))*sizeof(int));CHKPTRQ(gA);
  gB        = (int *)(gA + ldim);
  gAall     = (int *)(gB + ldim);
  gBall     = (int *)(gAall + gdim);

  /* Compute local parts of global orderings */
  ict = 0;
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
      /* gA = global number for 1 proc; gB = current global number */
      gA[ict] = i + j*M*dof;
      gB[ict] = start + ict;
      ict++;
    }
  }
  /* Broadcast the orderings */
  ierr = MPI_Allgatherv(gA,ldim,MPI_INT,gAall,ldims,bases,MPI_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(gB,ldim,MPI_INT,gBall,ldims,bases,MPI_INT,comm);CHKERRQ(ierr);
  for (i=0; i<gdim; i++) da->gtog1[gBall[i]] = gAall[i];
  ierr = PetscFree(gA);CHKERRQ(ierr);
  ierr = PetscFree(bases);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-da_view",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-da_view_draw",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,VIEWER_DRAW_(da->comm));CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAPrintHelp(da);CHKERRQ(ierr);}

  PetscPublishAll(da);  
#if defined(PETSC_HAVE_AMS)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)global,"AMSSetFieldBlock_C",
         "AMSSetFieldBlock_DA",AMSSetFieldBlock_DA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)local,"AMSSetFieldBlock_C",
         "AMSSetFieldBlock_DA",AMSSetFieldBlock_DA);CHKERRQ(ierr);
  if (((PetscObject)global)->amem > -1) {
    ierr = AMSSetFieldBlock_DA(((PetscObject)global)->amem,"values",global);CHKERRQ(ierr);
  }
#endif
#if defined(PETSC_HAVE_MATLAB)
  if (dof == 1) {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)local,"PetscMatlabEnginePut_C","VecMatlabEnginePut_DA2d",VecMatlabEnginePut_DA2d);CHKERRQ(ierr);
  }
#endif
  ierr = VecSetOperation(global,VECOP_VIEW,(void*)VecView_MPI_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(global,VECOP_LOADINTOVECTOR,(void*)VecLoadIntoVector_Binary_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAPrintHelp"
/*@
   DAPrintHelp - Prints command line options for DA.

   Collective on DA

   Input Parameters:
.  da - the distributed array

   Level: intermediate

.seealso: DACreate1d(), DACreate2d(), DACreate3d()

.keywords: DA, help

@*/
int DAPrintHelp(DA da)
{
  static PetscTruth called = PETSC_FALSE;
  MPI_Comm          comm;
  int               ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);

  comm = da->comm;
  if (!called) {
    ierr = (*PetscHelpPrintf)(comm,"General Distributed Array (DA) options:\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -da_view: print DA distribution to screen\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -da_view_draw: display DA in window\n");CHKERRQ(ierr);
    called = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DARefine"
/*@
   DARefine - Creates a new distributed array that is a refinement of a given
   distributed array.

   Collective on DA

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  daref - refined distributed array

   Level: advanced

   Note:
   Currently, refinement consists of just doubling the number of grid spaces
   in each dimension of the DA.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy()
@*/
int DARefine(DA da,DA *daref)
{
  int M,N,P,ierr;
  DA  da2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);

  M = 2*da->M - 1; N = 2*da->N - 1; P = 2*da->P - 1;
  if (da->dim == 1) {
    ierr = DACreate1d(da->comm,da->wrap,M,da->w,da->s,PETSC_NULL,&da2);CHKERRQ(ierr);
  } else if (da->dim == 2) {
    ierr = DACreate2d(da->comm,da->wrap,da->stencil_type,M,N,da->m,da->n,da->w,da->s,PETSC_NULL,
                      PETSC_NULL,&da2);CHKERRQ(ierr);
  } else if (da->dim == 3) {
    ierr = DACreate3d(da->comm,da->wrap,da->stencil_type,M,N,P,da->m,da->n,da->p,
           da->w,da->s,0,0,0,&da2);CHKERRQ(ierr);
  }
  *daref = da2;
  PetscFunctionReturn(0);
}
