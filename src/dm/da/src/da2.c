 
/* Peter Mell changed this file on 7/10/95 */
 
/*

    Code for manipulating distributed regular arrays in parallel.

*/
#include "daimpl.h"    /*I   "da.h"   I*/
#include "pinclude/pviewer.h"
#include "draw.h"
#include <math.h>

static int DAView_2d(PetscObject dain,Viewer ptr)
{
  DA          da = (DA) dain;
  PetscObject vobj = (PetscObject)ptr;
  int         mytid;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);

  MPI_Comm_rank(da->comm,&mytid); 

  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }

  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;

  if (vobj->cookie == VIEWER_COOKIE) {
    FILE *fd = ViewerFileGetPointer_Private(ptr);
    if (vobj->type == FILE_VIEWER) {
      MPIU_Seq_begin(da->comm,1);
      fprintf(fd,"Processor [%d] M %d N %d m %d n %d w %d s %d\n",mytid,da->M,
                 da->N,da->m,da->n,da->w,da->s);
      fprintf(fd,"X range %d %d Y range %d %d\n",da->xs,da->xe,da->ys,da->ye);
      fflush(fd);
      MPIU_Seq_end(da->comm,1);
    }
    else if (vobj->type == FILES_VIEWER) {

      if (!mytid) {
      }
      else {
      }
    }
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    DrawCtx win = (DrawCtx) ptr;
    double  ymin = -1*da->s-1, ymax = da->N+da->s;
    double  xmin = -1*da->s-1, xmax = da->M+da->s;
    double  x,y;
    int     base,*idx;
    char    node[10];
 
    DrawSetCoordinates(win,xmin,ymin,xmax,ymax);

    /* first processor draw all node lines */
    if (!mytid) {
      ymin = 0.0; ymax = da->N - 1;
      for ( xmin=0; xmin<da->M; xmin++ ) {
        DrawLine(win,xmin,ymin,xmin,ymax,DRAW_BLACK);
      }
      xmin = 0.0; xmax = da->M - 1;
      for ( ymin=0; ymin<da->N; ymin++ ) {
        DrawLine(win,xmin,ymin,xmax,ymin,DRAW_BLACK);
      }
    }
    DrawSyncFlush(win);
    MPI_Barrier(da->comm);

    /* draw my box */
    ymin = da->ys; ymax = da->ye - 1; xmin = da->xs/da->w; 
    xmax =(da->xe-1)/da->w;
    DrawLine(win,xmin,ymin,xmax,ymin,DRAW_RED);
    DrawLine(win,xmin,ymin,xmin,ymax,DRAW_RED);
    DrawLine(win,xmin,ymax,xmax,ymax,DRAW_RED);
    DrawLine(win,xmax,ymin,xmax,ymax,DRAW_RED);

    /* put in numbers */
    base = (da->base)/da->w;
    for ( y=ymin; y<=ymax; y++ ) {
      for ( x=xmin; x<=xmax; x++ ) {
        sprintf(node,"%d",base++);
        DrawText(win,x,y,DRAW_BLACK,node);
      }
    }

    DrawSyncFlush(win);
    MPI_Barrier(da->comm);

    if (da->stencil_type == DA_STENCIL_BOX) {

      /* overlay ghost numbers, useful for error checking */
      /* put in numbers */

      base = 0; idx = da->idx;
      ymin = da->Ys; ymax = da->Ye; xmin = da->Xs; xmax = da->Xe;
      for ( y=ymin; y<ymax; y++ ) {
        for ( x=xmin; x<xmax; x++ ) {
          if ((base % da->w) == 0) {
            sprintf(node,"%d",idx[base]/da->w);
            DrawText(win,x/da->w,y,DRAW_BLUE,node);
          }
          base++;
        }
      }        
    }
    else  /* Print ghost points with star stencil */
    {
      /* overlay ghost numbers, useful for error checking */
      /* put in numbers */

      /* Bottom part */
      base = 0; idx = da->idx;
      ymin = da->Ys; ymax = da->ys; xmin = da->xs; xmax = da->xe;
      for ( y=ymin; y<ymax; y++ ) {
        for ( x=xmin; x<xmax; x++ ) {
          if ((base % da->w) == 0) {
            sprintf(node,"%d",idx[base]/da->w);
            DrawText(win,x/da->w,y,DRAW_BLUE,node);
          }
          base++;
        }
      }      
  
      /* Middle part */
      ymin = da->ys; ymax = da->ye; xmin = da->Xs; xmax = da->Xe;
      for ( y=ymin; y<ymax; y++ ) {
        for ( x=xmin; x<xmax; x++ ) {
          if ((base % da->w) == 0) {
            sprintf(node,"%d",idx[base]/da->w);
            DrawText(win,x/da->w,y,DRAW_BLUE,node);
          }
          base++;
        }
      }      
  
      /* Top part */
      ymin = da->ye; ymax = da->Ye; xmin = da->xs; xmax = da->xe;
      for ( y=ymin; y<ymax; y++ ) {
        for ( x=xmin; x<xmax; x++ ) {
          if ((base % da->w) == 0) {
            sprintf(node,"%d",idx[base]/da->w);
            DrawText(win,x/da->w,y,DRAW_BLUE,node);
          }
          base++;
        }
      }      

    }      


    DrawSyncFlush(win);
  }
  return 0;
}

/*@C
    DACreate2d - Creates a two-dimensional regular array that is
    distributed across some processors.

   Input Parameters:
.  stencil_type - stencil type either DA_STENCIL_BOX or DA_STENCIL_STAR
.  M,N - global dimension in each direction of the array
.  m,n - corresponding number of processors in each dimension 
         (or PETSC_DECIDE to have calculated)
.  w - number of degress of freedom per node
.  s - stencil width
.  wrap - Are you using a periodic domain? Choose from 
$         DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC  

   Output Parameter:
.  inra - the resulting array object

.keywords: distributed array, create, two-dimensional
.seealso: DADestroy(), DAView()
@*/
int DACreate2d(MPI_Comm comm,DAPeriodicType wrap,DAStencilType stencil_type,
                int M, int N, int m,int n, int w, int s, DA *inra)
{
  int           mytid, numtid,xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,ierr,start,end;
  int           up,down,left,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn;
  int           xbase,*bases,j,x_t,y_t,s_t,base;
  int           s_x,s_y; /* s proportionalized to w */
  DA            da;
  Vec           local,global;
  VecScatterCtx ltog,gtol;
  IS            to,from;
  *inra = 0;

  PETSCHEADERCREATE(da,_DA,DA_COOKIE,0,comm);
  PLogObjectCreate(da);
  PLogObjectMemory(da,sizeof(struct _DA));

  MPI_Comm_size(comm,&numtid); 
  MPI_Comm_rank(comm,&mytid); 

  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int) sqrt( ((double)M)*((double)numtid)/((double)N) );
    if (m == 0) m = 1;
    while (m > 0) {
      n = numtid/m;
      if (m*n == numtid) break;
      m--;
    }
    if (M > N && m < n) {int _m = m; m = n; n = _m;}
    if (m*n != numtid)SETERRQ(1,"DaCreate2d:Internally Created Bad Partition");
  }
  else if (m*n != numtid) SETERRQ(1,"DACreate2d: Given Bad partition"); 

  if (M < m) SETERRQ(1,"DACreate2d: Partition in x direction is too fine!");
  if (N < n) SETERRQ(1,"DACreate2d: Partition in y direction is too fine!");

  /* determine local owned region */
  x = M/m + ((M % m) > (mytid % m));
  y = N/n + ((N % n) > (mytid/m));

  if (x < s) SETERRQ(1,"DACreate2d: Column width is too thin for stencil!");
  if (y < s) SETERRQ(1,"DACreate2d: Row width is too thin for stencil!");
  if ((M % m) > (mytid % m)) { xs = (mytid % m)*x; }
  else { xs = (M % m)*(x+1) + ((mytid % m)-(M % m))*x; }
  xe = xs + x;
  if ((N % n) > (mytid/m)) { ys = (mytid/m)*y; }
  else { ys = (N % n)*(y+1) + ((mytid/m)-(N % n))*y; }
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
  x   *= w;
  xs  *= w;
  xe  *= w;
  Xs  *= w;
  Xe  *= w;
  s_x = s*w;
  s_y = s;

  /* determine starting point of each processor */
  nn = x*y;
  bases = (int *) PETSCMALLOC( (numtid+1)*sizeof(int) ); CHKPTRQ(bases);
  MPI_Allgather(&nn,1,MPI_INT,bases+1,1,MPI_INT,comm);
  bases[0] = 0;
  for ( i=1; i<=numtid; i++ ) {
    bases[i] += bases[i-1];
  }

  /* allocate the base parallel and sequential vectors */
  ierr = VecCreateMPI(comm,x*y,PETSC_DECIDE,&global); CHKERRQ(ierr);
  if (stencil_type == DA_STENCIL_BOX)
    { ierr = VecCreateSequential(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys),&local);}
  else
    { ierr = VecCreateSequential(MPI_COMM_SELF,(Xe-Xs)*y+(Ye-ye)*x+(ys-Ys)*x,
                                 &local);}
  CHKERRQ(ierr);

  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  VecGetOwnershipRange(global,&start,&end);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,x*y,start,1,&to);CHKERRQ(ierr);

  if (stencil_type == DA_STENCIL_BOX) {
    left = xs - Xs; 
    down = ys - Ys; up    = down + y;
    from = 0;
    for ( i=down; i<up; i++ ) {
      ierr = ISAddStrideSequential(&from,x,left + i*(Xe-Xs),1); CHKERRQ(ierr);
    }
  }
  else { 
    left = xs - Xs; 
    from = 0;
    for ( i=0; i<y; i++ ) {
      ierr = ISAddStrideSequential(&from,x,(ys-Ys)*x+left + i*(Xe-Xs),1); 
      CHKERRQ(ierr);
    }
  }

  ierr = VecScatterCtxCreate(local,from,global,to,&ltog); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* global to local must include ghost points */
  if (stencil_type == DA_STENCIL_BOX)
    { ierr = ISCreateStrideSequential(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys),0,1,&to);}
  else
    { ierr = ISCreateStrideSequential(MPI_COMM_SELF,
                                  (Xe-Xs)*y + (Ye-ye)*x + (ys-Ys)*x,0,1,&to);}
  CHKERRQ(ierr); 

  /* determine who lies on each side of use stored in    n6 n7 n8
                                                         n3    n5
                                                         n0 n1 n2
  */

  /* Assume the Non-Periodic Case */
  n1 = mytid - m; 
  if (mytid % m) {
    n0 = n1 - 1; 
  }
  else {
    n0 = -1;
  }
  if ((mytid+1) % m) {
    n2 = n1 + 1;
    n5 = mytid + 1;
    n8 = mytid + m + 1; if (n8 >= m*n) n8 = -1;
  }
  else {
    n2 = -1; n5 = -1; n8 = -1;
  }
  if (mytid % m) {
    n3 = mytid - 1; 
    n6 = n3 + m; if (n6 >= m*n) n6 = -1;
  }
  else {
    n3 = -1; n6 = -1;
  }
  n7 = mytid + m; if (n7 >= m*n) n7 = -1;


  /* Modify for Periodic Cases */
  if (wrap == DA_YPERIODIC) {  /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = mytid + m * (n-1);
    if (n7 < 0) n7 = mytid - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = numtid - m + mytid - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (mytid%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = numtid - m + mytid + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (mytid%m)+1;
  } 
  else if (wrap == DA_XPERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = mytid + (m-1);
    if (n5 < 0) n5 = mytid - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = mytid-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = mytid-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = mytid+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = mytid+1;
  }
  else if (wrap == DA_XYPERIODIC) {

    /* Handle all four corners */
    if ((n6 < 0) && (n7 < 0) && (n3 < 0)) n6 = m-1;
    if ((n8 < 0) && (n7 < 0) && (n5 < 0)) n8 = 0;
    if ((n2 < 0) && (n5 < 0) && (n1 < 0)) n2 = numtid-m;
    if ((n0 < 0) && (n3 < 0) && (n1 < 0)) n0 = numtid-1;   

    /* Handle sides */

    /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = mytid + m * (n-1);
    if (n7 < 0) n7 = mytid - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = numtid - m + mytid - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (mytid%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = numtid - m + mytid + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (mytid%m)+1;

    /* Handle Left and Right Sides */
    if (n3 < 0) n3 = mytid + (m-1);
    if (n5 < 0) n5 = mytid - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = mytid-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = mytid-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = mytid+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = mytid+1;
  }

  if (stencil_type == DA_STENCIL_STAR) {n0 = n2 = n6 = n8 = -1;}

  idx = (int *)PETSCMALLOC((x+2*s_x)*(y+2*s_y)*sizeof(int));CHKPTRQ(idx);
  PLogObjectMemory(da,(x+2*s_x)*(y+2*s_y)*sizeof(int));
  nn = 0;

  xbase = bases[mytid];
  for ( i=1; i<=s_y; i++ ) {
    if (n0 >= 0) { /* left below */
      x_t = (M/m + ((M % m) > (n0 % m)))*w;
      y_t = N/n + ((N % n) > (n0/m));
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = N/n + ((N % n) > (n1/m));
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
    }
    if (n2 >= 0) { /* right below */
      x_t = (M/m + ((M % m) > (n2 % m)))*w;
      y_t = N/n + ((N % n) > (n2/m));
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  for ( i=0; i<y; i++ ) {
    if (n3 >= 0) { /* directly left */
      x_t = (M/m + ((M % m) > (n3 % m)))*w;
      y_t = y;
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }

    for ( j=0; j<x; j++ ) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = (M/m + ((M % m) > (n5 % m)))*w;
      y_t = y;
      s_t = bases[n5] + (i)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  for ( i=1; i<=s_y; i++ ) {
    if (n6 >= 0) { /* left above */
      x_t = (M/m + ((M % m) > (n6 % m)))*w;
      y_t = N/n + ((N % n) > (n6/m));
      s_t = bases[n6] + (i)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      y_t = N/n + ((N % n) > (n7/m));
      s_t = bases[n7] + (i-1)*x_t;
      for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = (M/m + ((M % m) > (n8 % m)))*w;
      y_t = N/n + ((N % n) > (n8/m));
      s_t = bases[n8] + (i-1)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  base = bases[mytid];
  PETSCFREE(bases);
  ierr = ISCreateSequential(comm,nn,idx,&from); CHKERRQ(ierr);
  ierr = VecScatterCtxCreate(global,from,local,to,&gtol); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,gtol);
  ISDestroy(to); ISDestroy(from);

  da->M  = M;  da->N  = N;  da->m  = m;  da->n  = n; da->w = w; da->s = s;
  da->xs = xs; da->xe = xe; da->ys = ys; da->ye = ye;
  da->Xs = Xs; da->Xe = Xe; da->Ys = Ys; da->Ye = Ye;

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
  *inra = da;
  return 0;
}

/*@
   DAGetCorners - Returns the global (x,y,z) indices of the lower left
   corner of the local region, excluding ghost points.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  x,y,z - the corner indices. y and z are optional.
.  m,n,p - widths in the corresponding directions. n and p are optional.

.keywords: distributed array, get, corners, nodes, local indices

.seealso: DAGetGhostCorners()
@*/
int DAGetCorners(DA da,int *x,int *y,int *z,int *m, int *n, int *p)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *x = da->xs; *m = da->xe - da->xs;
  if (y) *y = da->ys; if (n) *n = da->ye - da->ys;
  if (z) *z = da->zs; if (p) *p = da->ze - da->zs; 
  return 0;
} 

/*@
    DAGetGhostCorners - Returns the global (x,y,z) indices of the lower left
    corner of the local region, including ghost points.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  x,y,z - the corner indices. y and z are optional.
.  m,n,p - widths in the corresponding directions. n and p are optional.

.keywords: distributed array, get, ghost, corners, nodes, local indices

.seealso: DAGetCorners()
@*/
int DAGetGhostCorners(DA da,int *x,int *y,int *z,int *m, int *n, int *p)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *x = da->Xs; *m = da->Xe - da->Xs;
  if (y) *y = da->Ys; if (n) *n = da->Ye - da->Ys;
  if (z) *z = da->Zs; if (p) *p = da->Ze - da->Zs; 
  return 0;
}

/*@C
   DADestroy - Destroy a distributed array.

   Input Parameter:
.  da - the distributed array to destroy 

.keywords: distributed array, destroy

.seealso: DACreate2d()
@*/
int DADestroy(DA da)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  PLogObjectDestroy(da);
  PETSCFREE(da->idx);
  VecScatterCtxDestroy(da->ltog);
  VecScatterCtxDestroy(da->gtol);
  PETSCHEADERDESTROY(da);
  return 0;
}

/*@
   DALocalToGlobal - Maps values from the local patch back to the 
   global vector. The ghost points are discarded.

   Input Parameters:
.  da - the distributed array context
.  l  - the local values
.  mode - one of INSERTVALUES or ADDVALUES

   Output Parameter:
.  g - the global vector

.keywords: distributed array, local to global

.seealso: DAGlobalToLocalBegin(), DACreate2d()
@*/
int DALocalToGlobal(DA da,Vec l, InsertMode mode,Vec g)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  ierr = VecScatterBegin(l,g,mode,SCATTERALL,da->ltog); CHKERRQ(ierr);
  ierr = VecScatterEnd(l,g,mode,SCATTERALL,da->ltog); CHKERRQ(ierr);
  return 0;
}

/*@
   DAGlobalToLocalBegin - Maps values from the global vector to the local
   patch, the ghost points are included. Must be followed by 
   DAGlobalToLocalEnd() to complete the exchange.

   Input Parameters:
.  da - the distributed array context
.  g - the global vector
.  mode - one of INSERTVALUES or ADDVALUES

   Output Parameter:
.  l  - the local values

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  ierr = VecScatterBegin(g,l,mode,SCATTERALL,da->gtol); CHKERRQ(ierr);
  return 0;
}

/*@
   DAGlobalToLocalEnd - Maps values from the global vector to the local
   patch, the ghost points are included. Must be preceeded by 
   DAGlobalToLocalBegin().

   Input Parameters:
.  da - the distributed array context
.  g - the global vector
.  mode - one of INSERTVALUES or ADDVALUES

   Output Parameter:
.  l  - the local values

.keywords: distributed array, global to local, end

.seealso: DAGlobalToLocalBegin(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTERALL,da->gtol); CHKERRQ(ierr);
  return 0;
}

/*@C
   DAGetDistributedVector - Gets a distributed vector for a 
   distributed array.  Additional vectors of the same type can be 
   created with VecDuplicate().

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed vector

.keywords: distributed array, get, global, distributed, vector

.seealso: DAGetLocalVector()
@*/
int   DAGetDistributedVector(DA da,Vec* g)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *g = da->global;
  return 0;
}

/*@C
   DAGetLocalVector - Gets a local vector (including ghost points) for a 
   distributed array.  Additional vectors of the same type can be created 
   with VecDuplicate().

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  l - the distributed vector

.keywords: distributed array, get, local, vector

.seealso: DAGetDistributedVector()
@*/
int   DAGetLocalVector(DA da,Vec* l)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *l = da->local;
  return 0;
}


/*@
   DAView - Visualizes a distributed array object.

   Input Parameters:
.  da - the distributed array
.  ptr - an optional visualization context

   Notes:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpen() - output to a specified file
$    DrawOpenX() - output nonzero matrix structure to 
$         an X window display

.keywords: distributed array, view, visualize

.seealso: ViewerFileOpen(), DrawOpenX(), 
@*/
int DAView(DA da, Viewer v)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  return (*da->view)((PetscObject)da,v);
}  

/*@
   DAGetGlobalIndices - Returns the global node number of all local nodes,
   including ghost nodes.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  n - the number of local elements, including ghost nodes
.  idx - the global indices

.keywords: distributed array, get, global, indices, local to global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal()
@*/
int DAGetGlobalIndices(DA da, int *n,int **idx)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *n   = da->Nl;
  *idx = da->idx;
  return 0;
}

/*@C
   DAGetScatterCtx - Gets the local to global and local to global 
   vector scatter contexts for a distributed array.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ltog - local to global scatter context
.  gtol - global to local scatter context

.keywords: distributed array, get, scatter, context, global to local,
           local to global

.seealso: DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DAGetScatterCtx(DA da, VecScatterCtx *ltog,VecScatterCtx *gtol)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *ltog = da->ltog;
  *gtol = da->gtol;
  return 0;
}
 
