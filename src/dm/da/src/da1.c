
/* 
   This file was created by Peter Mell   6/30/95    

   Code for manipulating distributed regular 1d arrays in parallel.
*/


#include "daimpl.h"
#include "pviewer.h"
#include <math.h>

#define WRAP 1
#define NO_WRAP 0

/*@
    DACreate1d - Creates a one-dimensional regular array that is
    distributed across some processors.

   Input Parameters:
.  M - global dimension of the array
.  w - number of degress of freedom per node
.  s - stencil width  
.  wrap - Do you want ghost points to wrap around? 
          0=No Wrap 1=Wrap

   Output Parameter:
.  inra - the resulting array object

.keywords: distributed array, create
.seealso: DADestroy(), DAView()
@*/

int DACreate1d(MPI_Comm comm,int M, int w,
               int s, int wrap, DA *inra)
{
  int           mytid, numtid,xs,xe,x,Xs,Xe,ierr,start,end,m;
  int           i,*idx,nn;
  int           j,x_t,s_t;
  DA            da;
  Vec           local,global;
  VecScatterCtx ltog,gtol;
  IS            to,from;
  *inra = 0;

  MPI_Comm_size(comm,&numtid); 
  MPI_Comm_rank(comm,&mytid); 

  m = numtid;

  if (M < m)     SETERRQ(1,"DACreate1d: More processors than data points!");
  if ((M-1) < s) SETERRQ(1,"DACreate1d: Array is too small for stencil!");

  /* determine local owned region */
  x = M/m + ((M % m) > (mytid));

  if (mytid >= (M % m)) {xs = (mytid * (int) (M/m) + M % m);}
    else {xs = mytid * (int)(M/m) + mytid;}

  /* From now on x,s,xs,xe,Xs,Xe are the exact location in the array */

  x  *= w;
  s  *= w;  /* NOTE: I change s to be absolute stencil distance */
  xs *= w;
  xe = xs + x;

  /* determine ghost region */
  if (wrap) {
    Xs = xs - s; 
    Xe = xe + s;
  }
  else
  {
    if ((xs-s) >= 0)   Xs = xs-s;  else Xs = 0; 
    if ((xe+s) <= M*w) Xe = xe+s;  else Xe = M*w;    
  }

  /* allocate the base parallel and sequential vectors */
  ierr = VecCreateMPI(comm,x,PETSC_DECIDE,&global); CHKERRQ(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,(Xe-Xs),&local); CHKERRQ(ierr);
    
  /* Create Local to Global Vector Scatter Context */
  /* local to global inserts non-ghost point region into global */
  VecGetOwnershipRange(global,&start,&end);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,x,start,1,&to);   CHKERRQ(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,x,xs-Xs,1,&from); CHKERRQ(ierr);
  ierr = VecScatterCtxCreate(local,from,global,to,&ltog); CHKERRQ(ierr);
  ISDestroy(from); ISDestroy(to);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */

  ierr = ISCreateStrideSequential(MPI_COMM_SELF,(Xe-Xs),0,1,&to); CHKERRQ(ierr);
 
  idx = (int *) PETSCMALLOC( (x+2*s)*sizeof(int) ); CHKPTRQ(idx);  

  nn = 0;
  if (wrap) {    /* Handle all cases with wrap first */

    for (i=0; i<s; i++) {  /* Left ghost points */
      if ((xs-s+i)>=0) { idx[nn++] = xs-s+i;}
      else             { idx[nn++] = M*w+(xs-s+i);}
    }

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}  /* Non-ghost points */
    
    for (i=0; i<s; i++) { /* Right ghost points */
      if ((xe+i)<M*w) { idx [nn++] =  xe+i; }
      else              { idx [nn++] = (xe+i) - M*w;}
    }
  }

  else {      /* Now do all cases with no wrapping */

    if (s <= xs) {for (i=0; i<s; i++) {idx[nn++] = xs - s + i;}}
    else         {for (i=0; i<xs;  i++) {idx[nn++] = i;}}

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}
    
    if ((xe+s)<=M*w) {for (i=0;  i<s;     i++) {idx[nn++]=xe+i;}}
    else             {for (i=xe; i<(M*w); i++) {idx[nn++]=i;   }}
  }

  ierr = ISCreateSequential(comm,nn,idx,&from); CHKERRQ(ierr);
  ierr = VecScatterCtxCreate(global,from,local,to,&gtol); CHKERRQ(ierr);
  ISDestroy(to); ISDestroy(from);

  PETSCHEADERCREATE(da,_DA,DA_COOKIE,0,comm);
  PLogObjectCreate(da);
  da->M  = M; da->N = 1;  da->m  = m; da->n = 1; da->w = w; da->s = s/w;
  da->xs = xs; da->xe = xe; da->ys = 0; da->ye = 1;
  da->Xs = Xs; da->Xe = Xe; da->Ys = 0; da->Ye = 1;
  da->global = global; 
  da->local  = local;
  da->gtol   = gtol;
  da->ltog   = ltog;
  da->idx    = idx;
  da->Nl     = nn;
  da->base   = xs;
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
  VALIDHEADER(da,DA_COOKIE);
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
  VALIDHEADER(da,DA_COOKIE);
  *x = da->Xs; *m = da->Xe - da->Xs;
  if (y) *y = da->Ys; if (n) *n = da->Ye - da->Ys;
  if (z) *z = da->Zs; if (p) *p = da->Ze - da->Zs; 
  return 0;
}

/*@
   DADestroy - Destroy a distributed array.

   Input Parameter:
.  da - the distributed array to destroy 

.keywords: distributed array, destroy

.seealso: DACreate2d()
@*/
int DADestroy(DA da)
{
  VALIDHEADER(da,DA_COOKIE);
  PLogObjectDestroy(da);
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
  VALIDHEADER(da,DA_COOKIE);
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
  VALIDHEADER(da,DA_COOKIE);
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
  VALIDHEADER(da,DA_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTERALL,da->gtol); CHKERRQ(ierr);
  return 0;
}

/*@
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
  VALIDHEADER(da,DA_COOKIE);
  *g = da->global;
  return 0;
}

/*@
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
  VALIDHEADER(da,DA_COOKIE);
  *l = da->local;
  return 0;
}

#include "draw.h"
/*@
   DAView1d - Visualizes a distributed array object in 1d.

   Input Parameters:
.  da - the distributed array
.  ptr - an optional visualization context

   Notes:
   The available visualization contexts include
$     STDOUT_VIEWER - standard output (the default)
$     SYNC_STDOUT_VIEWER - synchronized standard
$        output, where only the first processor opens
$        the file.  All other processors send their 
$        data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpen() - output to a specified file
$    ViewerFileOpenSync() - synchronized output to a 
$         specified file
$    DrawOpenX() - output nonzero matrix structure to 
$         an X window display

.keywords: distributed array, view, visualize

.seealso: ViewerFileOpen(), ViewerFileOpenSync(), DrawOpenX(), 
@*/
int DAView1d(DA da,Viewer ptr)
{
  PetscObject vobj = (PetscObject)ptr;
  int         mytid;
  VALIDHEADER(da,DA_COOKIE);

  MPI_Comm_rank(da->comm,&mytid); 

  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER; vobj = (PetscObject) ptr;
  }

  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;

  if (vobj->cookie == VIEWER_COOKIE) {
    FILE *fd = ViewerFileGetPointer_Private(ptr);
    if (vobj->type == FILE_VIEWER) {
      MPIU_Seq_begin(da->comm,1);
      fprintf(fd,"Processor [%d] M %d N %d m %d n %d w %d s %d\n",mytid,da->M,
                 1,da->m,1,da->w,da->s);
      fprintf(fd,"X range %d %d Y range %d %d\n",da->xs,da->xe,1,1);
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
    double  ymin = -1,ymax = 1,xmin = -1,xmax = da->M,x,y;
    int     base,*idx;
    char    node[10];

    DrawSetCoordinates(win,xmin,ymin,xmax,ymax);

    /* first processor draws all node lines */

    if (!mytid) {
      ymin = 0.0; ymax = 0.3;

      for ( xmin=0; xmin<da->M; xmin++ ) {
         DrawLine(win,xmin,ymin,xmin,ymax,DRAW_BLACK,DRAW_BLACK);
      }

      xmin = 0.0; xmax = da->M - 1;
      DrawLine(win,xmin,ymin,xmax,ymin,DRAW_BLACK,DRAW_BLACK);
      DrawLine(win,xmin,ymax,xmax,ymax,DRAW_BLACK,DRAW_BLACK);
    }

    DrawSyncFlush(win); 
    MPI_Barrier(da->comm);

    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = da->xs / da->w; xmax = (da->xe / da->w)  - 1;
    DrawLine(win,xmin,ymin,xmax,ymin,DRAW_RED,DRAW_RED);
    DrawLine(win,xmin,ymin,xmin,ymax,DRAW_RED,DRAW_RED);
    DrawLine(win,xmin,ymax,xmax,ymax,DRAW_RED,DRAW_RED);
    DrawLine(win,xmax,ymin,xmax,ymax,DRAW_RED,DRAW_RED);

    /* Put in index numbers */
    base = da->base / da->w;
    for ( x=xmin; x<=xmax; x++ ) {
      sprintf(node,"%d",base++);
      DrawText(win,x,y,DRAW_RED,node);
    }

    DrawSyncFlush(win);
 
  }
  return 0;
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
  VALIDHEADER(da,DA_COOKIE);
  *n   = da->Nl;
  *idx = da->idx;
  return 0;
}

/*@
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
  VALIDHEADER(da,DA_COOKIE);
  *ltog = da->ltog;
  *gtol = da->gtol;
  return 0;
}









