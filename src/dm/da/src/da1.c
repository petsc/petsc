#ifndef lint
static char vcid[] = "$Id: da1.c,v 1.25 1996/01/26 04:35:39 bsmith Exp curfman $";
#endif

/* 
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95    
*/

#include "daimpl.h"     /*I  "da.h"   I*/
#include "pinclude/pviewer.h"   
#include <math.h>
#include "draw.h"      /*I  "draw.h"  I*/

static int DAView_1d(PetscObject pobj,Viewer ptr)
{
  DA da  = (DA) pobj;
  PetscObject vobj = (PetscObject)ptr;
  int         rank, ierr;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);

  MPI_Comm_rank(da->comm,&rank); 

  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }

  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;

  if (vobj->cookie == VIEWER_COOKIE) {
    FILE *fd;
    ierr = ViewerFileGetPointer(ptr,&fd); CHKERRQ(ierr);
    if (vobj->type == ASCII_FILE_VIEWER) {
      MPIU_Seq_begin(da->comm,1);
      fprintf(fd,"Processor [%d] M %d m %d w %d s %d\n",rank,da->M,
                 da->m,da->w,da->s);
      fprintf(fd,"X range %d %d Y range %d %d\n",da->xs,da->xe,1,1);
      fflush(fd);
      MPIU_Seq_end(da->comm,1);
    }
    else if (vobj->type == ASCII_FILES_VIEWER) {

      if (!rank) {
      }
      else {
      }
    }
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    Draw win = (Draw) ptr;
    double  ymin = -1,ymax = 1,xmin = -1,xmax = da->M,x;
    int     base;
    char    node[10];

    DrawSetCoordinates(win,xmin,ymin,xmax,ymax);

    /* first processor draws all node lines */

    if (!rank) {
      ymin = 0.0; ymax = 0.3;

      for ( xmin=0; xmin<da->M; xmin++ ) {
         DrawLine(win,xmin,ymin,xmin,ymax,DRAW_BLACK);
      }

      xmin = 0.0; xmax = da->M - 1;
      DrawLine(win,xmin,ymin,xmax,ymin,DRAW_BLACK);
      DrawLine(win,xmin,ymax,xmax,ymax,DRAW_BLACK);
    }

    DrawSyncFlush(win); 
    DrawPause(win);
    MPI_Barrier(da->comm);

    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = da->xs / da->w; xmax = (da->xe / da->w)  - 1;
    DrawLine(win,xmin,ymin,xmax,ymin,DRAW_RED);
    DrawLine(win,xmin,ymin,xmin,ymax,DRAW_RED);
    DrawLine(win,xmin,ymax,xmax,ymax,DRAW_RED);
    DrawLine(win,xmax,ymin,xmax,ymax,DRAW_RED);

    /* Put in index numbers */
    base = da->base / da->w;
    for ( x=xmin; x<=xmax; x++ ) {
      sprintf(node,"%d",base++);
      DrawText(win,x,ymin,DRAW_RED,node);
    }

    DrawSyncFlush(win);
    DrawPause(win); 
  }
  return 0;
}

/*@C
    DACreate1d - Creates a one-dimensional regular array that is
    distributed across some processors.

   Input Parameters:
.  comm - MPI communicator
.  wrap - Do you want ghost points to wrap around? Use one of
$         DA_NONPERIODIC, DA_XPERIODIC
.  M - global dimension of the array
.  w - number of degress of freedom per node
.  s - stencil width  

   Output Parameter:
.  inra - the resulting array object

.keywords: distributed array, create, one-dimensional

.seealso: DADestroy(), DAView(), DACreate2d(), DACreate3d()
@*/
int DACreate1d(MPI_Comm comm,DAPeriodicType wrap,int M,int w,int s,DA *inra)
{
  int           rank, size,xs,xe,x,Xs,Xe,ierr,start,end,m;
  int           i,*idx,nn,j,count,left;
  DA            da;
  Vec           local,global;
  VecScatter    ltog,gtol;
  IS            to,from;
  *inra = 0;

  PetscHeaderCreate(da,_DA,DA_COOKIE,0,comm);
  PLogObjectCreate(da);
  PLogObjectMemory(da,sizeof(struct _DA));
  da->dim = 1;

  MPI_Comm_size(comm,&size); 
  MPI_Comm_rank(comm,&rank); 

  m = size;

  if (M < m)     SETERRQ(1,"DACreate1d:More processors than data points!");
  if ((M-1) < s) SETERRQ(1,"DACreate1d:Array is too small for stencil!");

  /* determine local owned region */
  x = M/m + ((M % m) > (rank));

  if (rank >= (M % m)) {xs = (rank * (int) (M/m) + M % m);}
    else {xs = rank * (int)(M/m) + rank;}

  /* From now on x,s,xs,xe,Xs,Xe are the exact location in the array */

  x  *= w;
  s  *= w;  /* NOTE: I change s to be absolute stencil distance */
  xs *= w;
  xe = xs + x;

  /* determine ghost region */
  if (wrap == DA_XPERIODIC) {
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
  ierr = VecCreateSeq(MPI_COMM_SELF,(Xe-Xs),&local); CHKERRQ(ierr);
    
  /* Create Local to Global Vector Scatter Context */
  /* local to global inserts non-ghost point region into global */
  VecGetOwnershipRange(global,&start,&end);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,x,start,1,&to);CHKERRQ(ierr);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,x,xs-Xs,1,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(local,from,global,to,&ltog); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */

  ierr=ISCreateStrideSeq(MPI_COMM_SELF,(Xe-Xs),0,1,&to);CHKERRQ(ierr);
 
  idx = (int *) PetscMalloc( (x+2*s)*sizeof(int) ); CHKPTRQ(idx);  
  PLogObjectMemory(da,(x+2*s)*sizeof(int));

  nn = 0;
  if (wrap == DA_XPERIODIC) {    /* Handle all cases with wrap first */

    for (i=0; i<s; i++) {  /* Left ghost points */
      if ((xs-s+i)>=0) { idx[nn++] = xs-s+i;}
      else             { idx[nn++] = M*w+(xs-s+i);}
    }

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}  /* Non-ghost points */
    
    for (i=0; i<s; i++) { /* Right ghost points */
      if ((xe+i)<M*w) { idx [nn++] =  xe+i; }
      else            { idx [nn++] = (xe+i) - M*w;}
    }
  }

  else {      /* Now do all cases with no wrapping */

    if (s <= xs) {for (i=0; i<s; i++) {idx[nn++] = xs - s + i;}}
    else         {for (i=0; i<xs;  i++) {idx[nn++] = i;}}

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}
    
    if ((xe+s)<=M*w) {for (i=0;  i<s;     i++) {idx[nn++]=xe+i;}}
    else             {for (i=xe; i<(M*w); i++) {idx[nn++]=i;   }}
  }

  ierr = ISCreateSeq(comm,nn,idx,&from); CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,gtol);
  ISDestroy(to); ISDestroy(from);

  da->M      = M; da->N = 0;  da->m  = m; da->n = 0; da->w = w; da->s = s/w;
  da->xs     = xs; da->xe = xe; da->ys = 0; da->ye = 0;
  da->Xs     = Xs; da->Xe = Xe; da->Ys = 0; da->Ye = 0;

  PLogObjectParent(da,global);
  PLogObjectParent(da,local);

  da->global = global; 
  da->local  = local;
  da->gtol   = gtol;
  da->ltog   = ltog;
  da->idx    = idx;
  da->Nl     = nn;
  da->base   = xs;
  da->view   = DAView_1d;
  da->wrap   = wrap;
  da->stencil_type = DA_STENCIL_STAR;

  /* construct the local to local scatter context */
  /* 
      We simply remap the values in the from part of 
    global to local to read from an array with the ghost values 
    rather then from the plan array.
  */
  ierr = VecScatterCopy(gtol,&da->ltol); CHKERRQ(ierr);
  left  = xs - Xs;
  idx = (int *) PetscMalloc( x*sizeof(int) ); CHKPTRQ(idx);
  count = 0;
  for ( j=0; j<x; j++ ) {
    idx[count++] = left + j;
  }  
  ierr = VecScatterRemap(da->ltol,idx,PETSC_NULL); CHKERRQ(ierr); 
  PetscFree(idx);

  *inra = da;
  return 0;
}











