#ifndef lint
static char vcid[] = "$Id: da1.c,v 1.35 1996/04/17 23:01:40 curfman Exp curfman $";
#endif

/* 
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95    
*/

#include "daimpl.h"     /*I  "da.h"   I*/
#include "pinclude/pviewer.h"   
#include <math.h>
#include "draw.h"      /*I  "draw.h"  I*/

static int DAView_1d(PetscObject pobj,Viewer viewer)
{
  DA          da  = (DA) pobj;
  int         rank, ierr;
  ViewerType  vtype;

  PetscValidHeaderSpecific(da,DA_COOKIE);

  MPI_Comm_rank(da->comm,&rank); 

  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);


  if (vtype == ASCII_FILE_VIEWER) {
    FILE *fd;
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(da->comm,1);
    fprintf(fd,"Processor [%d] M %d m %d w %d s %d\n",rank,da->M,
                 da->m,da->w,da->s);
    fprintf(fd,"X range: %d %d\n",da->xs,da->xe);
    fflush(fd);
    PetscSequentialPhaseEnd(da->comm,1);
  }
  else if (vtype == DRAW_VIEWER) {
    Draw       draw;
    double     ymin = -1,ymax = 1,xmin = -1,xmax = da->M,x;
    int        base;
    char       node[10];
    PetscTruth isnull;

    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;

    DrawSetCoordinates(draw,xmin,ymin,xmax,ymax);

    /* first processor draws all node lines */

    if (!rank) {
      ymin = 0.0; ymax = 0.3;

      for ( xmin=0; xmin<da->M; xmin++ ) {
         DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_BLACK);
      }

      xmin = 0.0; xmax = da->M - 1;
      DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_BLACK);
      DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_BLACK);
    }

    DrawSyncFlush(draw); 
    DrawPause(draw);
    MPI_Barrier(da->comm);

    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = da->xs / da->w; xmax = (da->xe / da->w)  - 1;
    DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_RED);
    DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_RED);
    DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_RED);
    DrawLine(draw,xmax,ymin,xmax,ymax,DRAW_RED);

    /* Put in index numbers */
    base = da->base / da->w;
    for ( x=xmin; x<=xmax; x++ ) {
      sprintf(node,"%d",base++);
      DrawText(draw,x,ymin,DRAW_RED,node);
    }

    DrawSyncFlush(draw);
    DrawPause(draw); 
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
.  inra - the resulting distributed array object

   Options Database Key:
$  -da_view : call DAView() at the conclusion of DACreate1d()

.keywords: distributed array, create, one-dimensional

.seealso: DADestroy(), DAView(), DACreate2d(), DACreate3d()
@*/
int DACreate1d(MPI_Comm comm,DAPeriodicType wrap,int M,int w,int s,DA *inra)
{
  int        rank, size,xs,xe,x,Xs,Xe,ierr,start,end,m;
  int        i,*idx,nn,j,count,left,flg,gdim;
  DA         da;
  Vec        local,global;
  VecScatter ltog,gtol;
  IS         to,from;
  DF         df_local;
  *inra = 0;

  PetscHeaderCreate(da,_DA,DA_COOKIE,0,comm);
  PLogObjectCreate(da);
  PLogObjectMemory(da,sizeof(struct _DA));
  da->dim = 1;
  da->gtog1 = 0;

  MPI_Comm_size(comm,&size); 
  MPI_Comm_rank(comm,&rank); 

  m = size;

  if (M < m)     SETERRQ(1,"DACreate1d:More processors than data points!");
  if ((M-1) < s) SETERRQ(1,"DACreate1d:Array is too small for stencil!");

  /* determine locally owned region */
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

  da->M  = M;  da->N  = 1;  da->m  = m; da->n = 1;
  da->xs = xs; da->xe = xe; da->ys = 0; da->ye = 0; da->zs = 0; da->ze = 0;
  da->Xs = Xs; da->Xe = Xe; da->Ys = 0; da->Ye = 0; da->Zs = 0; da->Ze = 0;
  da->P  = 1;  da->p  = 1;  da->w = w; da->s = s/w;

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

  /* Construct the mapping from current global ordering to global
     ordering that would be used if only 1 processor were employed.
     This mapping is intended only for internal use by discrete
     function and matrix viewers.

     We don't really need this for 1D distributed arrays, since the
     ordering is the same regardless.  But for now we form it anyway
     so that the DFVec routines can all be used seamlessly.  Maybe
     we'll change in the near future.
   */
  ierr = VecGetSize(global,&gdim); CHKERRQ(ierr);
  da->gtog1 = (int *)PetscMalloc(gdim*sizeof(int)); CHKPTRQ(da->gtog1);
  for (i=0; i<gdim; i++) da->gtog1[i] = i;

  /* Create discrete function shell and associate with vectors in DA */
  /* Eventually will pass in optional labels for each component */
  ierr = DFShellCreateDA_Private(comm,PETSC_NULL,da,&da->dfshell); CHKERRQ(ierr);
  ierr = DFShellGetLocalDFShell(da->dfshell,&df_local);
  ierr = DFVecShellAssociate(da->dfshell,global); CHKERRQ(ierr);
  ierr = DFVecShellAssociate(df_local,local); CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-da_view",&flg); CHKERRQ(ierr);
  if (flg) {ierr = DAView(da,STDOUT_VIEWER_SELF); CHKERRQ(ierr);}

  *inra = da;
  return 0;
}











