/*$Id: da1.c,v 1.112 2000/04/12 04:26:20 bsmith Exp bsmith $*/

/* 
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95    
*/

#include "src/dm/da/daimpl.h"     /*I  "da.h"   I*/

#if defined (PETSC_HAVE_AMS)
EXTERN_C_BEGIN
extern int AMSSetFieldBlock_DA(AMS_Memory,char *,Vec);
EXTERN_C_END
#endif

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAView_1d"
int DAView_1d(DA da,Viewer viewer)
{
  int        rank,ierr;
  PetscTruth isascii,isdraw,isbinary;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(da->comm,&rank);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %d m %d w %d s %d\n",rank,da->M,
                 da->m,da->w,da->s);CHKERRQ(ierr);
    ierr = ViewerASCIISynchronizedPrintf(viewer,"X range: %d %d\n",da->xs,da->xe);CHKERRQ(ierr);
    ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    Draw       draw;
    double     ymin = -1,ymax = 1,xmin = -1,xmax = da->M,x;
    int        base;
    char       node[10];
    PetscTruth isnull;

    ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

    ierr = DrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draws all node lines */
    if (!rank) {
      int xmin_tmp;
      ymin = 0.0; ymax = 0.3;
      
      /* ADIC doesn't like doubles in a for loop */
      for (xmin_tmp =0; xmin_tmp < (int)da->M; xmin_tmp++) {
         ierr = DrawLine(draw,(double)xmin_tmp,ymin,(double)xmin_tmp,ymax,DRAW_BLACK);CHKERRQ(ierr);
      }

      xmin = 0.0; xmax = da->M - 1;
      ierr = DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_BLACK);CHKERRQ(ierr);
      ierr = DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_BLACK);CHKERRQ(ierr);
    }

    ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = DrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = da->xs / da->w; xmax = (da->xe / da->w)  - 1;
    ierr = DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawLine(draw,xmax,ymin,xmax,ymax,DRAW_RED);CHKERRQ(ierr);

    /* Put in index numbers */
    base = da->base / da->w;
    for (x=xmin; x<=xmax; x++) {
      sprintf(node,"%d",base++);
      ierr = DrawString(draw,x,ymin,DRAW_RED,node);CHKERRQ(ierr);
    }

    ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = DrawPause(draw);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = DAView_Binary(da,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for DA 1d",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

extern int DAPublish_Petsc(PetscObject);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DACreate1d"
/*@C
   DACreate1d - Creates an object that will manage the communication of  one-dimensional 
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  wrap - type of periodicity should the array have, if any. Use 
          either DA_NONPERIODIC or DA_XPERIODIC
.  M - global dimension of the array
.  dof - number of degrees of freedom per node
.  lc - array containing number of nodes in the X direction on each processor, 
        or PETSC_NULL. If non-null, must be of length as m.
-  s - stencil width  

   Output Parameter:
.  inra - the resulting distributed array object

   Options Database Key:
.  -da_view - Calls DAView() at the conclusion of DACreate1d()

   Level: beginner

   Notes:
   The array data itself is NOT stored in the DA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DACreateGlobalVector()
   and DACreateLocalVector() and calls to VecDuplicate() if more are needed.


.keywords: distributed array, create, one-dimensional

.seealso: DADestroy(), DAView(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DALocalToLocalBegin(), DALocalToLocalEnd(),
          DAGetInfo(), DACreateGlobalVector(), DACreateLocalVector(), DACreateNaturalVector(), DALoad(), DAView()

@*/
int DACreate1d(MPI_Comm comm,DAPeriodicType wrap,int M,int dof,int s,int *lc,DA *inra)
{
  int        rank,size,xs,xe,x,Xs,Xe,ierr,start,end,m;
  int        i,*idx,nn,j,left,gdim;
  PetscTruth flg1,flg2;
  DA         da;
  Vec        local,global;
  VecScatter ltog,gtol;
  IS         to,from;

  PetscFunctionBegin;
  *inra = 0;

  if (dof < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Must have 1 or more degrees of freedom per node: %d",dof);
  if (s < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Stencil width cannot be negative: %d",s);

  PetscHeaderCreate(da,_p_DA,int,DA_COOKIE,0,"DA",comm,DADestroy,DAView);
  PLogObjectCreate(da);
  da->bops->publish = DAPublish_Petsc;
  PLogObjectMemory(da,sizeof(struct _p_DA));
  da->dim        = 1;
  da->gtog1      = 0;
  da->localused  = PETSC_FALSE;
  da->globalused = PETSC_FALSE;
  da->fieldname  = (char**)PetscMalloc(dof*sizeof(char*));CHKPTRQ(da->fieldname);
  ierr = PetscMemzero(da->fieldname,dof*sizeof(char*));CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  m = size;

  if (M < m)     SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"More processors than data points! %d %d",m,M);
  if ((M-1) < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Array is too small for stencil! %d %d",M-1,s);

  /* 
     Determine locally owned region 
     xs is the first local node number, x is the number of local nodes 
  */
  if (!lc) {
    ierr = OptionsHasName(PETSC_NULL,"-da_partition_blockcomm",&flg1);CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-da_partition_nodes_at_end",&flg2);CHKERRQ(ierr);
    if (flg1) {      /* Block Comm type Distribution */
      xs = rank*M/m;
      x  = (rank + 1)*M/m - xs;
    } else if (flg2) { /* The odd nodes are evenly distributed across last nodes */
      x = (M + rank)/m;
      if (M/m == x) { xs = rank*x; }
      else          { xs = rank*(x-1) + (M+rank)%(x*m); }
    } else { /* The odd nodes are evenly distributed across the first k nodes */
      /* Regular PETSc Distribution */
      x = M/m + ((M % m) > rank);
      if (rank >= (M % m)) {xs = (rank * (int)(M/m) + M % m);}
      else                 {xs = rank * (int)(M/m) + rank;}
    }
  } else {
    x  = lc[rank];
    xs = 0;
    for (i=0; i<rank; i++) {
      xs += lc[i];
    }
    /* verify that data user provided is consistent */
    left = xs;
    for (i=rank; i<size; i++) {
      left += lc[i];
    }
    if (left != M) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Sum of lc across processors not equal to M %d %d",left,M);
    }
  }

  /* From now on x,s,xs,xe,Xs,Xe are the exact location in the array */
  x  *= dof;
  s  *= dof;  /* NOTE: here change s to be absolute stencil distance */
  xs *= dof;
  xe = xs + x;

  /* determine ghost region */
  if (wrap == DA_XPERIODIC) {
    Xs = xs - s; 
    Xe = xe + s;
  } else {
    if ((xs-s) >= 0)   Xs = xs-s;  else Xs = 0; 
    if ((xe+s) <= M*dof) Xe = xe+s;  else Xe = M*dof;    
  }

  /* allocate the base parallel and sequential vectors */
  ierr = VecCreateMPI(comm,x,PETSC_DECIDE,&global);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,dof);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,(Xe-Xs),&local);CHKERRQ(ierr);
  ierr = VecSetBlockSize(local,dof);CHKERRQ(ierr);
    
  /* Create Local to Global Vector Scatter Context */
  /* local to global inserts non-ghost point region into global */
  VecGetOwnershipRange(global,&start,&end);
  ierr = ISCreateStride(comm,x,start,1,&to);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,xs-Xs,1,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */
  ierr = ISCreateStride(comm,(Xe-Xs),0,1,&to);CHKERRQ(ierr);
 
  idx  = (int*)PetscMalloc((x+2*s)*sizeof(int));CHKPTRQ(idx);  
  PLogObjectMemory(da,(x+2*s)*sizeof(int));

  nn = 0;
  if (wrap == DA_XPERIODIC) {    /* Handle all cases with wrap first */

    for (i=0; i<s; i++) {  /* Left ghost points */
      if ((xs-s+i)>=0) { idx[nn++] = xs-s+i;}
      else             { idx[nn++] = M*dof+(xs-s+i);}
    }

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}  /* Non-ghost points */
    
    for (i=0; i<s; i++) { /* Right ghost points */
      if ((xe+i)<M*dof) { idx [nn++] =  xe+i; }
      else            { idx [nn++] = (xe+i) - M*dof;}
    }
  } else {      /* Now do all cases with no wrapping */

    if (s <= xs) {for (i=0; i<s; i++) {idx[nn++] = xs - s + i;}}
    else         {for (i=0; i<xs;  i++) {idx[nn++] = i;}}

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}
    
    if ((xe+s)<=M*dof) {for (i=0;  i<s;     i++) {idx[nn++]=xe+i;}}
    else             {for (i=xe; i<(M*dof); i++) {idx[nn++]=i;   }}
  }

  ierr = ISCreateGeneral(comm,nn,idx,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,gtol);
  ISDestroy(to); ISDestroy(from);

  da->M  = M;  da->N  = 1;  da->m  = m; da->n = 1;
  da->xs = xs; da->xe = xe; da->ys = 0; da->ye = 1; da->zs = 0; da->ze = 1;
  da->Xs = Xs; da->Xe = Xe; da->Ys = 0; da->Ye = 1; da->Zs = 0; da->Ze = 1;
  da->P  = 1;  da->p  = 1;  da->w = dof; da->s = s/dof;

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

  /* 
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  {
    ISLocalToGlobalMapping isltog;
    ierr        = ISLocalToGlobalMappingCreate(comm,nn,idx,&isltog);CHKERRQ(ierr);
    ierr        = VecSetLocalToGlobalMapping(da->global,isltog);CHKERRQ(ierr);
    da->ltogmap = isltog; 
    ierr        = PetscObjectReference((PetscObject)isltog);CHKERRQ(ierr);
    PLogObjectParent(da,isltog);
    ierr = ISLocalToGlobalMappingDestroy(isltog);CHKERRQ(ierr);
  }

  /* construct the local to local scatter context */
  /* 
      We simply remap the values in the from part of 
    global to local to read from an array with the ghost values 
    rather then from the plain array.
  */
  ierr = VecScatterCopy(gtol,&da->ltol);CHKERRQ(ierr);
  PLogObjectParent(da,da->ltol);
  left  = xs - Xs;
  idx   = (int*)PetscMalloc((Xe-Xs)*sizeof(int));CHKPTRQ(idx);
  for (j=0; j<Xe-Xs; j++) {
    idx[j] = left + j;
  }  
  ierr = VecScatterRemap(da->ltol,idx,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscFree(idx);CHKERRQ(ierr);

  /* 
     Build the natural ordering to PETSc ordering mappings.
  */
  {
    IS is;
    
    ierr = ISCreateStride(comm,da->xe-da->xs,da->base,1,&is);CHKERRQ(ierr);
    ierr = AOCreateBasicIS(is,is,&da->ao);CHKERRQ(ierr);
    PLogObjectParent(da,da->ao);
    ierr = ISDestroy(is);CHKERRQ(ierr);
  }

  /*
     Note the following will be removed soon. Since the functionality 
    is replaced by the above.
  */
  /* Construct the mapping from current global ordering to global
     ordering that would be used if only 1 processor were employed.
     This mapping is intended only for internal use by discrete
     function and matrix viewers.

     We don't really need this for 1D distributed arrays, since the
     ordering is the same regardless.  But for now we form it anyway
     Maybe we'll change in the near future.
   */
  ierr = VecGetSize(global,&gdim);CHKERRQ(ierr);
  da->gtog1 = (int *)PetscMalloc(gdim*sizeof(int));CHKPTRQ(da->gtog1);
  PLogObjectMemory(da,gdim*sizeof(int));
  for (i=0; i<gdim; i++) da->gtog1[i] = i;

  ierr = OptionsHasName(PETSC_NULL,"-da_view",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-da_view_draw",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,VIEWER_DRAW_(da->comm));CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAPrintHelp(da);CHKERRQ(ierr);}
  *inra = da;
  PetscPublishAll(da);  
#if defined(PETSC_HAVE_AMS)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)global,"AMSSetFieldBlock_C",
         "AMSSetFieldBlock_DA",(void*)AMSSetFieldBlock_DA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)local,"AMSSetFieldBlock_C",
         "AMSSetFieldBlock_DA",(void*)AMSSetFieldBlock_DA);CHKERRQ(ierr);
  if (((PetscObject)global)->amem > -1) {
    ierr = AMSSetFieldBlock_DA(((PetscObject)global)->amem,"values",global);CHKERRQ(ierr);
  }
#endif
  ierr = VecSetOperation(global,VECOP_VIEW,(void*)VecView_MPI_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(global,VECOP_LOADINTOVECTOR,(void*)VecLoadIntoVector_Binary_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


