#define PETSCDM_DLL
/* 
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95    
*/

#include "src/dm/da/daimpl.h"     /*I  "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAView_1d"
PetscErrorCode DAView_1d(DA da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscTruth     iascii,isdraw;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(da->comm,&rank);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D m %D w %D s %D\n",rank,da->M,
                 da->m,da->w,da->s);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D\n",da->xs,da->xe);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    PetscDraw       draw;
    double     ymin = -1,ymax = 1,xmin = -1,xmax = da->M,x;
    PetscInt        base;
    char       node[10];
    PetscTruth isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

    ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draws all node lines */
    if (!rank) {
      PetscInt xmin_tmp;
      ymin = 0.0; ymax = 0.3;
      
      /* ADIC doesn't like doubles in a for loop */
      for (xmin_tmp =0; xmin_tmp < da->M; xmin_tmp++) {
         ierr = PetscDrawLine(draw,(double)xmin_tmp,ymin,(double)xmin_tmp,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }

      xmin = 0.0; xmax = da->M - 1;
      ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }

    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = da->xs / da->w; xmax = (da->xe / da->w)  - 1;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);

    /* Put in index numbers */
    base = da->base / da->w;
    for (x=xmin; x<=xmax; x++) {
      sprintf(node,"%d",(int)base++);
      ierr = PetscDrawString(draw,x,ymin,PETSC_DRAW_RED,node);CHKERRQ(ierr);
    }

    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for DA 1d",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode DAPublish_Petsc(PetscObject);

#undef __FUNCT__  
#define __FUNCT__ "DACreate1d"
/*@C
   DACreate1d - Creates an object that will manage the communication of  one-dimensional 
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  wrap - type of periodicity should the array have, if any. Use 
          either DA_NONPERIODIC or DA_XPERIODIC
.  M - global dimension of the array (use -M to indicate that it may be set to a different value 
            from the command line with -da_grid_x <M>)
.  dof - number of degrees of freedom per node
.  lc - array containing number of nodes in the X direction on each processor, 
        or PETSC_NULL. If non-null, must be of length as m.
-  s - stencil width  

   Output Parameter:
.  inra - the resulting distributed array object

   Options Database Key:
+  -da_view - Calls DAView() at the conclusion of DACreate1d()
.  -da_grid_x <nx> - number of grid points in x direction; can set if M < 0
-  -da_refine_x - refinement factor 

   Level: beginner

   Notes:
   The array data itself is NOT stored in the DA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DACreateGlobalVector()
   and DACreateLocalVector() and calls to VecDuplicate() if more are needed.


.keywords: distributed array, create, one-dimensional

.seealso: DADestroy(), DAView(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(), DASetRefinementFactor(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DALocalToLocalBegin(), DALocalToLocalEnd(), DAGetRefinementFactor(),
          DAGetInfo(), DACreateGlobalVector(), DACreateLocalVector(), DACreateNaturalVector(), DALoad(), DAView()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreate1d(MPI_Comm comm,DAPeriodicType wrap,PetscInt M,PetscInt dof,PetscInt s,PetscInt *lc,DA *inra)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,*idx,nn,left,refine_x = 2,tM = M,xs,xe,x,Xs,Xe,start,end,m;
  PetscTruth     flg1,flg2;
  DA             da;
  Vec            local,global;
  VecScatter     ltog,gtol;
  IS             to,from;

  PetscFunctionBegin;
  PetscValidPointer(inra,7);
  *inra = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  if (dof < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dof);
  if (s < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",s);

  ierr = PetscOptionsBegin(comm,PETSC_NULL,"1d DA Options","DA");CHKERRQ(ierr);
    if (M < 0) {
      tM   = -M; 
      ierr = PetscOptionsInt("-da_grid_x","Number of grid points in x direction","DACreate1d",tM,&tM,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-da_refine_x","Refinement ratio in x direction","DASetRefinementFactor",refine_x,&refine_x,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  M = tM;

  ierr = PetscHeaderCreate(da,_p_DA,struct _DAOps,DA_COOKIE,0,"DA",comm,DADestroy,DAView);CHKERRQ(ierr);
  da->bops->publish           = DAPublish_Petsc;
  da->ops->createglobalvector = DACreateGlobalVector;
  da->ops->getinterpolation   = DAGetInterpolation;
  da->ops->getcoloring        = DAGetColoring;
  da->ops->getmatrix          = DAGetMatrix;
  da->ops->refine             = DARefine;
  ierr = PetscLogObjectMemory(da,sizeof(struct _p_DA));CHKERRQ(ierr);
  da->dim        = 1;
  da->interptype = DA_Q1;
  da->refine_x   = refine_x;
  ierr = PetscMalloc(dof*sizeof(char*),&da->fieldname);CHKERRQ(ierr);
  ierr = PetscMemzero(da->fieldname,dof*sizeof(char*));CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  m = size;

  if (M < m)     SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"More processors than data points! %D %D",m,M);
  if ((M-1) < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Array is too small for stencil! %D %D",M-1,s);

  /* 
     Determine locally owned region 
     xs is the first local node number, x is the number of local nodes 
  */
  if (!lc) {
    ierr = PetscOptionsHasName(PETSC_NULL,"-da_partition_blockcomm",&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-da_partition_nodes_at_end",&flg2);CHKERRQ(ierr);
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
      if (rank >= (M % m)) {xs = (rank * (PetscInt)(M/m) + M % m);}
      else                 {xs = rank * (PetscInt)(M/m) + rank;}
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
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Sum of lc across processors not equal to M %D %D",left,M);
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
  da->Nlocal = x;
  ierr = VecCreateMPIWithArray(comm,da->Nlocal,PETSC_DECIDE,0,&global);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,dof);CHKERRQ(ierr);
  da->nlocal = (Xe-Xs);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,da->nlocal,0,&local);CHKERRQ(ierr);
  ierr = VecSetBlockSize(local,dof);CHKERRQ(ierr);
    
  /* Create Local to Global Vector Scatter Context */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,start,1,&to);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,xs-Xs,1,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,to);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,ltog);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */
  ierr = ISCreateStride(comm,(Xe-Xs),0,1,&to);CHKERRQ(ierr);
 
  ierr = PetscMalloc((x+2*s)*sizeof(PetscInt),&idx);CHKERRQ(ierr);  
  ierr = PetscLogObjectMemory(da,(x+2*s)*sizeof(PetscInt));CHKERRQ(ierr);

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
  ierr = PetscLogObjectParent(da,to);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,gtol);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);

  da->M  = M;  da->N  = 1;  da->m  = m; da->n = 1;
  da->xs = xs; da->xe = xe; da->ys = 0; da->ye = 1; da->zs = 0; da->ze = 1;
  da->Xs = Xs; da->Xe = Xe; da->Ys = 0; da->Ye = 1; da->Zs = 0; da->Ze = 1;
  da->P  = 1;  da->p  = 1;  da->w = dof; da->s = s/dof;

  da->gtol         = gtol;
  da->ltog         = ltog;
  da->idx          = idx;
  da->Nl           = nn;
  da->base         = xs;
  da->ops->view    = DAView_1d;
  da->wrap         = wrap;
  da->stencil_type = DA_STENCIL_STAR;

  /* 
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  ierr = ISLocalToGlobalMappingCreateNC(comm,nn,idx,&da->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(da->ltogmap,da->w,&da->ltogmapb);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,da->ltogmap);CHKERRQ(ierr);

  da->ltol = PETSC_NULL;
  da->ao   = PETSC_NULL;

  ierr = PetscOptionsHasName(PETSC_NULL,"-da_view",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,PETSC_VIEWER_STDOUT_(da->comm));CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL,"-da_view_draw",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,PETSC_VIEWER_DRAW_(da->comm));CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAPrintHelp(da);CHKERRQ(ierr);}
  *inra = da;
  ierr = PetscPublishAll(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


