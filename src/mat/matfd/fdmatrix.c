
/*
   This is where the abstract matrix operations are defined that are
  used for finite difference computations of Jacobians using coloring.
*/

#include <petsc-private/matimpl.h>        /*I "petscmat.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringSetF"
PetscErrorCode  MatFDColoringSetF(MatFDColoring fd,Vec F)
{
  PetscFunctionBegin;
  fd->F = F;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringView_Draw_Zoom"
static PetscErrorCode MatFDColoringView_Draw_Zoom(PetscDraw draw,void *Aa)
{
  MatFDColoring  fd = (MatFDColoring)Aa;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      x,y;

  PetscFunctionBegin;

  /* loop over colors  */
  for (i=0; i<fd->ncolors; i++) {
    for (j=0; j<fd->nrows[i]; j++) {
      y = fd->M - fd->rows[i][j] - fd->rstart;
      x = fd->columnsforrow[i][j];
      ierr = PetscDrawRectangle(draw,x,y,x+1,y+1,i+1,i+1,i+1,i+1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringView_Draw"
static PetscErrorCode MatFDColoringView_Draw(MatFDColoring fd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isnull;
  PetscDraw      draw;
  PetscReal      xr,yr,xl,yl,h,w;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)fd,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);

  xr  = fd->N; yr = fd->M; h = yr/10.0; w = xr/10.0; 
  xr += w;     yr += h;    xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatFDColoringView_Draw_Zoom,fd);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)fd,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringView"
/*@C
   MatFDColoringView - Views a finite difference coloring context.

   Collective on MatFDColoring

   Input  Parameters:
+  c - the coloring context
-  viewer - visualization context

   Level: intermediate

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
.     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
        output where only the first processor opens
        the file.  All other processors send their 
        data to the first processor to print. 
-     PETSC_VIEWER_DRAW_WORLD - graphical display of nonzero structure

   Notes:
     Since PETSc uses only a small number of basic colors (currently 33), if the coloring
   involves more than 33 then some seemingly identical colors are displayed making it look
   like an illegal coloring. This is just a graphical artifact.

.seealso: MatFDColoringCreate()

.keywords: Mat, finite differences, coloring, view
@*/
PetscErrorCode  MatFDColoringView(MatFDColoring c,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          i,j;
  PetscBool         isdraw,iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c,MAT_FDCOLORING_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)c)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2); 
  PetscCheckSameComm(c,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (isdraw) { 
    ierr = MatFDColoringView_Draw(c,viewer);CHKERRQ(ierr);
  } else if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)c,viewer,"MatFDColoring Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Error tolerance=%G\n",c->error_rel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Umin=%G\n",c->umin);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Number of colors=%D\n",c->ncolors);CHKERRQ(ierr);

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_INFO) {
      for (i=0; i<c->ncolors; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Information for color %D\n",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    Number of columns %D\n",c->ncolumns[i]);CHKERRQ(ierr);
        for (j=0; j<c->ncolumns[i]; j++) {
          ierr = PetscViewerASCIIPrintf(viewer,"      %D\n",c->columns[i][j]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"    Number of rows %D\n",c->nrows[i]);CHKERRQ(ierr);
        for (j=0; j<c->nrows[i]; j++) {
          ierr = PetscViewerASCIIPrintf(viewer,"      %D %D \n",c->rows[i][j],c->columnsforrow[i][j]);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for MatFDColoring",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringSetParameters"
/*@
   MatFDColoringSetParameters - Sets the parameters for the sparse approximation of
   a Jacobian matrix using finite differences.

   Logically Collective on MatFDColoring

   The Jacobian is estimated with the differencing approximation
.vb
       F'(u)_{:,i} = [F(u+h*dx_{i}) - F(u)]/h where
       h = error_rel*u[i]                 if  abs(u[i]) > umin
         = +/- error_rel*umin             otherwise, with +/- determined by the sign of u[i]
       dx_{i} = (0, ... 1, .... 0)
.ve

   Input Parameters:
+  coloring - the coloring context
.  error_rel - relative error
-  umin - minimum allowable u-value magnitude

   Level: advanced

.keywords: Mat, finite differences, coloring, set, parameters

.seealso: MatFDColoringCreate(), MatFDColoringSetFromOptions()

@*/
PetscErrorCode  MatFDColoringSetParameters(MatFDColoring matfd,PetscReal error,PetscReal umin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);
  PetscValidLogicalCollectiveReal(matfd,error,2);
  PetscValidLogicalCollectiveReal(matfd,umin,3);
  if (error != PETSC_DEFAULT) matfd->error_rel = error;
  if (umin != PETSC_DEFAULT)  matfd->umin      = umin;
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringGetFunction"
/*@C
   MatFDColoringGetFunction - Gets the function to use for computing the Jacobian.

   Not Collective

   Input Parameters:
.  coloring - the coloring context

   Output Parameters:
+  f - the function
-  fctx - the optional user-defined function context

   Level: intermediate

.keywords: Mat, Jacobian, finite differences, set, function

.seealso: MatFDColoringCreate(), MatFDColoringSetFunction(), MatFDColoringSetFromOptions()

@*/
PetscErrorCode  MatFDColoringGetFunction(MatFDColoring matfd,PetscErrorCode (**f)(void),void **fctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);
  if (f) *f = matfd->f;
  if (fctx) *fctx = matfd->fctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringSetFunction"
/*@C
   MatFDColoringSetFunction - Sets the function to use for computing the Jacobian.

   Logically Collective on MatFDColoring

   Input Parameters:
+  coloring - the coloring context
.  f - the function
-  fctx - the optional user-defined function context

   Calling sequence of (*f) function:
    For SNES:    PetscErrorCode (*f)(SNES,Vec,Vec,void*)
    If not using SNES: PetscErrorCode (*f)(void *dummy,Vec,Vec,void*) and dummy is ignored

   Level: advanced

   Notes: This function is usually used automatically by SNES (when one uses SNESSetJacobian() with the argument
     SNESDefaultComputeJacobianColor()) and only needs to be used by someone computing a matrix via coloring directly by
     calling MatFDColoringApply()

   Fortran Notes:
    In Fortran you must call MatFDColoringSetFunction() for a coloring object to
  be used without SNES or within the SNES solvers.

.keywords: Mat, Jacobian, finite differences, set, function

.seealso: MatFDColoringCreate(), MatFDColoringGetFunction(), MatFDColoringSetFromOptions()

@*/
PetscErrorCode  MatFDColoringSetFunction(MatFDColoring matfd,PetscErrorCode (*f)(void),void *fctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);
  matfd->f    = f;
  matfd->fctx = fctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringSetFromOptions"
/*@
   MatFDColoringSetFromOptions - Sets coloring finite difference parameters from 
   the options database.

   Collective on MatFDColoring

   The Jacobian, F'(u), is estimated with the differencing approximation
.vb
       F'(u)_{:,i} = [F(u+h*dx_{i}) - F(u)]/h where
       h = error_rel*u[i]                 if  abs(u[i]) > umin
         = +/- error_rel*umin             otherwise, with +/- determined by the sign of u[i]
       dx_{i} = (0, ... 1, .... 0)
.ve

   Input Parameter:
.  coloring - the coloring context

   Options Database Keys:
+  -mat_fd_coloring_err <err> - Sets <err> (square root
           of relative error in the function)
.  -mat_fd_coloring_umin <umin> - Sets umin, the minimum allowable u-value magnitude
.  -mat_fd_type - "wp" or "ds" (see MATMFFD_WP or MATMFFD_DS)
.  -mat_fd_coloring_view - Activates basic viewing
.  -mat_fd_coloring_view_info - Activates viewing info
-  -mat_fd_coloring_view_draw - Activates drawing

    Level: intermediate

.keywords: Mat, finite differences, parameters

.seealso: MatFDColoringCreate(), MatFDColoringView(), MatFDColoringSetParameters()

@*/
PetscErrorCode  MatFDColoringSetFromOptions(MatFDColoring matfd)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  char           value[3];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)matfd);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_fd_coloring_err","Square root of relative error in function","MatFDColoringSetParameters",matfd->error_rel,&matfd->error_rel,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_fd_coloring_umin","Minimum allowable u magnitude","MatFDColoringSetParameters",matfd->umin,&matfd->umin,0);CHKERRQ(ierr);
    ierr = PetscOptionsString("-mat_fd_type","Algorithm to compute h, wp or ds","MatFDColoringCreate",matfd->htype,value,3,&flg);CHKERRQ(ierr);
    if (flg) {
      if (value[0] == 'w' && value[1] == 'p') matfd->htype = "wp";
      else if (value[0] == 'd' && value[1] == 's') matfd->htype = "ds";
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown finite differencing type %s",value);
    }
    /* not used here; but so they are presented in the GUI */
    ierr = PetscOptionsName("-mat_fd_coloring_view","Print entire datastructure used for Jacobian","None",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-mat_fd_coloring_view_info","Print number of colors etc for Jacobian","None",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-mat_fd_coloring_view_draw","Plot nonzero structure ofJacobian","None",0);CHKERRQ(ierr);

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)matfd);CHKERRQ(ierr);
  PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringView_Private"
PetscErrorCode MatFDColoringView_Private(MatFDColoring fd)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStdout(((PetscObject)fd)->comm,&viewer);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_fd_coloring_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MatFDColoringView(fd,viewer);CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_fd_coloring_view_info",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatFDColoringView(fd,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_fd_coloring_view_draw",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MatFDColoringView(fd,PETSC_VIEWER_DRAW_(((PetscObject)fd)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(((PetscObject)fd)->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringCreate" 
/*@
   MatFDColoringCreate - Creates a matrix coloring context for finite difference 
   computation of Jacobians.

   Collective on Mat

   Input Parameters:
+  mat - the matrix containing the nonzero structure of the Jacobian
-  iscoloring - the coloring of the matrix; usually obtained with MatGetColoring() or DMCreateColoring()

    Output Parameter:
.   color - the new coloring context
   
    Level: intermediate

.seealso: MatFDColoringDestroy(),SNESDefaultComputeJacobianColor(), ISColoringCreate(),
          MatFDColoringSetFunction(), MatFDColoringSetFromOptions(), MatFDColoringApply(),
          MatFDColoringView(), MatFDColoringSetParameters(), MatGetColoring(), DMCreateColoring()
@*/
PetscErrorCode  MatFDColoringCreate(Mat mat,ISColoring iscoloring,MatFDColoring *color)
{
  MatFDColoring  c;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_FDColoringCreate,mat,0,0,0);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  if (M != N) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Only for square matrices");

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscHeaderCreate(c,_p_MatFDColoring,int,MAT_FDCOLORING_CLASSID,0,"MatFDColoring","Jacobian computation via finite differences with coloring","Mat",comm,MatFDColoringDestroy,MatFDColoringView);CHKERRQ(ierr);

  c->ctype = iscoloring->ctype;

  if (mat->ops->fdcoloringcreate) {
    ierr = (*mat->ops->fdcoloringcreate)(mat,iscoloring,c);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Code not yet written for matrix type %s",((PetscObject)mat)->type_name);

  ierr = MatGetVecs(mat,PETSC_NULL,&c->w1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(c,c->w1);CHKERRQ(ierr);
  ierr = VecDuplicate(c->w1,&c->w2);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(c,c->w2);CHKERRQ(ierr);

  c->error_rel         = PETSC_SQRT_MACHINE_EPSILON;
  c->umin              = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  c->currentcolor      = -1;
  c->htype             = "wp";

  *color = c;
  ierr = PetscLogEventEnd(MAT_FDColoringCreate,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringDestroy"
/*@
    MatFDColoringDestroy - Destroys a matrix coloring context that was created
    via MatFDColoringCreate().

    Collective on MatFDColoring

    Input Parameter:
.   c - coloring context

    Level: intermediate

.seealso: MatFDColoringCreate()
@*/
PetscErrorCode  MatFDColoringDestroy(MatFDColoring *c)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  if (--((PetscObject)(*c))->refct > 0) {*c = 0; PetscFunctionReturn(0);}

  for (i=0; i<(*c)->ncolors; i++) {
    ierr = PetscFree((*c)->columns[i]);CHKERRQ(ierr);
    ierr = PetscFree((*c)->rows[i]);CHKERRQ(ierr);
    ierr = PetscFree((*c)->columnsforrow[i]);CHKERRQ(ierr);
    if ((*c)->vscaleforrow) {ierr = PetscFree((*c)->vscaleforrow[i]);CHKERRQ(ierr);} 
  }
  ierr = PetscFree((*c)->ncolumns);CHKERRQ(ierr);
  ierr = PetscFree((*c)->columns);CHKERRQ(ierr);
  ierr = PetscFree((*c)->nrows);CHKERRQ(ierr);
  ierr = PetscFree((*c)->rows);CHKERRQ(ierr);
  ierr = PetscFree((*c)->columnsforrow);CHKERRQ(ierr);
  ierr = PetscFree((*c)->vscaleforrow);CHKERRQ(ierr);
  ierr = VecDestroy(&(*c)->vscale);CHKERRQ(ierr);
  ierr = VecDestroy(&(*c)->w1);CHKERRQ(ierr);
  ierr = VecDestroy(&(*c)->w2);CHKERRQ(ierr);
  ierr = VecDestroy(&(*c)->w3);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringGetPerturbedColumns"
/*@C
    MatFDColoringGetPerturbedColumns - Returns the indices of the columns that
      that are currently being perturbed.

    Not Collective

    Input Parameters:
.   coloring - coloring context created with MatFDColoringCreate()

    Output Parameters:
+   n - the number of local columns being perturbed
-   cols - the column indices, in global numbering

   Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView(), MatFDColoringApply()

.keywords: coloring, Jacobian, finite differences
@*/
PetscErrorCode  MatFDColoringGetPerturbedColumns(MatFDColoring coloring,PetscInt *n,PetscInt *cols[])
{
  PetscFunctionBegin;
  if (coloring->currentcolor >= 0) {
    *n    = coloring->ncolumns[coloring->currentcolor];
    *cols = coloring->columns[coloring->currentcolor];
  } else {
    *n = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringApply"
/*@
    MatFDColoringApply - Given a matrix for which a MatFDColoring context 
    has been created, computes the Jacobian for a function via finite differences.

    Collective on MatFDColoring

    Input Parameters:
+   mat - location to store Jacobian
.   coloring - coloring context created with MatFDColoringCreate()
.   x1 - location at which Jacobian is to be computed
-   sctx - context required by function, if this is being used with the SNES solver then it is SNES object, otherwise it is null

    Options Database Keys:
+    -mat_fd_type - "wp" or "ds"  (see MATMFFD_WP or MATMFFD_DS)
.    -mat_fd_coloring_view - Activates basic viewing or coloring
.    -mat_fd_coloring_view_draw - Activates drawing of coloring
-    -mat_fd_coloring_view_info - Activates viewing of coloring info

    Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView(), MatFDColoringSetFunction()

.keywords: coloring, Jacobian, finite differences
@*/
PetscErrorCode  MatFDColoringApply(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;    
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_CLASSID,2);
  PetscValidHeaderSpecific(x1,VEC_CLASSID,3);
  if (!coloring->f) SETERRQ(((PetscObject)J)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call MatFDColoringSetFunction()");
  if (!J->ops->fdcoloringapply) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not supported for this matrix type %s",((PetscObject)J)->type_name);
  ierr = (*J->ops->fdcoloringapply)(J,coloring,x1,flag,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringApply_AIJ"
PetscErrorCode  MatFDColoringApply_AIJ(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*) = (PetscErrorCode (*)(void*,Vec,Vec,void *))coloring->f;
  PetscErrorCode ierr;
  PetscInt       k,start,end,l,row,col,srow,**vscaleforrow,m1,m2;
  PetscScalar    dx,*y,*xx,*w3_array;
  PetscScalar    *vscale_array;
  PetscReal      epsilon = coloring->error_rel,umin = coloring->umin,unorm; 
  Vec            w1=coloring->w1,w2=coloring->w2,w3;
  void           *fctx = coloring->fctx;
  PetscBool      flg = PETSC_FALSE;
  PetscInt       ctype=coloring->ctype,N,col_start=0,col_end=0;
  Vec            x1_tmp;

  PetscFunctionBegin;    
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_CLASSID,2);
  PetscValidHeaderSpecific(x1,VEC_CLASSID,3);
  if (!f) SETERRQ(((PetscObject)J)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call MatFDColoringSetFunction()");

  ierr = PetscLogEventBegin(MAT_FDColoringApply,coloring,J,x1,0);CHKERRQ(ierr);
  ierr = MatSetUnfactored(J);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_fd_coloring_dont_rezero",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscInfo(coloring,"Not calling MatZeroEntries()\n");CHKERRQ(ierr);
  } else {
    PetscBool  assembled;
    ierr = MatAssembled(J,&assembled);CHKERRQ(ierr);
    if (assembled) {
      ierr = MatZeroEntries(J);CHKERRQ(ierr);
    }
  }

  x1_tmp = x1; 
  if (!coloring->vscale){ 
    ierr = VecDuplicate(x1_tmp,&coloring->vscale);CHKERRQ(ierr);
  }
    
  /*
    This is a horrible, horrible, hack. 
  */
  if (coloring->F) {
    ierr = VecGetLocalSize(coloring->F,&m1);CHKERRQ(ierr);
    ierr = VecGetLocalSize(w1,&m2);CHKERRQ(ierr);
    if (m1 != m2) {  
      coloring->F = 0; 
      }    
    }   

  if (coloring->htype[0] == 'w') { /* tacky test; need to make systematic if we add other approaches to computing h*/
    ierr = VecNorm(x1_tmp,NORM_2,&unorm);CHKERRQ(ierr);
  }
  ierr = VecGetOwnershipRange(w1,&start,&end);CHKERRQ(ierr); /* OwnershipRange is used by ghosted x! */
      
  /* Set w1 = F(x1) */
  if (coloring->F) {
    w1          = coloring->F; /* use already computed value of function */
    coloring->F = 0; 
  } else {
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1_tmp,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
  }
      
  if (!coloring->w3) {
    ierr = VecDuplicate(x1_tmp,&coloring->w3);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(coloring,coloring->w3);CHKERRQ(ierr);
  }
  w3 = coloring->w3;

    /* Compute all the local scale factors, including ghost points */
  ierr = VecGetLocalSize(x1_tmp,&N);CHKERRQ(ierr);
  ierr = VecGetArray(x1_tmp,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  if (ctype == IS_COLORING_GHOSTED){
    col_start = 0; col_end = N;
  } else if (ctype == IS_COLORING_GLOBAL){
    xx = xx - start;
    vscale_array = vscale_array - start;
    col_start = start; col_end = N + start;
  }
  for (col=col_start; col<col_end; col++){ 
    /* Loop over each local column, vscale[col] = 1./(epsilon*dx[col]) */      
    if (coloring->htype[0] == 'w') {
      dx = 1.0 + unorm;
    } else {
      dx  = xx[col];
    }
    if (dx == (PetscScalar)0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
    if (dx < umin && dx >= 0.0)      dx = umin;
    else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
    if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
    else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
    dx               *= epsilon;
    vscale_array[col] = (PetscScalar)1.0/dx;
  } 
  if (ctype == IS_COLORING_GLOBAL)  vscale_array = vscale_array + start;      
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  if (ctype == IS_COLORING_GLOBAL){
    ierr = VecGhostUpdateBegin(coloring->vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(coloring->vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
    
  if (coloring->vscaleforrow) {
    vscaleforrow = coloring->vscaleforrow;
  } else SETERRQ(((PetscObject)J)->comm,PETSC_ERR_ARG_NULL,"Null Object: coloring->vscaleforrow");

  /*
    Loop over each color
  */
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  for (k=0; k<coloring->ncolors; k++) { 
    coloring->currentcolor = k;
    ierr = VecCopy(x1_tmp,w3);CHKERRQ(ierr);
    ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);
    if (ctype == IS_COLORING_GLOBAL) w3_array = w3_array - start;
    /*
      Loop over each column associated with color 
      adding the perturbation to the vector w3.
    */
    for (l=0; l<coloring->ncolumns[k]; l++) {
      col = coloring->columns[k][l];    /* local column of the matrix we are probing for */
      if (coloring->htype[0] == 'w') {
        dx = 1.0 + unorm;
      } else {
        dx  = xx[col];
      }
      if (dx == (PetscScalar)0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
      if (dx < umin && dx >= 0.0)      dx = umin;
      else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
      if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
      dx            *= epsilon;
      if (!PetscAbsScalar(dx)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Computed 0 differencing parameter");
      w3_array[col] += dx;
    } 
    if (ctype == IS_COLORING_GLOBAL) w3_array = w3_array + start;
    ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);

    /*
      Evaluate function at w3 = x1 + dx (here dx is a vector of perturbations)
                           w2 = F(x1 + dx) - F(x1)
    */
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);        
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr); 
        
    /*
      Loop over rows of vector, putting results into Jacobian matrix
    */
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    for (l=0; l<coloring->nrows[k]; l++) {
      row    = coloring->rows[k][l];             /* local row index */
      col    = coloring->columnsforrow[k][l];    /* global column index */
      y[row] *= vscale_array[vscaleforrow[k][l]];
      srow   = row + start;
      ierr   = MatSetValues(J,1,&srow,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
  } /* endof for each color */
  if (ctype == IS_COLORING_GLOBAL) xx = xx + start; 
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x1_tmp,&xx);CHKERRQ(ierr);
   
  coloring->currentcolor = -1;
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_FDColoringApply,coloring,J,x1,0);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_null_space_test",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MatNullSpaceTest(J->nullsp,J,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = MatFDColoringView_Private(coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
