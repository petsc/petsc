
/*
   This is where the abstract matrix operations are defined that are
  used for finite difference computations of Jacobians using coloring.
*/

#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/
#include <petsc/private/isimpl.h>

PetscErrorCode  MatFDColoringSetF(MatFDColoring fd,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (F) {
    ierr     = VecCopy(F,fd->w1);CHKERRQ(ierr);
    fd->fset = PETSC_TRUE;
  } else {
    fd->fset = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode MatFDColoringView_Draw_Zoom(PetscDraw draw,void *Aa)
{
  MatFDColoring  fd = (MatFDColoring)Aa;
  PetscErrorCode ierr;
  PetscInt       i,j,nz,row;
  PetscReal      x,y;
  MatEntry       *Jentry=fd->matentry;

  PetscFunctionBegin;
  /* loop over colors  */
  nz = 0;
  for (i=0; i<fd->ncolors; i++) {
    for (j=0; j<fd->nrows[i]; j++) {
      row = Jentry[nz].row;
      y   = fd->M - row - fd->rstart;
      x   = (PetscReal)Jentry[nz++].col;
      ierr = PetscDrawRectangle(draw,x,y,x+1,y+1,i+1,i+1,i+1,i+1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFDColoringView_Draw(MatFDColoring fd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isnull;
  PetscDraw      draw;
  PetscReal      xr,yr,xl,yl,h,w;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = fd->N; yr  = fd->M; h = yr/10.0; w = xr/10.0;
  xr  += w;     yr += h;    xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)fd,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatFDColoringView_Draw_Zoom,fd);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)fd,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)c),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(c,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (isdraw) {
    ierr = MatFDColoringView_Draw(c,viewer);CHKERRQ(ierr);
  } else if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)c,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Error tolerance=%g\n",(double)c->error_rel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Umin=%g\n",(double)c->umin);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Number of colors=%D\n",c->ncolors);CHKERRQ(ierr);

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_INFO) {
      PetscInt row,col,nz;
      nz = 0;
      for (i=0; i<c->ncolors; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Information for color %D\n",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    Number of columns %D\n",c->ncolumns[i]);CHKERRQ(ierr);
        for (j=0; j<c->ncolumns[i]; j++) {
          ierr = PetscViewerASCIIPrintf(viewer,"      %D\n",c->columns[i][j]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"    Number of rows %D\n",c->nrows[i]);CHKERRQ(ierr);
        if (c->matentry) {
          for (j=0; j<c->nrows[i]; j++) {
            row  = c->matentry[nz].row;
            col  = c->matentry[nz++].col;
            ierr = PetscViewerASCIIPrintf(viewer,"      %D %D \n",row,col);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   MatFDColoringSetParameters - Sets the parameters for the sparse approximation of
   a Jacobian matrix using finite differences.

   Logically Collective on MatFDColoring

   The Jacobian is estimated with the differencing approximation
.vb
       F'(u)_{:,i} = [F(u+h*dx_{i}) - F(u)]/h where
       htype = 'ds':
         h = error_rel*u[i]                 if  abs(u[i]) > umin
           = +/- error_rel*umin             otherwise, with +/- determined by the sign of u[i]
         dx_{i} = (0, ... 1, .... 0)

       htype = 'wp':
         h = error_rel * sqrt(1 + ||u||)
.ve

   Input Parameters:
+  coloring - the coloring context
.  error_rel - relative error
-  umin - minimum allowable u-value magnitude

   Level: advanced

.seealso: MatFDColoringCreate(), MatFDColoringSetFromOptions()

@*/
PetscErrorCode MatFDColoringSetParameters(MatFDColoring matfd,PetscReal error,PetscReal umin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);
  PetscValidLogicalCollectiveReal(matfd,error,2);
  PetscValidLogicalCollectiveReal(matfd,umin,3);
  if (error != PETSC_DEFAULT) matfd->error_rel = error;
  if (umin != PETSC_DEFAULT)  matfd->umin      = umin;
  PetscFunctionReturn(0);
}

/*@
   MatFDColoringSetBlockSize - Sets block size for efficient inserting entries of Jacobian matrix.

   Logically Collective on MatFDColoring

   Input Parameters:
+  coloring - the coloring context
.  brows - number of rows in the block
-  bcols - number of columns in the block

   Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringSetFromOptions()

@*/
PetscErrorCode MatFDColoringSetBlockSize(MatFDColoring matfd,PetscInt brows,PetscInt bcols)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);
  PetscValidLogicalCollectiveInt(matfd,brows,2);
  PetscValidLogicalCollectiveInt(matfd,bcols,3);
  if (brows != PETSC_DEFAULT) matfd->brows = brows;
  if (bcols != PETSC_DEFAULT) matfd->bcols = bcols;
  PetscFunctionReturn(0);
}

/*@
   MatFDColoringSetUp - Sets up the internal data structures of matrix coloring context for the later use.

   Collective on Mat

   Input Parameters:
+  mat - the matrix containing the nonzero structure of the Jacobian
.  iscoloring - the coloring of the matrix; usually obtained with MatGetColoring() or DMCreateColoring()
-  color - the matrix coloring context

   Level: beginner

   Notes: When the coloring type is IS_COLORING_LOCAL the coloring is in the local ordering of the unknowns.

.seealso: MatFDColoringCreate(), MatFDColoringDestroy()
@*/
PetscErrorCode MatFDColoringSetUp(Mat mat,ISColoring iscoloring,MatFDColoring color)
{
  PetscErrorCode ierr;
  PetscBool      eq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(color,MAT_FDCOLORING_CLASSID,3);
  if (color->setupcalled) PetscFunctionReturn(0);
  ierr = PetscObjectCompareId((PetscObject)mat,color->matid,&eq);CHKERRQ(ierr);
  if (!eq) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"Matrix used with MatFDColoringSetUp() must be that used with MatFDColoringCreate()");

  ierr = PetscLogEventBegin(MAT_FDColoringSetUp,mat,0,0,0);CHKERRQ(ierr);
  if (mat->ops->fdcoloringsetup) {
    ierr = (*mat->ops->fdcoloringsetup)(mat,iscoloring,color);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Code not yet written for matrix type %s",((PetscObject)mat)->type_name);

  color->setupcalled = PETSC_TRUE;
   ierr   = PetscLogEventEnd(MAT_FDColoringSetUp,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatFDColoringGetFunction - Gets the function to use for computing the Jacobian.

   Not Collective

   Input Parameters:
.  coloring - the coloring context

   Output Parameters:
+  f - the function
-  fctx - the optional user-defined function context

   Level: intermediate

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

   Notes:
    This function is usually used automatically by SNES (when one uses SNESSetJacobian() with the argument
     SNESComputeJacobianDefaultColor()) and only needs to be used by someone computing a matrix via coloring directly by
     calling MatFDColoringApply()

   Fortran Notes:
    In Fortran you must call MatFDColoringSetFunction() for a coloring object to
  be used without SNES or within the SNES solvers.

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
+  -mat_fd_coloring_err <err> - Sets <err> (square root of relative error in the function)
.  -mat_fd_coloring_umin <umin> - Sets umin, the minimum allowable u-value magnitude
.  -mat_fd_type - "wp" or "ds" (see MATMFFD_WP or MATMFFD_DS)
.  -mat_fd_coloring_view - Activates basic viewing
.  -mat_fd_coloring_view ::ascii_info - Activates viewing info
-  -mat_fd_coloring_view draw - Activates drawing

    Level: intermediate

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
  ierr = PetscOptionsReal("-mat_fd_coloring_err","Square root of relative error in function","MatFDColoringSetParameters",matfd->error_rel,&matfd->error_rel,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_fd_coloring_umin","Minimum allowable u magnitude","MatFDColoringSetParameters",matfd->umin,&matfd->umin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat_fd_type","Algorithm to compute h, wp or ds","MatFDColoringCreate",matfd->htype,value,sizeof(value),&flg);CHKERRQ(ierr);
  if (flg) {
    if (value[0] == 'w' && value[1] == 'p') matfd->htype = "wp";
    else if (value[0] == 'd' && value[1] == 's') matfd->htype = "ds";
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown finite differencing type %s",value);
  }
  ierr = PetscOptionsInt("-mat_fd_coloring_brows","Number of block rows","MatFDColoringSetBlockSize",matfd->brows,&matfd->brows,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_fd_coloring_bcols","Number of block columns","MatFDColoringSetBlockSize",matfd->bcols,&matfd->bcols,&flg);CHKERRQ(ierr);
  if (flg && matfd->bcols > matfd->ncolors) {
    /* input bcols cannot be > matfd->ncolors, thus set it as ncolors */
    matfd->bcols = matfd->ncolors;
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)matfd);CHKERRQ(ierr);
  PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatFDColoringSetType - Sets the approach for computing the finite difference parameter

   Collective on MatFDColoring

   Input Parameters:
+  coloring - the coloring context
-  type - either MATMFFD_WP or MATMFFD_DS

   Options Database Keys:
.  -mat_fd_type - "wp" or "ds"

   Note: It is goofy that the argument type is MatMFFDType since the MatFDColoring actually computes the matrix entries
         but the process of computing the entries is the same as as with the MatMFFD operation so we should reuse the names instead of
         introducing another one.

   Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringView(), MatFDColoringSetParameters()

@*/
PetscErrorCode  MatFDColoringSetType(MatFDColoring matfd,MatMFFDType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_CLASSID,1);
  /*
     It is goofy to handle the strings this way but currently there is no code to free a dynamically created matfd->htype
     and this function is being provided as patch for a release so it shouldn't change the implementaton
  */
  if (type[0] == 'w' && type[1] == 'p') matfd->htype = "wp";
  else if (type[0] == 'd' && type[1] == 's') matfd->htype = "ds";
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown finite differencing type %s",type);
  PetscFunctionReturn(0);
}

PetscErrorCode MatFDColoringViewFromOptions(MatFDColoring fd,const char prefix[],const char optionname[])
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (prefix) {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)fd),((PetscObject)fd)->options,prefix,optionname,&viewer,&format,&flg);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)fd),((PetscObject)fd)->options,((PetscObject)fd)->prefix,optionname,&viewer,&format,&flg);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = MatFDColoringView(fd,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   MatFDColoringCreate - Creates a matrix coloring context for finite difference
   computation of Jacobians.

   Collective on Mat

   Input Parameters:
+  mat - the matrix containing the nonzero structure of the Jacobian
-  iscoloring - the coloring of the matrix; usually obtained with MatColoringCreate() or DMCreateColoring()

    Output Parameter:
.   color - the new coloring context

    Level: intermediate

.seealso: MatFDColoringDestroy(),SNESComputeJacobianDefaultColor(), ISColoringCreate(),
          MatFDColoringSetFunction(), MatFDColoringSetFromOptions(), MatFDColoringApply(),
          MatFDColoringView(), MatFDColoringSetParameters(), MatColoringCreate(), DMCreateColoring(), MatFDColoringSetValues()
@*/
PetscErrorCode  MatFDColoringCreate(Mat mat,ISColoring iscoloring,MatFDColoring *color)
{
  MatFDColoring  c;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (!mat->assembled) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled by calls to MatAssemblyBegin/End();");
  ierr = PetscLogEventBegin(MAT_FDColoringCreate,mat,0,0,0);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  if (M != N) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Only for square matrices");
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscHeaderCreate(c,MAT_FDCOLORING_CLASSID,"MatFDColoring","Jacobian computation via finite differences with coloring","Mat",comm,MatFDColoringDestroy,MatFDColoringView);CHKERRQ(ierr);

  c->ctype = iscoloring->ctype;
  ierr     = PetscObjectGetId((PetscObject)mat,&c->matid);CHKERRQ(ierr);

  if (mat->ops->fdcoloringcreate) {
    ierr = (*mat->ops->fdcoloringcreate)(mat,iscoloring,c);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Code not yet written for matrix type %s",((PetscObject)mat)->type_name);

  ierr = MatCreateVecs(mat,NULL,&c->w1);CHKERRQ(ierr);
  /* Vec is used instensively in particular piece of scalar CPU code; won't benifit from bouncing back and forth to the GPU */
  ierr = VecBindToCPU(c->w1,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)c,(PetscObject)c->w1);CHKERRQ(ierr);
  ierr = VecDuplicate(c->w1,&c->w2);CHKERRQ(ierr);
  /* Vec is used instensively in particular piece of scalar CPU code; won't benifit from bouncing back and forth to the GPU */
  ierr = VecBindToCPU(c->w2,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)c,(PetscObject)c->w2);CHKERRQ(ierr);

  c->error_rel    = PETSC_SQRT_MACHINE_EPSILON;
  c->umin         = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  c->currentcolor = -1;
  c->htype        = "wp";
  c->fset         = PETSC_FALSE;
  c->setupcalled  = PETSC_FALSE;

  *color = c;
  ierr   = PetscObjectCompose((PetscObject)mat,"SNESMatFDColoring",(PetscObject)c);CHKERRQ(ierr);
  ierr   = PetscLogEventEnd(MAT_FDColoringCreate,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  MatFDColoring  color = *c;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  if (--((PetscObject)color)->refct > 0) {*c = NULL; PetscFunctionReturn(0);}

  /* we do not free the column arrays since their entries are owned by the ISs in color->isa */
  for (i=0; i<color->ncolors; i++) {
    ierr = ISDestroy(&color->isa[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(color->isa);CHKERRQ(ierr);
  ierr = PetscFree2(color->ncolumns,color->columns);CHKERRQ(ierr);
  ierr = PetscFree(color->nrows);CHKERRQ(ierr);
  if (color->htype[0] == 'w') {
    ierr = PetscFree(color->matentry2);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(color->matentry);CHKERRQ(ierr);
  }
  ierr = PetscFree(color->dy);CHKERRQ(ierr);
  if (color->vscale) {ierr = VecDestroy(&color->vscale);CHKERRQ(ierr);}
  ierr = VecDestroy(&color->w1);CHKERRQ(ierr);
  ierr = VecDestroy(&color->w2);CHKERRQ(ierr);
  ierr = VecDestroy(&color->w3);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    MatFDColoringGetPerturbedColumns - Returns the indices of the columns that
      that are currently being perturbed.

    Not Collective

    Input Parameters:
.   coloring - coloring context created with MatFDColoringCreate()

    Output Parameters:
+   n - the number of local columns being perturbed
-   cols - the column indices, in global numbering

   Level: advanced

   Note: IF the matrix type is BAIJ, then the block column indices are returned

   Fortran Note:
   This routine has a different interface for Fortran
$     #include <petsc/finclude/petscmat.h>
$          use petscmat
$          PetscInt, pointer :: array(:)
$          PetscErrorCode  ierr
$          MatFDColoring   i
$          call MatFDColoringGetPerturbedColumnsF90(i,array,ierr)
$      use the entries of array ...
$          call MatFDColoringRestorePerturbedColumnsF90(i,array,ierr)

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView(), MatFDColoringApply()

@*/
PetscErrorCode  MatFDColoringGetPerturbedColumns(MatFDColoring coloring,PetscInt *n,const PetscInt *cols[])
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
.    -mat_fd_coloring_view draw - Activates drawing of coloring
-    -mat_fd_coloring_view ::ascii_info - Activates viewing of coloring info

    Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView(), MatFDColoringSetFunction(), MatFDColoringSetValues()

@*/
PetscErrorCode  MatFDColoringApply(Mat J,MatFDColoring coloring,Vec x1,void *sctx)
{
  PetscErrorCode ierr;
  PetscBool      eq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_CLASSID,2);
  PetscValidHeaderSpecific(x1,VEC_CLASSID,3);
  ierr = PetscObjectCompareId((PetscObject)J,coloring->matid,&eq);CHKERRQ(ierr);
  if (!eq) SETERRQ(PetscObjectComm((PetscObject)J),PETSC_ERR_ARG_WRONG,"Matrix used with MatFDColoringApply() must be that used with MatFDColoringCreate()");
  if (!coloring->f) SETERRQ(PetscObjectComm((PetscObject)J),PETSC_ERR_ARG_WRONGSTATE,"Must call MatFDColoringSetFunction()");
  if (!J->ops->fdcoloringapply) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not supported for this matrix type %s",((PetscObject)J)->type_name);
  if (!coloring->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatFDColoringSetUp()");

  ierr = MatSetUnfactored(J);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_FDColoringApply,coloring,J,x1,0);CHKERRQ(ierr);
  ierr = (*J->ops->fdcoloringapply)(J,coloring,x1,sctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_FDColoringApply,coloring,J,x1,0);CHKERRQ(ierr);
  if (!coloring->viewed) {
    ierr = MatFDColoringViewFromOptions(coloring,NULL,"-mat_fd_coloring_view");CHKERRQ(ierr);
    coloring->viewed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}
