
#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMView_DA_2d"
PetscErrorCode DMView_DA_2d(DM da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      iascii,isdraw,isbinary;
  DM_DA          *dd = (DM_DA*)da->data;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscBool      ismatlab;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)da)->comm,&rank);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  if (iascii) {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_VTK && format != PETSC_VIEWER_ASCII_VTK_CELL) {
      DMDALocalInfo info;
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D N %D m %D n %D w %D s %D\n",rank,dd->M,dd->N,dd->m,dd->n,dd->w,dd->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D, Y range of indices: %D %D\n",info.xs,info.xs+info.xm,info.ys,info.ys+info.ym);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = DMView_DA_VTK(da,viewer);CHKERRQ(ierr);
    }
  } else if (isdraw) {
    PetscDraw  draw;
    double     ymin = -1*dd->s-1,ymax = dd->N+dd->s;
    double     xmin = -1*dd->s-1,xmax = dd->M+dd->s;
    double     x,y;
    PetscInt   base,*idx;
    char       node[10];
    PetscBool  isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
    if (!da->coordinates) {
      ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    }
    ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draw all node lines */
    if (!rank) {
      ymin = 0.0; ymax = dd->N - 1;
      for (xmin=0; xmin<dd->M; xmin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
      xmin = 0.0; xmax = dd->M - 1;
      for (ymin=0; ymin<dd->N; ymin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = dd->ys; ymax = dd->ye - 1; xmin = dd->xs/dd->w;
    xmax =(dd->xe-1)/dd->w;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);

    /* put in numbers */
    base = (dd->base)/dd->w;
    for (y=ymin; y<=ymax; y++) {
      for (x=xmin; x<=xmax; x++) {
        sprintf(node,"%d",(int)base++);
        ierr = PetscDrawString(draw,x,y,PETSC_DRAW_BLACK,node);CHKERRQ(ierr);
      }
    }

    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
    /* overlay ghost numbers, useful for error checking */
    /* put in numbers */

    base = 0; idx = dd->idx;
    ymin = dd->Ys; ymax = dd->Ye; xmin = dd->Xs; xmax = dd->Xe;
    for (y=ymin; y<ymax; y++) {
      for (x=xmin; x<xmax; x++) {
        if ((base % dd->w) == 0) {
          sprintf(node,"%d",(int)(idx[base]/dd->w));
          ierr = PetscDrawString(draw,x/dd->w,y,PETSC_DRAW_BLUE,node);CHKERRQ(ierr);
        }
        base++;
      }
    }
    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  } else if (isbinary){
    ierr = DMView_DA_Binary(da,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = DMView_DA_Matlab(da,viewer);CHKERRQ(ierr);
#endif
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for DMDA 1d",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

/*
      M is number of grid points
      m is number of processors

*/
#undef __FUNCT__
#define __FUNCT__ "DMDASplitComm2d"
PetscErrorCode  DMDASplitComm2d(MPI_Comm comm,PetscInt M,PetscInt N,PetscInt sw,MPI_Comm *outcomm)
{
  PetscErrorCode ierr;
  PetscInt       m,n = 0,x = 0,y = 0;
  PetscMPIInt    size,csize,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  csize = 4*size;
  do {
    if (csize % 4) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot split communicator of size %d tried %d %D %D",size,csize,x,y);
    csize   = csize/4;

    m = (PetscInt)(0.5 + sqrt(((double)M)*((double)csize)/((double)N)));
    if (!m) m = 1;
    while (m > 0) {
      n = csize/m;
      if (m*n == csize) break;
      m--;
    }
    if (M > N && m < n) {PetscInt _m = m; m = n; n = _m;}

    x = M/m + ((M % m) > ((csize-1) % m));
    y = (N + (csize-1)/m)/n;
  } while ((x < 4 || y < 4) && csize > 1);
  if (size != csize) {
    MPI_Group    entire_group,sub_group;
    PetscMPIInt  i,*groupies;

    ierr = MPI_Comm_group(comm,&entire_group);CHKERRQ(ierr);
    ierr = PetscMalloc(csize*sizeof(PetscInt),&groupies);CHKERRQ(ierr);
    for (i=0; i<csize; i++) {
      groupies[i] = (rank/csize)*csize + i;
    }
    ierr = MPI_Group_incl(entire_group,csize,groupies,&sub_group);CHKERRQ(ierr);
    ierr = PetscFree(groupies);CHKERRQ(ierr);
    ierr = MPI_Comm_create(comm,sub_group,outcomm);CHKERRQ(ierr);
    ierr = MPI_Group_free(&entire_group);CHKERRQ(ierr);
    ierr = MPI_Group_free(&sub_group);CHKERRQ(ierr);
    ierr = PetscInfo1(0,"DMDASplitComm2d:Creating redundant coarse problems of size %d\n",csize);CHKERRQ(ierr);
  } else {
    *outcomm = comm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAFunction"
static PetscErrorCode DMDAFunction(DM dm,Vec x,Vec F)
{
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAComputeFunction1(dm,localX,F,dm->ctx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalFunction"
/*@C
       DMDASetLocalFunction - Caches in a DM a local function.

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  lf - the local function

   Level: intermediate

   Notes:
      If you use SNESSetDM() and did not use SNESSetFunction() then the context passed to your function is the context set with DMSetApplicationContext()

   Developer Notes: It is possibly confusing which context is passed to the user function, it would be nice to unify them somehow.

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunctioni()
@*/
PetscErrorCode  DMDASetLocalFunction(DM da,DMDALocalFunction1 lf)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = DMSetFunction(da,DMDAFunction);CHKERRQ(ierr);
  dd->lf       = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalFunctioni"
/*@C
       DMDASetLocalFunctioni - Caches in a DM a local function that evaluates a single component

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  lfi - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction()
@*/
PetscErrorCode  DMDASetLocalFunctioni(DM da,PetscErrorCode (*lfi)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*))
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->lfi = lfi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalFunctionib"
/*@C
       DMDASetLocalFunctionib - Caches in a DM a block local function that evaluates a single component

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  lfi - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction()
@*/
PetscErrorCode  DMDASetLocalFunctionib(DM da,PetscErrorCode (*lfi)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*))
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->lfib = lfi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalAdicFunction_Private"
PetscErrorCode DMDASetLocalAdicFunction_Private(DM da,DMDALocalFunction1 ad_lf)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->adic_lf = ad_lf;
  PetscFunctionReturn(0);
}

/*MC
       DMDASetLocalAdicFunctioni - Caches in a DM a local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DMDASetLocalAdicFunctioni(DM da,PetscInt (ad_lf*)(DMDALocalInfo*,MatStencil*,void*,void*,void*)

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  ad_lfi - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction(),
          DMDASetLocalJacobian(), DMDASetLocalFunctioni()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalAdicFunctioni_Private"
PetscErrorCode DMDASetLocalAdicFunctioni_Private(DM da,PetscErrorCode (*ad_lfi)(DMDALocalInfo*,MatStencil*,void*,void*,void*))
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->adic_lfi = ad_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DMDASetLocalAdicMFFunctioni - Caches in a DM a local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode  DMDASetLocalAdicFunctioni(DM da,int (ad_lf*)(DMDALocalInfo*,MatStencil*,void*,void*,void*)

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  admf_lfi - the local matrix-free function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction(),
          DMDASetLocalJacobian(), DMDASetLocalFunctioni()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalAdicMFFunctioni_Private"
PetscErrorCode DMDASetLocalAdicMFFunctioni_Private(DM da,PetscErrorCode (*admf_lfi)(DMDALocalInfo*,MatStencil*,void*,void*,void*))
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->adicmf_lfi = admf_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DMDASetLocalAdicFunctionib - Caches in a DM a block local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DMDASetLocalAdicFunctionib(DM da,PetscInt (ad_lf*)(DMDALocalInfo*,MatStencil*,void*,void*,void*)

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  ad_lfi - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction(),
          DMDASetLocalJacobian(), DMDASetLocalFunctionib()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalAdicFunctionib_Private"
PetscErrorCode DMDASetLocalAdicFunctionib_Private(DM da,PetscErrorCode (*ad_lfi)(DMDALocalInfo*,MatStencil*,void*,void*,void*))
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->adic_lfib = ad_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DMDASetLocalAdicMFFunctionib - Caches in a DM a block local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode  DMDASetLocalAdicFunctionib(DM da,int (ad_lf*)(DMDALocalInfo*,MatStencil*,void*,void*,void*)

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  admf_lfi - the local matrix-free function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction(),
          DMDASetLocalJacobian(), DMDASetLocalFunctionib()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalAdicMFFunctionib_Private"
PetscErrorCode DMDASetLocalAdicMFFunctionib_Private(DM da,PetscErrorCode (*admf_lfi)(DMDALocalInfo*,MatStencil*,void*,void*,void*))
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->adicmf_lfib = admf_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DMDASetLocalAdicMFFunction - Caches in a DM a local function computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DMDASetLocalAdicMFFunction(DM da,DMDALocalFunction1 ad_lf)

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
-  ad_lf - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction(),
          DMDASetLocalJacobian()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalAdicMFFunction_Private"
PetscErrorCode DMDASetLocalAdicMFFunction_Private(DM da,DMDALocalFunction1 ad_lf)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dd->adicmf_lf = ad_lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAJacobianDefaultLocal"
PetscErrorCode DMDAJacobianLocal(DM dm,Vec x,Mat A,Mat B, MatStructure *str)
{
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = MatFDColoringApply(B,dm->fd,localX,str,dm);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (A != B) {
    ierr  = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *str = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDAJacobian"
static PetscErrorCode DMDAJacobian(DM dm,Vec x,Mat A,Mat B, MatStructure *str)
{
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAComputeJacobian1(dm,localX,B,dm->ctx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (A != B) {
    ierr  = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *str = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*@C
       DMDASetLocalJacobian - Caches in a DM a local Jacobian computation function

   Logically Collective on DMDA


   Input Parameter:
+  da - initial distributed array
-  lj - the local Jacobian

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction()
@*/
#undef __FUNCT__
#define __FUNCT__ "DMDASetLocalJacobian"
PetscErrorCode  DMDASetLocalJacobian(DM da,DMDALocalFunction1 lj)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = DMSetJacobian(da,DMDAJacobian);CHKERRQ(ierr);
  dd->lj    = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetLocalFunction"
/*@C
       DMDAGetLocalFunction - Gets from a DM a local function and its ADIC/ADIFOR Jacobian

   Note Collective

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  lf - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalJacobian(), DMDASetLocalFunction()
@*/
PetscErrorCode  DMDAGetLocalFunction(DM da,DMDALocalFunction1 *lf)
{
  DM_DA *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (lf) *lf = dd->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetLocalJacobian"
/*@C
       DMDAGetLocalJacobian - Gets from a DM a local jacobian

   Not Collective

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  lj - the local jacobian

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalJacobian()
@*/
PetscErrorCode  DMDAGetLocalJacobian(DM da,DMDALocalFunction1 *lj)
{
  DM_DA *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (lj) *lj = dd->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunction"
/*@
    DMDAComputeFunction - Evaluates a user provided function on each processor that
        share a DMDA

   Input Parameters:
+    da - the DM that defines the grid
.    vu - input vector
.    vfu - output vector
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

           This should eventually replace DMDAComputeFunction1

    Level: advanced

.seealso: DMDAComputeJacobian1WithAdic()

@*/
PetscErrorCode  DMDAComputeFunction(DM da,PetscErrorCode (*lf)(void),Vec vu,Vec vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u,*fu;
  DMDALocalInfo  info;
  PetscErrorCode (*f)(DMDALocalInfo*,void*,void*,void*) = (PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))lf;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vfu,&fu);CHKERRQ(ierr);

  ierr = (*f)(&info,u,fu,w);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,vfu,&fu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunctionLocal"
/*@C
   DMDAComputeFunctionLocal - This is a universal function evaluation routine for
   a local DM function.

   Collective on DMDA

   Input Parameters:
+  da - the DM context
.  func - The local function
.  X - input vector
.  F - function vector
-  ctx - A user context

   Level: intermediate

.seealso: DMDASetLocalFunction(), DMDASetLocalJacobian(), DMDASetLocalAdicFunction(), DMDASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode  DMDAComputeFunctionLocal(DM da, DMDALocalFunction1 func, Vec X, Vec F, void *ctx)
{
  Vec            localX;
  DMDALocalInfo  info;
  void           *u;
  void           *fu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&fu);CHKERRQ(ierr);
  ierr = (*func)(&info,u,fu,ctx);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&fu);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunctionLocalGhost"
/*@C
   DMDAComputeFunctionLocalGhost - This is a universal function evaluation routine for
   a local DM function, but the ghost values of the output are communicated and added.

   Collective on DMDA

   Input Parameters:
+  da - the DM context
.  func - The local function
.  X - input vector
.  F - function vector
-  ctx - A user context

   Level: intermediate

.seealso: DMDASetLocalFunction(), DMDASetLocalJacobian(), DMDASetLocalAdicFunction(), DMDASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode  DMDAComputeFunctionLocalGhost(DM da, DMDALocalFunction1 func, Vec X, Vec F, void *ctx)
{
  Vec            localX, localF;
  DMDALocalInfo  info;
  void           *u;
  void           *fu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localF);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localF,&fu);CHKERRQ(ierr);
  ierr = (*func)(&info,u,fu,ctx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,localF,&fu);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunction1"
/*@
    DMDAComputeFunction1 - Evaluates a user provided function on each processor that
        share a DMDA

   Input Parameters:
+    da - the DM that defines the grid
.    vu - input vector
.    vfu - output vector
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DMDAComputeJacobian1WithAdic()

@*/
PetscErrorCode  DMDAComputeFunction1(DM da,Vec vu,Vec vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u,*fu;
  DMDALocalInfo  info;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  if (!dd->lf) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_NULL,"DMDASetLocalFunction() never called");
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vfu,&fu);CHKERRQ(ierr);

  CHKMEMQ;
  ierr = (*dd->lf)(&info,u,fu,w);CHKERRQ(ierr);
  CHKMEMQ;

  ierr = DMDAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,vfu,&fu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunctioniTest1"
PetscErrorCode  DMDAComputeFunctioniTest1(DM da,void *w)
{
  Vec            vu,fu,fui;
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *ui;
  PetscRandom    rnd;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = VecSetRandom(vu,rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(da,&fu);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&fui);CHKERRQ(ierr);

  ierr = DMDAComputeFunction1(da,vu,fu,w);CHKERRQ(ierr);

  ierr = VecGetArray(fui,&ui);CHKERRQ(ierr);
  ierr = VecGetLocalSize(fui,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = DMDAComputeFunctioni1(da,i,vu,ui+i,w);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(fui,&ui);CHKERRQ(ierr);

  ierr = VecAXPY(fui,-1.0,fu);CHKERRQ(ierr);
  ierr = VecNorm(fui,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(((PetscObject)da)->comm,"Norm of difference in vectors %G\n",norm);CHKERRQ(ierr);
  ierr = VecView(fu,0);CHKERRQ(ierr);
  ierr = VecView(fui,0);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&fu);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&fui);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunctioni1"
/*@
    DMDAComputeFunctioni1 - Evaluates a user provided point-wise function

   Input Parameters:
+    da - the DM that defines the grid
.    i - the component of the function we wish to compute (must be local)
.    vu - input vector
.    vfu - output value
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DMDAComputeJacobian1WithAdic()

@*/
PetscErrorCode  DMDAComputeFunctioni1(DM da,PetscInt i,Vec vu,PetscScalar *vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u;
  DMDALocalInfo  info;
  MatStencil     stencil;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vu,&u);CHKERRQ(ierr);

  /* figure out stencil value from i */
  stencil.c = i % info.dof;
  stencil.i = (i % (info.xm*info.dof))/info.dof;
  stencil.j = (i % (info.xm*info.ym*info.dof))/(info.xm*info.dof);
  stencil.k = i/(info.xm*info.ym*info.dof);

  ierr = (*dd->lfi)(&info,&stencil,u,vfu,w);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeFunctionib1"
/*@
    DMDAComputeFunctionib1 - Evaluates a user provided point-block function

   Input Parameters:
+    da - the DM that defines the grid
.    i - the component of the function we wish to compute (must be local)
.    vu - input vector
.    vfu - output value
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DMDAComputeJacobian1WithAdic()

@*/
PetscErrorCode  DMDAComputeFunctionib1(DM da,PetscInt i,Vec vu,PetscScalar *vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u;
  DMDALocalInfo  info;
  MatStencil     stencil;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vu,&u);CHKERRQ(ierr);

  /* figure out stencil value from i */
  stencil.c = i % info.dof;
  if (stencil.c) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Point-block functions can only be called for the entire block");
  stencil.i = (i % (info.xm*info.dof))/info.dof;
  stencil.j = (i % (info.xm*info.ym*info.dof))/(info.xm*info.dof);
  stencil.k = i/(info.xm*info.ym*info.dof);

  ierr = (*dd->lfib)(&info,&stencil,u,vfu,w);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(new)
#undef __FUNCT__
#define __FUNCT__ "DMDAGetDiagonal_MFFD"
/*
  DMDAGetDiagonal_MFFD - Gets the diagonal for a matrix free matrix where local
    function lives on a DMDA

        y ~= (F(u + ha) - F(u))/h,
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
PetscErrorCode DMDAGetDiagonal_MFFD(DM da,Vec U,Vec a)
{
  PetscScalar    h,*aa,*ww,v;
  PetscReal      epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  PetscErrorCode ierr;
  PetscInt       gI,nI;
  MatStencil     stencil;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = (*ctx->func)(0,U,a,ctx->funcctx);CHKERRQ(ierr);
  ierr = (*ctx->funcisetbase)(U,ctx->funcctx);CHKERRQ(ierr);

  ierr = VecGetArray(U,&ww);CHKERRQ(ierr);
  ierr = VecGetArray(a,&aa);CHKERRQ(ierr);

  nI = 0;
    h  = ww[gI];
    if (h == 0.0) h = 1.0;
#if !defined(PETSC_USE_COMPLEX)
    if (h < umin && h >= 0.0)      h = umin;
    else if (h < 0.0 && h > -umin) h = -umin;
#else
    if (PetscAbsScalar(h) < umin && PetscRealPart(h) >= 0.0)     h = umin;
    else if (PetscRealPart(h) < 0.0 && PetscAbsScalar(h) < umin) h = -umin;
#endif
    h     *= epsilon;

    ww[gI] += h;
    ierr          = (*ctx->funci)(i,w,&v,ctx->funcctx);CHKERRQ(ierr);
    aa[nI]  = (v - aa[nI])/h;
    ww[gI] -= h;
    nI++;
  }
  ierr = VecRestoreArray(U,&ww);CHKERRQ(ierr);
  ierr = VecRestoreArray(a,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_ADIC)
EXTERN_C_BEGIN
#include <adic/ad_utils.h>
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeJacobian1WithAdic"
/*@C
    DMDAComputeJacobian1WithAdic - Evaluates a adiC provided Jacobian function on each processor that
        share a DMDA

   Input Parameters:
+    da - the DM that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

   Level: advanced

    Notes: Does NOT do ghost updates on vu upon entry

.seealso: DMDAComputeFunction1()

@*/
PetscErrorCode  DMDAComputeJacobian1WithAdic(DM da,Vec vu,Mat J,void *w)
{
  PetscErrorCode ierr;
  PetscInt       gtdof,tdof;
  PetscScalar    *ustart;
  DMDALocalInfo  info;
  void           *ad_u,*ad_f,*ad_ustart,*ad_fstart;
  ISColoring     iscoloring;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  PetscADResetIndep();

  /* get space for derivative objects.  */
  ierr = DMDAGetAdicArray(da,PETSC_TRUE,&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DMDAGetAdicArray(da,PETSC_FALSE,&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  ierr = VecGetArray(vu,&ustart);CHKERRQ(ierr);
  ierr = DMCreateColoring(da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);

  PetscADSetValueAndColor(ad_ustart,gtdof,iscoloring->colors,ustart);

  ierr = VecRestoreArray(vu,&ustart);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = PetscADIncrementTotalGradSize(iscoloring->n);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = PetscLogEventBegin(DMDA_LocalADFunction,0,0,0,0);CHKERRQ(ierr);
  ierr = (*dd->adic_lf)(&info,ad_u,ad_f,w);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMDA_LocalADFunction,0,0,0,0);CHKERRQ(ierr);

  /* stick the values into the matrix */
  ierr = MatSetValuesAdic(J,(PetscScalar**)ad_fstart);CHKERRQ(ierr);

  /* return space for derivative objects.  */
  ierr = DMDARestoreAdicArray(da,PETSC_TRUE,&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DMDARestoreAdicArray(da,PETSC_FALSE,&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAMultiplyByJacobian1WithAdic"
/*@C
    DMDAMultiplyByJacobian1WithAdic - Applies an ADIC-provided Jacobian function to a vector on
    each processor that shares a DMDA.

    Input Parameters:
+   da - the DM that defines the grid
.   vu - Jacobian is computed at this point (ghosted)
.   v - product is done on this vector (ghosted)
.   fu - output vector = J(vu)*v (not ghosted)
-   w - any user data

    Notes:
    This routine does NOT do ghost updates on vu upon entry.

   Level: advanced

.seealso: DMDAComputeFunction1()

@*/
PetscErrorCode  DMDAMultiplyByJacobian1WithAdic(DM da,Vec vu,Vec v,Vec f,void *w)
{
  PetscErrorCode ierr;
  PetscInt       i,gtdof,tdof;
  PetscScalar    *avu,*av,*af,*ad_vustart,*ad_fstart;
  DMDALocalInfo  info;
  void           *ad_vu,*ad_f;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* get space for derivative objects.  */
  ierr = DMDAGetAdicMFArray(da,PETSC_TRUE,&ad_vu,&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = DMDAGetAdicMFArray(da,PETSC_FALSE,&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);

  /* copy input vector into derivative object */
  ierr = VecGetArray(vu,&avu);CHKERRQ(ierr);
  ierr = VecGetArray(v,&av);CHKERRQ(ierr);
  for (i=0; i<gtdof; i++) {
    ad_vustart[2*i]   = avu[i];
    ad_vustart[2*i+1] = av[i];
  }
  ierr = VecRestoreArray(vu,&avu);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&av);CHKERRQ(ierr);

  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(1);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = (*dd->adicmf_lf)(&info,ad_vu,ad_f,w);CHKERRQ(ierr);

  /* stick the values into the vector */
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);
  for (i=0; i<tdof; i++) {
    af[i] = ad_fstart[2*i+1];
  }
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);

  /* return space for derivative objects.  */
  ierr = DMDARestoreAdicMFArray(da,PETSC_TRUE,&ad_vu,&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = DMDARestoreAdicMFArray(da,PETSC_FALSE,&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeJacobian1"
/*@
    DMDAComputeJacobian1 - Evaluates a local Jacobian function on each processor that
        share a DMDA

   Input Parameters:
+    da - the DM that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DMDAComputeFunction1()

@*/
PetscErrorCode  DMDAComputeJacobian1(DM da,Vec vu,Mat J,void *w)
{
  PetscErrorCode ierr;
  void           *u;
  DMDALocalInfo  info;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = (*dd->lj)(&info,u,J,w);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDAComputeJacobian1WithAdifor"
/*
    DMDAComputeJacobian1WithAdifor - Evaluates a ADIFOR provided Jacobian local function on each processor that
        share a DMDA

   Input Parameters:
+    da - the DM that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

.seealso: DMDAComputeFunction1()

*/
PetscErrorCode  DMDAComputeJacobian1WithAdifor(DM da,Vec vu,Mat J,void *w)
{
  PetscErrorCode  ierr;
  PetscInt        i,Nc,N;
  ISColoringValue *color;
  DMDALocalInfo   info;
  PetscScalar     *u,*g_u,*g_f,*f = 0,*p_u;
  ISColoring      iscoloring;
  DM_DA          *dd = (DM_DA*)da->data;
  void            (*lf)(PetscInt*,DMDALocalInfo*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*) =
                  (void (*)(PetscInt*,DMDALocalInfo*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*))*dd->adifor_lf;

  PetscFunctionBegin;
  ierr = DMCreateColoring(da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);
  Nc   = iscoloring->n;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  N    = info.gxm*info.gym*info.gzm*info.dof;

  /* get space for derivative objects.  */
  ierr  = PetscMalloc(Nc*info.gxm*info.gym*info.gzm*info.dof*sizeof(PetscScalar),&g_u);CHKERRQ(ierr);
  ierr  = PetscMemzero(g_u,Nc*info.gxm*info.gym*info.gzm*info.dof*sizeof(PetscScalar));CHKERRQ(ierr);
  p_u   = g_u;
  color = iscoloring->colors;
  for (i=0; i<N; i++) {
    p_u[*color++] = 1.0;
    p_u          += Nc;
  }
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nc*info.xm*info.ym*info.zm*info.dof,PetscScalar,&g_f,info.xm*info.ym*info.zm*info.dof,PetscScalar,&f);CHKERRQ(ierr);

  /* Seed the input array g_u with coloring information */

  ierr = VecGetArray(vu,&u);CHKERRQ(ierr);
  (lf)(&Nc,&info,u,g_u,&Nc,f,g_f,&Nc,w,&ierr);CHKERRQ(ierr);
  ierr = VecRestoreArray(vu,&u);CHKERRQ(ierr);

  /* stick the values into the matrix */
  /* PetscScalarView(Nc*info.xm*info.ym,g_f,0); */
  ierr = MatSetValuesAdifor(J,Nc,g_f);CHKERRQ(ierr);

  /* return space for derivative objects.  */
  ierr = PetscFree(g_u);CHKERRQ(ierr);
  ierr = PetscFree2(g_f,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAFormJacobianLocal"
/*@C
   DMDAFormjacobianLocal - This is a universal Jacobian evaluation routine for
   a local DM function.

   Collective on DMDA

   Input Parameters:
+  da - the DM context
.  func - The local function
.  X - input vector
.  J - Jacobian matrix
-  ctx - A user context

   Level: intermediate

.seealso: DMDASetLocalFunction(), DMDASetLocalJacobian(), DMDASetLocalAdicFunction(), DMDASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode  DMDAFormJacobianLocal(DM da, DMDALocalFunction1 func, Vec X, Mat J, void *ctx)
{
  Vec            localX;
  DMDALocalInfo  info;
  void           *u;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = (*func)(&info,u,J,ctx);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAMultiplyByJacobian1WithAD"
/*@C
    DMDAMultiplyByJacobian1WithAD - Applies a Jacobian function supplied by ADIFOR or ADIC
    to a vector on each processor that shares a DMDA.

   Input Parameters:
+    da - the DM that defines the grid
.    vu - Jacobian is computed at this point (ghosted)
.    v - product is done on this vector (ghosted)
.    fu - output vector = J(vu)*v (not ghosted)
-    w - any user data

    Notes:
    This routine does NOT do ghost updates on vu and v upon entry.

    Automatically calls DMDAMultiplyByJacobian1WithAdifor() or DMDAMultiplyByJacobian1WithAdic()
    depending on whether DMDASetLocalAdicMFFunction() or DMDASetLocalAdiforMFFunction() was called.

   Level: advanced

.seealso: DMDAComputeFunction1(), DMDAMultiplyByJacobian1WithAdifor(), DMDAMultiplyByJacobian1WithAdic()

@*/
PetscErrorCode  DMDAMultiplyByJacobian1WithAD(DM da,Vec u,Vec v,Vec f,void *w)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  if (dd->adicmf_lf) {
#if defined(PETSC_HAVE_ADIC)
    ierr = DMDAMultiplyByJacobian1WithAdic(da,u,v,f,w);CHKERRQ(ierr);
#else
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP_SYS,"Requires ADIC to be installed and cannot use complex numbers");
#endif
  } else if (dd->adiformf_lf) {
    ierr = DMDAMultiplyByJacobian1WithAdifor(da,u,v,f,w);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ORDER,"Must call DMDASetLocalAdiforMFFunction() or DMDASetLocalAdicMFFunction() before using");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDAMultiplyByJacobian1WithAdifor"
/*@C
    DMDAMultiplyByJacobian1WithAdifor - Applies a ADIFOR provided Jacobian function on each processor that
        share a DM to a vector

   Input Parameters:
+    da - the DM that defines the grid
.    vu - Jacobian is computed at this point (ghosted)
.    v - product is done on this vector (ghosted)
.    fu - output vector = J(vu)*v (not ghosted)
-    w - any user data

    Notes: Does NOT do ghost updates on vu and v upon entry

   Level: advanced

.seealso: DMDAComputeFunction1()

@*/
PetscErrorCode  DMDAMultiplyByJacobian1WithAdifor(DM da,Vec u,Vec v,Vec f,void *w)
{
  PetscErrorCode ierr;
  PetscScalar    *au,*av,*af,*awork;
  Vec            work;
  DMDALocalInfo  info;
  DM_DA          *dd = (DM_DA*)da->data;
  void           (*lf)(DMDALocalInfo*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*) =
                 (void (*)(DMDALocalInfo*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))*dd->adiformf_lf;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(da,&work);CHKERRQ(ierr);
  ierr = VecGetArray(u,&au);CHKERRQ(ierr);
  ierr = VecGetArray(v,&av);CHKERRQ(ierr);
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);
  ierr = VecGetArray(work,&awork);CHKERRQ(ierr);
  (lf)(&info,au,av,awork,af,w,&ierr);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&au);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&av);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);
  ierr = VecRestoreArray(work,&awork);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&work);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_DA_2D"
PetscErrorCode  DMSetUp_DA_2D(DM da)
{
  DM_DA            *dd = (DM_DA*)da->data;
  const PetscInt   M            = dd->M;
  const PetscInt   N            = dd->N;
  PetscInt         m            = dd->m;
  PetscInt         n            = dd->n;
  PetscInt         o            = dd->overlap;
  const PetscInt   dof          = dd->w;
  const PetscInt   s            = dd->s;
  DMDABoundaryType bx         = dd->bx;
  DMDABoundaryType by         = dd->by;
  DMDAStencilType  stencil_type = dd->stencil_type;
  PetscInt         *lx           = dd->lx;
  PetscInt         *ly           = dd->ly;
  MPI_Comm         comm;
  PetscMPIInt      rank,size;
  PetscInt         xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,start,end,IXs,IXe,IYs,IYe;
  PetscInt         up,down,left,right,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn,*idx_cpy;
  const PetscInt   *idx_full;
  PetscInt         xbase,*bases,*ldims,j,x_t,y_t,s_t,base,count;
  PetscInt         s_x,s_y; /* s proportionalized to w */
  PetscInt         sn0 = 0,sn2 = 0,sn6 = 0,sn8 = 0;
  Vec              local,global;
  VecScatter       ltog,gtol;
  IS               to,from,ltogis;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (dof < 1) SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dof);
  if (s < 0) SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",s);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (((Petsc64bitInt) M)*((Petsc64bitInt) N)*((Petsc64bitInt) dof) > (Petsc64bitInt) PETSC_MPI_INT_MAX) SETERRQ3(comm,PETSC_ERR_INT_OVERFLOW,"Mesh of %D by %D by %D (dof) is too large for 32 bit indices",M,N,dof);
#endif

  if (dof < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dof);
  if (s < 0) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",s);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  dd->dim         = 2;
  ierr = PetscMalloc(dof*sizeof(char*),&dd->fieldname);CHKERRQ(ierr);
  ierr = PetscMemzero(dd->fieldname,dof*sizeof(char*));CHKERRQ(ierr);

  if (m != PETSC_DECIDE) {
    if (m < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in X direction: %D",m);
    else if (m > size) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in X direction: %D %d",m,size);
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Y direction: %D",n);
    else if (n > size) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Y direction: %D %d",n,size);
  }

  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    if (n != PETSC_DECIDE) {
      m = size/n;
    } else if (m != PETSC_DECIDE) {
      n = size/m;
    } else {
      /* try for squarish distribution */
      m = (PetscInt)(0.5 + sqrt(((double)M)*((double)size)/((double)N)));
      if (!m) m = 1;
      while (m > 0) {
	n = size/m;
	if (m*n == size) break;
	m--;
      }
      if (M > N && m < n) {PetscInt _m = m; m = n; n = _m;}
    }
    if (m*n != size) SETERRQ(comm,PETSC_ERR_PLIB,"Unable to create partition, check the size of the communicator and input m and n ");
  } else if (m*n != size) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition");

  if (M < m) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %D %D",M,m);
  if (N < n) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %D %D",N,n);

  /*
     Determine locally owned region
     xs is the first local node number, x is the number of local nodes
  */
  if (!lx) {
    ierr = PetscMalloc(m*sizeof(PetscInt), &dd->lx);CHKERRQ(ierr);
    lx = dd->lx;
    for (i=0; i<m; i++) {
      lx[i] = M/m + ((M % m) > i);
    }
  }
  x  = lx[rank % m];
  xs = 0;
  for (i=0; i<(rank % m); i++) {
    xs += lx[i];
  }
#if defined(PETSC_USE_DEBUG)
  left = xs;
  for (i=(rank % m); i<m; i++) {
    left += lx[i];
  }
  if (left != M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Sum of lx across processors not equal to M: %D %D",left,M);
#endif

  /*
     Determine locally owned region
     ys is the first local node number, y is the number of local nodes
  */
  if (!ly) {
    ierr = PetscMalloc(n*sizeof(PetscInt), &dd->ly);CHKERRQ(ierr);
    ly = dd->ly;
    for (i=0; i<n; i++) {
      ly[i] = N/n + ((N % n) > i);
    }
  }
  y  = ly[rank/m];
  ys = 0;
  for (i=0; i<(rank/m); i++) {
    ys += ly[i];
  }
#if defined(PETSC_USE_DEBUG)
  left = ys;
  for (i=(rank/m); i<n; i++) {
    left += ly[i];
  }
  if (left != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Sum of ly across processors not equal to N: %D %D",left,N);
#endif

  /*
   check if the scatter requires more than one process neighbor or wraps around
   the domain more than once
  */
  if ((x < s+o) && ((m > 1) || (bx == DMDA_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local x-width of domain x %D is smaller than stencil width s %D",x,s+o);
  if ((y < s+o) && ((n > 1) || (by == DMDA_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local y-width of domain y %D is smaller than stencil width s %D",y,s+o);
  xe = xs + x;
  ye = ys + y;

  /* determine ghost region (Xs) and region scattered into (IXs)  */
  if (xs-s-o > 0) {
    Xs = xs - s - o; IXs = xs - s - o;
  } else {
    if (bx) {
      Xs = xs - s;
    } else {
      Xs = 0;
    }
    IXs = 0;
  }
  if (xe+s+o <= M) {
    Xe = xe + s + o; IXe = xe + s + o;
  } else {
    if (bx) {
      Xs = xs - s - o; Xe = xe + s;
    } else {
      Xe = M;
    }
    IXe = M;
  }

  if (bx == DMDA_BOUNDARY_PERIODIC) {
    IXs = xs - s - o;
    IXe = xe + s + o;
    Xs = xs - s - o;
    Xe = xe + s + o;
  }

  if (ys-s-o > 0) {
    Ys = ys - s - o; IYs = ys - s - o;
  } else {
    if (by) {
      Ys = ys - s;
    } else {
      Ys = 0;
    }
    IYs = 0;
  }
  if (ye+s+o <= N) {
    Ye = ye + s + o; IYe = ye + s + o;
  } else {
    if (by) {
      Ye = ye + s;
    } else {
      Ye = N;
    }
    IYe = N;
  }

  if (by == DMDA_BOUNDARY_PERIODIC) {
    IYs = ys - s - o;
    IYe = ye + s + o;
    Ys = ys - s - o;
    Ye = ye + s + o;
  }

  /* stencil length in each direction */
  s_x = s + o;
  s_y = s + o;

  /* determine starting point of each processor */
  nn    = x*y;
  ierr  = PetscMalloc2(size+1,PetscInt,&bases,size,PetscInt,&ldims);CHKERRQ(ierr);
  ierr  = MPI_Allgather(&nn,1,MPIU_INT,ldims,1,MPIU_INT,comm);CHKERRQ(ierr);
  bases[0] = 0;
  for (i=1; i<=size; i++) {
    bases[i] = ldims[i-1];
  }
  for (i=1; i<=size; i++) {
    bases[i] += bases[i-1];
  }
  base = bases[rank]*dof;

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = x*y*dof;
  ierr = VecCreateMPIWithArray(comm,dof,dd->Nlocal,PETSC_DECIDE,0,&global);CHKERRQ(ierr);
  dd->nlocal = (Xe-Xs)*(Ye-Ys)*dof;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dof,dd->nlocal,0,&local);CHKERRQ(ierr);

  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x*y*dof,start,1,&to);CHKERRQ(ierr);

  count = x*y;
  ierr = PetscMalloc(x*y*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  left = xs - Xs; right = left + x;
  down = ys - Ys; up = down + y;
  count = 0;
  for (i=down; i<up; i++) {
    for (j=left; j<right; j++) {
      idx[count++] = i*(Xe-Xs) + j;
    }
  }

  ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(dd,ltog);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);

  /* global to local must include ghost points within the domain,
     but not ghost points outside the domain that aren't periodic */
  if (stencil_type == DMDA_STENCIL_BOX || o > 0) {
    count = (IXe-IXs)*(IYe-IYs);
    ierr  = PetscMalloc(count*sizeof(PetscInt),&idx);CHKERRQ(ierr);

    left = IXs - Xs; right = left + (IXe-IXs);
    down = IYs - Ys; up = down + (IYe-IYs);
    count = 0;
    for (i=down; i<up; i++) {
      for (j=left; j<right; j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);

  } else {
    /* must drop into cross shape region */
    /*       ---------|
            |  top    |
         |---         ---| up
         |   middle      |
         |               |
         ----         ---- down
            | bottom  |
            -----------
         Xs xs        xe Xe */
    count = (ys-IYs)*x + y*(IXe-IXs) + (IYe-ye)*x;
    ierr  = PetscMalloc(count*sizeof(PetscInt),&idx);CHKERRQ(ierr);

    left = xs - Xs; right = left + x;
    down = ys - Ys; up = down + y;
    count = 0;
    /* bottom */
    for (i=(IYs-Ys); i<down; i++) {
      for (j=left; j<right; j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    /* middle */
    for (i=down; i<up; i++) {
      for (j=(IXs-Xs); j<(IXe-Xs); j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    /* top */
    for (i=up; i<up+IYe-ye; i++) {
      for (j=left; j<right; j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);
  }


  /* determine who lies on each side of us stored in    n6 n7 n8
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

  if (bx == DMDA_BOUNDARY_PERIODIC && by == DMDA_BOUNDARY_PERIODIC) {
  /* Modify for Periodic Cases */
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
  } else if (by == DMDA_BOUNDARY_PERIODIC) {  /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;
  } else if (bx == DMDA_BOUNDARY_PERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  }

  ierr = PetscMalloc(9*sizeof(PetscInt),&dd->neighbors);CHKERRQ(ierr);
  dd->neighbors[0] = n0;
  dd->neighbors[1] = n1;
  dd->neighbors[2] = n2;
  dd->neighbors[3] = n3;
  dd->neighbors[4] = rank;
  dd->neighbors[5] = n5;
  dd->neighbors[6] = n6;
  dd->neighbors[7] = n7;
  dd->neighbors[8] = n8;

  if (stencil_type == DMDA_STENCIL_STAR && o == 0) {
    /* save corner processor numbers */
    sn0 = n0; sn2 = n2; sn6 = n6; sn8 = n8;
    n0 = n2 = n6 = n8 = -1;
  }

  ierr = PetscMalloc((Xe-Xs)*(Ye-Ys)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(da,(Xe-Xs)*(Ye-Ys)*sizeof(PetscInt));CHKERRQ(ierr);

  nn = 0;
  xbase = bases[rank];
  for (i=1; i<=s_y; i++) {
    if (n0 >= 0) { /* left below */
      x_t = lx[n0 % m];
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
      x_t = lx[n2 % m];
      y_t = ly[(n2/m)];
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=0; i<y; i++) {
    if (n3 >= 0) { /* directly left */
      x_t = lx[n3 % m];
      /* y_t = y; */
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }

    for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = lx[n5 % m];
      /* y_t = y; */
      s_t = bases[n5] + (i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=1; i<=s_y; i++) {
    if (n6 >= 0) { /* left above */
      x_t = lx[n6 % m];
      /* y_t = ly[(n6/m)]; */
      s_t = bases[n6] + (i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      /* y_t = ly[(n7/m)]; */
      s_t = bases[n7] + (i-1)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = lx[n8 % m];
      /* y_t = ly[(n8/m)]; */
      s_t = bases[n8] + (i-1)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  ierr = ISCreateBlock(comm,dof,nn,idx,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  if (stencil_type == DMDA_STENCIL_STAR && o == 0) {
    n0 = sn0; n2 = sn2; n6 = sn6; n8 = sn8;
  }

  if (((stencil_type == DMDA_STENCIL_STAR) ||
       (bx && bx != DMDA_BOUNDARY_PERIODIC) ||
       (by && by != DMDA_BOUNDARY_PERIODIC)) && o == 0) {
    /*
        Recompute the local to global mappings, this time keeping the
      information about the cross corner processor numbers and any ghosted
      but not periodic indices.
    */
    nn = 0;
    xbase = bases[rank];
    for (i=1; i<=s_y; i++) {
      if (n0 >= 0) { /* left below */
        x_t = lx[n0 % m];
        y_t = ly[(n0/m)];
        s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      } else if (xs-Xs > 0 && ys-Ys > 0) {
        for (j=0; j<s_x; j++) { idx[nn++] = -1;}
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n1/m)];
        s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
      } else if (ys-Ys > 0) {
        for (j=0; j<x; j++) { idx[nn++] = -1;}
      }
      if (n2 >= 0) { /* right below */
        x_t = lx[n2 % m];
        y_t = ly[(n2/m)];
        s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      } else if (Xe-xe> 0 && ys-Ys > 0) {
        for (j=0; j<s_x; j++) { idx[nn++] = -1;}
      }
    }

    for (i=0; i<y; i++) {
      if (n3 >= 0) { /* directly left */
        x_t = lx[n3 % m];
        /* y_t = y; */
        s_t = bases[n3] + (i+1)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      } else if (xs-Xs > 0) {
        for (j=0; j<s_x; j++) { idx[nn++] = -1;}
      }

      for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

      if (n5 >= 0) { /* directly right */
        x_t = lx[n5 % m];
        /* y_t = y; */
        s_t = bases[n5] + (i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      } else if (Xe-xe > 0) {
        for (j=0; j<s_x; j++) { idx[nn++] = -1;}
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n6 >= 0) { /* left above */
        x_t = lx[n6 % m];
        /* y_t = ly[(n6/m)]; */
        s_t = bases[n6] + (i)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      } else if (xs-Xs > 0 && Ye-ye > 0) {
        for (j=0; j<s_x; j++) { idx[nn++] = -1;}
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        /* y_t = ly[(n7/m)]; */
        s_t = bases[n7] + (i-1)*x_t;
        for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
      } else if (Ye-ye > 0) {
        for (j=0; j<x; j++) { idx[nn++] = -1;}
      }
      if (n8 >= 0) { /* right above */
        x_t = lx[n8 % m];
        /* y_t = ly[(n8/m)]; */
        s_t = bases[n8] + (i-1)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      } else if (Xe-xe > 0 && Ye-ye > 0) {
        for (j=0; j<s_x; j++) { idx[nn++] = -1;}
      }
    }
  }
  /*
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  ierr = ISCreateBlock(comm,dof,nn,idx,PETSC_OWN_POINTER,&ltogis);CHKERRQ(ierr);
  ierr = PetscMalloc(nn*dof*sizeof(PetscInt),&idx_cpy);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(da,nn*dof*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISGetIndices(ltogis, &idx_full);
  ierr = PetscMemcpy(idx_cpy,idx_full,nn*dof*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISRestoreIndices(ltogis, &idx_full);
  ierr = ISLocalToGlobalMappingCreateIS(ltogis,&da->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,da->ltogmap);CHKERRQ(ierr);
  ierr = ISDestroy(&ltogis);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(da->ltogmap,dd->w,&da->ltogmapb);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,da->ltogmap);CHKERRQ(ierr);

  ierr = PetscFree2(bases,ldims);CHKERRQ(ierr);
  dd->m  = m;  dd->n  = n;
  /* note petsc expects xs/xe/Xs/Xe to be multiplied by #dofs in many places */
  dd->xs = xs*dof; dd->xe = xe*dof; dd->ys = ys; dd->ye = ye; dd->zs = 0; dd->ze = 1;
  dd->Xs = Xs*dof; dd->Xe = Xe*dof; dd->Ys = Ys; dd->Ye = Ye; dd->Zs = 0; dd->Ze = 1;

  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  dd->gtol      = gtol;
  dd->ltog      = ltog;
  dd->idx       = idx_cpy;
  dd->Nl        = nn*dof;
  dd->base      = base;
  da->ops->view = DMView_DA_2d;
  dd->ltol = PETSC_NULL;
  dd->ao   = PETSC_NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDACreate2d"
/*@C
   DMDACreate2d -  Creates an object that will manage the communication of  two-dimensional
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  bx,by - type of ghost nodes the array have.
         Use one of DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_PERIODIC.
.  stencil_type - stencil type.  Use either DMDA_STENCIL_BOX or DMDA_STENCIL_STAR.
.  M,N - global dimension in each direction of the array (use -M and or -N to indicate that it may be set to a different value
            from the command line with -da_grid_x <M> -da_grid_y <N>)
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
.  da - the resulting distributed array object

   Options Database Key:
+  -da_view - Calls DMView() at the conclusion of DMDACreate2d()
.  -da_grid_x <nx> - number of grid points in x direction, if M < 0
.  -da_grid_y <ny> - number of grid points in y direction, if N < 0
.  -da_processors_x <nx> - number of processors in x direction
.  -da_processors_y <ny> - number of processors in y direction
.  -da_refine_x <rx> - refinement ratio in x direction
.  -da_refine_y <ry> - refinement ratio in y direction
-  -da_refine <n> - refine the DMDA n times before creating, if M or N < 0


   Level: beginner

   Notes:
   The stencil type DMDA_STENCIL_STAR with width 1 corresponds to the
   standard 5-pt stencil, while DMDA_STENCIL_BOX with width 1 denotes
   the standard 9-pt stencil.

   The array data itself is NOT stored in the DMDA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DMCreateGlobalVector()
   and DMCreateLocalVector() and calls to VecDuplicate() if more are needed.

.keywords: distributed array, create, two-dimensional

.seealso: DMDestroy(), DMView(), DMDACreate1d(), DMDACreate3d(), DMGlobalToLocalBegin(), DMDAGetRefinementFactor(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMDALocalToLocalBegin(), DMDALocalToLocalEnd(), DMDASetRefinementFactor(),
          DMDAGetInfo(), DMCreateGlobalVector(), DMCreateLocalVector(), DMDACreateNaturalVector(), DMLoad(), DMDAGetOwnershipRanges()

@*/

PetscErrorCode  DMDACreate2d(MPI_Comm comm,DMDABoundaryType bx,DMDABoundaryType by,DMDAStencilType stencil_type,
                          PetscInt M,PetscInt N,PetscInt m,PetscInt n,PetscInt dof,PetscInt s,const PetscInt lx[],const PetscInt ly[],DM *da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreate(comm, da);CHKERRQ(ierr);
  ierr = DMDASetDim(*da, 2);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, N, 1);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*da, m, n, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*da, bx, by, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*da, stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*da, lx, ly, PETSC_NULL);CHKERRQ(ierr);
  /* This violates the behavior for other classes, but right now users expect negative dimensions to be handled this way */
  ierr = DMSetFromOptions(*da);CHKERRQ(ierr);
  ierr = DMSetUp(*da);CHKERRQ(ierr);
  ierr = DMView_DA_Private(*da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
