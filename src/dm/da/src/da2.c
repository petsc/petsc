#define PETSCDM_DLL
 
#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAView_2d"
PetscErrorCode DAView_2d(DA da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscTruth     iascii,isdraw;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)da)->comm,&rank);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_VTK && format != PETSC_VIEWER_ASCII_VTK_CELL) {
      DALocalInfo info;
      ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D N %D m %D n %D w %D s %D\n",rank,da->M,
                                                da->N,da->m,da->n,da->w,da->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D, Y range of indices: %D %D\n",info.xs,info.xs+info.xm,info.ys,info.ys+info.ym);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  } else if (isdraw) {
    PetscDraw       draw;
    double     ymin = -1*da->s-1,ymax = da->N+da->s;
    double     xmin = -1*da->s-1,xmax = da->M+da->s;
    double     x,y;
    PetscInt   base,*idx;
    char       node[10];
    PetscTruth isnull;
 
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
    if (!da->coordinates) {
      ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    }
    ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draw all node lines */
    if (!rank) {
      ymin = 0.0; ymax = da->N - 1;
      for (xmin=0; xmin<da->M; xmin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
      xmin = 0.0; xmax = da->M - 1;
      for (ymin=0; ymin<da->N; ymin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = da->ys; ymax = da->ye - 1; xmin = da->xs/da->w; 
    xmax =(da->xe-1)/da->w;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);

    /* put in numbers */
    base = (da->base)/da->w;
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

    base = 0; idx = da->idx;
    ymin = da->Ys; ymax = da->Ye; xmin = da->Xs; xmax = da->Xe;
    for (y=ymin; y<ymax; y++) {
      for (x=xmin; x<xmax; x++) {
        if ((base % da->w) == 0) {
          sprintf(node,"%d",(int)(idx[base]/da->w));
          ierr = PetscDrawString(draw,x/da->w,y,PETSC_DRAW_BLUE,node);CHKERRQ(ierr);
        }
        base++;
      }
    }        
    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for DA2d",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
      M is number of grid points 
      m is number of processors

*/
#undef __FUNCT__  
#define __FUNCT__ "DASplitComm2d"
PetscErrorCode PETSCDM_DLLEXPORT DASplitComm2d(MPI_Comm comm,PetscInt M,PetscInt N,PetscInt sw,MPI_Comm *outcomm)
{
  PetscErrorCode ierr;
  PetscInt       m,n = 0,x = 0,y = 0;
  PetscMPIInt    size,csize,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  csize = 4*size;
  do {
    if (csize % 4) SETERRQ4(PETSC_ERR_ARG_INCOMP,"Cannot split communicator of size %d tried %d %D %D",size,csize,x,y);
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
    ierr = PetscInfo1(0,"DASplitComm2d:Creating redundant coarse problems of size %d\n",csize);CHKERRQ(ierr);
  } else {
    *outcomm = comm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetElements_2d_P1"
PetscErrorCode DAGetElements_2d_P1(DA da,PetscInt *n,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscInt       i,j,cnt,xs,xe = da->xe,ys,ye = da->ye,Xs = da->Xs, Xe = da->Xe, Ys = da->Ys;

  PetscFunctionBegin;
  if (!da->e) {
    if (da->xs == Xs) xs = da->xs; else xs = da->xs - 1;
    if (da->ys == Ys) ys = da->ys; else ys = da->ys - 1;
    da->ne = 2*(xe - xs - 1)*(ye - ys - 1);
    ierr   = PetscMalloc((1 + 3*da->ne)*sizeof(PetscInt),&da->e);CHKERRQ(ierr);
    cnt    = 0;
    for (j=ys; j<ye-1; j++) {
      for (i=xs; i<xe-1; i++) {
        da->e[cnt]   = i - Xs + (j - Ys)*(Xe - Xs);
        da->e[cnt+1] = i - Xs + 1 + (j - Ys)*(Xe - Xs);
        da->e[cnt+2] = i - Xs + (j - Ys + 1)*(Xe - Xs);

        da->e[cnt+3] = i - Xs + 1 + (j - Ys + 1)*(Xe - Xs);
        da->e[cnt+4] = i - Xs + (j - Ys + 1)*(Xe - Xs);
        da->e[cnt+5] = i - Xs + 1 + (j - Ys)*(Xe - Xs);
        cnt += 6;
      }
    }
  }
  *n = da->ne;
  *e = da->e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalFunction"
/*@C
       DASetLocalFunction - Caches in a DA a local function. 

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  lf - the local function

   Level: intermediate

   Notes: The routine SNESDAFormFunction() uses this the cached function to evaluate the user provided function.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunctioni()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetLocalFunction(DA da,DALocalFunction1 lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->lf    = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalFunctioni"
/*@C
       DASetLocalFunctioni - Caches in a DA a local function that evaluates a single component

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  lfi - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetLocalFunctioni(DA da,PetscErrorCode (*lfi)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->lfi = lfi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalFunctionib"
/*@C
       DASetLocalFunctionib - Caches in a DA a block local function that evaluates a single component

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  lfi - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetLocalFunctionib(DA da,PetscErrorCode (*lfi)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->lfib = lfi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicFunction_Private"
PetscErrorCode DASetLocalAdicFunction_Private(DA da,DALocalFunction1 ad_lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->adic_lf = ad_lf;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicFunctioni - Caches in a DA a local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DASetLocalAdicFunctioni(DA da,PetscInt (ad_lf*)(DALocalInfo*,MatStencil*,void*,void*,void*)
   
   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  ad_lfi - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian(), DASetLocalFunctioni()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicFunctioni_Private"
PetscErrorCode DASetLocalAdicFunctioni_Private(DA da,PetscErrorCode (*ad_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->adic_lfi = ad_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicMFFunctioni - Caches in a DA a local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode  DASetLocalAdicFunctioni(DA da,int (ad_lf*)(DALocalInfo*,MatStencil*,void*,void*,void*)
   
   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  admf_lfi - the local matrix-free function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian(), DASetLocalFunctioni()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicMFFunctioni_Private"
PetscErrorCode DASetLocalAdicMFFunctioni_Private(DA da,PetscErrorCode (*admf_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->adicmf_lfi = admf_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicFunctionib - Caches in a DA a block local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DASetLocalAdicFunctionib(DA da,PetscInt (ad_lf*)(DALocalInfo*,MatStencil*,void*,void*,void*)
   
   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  ad_lfi - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian(), DASetLocalFunctionib()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicFunctionib_Private"
PetscErrorCode DASetLocalAdicFunctionib_Private(DA da,PetscErrorCode (*ad_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->adic_lfib = ad_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicMFFunctionib - Caches in a DA a block local functioni computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode  DASetLocalAdicFunctionib(DA da,int (ad_lf*)(DALocalInfo*,MatStencil*,void*,void*,void*)

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  admf_lfi - the local matrix-free function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian(), DASetLocalFunctionib()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicMFFunctionib_Private"
PetscErrorCode DASetLocalAdicMFFunctionib_Private(DA da,PetscErrorCode (*admf_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->adicmf_lfib = admf_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicMFFunction - Caches in a DA a local function computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DASetLocalAdicMFFunction(DA da,DALocalFunction1 ad_lf)

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  ad_lf - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicMFFunction_Private"
PetscErrorCode DASetLocalAdicMFFunction_Private(DA da,DALocalFunction1 ad_lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->adicmf_lf = ad_lf;
  PetscFunctionReturn(0);
}

/*@C
       DASetLocalJacobian - Caches in a DA a local Jacobian

   Collective on DA

   
   Input Parameter:
+  da - initial distributed array
-  lj - the local Jacobian

   Level: intermediate

   Notes: The routine SNESDAFormFunction() uses this the cached function to evaluate the user provided function.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction()
@*/
#undef __FUNCT__  
#define __FUNCT__ "DASetLocalJacobian"
PetscErrorCode PETSCDM_DLLEXPORT DASetLocalJacobian(DA da,DALocalFunction1 lj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->lj    = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalFunction"
/*@C
       DAGetLocalFunction - Gets from a DA a local function and its ADIC/ADIFOR Jacobian

   Collective on DA

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  lf - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalJacobian(), DASetLocalFunction()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetLocalFunction(DA da,DALocalFunction1 *lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (lf)       *lf = da->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalJacobian"
/*@C
       DAGetLocalJacobian - Gets from a DA a local jacobian

   Collective on DA

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  lj - the local jacobian

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalJacobian()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetLocalJacobian(DA da,DALocalFunction1 *lj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (lj) *lj = da->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunction"
/*@
    DAFormFunction - Evaluates a user provided function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector
.    vfu - output vector 
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

           This should eventually replace DAFormFunction1

    Level: advanced

.seealso: DAComputeJacobian1WithAdic()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunction(DA da,PetscErrorCode (*lf)(void),Vec vu,Vec vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u,*fu;
  DALocalInfo    info;
  PetscErrorCode (*f)(DALocalInfo*,void*,void*,void*) = (PetscErrorCode (*)(DALocalInfo*,void*,void*,void*))lf;
  
  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu,&fu);CHKERRQ(ierr);

  ierr = (*f)(&info,u,fu,w);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(pierr);
    pierr = DAVecRestoreArray(da,vfu,&fu);CHKERRQ(pierr);
  }
 CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu,&fu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctionLocal"
/*@C 
   DAFormFunctionLocal - This is a universal function evaluation routine for
   a local DA function.

   Collective on DA

   Input Parameters:
+  da - the DA context
.  func - The local function
.  X - input vector
.  F - function vector
-  ctx - A user context

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), DASetLocalAdicFunction(), DASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunctionLocal(DA da, DALocalFunction1 func, Vec X, Vec F, void *ctx)
{
  Vec            localX;
  DALocalInfo    info;
  void          *u;
  void          *fu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,F,&fu);CHKERRQ(ierr);
  ierr = (*func)(&info,u,fu,ctx);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DAVecRestoreArray(da,localX,&u);CHKERRQ(pierr);
    pierr = DAVecRestoreArray(da,F,&fu);CHKERRQ(pierr);
  }
 CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,F,&fu);CHKERRQ(ierr);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DARestoreLocalVector(da,&localX);CHKERRQ(pierr);
  }
 CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctionLocalGhost"
/*@C 
   DAFormFunctionLocalGhost - This is a universal function evaluation routine for
   a local DA function, but the ghost values of the output are communicated and added.

   Collective on DA

   Input Parameters:
+  da - the DA context
.  func - The local function
.  X - input vector
.  F - function vector
-  ctx - A user context

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), DASetLocalAdicFunction(), DASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunctionLocalGhost(DA da, DALocalFunction1 func, Vec X, Vec F, void *ctx)
{
  Vec            localX, localF;
  DALocalInfo    info;
  void          *u;
  void          *fu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da,&localF);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,localF,&fu);CHKERRQ(ierr);
  ierr = (*func)(&info,u,fu,ctx);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DAVecRestoreArray(da,localX,&u);CHKERRQ(pierr);
    pierr = DAVecRestoreArray(da,localF,&fu);CHKERRQ(pierr);
  }
  CHKERRQ(ierr);
  ierr = DALocalToGlobalBegin(da,localF,F);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(da,localF,F);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,localF,&fu);CHKERRQ(ierr);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DARestoreLocalVector(da,&localX);CHKERRQ(pierr);
  ierr = DARestoreLocalVector(da,&localF);CHKERRQ(ierr);
  }
  CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunction1"
/*@
    DAFormFunction1 - Evaluates a user provided function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector
.    vfu - output vector 
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAComputeJacobian1WithAdic()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunction1(DA da,Vec vu,Vec vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u,*fu;
  DALocalInfo    info;
  
  PetscFunctionBegin;

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu,&fu);CHKERRQ(ierr);

  CHKMEMQ;
  ierr = (*da->lf)(&info,u,fu,w);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(pierr);
    pierr = DAVecRestoreArray(da,vfu,&fu);CHKERRQ(pierr);
  }
  CHKERRQ(ierr);
  CHKMEMQ;

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu,&fu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctioniTest1"
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunctioniTest1(DA da,void *w)
{
  Vec            vu,fu,fui;
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *ui;
  PetscRandom    rnd;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = VecSetRandom(vu,rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rnd);CHKERRQ(ierr);

  ierr = DAGetGlobalVector(da,&fu);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da,&fui);CHKERRQ(ierr);
  
  ierr = DAFormFunction1(da,vu,fu,w);CHKERRQ(ierr);

  ierr = VecGetArray(fui,&ui);CHKERRQ(ierr);
  ierr = VecGetLocalSize(fui,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = DAFormFunctioni1(da,i,vu,ui+i,w);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(fui,&ui);CHKERRQ(ierr);

  ierr = VecAXPY(fui,-1.0,fu);CHKERRQ(ierr);
  ierr = VecNorm(fui,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(((PetscObject)da)->comm,"Norm of difference in vectors %G\n",norm);CHKERRQ(ierr);
  ierr = VecView(fu,0);CHKERRQ(ierr);
  ierr = VecView(fui,0);CHKERRQ(ierr);

  ierr = DARestoreLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&fu);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&fui);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctioni1"
/*@
    DAFormFunctioni1 - Evaluates a user provided point-wise function

   Input Parameters:
+    da - the DA that defines the grid
.    i - the component of the function we wish to compute (must be local)
.    vu - input vector
.    vfu - output value
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAComputeJacobian1WithAdic()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunctioni1(DA da,PetscInt i,Vec vu,PetscScalar *vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u;
  DALocalInfo    info;
  MatStencil     stencil;
  
  PetscFunctionBegin;

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);

  /* figure out stencil value from i */
  stencil.c = i % info.dof;
  stencil.i = (i % (info.xm*info.dof))/info.dof;
  stencil.j = (i % (info.xm*info.ym*info.dof))/(info.xm*info.dof);
  stencil.k = i/(info.xm*info.ym*info.dof);

  ierr = (*da->lfi)(&info,&stencil,u,vfu,w);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctionib1"
/*@
    DAFormFunctionib1 - Evaluates a user provided point-block function

   Input Parameters:
+    da - the DA that defines the grid
.    i - the component of the function we wish to compute (must be local)
.    vu - input vector
.    vfu - output value
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAComputeJacobian1WithAdic()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormFunctionib1(DA da,PetscInt i,Vec vu,PetscScalar *vfu,void *w)
{
  PetscErrorCode ierr;
  void           *u;
  DALocalInfo    info;
  MatStencil     stencil;
  
  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);

  /* figure out stencil value from i */
  stencil.c = i % info.dof;
  if (stencil.c) SETERRQ(PETSC_ERR_ARG_WRONG,"Point-block functions can only be called for the entire block");
  stencil.i = (i % (info.xm*info.dof))/info.dof;
  stencil.j = (i % (info.xm*info.ym*info.dof))/(info.xm*info.dof);
  stencil.k = i/(info.xm*info.ym*info.dof);

  ierr = (*da->lfib)(&info,&stencil,u,vfu,w);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(new)
#undef __FUNCT__  
#define __FUNCT__ "DAGetDiagonal_MFFD"
/*
  DAGetDiagonal_MFFD - Gets the diagonal for a matrix free matrix where local
    function lives on a DA

        y ~= (F(u + ha) - F(u))/h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
PetscErrorCode DAGetDiagonal_MFFD(DA da,Vec U,Vec a)
{
  PetscScalar    h,*aa,*ww,v;
  PetscReal      epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  PetscErrorCode ierr;
  PetscInt       gI,nI;
  MatStencil     stencil;
  DALocalInfo    info;
 
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
    
    ww[gI += h;
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
#include "adic/ad_utils.h"
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1WithAdic"
/*@C
    DAComputeJacobian1WithAdic - Evaluates a adiC provided Jacobian function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

   Level: advanced

    Notes: Does NOT do ghost updates on vu upon entry

.seealso: DAFormFunction1()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAComputeJacobian1WithAdic(DA da,Vec vu,Mat J,void *w)
{
  PetscErrorCode ierr;
  PetscInt       gtdof,tdof;
  PetscScalar    *ustart;
  DALocalInfo    info;
  void           *ad_u,*ad_f,*ad_ustart,*ad_fstart;
  ISColoring     iscoloring;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  PetscADResetIndep();

  /* get space for derivative objects.  */
  ierr = DAGetAdicArray(da,PETSC_TRUE,(void **)&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DAGetAdicArray(da,PETSC_FALSE,(void **)&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  ierr = VecGetArray(vu,&ustart);CHKERRQ(ierr);
  ierr = DAGetColoring(da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);

  PetscADSetValueAndColor(ad_ustart,gtdof,iscoloring->colors,ustart);

  ierr = VecRestoreArray(vu,&ustart);CHKERRQ(ierr);
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
  ierr = PetscADIncrementTotalGradSize(iscoloring->n);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = PetscLogEventBegin(DA_LocalADFunction,0,0,0,0);CHKERRQ(ierr);
  ierr = (*da->adic_lf)(&info,ad_u,ad_f,w);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DA_LocalADFunction,0,0,0,0);CHKERRQ(ierr);

  /* stick the values into the matrix */
  ierr = MatSetValuesAdic(J,(PetscScalar**)ad_fstart);CHKERRQ(ierr);

  /* return space for derivative objects.  */
  ierr = DARestoreAdicArray(da,PETSC_TRUE,(void **)&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DARestoreAdicArray(da,PETSC_FALSE,(void **)&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAdic"
/*@C
    DAMultiplyByJacobian1WithAdic - Applies an ADIC-provided Jacobian function to a vector on 
    each processor that shares a DA.

    Input Parameters:
+   da - the DA that defines the grid
.   vu - Jacobian is computed at this point (ghosted)
.   v - product is done on this vector (ghosted)
.   fu - output vector = J(vu)*v (not ghosted)
-   w - any user data

    Notes: 
    This routine does NOT do ghost updates on vu upon entry.

   Level: advanced

.seealso: DAFormFunction1()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAMultiplyByJacobian1WithAdic(DA da,Vec vu,Vec v,Vec f,void *w)
{
  PetscErrorCode ierr;
  PetscInt       i,gtdof,tdof;
  PetscScalar    *avu,*av,*af,*ad_vustart,*ad_fstart;
  DALocalInfo    info;
  void           *ad_vu,*ad_f;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* get space for derivative objects.  */
  ierr = DAGetAdicMFArray(da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = DAGetAdicMFArray(da,PETSC_FALSE,(void **)&ad_f,(void**)&ad_fstart,&tdof);CHKERRQ(ierr);

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

  ierr = (*da->adicmf_lf)(&info,ad_vu,ad_f,w);CHKERRQ(ierr);

  /* stick the values into the vector */
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);  
  for (i=0; i<tdof; i++) {
    af[i] = ad_fstart[2*i+1];
  }
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);  

  /* return space for derivative objects.  */
  ierr = DARestoreAdicMFArray(da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(da,PETSC_FALSE,(void **)&ad_f,(void**)&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1"
/*@
    DAComputeJacobian1 - Evaluates a local Jacobian function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAFormFunction1()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAComputeJacobian1(DA da,Vec vu,Mat J,void *w)
{
  PetscErrorCode ierr;
  void           *u;
  DALocalInfo    info;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = (*da->lj)(&info,u,J,w);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1WithAdifor"
/*
    DAComputeJacobian1WithAdifor - Evaluates a ADIFOR provided Jacobian local function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

.seealso: DAFormFunction1()

*/
PetscErrorCode PETSCDM_DLLEXPORT DAComputeJacobian1WithAdifor(DA da,Vec vu,Mat J,void *w)
{
  PetscErrorCode  ierr;
  PetscInt        i,Nc,N;
  ISColoringValue *color;
  DALocalInfo     info;
  PetscScalar     *u,*g_u,*g_f,*f = 0,*p_u;
  ISColoring      iscoloring;
  void            (*lf)(PetscInt*,DALocalInfo*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*) = 
                  (void (*)(PetscInt*,DALocalInfo*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*))*da->adifor_lf;

  PetscFunctionBegin;
  ierr = DAGetColoring(da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);
  Nc   = iscoloring->n;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
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
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
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
#define __FUNCT__ "DAFormJacobianLocal"
/*@C 
   DAFormjacobianLocal - This is a universal Jacobian evaluation routine for
   a local DA function.

   Collective on DA

   Input Parameters:
+  da - the DA context
.  func - The local function
.  X - input vector
.  J - Jacobian matrix
-  ctx - A user context

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), DASetLocalAdicFunction(), DASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAFormJacobianLocal(DA da, DALocalFunction1 func, Vec X, Mat J, void *ctx)
{
  Vec            localX;
  DALocalInfo    info;
  void          *u;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = (*func)(&info,u,J,ctx);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DAVecRestoreArray(da,localX,&u);CHKERRQ(pierr);
  }
  CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DARestoreLocalVector(da,&localX);CHKERRQ(pierr);
  }
  CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAD"
/*@C
    DAMultiplyByJacobian1WithAD - Applies a Jacobian function supplied by ADIFOR or ADIC
    to a vector on each processor that shares a DA.

   Input Parameters:
+    da - the DA that defines the grid
.    vu - Jacobian is computed at this point (ghosted)
.    v - product is done on this vector (ghosted)
.    fu - output vector = J(vu)*v (not ghosted)
-    w - any user data

    Notes: 
    This routine does NOT do ghost updates on vu and v upon entry.
           
    Automatically calls DAMultiplyByJacobian1WithAdifor() or DAMultiplyByJacobian1WithAdic()
    depending on whether DASetLocalAdicMFFunction() or DASetLocalAdiforMFFunction() was called.

   Level: advanced

.seealso: DAFormFunction1(), DAMultiplyByJacobian1WithAdifor(), DAMultiplyByJacobian1WithAdic()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAMultiplyByJacobian1WithAD(DA da,Vec u,Vec v,Vec f,void *w)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (da->adicmf_lf) {
#if defined(PETSC_HAVE_ADIC)
    ierr = DAMultiplyByJacobian1WithAdic(da,u,v,f,w);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_ERR_SUP_SYS,"Requires ADIC to be installed and cannot use complex numbers");
#endif
  } else if (da->adiformf_lf) {
    ierr = DAMultiplyByJacobian1WithAdifor(da,u,v,f,w);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ORDER,"Must call DASetLocalAdiforMFFunction() or DASetLocalAdicMFFunction() before using");
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAdifor"
/*@C
    DAMultiplyByJacobian1WithAdifor - Applies a ADIFOR provided Jacobian function on each processor that 
        share a DA to a vector

   Input Parameters:
+    da - the DA that defines the grid
.    vu - Jacobian is computed at this point (ghosted)
.    v - product is done on this vector (ghosted)
.    fu - output vector = J(vu)*v (not ghosted)
-    w - any user data

    Notes: Does NOT do ghost updates on vu and v upon entry

   Level: advanced

.seealso: DAFormFunction1()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAMultiplyByJacobian1WithAdifor(DA da,Vec u,Vec v,Vec f,void *w)
{
  PetscErrorCode ierr;
  PetscScalar    *au,*av,*af,*awork;
  Vec            work;
  DALocalInfo    info;
  void           (*lf)(DALocalInfo*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*) = 
                 (void (*)(DALocalInfo*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))*da->adiformf_lf;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = DAGetGlobalVector(da,&work);CHKERRQ(ierr); 
  ierr = VecGetArray(u,&au);CHKERRQ(ierr);
  ierr = VecGetArray(v,&av);CHKERRQ(ierr);
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);
  ierr = VecGetArray(work,&awork);CHKERRQ(ierr);
  (lf)(&info,au,av,awork,af,w,&ierr);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&au);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&av);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);
  ierr = VecRestoreArray(work,&awork);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&work);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "DACreate_2D"
PetscErrorCode PETSCDM_DLLEXPORT DACreate_2D(DA da)
{
  const PetscInt       dim          = da->dim;
  const PetscInt       M            = da->M;
  const PetscInt       N            = da->N;
  PetscInt             m            = da->m;
  PetscInt             n            = da->n;
  const PetscInt       dof          = da->w;
  const PetscInt       s            = da->s;
  const DAPeriodicType wrap         = da->wrap;
  const DAStencilType  stencil_type = da->stencil_type;
  PetscInt            *lx           = da->lx;
  PetscInt            *ly           = da->ly;
  MPI_Comm             comm;
  PetscMPIInt    rank,size;
  PetscInt       xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,start,end;
  PetscInt       up,down,left,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn;
  PetscInt       xbase,*bases,*ldims,j,x_t,y_t,s_t,base,count;
  PetscInt       s_x,s_y; /* s proportionalized to w */
  PetscInt       sn0 = 0,sn2 = 0,sn6 = 0,sn8 = 0;
  Vec            local,global;
  VecScatter     ltog,gtol;
  IS             to,from;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  if (dim != PETSC_DECIDE && dim != 2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Dimension should be 2: %D",dim);
  if (dof < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dof);
  if (s < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",s);

  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  da->ops->getelements = DAGetElements_2d_P1;

  da->dim         = 2;
  da->elementtype = DA_ELEMENT_P1;
  ierr = PetscMalloc(dof*sizeof(char*),&da->fieldname);CHKERRQ(ierr);
  ierr = PetscMemzero(da->fieldname,dof*sizeof(char*));CHKERRQ(ierr);

  if (m != PETSC_DECIDE) {
    if (m < 1) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in X direction: %D",m);}
    else if (m > size) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in X direction: %D %d",m,size);}
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Y direction: %D",n);}
    else if (n > size) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Y direction: %D %d",n,size);}
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
    if (m*n != size) SETERRQ(PETSC_ERR_PLIB,"Unable to create partition, check the size of the communicator and input m and n ");
  } else if (m*n != size) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition"); 

  if (M < m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %D %D",M,m);
  if (N < n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %D %D",N,n);

  /* 
     Determine locally owned region 
     xs is the first local node number, x is the number of local nodes 
  */
  if (!lx) {
    ierr = PetscMalloc(m*sizeof(PetscInt), &da->lx);CHKERRQ(ierr);
    lx = da->lx;
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
  if (left != M) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Sum of lx across processors not equal to M: %D %D",left,M);
  }
#endif

  /* 
     Determine locally owned region 
     ys is the first local node number, y is the number of local nodes 
  */
  if (!ly) {
    ierr = PetscMalloc(n*sizeof(PetscInt), &da->ly);CHKERRQ(ierr);
    ly = da->ly;
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
  if (left != N) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Sum of ly across processors not equal to N: %D %D",left,N);
  }
#endif

  if (x < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local x-width of domain x %D is smaller than stencil width s %D",x,s);
  if (y < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local y-width of domain y %D is smaller than stencil width s %D",y,s);
  xe = xs + x;
  ye = ys + y;

  /* determine ghost region */
  /* Assume No Periodicity */
  if (xs-s > 0) Xs = xs - s; else Xs = 0; 
  if (ys-s > 0) Ys = ys - s; else Ys = 0; 
  if (xe+s <= M) Xe = xe + s; else Xe = M; 
  if (ye+s <= N) Ye = ye + s; else Ye = N;

  /* X Periodic */
  if (DAXPeriodic(wrap)){
    Xs = xs - s; 
    Xe = xe + s; 
  }

  /* Y Periodic */
  if (DAYPeriodic(wrap)){
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

  /* allocate the base parallel and sequential vectors */
  da->Nlocal = x*y;
  ierr = VecCreateMPIWithArray(comm,da->Nlocal,PETSC_DECIDE,0,&global);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,dof);CHKERRQ(ierr);
  da->nlocal = (Xe-Xs)*(Ye-Ys);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,da->nlocal,0,&local);CHKERRQ(ierr);
  ierr = VecSetBlockSize(local,dof);CHKERRQ(ierr);


  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x*y,start,1,&to);CHKERRQ(ierr);

  left  = xs - Xs; down  = ys - Ys; up    = down + y;
  ierr = PetscMalloc(x*(up - down)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  count = 0;
  for (i=down; i<up; i++) {
    for (j=0; j<x/dof; j++) {
      idx[count++] = left + i*(Xe-Xs) + j*dof;
    }
  }
  ierr = ISCreateBlock(comm,dof,count,idx,&from);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);

  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,ltog);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);

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
    ierr  = PetscMalloc(count*sizeof(PetscInt)/dof,&idx);CHKERRQ(ierr);
    count = 0;
    for (i=0; i<down; i++) {
      for (j=0; j<xe-xs; j += dof) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    /* middle */
    for (i=down; i<up; i++) {
      for (j=0; j<Xe-Xs; j += dof) {
        idx[count++] = i*(Xe-Xs) + j;
      }
    }
    /* top */
    for (i=up; i<Ye-Ys; i++) {
      for (j=0; j<xe-xs; j += dof) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,&to);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr);
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
  ierr = PetscMalloc(9*sizeof(PetscInt),&da->neighbors);CHKERRQ(ierr);
  da->neighbors[0] = n0;
  da->neighbors[1] = n1;
  da->neighbors[2] = n2;
  da->neighbors[3] = n3;
  da->neighbors[4] = rank;
  da->neighbors[5] = n5;
  da->neighbors[6] = n6;
  da->neighbors[7] = n7;
  da->neighbors[8] = n8;

  if (stencil_type == DA_STENCIL_STAR) {
    /* save corner processor numbers */
    sn0 = n0; sn2 = n2; sn6 = n6; sn8 = n8; 
    n0 = n2 = n6 = n8 = -1;
  }

  ierr = PetscMalloc((x+2*s_x)*(y+2*s_y)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(da,(x+2*s_x)*(y+2*s_y)*sizeof(PetscInt));CHKERRQ(ierr);
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
      /* y_t = y; */
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }

    for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = lx[n5 % m]*dof;
      /* y_t = y; */
      s_t = bases[n5] + (i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=1; i<=s_y; i++) {
    if (n6 >= 0) { /* left above */
      x_t = lx[n6 % m]*dof;
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
      x_t = lx[n8 % m]*dof;
      /* y_t = ly[(n8/m)]; */
      s_t = bases[n8] + (i-1)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  base = bases[rank];
  {
    PetscInt nnn = nn/dof,*iidx;
    ierr = PetscMalloc(nnn*sizeof(PetscInt),&iidx);CHKERRQ(ierr);
    for (i=0; i<nnn; i++) {
      iidx[i] = idx[dof*i];
    }
    ierr = ISCreateBlock(comm,dof,nnn,iidx,&from);CHKERRQ(ierr);
    ierr = PetscFree(iidx);CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,gtol);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);

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
        /* y_t = y; */
        s_t = bases[n3] + (i+1)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }

      for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

      if (n5 >= 0) { /* directly right */
        x_t = lx[n5 % m]*dof;
        /* y_t = y; */
        s_t = bases[n5] + (i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n6 >= 0) { /* left above */
        x_t = lx[n6 % m]*dof;
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
        x_t = lx[n8 % m]*dof;
        /* y_t = ly[(n8/m)]; */
        s_t = bases[n8] + (i-1)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }
  }
  ierr = PetscFree2(bases,ldims);CHKERRQ(ierr); 

  da->m  = m;  da->n  = n;
  da->xs = xs; da->xe = xe; da->ys = ys; da->ye = ye; da->zs = 0; da->ze = 1;
  da->Xs = Xs; da->Xe = Xe; da->Ys = Ys; da->Ye = Ye; da->Zs = 0; da->Ze = 1;

  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);

  da->gtol      = gtol;
  da->ltog      = ltog;
  da->idx       = idx;
  da->Nl        = nn;
  da->base      = base;
  da->ops->view = DAView_2d;

  /* 
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  ierr = ISLocalToGlobalMappingCreateNC(comm,nn,idx,&da->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(da->ltogmap,da->w,&da->ltogmapb);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,da->ltogmap);CHKERRQ(ierr);

  da->ltol = PETSC_NULL;
  da->ao   = PETSC_NULL;

  ierr = PetscPublishAll(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "DACreate2d"
/*@C
   DACreate2d -  Creates an object that will manage the communication of  two-dimensional 
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  wrap - type of periodicity should the array have. 
         Use one of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, or DA_XYPERIODIC.
.  stencil_type - stencil type.  Use either DA_STENCIL_BOX or DA_STENCIL_STAR.
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
+  -da_view - Calls DAView() at the conclusion of DACreate2d()
.  -da_grid_x <nx> - number of grid points in x direction, if M < 0
.  -da_grid_y <ny> - number of grid points in y direction, if N < 0
.  -da_processors_x <nx> - number of processors in x direction
.  -da_processors_y <ny> - number of processors in y direction
.  -da_refine_x - refinement ratio in x direction
-  -da_refine_y - refinement ratio in y direction

   Level: beginner

   Notes:
   The stencil type DA_STENCIL_STAR with width 1 corresponds to the 
   standard 5-pt stencil, while DA_STENCIL_BOX with width 1 denotes
   the standard 9-pt stencil.

   The array data itself is NOT stored in the DA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DACreateGlobalVector()
   and DACreateLocalVector() and calls to VecDuplicate() if more are needed.

.keywords: distributed array, create, two-dimensional

.seealso: DADestroy(), DAView(), DACreate1d(), DACreate3d(), DAGlobalToLocalBegin(), DAGetRefinementFactor(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DALocalToLocalBegin(), DALocalToLocalEnd(), DASetRefinementFactor(),
          DAGetInfo(), DACreateGlobalVector(), DACreateLocalVector(), DACreateNaturalVector(), DALoad(), DAView(), DAGetOwnershipRanges()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreate2d(MPI_Comm comm,DAPeriodicType wrap,DAStencilType stencil_type,
                          PetscInt M,PetscInt N,PetscInt m,PetscInt n,PetscInt dof,PetscInt s,const PetscInt lx[],const PetscInt ly[],DA *da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DACreate(comm, da);CHKERRQ(ierr);
  ierr = DASetDim(*da, 2);CHKERRQ(ierr);
  ierr = DASetSizes(*da, M, N, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DASetNumProcs(*da, m, n, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DASetPeriodicity(*da, wrap);CHKERRQ(ierr);
  ierr = DASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DASetStencilType(*da, stencil_type);CHKERRQ(ierr);
  ierr = DASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DASetVertexDivision(*da, lx, ly, PETSC_NULL);CHKERRQ(ierr);
  /* This violates the behavior for other classes, but right now users expect negative dimensions to be handled this way */
  ierr = DASetFromOptions(*da);CHKERRQ(ierr);
  ierr = DASetType(*da, DA2D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
