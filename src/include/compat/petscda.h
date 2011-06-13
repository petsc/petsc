#ifndef _COMPAT_PETSC_DA_H
#define _COMPAT_PETSC_DA_H

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#include <petscda.h>
#endif

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#define DASetOwnershipRanges           DASetVertexDivision
#define DAGetLocalToGlobalMapping      DAGetISLocalToGlobalMapping
#define DAGetLocalToGlobalMappingBlock DAGetISLocalToGlobalMappingBlck
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,1,0)

#include "private/daimpl.h"
#undef DAGetElements
#undef DARestoreElements

#undef __FUNCT__
#define __FUNCT__ "DASetElementType"
static PetscErrorCode
DASetElementType_Compat(DA da, DAElementType etype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (da->elementtype != etype) {
    ierr = PetscFree(da->e);CHKERRQ(ierr);
    da->elementtype = etype;
    da->ne          = 0;
    da->e           = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}
#define DASetElementType DASetElementType_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetElementType"
static PetscErrorCode
DAGetElementType_Compat(DA da, DAElementType *etype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(etype,2);
  *etype = da->elementtype;
  PetscFunctionReturn(0);
}
#define DAGetElementType DAGetElementType_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetElements_1D"
static PetscErrorCode
DAGetElements_1D(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xe,Xs,Xe;
  PetscInt       cnt=0;
  PetscFunctionBegin;
  if (!da->e) {
    ierr = DAGetCorners(da,&xs,0,0,&xe,0,0);CHKERRQ(ierr);
    ierr = DAGetGhostCorners(da,&Xs,0,0,&Xe,0,0);CHKERRQ(ierr);
    xe += xs; Xe += Xs; if (xs != Xs) xs -= 1;
    da->ne = 1*(xe - xs - 1);
    ierr = PetscMalloc((1 + 2*da->ne)*sizeof(PetscInt),&da->e);CHKERRQ(ierr);
    for (i=xs; i<xe-1; i++) {
      da->e[cnt++] = (i-Xs  );
      da->e[cnt++] = (i-Xs+1);
    }
  }
  *nel = da->ne;
  *nen = 2;
  *e   = da->e;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DAGetElements_2D"
static PetscErrorCode
DAGetElements_2D(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xe,Xs,Xe;
  PetscInt       j,ys,ye,Ys,Ye;
  PetscInt       cnt=0, cell[4], ns=2, nn=3;
  PetscInt       c, split[] = {0,1,3,
			       2,3,1};
  PetscFunctionBegin;
  if (!da->e) {
    if (da->elementtype == DA_ELEMENT_P1) {ns=2; nn=3;}
    if (da->elementtype == DA_ELEMENT_Q1) {ns=1; nn=4;}
    ierr = DAGetCorners(da,&xs,&ys,0,&xe,&ye,0);CHKERRQ(ierr);
    ierr = DAGetGhostCorners(da,&Xs,&Ys,0,&Xe,&Ye,0);CHKERRQ(ierr);
    xe += xs; Xe += Xs; if (xs != Xs) xs -= 1;
    ye += ys; Ye += Ys; if (ys != Ys) ys -= 1;
    da->ne = ns*(xe - xs - 1)*(ye - ys - 1);
    ierr = PetscMalloc((1 + nn*da->ne)*sizeof(PetscInt),&da->e);CHKERRQ(ierr);
    for (j=ys; j<ye-1; j++) {
      for (i=xs; i<xe-1; i++) {
	cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs);
	cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs);
	cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs);
	cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs);
	if (da->elementtype == DA_ELEMENT_P1) {
	  for (c=0; c<ns*nn; c++)
	    da->e[cnt++] = cell[split[c]];
	}
	if (da->elementtype == DA_ELEMENT_Q1) {
	  for (c=0; c<ns*nn; c++)
	    da->e[cnt++] = cell[c];
	}
      }
    }
  }
  *nel = da->ne;
  *nen = nn;
  *e   = da->e;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DAGetElements_3D"
static PetscErrorCode
DAGetElements_3D(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xe,Xs,Xe;
  PetscInt       j,ys,ye,Ys,Ye;
  PetscInt       k,zs,ze,Zs,Ze;
  PetscInt       cnt=0, cell[8], ns=6, nn=4;
  PetscInt       c, split[] = {0,1,3,7,
			       0,1,7,4,
			       1,2,3,7,
			       1,2,7,6,
			       1,4,5,7,
			       1,5,6,7};
  PetscFunctionBegin;
  if (!da->e) {
    if (da->elementtype == DA_ELEMENT_P1) {ns=6; nn=4;}
    if (da->elementtype == DA_ELEMENT_Q1) {ns=1; nn=8;}
    ierr = DAGetCorners(da,&xs,&ys,&zs,&xe,&ye,&ze);CHKERRQ(ierr);
    ierr = DAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xe,&Ye,&Ze);CHKERRQ(ierr);
    xe += xs; Xe += Xs; if (xs != Xs) xs -= 1;
    ye += ys; Ye += Ys; if (ys != Ys) ys -= 1;
    ze += zs; Ze += Zs; if (zs != Zs) zs -= 1;
    da->ne = ns*(xe - xs - 1)*(ye - ys - 1)*(ze - zs - 1);
    ierr = PetscMalloc((1 + nn*da->ne)*sizeof(PetscInt),&da->e);CHKERRQ(ierr);
    for (k=zs; k<ze-1; k++) {
      for (j=ys; j<ye-1; j++) {
	for (i=xs; i<xe-1; i++) {
	  cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
	  cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
	  cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
	  cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
	  cell[4] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
	  cell[5] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
	  cell[6] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
	  cell[7] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
	  if (da->elementtype == DA_ELEMENT_P1) {
	    for (c=0; c<ns*nn; c++)
	      da->e[cnt++] = cell[split[c]];
	  }
	  if (da->elementtype == DA_ELEMENT_Q1) {
	    for (c=0; c<ns*nn; c++)
	      da->e[cnt++] = cell[c];
	  }
	}
      }
    }
  }
  *nel = da->ne;
  *nen = nn;
  *e   = da->e;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DAGetElements"
static PetscErrorCode
DAGetElements_Compat(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscInt       dim;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidIntPointer(nel,2);
  PetscValidIntPointer(nen,3);
  PetscValidPointer(e,4);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim==-1) {
    *nel = 0; *nen = 0; *e = PETSC_NULL;
  } else if (dim==1) {
    ierr = DAGetElements_1D(da,nel,nen,e);CHKERRQ(ierr);
  } else if (dim==2) {
    ierr = DAGetElements_2D(da,nel,nen,e);CHKERRQ(ierr);
  } else if (dim==3) {
    ierr = DAGetElements_3D(da,nel,nen,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define DAGetElements DAGetElements_Compat
#undef __FUNCT__
#define __FUNCT__ "DARestoreElements"
static PetscErrorCode
DARestoreElements_Compat(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidIntPointer(nel,2);
  PetscValidIntPointer(nen,3);
  PetscValidPointer(e,4);
  PetscFunctionReturn(0);
}
#define DARestoreElements DARestoreElements_Compat

#elif (PETSC_VERSION_(3,0,0))

#undef __FUNCT__
#define __FUNCT__ "DASetElementType"
static PetscErrorCode
DASetElementType_Compat(DA da, DAElementType etype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#define DASetElementType DASetElementType_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetElementType"
static PetscErrorCode
DAGetElementType_Compat(DA da, DAElementType *etype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(etype,2);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#define DAGetElementType DAGetElementType_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetElements"
static PetscErrorCode
DAGetElements_Compat(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscInt       dim;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidIntPointer(nel,2);
  PetscValidIntPointer(nen,3);
  PetscValidPointer(e,4);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  *nen = dim+1;
  ierr = DAGetElements(da,nel,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  DAGetElements
#define DAGetElements DAGetElements_Compat
#undef __FUNCT__
#define __FUNCT__ "DARestoreElements"
static PetscErrorCode
DARestoreElements_Compat(DA da,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidIntPointer(nel,2);
  PetscValidIntPointer(nen,3);
  PetscValidPointer(e,4);
  ierr = DARestoreElements(da,nel,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  DARestoreElements
#define DARestoreElements DARestoreElements_Compat
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "DASetCoordinates"
static PetscErrorCode DASetCoordinates_Compat(DA da,Vec c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidHeaderSpecific(c,VEC_COOKIE,2);
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = DASetCoordinates(da,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DASetCoordinates DASetCoordinates_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DAGetCoordinates"
static PetscErrorCode DAGetCoordinates_Compat(DA da,Vec *c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetCoordinates(da,c);CHKERRQ(ierr);
  if (*c) {ierr = PetscObjectDereference((PetscObject)*c);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetCoordinates DAGetCoordinates_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetCoordinateDA"
static PetscErrorCode DAGetCoordinateDA_Compat(DA da,DA *cda)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(da,cda);CHKERRQ(ierr);
  if (*cda) {ierr = PetscObjectDereference((PetscObject)*cda);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetCoordinateDA DAGetCoordinateDA_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetGhostedCoordinates"
static PetscErrorCode DAGetGhostedCoordinates_Compat(DA da,Vec *c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetGhostedCoordinates(da,c);CHKERRQ(ierr);
  if (*c) {ierr = PetscObjectDereference((PetscObject)*c);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetGhostedCoordinates DAGetGhostedCoordinates_Compat
#endif

#if PETSC_VERSION_(3,0,0)
#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalBoundingBox"
static PetscErrorCode DAGetLocalBoundingBox(DA da,PetscReal lmin[],PetscReal lmax[])
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#undef __FUNCT__  
#define __FUNCT__ "DAGetBoundingBox"
PetscErrorCode  DAGetBoundingBox(DA da,PetscReal gmin[],PetscReal gmax[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
typedef enum {
  DA_BOUNDARY_NONE,
  DA_BOUNDARY_GHOSTED,
  DA_BOUNDARY_MIRROR,
  DA_BOUNDARY_PERIODIC,
} DABoundaryType;
#undef __FUNCT__
#define __FUNCT__ "DA_Boundary2Periodic"
static PetscErrorCode
DA_Boundary2Periodic(PetscInt dim,
		     DABoundaryType x,DABoundaryType y,DABoundaryType z,
		     DAPeriodicType *_ptype)
{
  DAPeriodicType ptype = DA_NONPERIODIC;
  PetscFunctionBegin;
  if (dim < 3) z = DA_BOUNDARY_NONE;
  if (dim < 2) y = DA_BOUNDARY_NONE;
  /*  */ if (x==DA_BOUNDARY_NONE &&
	     y==DA_BOUNDARY_NONE &&
	     z==DA_BOUNDARY_NONE) {
    ptype = DA_NONPERIODIC;
  } else if (x==DA_BOUNDARY_GHOSTED ||
	     y==DA_BOUNDARY_GHOSTED ||
	     y==DA_BOUNDARY_GHOSTED) {
    if ((dim==1 && (x==DA_BOUNDARY_GHOSTED)) ||
	(dim==2 && (x==DA_BOUNDARY_GHOSTED &&
		    y==DA_BOUNDARY_GHOSTED)) ||
	(dim==3 && (x==DA_BOUNDARY_GHOSTED &&
		    y==DA_BOUNDARY_GHOSTED &&
		    z==DA_BOUNDARY_GHOSTED)))
      ptype = DA_XYZGHOSTED;
    else {
      SETERRQ(PETSC_ERR_SUP,
	      "Boundary type not supported in this PETSc version");
    }
  } else if(x==DA_BOUNDARY_PERIODIC &&
	    y==DA_BOUNDARY_PERIODIC &&
	    z==DA_BOUNDARY_PERIODIC) {
    ptype = DA_XYZPERIODIC;
  } else if(x==DA_BOUNDARY_PERIODIC &&
	    y==DA_BOUNDARY_PERIODIC) {
    ptype = DA_XYPERIODIC;
  } else if(x==DA_BOUNDARY_PERIODIC &&
	    z==DA_BOUNDARY_PERIODIC) {
    ptype = DA_XZPERIODIC;
  } else if(y==DA_BOUNDARY_PERIODIC &&
	    z==DA_BOUNDARY_PERIODIC) {
    ptype = DA_YZPERIODIC;
  } else if(x==DA_BOUNDARY_PERIODIC) {
    ptype = DA_XPERIODIC;
  } else if(y==DA_BOUNDARY_PERIODIC) {
    ptype = DA_YPERIODIC;
  } else if(z==DA_BOUNDARY_PERIODIC) {
    ptype = DA_ZPERIODIC;
  } else {
    SETERRQ(PETSC_ERR_SUP,"Boundary type not supported in this PETSc version");
  }
  *_ptype = ptype;
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(3,1,0)
#undef __FUNCT__
#define __FUNCT__ "DASetBoundaryType"
static PetscErrorCode
DASetBoundaryType(DA da,DABoundaryType x,DABoundaryType y,DABoundaryType z)
{
  PetscInt       dim;
  DAPeriodicType ptype = DA_NONPERIODIC;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DA_Boundary2Periodic(dim,x,y,z,&ptype);CHKERRQ(ierr);
  ierr = DASetPeriodicity(da,ptype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "DAGetInfo"
static PetscErrorCode
DAGetInfo_Compat(DA da,
		 PetscInt *dim,
		 PetscInt *M,PetscInt *N,PetscInt *P,
		 PetscInt *m,PetscInt *n,PetscInt *p,
		 PetscInt *dof,PetscInt *s,
		 DABoundaryType *btx,
		 DABoundaryType *bty,
		 DABoundaryType *btz,
		 DAStencilType *stype)
{
  DAPeriodicType ptype;
  DABoundaryType x=DA_BOUNDARY_NONE;
  DABoundaryType y=DA_BOUNDARY_NONE;
  DABoundaryType z=DA_BOUNDARY_NONE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = DAGetInfo(da,dim,M,N,P,m,n,p,dof,s,&ptype,stype);CHKERRQ(ierr);
  switch (ptype) {
  case DA_NONPERIODIC: break;
  case DA_XPERIODIC:   x = DA_BOUNDARY_PERIODIC; break;
  case DA_YPERIODIC:   y = DA_BOUNDARY_PERIODIC; break;
  case DA_ZPERIODIC:   z = DA_BOUNDARY_PERIODIC; break;
  case DA_XYPERIODIC:  x = y = DA_BOUNDARY_PERIODIC; break;
  case DA_XZPERIODIC:  x = z = DA_BOUNDARY_PERIODIC; break;
  case DA_YZPERIODIC:  y = z = DA_BOUNDARY_PERIODIC; break;
  case DA_XYZPERIODIC: x = y = z = DA_BOUNDARY_PERIODIC; break;
  case DA_XYZGHOSTED:  x = y = z = DA_BOUNDARY_GHOSTED; break;
  default: break;
  }
  if (btx) *btx = x;
  if (bty) *bty = y;
  if (btz) *btz = z;
  PetscFunctionReturn(0);
}
#define DAGetInfo DAGetInfo_Compat
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "DAGetInterpolationType"
static PetscErrorCode
DAGetInterpolationType_Compat(DA da,DAInterpolationType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(ctype,2);
#if PETSC_VERSION_(3,1,0)
  *ctype = da->interptype;
#else
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
#endif
  PetscFunctionReturn(0);
}
#define DAGetInterpolationType DAGetInterpolationType_Compat
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,0,0)

#define PetscDA_ERR_SUP(da) \
  PetscFunctionBegin; \
  PetscValidHeaderSpecific(da,DM_COOKIE,1); \
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() " \
          "not supported in this PETSc version"); \
  PetscFunctionReturn(PETSC_ERR_SUP);
/**/
#undef __FUNCT__
#define __FUNCT__ "DASetDim"
static PetscErrorCode DASetDim(DA da,...){PetscDA_ERR_SUP(da)}
#undef __FUNCT__
#define __FUNCT__ "DASetDof"
static PetscErrorCode DASetDof(DA da,...){PetscDA_ERR_SUP(da)}
#undef __FUNCT__
#define __FUNCT__ "DASetSizes"
static PetscErrorCode DASetSizes(DA da,...){PetscDA_ERR_SUP(da)}
#undef __FUNCT__
#define __FUNCT__ "DASetNumProcs"
static PetscErrorCode DASetNumProcs(DA da,...){PetscDA_ERR_SUP(da)}
#undef __FUNCT__
#define __FUNCT__ "DASetBoundaryType"
static PetscErrorCode DASetBoundaryType(DA da,...){PetscDA_ERR_SUP(da)}
#undef __FUNCT__
#define __FUNCT__ "DASetStencilType"
static PetscErrorCode DASetStencilType(DA da,...){PetscDA_ERR_SUP(da)}
#undef __FUNCT__
#define __FUNCT__ "DASetStencilWidth"
static PetscErrorCode DASetStencilWidth(DA da,...){PetscDA_ERR_SUP(da)}

#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)

#define DMDABoundaryType           DABoundaryType
#define DMDA_BOUNDARY_NONE         DA_BOUNDARY_NONE
#define DMDA_BOUNDARY_GHOSTED      DA_BOUNDARY_GHOSTED
#define DMDA_BOUNDARY_MIRROR       DA_BOUNDARY_MIRROR
#define DMDA_BOUNDARY_PERIODIC     DA_BOUNDARY_PERIODIC

#define DMDADirection              DADirection
#define DMDA_X                     DA_X
#define DMDA_Y                     DA_Y
#define DMDA_Z                     DA_Z

#define DMDAStencilType            DAStencilType
#define DMDA_STENCIL_STAR          DA_STENCIL_STAR
#define DMDA_STENCIL_BOX           DA_STENCIL_BOX

#define DMDAInterpolationType      DAInterpolationType
#define DMDA_Q0                    DA_Q0
#define DMDA_Q1                    DA_Q1

#define DMDAElementType            DAElementType
#define DMDA_ELEMENT_P1            DA_ELEMENT_P1
#define DMDA_ELEMENT_Q1            DA_ELEMENT_Q1

#define DMDACreate(a,b)                          DACreate(a,(DA*)b)
#define DMDASetDim(a,b)                          DASetDim((DA)a,b)
#define DMDASetDof(a,b)                          DASetDof((DA)a,b)
#define DMDASetSizes(a,x,y,z)                    DASetSizes((DA)a,x,y,z)
#define DMDASetNumProcs(a,x,y,z)                 DASetNumProcs((DA)a,x,y,z)
#define DMDASetOwnershipRanges(a,x,y,z)          DASetOwnershipRanges((DA)a,x,y,z)
#define DMDASetBoundaryType(a,x,y,z)             DASetBoundaryType((DA)a,x,y,z)
#define DMDASetStencilType(a,b)                  DASetStencilType((DA)a,b)
#define DMDASetStencilWidth(a,b)                 DASetStencilWidth((DA)a,b)

#define DMDAGetInfo(a,d,M,N,P,m,n,p,f,w,x,y,z,s) DAGetInfo((DA)a,d,M,N,P,m,n,p,f,w,x,y,z,s)
#define DMDAGetCorners(a,x,y,z,u,v,w)            DAGetCorners((DA)a,x,y,z,u,v,w)
#define DMDAGetGhostCorners(a,x,y,z,u,v,w)       DAGetGhostCorners((DA)a,x,y,z,u,v,w)
#define DMDAGetOwnershipRanges(a,x,y,z)          DAGetOwnershipRanges((DA)a,x,y,z)

#define DMDASetUniformCoordinates(a,x,y,z,u,v,w) DASetUniformCoordinates((DA)a,x,y,z,u,v,w)
#define DMDASetCoordinates(a,b)                  DASetCoordinates((DA)a,b)
#define DMDAGetCoordinates(a,b)                  DAGetCoordinates((DA)a,b)
#define DMDAGetCoordinateDA(a,b)                 DAGetCoordinateDA((DA)a,(DA*)b)
#define DMDAGetGhostedCoordinates(a,b)           DAGetGhostedCoordinates((DA)a,b)
#define DMDAGetBoundingBox(a,b,c)                DAGetBoundingBox((DA)a,b,c)
#define DMDAGetLocalBoundingBox(a,b,c)           DAGetLocalBoundingBox((DA)a,b,c)

#define DMDACreateNaturalVector(a,b)             DACreateNaturalVector((DA)a,b)
#define DMDAGlobalToNaturalBegin(a,b,c,d)        DAGlobalToNaturalBegin((DA)a,b,c,d)
#define DMDAGlobalToNaturalEnd(a,b,c,d)          DAGlobalToNaturalEnd((DA)a,b,c,d)
#define DMDANaturalToGlobalBegin(a,b,c,d)        DANaturalToGlobalBegin((DA)a,b,c,d)
#define DMDANaturalToGlobalEnd(a,b,c,d)          DANaturalToGlobalEnd((DA)a,b,c,d)
#define DMDALocalToLocalBegin(a,b,c,d)           DALocalToLocalBegin((DA)a,b,c,d)
#define DMDALocalToLocalEnd(a,b,c,d)             DALocalToLocalEnd((DA)a,b,c,d)

#define DMDAGetAO(a,b)                           DAGetAO((DA)a,b)
#define DMDAGetScatter(a,b,c,d)                  DAGetScatter((DA)a,b,c,d)

#define DMDASetRefinementFactor(a,x,y,z)         DASetRefinementFactor((DA)a,x,y,z)
#define DMDAGetRefinementFactor(a,x,y,z)         DAGetRefinementFactor((DA)a,x,y,z)
#define DMDASetInterpolationType(a,b)            DASetInterpolationType((DA)a,b)
#define DMDAGetInterpolationType(a,b)            DAGetInterpolationType((DA)a,b)
#define DMDASetElementType(a,b)                  DASetElementType((DA)a,b)
#define DMDAGetElementType(a,b)                  DAGetElementType((DA)a,b)
#define DMDAGetElements(a,b,c,d)                 DAGetElements((DA)a,b,c,d)
#define DMDARestoreElements(a,b,c,d)             DARestoreElements((DA)a,b,c,d)

#endif

/* ---------------------------------------------------------------- */

#endif /* _COMPAT_PETSC_DA_H */
