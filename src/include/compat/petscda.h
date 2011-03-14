#ifndef _COMPAT_PETSC_DA_H
#define _COMPAT_PETSC_DA_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#include <petscda.h>
#endif

#if PETSC_VERSION_(3,1,0)
#undef __FUNCT__
#define __FUNCT__ "DASetUp"
static PetscErrorCode
DASetUp_Compat(DA da)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = DASetFromOptions(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DASetUp DASetUp_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define DASetOwnershipRanges DASetVertexDivision
#define DAGetLocalToGlobalMapping      DAGetISLocalToGlobalMapping
#define DAGetLocalToGlobalMappingBlock DAGetISLocalToGlobalMappingBlck
#endif

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

/*#undef __FUNCT__
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
#define DAGetElementType DAGetElementType_Compat*/

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

#endif

/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_(3,0,0))
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
/*#define __FUNCT__ "DAGetElementType"
static PetscErrorCode
DAGetElementType_Compat(DA da, DAElementType *etype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(etype,2);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#define DAGetElementType DAGetElementType_Compat*/
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

#if (PETSC_VERSION_(3,0,0))
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

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DASetOptionsPrefix"
static PetscErrorCode DASetOptionsPrefix(DA da,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)da,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DASetFromOptions"
static PetscErrorCode DASetFromOptions(DA da) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DA Options","DA");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
typedef enum {
  DA_BOUNDARY_NONE,
  DA_BOUNDARY_GHOSTED,
  DA_BOUNDARY_MIRROR,
  DA_BOUNDARY_PERIODIC,
} DABoundaryType;
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DA_Boundary2Periodic"
static PetscErrorCode
DA_Boundary2Periodic(PetscInt dim,
                     DABoundaryType x,DABoundaryType y,DABoundaryType z,
                     DAPeriodicType *_ptype)
{
  DAPeriodicType ptype = DA_NONPERIODIC;
  PetscFunctionBegin;
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
    SETERRQ(PETSC_ERR_SUP,
	    "Boundary type not supported in this PETSc version");
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
#define DASetUp DASetUp_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DAGetInfo_Compat"
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

#endif /* _COMPAT_PETSC_DA_H */
