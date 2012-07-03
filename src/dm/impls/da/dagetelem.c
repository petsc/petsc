
#include <petsc-private/daimpl.h>     /*I  "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMDAGetElements_1D"
static PetscErrorCode DMDAGetElements_1D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  DM_DA          *da = (DM_DA*)dm->data;
  PetscInt       i,xs,xe,Xs,Xe;
  PetscInt       cnt=0;
  PetscFunctionBegin;
  if (!da->e) {
    ierr = DMDAGetCorners(dm,&xs,0,0,&xe,0,0);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(dm,&Xs,0,0,&Xe,0,0);CHKERRQ(ierr);
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
#define __FUNCT__ "DMDAGetElements_2D"
static PetscErrorCode DMDAGetElements_2D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  DM_DA          *da = (DM_DA*)dm->data;
  PetscInt       i,xs,xe,Xs,Xe;
  PetscInt       j,ys,ye,Ys,Ye;
  PetscInt       cnt=0, cell[4], ns=2, nn=3;
  PetscInt       c, split[] = {0,1,3,
                               2,3,1};
  PetscFunctionBegin;
  if (!da->e) {
    if (da->elementtype == DMDA_ELEMENT_P1) {ns=2; nn=3;}
    if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=4;}
    ierr = DMDAGetCorners(dm,&xs,&ys,0,&xe,&ye,0);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,0,&Xe,&Ye,0);CHKERRQ(ierr);
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
        if (da->elementtype == DMDA_ELEMENT_P1) {
          for (c=0; c<ns*nn; c++)
            da->e[cnt++] = cell[split[c]];
        }
        if (da->elementtype == DMDA_ELEMENT_Q1) {
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
#define __FUNCT__ "DMDAGetElements_3D"
static PetscErrorCode DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscErrorCode ierr;
  DM_DA          *da = (DM_DA*)dm->data;
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
    if (da->elementtype == DMDA_ELEMENT_P1) {ns=6; nn=4;}
    if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=8;}
    ierr = DMDAGetCorners(dm,&xs,&ys,&zs,&xe,&ye,&ze);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,&Zs,&Xe,&Ye,&Ze);CHKERRQ(ierr);
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
          if (da->elementtype == DMDA_ELEMENT_P1) {
            for (c=0; c<ns*nn; c++)
              da->e[cnt++] = cell[split[c]];
          }
          if (da->elementtype == DMDA_ELEMENT_Q1) {
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

/*@C
      DMDASetElementType - Sets the element type to be returned by DMDAGetElements()

    Not Collective

   Input Parameter:
.     da - the DMDA object

   Output Parameters:
.     etype - the element type, currently either DMDA_ELEMENT_P1 or ELEMENT_Q1

   Level: intermediate

.seealso: DMDAElementType, DMDAGetElementType(), DMDAGetElements(), DMDARestoreElements()
@*/
#undef __FUNCT__
#define __FUNCT__ "DMDASetElementType"
PetscErrorCode  DMDASetElementType(DM da, DMDAElementType etype)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(da,etype,2);
  if (dd->elementtype != etype) {
    ierr = PetscFree(dd->e);CHKERRQ(ierr);
    dd->elementtype = etype;
    dd->ne          = 0;
    dd->e           = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
      DMDAGetElementType - Gets the element type to be returned by DMDAGetElements()

    Not Collective

   Input Parameter:
.     da - the DMDA object

   Output Parameters:
.     etype - the element type, currently either DMDA_ELEMENT_P1 or ELEMENT_Q1

   Level: intermediate

.seealso: DMDAElementType, DMDASetElementType(), DMDAGetElements(), DMDARestoreElements()
@*/
#undef __FUNCT__
#define __FUNCT__ "DMDAGetElementType"
PetscErrorCode  DMDAGetElementType(DM da, DMDAElementType *etype)
{
  DM_DA *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(etype,2);
  *etype = dd->elementtype;
  PetscFunctionReturn(0);
}

/*@C
      DMDAGetElements - Gets an array containing the indices (in local coordinates)
                 of all the local elements

    Not Collective

   Input Parameter:
.     dm - the DM object

   Output Parameters:
+     nel - number of local elements
.     nen - number of element nodes
-     e - the local indices of the elements' vertices

   Level: intermediate

.seealso: DMDAElementType, DMDASetElementType(), DMDARestoreElements(), VecSetValuesLocal(), MatSetValuesLocal(), DMGlobalToLocalBegin(), DMLocalToGlobalBegin()
@*/
#undef __FUNCT__
#define __FUNCT__ "DMDAGetElements"
PetscErrorCode  DMDAGetElements(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  DM_DA          *da = (DM_DA*)dm->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (da->dim==-1) {
    *nel = 0; *nen = 0; *e = PETSC_NULL;
  } else if (da->dim==1) {
    ierr = DMDAGetElements_1D(dm,nel,nen,e);CHKERRQ(ierr);
  } else if (da->dim==2) {
    ierr = DMDAGetElements_2D(dm,nel,nen,e);CHKERRQ(ierr);
  } else if (da->dim==3) {
    ierr = DMDAGetElements_3D(dm,nel,nen,e);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",da->dim);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreElements"
/*@C
      DMDARestoreElements - Returns an array containing the indices (in local coordinates)
                 of all the local elements obtained with DMDAGetElements()

    Not Collective

   Input Parameter:
+     dm - the DM object
.     nel - number of local elements
.     nen - number of element nodes
-     e - the local indices of the elements' vertices

   Level: intermediate

.seealso: DMDAElementType, DMDASetElementType(), DMDAGetElements()
@*/
PetscErrorCode  DMDARestoreElements(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidIntPointer(nel,2);
  PetscValidIntPointer(nen,3);
  PetscValidPointer(e,4);
  /* XXX */
  PetscFunctionReturn(0);
}
