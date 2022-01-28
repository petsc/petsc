#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmswarmimpl.h>
#include "../src/dm/impls/swarm/data_bucket.h"

PetscErrorCode private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(PetscInt dim,PetscInt np[],PetscInt *_npoints,PetscReal **_xi)
{
  PetscErrorCode ierr;
  PetscReal      *xi;
  PetscInt       d,npoints=0,cnt;
  PetscReal      ds[] = {0.0,0.0,0.0};
  PetscInt       ii,jj,kk;

  PetscFunctionBegin;
  switch (dim) {
    case 1:
      npoints = np[0];
      break;
    case 2:
      npoints = np[0]*np[1];
      break;
    case 3:
      npoints = np[0]*np[1]*np[2];
      break;
  }
  for (d=0; d<dim; d++) {
    ds[d] = 2.0 / ((PetscReal)np[d]);
  }

  ierr = PetscMalloc1(dim*npoints,&xi);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      cnt = 0;
      for (ii=0; ii<np[0]; ii++) {
        xi[dim*cnt+0] = -1.0 + 0.5*ds[d] + ii*ds[0];
        cnt++;
      }
      break;

    case 2:
      cnt = 0;
      for (jj=0; jj<np[1]; jj++) {
        for (ii=0; ii<np[0]; ii++) {
          xi[dim*cnt+0] = -1.0 + 0.5*ds[0] + ii*ds[0];
          xi[dim*cnt+1] = -1.0 + 0.5*ds[1] + jj*ds[1];
          cnt++;
        }
      }
      break;

    case 3:
      cnt = 0;
      for (kk=0; kk<np[2]; kk++) {
        for (jj=0; jj<np[1]; jj++) {
          for (ii=0; ii<np[0]; ii++) {
            xi[dim*cnt+0] = -1.0 + 0.5*ds[0] + ii*ds[0];
            xi[dim*cnt+1] = -1.0 + 0.5*ds[1] + jj*ds[1];
            xi[dim*cnt+2] = -1.0 + 0.5*ds[2] + kk*ds[2];
            cnt++;
          }
        }
      }
      break;
  }
  *_npoints = npoints;
  *_xi = xi;
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmCreateCellLocalCoords_DA_Q1_Gauss(PetscInt dim,PetscInt np_1d,PetscInt *_npoints,PetscReal **_xi)
{
  PetscErrorCode  ierr;
  PetscQuadrature quadrature;
  const PetscReal *quadrature_xi;
  PetscReal       *xi;
  PetscInt        d,q,npoints_q;

  PetscFunctionBegin;
  ierr = PetscDTGaussTensorQuadrature(dim,1,np_1d,-1.0,1.0,&quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&npoints_q,&quadrature_xi,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*npoints_q,&xi);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    for (d=0; d<dim; d++) {
      xi[dim*q+d] = quadrature_xi[dim*q+d];
    }
  }
  ierr = PetscQuadratureDestroy(&quadrature);CHKERRQ(ierr);
  *_npoints = npoints_q;
  *_xi = xi;
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA_Q1(DM dm,DM dmc,PetscInt npoints,DMSwarmPICLayoutType layout)
{
  PetscErrorCode    ierr;
  PetscInt          dim,npoints_q;
  PetscInt          nel,npe,e,q,k,d;
  const PetscInt    *element_list;
  PetscReal         **basis;
  PetscReal         *xi;
  Vec               coor;
  const PetscScalar *_coor;
  PetscReal         *elcoor;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscInt          pcnt;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (layout) {
    case DMSWARMPIC_LAYOUT_REGULAR:
    {
      PetscInt np_dir[3];
      np_dir[0] = np_dir[1] = np_dir[2] = npoints;
      ierr = private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(dim,np_dir,&npoints_q,&xi);CHKERRQ(ierr);
    }
      break;
    case DMSWARMPIC_LAYOUT_GAUSS:
      ierr = private_DMSwarmCreateCellLocalCoords_DA_Q1_Gauss(dim,npoints,&npoints_q,&xi);CHKERRQ(ierr);
      break;

    case DMSWARMPIC_LAYOUT_SUBDIVISION:
    {
      PetscInt s,nsub;
      PetscInt np_dir[3];
      nsub = npoints;
      np_dir[0] = 1;
      for (s=0; s<nsub; s++) {
        np_dir[0] *= 2;
      }
      np_dir[1] = np_dir[0];
      np_dir[2] = np_dir[0];
      ierr = private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(dim,np_dir,&npoints_q,&xi);CHKERRQ(ierr);
    }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"A valid DMSwarmPIC layout must be provided");
  }

  ierr = DMDAGetElements(dmc,&nel,&npe,&element_list);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*npe,&elcoor);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints_q,&basis);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    ierr = PetscMalloc1(npe,&basis[q]);CHKERRQ(ierr);

    switch (dim) {
      case 1:
        basis[q][0] = 0.5*(1.0 - xi[dim*q+0]);
        basis[q][1] = 0.5*(1.0 + xi[dim*q+0]);
        break;
      case 2:
        basis[q][0] = 0.25*(1.0 - xi[dim*q+0])*(1.0 - xi[dim*q+1]);
        basis[q][1] = 0.25*(1.0 + xi[dim*q+0])*(1.0 - xi[dim*q+1]);
        basis[q][2] = 0.25*(1.0 + xi[dim*q+0])*(1.0 + xi[dim*q+1]);
        basis[q][3] = 0.25*(1.0 - xi[dim*q+0])*(1.0 + xi[dim*q+1]);
        break;

      case 3:
        basis[q][0] = 0.125*(1.0 - xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][1] = 0.125*(1.0 + xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][2] = 0.125*(1.0 + xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][3] = 0.125*(1.0 - xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][4] = 0.125*(1.0 - xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        basis[q][5] = 0.125*(1.0 + xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        basis[q][6] = 0.125*(1.0 + xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        basis[q][7] = 0.125*(1.0 - xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        break;
    }
  }

  ierr = DMSwarmSetLocalSizes(dm,npoints_q*nel,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dmc,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&_coor);CHKERRQ(ierr);
  pcnt = 0;
  for (e=0; e<nel; e++) {
    const PetscInt *element = &element_list[npe*e];

    for (k=0; k<npe; k++) {
      for (d=0; d<dim; d++) {
        elcoor[dim*k+d] = PetscRealPart(_coor[ dim*element[k] + d ]);
      }
    }

    for (q=0; q<npoints_q; q++) {
      for (d=0; d<dim; d++) {
        swarm_coor[dim*pcnt+d] = 0.0;
      }
      for (k=0; k<npe; k++) {
        for (d=0; d<dim; d++) {
          swarm_coor[dim*pcnt+d] += basis[q][k] * elcoor[dim*k+d];
        }
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
  }
  ierr = VecRestoreArrayRead(coor,&_coor);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMDARestoreElements(dmc,&nel,&npe,&element_list);CHKERRQ(ierr);

  ierr = PetscFree(xi);CHKERRQ(ierr);
  ierr = PetscFree(elcoor);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    ierr = PetscFree(basis[q]);CHKERRQ(ierr);
  }
  ierr = PetscFree(basis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA(DM dm,DM celldm,DMSwarmPICLayoutType layout,PetscInt layout_param)
{
  PetscErrorCode  ierr;
  DMDAElementType etype;
  PetscInt        dim;

  PetscFunctionBegin;
  ierr = DMDAGetElementType(celldm,&etype);CHKERRQ(ierr);
  ierr = DMGetDimension(celldm,&dim);CHKERRQ(ierr);
  switch (etype) {
    case DMDA_ELEMENT_P1:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DA support is not currently available for DMDA_ELEMENT_P1");
    case DMDA_ELEMENT_Q1:
      PetscAssertFalse(dim == 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Support only available for dim = 2, 3");
      ierr = private_DMSwarmInsertPointsUsingCellDM_DA_Q1(dm,celldm,layout_param,layout);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmProjectField_ApproxQ1_DA_2D(DM swarm,PetscReal *swarm_field,DM dm,Vec v_field)
{
  PetscErrorCode    ierr;
  Vec               v_field_l,denom_l,coor_l,denom;
  PetscScalar       *_field_l,*_denom_l;
  PetscInt          k,p,e,npoints,nel,npe;
  PetscInt          *mpfield_cell;
  PetscReal         *mpfield_coor;
  const PetscInt    *element_list;
  const PetscInt    *element;
  PetscScalar       xi_p[2],Ni[4];
  const PetscScalar *_coor;

  PetscFunctionBegin;
  ierr = VecZeroEntries(v_field);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&v_field_l);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&denom);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&denom_l);CHKERRQ(ierr);
  ierr = VecZeroEntries(v_field_l);CHKERRQ(ierr);
  ierr = VecZeroEntries(denom);CHKERRQ(ierr);
  ierr = VecZeroEntries(denom_l);CHKERRQ(ierr);

  ierr = VecGetArray(v_field_l,&_field_l);CHKERRQ(ierr);
  ierr = VecGetArray(denom_l,&_denom_l);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm,&coor_l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor_l,&_coor);CHKERRQ(ierr);

  ierr = DMDAGetElements(dm,&nel,&npe,&element_list);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);

  for (p=0; p<npoints; p++) {
    PetscReal         *coor_p;
    const PetscScalar *x0;
    const PetscScalar *x2;
    PetscScalar       dx[2];

    e = mpfield_cell[p];
    coor_p = &mpfield_coor[2*p];
    element = &element_list[npe*e];

    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0 = &_coor[2*element[0]];
    x2 = &_coor[2*element[2]];

    dx[0] = x2[0] - x0[0];
    dx[1] = x2[1] - x0[1];

    xi_p[0] = 2.0 * (coor_p[0] - x0[0])/dx[0] - 1.0;
    xi_p[1] = 2.0 * (coor_p[1] - x0[1])/dx[1] - 1.0;

    /* evaluate basis functions */
    Ni[0] = 0.25*(1.0 - xi_p[0])*(1.0 - xi_p[1]);
    Ni[1] = 0.25*(1.0 + xi_p[0])*(1.0 - xi_p[1]);
    Ni[2] = 0.25*(1.0 + xi_p[0])*(1.0 + xi_p[1]);
    Ni[3] = 0.25*(1.0 - xi_p[0])*(1.0 + xi_p[1]);

    for (k=0; k<npe; k++) {
      _field_l[ element[k] ] += Ni[k] * swarm_field[p];
      _denom_l[ element[k] ] += Ni[k];
    }
  }

  ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  ierr = DMDARestoreElements(dm,&nel,&npe,&element_list);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor_l,&_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(v_field_l,&_field_l);CHKERRQ(ierr);
  ierr = VecRestoreArray(denom_l,&_denom_l);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,v_field_l,ADD_VALUES,v_field);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,v_field_l,ADD_VALUES,v_field);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,denom_l,ADD_VALUES,denom);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,denom_l,ADD_VALUES,denom);CHKERRQ(ierr);

  ierr = VecPointwiseDivide(v_field,v_field,denom);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&v_field_l);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&denom_l);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&denom);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmProjectFields_DA(DM swarm,DM celldm,PetscInt project_type,PetscInt nfields,DMSwarmDataField dfield[],Vec vecs[])
{
  PetscErrorCode  ierr;
  PetscInt        f,dim;
  DMDAElementType etype;

  PetscFunctionBegin;
  ierr = DMDAGetElementType(celldm,&etype);CHKERRQ(ierr);
  PetscAssertFalse(etype == DMDA_ELEMENT_P1,PetscObjectComm((PetscObject)swarm),PETSC_ERR_SUP,"Only Q1 DMDA supported");

  ierr = DMGetDimension(swarm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2:
      for (f=0; f<nfields; f++) {
        PetscReal *swarm_field;

        ierr = DMSwarmDataFieldGetEntries(dfield[f],(void**)&swarm_field);CHKERRQ(ierr);
        ierr = DMSwarmProjectField_ApproxQ1_DA_2D(swarm,swarm_field,celldm,vecs[f]);CHKERRQ(ierr);
      }
      break;
    case 3:
      SETERRQ(PetscObjectComm((PetscObject)swarm),PETSC_ERR_SUP,"No support for 3D");
    default:
      break;
  }
  PetscFunctionReturn(0);
}
