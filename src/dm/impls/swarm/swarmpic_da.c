
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

PetscErrorCode private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(PetscInt dim,PetscInt np[],PetscInt *_npoints,PetscReal **_xi)
{
  PetscErrorCode ierr;
  PetscReal *xi;
  PetscInt d,npoints,cnt;
  PetscReal ds[] = {0.0,0.0,0.0};
  PetscInt ii,jj,kk;

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
  PetscErrorCode ierr;
  PetscQuadrature quadrature;
  const PetscReal *quadrature_xi;
  PetscReal *xi;
  PetscInt d,q,npoints_q;
  
  PetscFunctionBegin;
  ierr = PetscDTGaussTensorQuadrature(dim,np_1d,-1.0,1.0,&quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature,NULL,&npoints_q,&quadrature_xi,NULL);CHKERRQ(ierr);
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

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA_Q1(DM dm,DM dmc,PetscInt npoints,PetscInt layout)
{
  PetscErrorCode ierr;
  PetscInt dim,npoints_q;
  PetscInt nel,npe,e,q,k,d;
  const PetscInt *element_list;
  PetscReal **basis;
  PetscReal *xi;
  Vec coor;
  const PetscScalar *_coor;
  PetscReal *elcoor;
  PetscReal *swarm_coor;
  PetscInt *swarm_cellid;
  PetscInt pcnt;
  
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
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"A valid DMSwarmPIC layout must be provided");
      break;
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
        elcoor[dim*k+d] = _coor[ dim*element[k] + d ];
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
  PetscErrorCode ierr;
  DMDAElementType etype;
  PetscInt dim;
  
  PetscFunctionBegin;
  ierr = DMDAGetElementType(celldm,&etype);CHKERRQ(ierr);
  ierr = DMGetDimension(celldm,&dim);CHKERRQ(ierr);
  switch (etype) {
    case DMDA_ELEMENT_P1:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DA support is not currently available for DMDA_ELEMENT_P1");
      break;
    case DMDA_ELEMENT_Q1:
      if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Support only available for dim = 2, 3");
      ierr = private_DMSwarmInsertPointsUsingCellDM_DA_Q1(dm,celldm,layout_param,layout);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}
