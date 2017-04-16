
#include <petsc.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>


PetscErrorCode subdivide_triangle(PetscReal v1[2],PetscReal v2[2],PetscReal v3[3],PetscInt depth,PetscInt max,PetscReal xi[],PetscInt *np)
{
  PetscReal v12[2],v23[2],v31[2];
  PetscInt i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (depth == max) {
    PetscReal cx[2];
    
    //printf("centroid \n");
    cx[0] = (v1[0] + v2[0] + v3[0])/3.0;
    cx[1] = (v1[1] + v2[1] + v3[1])/3.0;
    
    xi[2*(*np)+0] = cx[0];
    xi[2*(*np)+1] = cx[1];
    (*np)++;
    PetscFunctionReturn(0);
  }
  
  /* calculate midpoints of each side */
  for (i = 0; i < 2; i++) {
    v12[i] = (v1[i]+v2[i])/2.0;
    v23[i] = (v2[i]+v3[i])/2.0;
    v31[i] = (v3[i]+v1[i])/2.0;
  }
  
  /* recursively subdivide new triangles */
  ierr = subdivide_triangle(v1,v12,v31,depth+1,max,xi,np);CHKERRQ(ierr);
  ierr = subdivide_triangle(v2,v23,v12,depth+1,max,xi,np);CHKERRQ(ierr);
  ierr = subdivide_triangle(v3,v31,v23,depth+1,max,xi,np);CHKERRQ(ierr);
  ierr = subdivide_triangle(v12,v23,v31,depth+1,max,xi,np);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX2D_SubDivide(DM dm,DM dmc,PetscInt nsub)
{
  PetscErrorCode ierr;
  const PetscInt dim = 2;
  PetscInt q,npoints_q,e,nel,npe,pcnt,ps,pe,d,k,depth;
  PetscReal *xi;
  PetscReal **basis;
  Vec coorlocal;
  PetscSection coordSection;
  PetscScalar *elcoor = NULL;
  PetscReal *swarm_coor;
  PetscInt *swarm_cellid;
  PetscReal v1[2],v2[2],v3[2];
  
  PetscFunctionBegin;
  
  npoints_q = 1;
  for (d=0; d<nsub; d++) { npoints_q *= 4; }
  ierr = PetscMalloc1(dim*npoints_q,&xi);CHKERRQ(ierr);
  
  v1[0] = 0.0;  v1[1] = 0.0;
  v2[0] = 1.0;  v2[1] = 0.0;
  v3[0] = 0.0;  v3[1] = 1.0;
  depth = 0;
  pcnt = 0;
  ierr = subdivide_triangle(v1,v2,v3,depth,nsub,xi,&pcnt);CHKERRQ(ierr);

  npe = 3; /* nodes per element (triangle) */
  ierr = PetscMalloc1(npoints_q,&basis);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    ierr = PetscMalloc1(npe,&basis[q]);CHKERRQ(ierr);
    
    basis[q][0] = 1.0 - xi[dim*q+0] - xi[dim*q+1];
    basis[q][1] = xi[dim*q+0];
    basis[q][2] = xi[dim*q+1];
  }
  
  /* 0->cell, 1->edge, 2->vert */
  ierr = DMPlexGetHeightStratum(dmc,0,&ps,&pe);CHKERRQ(ierr);
  nel = pe - ps;
  
  ierr = DMSwarmSetLocalSizes(dm,npoints_q*nel,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dmc,&coorlocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmc,&coordSection);CHKERRQ(ierr);
  
  pcnt = 0;
  for (e=0; e<nel; e++) {
    ierr = DMPlexVecGetClosure(dmc,coordSection,coorlocal,e,NULL,&elcoor);CHKERRQ(ierr);
    
    for (q=0; q<npoints_q; q++) {
      for (d=0; d<dim; d++) {
        swarm_coor[dim*pcnt+d] = 0.0;
        for (k=0; k<npe; k++) {
          swarm_coor[dim*pcnt+d] += basis[q][k] * elcoor[dim*k+d];
        }
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
    ierr = DMPlexVecRestoreClosure(dmc,coordSection,coorlocal,e,NULL,&elcoor);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  
  ierr = PetscFree(xi);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    ierr = PetscFree(basis[q]);CHKERRQ(ierr);
  }
  ierr = PetscFree(basis);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX2D_Regular(DM dm,DM dmc,PetscInt npoints)
{
  PetscErrorCode ierr;
  const PetscInt dim = 2;
  PetscInt ii,jj,q,npoints_q,e,nel,npe,pcnt,ps,pe,d,k;
  PetscReal *xi,ds,ds2;
  PetscReal **basis;
  Vec coorlocal;
  PetscSection coordSection;
  PetscScalar *elcoor = NULL;
  PetscReal *swarm_coor;
  PetscInt *swarm_cellid;

  PetscFunctionBegin;
  ierr = PetscMalloc1(dim*npoints*npoints,&xi);CHKERRQ(ierr);
  pcnt = 0;
  ds = 1.0/((PetscReal)(npoints-1));
  ds2 = 1.0/((PetscReal)(npoints));
  for (jj = 0; jj<npoints; jj++) {
    for (ii=0; ii<npoints-jj; ii++) {
      xi[dim*pcnt+0] = ii * ds;
      xi[dim*pcnt+1] = jj * ds;
      
      xi[dim*pcnt+0] *= (1.0 - 1.2*ds2);
      xi[dim*pcnt+1] *= (1.0 - 1.2*ds2);
      
      xi[dim*pcnt+0] += 0.35*ds2;
      xi[dim*pcnt+1] += 0.35*ds2;

      pcnt++;
    }
  }
  npoints_q = pcnt;
  
  npe = 3; /* nodes per element (triangle) */
  ierr = PetscMalloc1(npoints_q,&basis);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    ierr = PetscMalloc1(npe,&basis[q]);CHKERRQ(ierr);
    
    basis[q][0] = 1.0 - xi[dim*q+0] - xi[dim*q+1];
    basis[q][1] = xi[dim*q+0];
    basis[q][2] = xi[dim*q+1];
  }

  /* 0->cell, 1->edge, 2->vert */
  ierr = DMPlexGetHeightStratum(dmc,0,&ps,&pe);CHKERRQ(ierr);
  nel = pe - ps;
  
  ierr = DMSwarmSetLocalSizes(dm,npoints_q*nel,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dmc,&coorlocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmc,&coordSection);CHKERRQ(ierr);

  pcnt = 0;
  for (e=0; e<nel; e++) {
    ierr = DMPlexVecGetClosure(dmc,coordSection,coorlocal,e,NULL,&elcoor);CHKERRQ(ierr);
    
    for (q=0; q<npoints_q; q++) {
      for (d=0; d<dim; d++) {
        swarm_coor[dim*pcnt+d] = 0.0;
        for (k=0; k<npe; k++) {
          swarm_coor[dim*pcnt+d] += basis[q][k] * elcoor[dim*k+d];
        }
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
    ierr = DMPlexVecRestoreClosure(dmc,coordSection,coorlocal,e,NULL,&elcoor);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  
  ierr = PetscFree(xi);CHKERRQ(ierr);
  for (q=0; q<npoints_q; q++) {
    ierr = PetscFree(basis[q]);CHKERRQ(ierr);
  }
  ierr = PetscFree(basis);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX(DM dm,DM celldm,DMSwarmPICLayoutType layout,PetscInt layout_param)
{
  PetscErrorCode ierr;
  PetscInt dim;
  
  PetscFunctionBegin;
  ierr = DMGetDimension(celldm,&dim);CHKERRQ(ierr);
  if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"No 3D support for PLEX");
  switch (layout) {
    case DMSWARMPIC_LAYOUT_REGULAR:
      ierr = private_DMSwarmInsertPointsUsingCellDM_PLEX2D_Regular(dm,celldm,layout_param);CHKERRQ(ierr);
      break;
    case DMSWARMPIC_LAYOUT_GAUSS:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Gauss layout not supported for PLEX");
      break;
    case DMSWARMPIC_LAYOUT_SUBDIVISION:
      ierr = private_DMSwarmInsertPointsUsingCellDM_PLEX2D_SubDivide(dm,celldm,layout_param);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}
