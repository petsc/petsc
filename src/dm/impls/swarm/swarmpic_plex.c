#include <petscdm.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include "../src/dm/impls/swarm/data_bucket.h"

PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM,DM,PetscInt,PetscReal*xi);

static PetscErrorCode private_PetscFECreateDefault_scalar_pk1(DM dm, PetscInt dim, PetscBool isSimplex, PetscInt qorder, PetscFE *fem)
{
  const PetscInt  Nc = 1;
  PetscQuadrature q, fq;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order, quadPointsPerEdge;
  PetscBool       tensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
  /*ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);*/
  ierr = PetscSpacePolynomialSetTensor(P, tensor);CHKERRQ(ierr);
  /*ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);*/
  ierr = PetscSpaceSetType(P,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P,1,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(P, &order, NULL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(P, &tensor);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  /*ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix);CHKERRQ(ierr);*/
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, isSimplex), &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor);CHKERRQ(ierr);
  /*ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);*/
  ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), fem);CHKERRQ(ierr);
  /*ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix);CHKERRQ(ierr);*/
  /*ierr = PetscFESetFromOptions(*fem);CHKERRQ(ierr);*/
  ierr = PetscFESetType(*fem,PETSCFEBASIC);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  qorder = qorder >= 0 ? qorder : order;
  quadPointsPerEdge = PetscMax(qorder + 1,1);
  if (isSimplex) {
    ierr = PetscDTStroudConicalQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  }
  else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*fem, q);CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode subdivide_triangle(PetscReal v1[2],PetscReal v2[2],PetscReal v3[2],PetscInt depth,PetscInt max,PetscReal xi[],PetscInt *np)
{
  PetscReal      v12[2],v23[2],v31[2];
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (depth == max) {
    PetscReal cx[2];

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
  PetscInt       q,npoints_q,e,nel,npe,pcnt,ps,pe,d,k,depth;
  PetscReal      *xi;
  PetscReal      **basis;
  Vec            coorlocal;
  PetscSection   coordSection;
  PetscScalar    *elcoor = NULL;
  PetscReal      *swarm_coor;
  PetscInt       *swarm_cellid;
  PetscReal      v1[2],v2[2],v3[2];

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
          swarm_coor[dim*pcnt+d] += basis[q][k] * PetscRealPart(elcoor[dim*k+d]);
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

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX_SubDivide(DM dm,DM dmc,PetscInt nsub)
{
  PetscErrorCode  ierr;
  PetscInt        dim,nfaces,nbasis;
  PetscInt        q,npoints_q,e,nel,pcnt,ps,pe,d,k,r;
  PetscTabulation T;
  Vec             coorlocal;
  PetscSection    coordSection;
  PetscScalar     *elcoor = NULL;
  PetscReal       *swarm_coor;
  PetscInt        *swarm_cellid;
  const PetscReal *xiq;
  PetscQuadrature quadrature;
  PetscFE         fe,feRef;
  PetscBool       is_simplex;

  PetscFunctionBegin;
  ierr = DMGetDimension(dmc,&dim);CHKERRQ(ierr);
  is_simplex = PETSC_FALSE;
  ierr = DMPlexGetHeightStratum(dmc,0,&ps,&pe);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dmc, ps, &nfaces);CHKERRQ(ierr);
  if (nfaces == (dim+1)) { is_simplex = PETSC_TRUE; }

  ierr = private_PetscFECreateDefault_scalar_pk1(dmc, dim, is_simplex, 0, &fe);CHKERRQ(ierr);

  for (r=0; r<nsub; r++) {
    ierr = PetscFERefine(fe,&feRef);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(feRef,fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&feRef);CHKERRQ(ierr);
  }

  ierr = PetscFEGetQuadrature(fe,&quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature, NULL, NULL, &npoints_q, &xiq, NULL);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe,&nbasis);CHKERRQ(ierr);
  ierr = PetscFEGetCellTabulation(fe, 1, &T);CHKERRQ(ierr);

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
    ierr = DMPlexVecGetClosure(dmc,coordSection,coorlocal,ps+e,NULL,&elcoor);CHKERRQ(ierr);

    for (q=0; q<npoints_q; q++) {
      for (d=0; d<dim; d++) {
        swarm_coor[dim*pcnt+d] = 0.0;
        for (k=0; k<nbasis; k++) {
          swarm_coor[dim*pcnt+d] += T->T[0][q*nbasis + k] * PetscRealPart(elcoor[dim*k+d]);
        }
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
    ierr = DMPlexVecRestoreClosure(dmc,coordSection,coorlocal,ps+e,NULL,&elcoor);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);

  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX2D_Regular(DM dm,DM dmc,PetscInt npoints)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscInt       ii,jj,q,npoints_q,e,nel,npe,pcnt,ps,pe,d,k,nfaces;
  PetscReal      *xi,ds,ds2;
  PetscReal      **basis;
  Vec            coorlocal;
  PetscSection   coordSection;
  PetscScalar    *elcoor = NULL;
  PetscReal      *swarm_coor;
  PetscInt       *swarm_cellid;
  PetscBool      is_simplex;

  PetscFunctionBegin;
  ierr = DMGetDimension(dmc,&dim);CHKERRQ(ierr);
  PetscCheckFalse(dim != 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only 2D is supported");
  is_simplex = PETSC_FALSE;
  ierr = DMPlexGetHeightStratum(dmc,0,&ps,&pe);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dmc, ps, &nfaces);CHKERRQ(ierr);
  if (nfaces == (dim+1)) { is_simplex = PETSC_TRUE; }
  PetscCheckFalse(!is_simplex,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only the simplex is supported");

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
          swarm_coor[dim*pcnt+d] += basis[q][k] * PetscRealPart(elcoor[dim*k+d]);
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
  PetscInt       dim;

  PetscFunctionBegin;
  ierr = DMGetDimension(celldm,&dim);CHKERRQ(ierr);
  switch (layout) {
    case DMSWARMPIC_LAYOUT_REGULAR:
      PetscCheckFalse(dim == 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"No 3D support for REGULAR+PLEX");
      ierr = private_DMSwarmInsertPointsUsingCellDM_PLEX2D_Regular(dm,celldm,layout_param);CHKERRQ(ierr);
      break;
    case DMSWARMPIC_LAYOUT_GAUSS:
    {
      PetscInt npoints,npoints1,ps,pe,nfaces;
      const PetscReal *xi;
      PetscBool is_simplex;
      PetscQuadrature quadrature;

      is_simplex = PETSC_FALSE;
      ierr = DMPlexGetHeightStratum(celldm,0,&ps,&pe);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(celldm, ps, &nfaces);CHKERRQ(ierr);
      if (nfaces == (dim+1)) { is_simplex = PETSC_TRUE; }

      npoints1 = layout_param;
      if (is_simplex) {
        ierr = PetscDTStroudConicalQuadrature(dim,1,npoints1,-1.0,1.0,&quadrature);CHKERRQ(ierr);
      } else {
        ierr = PetscDTGaussTensorQuadrature(dim,1,npoints1,-1.0,1.0,&quadrature);CHKERRQ(ierr);
      }
      ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&npoints,&xi,NULL);CHKERRQ(ierr);
      ierr = private_DMSwarmSetPointCoordinatesCellwise_PLEX(dm,celldm,npoints,(PetscReal*)xi);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&quadrature);CHKERRQ(ierr);
    }
      break;
    case DMSWARMPIC_LAYOUT_SUBDIVISION:
      ierr = private_DMSwarmInsertPointsUsingCellDM_PLEX_SubDivide(dm,celldm,layout_param);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

/*
typedef struct {
  PetscReal x,y;
} Point2d;

static PetscErrorCode signp2d(Point2d p1, Point2d p2, Point2d p3,PetscReal *s)
{
  PetscFunctionBegin;
  *s = (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
  PetscFunctionReturn(0);
}
*/
/*
static PetscErrorCode PointInTriangle(Point2d pt, Point2d v1, Point2d v2, Point2d v3,PetscBool *v)
{
  PetscReal s1,s2,s3;
  PetscBool b1, b2, b3;

  PetscFunctionBegin;
  signp2d(pt, v1, v2,&s1); b1 = s1 < 0.0f;
  signp2d(pt, v2, v3,&s2); b2 = s2 < 0.0f;
  signp2d(pt, v3, v1,&s3); b3 = s3 < 0.0f;

  *v = ((b1 == b2) && (b2 == b3));
  PetscFunctionReturn(0);
}
*/
/*
static PetscErrorCode _ComputeLocalCoordinateAffine2d(PetscReal xp[],PetscReal coords[],PetscReal xip[],PetscReal *dJ)
{
  PetscReal x1,y1,x2,y2,x3,y3;
  PetscReal c,b[2],A[2][2],inv[2][2],detJ,od;

  PetscFunctionBegin;
  x1 = coords[2*0+0];
  x2 = coords[2*1+0];
  x3 = coords[2*2+0];

  y1 = coords[2*0+1];
  y2 = coords[2*1+1];
  y3 = coords[2*2+1];

  c = x1 - 0.5*x1 - 0.5*x1 + 0.5*x2 + 0.5*x3;
  b[0] = xp[0] - c;
  c = y1 - 0.5*y1 - 0.5*y1 + 0.5*y2 + 0.5*y3;
  b[1] = xp[1] - c;

  A[0][0] = -0.5*x1 + 0.5*x2;   A[0][1] = -0.5*x1 + 0.5*x3;
  A[1][0] = -0.5*y1 + 0.5*y2;   A[1][1] = -0.5*y1 + 0.5*y3;

  detJ = A[0][0]*A[1][1] - A[0][1]*A[1][0];
  *dJ = PetscAbsReal(detJ);
  od = 1.0/detJ;

  inv[0][0] =  A[1][1] * od;
  inv[0][1] = -A[0][1] * od;
  inv[1][0] = -A[1][0] * od;
  inv[1][1] =  A[0][0] * od;

  xip[0] = inv[0][0]*b[0] + inv[0][1]*b[1];
  xip[1] = inv[1][0]*b[0] + inv[1][1]*b[1];
  PetscFunctionReturn(0);
}
*/

static PetscErrorCode ComputeLocalCoordinateAffine2d(PetscReal xp[],PetscScalar coords[],PetscReal xip[],PetscReal *dJ)
{
  PetscReal x1,y1,x2,y2,x3,y3;
  PetscReal b[2],A[2][2],inv[2][2],detJ,od;

  PetscFunctionBegin;
  x1 = PetscRealPart(coords[2*0+0]);
  x2 = PetscRealPart(coords[2*1+0]);
  x3 = PetscRealPart(coords[2*2+0]);

  y1 = PetscRealPart(coords[2*0+1]);
  y2 = PetscRealPart(coords[2*1+1]);
  y3 = PetscRealPart(coords[2*2+1]);

  b[0] = xp[0] - x1;
  b[1] = xp[1] - y1;

  A[0][0] = x2-x1;   A[0][1] = x3-x1;
  A[1][0] = y2-y1;   A[1][1] = y3-y1;

  detJ = A[0][0]*A[1][1] - A[0][1]*A[1][0];
  *dJ = PetscAbsReal(detJ);
  od = 1.0/detJ;

  inv[0][0] =  A[1][1] * od;
  inv[0][1] = -A[0][1] * od;
  inv[1][0] = -A[1][0] * od;
  inv[1][1] =  A[0][0] * od;

  xip[0] = inv[0][0]*b[0] + inv[0][1]*b[1];
  xip[1] = inv[1][0]*b[0] + inv[1][1]*b[1];
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmProjectField_ApproxP1_PLEX_2D(DM swarm,PetscReal *swarm_field,DM dm,Vec v_field)
{
  PetscErrorCode  ierr;
  const PetscReal PLEX_C_EPS = 1.0e-8;
  Vec             v_field_l,denom_l,coor_l,denom;
  PetscInt        k,p,e,npoints;
  PetscInt        *mpfield_cell;
  PetscReal       *mpfield_coor;
  PetscReal       xi_p[2];
  PetscScalar     Ni[3];
  PetscSection    coordSection;
  PetscScalar     *elcoor = NULL;

  PetscFunctionBegin;
  ierr = VecZeroEntries(v_field);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&v_field_l);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&denom);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&denom_l);CHKERRQ(ierr);
  ierr = VecZeroEntries(v_field_l);CHKERRQ(ierr);
  ierr = VecZeroEntries(denom);CHKERRQ(ierr);
  ierr = VecZeroEntries(denom_l);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm,&coor_l);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm,&coordSection);CHKERRQ(ierr);

  ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);

  for (p=0; p<npoints; p++) {
    PetscReal   *coor_p,dJ;
    PetscScalar elfield[3];
    PetscBool   point_located;

    e       = mpfield_cell[p];
    coor_p  = &mpfield_coor[2*p];

    ierr = DMPlexVecGetClosure(dm,coordSection,coor_l,e,NULL,&elcoor);CHKERRQ(ierr);

/*
    while (!point_located && (failed_counter < 25)) {
      ierr = PointInTriangle(point, coords[0], coords[1], coords[2], &point_located);CHKERRQ(ierr);
      point.x = coor_p[0];
      point.y = coor_p[1];
      point.x += 1.0e-10 * (2.0 * rand()/((double)RAND_MAX)-1.0);
      point.y += 1.0e-10 * (2.0 * rand()/((double)RAND_MAX)-1.0);
      failed_counter++;
    }

    if (!point_located) {
        PetscPrintf(PETSC_COMM_SELF,"Failed to locate point (%1.8e,%1.8e) in local mesh (cell %D) with triangle coords (%1.8e,%1.8e) : (%1.8e,%1.8e) : (%1.8e,%1.8e) in %D iterations\n",point.x,point.y,e,coords[0].x,coords[0].y,coords[1].x,coords[1].y,coords[2].x,coords[2].y,failed_counter);
    }

    PetscCheckFalse(!point_located,PETSC_COMM_SELF,PETSC_ERR_SUP,"Failed to locate point (%1.8e,%1.8e) in local mesh (cell %D)",point.x,point.y,e);
    else {
      ierr = _ComputeLocalCoordinateAffine2d(coor_p,elcoor,xi_p,&dJ);CHKERRQ(ierr);
      xi_p[0] = 0.5*(xi_p[0] + 1.0);
      xi_p[1] = 0.5*(xi_p[1] + 1.0);

      PetscPrintf(PETSC_COMM_SELF,"[p=%D] x(%+1.4e,%+1.4e) -> mapped to element %D xi(%+1.4e,%+1.4e)\n",p,point.x,point.y,e,xi_p[0],xi_p[1]);

    }
*/

    ierr = ComputeLocalCoordinateAffine2d(coor_p,elcoor,xi_p,&dJ);CHKERRQ(ierr);
    /*
    PetscPrintf(PETSC_COMM_SELF,"[p=%D] x(%+1.4e,%+1.4e) -> mapped to element %D xi(%+1.4e,%+1.4e)\n",p,point.x,point.y,e,xi_p[0],xi_p[1]);
    */
    /*
     point_located = PETSC_TRUE;
    if (xi_p[0] < 0.0) {
      if (xi_p[0] > -PLEX_C_EPS) {
        xi_p[0] = 0.0;
      } else {
        point_located = PETSC_FALSE;
      }
    }
    if (xi_p[1] < 0.0) {
      if (xi_p[1] > -PLEX_C_EPS) {
        xi_p[1] = 0.0;
      } else {
        point_located = PETSC_FALSE;
      }
    }
    if (xi_p[1] > (1.0-xi_p[0])) {
      if ((xi_p[1] - 1.0 + xi_p[0]) < PLEX_C_EPS) {
        xi_p[1] = 1.0 - xi_p[0];
      } else {
        point_located = PETSC_FALSE;
      }
    }
    if (!point_located) {
      PetscPrintf(PETSC_COMM_SELF,"[Error] xi,eta = %+1.8e, %+1.8e\n",xi_p[0],xi_p[1]);
      PetscPrintf(PETSC_COMM_SELF,"[Error] Failed to locate point (%1.8e,%1.8e) in local mesh (cell %D) with triangle coords (%1.8e,%1.8e) : (%1.8e,%1.8e) : (%1.8e,%1.8e)\n",coor_p[0],coor_p[1],e,elcoor[0],elcoor[1],elcoor[2],elcoor[3],elcoor[4],elcoor[5]);
    }
    PetscCheckFalse(!point_located,PETSC_COMM_SELF,PETSC_ERR_SUP,"Failed to locate point (%1.8e,%1.8e) in local mesh (cell %D)",coor_p[0],coor_p[1],e);
    */

    Ni[0] = 1.0 - xi_p[0] - xi_p[1];
    Ni[1] = xi_p[0];
    Ni[2] = xi_p[1];

    point_located = PETSC_TRUE;
    for (k=0; k<3; k++) {
      if (PetscRealPart(Ni[k]) < -PLEX_C_EPS) point_located = PETSC_FALSE;
      if (PetscRealPart(Ni[k]) > (1.0+PLEX_C_EPS)) point_located = PETSC_FALSE;
    }
    if (!point_located) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[Error] xi,eta = %+1.8e, %+1.8e\n",(double)xi_p[0],(double)xi_p[1]);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"[Error] Failed to locate point (%1.8e,%1.8e) in local mesh (cell %D) with triangle coords (%1.8e,%1.8e) : (%1.8e,%1.8e) : (%1.8e,%1.8e)\n",(double)coor_p[0],(double)coor_p[1],e,(double)PetscRealPart(elcoor[0]),(double)PetscRealPart(elcoor[1]),(double)PetscRealPart(elcoor[2]),(double)PetscRealPart(elcoor[3]),(double)PetscRealPart(elcoor[4]),(double)PetscRealPart(elcoor[5]));CHKERRQ(ierr);
    }
    PetscCheckFalse(!point_located,PETSC_COMM_SELF,PETSC_ERR_SUP,"Failed to locate point (%1.8e,%1.8e) in local mesh (cell %D)",(double)coor_p[0],(double)coor_p[1],e);

    for (k=0; k<3; k++) {
      Ni[k] = Ni[k] * dJ;
      elfield[k] = Ni[k] * swarm_field[p];
    }
    ierr = DMPlexVecRestoreClosure(dm,coordSection,coor_l,e,NULL,&elcoor);CHKERRQ(ierr);

    ierr = DMPlexVecSetClosure(dm, NULL,v_field_l, e, elfield, ADD_VALUES);CHKERRQ(ierr);
    ierr = DMPlexVecSetClosure(dm, NULL,denom_l, e, Ni, ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);

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

PetscErrorCode private_DMSwarmProjectFields_PLEX(DM swarm,DM celldm,PetscInt project_type,PetscInt nfields,DMSwarmDataField dfield[],Vec vecs[])
{
  PetscErrorCode ierr;
  PetscInt       f,dim;

  PetscFunctionBegin;
  ierr = DMGetDimension(swarm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2:
      for (f=0; f<nfields; f++) {
        PetscReal *swarm_field;

        ierr = DMSwarmDataFieldGetEntries(dfield[f],(void**)&swarm_field);CHKERRQ(ierr);
        ierr = DMSwarmProjectField_ApproxP1_PLEX_2D(swarm,swarm_field,celldm,vecs[f]);CHKERRQ(ierr);
      }
      break;
    case 3:
      SETERRQ(PetscObjectComm((PetscObject)swarm),PETSC_ERR_SUP,"No support for 3D");
    default:
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM dm,DM dmc,PetscInt npoints,PetscReal xi[])
{
  PetscBool       is_simplex,is_tensorcell;
  PetscErrorCode  ierr;
  PetscInt        dim,nfaces,ps,pe,p,d,nbasis,pcnt,e,k,nel;
  PetscFE         fe;
  PetscQuadrature quadrature;
  PetscTabulation T;
  PetscReal       *xiq;
  Vec             coorlocal;
  PetscSection    coordSection;
  PetscScalar     *elcoor = NULL;
  PetscReal       *swarm_coor;
  PetscInt        *swarm_cellid;

  PetscFunctionBegin;
  ierr = DMGetDimension(dmc,&dim);CHKERRQ(ierr);

  is_simplex = PETSC_FALSE;
  is_tensorcell = PETSC_FALSE;
  ierr = DMPlexGetHeightStratum(dmc,0,&ps,&pe);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dmc, ps, &nfaces);CHKERRQ(ierr);

  if (nfaces == (dim+1)) { is_simplex = PETSC_TRUE; }

  switch (dim) {
    case 2:
      if (nfaces == 4) { is_tensorcell = PETSC_TRUE; }
      break;
    case 3:
      if (nfaces == 6) { is_tensorcell = PETSC_TRUE; }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only support for 2D, 3D");
  }

  /* check points provided fail inside the reference cell */
  if (is_simplex) {
    for (p=0; p<npoints; p++) {
      PetscReal sum;
      for (d=0; d<dim; d++) {
        PetscCheckFalse(xi[dim*p+d] < -1.0,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Points do not fail inside the simplex domain");
      }
      sum = 0.0;
      for (d=0; d<dim; d++) {
        sum += xi[dim*p+d];
      }
      PetscCheckFalse(sum > 0.0,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Points do not fail inside the simplex domain");
    }
  } else if (is_tensorcell) {
    for (p=0; p<npoints; p++) {
      for (d=0; d<dim; d++) {
        PetscCheckFalse(PetscAbsReal(xi[dim*p+d]) > 1.0,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Points do not fail inside the tensor domain [-1,1]^d");
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only support for d-simplex and d-tensorcell");

  ierr = PetscQuadratureCreate(PetscObjectComm((PetscObject)dm),&quadrature);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints*dim,&xiq);CHKERRQ(ierr);
  ierr = PetscArraycpy(xiq,xi,npoints*dim);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(quadrature,dim,1,npoints,(const PetscReal*)xiq,NULL);CHKERRQ(ierr);
  ierr = private_PetscFECreateDefault_scalar_pk1(dmc, dim, is_simplex, 0, &fe);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe,quadrature);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe,&nbasis);CHKERRQ(ierr);
  ierr = PetscFEGetCellTabulation(fe, 1, &T);CHKERRQ(ierr);

  /* for each cell, interpolate coordaintes and insert the interpolated points coordinates into swarm */
  /* 0->cell, 1->edge, 2->vert */
  ierr = DMPlexGetHeightStratum(dmc,0,&ps,&pe);CHKERRQ(ierr);
  nel = pe - ps;

  ierr = DMSwarmSetLocalSizes(dm,npoints*nel,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dmc,&coorlocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmc,&coordSection);CHKERRQ(ierr);

  pcnt = 0;
  for (e=0; e<nel; e++) {
    ierr = DMPlexVecGetClosure(dmc,coordSection,coorlocal,ps+e,NULL,&elcoor);CHKERRQ(ierr);

    for (p=0; p<npoints; p++) {
      for (d=0; d<dim; d++) {
        swarm_coor[dim*pcnt+d] = 0.0;
        for (k=0; k<nbasis; k++) {
          swarm_coor[dim*pcnt+d] += T->T[0][p*nbasis + k] * PetscRealPart(elcoor[dim*k+d]);
        }
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
    ierr = DMPlexVecRestoreClosure(dmc,coordSection,coorlocal,ps+e,NULL,&elcoor);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);

  ierr = PetscQuadratureDestroy(&quadrature);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
