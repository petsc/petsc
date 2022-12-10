#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmswarmimpl.h>
#include "../src/dm/impls/swarm/data_bucket.h"

PetscErrorCode private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(PetscInt dim, PetscInt np[], PetscInt *_npoints, PetscReal **_xi)
{
  PetscReal *xi;
  PetscInt   d, npoints = 0, cnt;
  PetscReal  ds[] = {0.0, 0.0, 0.0};
  PetscInt   ii, jj, kk;

  PetscFunctionBegin;
  switch (dim) {
  case 1:
    npoints = np[0];
    break;
  case 2:
    npoints = np[0] * np[1];
    break;
  case 3:
    npoints = np[0] * np[1] * np[2];
    break;
  }
  for (d = 0; d < dim; d++) ds[d] = 2.0 / ((PetscReal)np[d]);

  PetscCall(PetscMalloc1(dim * npoints, &xi));
  switch (dim) {
  case 1:
    cnt = 0;
    for (ii = 0; ii < np[0]; ii++) {
      xi[dim * cnt + 0] = -1.0 + 0.5 * ds[d] + ii * ds[0];
      cnt++;
    }
    break;

  case 2:
    cnt = 0;
    for (jj = 0; jj < np[1]; jj++) {
      for (ii = 0; ii < np[0]; ii++) {
        xi[dim * cnt + 0] = -1.0 + 0.5 * ds[0] + ii * ds[0];
        xi[dim * cnt + 1] = -1.0 + 0.5 * ds[1] + jj * ds[1];
        cnt++;
      }
    }
    break;

  case 3:
    cnt = 0;
    for (kk = 0; kk < np[2]; kk++) {
      for (jj = 0; jj < np[1]; jj++) {
        for (ii = 0; ii < np[0]; ii++) {
          xi[dim * cnt + 0] = -1.0 + 0.5 * ds[0] + ii * ds[0];
          xi[dim * cnt + 1] = -1.0 + 0.5 * ds[1] + jj * ds[1];
          xi[dim * cnt + 2] = -1.0 + 0.5 * ds[2] + kk * ds[2];
          cnt++;
        }
      }
    }
    break;
  }
  *_npoints = npoints;
  *_xi      = xi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode private_DMSwarmCreateCellLocalCoords_DA_Q1_Gauss(PetscInt dim, PetscInt np_1d, PetscInt *_npoints, PetscReal **_xi)
{
  PetscQuadrature  quadrature;
  const PetscReal *quadrature_xi;
  PetscReal       *xi;
  PetscInt         d, q, npoints_q;

  PetscFunctionBegin;
  PetscCall(PetscDTGaussTensorQuadrature(dim, 1, np_1d, -1.0, 1.0, &quadrature));
  PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &npoints_q, &quadrature_xi, NULL));
  PetscCall(PetscMalloc1(dim * npoints_q, &xi));
  for (q = 0; q < npoints_q; q++) {
    for (d = 0; d < dim; d++) xi[dim * q + d] = quadrature_xi[dim * q + d];
  }
  PetscCall(PetscQuadratureDestroy(&quadrature));
  *_npoints = npoints_q;
  *_xi      = xi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA_Q1(DM dm, DM dmc, PetscInt npoints, DMSwarmPICLayoutType layout)
{
  PetscInt           dim, npoints_q;
  PetscInt           nel, npe, e, q, k, d;
  const PetscInt    *element_list;
  PetscReal        **basis;
  PetscReal         *xi;
  Vec                coor;
  const PetscScalar *_coor;
  PetscReal         *elcoor;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscInt           pcnt;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  switch (layout) {
  case DMSWARMPIC_LAYOUT_REGULAR: {
    PetscInt np_dir[3];
    np_dir[0] = np_dir[1] = np_dir[2] = npoints;
    PetscCall(private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(dim, np_dir, &npoints_q, &xi));
  } break;
  case DMSWARMPIC_LAYOUT_GAUSS:
    PetscCall(private_DMSwarmCreateCellLocalCoords_DA_Q1_Gauss(dim, npoints, &npoints_q, &xi));
    break;

  case DMSWARMPIC_LAYOUT_SUBDIVISION: {
    PetscInt s, nsub;
    PetscInt np_dir[3];
    nsub      = npoints;
    np_dir[0] = 1;
    for (s = 0; s < nsub; s++) np_dir[0] *= 2;
    np_dir[1] = np_dir[0];
    np_dir[2] = np_dir[0];
    PetscCall(private_DMSwarmCreateCellLocalCoords_DA_Q1_Regular(dim, np_dir, &npoints_q, &xi));
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "A valid DMSwarmPIC layout must be provided");
  }

  PetscCall(DMDAGetElements(dmc, &nel, &npe, &element_list));
  PetscCall(PetscMalloc1(dim * npe, &elcoor));
  PetscCall(PetscMalloc1(npoints_q, &basis));
  for (q = 0; q < npoints_q; q++) {
    PetscCall(PetscMalloc1(npe, &basis[q]));

    switch (dim) {
    case 1:
      basis[q][0] = 0.5 * (1.0 - xi[dim * q + 0]);
      basis[q][1] = 0.5 * (1.0 + xi[dim * q + 0]);
      break;
    case 2:
      basis[q][0] = 0.25 * (1.0 - xi[dim * q + 0]) * (1.0 - xi[dim * q + 1]);
      basis[q][1] = 0.25 * (1.0 + xi[dim * q + 0]) * (1.0 - xi[dim * q + 1]);
      basis[q][2] = 0.25 * (1.0 + xi[dim * q + 0]) * (1.0 + xi[dim * q + 1]);
      basis[q][3] = 0.25 * (1.0 - xi[dim * q + 0]) * (1.0 + xi[dim * q + 1]);
      break;

    case 3:
      basis[q][0] = 0.125 * (1.0 - xi[dim * q + 0]) * (1.0 - xi[dim * q + 1]) * (1.0 - xi[dim * q + 2]);
      basis[q][1] = 0.125 * (1.0 + xi[dim * q + 0]) * (1.0 - xi[dim * q + 1]) * (1.0 - xi[dim * q + 2]);
      basis[q][2] = 0.125 * (1.0 + xi[dim * q + 0]) * (1.0 + xi[dim * q + 1]) * (1.0 - xi[dim * q + 2]);
      basis[q][3] = 0.125 * (1.0 - xi[dim * q + 0]) * (1.0 + xi[dim * q + 1]) * (1.0 - xi[dim * q + 2]);
      basis[q][4] = 0.125 * (1.0 - xi[dim * q + 0]) * (1.0 - xi[dim * q + 1]) * (1.0 + xi[dim * q + 2]);
      basis[q][5] = 0.125 * (1.0 + xi[dim * q + 0]) * (1.0 - xi[dim * q + 1]) * (1.0 + xi[dim * q + 2]);
      basis[q][6] = 0.125 * (1.0 + xi[dim * q + 0]) * (1.0 + xi[dim * q + 1]) * (1.0 + xi[dim * q + 2]);
      basis[q][7] = 0.125 * (1.0 - xi[dim * q + 0]) * (1.0 + xi[dim * q + 1]) * (1.0 + xi[dim * q + 2]);
      break;
    }
  }

  PetscCall(DMSwarmSetLocalSizes(dm, npoints_q * nel, -1));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));

  PetscCall(DMGetCoordinatesLocal(dmc, &coor));
  PetscCall(VecGetArrayRead(coor, &_coor));
  pcnt = 0;
  for (e = 0; e < nel; e++) {
    const PetscInt *element = &element_list[npe * e];

    for (k = 0; k < npe; k++) {
      for (d = 0; d < dim; d++) elcoor[dim * k + d] = PetscRealPart(_coor[dim * element[k] + d]);
    }

    for (q = 0; q < npoints_q; q++) {
      for (d = 0; d < dim; d++) swarm_coor[dim * pcnt + d] = 0.0;
      for (k = 0; k < npe; k++) {
        for (d = 0; d < dim; d++) swarm_coor[dim * pcnt + d] += basis[q][k] * elcoor[dim * k + d];
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
  }
  PetscCall(VecRestoreArrayRead(coor, &_coor));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMDARestoreElements(dmc, &nel, &npe, &element_list));

  PetscCall(PetscFree(xi));
  PetscCall(PetscFree(elcoor));
  for (q = 0; q < npoints_q; q++) PetscCall(PetscFree(basis[q]));
  PetscCall(PetscFree(basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA(DM dm, DM celldm, DMSwarmPICLayoutType layout, PetscInt layout_param)
{
  DMDAElementType etype;
  PetscInt        dim;

  PetscFunctionBegin;
  PetscCall(DMDAGetElementType(celldm, &etype));
  PetscCall(DMGetDimension(celldm, &dim));
  switch (etype) {
  case DMDA_ELEMENT_P1:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DA support is not currently available for DMDA_ELEMENT_P1");
  case DMDA_ELEMENT_Q1:
    PetscCheck(dim != 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Support only available for dim = 2, 3");
    PetscCall(private_DMSwarmInsertPointsUsingCellDM_DA_Q1(dm, celldm, layout_param, layout));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmProjectField_ApproxQ1_DA_2D(DM swarm, PetscReal *swarm_field, DM dm, Vec v_field)
{
  Vec                v_field_l, denom_l, coor_l, denom;
  PetscScalar       *_field_l, *_denom_l;
  PetscInt           k, p, e, npoints, nel, npe;
  PetscInt          *mpfield_cell;
  PetscReal         *mpfield_coor;
  const PetscInt    *element_list;
  const PetscInt    *element;
  PetscScalar        xi_p[2], Ni[4];
  const PetscScalar *_coor;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(v_field));

  PetscCall(DMGetLocalVector(dm, &v_field_l));
  PetscCall(DMGetGlobalVector(dm, &denom));
  PetscCall(DMGetLocalVector(dm, &denom_l));
  PetscCall(VecZeroEntries(v_field_l));
  PetscCall(VecZeroEntries(denom));
  PetscCall(VecZeroEntries(denom_l));

  PetscCall(VecGetArray(v_field_l, &_field_l));
  PetscCall(VecGetArray(denom_l, &_denom_l));

  PetscCall(DMGetCoordinatesLocal(dm, &coor_l));
  PetscCall(VecGetArrayRead(coor_l, &_coor));

  PetscCall(DMDAGetElements(dm, &nel, &npe, &element_list));
  PetscCall(DMSwarmGetLocalSize(swarm, &npoints));
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&mpfield_coor));
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_cellid, NULL, NULL, (void **)&mpfield_cell));

  for (p = 0; p < npoints; p++) {
    PetscReal         *coor_p;
    const PetscScalar *x0;
    const PetscScalar *x2;
    PetscScalar        dx[2];

    e       = mpfield_cell[p];
    coor_p  = &mpfield_coor[2 * p];
    element = &element_list[npe * e];

    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0 = &_coor[2 * element[0]];
    x2 = &_coor[2 * element[2]];

    dx[0] = x2[0] - x0[0];
    dx[1] = x2[1] - x0[1];

    xi_p[0] = 2.0 * (coor_p[0] - x0[0]) / dx[0] - 1.0;
    xi_p[1] = 2.0 * (coor_p[1] - x0[1]) / dx[1] - 1.0;

    /* evaluate basis functions */
    Ni[0] = 0.25 * (1.0 - xi_p[0]) * (1.0 - xi_p[1]);
    Ni[1] = 0.25 * (1.0 + xi_p[0]) * (1.0 - xi_p[1]);
    Ni[2] = 0.25 * (1.0 + xi_p[0]) * (1.0 + xi_p[1]);
    Ni[3] = 0.25 * (1.0 - xi_p[0]) * (1.0 + xi_p[1]);

    for (k = 0; k < npe; k++) {
      _field_l[element[k]] += Ni[k] * swarm_field[p];
      _denom_l[element[k]] += Ni[k];
    }
  }

  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_cellid, NULL, NULL, (void **)&mpfield_cell));
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&mpfield_coor));
  PetscCall(DMDARestoreElements(dm, &nel, &npe, &element_list));
  PetscCall(VecRestoreArrayRead(coor_l, &_coor));
  PetscCall(VecRestoreArray(v_field_l, &_field_l));
  PetscCall(VecRestoreArray(denom_l, &_denom_l));

  PetscCall(DMLocalToGlobalBegin(dm, v_field_l, ADD_VALUES, v_field));
  PetscCall(DMLocalToGlobalEnd(dm, v_field_l, ADD_VALUES, v_field));
  PetscCall(DMLocalToGlobalBegin(dm, denom_l, ADD_VALUES, denom));
  PetscCall(DMLocalToGlobalEnd(dm, denom_l, ADD_VALUES, denom));

  PetscCall(VecPointwiseDivide(v_field, v_field, denom));

  PetscCall(DMRestoreLocalVector(dm, &v_field_l));
  PetscCall(DMRestoreLocalVector(dm, &denom_l));
  PetscCall(DMRestoreGlobalVector(dm, &denom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode private_DMSwarmProjectFields_DA(DM swarm, DM celldm, PetscInt project_type, PetscInt nfields, DMSwarmDataField dfield[], Vec vecs[])
{
  PetscInt        f, dim;
  DMDAElementType etype;

  PetscFunctionBegin;
  PetscCall(DMDAGetElementType(celldm, &etype));
  PetscCheck(etype != DMDA_ELEMENT_P1, PetscObjectComm((PetscObject)swarm), PETSC_ERR_SUP, "Only Q1 DMDA supported");

  PetscCall(DMGetDimension(swarm, &dim));
  switch (dim) {
  case 2:
    for (f = 0; f < nfields; f++) {
      PetscReal *swarm_field;

      PetscCall(DMSwarmDataFieldGetEntries(dfield[f], (void **)&swarm_field));
      PetscCall(DMSwarmProjectField_ApproxQ1_DA_2D(swarm, swarm_field, celldm, vecs[f]));
    }
    break;
  case 3:
    SETERRQ(PetscObjectComm((PetscObject)swarm), PETSC_ERR_SUP, "No support for 3D");
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
