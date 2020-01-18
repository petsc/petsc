#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/
#include <petsc/private/vecimpl.h>

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>
#include <moab/ReadUtilIface.hpp>
#include <moab/MergeMesh.hpp>
#include <moab/CN.hpp>

typedef struct {
  // options
  PetscInt  A, B, C, M, N, K, dim;
  PetscInt  blockSizeVertexXYZ[3];              // Number of element blocks per partition
  PetscInt  blockSizeElementXYZ[3];
  PetscReal xyzbounds[6]; // the physical size of the domain
  bool      newMergeMethod, keep_skins, simplex, adjEnts;

  // compute params
  PetscReal dx, dy, dz;
  PetscInt  NX, NY, NZ, nex, ney, nez;
  PetscInt  q, xstride, ystride, zstride;
  PetscBool usrxyzgrid, usrprocgrid, usrrefgrid;
  PetscInt  fraction, remainder, cumfraction;
  PetscLogEvent generateMesh, generateElements, generateVertices, parResolve;

} DMMoabMeshGeneratorCtx;


PetscInt DMMoab_SetTensorElementConnectivity_Private(DMMoabMeshGeneratorCtx& genCtx, PetscInt offset, PetscInt corner, std::vector<PetscInt>& subent_conn, moab::EntityHandle *connectivity)
{
  switch (genCtx.dim) {
  case 1:
    subent_conn.resize(2);
    moab::CN::SubEntityVertexIndices(moab::MBEDGE, 1, 0, subent_conn.data());
    connectivity[offset + subent_conn[0]] = corner;
    connectivity[offset + subent_conn[1]] = corner + 1;
    break;
  case 2:
    subent_conn.resize(4);
    moab::CN::SubEntityVertexIndices(moab::MBQUAD, 2, 0, subent_conn.data());
    connectivity[offset + subent_conn[0]] = corner;
    connectivity[offset + subent_conn[1]] = corner + 1;
    connectivity[offset + subent_conn[2]] = corner + 1 + genCtx.ystride;
    connectivity[offset + subent_conn[3]] = corner + genCtx.ystride;
    break;
  case 3:
  default:
    subent_conn.resize(8);
    moab::CN::SubEntityVertexIndices(moab::MBHEX, 3, 0, subent_conn.data());
    connectivity[offset + subent_conn[0]] = corner;
    connectivity[offset + subent_conn[1]] = corner + 1;
    connectivity[offset + subent_conn[2]] = corner + 1 + genCtx.ystride;
    connectivity[offset + subent_conn[3]] = corner + genCtx.ystride;
    connectivity[offset + subent_conn[4]] = corner + genCtx.zstride;
    connectivity[offset + subent_conn[5]] = corner + 1 + genCtx.zstride;
    connectivity[offset + subent_conn[6]] = corner + 1 + genCtx.ystride + genCtx.zstride;
    connectivity[offset + subent_conn[7]] = corner + genCtx.ystride + genCtx.zstride;
    break;
  }
  return subent_conn.size();
}


PetscInt DMMoab_SetSimplexElementConnectivity_Private(DMMoabMeshGeneratorCtx& genCtx, PetscInt subelem, PetscInt offset, PetscInt corner, std::vector<PetscInt>& subent_conn, moab::EntityHandle *connectivity)
{
  PetscInt A, B, C, D, E, F, G, H, M;
  const PetscInt trigen_opts = 1; /* 1 - Aligned diagonally to right, 2 - Aligned diagonally to left, 3 - 4 elements per quad */
  A = corner;
  B = corner + 1;
  switch (genCtx.dim) {
  case 1:
    subent_conn.resize(2);  /* only linear EDGE supported now */
    moab::CN::SubEntityVertexIndices(moab::MBEDGE, 1, 0, subent_conn.data());
    connectivity[offset + subent_conn[0]] = A;
    connectivity[offset + subent_conn[1]] = B;
    break;
  case 2:
    C = corner + 1 + genCtx.ystride;
    D = corner +     genCtx.ystride;
    M = corner + 0.5; /* technically -- need to modify vertex generation */
    subent_conn.resize(3);  /* only linear TRI supported */
    moab::CN::SubEntityVertexIndices(moab::MBTRI, 2, 0, subent_conn.data());
    if (trigen_opts == 1) {
      if (subelem) { /* 0 1 2 of a QUAD */
        connectivity[offset + subent_conn[0]] = B;
        connectivity[offset + subent_conn[1]] = C;
        connectivity[offset + subent_conn[2]] = A;
      }
      else {        /* 2 3 0 of a QUAD */
        connectivity[offset + subent_conn[0]] = D;
        connectivity[offset + subent_conn[1]] = A;
        connectivity[offset + subent_conn[2]] = C;
      }
    }
    else if (trigen_opts == 2) {
      if (subelem) { /* 0 1 2 of a QUAD */
        connectivity[offset + subent_conn[0]] = A;
        connectivity[offset + subent_conn[1]] = B;
        connectivity[offset + subent_conn[2]] = D;
      }
      else {        /* 2 3 0 of a QUAD */
        connectivity[offset + subent_conn[0]] = C;
        connectivity[offset + subent_conn[1]] = D;
        connectivity[offset + subent_conn[2]] = B;
      }
    }
    else {
      switch (subelem) { /* 0 1 2 of a QUAD */
      case 0:
        connectivity[offset + subent_conn[0]] = A;
        connectivity[offset + subent_conn[1]] = B;
        connectivity[offset + subent_conn[2]] = M;
        break;
      case 1:
        connectivity[offset + subent_conn[0]] = B;
        connectivity[offset + subent_conn[1]] = C;
        connectivity[offset + subent_conn[2]] = M;
        break;
      case 2:
        connectivity[offset + subent_conn[0]] = C;
        connectivity[offset + subent_conn[1]] = D;
        connectivity[offset + subent_conn[2]] = M;
        break;
      case 3:
        connectivity[offset + subent_conn[0]] = D;
        connectivity[offset + subent_conn[1]] = A;
        connectivity[offset + subent_conn[2]] = M;
        break;
      }
    }
    break;
  case 3:
  default:
    C = corner + 1 + genCtx.ystride;
    D = corner +     genCtx.ystride;
    E = corner +                      genCtx.zstride;
    F = corner + 1 +                  genCtx.zstride;
    G = corner + 1 + genCtx.ystride + genCtx.zstride;
    H = corner +     genCtx.ystride + genCtx.zstride;
    subent_conn.resize(4);  /* only linear TET supported */
    moab::CN::SubEntityVertexIndices(moab::MBTET, 3, 0, subent_conn.data());
    switch (subelem) {
    case 0: /* 4 3 7 6 of a HEX */
      connectivity[offset + subent_conn[0]] = E;
      connectivity[offset + subent_conn[1]] = D;
      connectivity[offset + subent_conn[2]] = H;
      connectivity[offset + subent_conn[3]] = G;
      break;
    case 1: /* 0 1 2 5 of a HEX */
      connectivity[offset + subent_conn[0]] = A;
      connectivity[offset + subent_conn[1]] = B;
      connectivity[offset + subent_conn[2]] = C;
      connectivity[offset + subent_conn[3]] = F;
      break;
    case 2: /* 0 3 4 5 of a HEX */
      connectivity[offset + subent_conn[0]] = A;
      connectivity[offset + subent_conn[1]] = D;
      connectivity[offset + subent_conn[2]] = E;
      connectivity[offset + subent_conn[3]] = F;
      break;
    case 3: /* 2 6 3 5 of a HEX */
      connectivity[offset + subent_conn[0]] = C;
      connectivity[offset + subent_conn[1]] = G;
      connectivity[offset + subent_conn[2]] = D;
      connectivity[offset + subent_conn[3]] = F;
      break;
    case 4: /* 0 2 3 5 of a HEX */
      connectivity[offset + subent_conn[0]] = A;
      connectivity[offset + subent_conn[1]] = C;
      connectivity[offset + subent_conn[2]] = D;
      connectivity[offset + subent_conn[3]] = F;
      break;
    case 5: /* 3 6 4 5 of a HEX */
      connectivity[offset + subent_conn[0]] = D;
      connectivity[offset + subent_conn[1]] = G;
      connectivity[offset + subent_conn[2]] = E;
      connectivity[offset + subent_conn[3]] = F;
      break;
    }
    break;
  }
  return subent_conn.size();
}


std::pair<PetscInt, PetscInt> DMMoab_SetElementConnectivity_Private(DMMoabMeshGeneratorCtx& genCtx, PetscInt offset, PetscInt corner, moab::EntityHandle *connectivity)
{
  PetscInt vcount = 0;
  PetscInt simplices_per_tensor[4] = {0, 1, 2, 6};
  std::vector<PetscInt> subent_conn;  /* only linear edge, tri, tet supported now */
  subent_conn.reserve(27);
  PetscInt m, subelem;
  if (genCtx.simplex) {
    subelem = simplices_per_tensor[genCtx.dim];
    for (m = 0; m < subelem; m++) {
      vcount = DMMoab_SetSimplexElementConnectivity_Private(genCtx, m, offset, corner, subent_conn, connectivity);
      offset += vcount;
    }
  }
  else {
    subelem = 1;
    vcount = DMMoab_SetTensorElementConnectivity_Private(genCtx, offset, corner, subent_conn, connectivity);
  }
  return std::pair<PetscInt, PetscInt>(vcount * subelem, subelem);
}


PetscErrorCode DMMoab_GenerateVertices_Private(moab::Interface *mbImpl, moab::ReadUtilIface *iface, DMMoabMeshGeneratorCtx& genCtx, PetscInt m, PetscInt n, PetscInt k,
    PetscInt a, PetscInt b, PetscInt c, moab::Tag& global_id_tag, moab::EntityHandle& startv, moab::Range& uverts)
{
  PetscInt x, y, z, ix, nnodes;
  PetscErrorCode ierr;
  PetscInt ii, jj, kk;
  std::vector<PetscReal*> arrays;
  PetscInt* gids;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  /* we will generate (q*block+1)^3 vertices, and block^3 hexas; q is 1 for linear, 2 for quadratic
   * the global id of the vertices will come from m, n, k, a, b, c
   * x will vary from  m*A*q*block + a*q*block to m*A*q*block+(a+1)*q*block etc.
   */
  nnodes = genCtx.blockSizeVertexXYZ[0] * (genCtx.dim > 1 ? genCtx.blockSizeVertexXYZ[1] * (genCtx.dim > 2 ? genCtx.blockSizeVertexXYZ[2] : 1) : 1);
  ierr = PetscMalloc1(nnodes, &gids);CHKERRQ(ierr);

  merr = iface->get_node_coords(3, nnodes, 0, startv, arrays);MBERR("Can't get node coords.", merr);

  /* will start with the lower corner: */
  /* x = ( m * genCtx.A + a ) * genCtx.q * genCtx.blockSizeElementXYZ[0]; */
  /* y = ( n * genCtx.B + b ) * genCtx.q * genCtx.blockSizeElementXYZ[1]; */
  /* z = ( k * genCtx.C + c ) * genCtx.q * genCtx.blockSizeElementXYZ[2]; */

  x = ( m * genCtx.A + a ) * genCtx.q;
  y = ( n * genCtx.B + b ) * genCtx.q;
  z = ( k * genCtx.C + c ) * genCtx.q;
  PetscInfo3(NULL, "Starting offset for coordinates := %d, %d, %d\n", x, y, z);
  ix = 0;
  moab::Range verts(startv, startv + nnodes - 1);
  for (kk = 0; kk < (genCtx.dim > 2 ? genCtx.blockSizeVertexXYZ[2] : 1); kk++) {
    for (jj = 0; jj < (genCtx.dim > 1 ? genCtx.blockSizeVertexXYZ[1] : 1); jj++) {
      for (ii = 0; ii < genCtx.blockSizeVertexXYZ[0]; ii++, ix++) {
        /* set coordinates for the vertices */
        arrays[0][ix] = (x + ii) * genCtx.dx + genCtx.xyzbounds[0];
        arrays[1][ix] = (y + jj) * genCtx.dy + genCtx.xyzbounds[2];
        arrays[2][ix] = (z + kk) * genCtx.dz + genCtx.xyzbounds[4];
        PetscInfo3(NULL, "Creating vertex with coordinates := %f, %f, %f\n", arrays[0][ix], arrays[1][ix], arrays[2][ix]);

        /* If we want to set some tags on the vertices -> use the following entity handle definition:
           moab::EntityHandle v = startv + ix;
        */
        /* compute the global ID for vertex */
        gids[ix] = 1 + (x + ii) + (y + jj) * genCtx.NX + (z + kk) * (genCtx.NX * genCtx.NY);
      }
    }
  }
  /* set global ID data on vertices */
  mbImpl->tag_set_data(global_id_tag, verts, &gids[0]);
  verts.swap(uverts);
  ierr = PetscFree(gids);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMMoab_GenerateElements_Private(moab::Interface* mbImpl, moab::ReadUtilIface* iface, DMMoabMeshGeneratorCtx& genCtx, PetscInt m, PetscInt n, PetscInt k,
    PetscInt a, PetscInt b, PetscInt c, moab::Tag& global_id_tag, moab::EntityHandle startv, moab::Range& cells)
{
  moab::ErrorCode merr;
  PetscInt ix, ie, xe, ye, ze;
  PetscInt ii, jj, kk, nvperelem;
  PetscInt simplices_per_tensor[4] = {0, 1, 2, 6};
  PetscInt ntensorelems = genCtx.blockSizeElementXYZ[0] * (genCtx.dim > 1 ? genCtx.blockSizeElementXYZ[1] * (genCtx.dim > 2 ? genCtx.blockSizeElementXYZ[2] : 1) : 1); /*pow(genCtx.blockSizeElement,genCtx.dim);*/
  PetscInt nelems = ntensorelems;
  moab::EntityHandle starte; /* connectivity */
  moab::EntityHandle* conn;

  PetscFunctionBegin;
  switch (genCtx.dim) {
  case 1:
    nvperelem = 2;
    merr = iface->get_element_connect(nelems, 2, moab::MBEDGE, 0, starte, conn);MBERR("Can't get EDGE2 element connectivity.", merr);
    break;
  case 2:
    if (genCtx.simplex) {
      nvperelem = 3;
      nelems = ntensorelems * simplices_per_tensor[genCtx.dim];
      merr = iface->get_element_connect(nelems, 3, moab::MBTRI, 0, starte, conn);MBERR("Can't get TRI3 element connectivity.", merr);
    }
    else {
      nvperelem = 4;
      merr = iface->get_element_connect(nelems, 4, moab::MBQUAD, 0, starte, conn);MBERR("Can't get QUAD4 element connectivity.", merr);
    }
    break;
  case 3:
  default:
    if (genCtx.simplex) {
      nvperelem = 4;
      nelems = ntensorelems * simplices_per_tensor[genCtx.dim];
      merr = iface->get_element_connect(nelems, 4, moab::MBTET, 0, starte, conn);MBERR("Can't get TET4 element connectivity.", merr);
    }
    else {
      nvperelem = 8;
      merr = iface->get_element_connect(nelems, 8, moab::MBHEX, 0, starte, conn);MBERR("Can't get HEX8 element connectivity.", merr);
    }
    break;
  }

  ix = ie = 0; /* index now in the elements, for global ids */

  /* create a temporary range to store local element handles */
  moab::Range tmp(starte, starte + nelems - 1);
  std::vector<PetscInt> gids(nelems);

  /* identify the elements at the lower corner, for their global ids */
  xe = m * genCtx.A * genCtx.blockSizeElementXYZ[0] + a * genCtx.blockSizeElementXYZ[0];
  ye = (genCtx.dim > 1 ? n * genCtx.B * genCtx.blockSizeElementXYZ[1] + b * genCtx.blockSizeElementXYZ[1] : 0);
  ze = (genCtx.dim > 2 ? k * genCtx.C * genCtx.blockSizeElementXYZ[2] + c * genCtx.blockSizeElementXYZ[2] : 0);

  /* create owned elements requested by genCtx */
  for (kk = 0; kk < (genCtx.dim > 2 ? genCtx.blockSizeElementXYZ[2] : 1); kk++) {
    for (jj = 0; jj < (genCtx.dim > 1 ? genCtx.blockSizeElementXYZ[1] : 1); jj++) {
      for (ii = 0; ii < genCtx.blockSizeElementXYZ[0]; ii++) {

        moab::EntityHandle corner = startv + genCtx.q * ii + genCtx.q * jj * genCtx.ystride + genCtx.q * kk * genCtx.zstride;

        std::pair<PetscInt, PetscInt> entoffset = DMMoab_SetElementConnectivity_Private(genCtx, ix, corner, conn);

        for (PetscInt j = 0; j < entoffset.second; j++) {
          /* The entity handle for the particular element -> if we want to set some tags is
             moab::EntityHandle eh = starte + ie + j;
          */
          gids[ie + j] = 1 + ((xe + ii) + (ye + jj) * genCtx.nex + (ze + kk) * (genCtx.nex * genCtx.ney));
          /* gids[ie+j] = ie + j + ((xe + ii) + (ye + jj) * genCtx.nex + (ze + kk) * (genCtx.nex * genCtx.ney)); */
          /* gids[ie+j] = 1 + ie; */
          /* ie++; */
        }

        ix += entoffset.first;
        ie += entoffset.second;
      }
    }
  }
  if (genCtx.adjEnts) { /* we need to update adjacencies now, because some elements are new */
    merr = iface->update_adjacencies(starte, nelems, nvperelem, conn);MBERR("Can't update adjacencies", merr);
  }
  tmp.swap(cells);
  merr = mbImpl->tag_set_data(global_id_tag, cells, &gids[0]);MBERR("Can't set global ids to elements.", merr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMMBUtil_InitializeOptions(DMMoabMeshGeneratorCtx& genCtx, PetscInt dim, PetscBool simplex, PetscInt rank, PetscInt nprocs, const PetscReal* bounds, PetscInt nelems)
{
  PetscFunctionBegin;
  /* Initialize all genCtx data */
  genCtx.dim = dim;
  genCtx.simplex = simplex;
  genCtx.newMergeMethod = genCtx.keep_skins = genCtx.adjEnts = true;
  /* determine other global quantities for the mesh used for nodes increments */
  genCtx.q = 1;
  genCtx.fraction = genCtx.remainder = genCtx.cumfraction = 0;

  if (!genCtx.usrxyzgrid) { /* not overridden by genCtx - assume nele equally and that genCtx wants a uniform cube mesh */

    genCtx.fraction = nelems / nprocs; /* partition only by the largest dimension */
    genCtx.remainder = nelems % nprocs; /* remainder after partition which gets evenly distributed by round-robin */
    genCtx.cumfraction = (rank > 0 ? (genCtx.fraction) * (rank) + (rank - 1 < genCtx.remainder ? rank : genCtx.remainder ) : 0);
    if (rank < genCtx.remainder)    /* This process gets "fraction+1" elements */
      genCtx.fraction++;

    PetscInfo3(NULL, "Fraction = %D, Remainder = %D, Cumulative fraction = %D\n", genCtx.fraction, genCtx.remainder, genCtx.cumfraction);
    switch (genCtx.dim) {
    case 1:
      genCtx.blockSizeElementXYZ[0] = genCtx.fraction;
      genCtx.blockSizeElementXYZ[1] = 1;
      genCtx.blockSizeElementXYZ[2] = 1;
      break;
    case 2:
      genCtx.blockSizeElementXYZ[0] = nelems;
      genCtx.blockSizeElementXYZ[1] = genCtx.fraction;
      genCtx.blockSizeElementXYZ[2] = 1;
      break;
    case 3:
    default:
      genCtx.blockSizeElementXYZ[0] = nelems;
      genCtx.blockSizeElementXYZ[1] = nelems;
      genCtx.blockSizeElementXYZ[2] = genCtx.fraction;
      break;
    }
  }

  /* partition only by the largest dimension */
  /* Total number of local elements := genCtx.blockSizeElementXYZ[0]*(genCtx.dim>1? genCtx.blockSizeElementXYZ[1]*(genCtx.dim>2 ? genCtx.blockSizeElementXYZ[2]:1) :1); */
  if (bounds) {
    for (PetscInt i = 0; i < 6; i++)
      genCtx.xyzbounds[i] = bounds[i];
  }
  else {
    genCtx.xyzbounds[0] = genCtx.xyzbounds[2] = genCtx.xyzbounds[4] = 0.0;
    genCtx.xyzbounds[1] = genCtx.xyzbounds[3] = genCtx.xyzbounds[5] = 1.0;
  }

  if (!genCtx.usrprocgrid) {
    switch (genCtx.dim) {
    case 1:
      genCtx.M = nprocs;
      genCtx.N = genCtx.K = 1;
      break;
    case 2:
      genCtx.N = nprocs;
      genCtx.M = genCtx.K = 1;
      break;
    default:
      genCtx.K = nprocs;
      genCtx.M = genCtx.N = 1;
      break;
    }
  }

  if (!genCtx.usrrefgrid) {
    genCtx.A = genCtx.B = genCtx.C = 1;
  }

  /* more default values */
  genCtx.nex = genCtx.ney = genCtx.nez = 0;
  genCtx.xstride = genCtx.ystride = genCtx.zstride = 0;
  genCtx.NX = genCtx.NY = genCtx.NZ = 0;
  genCtx.nex = genCtx.ney = genCtx.nez = 0;
  genCtx.blockSizeVertexXYZ[0] = genCtx.blockSizeVertexXYZ[1] = genCtx.blockSizeVertexXYZ[2] = 1;

  switch (genCtx.dim) {
  case 3:
    genCtx.blockSizeVertexXYZ[0] = genCtx.q * genCtx.blockSizeElementXYZ[0] + 1;
    genCtx.blockSizeVertexXYZ[1] = genCtx.q * genCtx.blockSizeElementXYZ[1] + 1;
    genCtx.blockSizeVertexXYZ[2] = genCtx.q * genCtx.blockSizeElementXYZ[2] + 1;

    genCtx.nex = genCtx.M * genCtx.A * genCtx.blockSizeElementXYZ[0];   /* number of elements in x direction, used for global id on element */
    genCtx.dx = (genCtx.xyzbounds[1] - genCtx.xyzbounds[0]) / (nelems * genCtx.q); /* distance between 2 nodes in x direction */
    genCtx.NX = (genCtx.q * genCtx.nex + 1);
    genCtx.xstride = 1;
    genCtx.ney = genCtx.N * genCtx.B * genCtx.blockSizeElementXYZ[1];  /* number of elements in y direction  .... */
    genCtx.dy = (genCtx.xyzbounds[3] - genCtx.xyzbounds[2]) / (nelems * genCtx.q); /* distance between 2 nodes in y direction */
    genCtx.NY = (genCtx.q * genCtx.ney + 1);
    genCtx.ystride = genCtx.blockSizeVertexXYZ[0];
    genCtx.nez = genCtx.K * genCtx.C * genCtx.blockSizeElementXYZ[2];  /* number of elements in z direction  .... */
    genCtx.dz = (genCtx.xyzbounds[5] - genCtx.xyzbounds[4]) / (nelems * genCtx.q); /* distance between 2 nodes in z direction */
    genCtx.NZ = (genCtx.q * genCtx.nez + 1);
    genCtx.zstride = genCtx.blockSizeVertexXYZ[0] * genCtx.blockSizeVertexXYZ[1];
    break;
  case 2:
    genCtx.blockSizeVertexXYZ[0] = genCtx.q * genCtx.blockSizeElementXYZ[0] + 1;
    genCtx.blockSizeVertexXYZ[1] = genCtx.q * genCtx.blockSizeElementXYZ[1] + 1;
    genCtx.blockSizeVertexXYZ[2] = 0;

    genCtx.nex = genCtx.M * genCtx.A * genCtx.blockSizeElementXYZ[0];   /* number of elements in x direction, used for global id on element */
    genCtx.dx = (genCtx.xyzbounds[1] - genCtx.xyzbounds[0]) / (genCtx.nex * genCtx.q); /* distance between 2 nodes in x direction */
    genCtx.NX = (genCtx.q * genCtx.nex + 1);
    genCtx.xstride = 1;
    genCtx.ney = genCtx.N * genCtx.B * genCtx.blockSizeElementXYZ[1];  /* number of elements in y direction  .... */
    genCtx.dy = (genCtx.xyzbounds[3] - genCtx.xyzbounds[2]) / (nelems * genCtx.q); /* distance between 2 nodes in y direction */
    genCtx.NY = (genCtx.q * genCtx.ney + 1);
    genCtx.ystride = genCtx.blockSizeVertexXYZ[0];
    break;
  case 1:
    genCtx.blockSizeVertexXYZ[1] = genCtx.blockSizeVertexXYZ[2] = 0;
    genCtx.blockSizeVertexXYZ[0] = genCtx.q * genCtx.blockSizeElementXYZ[0] + 1;

    genCtx.nex = genCtx.M * genCtx.A * genCtx.blockSizeElementXYZ[0];   /* number of elements in x direction, used for global id on element */
    genCtx.dx = (genCtx.xyzbounds[1] - genCtx.xyzbounds[0]) / (nelems * genCtx.q); /* distance between 2 nodes in x direction */
    genCtx.NX = (genCtx.q * genCtx.nex + 1);
    genCtx.xstride = 1;
    break;
  }

  /* Lets check for some valid input */
  if (genCtx.dim < 1 || genCtx.dim > 3) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid topological dimension specified: %d.\n", genCtx.dim);
  if (genCtx.M * genCtx.N * genCtx.K != nprocs) SETERRQ4(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid [m, n, k] data: %d, %d, %d. Product must be equal to global size = %d.\n", genCtx.M, genCtx.N, genCtx.K, nprocs);
  /* validate the bounds data */
  if (genCtx.xyzbounds[0] >= genCtx.xyzbounds[1]) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "X-dim: Left boundary cannot be greater than right. [%G >= %G]\n", genCtx.xyzbounds[0], genCtx.xyzbounds[1]);
  if (genCtx.dim > 1 && (genCtx.xyzbounds[2] >= genCtx.xyzbounds[3])) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Y-dim: Left boundary cannot be greater than right. [%G >= %G]\n", genCtx.xyzbounds[2], genCtx.xyzbounds[3]);
  if (genCtx.dim > 2 && (genCtx.xyzbounds[4] >= genCtx.xyzbounds[5])) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Z-dim: Left boundary cannot be greater than right. [%G >= %G]\n", genCtx.xyzbounds[4], genCtx.xyzbounds[5]);

  PetscInfo3(NULL, "Local elements:= %d, %d, %d\n", genCtx.blockSizeElementXYZ[0], genCtx.blockSizeElementXYZ[1], genCtx.blockSizeElementXYZ[2]);
  PetscInfo3(NULL, "Local vertices:= %d, %d, %d\n", genCtx.blockSizeVertexXYZ[0], genCtx.blockSizeVertexXYZ[1], genCtx.blockSizeVertexXYZ[2]);
  PetscInfo3(NULL, "Local blocks/processors := %d, %d, %d\n", genCtx.A, genCtx.B, genCtx.C);
  PetscInfo3(NULL, "Local processors := %d, %d, %d\n", genCtx.M, genCtx.N, genCtx.K);
  PetscInfo3(NULL, "Local nexyz:= %d, %d, %d\n", genCtx.nex, genCtx.ney, genCtx.nez);
  PetscInfo3(NULL, "Local delxyz:= %g, %g, %g\n", genCtx.dx, genCtx.dy, genCtx.dz);
  PetscInfo3(NULL, "Local strides:= %d, %d, %d\n", genCtx.xstride, genCtx.ystride, genCtx.zstride);
  PetscFunctionReturn(0);
}

/*@C
  DMMoabCreateBoxMesh - Creates a mesh on the tensor product (box) of intervals with genCtx specified bounds.

  Collective

  Input Parameters:
+ comm - The communicator for the DM object
. dim - The spatial dimension
. bounds - The bounds of the box specified with [x-left, x-right, y-bottom, y-top, z-bottom, z-top] depending on the spatial dimension
. nele - The number of discrete elements in each direction
- user_nghost - The number of ghosted layers needed in the partitioned mesh

  Output Parameter:
. dm  - The DM object

  Level: beginner

.seealso: DMSetType(), DMCreate(), DMMoabLoadFromFile()
@*/
PetscErrorCode DMMoabCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool useSimplex, const PetscReal* bounds, PetscInt nele, PetscInt nghost, DM *dm)
{
  PetscErrorCode         ierr;
  moab::ErrorCode        merr;
  PetscInt               a, b, c, n, global_size, global_rank;
  DM_Moab               *dmmoab;
  moab::Interface       *mbImpl;
#ifdef MOAB_HAVE_MPI
  moab::ParallelComm    *pcomm;
#endif
  moab::ReadUtilIface   *readMeshIface;
  moab::Range            verts, cells, edges, faces, adj, dim3, dim2;
  DMMoabMeshGeneratorCtx genCtx;
  const PetscInt         npts = nele + 1;    /* Number of points in every dimension */

  moab::Tag              global_id_tag, part_tag, geom_tag, mat_tag, dir_tag, neu_tag;
  moab::Range            ownedvtx, ownedelms, localvtxs, localelms;
  moab::EntityHandle     regionset;
  PetscInt               ml = 0, nl = 0, kl = 0;

  PetscFunctionBegin;
  if (dim < 1 || dim > 3) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension argument for mesh: dim=[1,3].\n");

  ierr = PetscLogEventRegister("GenerateMesh", DM_CLASSID,   &genCtx.generateMesh);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AddVertices", DM_CLASSID,   &genCtx.generateVertices);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AddElements", DM_CLASSID,   &genCtx.generateElements);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ParResolve", DM_CLASSID,   &genCtx.parResolve);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(genCtx.generateMesh, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &global_size);CHKERRQ(ierr);
  /* total number of vertices in all dimensions */
  n = pow(npts, dim);

  /* do some error checking */
  if (n < 2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of points must be >= 2.\n");
  if (global_size > n) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of processors must be less than or equal to number of elements.\n");
  if (nghost < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of ghost layers cannot be negative.\n");

  /* Create the basic DMMoab object and keep the default parameters created by DM impls */
  ierr = DMMoabCreateMoab(comm, NULL, NULL, NULL, dm);CHKERRQ(ierr);

  /* get all the necessary handles from the private DM object */
  dmmoab = (DM_Moab*)(*dm)->data;
  mbImpl = dmmoab->mbiface;
#ifdef MOAB_HAVE_MPI
  pcomm = dmmoab->pcomm;
  global_rank = pcomm->rank();
#else
  global_rank = 0;
  global_size = 1;
#endif
  global_id_tag = dmmoab->ltog_tag;
  dmmoab->dim = dim;
  dmmoab->nghostrings = nghost;
  dmmoab->refct = 1;

  /* create a file set to associate all entities in current mesh */
  merr = mbImpl->create_meshset(moab::MESHSET_SET, dmmoab->fileset);MBERR("Creating file set failed", merr);

  /* No errors yet; proceed with building the mesh */
  merr = mbImpl->query_interface(readMeshIface);MBERRNM(merr);

  genCtx.M = genCtx.N = genCtx.K = 1;
  genCtx.A = genCtx.B = genCtx.C = 1;
  genCtx.blockSizeElementXYZ[0] = 0;
  genCtx.blockSizeElementXYZ[1] = 0;
  genCtx.blockSizeElementXYZ[2] = 0;

  ierr = PetscOptionsBegin(comm, "", "DMMoab Creation Options", "DMMOAB");CHKERRQ(ierr);
  /* Handle DMMoab spatial resolution */
  ierr = PetscOptionsInt("-dmb_grid_x", "Number of grid points in x direction", "DMMoabSetSizes", genCtx.blockSizeElementXYZ[0], &genCtx.blockSizeElementXYZ[0], &genCtx.usrxyzgrid);CHKERRQ(ierr);
  if (dim > 1) {ierr = PetscOptionsInt("-dmb_grid_y", "Number of grid points in y direction", "DMMoabSetSizes", genCtx.blockSizeElementXYZ[1], &genCtx.blockSizeElementXYZ[1], &genCtx.usrxyzgrid);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsInt("-dmb_grid_z", "Number of grid points in z direction", "DMMoabSetSizes", genCtx.blockSizeElementXYZ[2], &genCtx.blockSizeElementXYZ[2], &genCtx.usrxyzgrid);CHKERRQ(ierr);}

  /* Handle DMMoab parallel distibution */
  ierr = PetscOptionsInt("-dmb_processors_x", "Number of processors in x direction", "DMMoabSetNumProcs", genCtx.M, &genCtx.M, &genCtx.usrprocgrid);CHKERRQ(ierr);
  if (dim > 1) {ierr = PetscOptionsInt("-dmb_processors_y", "Number of processors in y direction", "DMMoabSetNumProcs", genCtx.N, &genCtx.N, &genCtx.usrprocgrid);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsInt("-dmb_processors_z", "Number of processors in z direction", "DMMoabSetNumProcs", genCtx.K, &genCtx.K, &genCtx.usrprocgrid);CHKERRQ(ierr);}

  /* Handle DMMoab block refinement */
  ierr = PetscOptionsInt("-dmb_refine_x", "Number of refinement blocks in x direction", "DMMoabSetRefinement", genCtx.A, &genCtx.A, &genCtx.usrrefgrid);
  if (dim > 1) {ierr = PetscOptionsInt("-dmb_refine_y", "Number of refinement blocks in y direction", "DMMoabSetRefinement", genCtx.B, &genCtx.B, &genCtx.usrrefgrid);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsInt("-dmb_refine_z", "Number of refinement blocks in z direction", "DMMoabSetRefinement", genCtx.C, &genCtx.C, &genCtx.usrrefgrid);CHKERRQ(ierr);}
  PetscOptionsEnd();

  ierr = DMMBUtil_InitializeOptions(genCtx, dim, useSimplex, global_rank, global_size, bounds, nele);CHKERRQ(ierr);

  //if(nele<nprocs) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The dimensional discretization size should be greater or equal to number of processors: %D < %D",nele,nprocs);

  if (genCtx.adjEnts) genCtx.keep_skins = true; /* do not delete anything - consumes more memory */

  /* determine m, n, k for processor rank */
  ml = nl = kl = 0;
  switch (genCtx.dim) {
  case 1:
    ml = (genCtx.cumfraction);
    break;
  case 2:
    nl = (genCtx.cumfraction);
    break;
  default:
    kl = (genCtx.cumfraction) / genCtx.q / genCtx.blockSizeElementXYZ[2] / genCtx.C; //genCtx.K
    break;
  }

  /*
   * so there are a total of M * A * blockSizeElement elements in x direction (so M * A * blockSizeElement + 1 verts in x direction)
   * so there are a total of N * B * blockSizeElement elements in y direction (so N * B * blockSizeElement + 1 verts in y direction)
   * so there are a total of K * C * blockSizeElement elements in z direction (so K * C * blockSizeElement + 1 verts in z direction)

   * there are ( M * A blockSizeElement )      *  ( N * B * blockSizeElement)      * (K * C * blockSizeElement )    hexas
   * there are ( M * A * blockSizeElement + 1) *  ( N * B * blockSizeElement + 1 ) * (K * C * blockSizeElement + 1) vertices
   * x is the first dimension that varies
   */

  /* generate the block at (a, b, c); it will represent a partition , it will get a partition tag */
  PetscInt dum_id = -1;
  merr = mbImpl->tag_get_handle("GLOBAL_ID", 1, moab::MB_TYPE_INTEGER, global_id_tag);MBERR("Getting Global_ID Tag handle failed", merr);

  merr = mbImpl->tag_get_handle(MATERIAL_SET_TAG_NAME, 1, moab::MB_TYPE_INTEGER, mat_tag);MBERR("Getting Material set Tag handle failed", merr);
  merr = mbImpl->tag_get_handle(DIRICHLET_SET_TAG_NAME, 1, moab::MB_TYPE_INTEGER, dir_tag);MBERR("Getting Dirichlet set Tag handle failed", merr);
  merr = mbImpl->tag_get_handle(NEUMANN_SET_TAG_NAME, 1, moab::MB_TYPE_INTEGER, neu_tag);MBERR("Getting Neumann set Tag handle failed", merr);

  merr = mbImpl->tag_get_handle("PARALLEL_PARTITION", 1, moab::MB_TYPE_INTEGER, part_tag, moab::MB_TAG_CREAT | moab::MB_TAG_SPARSE, &dum_id);MBERR("Getting Partition Tag handle failed", merr);

  /* lets create some sets */
  merr = mbImpl->tag_get_handle(GEOM_DIMENSION_TAG_NAME, 1, moab::MB_TYPE_INTEGER, geom_tag, moab::MB_TAG_CREAT | moab::MB_TAG_SPARSE, &dum_id);MBERRNM(merr);
  merr = mbImpl->create_meshset(moab::MESHSET_SET, regionset);MBERRNM(merr);
  ierr = PetscLogEventEnd(genCtx.generateMesh, 0, 0, 0, 0);CHKERRQ(ierr);

  for (a = 0; a < (genCtx.dim > 0 ? genCtx.A : genCtx.A); a++) {
    for (b = 0; b < (genCtx.dim > 1 ? genCtx.B : 1); b++) {
      for (c = 0; c < (genCtx.dim > 2 ? genCtx.C : 1); c++) {

        moab::EntityHandle startv;

        ierr = PetscLogEventBegin(genCtx.generateVertices, 0, 0, 0, 0);CHKERRQ(ierr);
        ierr = DMMoab_GenerateVertices_Private(mbImpl, readMeshIface, genCtx, ml, nl, kl, a, b, c, global_id_tag, startv, verts);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(genCtx.generateVertices, 0, 0, 0, 0);CHKERRQ(ierr);

        ierr = PetscLogEventBegin(genCtx.generateElements, 0, 0, 0, 0);CHKERRQ(ierr);
        ierr = DMMoab_GenerateElements_Private(mbImpl, readMeshIface, genCtx, ml, nl, kl, a, b, c, global_id_tag, startv, cells);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(genCtx.generateElements, 0, 0, 0, 0);CHKERRQ(ierr);

        PetscInt part_num = 0;
        switch (genCtx.dim) {
        case 3:
          part_num += (c + kl * genCtx.C) * (genCtx.M * genCtx.A * genCtx.N * genCtx.B);
        case 2:
          part_num += (b + nl * genCtx.B) * (genCtx.M * genCtx.A);
        case 1:
          part_num += (a + ml * genCtx.A);
          break;
        }

        moab::EntityHandle part_set;
        merr = mbImpl->create_meshset(moab::MESHSET_SET, part_set);MBERR("Can't create mesh set.", merr);

        merr = mbImpl->add_entities(part_set, verts);MBERR("Can't add vertices to set.", merr);
        merr = mbImpl->add_entities(part_set, cells);MBERR("Can't add entities to set.", merr);
        merr = mbImpl->add_entities(regionset, cells);MBERR("Can't add entities to set.", merr);

        /* if needed, add all edges and faces */
        if (genCtx.adjEnts)
        {
          if (genCtx.dim > 1) {
            merr = mbImpl->get_adjacencies(cells, 1, true, edges, moab::Interface::UNION);MBERR("Can't get edges", merr);
            merr = mbImpl->add_entities(part_set, edges);MBERR("Can't add edges to partition set.", merr);
          }
          if (genCtx.dim > 2) {
            merr = mbImpl->get_adjacencies(cells, 2, true, faces, moab::Interface::UNION);MBERR("Can't get faces", merr);
            merr = mbImpl->add_entities(part_set, faces);MBERR("Can't add faces to partition set.", merr);
          }
          edges.clear();
          faces.clear();
        }
        verts.clear(); cells.clear();

        merr = mbImpl->tag_set_data(part_tag, &part_set, 1, &part_num);MBERR("Can't set part tag on set", merr);
        if (dmmoab->fileset) {
          merr = mbImpl->add_parent_child(dmmoab->fileset, part_set);MBERR("Can't add part set to file set.", merr);
          merr = mbImpl->unite_meshset(dmmoab->fileset, part_set);MBERRNM(merr);
        }
        merr = mbImpl->add_entities(dmmoab->fileset, &part_set, 1);MBERRNM(merr);
      }
    }
  }

  merr = mbImpl->add_parent_child(dmmoab->fileset, regionset);MBERRNM(merr);

  /* Only in parallel: resolve shared entities between processors and exchange ghost layers */
  if (global_size > 1) {

    ierr = PetscLogEventBegin(genCtx.parResolve, 0, 0, 0, 0);CHKERRQ(ierr);

    merr = mbImpl->get_entities_by_dimension(dmmoab->fileset, genCtx.dim, cells);MBERR("Can't get all d-dimensional elements.", merr);
    merr = mbImpl->get_entities_by_dimension(dmmoab->fileset, 0, verts);MBERR("Can't get all vertices.", merr);

    if (genCtx.A * genCtx.B * genCtx.C != 1) { //  merge needed
      moab::MergeMesh mm(mbImpl);
      if (genCtx.newMergeMethod) {
        merr = mm.merge_using_integer_tag(verts, global_id_tag);MBERR("Can't merge with GLOBAL_ID tag", merr);
      }
      else {
        merr = mm.merge_entities(cells, 0.0001);MBERR("Can't merge with coordinates", merr);
      }
    }

#ifdef MOAB_HAVE_MPI
    /* check the handles */
    merr = pcomm->check_all_shared_handles();MBERRV(mbImpl, merr);

    /* resolve the shared entities by exchanging information to adjacent processors */
    merr = pcomm->resolve_shared_ents(dmmoab->fileset, cells, dim, dim - 1, NULL, &global_id_tag);MBERRV(mbImpl, merr);
    if (dmmoab->fileset) {
      merr = pcomm->exchange_ghost_cells(dim, 0, nghost, dim, true, false, &dmmoab->fileset);MBERRV(mbImpl, merr);
    }
    else {
      merr = pcomm->exchange_ghost_cells(dim, 0, nghost, dim, true, false);MBERRV(mbImpl, merr);
    }

    /* Reassign global IDs on all entities. */
    merr = pcomm->assign_global_ids(dmmoab->fileset, dim, 1, false, true, false);MBERRNM(merr);
#endif

    ierr = PetscLogEventEnd(genCtx.parResolve, 0, 0, 0, 0);CHKERRQ(ierr);
  }

  if (!genCtx.keep_skins) { // default is to delete the 1- and 2-dimensional entities
    // delete all quads and edges
    moab::Range toDelete;
    if (genCtx.dim > 1) {
      merr = mbImpl->get_entities_by_dimension(dmmoab->fileset, 1, toDelete);MBERR("Can't get edges", merr);
    }

    if (genCtx.dim > 2) {
      merr = mbImpl->get_entities_by_dimension(dmmoab->fileset, 2, toDelete);MBERR("Can't get faces", merr);
    }

#ifdef MOAB_HAVE_MPI
    merr = dmmoab->pcomm->delete_entities(toDelete) ;MBERR("Can't delete entities", merr);
#endif
  }

  /* set geometric dimension tag for regions */
  merr = mbImpl->tag_set_data(geom_tag, &regionset, 1, &dmmoab->dim);MBERRNM(merr);
  /* set default material ID for regions */
  int default_material = 1;
  merr = mbImpl->tag_set_data(mat_tag, &regionset, 1, &default_material);MBERRNM(merr);
  /*
    int default_dbc = 0;
    merr = mbImpl->tag_set_data(dir_tag, &vertexset, 1, &default_dbc);MBERRNM(merr);
  */
  PetscFunctionReturn(0);
}


PetscErrorCode DMMoab_GetReadOptions_Private(PetscBool by_rank, PetscInt numproc, PetscInt dim, PetscInt nghost, MoabReadMode mode, PetscInt dbglevel, const char* dm_opts, const char* extra_opts, const char** read_opts)
{
  char           *ropts;
  char           ropts_par[PETSC_MAX_PATH_LEN], ropts_pargh[PETSC_MAX_PATH_LEN];
  char           ropts_dbg[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(PETSC_MAX_PATH_LEN, &ropts);CHKERRQ(ierr);
  ierr = PetscMemzero(&ropts_par, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&ropts_pargh, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&ropts_dbg, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);

  /* do parallel read unless using only one processor */
  if (numproc > 1) {
    // ierr = PetscSNPrintf(ropts_par, PETSC_MAX_PATH_LEN, "PARALLEL=%s;PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;PARALLEL_RESOLVE_SHARED_ENTS;PARALLEL_GHOSTS=%d.0.1%s;",MoabReadModes[mode],dim,(by_rank ? ";PARTITION_BY_RANK":""));CHKERRQ(ierr);
    ierr = PetscSNPrintf(ropts_par, PETSC_MAX_PATH_LEN, "PARALLEL=%s;PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;PARALLEL_RESOLVE_SHARED_ENTS;%s", MoabReadModes[mode], (by_rank ? "PARTITION_BY_RANK;" : ""));CHKERRQ(ierr);
    if (nghost) {
      ierr = PetscSNPrintf(ropts_pargh, PETSC_MAX_PATH_LEN, "PARALLEL_GHOSTS=%d.0.%d;", dim, nghost);CHKERRQ(ierr);
    }
  }

  if (dbglevel) {
    if (numproc > 1) {
      ierr = PetscSNPrintf(ropts_dbg, PETSC_MAX_PATH_LEN, "CPUTIME;DEBUG_IO=%d;DEBUG_PIO=%d;", dbglevel, dbglevel);CHKERRQ(ierr);
    }
    else {
      ierr = PetscSNPrintf(ropts_dbg, PETSC_MAX_PATH_LEN, "CPUTIME;DEBUG_IO=%d;", dbglevel);CHKERRQ(ierr);
    }
  }

  ierr = PetscSNPrintf(ropts, PETSC_MAX_PATH_LEN, "%s%s%s%s%s", ropts_par, (nghost ? ropts_pargh : ""), ropts_dbg, (extra_opts ? extra_opts : ""), (dm_opts ? dm_opts : ""));CHKERRQ(ierr);
  *read_opts = ropts;
  PetscFunctionReturn(0);
}


/*@C
  DMMoabLoadFromFile - Creates a DM object by loading the mesh from a user specified file.

  Collective

  Input Parameters:
+ comm - The communicator for the DM object
. dim - The spatial dimension
. filename - The name of the mesh file to be loaded
- usrreadopts - The options string to read a MOAB mesh.

  Reference (Parallel Mesh Initialization: https://www.mcs.anl.gov/~fathom/moab-docs/html/contents.html#fivetwo)

  Output Parameter:
. dm  - The DM object

  Level: beginner

.seealso: DMSetType(), DMCreate(), DMMoabCreateBoxMesh()
@*/
PetscErrorCode DMMoabLoadFromFile(MPI_Comm comm, PetscInt dim, PetscInt nghost, const char* filename, const char* usrreadopts, DM *dm)
{
  moab::ErrorCode     merr;
  PetscInt            nprocs;
  DM_Moab            *dmmoab;
  moab::Interface    *mbiface;
#ifdef MOAB_HAVE_MPI
  moab::ParallelComm *pcomm;
#endif
  moab::Range         verts, elems;
  const char         *readopts;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 6);

  /* Create the basic DMMoab object and keep the default parameters created by DM impls */
  ierr = DMMoabCreateMoab(comm, NULL, NULL, NULL, dm);CHKERRQ(ierr);

  /* get all the necessary handles from the private DM object */
  dmmoab = (DM_Moab*)(*dm)->data;
  mbiface = dmmoab->mbiface;
#ifdef MOAB_HAVE_MPI
  pcomm = dmmoab->pcomm;
  nprocs = pcomm->size();
#else
  nprocs = 1;
#endif
  /* TODO: Decipher dimension based on the loaded mesh instead of getting from user */
  dmmoab->dim = dim;
  dmmoab->nghostrings = nghost;
  dmmoab->refct = 1;

  /* create a file set to associate all entities in current mesh */
  merr = dmmoab->mbiface->create_meshset(moab::MESHSET_SET, dmmoab->fileset);MBERR("Creating file set failed", merr);

  /* add mesh loading options specific to the DM */
  ierr = DMMoab_GetReadOptions_Private(dmmoab->partition_by_rank, nprocs, dim, nghost, dmmoab->read_mode,
                                       dmmoab->rw_dbglevel, dmmoab->extra_read_options, usrreadopts, &readopts);CHKERRQ(ierr);

  PetscInfo2(*dm, "Reading file %s with options: %s\n", filename, readopts);

  /* Load the mesh from a file. */
  if (dmmoab->fileset) {
    merr = mbiface->load_file(filename, &dmmoab->fileset, readopts);MBERRVM(mbiface, "Reading MOAB file failed.", merr);
  }
  else {
    merr = mbiface->load_file(filename, 0, readopts);MBERRVM(mbiface, "Reading MOAB file failed.", merr);
  }

#ifdef MOAB_HAVE_MPI
  /* Reassign global IDs on all entities. */
  /* merr = pcomm->assign_global_ids(dmmoab->fileset, dim, 1, true, true, true);MBERRNM(merr); */
#endif

  /* load the local vertices */
  merr = mbiface->get_entities_by_type(dmmoab->fileset, moab::MBVERTEX, verts, true);MBERRNM(merr);
  /* load the local elements */
  merr = mbiface->get_entities_by_dimension(dmmoab->fileset, dim, elems, true);MBERRNM(merr);

#ifdef MOAB_HAVE_MPI
  /* Everything is set up, now just do a tag exchange to update tags
     on all of the ghost vertexes */
  merr = pcomm->exchange_tags(dmmoab->ltog_tag, verts);MBERRV(mbiface, merr);
  merr = pcomm->exchange_tags(dmmoab->ltog_tag, elems);MBERRV(mbiface, merr);
  merr = pcomm->collective_sync_partition();MBERR("Collective sync failed", merr);
#endif

  PetscInfo3(*dm, "MOAB file '%s' was successfully loaded. Found %D vertices and %D elements.\n", filename, verts.size(), elems.size());
  ierr = PetscFree(readopts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
  DMMoabRenumberMeshEntities - Order and number all entities (vertices->elements) to be contiguously ordered
  in parallel

  Collective

  Input Parameters:
. dm  - The DM object

  Level: advanced

.seealso: DMSetUp(), DMCreate()
@*/
PetscErrorCode DMMoabRenumberMeshEntities(DM dm)
{
  moab::Range         verts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

#ifdef MOAB_HAVE_MPI
  /* Insert new points */
  moab::ErrorCode     merr;
  merr = ((DM_Moab*) dm->data)->pcomm->assign_global_ids(((DM_Moab*) dm->data)->fileset, 3, 0, false, true, false);MBERRNM(merr);
#endif
  PetscFunctionReturn(0);
}

