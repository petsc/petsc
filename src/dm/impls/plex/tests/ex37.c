static const char help[] = "Test of EGADSLite CAD functionality";

#include <petscdmplex.h>

#include <egads.h>
#include <petsc.h>

typedef struct {
  char filename[PETSC_MAX_PATH_LEN];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "EGADSPlex Problem Options", "EGADSLite");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The EGADSLite file", "ex9.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  DMLabel        bodyLabel, faceLabel, edgeLabel;
  PetscInt       cStart, cEnd, c;
  /* EGADSLite variables */
  ego            context, model, geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int            oclass, mtype, nbodies, *senses;
  int            b;
  /* PETSc variables */
  DM             dm;
  PetscInt       dim = -1, cdim = -1, numCorners = 0, numVertices = 0, numCells = 0;
  PetscInt      *cells  = NULL;
  PetscReal     *coords = NULL;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &ctx);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    /* Open EGADs file and load EGADs model data */
    ierr = EG_open(&context);CHKERRQ(ierr);
    ierr = EG_loadModel(context, 0, ctx.filename, &model);CHKERRQ(ierr);

    /* test bodyTopo functions */
    ierr = EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %d \n", nbodies);CHKERRQ(ierr);

    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nsh, Nf, Nl, l, Ne, e, Nv, v;

      /* Output Basic Model Topology */
      ierr = EG_getBodyTopos(body, NULL, SHELL, &Nsh, &objs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of SHELLS: %d \n", Nsh);CHKERRQ(ierr);

      ierr = EG_getBodyTopos(body, NULL, FACE,  &Nf, &objs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of FACES: %d \n", Nf);CHKERRQ(ierr);

      ierr = EG_getBodyTopos(body, NULL, LOOP,  &Nl, &lobjs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of LOOPS: %d \n", Nl);CHKERRQ(ierr);

      ierr = EG_getBodyTopos(body, NULL, EDGE,  &Ne, &objs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of EDGES: %d \n", Ne);CHKERRQ(ierr);

      ierr = EG_getBodyTopos(body, NULL, NODE,  &Nv, &objs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "   Number of NODES: %d \n", Nv);CHKERRQ(ierr);

      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];

        id   = EG_indexBodyTopo(body, loop);
        ierr = PetscPrintf(PETSC_COMM_SELF, "          LOOP ID: %d\n", id);CHKERRQ(ierr);

        /* Get EDGE info which associated with the current LOOP */
        ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];

          id = EG_indexBodyTopo(body, edge);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_SELF, "            EDGE ID: %d\n", id);CHKERRQ(ierr);

          double range[4] = {0., 0., 0., 0.};
          double point[3] = {0., 0., 0.};
          int    peri;

          ierr = EG_getRange(objs[e], range, &peri);
          ierr = PetscPrintf(PETSC_COMM_SELF, " Range = %lf, %lf, %lf, %lf \n", range[0], range[1], range[2], range[3]);

          /* Get NODE info which associated with the current EDGE */
          ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);CHKERRQ(ierr);

          for (v = 0; v < Nv; ++v) {
            ego    vertex = nobjs[v];
            double limits[4];
            int    dummy;

            ierr = EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses);CHKERRQ(ierr);
            id   = EG_indexBodyTopo(body, vertex);
            ierr = PetscPrintf(PETSC_COMM_SELF, "              NODE ID: %d \n", id);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "                 (x, y, z) = (%lf, %lf, %lf) \n", limits[0], limits[1], limits[2]);

            point[0] = point[0] + limits[0];
            point[1] = point[1] + limits[1];
            point[2] = point[2] + limits[2];
          }
        }
      }
    }

    /* ---------------------------------------------------------------------------------------------------
    Generate Petsc Plex
      Get all Nodes in model, record coordinates in a correctly formatted array
      Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array */

    /* Calculate cell and vertex sizes */
    ierr = EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses);CHKERRQ(ierr);
    numCells    = 0;
    numVertices = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l, Nv, v;

      ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
      ierr = EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs);CHKERRQ(ierr);
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];

        id = EG_indexBodyTopo(body, loop);
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numCells = PetscMax(id, numCells);
      }
      for (v = 0; v < Nv; ++v) {
        ego vertex = nobjs[v];

        id = EG_indexBodyTopo(body, vertex);
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numVertices = PetscMax(id, numVertices);
      }
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, "\nPLEX Input Array Checkouts\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " Total Number of Unique Cells    = %d \n", numCells);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, " Total Number of Unique Vertices = %d \n", numVertices);CHKERRQ(ierr);

    dim        = 2; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    cdim       = 3; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    numCorners = 3; /* TODO Check number of cell corners from EGADSLite */
    ierr = PetscMalloc2(numVertices*cdim, &coords, numCells*numCorners, &cells);CHKERRQ(ierr);

    /* Get vertex coordinates */
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nv, v;

      ierr = EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs);CHKERRQ(ierr);
      for (v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    dummy;

        ierr = EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses);CHKERRQ(ierr);
        id   = EG_indexBodyTopo(body, vertex);CHKERRQ(ierr);
        coords[(id-1)*cdim+0] = limits[0];
        coords[(id-1)*cdim+1] = limits[1];
        coords[(id-1)*cdim+2] = limits[2];
        ierr = PetscPrintf(PETSC_COMM_SELF, "    Node ID = %d \n", id);
        ierr = PetscPrintf(PETSC_COMM_SELF, "      (x,y,z) = (%lf, %lf, %lf) \n \n", coords[(id-1)*cdim+0], coords[(id-1)*cdim+1],coords[(id-1)*cdim+2]);
      }
    }

    /* Get cell vertices by traversing loops */
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l;

      ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ne, e, nc = 0, c;

        lid  = EG_indexBodyTopo(body, loop);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "    LOOP ID: %d \n", lid);CHKERRQ(ierr);
        ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int Nv, v;

          id   = EG_indexBodyTopo(body, edge);
          ierr = PetscPrintf(PETSC_COMM_SELF, "      EDGE ID: %d \n", id);CHKERRQ(ierr);
          if (mtype == DEGENERATE) {ierr = PetscPrintf(PETSC_COMM_SELF, "        EGDE %d is DEGENERATE \n", id);CHKERRQ(ierr);}
          ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);

          /* Add unique vertices to cells, this handles mtype == DEGENERATE fine */
          for (v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];

            id = EG_indexBodyTopo(body, vertex);
            for (c = 0; c < nc; ++c) if (cells[(lid-1)*numCorners+c] == id-1) break;
            if (c == nc) cells[(lid-1)*numCorners+nc++] = id-1;
          }
        }
        if (nc != numCorners) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of cell corners %D, should be %D", nc, numCorners);
        ierr = PetscPrintf(PETSC_COMM_SELF, "      LOOP Corner NODEs (");
        for (c = 0; c < numCorners; ++c) {
          if (c > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");}
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D", cells[(lid-1)*numCorners+c]);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, ")\n");
      }
    }
  }
  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD, dim, numCells, numVertices, numCorners, PETSC_TRUE, cells, cdim, coords, &dm);CHKERRQ(ierr);
  ierr = PetscFree2(coords, cells);CHKERRQ(ierr);
  {
    PetscContainer modelObj;

    ierr = PetscContainerCreate(PETSC_COMM_SELF, &modelObj);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(modelObj, model);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dm, "EGADS Model", (PetscObject) modelObj);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&modelObj);CHKERRQ(ierr);
  }
  ierr = DMCreateLabel(dm, "EGADS Body ID");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Body ID", &bodyLabel);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "EGADS Face ID");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Face ID", &faceLabel);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "EGADS Edge ID");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Edge ID", &edgeLabel);CHKERRQ(ierr);
  for (b = 0; b < nbodies; ++b) {
    ego body = bodies[b];
    int id, Nl, l;

    ierr = EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs);CHKERRQ(ierr);
    for (l = 0; l < Nl; ++l) {
      ego loop = lobjs[l];
      int lid, cell, Ne, e;

      lid  = EG_indexBodyTopo(body, loop);CHKERRQ(ierr);
      cell = lid-1;
      ierr = DMLabelSetValue(bodyLabel, cell, b);CHKERRQ(ierr);
      {
        ego *fobjs;
        int  Nf, fid;

        ierr = EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs);CHKERRQ(ierr);
        fid  = EG_indexBodyTopo(body, fobjs[0]);CHKERRQ(ierr);
        ierr = DMLabelSetValue(faceLabel, cell, fid);CHKERRQ(ierr);
      }

      ierr = EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses);CHKERRQ(ierr);
      for (e = 0; e < Ne; ++e) {
        ego             edge = objs[e];
        int             eid, Nv, v;
        PetscInt        support[2], numEdges;
        const PetscInt *edges;

        eid  = EG_indexBodyTopo(body, edge);
        ierr = EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses);
        if (Nv > 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge %d has %d vertices > 2", eid, Nv);
        for (v = 0; v < Nv; ++v) {
          ego vertex = nobjs[v];

          id   = EG_indexBodyTopo(body, vertex);
          ierr = DMLabelSetValue(edgeLabel, numCells + id-1, eid);CHKERRQ(ierr);
          support[v] = numCells + id-1;
        }
        if (Nv == 2) {
          ierr = DMPlexGetJoin(dm, 2, support, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "2 vertices should only bound 1 edge, not %D", numEdges);
          ierr = DMLabelSetValue(edgeLabel, edges[0], eid);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  clSize, cl, bval, fval;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
    ierr = DMLabelGetValue(bodyLabel, c, &bval);CHKERRQ(ierr);
    ierr = DMLabelGetValue(faceLabel, c, &fval);CHKERRQ(ierr);
    for (cl = 0; cl < clSize*2; cl += 2) {
      ierr = DMLabelSetValue(bodyLabel, closure[cl], bval);CHKERRQ(ierr);
      ierr = DMLabelSetValue(faceLabel, closure[cl], fval);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
  }
  ierr = DMLabelView(bodyLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(faceLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(edgeLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  /* Close EGADSlite file */
  ierr = EG_close(context);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: egads

  test:
    suffix: sphere_0
    filter: sed "s/DM_[0-9a-zA-Z]*_0/DM__0/g"
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -dm_view ::ascii_info_detail

TEST*/
