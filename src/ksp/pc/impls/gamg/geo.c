/*
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/

#if defined(PETSC_HAVE_TRIANGLE)
#if !defined(ANSI_DECLARATORS)
#define ANSI_DECLARATORS
#endif
#include <triangle.h>
#endif

#include <petscblaslapack.h>

/* Private context for the GAMG preconditioner */
typedef struct {
  PetscInt lid;            /* local vertex index */
  PetscInt degree;         /* vertex degree */
} GAMGNode;

static inline int petsc_geo_mg_compare(const void *a, const void *b)
{
  return (((GAMGNode*)a)->degree - ((GAMGNode*)b)->degree);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_GEO

   Input Parameter:
   .  pc - the preconditioner context
*/
PetscErrorCode PCSetCoordinates_GEO(PC pc, PetscInt ndm, PetscInt a_nloc, PetscReal *coords)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscInt       arrsz,bs,my0,kk,ii,nloc,Iend,aloc;
  Mat            Amat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Amat, MAT_CLASSID, 1);
  CHKERRQ(MatGetBlockSize(Amat, &bs));
  CHKERRQ(MatGetOwnershipRange(Amat, &my0, &Iend));
  aloc = (Iend-my0);
  nloc = (Iend-my0)/bs;

  PetscCheckFalse(nloc!=a_nloc && aloc!=a_nloc,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of local blocks %D must be %D or %D.",a_nloc,nloc,aloc);

  pc_gamg->data_cell_rows = 1;
  PetscCheckFalse(!coords && nloc > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Need coordinates for pc_gamg_type 'geo'.");
  pc_gamg->data_cell_cols = ndm; /* coordinates */

  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (!pc_gamg->data || (pc_gamg->data_sz != arrsz)) {
    CHKERRQ(PetscFree(pc_gamg->data));
    CHKERRQ(PetscMalloc1(arrsz+1, &pc_gamg->data));
  }
  for (kk=0; kk<arrsz; kk++) pc_gamg->data[kk] = -999.;
  pc_gamg->data[arrsz] = -99.;
  /* copy data in - column oriented */
  if (nloc == a_nloc) {
    for (kk = 0; kk < nloc; kk++) {
      for (ii = 0; ii < ndm; ii++) {
        pc_gamg->data[ii*nloc + kk] =  coords[kk*ndm + ii];
      }
    }
  } else { /* assumes the coordinates are blocked */
    for (kk = 0; kk < nloc; kk++) {
      for (ii = 0; ii < ndm; ii++) {
        pc_gamg->data[ii*nloc + kk] =  coords[bs*kk*ndm + ii];
      }
    }
  }
  PetscCheckFalse(pc_gamg->data[arrsz] != -99.,PETSC_COMM_SELF,PETSC_ERR_PLIB,"pc_gamg->data[arrsz %D] %g != -99.",arrsz,pc_gamg->data[arrsz]);
  pc_gamg->data_sz = arrsz;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetData_GEO

  Input Parameter:
   . pc -
*/
PetscErrorCode PCSetData_GEO(PC pc, Mat m)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"GEO MG needs coordinates");
}

/* -------------------------------------------------------------------------- */
/*
   PCSetFromOptions_GEO

  Input Parameter:
   . pc -
*/
PetscErrorCode PCSetFromOptions_GEO(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"GAMG-GEO options"));
  {
    /* -pc_gamg_sa_nsmooths */
    /* pc_gamg_sa->smooths = 0; */
    /* ierr = PetscOptionsInt("-pc_gamg_agg_nsmooths", */
    /*                        "smoothing steps for smoothed aggregation, usually 1 (0)", */
    /*                        "PCGAMGSetNSmooths_AGG", */
    /*                        pc_gamg_sa->smooths, */
    /*                        &pc_gamg_sa->smooths, */
    /*                        &flag);  */
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 triangulateAndFormProl

   Input Parameter:
   . selected_2 - list of selected local ID, includes selected ghosts
   . data_stride -
   . coords[2*data_stride] - column vector of local coordinates w/ ghosts
   . nselected_1 - selected IDs that go with base (1) graph includes selected ghosts
   . clid_lid_1[nselected_1] - lids of selected (c) nodes   ???????????
   . agg_lists_1 - list of aggregates selected_1 vertices of aggregate unselected vertices
   . crsGID[selected.size()] - global index for prolongation operator
   . bs - block size
  Output Parameter:
   . a_Prol - prolongation operator
   . a_worst_best - measure of worst missed fine vertex, 0 is no misses
*/
static PetscErrorCode triangulateAndFormProl(IS selected_2,PetscInt data_stride,PetscReal coords[],PetscInt nselected_1,const PetscInt clid_lid_1[],const PetscCoarsenData *agg_lists_1,
                                             const PetscInt crsGID[],PetscInt bs,Mat a_Prol,PetscReal *a_worst_best)
{
#if defined(PETSC_HAVE_TRIANGLE)
  PetscInt             jj,tid,tt,idx,nselected_2;
  struct triangulateio in,mid;
  const PetscInt       *selected_idx_2;
  PetscMPIInt          rank;
  PetscInt             Istart,Iend,nFineLoc,myFine0;
  int                  kk,nPlotPts,sid;
  MPI_Comm             comm;
  PetscReal            tm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)a_Prol,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(ISGetSize(selected_2, &nselected_2));
  if (nselected_2 == 1 || nselected_2 == 2) { /* 0 happens on idle processors */
    *a_worst_best = 100.0; /* this will cause a stop, but not globalized (should not happen) */
  } else *a_worst_best = 0.0;
  CHKERRMPI(MPIU_Allreduce(a_worst_best, &tm, 1, MPIU_REAL, MPIU_MAX, comm));
  if (tm > 0.0) {
    *a_worst_best = 100.0;
    PetscFunctionReturn(0);
  }
  CHKERRQ(MatGetOwnershipRange(a_Prol, &Istart, &Iend));
  nFineLoc = (Iend-Istart)/bs; myFine0 = Istart/bs;
  nPlotPts = nFineLoc; /* locals */
  /* triangle */
  /* Define input points - in*/
  in.numberofpoints          = nselected_2;
  in.numberofpointattributes = 0;
  /* get nselected points */
  CHKERRQ(PetscMalloc1(2*nselected_2, &in.pointlist));
  CHKERRQ(ISGetIndices(selected_2, &selected_idx_2));

  for (kk=0,sid=0; kk<nselected_2; kk++,sid += 2) {
    PetscInt lid = selected_idx_2[kk];
    in.pointlist[sid]   = coords[lid];
    in.pointlist[sid+1] = coords[data_stride + lid];
    if (lid>=nFineLoc) nPlotPts++;
  }
  PetscCheckFalse(sid != 2*nselected_2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"sid %D != 2*nselected_2 %D",sid,nselected_2);

  in.numberofsegments      = 0;
  in.numberofedges         = 0;
  in.numberofholes         = 0;
  in.numberofregions       = 0;
  in.trianglelist          = NULL;
  in.segmentmarkerlist     = NULL;
  in.pointattributelist    = NULL;
  in.pointmarkerlist       = NULL;
  in.triangleattributelist = NULL;
  in.trianglearealist      = NULL;
  in.segmentlist           = NULL;
  in.holelist              = NULL;
  in.regionlist            = NULL;
  in.edgelist              = NULL;
  in.edgemarkerlist        = NULL;
  in.normlist              = NULL;

  /* triangulate */
  mid.pointlist = NULL;          /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = NULL;
  mid.pointmarkerlist    = NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist       = NULL; /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = NULL;
  mid.neighborlist          = NULL; /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = NULL;
  mid.edgelist          = NULL; /* Needed only if -e switch used. */
  mid.edgemarkerlist    = NULL; /* Needed if -e used and -B not used. */
  mid.numberoftriangles = 0;

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                            */
  if (nselected_2 != 0) { /* inactive processor */
    char args[] = "npczQ"; /* c is needed ? */
    triangulate(args, &in, &mid, (struct triangulateio*) NULL);
    /* output .poly files for 'showme' */
    if (!PETSC_TRUE) {
      static int level = 1;
      FILE       *file; char fname[32];

      sprintf(fname,"C%d_%d.poly",level,rank); file = fopen(fname, "w");
      /*First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>*/
      fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for (kk=0,sid=0; kk<in.numberofpoints; kk++,sid += 2) {
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
      }
      /*One line: <# of segments> <# of boundary markers (0 or 1)> */
      fprintf(file, "%d  %d\n",0,0);
      /*Following lines: <segment #> <endpoint> <endpoint> [boundary marker] */
      /* One line: <# of holes> */
      fprintf(file, "%d\n",0);
      /* Following lines: <hole #> <x> <y> */
      /* Optional line: <# of regional attributes and/or area constraints> */
      /* Optional following lines: <region #> <x> <y> <attribute> <maximum area> */
      fclose(file);

      /* elems */
      sprintf(fname,"C%d_%d.ele",level,rank); file = fopen(fname, "w");
      /* First line: <# of triangles> <nodes per triangle> <# of attributes> */
      fprintf(file, "%d %d %d\n",mid.numberoftriangles,3,0);
      /* Remaining lines: <triangle #> <node> <node> <node> ... [attributes] */
      for (kk=0,sid=0; kk<mid.numberoftriangles; kk++,sid += 3) {
        fprintf(file, "%d %d %d %d\n",kk,mid.trianglelist[sid],mid.trianglelist[sid+1],mid.trianglelist[sid+2]);
      }
      fclose(file);

      sprintf(fname,"C%d_%d.node",level,rank); file = fopen(fname, "w");
      /* First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)> */
      /* fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0); */
      fprintf(file, "%d  %d  %d  %d\n",nPlotPts,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for (kk=0,sid=0; kk<in.numberofpoints; kk++,sid+=2) {
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
      }

      sid /= 2;
      for (jj=0; jj<nFineLoc; jj++) {
        PetscBool sel = PETSC_TRUE;
        for (kk=0; kk<nselected_2 && sel; kk++) {
          PetscInt lid = selected_idx_2[kk];
          if (lid == jj) sel = PETSC_FALSE;
        }
        if (sel) fprintf(file, "%d %e %e\n",sid++,coords[jj],coords[data_stride + jj]);
      }
      fclose(file);
      PetscCheckFalse(sid != nPlotPts,PETSC_COMM_SELF,PETSC_ERR_PLIB,"sid %D != nPlotPts %D",sid,nPlotPts);
      level++;
    }
  }
  CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[FIND_V],0,0,0,0));
  { /* form P - setup some maps */
    PetscInt clid,mm,*nTri,*node_tri;

    CHKERRQ(PetscMalloc2(nselected_2, &node_tri,nselected_2, &nTri));

    /* need list of triangles on node */
    for (kk=0; kk<nselected_2; kk++) nTri[kk] = 0;
    for (tid=0,kk=0; tid<mid.numberoftriangles; tid++) {
      for (jj=0; jj<3; jj++) {
        PetscInt cid = mid.trianglelist[kk++];
        if (nTri[cid] == 0) node_tri[cid] = tid;
        nTri[cid]++;
      }
    }
#define EPS 1.e-12
    /* find points and set prolongation */
    for (mm = clid = 0; mm < nFineLoc; mm++) {
      PetscBool ise;
      CHKERRQ(PetscCDEmptyAt(agg_lists_1,mm,&ise));
      if (!ise) {
        const PetscInt lid = mm;
        PetscScalar    AA[3][3];
        PetscBLASInt   N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
        PetscCDIntNd   *pos;

        CHKERRQ(PetscCDGetHeadPos(agg_lists_1,lid,&pos));
        while (pos) {
          PetscInt flid;
          CHKERRQ(PetscCDIntNdGetID(pos, &flid));
          CHKERRQ(PetscCDGetNextPos(agg_lists_1,lid,&pos));

          if (flid < nFineLoc) {  /* could be a ghost */
            PetscInt       bestTID = -1; PetscReal best_alpha = 1.e10;
            const PetscInt fgid    = flid + myFine0;
            /* compute shape function for gid */
            const PetscReal fcoord[3] = {coords[flid],coords[data_stride+flid],1.0};
            PetscBool       haveit    =PETSC_FALSE; PetscScalar alpha[3]; PetscInt clids[3];

            /* look for it */
            for (tid = node_tri[clid], jj=0;
                 jj < 5 && !haveit && tid != -1;
                 jj++) {
              for (tt=0; tt<3; tt++) {
                PetscInt cid2 = mid.trianglelist[3*tid + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = coords[lid2]; AA[tt][1] = coords[data_stride + lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }

              for (tt=0; tt<3; tt++) alpha[tt] = (PetscScalar)fcoord[tt];

              /* SUBROUTINE DGESV(N, NRHS, A, LDA, IPIV, B, LDB, INFO) */
              PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO));
              {
                PetscBool have=PETSC_TRUE;  PetscReal lowest=1.e10;
                for (tt = 0, idx = 0; tt < 3; tt++) {
                  if (PetscRealPart(alpha[tt]) > (1.0+EPS) || PetscRealPart(alpha[tt]) < -EPS) have = PETSC_FALSE;
                  if (PetscRealPart(alpha[tt]) < lowest) {
                    lowest = PetscRealPart(alpha[tt]);
                    idx    = tt;
                  }
                }
                haveit = have;
              }
              tid = mid.neighborlist[3*tid + idx];
            }

            if (!haveit) {
              /* brute force */
              for (tid=0; tid<mid.numberoftriangles && !haveit; tid++) {
                for (tt=0; tt<3; tt++) {
                  PetscInt cid2 = mid.trianglelist[3*tid + tt];
                  PetscInt lid2 = selected_idx_2[cid2];
                  AA[tt][0] = coords[lid2]; AA[tt][1] = coords[data_stride + lid2]; AA[tt][2] = 1.0;
                  clids[tt] = cid2; /* store for interp */
                }
                for (tt=0; tt<3; tt++) alpha[tt] = fcoord[tt];
                /* SUBROUTINE DGESV(N, NRHS, A, LDA, IPIV, B, LDB, INFO) */
                PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO));
                {
                  PetscBool have=PETSC_TRUE;  PetscReal worst=0.0, v;
                  for (tt=0; tt<3 && have; tt++) {
                    if (PetscRealPart(alpha[tt]) > 1.0+EPS || PetscRealPart(alpha[tt]) < -EPS) have=PETSC_FALSE;
                    if ((v=PetscAbs(PetscRealPart(alpha[tt])-0.5)) > worst) worst = v;
                  }
                  if (worst < best_alpha) {
                    best_alpha = worst; bestTID = tid;
                  }
                  haveit = have;
                }
              }
            }
            if (!haveit) {
              if (best_alpha > *a_worst_best) *a_worst_best = best_alpha;
              /* use best one */
              for (tt=0; tt<3; tt++) {
                PetscInt cid2 = mid.trianglelist[3*bestTID + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = coords[lid2]; AA[tt][1] = coords[data_stride + lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }
              for (tt=0; tt<3; tt++) alpha[tt] = fcoord[tt];
              /* SUBROUTINE DGESV(N, NRHS, A, LDA, IPIV, B, LDB, INFO) */
              PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO));
            }

            /* put in row of P */
            for (idx=0; idx<3; idx++) {
              PetscScalar shp = alpha[idx];
              if (PetscAbs(PetscRealPart(shp)) > 1.e-6) {
                PetscInt cgid = crsGID[clids[idx]];
                PetscInt jj   = cgid*bs, ii = fgid*bs; /* need to gloalize */
                for (tt=0; tt < bs; tt++, ii++, jj++) {
                  CHKERRQ(MatSetValues(a_Prol,1,&ii,1,&jj,&shp,INSERT_VALUES));
                }
              }
            }
          }
        } /* aggregates iterations */
        clid++;
      } /* a coarse agg */
    } /* for all fine nodes */

    CHKERRQ(ISRestoreIndices(selected_2, &selected_idx_2));
    CHKERRQ(MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY));

    CHKERRQ(PetscFree2(node_tri,nTri));
  }
  CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[FIND_V],0,0,0,0));
  free(mid.trianglelist);
  free(mid.neighborlist);
  free(mid.segmentlist);
  free(mid.segmentmarkerlist);
  free(mid.pointlist);
  free(mid.pointmarkerlist);
  CHKERRQ(PetscFree(in.pointlist));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject)a_Prol),PETSC_ERR_PLIB,"configure with TRIANGLE to use geometric MG");
#endif
}
/* -------------------------------------------------------------------------- */
/*
   getGIDsOnSquareGraph - square graph, get

   Input Parameter:
   . nselected_1 - selected local indices (includes ghosts in input Gmat1)
   . clid_lid_1 - [nselected_1] lids of selected nodes
   . Gmat1 - graph that goes with 'selected_1'
   Output Parameter:
   . a_selected_2 - selected local indices (includes ghosts in output a_Gmat_2)
   . a_Gmat_2 - graph that is squared of 'Gmat_1'
   . a_crsGID[a_selected_2.size()] - map of global IDs of coarse grid nodes
*/
static PetscErrorCode getGIDsOnSquareGraph(PC pc, PetscInt nselected_1,const PetscInt clid_lid_1[],const Mat Gmat1,IS *a_selected_2,Mat *a_Gmat_2,PetscInt **a_crsGID)
{
  PetscMPIInt    size;
  PetscInt       *crsGID, kk,my0,Iend,nloc;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)Gmat1,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(MatGetOwnershipRange(Gmat1,&my0,&Iend)); /* AIJ */
  nloc = Iend - my0; /* this does not change */

  if (size == 1) { /* not much to do in serial */
    CHKERRQ(PetscMalloc1(nselected_1, &crsGID));
    for (kk=0; kk<nselected_1; kk++) crsGID[kk] = kk;
    *a_Gmat_2 = NULL;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nselected_1,clid_lid_1,PETSC_COPY_VALUES,a_selected_2));
  } else {
    PetscInt    idx,num_fine_ghosts,num_crs_ghost,myCrs0;
    Mat_MPIAIJ  *mpimat2;
    Mat         Gmat2;
    Vec         locState;
    PetscScalar *cpcol_state;

    /* scan my coarse zero gid, set 'lid_state' with coarse GID */
    kk = nselected_1;
    CHKERRMPI(MPI_Scan(&kk, &myCrs0, 1, MPIU_INT, MPI_SUM, comm));
    myCrs0 -= nselected_1;

    if (a_Gmat_2) { /* output */
      /* grow graph to get wider set of selected vertices to cover fine grid, invalidates 'llist' */
      CHKERRQ(PCGAMGSquareGraph_GAMG(pc,Gmat1,&Gmat2));
      *a_Gmat_2 = Gmat2; /* output */
    } else Gmat2 = Gmat1;  /* use local to get crsGIDs at least */
    /* get coarse grid GIDS for selected (locals and ghosts) */
    mpimat2 = (Mat_MPIAIJ*)Gmat2->data;
    CHKERRQ(MatCreateVecs(Gmat2, &locState, NULL));
    CHKERRQ(VecSet(locState, (PetscScalar)(PetscReal)(-1))); /* set with UNKNOWN state */
    for (kk=0; kk<nselected_1; kk++) {
      PetscInt    fgid = clid_lid_1[kk] + my0;
      PetscScalar v    = (PetscScalar)(kk+myCrs0);
      CHKERRQ(VecSetValues(locState, 1, &fgid, &v, INSERT_VALUES)); /* set with PID */
    }
    CHKERRQ(VecAssemblyBegin(locState));
    CHKERRQ(VecAssemblyEnd(locState));
    CHKERRQ(VecScatterBegin(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecGetLocalSize(mpimat2->lvec, &num_fine_ghosts));
    CHKERRQ(VecGetArray(mpimat2->lvec, &cpcol_state));
    for (kk=0,num_crs_ghost=0; kk<num_fine_ghosts; kk++) {
      if ((PetscInt)PetscRealPart(cpcol_state[kk]) != -1) num_crs_ghost++;
    }
    CHKERRQ(PetscMalloc1(nselected_1+num_crs_ghost, &crsGID)); /* output */
    {
      PetscInt *selected_set;
      CHKERRQ(PetscMalloc1(nselected_1+num_crs_ghost, &selected_set));
      /* do ghost of 'crsGID' */
      for (kk=0,idx=nselected_1; kk<num_fine_ghosts; kk++) {
        if ((PetscInt)PetscRealPart(cpcol_state[kk]) != -1) {
          PetscInt cgid = (PetscInt)PetscRealPart(cpcol_state[kk]);
          selected_set[idx] = nloc + kk;
          crsGID[idx++]     = cgid;
        }
      }
      PetscCheckFalse(idx != (nselected_1+num_crs_ghost),PETSC_COMM_SELF,PETSC_ERR_PLIB,"idx %D != (nselected_1 %D + num_crs_ghost %D)",idx,nselected_1,num_crs_ghost);
      CHKERRQ(VecRestoreArray(mpimat2->lvec, &cpcol_state));
      /* do locals in 'crsGID' */
      CHKERRQ(VecGetArray(locState, &cpcol_state));
      for (kk=0,idx=0; kk<nloc; kk++) {
        if ((PetscInt)PetscRealPart(cpcol_state[kk]) != -1) {
          PetscInt cgid = (PetscInt)PetscRealPart(cpcol_state[kk]);
          selected_set[idx] = kk;
          crsGID[idx++]     = cgid;
        }
      }
      PetscCheckFalse(idx != nselected_1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"idx %D != nselected_1 %D",idx,nselected_1);
      CHKERRQ(VecRestoreArray(locState, &cpcol_state));

      if (a_selected_2 != NULL) { /* output */
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,(nselected_1+num_crs_ghost),selected_set,PETSC_OWN_POINTER,a_selected_2));
      } else {
        CHKERRQ(PetscFree(selected_set));
      }
    }
    CHKERRQ(VecDestroy(&locState));
  }
  *a_crsGID = crsGID; /* output */
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGGraph_GEO

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
  Output Parameter:
   . a_Gmat
*/
PetscErrorCode PCGAMGGraph_GEO(PC pc,Mat Amat,Mat *a_Gmat)
{
  PC_MG           *mg      = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscReal vfilter  = pc_gamg->threshold[0];
  MPI_Comm        comm;
  Mat             Gmat;
  PetscBool       set,flg,symm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)Amat,&comm));
  CHKERRQ(PetscLogEventBegin(PC_GAMGGraph_GEO,0,0,0,0));

  CHKERRQ(MatIsSymmetricKnown(Amat, &set, &flg));
  symm = (PetscBool)!(set && flg);

  CHKERRQ(PCGAMGCreateGraph(Amat, &Gmat));
  CHKERRQ(PCGAMGFilterGraph(&Gmat, vfilter, symm));

  *a_Gmat = Gmat;
  CHKERRQ(PetscLogEventEnd(PC_GAMGGraph_GEO,0,0,0,0));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCoarsen_GEO

  Input Parameter:
   . a_pc - this
   . a_Gmat - graph
  Output Parameter:
   . a_llist_parent - linked list from selected indices for data locality only
*/
PetscErrorCode PCGAMGCoarsen_GEO(PC a_pc,Mat *a_Gmat,PetscCoarsenData **a_llist_parent)
{
  PetscInt       Istart,Iend,nloc,kk,Ii,ncols;
  IS             perm;
  GAMGNode       *gnodes;
  PetscInt       *permute;
  Mat            Gmat  = *a_Gmat;
  MPI_Comm       comm;
  MatCoarsen     crs;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)a_pc,&comm));
  CHKERRQ(PetscLogEventBegin(PC_GAMGCoarsen_GEO,0,0,0,0));
  CHKERRQ(MatGetOwnershipRange(Gmat, &Istart, &Iend));
  nloc = (Iend-Istart);

  /* create random permutation with sort for geo-mg */
  CHKERRQ(PetscMalloc1(nloc, &gnodes));
  CHKERRQ(PetscMalloc1(nloc, &permute));

  for (Ii=Istart; Ii<Iend; Ii++) { /* locals only? */
    CHKERRQ(MatGetRow(Gmat,Ii,&ncols,NULL,NULL));
    {
      PetscInt lid = Ii - Istart;
      gnodes[lid].lid    = lid;
      gnodes[lid].degree = ncols;
    }
    CHKERRQ(MatRestoreRow(Gmat,Ii,&ncols,NULL,NULL));
  }
  if (PETSC_TRUE) {
    PetscRandom  rand;
    PetscBool    *bIndexSet;
    PetscReal    rr;
    PetscInt     iSwapIndex;

    CHKERRQ(PetscRandomCreate(comm,&rand));
    CHKERRQ(PetscCalloc1(nloc, &bIndexSet));
    for (Ii = 0; Ii < nloc; Ii++) {
      CHKERRQ(PetscRandomGetValueReal(rand,&rr));
      iSwapIndex = (PetscInt) (rr*nloc);
      if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii) {
        GAMGNode iTemp = gnodes[iSwapIndex];
        gnodes[iSwapIndex]    = gnodes[Ii];
        gnodes[Ii]            = iTemp;
        bIndexSet[Ii]         = PETSC_TRUE;
        bIndexSet[iSwapIndex] = PETSC_TRUE;
      }
    }
    CHKERRQ(PetscRandomDestroy(&rand));
    CHKERRQ(PetscFree(bIndexSet));
  }
  /* only sort locals */
  qsort(gnodes, nloc, sizeof(GAMGNode), petsc_geo_mg_compare);
  /* create IS of permutation */
  for (kk=0; kk<nloc; kk++) permute[kk] = gnodes[kk].lid; /* locals only */
  CHKERRQ(PetscFree(gnodes));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_OWN_POINTER, &perm));

  /* get MIS aggs */

  CHKERRQ(MatCoarsenCreate(comm, &crs));
  CHKERRQ(MatCoarsenSetType(crs, MATCOARSENMIS));
  CHKERRQ(MatCoarsenSetGreedyOrdering(crs, perm));
  CHKERRQ(MatCoarsenSetAdjacency(crs, Gmat));
  CHKERRQ(MatCoarsenSetStrictAggs(crs, PETSC_FALSE));
  CHKERRQ(MatCoarsenApply(crs));
  CHKERRQ(MatCoarsenGetData(crs, a_llist_parent));
  CHKERRQ(MatCoarsenDestroy(&crs));

  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(PetscLogEventEnd(PC_GAMGCoarsen_GEO,0,0,0,0));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCGAMGProlongator_GEO

 Input Parameter:
 . pc - this
 . Amat - matrix on this fine level
 . Graph - used to get ghost data for nodes in
 . selected_1 - [nselected]
 . agg_lists - [nselected]
 Output Parameter:
 . a_P_out - prolongation operator to the next level
 */
PetscErrorCode PCGAMGProlongator_GEO(PC pc,Mat Amat,Mat Gmat,PetscCoarsenData *agg_lists,Mat *a_P_out)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscInt dim      = pc_gamg->data_cell_cols, data_cols = pc_gamg->data_cell_cols;
  PetscInt       Istart,Iend,nloc,my0,jj,kk,ncols,nLocalSelected,bs,*clid_flid;
  Mat            Prol;
  PetscMPIInt    rank, size;
  MPI_Comm       comm;
  IS             selected_2,selected_1;
  const PetscInt *selected_idx;
  MatType        mtype;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)Amat,&comm));
  CHKERRQ(PetscLogEventBegin(PC_GAMGProlongator_GEO,0,0,0,0));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(MatGetOwnershipRange(Amat, &Istart, &Iend));
  CHKERRQ(MatGetBlockSize(Amat, &bs));
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  PetscCheckFalse((Iend-Istart) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"(Iend %D - Istart %D) % bs %D",Iend,Istart,bs);

  /* get 'nLocalSelected' */
  CHKERRQ(PetscCDGetMIS(agg_lists, &selected_1));
  CHKERRQ(ISGetSize(selected_1, &jj));
  CHKERRQ(PetscMalloc1(jj, &clid_flid));
  CHKERRQ(ISGetIndices(selected_1, &selected_idx));
  for (kk=0,nLocalSelected=0; kk<jj; kk++) {
    PetscInt lid = selected_idx[kk];
    if (lid<nloc) {
      CHKERRQ(MatGetRow(Gmat,lid+my0,&ncols,NULL,NULL));
      if (ncols>1) clid_flid[nLocalSelected++] = lid; /* fiter out singletons */
      CHKERRQ(MatRestoreRow(Gmat,lid+my0,&ncols,NULL,NULL));
    }
  }
  CHKERRQ(ISRestoreIndices(selected_1, &selected_idx));
  CHKERRQ(ISDestroy(&selected_1)); /* this is selected_1 in serial */

  /* create prolongator  matrix */
  CHKERRQ(MatGetType(Amat,&mtype));
  CHKERRQ(MatCreate(comm, &Prol));
  CHKERRQ(MatSetSizes(Prol,nloc*bs,nLocalSelected*bs,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSizes(Prol, bs, bs));
  CHKERRQ(MatSetType(Prol, mtype));
  CHKERRQ(MatSeqAIJSetPreallocation(Prol,3*data_cols,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(Prol,3*data_cols,NULL,3*data_cols,NULL));

  /* can get all points "removed" - but not on geomg */
  CHKERRQ(MatGetSize(Prol, &kk, &jj));
  if (!jj) {
    CHKERRQ(PetscInfo(pc,"ERROE: no selected points on coarse grid\n"));
    CHKERRQ(PetscFree(clid_flid));
    CHKERRQ(MatDestroy(&Prol));
    *a_P_out = NULL;  /* out */
    PetscFunctionReturn(0);
  }

  {
    PetscReal *coords;
    PetscInt  data_stride;
    PetscInt  *crsGID = NULL;
    Mat       Gmat2;

    PetscCheckFalse(dim != data_cols,PETSC_COMM_SELF,PETSC_ERR_PLIB,"dim %D != data_cols %D",dim,data_cols);
    /* grow ghost data for better coarse grid cover of fine grid */
    CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET5],0,0,0,0));
    /* messy method, squares graph and gets some data */
    CHKERRQ(getGIDsOnSquareGraph(pc, nLocalSelected, clid_flid, Gmat, &selected_2, &Gmat2, &crsGID));
    /* llist is now not valid wrt squared graph, but will work as iterator in 'triangulateAndFormProl' */
    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET5],0,0,0,0));
    /* create global vector of coorindates in 'coords' */
    if (size > 1) {
      CHKERRQ(PCGAMGGetDataWithGhosts(Gmat2, dim, pc_gamg->data, &data_stride, &coords));
    } else {
      coords      = (PetscReal*)pc_gamg->data;
      data_stride = pc_gamg->data_sz/pc_gamg->data_cell_cols;
    }
    CHKERRQ(MatDestroy(&Gmat2));

    /* triangulate */
    if (dim == 2) {
      PetscReal metric,tm;
      CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET6],0,0,0,0));
      CHKERRQ(triangulateAndFormProl(selected_2, data_stride, coords,nLocalSelected, clid_flid, agg_lists, crsGID, bs, Prol, &metric));
      CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET6],0,0,0,0));
      CHKERRQ(PetscFree(crsGID));

      /* clean up and create coordinates for coarse grid (output) */
      if (size > 1) CHKERRQ(PetscFree(coords));

      CHKERRMPI(MPIU_Allreduce(&metric, &tm, 1, MPIU_REAL, MPIU_MAX, comm));
      if (tm > 1.) { /* needs to be globalized - should not happen */
        CHKERRQ(PetscInfo(pc," failed metric for coarse grid %e\n",(double)tm));
        CHKERRQ(MatDestroy(&Prol));
      } else if (metric > .0) {
        CHKERRQ(PetscInfo(pc,"worst metric for coarse grid = %e\n",(double)metric));
      }
    } else SETERRQ(comm,PETSC_ERR_PLIB,"3D not implemented for 'geo' AMG");
    { /* create next coords - output */
      PetscReal *crs_crds;
      CHKERRQ(PetscMalloc1(dim*nLocalSelected, &crs_crds));
      for (kk=0; kk<nLocalSelected; kk++) { /* grab local select nodes to promote - output */
        PetscInt lid = clid_flid[kk];
        for (jj=0; jj<dim; jj++) crs_crds[jj*nLocalSelected + kk] = pc_gamg->data[jj*nloc + lid];
      }

      CHKERRQ(PetscFree(pc_gamg->data));
      pc_gamg->data    = crs_crds; /* out */
      pc_gamg->data_sz = dim*nLocalSelected;
    }
    CHKERRQ(ISDestroy(&selected_2));
  }

  *a_P_out = Prol;  /* out */
  CHKERRQ(PetscFree(clid_flid));
  CHKERRQ(PetscLogEventEnd(PC_GAMGProlongator_GEO,0,0,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_GAMG_GEO(PC pc)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCCreateGAMG_GEO

  Input Parameter:
   . pc -
*/
PetscErrorCode  PCCreateGAMG_GEO(PC pc)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->ops->setfromoptions = PCSetFromOptions_GEO;
  pc_gamg->ops->destroy        = PCDestroy_GAMG_GEO;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->graph             = PCGAMGGraph_GEO;
  pc_gamg->ops->coarsen           = PCGAMGCoarsen_GEO;
  pc_gamg->ops->prolongator       = PCGAMGProlongator_GEO;
  pc_gamg->ops->optprolongator    = NULL;
  pc_gamg->ops->createdefaultdata = PCSetData_GEO;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_GEO));
  PetscFunctionReturn(0);
}
