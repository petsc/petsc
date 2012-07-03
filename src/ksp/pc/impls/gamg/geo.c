/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

#if defined(PETSC_HAVE_TRIANGLE) 
#define REAL PetscReal
#include <triangle.h>
#endif

#include <assert.h>
#include <petscblaslapack.h>

/* Private context for the GAMG preconditioner */
typedef struct{
  PetscInt       lid;      /* local vertex index */
  PetscInt       degree;   /* vertex degree */
} GAMGNode;
int petsc_geo_mg_compare (const void *a, const void *b)
{
  return (((GAMGNode*)a)->degree - ((GAMGNode*)b)->degree);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_GEO

   Input Parameter:
   .  pc - the preconditioner context
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_GEO"
PetscErrorCode PCSetCoordinates_GEO( PC pc, PetscInt ndm, PetscInt a_nloc, PetscReal *coords )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;
  PetscInt       arrsz,bs,my0,kk,ii,nloc,Iend;
  Mat            Amat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific( Amat, MAT_CLASSID, 1 );
  ierr  = MatGetBlockSize( Amat, &bs );               CHKERRQ( ierr );

  ierr  = MatGetOwnershipRange( Amat, &my0, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-my0)/bs;

  if(nloc!=a_nloc)SETERRQ2(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Stokes not supported nloc = %d %d.",a_nloc,nloc);
  if((Iend-my0)%bs!=0) SETERRQ1(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Bad local size %d.",nloc);

  pc_gamg->data_cell_rows = 1;
  if( coords==0 && nloc > 0 ) {
    SETERRQ(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Need coordinates for pc_gamg_type 'geo'.");
  }
  pc_gamg->data_cell_cols = ndm; /* coordinates */

  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;
  
  /* create data - syntactic sugar that should be refactored at some point */
  if (pc_gamg->data==0 || (pc_gamg->data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->data );  CHKERRQ(ierr);
    ierr = PetscMalloc((arrsz+1)*sizeof(PetscReal), &pc_gamg->data ); CHKERRQ(ierr);
  }
  for(kk=0;kk<arrsz;kk++)pc_gamg->data[kk] = -999.;
  pc_gamg->data[arrsz] = -99.;
  /* copy data in - column oriented */
  for( kk = 0 ; kk < nloc ; kk++ ){
    for( ii = 0 ; ii < ndm ; ii++ ) {
      pc_gamg->data[ii*nloc + kk] =  coords[kk*ndm + ii];
    }
  }
  assert(pc_gamg->data[arrsz] == -99.);
    
  pc_gamg->data_sz = arrsz;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*
   PCSetData_GEO

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetData_GEO"
PetscErrorCode PCSetData_GEO( PC pc, Mat m )
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_LIB,"GEO MG needs coordinates");
}

/* -------------------------------------------------------------------------- */
/*
   PCSetFromOptions_GEO

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GEO"
PetscErrorCode PCSetFromOptions_GEO( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG-GEO options"); CHKERRQ(ierr);
  {
    /* -pc_gamg_sa_nsmooths */
    /* pc_gamg_sa->smooths = 0; */
    /* ierr = PetscOptionsInt("-pc_gamg_agg_nsmooths", */
    /*                        "smoothing steps for smoothed aggregation, usually 1 (0)", */
    /*                        "PCGAMGSetNSmooths_AGG", */
    /*                        pc_gamg_sa->smooths, */
    /*                        &pc_gamg_sa->smooths, */
    /*                        &flag);  */
    /* CHKERRQ(ierr); */
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
  /* call base class */
  ierr = PCSetFromOptions_GAMG( pc ); CHKERRQ(ierr);

  if( pc_gamg->verbose ) {
    MPI_Comm  wcomm = ((PetscObject)pc)->comm;
    PetscPrintf(wcomm,"[%d]%s done\n",0,__FUNCT__);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 triangulateAndFormProl

   Input Parameter:
   . selected_2 - list of selected local ID, includes selected ghosts
   . data_stride -
   . coords[2*data_stride] - column vector of local coordinates w/ ghosts
   . nselected_1 - selected IDs that go with base (1) graph
   . clid_lid_1[nselected_1] - lids of selected (c) nodes   ???????????
   . agg_lists_1 - list of aggregates 
   . crsGID[selected.size()] - global index for prolongation operator
   . bs - block size
  Output Parameter:
   . a_Prol - prolongation operator
   . a_worst_best - measure of worst missed fine vertex, 0 is no misses
*/
#undef __FUNCT__
#define __FUNCT__ "triangulateAndFormProl"
static PetscErrorCode triangulateAndFormProl( IS  selected_2, /* list of selected local ID, includes selected ghosts */
                                              const PetscInt data_stride,
                                              const PetscReal coords[], /* column vector of local coordinates w/ ghosts */
                                              const PetscInt nselected_1, /* list of selected local ID, includes selected ghosts */
                                              const PetscInt clid_lid_1[],
                                              const PetscCoarsenData *agg_lists_1, /* selected_1 vertices of aggregate unselected vertices */
                                              const PetscInt crsGID[],
                                              const PetscInt bs,
                                              Mat a_Prol, /* prolongation operator (output) */
                                              PetscReal *a_worst_best /* measure of worst missed fine vertex, 0 is no misses */
                                              )
{
#if defined(PETSC_HAVE_TRIANGLE) 
  PetscErrorCode       ierr;
  PetscInt             jj,tid,tt,idx,nselected_2;
  struct triangulateio in,mid;
  const PetscInt      *selected_idx_2;
  PetscMPIInt          mype,npe;
  PetscInt             Istart,Iend,nFineLoc,myFine0;
  int                  kk,nPlotPts,sid;
  MPI_Comm             wcomm = ((PetscObject)a_Prol)->comm;
  PetscReal            tm;
  PetscFunctionBegin;

  ierr = MPI_Comm_rank(wcomm,&mype);    CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);     CHKERRQ(ierr);
  ierr = ISGetSize( selected_2, &nselected_2 );        CHKERRQ(ierr);
  if(nselected_2 == 1 || nselected_2 == 2 ){ /* 0 happens on idle processors */
    *a_worst_best = 100.0; /* this will cause a stop, but not globalized (should not happen) */
  }
  else *a_worst_best = 0.0;
  ierr = MPI_Allreduce( a_worst_best, &tm, 1, MPIU_REAL, MPIU_MAX, wcomm );  CHKERRQ(ierr);
  if( tm > 0.0 ) {
    *a_worst_best = 100.0; 
    PetscFunctionReturn(0);
  }
  ierr = MatGetOwnershipRange( a_Prol, &Istart, &Iend );  CHKERRQ(ierr);
  nFineLoc = (Iend-Istart)/bs; myFine0 = Istart/bs;
  nPlotPts = nFineLoc; /* locals */
  /* traingle */
  /* Define input points - in*/
  in.numberofpoints = nselected_2;
  in.numberofpointattributes = 0;
  /* get nselected points */
  ierr = PetscMalloc( 2*(nselected_2)*sizeof(REAL), &in.pointlist ); CHKERRQ(ierr);
  ierr = ISGetIndices( selected_2, &selected_idx_2 );     CHKERRQ(ierr);

  for(kk=0,sid=0;kk<nselected_2;kk++,sid += 2){
    PetscInt lid = selected_idx_2[kk];
    in.pointlist[sid] = coords[lid];
    in.pointlist[sid+1] = coords[data_stride + lid];
    if(lid>=nFineLoc) nPlotPts++;
  }
  assert(sid==2*nselected_2);

  in.numberofsegments = 0;
  in.numberofedges = 0;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.trianglelist = 0;
  in.segmentmarkerlist = 0;
  in.pointattributelist = 0;
  in.pointmarkerlist = 0;
  in.triangleattributelist = 0;
  in.trianglearealist = 0;
  in.segmentlist = 0;
  in.holelist = 0;
  in.regionlist = 0;
  in.edgelist = 0;
  in.edgemarkerlist = 0;
  in.normlist = 0;
  /* triangulate */
  mid.pointlist = 0;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = 0;
  mid.pointmarkerlist = 0; /* Not needed if -N or -B switch used. */
  mid.trianglelist = 0;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = 0;
  mid.neighborlist = 0;         /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = 0;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = 0;
  mid.edgelist = 0;             /* Needed only if -e switch used. */
  mid.edgemarkerlist = 0;   /* Needed if -e used and -B not used. */
  mid.numberoftriangles = 0;

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                            */
  if(nselected_2 != 0){ /* inactive processor */
    char args[] = "npczQ"; /* c is needed ? */
    triangulate(args, &in, &mid, (struct triangulateio *) NULL );
    /* output .poly files for 'showme' */
    if( !PETSC_TRUE ) {
      static int level = 1;
      FILE *file; char fname[32];

      sprintf(fname,"C%d_%d.poly",level,mype); file = fopen(fname, "w");
      /*First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>*/
      fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++,sid += 2){
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
      sprintf(fname,"C%d_%d.ele",level,mype); file = fopen(fname, "w");
      /* First line: <# of triangles> <nodes per triangle> <# of attributes> */
      fprintf(file, "%d %d %d\n",mid.numberoftriangles,3,0);
      /* Remaining lines: <triangle #> <node> <node> <node> ... [attributes] */
      for(kk=0,sid=0;kk<mid.numberoftriangles;kk++,sid += 3){
        fprintf(file, "%d %d %d %d\n",kk,mid.trianglelist[sid],mid.trianglelist[sid+1],mid.trianglelist[sid+2]);
      }
      fclose(file);

      sprintf(fname,"C%d_%d.node",level,mype); file = fopen(fname, "w");
      /* First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)> */
      /* fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0); */
      fprintf(file, "%d  %d  %d  %d\n",nPlotPts,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++,sid+=2){
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
      }

      sid /= 2;
      for(jj=0;jj<nFineLoc;jj++){
        PetscBool sel = PETSC_TRUE;
        for( kk=0 ; kk<nselected_2 && sel ; kk++ ){
          PetscInt lid = selected_idx_2[kk];
          if( lid == jj ) sel = PETSC_FALSE;
        }
        if( sel ) {
          fprintf(file, "%d %e %e\n",sid++,coords[jj],coords[data_stride + jj]);
        }
      }
      fclose(file);
      assert(sid==nPlotPts);
      level++;
    }
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[FIND_V],0,0,0,0);CHKERRQ(ierr);
#endif
  { /* form P - setup some maps */
    PetscInt clid,mm,*nTri,*node_tri;

    ierr = PetscMalloc( nselected_2*sizeof(PetscInt), &node_tri ); CHKERRQ(ierr); 
    ierr = PetscMalloc( nselected_2*sizeof(PetscInt), &nTri ); CHKERRQ(ierr); 

    /* need list of triangles on node */
    for(kk=0;kk<nselected_2;kk++) nTri[kk] = 0;
    for(tid=0,kk=0;tid<mid.numberoftriangles;tid++){
      for(jj=0;jj<3;jj++) {
        PetscInt cid = mid.trianglelist[kk++];
        if( nTri[cid] == 0 ) node_tri[cid] = tid;
        nTri[cid]++;
      }
    } 
#define EPS 1.e-12
    /* find points and set prolongation */
    for( mm = clid = 0 ; mm < nFineLoc ; mm++ ){
      PetscBool ise;
      ierr = PetscCDEmptyAt(agg_lists_1,mm,&ise); CHKERRQ(ierr);
      if( !ise ) {
        const PetscInt lid = mm;
        //for(clid_iterator=0;clid_iterator<nselected_1;clid_iterator++){
        //PetscInt flid = clid_lid_1[clid_iterator]; assert(flid != -1);
        PetscScalar AA[3][3];
        PetscBLASInt N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
        PetscCDPos         pos;
        ierr = PetscCDGetHeadPos(agg_lists_1,lid,&pos); CHKERRQ(ierr);
        while(pos){              
          PetscInt flid; 
          ierr = PetscLLNGetID( pos, &flid ); CHKERRQ(ierr);
          ierr = PetscCDGetNextPos(agg_lists_1,lid,&pos); CHKERRQ(ierr);

          if( flid < nFineLoc ) {  /* could be a ghost */
            PetscInt bestTID = -1; PetscReal best_alpha = 1.e10;
            const PetscInt fgid = flid + myFine0;
            /* compute shape function for gid */
            const PetscReal fcoord[3] = {coords[flid],coords[data_stride+flid],1.0};
            PetscBool haveit=PETSC_FALSE; PetscScalar alpha[3]; PetscInt clids[3];
            /* look for it */
            for( tid = node_tri[clid], jj=0;
                 jj < 5 && !haveit && tid != -1;
                 jj++ ){
              for(tt=0;tt<3;tt++){
                PetscInt cid2 = mid.trianglelist[3*tid + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = coords[lid2]; AA[tt][1] = coords[data_stride + lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }

              for(tt=0;tt<3;tt++) alpha[tt] = (PetscScalar)fcoord[tt];

              /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
              LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
              {
                PetscBool have=PETSC_TRUE;  PetscReal lowest=1.e10;
                for( tt = 0, idx = 0 ; tt < 3 ; tt++ ) {
                  if( PetscRealPart(alpha[tt]) > (1.0+EPS) || PetscRealPart(alpha[tt]) < -EPS ) have = PETSC_FALSE;
                  if( PetscRealPart(alpha[tt]) < lowest ){
                    lowest = PetscRealPart(alpha[tt]);
                    idx = tt;
                  }
                }
                haveit = have;
              }
              tid = mid.neighborlist[3*tid + idx];
            }
            
            if( !haveit ) {
              /* brute force */
              for(tid=0 ; tid<mid.numberoftriangles && !haveit ; tid++ ){
                for(tt=0;tt<3;tt++){
                  PetscInt cid2 = mid.trianglelist[3*tid + tt];
                  PetscInt lid2 = selected_idx_2[cid2];
                  AA[tt][0] = coords[lid2]; AA[tt][1] = coords[data_stride + lid2]; AA[tt][2] = 1.0;
                  clids[tt] = cid2; /* store for interp */
                }
                for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
                /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
                LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
                {
                  PetscBool have=PETSC_TRUE;  PetscReal worst=0.0, v;
                  for(tt=0; tt<3 && have ;tt++) {
                    if( PetscRealPart(alpha[tt]) > 1.0+EPS || PetscRealPart(alpha[tt]) < -EPS ) have=PETSC_FALSE;
                    if( (v=PetscAbs(PetscRealPart(alpha[tt])-0.5)) > worst ) worst = v;
                  }
                  if( worst < best_alpha ) {
                    best_alpha = worst; bestTID = tid;
                  }
                  haveit = have;
                }
              }
            }
            if( !haveit ) {
              if( best_alpha > *a_worst_best ) *a_worst_best = best_alpha;
              /* use best one */
              for(tt=0;tt<3;tt++){
                PetscInt cid2 = mid.trianglelist[3*bestTID + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = coords[lid2]; AA[tt][1] = coords[data_stride + lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }
              for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
              /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
              LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
            }
            
            /* put in row of P */
            for(idx=0;idx<3;idx++){
              PetscScalar shp = alpha[idx];
              if( PetscAbs(PetscRealPart(shp)) > 1.e-6 ) {
                PetscInt cgid = crsGID[clids[idx]];
                PetscInt jj = cgid*bs, ii = fgid*bs; /* need to gloalize */
                for(tt=0 ; tt < bs ; tt++, ii++, jj++ ){
                  ierr = MatSetValues(a_Prol,1,&ii,1,&jj,&shp,INSERT_VALUES); CHKERRQ(ierr);
                }
              }
            }
          }
        } /* aggregates iterations */
        clid++;
      } /* a coarse agg */
    } /* for all fine nodes */
    
    ierr = ISRestoreIndices( selected_2, &selected_idx_2 );     CHKERRQ(ierr);
    ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PetscFree( node_tri );  CHKERRQ(ierr);
    ierr = PetscFree( nTri );  CHKERRQ(ierr);
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[FIND_V],0,0,0,0);CHKERRQ(ierr);
#endif
  free( mid.trianglelist );
  free( mid.neighborlist );
  ierr = PetscFree( in.pointlist );  CHKERRQ(ierr);

  PetscFunctionReturn(0);
#else
  SETERRQ(((PetscObject)a_Prol)->comm,PETSC_ERR_LIB,"configure with TRIANGLE to use geometric MG");
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
#undef __FUNCT__
#define __FUNCT__ "getGIDsOnSquareGraph"
static PetscErrorCode getGIDsOnSquareGraph( const PetscInt nselected_1,
                                            const PetscInt clid_lid_1[],
                                            const Mat Gmat1,
                                            IS *a_selected_2,
                                            Mat *a_Gmat_2,
                                            PetscInt **a_crsGID
                                            )
{
  PetscErrorCode ierr;
  PetscMPIInt    mype,npe;
  PetscInt       *crsGID, kk,my0,Iend,nloc;
  MPI_Comm       wcomm = ((PetscObject)Gmat1)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat1,&my0,&Iend); CHKERRQ(ierr); /* AIJ */
  nloc = Iend - my0; /* this does not change */
  
  if (npe == 1) { /* not much to do in serial */
    ierr = PetscMalloc( nselected_1*sizeof(PetscInt), &crsGID ); CHKERRQ(ierr);
    for(kk=0;kk<nselected_1;kk++) crsGID[kk] = kk;
    *a_Gmat_2 = 0;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nselected_1,clid_lid_1,PETSC_COPY_VALUES,a_selected_2);
    CHKERRQ(ierr);
  }
  else {
    PetscInt      idx,num_fine_ghosts,num_crs_ghost,myCrs0;
    Mat_MPIAIJ   *mpimat2; 
    Mat           Gmat2;
    Vec           locState;
    PetscScalar   *cpcol_state;

    /* scan my coarse zero gid, set 'lid_state' with coarse GID */
    kk = nselected_1;
    MPI_Scan( &kk, &myCrs0, 1, MPIU_INT, MPIU_SUM, wcomm );
    myCrs0 -= nselected_1;

    if( a_Gmat_2 ) { /* output */
      /* grow graph to get wider set of selected vertices to cover fine grid, invalidates 'llist' */
      ierr = MatTransposeMatMult(Gmat1, Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );   CHKERRQ(ierr);
      *a_Gmat_2 = Gmat2; /* output */
    }
    else Gmat2 = Gmat1;  /* use local to get crsGIDs at least */
    /* get coarse grid GIDS for selected (locals and ghosts) */
    mpimat2 = (Mat_MPIAIJ*)Gmat2->data;
    ierr = MatGetVecs( Gmat2, &locState, 0 );         CHKERRQ(ierr);
    ierr = VecSet( locState, (PetscScalar)(PetscReal)(-1) );  CHKERRQ(ierr); /* set with UNKNOWN state */
    for(kk=0;kk<nselected_1;kk++){
      PetscInt fgid = clid_lid_1[kk] + my0;
      PetscScalar v = (PetscScalar)(kk+myCrs0);
      ierr = VecSetValues( locState, 1, &fgid, &v, INSERT_VALUES );  CHKERRQ(ierr); /* set with PID */
    }
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetLocalSize( mpimat2->lvec, &num_fine_ghosts ); CHKERRQ(ierr);
    ierr = VecGetArray( mpimat2->lvec, &cpcol_state ); CHKERRQ(ierr); 
    for(kk=0,num_crs_ghost=0;kk<num_fine_ghosts;kk++){
      if( (PetscInt)PetscRealPart(cpcol_state[kk]) != -1 ) num_crs_ghost++;
    }
    ierr = PetscMalloc( (nselected_1+num_crs_ghost)*sizeof(PetscInt), &crsGID ); CHKERRQ(ierr); /* output */
    {
      PetscInt *selected_set;
      ierr = PetscMalloc( (nselected_1+num_crs_ghost)*sizeof(PetscInt), &selected_set ); CHKERRQ(ierr);
      /* do ghost of 'crsGID' */
      for(kk=0,idx=nselected_1;kk<num_fine_ghosts;kk++){
        if( (PetscInt)PetscRealPart(cpcol_state[kk]) != -1 ){
          PetscInt cgid = (PetscInt)PetscRealPart(cpcol_state[kk]);
          selected_set[idx] = nloc + kk;
          crsGID[idx++] = cgid;
        }
      }
      assert(idx==(nselected_1+num_crs_ghost));
      ierr = VecRestoreArray( mpimat2->lvec, &cpcol_state ); CHKERRQ(ierr);
      /* do locals in 'crsGID' */
      ierr = VecGetArray( locState, &cpcol_state ); CHKERRQ(ierr);
      for(kk=0,idx=0;kk<nloc;kk++){
        if( (PetscInt)PetscRealPart(cpcol_state[kk]) != -1 ){
          PetscInt cgid = (PetscInt)PetscRealPart(cpcol_state[kk]);
          selected_set[idx] = kk;
          crsGID[idx++] = cgid;
        }
      }
      assert(idx==nselected_1);
      ierr = VecRestoreArray( locState, &cpcol_state ); CHKERRQ(ierr);

      if( a_selected_2 != 0 ) { /* output */
        ierr = ISCreateGeneral(PETSC_COMM_SELF,(nselected_1+num_crs_ghost),selected_set,PETSC_OWN_POINTER,a_selected_2);
        CHKERRQ(ierr);
      }
      else {
        ierr = PetscFree( selected_set );  CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy( &locState );                    CHKERRQ(ierr);
  }
  *a_crsGID = crsGID; /* output */

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGgraph_GEO

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
  Output Parameter:
   . a_Gmat
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGgraph_GEO"
PetscErrorCode PCGAMGgraph_GEO( PC pc,
                                const Mat Amat,
                                Mat *a_Gmat
                                )
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscInt verbose = pc_gamg->verbose;
  const PetscReal vfilter = pc_gamg->threshold;
  PetscMPIInt    mype,npe;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  Mat            Gmat;
  PetscBool  set,flg,symm;
  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGGgraph_GEO,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);

  ierr = MatIsSymmetricKnown(Amat, &set, &flg);        CHKERRQ(ierr);
  symm = (PetscBool)!(set && flg);

  ierr  = PCGAMGCreateGraph( Amat, &Gmat ); CHKERRQ( ierr );
  ierr  = PCGAMGFilterGraph( &Gmat, vfilter, symm, verbose ); CHKERRQ( ierr );

  *a_Gmat = Gmat;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGGgraph_GEO,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGcoarsen_GEO

  Input Parameter:
   . a_pc - this
   . a_Gmat - graph
  Output Parameter:
   . a_llist_parent - linked list from selected indices for data locality only
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGcoarsen_GEO"
PetscErrorCode PCGAMGcoarsen_GEO( PC a_pc,
                                  Mat *a_Gmat,
                                  PetscCoarsenData **a_llist_parent
                                  )
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscInt       Istart,Iend,nloc,kk,Ii,ncols;
  PetscMPIInt    mype,npe;
  IS             perm;
  GAMGNode *gnodes;
  PetscInt *permute;
  Mat       Gmat = *a_Gmat;
  MPI_Comm  wcomm = ((PetscObject)Gmat)->comm;
  MatCoarsen crs;

  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGCoarsen_GEO,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Gmat, &Istart, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-Istart);

  /* create random permutation with sort for geo-mg */
  ierr = PetscMalloc( nloc*sizeof(GAMGNode), &gnodes ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &permute ); CHKERRQ(ierr);
  
  for (Ii=Istart; Ii<Iend; Ii++) { /* locals only? */
    ierr = MatGetRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
    {
      PetscInt lid = Ii - Istart;
      gnodes[lid].lid = lid;
      gnodes[lid].degree = ncols;
    }
    ierr = MatRestoreRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
  }
  /* randomize */
  srand(1); /* make deterministic */
  if( PETSC_TRUE ) {
    PetscBool *bIndexSet;
    ierr = PetscMalloc( nloc*sizeof(PetscBool), &bIndexSet ); CHKERRQ(ierr);
    for ( Ii = 0; Ii < nloc ; Ii++) bIndexSet[Ii] = PETSC_FALSE;
    for ( Ii = 0; Ii < nloc ; Ii++)
    {
      PetscInt iSwapIndex = rand()%nloc;
      if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii)
      {
        GAMGNode iTemp = gnodes[iSwapIndex];
        gnodes[iSwapIndex] = gnodes[Ii];
        gnodes[Ii] = iTemp;
        bIndexSet[Ii] = PETSC_TRUE;
        bIndexSet[iSwapIndex] = PETSC_TRUE;
      }
    }
    ierr = PetscFree( bIndexSet );  CHKERRQ(ierr);
  }
  /* only sort locals */
  qsort( gnodes, nloc, sizeof(GAMGNode), petsc_geo_mg_compare );
  /* create IS of permutation */
  for(kk=0;kk<nloc;kk++) { /* locals only */
    permute[kk] = gnodes[kk].lid;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_OWN_POINTER, &perm);
  CHKERRQ(ierr);
  
  ierr = PetscFree( gnodes );  CHKERRQ(ierr);
  
  /* get MIS aggs */

  ierr = MatCoarsenCreate( wcomm, &crs ); CHKERRQ(ierr);
  ierr = MatCoarsenSetType( crs, MATCOARSENMIS ); CHKERRQ(ierr);
  ierr = MatCoarsenSetGreedyOrdering( crs, perm ); CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency( crs, Gmat ); CHKERRQ(ierr);
  ierr = MatCoarsenSetVerbose( crs, pc_gamg->verbose ); CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs( crs, PETSC_FALSE ); CHKERRQ(ierr);
  ierr = MatCoarsenApply( crs ); CHKERRQ(ierr);
  ierr = MatCoarsenGetData( crs, a_llist_parent ); CHKERRQ(ierr);
  ierr = MatCoarsenDestroy( &crs ); CHKERRQ(ierr);

  ierr = ISDestroy( &perm );                    CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGCoarsen_GEO,0,0,0,0);CHKERRQ(ierr);
#endif
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
#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_GEO"
PetscErrorCode PCGAMGProlongator_GEO( PC pc,
                                      const Mat Amat,
                                      const Mat Gmat,
                                      PetscCoarsenData *agg_lists,
                                      Mat *a_P_out
                                      )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;  
  const PetscInt  verbose = pc_gamg->verbose;
  const PetscInt  dim = pc_gamg->data_cell_cols, data_cols = pc_gamg->data_cell_cols;
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,nloc,my0,jj,kk,ncols,nLocalSelected,bs,*clid_flid;
  Mat            Prol;
  PetscMPIInt    mype, npe;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  IS             selected_2,selected_1;
  const PetscInt *selected_idx;

  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGProlongator_GEO,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  ierr = MatGetBlockSize( Amat, &bs );               CHKERRQ( ierr );
  nloc = (Iend-Istart)/bs; my0 = Istart/bs; assert((Iend-Istart)%bs==0);

  /* get 'nLocalSelected' */
  ierr = PetscCDGetMIS( agg_lists, &selected_1 );        CHKERRQ(ierr);
  ierr = ISGetSize( selected_1, &jj );               CHKERRQ(ierr);
  ierr = PetscMalloc( jj*sizeof(PetscInt), &clid_flid ); CHKERRQ(ierr);
  ierr = ISGetIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<jj;kk++) {
    PetscInt lid = selected_idx[kk];
    if( lid<nloc ) {
      ierr = MatGetRow(Gmat,lid+my0,&ncols,0,0); CHKERRQ(ierr);
      if( ncols>1 ) { /* fiter out singletons */
        clid_flid[nLocalSelected++] = lid;
      }
      else assert(0); /* filtered in coarsening */
      ierr = MatRestoreRow(Gmat,lid+my0,&ncols,0,0); CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
  ierr = ISDestroy( &selected_1 ); CHKERRQ(ierr); /* this is selected_1 in serial */

  /* create prolongator, create P matrix */
  ierr = MatCreate( wcomm, &Prol ); CHKERRQ(ierr);
  ierr = MatSetSizes(Prol,nloc*bs,nLocalSelected*bs,PETSC_DETERMINE,PETSC_DETERMINE); 
  CHKERRQ(ierr);
  ierr = MatSetBlockSizes( Prol, bs, bs ); CHKERRQ(ierr);
  ierr = MatSetType( Prol, MATAIJ );   CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Prol,3*data_cols,PETSC_NULL);   CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Prol,3*data_cols,PETSC_NULL,3*data_cols,PETSC_NULL); CHKERRQ(ierr);
  /* ierr = MatCreateAIJ( wcomm,  */
  /*                      nloc*bs, nLocalSelected*bs, */
  /*                      PETSC_DETERMINE, PETSC_DETERMINE, */
  /*                      3*data_cols, PETSC_NULL,  */
  /*                      3*data_cols, PETSC_NULL, */
  /*                      &Prol ); */
  /* CHKERRQ(ierr); */
  
  /* can get all points "removed" - but not on geomg */
  ierr =  MatGetSize( Prol, &kk, &jj ); CHKERRQ(ierr);
  if( jj==0 ) {
    if( verbose ) {
      PetscPrintf(wcomm,"[%d]%s ERROE: no selected points on coarse grid\n",mype,__FUNCT__);
    }
    ierr = PetscFree( clid_flid );  CHKERRQ(ierr);
    ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
    *a_P_out = PETSC_NULL;  /* out */
    PetscFunctionReturn(0);
  }

  {
    PetscReal *coords; 
    PetscInt   data_stride;
    PetscInt  *crsGID = PETSC_NULL;
    Mat        Gmat2;

    assert(dim==data_cols); 
    /* grow ghost data for better coarse grid cover of fine grid */
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET5],0,0,0,0);CHKERRQ(ierr);
#endif
    /* messy method, squares graph and gets some data */
    ierr = getGIDsOnSquareGraph( nLocalSelected, clid_flid, Gmat, &selected_2, &Gmat2, &crsGID );
    CHKERRQ(ierr);
    /* llist is now not valid wrt squared graph, but will work as iterator in 'triangulateAndFormProl' */
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET5],0,0,0,0);CHKERRQ(ierr);
#endif
    /* create global vector of coorindates in 'coords' */
    if (npe > 1) {
      ierr = PCGAMGGetDataWithGhosts( Gmat2, dim, pc_gamg->data, &data_stride, &coords );
      CHKERRQ(ierr);
    }
    else {
      coords = (PetscReal*)pc_gamg->data;
      data_stride = pc_gamg->data_sz/pc_gamg->data_cell_cols;
    }
    ierr = MatDestroy( &Gmat2 );  CHKERRQ(ierr);

    /* triangulate */
    if( dim == 2 ) {
      PetscReal metric,tm;
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET6],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = triangulateAndFormProl( selected_2, data_stride, coords,
                                     nLocalSelected, clid_flid, agg_lists, crsGID, bs, Prol, &metric );
      CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET6],0,0,0,0); CHKERRQ(ierr);
#endif
      ierr = PetscFree( crsGID );  CHKERRQ(ierr);
      
      /* clean up and create coordinates for coarse grid (output) */
      if (npe > 1) ierr = PetscFree( coords ); CHKERRQ(ierr);
      
      ierr = MPI_Allreduce( &metric, &tm, 1, MPIU_REAL, MPIU_MAX, wcomm );  CHKERRQ(ierr);
      if( tm > 1. ) { /* needs to be globalized - should not happen */
        if( verbose ) {
          PetscPrintf(wcomm,"[%d]%s failed metric for coarse grid %e\n",mype,__FUNCT__,tm);
        }
        ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
        Prol = PETSC_NULL;
      }
      else if( metric > .0 ) {
        if( verbose ) {
          PetscPrintf(wcomm,"[%d]%s worst metric for coarse grid = %e\n",mype,__FUNCT__,metric);
        }
      }
    } else {
      SETERRQ(wcomm,PETSC_ERR_LIB,"3D not implemented for 'geo' AMG");
    }
    { /* create next coords - output */
      PetscReal *crs_crds;
      ierr = PetscMalloc( dim*nLocalSelected*sizeof(PetscReal), &crs_crds ); 
      CHKERRQ(ierr);
      for(kk=0;kk<nLocalSelected;kk++){/* grab local select nodes to promote - output */
        PetscInt lid = clid_flid[kk];
        for(jj=0;jj<dim;jj++) crs_crds[jj*nLocalSelected + kk] = pc_gamg->data[jj*nloc + lid];
      }

      ierr = PetscFree( pc_gamg->data ); CHKERRQ( ierr );
      pc_gamg->data = crs_crds; /* out */
      pc_gamg->data_sz = dim*nLocalSelected;
    }
    ierr = ISDestroy( &selected_2 ); CHKERRQ(ierr); 
  }

  *a_P_out = Prol;  /* out */
  ierr = PetscFree( clid_flid );  CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGProlongator_GEO,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCCreateGAMG_GEO

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_GEO"
PetscErrorCode  PCCreateGAMG_GEO( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc->ops->setfromoptions = PCSetFromOptions_GEO;
  /* pc->ops->destroy        = PCDestroy_GEO; */
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->graph = PCGAMGgraph_GEO;
  pc_gamg->coarsen = PCGAMGcoarsen_GEO;
  pc_gamg->prolongator = PCGAMGProlongator_GEO;
  pc_gamg->optprol = 0;
  pc_gamg->formkktprol = 0;

  pc_gamg->createdefaultdata = PCSetData_GEO;
  
  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
                                            "PCSetCoordinates_C",
                                            "PCSetCoordinates_GEO",
                                            PCSetCoordinates_GEO);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
