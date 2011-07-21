/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include "petscvec.h" 
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define REAL PetscReal
#include <triangle.h>

#include <assert.h>
#include <petscblaslapack.h>

typedef enum { NOT_DONE=-2, DELETED=-1 } NState;

/* Private context for the GAMG preconditioner */
typedef struct{
  PetscInt       m_lid;      // local vertex index
  PetscInt       m_degree;   // vertex degree
} GNode;

int compare (const void *a, const void *b)
{
  return (((GNode*)a)->m_degree - ((GNode*)b)->m_degree);
}
static const PetscMPIInt target = -1;
/* -------------------------------------------------------------------------- */
/*
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info.

   Input Parameter:
   . a_perm - serial permutation of rows of local to process in MIS
   . Gmat - glabal matrix of graph (data not defined)
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "maxIndSetAgg"
PetscErrorCode maxIndSetAgg( IS a_perm,
                             Mat a_Gmat,
                             IS *a_selected,
                             IS *a_locals_llist
                             )
{
  PetscErrorCode ierr;
  PetscBool      isSeq, isMPI;
  Mat_SeqAIJ    *matA, *matB = 0;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat)->comm;
  Vec            locState,ghostState;
  PetscInt       num_fine_ghosts,kk,n,i,j,*idx,*ii,iter,Iend,my0;
  Mat_MPIAIJ    *mpimat = 0;
  PetscScalar   *cpcol_proc,*cpcol_state;
  PetscMPIInt    mype, pe;
  const PetscInt *perm_ix;
  PetscInt nDone = 0, nselected = 0;
  const PetscInt nloc = a_Gmat->rmap->n;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)a_Gmat, MATSEQAIJ, &isSeq ); CHKERRQ(ierr);
  ierr = PetscTypeCompare( (PetscObject)a_Gmat, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)a_Gmat->data;
    matA = (Mat_SeqAIJ*)mpimat->A->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
  } else if (isSeq) {
    matA = (Mat_SeqAIJ*)a_Gmat->data;
  }
  assert( matA && !matA->compressedrow.use );
  assert( matB==0 || matB->compressedrow.use );
  /* get vector */
  ierr = MatGetVecs( a_Gmat, &locState, 0 );         CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(a_Gmat,&my0,&Iend);  CHKERRQ(ierr);
  if( mpimat ) {
    PetscScalar v = (PetscScalar)(mype);
    PetscInt idx;
    for(kk=0,idx=my0;kk<nloc;kk++,idx++) {
      ierr = VecSetValues( locState, 1, &idx, &v, INSERT_VALUES );  CHKERRQ(ierr); /* set with PID */
    }
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray( mpimat->lvec, &cpcol_proc ); CHKERRQ(ierr); /* get proc ID in 'cpcol_proc' */
    ierr = VecDuplicate( mpimat->lvec, &ghostState ); CHKERRQ(ierr); /* need 2nd compressed col. of off proc data */
    ierr = VecGetLocalSize( mpimat->lvec, &num_fine_ghosts ); CHKERRQ(ierr);
    ierr = VecSet( ghostState, (PetscScalar)(NOT_DONE) );  CHKERRQ(ierr); /* set with UNKNOWN state */
  }
  else num_fine_ghosts = 0;

  {  /* need an inverse map - locals */
    PetscInt lid_cprowID[nloc], lid_gid[nloc];
    PetscInt id_llist[nloc+num_fine_ghosts]; /* linked list with locality info - output */
    PetscScalar lid_state[nloc];
    for(kk=0;kk<nloc;kk++) {
      id_llist[kk] = -1; /* terminates linked lists */
      lid_cprowID[kk] = -1;
      lid_gid[kk] = kk + my0;
      lid_state[kk] =  (PetscScalar)(NOT_DONE);
    }
    for(/*void*/;kk<nloc+num_fine_ghosts;kk++) {
      id_llist[kk] = -1; /* terminates linked lists */
    }
    /* set index into cmpressed row 'lid_cprowID' */
    if( matB ) {
      PetscInt m = matB->compressedrow.nrows;
      ii = matB->compressedrow.i;
      for (i=0; i<m; i++) {
        PetscInt lid = matB->compressedrow.rindex[i];
        lid_cprowID[lid] = i;
      }
    }
    /* MIS */
    ierr = ISGetIndices( a_perm, &perm_ix );     CHKERRQ(ierr);
    iter = 0;
    while ( nDone < nloc || true ) { /* asyncronous not implemented */
      iter++;
      if( mpimat ) {
        ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
      }
      for(kk=0;kk<nloc;kk++){
        PetscInt lid = perm_ix[kk]; 
        NState state = (NState)lid_state[lid];
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"[%d]%s %d) try gid %d in state %s\n",mype,__FUNCT__,iter,lid+my0, (state==NOT_DONE) ? "not done" : (state!=DELETED) ? "selected" : "deleted");
        if( state == NOT_DONE ) {
          /* parallel test, delete if selected ghost */
          PetscBool isOK = PETSC_TRUE;
          if( (i=lid_cprowID[lid]) != -1 ) { /* if I have any ghost neighbors */
            ii = matB->compressedrow.i; n = ii[i+1] - ii[i];
            idx = matB->j + ii[i];
            for( j=0 ; j<n ; j++ ) {
              PetscInt cpid = idx[j]; /* compressed row ID in B mat */
              pe = (PetscInt)cpcol_proc[cpid];
              state = (NState)cpcol_state[cpid];
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"\t\t[%d]%s %d) check cpid=%d on pe %d, fo r local gid %d\n",mype,__FUNCT__,iter,cpid,pe,lid+my0);
              if( state == NOT_DONE && pe > mype ) {
                isOK = PETSC_FALSE; /* can not delete */
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"\t\t\t[%d]%s %d) skip gid %d\n",mype,__FUNCT__,iter,lid+my0);
              }
              else if( state!=DELETED && state!=NOT_DONE ) { /* lid is now deleted, do it */
                nDone++;  lid_state[lid] = (PetscScalar)DELETED; /* delete this */
                PetscInt lidj = nloc + cpid;
                id_llist[lid] = id_llist[lidj]; id_llist[lidj] = lid; /* insert 'lid' into head of llist */
                isOK = PETSC_FALSE; /* all done with this vertex */
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"\t\t\t\t[%d]%s %d) deleted gid %d from local ghost id %d on pe %d, \n",mype,__FUNCT__,iter,lid+my0,cpid,pe);
                break;
              }
            }
          } /* parallel test */
          if( isOK ){ /* select this vertex */
            nDone++;
            lid_state[lid] =  (PetscScalar)(lid+my0);  /* SELECTED state encoded with global index */
            nselected++;
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s select gid %d\n",mype,__FUNCT__,lid+my0);
            /* delete neighbors - local */
            ii = matA->i; n = ii[lid+1] - ii[lid]; idx = matA->j + ii[lid];
            for (j=0; j<n; j++) {
              PetscInt lidj = idx[j]; assert(lidj>=0 && lidj<nloc);
              state = (NState)lid_state[lidj];
              if( state == NOT_DONE ){
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"\t\t\t\t[%d]%s delete local %d with %d\n",mype,__FUNCT__,lidj+my0,lid+my0);
                nDone++; lid_state[lidj] = (PetscScalar)DELETED;  /* delete this */
                id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
              }
            }
            /* no need to delete ghost neighbors */
          }
        } /* not done vertex */
      } /* vertex loop */
      /* update ghost states and count todos */
      if( mpimat ) {
        ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
        /* put lid state in 'locState' */
        ierr = VecSetValues( locState, nloc, lid_gid, lid_state, INSERT_VALUES ); CHKERRQ(ierr);
        ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
        ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
        /* scatter states, check for done */
        ierr = VecScatterBegin(mpimat->Mvctx,locState,ghostState,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        i = nloc - nDone; assert(i>=0);
        MPI_Allreduce ( &i, &j, 1, MPI_INT, MPI_SUM, wcomm ); /* synchronous version */
        ierr =   VecScatterEnd(mpimat->Mvctx,locState,ghostState,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        if( j == 0 ) break;
      }
      else break; /* all done */
if(mype==target)PetscPrintf(PETSC_COMM_SELF,"[%d]%s %d) finished MIS loop %d left to do\n",mype,__FUNCT__,iter,i);
    } /* outer parallel MIS loop */
    ierr = ISRestoreIndices(a_perm,&perm_ix);     CHKERRQ(ierr);

    /* create output IS of data locality in linked list */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nloc+num_fine_ghosts,id_llist,PETSC_COPY_VALUES,a_locals_llist);   CHKERRQ(ierr);

    /* make 'a_selected' - output */
    if( mpimat ) {
      ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    }
    for (j=0; j<num_fine_ghosts; j++) {
      if(mype==target)PetscPrintf(PETSC_COMM_SELF,"[%d]%s ghost %d in state %e\n",mype,__FUNCT__,j, cpcol_state[j]);
      if( (NState)cpcol_state[j] != DELETED && (NState)cpcol_state[j] != NOT_DONE ) {
        nselected++;
      }
    }
    {
      PetscInt selected_set[nselected];
      for(kk=0,j=0;kk<nloc;kk++){
        NState state = (NState)lid_state[kk];
        if( state != DELETED && state != NOT_DONE ) {
          selected_set[j++] = kk;
        }
      }
      for (kk=0; kk<num_fine_ghosts; kk++) {
        if( (NState)cpcol_state[kk] != DELETED && (NState)cpcol_state[kk] != NOT_DONE ) {
          selected_set[j++] = nloc + kk;
        }
      }
      assert(j==nselected);
      ierr = ISCreateGeneral(PETSC_COMM_SELF, nselected, selected_set, PETSC_COPY_VALUES, a_selected );
      CHKERRQ(ierr);
    }
    if( mpimat ) {
      ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    }
  } /* scoping */

  if(mpimat){
    ierr = VecRestoreArray( mpimat->lvec, &cpcol_proc ); CHKERRQ(ierr);
    ierr = VecDestroy( &ghostState ); CHKERRQ(ierr);
  }

  ierr = VecDestroy( &locState );                    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 triangulateAndFormProl

   Input Parameter:
   . a_selected_2 - list of selected local ID, includes selected ghosts
   . a_coords_x[a_nSubDom]  - 
   . a_coords_y[a_nSubDom] - 
   . a_selected_1 - selected IDs that go with base (1) graph
   . a_locals_llist - linked list with (some) locality info of base graph
   . a_crsGID[a_selected.size()] - make of ghost nodes to global index for prolongation operator
  Output Parameter:
   . a_Prol - prolongation operator
*/
#undef __FUNCT__
#define __FUNCT__ "triangulateAndFormProl"
PetscErrorCode triangulateAndFormProl( IS  a_selected_2, /* list of selected local ID, includes selected ghosts */
                                       PetscReal a_coords_x[], /* serial vector of local coordinates w/ ghosts */
                                       PetscReal a_coords_y[], /* serial vector of local coordinates w/ ghosts */
                                       IS  a_selected_1, /* list of selected local ID, includes selected ghosts */
                                       IS  a_locals_llist, /* linked list from selected vertices of aggregate unselected vertices */
				       const PetscInt a_crsGID[],
                                       Mat a_Prol, /* prolongation operator (output) */
                                       PetscReal *a_worst_best /* measure of worst missed fine vertex, 0 is no misses */
                                       )
{
  PetscErrorCode ierr;
  PetscInt       kk,bs=1,jj,tid,tt,sid,idx,nselected_1,nselected_2,nPlotPts;
  struct triangulateio in,mid;
  const PetscInt *selected_idx_1,*selected_idx_2,*llist_idx;
  PetscMPIInt    mype,npe;
  PetscInt Istart,Iend,nFineLoc,myFine0;

  PetscFunctionBegin;
  *a_worst_best = 0.0;
  ierr = MPI_Comm_rank(((PetscObject)a_Prol)->comm,&mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)a_Prol)->comm,&npe);  CHKERRQ(ierr);
  ierr = ISGetLocalSize( a_selected_1, &nselected_1 );        CHKERRQ(ierr);
  ierr = ISGetLocalSize( a_selected_2, &nselected_2 );        CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Prol,&Istart,&Iend);  CHKERRQ(ierr);
  nFineLoc = (Iend-Istart)/bs; myFine0 = Istart/bs;
  nPlotPts = nFineLoc; /* locals */
  /* traingle */
  /* Define input points - in*/
  in.numberofpoints = nselected_2;
  in.numberofpointattributes = 0;
  /* get nselected points */
  ierr = PetscMalloc( 2*(nselected_2)*sizeof(REAL), &in.pointlist ); CHKERRQ(ierr);
  ierr = ISGetIndices( a_selected_2, &selected_idx_2 );     CHKERRQ(ierr);
  for(kk=0,sid=0;kk<nselected_2;kk++,sid += 2){
    PetscInt lid = selected_idx_2[kk];
    in.pointlist[sid] = a_coords_x[lid];
    in.pointlist[sid+1] = a_coords_y[lid];
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

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                            */
  {
    char args[] = "pczQ"; /* c is needed ? */
    triangulate(args, &in, &mid, (struct triangulateio *) NULL );
    /* output .poly files for 'showme' */
    if( !PETSC_TRUE ) {
      static int level = 0;
      FILE *file; char fname[32]; 
 
      sprintf(fname,"C%d_%d.poly",level,mype); file = fopen(fname, "w");
      /*First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>*/
      fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++){
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
        sid += 2;
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
      /*First line: <# of triangles> <nodes per triangle> <# of attributes> */
      fprintf(file, "%d %d %d\n",mid.numberoftriangles,3,0);
      /*Remaining lines: <triangle #> <node> <node> <node> ... [attributes]*/
      for(kk=0,sid=0;kk<mid.numberoftriangles;kk++){
        fprintf(file, "%d %d %d %d\n",kk,mid.trianglelist[sid],mid.trianglelist[sid+1],mid.trianglelist[sid+2]);
        sid += 3;
      }
      fclose(file);

      sprintf(fname,"C%d_%d.node",level,mype); file = fopen(fname, "w");
      /*First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>*/
      //fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0);
      fprintf(file, "%d  %d  %d  %d\n",nPlotPts,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++){
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
        sid += 2;
      }
      sid /= 2;
      for(jj=0;jj<nFineLoc;jj++){
        PetscBool sel = PETSC_TRUE;
        for( kk=0 ; kk<nselected_2 && sel ; kk++ ){
          PetscInt lid = selected_idx_2[kk];
          if( lid == jj ) sel = PETSC_FALSE;
        }
        if( sel ) {
          fprintf(file, "%d %e %e\n",sid,a_coords_x[jj],a_coords_y[jj]);
          sid++;
        }
      }
      fclose(file);
      assert(sid==nPlotPts);
      level++;
    }
  }

  { /* form P - setup some maps */
    PetscInt clid_iterator;
    PetscInt nTri[nselected_2], node_tri[nselected_2][8];
    /* need list of triangles on node*/
    for(kk=0;kk<nselected_2;kk++) nTri[kk] = -1;
    for(tid=0,kk=0;tid<mid.numberoftriangles;tid++){
      for(jj=0;jj<3;jj++) {
        PetscInt cid = mid.trianglelist[kk++];
        if( nTri[cid] < 8 ) node_tri[cid][nTri[cid]++] = tid;
      }
    }

    /* find points and set prolongation */
    ierr = ISGetIndices( a_selected_1, &selected_idx_1 );     CHKERRQ(ierr);
    ierr = ISGetIndices( a_locals_llist, &llist_idx );     CHKERRQ(ierr);
    for( clid_iterator = 0 ; clid_iterator < nselected_1 ; clid_iterator++ ){
      PetscInt flid = selected_idx_1[clid_iterator]; assert(flid != -1);
      do{
        if( flid < nFineLoc ) {  /*could be a ghost*/
          const PetscInt fgid = flid + myFine0;
          /* compute shape function for gid */
          const PetscReal fcoord[3] = { a_coords_x[flid], a_coords_y[flid], 1.0 };
          PetscBool haveit = PETSC_FALSE; PetscScalar alpha[3]; PetscInt clids[3];
          for(jj=0 ; jj<nTri[clid_iterator] && !haveit ; jj++) {
            PetscScalar AA[3][3];
            PetscInt tid = node_tri[clid_iterator][jj];
            for(tt=0;tt<3;tt++){
              PetscInt clid2 = mid.trianglelist[3*tid + tt];
              PetscInt lid2 = selected_idx_2[clid2]; /* get to coordinate through fine grid */
              AA[tt][0] = a_coords_x[lid2]; AA[tt][1] = a_coords_y[lid2]; AA[tt][2] = 1.0;
              clids[tt] = clid2; /* store for interp */
            }
            for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
            /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
            PetscBLASInt N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
            dgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
            PetscBool have=PETSC_TRUE;
#define EPS 1.e-5
            for(tt=0; tt<3 && have ;tt++) if(alpha[tt] > 1.0+EPS || alpha[tt] < 0.0-EPS ) have=PETSC_FALSE;
            haveit = have;
          }
          if(!haveit) {
            /* brute force */
            PetscInt bestTID = -1; PetscScalar best_alpha = 1.e10; 
            for(tid=0 ; tid<mid.numberoftriangles && !haveit ; tid++ ){
              PetscScalar AA[3][3];
              for(tt=0;tt<3;tt++){
                PetscInt cid2 = mid.trianglelist[3*tid + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = a_coords_x[lid2]; AA[tt][1] = a_coords_y[lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }
              for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
              /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
              PetscBLASInt N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
              dgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
              PetscBool have=PETSC_TRUE;  PetscScalar worst=0.0, v;
              for(tt=0; tt<3 && have ;tt++) {
#define EPS2 1.e-2
                if(alpha[tt] > 1.0+EPS2 || alpha[tt] < 0.0-EPS2 ) have=PETSC_FALSE;
                if( (v=PetscAbs(alpha[tt]-0.5)) > worst ) worst = v;
              }
              if( worst < best_alpha ) {
                best_alpha = worst; bestTID = tid;
              }
              haveit = have;
            }
            if( !haveit ) {
              if( best_alpha > *a_worst_best ) *a_worst_best = best_alpha;
              /* use best one */
              PetscScalar AA[3][3];
              for(tt=0;tt<3;tt++){
                PetscInt cid2 = mid.trianglelist[3*bestTID + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = a_coords_x[lid2]; AA[tt][1] = a_coords_y[lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }
              for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
              /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
              PetscBLASInt N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
              dgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
            }
          }
          /* put in row of P */
          for(idx=0;idx<3;idx++){
            PetscReal shp = alpha[idx];
            if( PetscAbs(shp) > 1.e-6 ) {
              PetscInt cgid = a_crsGID[clids[idx]];
              PetscInt jj = cgid*bs, ii = fgid*bs; /* need to gloalize */
              for(tt=0;tt<bs;tt++,ii++,jj++){
                ierr = MatSetValues(a_Prol,1,&ii,1,&jj,&shp,INSERT_VALUES); CHKERRQ(ierr);
              }
            }
          }
        } /* local vertex test */
      } while( (flid=llist_idx[flid]) != -1 );
    }
    ierr = ISRestoreIndices( a_selected_2, &selected_idx_2 );     CHKERRQ(ierr);
    ierr = ISRestoreIndices( a_selected_1, &selected_idx_1 );     CHKERRQ(ierr);
    ierr = ISRestoreIndices( a_locals_llist, &llist_idx );     CHKERRQ(ierr);
    ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  free( mid.trianglelist );
  ierr = PetscFree( in.pointlist );  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   growCrsSupport - square graph, get 

   Input Parameter:
   . a_selected_1 - selected local indices (includes ghosts in input a_Gmat_1)
   . a_Gmat1 - graph that goes with 'a_selected_1'
   Output Parameter:
   . a_selected_2 - selected local indices (includes ghosts in output a_Gmat_2)
   . a_Gmat_2 - graph that is squared of 'a_Gmat_1'
   . a_crsGID[a_selected_2.size()] - map of global IDs of coarse grid nodes
   . a_num_ghosts - number of fine ghost nodes in new a_Gmat_2 (for convience only)
*/
#undef __FUNCT__
#define __FUNCT__ "growCrsSupport"
PetscErrorCode growCrsSupport( const IS a_selected_1,
                               const Mat a_Gmat1,
                               IS *a_selected_2,
			       Mat *a_Gmat_2,
			       PetscInt **a_crsGID,
			       PetscInt *a_num_ghosts
			       )
{
  PetscMPIInt    ierr,mype,npe;
  PetscInt       *crsGID, kk,my0,Iend,nloc,nLocalSelected,nSelected_1;
  const PetscInt *selected_idx;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat1)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Gmat1,&my0,&Iend); CHKERRQ(ierr); /* AIJ */
  nloc = Iend - my0; /* this does not change */
  /* get 'nLocalSelected' */
  ierr = ISGetLocalSize( a_selected_1, &nSelected_1 );        CHKERRQ(ierr);
  ierr = ISGetIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<nSelected_1;kk++){
    PetscInt lid = selected_idx[kk];
    if(lid<nloc) nLocalSelected++;
  }
  ierr = ISRestoreIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);

  if (npe == 1) { /* not much to do in serial */
    *a_num_ghosts = 0;
    ierr = PetscMalloc( nLocalSelected*sizeof(PetscInt), &crsGID ); CHKERRQ(ierr);
    for(PetscInt kk=0;kk<nLocalSelected;kk++) crsGID[kk] = kk;
    *a_Gmat_2 = 0;
    *a_selected_2 = a_selected_1; /* needed? */
  }
  else {
    PetscInt      idx,num_fine_ghosts,num_crs_ghost,myCrs0 = 0; /* pe 0 not defined */
    Mat_MPIAIJ   *mpimat2;
    Mat           Gmat2;
    Vec           locState;
    PetscScalar   *cpcol_state;

    /* grow graph to get wider set of selected vertices to cover fine grid, invalidates 'llist', geo mg specific */
    ierr = MatMatMult(a_Gmat1, a_Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );   CHKERRQ(ierr);

    mpimat2 = (Mat_MPIAIJ*)Gmat2->data;
    ierr = VecGetLocalSize( mpimat2->lvec, a_num_ghosts );          CHKERRQ(ierr);
    /* scane my coarse zero gid, set 'lid_state' with coarse ID */
    MPI_Exscan( &nLocalSelected, &myCrs0, 1, MPI_INT, MPI_SUM, wcomm );

    ierr = MatGetVecs( Gmat2, &locState, 0 );         CHKERRQ(ierr);
    ierr = VecSet( locState, (PetscScalar)(NOT_DONE) );  CHKERRQ(ierr); /* set with UNKNOWN state */
    ierr = ISGetIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
    for(kk=0;kk<nLocalSelected;kk++){
      PetscInt fgid = selected_idx[kk] + my0;
      PetscScalar v = (PetscScalar)(kk+myCrs0);
      ierr = VecSetValues( locState, 1, &fgid, &v, INSERT_VALUES );  CHKERRQ(ierr); /* set with PID */
    }
    ierr = ISRestoreIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetLocalSize( mpimat2->lvec, &num_fine_ghosts ); CHKERRQ(ierr);
    *a_num_ghosts = num_fine_ghosts;
    ierr = VecGetArray( mpimat2->lvec, &cpcol_state ); CHKERRQ(ierr); /* get proc ID in 'cpcol_proc' */
    for(kk=0,num_crs_ghost=0;kk<num_fine_ghosts;kk++){
      if( (NState)cpcol_state[kk] != NOT_DONE ) num_crs_ghost++;
    }
    ierr = PetscMalloc( (nLocalSelected+num_crs_ghost)*sizeof(PetscInt), &crsGID ); CHKERRQ(ierr); /* output */
    {
      PetscInt selected_set[nLocalSelected+num_crs_ghost];
      /* do ghost of 'crsGID' */
      for(kk=0,idx=nLocalSelected;kk<num_fine_ghosts;kk++){
        if( (NState)cpcol_state[kk] != NOT_DONE ){
          PetscInt cgid = (PetscInt)cpcol_state[kk];
          selected_set[idx] = nloc + kk;
          crsGID[idx++] = cgid;
        }
      }
      assert(idx==(nLocalSelected+num_crs_ghost));
      ierr = VecRestoreArray( mpimat2->lvec, &cpcol_state ); CHKERRQ(ierr);
      /* do locals in 'crsGID' */
      ierr = VecGetArray( locState, &cpcol_state ); CHKERRQ(ierr);
      for(kk=0,idx=0;kk<nloc;kk++){
        if( (NState)cpcol_state[kk] != NOT_DONE ){
          PetscInt cgid = (PetscInt)cpcol_state[kk];
          selected_set[idx] = kk;
          crsGID[idx++] = cgid;
        }
      }
      assert(idx==nLocalSelected);
      ierr = VecRestoreArray( locState, &cpcol_state ); CHKERRQ(ierr);

      ierr = ISCreateGeneral(PETSC_COMM_SELF,(nLocalSelected+num_crs_ghost),selected_set,PETSC_COPY_VALUES,a_selected_2);
      CHKERRQ(ierr);
    }
    ierr = VecDestroy( &locState );                    CHKERRQ(ierr);

    *a_Gmat_2 = Gmat2; /* output */
  }
  *a_crsGID = crsGID;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   createProlongation

   Input Parameter:
   . a_Amat - matrix on this fine level
   . a_coords - coordinates
   . a_dim - a_dimention
  Output Parameter:
   . a_P_out - prolongation operator to the next level
   . a_coords_out - coordinates of coarse grid points
*/
#undef __FUNCT__
#define __FUNCT__ "createProlongation"
PetscErrorCode createProlongation( Mat a_Amat,
                                   PetscReal a_coords[],
                                   const PetscInt a_dim,
                                   Mat *a_P_out,
                                   PetscReal **a_coords_out,
                                   PetscBool *a_isOK
                                   )
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,nloc,bs,jj,dir,kk,sid,my0;
  Mat            Prol;
  PetscMPIInt    mype,npe;
  Mat            Gmat,Gmat2;
  MPI_Comm       wcomm = ((PetscObject)a_Amat)->comm;
  PetscBool      isSeq, isMPI;
  Mat_SeqAIJ    *matB = 0;

  PetscFunctionBegin;
  *a_isOK = PETSC_TRUE;
  ierr = PetscTypeCompare((PetscObject)a_Amat, MATSEQAIJ, &isSeq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)a_Amat, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
  if (isMPI) {
    Mat_MPIAIJ    *mpimat = (Mat_MPIAIJ*)a_Amat->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
  }
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Amat,&Istart,&Iend);    CHKERRQ(ierr); /* BAIJ */
  ierr = MatGetBlockSize(a_Amat, &bs);    CHKERRQ(ierr);
  nloc = (Iend - Istart)/bs; my0 = Istart/bs;

  /* scale Amat (this should be a scalar matrix even if Amat is blocked) */
  {
    Vec diag;
    ierr = MatGetVecs(a_Amat, &diag, 0);    CHKERRQ(ierr);
    ierr = MatGetDiagonal( a_Amat, diag );  CHKERRQ(ierr);
    ierr = VecReciprocal( diag );         CHKERRQ(ierr);
    ierr = VecSqrtAbs( diag );            CHKERRQ(ierr);
    ierr = MatDuplicate( a_Amat, MAT_COPY_VALUES, &Gmat ); CHKERRQ(ierr); /* AIJ */
    ierr = MatDiagonalScale( Gmat, diag, diag );CHKERRQ(ierr);
    ierr = VecDestroy( &diag );           CHKERRQ(ierr);
    if(bs > 1){
      /* need to reduce to scalar, keep 1. on diag (max norm) */
      SETERRQ(wcomm,PETSC_ERR_SUP,"GAMG called with BAIJ matrix");
    }
  }
  ierr = MatGetOwnershipRange(Gmat,&Istart,&Iend);CHKERRQ(ierr); /* use AIJ from here */
  /* filter Gmat */
  {
    ierr = MatCreateMPIAIJ(wcomm,nloc*bs,nloc*bs,PETSC_DECIDE,PETSC_DECIDE,11,PETSC_NULL,3,PETSC_NULL,&Gmat2);CHKERRQ(ierr);
    const PetscScalar *vals;  PetscScalar v; const PetscInt *idx; PetscInt ncols;
    for (Ii=Istart; Ii<Iend; Ii++) {
      ierr = MatGetRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
      for(jj=0;jj<ncols;jj++){
        if( (v=PetscAbs(vals[jj])) > 0.02 ) { // hard wired filter!!!
          ierr = MatSetValues(Gmat2,1,&Ii,1,&idx[jj],&v,INSERT_VALUES); CHKERRQ(ierr);
        }
        /*else PetscPrintf(PETSC_COMM_SELF,"\t%s filtered %d, v=%e\n",__FUNCT__,Ii,vals[jj]);*/
      }
      ierr = MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Gmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Gmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
    Gmat = Gmat2;

    /* force compressed row storage for B matrix */
    if (npe > 1) {
      Mat_MPIAIJ *mpimat = (Mat_MPIAIJ*)Gmat->data;
      Mat_SeqAIJ *Bmat = (Mat_SeqAIJ*) mpimat->B->data;
      Bmat->compressedrow.check = PETSC_TRUE;
      ierr = MatCheckCompressedRow(mpimat->B,&Bmat->compressedrow,Bmat->i,Gmat->rmap->n,-1.0);CHKERRQ(ierr);   CHKERRQ(ierr);
      assert( Bmat->compressedrow.use );
    }
  }
  /* view */
  if(PETSC_FALSE) {
    PetscViewer        viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, "Gmat.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Gmat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }
  {
    GNode gnodes[nloc];
    PetscInt ncols, permute[nloc];
    PetscInt num_ghosts,nLocalSelected;
    Vec tmp_crds;
    PetscReal *coords[a_dim];
    IS permIS, llist_1, selected_1, selected_2;
    const PetscInt *selected_idx;
    PetscInt *crsGID;

    /* Mat subMat = Gmat; */
    ierr = MatGetOwnershipRange(Gmat,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) { /* locals only? */
      ierr = MatGetRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
      {
        PetscInt lid = Ii - Istart;
        gnodes[lid].m_lid = lid;
        gnodes[lid].m_degree = ncols;
      }
      ierr = MatRestoreRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
    }
    qsort( gnodes, nloc, sizeof(GNode), compare ); /* only sort locals */
    /* create IS of permutation */
    for(kk=0;kk<nloc;kk++) { /* locals only */
      permute[kk] = gnodes[kk].m_lid;
    }
    ierr = ISCreateGeneral( PETSC_COMM_SELF, (Iend-Istart), permute, PETSC_COPY_VALUES, &permIS ); CHKERRQ(ierr);

    /* SELECT COARSE POINTS */
    ierr = maxIndSetAgg( permIS, Gmat, &selected_1, &llist_1 ); CHKERRQ(ierr);

    /* get 'nLocalSelected' */
    ierr = ISGetLocalSize( selected_1, &ncols );        CHKERRQ(ierr);
    ierr = ISGetIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
    for(kk=0,nLocalSelected=0;kk<ncols;kk++){
      PetscInt lid = selected_idx[kk];
      if(lid<nloc) nLocalSelected++;
    }
    ierr = ISRestoreIndices( selected_1, &selected_idx );     CHKERRQ(ierr);

    /* create prolongator, create P matrix */
    ierr = MatCreateMPIAIJ( wcomm, nloc*bs, nLocalSelected*bs,
                            PETSC_DETERMINE, PETSC_DETERMINE,
                            3, PETSC_NULL, 2, PETSC_NULL,
                            &Prol );
    CHKERRQ(ierr);

    /* grow ghost data for better coarse grid cover of fine grid */
    ierr = growCrsSupport( selected_1, Gmat, &selected_2, &Gmat2, &crsGID, &num_ghosts ); CHKERRQ(ierr);

    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
    /* llist is now not valid wrt squared graph, but will work as iterator in 'triangulateAndFormProl' */

    /* create global vector of coorindates in 'coords' */
    ierr = VecCreate( wcomm, &tmp_crds );               CHKERRQ(ierr);
    ierr = VecSetSizes( tmp_crds, nloc, PETSC_DECIDE ); CHKERRQ(ierr);
    ierr = VecSetFromOptions( tmp_crds );               CHKERRQ(ierr);
    for(dir=0; dir<a_dim; dir++) {
      ierr = PetscMalloc( (num_ghosts+nloc)*sizeof(PetscReal), &coords[dir] ); CHKERRQ(ierr);
      /* set local, and global */
      for(kk=0; kk<nloc; kk++) {
        PetscInt gid = my0 + kk;
        PetscReal crd = a_coords[kk*a_dim + dir];
        coords[dir][kk] = crd;
        ierr = VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES ); CHKERRQ(ierr);
      }
      ierr = VecAssemblyBegin( tmp_crds ); CHKERRQ(ierr);
      ierr = VecAssemblyEnd( tmp_crds ); CHKERRQ(ierr);
      /* get ghost coords */
      if (npe > 1) {
        Mat_MPIAIJ *mpimat = (Mat_MPIAIJ*)Gmat2->data;
        PetscScalar *coord_arr;
        ierr = VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr =   VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetLocalSize(mpimat->lvec,&kk); CHKERRQ(ierr); /* debug */
        assert( kk == num_ghosts );
        ierr = VecGetArray( mpimat->lvec, &coord_arr );   CHKERRQ(ierr);
        for(kk=nloc,jj=0;jj<num_ghosts;kk++,jj++){
          coords[dir][kk] = coord_arr[jj];
        }
        ierr = VecRestoreArray( mpimat->lvec, &coord_arr ); CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&tmp_crds); CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat2 );  CHKERRQ(ierr);

    /* triangulate */
    if( a_dim == 2 ) {
      PetscReal metric;
      ierr = triangulateAndFormProl( selected_2, coords[0], coords[1], selected_1, llist_1, crsGID, Prol, &metric ); 
      CHKERRQ(ierr);
      if( metric > 1.0 ) *a_isOK = PETSC_FALSE;
    } else {
      SETERRQ(wcomm,PETSC_ERR_LIB,"3D not implemented");
    }
    ierr = PetscFree( crsGID );  CHKERRQ(ierr);
    for(kk=0; kk<a_dim; kk++) {
      ierr = PetscFree( coords[kk] ); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&llist_1); CHKERRQ(ierr);
    ierr = ISDestroy(&permIS); CHKERRQ(ierr);
    { /* create next coords - output */
      PetscReal *crs_crds;
      ierr = PetscMalloc( a_dim*nLocalSelected*sizeof(PetscReal), &crs_crds ); CHKERRQ(ierr);
      ierr = ISGetIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
      for(kk=0,sid=0;kk<nLocalSelected;kk++){/* grab local select nodes to promote - output */
        PetscInt lid = selected_idx[kk];
        for(jj=0;jj<a_dim;jj++,sid++) crs_crds[sid] = a_coords[a_dim*lid+jj];
      }
      assert(sid==2*nLocalSelected);
      ierr = ISRestoreIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
      *a_coords_out = crs_crds; /* out */
    }
    ierr = ISDestroy( &selected_1 ); CHKERRQ(ierr);
    if (npe > 1) {
      ierr = ISDestroy( &selected_2 ); CHKERRQ(ierr); /* this is selected_1 in serial */
    }
  }

  *a_P_out = Prol;  /* out */
  PetscFunctionReturn(0);
}
