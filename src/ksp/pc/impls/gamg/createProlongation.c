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

typedef enum { DELETED=-1, SELECTED=-2, NOT_DONE=-3 } NState;

/* Private context for the GAMG preconditioner */
typedef struct{
  PetscInt       m_lid;      // local vertex index
  PetscInt       m_degree;   // vertex degree
} GNode;

int compare (const void *a, const void *b)
{
  return (((GNode*)a)->m_degree - ((GNode*)b)->m_degree);
}

/* -------------------------------------------------------------------------- */
/*
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info.

   Input Parameter:
   . IS perm - serial permutation of rows of local to process in MIS
   . Gmat - glabal matrix of graph (data not defined)
   Output Parameter:
   . selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "maxIndSetAgg"
PetscErrorCode maxIndSetAgg( IS perm, 
                             Mat Gmat, 
                             IS *selected, 
                             IS *locals_llist )
{
  PetscErrorCode ierr;
  PetscBool      isSeq, isMPI;
  Mat_SeqAIJ    *matA, *matB = 0;
  MPI_Comm       wcomm = ((PetscObject)Gmat)->comm;
  Vec            locState,ghostState;
  PetscInt       num_ghosts,kk,n,i,j,*idx,*ii;
  Mat_MPIAIJ    *mpimat = 0;
  PetscScalar   *lid_state,*cpcol_proc,*cpcol_state;
  PetscMPIInt    mype, pe;
  const PetscInt *perm_ix;
  PetscInt nDone = 0, nselected = 0;
  const PetscInt nloc = Gmat->rmap->n;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)Gmat, MATSEQAIJ, &isSeq ); CHKERRQ(ierr);
  ierr = PetscTypeCompare( (PetscObject)Gmat, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)Gmat->data;
    matA = (Mat_SeqAIJ*)mpimat->A->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
  } else if (isSeq) {
    matA = (Mat_SeqAIJ*)Gmat->data;
  }
  assert( matB==0 || matB->compressedrow.use );
  assert( matA && !matA->compressedrow.use );
  /* get vector */
  ierr = MatGetVecs( Gmat, &locState, 0 );         CHKERRQ(ierr);
  if( mpimat ) {
    ierr = VecSet( locState, (PetscScalar)(mype) );  CHKERRQ(ierr); /* set with PID */
    ierr = VecScatterBegin(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray( mpimat->lvec, &cpcol_proc ); CHKERRQ(ierr); /* get proc ID in 'cpcol_proc' */
    ierr = VecDuplicate( mpimat->lvec, &ghostState ); CHKERRQ(ierr); /* need 2nd compressed col. of off proc data */
    ierr = VecGetLocalSize( mpimat->lvec, &num_ghosts ); CHKERRQ(ierr);
    ierr = VecSet( ghostState, (PetscScalar)(NOT_DONE) );  CHKERRQ(ierr); /* set with UNKNOWN state */
  } 
  else num_ghosts = 0;
PetscPrintf(PETSC_COMM_SELF,"[%d]%s num_ghosts %d num_loc %d\n",mype,__FUNCT__,num_ghosts, Gmat->rmap->n);  
  ierr = VecSet( locState, (PetscScalar)(NOT_DONE) );  CHKERRQ(ierr); /* set with PID */

  {  /* need an inverse map - locals */
    PetscInt lid_cprowID[nloc+1];
    PetscInt id_llist[nloc+num_ghosts+1]; /* linked list with locality info - output */
    for(kk=0;kk<nloc;kk++) {
      id_llist[kk] = -1; /* terminates linked lists */
      lid_cprowID[kk] = -1;
    }
    for(/*void*/;kk<nloc+num_ghosts;kk++) {
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
    ierr = ISGetIndices( perm, &perm_ix );     CHKERRQ(ierr);
    while ( nDone < nloc || true ) { /* asyncronous not implemented */
      if( mpimat ) {
        ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
      }
      ierr = VecGetArray( locState, &lid_state );      CHKERRQ(ierr);
      for(kk=0;kk<nloc;kk++){
        PetscInt lid = perm_ix[kk]; assert(lid>=0 && lid < nloc );
        NState state = (NState)lid_state[lid];
        if( state == NOT_DONE ) {
          /* parallel test, delete if selected ghost */
          PetscBool isOK = PETSC_TRUE;
          if( (i=lid_cprowID[lid]) != -1 ) { /*if I have any ghost neighbors*/
            ii = matB->compressedrow.i; n = ii[i+1] - ii[i];
            idx = matB->j + ii[i];
            for( j=0 ; j<n ; j++ ) {
              PetscInt cpid = idx[j]; /* compressed row ID in B mat */
              pe = (PetscInt)cpcol_proc[cpid];
              state = (NState)cpcol_state[cpid];
PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s check cpid=%d on pe %d, \n",mype,__FUNCT__,cpid,pe);
              if( state == NOT_DONE && pe > mype ) {
                isOK = PETSC_FALSE; /* can not delete */
              }
              else if( state == SELECTED ) { /* this is now deleted, do it */
                assert(pe>mype);
                nDone++;  lid_state[lid] = (PetscScalar)DELETED; /* delete this */
                PetscInt lidj = nloc + cpid;
                id_llist[lid] = id_llist[lidj]; id_llist[lidj] = lid; /* insert 'lid' into head of llist */
                isOK = PETSC_FALSE; /* all done with this vertex */
                break;
              }
            }
          } /* parallel test */
          if( isOK ){ /* select this vertex */
            nDone++;  lid_state[lid] =  (PetscScalar)SELECTED;  nselected++; 
//PetscPrintf(PETSC_COMM_WORLD,"%s select %d\n",__FUNCT__,lid);
            /* delete neighbors - local */
            ii = matA->i; n = ii[lid+1] - ii[lid]; idx = matA->j + ii[lid];
            for (j=0; j<n; j++) {
              PetscInt lidj = idx[j]; assert(lidj>=0 && lidj<nloc);
              state = (NState)lid_state[lidj];
              if( state == NOT_DONE ){
//PetscPrintf(PETSC_COMM_WORLD,"\t%s delete local %d with %d\n",__FUNCT__,jj,kk);
                nDone++; lid_state[lidj] = (PetscScalar)DELETED;  /* delete this */
                id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
              }
            }
            /* no need to delete ghost neighbors !?!?! */
          } 
        } /* not done vertex */
      } /* vertex loop */
      if( mpimat ) {
        ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
      }
      ierr = VecRestoreArray( locState, &lid_state );      CHKERRQ(ierr);
      /* scatter states, check for done */
      i = nloc - nDone; assert(i>=0);
      MPI_Allreduce ( &i, &j, 1, MPI_INT, MPI_SUM, wcomm ); /* correct style ??? */
      if( j == 0 ) break; /* synchronous version !!! */
PetscPrintf(PETSC_COMM_WORLD,"%s finished MIS loop %d left to do \n",__FUNCT__,j);
      ierr = VecScatterBegin(mpimat->Mvctx,locState,ghostState,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr =   VecScatterEnd(mpimat->Mvctx,locState,ghostState,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    } /* outer parallel MIS loop */
    /* create output IS of data locality in linked list */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nloc+num_ghosts,id_llist,PETSC_COPY_VALUES,locals_llist);   CHKERRQ(ierr);
  } /* scoping trick */
  ierr = ISRestoreIndices(perm,&perm_ix);     CHKERRQ(ierr);
  
  /* create output IS, count ghost selecteds */
  if( mpimat ) {
    ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    for (j=0; j<num_ghosts; j++) {
      if(  (NState)cpcol_state[j] == SELECTED ) {
        nselected++;
      }
    }
    ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
  }
  {
    PetscInt selected_set[nselected];
    ierr = VecGetArray( locState, &lid_state ); CHKERRQ(ierr);
    for (j=0,i=0; j<nloc; j++) {
      if(  (NState)lid_state[j] == SELECTED ) {
        selected_set[i++] = j;
      }
    }
    ierr = VecRestoreArray( locState, &lid_state ); CHKERRQ(ierr);
    if( mpimat ) {
      ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
      for (j=0; j<num_ghosts; j++) {
        if(  (NState)cpcol_state[j] == SELECTED ) {
          selected_set[i++] = nloc + j;
        }
      }
      ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    }
    assert(i==nselected);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, nselected,selected_set,PETSC_COPY_VALUES,selected );
    CHKERRQ(ierr);
  }

  if(mpimat){
    ierr = VecRestoreArray( mpimat->lvec, &cpcol_proc ); CHKERRQ(ierr);
    ierr = VecDestroy( &ghostState ); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray( locState, &lid_state );    CHKERRQ(ierr);
  ierr = VecDestroy( &locState );                    CHKERRQ(ierr);
PetscPrintf(PETSC_COMM_SELF,"[%d]%s nselected %d\n",mype,__FUNCT__,nselected);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 triangulateAndFormProl

   Input Parameter:
   . selected
   . coords
   . locals_llist
  Output Parameter:
   . Prol - prolongation operator
*/
#undef __FUNCT__
#define __FUNCT__ "triangulateAndFormProl"
PetscErrorCode triangulateAndFormProl( IS  selected, /* list of selected local ID, includes selected ghosts */
                                       Vec coords, /* serial vector of local+ghost coordinates */
                                       IS  locals_llist, /* linked list from selected vertices of aggregate unselected vertices */
                                       Mat Prol /* prolongation operator (output) */
                                       )
{
  PetscErrorCode ierr;
  PetscInt       kk,bs=1,jj,tid,tt,sid,idx,nselected,nloc_wg;
  PetscScalar   *lid_crd;
  struct triangulateio in,mid;
  const PetscInt *selected_idx,*llist_idx;
  PetscMPIInt    mype;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)Prol)->comm,&mype);  CHKERRQ(ierr);
  ierr = ISGetLocalSize( locals_llist, &nloc_wg );        CHKERRQ(ierr);
  /* traingle */
  ierr = ISGetLocalSize( selected, &nselected );      CHKERRQ(ierr);
  /* Define input points - in*/
  in.numberofpoints = nselected;
  in.numberofpointattributes = 0;
  /* get nselected points */
  ierr = PetscMalloc( 2*(nselected)*sizeof(REAL), &in.pointlist ); CHKERRQ(ierr);
  ierr = VecGetArray( coords, &lid_crd );             CHKERRQ(ierr);
  ierr = ISGetIndices( selected, &selected_idx );     CHKERRQ(ierr);
  for(kk=0,sid=0;kk<nselected;kk++){
    PetscInt lid = selected_idx[kk];
    for(jj=0;jj<2;jj++,sid++) in.pointlist[sid] = lid_crd[2*lid+jj];
  }
  assert(sid==2*nselected);
  ierr = ISRestoreIndices( selected, &selected_idx );     CHKERRQ(ierr);
  ierr = VecRestoreArray( coords,&lid_crd);  CHKERRQ(ierr);

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
    if(PETSC_TRUE) {
      static int level = 0;
      FILE *file; char fname[32]; 
 
      sprintf(fname,"C%d_%d.poly",level,mype);
      file = fopen(fname, "w");
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
      fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++){
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
        sid += 2;
      }
      fclose(file);
      level++;
    }
  }

  { /* form P - setup some maps */
    PetscInt Istart,Iend,nloc,my0,clid_iterator;
    PetscInt nTri[nselected], node_tri[nselected][8];
    /* need list of triangles on node*/
    for(kk=0;kk<nselected;kk++) nTri[kk] = -1;
    for(tid=0,kk=0;tid<mid.numberoftriangles;tid++){
      for(jj=0;jj<3;jj++) {
        PetscInt cid = mid.trianglelist[kk++];
        if( nTri[cid] < 8 ) node_tri[cid][nTri[cid]++] = tid;
      }
    }
    ierr = MatGetOwnershipRange(Prol,&Istart,&Iend);  CHKERRQ(ierr);
    nloc = (Iend-Istart)/bs; my0 = Istart/bs;

    /* find points and set prolongation */
    ierr = ISGetIndices( selected, &selected_idx );     CHKERRQ(ierr);
    ierr = ISGetIndices( locals_llist, &llist_idx );     CHKERRQ(ierr);
    for( clid_iterator = 0 ; clid_iterator < nselected ; clid_iterator++ ){
      PetscInt flid = selected_idx[clid_iterator]; assert(flid != -1);
      do{
        if( flid < nloc ) {  /*could be a ghost*/
          const PetscInt fgid = flid + my0;
          /* compute shape function for gid */
          const PetscReal fcoord[3] = { lid_crd[2*flid], lid_crd[2*flid+1], 1.0 };
          PetscBool haveit = PETSC_FALSE; PetscScalar alpha[3]; PetscInt clids[3];
          for(jj=0 ; jj<nTri[clid_iterator] && !haveit ; jj++) {
            PetscScalar AA[3][3];
            PetscInt tid = node_tri[clid_iterator][jj];
            for(tt=0;tt<3;tt++){
              PetscInt clid2 = mid.trianglelist[3*tid + tt];
              PetscInt lid2 = selected_idx[clid2]; /* get to coordinate through fine grid */
              AA[tt][0] = lid_crd[2*lid2]; AA[tt][1] = lid_crd[2*lid2 + 1]; AA[tt][2] = 1.0;
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
                PetscInt lid2 = selected_idx[cid2];
                AA[tt][0] = lid_crd[2*lid2]; AA[tt][1] = lid_crd[2*lid2 + 1]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }
              for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
              /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
              PetscBLASInt N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
              dgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
              PetscBool have=PETSC_TRUE;  PetscScalar worst = 0.0,v;
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
              /* use best one */
              PetscScalar AA[3][3];
              for(tt=0;tt<3;tt++){
                PetscInt cid2 = mid.trianglelist[3*bestTID + tt];
                PetscInt lid2 = selected_idx[cid2];
                AA[tt][0] = lid_crd[2*lid2]; AA[tt][1] = lid_crd[2*lid2 + 1]; AA[tt][2] = 1.0;
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
              PetscInt cgid = clids[idx]; assert(mype==0); /* need global */
              PetscInt jj = cgid*bs, ii = fgid*bs; /* need to gloalize */
              for(tt=0;tt<bs;tt++,ii++,jj++){
                ierr = MatSetValues(Prol,1,&ii,1,&jj,&shp,INSERT_VALUES); CHKERRQ(ierr);
              }
            }
          }
        } /* local vertex test */
      } while( (flid=llist_idx[flid]) != -1 );
    }
    ierr = ISRestoreIndices( selected, &selected_idx );     CHKERRQ(ierr);
    ierr = ISRestoreIndices( locals_llist, &llist_idx );     CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  free( mid.trianglelist );
  ierr = PetscFree( in.pointlist );  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   createProlongation

   Input Parameter:
   . Amat - matrix on this fine level
   . a_coords - coordinates
   . dim - a_dimention
  Output Parameter:
   . P_out - prolongation operator to the next level
   . a_coords_out - coordinates of coarse grid points 
*/
#undef __FUNCT__
#define __FUNCT__ "createProlongation"
PetscErrorCode createProlongation( Mat Amat,
                                   PetscReal a_coords[],
                                   const PetscInt a_dim,
                                   Mat *P_out, 
                                   PetscReal **a_coords_out
                                   )
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,nloc,bs,jj,kk,sid,my0;
  Mat            Prol;
  PetscMPIInt    mype;
  Mat            Gmat;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  PetscBool      isSeq, isMPI;
  Mat_SeqAIJ    *matB = 0;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)Amat, MATSEQAIJ, &isSeq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)Amat, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
  if (isMPI) {
    Mat_MPIAIJ    *mpimat = (Mat_MPIAIJ*)Amat->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
  }
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);    CHKERRQ(ierr); /* BAIJ */
  ierr = MatGetBlockSize(Amat, &bs);    CHKERRQ(ierr);
  nloc = (Iend - Istart)/bs; my0 = Istart/bs;
PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s nloc=%d, bs=%d\n",mype,__FUNCT__,nloc,bs);
  /* scale Amat (this should be a scalar matrix even if Amat is blocked) */
  {
    Vec diag; 
    ierr = MatGetVecs(Amat, &diag, 0);    CHKERRQ(ierr);
    ierr = MatGetDiagonal( Amat, diag );  CHKERRQ(ierr);
    ierr = VecReciprocal( diag );         CHKERRQ(ierr);
    ierr = VecSqrtAbs( diag );            CHKERRQ(ierr);
    ierr = MatDuplicate( Amat, MAT_COPY_VALUES, &Gmat ); CHKERRQ(ierr); /* AIJ */
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
    Mat Gmat2;
    ierr = MatCreateSeqAIJ(wcomm,nloc*bs,nloc*bs,15,PETSC_NULL,&Gmat2);CHKERRQ(ierr);

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
  }
  /* modify matrix for fast coarsening in MIS */
  if(PETSC_FALSE){
    Mat Gmat2; /* this also symmetrizes - needed if Amat is !sym */
    ierr = MatCreateNormal( Gmat, &Gmat2 );    CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
    Gmat = Gmat2;
  }
  if(!PETSC_TRUE) {
    PetscViewer        viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, "Gmat.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Gmat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }
  {
    GNode gnodes[nloc+1];
    PetscInt ncols, permute[nloc+1];
    PetscInt nSelected;
    Vec crdsVec,ghostCrdVec;
    IS permIS, llist, selected;

    /* Mat subMat = Gmat; */
    ierr = MatGetOwnershipRange(Gmat,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) { /* locals only? */
      ierr = MatGetRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr); 
      PetscInt lid = Ii - Istart;
      gnodes[lid].m_lid = lid;
      gnodes[lid].m_degree = ncols;
      ierr = MatRestoreRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
    }
    qsort (gnodes, nloc, sizeof(GNode), compare ); /* only sort locals */
    /* create IS of permutation */ 
    for (Ii=Istart; Ii<Iend; Ii++) { /* locals only */
      PetscInt lid = Ii - Istart;
      permute[lid] = gnodes[lid].m_lid;
    }
    ierr = ISCreateGeneral( PETSC_COMM_SELF, (Iend-Istart), permute, PETSC_COPY_VALUES, &permIS ); CHKERRQ(ierr); 

    /* select coarse points */
    ierr = maxIndSetAgg( permIS, Gmat, &selected, &llist ); CHKERRQ(ierr);

    /* create prolongator */
    ierr = ISGetLocalSize( selected, &nSelected ); CHKERRQ(ierr); assert(mype==0); // need to count locals in parallel!!!
    ierr = MatCreateSeqAIJ( wcomm, nloc*bs, nSelected*bs, 15, PETSC_NULL, &Prol );CHKERRQ(ierr);
    /* create global vector of coorindates */
    ierr = VecCreate( wcomm, &crdsVec );   CHKERRQ(ierr);
    ierr = VecSetBlockSize( crdsVec, a_dim ); CHKERRQ(ierr);
    ierr = VecSetSizes( crdsVec, a_dim*nloc, PETSC_DECIDE ); CHKERRQ(ierr);
    ierr = VecSetFromOptions( crdsVec ); CHKERRQ(ierr);
    /* set local */
    for(kk=0; kk<nloc; kk++) {
      PetscInt gid = my0 + kk;
      ierr = VecSetValuesBlocked(crdsVec, 1, &gid, &a_coords[kk*a_dim], INSERT_VALUES ); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin( crdsVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( crdsVec ); CHKERRQ(ierr);
    /* grow crdVec like Gmat, into local 'ghostCrdVec' with ghost val  */
    assert(mype==0); // need to get ghost values in parallel!!!
    
    /* triangulate */
    if( a_dim == 2 ) {
      ierr = triangulateAndFormProl( selected, crdsVec, llist, Prol ); CHKERRQ(ierr);
    } else {
      SETERRQ(wcomm,PETSC_ERR_LIB,"3D not implemented");
    }
    ierr = VecDestroy(&crdsVec); CHKERRQ(ierr);
    ierr = ISDestroy(&llist); CHKERRQ(ierr);
    ierr = ISDestroy(&permIS); CHKERRQ(ierr);
    { /* create next coords - output */
      PetscReal *crs_crds;
      const PetscInt *selected_idx;
      ierr = PetscMalloc( a_dim*(nSelected)*sizeof(PetscReal), &crs_crds ); CHKERRQ(ierr);
      ierr = ISGetIndices( selected, &selected_idx );     CHKERRQ(ierr);
      for(kk=0,sid=0;kk<nSelected;kk++){/* grab local select nodes to promote - output */
        PetscInt lid = selected_idx[kk];
        for(jj=0;jj<a_dim;jj++,sid++) crs_crds[sid] = a_coords[a_dim*lid+jj];
      }
      assert(sid==2*nSelected);
      ierr = ISRestoreIndices( selected, &selected_idx );     CHKERRQ(ierr);
      *a_coords_out = crs_crds; /* out */
    }
    ierr = ISDestroy(&selected); CHKERRQ(ierr);
  }
  ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);

  *P_out = Prol;  /* out */
  PetscFunctionReturn(0);
}
