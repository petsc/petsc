From bsmith@mcs.anl.gov Wed Jul 28 12:11:14 1999
Status: RO
X-Status: 
Received: from mcs.anl.gov (cliff-9.mcs.anl.gov [140.221.9.17]) by antares.mcs.anl.gov (8.6.10/8.6.10)  with ESMTP
	id MAA06339; Wed, 28 Jul 1999 12:11:13 -0500
Received: (from daemon@localhost)
	by mcs.anl.gov (8.9.3/8.9.3) id MAA13256
	for petsc-maint-dist; Wed, 28 Jul 1999 12:10:38 -0500
Received: from vw.CS.Berkeley.EDU (vw.CS.Berkeley.EDU [128.32.46.238])
        by mcs.anl.gov (8.9.3/8.9.3) with ESMTP id MAA13252
        for <petsc-maint@mcs.anl.gov>; Wed, 28 Jul 1999 12:10:36 -0500
Received: from vw.CS.Berkeley.EDU (madams@localhost)
        by vw.CS.Berkeley.EDU (8.9.1a/8.9.1) with ESMTP id KAA04564;
        Wed, 28 Jul 1999 10:11:06 -0700 (PDT)
From: Mark Adams <madams@CS.Berkeley.EDU>
Message-Id: <199907281711.KAA04564@vw.CS.Berkeley.EDU>
X-Mailer: exmh version 2.0.2 2/24/98
To: petsc-maint@mcs.anl.gov
Cc: madams@vw.CS.Berkeley.EDU, petsc-maint@mcs.anl.gov
Subject: [PETSC #2769] MIS and coloring routines
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Date: Wed, 28 Jul 1999 10:11:06 -0700

Barry,

You mentioned a ways back an interest in maximal independent sets.  My MIS code
is old, ugly, (with lots of bells an whistles) and uses rather complex data
structures that evolved as I figured the MIS algorithm out and thus would not
be useful.  But, I just wrote a coloring routine (from scratch) that uses very
simple data structures, is very simple (once you figure the algorithm out), and
so it might be useful (its a simple coloring routine that just does MISs until
its done, BTW I've tested this and found that I get the same number of colors
as J and P on the test problem that is similar to my stuff, 27 point
stencil...).  This code is in C++ but should be easy to convert to C.  It uses
my hash table and you have a C version of this (in ctable.c) so this should not
be too difficult (one of my tables is a table of pointers - you should just
store indices in the table as casting a pointer to an integer is a problem on
the Origin/2000, or even and integer to a pointer...).  If you (or Satish,
or,...) have any questions feel free to ask.

Mark

typedef enum NdStTAG{SELECTED=0,DELETED,UNKNOWN}NodeState;

#undef __FUNC__  
#define __FUNC__ "NUMR_GRID::ColorVertices" 
/* NUMR_GRID::ColorVertices ************************************************
 *
 *  - colors nodes with colors 1,..,ncolors	
 *
 *  INPUT:
 *   - tag: MPI tag to use (uses tag,tag+1,...,tag+ncolors)
 *   - ncolors: the number of colors (out)
 *
 *  SIDE EFFECTS:
 *  RETURN: 
 *     - PETSc error code 
 */
int NUMR_GRID::ColorVertices( const int tag, int &ncolors )
{
  int ierr,todo,ii,xx,color,todo2,receive,doit,proc,*pb,ndone,jj,proctodo,its;
  int rbuf[6],sbuf[6]
  MPI_Status status; 
  TABLE      gid_ghost( 2*(nNodes - nLocalNd)+17 ); // number of ghosts+
  TABLE_INT  procTable, doneProcs, sentdoneProcs;
  TableiPos  ipos;
  
  // clear, make gid_ghost and procTable
  for( xx = 0 ; xx < nNodes ; xx++ ){           // "nodes" = [locals,ghosts]
    NUMR_NODE *curr = &nodes[xx];               // nLocalNd <= nNodes
    assert(curr->color()==0);    //curr->setColor(0);
    if( curr->IsGhost() ){                      // |ghosts| = nNodes-nLocalNd
      gid_ghost.Add( curr, curr->index.global()+1 ); 
      proc = curr->GetProc();                   // need processors 
      procTable.Add( proc + 1 );                // tables are 1 bases!!!!      

    }                                           
    else assert(xx < nLocalNd);
  }

  // colors
  proctodo = procTable.GetCount();
  for( todo = nLocalNd, color = 1 ; todo || proctodo ; color++ ) {
    // clear
    for( xx = 0, todo2 = 0 ; xx < nNodes ; xx++ ) {
      NUMR_NODE *curr = &nodes[xx];
      if( curr->color() == 0 ){ 
	curr->SetState(UNKNOWN);
	curr->Mark(FALSE);
	if( xx < nLocalNd ) todo2++; 
      }
    }
    // MIS, ignore colored (SELECTED) nodes
    for( its = 0 ; todo2 ; its++ ){
      // one MIS iteration
      for( xx = 0, ndone = 0 ; xx < nLocalNd ; xx++ ) {
	NUMR_NODE *curr = &nodes[xx];
	if( curr->GetState() == UNKNOWN ) {
	  // check for selectable
	  for( jj = 0, doit = TRUE; jj < curr->NumAdjac() && doit ; jj++ ) {
	    ii = curr->LidAdj[jj];  assert(ii>=0 && ii < nNodes);
	    NUMR_NODE *curr2 = &nodes[ii];
	    if( curr2->GetState() == SELECTED && curr2->color() == color ) {
	      doit = FALSE; // selected on this round
	      curr->SetState(DELETED);  assert(curr2->IsGhost());
	      todo2--;
	    }
	    else if( curr2->IsGhost() && curr2->GetState() == UNKNOWN ) {
	      if( curr2->GetProc() > myproc ) doit = FALSE;
	    }
	  }
	  if( doit ) { // select
	    curr->SetState(SELECTED); 
	    curr->setColor(color); todo--; assert( todo >= 0 );
	    todo2--;                       assert( todo2 >= 0 );
	    for( jj = 0; jj < curr->NumAdjac() ; jj++ ) { // delete
	      ii = curr->LidAdj[jj];  assert(ii>=0 && ii < nNodes);
	      NUMR_NODE *curr2 = &nodes[ii]; 
	      if( curr2->IsGhost() && curr2->GetState() != SELECTED ){
		proc = curr2->GetProc(); 
		pb = sbuf; *pb++ = SELECTED; *pb++ = curr2->index.global();
		*pb++ = curr->index.global(); *pb++ = color;
		// send(tag+color) < SELECTED, gid_DEL_gh, gid_SEL_loc, color >
		MPI_Send(sbuf,pb-sbuf,MPI_INT, proc, tag+color,Comm);
		if( curr2->GetState() == UNKNOWN ) curr2->SetState(DELETED);
	      }
	      if( curr2->GetState() == UNKNOWN ) {
		curr2->SetState(DELETED);  
		if( !curr2->IsGhost() ) todo2--; assert( todo2 >= 0 );
	      }	
	    }
	  }
	}
      }
      // receive(tag)
      do{ 
	ipos = procTable.GetHeadPosition();  
	while( ipos ) { 
	  proc = procTable.GetNext( ipos ) - 1;  // one based table!!!
	  do{ 
	    MPI_Iprobe( proc, tag+color, Comm, &receive, &status );
	    MPI_Get_count( &status, MPI_INT, &ii ); 
	    if( receive && ii > 1 ){
	      // recv(tag+color) <SEL,gid_DEL,gid2_SEL,color>|<DELETED,gid_gh>
	      ierr =  receiveState(gid_ghost,ndone,todo2,tag+color,proc,Comm);
	      CHKERRQ(ierr);
	    }
	    else if( receive && ii == 1 ) break; // capper, will recv latter
	  }while( receive && todo2 );
	}
      }while( ndone == 0 && todo2 );

      // send DELETED bound
      for(  xx = 0 ; xx < nLocalNd ; xx++ ) {
	NUMR_NODE *curr = &nodes[xx];
	if( curr->GetState() == DELETED && !curr->IsMarked() ) {
	  curr->Mark();
	  for( jj = 0; jj < curr->NumAdjac() ; jj++ ) { 
	    ii = curr->LidAdj[jj];  assert(ii>=0 && ii < nNodes);
	    NUMR_NODE *curr2 = &nodes[ii];
	    if( curr2->IsGhost() && curr2->GetState() != SELECTED ){
	      proc = curr2->GetProc();
	      pb = sbuf; *pb++ = DELETED; *pb++ = curr->index.global();
	      // send(tag+color) < DELETED, gid_loc >
	      MPI_Send( sbuf, pb-sbuf, MPI_INT, proc, tag+color, Comm);
	    }
	  }
	}
      }
    }

    // send capper
    ipos = procTable.GetHeadPosition();  
    while( ipos ) { 
      proc = procTable.GetNext(ipos) - 1; assert(proc>=0);
      if( todo != 0 || sentdoneProcs.Add(proc+1) ) {
	MPI_Send( &todo, 1, MPI_INT, proc, tag+color, Comm );
      }
    } 
    // receive(tag) for ghosts, and capper
    ipos = procTable.GetHeadPosition();  
    while( ipos ) { 
      proc = procTable.GetNext( ipos ) - 1;
      if( doneProcs.Find(proc+1) ) continue;
      while(1){ 
	MPI_Probe( proc, tag+color, Comm, &status );
	MPI_Get_count( &status, MPI_INT, &ii ); 
	if( ii > 1 ){
	  // recv(tag) <SEL,gid_DEL_loc,gid2_SEL_gh,color> or <DELETED,gid_gh>
	  ierr =  receiveState(gid_ghost,ndone,todo2,tag+color,proc,Comm);
	  CHKERRQ(ierr);
	}
	else if( ii == 1 ){
	  MPI_Recv( &jj, 1, MPI_INT, proc, tag+color, Comm, &status );
	  if( jj == 0 ) {
	    int add = doneProcs.Add(proc+1); assert(add); proctodo--;
	  }
	  break; // done with this proc
	}
      }
    }
  }

  // get max color and clear marks (optional)
  for( xx = 0, color = 0 ; xx < nLocalNd ; xx++ ) {
    NUMR_NODE *curr = &nodes[xx]; curr->Mark(FALSE);
    assert(curr->color() > 0);
    if( curr->color() > color ) color = curr->color();
    // debug
    for( jj = 0; jj < curr->NumAdjac() ; jj++ ) { 
      ii = curr->LidAdj[jj];  assert(ii>=0 && ii < nNodes);
      NUMR_NODE *curr2 = &nodes[ii];
      assert(curr2==curr || curr->color() != curr2->color());
    }
  }
  MPI_Allreduce( &color, &ncolors, 1, MPI_INT, MPI_MAX, Comm ); // barrier!
  
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "NUMR_GRID::receiveState" 
/* NUMR_GRID::receiveState ********************************************
 *
 *  INPUT:
 *   - gid_ghost:
 *   - ndone: (in/out)
 *   - todo2: (in/out)
 *   - tag: this and proc could just be the status ...
 *   - proc:
 *   - comm: 
 *
 *  SIDE EFFECTS:
 *  RETURN: 
 *     - PETSc error code 
 */
int NUMR_GRID::receiveState( TABLE &gid_ghost, int &ndone, int &todo2, 
			     const int tag, const int proc, MPI_Comm comm )
{
  int ii, cclr, gid, gid2, *pb, state;
  MPI_Status  status;
  
  // recv(tag) <SEL,gid_DEL_loc,gid2_SEL_gh,color> or <DELETED,gid_gh>
  MPI_Recv( rbuf, 6, MPI_INT, proc, tag, comm, &status );
  pb = rbuf; state = *pb++;  gid = *pb++;  
  if( state == (int)SELECTED ) {
    // recv(tag) < SELECTED, gid_DEL_loc, gid2_SEL_gh, color >
    gid2 = *pb++; cclr = *pb++; 
    MPI_Get_count( &status, MPI_INT, &ii ); assert(pb-rbuf == ii);
    // gid2_SELECTED_gh
    NUMR_NODE *curr2 = (NUMR_NODE*)gid_ghost.Find( gid2+1 ); assert(curr2);
    if( curr2->GetState() == UNKNOWN ) {
      curr2->SetState(SELECTED);     assert(curr2->color() == 0);
      curr2->setColor( cclr );
      ndone++; 
    }
    else assert( curr2->GetState() == SELECTED ); 
    // gid_DELETED_loc
    ii = gid - proc_gnode[myproc]; assert(ii>=0 && ii < nLocalNd);
    NUMR_NODE *curr = &nodes[ii]; 
    if( curr->GetState() == UNKNOWN ) {
      curr->SetState(DELETED); assert( curr->color() == 0 );
      todo2--;                       assert( todo2 >= 0 );
    } 
    else if( curr->GetState() == SELECTED ) assert(curr->color());
    else assert( curr->GetState() == DELETED );
  } 
  else { 
    MPI_Get_count( &status, MPI_INT, &ii ); assert(pb-rbuf == ii);
    assert(state == (int)DELETED );
    // recv(tag) < DELETED, gid_gh >
    NUMR_NODE *curr2 = (NUMR_NODE*)gid_ghost.Find( gid+1 ); assert(curr2);
    if( curr2->GetState() == UNKNOWN ) {
      curr2->SetState(DELETED); 
      ndone++;
    }
    else assert(curr2->GetState() == DELETED);
  }

  return 0;
}



