!
!  Parallel array sizes, including ghost points, for the
!  mesh coordinates.  Space is allocated in UserSetLocalMesh().

      double precision x(cx1:cxn,cy1:cyn,cz1:czn)
      double precision y(cx1:cxn,cy1:cyn,cz1:czn)
      double precision z(cx1:cxn,cy1:cyn,cz1:czn)

!  These parameters enable use of either the local or
!  global mesh; they are set in the routine parsetup().
!  For the parallel case, we clearly want to use only
!  the local part, but currently some of the post-processing
!  requires the global mesh.  This will eventually be upgraded.
!

