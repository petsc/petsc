c
c  Parallel array sizes, including ghost points, for the
c  mesh coordinates.  Space is allocated in UserSetLocalMesh().

      double precision x(cx1:cxn,cy1:cyn,cz1:czn)
      double precision y(cx1:cxn,cy1:cyn,cz1:czn)
      double precision z(cx1:cxn,cy1:cyn,cz1:czn)

c  These parameters enable use of either the local or
c  global mesh; they are set in the routine parsetup().
c  For the parallel case, we clearly want to use only
c  the local part, but currently some of the post-processing
c  requires the global mesh.  This will eventually be upgraded.
c

