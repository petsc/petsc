c
c  Parallel array sizes, including ghost points, for the
c  mesh coordinates.  Space is allocated in UserSetLocalMesh().

      Double x(cx1:cxn,cy1:cyn,cz1:czn)
      Double y(cx1:cxn,cy1:cyn,cz1:czn)
      Double z(cx1:cxn,cy1:cyn,cz1:czn)

c  These parameters enable use of either the local or
c  global mesh; they are set in the routine parsetup().
c  For the parallel case, we clearly want to use only
c  the local part, but currently some of the post-processing
c  requires the global mesh.  This will eventually be upgraded.
c
c  local:
c      Double x(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
c      Double y(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
c      Double z(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
c  global:
c      Double x(ni,nj,nk),y(ni,nj,nk),z(ni,nj,nk)
c
