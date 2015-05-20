function ex12(np,opt)
%
%   ex12(np)
% creates a series of vectors in PETSc and displays them in Matlab
%
% Run with option -on_error_attach_debugger to debug
%
%  Requires the Matlab mex routines in ${PETSC_DIR}/share/petsc/matlab and ${PETSC_DIR}/${PETSC_ARCH}/lib/matlab.
%  Make sure that ${PETSC_DIR}/share/petsc/matlab and ${PETSC_DIR}/${PETSC_ARCH}/lib/matlab is in your MATLABPATH or
%  $prefix/share/petsc/matlab and $prefix/lib/matlab if you ran ./configure with --prefix
%
if (nargin < 1)
  np = 1;
end
if (nargin < 2)
  opt = ' ';
end
time = 20;
launch(['./ex12 -time ' int2str(time)  opt],np);

p = PetscOpenSocket;
for i=1:time,
  v = PetscBinaryRead(p);
  plot(v);
  pause(1);
end;
close(p);
