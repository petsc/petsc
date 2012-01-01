function ex12(np,opt)
%
%   ex12(np) 
% creates a series of vectors in PETSc and displays them in Matlab
%
% Run with option -on_error_attach_debugger to debug
%
%  Requires the Matlab mex routines in ${PETSC_DIR}/bin/matlab.
%  Make sure that ${PETSC_DIR}/bin/matlab is in your Matlab PATH.
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
