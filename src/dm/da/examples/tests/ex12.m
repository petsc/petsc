function ex12(np,opt)
%
%   ex12(np) 
% creates a series of vectors in PETSc and displays them in Matlab
%
% Run with option -on_error_attach_debugger to debug
%
%  Requires the Matlab mex routines in ${PETSC_DIR}/bin/matlab. To make
% these cd to ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab and run make BOPT=g matlabcodes
% then make sure that ${PETSC_DIR}/bin/matlab is in your Matlab PATH.
%
if (nargin < 1)
  np = 1;
end
if (nargin < 2) 
  opt = ' ';
end
time = 20;
err = launch(['ex12 -time ' int2str(time) ' -viewer_socket_machine ' getenv('HOST') opt],np);
if (err ~= 0) then 
  return;
end

p = openport;
for i=1:time,
  v = receive(p);
  plot(v); 
  pause(1);
end;
closeport(p);
