function ex12(np,opt)
%
%   ex12(np) 
% creates a series of vectors in PETSc and displays them in Matlab
%
%  Requires the Matlab mex routines in src/sys/src/viewer/impls/socket. To make
% these cd to src/sys/src/viewer/impls/socket and run make BOPT=g matlabcodes
% then make sure that src/sys/src/viewer/impls/socket is in your Matlab PATH.
%
if (nargin < 1)
  np = 1;
end
if (nargin < 2) 
  opt = ' ';
end
time = 20;
err = launch(['ex12 -on_error_attach_debugger -time ' int2str(time) opt] ,np);
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
