function ex1(np)
%
%   ex1(np) 
% creates a series of vectors in PETSc and displays them in Matlab
%
%  Requires the Matlab mex routines in src/viewer/impls/matlab. To make
% these cd to src/viewer/impls/matlab and run make BOPT=g matlabcodes
% then make sure that src/viewer/impls/matlab is in your Matlab PATH.
%
time = 20;
err = launch(['ex1 -on_error_attach_debugger -time ' int2str(time)] ,np);
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
