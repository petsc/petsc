function ex1(np)
%
%   ex1(np) 
% creates a series of vectors in PETSc and displays them in Matlab
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
