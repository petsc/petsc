%
%  Matlab graphics routine that may be used with the main program.  This uses
%  sockets for faster visualization than writing intermediate output to a file.
%
%  Usage Instructions:
%  - First, compile Matlab programs in petsc/src/viewer/impls/matlab/ via
%      make BOPT=g openport closeport receive 
%    to generate the executables openport.mex4, closeport.mex4, and receive.mex4
%  - Be sure to set the MATLABPATH environmental variable with a command such as
%      setenv MATLABPATH $PETSC_DIR/src/viewer/impls/matlab
%    (or append this to your existing Matlab path)
%  - Then run both of the following:
%      - this script (mscript.m) in Matlab (via mscript)
%      - the example program on any machine, via
%            main -f <grid_name> -viewer_matlab_machine machine_name -matlab_graphics
%        for example,
%            main -f grids/B -viewer_matlab_machine merlin.mcs.anl.gov -matlab_graphics
%
% Open the connection to the MPI program
p = openport;        
%
% Receive the solution vector
solution = receive(p);
%
% Recive more solution vectors
tsteps = 4;
for i=1:tsteps,
  tsolution(:,i) = receive(p);
end
%
% Receive the vertex information (x,y) coordinates of vertices
vertices = receive(p);
%
% Receive list of vertices of all cells (4 per cell)
cells = receive(p);
%
% Close connection to MPI program
closeport(p);
%
% Construct arrays for x and y coordinates
[m,n] = size(vertices);
x = vertices(1:2:m);
y = vertices(2:2:m);
%
% Construct arrays for cells (4 vertices per cell)
[m,n] = size(cells);
m = m/4;
cells2 = zeros(4,m);
cells2(:) = cells + 1;
cellx = zeros(4,m);
celly = zeros(4,m);
cellz1 = zeros(4,m);
cellz2 = zeros(4,m);
for i=1:m,
  cellx(:,i) = x(cells2(:,i));
  celly(:,i) = y(cells2(:,i));
% remember in matlab indexing starts with 1 %
  cellz1(:,i) = solution(2*cells2(:,i)-1);
  cellz2(:,i) = solution(2*cells2(:,i));
end
%
figure(1)
fill3(cellx,celly,cellz1,cellz1)
%
figure(2)
fill3(cellx,celly,cellz2,cellz2)

for j= 1:tsteps,
  for i=1:m, 
    cellz1(:,i,j) = tsolution((2*cells2(:,i)-1),j);
    cellz2(:,i,j) = tsolution((2*cells2(:,i)),j);
  end
end

for j= 1:tsteps,
figure(1)
fill3(cellx,celly,cellz1(:,:,j),cellz1(:,:,j))
%
figure(2)
fill3(cellx,celly,cellz2(:,:,j),cellz2(:,:,j))
end


