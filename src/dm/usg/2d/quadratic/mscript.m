%
%  Matlab graphics routine that may be used with the main program
%
%  main -viewer_matlab_machine machinename -matlabgraphics
%
% First, compile programs in petsc/src/viewer/impls/matlab/
%    make BOPT=g openport closeport receive 
% to generate *.mex4
% Also, be sure to set the MATLABPATH environmental variable with a
% command such as
%    setenv MATLABPATH $PETSC_DIR/src/viewer/impls/matlab
%
% Open the connection to the MPI program
p = openport;        
%
% Receive the solution vector
solution = receive(p);
%
% Receive the vertex information (x,y) coordinates of vertices
vertices = receive(p);
%
% Receive list of vertices of all cells (6 per cell)
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
% Construct arrays for cells (6 vertices per cell)
[m,n] = size(cells);
m = m/6;
cells2 = zeros(6,m);
cells2(:) = cells + 1;
cellx = zeros(6,m);
celly = zeros(6,m);
cellz = zeros(6,m);
for i=1:m,
  cellx(:,i) = x(cells2(:,i));
  celly(:,i) = y(cells2(:,i));
  cellz(:,i) = solution(cells2(:,i));
end
%
figure(1)
fill(cellx,celly,cellz)
%
figure(2)
fill3(cellx,celly,cellz,cellz)

