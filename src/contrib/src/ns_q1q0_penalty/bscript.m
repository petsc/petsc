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

% Receive the vertex information (x,y) coordinates of vertices
cell_coords = receive(p);
% Receive list of vertices of all cells (4 per cell)
cell_df = receive(p);


%  Extract the x and y coords
N = length(cell_coords);
cells_x = cell_coords(1:2:N);
cells_y = cell_coords(2:2:N);

% Put each 4 into a row 
N = N/8;
cellx = zeros(4,N);
celly = zeros(4,N);
for i=1:N,
  cellx(:,i) = cells_x(4*i-3:4*i);
  celly(:,i) = cells_y(4*i-3 :4*i);
end
cell_df = cell_df + 1;

% now extract the solution at each point
cellz1 = zeros(4,N);
cellz2 = zeros(4,N);

flag = receive(p);

while flag > .5
  % Receive the solution vector
  solution = receive(p);
  flag = receive(p);	
  for i=1:N,	
    cellz1(:,i) = solution(cell_df( (8*i-7):2:8*i-1 ));
    cellz2(:,i) = solution(cell_df( (8*i-6):2:8*i  ));
  end
%
  figure(1)
  fill3(cellx,celly,cellz1,cellz1)
%
  figure(2)
  fill3(cellx,celly,cellz2,cellz2)
%
end

% Close connection to MPI program
closeport(p);


