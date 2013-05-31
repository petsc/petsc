function output = ex9busoutput(varargin)
%% Script to process and optionally plot output of 9 bus power grid dynamic 
%% simulation ex9bus.c
%% ex9busoutput returns a struct with machine dynamic variables, currents
%% and network voltages. Optionally plots the variables if input arguments
%% are provided. The inputs are strings with names of dynamic variables to plot
%%  Example:
%%  output = ex9busoutput('delta','w');
%%  will return output struct with all variables and plot the machine angle 
%% 'delta' and machine speed 'w' for all generators.
%%  Available variables that can be plotted are
%%  'delta' - generator angle
%%  'w'     - generator speed
%%  'Eqp'   - generator q-axis transient EMF
%%  'Edp'   - generator d-axis transient EMT
%%  'Id'    - generator d-axis current
%%  'Iq'    - generator q-axis current
%%  'Efd'   - exciter field voltage
%%  'RF'    - exciter stabilizer output
%%  'VR'    - exciter voltage regular output
%%  'Vm'    - network bus voltage magnitudes
%%  'Va'    - network bus voltage angles
%% Each variable is plotted on a seperate plot.
%% Note that ex9bus requires PETSc' Matlab utility function PetscBinaryRead 
%% to be in MATLAB's search path. PetscBinaryRead can be added to MATLAB's
%% search path by
%% >> petscbinmatlabpath = [PETSC_DIR,'/bin/matlab'];
%% >> addpath(petscbinmatlabpath);
%% where PETSC_DIR is location of PETSc directory.

ngen = 3;
nload = 3;
nbus = 9;
neqs_gen = 9*ngen;
t_idx = 1;
Eqp_idx = t_idx+[1:9:ngen*9];
Edp_idx = Eqp_idx+1;
delta_idx = Edp_idx+1;
w_idx = delta_idx+1;
Id_idx = w_idx+1;
Iq_idx = Id_idx+1;
Efd_idx = Iq_idx+1;
RF_idx = Efd_idx+1;
VR_idx = RF_idx+1;
VD_idx = neqs_gen + 1 + [1:2:2*nbus];
VQ_idx = neqs_gen + 1 + [2:2:2*nbus];

results = PetscBinaryRead('out.bin','double');
VD = results(VD_idx,:); VQ = results(VQ_idx,:);
Vm = sqrt(VD.^2 + VQ.^2);
Va = atan2(VQ,VD);
output.t =   results(t_idx,:);
output.Eqp = results(Eqp_idx,:);
output.Edp = results(Edp_idx,:);
output.delta = results(delta_idx,:);
output.w     = results(w_idx,:);
output.Id    = results(Id_idx,:);
output.Iq    = results(Iq_idx,:);
output.Efd   = results(Efd_idx,:);
output.RF    = results(RF_idx,:);
output.VR    = results(VR_idx,:);
output.Vm    = Vm;
output.Va    = Va;

if nargin
    for i = 1:nargin
        figure(i),plot(output.t,output.(varargin{i}));
        xlabel('Time (sec)');
        ylabel(varargin{i});
        legend(num2str((1:length(output.(varargin{i})(:,1)))'));
    end
end
