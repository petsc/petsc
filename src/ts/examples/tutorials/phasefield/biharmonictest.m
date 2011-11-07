
% Runs the heat equation solver for several mesh spacing and determines the order of accuracy of the method with respect to time

n = 5;
u = cell(n,1);
for i=1:n
        system(['./biharmonic -ts_monitor -snes_vi_monitor   -pc_type lu  -snes_max_it 250 -snes_converged_reason  -snes_ls_steptol 1.e-40  -ts_type cn    -da_refine  ' int2str(i+4) '     -kappa .00001 -ts_dt 2.96046e-07 -ts_monitor_solution -ts_monitor_solution_initial   -ts_final_time .011  -vi  -snes_rtol 1.e-15'])
%        system(['./biharmonic -ts_monitor -snes_monitor   -pc_type lu  -snes_max_it 250 -snes_converged_reason  -snes_ls_steptol 1.e-30  -ts_type cn    -da_refine  ' int2str(i+4) '  -vi  -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard -ts_monitor_solution -ts_monitor_solution_initial -ksp_monitor_true_residual -snes_ls_monitor -ts_final_time .011'])
%  system(['./biharmonic -ts_monitor -snes_ls_monitor -snes_vi_monitor -ksp_monitor_true_residual -snes_max_it 250  -snes_ls_steptol 1.e-30 -snes_rtol 1.e-25 -snes_atol 1.e-15 -pc_type lu -vi -snes_converged_reason   -ts_type cn    -da_refine ' int2str(i+4) '   -kappa .00001 -ts_dt 2.96046e-07 -cahn-hillard -ts_final_time 1. -ts_max_steps 30'])
  ii = 2^(i-1);
  ut = PetscBinaryRead('binaryoutput','precision','float128');
  u{i} = ut(1:ii:(320*ii));
end

ustar = (4*u{n} - u{n-1})/3;

for i=1:n
  norm(ustar - u{i})
end

   
