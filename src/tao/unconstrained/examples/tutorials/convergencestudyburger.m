fontsize_labels = 14;
fontsize_grid   = 12;
fontname = 'Times';

%
%   Uses for the initial guess the analytic solution; if one uses the other initial guess the code only convergences a very small amount
%   ./burgers_spectral  -ts_adapt_dt_max 3.e-3 -E 512 -N 2  -tao_grtol 1.e-12  -tao_gatol 0  -tao_max_it 30 -mu .001

n = [8 16 32 64 128 256 512];
e = [3.31018e-05 5.68753e-05 2.02074e-05 5.46704e-06 1.39818e-06 3.51656e-07 8.79429e-08];

p = [8    24    40    56    72];
f = [ 3.31018e-05 1.68731e-05 7.13636e-06 3.83633e-07 6.36686e-09];

hold off
clf
set(gca,'FontName',fontname)
set(gca,'FontSize',fontsize_grid)
set(gca,'FontSize',fontsize_labels)
loglog(n,e,'-+','Markersize',6,'LineWidth',2);
hold on
loglog(p,f,'-*','Markersize',6,'LineWidth',2);
xlabel('Number of GLL points');
ylabel('Analytic Error');
print('convergencestudy-burgers','-depsc');

