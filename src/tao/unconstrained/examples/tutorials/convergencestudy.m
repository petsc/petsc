
h = zeros(5,1);
e = zeros(5,1);
hold off
for i=1:5
  E = 8*2^i;
  syscommand = compose('./spectraladjointassimilation  -ts_adapt_dt_max 3.e-3 -E %d -N 2 -ncoeff 5  -a .1 -tao_grtol 1.e-12  -tao_gatol 1.e-12  -tao_max_it 7',[E])
  [status,result] = system(syscommand{1});
  eval(result);
  h(i) = history(8,2);
  e(i) = E;
  if (mod(i,2) == 1)
    yyaxis left
    semilogy(history(:,1),history(:,2));
    ylabel('Continuum Error');
    yyaxis right
    semilogy(history(:,1),history(:,3));
    ylabel('Objective function');
    hold on
  end
end
legend('16 elements','64 elements','256 elements','16 elements','64 elements','256 elements')
xlabel('Iteration')
print('convergencestudy-h','-dpdf');

hold off
clf
loglog(e,h,'-+');
xlabel('1/h');
ylabel('Continuum Error');
print('convergencestudy-h-2','-dpdf');


clf
h = zeros(4,1);
e = zeros(4,1);
for i=1:4
  N = 1 + 3*i;
  syscommand = compose('./spectraladjointassimilation  -ts_adapt_dt_max 3.e-3 -E 8 -N %d -ncoeff 5  -a .1 -tao_grtol 1.e-12  -tao_gatol 1.e-12  -tao_max_it 7',[N])
  [status,result] = system(syscommand{1});
  eval(result);
  e(i) = N;
  h(i) = history(8,2)
  yyaxis left
  semilogy(history(:,1),history(:,2));
  ylabel('Continuum Error');
  yyaxis right
  semilogy(history(:,1),history(:,3));
  ylabel('Objective function');
  hold on
end
legend('Order 2','Order 5','Order 8','Order 11','Order 2','Order 5','Order 8','Order 11')
xlabel('Iteration')
print('convergencestudy-p','-dpdf');

hold off
clf
loglog(e,h,'-+');
xlabel('Polynomial order');
ylabel('Continuum Error');
print('convergencestudy-p-2','-dpdf');
