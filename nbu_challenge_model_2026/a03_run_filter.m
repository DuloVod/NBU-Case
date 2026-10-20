close all; clear all; home;

%% Load model
tmp = load('results\a02_read_model\model.mat');
m = tmp.m;

disp(' ');
disp(['Used model file: ' get(m, 'FileName')]);

%% Load data, introduce ranges
load('results\a01_read_data\read_data.mat');

dq.firsthist = qq(2016,1); %qq(2015,1);
dq.lasthist = qq(2026,1); %qq(2025,3);

dq.firstfore = dq.lasthist + 1;
dq.lastfore = qq(2028,4);

dq.rnghist = dq.firsthist:dq.lasthist;
dq.rngfilt = dq.firsthist:dq.lastfore;
dq.rngfcast = dq.firstfore:dq.lastfore;

%% Multipliers for wartime structural breaks
mult = {};

mult.std_res_l_gdp_tnd = Series;
mult.std_res_l_gdp_tnd(dq.rngfilt) = 0;
mult.std_res_l_gdp_tnd(qq(2022,1):qq(2022,2)) = 0.75;

% mult.std_res_l_rer_tnd = Series;
% mult.std_res_l_rer_tnd(dq.rngfilt) = 0;
% mult.std_res_l_rer_tnd(qq(2022,1):qq(2022,2)) = 1;

%% Tunes on gaps (optional)
dq.L_GDP_GAP = Series;
dq.L_GDP_GAP(qq(2015,1)) = 0;
dq.L_GDP_GAP(qq(2022,2)) = -18;

% dq.L_RER_GAP = Series;
% dq.L_RER_GAP(qq(2022,3)) = 3; %5;

%% Filter
[mfilt, filtdb, se, delta, pe] = filter(m, dq, dq.rngfilt, ...
  'deviation',  false, ...
  'std',        mult, ...
  'relative',   true);

%% Save results into .mat and .csv
[~,~,~] = mkdir(['results\' mfilename]);
save(['results\' mfilename '\filter_data.mat'], 'filtdb', 'dq'); 
databank.toCSV(filtdb.mean,['results\' mfilename '\filter_data.csv']); 

disp([mfilename ' Done!']);

%% Quick plot
disp('Wait for a couple of figures');
a = filtdb.mean;

% Main trends
fig = figure ('name', 'Main trends','Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(dq.rnghist, [a.l_gdp a.l_gdp_tnd], 'linewidth', 2); grid on; title('GDP (ln*100)', 'fontsize', 20); highlight(dq.rngfcast);
subplot(1,2,2)
plot(dq.rnghist, [a.l_rer a.l_rer_tnd], 'linewidth', 2); grid on; title('RER (ln*100)', 'fontsize', 20); highlight(dq.rngfcast);
saveas(fig, ['results\' mfilename '\' 'gaps_trends' '.png']);

% Main variables
figure ('name', 'Main variables','Position', [100, 100, 1200, 600]);
subplot(2,2,1)
plot(dq.rnghist, [a.d4l_cpi a.dl_cpi_tar], 'linewidth', 2); grid on; title('CPI, % y-o-y'); highlight(dq.rngfcast); legend 'Inflation' '5% target';
subplot(2,2,2)
plot(dq.rnghist, exp(a.l_er/100), 'linewidth', 2); grid on; title ('UAH/USD'); highlight(dq.rngfcast);
subplot(2,2,3)
plot(dq.rnghist, [a.l_gdp a.l_gdp_tnd], 'linewidth', 2); grid on; title ('GDP (ln*100)'); highlight(dq.rngfcast); legend 'Level' 'Trend';
subplot(2,2,4)
plot(dq.rnghist, [a.ir_pol a.rir_pol], 'linewidth', 2); grid on; title('Interest rate, %'); highlight(dq.rngfcast); legend 'Nominal' 'Real';

% Inflation and interest rate
fig = figure('name', 'Inflation and interest rate', 'Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(dq.rnghist, [a.d4l_cpi a.dl_cpi_tar], 'linewidth', 2); grid on; title('CPI, % y-o-y', 'fontsize', 20); highlight(dq.rngfcast); legend({'Headline', '5% target'}, 'Location', 'best', 'fontsize', 14);
subplot(1,2,2)
plot(dq.rnghist, a.ir_pol, 'linewidth', 2); grid on; title('Interest rate, %', 'fontsize', 20); highlight(dq.rngfcast);
saveas(fig, ['results\' mfilename '\' 'cpi_ir' '.png']);

% ER
fig = figure('name', 'UAH/EUR exchange rate','Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(dq.rnghist, exp(a.l_er/100), 'linewidth', 2); grid on; title ('Nominal UAH/EUR exchange rate', 'fontsize', 16);
subplot(1,2,2)
plot(dq.rnghist, [a.l_rer a.l_rer_tnd], 'linewidth', 2); grid on; title('Real UAH/EUR exchange rate (RER)', 'fontsize', 16);
saveas(fig, ['results\' mfilename '\' 'ers' '.png']);

% ER
fig = figure('name', 'UAH/EUR exchange rate','Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(dq.rnghist, exp(a.l_er/100), 'linewidth', 2); grid on; %title ('Nominal UAH/USD exchange rate', 'fontsize', 16);
saveas(fig, ['results\' mfilename '\' 'secret' '.png']);

disp('The figures are ready');