close all; clear all; home;

%% Load model
tmp = load('results\a02_read_model\model.mat');
m = tmp.m;

%% Load filter data
load('results\a03_run_filter\filter_data.mat');
initdb = dbclip(filtdb, dq.rnghist);
db = filtdb.mean;
rngfcast = dq.rngfcast;

%% Change parameters on the forecasting horizon
mfcast = m;
% mfcast = solve(mfcast);
% mfcast = sstate(mfcast);
% sschk = chksstate(mfcast);

%% Set judgments
simplan = plan(mfcast, rngfcast);

%% External sector (external assumptions from the excel file for the whole forecasting horizon)
list_var_plan   = {'ir_ea', 'l_gdp_ea_gap', 'dl_cpi_ea'};
list_shck_plan  = {'res_ir_ea', 'res_l_gdp_ea_gap', 'res_dl_cpi_ea'};
                  
simplan = exogenize(simplan, list_var_plan, rngfcast);
simplan = endogenize(simplan, list_shck_plan, rngfcast);
dbext = filtdb.mean;
dbext = dbbatch(dbext, '$0', 'dbext.$0', 'stringList', list_var_plan, 'fresh', true);
initdb.mean = dbextend(initdb.mean, dbext);

%% Short-term forecast (optional for a couple of quarters if you have smth from other sources)
% simplan = exogenize(simplan, 'l_er', qq(2026,2):qq(2026,3));
% simplan = endogenize(simplan, 'res_dl_er_tar', qq(2026,2):qq(2026,3));
% initdb.mean.l_er(qq(2026,2):qq(2026,3))) = log([51.4 51.4)*100;

%% Alternative scenario 1 (depreciation)
simplan_alt1 = simplan;
initdb_alt1 = initdb;

simplan_alt1 = exogenize(simplan_alt1, 'dl_er_tar', qq(2026,2):qq(2027,1));
simplan_alt1 = endogenize(simplan_alt1, 'res_dl_er_tar', qq(2026,2):qq(2027,1));
initdb_alt1.mean.dl_er_tar(qq(2026,2):qq(2027,1)) = [9 9 9 9];

%% Alternative scenario 2 (appreciation)
simplan_alt2 = simplan;
initdb_alt2 = initdb;

simplan_alt2 = exogenize(simplan_alt2, 'dl_er_tar', qq(2026,2):qq(2027,1));
simplan_alt2 = endogenize(simplan_alt2, 'res_dl_er_tar', qq(2026,2):qq(2027,1));
initdb_alt2.mean.dl_er_tar(qq(2026,2):qq(2027,1)) = [-1 -1 -1 -1];

%% Run baseline and alternative forecasts
j = jforecast(mfcast, initdb, rngfcast, 'plan', simplan, 'anticipate', true);
j_alt1 = jforecast(mfcast, initdb_alt1, rngfcast, 'plan', simplan_alt1, 'anticipate', true);
j_alt2 = jforecast(mfcast, initdb_alt2, rngfcast, 'plan', simplan_alt2, 'anticipate', true);

fcastdb = struct();
fcastdb.mean = dbextend(initdb.mean, j.mean);
fcastdb.std  = dbextend(initdb.std, j.std);

fcastdb_alt1 = struct();
fcastdb_alt1.mean = dbextend(initdb.mean, j_alt1.mean);
fcastdb_alt1.std  = dbextend(initdb.std, j_alt1.std);

fcastdb_alt2 = struct();
fcastdb_alt2.mean = dbextend(initdb.mean, j_alt2.mean);
fcastdb_alt2.std  = dbextend(initdb.std, j_alt2.std);

%% Save results into .mat and .csv
[~,~,~] = mkdir(['results\' mfilename]);
save(['results\' mfilename '\fcast_data.mat'], 'fcastdb');
save(['results\' mfilename '\fcast_data_alt1.mat'], 'fcastdb_alt1');
save(['results\' mfilename '\fcast_data_alt2.mat'], 'fcastdb_alt2');
databank.toCSV(fcastdb.mean,['results\' mfilename '\fcast_data.csv']); 
databank.toCSV(fcastdb_alt1.mean,['results\' mfilename '\fcast_data_alt1.csv']);
databank.toCSV(fcastdb_alt2.mean,['results\' mfilename '\fcast_data_alt2.csv']);

disp([mfilename ' Done!']);

%% Quick plot
disp('Wait for a couple of figures');
b = fcastdb.mean;
a1 = fcastdb_alt1.mean;
a2 = fcastdb_alt2.mean;

rng = qq(2022,1):qq(2028,4);

% Main variables
fig = figure ('name', 'Main variables','Position', [100, 100, 1200, 600]);
subplot(2,2,1)
plot(rng, [b.d4l_cpi a1.d4l_cpi a2.d4l_cpi b.dl_cpi_tar], 'linewidth', 2); grid on; title('CPI, % y-o-y', 'fontsize', 16); highlight(dq.rngfcast); legend({'Baseline', 'Alternative1', 'Alternative2', '5% target'}, 'Location', 'best', 'fontsize', 13);
subplot(2,2,2)
plot(rng, [exp(b.l_er/100) exp(a1.l_er/100) exp(a2.l_er/100)], 'linewidth', 2); grid on; title ('UAH/EUR', 'fontsize', 16); highlight(dq.rngfcast); legend({'Baseline', 'Alternative1', 'Alternative2'}, 'Location', 'best', 'fontsize', 13);
subplot(2,2,3)
plot(rng, [b.l_gdp_gap a1.l_gdp_gap a2.l_gdp_gap], 'linewidth', 2); grid on; title ('GDP Gap, %', 'fontsize', 16); highlight(dq.rngfcast);
% plot(rng, [(b.l_gdp - b.l_gdp{-4}) (a1.l_gdp - a1.l_gdp{-4}) (a2.l_gdp - a2.l_gdp{-4})], 'linewidth', 2); grid on; title ('GDP, % y-o-y', 'fontsize', 16); highlight(dq.rngfcast); legend({'Baseline', 'Alternative1', 'Alternative2'}, 'Location', 'best', 'fontsize', 13);
% plot(rng, [b.l_gdp a1.l_gdp a2.l_gdp b.l_gdp_tnd], 'linewidth', 2); grid on; title ('GDP (ln*100)', 'fontsize', 16); highlight(dq.rngfcast); legend({'Baseline', 'Alternative1', 'Alternative2', 'Trend'}, 'Location', 'best', 'fontsize', 13);
subplot(2,2,4)
plot(rng, [b.ir_pol a1.ir_pol a2.ir_pol], 'linewidth', 2); grid on; title('Interest rate, %', 'fontsize', 16); highlight(dq.rngfcast); %legend({'Alternative1', 'Alternative2', 'Baseline'}, 'Location', 'best', 'fontsize', 13) ;
saveas(fig, ['results\' mfilename '\' 'scenarios' '.png']);

% Main variables baseline
fig = figure ('name', 'Main variables baseline','Position', [100, 100, 1200, 600]);
subplot(2,2,1)
plot(rng, [b.d4l_cpi b.dl_cpi_tar], 'linewidth', 2); grid on; title('CPI, % y-o-y', 'fontsize', 16); highlight(dq.rngfcast); legend({'Baseline', '5% target'}, 'Location', 'best', 'fontsize', 13);
subplot(2,2,2)
plot(rng, [exp(b.l_er/100)], 'linewidth', 2); grid on; title ('UAH/EUR', 'fontsize', 16); highlight(dq.rngfcast); 
subplot(2,2,3)
plot(rng, b.l_gdp_gap, 'linewidth', 2); grid on; title ('GDP Gap, %', 'fontsize', 16); highlight(dq.rngfcast);
subplot(2,2,4)
plot(rng, [b.ir_pol], 'linewidth', 2); grid on; title('Interest rate, %', 'fontsize', 16); highlight(dq.rngfcast);
saveas(fig, ['results\' mfilename '\' 'fcast' '.png']);

% Main trends
figure ('name', 'Main trends','Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(rng, [b.l_gdp b.l_gdp_tnd], 'linewidth', 2); grid on; title('GDP (ln*100)', 'fontsize', 20); highlight(dq.rngfcast);
subplot(1,2,2)
plot(rng, [b.l_rer b.l_rer_tnd], 'linewidth', 2); grid on; title('RER (ln*100)', 'fontsize', 20); highlight(dq.rngfcast);

% MCI and RMCI
fig = figure ('name', 'Real IR and ER','Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(rng, [b.rir_pol b.rir_tnd], 'linewidth', 2); grid on; title('RIR, %', 'fontsize', 20); highlight(dq.rngfcast);
subplot(1,2,2)
plot(rng, [b.l_rer b.l_rer_tnd], 'linewidth', 2); grid on; title('RER (ln*100)', 'fontsize', 20); highlight(dq.rngfcast);
saveas(fig, ['results\' mfilename '\' 'trends_fcast' '.png']);

% External variables
fig = figure ('name', 'External variables','Position', [100, 100, 1200, 600]);
subplot(1,2,1)
plot(rng, [(b.l_cpi_ea - b.l_cpi_ea{-4}) 2], 'linewidth', 2); grid on; title('CPI EA, % y-o-y', 'fontsize', 16); highlight(dq.rngfcast); legend({'Assumption', '2% target'}, 'Location', 'best', 'fontsize', 13);
% subplot(2,2,2)
% plot(rng, [b.prem_uip b.prem_uip_tnd], 'linewidth', 2); grid on; title ('Sov. risk premium, pp.', 'fontsize', 16); highlight(dq.rngfcast); legend({'Filtration', 'Trend'}, 'Location', 'best', 'fontsize', 13);

% ext = load('results\a01_read_data\read_data.mat'); ext = ext.dq;
% 
% subplot(2,2,3)
% plot(rng, [ext.L_GDP_EA (ext.L_GDP_EA - ext.L_GDP_EA_GAP)], 'linewidth', 2); grid on; title ('GDP EA (ln*100)', 'fontsize', 16); highlight(dq.rngfcast); legend({'Assumption', 'Trend'}, 'Location', 'best', 'fontsize', 13);
subplot(1,2,2)
plot(rng, [b.ir_ea], 'linewidth', 2); grid on; title('Interest rate EA, %', 'fontsize', 16); highlight(dq.rngfcast);
saveas(fig, ['results\' mfilename '\' 'ext' '.png']);

disp('The figures are ready');




