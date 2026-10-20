function p = param()

% MP regime switches
p.alpha = 1;
p.beta = 0.7; 0.8;

% Steady states
p.dl_cpi_tar = 5;
p.dl_rer_tnd_ss = -2;
p.dl_gdp_tnd_ss = 2;
p.prem_uip_tnd_ss = 4;

p.coef = 1;

p.dl_cpi_ea_tar = 2;
p.rir_ea_tnd_ss = 1;

% STDs
p.std_res_dl_cpi_c = 1;

p.std_res_ir_tlr = 1;
p.std_res_ir_uip = 1;

p.std_res_l_gdp_gap = 0.5;
p.std_res_dl_gdp_tnd = 0.25;
p.std_res_l_gdp_tnd = 0;

p.std_res_l_er_uip = 2.5;
p.std_res_dl_er_tar = 0.25;
p.std_res_dl_rer_tnd = 0.25;
p.std_res_l_rer_tnd = 0;

p.std_res_prem_uip_gap = 1.5;

p.std_res_ir_ea = 1;
p.std_res_dl_cpi_ea = 1;
p.std_res_l_gdp_ea_gap = 1;


end