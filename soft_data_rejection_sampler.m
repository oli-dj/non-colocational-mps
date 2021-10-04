%% Soft data conditioning using extended rejection sampler
% Requires mGstat in path, specifically 'mps_template.m' and 'channels.m' 
% https://github.com/cultpenguin/mGstat
%
% The prior is sampled from a MPS model conditioned to some ``hard'' data,
% while the likelihood is evaluated from the probability of those
% realizations being compatible with the ``soft'' data.
%
% Then compared to an MPS Estimation that takes non-co-locational soft data
% into account.
%
rng(1); % Set fixed random seed;

%% LOAD DATA
soft_data = 0;      %use soft data (here set to zero, since we will use the
                    %soft data for the rejection sampler, and not the prior

SDG = NaN(20,20,2); %Soft Data Grid

% 1st soft data point "pseudo hard"
SDG(17,5,1) = 0.999;
SDG(17,5,2) = 1-SDG(17,5,1);

% 2nd soft data point "mildly informative"
SDG(5,10,1) = 0.3;
SDG(5,10,2) = 1-SDG(5,10,1);

%3rd data point "informationless"
SDG(16,12,1) = 0.5;
SDG(16,12,2) = 0.5;


max_prob = max(SDG(17,5,:)).*max(SDG(5,10,:)).*max(SDG(16,12,:));
%max_prob = max(SDG(17,5,:));
% Simulation Grid Size (only set if no softdata is used).
sg_x = 0;
sg_y = 0;

%% Training Image
TI = channels;
TI = TI(1:3:end,1:3:end); %scale down strebelle

%% GENERAL OPTIONS

num_realizations = 1000;
%num_realizations = 5; %for testing
print = 1;
plots = 0;

%% Save options
make_save = 0; %save individual realizations
save_setup = 1; %save results and setup.
output_folder = 'soft_data_rejection_smpl';
filename_prefix = 'sim';

%% IMPALA OPTIONS

% PATH
% 0: raster, 1:random, 2: random preferential, 3: dist
pathtype = 1; 
I_fac = 4; % Only applies to preferential path

% Data template 
template_length = 1500;
template_shape = 1;
options.print = print;

% Min counts to make a cdpf, else use marginal distribution
options.threshold = 5;

% Use GPU instead of CPU for searching the list and calculating cdpf
options.GPU = 1; %Requires CUDA capable GPU

% Max number of non-colocated softdata to use (0, 1, 2, 3 ...)
% Note: Currently only works in GPU implementation
options.num_soft_nc = 0;
options.prob_factor_limit = 0;

% Capping (Max number of conditional data to search for in the SG)
% Set to template_length or more to disable.
options.cap = 15;

% Trimming
options.trimming = 0;
options.trim_size = 5;
options.trim_trigger = 10;
options.min_size = 10;

%% Training Image
dim = length(size(TI));
cat = unique(TI(:))';
num_cat = length(cat);
fprintf("There are %i different facies. \n",num_cat);

%Template
tau = mps_template(template_length,dim,template_shape);

tic
%Populate pattern library (list)
list = populate_impala_list(TI, tau );
time_elapsed = toc;
if print
    fprintf('Time to populate list: %8.3f seconds.\n', time_elapsed);
    fprintf("List length %i \n",size(list,1));
end

%Display list if small enough
if size(list,1) < 40
    print_impala_list( list );
end

%Simulation grid size
if (sg_x == 0) || (sg_y == 0)
    sg_x = size(SDG,1);
    sg_y = size(SDG,2);
end

%Soft data
% if (soft_data > 0)   
%     sg_x = size(SDG,1);
%     sg_y = size(SDG,2);
%     sg_z = size(SDG,3);
% else
%     switch dim
%         case 2
%             SDG = NaN(sg_x,sg_y,num_cat);
%         case 3
%             SDG = NaN(sg_x,sg_y,sg_z,num_cat);
%     end
% end

switch dim
    case 2
        SG = NaN(sg_x,sg_y);
    case 3
        SG = NaN(sg_x,sg_y,sg_z);
end

%% Init rejection sampler
num_accept = 0;
SG_rejection_sampler = zeros(size(SG));

% Flatten soft-data grid (for calculating likelihood)
SDG0 = SDG(:,:,1);
SDG1 = SDG(:,:,2);
SDG0 = SDG0(:);
SDG1 = SDG1(:);

%% HARD DATA:
SG(13,5) = 1;
SG(15,18) = 1;
SG_orig = SG; %Save original hard data.
%%

if ~exist('SDG','var')
     SDG = NaN([size(SG),length(cat)]);
end

SG_tot = zeros(size(SG));
SG_tot_rejection = zeros(size(SG));

if plots
    fig_current = figure();
    figure;imagesc(SG);
    xlabel('x')
    ylabel('y')
    axis image
    axis ij
    hold on;
    drawnow;
end

options_rejection=options;
options.num_soft_nc = sum(~isnan(SDG),[1,2,3])./num_cat;

%% Simulation
for i = 1:num_realizations
    fprintf('Now simulating...');
    %Generate new path
    tic
    switch  pathtype
        case 0
            [path, n_u] = raster_path(SG);
        case 1
            [path, n_u] = rand_path(SG);
        case 2
            [path, n_u] = pref_path(SG, SDG, I_fac);
        case 3
            [path, n_u] = dist_path(SG, tau);
    end
    time_elapsed = toc;
    if print
        fprintf('Time to generate random path: %8.3f seconds.\n',...
            time_elapsed);
    end
    
    %Pre-calculate random numbers
    rand_pre = rand(n_u,1);
    
tic;
    
    if options.GPU
        % Simulate without conditioning to soft data (for rejection
        % sampler)
        
        [SG_rejection, tauG, stats] = impala_core_gpu_soft(...
            SG_orig, NaN(size(SDG)), list, path, tau, rand_pre, cat,...
            options_rejection);
        fprintf('Time to simulate with hard data only: %8.3f seconds', toc);
        
        % Simulate *with* soft data
%         [SG, tauG, stats] = impala_core_gpu_soft(...
%             SG, SDG, list, path, tau, rand_pre, cat, options);
    else
        [SG, tauG, stats] = impala_core(...
            SG, SDG, list, path, tau, rand_pre, cat, options);
    end
    
    time_elapsed = toc;
    if print
        fprintf('Time to generate realization number %i: %8.3f seconds.\n', i, time_elapsed);
    end
    if plots
        imagesc(SG);
        drawnow;
    end
    %% Rejection sampler
    
    % flatten simulation grid
    SG_flat=SG_rejection(:);
    
    % calculate acceptance probability and normalize using max_prob
    
    prob_accept = (prod(max([(SDG0(~isnan(SDG0)).*(1-SG_flat(~isnan(SDG0)))),...
    (SDG1(~isnan(SDG0)).*(SG_flat(~isnan(SDG0))))]')))./max_prob;
    
    fprintf(['Prob_accept: ' num2str(prob_accept) '\n']);
    if  prob_accept > rand
          SG_rejection_sampler = SG_rejection_sampler + SG_rejection;
          num_accept = num_accept +1;
    
        if make_save
            saveRealization(SG,[filename_prefix num2str(i)],output_folder);
        end
    end
    % Save regardless of rejection sampler
%     SG_tot = SG_tot + SG; %with soft data
    SG_tot_rejection = SG_tot_rejection + SG_rejection; %without soft data
    last_i = i;
    %SG = NaN(size(SG)); 
    SG = SG_orig; 
    fprintf(['Number of accepted %i\n Acceptance ratio: ' num2str(num_accept/i) '\n'],num_accept);
end

if save_setup
    close all;
    save([output_folder '//' 'setup.mat']);
end

SG_tot = SG_tot./last_i;
SG_tot_rejection = SG_tot_rejection./last_i;
SG_rejection_sampler_tot = SG_rejection_sampler./num_accept;


%% Estimation
template_length = 1500;
template_shape = 1;

%Template
tau = mps_template(template_length,dim,template_shape);

tic
%Populate pattern library (list)
list = populate_impala_list(TI, tau );
time_elapsed = toc;
if print
    fprintf('Time to populate list: %8.3f seconds.\n', time_elapsed);
    fprintf("List length %i \n",size(list,1));
end


options_est = options;
[path, n_u] = raster_path(SG);
rand_pre = rand(n_u,1);
options_est.num_soft_nc = 3;

[CG, tauG, stats] = estimator_core_gpu_soft(SG, SDG, list, path,...
    tau, rand_pre, cat, options_est);

%%

figure;
[cmap] = generateColormap(TI,0,1);
subplot(3,3,1)
imagesc(CG(:,:,2));
title('Estimation, hard and soft data.','interpreter','latex');
% %title(sprintf('$$n_{soft} = %i$$',ncsds(j)),'interpreter','latex');
axis image
axis ij
colorbar;
colormap(cmap);
caxis([0,1]);
xlabel('x','interpreter','latex');
ylabel('y','interpreter','latex');
 

subplot(3,3,2)
imagesc(SG_tot);
% %title(sprintf('$$n_{soft} = %i$$',ncsds(j)),'interpreter','latex');
title('Simulation, hard and soft data.','interpreter','latex');
axis image
axis ij
colorbar;
colormap(cmap);
caxis([0,1]);
xlabel('x','interpreter','latex');
ylabel('y','interpreter','latex');
% 
subplot(3,3,3)
imagesc(SG_rejection_sampler_tot);
title(sprintf('Simulation, Rejection Sampler\n Accepted realizations: %i',num_accept),'interpreter','latex');
% %title(sprintf('$$n_{soft} = %i$$',ncsds(j)),'interpreter','latex');
axis image
axis ij
colorbar;
colormap(cmap);


subplot(3,3,4)
imagesc(SG_tot_rejection);
% %title(sprintf('$$n_{soft} = %i$$',ncsds(j)),'interpreter','latex');
title('Simulation, hard data only.','interpreter','latex');
axis image
axis ij
colorbar;
colormap(cmap);
caxis([0,1]);
xlabel('x','interpreter','latex');
ylabel('y','interpreter','latex');



%ca = gca;
%colormap(ca, flipud(gray));
% caxis([0,0.2]);
% xlabel('x','latex');
% ylabel('y','latex');