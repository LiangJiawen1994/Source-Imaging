function [S,par] = SISST(B,L,vertconn,cluster,varargin)
%% Source Imaging via Bayesian Modeling with Smoothness in Spatial and Temporal Domains
% Input: 
%      B(P x T):             M/EEG Measurement
%      L(P x D):             Leadfiled Matrix
%      vertconn(D x D):      Vertices Connection over the cortical mesh
%      cluster:              Overlapped indexes collections of vertices
% Output:
%      S:                    Estimated Sources

% Author : Liang Jiawen
% Date: 2022/6/15 

% Reference: [1] MEG source localization of spatially extended generators of epileptic activity: 
% comparing entropic and hierarchical bayesian approaches;
%  [2] Probabilistic algorithms for MEG/EEG source reconstruction using
%  temporal basis functions learned from data.
% [3] Bayesian Electromagnetic Spatio-Temporal Imaging of Extended Sources Based on Matrix Factorization

tic
%% Initial of Algorithm
[nSensor,nSource] = size(L);
nSnap = size(B,2);
if all(diag(vertconn)==1)
    vertconn = vertconn - speye(nSource);
end

C_noise = eye(nSensor);
MAX_ITER = 1e3;
epsilon = 1e-8;
prune = [1e-1,1e-6];
update = 'MacKay';

% Initialization of TBFs
K = 7;               
[~,~,D] = svd(B);
Phi = D(:,1:K)';


% get input argument values
if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'epsilon'
                epsilon = varargin{i+1};
            case 'max_iter'
                MAX_ITER = varargin{i+1};
            case 'update'
                update = varargin{i+1};
            case 'tbfs'
                Phi = varargin{i+1};
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
    end
end

% spatial smoothness
[~,W] = spatial_priorw(vertconn);
W = W./sum(W,2);

% temporal smoothness
M = smooth_constraint(nSnap,3);
MTM = M'*M;
xi = (M\Phi')';

% initialization of hyperparameters
K = size(Phi,1);
alpha = ones(K,1);
gamma = ones(numel(cluster),1);

% initialization of encoding coefficients
w = zeros(nSource,K);
w_re = w;

% initialization of auxiliary variables
mu = zeros(K,numel(cluster));
Diag_C = ones(K,1);
Diag_W = ones(K,1);

F = L*W;
FF = F;

evidence_array = zeros(1,MAX_ITER);

%% Iteration
for iter = 1:MAX_ITER
    %% check TBFs
    keep = alpha<min(alpha)/prune(1);
    alpha = alpha(keep);w_re = w_re(:,keep);mu = mu(keep,:);K = sum(keep);
    Diag_C = Diag_C(keep);Diag_W = Diag_W(keep);Phi = Phi(keep,:);xi = xi(keep,:);
    
    %% check clusters
    keep = gamma<min(gamma)/prune(2);
    gamma = gamma(keep);cluster = cluster(keep);mu = mu(:,keep);
    
    %% Construct gamma
    Gamma = zeros(1,nSource);
    for i = 1:numel(cluster)
        Gamma(cluster{i}) = Gamma(cluster{i}) + 1/gamma(i);
    end
    VertKeepList = ~(Gamma==0);
    F = FF(:,VertKeepList);w = w_re(VertKeepList,:);
    Gamma = 1./Gamma(VertKeepList);
    
    %% Source update
    FGFT = F./Gamma*F';
    temp = Phi*Phi'+diag(Diag_W);
    w_re = zeros(nSource,K);
    for k = 1:K
        index = setdiff(1:K,k);
        scale = temp(k,k);
        residual = B*M*xi(k,:)'- F*w(:,index)*temp(k,index)';
        Sigma_w = FGFT+C_noise/scale;
        w(:,k) = (F./Gamma)'/scale/Sigma_w*residual;
        for i = 1:numel(cluster)
            index = cluster{i};
            mu(k,i) = trace(FF(:,index)'/Sigma_w*FF(:,index));
        end
        Diag_C(k) = trace(FGFT/Sigma_w/scale);
        % FreeEnergy Compute
        evidence_array(iter) = evidence_array(iter) - 0.5*sum(log(eig(Sigma_w*scale))) - 0.5*w(:,k)'.*Gamma*w(:,k);
    end
    w_re(VertKeepList,:) = w;
    
    %% TBFs Update
    temp = w'*F'/C_noise*F*w+diag(Diag_C) ;
    for k=1:K
        index = setdiff(1:K,k);
        scale = temp(k,k);
        residual = B'*F*w(:,k)- M*xi(index,:)'*temp(index,k);
        Sigma_xi = inv(MTM*scale+alpha(k)*eye(nSnap));
        xi(k,:) = Sigma_xi*M'*residual;
        alpha(k) = nSnap/(xi(k,:)*xi(k,:)'+trace(Sigma_xi));
        Diag_W(k) = trace(MTM*Sigma_xi);
        % FreeEnergy Compute
        evidence_array(iter) = evidence_array(iter) + 0.5*nSnap*log(alpha(k))+ 0.5*sum(log(eig(Sigma_xi)))-0.5*(scale-Diag_C(k))*Diag_W(k);
    end
    Phi = xi*M';
    
    %% gamma update
    if strcmpi('MacKay', update)
        Ew2 = sum(w_re.^2,2);
        Gamma_recover = inf(1,nSource);
        Gamma_recover(VertKeepList) = Gamma;
        for i = 1:numel(cluster)
            index = cluster{i};
            gamma(i) = gamma(i)*1/(Gamma_recover(index).^2*Ew2(index)/sum(mu(:,i)));
        end
    elseif strcmpi('Convex', update)
        Ew2 = sum(w_re.^2,2);
        Gamma_recover = inf(1,nSource);
        Gamma_recover(VertKeepList) = Gamma;
        for i = 1:numel(cluster)
            index = cluster{i};
            gamma(i) = gamma(i)*1/sqrt(Gamma_recover(index).^2*Ew2(index)/sum(mu(:,i)));
        end
    else
        error('Not validate update method')
    end
    
    %% FreeEnergy Compute
    evidence_array(iter) = evidence_array(iter) -0.5*(trace((B-F*w*Phi)'/C_noise*(B-F*w*Phi)) + nSensor*nSnap*log(2*pi));
    if iter>1
        MSE = (evidence_array(iter)-evidence_array(iter-1))/abs(evidence_array(iter-1));
        fprintf('iter = %g, MSE = %g, remaindTBFs = %g, remaindcluster = %g, remaindvert = %g \n', iter,MSE,numel(alpha),numel(cluster),sum(VertKeepList));
        if abs(MSE)<epsilon
            evidence_array = evidence_array(1:iter);
            break
        end
    end
end

%% Recording
S = W*w_re*Phi;
par.W = W*w_re;
par.evidence_array = evidence_array;
par.cluster = cluster;
par.gamma = gamma;
par.Phi = Phi;
par.time = toc;
end


function [W,W1] = spatial_priorw(neighbors)
%   This function returns the W Green matrix from which the local 
%   covariance matrices will be obtained. 
%
%   INUPTS:
% 		- OPTIONS    	: structure of parameters
%       - neighbors		: neighbor matrix (with 0s on the diag)
%
%   OUTPUTS:
%		- OPTIONS		: Keep track of parameters
%		- W				: Green Matrix
%   
% 	Reference
%       Harrison et al., NIMG 38(4), 677-695 (2007
%       Friston et al., NIMG 39(3), 1104-1120 (2008)
%
%   Authors: LATIS team, 2011.
%% ==============================================
% License 
%
% BEst is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BEst is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BEst. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------
                                              
                                              
% % Default options settings
% Def_OPTIONS.solver = struct('spatial_smoothing', 0.6);
% 
% % Return empty OPTIONS structure
% if (nargin == 0)
%     OPTIONS = Def_OPTIONS;
%     return
% end
% % Check field names of passed OPTIONS and fill missing ones with default values
% OPTIONS = be_struct_copy_fields(OPTIONS, Def_OPTIONS, {'solver'}, 0);
% clear Def_OPTIONS
%%
% Parameters:
nb_vertices = size(neighbors,2);
rho = 0.6;%OPTIONS.solver.spatial_smoothing; % scalar that weight the adjacency matrix 
W   = speye(nb_vertices);


% % Add comment to result
% if rho && ~isempty(neighbors)
%     OPTIONS.automatic.Comment = [OPTIONS.automatic.Comment ' | smooth=' num2str(rho)];
% else
%     return
% end

% Preallocation of the sparse matrix
A  = neighbors - spdiags(sum(neighbors,2),0,nb_vertices,nb_vertices);
% A  = neighbors + speye(nb_vertices);
A0 = rho*A/2;
for i = 1:7
    W = W + A0;
    A0 = rho/2*A0*A / (i+1);
end
W1 = W.*(W > exp(-8));
W = W1'*W1; 
W1 = W1 - spdiags(diag(W1),0,nb_vertices,nb_vertices);
% W1 = W1';
return

end

function M = smooth_constraint(n,span)
N = zeros(n);
for i = span:n
    N(i,i-span+1:i) = 1/span;
end
M = inv(eye(n)-N);
end
