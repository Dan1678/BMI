function modelParameters = positionEstimatorTraining(training_data)
% positionEstimatorTraining.m
%
% Trains 8 separate Kalman filters (one for each reaching angle),
% using 20 ms binning and a 5D state = [x; y; vx; vy; 1].
% Also computes an average firing-rate signature for the first 300 ms
% to classify direction at test time.

  % Binning size
  binSize = 20;
  first300ms = 300;  % for direction classification

  [numTrials, numAngles] = size(training_data);

  % -------------------------------------------------------------------
  % 1) Compute average firing rate in first 300 ms for each angle
  %    (for direction classification)
  % -------------------------------------------------------------------
  numNeurons = size(training_data(1,1).spikes, 1);  % 98
  avgFRperAngle = zeros(numAngles, numNeurons);
  for k = 1:numAngles
      spikeCountsAll = [];
      for n = 1:numTrials
          T = size(training_data(n,k).spikes, 2);
          Tuse = min(first300ms, T);
          c = sum(training_data(n,k).spikes(:,1:Tuse), 2); % 98x1
          spikeCountsAll = [spikeCountsAll, c];
      end
      avgFRperAngle(k,:) = mean(spikeCountsAll,2)';
  end

  % Store this for classification
  modelParameters.avgFRperAngle = avgFRperAngle;
  modelParameters.first300ms    = first300ms;

  % -------------------------------------------------------------------
  % 2) Train a separate Kalman Filter for each angle
  % -------------------------------------------------------------------
  for k = 1:numAngles
      % Gather all trials for angle k
      X0_dyn = [];  % for dynamics regression
      X1_dyn = [];
      X_obs  = [];
      Z_obs  = [];
      initStates = [];

      for n = 1:numTrials
          trial = training_data(n,k);
          pos   = trial.handPos(1:2,:);   % (2 x T)
          spikes= trial.spikes;          % (98 x T)
          Ttrial= size(pos,2);

          nBins = floor(Ttrial / binSize);
          if nBins < 1, continue; end

          X_trial = zeros(5, nBins);   % [x; y; vx; vy; 1]
          Z_trial = zeros(numNeurons, nBins);

          for b = 1:nBins
              tStart = (b-1)*binSize + 1;
              tEnd   = b*binSize;

              posEnd   = pos(:, tEnd);
              if b == 1
                  posStart = pos(:,1);
              else
                  posStart = pos(:, (b-1)*binSize);
              end
              vel = (posEnd - posStart) / binSize;  % mm per 20 ms

              X_trial(:,b) = [posEnd; vel; 1];
              Z_trial(:,b) = sum(spikes(:, tStart:tEnd), 2);
          end

          if nBins >= 2
              X0_dyn = [X0_dyn, X_trial(:,1:end-1)];
              X1_dyn = [X1_dyn, X_trial(:,2:end)];
          end
          X_obs = [X_obs, X_trial];
          Z_obs = [Z_obs, Z_trial];

          initStates = [initStates, X_trial(:,1)];
      end

      % --- Fit the dynamics matrix A, forcing a block structure ---
      A = eye(5);
      % We assume a constant velocity model: x(t+1) = x(t) + vx(t)
      A(1,3) = 1;  % x += vx
      A(2,4) = 1;  % y += vy

      % Predicted next state for the dynamic regression
      X1_pred = A * X0_dyn;
      Resid_dyn = X1_dyn - X1_pred;
      N_dyn = size(Resid_dyn,2);
      W = (Resid_dyn * Resid_dyn') / max(N_dyn - 1, 1);

      % --- Fit observation model: Z = H * X + noise
      H_est = Z_obs * pinv(X_obs);
      Z_pred = H_est * X_obs;
      Resid_obs = Z_obs - Z_pred;
      N_obs = size(Resid_obs,2);
      Q = (Resid_obs * Resid_obs') / max(N_obs - 1, 1);

      initMean = mean(initStates,2);
      initCov  = cov(initStates');

      % Store the KF for this angle
      modelParameters.KF(k).A = A;
      modelParameters.KF(k).W = W;
      modelParameters.KF(k).H = H_est;
      modelParameters.KF(k).Q = Q;
      modelParameters.KF(k).initMean = initMean;
      modelParameters.KF(k).initCov  = initCov;
  end

  modelParameters.numAngles = numAngles;
  modelParameters.numNeurons= numNeurons;
  modelParameters.binSize   = binSize;
  modelParameters.stateDim  = 5;  % [x, y, vx, vy, 1]
end
