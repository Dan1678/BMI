function [x, y] = positionEstimator(test_data, modelParameters)
% positionEstimator.m
%
% Implements a "switching" approach with 8 separate KFs:
%   1) On the first call for a new trial, classify the angle using the
%      first 300 ms of spiking, then initialize that angle's KF.
%   2) On subsequent calls, do standard KF updates (in 20 ms bins)
%      using only the chosen angle's parameters.

  persistent trialState
  if isempty(trialState)
      trialState = struct();
  end

  tID = test_data.trialId;
  fieldID = ['tid_' num2str(tID)];

  binSize   = modelParameters.binSize;      % 20 ms
  numAngles = modelParameters.numAngles;
  % classification info
  avgFR     = modelParameters.avgFRperAngle;
  first300  = modelParameters.first300ms;

  if ~isfield(trialState, fieldID)
      % ============= CLASSIFY THE ANGLE =============
      direction_id = classifyDirection(test_data, avgFR, first300);

      % ============= Initialize the chosen KF =============
      KF = modelParameters.KF(direction_id);

      % We override the position in initMean with the actual startHandPos
      x_est = KF.initMean;
      x_est(1:2) = test_data.startHandPos;   % set x,y to real start
      % Optionally set velocity to 0 if you expect rest start:
      % x_est(3:4) = [0;0];

      P_est = KF.initCov;

      % Create a struct to store the trial's state
      s.direction_id = direction_id;
      s.x_est = x_est;
      s.P_est = P_est;
      s.lastBin = 0;  % we haven't processed any bins yet

      trialState.(fieldID) = s;
  end

  % Retrieve the trial state
  s = trialState.(fieldID);
  direction_id = s.direction_id;
  KF = modelParameters.KF(direction_id);  % the KF for this angle

  % Now we do the standard KF updates from lastBin+1 up to current bin
  T_current = size(test_data.spikes,2);
  currentBin = floor(T_current / binSize);

  % If there's new bins to process:
  if currentBin > s.lastBin
      A = KF.A;
      W = KF.W;
      H = KF.H;
      Q = KF.Q;
      stateDim = modelParameters.stateDim;
      obsDim   = modelParameters.numNeurons;
      epsilon  = 1e-6;
      I_obs    = eye(obsDim);

      x_est = s.x_est;
      P_est = s.P_est;

      for b = (s.lastBin+1) : currentBin
          % 1) Predict
          x_pred = A * x_est;
          P_pred = A * P_est * A' + W;

          % 2) Observation: sum of spikes in [tStart, tEnd] for this bin
          tStart = (b-1)*binSize + 1;
          tEnd   = b*binSize;
          z = sum(test_data.spikes(:, tStart:tEnd), 2);  % 98 x 1

          % 3) Update
          S_mat = H * P_pred * H' + Q + epsilon * I_obs;
          K = P_pred * H' / S_mat;
          innovation = z - H * x_pred;
          x_est = x_pred + K * innovation;
          P_est = (eye(stateDim) - K * H) * P_pred;
      end

      s.x_est = x_est;
      s.P_est = P_est;
      s.lastBin = currentBin;
      trialState.(fieldID) = s;
  end

  % Finally, output the decoded x,y
  x = s.x_est(1);
  y = s.x_est(2);
end

% ----------------------------------------------------------------------
function direction_id = classifyDirection(test_data, avgFR, first300)
% Use the first 300 ms spiking to find which of 8 angles is closest
% in Euclidean distance to the average FR profile.

  numAngles = size(avgFR,1);
  spikesSoFar = test_data.spikes;  % 98 x T
  T = size(spikesSoFar,2);
  Tuse = min(first300, T);

  cTest = sum(spikesSoFar(:,1:Tuse), 2);  % 98x1
  dists = zeros(numAngles,1);
  for k = 1:numAngles
      diffVec = avgFR(k,:)' - cTest;
      dists(k) = sum(diffVec.^2);
  end
  [~, direction_id] = min(dists);
end
