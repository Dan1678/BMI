function modelParameters = positionEstimatorTraining(training_data)

    numAngles  = size(training_data, 2); 
    numTrials  = size(training_data, 1);
    numNeurons = size(training_data(1,1).spikes, 1);

    %% Compute average trajectory for each angle
    maxTimePerAngle = zeros(numAngles,1);
    for k=1:numAngles
        allLengths = zeros(numTrials,1);
        for n=1:numTrials
            allLengths(n) = size(training_data(n,k).spikes,2);
        end
        maxTimePerAngle(k) = min(allLengths);
    end

    avgTrajectory = cell(numAngles,1);
    for k=1:numAngles
        T = maxTimePerAngle(k);
        sumPos = zeros(2, T);
        for n=1:numTrials
            sumPos = sumPos + training_data(n,k).handPos(1:2, 1:T);
        end
        avgTrajectory{k} = sumPos / numTrials; % 2 x T
    end

    %% Naive direction classifier: average firing in first 300 ms
    first300ms = 300;
    avgFRperAngle = zeros(numAngles, numNeurons);

    for k=1:numAngles
        spikeCountsAll = zeros(numTrials, numNeurons);
        for n=1:numTrials
            T = min(size(training_data(n,k).spikes,2), first300ms);
            spikeCountsAll(n,:) = sum(training_data(n,k).spikes(:,1:T), 2);
        end
        avgFRperAngle(k,:) = mean(spikeCountsAll,1);
    end

    %% Store in modelParameters
    modelParameters.avgTrajectory   = avgTrajectory;
    modelParameters.avgFRperAngle  = avgFRperAngle;
    modelParameters.numAngles      = numAngles;
    modelParameters.numNeurons     = numNeurons;
    modelParameters.first300ms     = first300ms;
    modelParameters.maxTimePerAngle= maxTimePerAngle;
end
