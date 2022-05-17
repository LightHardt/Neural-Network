#include "neuralNet.hpp"

int main()
{
    std::vector<unsigned int> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);

    NeuralNet net(topology);

    // Training Data
    std::vector<double> inputVals, targetVals, resultVals;
    inputVals.push_back(0);
    inputVals.push_back(0);
    targetVals.push_back(0);

    int turn = 1;

    // Train 0,0 is 0 
    // 1,1 is 1
    for(int training = 0; training < 1000; training++)
    {

        switch (turn)
        {
        case 1:
            inputVals[0] = 0.0;
            inputVals[1] = 0.0;
            targetVals[0] = 0.0;
            net.feedForward(inputVals);
            net.getResults(resultVals);
            cout << "Pass: " << training << endl;
            cout << "Input Vals: " << inputVals[0] <<","<< inputVals[1] <<endl;
            cout << "Target: " << targetVals[0] << endl;
            cout << "Result: " << resultVals[0] << endl;
                
            net.backProp(targetVals);
            // Report how well the training is working, average over recent samples:
            cout << "Net recent average error: "
                << net.getRecentAverageError() << endl;
            cout<<endl;
            break;
        case 2:
            inputVals[0] = 1.0;
            inputVals[1] = 1.0;
            targetVals[0] = 1.0;
            net.feedForward(inputVals);
            net.getResults(resultVals);
            cout << "Pass: " << training << endl;
            cout << "Input Vals: " << inputVals[0] <<","<< inputVals[1] <<endl;
            cout << "Target: " << targetVals[0] << endl;
            cout << "Result: " << resultVals[0] << endl;
                
            net.backProp(targetVals);
            // Report how well the training is working, average over recent samples:
            cout << "Net recent average error: "
                << net.getRecentAverageError() << endl;
            cout<<endl;
            break;
            
        default:
            break;
         }
         turn++;
         if(turn > 2)
            turn = 1;
    }

    return 0;
}