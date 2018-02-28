//neural network.cpp
#include "Net.hpp"
#include <iostream>
#include "TrainingData.hpp"


//show data in vector
void showVectorData(int i,string label, vector<double> &v){
    cout <<"No."<< i <<" "<< label << " ";
    for (unsigned int i = 0; i < v.size(); ++i){
        cout << v[i] << " ";
    }
    cout << endl;
}



//Main
int main(){
//Training Section-----------------------------------------
    TrainingData traingData("/Users/wayneyoung/Desktop/tesla.txt");
    
    //ex: {3, 2, 1}neural nodes???
    vector<unsigned int> topology{6, 6, 1};
    Net myNet(topology);
    vector<double> inputData;
    vector<double> targetData;
    vector<double> resultData;
    vector<double> finalData;
    double x_sum, x_bar, difference = 0.0, x, std, y=0.0;
    int trainingPass = 0;
    double avgerror = 0.0;
    
    cout <<"We have "<<topology[0]<<" inputs and "<<topology[2]<<" outputs"<<endl;
    cout <<endl;
    while (!traingData.iseof()){
        ++ trainingPass;
        if (traingData.getNextInputs(inputData) != topology[0]){
            //cout << inputData.size() << endl; for testing input data
            break;
        }
        showVectorData(trainingPass, "Inputs: ", inputData);
        myNet.feedForward(inputData);
        
        //Collect the actual results
        myNet.getResults(resultData);
        showVectorData(trainingPass, "Outputs: ", resultData);
        
        //Training
        traingData.getTargetOutputs(targetData);
        showVectorData(trainingPass, "Target: ",targetData);
        assert(targetData.size() == topology.back());
        myNet.backProp(targetData);
    
        //Report the training results
        cout << "Net recent avg error: "<< myNet.getRecentAvgError() << endl;
        cout << endl;
        avgerror += myNet.getRecentAvgError();
        //Collecting Result Data
        for(auto i = resultData.begin(); i != resultData.end(); ++i){
            finalData.push_back(*i);
        }
    }
    
    
    
    //Calculating Sum of Result Data
    for(auto i = finalData.begin(); i != finalData.end(); ++i){
        x_sum += *i;
    }
    //Calculating Avg of Result Data
    x_bar = x_sum/finalData.size();
    //Using this to calculate Std of Result Data
    for(auto i = finalData.begin(); i != finalData.end(); ++i){
        difference = (*i) - x_bar;
        y = difference*difference;
        x += y;
    }
    std = pow(x/finalData.size(), 0.5);
    
    
    cout <<"Standard Deviation of Result is: "<<std <<endl;
    cout <<"Average error is: "<< avgerror/(trainingPass - 1.0)<<endl;
    
//Testing section--------------------------------------
    cout <<"Do want to print out the final weights? 1 for yes, 2 for no...";
    int option;
    cin >> option;
    if (option == 1){
        double opw8, opbiasw8, ipw8, ipbiasw8, testingData, testingOutput, sum1, sum2;
        vector<double> oplayermx;
        vector<double> iplayermx;
        cout << "Please key in the output layer weights matrix: "<<endl;
        for (unsigned int i=0;i<topology[1];++i){
            cin >> opw8;
            oplayermx.push_back(opw8);
        }
        
        cout << "Please key in the output bias weights: "<<endl;
        cin >> opbiasw8;
        
        cout << "Please key in the input layer weights matrix: "<<endl;
        for (unsigned int i=0;i<topology[0];++i){
            cin >> ipw8;
            iplayermx.push_back(ipw8);
        }
        
        cout << "Please key in the output bias weights: "<<endl;
        cin >> ipbiasw8;
        
        cout << "Please key in the data you want to test: "<<endl;
        cin >> testingData;
        
        for (unsigned int i =0; i < iplayermx.size();++i){
            double a;
            a= testingData * iplayermx[i];
            sum1 += a;
        }
        
        for (unsigned int i =0; i < oplayermx.size();++i){
            double a;
            a= oplayermx[i]*tanh(ipbiasw8 + sum1 * testingData);
            sum2 += a;
        }
        
        testingOutput = opbiasw8 + sum2;
        
        cout <<"Testing output is: "<<testingOutput<<endl;
        cout << endl << "Done! BYEBYE"  << endl;
        
    }
//Testing section--------------------------------------
    if (option == 2){
        cout << endl << "Done! BYEBYE"  << endl;
    }
}




