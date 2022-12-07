#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {

private:

    int num_hashes_;
    std::vector<int> hashSeeds_;
    int amt_buckets_;
    std::map<int,std::vector<int>> bucketMatrixSpam_;
    std::map<int,std::vector<int>> bucketMatrixHam_;


    void initHashSeeds(){
        // Generate a random 16-bit number
        for(int i ; i < num_hashes_ ; i++){
            int rand16 = rand() % (65536 + 1);
            hashSeeds_.push_back(rand16);
        }
    }


    void initBucketMatrix(std::map<int,std::vector<int>> bucketMatrix){

    }


public:
    // TODO initialize your fields in the constructor
    NaiveBayesCountMin(int num_hashes, int log_num_buckets, double threshold) {
        // Set basic arguments
        this.

        // Generate num_hashes amount of different hash seeds

        // Initialize bucket Matrices

    }

    void update_(const Email &email)
    {
        // TODO implement this
    }

    double predict_(const Email& email) const
    {
        // TODO implement this
        return 0.0;
    }
};

} // namespace bdap
