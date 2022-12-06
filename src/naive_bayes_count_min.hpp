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
    std::vector<int> hashSeeds_;
    int amt_buckets_;
    std::map<int,std::vector<int>> bucketMatrixSpam_;
    std::map<int,std::vector<int>> bucketMatrixHam_;


public:
    // TODO initialize your fields in the constructor
    NaiveBayesCountMin(int num_hashes, int log_num_buckets, double threshold) {}

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
