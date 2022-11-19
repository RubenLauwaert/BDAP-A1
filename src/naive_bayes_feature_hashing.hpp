#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int log_num_buckets_;
    int amt_buckets_;
    std::vector<int> buckets_;

    // TODO add fields here

    int seed_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
        : log_num_buckets_(log_num_buckets)
        , amt_buckets_(std::pow(2,log_num_buckets_))
        , buckets_(std::vector(amt_buckets_,1))
        , seed_(0x249cd)
    {
        // TODO initialize the data structures here
        this->threshold = threshold;
    }

    void update_(const Email& email)
    {
        // TODO implement this

        EmailIter iter = EmailIter(email, this->ngram_k);
        while(!iter.is_done()){
            size_t index = this->hash(iter.next(),seed_) % amt_buckets_;
            buckets_[index] = buckets_[index] + 1;
            //std::cout << hash << "\n";
            
        }
        for(int bucket : buckets_){
            std::cout << bucket << "\n";
        }
    }

    double predict_(const Email& email) const
    {
        // TODO implement this
        return 0.0;
    }

    void print_weights() const
    {
        size_t n = (1 << log_num_buckets_);
        for (size_t i = 0; i < n; ++i)
        {
            std::cout << "w" <<i << " " << buckets_[i] << ", " << buckets_[n + i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        // TODO limit the range of the hash values here
        return hash;
    }
};

} // namespace bdap
