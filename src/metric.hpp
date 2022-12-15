#include "email.hpp"

namespace bdap {

    struct Accuracy {
        int n = 0;
        int correct = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
            ++n;
            correct += static_cast<int>(lab == pred);
        }

        double get_accuracy() const { return static_cast<double>(correct) / n; }
        double get_error() const { return 1.0 - get_accuracy(); }

        double get_score() const { return get_accuracy(); }
    };


    struct ConfusionMatrix {

        int t_p;
        int t_n;
        int f_p;
        int f_n;


        // Constructor
        ConfusionMatrix() :
        t_p(0),
        t_n(0),
        f_p(0),
        f_n(0){

        }

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);

            if(lab){
                (pred) ? (t_p++) : (f_n++);
            }else{
                (pred) ? (f_p++) : (t_n++);
            }
        }

        double getPrecision() const {
            return static_cast<double> (t_p) / (t_p + f_p);
        }

        double getRecall() const {
            return static_cast<double> (t_p) / (t_p + f_n);
        }

        double getF1Score() const {
            double precision = getPrecision();
            double recall = getRecall();
            return static_cast<double> (2 * precision * recall) / (precision + recall);
        }

    };





    // Precision Evaluation Metric

    /*
     * This metric measures the fraction of predicted spam emails that are actually spam.
     * It is calculated as the number of true positive predictions (emails that are correctly identified as spam)
     * divided by the total number of predicted spam emails.
     */

    struct Precision {

        ConfusionMatrix confusionMatrix;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails){
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email){
            confusionMatrix.evaluate(clf,email);
        }

        double get_score() const {
            return confusionMatrix.getPrecision();
        }

    };


    // Recall Evaluation Metric

    /*
     * This metric measures the fraction of actual spam emails that are correctly identified by the spam filter.
     * It is calculated as the number of true positive predictions divided by the total number of actual spam emails.
     */

    struct Recall {

        ConfusionMatrix confusionMatrix;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails){
            for (const Email& email : emails)
                evaluate(clf, email);
        };

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email){
            confusionMatrix.evaluate(clf,email);
        };

        double get_score() const {
            return confusionMatrix.getRecall();
        };

    };

    // F1 Score Evaluation Metric

    /*
     * This metric is the harmonic mean of precision and recall,
     * and is calculated as the product of precision and recall divided by the sum of precision and recall.
     * It provides a single score that balances precision and recall,
     * and is often used as a general-purpose metric for evaluating spam filters.
     */

    struct F1Score {
        // Parameters

        ConfusionMatrix confusionMatrix = {};

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails){
            for (const Email& email : emails)
                evaluate(clf, email);
        };

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email){
            confusionMatrix.evaluate(clf, email);
        };

        double get_score() const {
            return confusionMatrix.getF1Score();
        };

    };





} // namespace bdap
