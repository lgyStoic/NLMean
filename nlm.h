#ifndef _NLM_H_
#define _NLM_H_
#include <opencv2/opencv.hpp>
#include <iostream>
namespace nlm{
class NLMean{
public:
    explicit NLMean(int halfSearchSize, int halfTemplateSize, std::string imgPath, float h):
        h2_(h * h), halfSearchKernelSize_(halfSearchSize), halfTemplateKernelSize_(halfTemplateSize){
        srcImg_ = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        if (srcImg_.empty()) {
            std::cout << "error type in image or path equal empty" << std::endl;
            return;
        }
    }
    
    NLMean() = delete;
    NLMean(const NLMean &&) = delete;
    NLMean(const NLMean &) = delete;
    NLMean& operator=(const NLMean&) = delete;
    
    void CalculateNLMNaive(cv::Mat &restoredImg);
    
    
private:
    float h2_;
    cv::Mat srcImg_;
    cv::Mat padSrcImg_;
    int halfSearchKernelSize_;
    int halfTemplateKernelSize_;
    
};
}

#endif
