#ifndef _NLM_H_
#define _NLM_H_
#include <opencv2/opencv.hpp>
#include <iostream>
namespace nlm{



class NLMean{
public:
    explicit NLMean(int halfSearchSize, int halfTemplateSize, std::string imgPath, float h):
        h2_(h * h), halfSearchKernelSize_(halfSearchSize), halfTemplateKernelSize_(halfTemplateSize), borderSize_(halfSearchKernelSize_ + halfTemplateKernelSize_ + 1){
            srcImg_ = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
            for(int i = 0; i < 255 * 255; i++) {
                exps.push_back(exp(-i));
            }
            if (srcImg_.empty()) {
                std::cout << "error type in image or path equal empty" << std::endl;
                return;
            }
            cv::copyMakeBorder(srcImg_, padSrcImg_, borderSize_, borderSize_, borderSize_, borderSize_, cv::BORDER_REPLICATE);
    }
    
    NLMean() = delete;
    NLMean(const NLMean &&) = delete;
    NLMean(const NLMean &) = delete;
    NLMean& operator=(const NLMean&) = delete;
    
    // naive nlm
    void CalculateNLMNaive(cv::Mat &restoredImg);
    
    // fast nlm
    void CalcuateNLMFast(cv::Mat &restoreImg);
    
    //fast nlm
    void CalcuateNLMFastNeon(cv::Mat &restoreImg);
    
    
private:

    std::vector <float> exps;
    float h2_;
    cv::Mat srcImg_;
    cv::Mat padSrcImg_;
    int halfSearchKernelSize_;
    int halfTemplateKernelSize_;
    int borderSize_;
    
};
}

#endif
