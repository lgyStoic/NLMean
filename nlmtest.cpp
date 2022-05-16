#include "nlm.h"

int main() {
    std::string imgPath = "/Users/lgy/nlm_filter/lena.jpg";
    nlm::NLMean mean(15, 3, imgPath, 20);
    cv::Mat res;
    mean.CalculateNLMNaive(res);
    cv::imshow("res", res);
    cv::waitKey();
    return 0;
}
