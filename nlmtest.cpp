#include "nlm.h"
#include <chrono>

int main(int argc,char **argv) {

    std::string keys = 
    "{help h usage ? |      | input src image ,and result folder like 'src=aaa dst=bbbb' }"
    "{src | /home/lgy/NLMean/lena.jpg | image for denoise}"
    "{raw | /home/lgy/NLMean/lena_raw.jpg | image for denoise}"
    "{dst | /home/lgy/NLMean/ | dst folder for res}";

    cv::CommandLineParser parser(argc, argv, keys);
    
    std::string imgPath = parser.get<std::string>("src", "/home/lgy/NLMean/lena.jpg");

    std::string dstFolder = parser.get<std::string>("dst", "/home/lgy/NLMean/");
    nlm::NLMean mean(15, 3, imgPath, 20);
    cv::Mat res;
   cv::setNumThreads(0);
    auto start = std::chrono::steady_clock::now();
    mean.CalculateNLMNaive(res);
    auto end = std::chrono::steady_clock::now();
    auto passed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "nlm without optimize cost :" << passed << "ms" << std::endl;
    cv::imwrite(dstFolder + "nlmnaive.jpg", res);
#if defined(__x86_64__) 
    cv::imshow("naive", res);
    cv::waitKey();
#endif

    start = std::chrono::steady_clock::now();
    mean.CalcuateNLMFast(res);
    end = std::chrono::steady_clock::now();
    passed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "nlm using integral cost :" << passed << "ms" << std::endl;
    cv::imwrite(dstFolder + "nlmalgoptimize.jpg", res);
#if defined(__x86_64__) 
    cv::imshow("algfast", res);
    cv::waitKey();
#endif

    start = std::chrono::steady_clock::now();
    mean.CalcuateNLMFastNeon(res);
    end = std::chrono::steady_clock::now();
    passed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "nlm using neon and integral cost :" << passed << "ms" << std::endl;
    cv::imwrite(dstFolder + "nlmneonoptimize.jpg", res);
#if defined(__x86_64__) 
    cv::imshow("neon", res);
    cv::waitKey();
#endif

    cv::Mat src = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    start = std::chrono::steady_clock::now();
    cv::fastNlMeansDenoising(src, res, 20, 6, 30);
    end = std::chrono::steady_clock::now();
    passed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "opencv cost :" << passed << "ms" << std::endl;
    cv::imwrite(dstFolder + "nlmopencv.jpg", res);
#if defined(__x86_64__) 
    cv::imshow("opencv", res);
    cv::waitKey();
#endif

    std::string rawPath = parser.get<std::string>("raw");
    cv::Mat raw = cv::imread(rawPath, cv::IMREAD_GRAYSCALE);
    std::cout << "restore image psnr :"<< cv::PSNR(res, raw) << " , origin psnr :" << cv::PSNR(src, raw) << std::endl;
    return 0;
}
