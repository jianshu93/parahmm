# Parallel Hidden Markov Model

Team members:
- Yuchen Huo (<yhuo1@andrew.cmu.edu>)
- Danhao Guo (<danhaog@andrew.cmu.edu>)

# Final Report Temp
## Summary
We implemented the multi-core parallel version of hidden Markov model (HMM) algorithms:
1. Compute the probability of the observation sequence. (Forward / Backward Algorithm) 
2. Decode the observations to find hidden state sequence with the most probablility. (Viterbi Algorithm) 
3. Unsupervised training of hidden Markov mode parameters. (Baum-Welch Algorithm)

We tested our implementation on the 8 physical cores hyper-threading GHC machines. With AVX instructions, we achieved 6.62X and 5.60X speed up over baseline for single thread Forward and Viterbi algorithm. With 8 threads, we achieved 7.29X and 4.7X speed up over the AVX implementation. We achieved a total speed up of 66.48X over the baseline for Forward algorithm using 16 threads on the GHC machine.

## Technical Challenges
We optimization focus on two levels of parallelism: 1. SIMD parallelism 2. Multi-threading parallelism. Hidden Markov Model algorithms are similar to the inference procedure of the deep neural network, which is performed by a layer by layer fashion. Each layer depends on the previous layer, but in each layer, the computation is independent. Therefore, work can be easily divided for each thread and the speed up is close to linear. However, it is non-trivial to utilize SIMD parallelism in HMM algorithms. The challenge comes from several parts:
1. Since all transition and emission probability numbers are small, to avoid arithmetic instability, all the computation has to be performed in log-space. This makes it different from simple matrix multiplication which has been studied well.
2. Matrices have to be carefully chosen to avoid scatter and gather in the SIMD instructions and poor data locality.

![GitHub Logo](ViterbiSIMD.png)
*Single thread viterbi algorithm optimization*

# Middle Checkpoint
Our original project idea was to extend the Kernalized Correlation Filter (KCF) [2, 3], a fast object tracker, to the multicore platform. We downloaded the original paper’s code and built up the test framework using the [Need for Speed benchmark](http://ci2cv.net/nfs/index.html)  and [VOT2014 benchmark]( http://www.votchallenge.net/vot2014/dataset.html). We then performed some profiling on the sequential implementation. The result shows that most of the cpu time is spent in the discrete fourier transform and the HOG feature extraction which is consistent to our intuition. However, as in the object tracking tasks, the bounding box of the target object is not very large and after using HOG feature which compresses 4x4 cells into single feature descriptor, the dft computation is conducted on a quite small matrix. According to our measurement, the processing time for a single frame is about 8ms where dft time holds for 4-5ms. Since in object tracking task each frame’s computation depends on the result of the previous one, the dependency is pretty high. Therefore, we think this system is not very suitable for multithreading, since the overhead of spawning and synchronization would be larger than the actual computation. We actually try to use the FFTW library to replace the opencv implementation for the code, the performance becomes worse. Therefore, the only parallelism we could benefit from would be using SIMD instructions. We think this project actually could not get much performance improvement from the multicore platform and seems to be quite narrow in opptimization approaches. We decided to switch the project. 

# New proposal
## Summary
We are going to implement the parallel version of hidden Markov model (HMM) [1] training and classification algorithm utilizing SIMD and multithreading. HMM involves three basic problems: 
1. Compute the probability of the observation sequence. (Forward / Backward Algorithm) 
2. Decode the observations to find hidden state sequence with the most probablility. (Viterbi Algorithm) 
3. Unsupervised training of hidden Markov mode parameters. (Baum-Welch Algorithm)

## Background:
Markov model describes systems with randomly changes in which the future states of the system only depend on the current state instead of the events before the current state. HMM assumes the observations are assumed to be the result (emission) of unobserved hidden states in a Markov model. HHM are especially used in the applications of pattern recognition such as text tagging, semantics analytics and speech recognition. 

## Challenges:
1. Identify performance bottleneck and analyze the feasibility of parallelism on the part is not that easy.
2. For HMM with large state space, the space needed for parameters are too large to be loaded into L1 cache at the same time. Implementing parallel HMM with friendly data locality is also a big factor of performance improvement.

## Resources:
A sequential [implementation](https://github.com/chuan/chmm) of the HMM algorithms which also contains a cuda implementation.

## Our goals:
80% Goal:
Implementing a multi-threaded HMM with SIMD intrinsics especially for large state space which needs friendly data locality. (including baum–Welch algorithm, forward algorithm, backward algorithm and viterbi algorithm)

100% Goal:
Exploring bottleneck of HMM for relative small state space. Generalizing HMM algorithm optimization with different strategies for different state space configuration. 

## Platform choice:
We would develop and test our software on the GHC machines which have 8 cores and AVX2 support.

# Checkpoint Report:
We spent quite a lot time in setting up the experiment environment for the original project idea, profiling and a few more days to seek a new project idea, so we are kind of left behind. By now, we have selected a sequential HMM implementation as our code base and built up the test cases for benchmarking. We have identified the data dependency in all of the three algorithms and we believe that they may benefit from optimized multithread vector-matrix multiplication with the help of AVX operations.

Since it is a bit late in the whole schedule, we would try our best to catch up. We decide to implement the SIMD optimized version of all three algorithms by the end of this week. We shall spend one more week to implement multi-threaded version concerning data locality using OpenMP. Both of us have no more exams after 5/8, therefore we would work hard and spend all of our time to explore the remaining possible optimization in the project and try to make it able to accelerate problems with different working set as different model size might need different parallel degree.

## We plan to show several graphs on the competition
* The AVX instruction speedup of all three algorithms.
* The speed up of multicore HMM algorithms using different number of cores.
* The speed up of our implementation on different working set size. 

## Issues we concerned:
* It should be possible to accelerate long Markov chains with lots of hidden states and observation types though multithreading. However, we do search on the internet and find out there are not too many application that would make use of this kind of Hidden Markov model. We are not sure if this would be a big problem. This project would make use of many of the idea we learnt in the course but it might not be that useful for real workload. We do plan to measure the performance on different size of the working set, we may be able to implement a more agile algorithm suitable for different size of models.

## Modified time table for coming up tasks:
* Apr. 27th – Apr. 30th: Implementing SIMD version of forward/backward algorithm and Baum-Welch by Yuchen, SIMD version of Vertibi algorithm by Danhao.
* May. 1th – May. 3th: Taking exams. Try some matrix multiplication lib for Baum-Welch algorithm if possible.
* May. 4th – May.7th: Use OpenMP to parallel the algorithms and explore data locality in the algorithm by both Yuchen and Danhao.
* May. 9th – May. 11th: Explore other possible optimizations and try to make the implementation flexible for different size of working set. Documenting and prepare for presentation.


## Reference
[1] Rabiner, Lawrence, and B. Juang.
"An introduction to hidden Markov models." ieee assp magazine 3.1 (1986): 4-16.

# Original Proposal
## C++ Parallel KCF Tracker

## Summary
We are going to extend the Kernalized Correlation Filter (KCF) [2, 3], a fast object tracker, to the multicore platform.

## Background
Visual Object Tracking is one of the popular tasks in the computer vision area. There are a lot of different implementations focusing on how to improve the accuracy of the tracking algorithm. These implementations mostly target at the "real time" online systems, which have been aimed at approximate frame rate of 30 Frames Per Second(FPS), enough for previous devices and workloads. However, for higher frame rate cameras (240 FPS) or an offline video processing pipeline, 30 FPS still seems to be slow. We look through some of the benchmark results [4, 5] and find this Kernalized Correlation Filter which provide top speed on CPU and pretty good accuracy. We decide to further extend this algorithm to be able to benefit from multicore systems and be fast enough to be used offline.

By creating the circulant training data matrix, KCF uses Discrete Fourier Transform (DFT) to compute the close form solution of Ridge Regression and reduce both the storage and computation. It further uses HOG feature instead of raw pixels to gain better mean precision on the testing dataset. The remaining main computation lies in the dft and HOG feature extraction. There are plenty of fast parallel algorithms for dft and HOG feature extraction so we decide to build the parallel oject tracker based on the KCF implementation. 

Fast Fourier Transforms (FFT) is an efficient way for calculation of DFT by utilizing DFT's symmetry properties. Compared with O(N^2) computation complexity, the computational cost of FFT is only O(N log(N)) It is a key tool in most of digital signal processing systems including object tracking ones. In the KCF object tracking algorithm, FFT is heavily used for learning and evaluation to achieve lower computational complexity. For example, by utilizing FFT operation, KRLS learning algorithm is improved from O(N^4) to only O(N^2 log(N)) for NxN 2D images.

## Challenge
The biggest challenge is that we are not very familiar with the computer vision algorithm so it may take time for us to understand the implementation details of the KCF tracker. However, since we have the open source implementation, we may not need to understand every single lines in the code. Instead, we may identify the paralizable parts and do optimization on these parts.

Although FFT brings out great improvement on performance, it is still computational intensive with O(n log(n)) complexity. According to the properties of FFT algorithm itself, we believe that there should be some computational independent tasks inside it such as the DFT calculation of even terms and odd terms in each iteration. So, it may benefit from parallel implementation on multi-threaded processing. To implement FFT in parallel, the major points is that multi-dimensional FFT is utilized. In this regard, analyzing computation independence becomes tough. Meanwhile, applying such complicated algorithm with parallel technologies especially AVX instructions are pretty hard. We plan to combine both AVX and OpenMP to support different level of parallelism for FFT.

For HOG feature extraction, there have already been a few multicore CPU and GPU implementations for different computer vision tasks. We would be able to borrow some the ideas from them but their optimizations are often combined with the specific vision task algorithm, so we may need to develop our own version and some further optimization might be adopted to offer better performance for the KCF tracker.

KCF has done a lot of optimization to reduce the computation which makes it the fastest cpu object tracker implementation, we are not sure that the remaining computation would still be enough to gain benefits from multithreading. This might be a problem. We would try to identify this as soon as possible.

## Resources
The Author of KCF open source their [code](https://github.com/joaofaro/KCFcpp) on the github so we are able to build our parallel version based on their implementation. OpenCV is an open source library for computer vision. It provides tons of functions for real-time computer vision. The [code](https://github.com/opencv/opencv) is available on GitHub. we could build our parallel FFT implementation based on the top of OpenCV's source code. We would develop and test our software on the GHC machines which have 8 cores and we may further test the performance on Xeon Phi machines or develop the GPU version to see how our implementation scales if everything goes smoothly.

## Evaluation and Goals
Since KCF tracker currently is a sequential implementation, we hope our parallel version could achieve close to linear speedup comparing to the sequential version. The basic performance measure would just be the speedup/ core number graph. However, as we haven't identified the dependency in the current implementation, we are not very sure if this is possible to achieve the linear speedup.

We may try to use other libraries like FFTW to compare the performance to see if we have achieved a good enough parallel implementation. Opencv also have implemented the same object tracker, which we may use to perform the comparison.

Minimum Goal: 1. ReImplement FFT algorithm to support multicore and SIMD parallelism. The parallel FFT should be faster than the original FFT implementation in KCF and improve the overall performance of KCF algorithm on multicore machines. 2. Implement multithread and SIMD version of the HOG feature computation, along with parallel FFT, to improve the performance of KCF algorithm.

Expected Goal: 1. The parallel FFT could achieve sub-linear scalability with the increase of core number in the machine. 2. Perform enough experiment to make parallel HOG feature processing fast enough to reach some other implementation's speedup.

Stretch Goal: The parallel FFT could beat other excellent DFT calculation libraries, such as FFTW, in performance.

## Reference
[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,   
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[3] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,   
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

[4] H. Kiani Galoogahi, A. Fagg, C. Huang, D. Ramanan, S.Lucey,   
"Need for Speed: A Benchmark for Higher Frame Rate Object Tracking", 2017, arXiv preprint arXiv:1703.05884.

[5] Y. Wu and J. Lim and M. Yan,   
"Online Object Tracking: A Benchmark", CVPR 2013

## Schedule
Week 1 (Apr. 3rd): 

Deciding project idea; 
Investigation background (Papers and Code); 
Writing proposal. 

Week 2 (Apr. 10th): 

Implementing parallel FFT with OpenMP;
Implementing parallel HOG features extraction with OpenMP.

Week 3 (Apr. 17th): 

Incremental developing parallel FFT with AVX;
Incremental developing parallel HOG features extraction with AVX.
Integrating two implementations into KCF algorithm;

Week 4 (Apr. 24th):

Conducting experiments to evaluate ParaKCF on both performance improvement and scalability;
Tuning the implementation to improve the performance.

Week 5 (May 1st): 

Cleaning code;
Perfecting documentation;
Writing final report;
Preparing for the final competition.
