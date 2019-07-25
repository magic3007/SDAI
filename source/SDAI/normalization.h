#ifndef __NORMALLIZATION_H__
#define __NORMALLIZATION_H__
#include "configure.h"
#include <assert.h>

#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

template<int NB_FILTER>
class BatchNorm2d{
    public:
        TYPE_T scales[NB_FILTER];
        TYPE_T bias[NB_FILTER];
        TYPE_T means[NB_FILTER];
        TYPE_T reci_var2_eps[NB_FILTER]; //  \frac{1}{\sqrt{variance + \epsilon}}

        BatchNorm2d_DataStream(const TYPE_T *SCALES, const TYPE_T *BIAS, 
            const TYPE_T *MEANS, const TYPE_T *VARS, const float EPSILON=.000001f){
#if BATCHNORM2D_PERF_MODE == PERF_HIGH || BATCHNORM2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_MERGE
#endif
            for(int i = 0; i < NB_FILTER; i++)
                scales[i] = SCALES[i];
            for (int i = 0; i < NB_FILTER; i++)
                bias[i] = BIAS[i];
            for (int i= 0; i < NB_FILTER; i++)
                means[i] = MEANS[i];
            for (int i = 0; i < NB_FILTER; i++)
                reci_var2_eps[i] = (TYPE_T)(1. / sqrt(double(VARS[i]) + EPSILON) );
            // TODO: calculate \frac{1}{\sqrt{variance + \epsilon}} in CPU
        }

        inline TYPE_T forward(TYPE_T x, int k){
            #pragma HLS INLINE
            return scales[k] * (x - means[k]) * reci_var2_eps[k] + bias[k];
        }
}

}

#endif // !__NORMALLIZATION_H__
