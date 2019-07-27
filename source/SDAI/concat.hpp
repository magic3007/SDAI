#pragma once
#include "configure.h"
#include <assert.h>

#if DEBUG
#include <stdio.h>
using namespace std;
#endif

namespace SDAI
{

template<int ROW, int COL, int ... DIMS>
class Concat_DataStream{
        #define N_MAPS sizeof...(DIMS)
        int dims[N_MAPS];
        int offsets[N_MAPS];
        int OUT_DIM;
    public:
        Concat_DataStream() : dims{(DIMS)...} {
            offsets[0] = 0;
            for(int i = 1; i < N_MAPS; i++)
                offsets[i] = offsets[i-1] + dims[i];
            OUT_DIM = offsets[N_MAPS - 1] + dims[N_MAPS - 1];
#if DEBUG
            fprintf(stderr, "concat -> %d x %d x %d", ROW, COL, OUTPUT_DIM);
#endif
        }

        inline void array_copy(volatile TYPE_T *psrc, volatile TYPE_T *pdst, int len){
            #pragma HLS INLINE
            for(int i = 0; i < len; i++)
                pdst[i] = psrc[i];
        }

        void feedforward(TYPE_T* pdatas[N_MAPS], volatile TYPE_T *res){
            for(int i = 0; i < ROW; i++)
                for(int j = 0; j < COL; j++){
                    for(int k = 0; k < N_MAPS; k++){
                        #pragma HLS LOOP_FLATTEN
                        int input_index = i * COL * dims[k] + j * dims[k];
                        int output_index = i * COL * OUTPUT_DIM + j * OUTPUT_DIM + offsets[k];
                        int len = dims[k];
                        array_copy(&pdata[k][input_index], res + output_index, len);                
                    }
                }
        }
        #undef N_MAPS

};

}