#pragma once

#include <assert.h>
#include "mem.h"
#include "reshape.h"
#if DEBUG
#include <stdio.h>
using namespace std;
#endif

namespace SDAI
{
enum UPSAMPLING_MODE{
    NEAREST,
};
/**
 * TODO: support other upsampling mode, like 'linear', 'bilinear',
 *  'bicubic' and 'trilinear'.
 */

template<int INPUT_DIM, int ROW, int COL, int SCALE_FACTOR, UPSAMPLING_MODE=NEAREST,
    int OUT_ROW = ROW * SCALE_FACTOR, int OUT_COL = COL * SCALE_FACTOR>
class Upsampling2d_DataStream{
    public:
        Upsampling2d_DataStream(){
#if DEBUG
	fprintf(stderr, "upsampling          %d x %d/(%d, %d)  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR, 
		ROW, COL, INPUT_DIM, OUT_ROW, OUT_COL, INPUT_DIM);
#endif
        }

        void feedforward(volatile TYPE_T *data, volatile TYPE_T *res){
            Reshape_Stream_2D<COL, INPUT_DIM> stream;

            for(int i = 0; i < ROW; i++){
                stream.forward(&data[i * COL * INPUT_DIM]);
                for(int j = 0; j < COL; j++)
                    for(int delta_i = 0; delta_i < SCALE_FACTOR; delta_i++)
                        for(int delta_j = 0; delta_j < SCALE_FACTOR; delta_j++)
                            for(int k = 0; k < INPUT_DIM; k++){
                                int input_idx = j * INPUT_DIM + k;
                                int output_idx = (i * SCALE_FACTOR + delta_i) * OUT_COL * INPUT_DIM + (j * SCALE_FACTOR + delta_j) * INPUT_DIM + k;
                                res[output_idx] = stream.res[j * INPUT_DIM + k];
                        }
            }
            // TODO: optimize upsampling2d
        }
};
}