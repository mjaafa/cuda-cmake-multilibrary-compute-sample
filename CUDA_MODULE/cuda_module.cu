/* @brief
 *
 * Copyright (c)  2016 AZZAHRA Consulting France
 * The computer program contained herein contains proprietary
 * information which is the property of AZZAHRA Consulting France.
 * The program may be used and/or copied only with the written
 * permission of AZZAHRA Consulting France or in accordance with the
 * terms and conditions stipulated in the agreement/contract under
 * which the programs have been supplied.
 *
 * @author Mohamed Jaafar <mohamet.jaafar@gmail.com>
 */

/**
* @defgroup SAMPLE SAMPLE
* \{ */
/**
* @defgroup CUDA_MODULE CUDA_MODULE
* \{ */
/* ##########################################################################################
** #                                       INCLUDES                                         #
** ##########################################################################################*/
/**
* @defgroup SAMPLE SAMPLE
* \{ */
/**
* @defgroup CUDA_MODULE CUDA_MODULE
* \{ */
/* ##########################################################################################
** #                                       INCLUDES                                         #
** ##########################################################################################*/
#define MODULE "CUDA_MODULE"

#include "sample.h"
#include "sample_types.h"
/* ##########################################################################################
** #                                   DEFINES & MACROS                                     #
** ##########################################################################################*/


/* ##########################################################################################
** #                                       TYPEDEFS                                         #
** ##########################################################################################*/

/* ##########################################################################################
** #                                       FUNCTIONS                                        #
** ##########################################################################################*/

/*
 * KERNEL CUDA : test.
 *Kernel that executes on the CUDA device
 */

__global__ void cudaAdd(int *array_A, int *array_B, int *array_C)
{
        array_C[blockIdx.x] = array_A[blockIdx.x] + array_B[blockIdx.x];
}
extern "C"
{
SAMPLE_Error_t CUDA_MODULE_TestArrayAdd(int arraySize)
{
    SAMPLE_Error_t  z_ret = SAMPLE_RET_OK;
    unsigned int    Idx   = 0;
    int             *host_Array_A; //[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int             *host_Array_B; //[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int             *host_Array_C; //[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};


    int             *device_Array_A;
    int             *device_Array_B;
    int             *device_Array_C;
    int    	    arraySz = ((arraySize > 0) ? arraySize : 100);

    host_Array_A = (int*) malloc ((arraySz * sizeof (int)));
    host_Array_B = (int*) malloc ((arraySz * sizeof (int)));
    host_Array_C = (int*) malloc ((arraySz * sizeof (int)));

    srand (GOD_OF_RANDOM);

    for (Idx = 0; Idx < arraySz; Idx++)
    {
        host_Array_A[Idx] = (rand () % GOD_OF_RANDOM);
        host_Array_B[Idx] = (rand () % GOD_OF_RANDOM);
    }


    SAMPLE_LOG_INFO("size equal %d ", arraySz);
    /* cuda Part */
    if (cudaSuccess != cudaMalloc((void**) &device_Array_A, (arraySz * sizeof(int))))
    {
        SAMPLE_LOG_ERROR(" cannot allocate memory on device GPU");
        z_ret = SAMPLE_RET_CUDA_BASIC_MEM_ERROR;
        goto error;
    }
    /* else : continue */

    if (cudaSuccess != cudaMalloc((void**) &device_Array_B, (arraySz * sizeof(int))))
    {
        SAMPLE_LOG_ERROR(" cannot allocate memory on device GPU");
        z_ret = SAMPLE_RET_CUDA_BASIC_MEM_ERROR;
        goto error;
    }
    /* else : continue */

    if (cudaSuccess != cudaMalloc((void**) &device_Array_C, (arraySz * sizeof(int))))
    {
        SAMPLE_LOG_ERROR(" cannot allocate memory on device GPU");
        z_ret = SAMPLE_RET_CUDA_BASIC_MEM_ERROR;
        goto error;
    }
    /* else : continue */

    if (cudaSuccess != cudaMemcpy(device_Array_A,
                                  host_Array_A,
                                  (arraySz * sizeof(int)),
                                  cudaMemcpyHostToDevice))
    {
        SAMPLE_LOG_ERROR(" cannot cpoy memory on device GPU");
        z_ret = SAMPLE_RET_CUDA_BASIC_MEM_ERROR;
        goto error;
    }
    /* else : continue */

    if (cudaSuccess != cudaMemcpy(device_Array_B,
                                  host_Array_B,
                                  (arraySz * sizeof(int)),
                                  cudaMemcpyHostToDevice))
    {
        SAMPLE_LOG_ERROR(" cannot cpoy memory on device GPU");
        z_ret = SAMPLE_RET_CUDA_BASIC_MEM_ERROR;
        goto error;
    }
    /* else : continue */

    /* let the dogs out */
    cudaAdd <<<arraySz,1>>> (device_Array_A, device_Array_B, device_Array_C);

    if (cudaSuccess != cudaMemcpy(host_Array_C,
                                  device_Array_C,
                                  (arraySz * sizeof(int)),
                                  cudaMemcpyDeviceToHost))
    {
        SAMPLE_LOG_ERROR(" cannot cpoy memory on device GPU");
        z_ret = SAMPLE_RET_CUDA_BASIC_MEM_ERROR;
        goto error;
    }
    /* else : continue */

    /* get back the result */
    for (Idx = 0; Idx < arraySz; Idx++)
    {
        SAMPLE_LOG_INFO(" [%d] + [%d] = [%d]", host_Array_A[Idx],
                                               host_Array_B[Idx],
                                               host_Array_C[Idx]);
    }

    free(host_Array_A);
    free(host_Array_B);
    free(host_Array_C);
    cudaFree(device_Array_A);
    cudaFree(device_Array_B);
    cudaFree(device_Array_C);
error:
    return z_ret;
}
}
// CUDA_MODULE
/** \} */
// SAMPLE
/** \} */

