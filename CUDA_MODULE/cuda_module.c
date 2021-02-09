/* @brief
 *
 * Copyright (c)  2016 
 * The computer program contained herein contains proprietary
 * information which is the property of AZZAHRA Consulting France.
 * The program may be used and/or copied only with the written
 * permission of Mohamed JAAFAR accordance with the
 * terms and conditions stipulated in the agreement/contract under
 * which the programs have been supplied.
 *
 * @author Mohamed Jaafar <mohamet.jaafar@protonmail.ch>
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
#define MODULE "CUDA_MODULE"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#include "sample.h"
#include "sample_types.h"
#include "cuda_module.h"
#include "cuda_module.cuh"
#include "cuda_module_internal.h"

/* ##########################################################################################
** #                                   DEFINES & MACROS                                     #
** ##########################################################################################*/

/**
** @brief defines the parsed fileData.
*/
//PRIVATE GArray *memoryHolder;

/* ##########################################################################################
** #                                       TYPEDEFS                                         #
** ##########################################################################################*/

/* ##########################################################################################
** #                                       FUNCTIONS                                        #
** ##########################################################################################*/

/**
 *
 * \brief   server's main.
 *
 * \param   void                    : no args needed.
 * \return  error code
 * \author  mohamet.jaafar\@gmail.com
 * \date    2016
 */

void * CUDA_MODULE_entry_point (void * params)
{
    SAMPLE_syncDevice_t *syncDev = (SAMPLE_syncDevice_t*) params;
    SAMPLE_LOG_ERROR (" pointers syncDev = %p ", syncDev);
    SAMPLE_LOG_WARNING(" cuda worker is ready now !");

    while (SAMPLE_TURN_ME_ON_AND_ON)
    {
        /* prepare to cpy to device memory : crtitical section */
        pthread_mutex_lock(&syncDev->deviceLock);
        pthread_cond_wait(&syncDev->deviceTrigger, &syncDev->deviceLock);
        SAMPLE_LOG_MSG(" ORDER FROM USER RECEIVED !!");
        /* TODO : workers type exec */
        if (SAMPLE_DATA_INT_TYPE == syncDev->dataType)
        {
            SAMPLE_LOG_INFO(" received size array = %d ", (int)syncDev->data);

            if (SAMPLE_RET_OK != CUDA_MODULE_TestArrayAdd(syncDev->data))
            {
                SAMPLE_LOG_ERROR(" something went wrong !");
            }
            /* else : continue */
        }
        /* else : continue */

        pthread_mutex_unlock(&syncDev->deviceLock);
    }
    SAMPLE_LOG_MSG("exit");
    return (NULL);
}


/**
 *
 * \brief   cuda init to check props and prhibs.
 *
 * \param   void                            user_data   : User data passed.
 * \return  error type SAMPLE_RET_OK;
 * \author  mohamet.jaafar\@gmail.com
 * \date    2016
 */
SAMPLE_Error_t CUDA_MODULE_Init_Cuda_GPU(void)
{
    cudaError_t             zCuda_ret        = cudaSuccess;
    int                     devID            = CUDA_MODULE_DEVICE_ID;
    int                     devCount         = 0;
    SAMPLE_Error_t          z_ret            = SAMPLE_RET_OK;
    struct cudaDeviceProp   cudaDeviceProp;
    u_int16_t               cudaDevCaps      = 0x0;
    int                     deviceIndex      = 0;

    SAMPLE_LOG_INFO("[CUDA_MODULE - module Using CUDA] - Starting...\n");

    zCuda_ret = cuInit(devID);
    if( cudaSuccess != zCuda_ret)
    {
        SAMPLE_LOG_WARNING("CUDA - error when init device cuda %s :: %d \n", cudaGetErrorString(zCuda_ret), zCuda_ret);
    }
    else
    {
        SAMPLE_LOG_MSG("CUDA - Device init on the workstation %d  :: %s \n", devCount, cudaGetErrorString(zCuda_ret));
    }

    zCuda_ret = cudaGetDeviceCount(&devCount);
    if( cudaSuccess != zCuda_ret)
    {
        SAMPLE_LOG_WARNING("CUDA - error when getting device count %s :: %d \n", cudaGetErrorString(zCuda_ret), zCuda_ret);
    }
    else
    {
        SAMPLE_LOG_MSG("CUDA - Device count on the workstation %d  :: %s \n", devCount, cudaGetErrorString(zCuda_ret));
    }
    zCuda_ret = cudaGetDevice(&deviceIndex);
    if( cudaSuccess != zCuda_ret)
    {
        SAMPLE_LOG_WARNING("CUDA - error when setting device ID %s :: %d \n", cudaGetErrorString(zCuda_ret), zCuda_ret);
    }
    else
    {
        SAMPLE_LOG_MSG("CUDA - Device count on the workstation %d  :: %s \n", deviceIndex, cudaGetErrorString(zCuda_ret));
    }

    zCuda_ret = cudaSetDevice(devID);
    if (cudaSuccess != zCuda_ret)
    {
        SAMPLE_LOG_WARNING("CUDA - error when setting device ID %s :: %d \n", cudaGetErrorString(zCuda_ret), zCuda_ret);
        /* ignore the error use the default */
    }
    /* else : continue */

    zCuda_ret = cudaGetDevice(&devID);
    if (cudaSuccess != zCuda_ret)
    {
        SAMPLE_LOG_WARNING("cudaGetDevice returned error %s :: %d \n", cudaGetErrorString(zCuda_ret), zCuda_ret);
        /* ignore the error try to get it anyway */
    }
    /* else : continue */

    zCuda_ret = cudaGetDeviceProperties(&cudaDeviceProp, devID);
    if(cudaSuccess != zCuda_ret)
    {
        SAMPLE_LOG_ERROR("cudaGetDeviceProperties returned error %s :: %d \n", cudaGetErrorString(zCuda_ret), zCuda_ret);
    }
    else
    {
        SAMPLE_LOG_INFO("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, cudaDeviceProp.name, cudaDeviceProp.major, cudaDeviceProp.minor);
    }

    if (cudaComputeModeProhibited == cudaDeviceProp.computeMode)
    {
        SAMPLE_LOG_ERROR("Error: device is running in <Compute Mode Prohibited>, no threads can be used ::cudaSetDevice().\n");
        z_ret = SAMPLE_RET_CUDA_INIT_ERROR;
        goto error;
    }

    /*
    ** get GPU memory props.
    */
    cudaDevCaps |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_MEMORY_CAPA_FIELD);
    cudaDevCaps |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD);

    CUDA_MODULE_DEVICE_CAPA_UPDATE(cudaDeviceProp, cudaDevCaps);

    if ((CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_MEMORY_CAPA_FIELD) != \
        (cudaDevCaps & ( CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_MEMORY_CAPA_FIELD)))
    {
        SAMPLE_LOG_INFO("cuda driver memory caps = 0x%x \n", cudaDevCaps & CUDA_MODULE_MEM_CAPS_MASK);
    }
    else
    {
        SAMPLE_LOG_INFO("cuda driver :: unable to  catch memory caps 0x%x \n", cudaDevCaps);
    }

    if ((CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD) != \
        (cudaDevCaps & ( CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD)))
    {
        SAMPLE_LOG_INFO("cuda driver kernel exec caps present = 0x%x \n", cudaDevCaps & CUDA_MODULE_KERN_EXEC_CAPS_MASK);
    }
    else
    {
        SAMPLE_LOG_INFO("cuda driver :: unable to  catch kernel exec caps 0x%x \n", cudaDevCaps);
    }
error:
    return z_ret;
}

// CUDA_MODULE
/** \} */
// SAMPLE
/** \} */

