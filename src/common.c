/* @brief
 *
 * Copyright (c)  2016 France
 * The computer program contained herein contains proprietary
 * information which is the property of Mohamed JAAFAR.
 * The program may be used and/or copied only with the written
 * permission Mohamed JAAFAR or in accordance with the
 * terms and conditions stipulated in the agreement/contract under
 * which the programs have been supplied.
 *
 * @author Mohamed Jaafar <mohamet.jaafar@gmail.com>
 */

/**
* @defgroup SAMPLE SAMPLE
* \{ */
/**
* @defgroup COMMON_MODULE COMMON_MODULE
* \{ */
/* ##########################################################################################
** #                                       INCLUDES                                         #
** ##########################################################################################*/
#define MODULE "COMMON_MODULE"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <glib.h>
#include <gio/gio.h>
#include <string.h>
#include <pthread.h>

#include "sample.h"
#include "sample_types.h"
#ifdef BUILD_C_MODULE_API
#include "c_module.h"
#endif /* BUILD_C_MODULE_API */

#ifdef BUILD_CUDA_MODULE_API
#include "cuda_module.h"
#endif /* BUILD_CUDA_MODULE_API */
/* ##########################################################################################
** #                                   DEFINES & MACROS                                     #
** ##########################################################################################*/

/* ##########################################################################################
** #                                       TYPEDEFS                                         #
** ##########################################################################################*/
/**
** @brief defines the structure that holds the common api context.
*/
typedef struct WRAPPER_COMMON_context_s
{
#ifdef BUILD_C_MODULE_API
    pthread_t               c_module_service_id;          /*!< thread id for notifier                 */
#endif /* BUILD_C_MODULE_API */
#ifdef BUILD_CUDA_MODULE_API
    pthread_t               cuda_module_service_id;       /*!< thread id for notifier                 */
    SAMPLE_syncDevice_t     syncDev;                      /*!< synch push orders tp device            */
#endif /* BUILD_CUDA_MODULE_API */
} SAMPLE_api_context_t;

SAMPLE_api_context_t     sample_context;
/* ##########################################################################################
** #                                       FUNCTIONS                                        #
** ##########################################################################################*/

/*##############################################################################*/

int main (void)
{
    SAMPLE_Error_t          z_ret   = SAMPLE_RET_OK;
    GError                  *error  = NULL;
    static pthread_attr_t   threadAttr;
    pthread_mutexattr_t     mutexAttr;
    pthread_condattr_t      condAttr;
    char                    order[2];
    struct sched_param      sched_param;
    int                     sizeArray;


    memset(&sample_context, 0, sizeof(struct WRAPPER_COMMON_context_s));

    z_ret = (pthread_mutexattr_init( &mutexAttr ));
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR("Cannot initialize mutex attributes (err = %d) \n", z_ret);
       return z_ret;
    }

    z_ret = (pthread_mutex_init(&(sample_context.syncDev.deviceLock), &mutexAttr));
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR("Cannot initialize mutex (err = %d)\n", z_ret);
       return z_ret;
    }

    pthread_mutexattr_destroy(&mutexAttr);

    z_ret = (pthread_cond_init(&(sample_context.syncDev.deviceTrigger), &condAttr));
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR("Cannot initialize cond (err = %d)\n", z_ret);
       return z_ret;
    }

    pthread_condattr_destroy(&condAttr);
#ifdef BUILD_C_MODULE_API
    /* create processor */
    z_ret = (pthread_attr_init(&threadAttr));
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR("pthread_attr_init() failed (err = %d)", z_ret);
       return z_ret;
    }

    z_ret = (pthread_create(&sample_context. c_module_service_id,
                               &threadAttr, C_MODULE_entry_point,
                               NULL) < 0);
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR ("pthread_create C_MODULE_entry_point Thread failed\n");
       return z_ret;
    }

    /* cleanup */
    pthread_attr_destroy(&threadAttr);

#endif /* BUILD_C_MODULE_API */
#ifdef BUILD_CUDA_MODULE_API
    z_ret = CUDA_MODULE_Init_Cuda_GPU();
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR("CUDA_MODULE_Init_Cuda_GPU failed (err = %d)", z_ret);
       return z_ret;
    }

    z_ret = (pthread_attr_init(&threadAttr));
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR("pthread_attr_init() failed (err = %d)", z_ret);
       return z_ret;
    }

   /* pthread priority fixing */
   z_ret = pthread_attr_getschedparam (&threadAttr, &sched_param);
   sched_param.sched_priority  = CUDA_MODULE_WORKER_PRIORITY;
   if (z_ret != SAMPLE_RET_OK)
   {
       SAMPLE_LOG_ERROR("error = %d %s \n", z_ret ,strerror(z_ret));
       goto error;
   }

   z_ret = pthread_attr_setinheritsched(&threadAttr, PTHREAD_EXPLICIT_SCHED);
   if (z_ret != SAMPLE_RET_OK)
   {
       SAMPLE_LOG_ERROR("error = %d %s \n", z_ret, strerror(z_ret));
       goto error;
   }

   z_ret = pthread_attr_setschedpolicy(&threadAttr, SCHED_RR);
   if (z_ret != SAMPLE_RET_OK)
   {
       SAMPLE_LOG_ERROR("error = %d %s \n", z_ret, strerror(z_ret));
       goto error;
   }

   z_ret = pthread_attr_setschedparam(&threadAttr,&sched_param);
   if (z_ret != SAMPLE_RET_OK)
   {
       SAMPLE_LOG_ERROR("error = %d %s \n", z_ret ,strerror(z_ret));
       goto error;
   }

    SAMPLE_LOG_ERROR (" pointers syncDev = %p ", &(sample_context.syncDev));

    z_ret = (pthread_create(&sample_context.cuda_module_service_id,
                               &threadAttr, CUDA_MODULE_entry_point,
                               &(sample_context.syncDev)) < 0);
    if(SAMPLE_RET_OK != z_ret)
    {
       SAMPLE_LOG_ERROR ("pthread_create SAMPLE_entry_point Thread failed \n");
       goto error;
    }

    /* cleanup */
    pthread_attr_destroy(&threadAttr);
    pthread_mutexattr_destroy(&mutexAttr);

#endif /* BUILD_CUDA_MODULE_API */
#ifdef BUILD_C_MODULE_API
    pthread_join(sample_context.c_module_service_id, NULL);
#endif /* BUILD_C_MODULE_API */
#ifdef BUILD_MODULE_API
    pthread_join(sample_context.cuda_module_service_id, NULL);
#endif /* BUILD_MODULE_API */

    /* Start taking orders */

    SAMPLE_LOG_INFO(" Threads are ready for work !");


#ifdef BUILD_FOR_NVIDIA_VISUAL_PROFILER
        order[0] = '1';
#endif /* BUILD_FOR_NVIDIA_VISUAL_PROFILER */
    while(SAMPLE_TURN_ME_ON_AND_ON)
    {
    /* ALL you want goes here */
        SAMPLE_LOG_MSG(" Give order please just to block main process dummy enter would work !");

        SAMPLE_LOG_WARNING("please enter request !");
        SAMPLE_LOG_WARNING("1/ make cuda work ");
        SAMPLE_LOG_WARNING("2/ exit");
        /* for sync main is not sleepy*/
        usleep(10);
        pthread_mutex_lock(&(sample_context.syncDev.deviceLock));
#ifndef BUILD_FOR_NVIDIA_VISUAL_PROFILER
        scanf("%c", &order[0]);
#endif /* !BUILD_FOR_NVIDIA_VISUAL_PROFILER */
        order[1] = '\0';
        if (!strcmp(order, "1"))
        {
            SAMPLE_LOG_WARNING(" give array size ");
#ifndef BUILD_FOR_NVIDIA_VISUAL_PROFILER
            scanf("%d", &sizeArray);
#else
            sizeArray = 100;
#endif /* !BUILD_FOR_NVIDIA_VISUAL_PROFILER */

            sample_context.syncDev.data = sizeArray;
            sample_context.syncDev.dataType = SAMPLE_DATA_INT_TYPE;
            pthread_cond_signal(&(sample_context.syncDev.deviceTrigger));

#ifdef BUILD_FOR_NVIDIA_VISUAL_PROFILER
            order[0] = '2';
#endif /* BUILD_FOR_NVIDIA_VISUAL_PROFILER */
        }
        else if (!strcmp(order, "2"))
        {
            exit(0);
        }
        /*else : continue wrong choice */

        pthread_mutex_unlock(&(sample_context.syncDev.deviceLock));
    }


error:
    return z_ret;
}
// CUDA_MODULE
/** \} */
// SAMPLE
/** \} */

