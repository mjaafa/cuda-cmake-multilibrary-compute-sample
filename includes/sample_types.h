#ifndef SAMPLE_TYPES_H
# define SAMPLE_TYPES_H
/* @brief
 *
 * Copyright (c) 2016 France
 *
 * The computer program contained herein contains proprietary
 * information which is the property of Mohamed JAAFAR.
 * The program may be used and/or copied only with the written 
 * permission of Mohamed JAAFAR or in accordance with the
 * terms and conditions stipulated in the agreement/contract under
 * which the programs have been supplied.
 *
 * @author Mohamed Jaafar <mohamet.jaafar@gmail.com>
 */

/**
* @defgroup SAMPLE SAMPLE
* \{ */
/**
* @defgroup SAMPLE_TYPES SAMPLE_TYPES
* \{ */ 


# ifdef __cplusplus
extern "C" {
# endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ##########################################################################################
** #                                        MACROS                                          #
** ##########################################################################################*/
/**
** @brief defines macro for more secured asprintf.
*/
#define Sasprintf(write_to, ...) {\
char *tmp_string_for_extend = write_to; \
    asprintf(&(write_to), __VA_ARGS__); \
free(tmp_string_for_extend); \
}

/**
** @brief defines macro for unused variables.
*/
#define UNUSED_VARIABLE(x) (x=x)

/**
** @brief defines macro for unused pointers.
*/
#define UNUSED_POINTER(p)  (*p)

/**
** @brief defines macro for private functions.
*/
#define PRIVATE             static

/**
** @brief defines macro for private debug system using module restrictions.
*/

#define SAMPLE_LOG_ERROR( ...)          \
    printf("\033[31m"); printf("[%s]%s : %d > ",MODULE,__FUNCTION__,__LINE__);\
    printf( __VA_ARGS__);printf("\033[0m \n")
#define SAMPLE_LOG_WARNING( ...)        \
    printf("\033[33m"); printf("[%s]%s : %d > ",MODULE,__FUNCTION__,__LINE__);\
    printf( __VA_ARGS__);printf("\033[0m \n")
#define SAMPLE_LOG_INFO( ...)           \
    printf("\033[32m"); printf("[%s]%s : %d > ",MODULE,__FUNCTION__,__LINE__);\
    printf( __VA_ARGS__);printf("\033[0m \n")
#define SAMPLE_LOG_MSG( ...)            \
    printf("\033[34m"); printf("[%s]%s : %d > ",MODULE,__FUNCTION__,__LINE__);\
    printf( __VA_ARGS__);printf("\033[0m \n")
#define SAMPLE_LOG_PRINTF( ...)         \
    printf("\033[36m"); printf("[%s]%s : %d > ",MODULE,__FUNCTION__,__LINE__);\
    printf( __VA_ARGS__);printf("\033[0m \n")
/**
**\brief MAY the 4th be with you.
*/
#define SAMPLE_TURN_ME_ON_AND_ON        54

/**
**\brief random max value.
*/
#define GOD_OF_RANDOM                   100
/* ##########################################################################################
** #                                       TYPEDEFS                                         #
** ##########################################################################################*/

/**
** @brief enumerates the error types in SAMPLE module.
*/
typedef enum SAMPLE_Error_e
{
    SAMPLE_RET_OK                        ,       /*!< no error                               */
    SAMPLE_RET_DBUS_ERROR                ,       /*!< specific dbus error                    */
    SAMPLE_RET_CONNECT_SRV_ERROR         ,       /*!< specific connection to server error    */
    SAMPLE_RET_COMPRESSION_RAINBOW_ERROR ,
    SAMPLE_RET_CUDA_INIT_ERROR           ,
    SAMPLE_RET_CUDA_BASIC_MEM_ERROR      ,
    SAMPLE_RET_MAX

}SAMPLE_Error_t;

#ifdef BUILD_CUDA_MODULE_API
typedef enum SAMPLE_DevDATA_e
{
    SAMPLE_DATA_INT_TYPE  ,
    SAMPLE_DATA_INT_CHAR  ,
    SAMPLE_DATA_INT_STRING,
    SAMPLE_DATA_INT_MAX
} SAMPLE_DevDATA_t;

typedef struct SAMPLE_syncDevice_s
{
    pthread_mutex_t     deviceLock;
    pthread_cond_t      deviceTrigger;
    unsigned int        data;
    SAMPLE_DevDATA_t    dataType;
} SAMPLE_syncDevice_t;
#endif /* BUILD_CUDA_MODULE_API*/
# ifdef __cplusplus
}
# endif
// SAMPLE_TYPES
/** \} */
// SAMPLE
/** \} */
#endif /* !SAMPLE_TYPES_H */
