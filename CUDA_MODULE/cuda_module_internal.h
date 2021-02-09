#ifndef CUDA_MODULE_INTERNAL_H
# define CUDA_MODULE_INTERNAL_H
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


# ifdef __cplusplus
extern "C" {
# endif
/* ##########################################################################################
** #                                       INCLUDES                                         #
** ##########################################################################################*/
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

/* ##########################################################################################
** #                                   DEFINES & MACROS                                     #
** ##########################################################################################*/

/**
** @brief defines the parsed fileData.
*/

/* ##########################################################################################
** #                                       TYPEDEFS                                         #
** ##########################################################################################*/
/**
** \brief defines the cuda device ID (default 0)
*/
#define CUDA_MODULE_DEVICE_ID 0 
/**
** \brief enumeration defines the device capabilities used when performing cmd buffer from host
** \par                       to cuda driver : UVA, ...
*/
typedef enum CUDA_MODULE_devCaps_e
{
    CUDA_DEVICE_MEMORY_CAPA_FIELD               = 0x0 ,
    CUDA_DEVICE_UVA_CAPA_FIELD                        ,
    CUDA_DEVICE_PAGE_MEM_ACCESS_CAPA_FIELD            ,
    CUDA_DEVICE_MAP_HOST_MEM_CAPA_FIELD               ,
    CUDA_DEVICE_MEM_CCRNT_WITH_HOST_CAPA_FIELD        ,
    CUDA_DEVICE_MEM_MANAGED_CAPA_FIELD                ,
    CUDA_DEVICE_MEM_GLOBAL_L1_CACHE_CAPA_FIELD        ,
    CUDA_DEVICE_MEM_LOCALS_L1_CACHE_CAPA_FIELD        ,

    CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD          = 0x08,
    CUDA_DEVICE_KERNEL_CCRNT_EXEC_CAPA_FIELD          ,
    CUDA_DEVICE_KERNEL_EXEC_TIMEOUT_CAPA_FIELD        ,
    CUDA_DEVICE_STREAM_PRIO_CAPA_FIELD                ,
    CUDA_DEVICE_MULTI_GPU_CAPA_FIELD                  ,

    CUDA_DEVICE_CAPA_FIELD_MAX

} CUDA_MODULE_devCaps_t;

/**
** \brief defines the memory caps mask .
*/
#define CUDA_MODULE_MEM_CAPS_MASK                   (0xFF)
/**
** \brief defines the kernel exec caps mask .
*/
#define CUDA_MODULE_KERN_EXEC_CAPS_MASK             (0xFF00)
/**
** \brief defines value set if a capability is present : to be put in the right field
** \par   for every capability.
*/
#define CUDA_MODULE_DEVICE_CAPA_IS_PRESENT        (1)

#ifdef CUDA_MANAGED_ACCESS_ACTIVE 
       if (capaCuda.pageableMemoryAccess)                               \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_PAGE_MEM_ACCESS_CAPA_FIELD);      \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.concurrentManagedAccess)                            \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_MEM_CCRNT_WITH_HOST_CAPA_FIELD);  \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.managedMemSupported)                                \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_MEM_MANAGED_CAPA_FIELD);          \
       }                                                                \
       /* else :continue */                                             
#endif

#define CUDA_MODULE_DEVICE_CAPA_UPDATE(capaCuda, capaHolder)                                    \
   if ((CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_MEMORY_CAPA_FIELD) ==                 \
        (capaHolder & (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_MEMORY_CAPA_FIELD)))   \
   {                                                                                            \
       if (capaCuda.unifiedAddressing)                                  \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_UVA_CAPA_FIELD);                  \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.canMapHostMemory)                                   \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_MAP_HOST_MEM_CAPA_FIELD);         \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.globalL1CacheSupported)                             \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_MEM_GLOBAL_L1_CACHE_CAPA_FIELD);  \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.localL1CacheSupported)                              \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_MEM_LOCALS_L1_CACHE_CAPA_FIELD);  \
       }                                                                \
       /* else :continue */                                             \
                                                                        \
       capaHolder &= ~(CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<            \
                          CUDA_DEVICE_MEMORY_CAPA_FIELD);               \
   } \
   if ((CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD) ==                 \
        (capaHolder & (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT << CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD)))   \
   {                                                                                                 \
       if (capaCuda.concurrentKernels)                                  \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_KERNEL_CCRNT_EXEC_CAPA_FIELD);    \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.kernelExecTimeoutEnabled)                           \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_KERNEL_EXEC_TIMEOUT_CAPA_FIELD);  \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.streamPrioritiesSupported)                          \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_STREAM_PRIO_CAPA_FIELD);          \
       }                                                                \
       /* else :continue */                                             \
       if (capaCuda.isMultiGpuBoard)                                    \
       {                                                                \
           capaHolder |= (CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<         \
                          CUDA_DEVICE_MULTI_GPU_CAPA_FIELD);            \
       }                                                                \
       /* else :continue */                                             \
                                                                        \
       capaHolder &= ~(CUDA_MODULE_DEVICE_CAPA_IS_PRESENT <<            \
                          CUDA_DEVICE_KERNEL_EXEC_CAPA_FIELD);          \
   }

/* ##########################################################################################
** #                                       FUNCTIONS                                        #
** ##########################################################################################*/

# ifdef __cplusplus
}
# endif

#endif /* !CUDA_MODULE_INTERNAL_H */
