// Offline fixture only: write real-SDK init data and mmap IPC; never a project client.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "app_ipc.h"
#include "shmem.h"

int main(int argc, char **argv)
{
    if (argc != 4 || chdir(argv[1]))
        return 2;
    APP_INIT_DATA aid;
    aid.major_version = 7;
    aid.minor_version = 24;
    aid.release = 1;
    aid.app_version = 410;
    std::strcpy(aid.app_name, "genefer19");
    std::strcpy(aid.authenticator, "OFFLINE_FIXTURE_NOT_AN_ACCOUNT_KEY");
    std::strcpy(aid.wu_name, "offline_gfn19");
    std::strcpy(aid.result_name, "offline_gfn19_0");
    std::strcpy(aid.user_name, "offline fixture");
    if (std::strlen(argv[3]) >= sizeof(aid.project_dir))
        return 3;
    std::strcpy(aid.project_dir, argv[3]);
    aid.userid = 1;
    aid.hostid = 1;
    aid.slot = 0;
    aid.client_pid = std::atoi(argv[2]);
    aid.shmem_seg_name = -1;
    aid.checkpoint_period = 5;
    aid.fraction_done_start = 0;
    aid.fraction_done_end = 1;
    aid.ncpus = 1;
    std::strcpy(aid.gpu_type, "NVIDIA");
    aid.gpu_device_num = 0;
    aid.gpu_opencl_dev_index = 0;
    aid.gpu_usage = 1;
    aid.rsc_fpops_est = 1e15;
    aid.rsc_fpops_bound = 1e20;
    aid.rsc_memory_bound = 8e9;
    aid.rsc_disk_bound = 8e9;
    void *memory = nullptr;
    if (create_shmem_mmap(MMAPPED_FILE_NAME, sizeof(SHARED_MEM), &memory))
        return 4;
    std::memset(memory, 0, sizeof(SHARED_MEM));
    FILE *output = std::fopen(INIT_DATA_FILE, "w");
    if (!output)
        return 5;
    int result = write_init_data_file(output, aid);
    if (std::fclose(output))
        return 6;
    detach_shmem_mmap(memory, sizeof(SHARED_MEM));
    return result;
}
