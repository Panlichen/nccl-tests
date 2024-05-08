import os 


AR = {}
AG = {}
B = {}
R = {}
RS = {}
OPS=[AR,AG,B,R,RS]
ops_str=['all_reduce','all_gather','broadcast','reduce','reduce_scatter']
buffer_sizes = ["64", "128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M","64M","128M","256M","512M","1G","2G","4G"]
# 设置环境变量
def setEnv():
    os.environ['LD_LIBRARY_PATH'] = "/data/run01/scw6cab/OCCL/ofccl/build/lib:/data/apps/cuda/12.1/lib64"
    # os.system("which mpirun")
        
    # os.environ['LD_LIBRARY_PATH'] = "/data/home/panlichen/zrk/ofcclProj/ofccl/build/lib/:/data/home/panlichen/zrk/mpi/lib/"
    os.environ['NCCL_IGNORE_DISABLED_P2P']="1"
    os.environ['NCCL_PROTO'] = "Simple"
    os.environ['NCCL_ALGO'] = "Ring"

    os.environ['CHECK']="0"
    os.environ['SHOW_ALL_PREPARED_COLL']="0"

    os.environ['RECV_SUCCESS_FACTOR'] = "5"
    os.environ['RECV_SUCCESS_THRESHOLD']="100000000"
    os.environ['TOLERANT_UNPROGRESSED_CNT'] = "100000"
    os.environ['BASE_CTX_SWITCH_THRESHOLD'] = "800000"
    os.environ['NUM_TRY_TASKQ_HEAD'] = "6"
    os.environ['DEV_TRY_ROUND'] ="10"
    os.environ['CHECK_REMAINING_SQE_INTERVAL']="10000"
    
    os.system("echo RECV_SUCCESS_FACTOR=$RECV_SUCCESS_FACTOR")
    os.system("echo RECV_SUCCESS_THRESHOLD=$RECV_SUCCESS_THRESHOLD")
    os.system("echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT")
    os.system("echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD")
    os.system("echo NUM_TRY_TASKQ_HEAD=$NUM_TRY_TASKQ_HEAD")
    os.system("echo DEV_TRY_ROUND=$DEV_TRY_ROUND")
    os.system("echo CHECK_REMAINING_SQE_INTERVAL=$CHECK_REMAINING_SQE_INTERVAL")


def runTest(buffer_sizes,ITER,machines,i,RES_DIR):
     

    op = OPS[i]
    #input("input anyword to continue")
    op['nccl_rawData']={}
    op['occl_rawData']={}
    
    for iter in ITER:
        # data path
        op['nccl_rawData'][iter]=(RES_DIR+"/nccl_"+ops_str[i]+str(iter)+".txt")
        op['occl_rawData'][iter]=(RES_DIR+"/occl_"+ops_str[i]+str(iter)+".txt")
        # nccl run test
        op['nccl_run']="../build/"+ops_str[i]+"_perf"
        op['occl_run']="../build/ofccl_"+ops_str[i]+"_perf"
        for a in buffer_sizes:
            cmd=op['nccl_run']+" -b "+str(a)+" -e "+str(a)+" -f 2 -t 2 -g 1 -n 5 -w 2 -c 0  >>"+ op['nccl_rawData'][iter]
            print(cmd)
            os.system(cmd)
            
            # os.system("sleep 10s")
            #input("input anyword to continue") 
            cmd=op['occl_run']+" -b "+str(a)+" -e "+str(a)+" -f 2 -t 2 -g 1 -n 5 -w 2 -c 0  >>"+ op['occl_rawData'][iter]
            print(cmd)
            os.system(cmd)
            # os.system("sleep 10s")

        # os.system("sleep 60s")
        #input("enter any key to continue")
                

def main():
    setEnv()

    RES_DIR = "./4090_2card_para" 
    if os.path.exists(RES_DIR ):
        os.system("rm -r " + RES_DIR )
    os.mkdir(RES_DIR)
    
    for i in [0,1,2,3,4]:
        runTest(buffer_sizes,[0],4,i,RES_DIR)
    

if __name__ == "__main__":
    main()



