import os 


AR = {}
AG = {}
B = {}
R = {}
RS = {}
OPS=[AR,AG,B,R,RS]
ops_str=['all_reduce','all_gather','broadcast','reduce','reduce_scatter']
buffer_sizes = ["64", "128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M","64M","128M","256M","512M","1G"]
# 设置环境变量
def setEnv():
    os.environ['PATH'] = "/data/home/panlichen/zrk/mpi/bin:$PATH"
    os.system("which mpirun")
        
    os.environ['LD_LIBRARY_PATH'] = "/data/home/panlichen/zrk/mpi/lib:/data/home/panlichen/zrk/work2/ofccl/build/lib"
    os.environ['NCCL_IGNORE_DISABLED_P2P']="1"
    # os.environ['NCCL_PROTO'] = "Simple"
    # os.environ['NCCL_ALGO'] = "Ring"


def runTest(buffer_sizes,ITER,machines,i):

    RES_DIR = "./mpi_res_4hosts" 
    if os.path.exists(RES_DIR ):
        os.system("rm -r " + RES_DIR )
    os.mkdir(RES_DIR)

    op = OPS[i]
    #input("input anyword to continue")
    op['nccl_rawData']={}
    
    for iter in ITER:
        # data path
        op['nccl_rawData'][iter]=(RES_DIR+"/nccl_"+ops_str[i]+str(iter)+".txt")
        # nccl run test
        op['nccl_run']="../build/"+ops_str[i]+"_perf"
        for a in buffer_sizes:
            cmd="mpirun -np "+str(machines)+" -f  ../4node.txt "+op['nccl_run']+" -b "+str(a)+" -e "+str(a)+" -f 2 -t 8 -g 1 -n 5 -w 2 -c 0  >>"+ op['nccl_rawData'][iter]
            print(cmd)
            os.system(cmd)
            
            os.system("sleep 10s")
        os.system("sleep 60s")
        #input("enter any key to continue")
                

def main():
    setEnv() 

    # for i in [0,1,2,3,4]:
    #     runTest(buffer_sizes,[0,1,2,3,4,5],4,i)

    runTest(buffer_sizes,[0,1,2,3,4,5],4,0)
    

if __name__ == "__main__":
    main()



